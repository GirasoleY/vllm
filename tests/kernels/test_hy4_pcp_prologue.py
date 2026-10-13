# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HY V4 PCP prologue kernels must write exactly the bytes of the kernel
chain they replace: rms_norm + rotary_embedding + concat_and_cache_mla
(nvfp4_ds_mla), _k_norm_rope + indexer_k_quant_and_cache, and rotary on q_pe
+ _fused_indexer_q_rope_quant. Single GPU: the multicast kernel writes into a
local staging buffer (no multicast) and the insert's signal is preset."""

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.hy_v4.nvidia.indexer_fusion import (
    fused_indexer_k_norm_rope,
    fused_indexer_q_rope_quant,
)
from vllm.platforms import current_platform

HEADS, IDX_HEADS, BLOCK, MAX_POS = 64, 32, 64, 1 << 16
SOFTMAX_SCALE, HEAD_SCALE = 128**-0.5, IDX_HEADS**-0.5
ROW_BYTES = 352 + 128 + 4

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100)
    or not hasattr(torch.ops._C, "hy4_pcp_kv_norm_rope_quant_multicast"),
    reason="HY4 PCP prologue kernels need SM100",
)


def _cos_sin(dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    inv = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    f = torch.arange(MAX_POS, dtype=torch.float32)[:, None] * inv[None, :]
    f = f + 0.01 * torch.randn(f.shape, generator=g)
    return torch.cat([f.cos(), f.sin()], -1).to("cuda", dtype)


def make_inputs(num_tokens: int, cache_dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    gw = torch.Generator(device="cuda").manual_seed(1234)

    def rand(*shape, scale=2.0):
        x = torch.randn(*shape, generator=g, device="cuda") * scale
        return x.to(torch.bfloat16)

    # Strided views, as in the model: kv_lora inside the fused qkv_a output,
    # indexer K / weights inside the wk_weights_proj output.
    qkv = rand(num_tokens, 2048 + 576 + 64)
    kw = rand(num_tokens, 128 + IDX_HEADS)
    blocks = (num_tokens + BLOCK - 1) // BLOCK + 4
    perm = torch.randperm(blocks * BLOCK, generator=torch.Generator().manual_seed(seed))
    return SimpleNamespace(
        T=num_tokens,
        kv_lora=qkv[:, 2048 : 2048 + 576],
        kw=kw,
        positions=torch.randint(0, MAX_POS, (num_tokens,), generator=g, device="cuda"),
        cos_sin=_cos_sin(cache_dtype, 1),
        idx_cos_sin=_cos_sin(cache_dtype, 2),
        kv_w=(1 + 0.1 * torch.randn(512, generator=gw, device="cuda")).to(
            torch.bfloat16
        ),
        norm=SimpleNamespace(
            weight=1 + 0.1 * torch.randn(128, generator=gw, device="cuda"),
            bias=0.1 * torch.randn(128, generator=gw, device="cuda"),
            eps=1e-6,
        ),
        q=rand(num_tokens, HEADS, 256),
        iq=rand(num_tokens, IDX_HEADS, 128),
        slots=perm[:num_tokens].to("cuda"),
        blocks=blocks,
    )


def empty_caches(x):
    mla = torch.zeros(x.blocks, BLOCK, 352, dtype=torch.uint8, device="cuda")
    idx = torch.zeros(x.blocks, BLOCK, 132, dtype=torch.uint8, device="cuda")
    return mla, idx


def reference_k(x, mla, idx, slots):
    kv_n = torch.empty(x.T, 512, dtype=torch.bfloat16, device="cuda")
    ops.rms_norm(kv_n, x.kv_lora[:, :512], x.kv_w, 1e-6)
    k_pe = x.kv_lora[:, 512:].unsqueeze(1).clone()
    ops.rotary_embedding(x.positions, k_pe.clone(), k_pe, 64, x.cos_sin, False)
    ik = fused_indexer_k_norm_rope(
        x.kw[:, :128], x.norm, x.positions, x.idx_cos_sin, False
    )
    one = torch.ones(1, device="cuda")
    ops.concat_and_cache_mla(kv_n, k_pe.squeeze(1), mla, slots, "nvfp4_ds_mla", one)
    ops.indexer_k_quant_and_cache(ik, idx, slots, 128, "ue8m0")


def fused_k(x, mla, idx, slots, idx_slots=None):
    rows = x.T
    slot_bytes = (rows * ROW_BYTES + 15) // 16 * 16
    staging = torch.zeros(2 * slot_bytes, dtype=torch.uint8, device="cuda")
    state = torch.zeros(3, dtype=torch.int32, device="cuda")
    signal = torch.full((1,), 1 << 30, dtype=torch.int32, device="cuda")
    for _ in range(2):  # the second exchange uses the other staging slot
        torch.ops._C.hy4_pcp_kv_norm_rope_quant_multicast(
            x.kv_lora, x.kw[:, :128], x.positions, x.cos_sin, x.idx_cos_sin,
            x.kv_w, 1e-6, x.norm.weight, x.norm.bias, 1e-6, state,
            staging.data_ptr(), 0, 1, 0, x.T, 64, 8, False, False,
        )  # fmt: skip
        torch.ops._C.hy4_pcp_kv_wait_cache_insert(
            staging, signal, state, slots, mla, idx, idx_slots, 1, x.T, x.T,
            max(1, (x.T + 7) // 8), False,
        )  # fmt: skip
    torch.accelerator.synchronize()
    assert state.tolist() == [2, 0, 0]


def _rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    return cache.view(-1, cache.shape[-1])[slots]


def _idx_rows(cache: torch.Tensor, slots: torch.Tensor):
    flat = cache.view(cache.shape[0], -1)
    values = flat[:, : BLOCK * 128].reshape(-1, 128)[slots]
    scales = flat[:, BLOCK * 128 :].reshape(-1, 4)[slots]
    return values, scales


@pytest.mark.parametrize("num_tokens", [1, 7, 128, 1000, 4096])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_kv_multicast_cache_insert_matches_reference(num_tokens, cache_dtype):
    x = make_inputs(num_tokens, cache_dtype)
    mla_ref, idx_ref = empty_caches(x)
    reference_k(x, mla_ref, idx_ref, x.slots)
    mla, idx = empty_caches(x)
    # The indexer cache gets its own slot mapping.
    idx_slots = x.slots.flip(0).contiguous()
    fused_k(x, mla, idx, x.slots, idx_slots)
    torch.testing.assert_close(
        _rows(mla, x.slots), _rows(mla_ref, x.slots), atol=0, rtol=0
    )
    v, s = _idx_rows(idx, idx_slots)
    v_ref, s_ref = _idx_rows(idx_ref, x.slots)
    torch.testing.assert_close(v, v_ref, atol=0, rtol=0)
    torch.testing.assert_close(s, s_ref, atol=0, rtol=0)


@pytest.mark.parametrize("num_tokens", [1, 7, 128, 1000, 4096])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("in_place", [False, True])
def test_q_rope_indexer_quant_matches_reference(num_tokens, cache_dtype, in_place):
    x = make_inputs(num_tokens, cache_dtype)
    q_ref = x.q.clone()
    k_pe = x.kv_lora[:, 512:].unsqueeze(1).clone()
    ops.rotary_embedding(x.positions, q_ref[..., 192:], k_pe, 64, x.cos_sin, False)
    iq8_ref, iw_ref = fused_indexer_q_rope_quant(
        x.positions, x.iq, x.idx_cos_sin, x.kw[:, 128:], SOFTMAX_SCALE, HEAD_SCALE,
        False,
    )  # fmt: skip

    iq8 = torch.empty_like(iq8_ref)
    iw = torch.empty_like(iw_ref)
    if in_place:
        q = x.q.clone()
        q_out = q
    else:
        q = x.q
        q_out = torch.zeros(num_tokens, HEADS, 576, dtype=torch.bfloat16, device="cuda")
    # The top-k buffer has more rows than tokens; only the first T reset.
    topk = torch.randint(0, 1 << 20, (num_tokens + 3, 2048), dtype=torch.int32,
                         device="cuda")  # fmt: skip
    topk_tail = topk[num_tokens:].clone()
    torch.ops._C.hy4_q_rope_indexer_quant(
        q, q_out, x.positions, x.cos_sin, x.iq, x.kw[:, 128:], x.idx_cos_sin, iq8,
        iw, topk[:num_tokens], SOFTMAX_SCALE, HEAD_SCALE, False,
    )  # fmt: skip
    torch.accelerator.synchronize()
    assert bool((topk[:num_tokens] == -1).all())
    torch.testing.assert_close(topk[num_tokens:], topk_tail, atol=0, rtol=0)
    pe = q_out[..., q_out.shape[-1] - 64 :]
    torch.testing.assert_close(pe, q_ref[..., 192:], atol=0, rtol=0)
    if in_place:
        torch.testing.assert_close(q[..., :192], x.q[..., :192], atol=0, rtol=0)
    torch.testing.assert_close(
        iq8.view(torch.uint8), iq8_ref.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(iw, iw_ref, atol=0, rtol=0)


def _mrope_positions(num_tokens: int, seed: int) -> torch.Tensor:
    """[3, T] positions: text tokens (equal rows) around an image grid."""
    g = torch.Generator().manual_seed(seed)
    pos = torch.arange(num_tokens).repeat(3, 1) + 100
    img = slice(num_tokens // 4, num_tokens // 4 + num_tokens // 2)
    n = img.stop - img.start
    pos[1, img] += torch.randint(0, 64, (n,), generator=g)
    pos[2, img] += torch.randint(0, 64, (n,), generator=g)
    return pos.cuda()


@pytest.mark.parametrize("num_tokens", [1, 7, 1000])
@pytest.mark.parametrize("interleaved", [True, False])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_token_rope_applies_mrope(
    num_tokens, interleaved, cache_dtype, default_vllm_config
):
    """RoPE through token_rope's (arange, per-token rows) equals M-RoPE, and
    text rows equal the 1-D cache rows."""
    from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
    from vllm.models.hy_v4.nvidia.attention import set_token_rope, token_rope

    rope = MRotaryEmbedding(
        64, 64, 4096, 10000, False, cache_dtype, [12, 10, 10], interleaved
    ).cuda()
    positions = _mrope_positions(num_tokens, 0)
    g = torch.Generator(device="cuda").manual_seed(1)
    q = torch.randn(num_tokens, 4 * 64, generator=g, device="cuda", dtype=cache_dtype)
    k = torch.randn(num_tokens, 64, generator=g, device="cuda", dtype=cache_dtype)
    set_token_rope(rope, positions, q)
    cache = rope.cos_sin_cache
    rope_pos, rows = token_rope(rope, positions, cache)
    text = (positions[0] == positions[1]) & (positions[0] == positions[2])
    torch.testing.assert_close(rows[text], cache[positions[0, text]], atol=0, rtol=0)

    q_ref, k_ref = rope.forward_native(positions, q.clone(), k.clone())
    q_out, k_out = q.clone(), k.clone()
    ops.rotary_embedding(rope_pos, q_out, k_out, 64, rows, False)
    # fp32: forward_native and the CUDA kernel round differently (<= 0.5 ulp).
    tol = 1e-6 if cache_dtype == torch.float32 else 1e-2
    torch.testing.assert_close(q_out, q_ref, atol=tol, rtol=tol)
    torch.testing.assert_close(k_out, k_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("num_tokens", [7, 1000])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
def test_prologue_kernels_take_token_rows(num_tokens, cache_dtype):
    """The prologue kernels read only cache[positions[t]]: (arange, gathered
    rows) gives exactly the bytes of (positions, cache)."""
    x = make_inputs(num_tokens, cache_dtype)
    rows = SimpleNamespace(
        **{
            **vars(x),
            "positions": torch.arange(num_tokens, device="cuda"),
            "cos_sin": x.cos_sin[x.positions].contiguous(),
            "idx_cos_sin": x.idx_cos_sin[x.positions].contiguous(),
        }
    )
    mla_ref, idx_ref = empty_caches(x)
    fused_k(x, mla_ref, idx_ref, x.slots)
    mla, idx = empty_caches(x)
    fused_k(rows, mla, idx, x.slots)
    torch.testing.assert_close(mla, mla_ref, atol=0, rtol=0)
    torch.testing.assert_close(idx, idx_ref, atol=0, rtol=0)

    outs = []
    for y in (x, rows):
        q = y.q.clone()
        iq8 = torch.empty(num_tokens, IDX_HEADS, 128, dtype=torch.float8_e4m3fn,
                          device="cuda")  # fmt: skip
        iw = torch.empty(num_tokens, IDX_HEADS, dtype=torch.float32, device="cuda")
        topk = torch.zeros(num_tokens, 2048, dtype=torch.int32, device="cuda")
        torch.ops._C.hy4_q_rope_indexer_quant(
            q, q, y.positions, y.cos_sin, y.iq, y.kw[:, 128:], y.idx_cos_sin, iq8,
            iw, topk, SOFTMAX_SCALE, HEAD_SCALE, False,
        )  # fmt: skip
        outs.append((q, iq8.view(torch.uint8), iw))
    for a, b in zip(*outs):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
