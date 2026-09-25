# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA context parallelism (KCP) under hybrid PCP.

The end-to-end test runs the real layer on every emulated PCP rank of one GPU
and compares outputs and replicated caches with the ordinary unpartitioned
path. Rank threads run one at a time; they switch only inside collectives.
"""

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only kernels"),
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="KCP CuTe summaries require SM100-family GPUs",
    ),
]

LOWER_BOUND = -5.0


@pytest.mark.parametrize("counts", [[8], [8, 8, 8], [8, 3, 0], [1, 8, 3, 0]])
def test_kcp_merge_preserves_partial_states_and_contiguous_output(counts):
    """Ordered FP32 merge with identity-filled absent slots matches a reference."""
    from vllm.models.glm5next.nvidia.ops.kcp import kcp_merge_states

    torch.manual_seed(7)
    slots, heads, dim = 8, 2, 128
    summaries = torch.randn(slots, len(counts), heads, dim, 2 * dim, device="cuda")
    summaries *= 0.003
    summaries[..., dim:] += 0.95 * torch.eye(dim, device="cuda")
    for request, count in enumerate(counts):
        # Absent slots hold a zero S_ext and the identity transition.
        summaries[count:, request] = 0
        summaries[count:, request, :, :, dim:] = torch.eye(dim, device="cuda")
    original = summaries.clone()
    base = torch.randn(len(counts), heads, dim, dim, device="cuda") * 0.2
    expected_inits = base.new_empty(len(counts), slots, heads, dim, dim)
    expected_final = torch.empty_like(base)
    for request, count in enumerate(counts):
        state = base[request]
        for slot in range(slots):
            expected_inits[request, slot] = state
            if slot < count:
                summary = original[slot, request]
                state = state @ summary[..., dim:].transpose(-1, -2)
                state = state + summary[..., :dim].transpose(-1, -2)
        expected_final[request] = state

    order = [slot for rank in range(slots // 2) for slot in (rank, slots - 1 - rank)]
    summaries = summaries[order].contiguous()
    original_transition = summaries[..., dim:].clone()
    from vllm.v1.worker.gpu.pcp_manager import merge_source_rows

    pairs = [(n, c) for n in range(len(counts)) for c in range(slots)]
    rows = torch.tensor(merge_source_rows(len(counts), slots, pairs), device="cuda")
    initial, final = kcp_merge_states(summaries, base, rows, len(pairs))
    torch.testing.assert_close(
        initial.view(len(counts), slots, heads, dim, dim),
        expected_inits,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(final, expected_final, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        summaries[..., dim:], original_transition, atol=0, rtol=0
    )
    assert final.is_contiguous()


@pytest.mark.parametrize("batch", ["ragged", "empty", "none"])
@pytest.mark.parametrize(
    "dim,value_dim,chunk_size",
    [(64, 80, 16), (96, 80, 32), (128, 80, 64), (128, 128, 64), (192, 128, 64)],
)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="KCP CuTe summaries require SM100-family GPUs",
)
@torch.inference_mode()
def test_kcp_summaries_match_reference_and_preserve_destinations(
    dim, value_dim, chunk_size, batch
):
    """Check S arithmetic/rounding and the FP32 M chain, including empty segments."""
    from itertools import accumulate

    from vllm.models.glm5next.nvidia.ops.kcp import kcp_compute_summaries

    torch.manual_seed(57458)
    lengths = [0, 1, chunk_size + 3, 8 * chunk_size + 1]
    if batch != "ragged":
        lengths = [0, 0] if batch == "empty" else []
    tokens, heads = sum(lengths), 2
    shape = (1, tokens, heads, dim)
    kg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.025
    w = torch.randn_like(kg) * 0.025
    u = torch.randn((1, tokens, heads, value_dim), dtype=kg.dtype, device="cuda")
    gate = -torch.rand(shape, device="cuda") * 0.03
    offsets = [0] + list(accumulate(lengths))
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    dense = kcp_compute_summaries(kg, u, w, gate, cu, chunk_size)
    out = torch.full(
        (len(lengths) + 3, heads, dim, value_dim + dim), 7.0, device="cuda"
    )
    indices = torch.arange(len(lengths), 0, -1, device="cuda", dtype=torch.int64)
    out[indices] = float("nan")
    expected = out.clone()
    expected[indices] = dense
    actual = kcp_compute_summaries(
        kg, u, w, gate, cu, chunk_size, out=out, output_indices=indices
    )
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual = dense.cpu().double()
    # Independent CPU FP64 products; preserve the S producer's BF16 carry
    # and residual conversions explicitly. No candidate kernel forms the oracle.
    kg, u, w, gate = (x[0].cpu().double() for x in (kg, u, w, gate))
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = torch.zeros(heads, dim, value_dim, dtype=torch.float64)
        transfer = torch.eye(dim, dtype=torch.float64).expand(heads, dim, dim).clone()
        for begin in range(start, end, chunk_size):
            stop = min(begin + chunk_size, end)
            keys = kg[begin:stop].transpose(0, 1)
            weights = w[begin:stop].transpose(0, 1)
            values = u[begin:stop].transpose(0, 1)
            decay = gate[stop - 1].exp2()
            residual = values - weights @ state.to(torch.bfloat16).double()
            state = (
                decay[..., None] * state
                + keys.transpose(-1, -2) @ residual.to(torch.bfloat16).double()
            )
            transition = torch.diag_embed(decay) - keys.transpose(-1, -2) @ weights
            transfer = transition @ transfer
        summary = actual[seq, ..., :value_dim]
        if end - start <= 2 * chunk_size:
            torch.testing.assert_close(summary, state, atol=3e-4, rtol=1e-2)
        else:
            # FP32/FP64 reductions can cross repeated BF16 rounding boundaries.
            # Bound each head's aggregate error and isolated element errors.
            error = summary - state
            assert torch.all(
                torch.linalg.vector_norm(error, dim=(-2, -1))
                <= 2e-4 * torch.linalg.vector_norm(state, dim=(-2, -1))
            )
            assert error.abs().max() < 2e-3
        torch.testing.assert_close(
            actual[seq, ..., value_dim:], transfer, atol=2e-6, rtol=1e-4
        )


@pytest.mark.parametrize("use_tma", [False, True])
@pytest.mark.parametrize("gate_shift", [0.0, -4.0, -10.0])
@pytest.mark.parametrize(
    "dtype,heads,dim,chunk_size,safe_gate,strided,lower_bound,gate_offset,uniform",
    [
        (torch.bfloat16, 64, 128, 64, True, False, LOWER_BOUND, 0.0, False),
        (torch.float16, 2, 96, 32, True, True, LOWER_BOUND, 0.0, False),
        (torch.float32, 1, 192, 16, False, True, LOWER_BOUND, 0.0, False),
        (torch.bfloat16, 2, 128, 64, True, False, -10.0, 12.0, False),
        # Bare exp factors are normal, but scaled BF16 operands lose precision.
        (torch.bfloat16, 2, 128, 64, True, False, -5.4542, 80.0, True),
    ],
)
@torch.inference_mode()
def test_flashkda_prepare_matches_fla_for_ragged_segments(
    monkeypatch,
    use_tma,
    gate_shift,
    dtype,
    heads,
    dim,
    chunk_size,
    safe_gate,
    strided,
    lower_bound,
    gate_offset,
    uniform,
):
    """Native preparation covers full, short and empty segments without fallback."""
    pytest.importorskip("vllm._flashkda_C")
    if not hasattr(torch.ops._flashkda_C, "kcp_prepare"):
        pytest.skip("FlashKDA extension was built without KCP preparation")
    import importlib

    from vllm.models.glm5next.common.kda import _cast_sigmoid
    from vllm.models.glm5next.nvidia.ops.third_party.kda import kernels
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
        prepare_kcp,
    )
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril

    solve = importlib.import_module(
        "vllm.third_party.flash_linear_attention.ops.solve_tril"
    )
    prepare_module = importlib.import_module(prepare_kcp.__module__)
    if use_tma:
        from vllm.triton_utils.allocation import set_triton_allocator

        set_triton_allocator(torch.device("cuda"))
    monkeypatch.setattr(solve, "is_tma_supported", use_tma)

    class PoisonedStorage:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *args, **kwargs):
            return torch.empty(*args, **kwargs).fill_(float("nan"))

        def empty_like(self, *args, **kwargs):
            return torch.empty_like(*args, **kwargs).fill_(float("nan"))

    for module in (kernels, solve, prepare_module):
        monkeypatch.setattr(module, "torch", PoisonedStorage())
    torch.manual_seed(9202104)
    tokens = 2 * chunk_size + 1
    inputs_shape = (1, tokens, heads, dim * (2 if strided else 1))
    q, k, raw_g = [
        torch.randn(inputs_shape, device="cuda", dtype=dtype) for _ in range(3)
    ]
    if strided:
        q, k, raw_g = (x[..., ::2] for x in (q, k, raw_g))
    raw_g.mul_(0.1).add_(gate_shift + gate_offset)
    beta_raw = torch.randn(1, tokens, 128, device="cuda", dtype=dtype)[
        ..., 17 : 17 + heads
    ]
    a_log = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads * dim, device="cuda") * 0.1
    if uniform:
        q.fill_(1)
        k.fill_(1)
        beta_raw.zero_()
        a_log.zero_()
        bias.zero_()
    inputs = q, k, raw_g, beta_raw, a_log, bias
    cu = torch.tensor([0, 0, 1, chunk_size, tokens], device="cuda", dtype=torch.int64)
    chunks = torch.tensor(
        [[1, 0], [2, 0], [3, 0], [3, 1]], device="cuda", dtype=torch.int32
    )
    metadata = dict(cu_seqlens=cu, chunk_indices=chunks, chunk_size=chunk_size)
    actual = prepare_kcp(
        *inputs,
        cu,
        chunks,
        lower_bound=lower_bound if safe_gate else None,
        chunk_size=chunk_size,
    )
    qn, kn, beta = (
        l2norm_fwd(q.contiguous()),
        l2norm_fwd(k.contiguous()),
        _cast_sigmoid(beta_raw),
    )
    gate = kernels.fused_kda_gate_chunk_cumsum(
        raw_g.contiguous(),
        a_log,
        bias,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        **metadata,
    )
    a, aqk = kernels.chunk_kda_scaled_dot_kkt_fwd(
        qn,
        kn,
        gate,
        beta,
        scale=dim**-0.5,
        output_dtype=torch.float32,
        **metadata,
    )
    # Poisoning covers padding and upper triangles in both preparation paths.
    for value in (actual[4], actual[5], a, aqk):
        assert torch.isfinite(value).all()
    bf16_inverse = solve_tril(
        a, cu_seqlens=cu, chunk_indices=chunks, output_dtype=torch.bfloat16
    )
    assert torch.isfinite(bf16_inverse).all()
    # Compare the complete inverse; native preparation has inverted its diagonal.
    inverse_args = dict(cu_seqlens=cu, chunk_indices=chunks, output_dtype=torch.float32)
    actual = (
        *actual[:4],
        solve_tril(actual[4], diagonal_inverted=True, **inverse_args),
        actual[5],
    )
    a = solve_tril(a, **inverse_args)
    # Approximate gate activation and BF16 MMA differ from FLA materialization.
    tolerances = [
        (0, 0),
        (0, 0),
        (1e-3, 2e-3),
        (0, 4e-6),
        (1e-2, 1e-3),
        (1e-2, 2e-4),
    ]
    for value, expected, (rtol, atol) in zip(
        actual, (qn, kn, gate, beta, a, aqk), tolerances, strict=True
    ):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)
    empty = [x[:, :0] for x in inputs[:4]] + list(inputs[4:])
    empty_cu = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    empty_chunks = torch.empty((0, 2), device="cuda", dtype=torch.int64)
    result = prepare_kcp(
        *empty,
        empty_cu,
        empty_chunks,
        lower_bound=lower_bound if safe_gate else None,
        chunk_size=chunk_size,
    )
    assert all(value.numel() == 0 for value in result)


class _EmulatedGroup:
    """Serialized rank threads that exchange tensors only in all_gather."""

    def __init__(self, world: int):
        self.world = world
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(world)
        self.parts: list[torch.Tensor | None] = [None] * world
        self.local = threading.local()

    def _wait(self):
        self.lock.release()
        self.barrier.wait()
        self.lock.acquire()

    def all_gather(self, tensor, dim=0):
        self.parts[self.local.rank] = tensor
        self._wait()
        gathered = torch.cat(self.parts, dim)
        self._wait()
        return gathered

    def run(self, fn):
        errors = []

        def body(rank):
            self.local.rank = rank
            with self.lock:
                try:
                    fn(rank)
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                    self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]


def _layer(heads, dim, cache):
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    torch.manual_seed(1)
    channels = heads * dim
    convs = [
        SimpleNamespace(weight=torch.randn(channels, 1, 4, device="cuda") * 0.3)
        for _ in range(3)
    ]
    layer = SimpleNamespace(
        prefix="kda",
        kv_cache=cache,
        q_conv1d=SimpleNamespace(weight=convs[0].weight, bias=None),
        k_conv1d=convs[1],
        v_conv1d=convs[2],
        _merged_conv_weight=None,
        _conv_state_dim_first=True,
        A_log=0.5 * torch.randn(heads, device="cuda"),
        dt_bias=0.1 * torch.randn(channels, device="cuda"),
        local_num_heads=heads,
        head_dim=dim,
        local_projection_size=channels,
        kda_safe_gate=True,
        kda_lower_bound=LOWER_BOUND,
        kda_prefill_backend="triton",
    )
    for name in ("_forward", "_forward_kcp", "_conv_state_and_weights"):
        method = getattr(Glm5NextLinearAttention, name)
        setattr(layer, name, MethodType(getattr(method, "__wrapped__", method), layer))
    return layer


def _reference_metadata(lengths, computed, slots, num_decodes):
    """Ordinary GDN metadata of the unpartitioned batch, decodes first."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    qsl = torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32)
    nums_dict, batch_ptr, offsets = compute_causal_conv1d_metadata(
        qsl, device=torch.device("cuda")
    )
    return GDNAttentionMetadata(
        num_prefills=len(lengths) - num_decodes,
        num_prefill_tokens=int(sum(lengths[num_decodes:])),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(sum(lengths)),
        has_initial_state=torch.tensor(computed, device="cuda") > 0,
        non_spec_query_start_loc=qsl.cuda(),
        non_spec_state_indices_tensor=slots,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offsets,
    )


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "lengths,computed,num_decodes",
    [
        # Two decodes, a short prefill with empty slots, a continued prefill
        # and a fresh prefill spanning several chunks.
        ([1, 1, 5, 150, 70], [8, 3, 0, 64, 0], 2),
        # One short continued prefill: some ranks hold no prefill tokens.
        ([3], [5], 0),
    ],
)
@torch.inference_mode()
def test_kcp_layer_matches_unpartitioned_forward(
    monkeypatch, world, lengths, computed, num_decodes
):
    from vllm.models.glm5next.common import kda
    from vllm.models.glm5next.nvidia.ops import kcp
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    torch.manual_seed(5)
    heads, dim, halo, padding = 2, 128, 3, 2
    channels = heads * dim
    num_slots = 2 * len(lengths) + 1
    # Distinct, shuffled cache slots; slot 0 is NULL_BLOCK_ID and the spare
    # slots must stay untouched.
    slots = torch.randperm(num_slots - 1, device="cuda")[: len(lengths)] + 1
    conv = torch.randn(num_slots, 3 * channels, halo, device="cuda").bfloat16()
    state = torch.randn(num_slots, heads, dim, dim, device="cuda") * 0.1
    tokens = sum(lengths)
    qkv = torch.randn(tokens, 3 * channels, device="cuda").bfloat16()
    gate = torch.randn(1, tokens, heads, dim, device="cuda").bfloat16()
    beta = torch.randn(1, tokens, heads, device="cuda").bfloat16()
    prefilling = np.arange(len(lengths)) >= num_decodes

    def forward(layer, metadata, qkv, gate, beta, out):
        context = SimpleNamespace(attn_metadata={"kda": metadata})
        monkeypatch.setattr(kda, "get_forward_context", lambda: context)
        layer._forward(qkv, gate, beta, out)

    reference_cache = (conv.clone(), state.clone())
    reference = torch.empty(1, tokens, heads, dim, device="cuda").bfloat16()
    forward(
        _layer(heads, dim, reference_cache),
        _reference_metadata(lengths, computed, slots, num_decodes),
        qkv,
        gate,
        beta,
        reference,
    )

    qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    global_batch = SimpleNamespace(
        num_reqs=len(lengths),
        num_scheduled_tokens=np.asarray(lengths, dtype=np.int32),
        is_prefilling_np=prefilling,
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        query_start_loc_np=qsl,
        num_draft_tokens_per_req=None,
    )
    group = _EmulatedGroup(world)
    monkeypatch.setattr(kda, "get_pcp_group", lambda: group)
    monkeypatch.setattr(kcp, "get_pcp_group", lambda: group)
    contexts: dict[int, SimpleNamespace] = {}
    monkeypatch.setattr(kda, "get_forward_context", lambda: contexts[group.local.rank])
    results = {}

    def run_rank(rank):
        manager = PCPManager(world, rank, torch.device("cuda"))
        manager._global_batch = global_batch
        segments = manager._get_rank_segments(rank, lengths, prefilling, qsl)
        manager._local_segments = tuple(segments)
        plan = manager.build_hybrid_plan(halo)
        index = torch.tensor(
            [i for s in segments for i in range(*s.global_batch_slice.indices(tokens))],
            device="cuda",
            dtype=torch.int64,
        )
        pad = torch.zeros(padding, dtype=torch.int64, device="cuda")
        local = torch.cat((index, pad))
        metadata = _reference_metadata(lengths, computed, slots, num_decodes)
        metadata.cp_plan = plan.with_state_indices(slots)
        contexts[rank] = SimpleNamespace(attn_metadata={"kda": metadata})
        cache = (conv.clone(), state.clone())
        out = torch.full((1, len(local), heads, dim), 7.0, device="cuda").bfloat16()
        _layer(heads, dim, cache)._forward(
            qkv[local], gate[:, local], beta[:, local], out
        )
        results[rank] = (index, out, cache)

    group.run(run_rank)

    spare = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    spare[slots] = False
    for index, out, (rank_conv, rank_state) in results.values():
        torch.testing.assert_close(
            out[0, : len(index)], reference[0, index], atol=4e-3, rtol=2e-2
        )
        assert (out[0, len(index) :] == 0).all()
        torch.testing.assert_close(rank_conv, reference_cache[0], atol=0, rtol=0)
        # Decodes use the recurrent kernel and short prefills are stitched from
        # BF16 segment summaries, so states differ within output tolerance.
        torch.testing.assert_close(
            rank_state[slots], reference_cache[1][slots], atol=4e-3, rtol=2e-2
        )
        assert torch.equal(rank_state[spare], state[spare])
