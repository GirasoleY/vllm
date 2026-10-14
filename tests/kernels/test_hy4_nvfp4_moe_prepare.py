# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""prepare_nvfp4_megamoe_inputs must write exactly the bytes of the chain it
replaces: the standalone NVFP4 input quant kernel, plus either
prepare_megamoe_inputs (block-FP8 shared expert) or the torch copies and
padding masks (BF16 / no shared expert)."""

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.models.hy_v4.nvidia.nvfp4_moe import prepare_nvfp4_megamoe_inputs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

HIDDEN, TOP_K, NUM_EXPERTS = 6144, 8, 256

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="NVFP4 MegaMoE input staging needs SM100",
)


@triton.jit
def _ref_quantize_nvfp4_input(X, GlobalScale, Q, SF, H: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1) * 8 + tl.arange(0, 8)
    pair = tl.arange(0, 8)
    offset = block[:, None] * 16 + pair[None, :] * 2
    x0 = tl.load(X + row * H + offset, offset < H, 0).to(tl.float32)
    x1 = tl.load(X + row * H + offset + 1, offset < H, 0).to(tl.float32)
    scale = tl.load(GlobalScale)
    amax = tl.max(tl.maximum(tl.abs(x0), tl.abs(x1)), 1)
    sf = tl.minimum(448.0, tl.maximum(2.0**-9, tl.div_rn(amax, 6.0 * scale)))
    sf = sf.to(tl.float8e4nv).to(tl.float32)
    denom = sf[:, None] * scale
    x0 = tl.div_rn(x0, denom)
    x1 = tl.div_rn(x1, denom)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
        " cvt.u32.u8 $0, tmp; }",
        constraints="=r,f,f",
        args=[x0, x1],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(Q + row * (H // 2) + offset // 2, packed.to(tl.uint8), offset < H)
    tl.store(SF + row * (H // 16) + block, sf, block < H // 16)


def _buffers(m: int, shared: str | None, shared_block_m: int):
    sf_rows = triton.cdiv(m, shared_block_m) * triton.cdiv(shared_block_m, 128) * 128
    bufs = {
        "x": torch.full((m, HIDDEN // 2), 0xAB, dtype=torch.uint8, device="cuda"),
        "x_sf": torch.full((m, HIDDEN // 16), 0xCD, dtype=torch.uint8, device="cuda"),
        "topk_idx": torch.full((m, TOP_K), -7, dtype=torch.int64, device="cuda"),
        "topk_weights": torch.full((m, TOP_K), -7.0, device="cuda"),
    }
    if shared == "fp8":
        bufs["shared_x"] = torch.full(
            (m, HIDDEN), 0x5A, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        # DeepGEMM's MN-major packed UE8M0 layout.
        bufs["shared_x_sf"] = torch.full(
            (HIDDEN // 128, sf_rows), 0x3C3C3C3C, dtype=torch.int32, device="cuda"
        ).t()
    elif shared == "bf16":
        bufs["shared_x"] = torch.full(
            (m, HIDDEN), -3.0, dtype=torch.bfloat16, device="cuda"
        )
    return bufs


def _reference(hidden, scale, topk_weights, topk_ids, padding, shared, block_m):
    m = hidden.shape[0]
    out = _buffers(m, shared, block_m)
    _ref_quantize_nvfp4_input[(m, HIDDEN // 128)](
        hidden, scale, out["x"], out["x_sf"].view(torch.float8_e4m3fn), HIDDEN
    )
    if shared == "fp8":
        scratch = torch.empty((m, HIDDEN // 128), dtype=torch.int32, device="cuda")
        prepare_megamoe_inputs(
            hidden,
            topk_weights,
            topk_ids,
            out["shared_x"],
            scratch,
            out["topk_idx"],
            out["topk_weights"],
            is_padding=padding,
            shared_x_sf=out["shared_x_sf"],
            shared_block_m=block_m,
        )
    else:
        out["topk_idx"].copy_(topk_ids)
        out["topk_weights"].copy_(topk_weights)
        if padding is not None:
            out["topk_idx"].masked_fill_(padding[:, None], -1)
            out["topk_weights"].masked_fill_(padding[:, None], 0)
        if shared == "bf16":
            out["shared_x"].copy_(hidden)
    return out


@pytest.mark.parametrize("m", [1, 7, 64, 200, 333, 1024])
@pytest.mark.parametrize(
    "shared,shared_block_m", [(None, 16), ("bf16", 16), ("fp8", 16), ("fp8", 192)]
)
@pytest.mark.parametrize("with_padding", [False, True])
@pytest.mark.parametrize("scale_mul", [1.0, 1e-3, 1e3])
def test_prepare_nvfp4_megamoe_inputs_matches_reference(
    m, shared, shared_block_m, with_padding, scale_mul
):
    g = torch.Generator(device="cuda").manual_seed(m)
    hidden = torch.randn(m, HIDDEN, generator=g, device="cuda") * 3
    hidden[0, :128] = 0
    hidden[0, 128:256] = -0.0
    hidden = hidden.to(torch.bfloat16)
    # scale_mul != 1 drives the E4M3 block scale into its 448 / 2^-9 clamps.
    scale = (hidden.abs().amax().float() / (6 * 448) * scale_mul).reshape(1)
    topk_ids = torch.randint(
        0, NUM_EXPERTS, (m, TOP_K), generator=g, device="cuda", dtype=torch.int32
    )
    topk_weights = torch.rand(m, TOP_K, generator=g, device="cuda")
    padding = None
    if with_padding:
        padding = torch.rand(m, generator=g, device="cuda") < 0.3

    ref = _reference(
        hidden, scale, topk_weights, topk_ids, padding, shared, shared_block_m
    )
    out = _buffers(m, shared, shared_block_m)
    prepare_nvfp4_megamoe_inputs(
        hidden,
        scale,
        topk_weights,
        topk_ids,
        out["x"],
        out["x_sf"].view(torch.float8_e4m3fn),
        out["topk_idx"],
        out["topk_weights"],
        is_padding=padding,
        shared_x=out.get("shared_x"),
        shared_x_sf=out.get("shared_x_sf"),
        shared_block_m=shared_block_m if shared == "fp8" else None,
    )
    for name, expected in ref.items():
        actual = out[name]
        if actual.dtype.is_floating_point:
            actual = actual.view(
                torch.uint8
                if actual.element_size() == 1
                else torch.int16
                if actual.element_size() == 2
                else torch.int32
            )
            expected = expected.view(actual.dtype)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=name)
