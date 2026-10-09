# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The direct MLA query path (absorb bmm written into the 576-wide query
buffer, then a q_pe copy) must produce the same bytes as the bmm into a
contiguous buffer followed by concat_mla_q."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.attention.mla_attention import _copy_rows_16b
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")

HEADS, NOPE, LORA, ROPE = 64, 192, 512, 64


@pytest.mark.parametrize("num_tokens", [1, 7, 256, 4096])
def test_direct_q_matches_concat(num_tokens):
    torch.manual_seed(0)
    q = torch.randn(num_tokens, HEADS, NOPE + ROPE, device="cuda").to(torch.bfloat16)
    w_uk_t = (torch.randn(HEADS, NOPE, LORA, device="cuda") * NOPE**-0.5).to(
        torch.bfloat16
    )
    q_nope = q[..., :NOPE].transpose(0, 1)
    q_pe = q[..., NOPE:]

    ql_nope = torch.empty(num_tokens, HEADS, LORA, dtype=q.dtype, device="cuda")
    torch.bmm(q_nope, w_uk_t, out=ql_nope.transpose(0, 1))
    expected = torch.empty(num_tokens, HEADS, LORA + ROPE, dtype=q.dtype, device="cuda")
    ops.concat_mla_q(ql_nope, q_pe, expected)

    direct = torch.empty_like(expected)
    torch.bmm(q_nope, w_uk_t, out=direct[..., :LORA].transpose(0, 1))
    _copy_rows_16b(direct[..., LORA:], q_pe)
    torch.testing.assert_close(direct, expected, atol=0, rtol=0)


@pytest.mark.parametrize("offset", [0, 1, 8])
@pytest.mark.parametrize("width", [64, 60])
def test_copy_rows_16b(offset, width):
    """Exact for every layout; reinterprets only when rows and strides allow."""
    src_base = torch.randn(33, 8, 256 + 16, device="cuda").to(torch.bfloat16)
    dst_base = torch.zeros(33, 8, 576 + 16, dtype=torch.bfloat16, device="cuda")
    src = src_base[..., 192 + offset : 192 + offset + width]
    dst = dst_base[..., 512 + offset : 512 + offset + width]
    _copy_rows_16b(dst, src)
    torch.testing.assert_close(dst, src, atol=0, rtol=0)
    # Nothing outside the destination view is written.
    untouched = dst_base.clone()
    untouched[..., 512 + offset : 512 + offset + width] = 0
    assert not untouched.any()
