# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored GVR prefill top-k against top_k_per_row_prefill."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask,
)
from vllm.platforms import current_platform
from vllm.third_party.trtllm_gvr_topk import (
    gvr_topk_decode_self_sampling_host as gvr_topk,
)

TOPK = 2048


def _selected_values(logits, idx, ks):
    vals = logits.gather(1, (idx.long() + ks.long().unsqueeze(1)).clamp(min=0))
    vals = torch.where(idx >= 0, vals, float("-inf"))
    return vals.sort(dim=1, descending=True).values


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="GVR requires SM100/SM103",
)
# Rows 1 / 149 / 2050 hit the three GVR row tiers.
@pytest.mark.parametrize("num_rows", [1, 149, 2050])
@pytest.mark.parametrize("num_cols", [40000, 300000])
@pytest.mark.parametrize("row_start", [0, 1024])
# masked: -inf outside 8 candidate blocks of 64 keys, < TOPK finite keys per row.
@pytest.mark.parametrize("data", ["random", "ties", "masked"])
@torch.inference_mode()
def test_gvr_prefill_topk(num_rows, num_cols, row_start, data):
    torch.manual_seed(0)
    width = -(-(num_cols + row_start) // 256) * 256
    logits = torch.randn(num_rows, width, device="cuda")
    if data == "ties":
        logits = logits.bfloat16().float()
    # Causal tail: row i sees num_cols - num_rows + i + 1 keys.
    lens = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    lens = lens + num_cols - num_rows + 1
    ks = torch.full_like(lens, row_start)
    ke = ks + lens
    if data == "masked":
        blocks = torch.rand(num_rows, num_cols // 64, device="cuda").topk(8).indices
        apply_candidate_mask(logits, ks, ke, blocks.to(torch.int32), 64)

    ref = torch.full((num_rows, TOPK), -1, dtype=torch.int32, device="cuda")
    ops.top_k_per_row_prefill(
        logits, ks, ke, ref, num_rows, logits.stride(0), logits.stride(1), TOPK
    )
    out = torch.full_like(ref, 7)  # every slot must be written
    gvr_topk.run_prefill(logits, ks, ke, out)

    assert ((out >= -1) & (out < lens.unsqueeze(1))).all()
    assert ((out >= 0).sum(1) == (ref >= 0).sum(1)).all()
    srt = out.sort(dim=1).values
    assert not ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any()
    # Equal selected scores: index sets may differ only on ties.
    torch.testing.assert_close(
        _selected_values(logits, out, ks),
        _selected_values(logits, ref, ks),
        rtol=0,
        atol=0,
    )
