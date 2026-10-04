# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the replicated-embedding fused gather/norm kernels
(``vllm.model_executor.layers.fused_embed_norm``).

The guarantee: enabling ``VLLM_REPLICATE_EMBED`` (replicated table + fused
kernels) must not change model outputs. The gathered residual is bit-exact and
the fused norms match the unfused reference.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_embed_norm import (
    fused_embed_eh_norm,
    fused_embed_norm,
)

# The model-local (untouched) eh-norm the replicate path must match.
from vllm.models.deepseek_v32.common.kernels import fused_eh_norm
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

DTYPE = torch.bfloat16
VOCAB, HIDDEN, NUM_TOKENS, EPS = 8192, 4096, 129, 1e-6

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="fused embed/norm Triton kernels require a CUDA/ROCm device",
)


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    # Full-precision (fp32) reference RMSNorm.
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    return x.float() * torch.rsqrt(var + eps) * w.float()


@requires_cuda
@torch.inference_mode()
def test_fused_embed_norm_matches_reference():
    """Main-model fusion: the residual is the exact gather and the second output
    is a correct RMSNorm. The norm matches a full-precision reference to ~2 bf16
    ulp (rtol 1e-2) -- that gap is bf16 rounding, not the kernel."""
    set_random_seed(13)
    table = torch.randn(VOCAB, HIDDEN, dtype=DTYPE, device="cuda")
    ids = torch.randint(0, VOCAB, (NUM_TOKENS,), dtype=torch.int32, device="cuda")
    weight = torch.empty(HIDDEN, dtype=DTYPE, device="cuda").normal_(1.0, 0.1)

    residual, normed = fused_embed_norm(ids, table, chain_weight=weight, eps=EPS)

    embeds = table[ids.long()]
    torch.testing.assert_close(residual, embeds, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        normed.float(), _rmsnorm(embeds, weight, EPS), atol=1e-3, rtol=1e-2
    )


@requires_cuda
@torch.inference_mode()
def test_fused_embed_eh_norm_matches_reference():
    """MTP fusion (folded gather) is bit-exact vs gathering the embeds and
    feeding the untouched model-local ``fused_eh_norm``."""
    set_random_seed(13)
    table = torch.randn(VOCAB, HIDDEN, dtype=DTYPE, device="cuda")
    ids = torch.randint(0, VOCAB, (NUM_TOKENS,), dtype=torch.int32, device="cuda")
    prev = torch.randn(NUM_TOKENS, HIDDEN, dtype=DTYPE, device="cuda")
    enorm_w = torch.randn(HIDDEN, dtype=DTYPE, device="cuda")
    hnorm_w = torch.randn(HIDDEN, dtype=DTYPE, device="cuda")
    positions = torch.arange(NUM_TOKENS, device="cuda")  # includes pos 0

    fused = fused_embed_eh_norm(positions, ids, table, prev, enorm_w, hnorm_w, EPS)
    ref = fused_eh_norm(positions, table[ids.long()], prev, enorm_w, hnorm_w, EPS)

    torch.testing.assert_close(fused, ref, atol=0.0, rtol=0.0)


@requires_cuda
@pytest.mark.parametrize("n", [0, 1, 16, 32, 128])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.inference_mode()
def test_hyv4_eh_norm_matches_unfused(n, strided, eps, default_vllm_config):
    """Preserve HY4's unmasked embedding and independent norm weights."""
    from vllm.model_executor.layers.fused_embed_norm import (
        fused_eh_norm as fused_hyv4_eh_norm,
    )
    from vllm.model_executor.layers.layernorm import RMSNorm

    set_random_seed(42)
    h = 6144
    step = 2 if strided else 1
    e = torch.randn(n * step, h, device="cuda", dtype=DTYPE)[::step]
    p = torch.randn_like(e) * 3
    en = RMSNorm(h, eps=eps).to(device="cuda", dtype=DTYPE)
    hn = RMSNorm(h, eps=eps).to(device="cuda", dtype=DTYPE)
    en.weight.normal_(1, 0.2)
    hn.weight.normal_(1, 0.2)
    before = e.clone(), p.clone()
    actual = fused_hyv4_eh_norm(e, p, en.weight, hn.weight, eps)
    from vllm import _custom_ops as ops

    en_out, hn_out = torch.empty_like(e), torch.empty_like(p)
    if n:
        ops.rms_norm(en_out, e, en.weight, eps)
        ops.rms_norm(hn_out, p, hn.weight, eps)
    expected = torch.cat([en_out, hn_out], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(e, before[0], rtol=0, atol=0)
    torch.testing.assert_close(p, before[1], rtol=0, atol=0)
    if n:
        assert actual[0, :h].abs().max() > 0
