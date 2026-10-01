# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check HY4 fusion rounding, scale layout, trailing RoPE and routing weights."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_packed_for_deepgemm,
)
from vllm.models.hy_v4.nvidia.prefill import gate_quant, norm_quant

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("tokens", [1, 7, 128, 8192])
@pytest.mark.parametrize("head_dim", [1, 128])
def test_gate_quant_matches_bf16_rounding(tokens, head_dim):
    torch.manual_seed(42)
    x = torch.randn(tokens + 1, 16384, device="cuda", dtype=torch.bfloat16)[1:]
    gate = torch.randn(tokens, 16384 // head_dim + 128, device="cuda", dtype=x.dtype)
    gate = gate[:, : 16384 // head_dim]
    x[:, :128] = 0
    gate[:, :8] = torch.tensor([-80, -20, -5, 0, 5, 20, 80, 100], device="cuda")
    ref = x * gate.sigmoid().repeat_interleave(head_dim, dim=1)
    expected = per_token_group_quant_fp8_packed_for_deepgemm(ref, 128, use_ue8m0=True)
    actual = gate_quant(x, gate, head_dim)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    torch.testing.assert_close(actual[0].float(), expected[0].float(), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [1, 7, 128, 8192])
def test_norm_quant_strided_latent(tokens):
    from types import SimpleNamespace

    torch.manual_seed(42)
    x = torch.randn(tokens, 2624, device="cuda", dtype=torch.bfloat16)[:, :2048]
    w = torch.randn(2048, device="cuda", dtype=x.dtype)
    norm = SimpleNamespace(weight=w, variance_epsilon=1e-6)
    from vllm import _custom_ops as ops

    ref = torch.empty_like(x)
    ops.rms_norm(ref, x, w, norm.variance_epsilon)
    expected = per_token_group_quant_fp8_packed_for_deepgemm(ref, 128, use_ue8m0=True)
    actual = norm_quant(x, norm)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    # Reduction order may move values across a BF16 rounding boundary.
    torch.testing.assert_close(
        actual[0].float(), expected[0].float(), rtol=0.15, atol=2
    )


@pytest.mark.parametrize("tokens", [1, 17, 8192])
def test_indexer_rotates_trailing_dimensions(tokens):
    from vllm.models.hy_v4.nvidia.indexer_fusion import fused_indexer_q_rope_quant

    torch.manual_seed(42)
    q = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1024, 64, device="cuda")
    pos = torch.randint(1024, (tokens,), device="cuda")
    weights = torch.randn(tokens, 160, device="cuda", dtype=q.dtype)[:, 128:]
    ref = q.clone()
    cos, sin = (
        cache[pos, :32, None].transpose(1, 2),
        cache[pos, 32:, None].transpose(1, 2),
    )
    even, odd = q[..., 64::2].float(), q[..., 65::2].float()
    ref[..., 64::2] = even * cos - odd * sin
    ref[..., 65::2] = odd * cos + even * sin
    quant, scale = per_token_group_quant_fp8(ref.reshape(-1, 128), 128, use_ue8m0=True)
    expected_weights = weights * scale.reshape(tokens, 32) * 128**-0.5 * 32**-0.5
    actual, actual_weights = fused_indexer_q_rope_quant(
        pos, q, cache, weights, 128**-0.5, 32**-0.5, False
    )
    torch.testing.assert_close(
        actual.float(), quant.view_as(actual).float(), rtol=0, atol=0
    )
    torch.testing.assert_close(actual_weights, expected_weights)


@pytest.mark.parametrize("renormalize", [False, True])
def test_router_uses_bias_only_for_selection(renormalize):
    from vllm.model_executor.layers.fused_moe.router.router_factory import (
        create_fused_moe_router,
    )

    torch.manual_seed(42)
    logits = torch.randn(8192, 256, device="cuda")
    bias = torch.randn(256, device="cuda") * 0.1
    router = create_fused_moe_router(
        top_k=8,
        global_num_experts=256,
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        renormalize=renormalize,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
    )
    weights, ids = router._compute_routing(logits, logits, torch.int64)
    scores = logits.sigmoid()
    expected_ids = (scores + bias).topk(8, dim=-1).indices
    assert torch.equal(ids.sort(-1).values, expected_ids.sort(-1).values)
    expected = scores.gather(1, ids.long())
    if renormalize:
        expected /= expected.sum(-1, keepdim=True)
    torch.testing.assert_close(weights, expected * 2.5)


@pytest.mark.parametrize("dynamic", [False, True])
def test_gate_projection_fusion_preserves_shards_and_gemm(dynamic):
    from types import SimpleNamespace

    from torch import nn

    from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
        DeepGemmFp8BlockScaledMMKernel,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_fp8_weight_block,
    )
    from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
    from vllm.models.hy_v4.nvidia.prefill import fp8_linear

    torch.manual_seed(42)
    kernel = DeepGemmFp8BlockScaledMMKernel.__new__(DeepGemmFp8BlockScaledMMKernel)
    kernel.config = SimpleNamespace(out_dtype=torch.bfloat16)
    kernel.use_deep_gemm_e8m0 = True

    def projection(n):
        layer = nn.Module()
        layer.bias = None
        selected = kernel
        if dynamic:
            from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
                FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
            )

            selected = FlashInferFp8DeepGEMMDynamicBlockScaledKernel.__new__(
                FlashInferFp8DeepGEMMDynamicBlockScaledKernel
            )
            selected.fallback = kernel
        layer.quant_method = SimpleNamespace(kernel=selected)
        w = torch.randn(n, 6144, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full(((n + 127) // 128, 48), 0.01, device="cuda")
        w, scale = deepgemm_post_process_fp8_weight_block(w, scale, (128, 128), True)
        layer.weight = nn.Parameter(w, requires_grad=False)
        layer.weight_scale_inv = nn.Parameter(scale, requires_grad=False)
        return layer

    attn = HYV4MLAAttention.__new__(HYV4MLAAttention)
    nn.Module.__init__(attn)
    attn.fused_qkv_a_proj = projection(2624)
    attn.linear_gate = projection(16384)
    attn.use_hpc_gated_mla = False
    x = torch.randn(128, 6144, device="cuda", dtype=torch.bfloat16)
    quant = per_token_group_quant_fp8_packed_for_deepgemm(x, 128, use_ue8m0=True)
    expected = torch.cat(
        (fp8_linear(attn.fused_qkv_a_proj, quant), fp8_linear(attn.linear_gate, quant)),
        dim=-1,
    )
    attn.prepare_gate_fusion()
    actual = kernel.apply_block_scaled_mm(
        quant[0], attn._qkv_gate_weight, quant[1], attn._qkv_gate_scale
    )
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    assert attn.linear_gate.weight.data_ptr() == attn._qkv_gate_weight[2624:].data_ptr()
    assert attn.fused_qkv_a_proj.weight.data_ptr() == attn._qkv_gate_weight.data_ptr()


@pytest.mark.parametrize("tokens", [1, 17, 8192])
@pytest.mark.parametrize("neox", [False, True])
def test_indexer_key_norm_and_trailing_rope(tokens, neox):
    from types import SimpleNamespace

    from vllm.models.hy_v4.nvidia.indexer_fusion import fused_indexer_k_norm_rope

    torch.manual_seed(42)
    k = torch.randn(tokens, 160, device="cuda", dtype=torch.bfloat16)[:, :128]
    norm = torch.nn.LayerNorm(128, eps=1e-6, device="cuda", dtype=k.dtype)
    norm.weight.data.normal_()
    norm.bias.data.normal_()
    cache = torch.randn(1024, 64, device="cuda")
    positions = torch.randint(1024, (tokens,), device="cuda")
    rope = SimpleNamespace(cos_sin_cache=cache, is_neox_style=neox)
    ref = norm(k)
    if neox:
        first, second = slice(64, 96), slice(96, 128)
    else:
        first, second = slice(64, None, 2), slice(65, None, 2)
    x, y = ref[:, first].float(), ref[:, second].float()
    cos, sin = cache[positions, :32], cache[positions, 32:]
    ref[:, first] = x * cos - y * sin
    ref[:, second] = y * cos + x * sin
    actual = fused_indexer_k_norm_rope(k, norm, positions, rope)
    torch.testing.assert_close(actual, ref, rtol=0.016, atol=0.016)
