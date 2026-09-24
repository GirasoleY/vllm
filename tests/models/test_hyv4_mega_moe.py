# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run under torchrun on four SM100 GPUs; tests the Hy4 routed-expert adapter."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 4 or not torch.cuda.is_available(),
    reason="Run with torchrun --nproc-per-node=4 on SM100",
)


class _Gate(nn.Linear):
    def forward(self, x):
        return super().forward(x.float()), None


@pytest.fixture
def should_do_global_cleanup_after_test():
    # The module-scoped fixture owns the communication group across both cases.
    return False


def _fp8_roundtrip(x):
    """Independent per-32 UE8M0 activation quantization reference."""
    groups = x.float().reshape(x.shape[0], -1, 32)
    scale = torch.exp2(
        torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    quant = (groups / scale[..., None]).to(torch.float8_e4m3fn).float()
    return (quant * scale[..., None]).reshape_as(x)


def _unpack_fp4(packed, raw_scale):
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    decoded = values[(codes & 7).long()] * torch.where((codes & 8) != 0, -1, 1)
    scales = torch.exp2(raw_scale.float() - 127).repeat_interleave(32, dim=-1)
    return decoded * scales


@pytest.fixture(scope="module")
def ep_group():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("DeepGEMM MegaMoE requires SM100")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    yield SimpleNamespace(
        device_group=dist.group.WORLD, world_size=dist.get_world_size()
    )
    dist.destroy_process_group()


@pytest.mark.parametrize("tokens", [4, 128])
def test_hyv4_mxfp4_real_routing_matches_dequantized_reference(
    monkeypatch, tokens, ep_group
):
    from vllm.model_executor.layers.fused_moe.router.router_factory import (
        create_fused_moe_router,
    )
    from vllm.models.deepseek_v4.nvidia import model as dsv4
    from vllm.models.hy_v4.nvidia.moe import HYV4MoEFused

    rank, world = dist.get_rank(), dist.get_world_size()
    monkeypatch.setattr(dsv4, "get_ep_group", lambda: ep_group)
    monkeypatch.delenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", raising=False)
    torch.manual_seed(1234 + rank)
    h, intermediate, experts, topk = 6144, 2048, 256, 8
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max(32, tokens)),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with torch.device("cuda"):
        adapter = dsv4.DeepseekV4MegaMoEExperts(
            config,
            num_experts=experts,
            num_local_experts=experts // world,
            experts_start_idx=rank * (experts // world),
            top_k=topk,
            hidden_size=h,
            intermediate_size=intermediate,
            prefix="model.layers.1.mlp.experts",
        )
        # Values cover both signs and all E2M1 levels. Large inputs exercise clamp.
        adapter.w13_weight.random_(0, 256)
        adapter.w2_weight.random_(0, 256)
        adapter.w13_weight_scale.fill_(121)
        adapter.w2_weight_scale.fill_(121)
        w13, s13 = adapter.w13_weight.clone(), adapter.w13_weight_scale.clone()
        w2, s2 = adapter.w2_weight.clone(), adapter.w2_weight_scale.clone()
        moe = HYV4MoEFused.__new__(HYV4MoEFused)
        nn.Module.__init__(moe)
        moe.use_mega_moe = True
        moe.moe_swiglu_limit = 10.0
        moe.experts = adapter
        moe.shared_experts = None
        moe.gate = _Gate(h, experts, bias=False, dtype=torch.float32)
        moe.gate.requires_grad_(False)
        moe.gate.weight.data.normal_(std=0.005)
        bias = torch.randn(experts) * 0.1
        moe.mega_router = create_fused_moe_router(
            top_k=topk,
            global_num_experts=experts,
            renormalize=True,
            use_grouped_topk=True,
            num_expert_group=1,
            topk_group=1,
            scoring_func="sigmoid",
            routed_scaling_factor=2.827,
            e_score_correction_bias=bias,
        )
        x = torch.randn(tokens, h, dtype=torch.bfloat16) * 4
        scores = moe.gate(x)[0].sigmoid()
        ids = (scores + bias).topk(topk, dim=-1).indices
        weights = scores.gather(1, ids)
        weights = weights / weights.sum(-1, keepdim=True) * 2.827
        actual_weights, actual_ids = moe.mega_router.select_experts(
            x, moe.gate(x)[0], topk_indices_dtype=torch.int64
        )
        # Compare sorted pairs: selection order is not part of the router contract.
        expected_order = ids.argsort(-1)
        actual_order = actual_ids.argsort(-1)
        torch.testing.assert_close(
            ids.gather(1, expected_order), actual_ids.gather(1, actual_order)
        )
        torch.testing.assert_close(
            weights.gather(1, expected_order), actual_weights.gather(1, actual_order)
        )
        gathered = []
        for value in (x, ids, weights):
            parts = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(parts, value)
            gathered.append(torch.cat(parts))
        all_x, all_ids, all_weights = gathered
        qx = _fp8_roundtrip(all_x)
        reference = torch.zeros(world * tokens, h, dtype=torch.float32)
        clamped = 0
        for local_expert in range(experts // world):
            expert = rank * (experts // world) + local_expert
            rows, slots = torch.where(all_ids == expert)
            if rows.numel() == 0:
                continue
            first = qx[rows] @ _unpack_fp4(w13[local_expert], s13[local_expert]).T
            gate, up = first.to(torch.bfloat16).float().chunk(2, -1)
            clamped += ((gate > 10) | (up.abs() > 10)).sum().item()
            act = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            act = _fp8_roundtrip(act * all_weights[rows, slots, None])
            second = act @ _unpack_fp4(w2[local_expert], s2[local_expert]).T
            reference.index_add_(0, rows, second.to(torch.bfloat16).float())
        assert clamped > 0, "The fixture must exercise the Hy4 activation clamp"
        dist.all_reduce(reference)
        expected = reference[rank * tokens : (rank + 1) * tokens]
        adapter.finalize_weights()
        torch.accelerator.synchronize()
        dist.barrier()
        with torch.no_grad():
            output = moe(x).float()
        assert torch.isfinite(output).all()
        relative_l2 = (output - expected).norm() / expected.norm()
        print(
            f"rank={rank} tokens={tokens} relative_l2={relative_l2.item():.6f} "
            f"clamped={clamped}",
            flush=True,
        )
        assert relative_l2 < 0.02
        # The warmed path must also work in a decode CUDA graph.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = moe(x)
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(graph_output.float(), output, rtol=0, atol=0)
    dist.barrier()
    buffer = adapter.get_symm_buffer()
    buffer.destroy()
    for key, value in list(adapter._symm_buffer_cache.items()):
        if value is buffer:
            del adapter._symm_buffer_cache[key]
