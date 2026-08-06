# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist
import torch.nn.functional as F

import vllm.models.kimi_k3.nvidia.latent_moe_runner as latent_moe_runner
from tests.utils import (
    init_test_distributed_environment,
    multi_gpu_test,
    multi_process_parallel,
)
from vllm.distributed import get_tp_group
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExpertsOrder,
)
from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
from vllm.models.kimi_k3.nvidia.latent_moe_runner import LatentMoERunner
from vllm.models.kimi_k3.nvidia.ops.latent_moe_tail import KimiK3LatentMoETailOp
from vllm.platforms import current_platform

HIDDEN_SIZE = 7168
LATENT_SIZE = 3584
EPS = 0.1


class _SharedExpertsOrderStub:
    def __init__(self, order: SharedExpertsOrder) -> None:
        self.order = order

    def _determine_shared_experts_order(
        self, hidden_states: torch.Tensor
    ) -> SharedExpertsOrder:
        return self.order


@pytest.mark.parametrize(
    ("num_tokens", "dtype", "order", "expected"),
    [
        (256, torch.bfloat16, SharedExpertsOrder.NO_OVERLAP, False),
        (257, torch.bfloat16, SharedExpertsOrder.NO_OVERLAP, True),
        (512, torch.bfloat16, SharedExpertsOrder.NO_OVERLAP, True),
        (513, torch.bfloat16, SharedExpertsOrder.NO_OVERLAP, False),
        (448, torch.float32, SharedExpertsOrder.NO_OVERLAP, False),
        (448, torch.bfloat16, SharedExpertsOrder.MULTI_STREAM_OVERLAPPED, False),
    ],
)
def test_shared_expert_allreduce_overlap_window(
    monkeypatch: pytest.MonkeyPatch,
    num_tokens: int,
    dtype: torch.dtype,
    order: SharedExpertsOrder,
    expected: bool,
) -> None:
    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256")
    runner = object.__new__(LatentMoERunner)
    runner.__dict__["_shared_expert_ar_max_tokens"] = 512
    runner.__dict__["_shared_experts"] = _SharedExpertsOrderStub(order)
    runner.__dict__["moe_config"] = SimpleNamespace(
        tp_size=8, dp_size=1, ep_size=1, pcp_size=1
    )
    runner.__dict__["enable_k3_latent_moe_tail_fusion"] = False
    hidden_states = torch.empty(num_tokens, 1, dtype=dtype)
    assert runner._use_shared_expert_allreduce_overlap(hidden_states) is expected


def test_shared_expert_allreduce_overlap_runner_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline: list[str] = []

    class SharedExpertsStub(_SharedExpertsOrderStub):
        output: torch.Tensor

        def __call__(
            self, hidden_states: torch.Tensor, order: SharedExpertsOrder
        ) -> None:
            assert order is SharedExpertsOrder.NO_OVERLAP
            timeline.append("shared")
            self.output = hidden_states + 1

    class RoutedExpertsStub:
        def forward_monolithic(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            timeline.append("routed")
            return x * 3

    class TPGroupStub:
        def all_reduce_shared_expert_pynccl(
            self, tensor: torch.Tensor
        ) -> torch.Tensor:
            timeline.append("shared_ar")
            return tensor.mul_(2)

    def execute_aux_first(default_fn, aux_fns, *args, **kwargs):
        aux_results = [fn() for fn in aux_fns]
        return default_fn(), aux_results

    def primary_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        timeline.append("latent_ar")
        return tensor * 2

    monkeypatch.setenv("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256")
    monkeypatch.setattr(latent_moe_runner, "get_tp_group", lambda: TPGroupStub())
    monkeypatch.setattr(latent_moe_runner, "aux_stream", object)
    monkeypatch.setattr(
        latent_moe_runner, "execute_in_parallel", execute_aux_first
    )
    monkeypatch.setattr(
        latent_moe_runner,
        "tensor_model_parallel_all_reduce",
        primary_all_reduce,
    )

    runner = object.__new__(LatentMoERunner)
    runner.__dict__.update(
        _shared_expert_ar_max_tokens=512,
        _shared_experts=SharedExpertsStub(SharedExpertsOrder.NO_OVERLAP),
        _shared_ar_events=(object(), object()),
        routed_experts=RoutedExpertsStub(),
        routed_output_transform=SimpleNamespace(
            norm=None,
            up_proj=SimpleNamespace(weight=torch.eye(2, dtype=torch.bfloat16)),
        ),
        moe_config=SimpleNamespace(
            tp_size=8,
            dp_size=1,
            ep_size=1,
            pcp_size=1,
            skip_final_all_reduce=False,
            is_sequence_parallel=False,
        ),
        enable_k3_latent_moe_tail_fusion=False,
        _fused_output_is_reduced=False,
    )
    hidden_states = torch.ones(257, 2, dtype=torch.bfloat16)
    shared_input = torch.full_like(hidden_states, 2)

    shared_output, fused_output = runner._apply_quant_method(
        hidden_states,
        torch.empty(257, 1, dtype=torch.bfloat16),
        shared_input,
    )
    actual = runner._shared_allreduce_tail(fused_output, shared_output, None)

    expected = hidden_states * 6 + (shared_input + 1) * 2
    torch.testing.assert_close(actual, expected)
    assert timeline == ["shared", "shared_ar", "routed", "latent_ar"]


@ray.remote(num_gpus=1, max_calls=1)
def _test_latent_moe_tail_worker(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    pp_size: int,
    rank: int,
    distributed_init_port: str,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(
        tp_size,
        pp_size,
        rank,
        distributed_init_port,
    )

    torch.manual_seed(0)
    rms_weight = 1 + 0.1 * torch.randn(
        LATENT_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    up_weight = (
        torch.randn(
            HIDDEN_SIZE,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        / LATENT_SIZE**0.5
    )

    group = get_tp_group().device_group
    op = KimiK3LatentMoETailOp.initialize(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        dtype=torch.bfloat16,
        device=device,
        rms_eps=EPS,
    )
    cutedsl_warmup()

    for iteration, num_tokens in enumerate((1, 5, 8, 16, 5)):
        torch.manual_seed(100 * iteration + rank + 1)
        routed_output = torch.randn(
            num_tokens,
            LATENT_SIZE,
            device=device,
            dtype=torch.bfloat16,
        ).mul_(0.01)
        shared_output = torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )

        routed_reference = routed_output.clone()
        shared_reference = shared_output.clone()
        dist.all_reduce(routed_reference, group=group)
        dist.all_reduce(shared_reference, group=group)
        expected = F.linear(
            F.rms_norm(
                routed_reference,
                (LATENT_SIZE,),
                rms_weight,
                EPS,
            ),
            up_weight,
        )
        expected.add_(shared_reference)

        actual = op(
            routed_output,
            shared_output,
            rms_weight,
            up_weight,
        )
        torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
        assert actual.is_contiguous()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(
            routed_output,
            shared_output,
            rms_weight,
            up_weight,
        )
    graph.replay()
    torch.testing.assert_close(graph_output, expected, atol=8e-2, rtol=3e-2)


def _run_latent_moe_tail_test(
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
) -> None:
    if not current_platform.is_device_capability_family(100):
        pytest.skip("K3 latent-MoE tail fusion requires SM100")
    multi_process_parallel(
        monkeypatch,
        tp_size,
        1,
        _test_latent_moe_tail_worker,
    )


@multi_gpu_test(num_gpus=8)
def test_latent_moe_tail_tp8_matches_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_latent_moe_tail_test(monkeypatch, 8)


@multi_gpu_test(num_gpus=16)
def test_latent_moe_tail_tp16_matches_native_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_latent_moe_tail_test(monkeypatch, 16)
