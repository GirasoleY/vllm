# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise weight reconstruction with real tensors and Gloo collectives."""

from collections.abc import Callable
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from tests.kernels.quantization.nvfp4_utils import convert_swizzled_to_linear
from vllm.config import (
    DeviceConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.quantization import QuantSpec
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.pcp_o_proj_tp import PCPOProjTP
from vllm.model_executor.layers.quantization.modelopt import (
    CkptCtx,
    ModelOptLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.weight_gather import (
    GatheredWeights,
    WeightGatherSpec,
    WeightGatherWorkspace,
)
from vllm.platforms import current_platform


def _check_dense(rank: int, contiguous: bool) -> None:
    buffers = WeightGatherWorkspace()
    weight = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    x = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    # Sequential layers share the full-weight workspace, but retain independent
    # resident parameters. Updating a resident parameter must refresh staging.
    layers = [torch.nn.Linear(4, 3, bias=False) for _ in range(2)]
    specs = (WeightGatherSpec("weight", gather_dim=1, make_full_contiguous=contiguous),)
    switches = [GatheredWeights(layer, lambda _: specs, 2) for layer in layers]
    for switch in switches:
        switch.initialize(buffers)
    assert switches[0].parts is not None and switches[1].parts is not None
    assert (
        switches[0].parts["weight"].full_tensor.data_ptr()
        == switches[1].parts["weight"].full_tensor.data_ptr()
    )
    for index, (layer, switch) in enumerate(zip(layers, switches)):
        local = (weight + index)[:, rank * 4 : (rank + 1) * 4]
        with torch.no_grad():
            layer.weight.copy_(local)
        switch.gather(dist.group.WORLD)
        with switch.full_weight():
            torch.testing.assert_close(
                layer(x), F.linear(x, weight + index), rtol=0, atol=0
            )
            if contiguous:
                assert layer.weight.is_contiguous()
        torch.testing.assert_close(layer.weight, local, rtol=0, atol=0)

    layer, switch = layers[0], switches[0]
    with torch.no_grad():
        layer.weight.add_(2)
    local = layer.weight.clone()
    switch.gather(dist.group.WORLD)
    with (
        pytest.raises(RuntimeError, match="consumer failed"),
        switch.full_weight(),
    ):
        torch.testing.assert_close(layer(x), F.linear(x, weight + 2), rtol=0, atol=0)
        raise RuntimeError("consumer failed")
    torch.testing.assert_close(layer.weight, local, rtol=0, atol=0)


def _check_nvfp4(rank: int, output_size: int, scale_cols: int) -> None:
    # Use the existing test oracle to decode the gathered swizzled scales.
    padded_n = (output_size + 127) // 128 * 128
    padded_k = (scale_cols + 3) // 4 * 4
    local = (torch.arange(padded_n * padded_k) % 64 + rank * 64).to(torch.uint8)
    layer = torch.nn.Module()
    layer.input_size_per_partition = scale_cols * 16
    layer.output_size_per_partition = output_size
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            local.view(torch.float8_e4m3fn).reshape(padded_n, padded_k),
            requires_grad=False,
        ),
    )
    before = layer.weight_scale.view(torch.uint8).clone()
    logical = convert_swizzled_to_linear(
        layer.weight_scale, output_size, scale_cols * 16, 16
    )[:, :scale_cols]
    expected = [torch.empty_like(logical.view(torch.uint8)) for _ in range(2)]
    dist.all_gather(expected, logical.contiguous().view(torch.uint8))
    switch = GatheredWeights(
        layer,
        lambda _: (ModelOptLinearMethod.weight_gather_specs[1],),
        2,
    )
    switch.initialize(WeightGatherWorkspace())
    switch.gather(dist.group.WORLD)
    with switch.full_weight():
        full = convert_swizzled_to_linear(
            layer.weight_scale, output_size, scale_cols * 32, 16
        )[:, : scale_cols * 2]
        assert torch.equal(full.view(torch.uint8), torch.cat(expected, dim=1))
    assert torch.equal(layer.weight_scale.view(torch.uint8), before)


def _gather_worker(
    rank: int, init_file: str, check: Callable[..., None], args: tuple
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        check(rank, *args)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def _run_distributed(tmp_path, check: Callable[..., None], *args) -> None:
    mp.spawn(_gather_worker, args=(str(tmp_path / "gloo"), check, args), nprocs=2)


def _check_controller(rank: int) -> None:
    device = torch.device(f"cuda:{rank}" if current_platform.is_cuda_alike() else "cpu")
    if device.type == "cuda":
        torch.accelerator.set_device_index(rank)
    config = VllmConfig(
        device_config=DeviceConfig(device=device),
        parallel_config=ParallelConfig(
            prefill_context_parallel_size=2,
            distributed_executor_backend="mp",
            disable_custom_all_reduce=True,
        ),
    )
    with set_current_vllm_config(config):
        backend = "nccl" if device.type == "cuda" else "gloo"
        init_distributed_environment(
            world_size=2, rank=rank, local_rank=rank, backend=backend
        )
        initialize_model_parallel(
            prefill_context_model_parallel_size=2, backend=backend
        )
        owner = torch.nn.Module()
        with device:
            owner.o_proj = RowParallelLinear(
                8,
                3,
                bias=False,
                params_dtype=torch.float32,
                tp_rank=rank,
                tp_size=2,
                reduce_results=False,
            )
        owner.o_proj_tp = PCPOProjTP(owner.o_proj)
        GatheredWeights.initialize_model(owner)
        weight = torch.arange(24, dtype=torch.float32, device=device).reshape(3, 8)
        owner.o_proj.weight_loader_v2(owner.o_proj.weight, weight)
        assert list(owner.state_dict()) == ["o_proj.weight"]
        x = torch.arange(16, dtype=torch.float32, device=device).reshape(2, 8)
        for has_prefill in (True, False, True):
            with set_forward_context(None, config, global_has_prefill=has_prefill):
                owner.o_proj_tp.pre_attention_hook()
                out, _ = owner.o_proj_tp(x)
            torch.testing.assert_close(out, F.linear(x, weight), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="must call"):
            owner.o_proj_tp(x)
        with (
            set_forward_context({"attention": None}, config),
            pytest.raises(RuntimeError, match="global prefill decision"),
        ):
            owner.o_proj_tp.pre_attention_hook()


@pytest.mark.parametrize("contiguous", [False, True])
def test_dense_gather_shares_workspace_and_restores_local_weights(tmp_path, contiguous):
    _run_distributed(tmp_path, _check_dense, contiguous)


@pytest.mark.parametrize("output_size,scale_cols", [(256, 4), (131, 3), (128, 5)])
def test_nvfp4_gather_preserves_scale_bits_and_padding(
    tmp_path, output_size, scale_cols
):
    _run_distributed(tmp_path, _check_nvfp4, output_size, scale_cols)


def test_pcp_projection_transitions_preserve_output_and_checkpoint_names(tmp_path):
    _run_distributed(tmp_path, _check_controller)


@pytest.mark.parametrize("dim", [-3, 2])
def test_weight_gather_rejects_invalid_shard_axis(dim):
    with pytest.raises(ValueError, match="Invalid gather dimension"):
        layer = torch.nn.Linear(4, 3, bias=False)
        weights = GatheredWeights(
            layer, lambda _: (WeightGatherSpec("weight", gather_dim=dim),), 2
        )
        weights.initialize(WeightGatherWorkspace())


@pytest.mark.parametrize("activation,group_size", [(None, 16), (kNvfp4Dynamic, 32)])
def test_modelopt_weight_gather_rejects_unsupported_quant_spec(activation, group_size):
    method = ModelOptLinearMethod(
        QuantSpec(weight=kNvfp4Static, activation=activation),
        CkptCtx(group_size=group_size),
    )
    with pytest.raises(RuntimeError, match="only ModelOpt W4A4 NVFP4"):
        method.get_weight_gather_specs(torch.nn.Linear(4, 3, bias=False))
