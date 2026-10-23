# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP O-Proj communication policy over an ordinary row-parallel linear."""

import enum
from typing import TypedDict, cast

import torch
from torch.nn.parameter import Parameter

from vllm.config import VllmConfig
from vllm.distributed import (
    divide,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import LinearMethodBase, RowParallelLinear
from vllm.model_executor.layers.weight_gather import (
    GatheredWeights,
)


class _LinearShardKwargs(TypedDict, total=False):
    tp_rank: int
    tp_size: int


class PCPOProjPath(enum.Enum):
    WEIGHT_ALL_GATHER = "weight_all_gather"
    OUTPUT_REDUCE = "output_reduce"


class PCPOProjTP(torch.nn.Module):
    """Own weight prefetch and PCP reduction while preserving checkpoint names.

    The attention module registers the linear as ``o_proj`` and this controller
    as ``o_proj_tp``. Gathered weights reference the linear without registering
    it again, preserving its parameter names in the model's state dict.
    """

    @staticmethod
    def linear_shard_kwargs(vllm_config: VllmConfig) -> _LinearShardKwargs:
        if not vllm_config.parallel_config.enable_pcp_o_proj_tp:
            return {}
        group = get_pcp_group()
        return {
            "tp_rank": get_tensor_model_parallel_rank() * group.world_size
            + group.rank_in_group,
            "tp_size": get_tensor_model_parallel_world_size() * group.world_size,
        }

    @classmethod
    def maybe_create(
        cls, linear: RowParallelLinear, vllm_config: VllmConfig
    ) -> "PCPOProjTP | None":
        return cls(linear) if vllm_config.parallel_config.enable_pcp_o_proj_tp else None

    def __init__(self, linear: RowParallelLinear) -> None:
        super().__init__()
        if linear.bias is not None or not linear.input_is_parallel:
            raise ValueError(
                "PCP O-Proj TP requires bias=False and input_is_parallel=True."
            )
        self.pcp_group = get_pcp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.pcp_group.world_size <= 1:
            raise ValueError("PCP O-Proj TP requires prefill context parallelism.")
        if linear.tp_size != self.tp_size * self.pcp_group.world_size:
            raise ValueError("Build the linear with PCPOProjTP.linear_shard_kwargs().")
        self.tp_input_size = divide(linear.input_size, self.tp_size)
        quant_method = linear.quant_method
        assert isinstance(quant_method, LinearMethodBase)
        self.weights = GatheredWeights(
            linear, quant_method.get_weight_gather_specs, self.pcp_group.world_size
        )
        self._path: PCPOProjPath | None = None

    @property
    def linear(self) -> RowParallelLinear:
        return cast(RowParallelLinear, self.weights.layer)

    @staticmethod
    def select_path(has_prefill: bool) -> PCPOProjPath:
        return (
            PCPOProjPath.WEIGHT_ALL_GATHER
            if has_prefill
            else PCPOProjPath.OUTPUT_REDUCE
        )

    def pre_attention_hook(self) -> None:
        """Launch weight prefetch using the runner's PCP-global batch decision."""
        if self._path is not None:
            raise RuntimeError("The previous O-Proj prefetch has not been consumed.")
        context = get_forward_context()
        has_prefill = context.global_has_prefill
        if has_prefill is None:
            if context.attn_metadata is not None:
                raise RuntimeError("PCP O-Proj requires a global prefill decision.")
            has_prefill = False  # Metadata-free profiling runs replicated inputs.
        assert self.weights.initialized, "Initialize O-Proj after loading weights."
        path = self.select_path(has_prefill)
        if path is PCPOProjPath.WEIGHT_ALL_GATHER:
            self.weights.gather(self.pcp_group.device_group)
        self._path = path

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        path, self._path = self._path, None
        if path is None:
            raise RuntimeError("Attention must call PCPOProjTP.pre_attention_hook().")
        assert self.weights.initialized
        if input_.shape[-1] != self.tp_input_size:
            if path is PCPOProjPath.WEIGHT_ALL_GATHER:
                self.weights.wait()
            raise ValueError(f"PCP O-Proj expected input width {self.tp_input_size}.")

        linear = self.linear
        if path is PCPOProjPath.WEIGHT_ALL_GATHER:
            with self.weights.full_weight():
                local_size = linear.input_size_per_partition
                linear.input_size_per_partition = self.tp_input_size
                try:
                    output = linear.quant_method.apply(linear, input_, None)
                finally:
                    linear.input_size_per_partition = local_size
        else:
            size = linear.input_size_per_partition
            local_input = input_.narrow(
                -1, self.pcp_group.rank_in_group * size, size
            ).contiguous()
            output = linear.quant_method.apply(linear, local_input, None)
            output = self.pcp_group.all_reduce(output)

        if linear.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        return (output, None) if linear.return_bias else output
