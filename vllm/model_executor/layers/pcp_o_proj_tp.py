# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP O-Proj communication policy over an ordinary row-parallel linear."""

import enum
import weakref
from typing import TypedDict

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
from vllm.model_executor.layers.quantization.tp_weight_switch import (
    TPWeightBufferPool,
    TPWeightSwitch,
)


class _LinearShardKwargs(TypedDict, total=False):
    tp_rank: int
    tp_size: int


class PCPOProjPath(enum.Enum):
    WEIGHT_ALL_GATHER = "weight_all_gather"
    TOKEN_EXCHANGE = "token_exchange"


class PCPOProjTP(torch.nn.Module):
    """Own weight prefetch and PCP reduction while preserving checkpoint names.

    The attention module registers the linear as ``o_proj`` and this controller
    as ``o_proj_tp``. The controller holds a weak reference to that linear, so
    it adds no parameter aliases to the model's state dict.
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
        self._linear = weakref.ref(linear)
        self.pcp_group = get_pcp_group()
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.pcp_group.world_size <= 1:
            raise ValueError("PCP O-Proj TP requires prefill context parallelism.")
        if linear.tp_size != self.tp_size * self.pcp_group.world_size:
            raise ValueError("Build the linear with PCPOProjTP.linear_shard_kwargs().")
        self.tp_input_size = divide(linear.input_size, self.tp_size)
        self._weights: TPWeightSwitch | None = None
        self._path: PCPOProjPath | None = None

    @property
    def linear(self) -> RowParallelLinear:
        linear = self._linear()
        assert linear is not None, "The attention module must own its O-Proj linear."
        return linear

    @staticmethod
    def process_weights_after_loading(model: torch.nn.Module) -> None:
        """Initialize all controllers with a workspace owned by this model."""
        buffers = TPWeightBufferPool()
        for module in model.modules():
            if isinstance(module, PCPOProjTP):
                module.enable_weight_switch(buffers)

    def enable_weight_switch(self, buffers: TPWeightBufferPool) -> None:
        if self._path is not None:
            raise RuntimeError("Cannot rebuild O-Proj weights during a forward.")
        linear = self.linear
        quant_method = linear.quant_method
        assert isinstance(quant_method, LinearMethodBase)
        self._weights = TPWeightSwitch(
            linear,
            quant_method.get_tp_weight_switch_specs(linear),
            self.pcp_group.world_size,
            buffers,
        )

    @staticmethod
    def select_path(has_prefill: bool) -> PCPOProjPath:
        return (
            PCPOProjPath.WEIGHT_ALL_GATHER
            if has_prefill
            else PCPOProjPath.TOKEN_EXCHANGE
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
        assert self._weights is not None, "Initialize O-Proj after loading weights."
        path = self.select_path(has_prefill)
        if path is PCPOProjPath.WEIGHT_ALL_GATHER:
            self._weights.gather(self.pcp_group.device_group)
        self._path = path

    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        path, self._path = self._path, None
        if path is None:
            raise RuntimeError("Attention must call PCPOProjTP.pre_attention_hook().")
        assert self._weights is not None
        if input_.shape[-1] != self.tp_input_size:
            if path is PCPOProjPath.WEIGHT_ALL_GATHER:
                self._weights.wait()
            raise ValueError(f"PCP O-Proj expected input width {self.tp_input_size}.")

        linear = self.linear
        if path is PCPOProjPath.WEIGHT_ALL_GATHER:
            with self._weights.full_weight(linear):
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
