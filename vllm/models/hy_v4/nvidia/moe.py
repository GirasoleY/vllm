# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense FFN and MoE blocks for HY V4 (NVIDIA)."""

import copy
from contextlib import AbstractContextManager, nullcontext

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    silu_mul_quant_fp8_packed_triton,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_reduce_scatter,
)
from vllm.models.hy_v4.nvidia.prefill import fp8_linear, packed_fp8_kernel

logger = init_logger(__name__)


def use_hyv4_sequence_parallel(vllm_config: VllmConfig) -> bool:
    """Whether TEP MegaMoE shards the residual stream across TP ranks.

    MegaMoE dispatches each token over the EP group, so TP ranks must own
    distinct tokens. Attention all-gathers its input and reduce-scatters its
    output instead of all-reducing, keeping one collective per direction.
    """
    parallel_config = vllm_config.parallel_config
    return (
        vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        and parallel_config.pipeline_parallel_size == 1
        and parallel_config.enable_expert_parallel
        and parallel_config.tensor_parallel_size > 1
    )


class HYV4FeedForward(nn.Module):
    """Dense SwiGLU feed-forward block.

    Used both for the dense decoder layers and for the MoE shared experts.
    The routed-expert SwiGLU clamp does not apply here, matching the reference
    implementation.

    Under sequence parallel each rank owns a token shard. ``replicate`` keeps
    full weights so the shard is finished locally; otherwise the weights stay
    TP-sharded and ``forward`` all-gathers the tokens, computes this rank's
    partial, and reduce-scatters back to the shard.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        use_sequence_parallel: bool = False,
        replicate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.shard_sequence_parallel = use_sequence_parallel and not replicate
        disable_tp = use_sequence_parallel and replicate
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=disable_tp,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results and not use_sequence_parallel,
            disable_tp=disable_tp,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shard_sequence_parallel:
            x = sp_all_gather(x)
        gate_up, _ = self.gate_up_proj(x)
        if packed_fp8_kernel(self.down_proj, gate_up.shape[0]) is not None:
            out = silu_mul_quant_fp8_packed_triton(gate_up)
        else:
            out = self.act_fn(gate_up)
        out = fp8_linear(self.down_proj, out)
        if self.shard_sequence_parallel:
            out = sp_reduce_scatter(out)
        return out


class HYV4MoEFused(nn.Module):
    """HY V4 MoE layer with optional SwiGLU clamp support.

    When ``config.swiglu_limit > 0`` the routed experts use a clamped SwiGLU::

        gate = clamp(gate, max=limit)
        up = clamp(up, -limit, limit)
        output = silu(gate) * up

    Dense layers and shared experts are NOT clamped. The clamp is forwarded to
    ``FusedMoEFactory`` as ``swiglu_limit`` so it travels through the quant
    config to every expert backend instead of mutating global state.

    NOTE: The reference implementation additionally supports HPC gate kernels,
    TPCP/MLPSP sharded shared experts and a side-stream overlap for the shared
    experts. Those depend on infrastructure that is absent here, so this port
    keeps only the standard path. TODO: restore once available.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        enable_eplb: bool = False,
        is_mtp: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        top_k = config.num_experts_per_tok
        intermediate_size = config.expert_hidden_dim
        router_scaling_factor = getattr(config, "router_scaling_factor", 1.0)
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = GateLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            prefix=f"{prefix}.gate",
        )

        requested_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        self.use_mega_moe = requested_mega_moe and not is_mtp
        self.use_sequence_parallel = self.use_mega_moe and use_hyv4_sequence_parallel(
            vllm_config
        )

        self.shared_experts: HYV4FeedForward | None
        if config.num_shared_experts > 0:
            self.shared_experts = HYV4FeedForward(
                hidden_size=config.hidden_size,
                intermediate_size=config.expert_hidden_dim * config.num_shared_experts,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
                reduce_results=False,
                use_sequence_parallel=self.use_sequence_parallel,
                replicate=True,
            )
        else:
            self.shared_experts = None

        self.expert_bias = nn.Parameter(
            torch.empty(config.num_experts, dtype=torch.float32)
        )

        raw_swiglu_limit = getattr(config, "swiglu_limit", 0)
        use_swiglu_clamp = bool(raw_swiglu_limit) and float(raw_swiglu_limit) > 0
        moe_swiglu_limit = float(raw_swiglu_limit) if use_swiglu_clamp else None

        self.moe_swiglu_limit = moe_swiglu_limit
        self.prefix = prefix
        if self.use_mega_moe:
            if getattr(quant_config, "store_dtype", None) != "mxfp4":
                raise ValueError("HYV4 MegaMoE requires MXFP4 routed-expert weights.")
            parallel_config = vllm_config.parallel_config
            if not parallel_config.enable_expert_parallel:
                raise ValueError("HYV4 MegaMoE requires --enable-expert-parallel.")
            if self.tp_size != 1 and not self.use_sequence_parallel:
                raise ValueError(
                    "HYV4 MegaMoE with TP>1 requires sequence parallel (PP=1)."
                )
            if (
                self.enable_eplb
                or parallel_config.enable_eplb
                or self.n_redundant_experts
            ):
                raise ValueError("HYV4 MegaMoE EPLB is not implemented yet.")
            if self.n_physical_experts % self.ep_size:
                raise ValueError("HYV4 MegaMoE requires evenly sharded experts.")
            from vllm.model_executor.layers.fused_moe.router.router_factory import (
                create_fused_moe_router,
            )
            from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts

            # Shared experts hold full weights on every rank (replicated under
            # sequence parallel), so they can run inside the MegaMoE kernel.
            fuse_shared_experts = self.shared_experts is not None and (
                self.use_sequence_parallel or self.tp_size == 1
            )
            self.experts = DeepseekV4MegaMoEExperts(
                vllm_config,
                num_experts=self.n_physical_experts,
                num_local_experts=self.n_local_physical_experts,
                experts_start_idx=self.physical_expert_start,
                top_k=top_k,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                num_shared_experts=(
                    config.num_shared_experts if fuse_shared_experts else 0
                ),
                clamp_shared_experts=False,
                prefix=f"{prefix}.experts",
            )
            self.mega_router = create_fused_moe_router(
                top_k=top_k,
                global_num_experts=self.n_routed_experts,
                renormalize=config.route_norm,
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                scoring_func="sigmoid",
                routed_scaling_factor=router_scaling_factor,
                e_score_correction_bias=self.expert_bias,
            )
            logger.info_once("HYV4 routed experts use DeepGEMM MegaMoE.")
            return

        # The FP8 draft must not inherit the target's MXFP4-only model backend.
        factory_context: AbstractContextManager = nullcontext()
        if requested_mega_moe:
            draft_config = copy.copy(vllm_config)
            draft_config.kernel_config = copy.copy(vllm_config.kernel_config)
            draft_config.kernel_config.moe_backend = "auto"
            factory_context = set_current_vllm_config(draft_config)

        with factory_context:
            self.experts = FusedMoEFactory(
                num_experts=self.n_routed_experts,
                top_k=top_k,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                renormalize=config.route_norm,
                quant_config=quant_config,
                prefix=f"{prefix}.experts",
                enable_eplb=self.enable_eplb,
                num_redundant_experts=self.n_redundant_experts,
                scoring_func="sigmoid",
                use_grouped_topk=True,
                num_expert_group=1,
                topk_group=1,
                routed_scaling_factor=router_scaling_factor,
                e_score_correction_bias=self.expert_bias,
                shared_experts=self.shared_experts,
                swiglu_limit=moe_swiglu_limit,
            )
        if use_swiglu_clamp:
            logger.debug_once(
                "HYV4MoEFused: swiglu_limit=%.1f enabled for routed experts",
                moe_swiglu_limit,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # router_logits: (num_tokens, n_experts); the gate stores fp32 weights
        # and emits fp32 logits, so no explicit upcast is needed here.
        router_logits, _ = self.gate(hidden_states)

        if self.use_mega_moe:
            topk_weights, topk_ids = self.mega_router.select_experts(
                hidden_states, router_logits, topk_indices_dtype=torch.int64
            )
            final_hidden_states = self.experts(
                hidden_states,
                topk_weights,
                topk_ids,
                activation_clamp=self.moe_swiglu_limit,
            )
            if (
                self.shared_experts is not None
                and not self.experts.has_fused_shared_experts
            ):
                final_hidden_states = final_hidden_states + self.shared_experts(
                    hidden_states
                )
        else:
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )
        return final_hidden_states.view(orig_shape)
