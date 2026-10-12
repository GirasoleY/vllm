# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HYV4 NVFP4 routed experts with BF16 or block-FP8 shared experts."""

import torch
import torch.distributed as dist
from torch import nn

from vllm import envs
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import prepare_megamoe_inputs
from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_nvfp4_input(X, GlobalScale, Q, SF, H: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1) * 8 + tl.arange(0, 8)
    pair = tl.arange(0, 8)
    offset = block[:, None] * 16 + pair[None, :] * 2
    x0 = tl.load(X + row * H + offset, offset < H, 0).to(tl.float32)
    x1 = tl.load(X + row * H + offset + 1, offset < H, 0).to(tl.float32)
    scale = tl.load(GlobalScale)
    amax = tl.max(tl.maximum(tl.abs(x0), tl.abs(x1)), 1)
    sf = tl.minimum(448.0, tl.maximum(2.0**-9, tl.div_rn(amax, 6.0 * scale)))
    sf = sf.to(tl.float8e4nv).to(tl.float32)
    denom = sf[:, None] * scale
    x0 = tl.div_rn(x0, denom)
    x1 = tl.div_rn(x1, denom)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
        " cvt.u32.u8 $0, tmp; }",
        constraints="=r,f,f",
        args=[x0, x1],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(Q + row * (H // 2) + offset // 2, packed.to(tl.uint8), offset < H)
    tl.store(SF + row * (H // 16) + block, sf, block < H // 16)


class HYV4NVFP4MegaMoEExperts(DeepseekV4MegaMoEExperts):
    # BF16 and FP8 shared experts have different symmetric-buffer layouts.
    _nvfp4_buffer_cache: dict[tuple, object] = {}

    def __init__(self, vllm_config, **kwargs):
        super().__init__(vllm_config, **kwargs)
        self.mma_type = "nvfp4"
        self.weight_block_size = (1, 16)
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        for name, shape, dtype in (
            ("w13_weight_scale", (e, 2 * i, h // 16), torch.float8_e4m3fn),
            ("w2_weight_scale", (e, h, i // 16), torch.float8_e4m3fn),
            ("w13_weight_scale_2", (e, 2), torch.float32),
            ("w2_weight_scale_2", (e,), torch.float32),
            ("w13_input_scale", (e, 2), torch.float32),
            ("w2_input_scale", (e,), torch.float32),
        ):
            param = nn.Parameter(
                torch.full(shape, float("nan"), dtype=dtype), requires_grad=False
            )
            set_weight_attrs(param, {"weight_loader": self.weight_loader})
            setattr(self, name, param)
        self.shared_dtype = torch.float8_e4m3fn
        self._nv_shared_l1 = None
        self._nv_shared_l2 = None

    def weight_loader(
        self,
        param,
        loaded_weight,
        weight_name,
        shard_id,
        expert_id,
        return_success=False,
    ):
        if not weight_name.endswith(("weight_scale_2", "input_scale")):
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        local_ids = self._map_global_expert_id(expert_id)
        if loaded_weight.numel() != 1:
            raise ValueError(f"NVFP4 global scale must be scalar: {weight_name}")
        for local_id in local_ids:
            if shard_id in ("w1", "w3") and "w13_" in weight_name:
                dst = param.data[local_id, 0 if shard_id == "w1" else 1]
            elif shard_id == "w2" and "w2_" in weight_name:
                dst = param.data[local_id]
            else:
                raise ValueError(f"Invalid NVFP4 scale shard: {weight_name}/{shard_id}")
            dst.copy_(loaded_weight.reshape(()))
        return bool(local_ids) if return_success else None

    def finalize_weights(self, shared_experts=None):
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        if not hasattr(dg, "nvfp4_mega_moe"):
            raise RuntimeError("Rebuild DeepGEMM with NVFP4 MegaMoE support")
        if self._transformed_l1_weights is None:
            self._check_runtime_supported()
            for name in (
                "w13_input_scale",
                "w2_input_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
            ):
                scale = getattr(self, name).data
                if not torch.isfinite(scale).all() or not (scale > 0).all():
                    raise ValueError(
                        f"Missing or invalid NVFP4 scale: {self.prefix}.{name}"
                    )
            if not torch.equal(
                self.w13_weight_scale_2[:, 0], self.w13_weight_scale_2[:, 1]
            ):
                raise ValueError(
                    "NVFP4 MegaMoE requires equal gate/up global weight scales"
                )
            # Every EP rank must quantize with the same tensor-wide activation
            # scales; weight global scales remain per expert.
            activation_scales = torch.stack(
                (self.w13_input_scale.max(), self.w2_input_scale.max())
            )
            dist.all_reduce(
                activation_scales,
                op=dist.ReduceOp.MAX,
                group=get_ep_group().device_group,
            )
            self.input_global_scale = activation_scales[:1].contiguous()
            self.l2_activation_scale = activation_scales[1].item()
            self.l1_alpha = (
                self.w13_weight_scale_2[:, 0] * activation_scales[0]
            ).contiguous()
            self.l2_alpha = (self.w2_weight_scale_2 * activation_scales[1]).contiguous()

            def pack_scale(scale):
                values = scale.float()
                if not torch.isfinite(values).all() or (values < 0).any():
                    raise ValueError(
                        f"Missing or invalid NVFP4 block scales: {self.prefix}"
                    )
                # Four E4M3 bytes per int32, with MN-major strides for TMA.
                packed = scale.contiguous().view(torch.int32)
                return packed.transpose(-1, -2).contiguous().transpose(-1, -2)

            self._transformed_l1_weights, self._transformed_l2_weights = (
                dg.transform_weights_for_mega_moe(
                    (
                        self.w13_weight.data.view(torch.int8),
                        pack_scale(self.w13_weight_scale.data),
                    ),
                    (
                        self.w2_weight.data.view(torch.int8),
                        pack_scale(self.w2_weight_scale.data),
                    ),
                )
            )
            for name in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            ):
                setattr(self, name, None)

        if (
            shared_experts is None
            or not self.num_shared_experts
            or self.has_fused_shared_experts
        ):
            return
        gate_up = shared_experts.gate_up_proj.weight.data
        down = shared_experts.down_proj.weight.data
        if gate_up.dtype == down.dtype == torch.bfloat16:
            i = self.intermediate_size * self.num_shared_experts
            if gate_up.shape != (2 * i, self.hidden_size) or down.shape != (
                self.hidden_size,
                i,
            ):
                raise ValueError("NVFP4 MegaMoE needs replicated shared-expert weights")
            self.shared_dtype = torch.bfloat16
            self._nv_shared_l1, self._nv_shared_l2 = dg.transform_weights_for_mega_moe(
                gate_up, down
            )
        elif gate_up.dtype == down.dtype == torch.float8_e4m3fn:
            self._finalize_shared_expert_weights(dg, shared_experts)
            self._nv_shared_l1 = self._transformed_shared_l1_weights
            self._nv_shared_l2 = self._transformed_shared_l2_weights
        else:
            raise ValueError(
                f"Unsupported NVFP4 shared expert dtypes: {gate_up.dtype}, {down.dtype}"
            )

    @property
    def has_fused_shared_experts(self):
        return self._nv_shared_l1 is not None

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        group = get_ep_group().device_group
        shared = self.num_shared_experts if self.has_fused_shared_experts else 0
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            shared,
            self.shared_dtype,
        )
        if key not in self._nvfp4_buffer_cache:
            dg = _import_deep_gemm()
            buffer = dg.NVFP4SymmBuffer(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                shared,
                shared_dtype=self.shared_dtype,
            )
            if shared and self.shared_dtype == torch.float8_e4m3fn:
                buffer.shared_sf_scratch = torch.empty(
                    (buffer.num_max_tokens_per_rank, self.hidden_size // 128),
                    dtype=torch.int32,
                    device=buffer.x.device,
                )
            self._nvfp4_buffer_cache[key] = buffer
        return self._nvfp4_buffer_cache[key]

    def forward(
        self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True
    ):
        from vllm.utils.deep_gemm import _import_deep_gemm

        dg = _import_deep_gemm()
        m = hidden_states.shape[0]
        if m > self.max_num_tokens:
            raise ValueError(
                f"NVFP4 MegaMoE capacity exceeded: {m} > {self.max_num_tokens}"
            )
        buffer = self.get_symm_buffer()
        padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            padding = get_forward_context().is_padding
            if padding is not None:
                padding = padding[:m]
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        if m:
            # Exact division preserves ties-to-even at E2M1 boundaries.
            _quantize_nvfp4_input[(m, triton.cdiv(self.hidden_size, 128))](
                hidden_states.contiguous(),
                self.input_global_scale,
                buffer.x[:m].view(torch.uint8),
                buffer.x_sf[:m].view(torch.float8_e4m3fn),
                self.hidden_size,
                num_warps=4,
            )
            if (
                self.has_fused_shared_experts
                and self.shared_dtype == torch.float8_e4m3fn
            ):
                block_m = dg.get_block_m_for_nvfp4_mega_moe(
                    get_ep_group().world_size,
                    self.num_experts,
                    buffer.num_max_tokens_per_rank,
                    m,
                    self.top_k,
                )
                prepare_megamoe_inputs(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    buffer.shared_l1_acts[:m],
                    buffer.shared_sf_scratch[:m],
                    buffer.topk_idx[:m],
                    buffer.topk_weights[:m],
                    is_padding=padding,
                    shared_x_sf=buffer.shared_l1_acts_sf,
                    shared_block_m=block_m,
                )
            else:
                buffer.topk_idx[:m].copy_(topk_ids)
                buffer.topk_weights[:m].copy_(topk_weights)
                if padding is not None:
                    buffer.topk_idx[:m].masked_fill_(padding[:, None], -1)
                    buffer.topk_weights[:m].masked_fill_(padding[:, None], 0)
                if self.has_fused_shared_experts:
                    buffer.shared_l1_acts[:m].copy_(hidden_states)
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        dg.nvfp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            buffer,
            shared_l1_weights=self._nv_shared_l1,
            shared_l2_weights=self._nv_shared_l2,
            activation_clamp=activation_clamp,
            shared_activation_clamp=float("inf"),
            activation_alpha=1.0,
            activation_beta=0.0,
            fast_math=fast_math,
            l1_alpha=self.l1_alpha,
            l2_alpha=self.l2_alpha,
            l2_activation_scale=self.l2_activation_scale,
        )
        return y


HYV4NVFP4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
