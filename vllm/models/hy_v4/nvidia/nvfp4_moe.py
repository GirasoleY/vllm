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
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _prepare_nvfp4_megamoe_inputs_kernel(
    hidden_states,
    global_scale,
    x,
    x_sf,
    shared_x,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    shared_x_stride_m: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    num_tokens,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
) -> None:
    BLOCK_K: tl.constexpr = 128
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_id < num_tokens
    k_block_id = tl.program_id(1)
    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    hidden_bf16 = tl.load(
        hidden_states + token_id[:, None] * hidden_stride_m + k_offsets[None, :],
        mask=token_mask[:, None],
        # div.rn.f32 takes a slow path for zero dividends.
        other=1.0,
    )
    hidden = hidden_bf16.to(tl.float32)

    # NVFP4: 16-element blocks with E4M3 scales. Exact division preserves
    # ties-to-even at E2M1 boundaries.
    scale = tl.load(global_scale)
    x0, x1 = tl.split(tl.reshape(hidden, [BLOCK_M, BLOCK_K // 2, 2]))
    pair_amax = tl.maximum(tl.abs(x0), tl.abs(x1))
    amax = tl.max(tl.reshape(pair_amax, [BLOCK_M, BLOCK_K // 16, 8]), axis=2)
    sf = tl.minimum(448.0, tl.maximum(2.0**-9, tl.div_rn(amax, 6.0 * scale)))
    sf = sf.to(tl.float8e4nv).to(tl.float32)
    denom = (sf * scale)[:, :, None]
    x0 = tl.div_rn(tl.reshape(x0, [BLOCK_M, BLOCK_K // 16, 8]), denom)
    x1 = tl.div_rn(tl.reshape(x1, [BLOCK_M, BLOCK_K // 16, 8]), denom)
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, $2, $1;"
        " cvt.u32.u8 $0, tmp; }",
        constraints="=r,f,f",
        args=[x0, x1],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    packed = tl.reshape(packed, [BLOCK_M, BLOCK_K // 2]).to(tl.uint8)
    q_offsets = k_block_id * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)
    tl.store(
        x + token_id[:, None] * x_stride_m + q_offsets[None, :],
        packed,
        mask=token_mask[:, None],
    )
    sf_offsets = k_block_id * (BLOCK_K // 16) + tl.arange(0, BLOCK_K // 16)
    tl.store(
        x_sf + token_id[:, None] * x_sf_stride_m + sf_offsets[None, :],
        sf.to(tl.float8e4nv),
        mask=token_mask[:, None],
    )

    if shared_x_sf is not None:
        # Block-FP8 shared expert: 32-element groups with UE8M0 scales, packed
        # four per int32 in DeepGEMM's MN-major, BLOCK_M-permuted layout.
        groups = tl.reshape(hidden, [BLOCK_M, BLOCK_K // 32, 32])
        group_amax = tl.maximum(tl.max(tl.abs(groups), axis=2), 1.0e-4)
        scale_bits = (group_amax / 448.0).to(tl.uint32, bitcast=True)
        scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
            tl.uint32
        )
        scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
        rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)
        fp8 = tl.reshape(groups * (1.0 / rounded_scale)[:, :, None], [BLOCK_M, BLOCK_K])
        tl.store(
            shared_x + token_id[:, None] * shared_x_stride_m + k_offsets[None, :],
            fp8.to(tl.float8e4nv),
            mask=token_mask[:, None],
        )
        packed_scale = tl.sum(
            scale_exp << (tl.arange(0, BLOCK_K // 32)[None, :] * 8), axis=1
        ).to(tl.int32)
        m_in_block = token_id % SHARED_BLOCK_M
        aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
        shared_row = (
            (token_id // SHARED_BLOCK_M) * aligned_block_m
            + (m_in_block // 128) * 128
            + (m_in_block % 32) * 4
            + (m_in_block % 128) // 32
        )
        tl.store(
            shared_x_sf
            + shared_row * shared_x_sf_stride_m
            + k_block_id * shared_x_sf_stride_k,
            packed_scale,
            mask=token_mask,
        )
    elif shared_x is not None:
        tl.store(
            shared_x + token_id[:, None] * shared_x_stride_m + k_offsets[None, :],
            hidden_bf16,
            mask=token_mask[:, None],
        )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = token_mask[:, None] & (topk_offsets[None, :] < top_k)
        token_is_padding = tl.full((BLOCK_M,), False, tl.int1)
        if is_padding is not None:
            token_is_padding = tl.load(
                is_padding + token_id * is_padding_stride_m,
                mask=token_mask,
                other=True,
            )
        ids = tl.load(
            topk_ids + token_id[:, None] * topk_ids_stride_m + topk_offsets[None, :],
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        ids = tl.where(token_is_padding[:, None], -1, ids)
        tl.store(
            topk_idx_out
            + token_id[:, None] * topk_idx_stride_m
            + topk_offsets[None, :],
            ids,
            mask=topk_mask,
        )
        weights = tl.load(
            topk_weights
            + token_id[:, None] * topk_weights_stride_m
            + topk_offsets[None, :],
            mask=topk_mask,
            other=0.0,
        )
        weights = tl.where(token_is_padding[:, None], 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id[:, None] * topk_weights_out_stride_m
            + topk_offsets[None, :],
            weights,
            mask=topk_mask,
        )


def prepare_nvfp4_megamoe_inputs(
    hidden_states: torch.Tensor,
    global_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: torch.Tensor | None = None,
    shared_x: torch.Tensor | None = None,
    shared_x_sf: torch.Tensor | None = None,
    shared_block_m: int | None = None,
) -> None:
    """Stage NVFP4 MegaMoE inputs in one pass over ``hidden_states``.

    Writes the NVFP4 routed-expert activations (``x``: packed E2M1 bytes,
    ``x_sf``: E4M3 scales), the shared-expert activations (``shared_x``:
    block-FP8 when ``shared_x_sf`` is given, else a BF16 copy) and the top-k
    routing with padded tokens masked to expert -1 / weight 0.
    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    assert hidden_size % 128 == 0
    assert hidden_states.stride(1) == 1
    assert topk_weights.shape == topk_ids.shape
    assert (shared_x_sf is None) == (shared_block_m is None)
    assert shared_x_sf is None or shared_x is not None
    top_k = topk_ids.shape[1]
    if num_tokens < 8:
        block_m, num_warps = 1, 1
    elif num_tokens < 128:
        block_m, num_warps = 4, 4
    else:
        block_m, num_warps = 8, 4 if num_tokens < 256 else 2
    _prepare_nvfp4_megamoe_inputs_kernel[
        (triton.cdiv(num_tokens, block_m), hidden_size // 128)
    ](
        hidden_states,
        global_scale,
        x,
        x_sf,
        shared_x,
        shared_x_sf,
        topk_ids,
        topk_weights,
        is_padding,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        x.stride(0),
        x_sf.stride(0),
        shared_x.stride(0) if shared_x is not None else 0,
        shared_x_sf.stride(0) if shared_x_sf is not None else 0,
        shared_x_sf.stride(1) if shared_x_sf is not None else 0,
        topk_ids.stride(0),
        topk_weights.stride(0),
        is_padding.stride(0) if is_padding is not None else 0,
        topk_idx_out.stride(0),
        topk_weights_out.stride(0),
        num_tokens,
        top_k,
        BLOCK_M=block_m,
        BLOCK_TOPK=triton.next_power_of_2(top_k),
        SHARED_BLOCK_M=shared_block_m or 1,
        num_warps=num_warps,
    )


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
            shared_x = shared_x_sf = block_m = None
            if self.has_fused_shared_experts:
                shared_x = buffer.shared_l1_acts[:m]
            if shared_x is not None and self.shared_dtype == torch.float8_e4m3fn:
                shared_x_sf = buffer.shared_l1_acts_sf
                block_m = dg.get_block_m_for_nvfp4_mega_moe(
                    get_ep_group().world_size,
                    self.num_experts,
                    buffer.num_max_tokens_per_rank,
                    m,
                    self.top_k,
                )
            prepare_nvfp4_megamoe_inputs(
                hidden_states.contiguous(),
                self.input_global_scale,
                topk_weights,
                topk_ids,
                buffer.x[:m].view(torch.uint8),
                buffer.x_sf[:m].view(torch.float8_e4m3fn),
                buffer.topk_idx[:m],
                buffer.topk_weights[:m],
                is_padding=padding,
                shared_x=shared_x,
                shared_x_sf=shared_x_sf,
                shared_block_m=block_m,
            )
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
