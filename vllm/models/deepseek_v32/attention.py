# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.pcp import (
    finalize_mla_pcp_decode,
    maybe_gather_mla_latent_cache_inputs,
)
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import (
    SparseAttnIndexer,
    sparse_attn_indexer,
)
from vllm.model_executor.models.deepseek_v2 import (
    DeepSeekV2FusedQkvAProjLinear,
    DeepseekV32IndexerCache,
    yarn_get_mscale,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata


class DeepseekV32Indexer(nn.Module):
    indexer_cache_cls = DeepseekV32IndexerCache

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )

        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.head_dim, self.n_head],
            bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.wk_weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128
        self.topk_indices_buffer = topk_indices_buffer

        assert cache_config is not None, "DeepSeek V3.2 indexer requires cache_config"
        self.k_cache = type(self).indexer_cache_cls(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size,
        )

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)

        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        kw, _ = self.wk_weights_proj(hidden_states)
        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))

        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        q = q.view(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
        q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
        q_scale = q_scale.view(-1, self.n_head, 1)

        weights = (
            weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
        )
        weights = weights.squeeze(-1)

        return self.indexer_op(hidden_states, q_fp8, k, weights)


class DeepseekV32Attention(MLAAttention):
    indexer: "DeepseekV32Indexer | None"
    indexer_cls: "type[DeepseekV32Indexer]" = DeepseekV32Indexer

    require_fp8_kv_cache: bool = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        attn_backend: "type | None" = None,
    ) -> None:
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        hidden_size = config.hidden_size
        qk_nope_head_dim = config.qk_nope_head_dim
        qk_rope_head_dim = config.qk_rope_head_dim
        v_head_dim = config.v_head_dim
        q_lora_rank = config.q_lora_rank
        kv_lora_rank = config.kv_lora_rank
        num_heads = config.num_attention_heads

        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        num_local_heads = num_heads // tp_size
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        scaling = qk_head_dim**-0.5
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # DSA checkpoints may use plain ("default") or yarn-scaled RoPE.
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        if config.rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            scaling = scaling * mscale * mscale

        layer_id = extract_layer_index(prefix)
        index_topk_freq = getattr(config, "index_topk_freq", 1)
        index_topk_pattern = getattr(config, "index_topk_pattern", None)
        index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)
        if index_topk_pattern is None:
            skip_topk = (
                max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
            )
        elif 0 <= layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        else:
            skip_topk = False

        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = num_hidden_layers is not None and layer_id >= num_hidden_layers

        kv_b_proj = ColumnParallelLinear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        indexer = None
        if not skip_topk or is_mtp_layer:
            indexer = type(self).indexer_cls(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                prefix=f"{prefix}.indexer",
            )

        super().__init__(
            num_heads=num_local_heads,
            scale=scaling,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            use_sparse=True,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            attn_backend=attn_backend,
        )

        self.num_local_heads = num_local_heads
        self.qk_head_dim = qk_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.skip_topk = False
        enable_short_prefill_scoring_skip = (
            not is_mtp_layer
            and not skip_topk
            and not self.use_pcp
            and current_platform.is_cuda()
        )
        self._dense_mha_metadata_layer_name = (
            self.layer_name if enable_short_prefill_scoring_skip else ""
        )

        if self.require_fp8_kv_cache:
            assert is_quantized_kv_cache(self.kv_cache_dtype), (
                "deepseek_v32 (nvidia) requires an fp8 KV cache served by a sparse "
                "MLA backend. Launch with --kv-cache-dtype fp8 (FlashInfer sparse) "
                "or --kv-cache-dtype fp8_ds_mla (FlashMLA sparse)."
            )
            self._fp8_query = self.impl.supports_quant_query_input
            if not self._fp8_query:
                assert self.kv_cache_dtype == "fp8_ds_mla", (
                    "deepseek_v32 (nvidia) on a bf16-query sparse MLA backend "
                    "(FlashMLA sparse) requires the fp8_ds_mla KV cache layout. "
                    "Launch with --kv-cache-dtype fp8_ds_mla."
                )

            self._fp8_kv_needs_view = self.kv_cache_dtype != "fp8_ds_mla"

        self._index_rope_interleave = getattr(config, "indexer_rope_interleave", False)

        # Remaining MLA projections (registered on this module).
        self.fused_qkv_a_proj = DeepSeekV2FusedQkvAProjLinear(
            hidden_size,
            [q_lora_rank, kv_lora_rank + qk_rope_head_dim],
            quant_config=quant_config,
            prefix=f"{prefix}.fused_qkv_a_proj",
        )
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            q_lora_rank,
            num_heads * qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=config.rms_norm_eps)

        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        # Lightning indexer uses its own RoPE; interleave maps to non-NeoX.
        self.indexer_rope_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_pcp:
            return self._forward_pcp(positions, hidden_states)

        # Captured: A-projections (+ indexer A-GEMM on indexer layers).
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.indexer is not None and not self.skip_topk:
            kw = self.indexer.wk_weights_proj(hidden_states)[0]
            index_k = kw[:, : self.indexer.head_dim]
            index_weights = kw[:, self.indexer.head_dim :]
        else:
            index_k = None
            index_weights = None

        num_tokens = hidden_states.shape[0]
        output = torch.empty(
            (num_tokens, self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if isinstance(attn_metadata_raw, dict):
            attn_metadata = attn_metadata_raw.get(self.layer_name)
        elif isinstance(attn_metadata_raw, list):
            attn_metadata = attn_metadata_raw[0].get(self.layer_name)
        else:
            attn_metadata = attn_metadata_raw

        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        mla_slot = slot_mapping.get(self.layer_name)

        if self.indexer is not None and not self.skip_topk:
            has_indexer = True
            indexer_k_norm_w = self.indexer.k_norm.weight
            indexer_k_norm_bias = self.indexer.k_norm.bias
            indexer_k_norm_eps = self.indexer.k_norm.eps
            indexer_k_rope_cos_sin_cache = self.indexer_rope_emb.cos_sin_cache
            indexer_k_cache = self.indexer.k_cache.kv_cache
            indexer_softmax_scale = self.indexer.softmax_scale
            indexer_n_head_scale = self.indexer.n_head**-0.5
        else:
            has_indexer = False
            indexer_k_norm_w = None
            indexer_k_norm_bias = None
            indexer_k_norm_eps = 1e-6
            indexer_k_rope_cos_sin_cache = None
            indexer_k_cache = None
            indexer_softmax_scale = 0.0
            indexer_n_head_scale = 0.0

        if attn_metadata is None:
            mla_kv_cache = None
            mla_k_scale = None
            indexer_k_cache = None
            mla_slot = None
        else:
            mla_kv_cache = self.kv_cache
            mla_k_scale = self._k_scale

        q_c = fused_norm_rope(
            positions,
            q_c,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            k_pe,
            self.rotary_emb.cos_sin_cache,
            index_k,
            indexer_k_norm_w,
            indexer_k_norm_bias,
            indexer_k_norm_eps,
            indexer_k_rope_cos_sin_cache,
            self.topk_indices_buffer,
            slot_mapping=mla_slot,
            indexer_k_cache=indexer_k_cache,
            mla_kv_cache=mla_kv_cache,
            mla_kv_cache_dtype=self.kv_cache_dtype,
            mla_k_scale=mla_k_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
        )

        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_nope = q_nope.transpose(0, 1)
        ql_nope = torch.bmm(q_nope, self.W_UK_T).transpose(0, 1)

        if self.indexer is not None and not self.skip_topk:
            index_q = self.indexer.wq_b(q_c)[0]
            index_q = index_q.view(-1, self.indexer.n_head, self.indexer.head_dim)
        else:
            index_q = None

        index_q_fp8, index_weights_out, mqa_q = fused_q(
            positions,
            q_pe,
            self.rotary_emb.cos_sin_cache,
            index_q,
            self.indexer_rope_emb.cos_sin_cache if has_indexer else None,
            ql_nope,
            self._q_scale,
            index_weights,
            indexer_softmax_scale,
            indexer_n_head_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
            quantize_mqa=self._fp8_query,
        )

        self._sparse_indexer_and_attn(
            q_c,
            index_q_fp8,
            index_weights_out,
            ql_nope,
            mqa_q,
            output,
        )
        return self.o_proj(output)[0]

    @eager_break_during_capture
    def _sparse_indexer_and_attn(
        self,
        q_c: torch.Tensor,
        index_q_fp8: torch.Tensor | None,
        index_weights_out: torch.Tensor | None,
        ql_nope: torch.Tensor,
        mqa_q: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        if self.indexer is not None and not self.skip_topk:
            sparse_attn_indexer(
                q_c,
                self.indexer.k_cache.prefix,
                self.indexer.k_cache.kv_cache,
                index_q_fp8,
                None,  # q_scale folded into weights on the fp8 path
                None,  # k unused when skip_k_cache_insert=True
                index_weights_out,
                self.indexer.quant_block_size,
                self.indexer.scale_fmt,
                self.indexer.topk_tokens,
                self.indexer.head_dim,
                self.indexer.max_model_len,
                self.indexer.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=True,
                use_pcp=False,
                dense_mha_metadata_layer_name=self._dense_mha_metadata_layer_name,
                use_fp4_cache=False,
                # fused_norm_rope already cleared the topk buffer this forward.
                skip_topk_buffer_clear=True,
            )

        attn_metadata_raw = get_forward_context().attn_metadata
        if isinstance(attn_metadata_raw, dict):
            resolved_metadata = attn_metadata_raw.get(self.layer_name)
        elif isinstance(attn_metadata_raw, list):
            resolved_metadata = attn_metadata_raw[0].get(self.layer_name)
        else:
            resolved_metadata = attn_metadata_raw
        attn_metadata = cast("MLACommonMetadata | None", resolved_metadata)

        if attn_metadata is None:
            output.zero_()
            return

        num_actual = attn_metadata.num_actual_tokens  # type: ignore[attr-defined]
        kv_cache = self.kv_cache
        if self._fp8_kv_needs_view:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        if self._fp8_query:
            # FlashInfer sparse: single packed fp8 query.
            mqa_q_arg: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = mqa_q[
                :num_actual
            ]
        else:
            mqa_q_arg = (ql_nope[:num_actual], mqa_q[:num_actual])
        attn_out, _ = self.impl.forward_mqa(  # type: ignore[attr-defined]
            mqa_q_arg, kv_cache, attn_metadata, self
        )

        # NOTE(woosuk): While the below does not need to be in the eager region,
        # we put it here to avoid copying the attention output. Move this back to the
        # captured region once forward_mqa supports `out` argument.
        x = attn_out.view(
            num_actual, self.num_local_heads, self.kv_lora_rank
        ).transpose(0, 1)
        out = (
            output[:num_actual]
            .view(num_actual, self.num_local_heads, self.v_head_dim)
            .transpose(0, 1)
        )
        torch.bmm(x, self.W_UV, out=out)

    def _forward_pcp(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DSV32 fused sparse-MLA prefill with PCP cache collectives.

        Decode stays on the portable MLA path for now. Prefill reuses DSV32's
        fused normalization, RoPE, query packing, and indexer-query preparation,
        then hands the materialized local K rows to the existing PCP gathers.
        """
        attn_metadata_raw = get_forward_context().attn_metadata
        if isinstance(attn_metadata_raw, dict):
            resolved_metadata = attn_metadata_raw.get(self.layer_name)
        elif isinstance(attn_metadata_raw, list):
            resolved_metadata = attn_metadata_raw[0].get(self.layer_name)
        else:
            resolved_metadata = attn_metadata_raw
        attn_metadata = cast("MLACommonMetadata | None", resolved_metadata)
        # V2's post-KV-allocation kernel warmup is a 32-request, two-token
        # prefill. Keep that tiny shape on the established portable path; this
        # fusion targets long prefill, where its launch/copy savings matter.
        if (
            attn_metadata is None
            or attn_metadata.num_decode_tokens
            or hidden_states.shape[0] < 1024
        ):
            return self._forward_pcp_portable(positions, hidden_states)

        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        if self.indexer is not None and not self.skip_topk:
            kw = self.indexer.wk_weights_proj(hidden_states)[0]
            index_k = kw[:, : self.indexer.head_dim]
            index_weights = kw[:, self.indexer.head_dim :]
            has_indexer = True
            index_k_norm_w = self.indexer.k_norm.weight
            index_k_norm_bias = self.indexer.k_norm.bias
            index_k_norm_eps = self.indexer.k_norm.eps
            index_k_rope_cache = self.indexer_rope_emb.cos_sin_cache
            index_k_out = torch.empty_like(index_k)
            indexer_softmax_scale = self.indexer.softmax_scale
            indexer_head_scale = self.indexer.n_head**-0.5
        else:
            index_k = None
            index_weights = None
            has_indexer = False
            index_k_norm_w = None
            index_k_norm_bias = None
            index_k_norm_eps = 1e-6
            index_k_rope_cache = None
            index_k_out = None
            indexer_softmax_scale = 0.0
            indexer_head_scale = 0.0

        q_c_out = torch.empty_like(q_c)
        kv_c_out = torch.empty_like(kv_c)
        k_pe_out = torch.empty_like(k_pe)
        q_c = fused_norm_rope(
            positions,
            q_c,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.variance_epsilon,
            kv_c,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            k_pe,
            self.rotary_emb.cos_sin_cache,
            index_k,
            index_k_norm_w,
            index_k_norm_bias,
            index_k_norm_eps,
            index_k_rope_cache,
            self.topk_indices_buffer,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
            q_c_out=q_c_out,
            kv_c_out=kv_c_out,
            k_pe_out=k_pe_out,
            index_k_out=index_k_out,
        )

        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        ql_nope = torch.bmm(q_nope.transpose(0, 1), self.W_UK_T).transpose(0, 1)
        if has_indexer:
            assert self.indexer is not None
            index_q = self.indexer.wq_b(q_c)[0].view(
                -1, self.indexer.n_head, self.indexer.head_dim
            )
        else:
            index_q = None
        index_q_fp8, index_weights_out, mqa_q = fused_q(
            positions,
            q_pe,
            self.rotary_emb.cos_sin_cache,
            index_q,
            self.indexer_rope_emb.cos_sin_cache if has_indexer else None,
            ql_nope,
            self._q_scale,
            index_weights,
            indexer_softmax_scale,
            indexer_head_scale,
            has_indexer=has_indexer,
            index_rope_interleave=self._index_rope_interleave,
            quantize_mqa=self._fp8_query,
        )

        if has_indexer:
            assert self.indexer is not None and index_k_out is not None
            self.indexer.indexer_op(
                hidden_states,
                index_q_fp8,
                index_k_out,
                index_weights_out,
            )

        forward_context = get_forward_context()
        slot_mapping = forward_context.slot_mapping
        assert isinstance(slot_mapping, dict)
        kv_for_cache, kpe_for_cache, cache_slot_mapping = (
            maybe_gather_mla_latent_cache_inputs(
                kv_c_out,
                k_pe_out.unsqueeze(1),
                slot_mapping.get(self.layer_name),
                0,
                True,
            )
        )
        self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_for_cache,
            kpe_for_cache,
            self.kv_cache,
            cache_slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
        )

        num_actual = attn_metadata.num_actual_tokens
        kv_cache = self.kv_cache
        if self._fp8_kv_needs_view:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)
        mqa_q_arg: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        if self._fp8_query:
            mqa_q_arg = mqa_q[:num_actual]
        else:
            mqa_q_arg = (ql_nope[:num_actual], mqa_q[:num_actual])
        attn_out, lse = self.impl.forward_mqa(  # type: ignore[attr-defined]
            mqa_q_arg, kv_cache, attn_metadata, self
        )
        if self.impl.dcp_world_size > 1:
            assert lse is not None and self.dcp_manager is not None
            seq_lens = (
                attn_metadata.decode.seq_lens
                if attn_metadata.decode is not None
                else cast(torch.Tensor, attn_metadata.seq_lens)[  # type: ignore[attr-defined]
                    : attn_metadata.num_decodes
                ]
            )
            query_start_loc = attn_metadata.query_start_loc[
                : attn_metadata.num_decodes + 1
            ]
            attn_out = self.dcp_manager.combine(
                attn_out,
                lse,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
            )
            attn_out = finalize_mla_pcp_decode(attn_out, self.num_heads)

        output = torch.empty(
            (hidden_states.shape[0], self.num_local_heads * self.v_head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._v_up_proj(attn_out, out=output[:num_actual])
        if num_actual < output.shape[0]:
            output[num_actual:].zero_()
        return self.o_proj(output)[0]

    def _forward_pcp_portable(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Existing portable PCP path, retained for decode/profile fallback."""
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_c, k_pe = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        q_c = self.q_a_layernorm(q_c)
        kv_c = self.kv_a_layernorm(kv_c)
        k_pe = k_pe.unsqueeze(1)

        q = self.q_b_proj(q_c)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if self.indexer is not None and not self.skip_topk:
            self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)

        attn_out = super().forward(
            q,
            kv_c,
            k_pe,
            output_shape=(
                hidden_states.shape[0],
                self.num_local_heads * self.v_head_dim,
            ),
        )
        return self.o_proj(attn_out)[0]
