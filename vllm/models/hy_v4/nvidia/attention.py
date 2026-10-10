# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA attention and lightning indexer for HY V4 (NVIDIA).

NOTE: The reference implementation also offers a fused MLA preprocessing
wrapper, which is not available here, so this port keeps the eager MLA path
with an FP8 indexer cache. TODO: restore the fused preprocessing once the
kernel lands. The HPC gated-MLA GEMM *is* wired up: it replaces the eager
output gating when the hpc package is installed, ``VLLM_ENABLE_HPC_OPS=1``
and the elementwise/dtype/alignment constraints hold.

The per-head learnable sink is supported through `.flashmla_sparse`, which
subclasses the platform's sparse MLA backend to forward ``attn_sink``.

Prefill context parallelism (PCP) reuses the generic ``MLAAttention`` and
``SparseAttnIndexer`` cache gathers.
"""

from typing import cast

import regex as re
import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.model_executor.layers.hpc import hpc_gated_mla_gemm, hpc_gated_mla_supported
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
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.models.hy_v4.nvidia.indexer_fusion import (
    fused_indexer_k_norm_rope,
    fused_indexer_q_rope_quant,
)
from vllm.models.hy_v4.nvidia.prefill import (
    fp8_linear,
    gate_quant,
    norm_quant,
    packed_fp8_kernel,
)
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import execute_in_parallel
from vllm.utils.torch_utils import aux_stream
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.mla.index_group import SparseMLAIndexGroupBuilder
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.ops.pcp import (
    HY4PCPKVExchange,
    init_hy4_pcp_kv_exchange,
)
from vllm.v1.attention.selector import get_attn_backend

logger = init_logger(__name__)

_SPARSE_LAYER_TYPES = (
    "sparse_attention",
    "sparse",
    "deepseek_sparse_attention",
    "indexed_attention",
)
_WEIGHT_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def compute_skip_topk_layers(config: PretrainedConfig) -> set[int]:
    """Return the backbone layers that reuse a previous layer's top-k indices.

    A "shared" indexer layer performs sparse attention with the indices computed
    by the closest preceding "full" indexer layer, so it does not build its own
    indexer and its checkpoint indexer weights must be skipped.

    Args:
        config: The model config.

    Returns:
        The set of layer indices that share another layer's top-k indices.

    Raises:
        ValueError: If ``indexer_types`` has the wrong length or an unknown
            entry, or if ``index_topk_freq`` is not a positive integer.

    """
    if not hasattr(config, "index_topk"):
        return set()

    num_hidden_layers = config.num_hidden_layers
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None:
        if len(indexer_types) != num_hidden_layers:
            raise ValueError(
                "indexer_types must contain one entry per hidden layer: "
                f"expected {num_hidden_layers}, got {len(indexer_types)}."
            )
        invalid_types = sorted(set(indexer_types) - {"full", "shared"})
        if invalid_types:
            raise ValueError(
                f"indexer_types only supports 'full' and 'shared', got {invalid_types}."
            )
        return {
            layer_idx
            for layer_idx, indexer_type in enumerate(indexer_types)
            if indexer_type == "shared"
        }

    freq = getattr(config, "index_topk_freq", 1)
    if not isinstance(freq, int) or freq <= 0:
        raise ValueError(f"index_topk_freq must be a positive integer, got {freq!r}.")
    pattern = getattr(config, "index_topk_pattern", None)
    offset = getattr(config, "index_skip_topk_offset", 2)
    skip_layers: set[int] = set()
    for layer_idx in range(num_hidden_layers):
        if pattern is None:
            if max(layer_idx - offset + 1, 0) % freq != 0:
                skip_layers.add(layer_idx)
        elif 0 <= layer_idx < len(pattern) and pattern[layer_idx] == "S":
            skip_layers.add(layer_idx)
    return skip_layers


def is_skip_topk_indexer_weight(weight_name: str, skip_topk_layers: set[int]) -> bool:
    """Return whether an indexer weight belongs to a top-k sharing layer.

    Args:
        weight_name: Checkpoint weight name.
        skip_topk_layers: Result of `compute_skip_topk_layers`.

    Returns:
        True when the weight is an indexer weight of a layer that has no
        indexer module and therefore must be dropped.

    """
    if ".indexer." not in weight_name or not skip_topk_layers:
        return False
    match = _WEIGHT_LAYER_INDEX_RE.search(weight_name)
    return match is not None and int(match.group(1)) in skip_topk_layers


class Indexer(nn.Module):
    """Lightning indexer selecting the top-k tokens for sparse MLA."""

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
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank

        # No tensor parallelism, just replicated.
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        # Fused wk + weights_proj: single GEMM producing [head_dim + n_head].
        # FP8 wk weights are upcast to BF16 while loading to keep the fusion.
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

        # FP8 naive cache: values in fp8 plus one fp32 scale per
        # ``quant_block_size`` elements.
        assert cache_config is not None, "HYV4 indexer requires cache_config"
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix

        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

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

    def project_k_and_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fused wk + weights_proj GEMM; depends only on ``hidden_states``."""
        return self.wk_weights_proj(hidden_states)[0]

    def prepare_inputs(
        self,
        qr: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kw: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the quantized query, key and per-head weights of the indexer.

        Token-wise only (no attention metadata), so it can be CUDA-graph
        captured and overlapped with the MLA query projection.
        """
        q = fp8_linear(self.wq_b, qr)
        q = q.view(-1, self.n_head, self.head_dim)
        if (
            q.dtype == torch.bfloat16
            and self.head_dim == 128
            and self.rope_dim == 64
            and self.scale_fmt == "ue8m0"
            and current_platform.is_cuda()
            and not envs.VLLM_BATCH_INVARIANT
        ):
            k = fused_indexer_k_norm_rope(
                kw[:, : self.head_dim],
                self.k_norm,
                positions,
                rotary_emb,
            )
            q_fp8, weights = fused_indexer_q_rope_quant(
                positions,
                q,
                rotary_emb.cos_sin_cache,
                kw[:, self.head_dim :],
                self.softmax_scale,
                self.n_head**-0.5,
                rotary_emb.is_neox_style,
            )
            return q_fp8, k, weights
        # Checkpoint (PTM) layout: pe occupies the LAST rope_dim dims.
        q_nope, q_pe = torch.split(
            q, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        k = kw[:, : self.head_dim]
        weights = kw[:, self.head_dim :]

        k = self.k_norm(k)
        k_nope, k_pe = torch.split(
            k, [self.head_dim - self.rope_dim, self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # RoPE (NeoX) can introduce extra leading dims, so flatten back to the
        # token-major shapes.
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # Reassemble with the original physical layout: no_pe first, pe last.
        q = torch.cat([q_nope, q_pe], dim=-1)
        # ``k_pe`` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_nope, k_pe.squeeze(-2)], dim=-1)

        # Only q is quantized here; k quantization is fused with cache insertion.
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

        return q_fp8, k, weights


class HYV4MLAAttention(nn.Module):
    """Multi-head latent attention with optional sparse lightning indexer.

    Main reference: the DeepSeek-V2 paper and the FlashInfer implementation
    (https://arxiv.org/abs/2405.04434). HY V4 additionally supports an output
    gate (``gated_mla``) and a per-head learnable attention sink.

    The sink is applied by binding the sink-capable backend from
    `.flashmla_sparse`; if no backend on this platform can consume sinks, the
    weight is still loaded but the bias is disabled with a warning.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        layer_idx: int = 0,
        index_group_builder: SparseMLAIndexGroupBuilder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.layer_idx = layer_idx
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank

        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size

        self.layer_id = int(prefix.split(".")[-2])
        layer_types = getattr(config, "layer_types", None)
        requested_sparse = (
            hasattr(config, "index_topk")
            and layer_types is not None
            and self.layer_id < len(layer_types)
            and layer_types[self.layer_id] in _SPARSE_LAYER_TYPES
        )
        # Only actual sparse layers may share another layer's top-k indices.
        self.skip_topk = requested_sparse and self.layer_id in compute_skip_topk_layers(
            config
        )
        # The skip pattern only governs backbone layers. MTP/nextn layers
        # (layer_id >= num_hidden_layers) always build a full indexer: they
        # compute indices at draft step 0 and toggle at runtime.
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        is_mtp_layer = (
            num_hidden_layers is not None and self.layer_id >= num_hidden_layers
        )
        self.create_indexer = requested_sparse and (not self.skip_topk or is_mtp_layer)
        self.is_sparse = requested_sparse

        # Do not silently degrade sparse layers into dense attention. Probe the
        # sparse MLA backend directly and fail fast with the real error.
        kv_cache_dtype = cache_config.cache_dtype if cache_config else "auto"
        if self.is_sparse:
            try:
                get_attn_backend(
                    head_size=self.kv_lora_rank + self.qk_rope_head_dim,
                    dtype=torch.get_default_dtype(),
                    kv_cache_dtype=kv_cache_dtype,
                    use_mla=True,
                    has_sink=False,
                    use_sparse=True,
                    num_heads=self.num_local_heads,
                )
            except Exception as exc:
                raise RuntimeError(
                    "HYV4 sparse attention was requested, but no valid sparse MLA "
                    "backend is available for current runtime/config. "
                    "Refusing to fall back to dense attention."
                ) from exc

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.fused_qkv_a_proj = None
        self.kv_a_proj_with_mqa = None
        if self.q_lora_rank is not None:
            # ``q_a_proj`` and ``kv_a_proj_with_mqa`` read the same
            # ``hidden_states`` and are both TP-replicated, so they run as one
            # GEMM. The checkpoint keeps them separate; `load_weights` merges
            # them through ``stacked_params_mapping``.
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        self.q_a_layernorm = None
        self.q_b_proj = None
        self.q_proj = None
        if self.q_lora_rank is not None:
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )
        self.indexer_rope_emb: nn.Module | None
        self.indexer: Indexer | None
        if self.create_indexer:
            # The checkpoint stores indexer q_pe/k_pe in interleaved
            # (Megatron/PTM) layout, so the indexer must use interleaved RoPE
            # (is_neox_style=False) like the main attention path. Using NeoX
            # here loses the relative-position dependence and corrupts the DSA
            # top-k selection.
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=False,
            )
            # The indexer projects its queries from the MLA q_lora activations,
            # so a sparse layer requires a query down-projection.
            assert q_lora_rank is not None, (
                "HYV4 sparse attention requires q_lora_rank to be set"
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None

        self.gated_mla = bool(getattr(config, "gated_mla", False))
        self.linear_gate: ColumnParallelLinear | None
        if self.gated_mla:
            if config.gating_type == "headwise":
                self.gate_projection_size_per_head = 1
            elif config.gating_type == "elementwise":
                self.gate_projection_size_per_head = self.v_head_dim
            else:
                raise ValueError(f"Unknown gating type: {config.gating_type}")
            self.linear_gate = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.gate_projection_size_per_head,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_gate",
            )
            self.use_hpc_gated_mla = hpc_gated_mla_supported(
                config.gating_type, self.linear_gate
            )
        else:
            self.linear_gate = None
            self.use_hpc_gated_mla = False
        self.prefix = prefix
        self.register_buffer("_qkv_gate_weight", None, persistent=False)
        self.register_buffer("_qkv_gate_scale", None, persistent=False)

        # Overlaps the hidden_states-only side projections with the MLA query
        # path, following DeepSeek-V3.2 (DSA) in TensorRT-LLM.
        self.aux_stream = aux_stream() if current_platform.is_cuda() else None
        self.stream_events = [torch.cuda.Event() for _ in range(4)]

        # Per-head learnable attention sink. Created BEFORE ``MLAAttention`` so
        # it can be forwarded as the ``sinks`` impl kwarg. The parameter always
        # holds the local TP shard.
        self.learnable_sink = bool(getattr(config, "learnable_sink", False))
        sinks = None
        sink_backend: type[AttentionBackend] | None = None
        if self.learnable_sink:
            sink_backend = self._resolve_sink_backend(kv_cache_dtype)
            enable_sink = sink_backend is not None
            self.learnable_sink_param = nn.Parameter(
                torch.empty(
                    self.num_local_heads,
                    # The kernels require fp32 sinks; the disabled path keeps
                    # the checkpoint dtype since the value is never consumed.
                    dtype=torch.float32 if enable_sink else torch.bfloat16,
                )
            )
            if enable_sink:
                sinks = self.learnable_sink_param
                self._force_sparse_mqa()

        extra_impl_args = {} if sinks is None else {"sinks": sinks}
        self.mla_attn = MLAAttention(
            num_heads=self.num_local_heads,
            scale=self.scaling,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indices_buffer=topk_indices_buffer,
            index_group_builder=index_group_builder,
            attn_backend=sink_backend,
            **extra_impl_args,
        )
        self.kv_exchange = self._init_hy4_kv_exchange(vllm_config)

    def prepare_gate_fusion(self) -> None:
        """Concatenate the gate projection onto the q/kv projection weights.

        Builds ``_qkv_gate_weight`` and ``_qkv_gate_scale`` by stacking the
        ``fused_qkv_a_proj`` and ``linear_gate`` FP8 weights and block scales
        along the output dimension, so prefill can run both as one GEMM on a
        shared quantized input. Both layers' ``weight`` parameters are
        re-pointed at views of the concatenated buffer, keeping their names and
        shapes for weight reloads. Leaves the buffers None, and the forward on
        separate GEMMs, unless both layers run on the DeepGEMM UE8M0 kernel on
        CUDA and the HPC gated-MLA path is off.
        """
        qkv_kernel = packed_fp8_kernel(self.fused_qkv_a_proj)
        gate_kernel = packed_fp8_kernel(self.linear_gate)
        if qkv_kernel is None or gate_kernel is None or self.use_hpc_gated_mla:
            return
        qkv = qkv_kernel._get_layer_params(self.fused_qkv_a_proj)
        gate = gate_kernel._get_layer_params(self.linear_gate)
        if (
            qkv.weight.device.type != "cuda"
            or gate.weight.device.type != "cuda"
            or qkv.block_scale.dtype != torch.int32
            or gate.block_scale.dtype != torch.int32
            or qkv.block_scale.shape[0] != qkv.weight.shape[0]
            or gate.block_scale.shape[0] != gate.weight.shape[0]
        ):
            return
        self._qkv_gate_weight = torch.cat((qkv.weight, gate.weight), dim=0)
        self._qkv_gate_scale = (
            torch.cat((qkv.block_scale, gate.block_scale), dim=0).T.contiguous().T
        )
        # Both projections retain their original names and shapes for reloads.
        from vllm.model_executor.utils import replace_parameter

        split = qkv.weight.shape[0]
        replace_parameter(
            self.fused_qkv_a_proj, "weight", self._qkv_gate_weight[:split]
        )
        replace_parameter(self.linear_gate, "weight", self._qkv_gate_weight[split:])

    @property
    def topk_indices_buffer(self) -> torch.Tensor | None:
        """The sparse top-k index buffer this layer reads from.

        The real consumer is ``mla_attn.impl``, an ``AttentionImpl`` rather than
        an ``nn.Module``, so a plain ``named_modules()`` walk cannot reach it.
        The MTP proposer rebinds the draft layers onto the target model's buffer
        through such a walk, so expose it here: without this the draft attention
        would keep reading its own buffer while the indices it needs are written
        to the target's.
        """
        return getattr(self.mla_attn.impl, "topk_indices_buffer", None)

    @topk_indices_buffer.setter
    def topk_indices_buffer(self, buffer: torch.Tensor) -> None:
        self.mla_attn.impl.topk_indices_buffer = buffer  # type: ignore[attr-defined]

    def _resolve_sink_backend(
        self, kv_cache_dtype: str
    ) -> type[AttentionBackend] | None:
        """Return an MLA backend that can apply this layer's learnable sink.

        The sink is part of the architecture, so a backend that cannot apply it
        changes the model's output. Resolution order:

        1. If the backend the selector would pick already advertises
           `supports_sink`, keep it — this also honours an explicit
           ``--attention-backend`` choice.
        2. Otherwise fall back to the sink-capable ``FLASHMLA_SPARSE`` subclass
           in `.flashmla_sparse`, whose kernels accept ``attn_sink``, provided
           it validates against the current runtime configuration.
        3. Otherwise give up on the bias rather than failing the load.

        Args:
            kv_cache_dtype: The layer's KV cache dtype string.

        Returns:
            The backend class to bind, or None when no sink-capable backend is
            available; the caller then loads the sink weight but disables the
            bias.

        """
        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
            selected_cls = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=self.is_sparse,
                num_heads=self.num_local_heads,
            )
        except Exception as exc:
            # Stringify before logging: warning_once dedupes on the arguments,
            # and a fresh exception object per layer would defeat it.
            logger.warning_once(
                "HYV4 failed to select an MLA backend for the learnable sink "
                "(%s); the sink parameter is loaded but the sink bias is "
                "disabled.",
                str(exc),
            )
            return None

        if selected_cls.supports_sink():
            return selected_cls

        from .flashmla_sparse import HYV4FlashMLASparseBackend

        # Mirror how the selector derives the configuration-dependent inputs so
        # this check accepts exactly what the backend would accept at runtime.
        capability = current_platform.get_device_capability()
        if capability is None:
            logger.warning_once(
                "HYV4 learnable sink is unavailable: the device compute "
                "capability is unknown. The sink parameter is loaded but the "
                "sink bias is disabled."
            )
            return None
        vllm_config = get_current_vllm_config()
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        block_size = (
            cache_config.block_size
            if cache_config is not None and cache_config.user_specified_block_size
            else None
        )
        invalid_reasons = HYV4FlashMLASparseBackend.validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=cast(CacheDType, kv_cache_dtype),
            block_size=block_size,
            use_mla=True,
            has_sink=True,
            use_sparse=self.is_sparse,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=capability,
            attn_type=AttentionType.DECODER,
            use_pcp=parallel_config.prefill_context_parallel_size > 1,
            use_dcp=parallel_config.decode_context_parallel_size > 1,
        )
        if invalid_reasons:
            logger.warning_once(
                "HYV4 learnable sink is unavailable: the selected backend %s "
                "cannot apply sinks and the sink-capable FLASHMLA_SPARSE path "
                "is invalid here (%s). The sink parameter is loaded but the "
                "sink bias is disabled.",
                selected_cls.get_name(),
                ", ".join(invalid_reasons),
            )
            return None

        logger.info_once(
            "HYV4 learnable sink enabled: using the sink-capable "
            "FLASHMLA_SPARSE impl instead of %s, which cannot apply sinks.",
            selected_cls.get_name(),
        )
        return HYV4FlashMLASparseBackend

    def _force_sparse_mqa(self) -> None:
        """Keep every token on the sink-capable sparse MQA path.

        ``_resolve_sink_backend`` only binds the backend that serves decode.
        ``MLAAttention`` additionally routes short prefills to a separate dense
        MLA prefill backend, and none of those accept ``attn_sink``, so prefill
        would silently drop the sink while decode applies it. Such a partially
        applied sink is not the trained architecture and corrupts the output, so
        opt out of the dense split instead.

        Prefills up to ``index_topk`` keep every token inside the sparse top-k,
        making the sparse path numerically equivalent to the dense one apart from
        also applying the sink.
        """
        attention_config = get_current_vllm_config().attention_config
        if attention_config.sparse_mla_force_mqa:
            return
        attention_config.sparse_mla_force_mqa = True
        logger.info_once(
            "HYV4 learnable sink enabled: forcing sparse MQA for prefill too, "
            "as the dense MLA prefill backends cannot apply sinks."
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        indexer = self.indexer if self.is_sparse and not self.skip_topk else None
        enable_multi_stream = (
            self.aux_stream is not None
            and hidden_states.shape[0] <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
        )
        qkv_kernel = packed_fp8_kernel(self.fused_qkv_a_proj, hidden_states.shape[0])
        fuse_gate_projection = (
            self._qkv_gate_weight is not None
            and not enable_multi_stream
            and qkv_kernel is not None
        )
        linear_gate = (
            None if self.use_hpc_gated_mla or fuse_gate_projection else self.linear_gate
        )
        aux_streams = [self.aux_stream] if self.aux_stream is not None else None
        events = self.stream_events

        shared_input: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = hidden_states
        if qkv_kernel is not None and (
            fuse_gate_projection
            or packed_fp8_kernel(linear_gate, hidden_states.shape[0]) is not None
        ):
            shared_input = qkv_kernel.quant_fp8(hidden_states)

        def side_projections() -> tuple[torch.Tensor | None, torch.Tensor | None]:
            indexer_kw = (
                indexer.project_k_and_weights(hidden_states) if indexer else None
            )
            gate_score = fp8_linear(linear_gate, shared_input) if linear_gate else None
            return indexer_kw, gate_score

        # The prefill projection includes the gate; decode can overlap its
        # separate gate GEMM and the indexer projection with the query path.
        (q_c, kv_lora, fused_gate), [side_out] = execute_in_parallel(
            lambda: self._project_qkv_a(shared_input, fuse_gate_projection),
            [side_projections if indexer or linear_gate else None],
            events[0],
            [events[1]],
            aux_streams,
            enable=enable_multi_stream,
        )
        indexer_kw, gate_score = side_out if side_out is not None else (None, None)
        if fused_gate is not None:
            gate_score = fused_gate

        def indexer_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            assert indexer is not None and q_c is not None and indexer_kw is not None
            return indexer.prepare_inputs(
                q_c, indexer_kw, positions, self.indexer_rope_emb
            )

        kv_exchange = (
            self.kv_exchange
            if self.kv_exchange is not None
            and llama_4_scaling is None
            and self._is_sparse_mla_prefill(kv_lora.shape[0])
            else None
        )
        if kv_exchange is not None:
            q, kv_c_normed, k_pe, index_inputs = self._project_q_and_kv_fused(
                kv_exchange, positions, q_c, kv_lora, indexer_kw
            )
        else:
            # Phase 2: MLA q up-projection + RoPE on the default stream; the
            # indexer q/k preparation on the aux stream.
            (q, kv_c_normed, k_pe), [index_inputs] = execute_in_parallel(
                lambda: self._project_q_and_kv(
                    positions, hidden_states, q_c, kv_lora, llama_4_scaling
                ),
                [indexer_inputs if indexer else None],
                events[2],
                [events[3]],
                aux_streams,
                enable=enable_multi_stream,
            )
        index_q, index_k, index_weights = (
            index_inputs if index_inputs is not None else (None, None, None)
        )

        output_shape = (
            hidden_states.shape[0],
            self.num_local_heads * self.v_head_dim,
        )
        # Single coarse eager break covering the indexer and MLA attention, as
        # the breakable cudagraph contract requires: everything that reads
        # per-batch metadata runs in one eager segment.
        attn_out = torch.empty(
            output_shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        self._indexer_and_attn(
            hidden_states,
            index_q,
            index_k,
            index_weights,
            q,
            kv_c_normed,
            k_pe,
            attn_out,
            kv_exchange,
        )

        if self.gated_mla and self.linear_gate is not None:
            if self.use_hpc_gated_mla:
                # Projection, sigmoid and the product in one launch. The gate
                # is column-parallel and unbiased, so its local weight shard
                # maps straight onto the local attn_out columns.
                assert hidden_states.is_contiguous() and attn_out.is_contiguous()
                attn_out = hpc_gated_mla_gemm(
                    hidden_states,
                    self.linear_gate.weight,
                    attn_out,
                )
            else:
                assert gate_score is not None
                if packed_fp8_kernel(self.o_proj, attn_out.shape[0]) is not None:
                    attn_out = gate_quant(
                        attn_out,
                        gate_score,
                        self.v_head_dim if self.config.gating_type == "headwise" else 1,
                    )
                elif self.config.gating_type == "headwise":
                    gate_score = gate_score.unsqueeze(-1)
                    attn_out = attn_out.reshape(
                        *attn_out.shape[:-1], -1, self.v_head_dim
                    )
                    attn_out = attn_out * torch.sigmoid(gate_score)
                    attn_out = attn_out.reshape(*attn_out.shape[:-2], -1)
                else:
                    attn_out = attn_out * torch.sigmoid(gate_score)

        return fp8_linear(self.o_proj, attn_out)

    def _init_hy4_kv_exchange(self, vllm_config: VllmConfig) -> HY4PCPKVExchange | None:
        """The fused PCP prologue's exchange, if this layer supports it."""
        if (
            vllm_config.parallel_config.prefill_context_parallel_size <= 1
            or envs.VLLM_HY4_PCP_PROLOGUE == "0"
        ):
            return None
        mla = self.mla_attn
        indexer = self.indexer if self.is_sparse and not self.skip_topk else None
        ok = (
            self.q_lora_rank is not None
            and mla.kv_cache_dtype == "nvfp4_ds_mla"
            and mla.hisparse_cache is None
            and self.kv_lora_rank == 512
            and self.qk_nope_head_dim == 192
            and self.qk_rope_head_dim == 64
            and type(self.rotary_emb) is RotaryEmbedding
            and not self.rotary_emb.is_neox_style
            and self.kv_a_layernorm.weight.dtype == torch.bfloat16
        )
        if ok and indexer is not None:
            ok = (
                indexer.head_dim == 128
                and indexer.rope_dim == 64
                and indexer.scale_fmt == "ue8m0"
                and not indexer.indexer_op.use_fp4_cache
                and not envs.VLLM_BATCH_INVARIANT
                and self.indexer_rope_emb is not None
                and not self.indexer_rope_emb.is_neox_style
                and indexer.k_norm.weight.dtype == torch.float32
                and indexer.k_norm.bias is not None
                and indexer.k_norm.bias.dtype == torch.float32
            )
        if not ok:
            message = f"HY4 PCP prologue does not support layer {mla.layer_name}"
            if envs.VLLM_HY4_PCP_PROLOGUE == "1":
                raise ValueError(f"{message} (VLLM_HY4_PCP_PROLOGUE=1).")
            logger.info_once(f"{message}; using the unfused path.")
            return None
        return init_hy4_pcp_kv_exchange(vllm_config)

    def _is_sparse_mla_prefill(self, num_tokens: int) -> bool:
        """Pure prefill through sparse MQA. Agrees across PCP ranks: metadata
        is global and token counts are padded."""
        assert self.kv_exchange is not None
        if (
            torch.cuda.is_current_stream_capturing()
            or not 0 < num_tokens <= self.kv_exchange.max_local_tokens
        ):
            return False
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        metadata = attn_metadata.get(self.mla_attn.layer_name)
        return (
            metadata is not None
            and getattr(metadata, "num_decode_tokens", 1) == 0
            # Dense MHA prefill reads the local K, not the cache.
            and not self.mla_attn._use_sparse_mha(cast(MLACommonMetadata, metadata))
        )

    def _project_q_and_kv_fused(
        self,
        kv_exchange: HY4PCPKVExchange,
        positions: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        kv_lora: torch.Tensor,
        indexer_kw: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, None, torch.Tensor] | None,
    ]:
        """Multicast K as cache rows on a side stream; meanwhile the Q GEMMs,
        then one kernel for q_pe RoPE, indexer Q RoPE + quant and the
        indexer's top-k row reset."""
        assert self.q_b_proj is not None and q_c is not None
        # K and Q share the activation dtype, so one cos/sin cache each.
        cos_sin = self.rotary_emb._match_cos_sin_cache_dtype(kv_lora)
        indexer = self.indexer if indexer_kw is not None else None
        index_k = index_w = index_cos_sin = k_norm = None
        if indexer is not None:
            assert indexer_kw is not None
            index_k, index_w = indexer_kw.split(
                [indexer.head_dim, indexer_kw.shape[-1] - indexer.head_dim], dim=-1
            )
            index_rope = cast(RotaryEmbedding, self.indexer_rope_emb)
            index_cos_sin = index_rope._match_cos_sin_cache_dtype(kv_lora)
            k_norm = indexer.k_norm
        kv_exchange.multicast(
            kv_lora,
            index_k,
            positions,
            cos_sin,
            index_cos_sin,
            self.kv_a_layernorm.weight,
            self.kv_a_layernorm.variance_epsilon,
            k_norm.weight if k_norm is not None else None,
            k_norm.bias if k_norm is not None else None,
            k_norm.eps if k_norm is not None else 0.0,
        )

        q = fp8_linear(self.q_b_proj, q_c)
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        index_q = index_q_fp8 = index_weights = topk_indices = None
        softmax_scale = head_scale = 1.0
        if indexer is not None:
            index_q = fp8_linear(indexer.wq_b, q_c).view(
                -1, indexer.n_head, indexer.head_dim
            )
            index_q_fp8 = torch.empty_like(index_q, dtype=current_platform.fp8_dtype())
            index_weights = torch.empty(
                index_q.shape[:2], dtype=torch.float32, device=index_q.device
            )
            softmax_scale, head_scale = indexer.softmax_scale, indexer.n_head**-0.5
            topk_indices = indexer.indexer_op.topk_indices_buffer[: q.shape[0]]
        torch.ops._C.hy4_q_rope_indexer_quant(
            q,
            q,
            positions,
            cos_sin,
            index_q,
            index_w,
            index_cos_sin,
            index_q_fp8,
            index_weights,
            topk_indices,
            softmax_scale,
            head_scale,
            False,
        )
        index_inputs = None
        if index_q_fp8 is not None:
            assert index_weights is not None
            index_inputs = (index_q_fp8, None, index_weights)
        # Unused: sparse MQA reads K from the cache.
        kv_c_normed = kv_lora[..., : self.kv_lora_rank]
        k_pe = kv_lora[..., self.kv_lora_rank :].unsqueeze(1)
        return q, kv_c_normed, k_pe, index_inputs

    def _project_qkv_a(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        fuse_gate_projection: bool = False,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Return normalized q-lora, kv-lora and the optional fused gate."""
        if self.q_lora_rank is None:
            assert self.kv_a_proj_with_mqa is not None
            return None, self.kv_a_proj_with_mqa(hidden_states)[0], None
        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None
        gate = None
        if fuse_gate_projection:
            kernel = packed_fp8_kernel(self.fused_qkv_a_proj)
            assert kernel is not None and isinstance(hidden_states, tuple)
            projected = kernel.apply_block_scaled_mm(
                hidden_states[0],
                self._qkv_gate_weight,
                hidden_states[1],
                self._qkv_gate_scale,
            )
            split = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
            gate_start = self.fused_qkv_a_proj.weight.shape[0]
            gate_size = self.num_local_heads * self.gate_projection_size_per_head
            qkv_lora = projected[:, :split]
            gate = projected[:, gate_start : gate_start + gate_size]
        else:
            qkv_lora = fp8_linear(self.fused_qkv_a_proj, hidden_states)
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        if packed_fp8_kernel(self.q_b_proj, q_c.shape[0]) is not None and (
            self.indexer is None
            or packed_fp8_kernel(self.indexer.wq_b, q_c.shape[0]) is not None
        ):
            return norm_quant(q_c, self.q_a_layernorm), kv_lora, gate
        return self.q_a_layernorm(q_c), kv_lora, gate

    def _project_q_and_kv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_c: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        kv_lora: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q_c is not None:
            assert self.q_b_proj is not None
            q = fp8_linear(self.q_b_proj, q_c)
        else:
            assert self.q_proj is not None
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # Add a head dim of 1 to k_pe.
        k_pe = k_pe.unsqueeze(1)
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )

        if llama_4_scaling is not None:
            q *= llama_4_scaling
        return q, kv_c_normed, k_pe

    @eager_break_during_capture
    def _indexer_and_attn(
        self,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, heads * v_head_dim], written in place
        kv_exchange: HY4PCPKVExchange | None = None,
    ) -> None:
        """Run the lightning indexer and MLA attention in one eager segment.

        Both read per-batch attention metadata, so under the breakable cudagraph
        they must not be captured. Keeping them in a single break (instead of one
        break each) also means the attention inputs never have to survive a
        capture-segment boundary. The nested ``sparse_attn_indexer`` and
        ``unified_mla_attention_with_output`` breaks short-circuit here, since
        the capture is no longer active inside an eager segment. The indexer
        q/k/weights are token-wise and prepared before the break.
        """
        # The exchange fills the caches; skip the layers' own inserts.
        cache_filled = kv_exchange is not None
        if kv_exchange is not None:
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            slot_mappings = forward_context.slot_mapping
            assert isinstance(attn_metadata, dict) and isinstance(slot_mappings, dict)
            index_cache = index_slot_mapping = None
            if index_q is not None:
                assert self.indexer is not None
                k_cache = self.indexer.k_cache
                index_metadata = attn_metadata[k_cache.prefix]
                assert isinstance(index_metadata, DeepseekV32IndexerMetadata)
                index_cache, index_slot_mapping = (
                    k_cache.kv_cache,
                    index_metadata.slot_mapping,
                )
            kv_exchange.wait_cache_insert(
                slot_mappings[self.mla_attn.layer_name],
                self.mla_attn.kv_cache,
                index_cache,
                index_slot_mapping,
            )
        self.mla_attn.skip_kv_cache_update = cache_filled
        if index_q is not None:
            assert self.indexer is not None and index_weights is not None
            self.indexer.indexer_op.skip_k_cache_insert = cache_filled
            # The Q kernel already reset the top-k rows on the exchange path.
            self.indexer.indexer_op.skip_topk_buffer_clear = cache_filled
            self.indexer.indexer_op(hidden_states, index_q, index_k, index_weights)
        if self.is_sparse:
            self.mla_attn.impl.record_logical_topk_ready()  # type: ignore[attr-defined]
        result = self.mla_attn(
            q,
            kv_c_normed,
            k_pe,
            output_shape=out.shape,
            output=out,
        )
        if result.data_ptr() != out.data_ptr():
            out.copy_(result)
