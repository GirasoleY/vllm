# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sink-capable FlashMLA sparse backend for HY V4 (NVIDIA).

HY V4 adds a per-head learnable attention sink on top of sparse MLA. The
vendored FlashMLA kernels already accept an ``attn_sink`` argument, but vLLM's
shared ``FLASHMLA_SPARSE`` backend neither advertises sink support nor forwards
the tensor, so the bias would be silently dropped.

This module supplies the missing wiring inside the model package, mirroring how
`vllm.models.deepseek_v4.nvidia.flashinfer_sparse` hands ``sinks`` to the
FlashInfer sparse MLA kernels: subclass the platform backend, declare
`supports_sink`, and thread the sink into every kernel call.

The subclass intentionally keeps the inherited ``get_name()``
(``"FLASHMLA_SPARSE"``). Several shared code paths key off that exact string —
`_canonicalize_sparse_mla_kv_cache_dtype` promotes a quantized KV cache to
``fp8_ds_mla`` for it, and `FlashMLASparseImpl` asserts that layout — so a new
name would silently change KV cache behaviour. ``supports_sink``, the two
kernel wrappers, and the direct head-padded query buffer differ from the
parent.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# The BF16 sparse prefill kernel needs a head count that is a multiple of 128
# on Blackwell; HY V4 has 64, so its query is head-padded and the padding
# heads must stay zero (the kernel reduces across the head group).
_DIRECT_Q_BUFFERS: dict[tuple, torch.Tensor] = {}


def _direct_q_buffer(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The shared zero-initialised head-padded query buffer for *shape*."""
    key = (tuple(shape), dtype, str(device))
    buf = _DIRECT_Q_BUFFERS.get(key)
    if buf is None:
        buf = torch.zeros(shape, dtype=dtype, device=device)
        _DIRECT_Q_BUFFERS[key] = buf
    return buf


def _padded_q_view_offset(
    q: torch.Tensor, base: torch.Tensor, num_heads: int
) -> int | None:
    """Token offset of *q* inside *base* when *q* is a ``base[a:b, :num_heads]``
    view (as handed out by ``get_mqa_q_buffer``); None otherwise."""
    if q.dtype != base.dtype or q.dim() != 3 or q.shape[1] != num_heads:
        return None
    if q.stride() != base[:, :num_heads, :].stride():
        return None
    row_bytes = base.stride(0) * base.element_size()
    delta = q.data_ptr() - base.data_ptr()
    if delta < 0 or delta % row_bytes != 0:
        return None
    tok0 = delta // row_bytes
    if tok0 + q.shape[0] > base.shape[0]:
        return None
    return tok0


class HYV4FlashMLASparseImpl(FlashMLASparseImpl):
    """FlashMLA sparse impl that applies HY V4's per-head learnable sink.

    The sink enters as the ``sinks`` impl kwarg of
    `vllm.model_executor.layers.attention.MLAAttention` and is consumed by the
    FlashMLA kernels, which fold it into the softmax denominator:
    ``out *= exp(lse) / (exp(lse) + exp(sink))``.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        # ``SparseMLACommonImpl`` takes explicit keyword arguments only, so the
        # sink has to be removed before the base classes see``mla_args``.
        sinks: torch.Tensor | None = mla_args.pop("sinks", None)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            topk_indices_buffer=topk_indices_buffer,
            indexer=indexer,
            **mla_args,
        )
        self._validate_sinks(sinks, num_heads)
        self.sinks = sinks

    @staticmethod
    def _validate_sinks(sinks: torch.Tensor | None, num_heads: int) -> None:
        """Reject sink tensors the FlashMLA kernels cannot consume.

        Args:
            sinks: Candidate sink tensor, or None when the layer has no sink.
            num_heads: Local (TP-sharded) query head count of this layer.

        Raises:
            ValueError: If the dtype is not float32 or the shape is not
                ``(num_heads,)``.

        """
        if sinks is None:
            return
        if sinks.dtype != torch.float32:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have dtype "
                f"torch.float32, but got {sinks.dtype}."
            )
        if sinks.ndim != 1 or sinks.shape[0] != num_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks must have shape "
                f"({num_heads},), but got {tuple(sinks.shape)}."
            )

    def _sinks_for_query(
        self,
        q: torch.Tensor,
        head_dim: int,
        kernel_heads: int,
    ) -> torch.Tensor | None:
        """Return the sink laid out for the kernel's query head count.

        Args:
            q: Query tensor, before any head padding.
            head_dim: Axis of ``q`` holding the query heads.
            kernel_heads: Head count the kernel is invoked with, which may
                exceed the query head count because of padding.

        Returns:
            The sink tensor padded to ``kernel_heads`` with ``-inf`` (a no-op
            sink) for the padded lanes, or None when the layer has no sink.

        Raises:
            ValueError: If the sink and query head layouts disagree, or if they
                live on different devices.

        """
        sinks = self.sinks
        if sinks is None:
            return None

        query_heads = q.shape[head_dim]
        if sinks.shape[0] != query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sink head count must match the "
                f"runtime query layout: sinks={sinks.shape[0]}, "
                f"query_heads={query_heads}. The sink must use the same "
                "unpadded head layout as the query."
            )
        if sinks.device != q.device:
            raise ValueError(
                "HYV4 FlashMLA sparse attention sinks and query must be on the "
                f"same device, but got sinks={sinks.device}, query={q.device}."
            )
        if kernel_heads < query_heads:
            raise ValueError(
                "HYV4 FlashMLA sparse kernel head count cannot be smaller than "
                f"the runtime query layout: query_heads={query_heads}, "
                f"kernel_heads={kernel_heads}."
            )
        if kernel_heads == query_heads:
            return sinks

        # Mirror the query padding the kernels require. Reading ``sinks`` here
        # (rather than caching a padded copy at construction time) keeps the
        # values correct for weights loaded after the module is built, and the
        # allocation plus copy are captured in the CUDA graph.
        padded_sinks = sinks.new_full((kernel_heads,), float("-inf"))
        padded_sinks[:query_heads] = sinks
        return padded_sinks

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads
        attn_sink = self._sinks_for_query(q, head_dim=2, kernel_heads=padded_num_heads)

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
            attn_sink=attn_sink,
        )

        # Slice output and lse back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]
            lse = lse[:, :actual_num_heads, :]

        return out, lse

    def _direct_q_base(self) -> torch.Tensor | None:
        """The dedicated head-padded query buffer for this layer's shape, or
        None when the direct-query path does not apply."""
        if not is_quantized_kv_cache(self.kv_cache_dtype):
            return None
        if self.pcp_dcp_kv_gather or self.num_heads % self.prefill_padding == 0:
            return None
        # workspace_specs[-3] is (max_tokens, padded_prefill_query_heads, head_size).
        shape, dtype = self.workspace_specs[-3]
        device = torch.device("cuda", torch.accelerator.current_device_index())
        return _direct_q_buffer(shape, dtype, device)

    def get_mqa_q_buffer(
        self, num_tokens: int, attn_metadata: FlashMLASparseMetadata | None
    ) -> torch.Tensor | None:
        """(num_tokens, num_heads, head_size) view into the dedicated head-padded
        query buffer for the MLA layer to write the absorbed query into directly.
        None when the direct-query path does not apply (bf16 cache, FP8 mixed
        batch, PCP+DCP, no padding needed) or the batch has decode rows."""
        if attn_metadata is None or getattr(attn_metadata, "fp8_use_mixed_batch", True):
            return None
        # Pure-prefill batches only: the FP8 decode kernel wants a contiguous
        # query.
        if getattr(attn_metadata, "num_decode_tokens", None) != 0:
            return None
        base = self._direct_q_base()
        if base is None or num_tokens > base.shape[0]:
            return None
        return base[:num_tokens, : self.num_heads, :]

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
        actual_num_heads: int | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell. Pad from q's head count, not
        # self.num_heads: under DCP the heads are all-gathered before this.
        if actual_num_heads is None:
            actual_num_heads = q.shape[1]
        padded_num_heads = (
            (actual_num_heads + self.prefill_padding - 1)
            // self.prefill_padding
            * self.prefill_padding
        )
        # q_concat_buffer may already include the kernel-required padding.
        # The sink remains defined only for the real query heads.
        attn_sink = self._sinks_for_query(
            q[:, :actual_num_heads], head_dim=1, kernel_heads=padded_num_heads
        )

        if q.shape[1] < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for BF16 sparse prefill kernel"
            )
            # The padded lanes must be zero (not merely allocated): topk_indices
            # is shared by all heads, so the kernel reduces across the head
            # group and NaNs from uninitialized memory would leak into the real
            # heads.
            base = self._direct_q_base() if out is not None else None
            tok0 = (
                _padded_q_view_offset(q, base, actual_num_heads)
                if base is not None
                else None
            )
            if tok0 is not None and base is not None:
                q = base[tok0 : tok0 + num_tokens]
            else:
                if out is None:
                    q_padded = q.new_zeros((q.shape[0], padded_num_heads, q.shape[2]))
                else:
                    *_, q_padded, _, _ = current_workspace_manager().get_simultaneous(
                        *self.workspace_specs
                    )
                    q_padded = q_padded[:num_tokens]
                    q_padded[:, actual_num_heads:].zero_()
                q_padded[:, :actual_num_heads, :] = q
                q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, lse = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
            out=out,
        )

        output = output[:, :actual_num_heads, :]
        lse = lse[:, :actual_num_heads]
        return output, lse


class HYV4FlashMLASparseBackend(FlashMLASparseBackend):
    """``FLASHMLA_SPARSE`` with attention-sink support for HY V4.

    Keeps the parent's name, metadata and builder; only the impl class and the
    sink capability differ. See the module docstring for why the name is reused.
    """

    @staticmethod
    def get_impl_cls() -> type[HYV4FlashMLASparseImpl]:
        return HYV4FlashMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
