# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.models.glm5next.common.sparse_indexer import (
    RADIX_TOPK_WORKSPACE_SIZE,
    _build_decode_scatter_indices,
    _decode_topk_seq_lens,
    _fill_causal_indices,
    _fill_short_decode_causal_indices,
    _gather_workspace_shapes,
    _scatter_decode_tokens_by_request,
    kv_cache_as_quant_view,
)
from vllm.models.glm5next.nvidia.ops import kpool_compress as kpool_ops
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops

logger = init_logger(__name__)

# kpool write helper: form pools from the current token batch and compress them
# into the index K cache via the fused Triton kernel.


def _kpool_compress_insert(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Pool ``kpool`` consecutive tokens into one fp8 K and write at pool slots.

    ``slot_mapping`` is pool-granular (compress_ratio == kpool on the spec):
    only the *last* token of each complete pool carries a valid (>=0) slot;
    intra-pool tokens are -1. Every position is treated as a pool-completion
    candidate and non-completions are masked off inside the kernel. Compacting
    the valid rows first costs two device syncs on the eager prefill path and
    buys nothing numerically. Assumes pool-aligned chunk starts.
    """
    n = slot_mapping.shape[0]
    # No pool can complete in a batch smaller than one pool; also keeps the
    # clamped gather indices below in bounds.
    if n < kpool:
        return
    pos = torch.arange(n, device=k.device)
    valid = slot_mapping >= 0
    # Drop pools whose start falls before the batch (leading padding); their
    # gate/k data is undefined anyway.
    write_mask = valid & (pos >= kpool - 1)
    offs = torch.arange(kpool, device=k.device)
    idx = (pos - (kpool - 1)).clamp_min(0)[:, None] + offs[None, :]
    kpool_ops.kpool_compress_and_write_cache(
        kv_cache,
        k[idx],  # [n, kpool, head_dim]
        gate_score[idx],
        ape,
        slot_mapping.to(torch.int64),
        pool_size=kpool,
        head_dim=head_dim,
        write_mask=write_mask,
        round_scale=round_scale,
        write_cache=True,
        return_compressed=False,
    )


def _kpool_prefill_cache_update(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    slot_mapping: torch.Tensor,
    tail_slot_mapping: torch.Tensor | None,
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor | None,
    compress_ape: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Write complete pools and persist every request's incomplete tail."""
    _kpool_compress_insert(
        k,
        gate_score,
        compress_ape,
        kv_cache,
        slot_mapping,
        index_kpool,
        head_dim,
        round_scale=round_scale,
    )
    if tail_slot_mapping is not None and tail_kv_cache is not None:
        kpool_ops.kpool_seed_tail_cache(
            tail_kv_cache,
            k,
            gate_score,
            tail_slot_mapping,
            index_kpool,
            head_dim,
        )


def _kpool_decode_cache_update(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    tail_slot_mapping: torch.Tensor | None,
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor | None,
    compress_ape: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    round_scale: bool,
    *,
    num_requests: int,
    num_decode_tokens: int,
    use_uniform: bool,
    group_lens: torch.Tensor | None,
    lmax: int,
) -> None:
    """Group request tokens in position order before updating the raw cache."""
    if tail_slot_mapping is None or tail_kv_cache is None:
        return
    dec_k = k[:num_decode_tokens]
    dec_gate = gate_score[:num_decode_tokens]
    dec_slot = slot_mapping[:num_decode_tokens]
    dec_pos = positions[:num_decode_tokens].to(torch.int32)
    dec_tail = tail_slot_mapping[:num_decode_tokens]
    if use_uniform:
        shape2 = (num_requests, num_decode_tokens // num_requests)
        dec_k = dec_k.view(*shape2, head_dim)
        dec_gate = dec_gate.view(*shape2, head_dim)
        dec_slot = dec_slot.view(shape2)
        dec_pos = dec_pos.view(shape2)
        dec_tail = dec_tail.view(shape2)
    else:
        assert group_lens is not None
        scatter_idx = _build_decode_scatter_indices(
            group_lens, num_requests, num_decode_tokens
        )
        dec_k, dec_gate, dec_slot, dec_pos, dec_tail = (
            _scatter_decode_tokens_by_request(
                tensor, pad, num_requests, lmax, scatter_idx
            )
            for tensor, pad in (
                (dec_k, 0),
                (dec_gate, 0),
                (dec_slot, -1),
                (dec_pos, -1),
                (dec_tail, -1),
            )
        )
    kpool_ops.kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache,
        tail_kv_cache,
        dec_tail,
        dec_k,
        dec_gate,
        compress_ape,
        dec_slot,
        dec_pos,
        index_kpool,
        head_dim,
        round_scale=round_scale,
    )


def _kpool_pcp_cache_update(
    pcp_mgr: "PCPManager",
    k: torch.Tensor,
    gate_score: torch.Tensor,
    slot_mapping: torch.Tensor,
    tail_slot_mapping: torch.Tensor | None,
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor | None,
    compress_ape: torch.Tensor,
    index_kpool: int,
    head_dim: int,
    round_scale: bool,
) -> None:
    """Restore sequence order before writing the replicated kpool caches."""
    global_batch = pcp_mgr.global_batch
    assert global_batch is not None
    assert slot_mapping.shape[0] == pcp_mgr.pcp_world_size * k.shape[0], (
        "kpool PCP write requires the group-expanded slot mapping matching "
        f"the local batch: {slot_mapping.shape[0]} != "
        f"{pcp_mgr.pcp_world_size} * {k.shape[0]}"
    )
    k_global = pcp_mgr.restore_hidden_states(k, require_partition=True)
    gate_global = pcp_mgr.restore_hidden_states(gate_score, require_partition=True)
    slot_global = pcp_mgr.reorder_gathered_to_global(slot_mapping)
    num_tokens = slot_global.shape[0]
    assert k_global.shape[0] == num_tokens

    # Cache writes follow global request state, including short prefill chunks,
    # rather than the local query classification used for indexer scoring.
    is_prefilling = global_batch.is_prefilling_np
    num_decode_reqs = int((~is_prefilling).sum())
    assert not is_prefilling[:num_decode_reqs].any()
    decode_lens_np = global_batch.num_scheduled_tokens[:num_decode_reqs]
    num_decode_tokens = int(decode_lens_np.sum())
    tail_global = None
    if tail_slot_mapping is not None and tail_kv_cache is not None:
        tail_global = pcp_mgr.reorder_gathered_to_global(tail_slot_mapping)

    if num_tokens > num_decode_tokens:
        prefill_slice = slice(num_decode_tokens, num_tokens)
        _kpool_prefill_cache_update(
            k_global[prefill_slice],
            gate_global[prefill_slice],
            slot_global[prefill_slice],
            tail_global[prefill_slice] if tail_global is not None else None,
            kv_cache,
            tail_kv_cache,
            compress_ape,
            index_kpool,
            head_dim,
            round_scale=round_scale,
        )
    if num_decode_tokens > 0 and tail_global is not None:
        lmax = int(decode_lens_np.max())
        use_uniform = int(decode_lens_np.min()) == lmax
        group_lens = (
            None if use_uniform else torch.from_numpy(decode_lens_np).to(k.device)
        )
        _kpool_decode_cache_update(
            k_global,
            gate_global,
            slot_global,
            global_batch.positions,
            tail_global,
            kv_cache,
            tail_kv_cache,
            compress_ape,
            index_kpool,
            head_dim,
            round_scale=round_scale,
            num_requests=num_decode_reqs,
            num_decode_tokens=num_decode_tokens,
            use_uniform=use_uniform,
            group_lens=group_lens,
            lmax=lmax,
        )


@eager_break_during_capture
def sparse_attn_indexer_kpool(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_fp4_cache: bool = False,
    # kpool params (Plan-A: gate is consumed at write time and read back at
    # topk time to softmax-weight the pool).
    gate_score: torch.Tensor | None = None,
    compress_ape: torch.Tensor | None = None,
    index_kpool: int = 1,
    positions: torch.Tensor | None = None,
    # Paged tail cache (in-progress pool's raw K + gate score), replacing the
    # transient _DECODE_TAIL ring. tail_prefix resolves attn_metadata[tail_prefix]
    # for the tail group's token-granular slot_mapping. None on the dummy/profiling
    # path and when the tail cache is disabled.
    tail_kv_cache: torch.Tensor | None = None,
    tail_prefix: str | None = None,
    use_pcp: bool = False,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        current_workspace_manager().get_simultaneous(
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        )

        # Reserve profiler-visible memory for the worst-case decode logits,
        # whose shape is [B * next_n, max_model_len]. This profiling branch
        # returns before invoking the logits kernel itself.
        cfg = get_current_vllm_config_or_none()
        worst_decode_tokens = 0
        if cfg is not None:
            sched = cfg.scheduler_config
            num_spec = (
                cfg.speculative_config.num_speculative_tokens
                if cfg.speculative_config is not None
                else 0
            )
            worst_decode_tokens = min(
                sched.max_num_seqs * (num_spec + 1),
                sched.max_num_batched_tokens,
            )
        # float32 logits -> 4 bytes/element; uint8 sentinel so elems == bytes.
        decode_logits_elems = worst_decode_tokens * max_model_len * 4
        prefill_cap_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        max_logits_elems = max(decode_logits_elems, prefill_cap_elems)
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        return topk_indices_buffer
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        if use_pcp and num_tokens > k.shape[0]:
            num_tokens //= get_pcp_group().world_size
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        if index_kpool > 1 and gate_score is not None and compress_ape is not None:
            if use_pcp:
                # Pooling requires sequence order across zig-zag PCP chunks.
                pcp_mgr = get_forward_context().pcp_manager
                if pcp_mgr is not None and pcp_mgr.global_batch is not None:
                    tail_slot_mapping = None
                    if tail_prefix is not None:
                        tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                        if tail_meta is not None:
                            assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                            tail_slot_mapping = tail_meta.slot_mapping
                    _kpool_pcp_cache_update(
                        pcp_mgr,
                        k,
                        gate_score,
                        slot_mapping,
                        tail_slot_mapping,
                        kv_cache,
                        tail_kv_cache,
                        compress_ape,
                        index_kpool,
                        head_dim,
                        round_scale=(scale_fmt is not None),
                    )
            else:
                n_prefill = num_tokens - num_decode_tokens
                if n_prefill > 0:
                    prefill_slice = slice(num_decode_tokens, num_tokens)
                    tail_slot_mapping = None
                    if tail_kv_cache is not None and tail_prefix is not None:
                        tail_meta = attn_metadata.get(_resolve_layer_name(tail_prefix))
                        if tail_meta is not None:
                            assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
                            tail_slot_mapping = tail_meta.slot_mapping[prefill_slice]
                    _kpool_prefill_cache_update(
                        k[prefill_slice],
                        gate_score[prefill_slice],
                        slot_mapping[prefill_slice],
                        tail_slot_mapping,
                        kv_cache,
                        tail_kv_cache,
                        compress_ape,
                        index_kpool,
                        head_dim,
                        round_scale=(scale_fmt is not None),
                    )
        else:
            # standard: per-token fp8 quant + scatter (all tokens).
            assert scale_fmt is not None
            cache_k, cache_slot_mapping = maybe_gather_indexer_k(
                k, slot_mapping, num_decode_tokens, use_pcp
            )
            ops.indexer_k_quant_and_cache(
                cache_k,
                kv_cache,
                cache_slot_mapping,
                quant_block_size,
                scale_fmt,
            )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Short sequences select every pool, so skip sparse scoring and fill
        # the top-k buffer with all causal token indices. The index-K cache was
        # already written above.
        n_prefill_sf = num_tokens - num_decode_tokens
        # Host-side short-prefill predicate: max_prefill_seq_len is computed
        # in the metadata builder (exact for prefill rows) and equals
        # positions[prefill_slice].max() + 1, so this replaces a
        # positions.max().item() device sync per layer. -1 (unknown metadata)
        # falls back to the device-side check.
        if prefill_metadata.max_prefill_seq_len >= 0:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and prefill_metadata.max_prefill_seq_len <= topk_tokens
            )
        else:
            short_prefill = (
                n_prefill_sf > 0
                and positions is not None
                and int(positions[num_decode_tokens:num_tokens].max().item()) + 1
                <= topk_tokens
            )
        if short_prefill:
            # short_prefill is only True when positions is not None (above),
            # but narrow explicitly for the indexer below.
            assert positions is not None
            _pos = positions[num_decode_tokens:num_tokens].to(torch.int32)
            _buf = topk_indices_buffer[num_decode_tokens:num_tokens]
            _fill_causal_indices(_buf, _pos)

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        k_quant_full, k_scale_full = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
        )
        for chunk in prefill_metadata.chunks if not short_prefill else ():
            k_quant = k_quant_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]

            if not chunk.skip_kv_gather:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )

            q_slice = q_quant[chunk.token_start : chunk.token_end]
            q_scale_slice = (
                q_scale[chunk.token_start : chunk.token_end]
                if q_scale is not None
                else None
            )
            # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
            # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
            if use_fp4_cache:
                q_slice_cast = q_slice.view(torch.int8)
                k_quant_cast = k_quant.view(torch.int8)
                k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
            else:
                q_slice_cast = q_slice
                k_quant_cast = k_quant
                k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
            from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

            logits = fp8_fp4_mqa_logits(
                (q_slice_cast, q_scale_slice),
                (k_quant_cast, k_scale_cast),
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                clean_logits=False,
            )
            num_rows = logits.shape[0]

            # kpool: logits are pool-granular (compress_ratio == index_kpool),
            # so topk selects pools. We pick topk_tokens // kpool pools then
            # expand each pool back to its kpool constituent tokens.
            select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
            if index_kpool > 1:
                pool_topk = torch.empty(
                    (num_rows, select_k), dtype=torch.int32, device=logits.device
                )
                topk_dst = pool_topk
            else:
                topk_dst = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]

            torch.ops._C.top_k_per_row_prefill(
                logits,
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )

            if index_kpool > 1:
                pool_ids = pool_topk.to(torch.int64)
                if positions is not None:
                    # Fused expand-pools + append-tail into one Triton kernel
                    # (replaces ~25 elementwise ops). seq_len is token-granular
                    # (pos+1); the kernel derives pool_len internally.
                    q_seq = (
                        positions[chunk.token_start : chunk.token_end].to(torch.int32)
                        + 1
                    )
                    expanded = kpool_ops.expand_pools_and_append_tail(
                        pool_ids, q_seq, index_kpool
                    )
                else:
                    valid = pool_ids >= 0
                    expanded = kpool_ops.expand_pools_to_tokens(
                        pool_ids, valid, topk_tokens, index_kpool
                    )
                topk_indices_buffer[
                    chunk.token_start : chunk.token_end, : expanded.shape[-1]
                ] = expanded

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache_raw = kv_cache  # raw [num_blocks, block_size, head_dim+4] for writes
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)

        # Update the tail before reading logits; completed pools are compressed
        # into the slot supplied by slot_mapping.
        # Spec verification groups tokens by request and preserves position
        # order so each token is stashed before the next completes its pool.
        # Positions must remain token-granular because the kernel derives the
        # pool phase and tail index from ``pos % kpool``.
        if (
            index_kpool > 1
            and gate_score is not None
            and compress_ape is not None
            and positions is not None
            and not skip_k_cache_insert
            and not use_pcp
        ):
            num_requests = attn_metadata_narrowed.num_decodes
            # Kpool writes must recover the original request grouping after the
            # indexer's flattened decode path. Host metadata avoids a CUDA graph
            # sync when choosing the uniform or padded layout.
            per_req_lens = decode_metadata.per_req_decode_lens
            if per_req_lens is not None:
                use_uniform = (
                    decode_metadata.decode_is_uniform
                    and num_decode_tokens
                    == num_requests * decode_metadata.write_max_decode_len
                )
                group_lens = per_req_lens
                lmax = decode_metadata.write_max_decode_len
            else:
                # Legacy metadata without per-request lens: fall back to the
                # host-side requires_padding flag. Unreached now (per-request
                # lens is always populated for decode), kept defensive.
                use_uniform = not decode_metadata.requires_padding
                group_lens = decode_metadata.decode_lens
                lmax = int(decode_metadata.decode_lens.max().item())
            tail_meta = (
                attn_metadata.get(_resolve_layer_name(tail_prefix))
                if tail_prefix is not None
                else None
            )
            if tail_meta is not None:
                assert isinstance(tail_meta, DeepseekV32IndexerMetadata)
            _kpool_decode_cache_update(
                k,
                gate_score,
                slot_mapping,
                positions,
                tail_meta.slot_mapping if tail_meta is not None else None,
                kv_cache_raw,
                tail_kv_cache,
                compress_ape,
                index_kpool,
                head_dim,
                round_scale=(scale_fmt is not None),
                num_requests=num_requests,
                num_decode_tokens=num_decode_tokens,
                use_uniform=use_uniform,
                group_lens=group_lens,
                lmax=lmax,
            )
        if current_platform.is_cuda_alike() and _fill_short_decode_causal_indices(
            topk_indices_buffer,
            positions,
            num_decode_tokens,
            attn_metadata_narrowed.max_seq_len,
            topk_tokens,
        ):
            return topk_indices_buffer
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # Padding also covers short chunked prefills classified as decode.
            # MXFP4 uses zero-byte padding so padded slots dequantize to zero.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
            padded_weights = pack_seq_triton(
                weights[:num_decode_tokens], decode_lens, pad_value=0
            ).reshape(-1, *weights.shape[1:])
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
            padded_weights = weights[:num_decode_tokens]
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

        logits = fp8_fp4_paged_mqa_logits(
            (padded_q_quant_cast, padded_q_scale),
            kv_cache,
            padded_weights[:num_padded_tokens],
            seq_lens,
            decode_metadata.block_table,
            decode_metadata.schedule_metadata,
            max_model_len=max_model_len,
            clean_logits=False,
        )
        num_rows = logits.shape[0]
        # kpool: logits are pool-granular -> select topk_tokens//kpool pools,
        # then expand each pool back to its kpool tokens.
        select_k = topk_tokens // index_kpool if index_kpool > 1 else topk_tokens
        if index_kpool > 1:
            pool_topk = torch.empty(
                (num_rows, select_k), dtype=torch.int32, device=logits.device
            )
            topk_dst = pool_topk
        else:
            topk_dst = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
            workspace_manager = current_workspace_manager()
            (topk_workspace,) = workspace_manager.get_simultaneous(
                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_dst,
                topk_workspace,
                select_k,
                attn_metadata_narrowed.max_seq_len,
            )
        else:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                seq_lens,
                topk_dst,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                select_k,
            )

        # Resolve to token-level indices in the output buffer.
        if index_kpool > 1:
            pool_ids = pool_topk.to(torch.int64)
            n = pool_topk.shape[0]
            # Decode seq_lens are pool-granular; recover token lengths from
            # positions using the padded [B, next_n] row layout when needed.
            if positions is not None:
                dec_seq = _decode_topk_seq_lens(
                    positions,
                    decode_lens,
                    num_decode_tokens,
                    batch_size,
                    next_n,
                    decode_metadata.requires_padding,
                )
            else:
                dec_seq = decode_metadata.seq_lens[:n]
                if dec_seq.ndim == 2:
                    dec_seq = dec_seq[:, -1]
                dec_seq = dec_seq.to(torch.int32)
            out = kpool_ops.expand_pools_and_append_tail(pool_ids, dec_seq, index_kpool)
        else:
            out = topk_dst

        if decode_metadata.requires_padding:
            # Drop padded query rows introduced by the next_n padding above.
            out = unpack_seq_triton(
                out.reshape(batch_size, -1, out.shape[-1]), decode_lens
            )
        topk_indices_buffer[: out.shape[0], : out.shape[-1]] = out

    return topk_indices_buffer


@CustomOp.register("sparse_attn_indexer_kpool")
class SparseAttnIndexerKpool(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        tail_cache=None,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.tail_cache = tail_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM to be installed."
            )
        _cfg = get_current_vllm_config_or_none()
        _parallel = _cfg.parallel_config if _cfg is not None else None
        self.use_pcp = (
            _parallel is not None and _parallel.prefill_context_parallel_size > 1
        )
        if (
            _parallel is not None
            and _parallel.prefill_context_parallel_size > 1
            and _parallel.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "SparseAttnIndexerKpool does not support PCP+DCP."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        return self.forward_cuda(
            hidden_states,
            q_quant,
            k,
            weights,
            gate_score=gate_score,
            compress_ape=compress_ape,
            index_kpool=index_kpool,
            positions=positions,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor,
        weights: torch.Tensor,
        *,
        gate_score: torch.Tensor | None = None,
        compress_ape: torch.Tensor | None = None,
        index_kpool: int = 1,
        positions: torch.Tensor | None = None,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return sparse_attn_indexer_kpool(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_fp4_cache,
            gate_score,
            compress_ape,
            index_kpool,
            positions,
            self.tail_cache.kv_cache if self.tail_cache is not None else None,
            self.tail_cache.prefix if self.tail_cache is not None else None,
            use_pcp=self.use_pcp,
        )
