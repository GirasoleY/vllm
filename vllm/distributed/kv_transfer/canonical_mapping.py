# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of connector-independent canonical KV page mappings.

This is the only layer that reasons about TP/DCP/PCP placement; connectors
consume byte mappings. An attention canonical page contains all KV heads and
all ``block_size * dcp`` persistent tokens in the worker's page encoding. PCP
partitions forward computation but replicates persistent KV, so it contributes
writer replicas rather than token extent. A recurrent-state canonical page
contains the complete checkpoint state without TP sharding. Uncertifiable
layers get an opaque mapping that is only compatible with an identical
placement.
"""

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    CopyRun,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Version of the canonical byte format; bump on any layout change. Version 2
# records PCP as a persistent-KV replica axis instead of a token-sharding axis.
CANONICAL_FORMAT_VERSION = 2


def canonical_format_id(kv_cache_layout: str) -> str:
    """Identity of the canonical byte format, for namespacing persisted KV.
    Canonical pages keep the worker's KV layout family, so the id couples the
    format version with that family; consumers must match it exactly. The family
    keeps its historical NHD/HND spelling so ids stay stable for KV persisted
    before the layout enum existed."""
    layout = KVCacheLayout[kv_cache_layout]
    legacy = {KVCacheLayout.LBNHC: "nhd", KVCacheLayout.LBHNC: "hnd"}
    family = legacy.get(layout, layout.name.lower())
    return f"v{CANONICAL_FORMAT_VERSION}-{family}"


@dataclass(frozen=True)
class _RankContext:
    """Sharding parameters of one worker rank within the offload group."""

    tp_size: int
    dcp_size: int
    pcp_size: int
    interleave: int
    total_kv_heads: int
    rank: int
    dcp_rank: int

    @property
    def cp_size(self) -> int:
        """Number of token shards represented in one canonical page.

        DCP shards persistent KV tokens. MRV2 PCP only partitions the forward
        computation: cache inputs are gathered across PCP before insertion, so
        every PCP rank retains the same persistent KV page.
        """
        return self.dcp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def total_cp_rank(self) -> int:
        """Persistent-KV token-shard rank within :attr:`cp_size`."""
        return self.dcp_rank


def native_vllm_dcp_rank(
    *,
    tp_size: int,
    tp_rank: int,
    dcp_size: int,
    pcp_size: int = 1,
    pcp_rank: int = 0,
) -> int:
    """Return the DCP rank assigned by vLLM's native process groups.

    DCP is an overlay and never adds a Cartesian worker axis. Without PCP it
    subdivides TP. With PCP it may be disabled, span PCP, or span the complete
    TP x PCP grid. Keeping this adapter explicit prevents callers from treating
    ``dcp_rank`` as another independently flattened coordinate.

    Raises:
        ValueError: If a size/rank is invalid or the topology is not one that
            vLLM's ``ParallelConfig`` can instantiate.
    """
    for name, size in (("tp", tp_size), ("dcp", dcp_size), ("pcp", pcp_size)):
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"{name}_size must be a positive integer")
    for name, rank, size in (
        ("tp", tp_rank, tp_size),
        ("pcp", pcp_rank, pcp_size),
    ):
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < size:
            raise ValueError(f"{name}_rank must be in [0, {size})")

    if pcp_size == 1:
        if tp_size % dcp_size:
            raise ValueError(
                f"tp_size={tp_size} must be divisible by dcp_size={dcp_size}"
            )
        return tp_rank % dcp_size

    valid_dcp_sizes = {1, pcp_size, tp_size * pcp_size}
    if dcp_size not in valid_dcp_sizes:
        raise ValueError(
            "with PCP, dcp_size must be 1, pcp_size, or tp_size * pcp_size; "
            f"got TP={tp_size}, PCP={pcp_size}, DCP={dcp_size}"
        )
    if dcp_size == 1:
        return 0
    if dcp_size == pcp_size:
        return pcp_rank
    # parallel_state transposes PCP and TP before flattening a full-grid DCP
    # group, so PCP is the minor coordinate.
    return tp_rank * pcp_size + pcp_rank


@dataclass(frozen=True)
class ByteRegion:
    """A semantic byte region within a page that repeats once per token.

    Canonical coordinates are page-span independent.  Compact canonical
    storage may use a different offset/stride (notably for HND pages).
    """

    local_offset: int
    canonical_offset: int
    bytes_per_token: int
    canonical_token_stride: int
    canonical_region: int = 0
    canonical_storage_offset: int | None = None
    canonical_storage_token_stride: int | None = None


def _canonical_region_token_strides(
    regions: list[ByteRegion],
) -> tuple[tuple[int, int], ...]:
    """Return deterministic semantic-region geometry for page placement."""
    strides: dict[int, int] = {}
    for region in regions:
        previous = strides.setdefault(
            region.canonical_region, region.canonical_token_stride
        )
        if previous != region.canonical_token_stride:
            raise ValueError(
                f"canonical region {region.canonical_region} has conflicting "
                "token strides"
            )
    return tuple(sorted(strides.items()))


def _coalesce_runs(runs: list[CopyRun]) -> tuple[CopyRun, ...]:
    """Collapse contiguous fragments within and across runs to minimize the
    number of copy ops (e.g. a single-rank mapping becomes one whole-page run).
    """
    out: list[CopyRun] = []
    for run in runs:
        storage_offset = (
            run.canonical_offset
            if run.canonical_storage_offset is None
            else run.canonical_storage_offset
        )
        storage_stride = (
            run.canonical_stride
            if run.canonical_storage_stride is None
            else run.canonical_storage_stride
        )
        if (
            run.num_fragments > 1
            and run.local_stride == run.fragment_size
            and run.canonical_stride == run.fragment_size
            and storage_stride == run.fragment_size
        ):
            size = run.fragment_size * run.num_fragments
            run = CopyRun(
                run.local_offset,
                run.canonical_offset,
                size,
                1,
                size,
                size,
                run.canonical_region,
                storage_offset,
                size,
            )
        prev = out[-1] if out else None
        prev_storage_offset = (
            prev.canonical_offset
            if prev is not None and prev.canonical_storage_offset is None
            else prev.canonical_storage_offset
            if prev is not None
            else None
        )
        if (
            prev is not None
            and prev.num_fragments == 1
            and run.num_fragments == 1
            and prev.canonical_region == run.canonical_region
            and prev.local_offset + prev.fragment_size == run.local_offset
            and prev.canonical_offset + prev.fragment_size == run.canonical_offset
            and prev_storage_offset is not None
            and prev_storage_offset + prev.fragment_size == storage_offset
        ):
            size = prev.fragment_size + run.fragment_size
            out[-1] = CopyRun(
                prev.local_offset,
                prev.canonical_offset,
                size,
                1,
                size,
                size,
                prev.canonical_region,
                prev_storage_offset,
                size,
            )
        else:
            out.append(run)
    return tuple(out)


def _local_to_canonical_token(local_idx: int, ctx: _RankContext) -> int:
    """Canonical position of one of this rank's local token indices."""
    chunk, pos_in_chunk = divmod(local_idx, ctx.interleave)
    return (chunk * ctx.cp_size + ctx.total_cp_rank) * ctx.interleave + pos_in_chunk


def _interleave_cp_tokens(
    regions: list[ByteRegion],
    num_tokens: int,
    ctx: _RankContext,
) -> tuple[CopyRun, ...]:
    """Place each region's num_tokens rows at their canonical token positions,
    one run per chunk of interleaved tokens."""
    runs: list[CopyRun] = []
    for region in regions:
        storage_offset = (
            region.canonical_offset
            if region.canonical_storage_offset is None
            else region.canonical_storage_offset
        )
        storage_stride = (
            region.canonical_token_stride
            if region.canonical_storage_token_stride is None
            else region.canonical_storage_token_stride
        )
        if ctx.cp_size == 1:
            runs.append(
                CopyRun(
                    region.local_offset,
                    region.canonical_offset,
                    region.bytes_per_token,
                    num_tokens,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                    region.canonical_region,
                    storage_offset,
                    storage_stride,
                )
            )
            continue
        for chunk_start in range(0, num_tokens, ctx.interleave):
            canonical_token = _local_to_canonical_token(chunk_start, ctx)
            runs.append(
                CopyRun(
                    region.local_offset + chunk_start * region.bytes_per_token,
                    region.canonical_offset
                    + canonical_token * region.canonical_token_stride,
                    region.bytes_per_token,
                    ctx.interleave,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                    region.canonical_region,
                    storage_offset + canonical_token * storage_stride,
                    storage_stride,
                )
            )
    return _coalesce_runs(runs)


def _packed_kv_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """K and V adjacent per (token, head), in NHD or HND stride order."""
    bs, heads = spec.block_size, spec.num_kv_heads
    elem = kv_cache.element_size()
    head_elems = 2 * spec.head_size
    if heads * bs * head_elems * elem != spec.real_page_size_bytes:
        return None
    _, head_stride, token_stride, inner_stride = kv_cache.stride()
    if inner_stride != 1:
        return None
    head_bytes = head_elems * elem
    token_row_bytes = heads * head_bytes

    if head_stride == head_elems and token_stride == heads * head_elems:  # NHD
        return [
            ByteRegion(
                local_offset=0,
                canonical_offset=head_shard * token_row_bytes,
                bytes_per_token=token_row_bytes,
                canonical_token_stride=num_head_shards * token_row_bytes,
            )
        ]

    if head_stride == bs * head_elems and token_stride == head_elems:  # HND
        canonical_span = bs * cp_size  # canonical tokens per offloaded block
        total_heads = num_head_shards * heads
        return [
            ByteRegion(
                local_offset=head * bs * head_bytes,
                # Connector coordinates are always token-major, independent
                # of the endpoint's physical NHD/HND layout.  Keeping every
                # head in region 0 lets a token-row NHD run intersect these
                # per-head HND runs directly, without a packing kernel.
                canonical_offset=(head_shard * heads + head) * head_bytes,
                bytes_per_token=head_bytes,
                canonical_token_stride=total_heads * head_bytes,
                canonical_region=0,
                canonical_storage_offset=(head_shard * heads + head)
                * canonical_span
                * head_bytes,
                canonical_storage_token_stride=head_bytes,
            )
            for head in range(heads)
        ]
    return None


def _split_kv_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """K and V in separate page halves, in NHD or HND stride order."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    elem = kv_cache.element_size()
    if 2 * bs * heads * head_size * elem != spec.real_page_size_bytes:
        return None
    _, half_stride, token_stride, head_stride, inner_stride = kv_cache.stride()
    if inner_stride != 1 or half_stride != bs * heads * head_size:
        return None
    head_bytes = head_size * elem
    token_row_bytes = heads * head_bytes
    canonical_span = bs * cp_size  # canonical tokens per offloaded block

    if token_stride == heads * head_size and head_stride == head_size:  # NHD
        canonical_half_bytes = canonical_span * num_head_shards * token_row_bytes
        return [
            ByteRegion(
                local_offset=half * bs * token_row_bytes,
                canonical_offset=head_shard * token_row_bytes,
                bytes_per_token=token_row_bytes,
                canonical_token_stride=num_head_shards * token_row_bytes,
                canonical_region=half,
                canonical_storage_offset=half * canonical_half_bytes
                + head_shard * token_row_bytes,
                canonical_storage_token_stride=num_head_shards * token_row_bytes,
            )
            for half in range(2)  # K, then V
        ]

    if head_stride == bs * head_size and token_stride == head_size:  # HND
        local_half_bytes = bs * heads * head_bytes
        total_heads = num_head_shards * heads
        return [
            ByteRegion(
                local_offset=half * local_half_bytes + head * bs * head_bytes,
                # Use the same token-major semantic coordinate as NHD while
                # retaining the historical head-major compact-storage offset.
                canonical_offset=(head_shard * heads + head) * head_bytes,
                bytes_per_token=head_bytes,
                canonical_token_stride=total_heads * head_bytes,
                canonical_region=half,
                canonical_storage_offset=(
                    half * total_heads + head_shard * heads + head
                )
                * canonical_span
                * head_bytes,
                canonical_storage_token_stride=head_bytes,
            )
            for half in range(2)
            for head in range(heads)
        ]
    return None


def _attention_byte_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    num_blocks: int,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """Byte regions of an attention page, given this rank's head shard.
    None when the physical layout is not recognized (fail closed)."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    if tuple(kv_cache.shape) == (num_blocks, heads, bs, 2 * head_size):
        return _packed_kv_regions(kv_cache, spec, head_shard, num_head_shards, cp_size)
    if tuple(kv_cache.shape) == (num_blocks, 2, bs, heads, head_size):
        return _split_kv_regions(kv_cache, spec, head_shard, num_head_shards, cp_size)
    return None


def _is_certified_mamba_page(
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    spec: MambaSpec,
    num_blocks: int,
) -> bool:
    """Return whether ``kv_cache`` exposes the raw packed Mamba page format."""
    if (
        num_blocks <= 0
        or not isinstance(kv_cache, torch.Tensor)
        or kv_cache.layout is not torch.strided
    ):
        return False
    if kv_cache.dtype is not torch.int8 or tuple(kv_cache.shape[:3]) != (
        num_blocks,
        1,
        1,
    ):
        return False
    if kv_cache.ndim != 4 or kv_cache.shape[3] != spec.real_page_size_bytes:
        return False
    if kv_cache.stride(3) != 1:
        return False
    page_stride = kv_cache.stride(0) * kv_cache.element_size()
    return page_stride >= spec.page_size_bytes and kv_cache[0].is_contiguous()


def _gdn_state_component_sizes(spec: MambaSpec) -> tuple[int, ...] | None:
    """Return certified local GDN/KDA component byte sizes.

    The raw page is packed component-first by ``MambaBase.bind_kv_cache``.
    GDN's convolution state itself packs Q, K, and V projections, so those
    three projections must be separate canonical regions when TP sizes differ.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    if (
        spec.tp_replicated
        or spec.mamba_type is not MambaAttentionBackendEnum.GDN_ATTN
        or get_conv_state_layout() != "DS"
        or len(spec.shapes) not in (2, 4)
        or len(spec.shapes) != len(spec.dtypes)
    ):
        return None
    if any(
        not shape
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in shape
        )
        for shape in spec.shapes
    ):
        return None

    conv_shape, temporal_shape = spec.shapes[:2]
    if len(conv_shape) != 2 or len(temporal_shape) != 3:
        return None
    local_heads, head_v_dim, head_k_dim = temporal_shape
    local_conv_dim, conv_rows = conv_shape
    value_dim = local_heads * head_v_dim
    key_remainder = local_conv_dim - value_dim
    if key_remainder <= 0 or key_remainder % 2:
        return None
    key_dim = key_remainder // 2

    if len(spec.shapes) == 4:
        recover_state, recover_inputs = spec.shapes[2:]
        if (
            head_v_dim != head_k_dim
            or len(recover_state) != 3
            or len(recover_inputs) != 3
            or recover_state[0] != local_heads
            or recover_state[2] != head_v_dim
            or recover_inputs != (local_heads, recover_state[1], 2 * head_v_dim)
        ):
            return None

    conv_element_size = torch.empty((), dtype=spec.dtypes[0]).element_size()
    component_sizes = [
        key_dim * conv_rows * conv_element_size,
        key_dim * conv_rows * conv_element_size,
        value_dim * conv_rows * conv_element_size,
    ]
    for shape, dtype in zip(spec.shapes[1:], spec.dtypes[1:]):
        component_sizes.append(
            torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        )
    if sum(component_sizes) != spec.real_page_size_bytes:
        return None
    return tuple(component_sizes)


def _mamba_layer_mapping(
    spec: MambaSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    ctx: _RankContext,
) -> CanonicalPageMapping | None:
    """Certified recurrent-state placement, independent of DCP ownership.

    Mamba state pages are checkpoints rather than token extents, so these
    mappings intentionally omit ``canonical_token_span``. A state-aware
    request planner must bind each allocator page to its checkpoint position.
    """
    if not _is_certified_mamba_page(kv_cache, spec, num_blocks):
        return None

    pcp_rank = ctx.rank // ctx.tp_size
    local_page_size = spec.real_page_size_bytes
    if spec.tp_replicated:
        return CanonicalPageMapping(
            canonical_page_size_bytes=local_page_size,
            local_page_size_bytes=local_page_size,
            runs=(
                CopyRun(
                    0,
                    0,
                    local_page_size,
                    1,
                    local_page_size,
                    local_page_size,
                ),
            ),
            num_writers=ctx.tp_size * ctx.pcp_size,
            writer_index=ctx.rank,
            parallelism_agnostic=True,
        )

    component_sizes = _gdn_state_component_sizes(spec)
    if component_sizes is None:
        return None

    runs: list[CopyRun] = []
    local_offset = 0
    canonical_storage_offset = 0
    for canonical_region, local_size in enumerate(component_sizes):
        rank_offset = ctx.tp_rank * local_size
        runs.append(
            CopyRun(
                local_offset=local_offset,
                canonical_offset=rank_offset,
                fragment_size=local_size,
                num_fragments=1,
                local_stride=local_size,
                canonical_stride=local_size,
                canonical_region=canonical_region,
                canonical_storage_offset=canonical_storage_offset + rank_offset,
                canonical_storage_stride=local_size,
            )
        )
        local_offset += local_size
        canonical_storage_offset += ctx.tp_size * local_size

    return CanonicalPageMapping(
        canonical_page_size_bytes=canonical_storage_offset,
        local_page_size_bytes=local_page_size,
        runs=tuple(runs),
        num_writers=ctx.pcp_size,
        writer_index=pcp_rank,
        parallelism_agnostic=ctx.tp_size == 1,
    )


def _layer_mapping(
    spec: KVCacheSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    ctx: _RankContext,
) -> CanonicalPageMapping | None:
    """Certified mapping for one layer at one rank, or None (fail closed).

    The static attention mapping covers TP, TP-overlaid DCP, and PCP replication
    by itself. Native vLLM DCP and PCP are not independent axes when both are
    enabled, and attention's request-time ownership cannot be represented by
    this page map. Such attention deployments intentionally use the opaque
    fallback. Recurrent state has no DCP token ownership and may still be
    certified.
    """
    if not isinstance(spec, (AttentionSpec, MambaSpec)):
        return None
    if (
        not isinstance(ctx.rank, int)
        or isinstance(ctx.rank, bool)
        or not 0 <= ctx.rank < ctx.tp_size * ctx.pcp_size
    ):
        return None
    try:
        expected_dcp_rank = native_vllm_dcp_rank(
            tp_size=ctx.tp_size,
            tp_rank=ctx.tp_rank,
            dcp_size=ctx.dcp_size,
            pcp_size=ctx.pcp_size,
            pcp_rank=ctx.rank // ctx.tp_size,
        )
    except ValueError:
        return None
    if ctx.dcp_rank != expected_dcp_rank:
        return None
    if isinstance(spec, MambaSpec):
        return _mamba_layer_mapping(spec, kv_cache, num_blocks, ctx)

    bs = spec.block_size
    page = spec.real_page_size_bytes
    if (
        ctx.interleave <= 0
        or ctx.interleave > bs
        or bs % ctx.interleave
        or (ctx.dcp_size > 1 and ctx.pcp_size > 1)
    ):
        return None

    if isinstance(spec, MLAAttentionSpec):
        # TP and PCP replicate the latent; DCP alone shards its tokens.
        if (
            spec.tokens_per_state != 1
            or page % bs
            or ctx.tp_size % ctx.dcp_size
            or spec.kv_quant_mode.is_per_token_head
        ):
            return None
        row = page // bs
        regions = [ByteRegion(0, 0, row, row)]
        writers_per_pcp_rank = ctx.tp_size // ctx.dcp_size
        pcp_rank = ctx.rank // ctx.tp_size
        return CanonicalPageMapping(
            canonical_page_size_bytes=ctx.cp_size * page,
            local_page_size_bytes=page,
            runs=_interleave_cp_tokens(regions, bs, ctx),
            num_writers=writers_per_pcp_rank * ctx.pcp_size,
            writer_index=(
                pcp_rank * writers_per_pcp_rank + ctx.tp_rank // ctx.dcp_size
            ),
            parallelism_agnostic=ctx.cp_size == 1,
            canonical_token_span=bs * ctx.cp_size,
            canonical_region_token_strides=_canonical_region_token_strides(regions),
        )

    if spec.kv_quant_mode.is_per_token_head or not isinstance(kv_cache, torch.Tensor):
        return None
    total, tp = ctx.total_kv_heads, ctx.tp_size
    if spec.num_kv_heads != max(1, total // tp):
        return None
    if total >= tp:
        if total % tp:
            return None
        num_head_shards, replication = tp, 1
    else:
        if tp % total:
            return None
        num_head_shards, replication = total, tp // total
    # DCP shards tokens across ranks holding replicated KV. PCP replicates each
    # resulting TP/DCP placement because PCP cache inputs are gathered before
    # insertion.
    if replication % ctx.dcp_size:
        return None

    head_shard = ctx.tp_rank // replication
    attention_regions = _attention_byte_regions(
        kv_cache, spec, num_blocks, head_shard, num_head_shards, ctx.cp_size
    )
    if attention_regions is None:
        return None
    writers_per_pcp_rank = replication // ctx.dcp_size
    pcp_rank = ctx.rank // ctx.tp_size
    return CanonicalPageMapping(
        canonical_page_size_bytes=ctx.cp_size * num_head_shards * page,
        local_page_size_bytes=page,
        runs=_interleave_cp_tokens(attention_regions, bs, ctx),
        num_writers=writers_per_pcp_rank * ctx.pcp_size,
        writer_index=(
            pcp_rank * writers_per_pcp_rank
            + (ctx.tp_rank % replication) // ctx.dcp_size
        ),
        parallelism_agnostic=ctx.cp_size == 1,
        canonical_token_span=bs * ctx.cp_size,
        canonical_region_token_strides=_canonical_region_token_strides(
            attention_regions
        ),
    )


def _opaque_cache_layout_fingerprint(
    spec: KVCacheSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    kv_cache_layout: str | None = None,
) -> str:
    """Fingerprint the complete rank-local page encoding, excluding capacity.

    ``kv_cache_layout`` records the declared semantic layout as well as the
    concrete tensor view.  The latter normally distinguishes layouts on its
    own, but degenerate shapes can have identical strides while assigning a
    different meaning to the same axes.
    """

    def tensor_layout(tensor: torch.Tensor) -> tuple[object, ...]:
        return (
            str(tensor.dtype),
            str(tensor.layout),
            tuple(tensor.shape[1:]),
            tuple(tensor.stride()[1:]),
        )

    cache_layout: object
    if isinstance(kv_cache, torch.Tensor):
        cache_layout = tensor_layout(kv_cache)
    elif isinstance(kv_cache, list):
        cache_layout = tuple(tensor_layout(tensor) for tensor in kv_cache)
    else:
        cache_layout = None
    description = (
        f"{type(spec).__module__}.{type(spec).__qualname__}",
        repr(spec),
        kv_cache_layout,
        cache_layout,
    )
    return hashlib.sha256(repr(description).encode()).hexdigest()


def _opaque_fallback_mapping(
    page_size_bytes: int,
    *,
    tp_size: int,
    tp_rank: int,
    dcp_size: int,
    dcp_rank: int,
    pcp_size: int,
    pcp_rank: int,
    interleave: int,
    total_kv_heads: int,
    cache_layout_fingerprint: str,
) -> CanonicalPageMapping:
    """Place a page whole in an identical-topology-only canonical layout."""
    expected_dcp_rank = native_vllm_dcp_rank(
        tp_size=tp_size,
        tp_rank=tp_rank,
        dcp_size=dcp_size,
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
    )
    if dcp_rank != expected_dcp_rank:
        raise ValueError(
            f"dcp_rank={dcp_rank} does not match vLLM native DCP rank "
            f"{expected_dcp_rank}"
        )
    rank = pcp_rank * tp_size + tp_rank
    num_ranks = tp_size * pcp_size
    run = CopyRun(
        0, rank * page_size_bytes, page_size_bytes, 1, page_size_bytes, page_size_bytes
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=num_ranks * page_size_bytes,
        local_page_size_bytes=page_size_bytes,
        runs=(run,),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=False,
        is_opaque=True,
        opaque_layout_signature=(
            "opaque-rank-major:v2:axes=pcp,tp:dcp=vllm-native:"
            f"tp={tp_size}:dcp-size={dcp_size}:pcp={pcp_size}:"
            f"interleave={interleave}:total-kv-heads={total_kv_heads}:"
            f"page={page_size_bytes}:layout={cache_layout_fingerprint}"
        ),
    )


def _run_intervals(runs: tuple[CopyRun, ...], canonical: bool) -> list[tuple[int, int]]:
    intervals = []
    for run in runs:
        offset = run.storage_offset if canonical else run.local_offset
        stride = run.storage_stride if canonical else run.local_stride
        for i in range(run.num_fragments):
            start = offset + i * stride
            intervals.append((start, start + run.fragment_size))
    return sorted(intervals)


def _is_exact_partition(intervals: list[tuple[int, int]], size: int) -> bool:
    return (
        bool(intervals)
        and intervals[0][0] == 0
        and intervals[-1][1] == size
        and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    )


def _verify_tiling(layer_name: str, per_rank: list[CanonicalPageMapping]) -> None:
    """Whichever ranks a block elects as writers must tile the canonical page
    exactly once, and each rank's runs must cover exactly its local page."""
    if not per_rank:
        raise ValueError(f"layer {layer_name} has no rank mappings")
    size = per_rank[0].canonical_page_size_bytes
    num_writers = per_rank[0].num_writers
    if size <= 0:
        raise ValueError(f"layer {layer_name} has a non-positive canonical page size")
    if num_writers <= 0:
        raise ValueError(f"layer {layer_name} has a non-positive writer count")
    opaque_signature = per_rank[0].opaque_layout_signature
    for rank, mapping in enumerate(per_rank):
        if mapping.canonical_page_size_bytes != size:
            raise ValueError(
                f"rank {rank} of layer {layer_name} has canonical page size "
                f"{mapping.canonical_page_size_bytes}, expected {size}"
            )
        if mapping.num_writers != num_writers:
            raise ValueError(
                f"rank {rank} of layer {layer_name} has {mapping.num_writers} "
                f"writers, expected {num_writers}"
            )
        if not 0 <= mapping.writer_index < num_writers:
            raise ValueError(
                f"rank {rank} of layer {layer_name} has invalid writer index "
                f"{mapping.writer_index}"
            )
        if mapping.opaque_layout_signature != opaque_signature:
            raise ValueError(
                f"rank {rank} of layer {layer_name} has a different opaque "
                "layout signature"
            )
        local = _run_intervals(mapping.runs, canonical=False)
        if not _is_exact_partition(local, mapping.local_page_size_bytes):
            raise ValueError(
                f"runs do not cover the local page of rank {rank}, layer {layer_name}"
            )
    for block_id in range(num_writers):
        stored: list[tuple[int, int]] = []
        for mapping in per_rank:
            if mapping.is_writer(block_id):
                stored += _run_intervals(mapping.runs, canonical=True)
        stored.sort()
        if not _is_exact_partition(stored, size):
            raise ValueError(
                f"writers of block {block_id} do not tile the canonical page "
                f"of layer {layer_name}"
            )


def _unpadded_page_size(spec: KVCacheSpec) -> int | None:
    if isinstance(spec, AttentionSpec):
        return spec.unpadded_page_size_bytes
    if isinstance(spec, MambaSpec):
        return replace(spec, page_size_padded=None).page_size_bytes
    return None


def _physical_page_spec(
    spec: KVCacheSpec,
    physical_pages_per_logical: int,
) -> KVCacheSpec:
    """Return the spec of one kernel page within a scheduler page."""
    if physical_pages_per_logical == 1 or not isinstance(spec, AttentionSpec):
        return spec
    if spec.block_size % physical_pages_per_logical:
        raise ValueError(
            "physical_pages_per_logical must divide every attention block "
            f"size; got block_size={spec.block_size}, "
            f"physical_pages_per_logical={physical_pages_per_logical}"
        )
    if spec.page_size_padded is not None:
        raise ValueError("padded attention pages cannot be split into kernel pages")

    physical_spec = spec.copy_with_new_block_size(
        spec.block_size // physical_pages_per_logical
    )
    if (
        not isinstance(physical_spec, AttentionSpec)
        or physical_spec.page_size_padded is not None
        or physical_spec.real_page_size_bytes * physical_pages_per_logical
        != spec.real_page_size_bytes
    ):
        raise ValueError(
            "attention page geometry cannot be divided into equal physical pages"
        )
    return physical_spec


def derive_rank_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    *,
    tp_rank: int,
    pcp_rank: int = 0,
    dcp_rank: int | None = None,
    physical_pages_per_logical: int = 1,
) -> dict[str, CanonicalPageMapping]:
    """Per-layer canonical page mappings for one native vLLM worker.

    PP, DP, and EP are deliberately absent from the byte mapping. PP selects
    which layers this worker owns, DP selects the request replica, and EP does
    not place attention KV. DCP is an overlay on the TP x PCP process grid, not
    an additional axis. Callers may supply ``dcp_rank`` explicitly; when it is
    omitted, :func:`native_vllm_dcp_rank` is the documented adapter. An
    explicit value must match that native assignment because this derivation
    does not have a generic DCP-group membership manifest.

    The certified static mapping supports TP, TP-overlaid DCP without PCP, and
    PCP replication without DCP. Simultaneous PCP and DCP uses an opaque
    identical-topology mapping rather than inventing Cartesian token ownership.

    This function certifies byte placement for an already-instantiated cache;
    it does not select an attention backend. Backend-specific CP restrictions
    must therefore have passed during backend construction. The universal
    storage invariant enforced here is that a configured CP interleave is a
    positive divisor of every attention layer's physical block size. When an
    attention backend subdivides one scheduler block, each returned mapping
    describes one of those equal physical pages. Recurrent state ignores DCP
    token ownership: replicated state elects one TP/PCP writer, while certified
    GDN state is sharded only by TP and replicated across PCP.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    dcp_size = parallel_config.decode_context_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    group_size = tp_size * pcp_size

    interleave = parallel_config.cp_kv_cache_interleave_size
    if (
        not isinstance(physical_pages_per_logical, int)
        or isinstance(physical_pages_per_logical, bool)
        or physical_pages_per_logical <= 0
    ):
        raise ValueError("physical_pages_per_logical must be a positive integer")
    if not isinstance(interleave, int) or isinstance(interleave, bool):
        raise ValueError("cp_kv_cache_interleave_size must be an integer")
    if interleave <= 0:
        raise ValueError("cp_kv_cache_interleave_size must be positive")
    if dcp_size > 1 or pcp_size > 1:
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            group_spec = kv_cache_group.kv_cache_spec
            per_layer_specs = (
                group_spec.kv_cache_specs
                if isinstance(group_spec, UniformTypeKVCacheSpecs)
                else dict.fromkeys(kv_cache_group.layer_names, group_spec)
            )
            for layer_name, spec in per_layer_specs.items():
                physical_spec = _physical_page_spec(
                    spec,
                    physical_pages_per_logical,
                )
                if isinstance(physical_spec, AttentionSpec) and (
                    interleave > physical_spec.block_size
                    or physical_spec.block_size % interleave
                ):
                    raise ValueError(
                        "cp_kv_cache_interleave_size must be no greater than and "
                        "divide every physical attention block size; "
                        f"layer {layer_name!r} has "
                        f"physical_block_size={physical_spec.block_size}, "
                        f"interleave={interleave}"
                    )

    expected_dcp_rank = native_vllm_dcp_rank(
        tp_size=tp_size,
        tp_rank=tp_rank,
        dcp_size=dcp_size,
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
    )
    if dcp_rank is None:
        dcp_rank = expected_dcp_rank
    elif (
        not isinstance(dcp_rank, int)
        or isinstance(dcp_rank, bool)
        or not 0 <= dcp_rank < dcp_size
    ):
        raise ValueError(f"dcp_rank must be in [0, {dcp_size})")
    elif dcp_rank != expected_dcp_rank:
        raise ValueError(
            f"dcp_rank={dcp_rank} does not match vLLM native DCP rank "
            f"{expected_dcp_rank} for TP rank {tp_rank}, PCP rank {pcp_rank}"
        )

    def ctx(rank: int) -> _RankContext:
        rank_tp = rank % tp_size
        rank_pcp = rank // tp_size
        return _RankContext(
            tp_size=tp_size,
            dcp_size=dcp_size,
            pcp_size=pcp_size,
            interleave=interleave,
            total_kv_heads=vllm_config.model_config.get_total_num_kv_heads(),
            rank=rank,
            dcp_rank=native_vllm_dcp_rank(
                tp_size=tp_size,
                tp_rank=rank_tp,
                dcp_size=dcp_size,
                pcp_size=pcp_size,
                pcp_rank=rank_pcp,
            ),
        )

    my_rank = pcp_rank * tp_size + tp_rank
    logical_num_blocks = kv_cache_config.num_blocks

    mappings: dict[str, CanonicalPageMapping] = {}
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        group_kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
            per_layer_specs = group_kv_cache_spec.kv_cache_specs
        else:
            per_layer_specs = {}
        for layer_name in kv_cache_group.layer_names:
            logical_spec = per_layer_specs.get(layer_name, group_kv_cache_spec)
            spec = _physical_page_spec(
                logical_spec,
                physical_pages_per_logical,
            )
            num_blocks = logical_num_blocks * (
                physical_pages_per_logical
                if isinstance(logical_spec, AttentionSpec)
                else 1
            )
            per_rank: list[CanonicalPageMapping] = []
            for rank in range(group_size):
                mapping = _layer_mapping(
                    spec, kv_caches.get(layer_name), num_blocks, ctx(rank)
                )
                if mapping is None:
                    break
                per_rank.append(mapping)
            if len(per_rank) != group_size:
                page = _unpadded_page_size(spec)
                if page is None:
                    continue
                layout_fingerprint = _opaque_cache_layout_fingerprint(
                    spec,
                    kv_caches.get(layer_name),
                    kv_cache_config.kv_cache_layout,
                )
                per_rank = [
                    _opaque_fallback_mapping(
                        page,
                        tp_size=tp_size,
                        tp_rank=rank % tp_size,
                        dcp_size=dcp_size,
                        dcp_rank=ctx(rank).dcp_rank,
                        pcp_size=pcp_size,
                        pcp_rank=rank // tp_size,
                        interleave=interleave,
                        total_kv_heads=(
                            vllm_config.model_config.get_total_num_kv_heads()
                        ),
                        cache_layout_fingerprint=layout_fingerprint,
                    )
                    for rank in range(group_size)
                ]
            _verify_tiling(layer_name, per_rank)
            mappings[layer_name] = per_rank[my_rank]
    return mappings


def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
) -> dict[str, CanonicalPageMapping]:
    """Backward-compatible offloading derivation for the current worker group.

    Offloading still requires its process group to be exactly the TP x PCP
    placement grid. Connectors that route PP/DP independently should call
    :func:`derive_rank_canonical_mappings` with explicit rank coordinates.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    group_size = tp_size * pcp_size
    if parallel_config.world_size != group_size:
        return {}
    rank = parallel_config.rank
    if not 0 <= rank < group_size:
        return {}
    return derive_rank_canonical_mappings(
        vllm_config,
        kv_cache_config,
        kv_caches,
        tp_rank=rank % tp_size,
        pcp_rank=rank // tp_size,
        dcp_rank=native_vllm_dcp_rank(
            tp_size=tp_size,
            tp_rank=rank % tp_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pcp_size=pcp_size,
            pcp_rank=rank // tp_size,
        ),
    )


__all__ = [
    "CANONICAL_FORMAT_VERSION",
    "canonical_format_id",
    "derive_canonical_mappings",
    "derive_rank_canonical_mappings",
    "native_vllm_dcp_rank",
]
