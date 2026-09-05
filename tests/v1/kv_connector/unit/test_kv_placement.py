# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import random
from dataclasses import FrozenInstanceError

import pytest

import vllm.distributed.kv_transfer.kv_placement as kv_placement_module
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    ConnectorCapabilities,
    CopyRun,
    KVFormatManifest,
    KVGroupFormat,
    KVRange,
    LayerPageMapping,
    PagePlacement,
    RankPlacementManifest,
    TransferRun,
    compose_page_placements,
    iter_page_placement_transfer_runs,
    validate_kv_ranges,
)


def _mapping(
    canonical_size: int,
    local_size: int,
    *runs: CopyRun,
    num_writers: int = 1,
    writer_index: int = 0,
    canonical_token_span: int | None = None,
    canonical_region_token_strides: tuple[tuple[int, int], ...] = (),
    is_opaque: bool = False,
    opaque_layout_signature: str | None = None,
) -> CanonicalPageMapping:
    return CanonicalPageMapping(
        canonical_page_size_bytes=canonical_size,
        local_page_size_bytes=local_size,
        runs=runs,
        num_writers=num_writers,
        writer_index=writer_index,
        canonical_token_span=canonical_token_span,
        canonical_region_token_strides=canonical_region_token_strides,
        is_opaque=is_opaque,
        opaque_layout_signature=opaque_layout_signature,
    )


def _identity(size: int, **kwargs) -> CanonicalPageMapping:
    return _mapping(
        size,
        size,
        CopyRun(0, 0, size, 1, size, size),
        **kwargs,
    )


def _placement(
    rank: int,
    page_id: int,
    mapping: CanonicalPageMapping,
    *,
    canonical_base: int = 0,
    local_base: int = 0,
    canonical_page_index: int = 0,
    canonical_space_id: str = "test-canonical-space",
    first_token: int | None = None,
    valid_token_offset: int = 0,
    valid_token_count: int | None = None,
) -> PagePlacement:
    return PagePlacement(
        rank=rank,
        local_page_id=page_id,
        canonical_page_index=canonical_page_index,
        mapping=mapping,
        canonical_space_id=canonical_space_id,
        first_token=first_token,
        valid_token_offset=valid_token_offset,
        valid_token_count=valid_token_count,
        canonical_base=canonical_base,
        local_base=local_base,
    )


def _group(
    group_id: int = 0,
    layer_names: tuple[str, ...] = ("layers.0.attn",),
    semantic_id: str | None = None,
) -> KVGroupFormat:
    return KVGroupFormat(
        group_id=group_id,
        semantic_id=semantic_id or f"attention-group-{group_id}",
        kind="mla",
        layer_names=layer_names,
        canonical_page_token_span=512,
        dtype="fp8_e4m3fn",
        canonical_page_size_bytes=4096,
        format_id="v1-mla-latent-mxfp8",
        quantization="mxfp8",
        scale_dtype="fp8_e8m0fnu",
        scale_granularity="32-elements",
    )


def _format_manifest(*groups: KVGroupFormat) -> KVFormatManifest:
    return KVFormatManifest(
        version=1,
        model_fingerprint="sha256:model-config-and-weights",
        groups=groups or (_group(),),
    )


def _rank_mapping() -> CanonicalPageMapping:
    return _identity(
        16,
        canonical_token_span=16,
        canonical_region_token_strides=((0, 1),),
    )


def _rank_format_manifest() -> KVFormatManifest:
    return _format_manifest(
        KVGroupFormat(
            group_id=0,
            semantic_id="attention-group-0",
            kind="mla",
            layer_names=("layers.0.attn",),
            canonical_page_token_span=16,
            dtype="uint8",
            canonical_page_size_bytes=16,
            format_id="test-token-major",
        )
    )


def _rank_manifest(**overrides) -> RankPlacementManifest:
    values = {
        "version": 1,
        "deployment_id": "prefill-deployment",
        "topology_generation": 7,
        "worker_id": "prefill-worker-3",
        "worker_incarnation": "boot-4d6388",
        "format_manifest_fingerprint": _rank_format_manifest().fingerprint(),
        "rank": 3,
        "tp_size": 8,
        "tp_rank": 3,
        "dcp_size": 8,
        "dcp_rank": 3,
        "dcp_group_id": "dp2-pp0-dcp0",
        "pcp_size": 1,
        "pcp_rank": 0,
        "pp_size": 2,
        "pp_rank": 1,
        "dp_size": 4,
        "dp_rank": 2,
        "dp_group_id": "deployment-dp",
        "ep_size": 8,
        "ep_rank": 6,
        "cp_interleave": 1,
        "layer_range": (0, 1),
        "mappings": (
            LayerPageMapping("layers.0.attn", 0, "attention-group-0", _rank_mapping()),
        ),
    }
    values.update(overrides)
    return RankPlacementManifest(**values)


def test_identity_composition_preserves_page_ids_and_bases():
    mapping = _identity(16)

    plan = compose_page_placements(
        [_placement(3, 17, mapping, local_base=32)],
        [_placement(9, 41, mapping, local_base=64)],
    )

    assert plan == (
        TransferRun(
            source_rank=3,
            destination_rank=9,
            source_page_id=17,
            destination_page_id=41,
            source_offset=32,
            destination_offset=64,
            fragment_size=16,
            fragment_count=1,
            source_stride=16,
            destination_stride=16,
        ),
    )


def test_scatter_and_gather_compose_to_affine_runs():
    full = _identity(16)
    even = _mapping(16, 8, CopyRun(0, 0, 4, 2, 4, 8))
    odd = _mapping(16, 8, CopyRun(0, 4, 4, 2, 4, 8))

    scatter = compose_page_placements(
        [_placement(0, 100, full)],
        [_placement(1, 200, even), _placement(2, 201, odd)],
    )
    assert scatter == (
        TransferRun(0, 1, 100, 200, 0, 0, 4, 2, 8, 4),
        TransferRun(0, 2, 100, 201, 4, 0, 4, 2, 8, 4),
    )

    gather = compose_page_placements(
        [_placement(1, 200, even), _placement(2, 201, odd)],
        [_placement(0, 100, full)],
    )
    assert gather == (
        TransferRun(1, 0, 200, 100, 0, 0, 4, 2, 4, 8),
        TransferRun(2, 0, 201, 100, 0, 4, 4, 2, 4, 8),
    )


def test_different_canonical_page_spans_compose_by_absolute_base():
    small = _identity(8)
    large = _identity(16)

    plan = compose_page_placements(
        [
            _placement(0, 10, small, canonical_base=32, local_base=2),
            _placement(0, 11, small, canonical_base=40, local_base=4),
        ],
        [_placement(1, 20, large, canonical_base=32, local_base=6)],
    )

    assert plan == (
        TransferRun(0, 1, 10, 20, 2, 6, 8, 1, 8, 8),
        TransferRun(0, 1, 11, 20, 4, 14, 8, 1, 8, 8),
    )


def test_semantic_regions_compose_head_major_pages_with_different_spans():
    # Compact source pages contain [h0t0,h0t1,h1t0,h1t1].  Semantic regions
    # separate the heads so two such pages compose into the destination's
    # [h0t0..h0t3,h1t0..h1t3] layout without a staging transpose.
    source_mapping = _mapping(
        4,
        4,
        CopyRun(0, 0, 2, 1, 2, 2, 0, 0, 2),
        CopyRun(2, 0, 2, 1, 2, 2, 1, 2, 2),
        canonical_token_span=2,
        canonical_region_token_strides=((0, 1), (1, 1)),
    )
    destination_mapping = _mapping(
        8,
        8,
        CopyRun(0, 0, 4, 1, 4, 4, 0, 0, 4),
        CopyRun(4, 0, 4, 1, 4, 4, 1, 4, 4),
        canonical_token_span=4,
        canonical_region_token_strides=((0, 1), (1, 1)),
    )

    plan = compose_page_placements(
        [
            _placement(0, 10, source_mapping, first_token=0),
            _placement(0, 11, source_mapping, first_token=2),
        ],
        [_placement(1, 20, destination_mapping, first_token=0)],
    )

    assert plan == (
        TransferRun(0, 1, 10, 20, 0, 0, 2, 2, 2, 4),
        TransferRun(0, 1, 11, 20, 0, 2, 2, 2, 2, 4),
    )


def _cp_mapping(
    cp_size: int,
    cp_rank: int,
    local_tokens: int,
    interleave: int,
) -> CanonicalPageMapping:
    assert local_tokens % interleave == 0
    runs = []
    for local_start in range(0, local_tokens, interleave):
        chunk = local_start // interleave
        canonical_start = (chunk * cp_size + cp_rank) * interleave
        runs.append(
            CopyRun(
                local_start,
                canonical_start,
                interleave,
                1,
                interleave,
                interleave,
            )
        )
    return _mapping(
        local_tokens * cp_size,
        local_tokens,
        *runs,
        canonical_token_span=local_tokens * cp_size,
        canonical_region_token_strides=((0, 1),),
    )


def test_nondivisible_dcp_sizes_and_interleaves_compose_with_partial_tail():
    # DCP2/I1 uses 4 local tokens per 8-token canonical page. DCP3/I2 uses
    # 4 local tokens per 12-token page. Neither DCP size nor page span divides
    # the other; a 23-token request still composes directly through token IDs.
    sources = [
        _placement(
            rank,
            page,
            _cp_mapping(2, rank, 4, 1),
            first_token=page * 8,
            valid_token_count=min(8, max(0, 23 - page * 8)),
            canonical_page_index=page,
        )
        for page in range(3)
        for rank in range(2)
    ]
    destinations = [
        _placement(
            10 + rank,
            page,
            _cp_mapping(3, rank, 4, 2),
            first_token=page * 12,
            valid_token_count=min(12, max(0, 23 - page * 12)),
            canonical_page_index=page,
        )
        for page in range(2)
        for rank in range(3)
    ]

    plan = compose_page_placements(sources, destinations)

    assert sum(run.fragment_size * run.fragment_count for run in plan) == 23
    assert {run.source_rank for run in plan} == {0, 1}
    assert {run.destination_rank for run in plan} == {10, 11, 12}


def _tp_head_mapping(
    tp_size: int,
    tp_rank: int,
    *,
    total_heads: int,
    tokens: int = 2,
) -> CanonicalPageMapping:
    if total_heads >= tp_size:
        assert total_heads % tp_size == 0
        local_heads = total_heads // tp_size
        replication = 1
        first_head = tp_rank * local_heads
    else:
        assert tp_size % total_heads == 0
        local_heads = 1
        replication = tp_size // total_heads
        first_head = tp_rank // replication
    runs = tuple(
        CopyRun(
            local_head * tokens,
            0,
            tokens,
            1,
            tokens,
            tokens,
            first_head + local_head,
            (first_head + local_head) * tokens,
            tokens,
        )
        for local_head in range(local_heads)
    )
    return _mapping(
        total_heads * tokens,
        local_heads * tokens,
        *runs,
        num_writers=replication,
        writer_index=tp_rank % replication,
        canonical_token_span=tokens,
        canonical_region_token_strides=tuple((head, 1) for head in range(total_heads)),
    )


@pytest.mark.parametrize(("source_tp", "destination_tp"), [(3, 4), (4, 3)])
def test_nondivisible_tp_sizes_compose_directly_by_semantic_head(
    source_tp: int, destination_tp: int
):
    total_heads = 12
    sources = [
        _placement(
            rank,
            0,
            _tp_head_mapping(source_tp, rank, total_heads=total_heads),
            first_token=0,
        )
        for rank in range(source_tp)
    ]
    destinations = [
        _placement(
            20 + rank,
            0,
            _tp_head_mapping(destination_tp, rank, total_heads=total_heads),
            first_token=0,
        )
        for rank in range(destination_tp)
    ]

    plan = compose_page_placements(sources, destinations)

    assert sum(run.fragment_size * run.fragment_count for run in plan) == 24
    assert {run.source_rank for run in plan} == set(range(source_tp))
    assert {run.destination_rank for run in plan} == {
        20 + rank for rank in range(destination_tp)
    }


@pytest.mark.parametrize(
    ("canonical_page_index", "expected_sources"),
    [(0, {0, 2}), (1, {1, 3})],
)
def test_gqa_replica_election_rotates_without_duplicate_reads(
    canonical_page_index: int, expected_sources: set[int]
):
    sources = [
        _placement(
            rank,
            canonical_page_index,
            _tp_head_mapping(4, rank, total_heads=2),
            first_token=canonical_page_index * 2,
            canonical_page_index=canonical_page_index,
        )
        for rank in range(4)
    ]
    destinations = [
        _placement(
            10 + rank,
            canonical_page_index,
            _tp_head_mapping(2, rank, total_heads=2),
            first_token=canonical_page_index * 2,
            canonical_page_index=canonical_page_index,
        )
        for rank in range(2)
    ]

    plan = compose_page_placements(sources, destinations)

    assert {run.source_rank for run in plan} == expected_sources
    assert sum(run.fragment_size * run.fragment_count for run in plan) == 4


def test_adjacent_intersections_coalesce_across_copy_run_boundaries():
    split_source = _mapping(
        16,
        16,
        CopyRun(0, 0, 4, 1, 4, 4),
        CopyRun(4, 4, 8, 1, 8, 8),
        CopyRun(12, 12, 4, 1, 4, 4),
    )

    plan = compose_page_placements(
        [_placement(4, 7, split_source)],
        [_placement(5, 8, _identity(16))],
    )

    assert plan == (TransferRun(4, 5, 7, 8, 0, 0, 16, 1, 16, 16),)


def test_duplicate_elected_source_bytes_are_rejected():
    source = _placement(0, 0, _identity(8))
    duplicate = _placement(1, 1, _identity(8))
    destination = _placement(2, 2, _identity(8))

    with pytest.raises(ValueError, match="duplicate|overlapping"):
        compose_page_placements([source, duplicate], [destination])


def test_source_local_aliases_are_rejected():
    mapping = _identity(8)
    aliases = [
        _placement(0, 7, mapping, canonical_base=0),
        _placement(0, 7, mapping, canonical_base=8),
    ]

    with pytest.raises(ValueError, match="source page.*overlapping local bytes"):
        compose_page_placements(aliases, [_placement(1, 9, _identity(16))])


def test_incompatible_canonical_spaces_are_rejected():
    with pytest.raises(ValueError, match="incompatible canonical spaces"):
        compose_page_placements(
            [_placement(0, 0, _identity(8), canonical_space_id="fp8:model-a")],
            [_placement(1, 0, _identity(8), canonical_space_id="bf16:model-a")],
        )


def test_opaque_mappings_require_the_exact_same_layout_signature():
    source = _mapping(
        8,
        8,
        CopyRun(0, 0, 8, 1, 8, 8),
        is_opaque=True,
        opaque_layout_signature="opaque:tp8:hnd",
    )
    destination = _mapping(
        8,
        8,
        CopyRun(0, 0, 8, 1, 8, 8),
        is_opaque=True,
        opaque_layout_signature="opaque:tp4:hnd",
    )

    with pytest.raises(ValueError, match="identical opaque layout signatures"):
        compose_page_placements(
            [_placement(0, 0, source)],
            [_placement(1, 0, destination)],
        )


def test_source_gap_required_by_destination_is_rejected():
    source_page = _identity(4)
    sources = [
        _placement(0, 0, source_page, canonical_base=0),
        _placement(0, 1, source_page, canonical_base=8),
    ]

    with pytest.raises(ValueError, match="canonical gap"):
        compose_page_placements(sources, [_placement(1, 0, _identity(12))])


def test_destination_local_gap_is_rejected_per_page():
    destination_with_gap = _mapping(
        8,
        8,
        CopyRun(0, 0, 3, 1, 3, 3),
        CopyRun(4, 4, 4, 1, 4, 4),
    )

    with pytest.raises(ValueError, match="local page.*gap"):
        compose_page_placements(
            [_placement(0, 0, _identity(8))],
            [_placement(1, 0, destination_with_gap)],
        )


def test_destination_replication_across_ranks_is_allowed():
    source = _placement(0, 5, _identity(8))
    destinations = [
        _placement(1, 6, _identity(8)),
        _placement(2, 7, _identity(8)),
    ]

    plan = compose_page_placements([source], destinations)

    assert plan == (
        TransferRun(0, 1, 5, 6, 0, 0, 8, 1, 8, 8),
        TransferRun(0, 2, 5, 7, 0, 0, 8, 1, 8, 8),
    )


@pytest.mark.parametrize(
    ("canonical_page_index", "expected_source_rank"),
    [(10, 20), (11, 21)],
)
def test_replica_writer_election_filters_sources(
    canonical_page_index: int,
    expected_source_rank: int,
):
    replicas = [
        _placement(
            20 + writer_index,
            30 + writer_index,
            _identity(8, num_writers=2, writer_index=writer_index),
            canonical_page_index=canonical_page_index,
        )
        for writer_index in range(2)
    ]

    plan = compose_page_placements(
        replicas,
        [_placement(40, 50, _identity(8), canonical_page_index=canonical_page_index)],
    )

    assert len(plan) == 1
    assert plan[0].source_rank == expected_source_rank


def test_direct_plan_has_no_segment_count_cutoff():
    # Deliberately exceed the common 4096-descriptor threshold.  Each source
    # page is a distinct registration, so these fragments cannot be coalesced.
    segment_count = 4097
    one_byte = _identity(1)
    sources = [
        _placement(
            0,
            page_id,
            one_byte,
            canonical_base=page_id,
            canonical_page_index=page_id,
        )
        for page_id in range(segment_count)
    ]
    destination = _placement(1, 0, _identity(segment_count))

    plan = compose_page_placements(sources, [destination])

    assert len(plan) == segment_count
    assert all(run.fragment_size == run.fragment_count == 1 for run in plan)


def _copy_bytes(
    runs: tuple[TransferRun, ...] | list[TransferRun],
) -> set[tuple[int, int, int, int, int, int]]:
    return {
        (
            run.source_rank,
            run.source_page_id,
            run.source_offset + fragment * run.source_stride + byte,
            run.destination_rank,
            run.destination_page_id,
            run.destination_offset + fragment * run.destination_stride + byte,
        )
        for run in runs
        for fragment in range(run.fragment_count)
        for byte in range(run.fragment_size)
    }


def test_streaming_composer_bounds_one_page_fragment_expansion(monkeypatch):
    # One compact page, rather than many windowed pages, is adversarial here.
    # Its canonical order alternates between the two halves of source memory,
    # preventing the intersections from collapsing into one contiguous copy.
    half = 5000
    source_mapping = _mapping(
        2 * half,
        2 * half,
        CopyRun(0, 0, 1, half, 1, 2, 0, 0, 1),
        CopyRun(half, 1, 1, half, 1, 2, 0, half, 1),
    )
    destination_mapping = _identity(2 * half)
    sources = (_placement(0, 7, source_mapping),)
    destinations = (_placement(1, 8, destination_mapping),)
    eager = compose_page_placements(sources, destinations)

    buffered_sizes: list[int] = []
    original_compress = kv_placement_module._compress_copies

    def observe_compress(copies):
        buffered_sizes.append(len(copies))
        return original_compress(copies)

    monkeypatch.setattr(kv_placement_module, "_compress_copies", observe_compress)
    monkeypatch.setattr(
        kv_placement_module,
        "_validate_and_expand",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("streaming composition must not eagerly expand pages")
        ),
    )

    stream = iter_page_placement_transfer_runs(
        sources,
        destinations,
        max_buffered_copy_fragments=4096,
    )
    first = next(stream)
    streamed = (first, *stream)

    assert buffered_sizes == [4096, 4096, 1808]
    assert max(buffered_sizes) == 4096
    assert _copy_bytes(streamed) == _copy_bytes(eager)


def test_streaming_composer_validates_late_gap_before_first_emission():
    sources = (
        _placement(0, 0, _identity(4), canonical_base=0),
        _placement(0, 1, _identity(4), canonical_base=8),
    )
    stream = iter_page_placement_transfer_runs(
        sources,
        (_placement(1, 0, _identity(12)),),
        max_buffered_copy_fragments=2,
    )

    with pytest.raises(ValueError, match="canonical gap"):
        next(stream)


def test_streaming_dcp_window_scans_sources_per_replica_lane_not_page(
    monkeypatch,
):
    page_count = 64
    cp_size = 8
    local_tokens = 4
    page_span = cp_size * local_tokens
    mappings = tuple(
        _cp_mapping(cp_size, rank, local_tokens, 1) for rank in range(cp_size)
    )
    sources = tuple(
        _placement(
            rank,
            page,
            mappings[rank],
            first_token=page * page_span,
            valid_token_count=page_span,
            canonical_page_index=page,
        )
        for page in range(page_count)
        for rank in range(cp_size)
    )
    destinations = tuple(
        _placement(
            10 + rank,
            page,
            mappings[rank],
            first_token=page * page_span,
            valid_token_count=page_span,
            canonical_page_index=page,
        )
        for page in range(page_count)
        for rank in range(cp_size)
    )

    intersection_passes = 0
    original_intersections = kv_placement_module._iter_intersections

    def observe_intersections(*args, **kwargs):
        nonlocal intersection_passes
        intersection_passes += 1
        yield from original_intersections(*args, **kwargs)

    monkeypatch.setattr(
        kv_placement_module,
        "_iter_intersections",
        observe_intersections,
    )
    streamed = tuple(
        iter_page_placement_transfer_runs(
            sources,
            destinations,
            max_buffered_copy_fragments=4096,
        )
    )

    # DCP ranks partition canonical tokens, so all 512 destination placements
    # share one disjoint lane. There is one preflight pass and one emit pass,
    # rather than 2 * 512 complete source scans.
    assert intersection_passes == 2
    assert (
        sum(run.fragment_size * run.fragment_count for run in streamed)
        == page_count * page_span
    )


def test_streaming_composer_randomized_differential():
    seed = 0xC0FFEE
    rng = random.Random(seed)
    for _case in range(64):
        source_cp = rng.choice((1, 2, 3, 4))
        destination_cp = rng.choice((1, 2, 3, 4))
        source_interleave = rng.choice((1, 2, 4))
        destination_interleave = rng.choice((1, 2, 4))
        source_local_tokens = source_interleave * rng.randint(1, 4)
        destination_local_tokens = destination_interleave * rng.randint(1, 4)
        source_span = source_cp * source_local_tokens
        destination_span = destination_cp * destination_local_tokens
        token_count = rng.randint(1, 3 * max(source_span, destination_span) + 1)
        replica_count = rng.choice((1, 1, 2))

        source_mappings = tuple(
            _cp_mapping(
                source_cp,
                rank,
                source_local_tokens,
                source_interleave,
            )
            for rank in range(source_cp)
        )
        destination_mappings = tuple(
            _cp_mapping(
                destination_cp,
                rank,
                destination_local_tokens,
                destination_interleave,
            )
            for rank in range(destination_cp)
        )
        sources = [
            _placement(
                rank,
                page,
                source_mappings[rank],
                first_token=page * source_span,
                valid_token_count=min(
                    source_span,
                    max(0, token_count - page * source_span),
                ),
                canonical_page_index=page,
            )
            for page in range((token_count + source_span - 1) // source_span)
            for rank in range(source_cp)
        ]
        destinations = [
            _placement(
                100 + replica * destination_cp + rank,
                page,
                destination_mappings[rank],
                first_token=page * destination_span,
                valid_token_count=min(
                    destination_span,
                    max(0, token_count - page * destination_span),
                ),
                canonical_page_index=page,
            )
            for replica in range(replica_count)
            for page in range((token_count + destination_span - 1) // destination_span)
            for rank in range(destination_cp)
        ]
        rng.shuffle(sources)
        rng.shuffle(destinations)

        eager = compose_page_placements(sources, destinations)
        buffer_limit = rng.choice((1, 2, 3, 5, 7))
        streamed = tuple(
            iter_page_placement_transfer_runs(
                sources,
                destinations,
                max_buffered_copy_fragments=buffer_limit,
            )
        )
        reordered = tuple(
            iter_page_placement_transfer_runs(
                tuple(reversed(sources)),
                tuple(reversed(destinations)),
                max_buffered_copy_fragments=buffer_limit,
            )
        )

        assert _copy_bytes(streamed) == _copy_bytes(eager), (
            seed,
            _case,
        )
        assert streamed == reordered, (seed, _case)
        assert (
            sum(run.fragment_size * run.fragment_count for run in streamed)
            == token_count * replica_count
        )


def test_format_manifest_is_frozen_and_json_round_trips():
    manifest = _format_manifest(
        _group(),
        KVGroupFormat(
            group_id=1,
            semantic_id="mamba-group-1",
            kind="mamba",
            layer_names=("layers.1.mixer",),
            canonical_page_token_span=512,
            dtype="float16",
            canonical_page_size_bytes=8192,
            format_id="v1-mamba-gdn-state",
        ),
    )

    payload = json.loads(json.dumps(manifest.to_dict()))

    assert KVFormatManifest.from_dict(payload) == manifest
    assert manifest.group(1).kind == "mamba"
    assert manifest.semantic_group("mamba-group-1").group_id == 1
    with pytest.raises(FrozenInstanceError):
        manifest.version = 2


def test_format_manifest_rejects_duplicate_group_and_layer_identity():
    with pytest.raises(ValueError, match="group_id values must be unique"):
        _format_manifest(_group(), _group())

    with pytest.raises(ValueError, match="layer names must be unique"):
        _format_manifest(_group(), _group(1))

    duplicate_semantic_id = _group(
        1, ("layers.1.attn",), semantic_id="attention-group-0"
    )
    with pytest.raises(ValueError, match="semantic_id values must be unique"):
        _format_manifest(_group(), duplicate_semantic_id)

    payload = _format_manifest().to_dict()
    payload["unversioned_extension"] = True
    with pytest.raises(ValueError, match="unknown=.*unversioned_extension"):
        KVFormatManifest.from_dict(payload)


@pytest.mark.parametrize("version", [True, 1.0])
def test_protocol_versions_require_an_actual_integer(version):
    with pytest.raises(ValueError, match="version must be a non-negative integer"):
        KVFormatManifest(
            version=version,
            model_fingerprint="model-v1",
            groups=(_group(),),
        )
    with pytest.raises(ValueError, match="version must be a non-negative integer"):
        _rank_manifest(version=version)


def test_rank_placement_manifest_records_all_topology_axes_and_round_trips():
    manifest = _rank_manifest()

    payload = json.loads(json.dumps(manifest.to_dict()))

    assert RankPlacementManifest.from_dict(payload) == manifest
    assert manifest.mapping_for("layers.0.attn") == _rank_mapping()
    assert manifest.dp_rank == 2
    # EP is retained for diagnostics; page mappings contain no EP coordinate.
    assert manifest.ep_rank == 6
    manifest.validate_format(_rank_format_manifest())


def test_canonical_space_ignores_local_group_and_page_geometry():
    local = _format_manifest(_group())
    remote_group = KVGroupFormat(
        group_id=7,
        semantic_id="attention-group-0",
        kind="mla",
        layer_names=("layers.0.attn",),
        canonical_page_token_span=256,
        dtype="fp8_e4m3fn",
        canonical_page_size_bytes=2048,
        format_id="v1-mla-latent-mxfp8",
        quantization="mxfp8",
        scale_dtype="fp8_e8m0fnu",
        scale_granularity="32-elements",
    )
    remote = _format_manifest(remote_group)

    assert local.fingerprint() != remote.fingerprint()
    assert local.canonical_space_id("attention-group-0") == (
        remote.canonical_space_id("attention-group-0")
    )


def test_canonical_space_ignores_layer_enumeration_order():
    first = _format_manifest(_group(0, ("layers.0.attn", "layers.1.attn")))
    second = _format_manifest(
        _group(
            9,
            ("layers.1.attn", "layers.0.attn"),
            semantic_id="attention-group-0",
        )
    )

    assert first.fingerprint() != second.fingerprint()
    assert first.canonical_space_id("attention-group-0") == (
        second.canonical_space_id("attention-group-0")
    )


def test_rank_placement_manifest_rejects_invalid_coordinates_and_mappings():
    with pytest.raises(ValueError, match="dcp_rank"):
        _rank_manifest(dcp_size=2, dcp_rank=2)

    duplicate_mappings = (
        LayerPageMapping("layers.0.attn", 0, "attention-group-0", _rank_mapping()),
        LayerPageMapping("layers.0.attn", 0, "attention-group-0", _rank_mapping()),
    )
    with pytest.raises(ValueError, match="mapping layer names must be unique"):
        _rank_manifest(mappings=duplicate_mappings)

    malformed = _mapping(8, 8, CopyRun(0, 0, 7, 1, 7, 7))
    with pytest.raises(ValueError, match="local page.*gap"):
        LayerPageMapping("layers.0.attn", 0, "attention-group-0", malformed)

    with pytest.raises(ValueError, match="inside layer_range"):
        _rank_manifest(
            mappings=(
                LayerPageMapping(
                    "layers.4.attn", 4, "attention-group-0", _rank_mapping()
                ),
            )
        )


def test_compact_affine_mapping_has_no_expanded_fragment_limit(monkeypatch):
    # This is one past the former logical-fragment ceiling. The compact wire
    # shape is still one CopyRun, so fragmentation must affect only the number
    # of direct batches, not admission or validation memory.
    fragment_count = 1_048_577
    mapping = _mapping(
        fragment_count,
        fragment_count,
        CopyRun(0, 0, 1, fragment_count, 1, 1),
        canonical_token_span=fragment_count,
        canonical_region_token_strides=((0, 1),),
    )
    layer_mapping = LayerPageMapping("layers.0.attn", 0, "attention-group-0", mapping)
    manifest = _rank_manifest(mappings=(layer_mapping,))

    payload = json.loads(json.dumps(manifest.to_dict()))
    assert RankPlacementManifest.from_dict(payload) == manifest

    monkeypatch.setattr(
        kv_placement_module,
        "_validate_and_expand",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("streaming planning must not expand the affine run")
        ),
    )
    source = _placement(
        0,
        0,
        mapping,
        first_token=0,
        valid_token_count=1,
    )
    destination = _placement(
        1,
        0,
        mapping,
        first_token=0,
        valid_token_count=1,
    )

    assert tuple(
        iter_page_placement_transfer_runs(
            (source,),
            (destination,),
            max_buffered_copy_fragments=1,
        )
    ) == (
        TransferRun(
            source_rank=0,
            destination_rank=1,
            source_page_id=0,
            destination_page_id=0,
            source_offset=0,
            destination_offset=0,
            fragment_size=1,
            fragment_count=1,
            source_stride=1,
            destination_stride=1,
        ),
    )


def test_compact_periodic_mapping_validation_never_expands_fragments(monkeypatch):
    fragment_count = 10**12
    mapping = _mapping(
        2 * fragment_count,
        2 * fragment_count,
        CopyRun(0, 0, 1, fragment_count, 1, 2),
        CopyRun(fragment_count, 1, 1, fragment_count, 1, 2),
        canonical_token_span=2 * fragment_count,
        canonical_region_token_strides=((0, 1),),
    )

    def reject_fragment_expansion(*_args, **_kwargs):
        raise AssertionError("compact validation must not enqueue fragments")

    monkeypatch.setattr(
        kv_placement_module.heapq, "heappush", reject_fragment_expansion
    )

    layer_mapping = LayerPageMapping("layers.0.attn", 0, "attention-group-0", mapping)
    manifest = _rank_manifest(mappings=(layer_mapping,))
    assert RankPlacementManifest.from_dict(manifest.to_dict()) == manifest


def test_connector_capabilities_are_data_not_fragmentation_policy():
    capabilities = ConnectorCapabilities(
        contiguous_copy=True,
        strided_copy=True,
        scatter_gather=True,
        gpu_pack_unpack=False,
        supports_read=True,
        supports_write=False,
        max_segments_per_batch=1,
    )

    payload = json.loads(json.dumps(capabilities.to_dict()))

    assert ConnectorCapabilities.from_dict(payload) == capabilities
    assert capabilities.max_segments_per_batch == 1
    with pytest.raises(ValueError, match="reads, writes, or both"):
        ConnectorCapabilities(True, True, True, False, False, False)
    with pytest.raises(ValueError, match="max_segments_per_batch"):
        ConnectorCapabilities(True, True, True, False, True, False, 0)


def test_kv_ranges_carry_partial_valid_coverage_and_are_unique():
    manifest = _format_manifest(
        _group(),
        _group(1, ("layers.1.attn",)),
    )
    padded = KVRange(
        semantic_group_id="attention-group-0",
        first_token=32768,
        token_count=4096,
        valid_token_count=4073,
    )
    second = KVRange("attention-group-1", 32768, 4096, 4096)
    sparse_tail = KVRange("attention-group-0", 40960, 512, 500)

    assert padded.end_token == 36864
    assert padded.valid_end_token == 36841
    assert KVRange.from_dict(json.loads(json.dumps(padded.to_dict()))) == padded
    assert validate_kv_ranges([sparse_tail, second, padded], manifest) == (
        padded,
        sparse_tail,
        second,
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_kv_ranges(
            [padded, KVRange("attention-group-0", 36863, 8, 8)], manifest
        )
    with pytest.raises(ValueError, match="unknown KV groups"):
        validate_kv_ranges([KVRange("unknown-group", 0, 1, 1)], manifest)
    with pytest.raises(ValueError, match="must not exceed"):
        KVRange("attention-group-0", 0, 7, 8)
