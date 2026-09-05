# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.canonical_mapping import (
    _layer_mapping,
    _opaque_cache_layout_fingerprint,
    _opaque_fallback_mapping,
    _RankContext,
    _verify_tiling,
    derive_canonical_mappings,
    derive_rank_canonical_mappings,
    native_vllm_dcp_rank,
)
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    CopyRun,
    PagePlacement,
    compose_page_placements,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVQuantMode,
    MambaSpec,
    MLAAttentionSpec,
)

NUM_BLOCKS = 3


def _ctx(
    rank,
    tp=1,
    dcp=1,
    pcp=1,
    interleave=1,
    total=None,
    heads=2,
    dcp_rank=None,
):
    tp_rank = rank % tp
    pcp_rank = rank // tp
    return _RankContext(
        tp_size=tp,
        dcp_size=dcp,
        pcp_size=pcp,
        interleave=interleave,
        total_kv_heads=heads * tp if total is None else total,
        rank=rank,
        dcp_rank=(
            native_vllm_dcp_rank(
                tp_size=tp,
                tp_rank=tp_rank,
                dcp_size=dcp,
                pcp_size=pcp,
                pcp_rank=pcp_rank,
            )
            if dcp_rank is None
            else dcp_rank
        ),
    )


def _full_spec(num_kv_heads: int = 2, **kwargs) -> FullAttentionSpec:
    # block_size=4, head_dim=64, int8
    return TritonAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=4,
            num_kv_heads=num_kv_heads,
            head_size=64,
            dtype=torch.int8,
            **kwargs,
        )
    )


def _mla_spec(**kwargs) -> MLAAttentionSpec:
    # page = 4 * 64 = 256B, one 64B latent row per token
    return MLAAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=64, dtype=torch.int8, **kwargs
    )


def _split_nhd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    )


def _split_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        2,
        spec.num_kv_heads,
        spec.block_size,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 1, 3, 2, 4)


def _packed_nhd_cache(spec) -> torch.Tensor:
    """Logical (num_blocks, heads, block_size, 2 * head_size) over an NHD
    physical layout — the FlashAttention/FlashInfer/Triton/Flex form."""
    return torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        spec.num_kv_heads,
        2 * spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3)


def _packed_hnd_cache(spec) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        spec.num_kv_heads,
        spec.block_size,
        2 * spec.head_size,
        dtype=torch.int8,
    )


def _gdn_spec(tp: int, *, recover_ssm: bool = False) -> MambaSpec:
    local_heads = 8 // tp
    head_dim = 4
    shapes: tuple[tuple[int, ...], ...] = (
        (3 * local_heads * head_dim, 3),
        (local_heads, head_dim, head_dim),
    )
    dtypes: tuple[torch.dtype, ...] = (torch.uint8, torch.float32)
    if recover_ssm:
        shapes += (
            (local_heads, 4, head_dim),
            (local_heads, 4, 2 * head_dim),
        )
        dtypes += (torch.float32, torch.uint8)
    return MambaSpec(
        block_size=32,
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )


def _mamba_cache(spec: MambaSpec) -> torch.Tensor:
    raw = torch.empty(NUM_BLOCKS * spec.page_size_bytes, dtype=torch.int8)
    return torch.as_strided(
        raw,
        (NUM_BLOCKS, 1, 1, spec.real_page_size_bytes),
        (
            spec.page_size_bytes,
            spec.real_page_size_bytes,
            spec.real_page_size_bytes,
            1,
        ),
    )


CACHE_BUILDERS = {
    "split_nhd": _split_nhd_cache,
    "split_hnd": _split_hnd_cache,
    "packed_nhd": _packed_nhd_cache,
    "packed_hnd": _packed_hnd_cache,
}


def _try_mapping(spec, kv_cache, ctx) -> CanonicalPageMapping | None:
    return _layer_mapping(spec, kv_cache, NUM_BLOCKS, ctx)


def _mapping(spec, kv_cache, ctx) -> CanonicalPageMapping:
    mapping = _try_mapping(spec, kv_cache, ctx)
    assert mapping is not None
    return mapping


def _triples(runs: tuple[CopyRun, ...]) -> list[tuple[int, int, int]]:
    """Expand runs into compact canonical-storage copy triples."""
    out = []
    for run in runs:
        for i in range(run.num_fragments):
            out.append(
                (
                    run.local_offset + i * run.local_stride,
                    run.storage_offset + i * run.storage_stride,
                    run.fragment_size,
                )
            )
    return out


def _semantic_triples(
    runs: tuple[CopyRun, ...],
) -> list[tuple[int, int, int, int]]:
    """Expand page-span-independent connector coordinates."""
    return [
        (
            run.canonical_region,
            run.local_offset + i * run.local_stride,
            run.canonical_offset + i * run.canonical_stride,
            run.fragment_size,
        )
        for run in runs
        for i in range(run.num_fragments)
    ]


def _semantic_page_values(placement: PagePlacement) -> list[tuple[int, int]]:
    """Materialize symbolic canonical bytes in one endpoint-local page."""
    mapping = placement.mapping
    assert placement.first_token is not None
    values: list[tuple[int, int] | None] = [None] * mapping.local_page_size_bytes
    for run in mapping.runs:
        page_base = placement.first_token * mapping.token_stride(run.canonical_region)
        for fragment_index in range(run.num_fragments):
            local_start = run.local_offset + fragment_index * run.local_stride
            canonical_start = (
                page_base + run.canonical_offset + fragment_index * run.canonical_stride
            )
            for byte_index in range(run.fragment_size):
                assert values[local_start + byte_index] is None
                values[local_start + byte_index] = (
                    run.canonical_region,
                    canonical_start + byte_index,
                )
    assert all(value is not None for value in values)
    return [value for value in values if value is not None]


def _assert_symbolic_plan_copies_layout(
    sources: list[PagePlacement], destinations: list[PagePlacement]
) -> None:
    """Execute a composed plan and compare the destination byte semantics."""
    plan = compose_page_placements(sources, destinations)
    source_pages = {
        (placement.rank, placement.local_page_id): _semantic_page_values(placement)
        for placement in sources
    }
    actual: dict[tuple[int, int], list[tuple[int, int] | None]] = {
        (placement.rank, placement.local_page_id): [None]
        * placement.mapping.local_page_size_bytes
        for placement in destinations
    }
    expected = {
        (placement.rank, placement.local_page_id): _semantic_page_values(placement)
        for placement in destinations
    }
    for run in plan:
        source = source_pages[(run.source_rank, run.source_page_id)]
        destination = actual[(run.destination_rank, run.destination_page_id)]
        for fragment_index in range(run.fragment_count):
            source_start = run.source_offset + fragment_index * run.source_stride
            destination_start = (
                run.destination_offset + fragment_index * run.destination_stride
            )
            destination[destination_start : destination_start + run.fragment_size] = (
                source[source_start : source_start + run.fragment_size]
            )
    assert actual == expected


# ---------------------------------------------------------------------------
# TP-only placement (byte-compatible with the uniform interleave layout)
# ---------------------------------------------------------------------------


def test_split_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _split_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert mapping.canonical_page_size_bytes == 4 * 1024
    assert mapping.parallelism_agnostic
    k_dst = [256, 768, 1280, 1792]
    assert _triples(mapping.runs) == [
        (local, canonical, 128)
        for local, canonical in zip(
            [0, 128, 256, 384, 512, 640, 768, 896],
            k_dst + [2048 + o for o in k_dst],
        )
    ]
    # Heads are sharded, not replicated: this rank writes every block
    assert mapping.num_writers == 1


def test_packed_nhd_placement_rank2_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_nhd_cache(spec), _ctx(rank=2, tp=4))
    assert _triples(mapping.runs) == [
        (0, 512, 256),
        (256, 1536, 256),
        (512, 2560, 256),
        (768, 3584, 256),
    ]


def test_packed_hnd_placement_rank1_of_4():
    spec = _full_spec()
    mapping = _mapping(spec, _packed_hnd_cache(spec), _ctx(rank=1, tp=4))
    # Compact storage remains two contiguous head-major regions. Semantic
    # token-major coordinates retain one direct fragment per token.
    assert _triples(mapping.runs) == [
        (local, canonical, 128)
        for local, canonical in zip(
            range(0, 1024, 128),
            range(1024, 2048, 128),
        )
    ]
    # Direct-transfer coordinates are token-major even though compact offload
    # storage remains head-major. This lets NHD and HND endpoints intersect.
    assert _semantic_triples(mapping.runs) == [
        (0, 0, 256, 128),
        (0, 128, 1280, 128),
        (0, 256, 2304, 128),
        (0, 384, 3328, 128),
        (0, 512, 384, 128),
        (0, 640, 1408, 128),
        (0, 768, 2432, 128),
        (0, 896, 3456, 128),
    ]


def test_split_hnd_uses_layout_independent_token_major_coordinates():
    spec = _full_spec()
    mapping = _mapping(spec, _split_hnd_cache(spec), _ctx(rank=1, tp=4))

    assert _semantic_triples(mapping.runs) == [
        (0, 0, 128, 64),
        (0, 64, 640, 64),
        (0, 128, 1152, 64),
        (0, 192, 1664, 64),
        (0, 256, 192, 64),
        (0, 320, 704, 64),
        (0, 384, 1216, 64),
        (0, 448, 1728, 64),
        (1, 512, 128, 64),
        (1, 576, 640, 64),
        (1, 640, 1152, 64),
        (1, 704, 1664, 64),
        (1, 768, 192, 64),
        (1, 832, 704, 64),
        (1, 896, 1216, 64),
        (1, 960, 1728, 64),
    ]


@pytest.mark.parametrize(
    (
        "source_form",
        "destination_form",
        "source_tp",
        "source_dcp",
        "dest_tp",
        "dest_dcp",
    ),
    [
        ("packed_nhd", "packed_hnd", 4, 2, 2, 1),
        ("packed_hnd", "packed_nhd", 2, 1, 4, 2),
        ("split_nhd", "split_hnd", 4, 2, 2, 1),
        ("split_hnd", "split_nhd", 2, 1, 4, 2),
    ],
)
def test_nhd_hnd_compose_directly_across_tp_and_dcp(
    source_form: str,
    destination_form: str,
    source_tp: int,
    source_dcp: int,
    dest_tp: int,
    dest_dcp: int,
):
    """Layout conversion is a segmented copy, including changed CP geometry."""
    total_heads = 2
    total_tokens = 8

    def endpoint(
        form: str, tp: int, dcp: int, *, destination: bool
    ) -> list[PagePlacement]:
        spec = _full_spec(num_kv_heads=max(1, total_heads // tp))
        cache = CACHE_BUILDERS[form](spec)
        rank_mappings = [
            _mapping(
                spec,
                cache,
                _ctx(rank, tp=tp, dcp=dcp, total=total_heads),
            )
            for rank in range(tp)
        ]
        page_span = rank_mappings[0].canonical_token_span
        assert page_span is not None and total_tokens % page_span == 0
        rank_base = 100 if destination else 0
        return [
            PagePlacement(
                rank=rank_base + rank,
                local_page_id=page_index,
                canonical_page_index=page_index,
                mapping=mapping,
                canonical_space_id="mha-token-major",
                first_token=page_index * page_span,
            )
            for page_index in range(total_tokens // page_span)
            for rank, mapping in enumerate(rank_mappings)
        ]

    _assert_symbolic_plan_copies_layout(
        endpoint(source_form, source_tp, source_dcp, destination=False),
        endpoint(destination_form, dest_tp, dest_dcp, destination=True),
    )


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_single_rank_compact_storage_is_contiguous(form):
    spec = _full_spec()
    mapping = _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank=0))
    assert mapping.canonical_page_size_bytes == 1024
    triples = _triples(mapping.runs)
    assert sum(size for _, _, size in triples) == 1024
    assert all(
        left_local + left_size == right_local
        and left_canonical + left_size == right_canonical
        for (left_local, left_canonical, left_size), (
            right_local,
            right_canonical,
            _,
        ) in zip(triples, triples[1:])
    )


# ---------------------------------------------------------------------------
# Replication and writer election
# ---------------------------------------------------------------------------


def test_gqa_replicated_heads_rotate_writer():
    # total 2 KV heads on tp=4: replication factor 2, head shard = rank // 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, total=2)  # noqa: E731
    rank2 = _mapping(spec, cache, ctx(2))
    rank3 = _mapping(spec, cache, ctx(3))
    assert rank2.canonical_page_size_bytes == 2 * 512
    # K region: head shard 1 at 64B offsets within 128B token rows
    assert _triples(rank2.runs)[:4] == [
        (0, 64, 64),
        (64, 192, 64),
        (128, 320, 64),
        (192, 448, 64),
    ]
    # Same head shard, so identical bytes: the two take alternate blocks
    assert rank3.runs == rank2.runs
    assert (rank2.is_writer(0), rank3.is_writer(0)) == (True, False)
    assert (rank2.is_writer(1), rank3.is_writer(1)) == (False, True)
    _verify_tiling("gqa", [_mapping(spec, cache, ctx(r)) for r in range(4)])


def test_mla_replicas_rotate_writer():
    spec = _mla_spec()
    rank0 = _mapping(spec, None, _ctx(rank=0, tp=2))
    rank1 = _mapping(spec, None, _ctx(rank=1, tp=2))
    # Latent pages are stored once per block, not once per rank
    assert rank0.canonical_page_size_bytes == 256
    assert _triples(rank0.runs) == [(0, 0, 256)]
    assert rank1.runs == rank0.runs
    assert (rank0.is_writer(0), rank1.is_writer(0)) == (True, False)
    assert (rank0.is_writer(1), rank1.is_writer(1)) == (False, True)


def test_gdn_state_composes_across_tp_and_ignores_dcp(monkeypatch):
    """GDN components, not whole rank pages, form the canonical state."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    source_spec = _gdn_spec(8, recover_ssm=True)
    source_spec = replace(
        source_spec,
        page_size_padded=source_spec.real_page_size_bytes + 64,
    )
    source_cache = _mamba_cache(source_spec)
    sources = [
        _mapping(source_spec, source_cache, _ctx(rank, tp=8, dcp=8))
        for rank in range(8)
    ]
    sources_without_dcp = [
        _mapping(source_spec, source_cache, _ctx(rank, tp=8)) for rank in range(8)
    ]
    destination_spec = _gdn_spec(1, recover_ssm=True)
    destination_spec = replace(
        destination_spec,
        page_size_padded=destination_spec.real_page_size_bytes + 128,
    )
    destination = _mapping(
        destination_spec,
        _mamba_cache(destination_spec),
        _ctx(0),
    )

    assert sources == sources_without_dcp
    assert all(mapping.canonical_token_span is None for mapping in sources)
    assert {run.canonical_region for run in sources[0].runs} == set(range(6))
    assert sources[0].canonical_page_size_bytes == (
        destination.canonical_page_size_bytes
    )
    _verify_tiling("gdn-state", sources)

    reference = bytes(
        (13 + 29 * index) % 256
        for index in range(destination.canonical_page_size_bytes)
    )
    source_pages = [_load_one(mapping, reference) for mapping in sources]
    assert (
        _store_all(
            sources,
            source_pages,
            destination.canonical_page_size_bytes,
        )
        == reference
    )
    assert _load_one(destination, reference) == reference

    runs = compose_page_placements(
        tuple(
            PagePlacement(
                rank=rank,
                local_page_id=7,
                canonical_page_index=0,
                mapping=mapping,
                canonical_space_id="test:gdn-state-v1",
            )
            for rank, mapping in enumerate(sources)
        ),
        (
            PagePlacement(
                rank=0,
                local_page_id=11,
                canonical_page_index=0,
                mapping=destination,
                canonical_space_id="test:gdn-state-v1",
            ),
        ),
    )
    assert {run.source_rank for run in runs} == set(range(8))
    assert {run.destination_rank for run in runs} == {0}
    destination_page = bytearray(destination.local_page_size_bytes)
    for run in runs:
        for fragment_index in range(run.fragment_count):
            source_offset = run.source_offset + fragment_index * run.source_stride
            destination_offset = (
                run.destination_offset + fragment_index * run.destination_stride
            )
            destination_page[
                destination_offset : destination_offset + run.fragment_size
            ] = source_pages[run.source_rank][
                source_offset : source_offset + run.fragment_size
            ]
    assert bytes(destination_page) == reference


def test_gdn_state_pcp_replicas_rotate_without_changing_bytes(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    spec = _gdn_spec(2)
    cache = _mamba_cache(spec)
    mappings = [_mapping(spec, cache, _ctx(rank, tp=2, pcp=2)) for rank in range(4)]

    assert mappings[0].runs == mappings[2].runs
    assert mappings[1].runs == mappings[3].runs
    assert [mapping.writer_index for mapping in mappings] == [0, 0, 1, 1]
    assert [mapping.is_writer(0) for mapping in mappings] == [True, True, False, False]
    assert [mapping.is_writer(1) for mapping in mappings] == [False, False, True, True]
    _verify_tiling("gdn-pcp", mappings)


def test_tp_replicated_recurrent_state_uses_one_whole_page():
    spec = MambaSpec(
        block_size=32,
        shapes=((12, 3),),
        dtypes=(torch.uint8,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        mamba_cache_mode="align",
        tp_replicated=True,
    )
    cache = _mamba_cache(spec)
    mappings = [_mapping(spec, cache, _ctx(rank, tp=4, dcp=2)) for rank in range(4)]

    assert all(mapping.runs == mappings[0].runs for mapping in mappings)
    assert _triples(mappings[0].runs) == [(0, 0, spec.real_page_size_bytes)]
    assert all(mapping.parallelism_agnostic for mapping in mappings)
    assert [mapping.num_writers for mapping in mappings] == [4] * 4
    assert [mapping.writer_index for mapping in mappings] == list(range(4))
    _verify_tiling("replicated-state", mappings)


def test_gdn_state_mapper_fails_closed_for_uncertified_layout(monkeypatch):
    spec = _gdn_spec(2, recover_ssm=True)
    cache = _mamba_cache(spec)
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "SD",
    )
    assert _try_mapping(spec, cache, _ctx(0, tp=2, dcp=2)) is None

    malformed = replace(spec, shapes=(*spec.shapes[:3], (4, 4, 7)))
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    assert (
        _try_mapping(
            malformed,
            _mamba_cache(malformed),
            _ctx(0, tp=2, dcp=2),
        )
        is None
    )


# ---------------------------------------------------------------------------
# DCP / PCP token sharding
# ---------------------------------------------------------------------------


def test_dcp_interleaves_tokens_within_replicas():
    # tp=4, dcp=2, total 2 KV heads: head shard = rank // 2, cp rank = rank % 2
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    ctx = lambda rank: _ctx(rank, tp=4, dcp=2, total=2)  # noqa: E731
    per_rank = [_mapping(spec, cache, ctx(rank)) for rank in range(4)]
    assert all(m is not None for m in per_rank)
    # 8 canonical tokens x 2 heads x 64B per region
    assert per_rank[0].canonical_page_size_bytes == 2048
    assert not per_rank[0].parallelism_agnostic
    # rank 2 = head shard 1, cp rank 0: K tokens 0,2,4,6 at head offset 64
    assert _triples(per_rank[2].runs)[:4] == [
        (0, 64, 64),
        (64, 320, 64),
        (128, 576, 64),
        (192, 832, 64),
    ]
    # every rank contributes (dcp == replication: no residual replicas)
    assert all(m.num_writers == 1 for m in per_rank)
    _verify_tiling("dcp", per_rank)


def test_mla_dcp_shards_latent_tokens():
    spec = _mla_spec()
    ctx = lambda rank: _ctx(rank, tp=2, dcp=2)  # noqa: E731
    rank0 = _mapping(spec, None, ctx(0))
    rank1 = _mapping(spec, None, ctx(1))
    assert rank0.canonical_page_size_bytes == 512
    assert _triples(rank0.runs) == [(o, 2 * o, 64) for o in (0, 64, 128, 192)]
    assert _triples(rank1.runs) == [(o, 2 * o + 64, 64) for o in (0, 64, 128, 192)]
    _verify_tiling("mla-dcp", [rank0, rank1])


def test_pcp_replicates_tp_head_shards():
    # PCP partitions forward compute, but all-gathered cache inputs leave the
    # persistent TP placement fully replicated across PCP ranks.
    spec = _full_spec(num_kv_heads=1)
    cache = _packed_nhd_cache(spec)
    tp_only = [_mapping(spec, cache, _ctx(rank, tp=2, total=2)) for rank in range(2)]
    per_rank = [
        _mapping(spec, cache, _ctx(rank, tp=2, pcp=2, total=2)) for rank in range(4)
    ]
    assert [mapping.runs for mapping in per_rank] == [
        tp_only[0].runs,
        tp_only[1].runs,
        tp_only[0].runs,
        tp_only[1].runs,
    ]
    assert all(mapping.canonical_token_span == spec.block_size for mapping in per_rank)
    assert all(mapping.num_writers == 2 for mapping in per_rank)
    assert [mapping.writer_index for mapping in per_rank] == [0, 0, 1, 1]
    _verify_tiling("pcp", per_rank)


def test_interleave_chunks_stay_contiguous():
    # interleave=2: chunks of 2 tokens alternate between the 2 cp ranks and
    # coalesce into one contiguous fragment per chunk
    spec = _mla_spec()
    mapping = _mapping(spec, None, _ctx(rank=0, tp=2, dcp=2, interleave=2))
    assert _triples(mapping.runs) == [(0, 0, 128), (128, 256, 128)]
    _verify_tiling(
        "interleave",
        [
            _mapping(spec, None, _ctx(rank, tp=2, dcp=2, interleave=2))
            for rank in range(2)
        ],
    )


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_all_ranks_tile_canonical_page(form):
    spec = _full_spec()
    per_rank = [
        _mapping(spec, CACHE_BUILDERS[form](spec), _ctx(rank, tp=4))
        for rank in range(4)
    ]
    _verify_tiling("layer", per_rank)


# ---------------------------------------------------------------------------
# Byte-level round trips
# ---------------------------------------------------------------------------


def _store_all(mappings, pages, size: int, block_id: int = 0) -> bytes:
    buf = bytearray(size)
    for mapping, page in zip(mappings, pages):
        if not mapping.is_writer(block_id):
            continue
        for local, canonical, n in _triples(mapping.runs):
            buf[canonical : canonical + n] = page[local : local + n]
    return bytes(buf)


def _load_one(mapping, canonical_bytes: bytes) -> bytes:
    page = bytearray(mapping.local_page_size_bytes)
    for local, canonical, n in _triples(mapping.runs):
        page[local : local + n] = canonical_bytes[canonical : canonical + n]
    return bytes(page)


@pytest.mark.parametrize("form", sorted(CACHE_BUILDERS))
def test_cross_tp_store_load(form):
    """Bytes stored under one TP size are the bytes another TP size loads."""
    total_heads, canonical_size = 8, 4096
    reference = bytes((7 + 31 * i) % 256 for i in range(canonical_size))

    def mappings_at(tp: int):
        spec = _full_spec(num_kv_heads=total_heads // tp)
        cache = CACHE_BUILDERS[form](spec)
        return [
            _mapping(spec, cache, _ctx(rank, tp=tp, total=total_heads))
            for rank in range(tp)
        ]

    for tp in (4, 2, 1):
        mappings = mappings_at(tp)
        assert all(m is not None for m in mappings)
        pages = [_load_one(m, reference) for m in mappings]
        assert _store_all(mappings, pages, canonical_size) == reference


def test_cp_round_trip():
    # tp=4 / dcp=2 / 2 KV heads: 4 workers jointly hold one canonical page
    spec = _full_spec(num_kv_heads=1)
    cache = _split_nhd_cache(spec)
    mappings = [
        _mapping(spec, cache, _ctx(rank, tp=4, dcp=2, total=2)) for rank in range(4)
    ]
    reference = bytes((3 + 17 * i) % 256 for i in range(2048))
    pages = [_load_one(m, reference) for m in mappings]
    assert _store_all(mappings, pages, 2048) == reference


def test_replica_rotation_round_trip():
    """Whichever replica a block elects reproduces the same canonical page."""
    spec = _mla_spec()
    mappings = [_mapping(spec, None, _ctx(rank, tp=2)) for rank in range(2)]
    reference = bytes((5 + 11 * i) % 256 for i in range(256))
    pages = [_load_one(m, reference) for m in mappings]
    for block_id in (0, 1):
        assert _store_all(mappings, pages, 256, block_id) == reference


def test_pcp_replica_rotation_round_trip():
    """Writer election rotates across TP and PCP replicas of persistent KV."""
    spec = _mla_spec()
    mappings = [_mapping(spec, None, _ctx(rank, tp=2, pcp=2)) for rank in range(4)]
    reference = bytes((17 + 23 * index) % 256 for index in range(256))
    pages = [_load_one(mapping, reference) for mapping in mappings]

    assert [mapping.num_writers for mapping in mappings] == [4] * 4
    assert [mapping.writer_index for mapping in mappings] == [0, 1, 2, 3]
    for block_id in range(4):
        assert _store_all(mappings, pages, 256, block_id) == reference


@pytest.mark.parametrize(
    ("source_topology", "destination_topology"),
    [
        ((2, 1, 2), (2, 1, 1)),
        ((2, 1, 1), (2, 1, 2)),
        ((2, 1, 2), (2, 2, 1)),
        ((2, 2, 1), (2, 1, 2)),
    ],
    ids=("pcp-to-tp", "tp-to-pcp", "pcp-to-dcp", "dcp-to-pcp"),
)
@pytest.mark.parametrize("attention_kind", ("mla", "mha"))
def test_pcp_replica_composes_directly_with_tp_and_dcp(
    source_topology: tuple[int, int, int],
    destination_topology: tuple[int, int, int],
    attention_kind: str,
):
    """PCP replica pages copy directly to and from TP/DCP byte placement."""
    spec = _mla_spec() if attention_kind == "mla" else _full_spec(num_kv_heads=1)
    cache = None if attention_kind == "mla" else _packed_nhd_cache(spec)
    total_kv_heads = None if attention_kind == "mla" else 1
    total_tokens = 8

    def endpoint(
        topology: tuple[int, int, int], *, destination: bool
    ) -> list[PagePlacement]:
        tp, dcp, pcp = topology
        rank_mappings = [
            _mapping(
                spec,
                cache,
                _ctx(
                    rank,
                    tp=tp,
                    dcp=dcp,
                    pcp=pcp,
                    total=total_kv_heads,
                ),
            )
            for rank in range(tp * pcp)
        ]
        page_span = rank_mappings[0].canonical_token_span
        assert page_span is not None and total_tokens % page_span == 0
        rank_base = 100 if destination else 0
        return [
            PagePlacement(
                rank=rank_base + rank,
                local_page_id=page_index,
                canonical_page_index=page_index,
                mapping=mapping,
                canonical_space_id=f"{attention_kind}-token-major",
                first_token=page_index * page_span,
            )
            for page_index in range(total_tokens // page_span)
            for rank, mapping in enumerate(rank_mappings)
        ]

    _assert_symbolic_plan_copies_layout(
        endpoint(source_topology, destination=False),
        endpoint(destination_topology, destination=True),
    )


# ---------------------------------------------------------------------------
# Fail-closed gates
# ---------------------------------------------------------------------------


def test_fail_closed_cases():
    spec = _full_spec()
    nhd = _split_nhd_cache(spec)
    # Spec heads inconsistent with total heads / tp
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, total=2)) is None
    # tp not divisible by total KV heads
    one_head = _full_spec(num_kv_heads=1)
    assert (
        _try_mapping(one_head, _split_nhd_cache(one_head), _ctx(0, tp=3, total=2))
        is None
    )
    # DCP wider than the KV replication factor (tokens would shard across
    # ranks holding different heads)
    assert _try_mapping(spec, nhd, _ctx(0, tp=4, dcp=2)) is None
    # Interleave must divide the block size
    assert _try_mapping(spec, nhd, _ctx(0, tp=2, dcp=2, interleave=3, total=2)) is None
    assert _try_mapping(spec, nhd, _ctx(0, tp=2, interleave=0, total=2)) is None
    # Native PCP and DCP overlap process axes rather than forming a Cartesian
    # token grid. The static mapper must not manufacture that ownership.
    assert _try_mapping(_mla_spec(), None, _ctx(0, tp=2, dcp=2, pcp=2)) is None
    # Per-token-head scales are packed with the data
    quant_spec = _full_spec(kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD)
    assert _try_mapping(quant_spec, _split_nhd_cache(quant_spec), _ctx(0, tp=4)) is None
    # Compressed MLA slots are not 1:1 with tokens
    assert (
        _try_mapping(_mla_spec(tokens_per_state=2), None, _ctx(0, tp=2, dcp=2)) is None
    )
    # Unrecognized physical layouts
    swapped = torch.zeros(
        NUM_BLOCKS,
        spec.block_size,
        2,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    ).permute(0, 2, 1, 3, 4)
    assert _try_mapping(spec, swapped, _ctx(0, tp=4)) is None
    # Non-attention specs
    assert _try_mapping(KVCacheSpec(block_size=4), None, _ctx(0, tp=4, total=8)) is None


def test_opaque_fallback_places_page_whole():
    def opaque(tp_rank: int, **overrides) -> CanonicalPageMapping:
        kwargs = {
            "tp_size": 4,
            "tp_rank": tp_rank,
            "dcp_size": 1,
            "dcp_rank": 0,
            "pcp_size": 1,
            "pcp_rank": 0,
            "interleave": 1,
            "total_kv_heads": 8,
            "cache_layout_fingerprint": "layout-a",
        }
        kwargs.update(overrides)
        return _opaque_fallback_mapping(1024, **kwargs)

    mapping = opaque(2)
    assert mapping.canonical_page_size_bytes == 4096
    assert not mapping.parallelism_agnostic
    assert _triples(mapping.runs) == [(0, 2048, 1024)]
    assert mapping.num_writers == 1
    _verify_tiling("opaque", [opaque(rank) for rank in range(4)])


def test_opaque_signature_captures_topology_and_page_layout():
    def signature(**overrides) -> str:
        kwargs = {
            "tp_size": 4,
            "tp_rank": 0,
            "dcp_size": 1,
            "dcp_rank": 0,
            "pcp_size": 1,
            "pcp_rank": 0,
            "interleave": 1,
            "total_kv_heads": 8,
            "cache_layout_fingerprint": "layout-a",
        }
        kwargs.update(overrides)
        value = _opaque_fallback_mapping(1024, **kwargs).opaque_layout_signature
        assert value is not None
        return value

    baseline = signature()
    assert signature(tp_rank=1) == baseline
    assert signature(dcp_size=2, dcp_rank=0) != baseline
    assert (
        signature(
            tp_size=2,
            dcp_size=1,
            pcp_size=2,
            pcp_rank=0,
        )
        != baseline
    )
    assert signature(interleave=2) != baseline
    assert signature(total_kv_heads=4) != baseline
    assert signature(cache_layout_fingerprint="layout-b") != baseline


def test_opaque_cache_fingerprint_captures_page_layout_not_capacity():
    spec = _full_spec()
    nhd = _split_nhd_cache(spec)
    same_layout_more_pages = torch.zeros(
        NUM_BLOCKS + 2,
        2,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
        dtype=torch.int8,
    )
    hnd = _split_hnd_cache(spec)

    baseline = _opaque_cache_layout_fingerprint(spec, nhd)
    assert _opaque_cache_layout_fingerprint(spec, same_layout_more_pages) == baseline
    assert _opaque_cache_layout_fingerprint(spec, hnd) != baseline
    assert _opaque_cache_layout_fingerprint(spec, nhd, "LBHNC") != baseline


# ---------------------------------------------------------------------------
# derive_canonical_mappings end to end
# ---------------------------------------------------------------------------


def _vllm_config(tp=1, dcp=1, pcp=1, pp=1, interleave=1, total_kv_heads=2):
    config = MagicMock()
    config.parallel_config.tensor_parallel_size = tp
    config.parallel_config.decode_context_parallel_size = dcp
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.cp_kv_cache_interleave_size = interleave
    config.parallel_config.world_size = pp * tp * pcp
    config.parallel_config.rank = 0
    config.model_config.get_total_num_kv_heads.return_value = total_kv_heads
    return config


def _kv_cache_config(groups):
    config = MagicMock()
    config.kv_cache_groups = groups
    config.num_blocks = NUM_BLOCKS
    return config


def test_derive_mixed_model_with_dcp():
    attn_spec = _full_spec(num_kv_heads=1)
    mla_spec = _mla_spec()
    quant_spec = _full_spec(
        num_kv_heads=1, kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD
    )
    kv_cache_config = _kv_cache_config(
        [
            KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec),
            KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=mla_spec),
            KVCacheGroupSpec(layer_names=["quant"], kv_cache_spec=quant_spec),
        ]
    )
    kv_caches = {
        "attn": _split_nhd_cache(attn_spec),
        "quant": _split_nhd_cache(quant_spec),
    }
    mappings = derive_canonical_mappings(
        _vllm_config(tp=4, dcp=2, total_kv_heads=2), kv_cache_config, kv_caches
    )
    assert set(mappings) == {"attn", "mla", "quant"}
    assert not mappings["attn"].parallelism_agnostic
    assert mappings["attn"].runs
    # Uncertifiable layers degrade to an opaque page, never disappear
    assert not mappings["quant"].parallelism_agnostic
    assert (
        mappings["quant"].canonical_page_size_bytes
        == 4 * quant_spec.unpadded_page_size_bytes
    )


def test_rank_derivation_certifies_gdn_state_without_dcp_token_sharding(monkeypatch):
    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.mamba_utils.get_conv_state_layout",
        lambda: "DS",
    )
    spec = _gdn_spec(8, recover_ssm=True)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["gdn"], kv_cache_spec=spec)]
    )

    mapping = derive_rank_canonical_mappings(
        _vllm_config(tp=8, dcp=8),
        kv_cache_config,
        {"gdn": _mamba_cache(spec)},
        tp_rank=5,
    )["gdn"]

    assert not mapping.is_opaque
    assert mapping.canonical_token_span is None
    assert mapping == _mapping(
        spec,
        _mamba_cache(spec),
        _ctx(5, tp=8, dcp=8),
    )


def test_derive_refuses_foreign_worker_groups():
    attn_spec = _full_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["attn"], kv_cache_spec=attn_spec)]
    )
    kv_caches = {"attn": _split_nhd_cache(attn_spec)}
    assert (
        derive_canonical_mappings(
            _vllm_config(tp=2, pp=2, total_kv_heads=4), kv_cache_config, kv_caches
        )
        == {}
    )


def test_explicit_rank_derivation_is_independent_of_pipeline_parallelism():
    spec = _mla_spec()
    config = _vllm_config(tp=2, dcp=2, pp=3)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )

    mapping = derive_rank_canonical_mappings(
        config,
        kv_cache_config,
        {},
        tp_rank=1,
        dcp_rank=1,
    )["mla"]

    expected = _mapping(spec, None, _ctx(rank=1, tp=2, dcp=2))
    assert mapping == expected


def test_default_dcp_adapter_matches_explicit_native_rank():
    spec = _mla_spec()
    config = _vllm_config(tp=4, dcp=2)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )

    implicit = derive_rank_canonical_mappings(config, kv_cache_config, {}, tp_rank=3)
    explicit = derive_rank_canonical_mappings(
        config, kv_cache_config, {}, tp_rank=3, dcp_rank=1
    )
    assert implicit == explicit


def test_rank_derivation_maps_one_physical_attention_page():
    spec = _mla_spec()
    config = _vllm_config(tp=2, dcp=2, interleave=1)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )

    mapping = derive_rank_canonical_mappings(
        config,
        kv_cache_config,
        {"mla": torch.empty(NUM_BLOCKS * 2, 1, 2, 64, dtype=torch.int8)},
        tp_rank=1,
        physical_pages_per_logical=2,
    )["mla"]

    assert mapping.local_page_size_bytes == spec.real_page_size_bytes // 2
    assert mapping.canonical_page_size_bytes == spec.real_page_size_bytes
    assert mapping.canonical_token_span == spec.block_size
    assert [run.canonical_offset for run in mapping.runs] == [64, 192]


def test_rank_derivation_rejects_physical_page_geometry_without_equal_split():
    config = _vllm_config(tp=2, dcp=2, interleave=2)
    spec = _mla_spec()
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )
    with pytest.raises(ValueError, match="physical attention block size"):
        derive_rank_canonical_mappings(
            config,
            kv_cache_config,
            {},
            tp_rank=0,
            physical_pages_per_logical=4,
        )

    padded = replace(spec, page_size_padded=spec.page_size_bytes + 64)
    padded_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=padded)]
    )
    with pytest.raises(ValueError, match="padded attention pages"):
        derive_rank_canonical_mappings(
            _vllm_config(),
            padded_config,
            {},
            tp_rank=0,
            physical_pages_per_logical=2,
        )


@pytest.mark.parametrize(
    ("tp", "tp_rank", "dcp", "pcp", "pcp_rank", "expected"),
    [
        (8, 5, 4, 1, 0, 1),
        (4, 3, 1, 2, 1, 0),
        (4, 3, 2, 2, 1, 1),
        (4, 3, 8, 2, 1, 7),
    ],
)
def test_native_vllm_dcp_rank_matches_process_group_layout(
    tp: int,
    tp_rank: int,
    dcp: int,
    pcp: int,
    pcp_rank: int,
    expected: int,
):
    assert (
        native_vllm_dcp_rank(
            tp_size=tp,
            tp_rank=tp_rank,
            dcp_size=dcp,
            pcp_size=pcp,
            pcp_rank=pcp_rank,
        )
        == expected
    )


def test_combined_pcp_dcp_derivation_falls_back_opaque():
    spec = _mla_spec()
    config = _vllm_config(tp=2, dcp=2, pcp=2)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )
    mapping = derive_rank_canonical_mappings(
        config,
        kv_cache_config,
        {},
        tp_rank=1,
        pcp_rank=1,
        dcp_rank=1,
    )["mla"]
    assert mapping.is_opaque


@pytest.mark.parametrize(
    ("tp_rank", "pcp_rank", "message"),
    [(-1, 0, "tp_rank"), (2, 0, "tp_rank"), (0, -1, "pcp_rank"), (0, 2, "pcp_rank")],
)
def test_explicit_rank_derivation_validates_coordinates(
    tp_rank: int, pcp_rank: int, message: str
):
    spec = _mla_spec()
    config = _vllm_config(tp=2, pcp=2)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )
    with pytest.raises(ValueError, match=message):
        derive_rank_canonical_mappings(
            config,
            kv_cache_config,
            {},
            tp_rank=tp_rank,
            pcp_rank=pcp_rank,
        )


def test_explicit_rank_derivation_rejects_non_native_dcp_identity():
    spec = _mla_spec()
    config = _vllm_config(tp=4, dcp=2)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )
    with pytest.raises(ValueError, match="does not match vLLM native DCP rank"):
        derive_rank_canonical_mappings(
            config,
            kv_cache_config,
            {},
            tp_rank=1,
            dcp_rank=0,
        )


def test_native_dcp_adapter_rejects_uninstantiable_topology():
    with pytest.raises(ValueError, match="must be divisible"):
        native_vllm_dcp_rank(tp_size=3, tp_rank=0, dcp_size=2)
    with pytest.raises(ValueError, match="with PCP"):
        native_vllm_dcp_rank(
            tp_size=4,
            tp_rank=0,
            dcp_size=4,
            pcp_size=2,
            pcp_rank=0,
        )


@pytest.mark.parametrize("interleave", [0, 3, 5])
def test_rank_derivation_rejects_uninstantiable_cp_interleave(interleave: int):
    spec = _mla_spec()
    config = _vllm_config(tp=2, dcp=2, interleave=interleave)
    kv_cache_config = _kv_cache_config(
        [KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec)]
    )

    with pytest.raises(ValueError, match="cp_kv_cache_interleave_size"):
        derive_rank_canonical_mappings(
            config,
            kv_cache_config,
            {},
            tp_rank=0,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda mapping: replace(mapping, canonical_page_size_bytes=257),
            "canonical page size",
        ),
        (lambda mapping: replace(mapping, num_writers=2), "writers"),
        (
            lambda mapping: replace(
                mapping,
                runs=(replace(mapping.runs[0], local_offset=1),),
            ),
            "do not cover the local page",
        ),
    ],
)
def test_verify_tiling_raises_value_error_for_invalid_rank_mapping(mutate, message):
    mapping = _mapping(_mla_spec(), None, _ctx(rank=0))
    with pytest.raises(ValueError, match=message):
        _verify_tiling("mla", [mapping, mutate(mapping)])


def test_verify_tiling_rejects_empty_mapping_set():
    with pytest.raises(ValueError, match="no rank mappings"):
        _verify_tiling("empty", [])
