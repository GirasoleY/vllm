# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge import (
    build_nixl_read_request_plan,
    iter_nixl_read_plan_windows,
    nixl_read_request_plan_digest,
    validate_complete_nixl_placement_endpoint,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.runtime_placement import (
    NixlRuntimePlacementUnsupported,
    build_runtime_nixl_placement,
    finalize_nixl_placement_cohort,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)


def _config(
    *,
    pp_size: int = 1,
    pcp_size: int = 1,
    tp_size: int = 1,
    dcp_size: int = 1,
    total_kv_heads: int = 1,
):
    parallel = SimpleNamespace(
        pipeline_parallel_size=pp_size,
        prefill_context_parallel_size=pcp_size,
        tensor_parallel_size=tp_size,
        decode_context_parallel_size=dcp_size,
        data_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    model = SimpleNamespace(
        get_total_num_kv_heads=lambda: total_kv_heads,
        compute_hash=lambda: "model-v1",
    )
    return SimpleNamespace(parallel_config=parallel, model_config=model)


def _cache_config(
    kv_cache_layout: str = "LBNHC",
    layer_name: str = "model.layers.0.self_attn",
    num_blocks: int = 3,
) -> KVCacheConfig:
    spec = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.uint8,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=(KVCacheGroupSpec([layer_name], spec),),
        kv_cache_layout=kv_cache_layout,
    )


def test_runtime_builder_advertises_certified_direct_mla_pages():
    placement = build_runtime_nixl_placement(
        vllm_config=_config(),
        kv_cache_config=_cache_config(),
        caches={
            "model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)
        },
        deployment_id="decode",
        topology_generation=0,
        worker_id="decode-tp0",
        worker_incarnation="boot-1",
        tp_rank=0,
    )

    group = placement.format_manifest.groups[0]
    assert group.kind == "mla"
    assert group.canonical_page_token_span == 4
    assert group.canonical_page_size_bytes == 8
    assert placement.rank_placement.mappings[0].mapping.parallelism_agnostic
    assert placement.capabilities.scatter_gather
    assert placement.capabilities.max_segments_per_batch == 4096


def test_runtime_builder_accepts_contiguous_mla_pages_with_outer_stride():
    raw = torch.empty(48, dtype=torch.uint8)
    cache = torch.as_strided(raw, (3, 1, 4, 2), (16, 8, 2, 1))
    assert cache[0].is_contiguous()
    placement = build_runtime_nixl_placement(
        vllm_config=_config(),
        kv_cache_config=_cache_config(),
        caches={"model.layers.0.self_attn": cache},
        deployment_id="decode",
        topology_generation=0,
        worker_id="decode-tp0",
        worker_incarnation="boot-1",
        tp_rank=0,
    )

    assert placement.page_registration_templates[0].page_stride == 16
    assert placement.page_registration_templates[0].page_size_bytes == 8


def test_runtime_mla_canonical_space_is_independent_of_local_layout_enum():
    cache = torch.empty((3, 1, 4, 2), dtype=torch.uint8)
    placements = tuple(
        build_runtime_nixl_placement(
            vllm_config=_config(),
            kv_cache_config=_cache_config(layout),
            caches={"model.layers.0.self_attn": cache},
            deployment_id=deployment_id,
            topology_generation=0,
            worker_id=f"{deployment_id}-tp0",
            worker_incarnation=f"{deployment_id}-boot",
            tp_rank=0,
        )
        for deployment_id, layout in (
            ("prefill", "LBNHC"),
            ("decode", "LBHNC"),
        )
    )

    source, destination = placements
    semantic_id = source.format_manifest.groups[0].semantic_id
    assert source.format_manifest.groups[0].format_id == (
        "kv-placement-v1:mla:latent-token-row"
    )
    assert source.format_manifest.canonical_space_id(semantic_id) == (
        destination.format_manifest.canonical_space_id(semantic_id)
    )
    plan = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([0],),
        destination_block_ids=([1],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )
    assert tuple(iter_nixl_read_plan_windows(plan))[0].layer_plan.runs


def test_runtime_mha_format_ignores_stride_order_but_not_kv_storage_family():
    layer_name = "model.layers.0.self_attn"
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=3,
        dtype=torch.uint8,
    )

    def placement(cache: torch.Tensor, deployment: str):
        cache_config = KVCacheConfig(
            num_blocks=3,
            kv_cache_tensors=[],
            kv_cache_groups=(KVCacheGroupSpec([layer_name], spec),),
            kv_cache_layout="LBNHC",
        )
        return build_runtime_nixl_placement(
            vllm_config=_config(total_kv_heads=2),
            kv_cache_config=cache_config,
            caches={layer_name: cache},
            deployment_id=deployment,
            topology_generation=0,
            worker_id=f"{deployment}-tp0",
            worker_incarnation=f"{deployment}-boot",
            tp_rank=0,
        )

    # Both views have logical [block, K/V, token, head, dim] shape. The first
    # is token-major and the second is a head-major view of equal-size pages.
    split_nhd = torch.empty((3, 2, 4, 2, 3), dtype=torch.uint8)
    split_hnd_storage = torch.empty((3, 2, 2, 4, 3), dtype=torch.uint8)
    split_hnd = torch.as_strided(
        split_hnd_storage,
        (3, 2, 4, 2, 3),
        (48, 24, 3, 12, 1),
    )
    packed_hnd = torch.empty((3, 2, 4, 6), dtype=torch.uint8)

    split_formats = {
        placement(cache, deployment).format_manifest.groups[0].format_id
        for cache, deployment in (
            (split_nhd, "split-nhd"),
            (split_hnd, "split-hnd"),
        )
    }
    packed_format = placement(packed_hnd, "packed").format_manifest.groups[0].format_id

    assert split_formats == {"kv-placement-v1:mha:split-kv:token-head-row"}
    assert packed_format == "kv-placement-v1:mha:packed-kv:token-head-row"


def test_runtime_builder_advertises_only_transfer_enabled_groups():
    transferred = "model.layers.0.self_attn"
    local_only = "model.layers.0.draft_attn"
    spec = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.uint8,
    )
    cache_config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=(
            KVCacheGroupSpec([transferred], spec),
            KVCacheGroupSpec([local_only], spec, enable_kv_transfer=False),
        ),
        kv_cache_layout="LBNHC",
    )

    placement = build_runtime_nixl_placement(
        vllm_config=_config(),
        kv_cache_config=cache_config,
        caches={
            transferred: torch.empty((3, 1, 4, 2), dtype=torch.uint8),
            # A disabled group may use unrelated physical page geometry.
            local_only: torch.empty((6, 1, 2, 2), dtype=torch.uint8),
        },
        deployment_id="decode",
        topology_generation=0,
        worker_id="decode-tp0",
        worker_incarnation="boot-1",
        tp_rank=0,
    )

    assert placement.format_manifest.groups[0].layer_names == (transferred,)
    assert tuple(item.layer_name for item in placement.rank_placement.mappings) == (
        transferred,
    )


def test_runtime_builder_advertises_pcp_as_persistent_kv_replicas():
    cache = {"model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)}
    placements = tuple(
        build_runtime_nixl_placement(
            vllm_config=_config(pcp_size=2),
            kv_cache_config=_cache_config(),
            caches=cache,
            deployment_id="decode",
            topology_generation=0,
            worker_id=f"decode-pcp{pcp_rank}-tp0",
            worker_incarnation=f"boot-{pcp_rank}",
            tp_rank=0,
            pcp_rank=pcp_rank,
        )
        for pcp_rank in range(2)
    )

    ranks = [placement.rank_placement for placement in placements]
    assert [(rank.pcp_size, rank.pcp_rank) for rank in ranks] == [(2, 0), (2, 1)]
    assert [rank.rank for rank in ranks] == [0, 1]
    assert [rank.dcp_rank for rank in ranks] == [0, 0]
    assert [rank.dcp_group_id for rank in ranks] == [
        "decode:dp0:pp0:pcp0:dcp0",
        "decode:dp0:pp0:pcp1:dcp0",
    ]
    mappings = [rank.mappings[0].mapping for rank in ranks]
    assert mappings[0].runs == mappings[1].runs
    assert [mapping.canonical_token_span for mapping in mappings] == [4, 4]
    assert [mapping.num_writers for mapping in mappings] == [2, 2]
    assert [mapping.writer_index for mapping in mappings] == [0, 1]

    finalized = finalize_nixl_placement_cohort(placements)
    assert validate_complete_nixl_placement_endpoint(finalized) == finalized


def test_runtime_builder_rejects_combined_pcp_dcp():
    cache = {"model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)}
    with pytest.raises(NixlRuntimePlacementUnsupported, match="PCP and DCP"):
        build_runtime_nixl_placement(
            vllm_config=_config(pcp_size=2, tp_size=2, dcp_size=2),
            kv_cache_config=_cache_config(),
            caches=cache,
            deployment_id="decode",
            topology_generation=0,
            worker_id="decode-pcp0-tp0",
            worker_incarnation="boot-0",
            tp_rank=0,
            pcp_rank=0,
        )


def test_pp_pcp_tp_cohort_is_complete_and_preserves_layer_ownership():
    pp_size = 2
    pcp_size = 2
    tp_size = 2
    num_blocks = 4
    stage_layers = (
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    )

    def make_endpoint(deployment_id: str):
        candidates = []
        for pp_rank, layer_name in enumerate(stage_layers):
            cache_config = _cache_config(
                layer_name=layer_name,
                num_blocks=num_blocks,
            )
            for pcp_rank in range(pcp_size):
                for tp_rank in range(tp_size):
                    candidates.append(
                        build_runtime_nixl_placement(
                            vllm_config=_config(
                                pp_size=pp_size,
                                pcp_size=pcp_size,
                                tp_size=tp_size,
                            ),
                            kv_cache_config=cache_config,
                            caches={
                                layer_name: torch.empty(
                                    (num_blocks, 1, 4, 2), dtype=torch.uint8
                                )
                            },
                            deployment_id=deployment_id,
                            topology_generation=0,
                            worker_id=(
                                f"{deployment_id}-pp{pp_rank}-pcp{pcp_rank}-tp{tp_rank}"
                            ),
                            worker_incarnation=(
                                f"{deployment_id}-boot-{pp_rank}-{pcp_rank}-{tp_rank}"
                            ),
                            tp_rank=tp_rank,
                            pcp_rank=pcp_rank,
                            pp_rank=pp_rank,
                        )
                    )
        endpoint = finalize_nixl_placement_cohort(candidates)
        assert endpoint == validate_complete_nixl_placement_endpoint(endpoint)
        return endpoint, candidates

    source, source_candidates = make_endpoint("prefill")
    destination, _ = make_endpoint("decode")

    expected_coordinates = {
        (pp_rank, pcp_rank, tp_rank)
        for pp_rank in range(pp_size)
        for pcp_rank in range(pcp_size)
        for tp_rank in range(tp_size)
    }
    assert {
        (
            worker.rank_placement.pp_rank,
            worker.rank_placement.pcp_rank,
            worker.rank_placement.tp_rank,
        )
        for worker in source
    } == expected_coordinates
    assert tuple(worker.rank_placement.rank for worker in source) == tuple(range(8))
    assert all(
        worker.format_manifest.groups[0].layer_names == stage_layers
        for worker in source
    )

    ranks_by_stage: dict[int, set[int]] = {}
    for pp_rank, layer_name in enumerate(stage_layers):
        owners = [
            worker
            for worker in source
            if any(
                mapping.layer_name == layer_name
                for mapping in worker.rank_placement.mappings
            )
        ]
        assert {
            (
                worker.rank_placement.pp_rank,
                worker.rank_placement.pcp_rank,
                worker.rank_placement.tp_rank,
            )
            for worker in owners
        } == {
            (pp_rank, pcp_rank, tp_rank)
            for pcp_rank in range(pcp_size)
            for tp_rank in range(tp_size)
        }
        mappings = [worker.rank_placement.mappings[0].mapping for worker in owners]
        assert {mapping.num_writers for mapping in mappings} == {pcp_size * tp_size}
        assert {mapping.writer_index for mapping in mappings} == set(
            range(pcp_size * tp_size)
        )
        ranks_by_stage[pp_rank] = {worker.rank_placement.rank for worker in owners}

    assert {worker.rank_placement.topology_generation for worker in source} == {
        worker.rank_placement.topology_generation
        for worker in finalize_nixl_placement_cohort(tuple(reversed(source_candidates)))
    }

    plan = build_nixl_read_request_plan(
        source_workers=source,
        destination_workers=destination,
        source_block_ids=(list(range(num_blocks)),),
        destination_block_ids=(list(range(num_blocks)),),
        destination_prefix_blocks=(0,),
        remote_num_tokens=num_blocks * 4,
    )
    assert plan.planning_context.source_expected_participant_count == 8
    assert plan.destination_expected_participant_count == 8
    runs_by_layer = {
        window.layer_plan.layer_name: window.layer_plan.runs
        for window in iter_nixl_read_plan_windows(plan)
    }
    for pp_rank, layer_name in enumerate(stage_layers):
        runs = runs_by_layer[layer_name]
        assert {run.source_rank for run in runs} == ranks_by_stage[pp_rank]
        assert {run.destination_rank for run in runs} == ranks_by_stage[pp_rank]


def test_runtime_builder_advertises_real_pp_and_ep_coordinates():
    layer_name = "model.layers.8.self_attn"
    placement = build_runtime_nixl_placement(
        vllm_config=_config(pp_size=2, tp_size=2),
        kv_cache_config=_cache_config(layer_name=layer_name),
        caches={layer_name: torch.empty((3, 1, 4, 2), dtype=torch.uint8)},
        deployment_id="decode",
        topology_generation=0,
        worker_id="decode-pp1-tp1",
        worker_incarnation="boot-1",
        tp_rank=1,
        pp_rank=1,
        ep_size=2,
        ep_rank=1,
    )

    rank = placement.rank_placement
    assert (rank.pp_size, rank.pp_rank) == (2, 1)
    assert (rank.ep_size, rank.ep_rank) == (2, 1)
    assert rank.rank == 3
    assert rank.dcp_group_id == "decode:dp0:pp1:dcp1"


def test_runtime_builder_rejects_uncertified_mla_semantics_and_strides():
    layer_name = "model.layers.0.self_attn"
    cache = {layer_name: torch.empty((3, 1, 4, 2), dtype=torch.uint8)}
    hidden_spec = HiddenStateCacheSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.uint8,
    )
    hidden_config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=(KVCacheGroupSpec([layer_name], hidden_spec),),
        kv_cache_layout="LBNHC",
    )
    with pytest.raises(NixlRuntimePlacementUnsupported, match="explicitly supported"):
        build_runtime_nixl_placement(
            vllm_config=_config(),
            kv_cache_config=hidden_config,
            caches={layer_name: torch.empty((3, 1, 4, 2), dtype=torch.uint8)},
            deployment_id="decode",
            topology_generation=0,
            worker_id="decode-tp0",
            worker_incarnation="boot-1",
            tp_rank=0,
        )

    strided_cache = torch.empty((3, 1, 4, 4), dtype=torch.uint8)[..., ::2]
    assert not strided_cache[0].is_contiguous()
    with pytest.raises(NixlRuntimePlacementUnsupported, match="exact contiguous"):
        build_runtime_nixl_placement(
            vllm_config=_config(),
            kv_cache_config=_cache_config(),
            caches={layer_name: strided_cache},
            deployment_id="decode",
            topology_generation=0,
            worker_id="decode-tp0",
            worker_incarnation="boot-1",
            tp_rank=0,
        )

    sliding_spec = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.uint8,
        sliding_window=8,
    )
    sliding_config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=(KVCacheGroupSpec(["model.layers.0.self_attn"], sliding_spec),),
        kv_cache_layout="LBNHC",
    )
    with pytest.raises(NixlRuntimePlacementUnsupported, match="stable full-attention"):
        build_runtime_nixl_placement(
            vllm_config=_config(),
            kv_cache_config=sliding_config,
            caches=cache,
            deployment_id="decode",
            topology_generation=0,
            worker_id="decode-tp0",
            worker_incarnation="boot-1",
            tp_rank=0,
        )


def test_runtime_builder_advertises_physical_kernel_pages():
    layer_name = "model.layers.0.self_attn"
    placement = build_runtime_nixl_placement(
        vllm_config=_config(),
        kv_cache_config=_cache_config(),
        caches={layer_name: torch.empty((6, 1, 2, 2), dtype=torch.uint8)},
        deployment_id="decode",
        topology_generation=0,
        worker_id="decode-tp0",
        worker_incarnation="boot-1",
        tp_rank=0,
        physical_pages_per_logical=2,
    )

    group = placement.format_manifest.groups[0]
    mapping = placement.rank_placement.mappings[0].mapping
    registration = placement.page_registration_templates[0]
    assert group.canonical_page_token_span == 2
    assert group.canonical_page_size_bytes == 4
    assert mapping.local_page_size_bytes == 4
    assert registration.num_pages == 6
    assert registration.page_size_bytes == 4


def test_runtime_builder_rejects_wrong_physical_page_capacity():
    with pytest.raises(
        NixlRuntimePlacementUnsupported, match="contain 6 physical pages"
    ):
        build_runtime_nixl_placement(
            vllm_config=_config(),
            kv_cache_config=_cache_config(),
            caches={
                "model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)
            },
            deployment_id="decode",
            topology_generation=0,
            worker_id="decode-tp0",
            worker_incarnation="boot-1",
            tp_rank=0,
            physical_pages_per_logical=2,
        )


def test_runtime_bridge_maps_tp1_pages_into_both_dcp2_ranks():
    cache_config = _cache_config()
    caches = {"model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)}
    source = build_runtime_nixl_placement(
        vllm_config=_config(),
        kv_cache_config=cache_config,
        caches=caches,
        deployment_id="prefill",
        topology_generation=0,
        worker_id="prefill-tp0",
        worker_incarnation="prefill-boot",
        tp_rank=0,
    )
    destinations = validate_complete_nixl_placement_endpoint(
        tuple(
            build_runtime_nixl_placement(
                vllm_config=_config(tp_size=2, dcp_size=2),
                kv_cache_config=cache_config,
                caches=caches,
                deployment_id="decode",
                topology_generation=0,
                worker_id=f"decode-tp{rank}",
                worker_incarnation=f"decode-boot-{rank}",
                tp_rank=rank,
            )
            for rank in range(2)
        )
    )

    plan = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=destinations,
        source_block_ids=([0, 1],),
        destination_block_ids=([2],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=8,
    )

    runs = tuple(iter_nixl_read_plan_windows(plan))[0].layer_plan.runs
    assert {run.destination_rank for run in runs} == {0, 1}
    assert {
        rank: sum(
            run.fragment_size * run.fragment_count
            for run in runs
            if run.destination_rank == rank
        )
        for rank in range(2)
    } == {0: 8, 1: 8}
    assert plan.destination_expected_participant_count == 2


def test_runtime_bridge_reads_rotating_pcp_replicas_directly_into_tp1():
    cache_config = _cache_config()
    caches = {"model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)}
    sources = finalize_nixl_placement_cohort(
        tuple(
            build_runtime_nixl_placement(
                vllm_config=_config(pcp_size=2),
                kv_cache_config=cache_config,
                caches=caches,
                deployment_id="prefill",
                topology_generation=0,
                worker_id=f"prefill-pcp{pcp_rank}-tp0",
                worker_incarnation=f"prefill-boot-{pcp_rank}",
                tp_rank=0,
                pcp_rank=pcp_rank,
            )
            for pcp_rank in range(2)
        )
    )
    destination = build_runtime_nixl_placement(
        vllm_config=_config(),
        kv_cache_config=cache_config,
        caches=caches,
        deployment_id="decode",
        topology_generation=0,
        worker_id="decode-tp0",
        worker_incarnation="decode-boot",
        tp_rank=0,
    )

    plan = build_nixl_read_request_plan(
        source_workers=sources,
        destination_workers=(destination,),
        source_block_ids=([0, 1],),
        destination_block_ids=([1, 2],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=8,
    )

    runs = tuple(
        run
        for window in iter_nixl_read_plan_windows(plan)
        for run in window.layer_plan.runs
    )
    assert {run.source_rank for run in runs} == {0, 1}
    assert {run.destination_rank for run in runs} == {0}
    assert sum(run.fragment_size * run.fragment_count for run in runs) == 16
    assert plan.planning_context.source_expected_participant_count == 2
    assert plan.destination_expected_participant_count == 1


def test_placement_cohort_generation_is_order_independent_and_restart_sensitive():
    cache_config = _cache_config()
    caches = {"model.layers.0.self_attn": torch.empty((3, 1, 4, 2), dtype=torch.uint8)}
    placements = tuple(
        build_runtime_nixl_placement(
            vllm_config=_config(tp_size=2, dcp_size=2),
            kv_cache_config=cache_config,
            caches=caches,
            deployment_id="decode",
            topology_generation=0,
            worker_id=f"decode-tp{rank}",
            worker_incarnation=f"decode-boot-{rank}",
            tp_rank=rank,
        )
        for rank in range(2)
    )

    finalized = finalize_nixl_placement_cohort(placements)
    reordered = finalize_nixl_placement_cohort(tuple(reversed(placements)))
    generations = {
        placement.rank_placement.topology_generation for placement in finalized
    }

    assert len(generations) == 1
    generation = generations.pop()
    assert generation > 0
    assert {
        placement.rank_placement.topology_generation for placement in reordered
    } == {generation}

    restarted = replace(
        placements[1],
        rank_placement=replace(
            placements[1].rank_placement,
            worker_incarnation="decode-restarted",
        ),
    )
    restarted_generation = finalize_nixl_placement_cohort((placements[0], restarted))[
        0
    ].rank_placement.topology_generation
    assert restarted_generation != generation


def test_pp_cohort_normalizes_stage_local_formats_across_all_tp_ranks():
    pp_size = 2
    tp_size = 2
    stage_layers = (
        "model.layers.0.self_attn",
        "model.layers.1.self_attn",
    )
    candidates = []
    for pp_rank, layer_name in enumerate(stage_layers):
        cache_config = _cache_config(layer_name=layer_name)
        cache = torch.empty((3, 1, 4, 2), dtype=torch.uint8)
        for tp_rank in range(tp_size):
            candidates.append(
                build_runtime_nixl_placement(
                    vllm_config=_config(pp_size=pp_size, tp_size=tp_size),
                    kv_cache_config=cache_config,
                    caches={layer_name: cache},
                    deployment_id="decode",
                    topology_generation=0,
                    worker_id=f"decode-pp{pp_rank}-tp{tp_rank}",
                    worker_incarnation=f"boot-{pp_rank}-{tp_rank}",
                    tp_rank=tp_rank,
                    pp_rank=pp_rank,
                )
            )

    finalized = finalize_nixl_placement_cohort(candidates)
    endpoint = validate_complete_nixl_placement_endpoint(finalized)
    reversed_endpoint = finalize_nixl_placement_cohort(tuple(reversed(candidates)))

    assert tuple(worker.rank_placement.rank for worker in endpoint) == (0, 1, 2, 3)
    assert all(
        worker.format_manifest.groups[0].layer_names == stage_layers
        for worker in endpoint
    )
    semantic_ids = {
        mapping.semantic_group_id
        for worker in endpoint
        for mapping in worker.rank_placement.mappings
    }
    assert semantic_ids == {endpoint[0].format_manifest.groups[0].semantic_id}
    assert {
        worker.rank_placement.pp_rank: worker.rank_placement.layer_range
        for worker in endpoint
    } == {0: (0, 1), 1: (1, 2)}
    assert {worker.rank_placement.topology_generation for worker in endpoint} == {
        worker.rank_placement.topology_generation for worker in reversed_endpoint
    }


def test_pp_request_plan_has_exact_participants_and_order_stable_digest():
    def endpoint(deployment_id: str):
        candidates = []
        for pp_rank in range(2):
            layer_name = f"model.layers.{pp_rank}.self_attn"
            candidates.append(
                build_runtime_nixl_placement(
                    vllm_config=_config(pp_size=2),
                    kv_cache_config=_cache_config(layer_name=layer_name),
                    caches={layer_name: torch.empty((3, 1, 4, 2), dtype=torch.uint8)},
                    deployment_id=deployment_id,
                    topology_generation=0,
                    worker_id=f"{deployment_id}-pp{pp_rank}-tp0",
                    worker_incarnation=f"{deployment_id}-boot-{pp_rank}",
                    tp_rank=0,
                    pp_rank=pp_rank,
                )
            )
        return finalize_nixl_placement_cohort(candidates)

    source = endpoint("prefill")
    destination = endpoint("decode")
    plan = build_nixl_read_request_plan(
        source_workers=source,
        destination_workers=destination,
        source_block_ids=([0],),
        destination_block_ids=([1],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )
    reordered_plan = build_nixl_read_request_plan(
        source_workers=tuple(reversed(source)),
        destination_workers=tuple(reversed(destination)),
        source_block_ids=([0],),
        destination_block_ids=([1],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )

    assert plan.planning_context.source_expected_participant_count == 2
    assert plan.destination_expected_participant_count == 2
    assert nixl_read_request_plan_digest(plan) == nixl_read_request_plan_digest(
        reordered_plan
    )
