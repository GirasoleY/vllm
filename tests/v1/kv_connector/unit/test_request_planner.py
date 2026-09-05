# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.distributed.kv_transfer.request_planner as request_planner_module
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    CopyRun,
    KVFormatManifest,
    KVGroupFormat,
    KVRange,
    LayerPageMapping,
    RankPlacementManifest,
    TransferRun,
)
from vllm.distributed.kv_transfer.request_planner import (
    EndpointRoute,
    LocalPageAllocation,
    build_request_transfer_context,
    build_request_transfer_plan,
    iter_request_transfer_layer_runs,
    plan_request_transfers,
)


def _identity(
    span: int = 4,
    *,
    num_writers: int = 1,
    writer_index: int = 0,
) -> CanonicalPageMapping:
    return CanonicalPageMapping(
        canonical_page_size_bytes=span,
        local_page_size_bytes=span,
        runs=(CopyRun(0, 0, span, 1, span, span),),
        num_writers=num_writers,
        writer_index=writer_index,
        canonical_token_span=span,
        canonical_region_token_strides=((0, 1),),
    )


def _shard(offset: int, span: int = 8) -> CanonicalPageMapping:
    return CanonicalPageMapping(
        canonical_page_size_bytes=span,
        local_page_size_bytes=span // 2,
        runs=(CopyRun(0, offset, 1, span // 2, 1, 2),),
        num_writers=1,
        writer_index=0,
        canonical_token_span=span,
        canonical_region_token_strides=((0, 1),),
    )


def _group(
    group_id: int,
    semantic_id: str,
    layers: tuple[str, ...],
    *,
    span: int = 4,
) -> KVGroupFormat:
    return KVGroupFormat(
        group_id=group_id,
        semantic_id=semantic_id,
        kind="mla",
        layer_names=layers,
        canonical_page_token_span=span,
        dtype="uint8",
        canonical_page_size_bytes=span,
        format_id="test-token-major",
    )


def _format(
    *groups: KVGroupFormat,
    model_fingerprint: str = "model-v1",
) -> KVFormatManifest:
    return KVFormatManifest(
        version=1,
        model_fingerprint=model_fingerprint,
        groups=groups,
    )


def _worker(
    format_manifest: KVFormatManifest,
    worker_id: str,
    rank: int,
    layers: tuple[tuple[str, int, str, CanonicalPageMapping], ...],
    *,
    deployment_id: str,
    generation: int = 3,
    dp_rank: int = 0,
    dp_size: int = 1,
    dp_group_id: str | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> RankPlacementManifest:
    layer_indices = [layer_index for _, layer_index, _, _ in layers]
    return RankPlacementManifest(
        version=1,
        deployment_id=deployment_id,
        topology_generation=generation,
        worker_id=worker_id,
        worker_incarnation=f"{worker_id}-boot",
        format_manifest_fingerprint=format_manifest.fingerprint(),
        rank=rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dcp_size=1,
        dcp_rank=0,
        dcp_group_id=f"{deployment_id}-dcp",
        pcp_size=1,
        pcp_rank=0,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=dp_size,
        dp_rank=dp_rank,
        dp_group_id=dp_group_id or f"{deployment_id}-dp",
        ep_size=ep_size,
        ep_rank=ep_rank,
        cp_interleave=1,
        layer_range=(min(layer_indices), max(layer_indices) + 1),
        mappings=tuple(
            LayerPageMapping(layer_name, layer_index, semantic_id, mapping)
            for layer_name, layer_index, semantic_id, mapping in layers
        ),
    )


def _route(
    deployment_id: str,
    *,
    generation: int = 3,
    dp_rank: int = 0,
) -> EndpointRoute:
    return EndpointRoute(
        deployment_id=deployment_id,
        topology_generation=generation,
        dp_group_id=f"{deployment_id}-dp",
        dp_rank=dp_rank,
    )


def _page(
    worker: RankPlacementManifest,
    layer_name: str,
    local_page_id: int,
    *,
    first_token: int = 0,
    canonical_page_index: int = 0,
    local_base: int = 0,
) -> LocalPageAllocation:
    return LocalPageAllocation(
        worker_id=worker.worker_id,
        worker_incarnation=worker.worker_incarnation,
        layer_name=layer_name,
        local_page_id=local_page_id,
        canonical_page_index=canonical_page_index,
        first_token=first_token,
        local_base=local_base,
    )


def _plan(
    source_format: KVFormatManifest,
    destination_format: KVFormatManifest,
    source_workers: tuple[RankPlacementManifest, ...],
    destination_workers: tuple[RankPlacementManifest, ...],
    source_pages: tuple[LocalPageAllocation, ...],
    destination_pages: tuple[LocalPageAllocation, ...],
    ranges: tuple[KVRange, ...],
    *,
    source_dp_rank: int = 0,
    destination_dp_rank: int = 0,
):
    return plan_request_transfers(
        source_format=source_format,
        destination_format=destination_format,
        source_route=_route("source", dp_rank=source_dp_rank),
        destination_route=_route("destination", dp_rank=destination_dp_rank),
        source_workers=source_workers,
        destination_workers=destination_workers,
        source_pages=source_pages,
        destination_pages=destination_pages,
        ranges=ranges,
    )


def test_token_ownership_check_is_constant_per_affine_run(monkeypatch):
    fragment_count = 10**12
    mapping = CanonicalPageMapping(
        canonical_page_size_bytes=2 * fragment_count,
        local_page_size_bytes=fragment_count,
        runs=(CopyRun(0, 0, 1, fragment_count, 1, 2),),
        num_writers=1,
        writer_index=0,
        canonical_token_span=2 * fragment_count,
        canonical_region_token_strides=((0, 1),),
    )

    def reject_fragment_expansion(*_args, **_kwargs):
        raise AssertionError("ownership checks must not expand affine fragments")

    monkeypatch.setattr(
        request_planner_module,
        "range",
        reject_fragment_expansion,
        raising=False,
    )

    assert request_planner_module._mapping_owns_tokens(
        mapping,
        page_first_token=0,
        first_token=2 * fragment_count - 2,
        end_token=2 * fragment_count - 1,
    )
    assert not request_planner_module._mapping_owns_tokens(
        mapping,
        page_first_token=0,
        first_token=2 * fragment_count - 1,
        end_token=2 * fragment_count,
    )


def _request_plan(
    source_format: KVFormatManifest,
    destination_format: KVFormatManifest,
    source_workers: tuple[RankPlacementManifest, ...],
    destination_workers: tuple[RankPlacementManifest, ...],
    source_pages: tuple[LocalPageAllocation, ...],
    destination_pages: tuple[LocalPageAllocation, ...],
    ranges: tuple[KVRange, ...],
    *,
    source_dp_rank: int = 0,
    destination_dp_rank: int = 0,
):
    return build_request_transfer_plan(
        source_format=source_format,
        destination_format=destination_format,
        source_route=_route("source", dp_rank=source_dp_rank),
        destination_route=_route("destination", dp_rank=destination_dp_rank),
        source_workers=source_workers,
        destination_workers=destination_workers,
        source_pages=source_pages,
        destination_pages=destination_pages,
        ranges=ranges,
    )


def test_groups_match_by_semantic_id_despite_local_ids_and_order():
    source_format = _format(
        _group(0, "state", ("state.0",)),
        _group(7, "attention", ("layers.0",)),
    )
    destination_format = _format(
        _group(2, "attention", ("layers.0",)),
        _group(9, "state", ("state.0",)),
    )
    mapping = _identity()
    source = _worker(
        source_format,
        "source-0",
        10,
        (
            ("layers.0", 0, "attention", mapping),
            ("state.0", 1, "state", mapping),
        ),
        deployment_id="source",
    )
    destination = _worker(
        destination_format,
        "destination-0",
        20,
        (
            ("layers.0", 0, "attention", mapping),
            ("state.0", 1, "state", mapping),
        ),
        deployment_id="destination",
    )

    plans = _plan(
        source_format,
        destination_format,
        (source,),
        (destination,),
        (_page(source, "layers.0", 100),),
        (_page(destination, "layers.0", 200),),
        (KVRange("attention", 0, 4, 4),),
    )

    assert len(plans) == 1
    assert plans[0].semantic_group_id == "attention"
    assert plans[0].source_group_id == 7
    assert plans[0].destination_group_id == 2
    assert plans[0].runs == (TransferRun(10, 20, 100, 200, 0, 0, 4, 1, 4, 4),)


def test_pp_ownership_is_intersected_per_layer():
    format_manifest = _format(_group(0, "attention", ("layers.0", "layers.1")))
    mapping = _identity()
    source_0 = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        pp_size=2,
        pp_rank=0,
    )
    source_1 = _worker(
        format_manifest,
        "source-1",
        11,
        (("layers.1", 1, "attention", mapping),),
        deployment_id="source",
        pp_size=2,
        pp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (
            ("layers.0", 0, "attention", mapping),
            ("layers.1", 1, "attention", mapping),
        ),
        deployment_id="destination",
    )

    plans = _plan(
        format_manifest,
        format_manifest,
        (source_0, source_1),
        (destination,),
        (
            _page(source_0, "layers.0", 100),
            _page(source_1, "layers.1", 101),
        ),
        (
            _page(destination, "layers.0", 200),
            _page(destination, "layers.1", 201),
        ),
        (KVRange("attention", 0, 4, 4),),
    )

    assert [(plan.layer_name, plan.runs[0].source_rank) for plan in plans] == [
        ("layers.0", 10),
        ("layers.1", 11),
    ]
    assert {plan.runs[0].destination_rank for plan in plans} == {20}


def test_dp_selects_one_replica_while_ep_is_diagnostic_only():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    mapping = _identity()
    source_0 = _worker(
        format_manifest,
        "source-dp0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        dp_rank=0,
        dp_size=2,
        ep_size=4,
        ep_rank=3,
    )
    source_1 = _worker(
        format_manifest,
        "source-dp1",
        11,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        dp_rank=1,
        dp_size=2,
        ep_size=8,
        ep_rank=6,
    )
    destination_0 = _worker(
        format_manifest,
        "destination-dp0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
        dp_rank=0,
        dp_size=2,
        ep_size=2,
        ep_rank=1,
    )
    destination_1 = _worker(
        format_manifest,
        "destination-dp1",
        21,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
        dp_rank=1,
        dp_size=2,
    )

    plans = _plan(
        format_manifest,
        format_manifest,
        (source_0, source_1),
        (destination_0, destination_1),
        (
            _page(source_0, "layers.0", 100),
            _page(source_1, "layers.0", 101),
        ),
        (
            _page(destination_0, "layers.0", 200),
            _page(destination_1, "layers.0", 201),
        ),
        (KVRange("attention", 0, 4, 4),),
        source_dp_rank=1,
        destination_dp_rank=0,
    )

    assert plans[0].runs == (TransferRun(11, 20, 101, 200, 0, 0, 4, 1, 4, 4),)


def test_sparse_ranges_clip_holes_and_partial_tail_pages():
    format_manifest = _format(_group(0, "attention", ("layers.0",), span=8))
    mapping = _identity(8)
    source = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
    )

    plans = _plan(
        format_manifest,
        format_manifest,
        (source,),
        (destination,),
        (
            _page(source, "layers.0", 100),
            _page(source, "layers.0", 101, first_token=8, canonical_page_index=1),
        ),
        (
            _page(destination, "layers.0", 200),
            _page(
                destination,
                "layers.0",
                201,
                first_token=8,
                canonical_page_index=1,
            ),
        ),
        (
            KVRange("attention", 0, 8, 3),
            KVRange("attention", 8, 8, 2),
        ),
    )

    assert plans[0].runs == (
        TransferRun(10, 20, 100, 200, 0, 0, 3, 1, 3, 3),
        TransferRun(10, 20, 101, 201, 0, 0, 2, 1, 2, 2),
    )


def test_tail_does_not_require_a_page_from_a_rank_that_owns_no_valid_bytes():
    format_manifest = _format(_group(0, "attention", ("layers.0",), span=8))
    source_even = _worker(
        format_manifest,
        "source-even",
        10,
        (("layers.0", 0, "attention", _shard(0)),),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    source_odd = _worker(
        format_manifest,
        "source-odd",
        11,
        (("layers.0", 0, "attention", _shard(1)),),
        deployment_id="source",
        tp_size=2,
        tp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination",
        20,
        (("layers.0", 0, "attention", _identity(8)),),
        deployment_id="destination",
    )

    plans = _plan(
        format_manifest,
        format_manifest,
        (source_even, source_odd),
        (destination,),
        (_page(source_even, "layers.0", 100),),
        (_page(destination, "layers.0", 200),),
        (KVRange("attention", 0, 8, 1),),
    )

    assert plans[0].runs == (TransferRun(10, 20, 100, 200, 0, 0, 1, 1, 1, 1),)


def test_non_elected_source_replica_does_not_need_a_transfer_page():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    elected = _worker(
        format_manifest,
        "source-writer-0",
        10,
        (
            (
                "layers.0",
                0,
                "attention",
                _identity(num_writers=2, writer_index=0),
            ),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    replica = _worker(
        format_manifest,
        "source-writer-1",
        11,
        (
            (
                "layers.0",
                0,
                "attention",
                _identity(num_writers=2, writer_index=1),
            ),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination",
        20,
        (("layers.0", 0, "attention", _identity()),),
        deployment_id="destination",
    )

    request_plan = _request_plan(
        format_manifest,
        format_manifest,
        (elected, replica),
        (destination,),
        (_page(elected, "layers.0", 100),),
        (_page(destination, "layers.0", 200),),
        (KVRange("attention", 0, 4, 4),),
    )

    assert request_plan.layers[0].runs == (
        TransferRun(10, 20, 100, 200, 0, 0, 4, 1, 4, 4),
    )
    assert request_plan.source_participant_worker_ids == (
        "source-writer-0",
        "source-writer-1",
    )


def test_request_plan_exposes_non_elected_and_full_hit_participants():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    elected = _worker(
        format_manifest,
        "source-writer-0",
        10,
        (
            (
                "layers.0",
                0,
                "attention",
                _identity(num_writers=2, writer_index=0),
            ),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    replica = _worker(
        format_manifest,
        "source-writer-1",
        11,
        (
            (
                "layers.0",
                0,
                "attention",
                _identity(num_writers=2, writer_index=1),
            ),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination",
        20,
        (("layers.0", 0, "attention", _identity()),),
        deployment_id="destination",
    )

    request_plan = _request_plan(
        format_manifest,
        format_manifest,
        (elected, replica),
        (destination,),
        (),
        (),
        (KVRange("attention", 0, 4, 0),),
    )

    assert request_plan.layers == ()
    assert request_plan.source_participant_worker_ids == (
        "source-writer-0",
        "source-writer-1",
    )
    assert request_plan.destination_participant_worker_ids == ("destination",)
    assert request_plan.source_expected_participant_count == 2
    assert request_plan.destination_expected_participant_count == 1
    assert request_plan.source_route.topology_generation == 3
    assert request_plan.source_participants[1].worker_incarnation == (
        "source-writer-1-boot"
    )


def test_composed_multilayer_request_keeps_direct_segmented_runs():
    format_manifest = _format(_group(0, "attention", ("layers.0", "layers.1"), span=8))
    even = _shard(0)
    odd = _shard(1)
    full = _identity(8)
    source_even = _worker(
        format_manifest,
        "source-even",
        10,
        (
            ("layers.0", 0, "attention", even),
            ("layers.1", 1, "attention", even),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    source_odd = _worker(
        format_manifest,
        "source-odd",
        11,
        (
            ("layers.0", 0, "attention", odd),
            ("layers.1", 1, "attention", odd),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination",
        20,
        (
            ("layers.0", 0, "attention", full),
            ("layers.1", 1, "attention", full),
        ),
        deployment_id="destination",
    )

    source_pages = tuple(
        _page(worker, layer, page_id)
        for worker, first_page_id in ((source_even, 100), (source_odd, 110))
        for layer, page_id in (
            ("layers.0", first_page_id),
            ("layers.1", first_page_id + 1),
        )
    )
    destination_pages = (
        _page(destination, "layers.0", 200),
        _page(destination, "layers.1", 201),
    )
    plans = _plan(
        format_manifest,
        format_manifest,
        (source_even, source_odd),
        (destination,),
        source_pages,
        destination_pages,
        (KVRange("attention", 0, 8, 8),),
    )

    assert len(plans) == 2
    assert all(len(plan.runs) == 2 for plan in plans)
    assert plans[0].runs == (
        TransferRun(10, 20, 100, 200, 0, 0, 1, 4, 1, 2),
        TransferRun(11, 20, 110, 200, 0, 1, 1, 4, 1, 2),
    )
    assert plans[1].runs == (
        TransferRun(10, 20, 101, 201, 0, 0, 1, 4, 1, 2),
        TransferRun(11, 20, 111, 201, 0, 1, 1, 4, 1, 2),
    )


def test_duplicate_worker_and_unknown_page_worker_fail_closed():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    mapping = _identity()
    source = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
    )
    kwargs = {
        "source_format": format_manifest,
        "destination_format": format_manifest,
        "source_workers": (source,),
        "destination_workers": (destination,),
        "source_pages": (_page(source, "layers.0", 100),),
        "destination_pages": (_page(destination, "layers.0", 200),),
        "ranges": (KVRange("attention", 0, 4, 4),),
    }

    with pytest.raises(ValueError, match="duplicate worker"):
        _plan(**kwargs | {"source_workers": (source, source)})

    unknown_page = LocalPageAllocation("missing", "missing-boot", "layers.0", 100, 0, 0)
    with pytest.raises(ValueError, match="unknown worker"):
        _plan(**kwargs | {"source_pages": (unknown_page,)})


def test_incomplete_selected_replica_membership_fails_closed():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    mapping = _identity()
    source_rank_0 = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
    )

    with pytest.raises(ValueError, match="incomplete placement membership"):
        _plan(
            format_manifest,
            format_manifest,
            (source_rank_0,),
            (destination,),
            (_page(source_rank_0, "layers.0", 100),),
            (_page(destination, "layers.0", 200),),
            (KVRange("attention", 0, 4, 4),),
        )


def test_incomplete_tp_layer_placement_fails_at_endpoint_bind():
    format_manifest = _format(_group(0, "attention", ("layers.0", "layers.1")))
    mapping = _identity()
    source_rank_0 = _worker(
        format_manifest,
        "source-0",
        10,
        (
            ("layers.0", 0, "attention", mapping),
            ("layers.1", 1, "attention", mapping),
        ),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    source_rank_1 = _worker(
        format_manifest,
        "source-1",
        11,
        (("layers.1", 1, "attention", mapping),),
        deployment_id="source",
        tp_size=2,
        tp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (
            ("layers.0", 0, "attention", mapping),
            ("layers.1", 1, "attention", mapping),
        ),
        deployment_id="destination",
    )

    with pytest.raises(ValueError, match="incomplete PCP/TP placement"):
        _plan(
            format_manifest,
            format_manifest,
            (source_rank_0, source_rank_1),
            (destination,),
            (),
            (),
            (KVRange("attention", 0, 4, 4),),
        )


def test_layer_owned_by_multiple_pp_ranks_fails_at_endpoint_bind():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    mapping = _identity()
    source_pp_0 = _worker(
        format_manifest,
        "source-pp-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        pp_size=2,
        pp_rank=0,
    )
    source_pp_1 = _worker(
        format_manifest,
        "source-pp-1",
        11,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        pp_size=2,
        pp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
    )

    with pytest.raises(ValueError, match="owned by exactly one PP rank"):
        _plan(
            format_manifest,
            format_manifest,
            (source_pp_0, source_pp_1),
            (destination,),
            (),
            (),
            (KVRange("attention", 0, 4, 4),),
        )


def test_stale_worker_generation_and_page_incarnation_fail_closed():
    format_manifest = _format(_group(0, "attention", ("layers.0",)))
    mapping = _identity()
    source = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
        generation=2,
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
    )

    with pytest.raises(ValueError, match="topology generation"):
        _plan(
            format_manifest,
            format_manifest,
            (source,),
            (destination,),
            (_page(source, "layers.0", 100),),
            (_page(destination, "layers.0", 200),),
            (KVRange("attention", 0, 4, 4),),
        )

    source = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
    )
    stale_page = LocalPageAllocation(
        source.worker_id,
        "old-boot",
        "layers.0",
        100,
        0,
        0,
    )
    with pytest.raises(ValueError, match="stale worker incarnation"):
        _plan(
            format_manifest,
            format_manifest,
            (source,),
            (destination,),
            (stale_page,),
            (_page(destination, "layers.0", 200),),
            (KVRange("attention", 0, 4, 4),),
        )


def test_missing_layer_or_page_coverage_fails_closed():
    format_manifest = _format(_group(0, "attention", ("layers.0", "layers.1")))
    mapping = _identity()
    source = _worker(
        format_manifest,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
    )
    destination = _worker(
        format_manifest,
        "destination-0",
        20,
        (
            ("layers.0", 0, "attention", mapping),
            ("layers.1", 1, "attention", mapping),
        ),
        deployment_id="destination",
    )

    with pytest.raises(ValueError, match="incomplete layer ownership"):
        _plan(
            format_manifest,
            format_manifest,
            (source,),
            (destination,),
            (_page(source, "layers.0", 100),),
            (
                _page(destination, "layers.0", 200),
                _page(destination, "layers.1", 201),
            ),
            (KVRange("attention", 0, 4, 4),),
        )

    source_full = _worker(
        format_manifest,
        "source-0",
        10,
        (
            ("layers.0", 0, "attention", mapping),
            ("layers.1", 1, "attention", mapping),
        ),
        deployment_id="source",
    )
    with pytest.raises(ValueError, match="missing page coverage"):
        _plan(
            format_manifest,
            format_manifest,
            (source_full,),
            (destination,),
            (_page(source_full, "layers.0", 100),),
            (
                _page(destination, "layers.0", 200),
                _page(destination, "layers.1", 201),
            ),
            (KVRange("attention", 0, 4, 4),),
        )


def test_incompatible_canonical_spaces_fail_closed():
    source_format = _format(_group(0, "attention", ("layers.0",)))
    destination_format = _format(
        _group(0, "attention", ("layers.0",)),
        model_fingerprint="different-model",
    )
    mapping = _identity()
    source = _worker(
        source_format,
        "source-0",
        10,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="source",
    )
    destination = _worker(
        destination_format,
        "destination-0",
        20,
        (("layers.0", 0, "attention", mapping),),
        deployment_id="destination",
    )

    with pytest.raises(ValueError, match="incompatible canonical spaces"):
        _plan(
            source_format,
            destination_format,
            (source,),
            (destination,),
            (_page(source, "layers.0", 100),),
            (_page(destination, "layers.0", 200),),
            (KVRange("attention", 0, 4, 4),),
        )


def test_streaming_layer_planner_matches_eager_copy_bytes():
    format_manifest = _format(_group(0, "attention", ("layers.0",), span=8))
    even = _shard(0)
    odd = _shard(1)
    full = _identity(8)
    source_even = _worker(
        format_manifest,
        "source-even",
        10,
        (("layers.0", 0, "attention", even),),
        deployment_id="source",
        tp_size=2,
        tp_rank=0,
    )
    source_odd = _worker(
        format_manifest,
        "source-odd",
        11,
        (("layers.0", 0, "attention", odd),),
        deployment_id="source",
        tp_size=2,
        tp_rank=1,
    )
    destination = _worker(
        format_manifest,
        "destination",
        20,
        (("layers.0", 0, "attention", full),),
        deployment_id="destination",
    )
    source_pages = (
        _page(source_even, "layers.0", 100),
        _page(source_odd, "layers.0", 110),
    )
    destination_pages = (_page(destination, "layers.0", 200),)
    ranges = (KVRange("attention", 0, 8, 8),)
    context = build_request_transfer_context(
        source_format=format_manifest,
        destination_format=format_manifest,
        source_route=_route("source"),
        destination_route=_route("destination"),
        source_workers=(source_even, source_odd),
        destination_workers=(destination,),
        ranges=ranges,
    )
    eager = (
        _request_plan(
            format_manifest,
            format_manifest,
            (source_even, source_odd),
            (destination,),
            source_pages,
            destination_pages,
            ranges,
        )
        .layers[0]
        .runs
    )

    streamed = tuple(
        iter_request_transfer_layer_runs(
            context,
            context.layers[0],
            source_pages=source_pages,
            destination_pages=destination_pages,
            max_buffered_copy_fragments=2,
        )
    )

    def bytes_for(runs):
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

    assert bytes_for(streamed) == bytes_for(eager)
    assert sum(run.fragment_count for run in streamed) == 8


def test_streaming_destination_rank_filter_preserves_full_cohort_validation():
    format_manifest = _format(_group(0, "attention", ("layers.0",), span=8))
    source = _worker(
        format_manifest,
        "source",
        10,
        (("layers.0", 0, "attention", _identity(8)),),
        deployment_id="source",
    )
    destination_even = _worker(
        format_manifest,
        "destination-even",
        20,
        (("layers.0", 0, "attention", _shard(0)),),
        deployment_id="destination",
        tp_size=2,
        tp_rank=0,
    )
    destination_odd = _worker(
        format_manifest,
        "destination-odd",
        21,
        (("layers.0", 0, "attention", _shard(1)),),
        deployment_id="destination",
        tp_size=2,
        tp_rank=1,
    )
    source_pages = (_page(source, "layers.0", 100),)
    destination_pages = (
        _page(destination_even, "layers.0", 200),
        _page(destination_odd, "layers.0", 201),
    )
    context = build_request_transfer_context(
        source_format=format_manifest,
        destination_format=format_manifest,
        source_route=_route("source"),
        destination_route=_route("destination"),
        source_workers=(source,),
        destination_workers=(destination_even, destination_odd),
        ranges=(KVRange("attention", 0, 8, 8),),
    )

    runs = tuple(
        iter_request_transfer_layer_runs(
            context,
            context.layers[0],
            source_pages=source_pages,
            destination_pages=destination_pages,
            destination_rank=20,
        )
    )

    assert {run.destination_rank for run in runs} == {20}
    assert sum(run.fragment_size * run.fragment_count for run in runs) == 4
    with pytest.raises(ValueError, match="missing page coverage"):
        tuple(
            iter_request_transfer_layer_runs(
                context,
                context.layers[0],
                source_pages=source_pages,
                destination_pages=(destination_pages[0],),
                destination_rank=20,
            )
        )
