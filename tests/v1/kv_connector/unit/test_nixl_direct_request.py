# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.direct_request import (
    MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH,
    iter_nixl_request_direct_batches,
    iter_nixl_request_layer_direct_batches,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlPageRegistrationTemplate,
    NixlPlacementMetadata,
)
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    ConnectorCapabilities,
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
    LayerTransferPlan,
    LocalPageAllocation,
    RequestTransferPlan,
    build_request_transfer_context,
)
from vllm.distributed.kv_transfer.transfer_completion import WorkerIdentity


def _mapping() -> CanonicalPageMapping:
    return CanonicalPageMapping(
        canonical_page_size_bytes=8,
        local_page_size_bytes=8,
        runs=(CopyRun(0, 0, 8, 1, 8, 8),),
        num_writers=1,
        writer_index=0,
        canonical_token_span=8,
        canonical_region_token_strides=((0, 1),),
    )


def _format() -> KVFormatManifest:
    return KVFormatManifest(
        version=1,
        model_fingerprint="model-v1",
        groups=(
            KVGroupFormat(
                group_id=0,
                semantic_id="attention",
                kind="mla",
                layer_names=("layers.0", "layers.1"),
                canonical_page_token_span=8,
                dtype="uint8",
                canonical_page_size_bytes=8,
                format_id="test-token-major",
            ),
        ),
    )


def _worker(
    worker_id: str,
    rank: int,
    deployment_id: str,
    bases: tuple[int, int],
) -> NixlPlacementMetadata:
    format_manifest = _format()
    incarnation = f"{worker_id}-boot"
    placement = RankPlacementManifest(
        version=1,
        deployment_id=deployment_id,
        topology_generation=3,
        worker_id=worker_id,
        worker_incarnation=incarnation,
        format_manifest_fingerprint=format_manifest.fingerprint(),
        rank=rank,
        tp_size=1,
        tp_rank=0,
        dcp_size=1,
        dcp_rank=0,
        dcp_group_id=f"{deployment_id}-dcp",
        pcp_size=1,
        pcp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        dp_group_id=f"{deployment_id}-dp",
        ep_size=1,
        ep_rank=0,
        cp_interleave=1,
        layer_range=(0, 2),
        mappings=tuple(
            LayerPageMapping(layer, index, "attention", _mapping())
            for index, layer in enumerate(("layers.0", "layers.1"))
        ),
    )
    return NixlPlacementMetadata(
        format_manifest=format_manifest,
        rank_placement=placement,
        capabilities=ConnectorCapabilities(
            contiguous_copy=True,
            strided_copy=True,
            scatter_gather=True,
            gpu_pack_unpack=False,
            supports_read=True,
            supports_write=False,
            max_segments_per_batch=3,
        ),
        page_registration_templates=tuple(
            NixlPageRegistrationTemplate(
                layer_name=layer,
                base_address=base,
                page_stride=16,
                page_size_bytes=8,
                num_pages=8,
                device_id=rank,
            )
            for layer, base in zip(("layers.0", "layers.1"), bases)
        ),
    )


def _allocation(
    worker: NixlPlacementMetadata, layer: str, page_id: int
) -> LocalPageAllocation:
    placement = worker.rank_placement
    return LocalPageAllocation(
        worker_id=placement.worker_id,
        worker_incarnation=placement.worker_incarnation,
        layer_name=layer,
        local_page_id=page_id,
        canonical_page_index=0,
        first_token=0,
    )


def _request_plan(
    source: NixlPlacementMetadata, destination: NixlPlacementMetadata
) -> RequestTransferPlan:
    layers = tuple(
        LayerTransferPlan(
            layer_name=layer,
            layer_index=index,
            semantic_group_id="attention",
            source_group_id=0,
            destination_group_id=0,
            runs=(
                TransferRun(
                    source_rank=source.rank_placement.rank,
                    destination_rank=destination.rank_placement.rank,
                    source_page_id=2,
                    destination_page_id=3,
                    source_offset=0,
                    destination_offset=0,
                    fragment_size=1,
                    fragment_count=8,
                    source_stride=1,
                    destination_stride=1,
                ),
            ),
        )
        for index, layer in enumerate(("layers.0", "layers.1"))
    )
    return RequestTransferPlan(
        layers=layers,
        source_route=EndpointRoute("source", 3, "source-dp", 0),
        destination_route=EndpointRoute("destination", 3, "destination-dp", 0),
        source_participants=(
            WorkerIdentity(
                source.rank_placement.worker_id,
                source.rank_placement.worker_incarnation,
            ),
        ),
        destination_participants=(
            WorkerIdentity(
                destination.rank_placement.worker_id,
                destination.rank_placement.worker_incarnation,
            ),
        ),
    )


def test_request_lowering_streams_bounded_direct_batches_per_layer():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))
    allocations = {
        "source": tuple(
            _allocation(source, layer, 2) for layer in ("layers.0", "layers.1")
        ),
        "destination": tuple(
            _allocation(destination, layer, 3) for layer in ("layers.0", "layers.1")
        ),
    }

    batches = tuple(
        iter_nixl_request_direct_batches(
            _request_plan(source, destination),
            source_workers=(source,),
            destination_workers=(destination,),
            source_allocations=allocations["source"],
            destination_allocations=allocations["destination"],
            max_segments_per_batch=3,
        )
    )

    assert [(item.layer_name, item.batch.segment_count) for item in batches] == [
        ("layers.0", 3),
        ("layers.0", 3),
        ("layers.0", 2),
        ("layers.1", 3),
        ("layers.1", 3),
        ("layers.1", 2),
    ]
    # The same allocator page IDs are resolved against each named layer's
    # independent base address.
    assert batches[0].batch.source_descriptors[0] == (1000 + 2 * 16, 1, 10)
    assert batches[3].batch.source_descriptors[0] == (2000 + 2 * 16, 1, 10)
    assert batches[0].batch.destination_descriptors[0] == (3000 + 3 * 16, 1, 20)
    assert all(
        item.source_group_id == item.destination_group_id == 0 for item in batches
    )


def test_request_lowering_rejects_missing_participant_and_stale_allocation():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))
    plan = _request_plan(source, destination)

    with pytest.raises(ValueError, match="worker set.*missing"):
        tuple(
            iter_nixl_request_direct_batches(
                plan,
                source_workers=(),
                destination_workers=(destination,),
                source_allocations=(),
                destination_allocations=(),
            )
        )

    stale = LocalPageAllocation("source-0", "old-boot", "layers.0", 2, 0, 0)
    with pytest.raises(ValueError, match="allocation.*stale"):
        tuple(
            iter_nixl_request_direct_batches(
                plan,
                source_workers=(source,),
                destination_workers=(destination,),
                source_allocations=(stale,),
                destination_allocations=(),
            )
        )


def test_request_lowering_rejects_page_id_outside_registration():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))

    with pytest.raises(ValueError, match="page_id must be"):
        tuple(
            iter_nixl_request_direct_batches(
                _request_plan(source, destination),
                source_workers=(source,),
                destination_workers=(destination,),
                source_allocations=(
                    _allocation(source, "layers.0", 8),
                    _allocation(source, "layers.1", 2),
                ),
                destination_allocations=tuple(
                    _allocation(destination, layer, 3)
                    for layer in ("layers.0", "layers.1")
                ),
            )
        )


def test_request_lowering_validates_route_and_allocation_base():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))
    plan = _request_plan(source, destination)
    wrong_route = replace(
        plan,
        destination_route=EndpointRoute("destination", 4, "destination-dp", 0),
    )

    with pytest.raises(ValueError, match="topology generation"):
        tuple(
            iter_nixl_request_direct_batches(
                wrong_route,
                source_workers=(source,),
                destination_workers=(destination,),
                source_allocations=(),
                destination_allocations=(),
            )
        )

    based = replace(_allocation(source, "layers.0", 2), local_base=1)
    with pytest.raises(ValueError, match="nonzero local_base"):
        tuple(
            iter_nixl_request_direct_batches(
                plan,
                source_workers=(source,),
                destination_workers=(destination,),
                source_allocations=(based,),
                destination_allocations=(),
            )
        )


def test_capabilities_only_change_direct_submission_chunking():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))
    source = replace(
        source,
        capabilities=replace(
            source.capabilities,
            scatter_gather=False,
            max_segments_per_batch=None,
        ),
    )
    allocations = tuple(
        _allocation(worker, layer, page_id)
        for worker, page_id in ((source, 2), (destination, 3))
        for layer in ("layers.0", "layers.1")
    )

    batches = tuple(
        iter_nixl_request_direct_batches(
            _request_plan(source, destination),
            source_workers=(source,),
            destination_workers=(destination,),
            source_allocations=allocations[:2],
            destination_allocations=allocations[2:],
            max_segments_per_batch=8,
        )
    )

    assert len(batches) == 16
    assert all(item.batch.segment_count == 1 for item in batches)

    no_read = replace(
        source,
        capabilities=replace(
            source.capabilities,
            supports_read=False,
            supports_write=True,
        ),
    )
    with pytest.raises(ValueError, match="does not support NIXL READ"):
        tuple(
            iter_nixl_request_direct_batches(
                _request_plan(no_read, destination),
                source_workers=(no_read,),
                destination_workers=(destination,),
                source_allocations=(),
                destination_allocations=(),
            )
        )


def test_request_lowering_enforces_local_hard_descriptor_ceiling():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))
    source = replace(
        source,
        capabilities=replace(source.capabilities, max_segments_per_batch=None),
    )
    destination = replace(
        destination,
        capabilities=replace(destination.capabilities, max_segments_per_batch=None),
    )
    request = _request_plan(source, destination)
    repeated_run = replace(request.layers[0].runs[0], fragment_count=1)
    request = replace(
        request,
        layers=(
            replace(
                request.layers[0],
                runs=(repeated_run,) * (MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH + 1),
            ),
        ),
    )

    batches = tuple(
        iter_nixl_request_direct_batches(
            request,
            source_workers=(source,),
            destination_workers=(destination,),
            source_allocations=(_allocation(source, "layers.0", 2),),
            destination_allocations=(_allocation(destination, "layers.0", 3),),
        )
    )

    assert [batch.batch.segment_count for batch in batches] == [
        MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH,
        1,
    ]


def test_streaming_request_layer_composes_and_lowers_without_eager_plan():
    source = _worker("source-0", 10, "source", (1000, 2000))
    destination = _worker("destination-0", 20, "destination", (3000, 4000))
    context = build_request_transfer_context(
        source_format=source.format_manifest,
        destination_format=destination.format_manifest,
        source_route=EndpointRoute("source", 3, "source-dp", 0),
        destination_route=EndpointRoute("destination", 3, "destination-dp", 0),
        source_workers=(source.rank_placement,),
        destination_workers=(destination.rank_placement,),
        ranges=(KVRange("attention", 0, 8, 8),),
    )
    source_page = _allocation(source, "layers.0", 2)
    destination_page = _allocation(destination, "layers.0", 3)

    batches = tuple(
        iter_nixl_request_layer_direct_batches(
            context,
            context.layers[0],
            source_workers=(source,),
            destination_workers=(destination,),
            source_allocations=(source_page,),
            destination_allocations=(destination_page,),
            destination_rank=20,
        )
    )

    assert len(batches) == 1
    assert batches[0].layer_name == "layers.0"
    assert batches[0].batch.source_descriptors == ((1000 + 2 * 16, 8, 10),)
    assert batches[0].batch.destination_descriptors == ((3000 + 3 * 16, 8, 20),)
    assert batches[0].batch.batch_count is None
    assert batches[0].batch.requires_aggregate_completion
