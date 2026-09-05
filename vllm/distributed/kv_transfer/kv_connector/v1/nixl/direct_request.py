# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lower a generic request plan into layer-scoped direct NIXL batches."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlPlacementMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.segmented import (
    NixlDirectDescriptorBatch,
    NixlPageRegistration,
    iter_nixl_direct_descriptor_batches,
    iter_nixl_direct_descriptor_batches_streaming,
)
from vllm.distributed.kv_transfer.kv_placement import KVRange
from vllm.distributed.kv_transfer.request_planner import (
    EndpointRoute,
    LocalPageAllocation,
    RequestLayerSpec,
    RequestTransferContext,
    RequestTransferPlan,
    iter_request_transfer_layer_runs,
)
from vllm.distributed.kv_transfer.transfer_completion import WorkerIdentity

MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH = 4096


@dataclass(frozen=True)
class NixlLayerDirectBatch:
    """One request layer's direct batch plus its endpoint-local group IDs."""

    layer_name: str
    semantic_group_id: str
    source_group_id: int
    destination_group_id: int
    batch: NixlDirectDescriptorBatch


@dataclass(frozen=True)
class _EndpointIndex:
    worker_by_id: dict[str, NixlPlacementMetadata]
    worker_id_by_rank: dict[int, str]
    max_segments_per_batch: int | None


def _index_endpoint(
    label: str,
    workers: Sequence[NixlPlacementMetadata],
    expected_participants: Sequence[WorkerIdentity],
    expected_route: EndpointRoute,
    operation: str,
) -> _EndpointIndex:
    expected_by_id = {item.worker_id: item for item in expected_participants}
    if len(expected_by_id) != len(expected_participants):
        raise ValueError(f"{label} participants contain duplicate worker IDs")

    worker_by_id: dict[str, NixlPlacementMetadata] = {}
    worker_id_by_rank: dict[int, str] = {}
    batch_limits: list[int] = []
    for worker in workers:
        if not isinstance(worker, NixlPlacementMetadata):
            raise ValueError(
                f"{label} workers must contain NixlPlacementMetadata values"
            )
        placement = worker.rank_placement
        if placement.deployment_id != expected_route.deployment_id:
            raise ValueError(
                f"{label} worker {placement.worker_id!r} belongs to deployment "
                f"{placement.deployment_id!r}, not "
                f"{expected_route.deployment_id!r}"
            )
        if placement.topology_generation != expected_route.topology_generation:
            raise ValueError(
                f"{label} worker {placement.worker_id!r} belongs to topology "
                f"generation {placement.topology_generation}, not "
                f"{expected_route.topology_generation}"
            )
        if (
            placement.dp_group_id != expected_route.dp_group_id
            or placement.dp_rank != expected_route.dp_rank
        ):
            raise ValueError(
                f"{label} worker {placement.worker_id!r} is outside the "
                "request's selected DP route"
            )
        capabilities = worker.capabilities
        supports_operation = (
            capabilities.supports_read
            if operation == "READ"
            else capabilities.supports_write
        )
        if not supports_operation:
            raise ValueError(
                f"{label} worker {placement.worker_id!r} does not support "
                f"NIXL {operation}"
            )
        if not capabilities.contiguous_copy:
            raise ValueError(
                f"{label} worker {placement.worker_id!r} cannot execute "
                "contiguous direct segments"
            )
        if not capabilities.scatter_gather:
            # Lack of a multi-segment primitive changes only submission
            # granularity: every fragment remains a direct one-segment copy.
            batch_limits.append(1)
        elif capabilities.max_segments_per_batch is not None:
            batch_limits.append(capabilities.max_segments_per_batch)
        if placement.worker_id in worker_by_id:
            raise ValueError(
                f"{label} endpoint has duplicate worker {placement.worker_id!r}"
            )
        previous_worker_id = worker_id_by_rank.setdefault(
            placement.rank, placement.worker_id
        )
        if previous_worker_id != placement.worker_id:
            raise ValueError(
                f"{label} endpoint has duplicate transfer rank {placement.rank}"
            )
        worker_by_id[placement.worker_id] = worker

    if set(worker_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(worker_by_id))
        unexpected = sorted(set(worker_by_id) - set(expected_by_id))
        raise ValueError(
            f"{label} worker set does not match request participants: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for worker_id, identity in expected_by_id.items():
        if (
            worker_by_id[worker_id].rank_placement.worker_incarnation
            != identity.worker_incarnation
        ):
            raise ValueError(
                f"{label} worker {worker_id!r} has a stale or wrong incarnation"
            )
    return _EndpointIndex(
        worker_by_id,
        worker_id_by_rank,
        min(batch_limits) if batch_limits else None,
    )


def _page_registrations(
    label: str,
    endpoint: _EndpointIndex,
    allocations: Sequence[LocalPageAllocation],
) -> dict[str, dict[tuple[int, int], NixlPageRegistration]]:
    registrations: dict[str, dict[tuple[int, int], NixlPageRegistration]] = {}
    for allocation in allocations:
        if not isinstance(allocation, LocalPageAllocation):
            raise ValueError(
                f"{label} allocations must contain LocalPageAllocation values"
            )
        try:
            worker = endpoint.worker_by_id[allocation.worker_id]
        except KeyError as error:
            raise ValueError(
                f"{label} allocation references non-participant worker "
                f"{allocation.worker_id!r}"
            ) from error
        placement = worker.rank_placement
        if allocation.worker_incarnation != placement.worker_incarnation:
            raise ValueError(
                f"{label} allocation for worker {allocation.worker_id!r} is stale"
            )
        if allocation.local_base != 0:
            raise ValueError(
                "generic NIXL page templates do not yet support a nonzero "
                f"local_base (worker {allocation.worker_id!r}, layer "
                f"{allocation.layer_name!r})"
            )
        templates = {
            template.layer_name: template
            for template in worker.page_registration_templates
        }
        try:
            template = templates[allocation.layer_name]
        except KeyError as error:
            raise ValueError(
                f"{label} worker {allocation.worker_id!r} has no page template "
                f"for layer {allocation.layer_name!r}"
            ) from error
        registration = NixlPageRegistration(
            base_address=template.page_address(allocation.local_page_id),
            length=template.page_size_bytes,
            device_id=template.device_id,
        )
        layer_registrations = registrations.setdefault(allocation.layer_name, {})
        key = (placement.rank, allocation.local_page_id)
        previous = layer_registrations.setdefault(key, registration)
        if previous != registration:
            raise ValueError(f"{label} allocation {key} has conflicting registration")
    return registrations


def iter_nixl_request_direct_batches(
    request_plan: RequestTransferPlan,
    *,
    source_workers: Sequence[NixlPlacementMetadata],
    destination_workers: Sequence[NixlPlacementMetadata],
    source_allocations: Sequence[LocalPageAllocation],
    destination_allocations: Sequence[LocalPageAllocation],
    operation: str = "READ",
    max_segments_per_batch: int | None = None,
) -> Iterator[NixlLayerDirectBatch]:
    """Stream all layer batches in a request without changing copy policy.

    The planner and page registrations stay layer-scoped, so allocator page
    IDs may be reused by different layers. ``max_segments_per_batch`` is
    forwarded solely as a bounded NIXL submission size; any remaining
    fragments become additional direct batches.
    """
    if not isinstance(request_plan, RequestTransferPlan):
        raise ValueError("request_plan must be a RequestTransferPlan")
    if operation not in ("READ", "WRITE"):
        raise ValueError("operation must be 'READ' or 'WRITE'")
    source = _index_endpoint(
        "source",
        source_workers,
        request_plan.source_participants,
        request_plan.source_route,
        operation,
    )
    destination = _index_endpoint(
        "destination",
        destination_workers,
        request_plan.destination_participants,
        request_plan.destination_route,
        operation,
    )
    advertised_limits = [MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH]
    advertised_limits.extend(
        limit
        for limit in (
            source.max_segments_per_batch,
            destination.max_segments_per_batch,
        )
        if limit is not None
    )
    if max_segments_per_batch is not None:
        advertised_limits.append(max_segments_per_batch)
    effective_batch_limit = min(advertised_limits) if advertised_limits else None
    source_pages = _page_registrations("source", source, source_allocations)
    destination_pages = _page_registrations(
        "destination", destination, destination_allocations
    )

    for layer_plan in request_plan.layers:
        for batch in iter_nixl_direct_descriptor_batches(
            layer_plan.runs,
            source_pages.get(layer_plan.layer_name, {}),
            destination_pages.get(layer_plan.layer_name, {}),
            max_segments_per_batch=effective_batch_limit,
        ):
            yield NixlLayerDirectBatch(
                layer_name=layer_plan.layer_name,
                semantic_group_id=layer_plan.semantic_group_id,
                source_group_id=layer_plan.source_group_id,
                destination_group_id=layer_plan.destination_group_id,
                batch=batch,
            )


def iter_nixl_request_layer_direct_batches(
    context: RequestTransferContext,
    layer: RequestLayerSpec,
    *,
    source_workers: Sequence[NixlPlacementMetadata],
    destination_workers: Sequence[NixlPlacementMetadata],
    source_allocations: Sequence[LocalPageAllocation],
    destination_allocations: Sequence[LocalPageAllocation],
    ranges: Sequence[KVRange] | None = None,
    operation: str = "READ",
    max_segments_per_batch: int | None = None,
    destination_rank: int | None = None,
) -> Iterator[NixlLayerDirectBatch]:
    """Compose and lower one layer without materializing its complete plan.

    Composition uses the connector's hard segment ceiling as its lookahead so
    a peer advertising a small submission limit does not prevent adjacent
    fragments from coalescing. Descriptor buffering still obeys the smaller
    effective endpoint limit. Additional fragmentation always produces more
    direct batches.
    """
    if not isinstance(context, RequestTransferContext):
        raise ValueError("context must be a RequestTransferContext")
    if operation not in ("READ", "WRITE"):
        raise ValueError("operation must be 'READ' or 'WRITE'")
    source = _index_endpoint(
        "source",
        source_workers,
        context.source_participants,
        context.source_route,
        operation,
    )
    destination = _index_endpoint(
        "destination",
        destination_workers,
        context.destination_participants,
        context.destination_route,
        operation,
    )
    advertised_limits = [MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH]
    advertised_limits.extend(
        limit
        for limit in (
            source.max_segments_per_batch,
            destination.max_segments_per_batch,
        )
        if limit is not None
    )
    if max_segments_per_batch is not None:
        advertised_limits.append(max_segments_per_batch)
    effective_batch_limit = min(advertised_limits)

    source_pages = _page_registrations("source", source, source_allocations)
    destination_pages = _page_registrations(
        "destination", destination, destination_allocations
    )
    transfer_runs = iter_request_transfer_layer_runs(
        context,
        layer,
        source_pages=source_allocations,
        destination_pages=destination_allocations,
        ranges=ranges,
        max_buffered_copy_fragments=MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH,
        destination_rank=destination_rank,
    )
    for batch in iter_nixl_direct_descriptor_batches_streaming(
        transfer_runs,
        source_pages.get(layer.layer_name, {}),
        destination_pages.get(layer.layer_name, {}),
        max_segments_per_batch=effective_batch_limit,
        max_buffered_segments=effective_batch_limit,
    ):
        yield NixlLayerDirectBatch(
            layer_name=layer.layer_name,
            semantic_group_id=layer.semantic_group_id,
            source_group_id=layer.source_group_id,
            destination_group_id=layer.destination_group_id,
            batch=batch,
        )


__all__ = [
    "MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH",
    "NixlLayerDirectBatch",
    "iter_nixl_request_layer_direct_batches",
    "iter_nixl_request_direct_batches",
]
