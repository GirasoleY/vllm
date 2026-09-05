# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge NIXL request metadata to generic KV placement plans.

The scheduler-facing NIXL protocol currently carries positional block-ID
groups: a remote prefix beginning at token zero and an unhashed destination
suffix beginning after the local prefix-cache hit.  This module is the narrow
adapter from that representation to semantic KV ranges and worker-local page
allocations.  The generic planner remains independent of NIXL and allocator
block IDs remain endpoint-local.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.direct_request import (
    MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH,
    iter_nixl_request_layer_direct_batches,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlPlacementMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.segmented import (
    NixlEphemeralDlistTracker,
    NixlPreparedDirectBatch,
    prepare_nixl_direct_batch,
)
from vllm.distributed.kv_transfer.kv_placement import KVFormatManifest, KVRange
from vllm.distributed.kv_transfer.request_planner import (
    EndpointRoute,
    LayerTransferPlan,
    LocalPageAllocation,
    RequestLayerSpec,
    RequestTransferContext,
    RequestTransferPlan,
    build_request_transfer_context,
    build_request_transfer_plan,
    plan_request_transfer_layer,
)
from vllm.distributed.kv_transfer.transfer_completion import (
    MAX_TRANSFER_PARTICIPANTS,
    TransferCompletionNotification,
    WorkerIdentity,
    participant_set_digest,
)

# The wire API remains a two-tuple. The second value is the PP-local
# ``pcp_rank * tp_size + tp_rank`` placement rank, not necessarily a TP rank.
HandshakeCoordinate = tuple[int, int]
BlockIdGroups = Sequence[Sequence[int]]
NIXL_DIRECT_COMPLETION_PREFIX = b"NIXL_DIRECT_COMPLETE_V1:"
MAX_NIXL_COMPLETION_BYTES = 1024 * 1024
DEFAULT_NIXL_CANONICAL_PAGE_WINDOW = 64

# Address, per-page payload, distance between pages, page capacity, device.
_RegisteredRegion = tuple[int, int, int, int, int]


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _ceil_div(dividend: int, divisor: int) -> int:
    return -(-dividend // divisor)


def _floor_sum(count: int, modulus: int, step: int, offset: int) -> int:
    """Return ``sum(floor((step * i + offset) / modulus))`` compactly."""
    total = 0
    while True:
        if step >= modulus:
            total += (count - 1) * count * (step // modulus) // 2
            step %= modulus
        if offset >= modulus:
            total += count * (offset // modulus)
            offset %= modulus
        maximum = step * count + offset
        if maximum < modulus:
            return total
        count = maximum // modulus
        offset = maximum % modulus
        modulus, step = step, modulus


def _modular_progression_hits_prefix(
    *,
    first: int,
    step: int,
    modulus: int,
    count: int,
    limit: int,
) -> bool:
    """Return whether one finite modular progression contains ``<= limit``."""
    if limit >= modulus - 1:
        return True
    first %= modulus
    step %= modulus
    threshold = limit + 1
    at_or_above = _floor_sum(
        count, modulus, step, first + modulus - threshold
    ) - _floor_sum(count, modulus, step, first)
    return at_or_above < count


def _registered_regions_overlap(
    first: _RegisteredRegion,
    second: _RegisteredRegion,
) -> bool:
    """Exactly compare two finite affine streams of half-open page intervals."""
    first_base, first_size, first_stride, first_count, first_device = first
    second_base, second_size, second_stride, second_count, second_device = second
    if first_device != second_device:
        return False

    first_extent_end = first_base + (first_count - 1) * first_stride + first_size
    second_extent_end = second_base + (second_count - 1) * second_stride + second_size
    if first_extent_end <= second_base or second_extent_end <= first_base:
        return False

    # For second page ``j``, an overlapping first-page start must be a multiple
    # of ``first_stride`` in [delta + j * second_stride - first_size + 1,
    # delta + j * second_stride + second_size - 1], relative to first_base.
    # Restrict j to intervals intersecting the bounded first-page start range.
    delta = second_base - first_base
    last_first_start = (first_count - 1) * first_stride
    second_index_min = max(
        0,
        _ceil_div(-second_size + 1 - delta, second_stride),
    )
    second_index_max = min(
        second_count - 1,
        (last_first_start + first_size - 1 - delta) // second_stride,
    )
    if second_index_min > second_index_max:
        return False

    # Intervals containing either endpoint of the bounded first progression
    # trivially overlap. The remaining intervals lie strictly between them.
    if second_index_min <= min(
        second_index_max,
        (first_size - 1 - delta) // second_stride,
    ):
        return True
    if (
        max(
            second_index_min,
            _ceil_div(
                last_first_start - second_size + 1 - delta,
                second_stride,
            ),
        )
        <= second_index_max
    ):
        return True

    middle_index_min = max(
        second_index_min,
        _ceil_div(first_size - delta, second_stride),
    )
    middle_index_max = min(
        second_index_max,
        (last_first_start - second_size - delta) // second_stride,
    )
    if middle_index_min > middle_index_max:
        return False

    interval_width = first_size + second_size - 2
    first_upper_bound = delta + middle_index_min * second_stride + second_size - 1
    return _modular_progression_hits_prefix(
        first=first_upper_bound,
        step=second_stride,
        modulus=first_stride,
        count=middle_index_max - middle_index_min + 1,
        limit=interval_width,
    )


def _require_disjoint_registered_pages(
    regions: Sequence[_RegisteredRegion],
    description: str,
) -> None:
    """Reject overlapping pages without expanding attacker-sized page streams."""
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            if _registered_regions_overlap(first, second):
                raise ValueError(f"{description} contain overlapping memory pages")


def _validate_placement_registration_binding(
    metadata: NixlAgentMetadata,
    placement: NixlPlacementMetadata,
) -> None:
    """Bind named placement pages to the legacy NIXL registration regions.

    Legacy metadata does not carry layer names, and its region order follows
    cache registration order rather than placement-manifest order.  Therefore
    this check deliberately compares an unordered, unique set of complete
    region signatures instead of zipping the two representations.

    The currently activated generic runtime slice has exactly one registered
    region per named attention layer.  Dense virtual pages, compressed aliases,
    and layouts that split one layer over multiple legacy regions cannot be
    proven from this wire schema and must fail closed rather than guessing an
    ordering or containment relationship.
    """
    rank_placement = placement.rank_placement
    if rank_placement.deployment_id != metadata.engine_id:
        raise ValueError(
            f"placement deployment {rank_placement.deployment_id!r} does not "
            f"match NIXL engine {metadata.engine_id!r}"
        )

    _require_positive_int(metadata.num_blocks, "remote NIXL num_blocks")
    _require_nonnegative_int(metadata.device_id, "remote NIXL device_id")
    region_count = len(metadata.kv_caches_base_addr)
    if not (region_count == len(metadata.block_lens) == len(metadata.block_strides)):
        raise ValueError(
            "legacy NIXL registered-region address, length, and stride arrays "
            "must have equal lengths"
        )
    if region_count != len(placement.page_registration_templates):
        raise ValueError(
            "generic placement requires exactly one named page template per "
            "legacy registered region; compressed, aliased, and multi-region "
            "layouts must not advertise this protocol slice"
        )

    legacy_regions: list[_RegisteredRegion] = []
    for region_index, (base_address, page_size, page_stride) in enumerate(
        zip(
            metadata.kv_caches_base_addr,
            metadata.block_lens,
            metadata.block_strides,
        )
    ):
        _require_nonnegative_int(
            base_address, f"legacy registered region {region_index} base address"
        )
        _require_positive_int(
            page_size, f"legacy registered region {region_index} page size"
        )
        _require_positive_int(
            page_stride, f"legacy registered region {region_index} page stride"
        )
        if page_stride < page_size:
            raise ValueError(
                f"legacy registered region {region_index} page stride is smaller "
                "than its page size"
            )
        legacy_regions.append(
            (
                base_address,
                page_size,
                page_stride,
                metadata.num_blocks,
                metadata.device_id,
            )
        )
    if len({region[0] for region in legacy_regions}) != len(legacy_regions):
        raise ValueError(
            "legacy NIXL registered regions contain an ambiguous base-address alias"
        )
    _require_disjoint_registered_pages(
        legacy_regions,
        "legacy NIXL registered regions",
    )

    template_regions: dict[_RegisteredRegion, str] = {}
    template_layers_by_base: dict[int, str] = {}
    for template in placement.page_registration_templates:
        aliased_layer = template_layers_by_base.get(template.base_address)
        if aliased_layer is not None:
            raise ValueError(
                f"placement layers {aliased_layer!r} and {template.layer_name!r} "
                "ambiguously alias one registered base address"
            )
        template_layers_by_base[template.base_address] = template.layer_name
        signature = (
            template.base_address,
            template.page_size_bytes,
            template.page_stride,
            template.num_pages,
            template.device_id,
        )
        template_regions[signature] = template.layer_name

    _require_disjoint_registered_pages(
        tuple(template_regions),
        "placement page templates",
    )

    legacy_region_set = set(legacy_regions)
    for signature, layer_name in template_regions.items():
        if signature not in legacy_region_set:
            raise ValueError(
                f"placement page template for layer {layer_name!r} is not exactly "
                "bound to a legacy NIXL registered region"
            )


@dataclass(frozen=True)
class NixlRemotePlacementIndex:
    """Remote placement payloads and NIXL agents keyed by transfer rank."""

    engine_id: str
    workers: tuple[NixlPlacementMetadata, ...]
    agent_names: tuple[tuple[int, str], ...]
    physical_pages_per_logical: int

    @property
    def agent_names_by_rank(self) -> dict[int, str]:
        return dict(self.agent_names)


@dataclass(frozen=True)
class NixlReadRequestPlan:
    """Compact inputs for incremental direct NIXL request planning."""

    planning_context: RequestTransferContext
    source_workers: tuple[NixlPlacementMetadata, ...]
    destination_workers: tuple[NixlPlacementMetadata, ...]
    source_block_ids: tuple[tuple[int, ...], ...]
    destination_block_ids: tuple[tuple[int, ...], ...]
    destination_prefix_blocks: tuple[int, ...]
    source_physical_pages_per_logical: int
    destination_physical_pages_per_logical: int
    max_segments_per_batch: int | None
    max_pages_per_window: int

    def __post_init__(self) -> None:
        if (
            tuple(worker.rank_placement for worker in self.source_workers)
            != self.planning_context.source_workers
        ):
            raise ValueError("source workers do not match the planning context")
        if (
            tuple(worker.rank_placement for worker in self.destination_workers)
            != self.planning_context.destination_workers
        ):
            raise ValueError("destination workers do not match the planning context")
        _require_positive_int(
            self.source_physical_pages_per_logical,
            "source_physical_pages_per_logical",
        )
        _require_positive_int(
            self.destination_physical_pages_per_logical,
            "destination_physical_pages_per_logical",
        )
        _require_positive_int(self.max_pages_per_window, "max_pages_per_window")
        if self.max_segments_per_batch is not None:
            _require_positive_int(self.max_segments_per_batch, "max_segments_per_batch")
            if self.max_segments_per_batch > MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH:
                raise ValueError(
                    "max_segments_per_batch exceeds the local hard ceiling"
                )

    @property
    def ranges(self) -> tuple[KVRange, ...]:
        """Return validated canonical request ranges."""
        return self.planning_context.ranges

    @property
    def source_route(self) -> EndpointRoute:
        """Return the selected source endpoint route."""
        return self.planning_context.source_route

    @property
    def destination_route(self) -> EndpointRoute:
        """Return the selected destination endpoint route."""
        return self.planning_context.destination_route

    @property
    def source_participants(self) -> tuple[WorkerIdentity, ...]:
        """Return the immutable source completion participants."""
        return self.planning_context.source_participants

    @property
    def destination_participants(self) -> tuple[WorkerIdentity, ...]:
        """Return the immutable destination completion participants."""
        return self.planning_context.destination_participants

    @property
    def destination_expected_participant_count(self) -> int:
        """Return the destination-side completion barrier size."""
        return self.planning_context.destination_expected_participant_count


@dataclass(frozen=True)
class NixlReadPlanWindow:
    """One layer and bounded canonical-page window ready for NIXL lowering."""

    layer_plan: LayerTransferPlan
    source_allocations: tuple[LocalPageAllocation, ...]
    destination_allocations: tuple[LocalPageAllocation, ...]
    kv_range: KVRange


@dataclass(frozen=True)
class NixlPreparedReadRequest:
    """All direct batches prepared atomically but not yet submitted."""

    request: NixlReadRequestPlan
    batches: tuple[NixlPreparedDirectBatch, ...]

    @property
    def transfer_handles(self) -> tuple[int, ...]:
        return tuple(batch.transfer_handle for batch in self.batches)


@dataclass(frozen=True)
class NixlDirectCompletionEnvelope:
    """One generic completion and its exact destination participant set."""

    notification: TransferCompletionNotification
    expected_participants: tuple[WorkerIdentity, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.notification, TransferCompletionNotification):
            raise ValueError("notification must be a TransferCompletionNotification")
        participants = tuple(self.expected_participants)
        object.__setattr__(self, "expected_participants", participants)
        participant_set_digest(participants)
        if self.notification.expected_participant_count != len(participants):
            raise ValueError(
                "completion participant set does not match the advertised count"
            )
        expected_sender = next(
            (
                participant
                for participant in participants
                if participant.worker_id == self.notification.sender_worker_id
            ),
            None,
        )
        if (
            expected_sender is None
            or expected_sender.worker_incarnation
            != self.notification.sender_worker_incarnation
        ):
            raise ValueError("completion sender is not an expected participant")

    def encode(self) -> bytes:
        """Return a strict, versioned NIXL notification payload."""
        body = {
            "notification": self.notification.to_dict(),
            "expected_participants": [
                {
                    "worker_id": participant.worker_id,
                    "worker_incarnation": participant.worker_incarnation,
                }
                for participant in self.expected_participants
            ],
        }
        payload = (
            NIXL_DIRECT_COMPLETION_PREFIX
            + json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        )
        if len(payload) > MAX_NIXL_COMPLETION_BYTES:
            raise ValueError(
                "generic NIXL completion exceeds the maximum wire size of "
                f"{MAX_NIXL_COMPLETION_BYTES} bytes"
            )
        return payload

    @classmethod
    def decode(cls, payload: bytes) -> "NixlDirectCompletionEnvelope":
        """Decode a generic NIXL completion, rejecting loose wire shapes."""
        if not isinstance(payload, bytes):
            raise ValueError("payload is not a generic NIXL completion")
        if len(payload) > MAX_NIXL_COMPLETION_BYTES:
            raise ValueError(
                "generic NIXL completion exceeds the maximum wire size of "
                f"{MAX_NIXL_COMPLETION_BYTES} bytes"
            )
        if not payload.startswith(NIXL_DIRECT_COMPLETION_PREFIX):
            raise ValueError("payload is not a generic NIXL completion")
        try:
            data = json.loads(payload[len(NIXL_DIRECT_COMPLETION_PREFIX) :])
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            raise ValueError("generic NIXL completion is not valid JSON") from error
        if not isinstance(data, dict) or set(data) != {
            "notification",
            "expected_participants",
        }:
            raise ValueError("generic NIXL completion has invalid envelope fields")
        raw_participants = data["expected_participants"]
        if not isinstance(raw_participants, list):
            raise ValueError("generic NIXL completion participants must be an array")
        if len(raw_participants) > MAX_TRANSFER_PARTICIPANTS:
            raise ValueError(
                "generic NIXL completion must contain at most "
                f"{MAX_TRANSFER_PARTICIPANTS} participants"
            )
        notification = TransferCompletionNotification.from_dict(data["notification"])
        if len(raw_participants) != notification.expected_participant_count:
            raise ValueError(
                "completion participant set does not match the advertised count"
            )
        participants: list[WorkerIdentity] = []
        for item in raw_participants:
            if not isinstance(item, dict) or set(item) != {
                "worker_id",
                "worker_incarnation",
            }:
                raise ValueError("generic NIXL completion has an invalid participant")
            participants.append(WorkerIdentity(**item))
        return cls(
            notification=notification,
            expected_participants=tuple(participants),
        )


def nixl_read_request_plan_digest(request: NixlReadRequestPlan) -> str:
    """Bind completion to the compact inputs that determine every byte run."""
    if not isinstance(request, NixlReadRequestPlan):
        raise ValueError("request must be a NixlReadRequestPlan")
    context = request.planning_context
    data = {
        "version": 2,
        "source_format": request.source_workers[0].format_manifest.fingerprint(),
        "destination_format": (
            request.destination_workers[0].format_manifest.fingerprint()
        ),
        "source_route": asdict(context.source_route),
        "destination_route": asdict(context.destination_route),
        "source_workers": [
            worker.to_dict()
            for worker in sorted(
                request.source_workers,
                key=lambda worker: worker.rank_placement.worker_id,
            )
        ],
        "destination_workers": [
            worker.to_dict()
            for worker in sorted(
                request.destination_workers,
                key=lambda worker: worker.rank_placement.worker_id,
            )
        ],
        "source_participants": [
            asdict(participant) for participant in context.source_participants
        ],
        "destination_participants": [
            asdict(participant) for participant in context.destination_participants
        ],
        "source_block_ids": request.source_block_ids,
        "destination_block_ids": request.destination_block_ids,
        "destination_prefix_blocks": request.destination_prefix_blocks,
        "source_physical_pages_per_logical": (
            request.source_physical_pages_per_logical
        ),
        "destination_physical_pages_per_logical": (
            request.destination_physical_pages_per_logical
        ),
        "ranges": [asdict(kv_range) for kv_range in request.ranges],
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{sha256(encoded).hexdigest()}"


def validate_complete_nixl_placement_endpoint(
    workers: Sequence[NixlPlacementMetadata],
    *,
    dp_rank: int | None = None,
) -> tuple[NixlPlacementMetadata, ...]:
    """Validate and deterministically order one complete placement endpoint.

    Every pull worker plans against the complete destination TP replica, then
    executes only the runs targeting itself. This helper lets the runtime
    validate a one-time local metadata gather with the same strict endpoint
    invariants as request planning, without manufacturing request pages.
    """
    format_manifest, route, selected = _select_endpoint("endpoint", workers, dp_rank)
    # Empty ranges require no page allocations, but endpoint binding still
    # validates the exact PP x PCP x TP membership and all rank coordinates.
    build_request_transfer_plan(
        source_format=format_manifest,
        destination_format=format_manifest,
        source_route=route,
        destination_route=route,
        source_workers=tuple(worker.rank_placement for worker in selected),
        destination_workers=tuple(worker.rank_placement for worker in selected),
        source_pages=(),
        destination_pages=(),
        ranges=(),
    )
    return tuple(sorted(selected, key=lambda worker: worker.rank_placement.rank))


def index_remote_nixl_placements(
    metadata_by_coordinate: Mapping[HandshakeCoordinate, NixlAgentMetadata],
    agent_names_by_coordinate: Mapping[HandshakeCoordinate, str],
) -> NixlRemotePlacementIndex:
    """Bind two-axis handshakes to flattened placement ranks.

    The returned rank keys are the PP x PCP x TP ranks advertised by the
    placement protocol. A handshake coordinate is
    ``(pp_rank, pcp_rank * tp_size + tp_rank)``; flattening PCP into the second
    coordinate preserves the existing two-tuple wire API without aliasing PCP
    workers.
    """
    if set(metadata_by_coordinate) != set(agent_names_by_coordinate):
        missing_metadata = sorted(
            set(agent_names_by_coordinate) - set(metadata_by_coordinate)
        )
        missing_agents = sorted(
            set(metadata_by_coordinate) - set(agent_names_by_coordinate)
        )
        raise ValueError(
            "remote placement and agent coordinates differ: "
            f"missing_metadata={missing_metadata}, missing_agents={missing_agents}"
        )
    if not metadata_by_coordinate:
        raise ValueError("remote placement index must not be empty")

    engine_ids: set[str] = set()
    workers: list[NixlPlacementMetadata] = []
    agent_names: dict[int, str] = {}
    worker_ids: set[str] = set()
    worker_incarnations: set[str] = set()
    seen_agent_names: set[str] = set()
    topology: tuple[int, int, int, int, str, int] | None = None
    physical_pages_per_logical: int | None = None
    for coordinate in sorted(metadata_by_coordinate):
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 2
            or any(
                not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
                for rank in coordinate
            )
        ):
            raise ValueError(
                "NIXL handshake coordinates must be non-negative "
                "(pp_rank, PP-local placement rank) pairs"
            )
        pp_rank, pp_local_rank = coordinate
        metadata = metadata_by_coordinate[coordinate]
        if not isinstance(metadata, NixlAgentMetadata):
            raise ValueError("remote metadata must contain NixlAgentMetadata values")
        placement = metadata.placement_metadata
        if placement is None:
            raise ValueError(
                f"remote NIXL worker at {coordinate} did not advertise placement"
            )
        _validate_placement_registration_binding(metadata, placement)
        rank_placement = placement.rank_placement
        if (
            rank_placement.dcp_size != metadata.dcp_size
            or rank_placement.pcp_size != metadata.pcp_size
            or rank_placement.cp_interleave != metadata.cp_kv_cache_interleave_size
        ):
            raise ValueError(
                "remote placement topology does not match legacy NIXL metadata"
            )
        if physical_pages_per_logical is None:
            physical_pages_per_logical = metadata.physical_blocks_per_logical_kv_block
        elif (
            physical_pages_per_logical != metadata.physical_blocks_per_logical_kv_block
        ):
            raise ValueError(
                "remote placements advertise inconsistent allocator page geometry"
            )
        expected_pp_local_rank = (
            rank_placement.pcp_rank * rank_placement.tp_size + rank_placement.tp_rank
        )
        if (rank_placement.pp_rank, expected_pp_local_rank) != coordinate:
            raise ValueError(
                f"remote placement coordinate for worker "
                f"{rank_placement.worker_id!r} does not match handshake "
                f"coordinate {coordinate}"
            )
        expected_rank = (
            pp_rank * rank_placement.pcp_size * rank_placement.tp_size + pp_local_rank
        )
        if rank_placement.rank != expected_rank:
            raise ValueError(
                f"remote placement rank {rank_placement.rank} does not match "
                f"flattened handshake rank {expected_rank}"
            )
        worker_topology = (
            rank_placement.pp_size,
            rank_placement.pcp_size,
            rank_placement.tp_size,
            rank_placement.dp_rank,
            rank_placement.dp_group_id,
            rank_placement.topology_generation,
        )
        if topology is None:
            topology = worker_topology
        elif topology != worker_topology:
            raise ValueError("remote placements advertise inconsistent topology")
        if rank_placement.worker_id in worker_ids:
            raise ValueError(
                f"duplicate remote placement worker {rank_placement.worker_id!r}"
            )
        worker_ids.add(rank_placement.worker_id)
        if rank_placement.worker_incarnation in worker_incarnations:
            raise ValueError(
                "duplicate remote placement worker incarnation "
                f"{rank_placement.worker_incarnation!r}"
            )
        worker_incarnations.add(rank_placement.worker_incarnation)
        if rank_placement.rank in agent_names:
            raise ValueError(f"duplicate remote placement rank {rank_placement.rank}")
        agent_name = agent_names_by_coordinate[coordinate]
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("remote NIXL agent names must be non-empty strings")
        if agent_name != rank_placement.worker_incarnation:
            raise ValueError(
                f"remote NIXL agent {agent_name!r} does not match placement "
                f"worker incarnation {rank_placement.worker_incarnation!r}"
            )
        if agent_name in seen_agent_names:
            raise ValueError(f"duplicate remote NIXL agent name {agent_name!r}")
        seen_agent_names.add(agent_name)
        agent_names[rank_placement.rank] = agent_name
        engine_ids.add(metadata.engine_id)
        workers.append(placement)

    if len(engine_ids) != 1:
        raise ValueError("remote placement metadata mixes multiple NIXL engines")
    assert topology is not None
    assert physical_pages_per_logical is not None
    pp_size, pcp_size, tp_size, _, _, _ = topology
    expected_coordinates = {
        (pp_rank, pp_local_rank)
        for pp_rank in range(pp_size)
        for pp_local_rank in range(pcp_size * tp_size)
    }
    if set(metadata_by_coordinate) != expected_coordinates:
        missing = sorted(expected_coordinates - set(metadata_by_coordinate))
        unexpected = sorted(set(metadata_by_coordinate) - expected_coordinates)
        raise ValueError(
            "remote placement handshake is incomplete: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return NixlRemotePlacementIndex(
        engine_id=engine_ids.pop(),
        workers=tuple(sorted(workers, key=lambda item: item.rank_placement.rank)),
        agent_names=tuple(sorted(agent_names.items())),
        physical_pages_per_logical=physical_pages_per_logical,
    )


def _select_endpoint(
    label: str,
    workers: Sequence[NixlPlacementMetadata],
    dp_rank: int | None,
) -> tuple[KVFormatManifest, EndpointRoute, tuple[NixlPlacementMetadata, ...]]:
    workers = tuple(workers)
    if not workers or any(
        not isinstance(worker, NixlPlacementMetadata) for worker in workers
    ):
        raise ValueError(f"{label} workers must contain NixlPlacementMetadata values")
    placements = tuple(worker.rank_placement for worker in workers)
    if dp_rank is None:
        dp_ranks = {placement.dp_rank for placement in placements}
        if len(dp_ranks) != 1:
            raise ValueError(f"{label} endpoint requires an explicit DP route")
        dp_rank = dp_ranks.pop()
    _require_nonnegative_int(dp_rank, f"{label} dp_rank")
    selected = tuple(
        worker for worker in workers if worker.rank_placement.dp_rank == dp_rank
    )
    if not selected:
        raise ValueError(f"{label} DP route {dp_rank} has no placement workers")

    first = selected[0]
    first_placement = first.rank_placement
    route = EndpointRoute(
        deployment_id=first_placement.deployment_id,
        topology_generation=first_placement.topology_generation,
        dp_group_id=first_placement.dp_group_id,
        dp_rank=dp_rank,
    )
    format_manifest = first.format_manifest
    for worker in selected:
        placement = worker.rank_placement
        if (
            placement.deployment_id != route.deployment_id
            or placement.topology_generation != route.topology_generation
            or placement.dp_group_id != route.dp_group_id
        ):
            raise ValueError(f"{label} workers do not share one endpoint generation")
        if worker.format_manifest != format_manifest:
            raise ValueError(f"{label} workers advertise inconsistent KV formats")
        capabilities = worker.capabilities
        if not (capabilities.supports_read and capabilities.contiguous_copy):
            raise ValueError(
                f"{label} worker {placement.worker_id!r} does not support "
                "segmented direct READ"
            )
    return format_manifest, route, selected


def _validate_positional_groups(
    label: str,
    manifest: KVFormatManifest,
    block_ids: BlockIdGroups,
) -> tuple[tuple[int, ...], ...]:
    group_ids = {group.group_id for group in manifest.groups}
    if group_ids != set(range(len(manifest.groups))):
        raise ValueError(f"{label} KV group IDs must be contiguous positional indices")
    if isinstance(block_ids, (str, bytes)) or len(block_ids) != len(manifest.groups):
        raise ValueError(
            f"{label} block IDs must contain exactly {len(manifest.groups)} groups"
        )
    normalized: list[tuple[int, ...]] = []
    for group_id, group in enumerate(block_ids):
        if isinstance(group, (str, bytes)):
            raise ValueError(f"{label} block group {group_id} must be a sequence")
        values = tuple(group)
        for block_id in values:
            _require_nonnegative_int(block_id, f"{label} block_id")
        if len(set(values)) != len(values):
            raise ValueError(f"{label} block group {group_id} contains duplicates")
        normalized.append(values)
    return tuple(normalized)


def _prefix_counts(
    manifest: KVFormatManifest,
    values: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != len(manifest.groups):
        raise ValueError(
            "destination prefix counts must contain exactly one value per KV group"
        )
    normalized = tuple(values)
    for value in normalized:
        _require_nonnegative_int(value, "destination prefix block count")
    return normalized


def _expand_physical_page_ids(
    block_ids: tuple[tuple[int, ...], ...],
    physical_pages_per_logical: int,
) -> tuple[tuple[int, ...], ...]:
    """Expand scheduler block IDs into consecutive kernel-page IDs."""
    return tuple(
        tuple(
            block_id * physical_pages_per_logical + physical_index
            for block_id in group
            for physical_index in range(physical_pages_per_logical)
        )
        for group in block_ids
    )


def select_nixl_destination_prefix_blocks(
    values: Sequence[int],
    *,
    transfer_group_ids: Sequence[int],
    total_group_count: int,
) -> tuple[int, ...]:
    """Project scheduler-wide prefix counts onto transferred KV groups.

    NIXL exchanges block-ID groups after ``KVCacheConfig`` has removed groups
    with ``enable_kv_transfer=False``. Scheduler prefix-hit counts predate that
    projection and are normally indexed by every cache group. Empty values are
    used by the full-prefix-hit/abort path; an already-projected tuple is also
    accepted for compatibility with direct worker tests and future schedulers.
    """
    _require_nonnegative_int(total_group_count, "total_group_count")
    if isinstance(values, (str, bytes)) or isinstance(transfer_group_ids, (str, bytes)):
        raise ValueError("prefix counts and transfer group IDs must be sequences")
    group_ids = tuple(transfer_group_ids)
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("transfer group IDs must be unique")
    for group_id in group_ids:
        _require_nonnegative_int(group_id, "transfer group ID")
        if group_id >= total_group_count:
            raise ValueError("transfer group ID is outside the cache group list")

    normalized = tuple(values)
    for value in normalized:
        _require_nonnegative_int(value, "destination prefix block count")
    if not normalized:
        return tuple(0 for _ in group_ids)
    if len(normalized) == total_group_count:
        return tuple(normalized[group_id] for group_id in group_ids)
    if len(normalized) == len(group_ids):
        return normalized
    raise ValueError(
        "destination prefix counts must be empty, scheduler-wide, or "
        "transfer-group-only"
    )


def _make_allocations(
    workers: Sequence[NixlPlacementMetadata],
    manifest: KVFormatManifest,
    block_ids: tuple[tuple[int, ...], ...],
    first_page_indices: tuple[int, ...],
) -> tuple[LocalPageAllocation, ...]:
    allocations: list[LocalPageAllocation] = []
    for worker in workers:
        placement = worker.rank_placement
        for layer_mapping in placement.mappings:
            group = manifest.semantic_group(layer_mapping.semantic_group_id)
            page_span = group.canonical_page_token_span
            first_page_index = first_page_indices[group.group_id]
            for local_index, page_id in enumerate(block_ids[group.group_id]):
                page_index = first_page_index + local_index
                allocations.append(
                    LocalPageAllocation(
                        worker_id=placement.worker_id,
                        worker_incarnation=placement.worker_incarnation,
                        layer_name=layer_mapping.layer_name,
                        local_page_id=page_id,
                        canonical_page_index=page_index,
                        first_token=page_index * page_span,
                    )
                )
    return tuple(allocations)


def _request_ranges(
    source_format: KVFormatManifest,
    destination_format: KVFormatManifest,
    destination_block_ids: tuple[tuple[int, ...], ...],
    destination_prefix_blocks: tuple[int, ...],
    remote_num_tokens: int,
    destination_physical_pages_per_logical: int,
) -> tuple[KVRange, ...]:
    source_semantic_ids = {group.semantic_id for group in source_format.groups}
    destination_semantic_ids = {
        group.semantic_id for group in destination_format.groups
    }
    if source_semantic_ids != destination_semantic_ids:
        raise ValueError(
            "source and destination advertise different semantic KV groups"
        )

    ranges: list[KVRange] = []
    for group in destination_format.groups:
        first_token = (
            destination_prefix_blocks[group.group_id]
            * destination_physical_pages_per_logical
            * group.canonical_page_token_span
        )
        token_count = (
            len(destination_block_ids[group.group_id])
            * destination_physical_pages_per_logical
            * group.canonical_page_token_span
        )
        # NIXL's scheduler protocol deliberately uses an empty destination
        # block vector for both a full prefix hit and an aborted request that
        # only needs to release the producer lease. Preserve that notify-only
        # meaning even when the abort path has no prefix-count metadata.
        if token_count > 0 and remote_num_tokens <= first_token:
            raise ValueError(
                f"non-empty destination KV group {group.semantic_id!r} starts at "
                f"token {first_token}, but the remote request contains only "
                f"{remote_num_tokens} tokens"
            )
        valid_token_count = 0 if token_count == 0 else remote_num_tokens - first_token
        ranges.append(
            KVRange(
                semantic_group_id=group.semantic_id,
                first_token=first_token,
                token_count=token_count,
                valid_token_count=valid_token_count,
            )
        )
    return tuple(ranges)


def _physical_page_id(
    block_ids: tuple[int, ...],
    physical_pages_per_logical: int,
    relative_page_index: int,
) -> int:
    logical_index, physical_index = divmod(
        relative_page_index, physical_pages_per_logical
    )
    return block_ids[logical_index] * physical_pages_per_logical + physical_index


def _make_layer_window_allocations(
    workers: Sequence[NixlPlacementMetadata],
    manifest: KVFormatManifest,
    layer: RequestLayerSpec,
    block_ids: tuple[tuple[int, ...], ...],
    first_page_index: int,
    physical_pages_per_logical: int,
    first_token: int,
    end_token: int,
) -> tuple[LocalPageAllocation, ...]:
    """Materialize only pages intersecting one layer's canonical window."""
    group = manifest.semantic_group(layer.semantic_group_id)
    page_span = group.canonical_page_token_span
    available_page_count = len(block_ids[group.group_id]) * physical_pages_per_logical
    available_end_page = first_page_index + available_page_count
    window_first_page = max(first_page_index, first_token // page_span)
    window_end_page = min(
        available_end_page,
        (end_token + page_span - 1) // page_span,
    )

    allocations: list[LocalPageAllocation] = []
    for worker in workers:
        placement = worker.rank_placement
        if not any(
            mapping.layer_name == layer.layer_name for mapping in placement.mappings
        ):
            continue
        for page_index in range(window_first_page, window_end_page):
            relative_page_index = page_index - first_page_index
            allocations.append(
                LocalPageAllocation(
                    worker_id=placement.worker_id,
                    worker_incarnation=placement.worker_incarnation,
                    layer_name=layer.layer_name,
                    local_page_id=_physical_page_id(
                        block_ids[group.group_id],
                        physical_pages_per_logical,
                        relative_page_index,
                    ),
                    canonical_page_index=page_index,
                    first_token=page_index * page_span,
                )
            )
    return tuple(allocations)


def _iter_layer_token_windows(
    request: NixlReadRequestPlan,
    layer: RequestLayerSpec,
) -> Iterator[KVRange]:
    source_group = request.source_workers[0].format_manifest.semantic_group(
        layer.semantic_group_id
    )
    destination_group = request.destination_workers[0].format_manifest.semantic_group(
        layer.semantic_group_id
    )
    source_span = source_group.canonical_page_token_span
    destination_span = destination_group.canonical_page_token_span
    for kv_range in request.ranges:
        if kv_range.semantic_group_id != layer.semantic_group_id:
            continue
        cursor = kv_range.first_token
        while cursor < kv_range.valid_end_token:
            source_first_page = cursor // source_span
            destination_first_page = cursor // destination_span
            end_token = min(
                kv_range.valid_end_token,
                (source_first_page + request.max_pages_per_window) * source_span,
                (destination_first_page + request.max_pages_per_window)
                * destination_span,
            )
            if end_token <= cursor:
                raise AssertionError("canonical page window did not advance")
            yield KVRange(
                semantic_group_id=layer.semantic_group_id,
                first_token=cursor,
                token_count=end_token - cursor,
                valid_token_count=end_token - cursor,
            )
            cursor = end_token


def _layer_window_allocations(
    request: NixlReadRequestPlan,
    layer: RequestLayerSpec,
    kv_range: KVRange,
) -> tuple[tuple[LocalPageAllocation, ...], tuple[LocalPageAllocation, ...]]:
    source_format = request.source_workers[0].format_manifest
    destination_format = request.destination_workers[0].format_manifest
    destination_group = destination_format.semantic_group(layer.semantic_group_id)
    destination_first_page_index = (
        request.destination_prefix_blocks[destination_group.group_id]
        * request.destination_physical_pages_per_logical
    )
    source_allocations = _make_layer_window_allocations(
        request.source_workers,
        source_format,
        layer,
        request.source_block_ids,
        0,
        request.source_physical_pages_per_logical,
        kv_range.first_token,
        kv_range.valid_end_token,
    )
    destination_allocations = _make_layer_window_allocations(
        request.destination_workers,
        destination_format,
        layer,
        request.destination_block_ids,
        destination_first_page_index,
        request.destination_physical_pages_per_logical,
        kv_range.first_token,
        kv_range.valid_end_token,
    )
    return source_allocations, destination_allocations


def iter_nixl_read_plan_windows(
    request: NixlReadRequestPlan,
) -> Iterator[NixlReadPlanWindow]:
    """Plan one layer and bounded canonical-page window per iteration."""
    if not isinstance(request, NixlReadRequestPlan):
        raise ValueError("request must be a NixlReadRequestPlan")
    for layer in request.planning_context.layers:
        for kv_range in _iter_layer_token_windows(request, layer):
            source_allocations, destination_allocations = _layer_window_allocations(
                request, layer, kv_range
            )
            yield NixlReadPlanWindow(
                layer_plan=plan_request_transfer_layer(
                    request.planning_context,
                    layer,
                    source_pages=source_allocations,
                    destination_pages=destination_allocations,
                    ranges=(kv_range,),
                ),
                source_allocations=source_allocations,
                destination_allocations=destination_allocations,
                kv_range=kv_range,
            )


def _max_segments_hint(workers: Sequence[NixlPlacementMetadata]) -> int | None:
    limits = [MAX_NIXL_DIRECT_SEGMENTS_PER_BATCH]
    for worker in workers:
        capabilities = worker.capabilities
        if not capabilities.scatter_gather:
            limits.append(1)
        elif capabilities.max_segments_per_batch is not None:
            limits.append(capabilities.max_segments_per_batch)
    return min(limits) if limits else None


def _validate_registered_block_ids(
    label: str,
    workers: Sequence[NixlPlacementMetadata],
    manifest: KVFormatManifest,
    block_ids: tuple[tuple[int, ...], ...],
    physical_pages_per_logical: int,
) -> None:
    """Validate compact scheduler IDs against every addressed layer region."""
    capacities_by_group: dict[str, list[int]] = {
        group.semantic_id: [] for group in manifest.groups
    }
    for worker in workers:
        templates = {
            template.layer_name: template
            for template in worker.page_registration_templates
        }
        for mapping in worker.rank_placement.mappings:
            capacities_by_group[mapping.semantic_group_id].append(
                templates[mapping.layer_name].num_pages
            )

    for group in manifest.groups:
        capacities = capacities_by_group[group.semantic_id]
        if not capacities:
            raise ValueError(
                f"{label} KV group {group.semantic_id!r} has no registered pages"
            )
        physical_count = len(block_ids[group.group_id]) * physical_pages_per_logical
        for capacity in capacities:
            if physical_count > capacity:
                raise ValueError(
                    f"{label} KV group {group.semantic_id!r} requests "
                    f"{physical_count} physical pages from a {capacity}-page region"
                )
        minimum_capacity = min(capacities)
        for block_id in block_ids[group.group_id]:
            first_physical_id = block_id * physical_pages_per_logical
            final_physical_id = first_physical_id + physical_pages_per_logical - 1
            if final_physical_id >= minimum_capacity:
                raise ValueError(
                    f"{label} block {block_id} expands to physical page "
                    f"{final_physical_id}, outside a {minimum_capacity}-page region"
                )


def _validate_semantic_page_capacity(
    label: str,
    manifest: KVFormatManifest,
    block_ids: tuple[tuple[int, ...], ...],
    first_page_indices: tuple[int, ...],
    physical_pages_per_logical: int,
    ranges: Sequence[KVRange],
    *,
    include_allocated_tail: bool,
) -> None:
    """Prove request token coverage from compact page counts before planning."""
    ranges_by_group: dict[str, list[KVRange]] = {
        group.semantic_id: [] for group in manifest.groups
    }
    for kv_range in ranges:
        ranges_by_group[kv_range.semantic_group_id].append(kv_range)

    for group in manifest.groups:
        group_ranges = ranges_by_group[group.semantic_id]
        if not group_ranges:
            continue
        page_span = group.canonical_page_token_span
        available_first = first_page_indices[group.group_id] * page_span
        available_end = available_first + (
            len(block_ids[group.group_id]) * physical_pages_per_logical * page_span
        )
        required_ranges = [
            (
                kv_range.first_token,
                kv_range.end_token
                if include_allocated_tail
                else kv_range.valid_end_token,
            )
            for kv_range in group_ranges
            if include_allocated_tail or kv_range.valid_token_count > 0
        ]
        if not required_ranges:
            continue
        required_first = min(start for start, _ in required_ranges)
        required_end = max(end for _, end in required_ranges)
        if required_first < available_first or required_end > available_end:
            raise ValueError(
                f"{label} KV group {group.semantic_id!r} does not cover requested "
                f"canonical tokens [{required_first}, {required_end}); available "
                f"coverage is [{available_first}, {available_end})"
            )


def build_nixl_read_request_plan(
    *,
    source_workers: Sequence[NixlPlacementMetadata],
    destination_workers: Sequence[NixlPlacementMetadata],
    source_block_ids: BlockIdGroups,
    destination_block_ids: BlockIdGroups,
    destination_prefix_blocks: Sequence[int],
    remote_num_tokens: int,
    source_physical_pages_per_logical: int = 1,
    destination_physical_pages_per_logical: int = 1,
    max_pages_per_window: int = DEFAULT_NIXL_CANONICAL_PAGE_WINDOW,
    source_dp_rank: int | None = None,
    destination_dp_rank: int | None = None,
) -> NixlReadRequestPlan:
    """Validate one NIXL pull request and retain compact planning inputs.

    ``max_segments_per_batch`` in the result is only the smallest advertised
    submission-size hint. Fragmentation and page-window boundaries only create
    additional direct transfers; neither can select packing or staging.
    """
    _require_nonnegative_int(remote_num_tokens, "remote_num_tokens")
    _require_positive_int(
        source_physical_pages_per_logical,
        "source_physical_pages_per_logical",
    )
    _require_positive_int(
        destination_physical_pages_per_logical,
        "destination_physical_pages_per_logical",
    )
    _require_positive_int(max_pages_per_window, "max_pages_per_window")
    source_format, source_route, source_workers = _select_endpoint(
        "source", source_workers, source_dp_rank
    )
    destination_format, destination_route, destination_workers = _select_endpoint(
        "destination", destination_workers, destination_dp_rank
    )
    source_block_ids = _validate_positional_groups(
        "source", source_format, source_block_ids
    )
    destination_block_ids = _validate_positional_groups(
        "destination", destination_format, destination_block_ids
    )
    if remote_num_tokens == 0 and any(source_block_ids) and any(destination_block_ids):
        raise ValueError(
            "remote_num_tokens is required for a non-empty generic NIXL read"
        )
    destination_prefix_blocks = _prefix_counts(
        destination_format, destination_prefix_blocks
    )
    _validate_registered_block_ids(
        "source",
        source_workers,
        source_format,
        source_block_ids,
        source_physical_pages_per_logical,
    )
    _validate_registered_block_ids(
        "destination",
        destination_workers,
        destination_format,
        destination_block_ids,
        destination_physical_pages_per_logical,
    )
    ranges = _request_ranges(
        source_format,
        destination_format,
        destination_block_ids,
        destination_prefix_blocks,
        remote_num_tokens,
        destination_physical_pages_per_logical,
    )
    _validate_semantic_page_capacity(
        "source",
        source_format,
        source_block_ids,
        tuple(0 for _ in source_format.groups),
        source_physical_pages_per_logical,
        ranges,
        include_allocated_tail=False,
    )
    _validate_semantic_page_capacity(
        "destination",
        destination_format,
        destination_block_ids,
        tuple(
            destination_prefix_blocks[group_id] * destination_physical_pages_per_logical
            for group_id in range(len(destination_format.groups))
        ),
        destination_physical_pages_per_logical,
        ranges,
        include_allocated_tail=True,
    )
    planning_context = build_request_transfer_context(
        source_format=source_format,
        destination_format=destination_format,
        source_route=source_route,
        destination_route=destination_route,
        source_workers=tuple(worker.rank_placement for worker in source_workers),
        destination_workers=tuple(
            worker.rank_placement for worker in destination_workers
        ),
        ranges=ranges,
    )
    return NixlReadRequestPlan(
        planning_context=planning_context,
        source_workers=source_workers,
        destination_workers=destination_workers,
        source_block_ids=source_block_ids,
        destination_block_ids=destination_block_ids,
        destination_prefix_blocks=destination_prefix_blocks,
        source_physical_pages_per_logical=source_physical_pages_per_logical,
        destination_physical_pages_per_logical=(destination_physical_pages_per_logical),
        max_segments_per_batch=_max_segments_hint(
            (*source_workers, *destination_workers)
        ),
        max_pages_per_window=max_pages_per_window,
    )


def materialize_nixl_read_request_plan(
    request: NixlReadRequestPlan,
) -> tuple[
    RequestTransferPlan,
    tuple[LocalPageAllocation, ...],
    tuple[LocalPageAllocation, ...],
]:
    """Eagerly materialize a compact plan for diagnostics and unit tests.

    The NIXL runtime deliberately does not call this helper.
    """
    if not isinstance(request, NixlReadRequestPlan):
        raise ValueError("request must be a NixlReadRequestPlan")
    source_physical_page_ids = _expand_physical_page_ids(
        request.source_block_ids,
        request.source_physical_pages_per_logical,
    )
    destination_physical_page_ids = _expand_physical_page_ids(
        request.destination_block_ids,
        request.destination_physical_pages_per_logical,
    )
    destination_prefix_pages = tuple(
        count * request.destination_physical_pages_per_logical
        for count in request.destination_prefix_blocks
    )
    source_allocations = _make_allocations(
        request.source_workers,
        request.source_workers[0].format_manifest,
        source_physical_page_ids,
        tuple(0 for _ in request.source_block_ids),
    )
    destination_allocations = _make_allocations(
        request.destination_workers,
        request.destination_workers[0].format_manifest,
        destination_physical_page_ids,
        destination_prefix_pages,
    )
    context = request.planning_context
    transfer_plan = build_request_transfer_plan(
        source_format=context.source_format,
        destination_format=context.destination_format,
        source_route=context.source_route,
        destination_route=context.destination_route,
        source_workers=context.source_workers,
        destination_workers=context.destination_workers,
        source_pages=source_allocations,
        destination_pages=destination_allocations,
        ranges=request.ranges,
    )
    return transfer_plan, source_allocations, destination_allocations


def prepare_nixl_read_request(
    request: NixlReadRequestPlan,
    remote: NixlRemotePlacementIndex,
    *,
    nixl_wrapper: object,
    tracker: NixlEphemeralDlistTracker,
    local_transfer_rank: int,
    memory_type: str,
    max_segments_per_batch: int | None = None,
) -> NixlPreparedReadRequest:
    """Prepare every direct READ batch before the caller submits any of them.

    Preparation consumes the lowering iterator one bounded batch at a time.
    If a later batch cannot be prepared, all earlier descriptor lists are
    released while no transfer is in flight.  Successful handles are returned
    unsubmitted so the worker can publish the complete sibling set before
    starting it and aggregate terminal state at request scope.
    """
    prepared: list[NixlPreparedDirectBatch] = []
    try:
        for batch in iter_prepare_nixl_read_request(
            request,
            remote,
            nixl_wrapper=nixl_wrapper,
            tracker=tracker,
            local_transfer_rank=local_transfer_rank,
            memory_type=memory_type,
            max_segments_per_batch=max_segments_per_batch,
        ):
            prepared.append(batch)
    except Exception:
        for batch in prepared:
            tracker.release(batch.transfer_handle)
        raise
    return NixlPreparedReadRequest(request=request, batches=tuple(prepared))


def _validate_remote_index_request_binding(
    request: NixlReadRequestPlan,
    remote: NixlRemotePlacementIndex,
) -> None:
    """Bind lowering to the exact source endpoint indexed during handshake."""
    if remote.engine_id != request.source_route.deployment_id:
        raise ValueError("remote placement index has the wrong source deployment")
    if remote.physical_pages_per_logical != request.source_physical_pages_per_logical:
        raise ValueError("remote placement index has stale allocator page geometry")

    expected_by_id = {
        worker.rank_placement.worker_id: worker for worker in request.source_workers
    }
    indexed_by_id = {
        worker.rank_placement.worker_id: worker for worker in remote.workers
    }
    if (
        len(expected_by_id) != len(request.source_workers)
        or len(indexed_by_id) != len(remote.workers)
        or expected_by_id.keys() != indexed_by_id.keys()
    ):
        raise ValueError("remote placement index does not match request source workers")

    mismatched_workers = sorted(
        worker_id
        for worker_id, expected in expected_by_id.items()
        if indexed_by_id[worker_id] != expected
    )
    if mismatched_workers:
        raise ValueError(
            "remote placement index has stale source identity, format, topology, "
            f"capability, or registration metadata for workers {mismatched_workers}"
        )

    expected_ranks = {worker.rank_placement.rank for worker in request.source_workers}
    agent_names = remote.agent_names_by_rank
    if (
        len(agent_names) != len(remote.agent_names)
        or set(agent_names) != expected_ranks
        or any(not isinstance(name, str) or not name for name in agent_names.values())
        or len(set(agent_names.values())) != len(agent_names)
    ):
        raise ValueError(
            "remote placement index agent bindings do not match request source ranks"
        )


def iter_prepare_nixl_read_request(
    request: NixlReadRequestPlan,
    remote: NixlRemotePlacementIndex,
    *,
    nixl_wrapper: object,
    tracker: NixlEphemeralDlistTracker,
    local_transfer_rank: int,
    memory_type: str,
    max_segments_per_batch: int | None = None,
) -> Iterator[NixlPreparedDirectBatch]:
    """Lazily prepare local direct READ batches for bounded execution.

    Each iteration owns at most one newly prepared transfer. The caller must
    either submit and retain its handle or release it before requesting the
    next batch. Fragmentation only produces more iterations; it never changes
    the direct-copy policy.
    """
    if not isinstance(request, NixlReadRequestPlan):
        raise ValueError("request must be a NixlReadRequestPlan")
    if not isinstance(remote, NixlRemotePlacementIndex):
        raise ValueError("remote must be a NixlRemotePlacementIndex")
    _validate_remote_index_request_binding(request, remote)

    destination_ranks = {
        worker.rank_placement.rank for worker in request.destination_workers
    }
    if local_transfer_rank not in destination_ranks:
        raise ValueError(
            f"local transfer rank {local_transfer_rank} is not a request "
            "destination participant"
        )

    batch_limit = (
        request.max_segments_per_batch
        if max_segments_per_batch is None
        else max_segments_per_batch
    )
    remote_agents = remote.agent_names_by_rank

    for layer in request.planning_context.layers:
        for kv_range in _iter_layer_token_windows(request, layer):
            source_allocations, destination_allocations = _layer_window_allocations(
                request, layer, kv_range
            )
            for layer_batch in iter_nixl_request_layer_direct_batches(
                request.planning_context,
                layer,
                source_workers=request.source_workers,
                destination_workers=request.destination_workers,
                source_allocations=source_allocations,
                destination_allocations=destination_allocations,
                ranges=(kv_range,),
                operation="READ",
                max_segments_per_batch=batch_limit,
                destination_rank=local_transfer_rank,
            ):
                descriptor_batch = layer_batch.batch
                remote_rank, _, _ = descriptor_batch.transfer_sides(
                    "READ", local_transfer_rank
                )
                try:
                    remote_agent_name = remote_agents[remote_rank]
                except KeyError as error:
                    raise ValueError(
                        f"missing NIXL remote agent for transfer rank {remote_rank}"
                    ) from error
                yield prepare_nixl_direct_batch(
                    nixl_wrapper,
                    tracker,
                    descriptor_batch,
                    operation="READ",
                    local_rank=local_transfer_rank,
                    remote_agent_name=remote_agent_name,
                    memory_type=memory_type,
                )


__all__ = [
    "DEFAULT_NIXL_CANONICAL_PAGE_WINDOW",
    "MAX_NIXL_COMPLETION_BYTES",
    "NIXL_DIRECT_COMPLETION_PREFIX",
    "NixlDirectCompletionEnvelope",
    "NixlReadPlanWindow",
    "NixlReadRequestPlan",
    "NixlPreparedReadRequest",
    "NixlRemotePlacementIndex",
    "build_nixl_read_request_plan",
    "index_remote_nixl_placements",
    "iter_nixl_read_plan_windows",
    "iter_prepare_nixl_read_request",
    "materialize_nixl_read_request_plan",
    "nixl_read_request_plan_digest",
    "prepare_nixl_read_request",
    "select_nixl_destination_prefix_blocks",
    "validate_complete_nixl_placement_endpoint",
]
