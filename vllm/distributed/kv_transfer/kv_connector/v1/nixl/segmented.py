# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt canonical KV placement plans to direct NIXL descriptors.

The shared placement planner emits affine :class:`TransferRun` objects.  NIXL
accepts paired lists of contiguous ``(address, length, device_id)``
descriptors, so each affine fragment becomes one descriptor pair here.

Descriptor count is deliberately not a validity constraint.  A connector may
split a large plan into multiple direct transfers with
``max_segments_per_batch``, but this module never selects staging, packing, or
rejection merely because a plan is fragmented.
"""

import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vllm.distributed.kv_transfer.kv_placement import TransferRun

NixlPageKey = tuple[int, int]
NixlDescriptor = tuple[int, int, int]

logger = logging.getLogger(__name__)


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class NixlPageRegistration:
    """Addressable extent for one rank-local KV page.

    ``base_address`` and ``length`` describe a slice of memory already
    registered with the NIXL agent.  The adapter only derives transfer
    descriptors; it does not own the registration or its lifetime.  Page IDs
    are scoped to one cache layer/region, so callers compose and lower each
    advertised layer mapping independently.
    """

    base_address: int
    length: int
    device_id: int

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.base_address, "base_address")
        _require_positive_int(self.length, "length")
        _require_nonnegative_int(self.device_id, "device_id")


@dataclass(frozen=True)
class NixlDirectDescriptorBatch:
    """One direct scatter/gather transfer between a source/destination pair.

    ``batch_index`` identifies peer-local emission order. ``batch_count`` is
    exact for the eager lowering API and ``None`` for a streaming input whose
    future length is intentionally not consumed to compute informational
    metadata. Callers must aggregate every batch across all request
    layers/cache groups before sending one request-completion notification.
    Attaching the same notification to each NIXL transfer would permit
    premature KV reuse.
    """

    source_rank: int
    destination_rank: int
    source_descriptors: tuple[NixlDescriptor, ...]
    destination_descriptors: tuple[NixlDescriptor, ...]
    batch_index: int = 0
    batch_count: int | None = 1

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.source_rank, "source_rank")
        _require_nonnegative_int(self.destination_rank, "destination_rank")
        _require_nonnegative_int(self.batch_index, "batch_index")
        if self.batch_count is not None:
            _require_positive_int(self.batch_count, "batch_count")
        if self.batch_count is not None and self.batch_index >= self.batch_count:
            raise ValueError("batch_index must be in [0, batch_count)")
        if not self.source_descriptors:
            raise ValueError("a descriptor batch must not be empty")
        if len(self.source_descriptors) != len(self.destination_descriptors):
            raise ValueError("source and destination descriptor counts must match")
        for side, descriptors in (
            ("source", self.source_descriptors),
            ("destination", self.destination_descriptors),
        ):
            for descriptor in descriptors:
                if len(descriptor) != 3:
                    raise ValueError(f"{side} NIXL descriptors must have three fields")
                address, length, device_id = descriptor
                _require_nonnegative_int(address, f"{side} descriptor address")
                _require_positive_int(length, f"{side} descriptor length")
                _require_nonnegative_int(device_id, f"{side} descriptor device_id")

        for source, destination in zip(
            self.source_descriptors, self.destination_descriptors
        ):
            if source[1] != destination[1]:
                raise ValueError("paired NIXL descriptors must have equal lengths")

    @property
    def segment_count(self) -> int:
        return len(self.source_descriptors)

    @property
    def total_bytes(self) -> int:
        return sum(descriptor[1] for descriptor in self.source_descriptors)

    @property
    def requires_aggregate_completion(self) -> bool:
        """Whether this batch belongs to a multi-transfer completion group."""
        return self.batch_count is None or self.batch_count > 1

    def transfer_sides(
        self, operation: str, local_rank: int
    ) -> tuple[int, tuple[NixlDescriptor, ...], tuple[NixlDescriptor, ...]]:
        """Orient descriptors as ``(remote_rank, local, remote)`` for NIXL.

        A pull connector issues ``READ`` from a remote source into its local
        destination.  A push connector issues ``WRITE`` from its local source
        into a remote destination.
        """
        _require_nonnegative_int(local_rank, "local_rank")
        if operation not in ("READ", "WRITE"):
            raise ValueError(
                "NIXL operation must use the canonical value 'READ' or 'WRITE', "
                f"got {operation!r}"
            )
        if operation == "READ":
            if local_rank != self.destination_rank:
                raise ValueError(
                    "READ must be submitted by the destination rank "
                    f"{self.destination_rank}, got {local_rank}"
                )
            return (
                self.source_rank,
                self.destination_descriptors,
                self.source_descriptors,
            )
        if operation == "WRITE":
            if local_rank != self.source_rank:
                raise ValueError(
                    "WRITE must be submitted by the source rank "
                    f"{self.source_rank}, got {local_rank}"
                )
            return (
                self.destination_rank,
                self.source_descriptors,
                self.destination_descriptors,
            )
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class NixlPreparedDirectBatch:
    """A direct descriptor batch prepared as a NIXL transfer.

    The associated local and remote descriptor-list handles are owned by the
    :class:`NixlEphemeralDlistTracker`; callers track only ``transfer_handle``
    in their normal request completion data structure.
    """

    descriptor_batch: NixlDirectDescriptorBatch
    transfer_handle: int


@dataclass(frozen=True)
class _EphemeralDlists:
    local_handle: Any
    remote_handle: Any


class NixlEphemeralDlistTracker:
    """Own request-scoped NIXL descriptor lists until their transfer ends.

    Existing NIXL KV paths use long-lived descriptor lists created during the
    handshake.  Segmented direct copies instead prepare exact descriptor lists
    for a request batch.  This tracker is deliberately opt-in: releasing an
    untracked transfer returns ``False``, allowing the worker to preserve its
    existing static-handle behavior.

    Entries are removed under a lock *before* calling into NIXL.  Consequently
    the tracker itself attempts to release each ephemeral resource at most
    once.  Cleanup errors are logged and suppressed so a second tracker call
    does not retry already-invalid handles.
    """

    def __init__(self, nixl_wrapper: Any):
        self._nixl_wrapper = nixl_wrapper
        self._by_transfer: dict[int, _EphemeralDlists] = {}
        self._lock = threading.Lock()
        self._closed = False

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._by_transfer)

    def uses_wrapper(self, nixl_wrapper: Any) -> bool:
        """Return whether resources will be released through this wrapper."""
        return self._nixl_wrapper is nixl_wrapper

    def track(
        self,
        transfer_handle: int,
        local_dlist_handle: Any,
        remote_dlist_handle: Any,
    ) -> None:
        """Take ownership of both descriptor lists for ``transfer_handle``."""
        with self._lock:
            if self._closed:
                raise RuntimeError("ephemeral NIXL descriptor tracker is closed")
            if transfer_handle in self._by_transfer:
                raise ValueError(
                    f"transfer handle {transfer_handle!r} is already tracked"
                )
            self._by_transfer[transfer_handle] = _EphemeralDlists(
                local_dlist_handle, remote_dlist_handle
            )

    def release(self, transfer_handle: int) -> bool:
        """Release a tracked transfer and both descriptor lists exactly once.

        Returns ``False`` when the transfer uses static descriptor lists or was
        already released.  In that case this tracker does not call NIXL.
        """
        with self._lock:
            resources = self._by_transfer.pop(transfer_handle, None)
        if resources is None:
            return False
        self._release_resources(transfer_handle, resources)
        return True

    def release_all(self) -> int:
        """Close the tracker and release every still-owned transfer resource."""
        with self._lock:
            self._closed = True
            resources = tuple(self._by_transfer.items())
            self._by_transfer.clear()
        for transfer_handle, dlists in resources:
            self._release_resources(transfer_handle, dlists)
        return len(resources)

    def _release_resources(
        self, transfer_handle: int, resources: _EphemeralDlists
    ) -> None:
        self._release_safely(
            self._nixl_wrapper.release_xfer_handle,
            transfer_handle,
            "transfer",
        )
        self._release_safely(
            self._nixl_wrapper.release_dlist_handle,
            resources.local_handle,
            "local descriptor list",
        )
        self._release_safely(
            self._nixl_wrapper.release_dlist_handle,
            resources.remote_handle,
            "remote descriptor list",
        )

    @staticmethod
    def _release_safely(release: Any, handle: Any, kind: str) -> None:
        try:
            release(handle)
        except Exception:
            logger.warning("Failed to release ephemeral NIXL %s", kind, exc_info=True)


def _release_untracked_dlist(nixl_wrapper: Any, handle: Any) -> None:
    if handle is None:
        return
    try:
        nixl_wrapper.release_dlist_handle(handle)
    except Exception:
        logger.warning(
            "Failed to clean up untracked NIXL descriptor list", exc_info=True
        )


def prepare_nixl_direct_batch(
    nixl_wrapper: Any,
    tracker: NixlEphemeralDlistTracker,
    batch: NixlDirectDescriptorBatch,
    *,
    operation: str,
    local_rank: int,
    remote_agent_name: str,
    memory_type: str,
) -> NixlPreparedDirectBatch:
    """Prepare one descriptor batch and transfer ownership to ``tracker``.

    This deliberately does not start the transfer and does not attach a NIXL
    notification.  The worker must start and track every prepared batch, then
    emit one completion notification only after all request batches finish.
    """
    if not tracker.uses_wrapper(nixl_wrapper):
        raise ValueError("ephemeral descriptor tracker belongs to another wrapper")
    _, local_descriptors, remote_descriptors = batch.transfer_sides(
        operation, local_rank
    )
    local_dlist_handle = None
    remote_dlist_handle = None
    transfer_handle = None
    try:
        local_xfer_descriptors = nixl_wrapper.get_xfer_descs(
            list(local_descriptors), memory_type
        )
        local_dlist_handle = nixl_wrapper.prep_xfer_dlist(
            "NIXL_INIT_AGENT", local_xfer_descriptors
        )
        if local_dlist_handle is None:
            raise RuntimeError("NIXL failed to prepare the local descriptor list")
        remote_xfer_descriptors = nixl_wrapper.get_xfer_descs(
            list(remote_descriptors), memory_type
        )
        remote_dlist_handle = nixl_wrapper.prep_xfer_dlist(
            remote_agent_name, remote_xfer_descriptors
        )
        if remote_dlist_handle is None:
            raise RuntimeError("NIXL failed to prepare the remote descriptor list")
        descriptor_ids = np.arange(batch.segment_count, dtype=np.int32)
        transfer_handle = nixl_wrapper.make_prepped_xfer(
            operation,
            local_dlist_handle,
            descriptor_ids,
            remote_dlist_handle,
            descriptor_ids,
        )
        if transfer_handle is None:
            raise RuntimeError("NIXL failed to prepare a segmented direct transfer")
        tracker.track(
            transfer_handle,
            local_dlist_handle,
            remote_dlist_handle,
        )
    except Exception:
        if transfer_handle is not None:
            try:
                nixl_wrapper.release_xfer_handle(transfer_handle)
            except Exception:
                logger.warning(
                    "Failed to clean up untracked NIXL transfer", exc_info=True
                )
        _release_untracked_dlist(nixl_wrapper, local_dlist_handle)
        _release_untracked_dlist(nixl_wrapper, remote_dlist_handle)
        raise

    return NixlPreparedDirectBatch(batch, transfer_handle)


def prepare_nixl_direct_batches(
    nixl_wrapper: Any,
    tracker: NixlEphemeralDlistTracker,
    batches: Sequence[NixlDirectDescriptorBatch],
    *,
    operation: str,
    local_rank: int,
    remote_agents: Mapping[int, str],
    memory_type: str,
) -> tuple[NixlPreparedDirectBatch, ...]:
    """Prepare a logical group before the caller starts any transfer.

    Preparing the whole group first gives setup atomicity: if a later batch or
    peer cannot be prepared, every earlier transfer and ephemeral descriptor
    list from this call is released.  Successful results still represent only
    direct NIXL copies, regardless of the number of batches.
    """
    prepared: list[NixlPreparedDirectBatch] = []
    try:
        for batch in batches:
            remote_rank, _, _ = batch.transfer_sides(operation, local_rank)
            try:
                remote_agent_name = remote_agents[remote_rank]
            except KeyError as error:
                raise ValueError(
                    f"missing NIXL remote agent for rank {remote_rank}"
                ) from error
            prepared.append(
                prepare_nixl_direct_batch(
                    nixl_wrapper,
                    tracker,
                    batch,
                    operation=operation,
                    local_rank=local_rank,
                    remote_agent_name=remote_agent_name,
                    memory_type=memory_type,
                )
            )
    except Exception:
        for item in prepared:
            tracker.release(item.transfer_handle)
        raise
    return tuple(prepared)


def _validate_run(run: TransferRun) -> None:
    for name in (
        "source_rank",
        "destination_rank",
        "source_page_id",
        "destination_page_id",
        "source_offset",
        "destination_offset",
        "source_stride",
        "destination_stride",
    ):
        _require_nonnegative_int(getattr(run, name), name)
    _require_positive_int(run.fragment_size, "fragment_size")
    _require_positive_int(run.fragment_count, "fragment_count")


def _lookup_page(
    pages: Mapping[NixlPageKey, NixlPageRegistration],
    rank: int,
    page_id: int,
    side: str,
) -> NixlPageRegistration:
    key = (rank, page_id)
    try:
        page = pages[key]
    except KeyError as error:
        raise ValueError(f"missing {side} page registration for {key}") from error
    if not isinstance(page, NixlPageRegistration):
        raise ValueError(
            f"{side} page registration for {key} must be NixlPageRegistration"
        )
    return page


def _validate_extent(
    *,
    offset: int,
    stride: int,
    fragment_size: int,
    fragment_count: int,
    page: NixlPageRegistration,
    side: str,
    rank: int,
    page_id: int,
) -> None:
    final_end = offset + (fragment_count - 1) * stride + fragment_size
    if final_end > page.length:
        raise ValueError(
            f"{side} transfer extent [{offset}, {final_end}) exceeds rank {rank} "
            f"page {page_id} length {page.length}"
        )


def _iter_run_descriptors(
    run: TransferRun,
    source_pages: Mapping[NixlPageKey, NixlPageRegistration],
    destination_pages: Mapping[NixlPageKey, NixlPageRegistration],
) -> Iterator[tuple[NixlDescriptor, NixlDescriptor]]:
    _validate_run(run)
    source_page = _lookup_page(
        source_pages, run.source_rank, run.source_page_id, "source"
    )
    destination_page = _lookup_page(
        destination_pages,
        run.destination_rank,
        run.destination_page_id,
        "destination",
    )
    _validate_extent(
        offset=run.source_offset,
        stride=run.source_stride,
        fragment_size=run.fragment_size,
        fragment_count=run.fragment_count,
        page=source_page,
        side="source",
        rank=run.source_rank,
        page_id=run.source_page_id,
    )
    _validate_extent(
        offset=run.destination_offset,
        stride=run.destination_stride,
        fragment_size=run.fragment_size,
        fragment_count=run.fragment_count,
        page=destination_page,
        side="destination",
        rank=run.destination_rank,
        page_id=run.destination_page_id,
    )

    for i in range(run.fragment_count):
        yield (
            (
                source_page.base_address + run.source_offset + i * run.source_stride,
                run.fragment_size,
                source_page.device_id,
            ),
            (
                destination_page.base_address
                + run.destination_offset
                + i * run.destination_stride,
                run.fragment_size,
                destination_page.device_id,
            ),
        )


def iter_nixl_direct_descriptor_batches(
    transfer_runs: Sequence[TransferRun],
    source_pages: Mapping[NixlPageKey, NixlPageRegistration],
    destination_pages: Mapping[NixlPageKey, NixlPageRegistration],
    *,
    max_segments_per_batch: int | None = None,
) -> Iterator[NixlDirectDescriptorBatch]:
    """Stream bounded direct descriptor batches grouped by peer pair.

    Only one peer-local batch of expanded descriptors is retained at a time
    when ``max_segments_per_batch`` is set. This is important for DCP
    interleave 1, where an affine placement can expand into many small RDMA
    segments. Fragmentation changes submission count, never transfer policy.
    """
    if max_segments_per_batch is not None:
        _require_positive_int(max_segments_per_batch, "max_segments_per_batch")

    by_endpoints: dict[tuple[int, int], list[TransferRun]] = defaultdict(list)
    for run in transfer_runs:
        if not isinstance(run, TransferRun):
            raise ValueError("transfer_runs must contain TransferRun values")
        _validate_run(run)
        by_endpoints[(run.source_rank, run.destination_rank)].append(run)

    for (source_rank, destination_rank), peer_runs in sorted(by_endpoints.items()):
        segment_count = sum(run.fragment_count for run in peer_runs)
        batch_size = max_segments_per_batch or segment_count
        batch_count = (segment_count + batch_size - 1) // batch_size
        source_descriptors: list[NixlDescriptor] = []
        destination_descriptors: list[NixlDescriptor] = []
        batch_index = 0
        for run in peer_runs:
            for source, destination in _iter_run_descriptors(
                run, source_pages, destination_pages
            ):
                source_descriptors.append(source)
                destination_descriptors.append(destination)
                if len(source_descriptors) != batch_size:
                    continue
                yield NixlDirectDescriptorBatch(
                    source_rank=source_rank,
                    destination_rank=destination_rank,
                    source_descriptors=tuple(source_descriptors),
                    destination_descriptors=tuple(destination_descriptors),
                    batch_index=batch_index,
                    batch_count=batch_count,
                )
                source_descriptors.clear()
                destination_descriptors.clear()
                batch_index += 1
        if source_descriptors:
            yield NixlDirectDescriptorBatch(
                source_rank=source_rank,
                destination_rank=destination_rank,
                source_descriptors=tuple(source_descriptors),
                destination_descriptors=tuple(destination_descriptors),
                batch_index=batch_index,
                batch_count=batch_count,
            )


def iter_nixl_direct_descriptor_batches_streaming(
    transfer_runs: Iterable[TransferRun],
    source_pages: Mapping[NixlPageKey, NixlPageRegistration],
    destination_pages: Mapping[NixlPageKey, NixlPageRegistration],
    *,
    max_segments_per_batch: int = 4096,
    max_buffered_segments: int | None = None,
) -> Iterator[NixlDirectDescriptorBatch]:
    """Lower a run stream with a global bound across all peer buffers.

    Unlike :func:`iter_nixl_direct_descriptor_batches`, this function never
    groups or counts the complete input first. At most
    ``max_buffered_segments`` descriptor pairs are retained across *all* peer
    pairs (defaulting to one batch). A full peer buffer is emitted directly;
    global pressure emits the largest buffered peer deterministically. No
    fragmentation level selects packing, staging, or rejection.

    Since discovering the final peer-local batch count would require
    consuming the stream, emitted batches use ``batch_count=None``. Runtime
    completion is request-aggregate based and does not depend on this
    informational count.
    """
    _require_positive_int(max_segments_per_batch, "max_segments_per_batch")
    if max_buffered_segments is None:
        max_buffered_segments = max_segments_per_batch
    _require_positive_int(max_buffered_segments, "max_buffered_segments")

    buffers: dict[
        tuple[int, int],
        tuple[list[NixlDescriptor], list[NixlDescriptor]],
    ] = {}
    batch_indices: dict[tuple[int, int], int] = defaultdict(int)
    buffered_segment_count = 0

    def pop_batch(peer: tuple[int, int]) -> NixlDirectDescriptorBatch:
        nonlocal buffered_segment_count
        source_descriptors, destination_descriptors = buffers.pop(peer)
        buffered_segment_count -= len(source_descriptors)
        batch_index = batch_indices[peer]
        batch_indices[peer] = batch_index + 1
        return NixlDirectDescriptorBatch(
            source_rank=peer[0],
            destination_rank=peer[1],
            source_descriptors=tuple(source_descriptors),
            destination_descriptors=tuple(destination_descriptors),
            batch_index=batch_index,
            batch_count=None,
        )

    for run in transfer_runs:
        if not isinstance(run, TransferRun):
            raise ValueError("transfer_runs must contain TransferRun values")
        peer = (run.source_rank, run.destination_rank)
        for source, destination in _iter_run_descriptors(
            run, source_pages, destination_pages
        ):
            if buffered_segment_count == max_buffered_segments:
                flush_peer = min(
                    buffers,
                    key=lambda candidate: (
                        -len(buffers[candidate][0]),
                        candidate,
                    ),
                )
                yield pop_batch(flush_peer)

            source_descriptors, destination_descriptors = buffers.setdefault(
                peer, ([], [])
            )
            source_descriptors.append(source)
            destination_descriptors.append(destination)
            buffered_segment_count += 1
            if len(source_descriptors) == max_segments_per_batch:
                yield pop_batch(peer)

    for peer in sorted(buffers):
        yield pop_batch(peer)


def build_nixl_direct_descriptor_batches(
    transfer_runs: Sequence[TransferRun],
    source_pages: Mapping[NixlPageKey, NixlPageRegistration],
    destination_pages: Mapping[NixlPageKey, NixlPageRegistration],
    *,
    max_segments_per_batch: int | None = None,
) -> tuple[NixlDirectDescriptorBatch, ...]:
    """Lower transfer runs into per-peer direct NIXL descriptor batches.

    ``max_segments_per_batch`` controls only the number of descriptor pairs in
    each returned batch.  All fragments remain direct copies, and every input
    fragment appears exactly once in the output even when the limit is small.
    """
    return tuple(
        iter_nixl_direct_descriptor_batches(
            transfer_runs,
            source_pages,
            destination_pages,
            max_segments_per_batch=max_segments_per_batch,
        )
    )


__all__ = [
    "NixlDescriptor",
    "NixlDirectDescriptorBatch",
    "NixlEphemeralDlistTracker",
    "NixlPageKey",
    "NixlPageRegistration",
    "NixlPreparedDirectBatch",
    "build_nixl_direct_descriptor_batches",
    "iter_nixl_direct_descriptor_batches",
    "iter_nixl_direct_descriptor_batches_streaming",
    "prepare_nixl_direct_batch",
    "prepare_nixl_direct_batches",
]
