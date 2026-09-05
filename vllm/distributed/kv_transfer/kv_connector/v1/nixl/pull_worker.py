# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import time
from collections import deque
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge import (
    NIXL_DIRECT_COMPLETION_PREFIX,
    NixlDirectCompletionEnvelope,
    build_nixl_read_request_plan,
    iter_prepare_nixl_read_request,
    nixl_read_request_plan_digest,
    select_nixl_destination_prefix_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.segmented import (
    NixlPreparedDirectBatch,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.distributed.kv_transfer.transfer_completion import (
    KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
    CompletionStatus,
    TransferCompletionNotification,
    TransferCompletionTracker,
    WorkerIdentity,
    participant_set_digest,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Slack (seconds) subtracted from D's exported block-expiry deadline on the turn-2
# readback, absorbing clock-offset error and read latency.
_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0


@dataclass
class _DirectReadBatchWindow:
    """Lazy batches belonging to one generic direct-read request."""

    batches: Iterator[NixlPreparedDirectBatch]
    remote_engine_id: str | None = None
    exhausted: bool = False
    failed: bool = False

    def close(self) -> None:
        close = getattr(self.batches, "close", None)
        if close is not None:
            close()


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._direct_read_notifications: dict[str, tuple[bytes, tuple[str, ...]]] = {}
        self._expected_direct_transfer_ids: dict[str, str] = {}
        self._expected_direct_participant_counts: dict[str, int] = {}
        self._expected_direct_participants: dict[str, tuple[WorkerIdentity, ...]] = {}
        self._direct_completion_trackers: dict[str, TransferCompletionTracker] = {}
        self._direct_completion_participant_digests: dict[str, str] = {}
        self._direct_completion_sender_bindings: dict[
            str, dict[str, WorkerIdentity]
        ] = {}
        max_inflight = self.kv_transfer_config.get_from_extra_config(
            "max_inflight_batches", 8
        )
        if (
            not isinstance(max_inflight, int)
            or isinstance(max_inflight, bool)
            or max_inflight <= 0
        ):
            raise ValueError("max_inflight_batches must be a positive integer")
        self._max_inflight_direct_batches = max_inflight
        self._direct_read_batch_windows: dict[str, _DirectReadBatchWindow] = {}
        self._direct_read_refill_queue: deque[str] = deque()

    def shutdown(self) -> None:
        """Close lazy direct-read iterators before releasing NIXL resources."""
        windows = getattr(self, "_direct_read_batch_windows", {})
        for state in tuple(windows.values()):
            state.close()
        windows.clear()
        refill_queue = getattr(self, "_direct_read_refill_queue", None)
        if refill_queue is not None:
            refill_queue.clear()
        super().shutdown()

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # Always store metadata for failure recovery. Serialize replacement
            # with background handshake callbacks so a late retry cannot start
            # or fail a newer attempt carrying the same request ID.
            with self._handshake_lock:
                if self._recving_metadata.get(req_id) is not None:
                    # The engine contract does not permit concurrent attempts
                    # with one request ID, and all receive and terminal state is
                    # keyed by it. Preserve even a handshake-pending attempt:
                    # its callback may already have queued a request-ID-only
                    # terminal result that must not apply to newer metadata.
                    logger.error(
                        "Ignoring overlapping NIXL receive attempt for request "
                        "ID %s until the original attempt is terminal",
                        req_id,
                    )
                    continue
                self._recving_metadata[req_id] = meta
                # Every request revalidates the complete endpoint specification;
                # engine_id alone is not a reusable registration identity.
                self._background_nixl_handshake(req_id, remote_engine_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            req_id, ready_meta = self._ready_requests.get_nowait()
            if self._recving_metadata.get(req_id) is ready_meta:
                self._read_blocks_for_req(req_id, ready_meta)

        # All requests made ready by this scheduler step are now registered.
        # Fill the worker-global credit pool only after that registration pass,
        # so the first request in the batch cannot consume every credit before
        # its peers become visible.
        self._refill_direct_read_batch_windows()

        if self.pcp_rank > 0 and not self._enable_generic_placement:
            return

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            self._clear_direct_completion_state(req_id)
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        # Deadlines are stamped with the scheduler process's perf_counter,
        # which is not comparable to ours when the worker runs in another
        # process on another node (perf_counter epochs differ by boot time).
        # Rebase the remaining TTL onto our clock; broadcast latency only
        # lengthens the lease, which is the safe direction. A cross-node
        # epoch gap larger than the TTL otherwise expires the lease on
        # arrival and the blocks are freed before D reads them.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                transfer_id = metadata.reqs_to_send_transfer_ids.get(req_id)
                expected_participant_count = (
                    metadata.reqs_to_send_expected_participant_counts.get(req_id)
                )
                expected_participants = metadata.reqs_to_send_expected_participants.get(
                    req_id, ()
                )
                has_completion_contract = (
                    isinstance(expected_participant_count, int)
                    and not isinstance(expected_participant_count, bool)
                    and expected_participant_count > 0
                    and len(expected_participants) == expected_participant_count
                )
                if (
                    transfer_id
                    and self._local_placement_metadata is not None
                    and has_completion_contract
                ):
                    # New scheduler metadata defines a new immutable attempt.
                    # Reset any incomplete aggregate retained for a reused ID.
                    assert expected_participant_count is not None
                    self._expected_direct_transfer_ids[req_id] = transfer_id
                    self._expected_direct_participant_counts[req_id] = (
                        expected_participant_count
                    )
                    participant_contracts = getattr(
                        self, "_expected_direct_participants", None
                    )
                    if participant_contracts is None:
                        participant_contracts = self._expected_direct_participants = {}
                    participant_contracts[req_id] = tuple(expected_participants)
                    self._direct_completion_trackers.pop(req_id, None)
                    self._direct_completion_participant_digests.pop(req_id, None)
                    self._direct_completion_sender_bindings.pop(req_id, None)
                else:
                    # A transfer attempt is generic only when the producer has
                    # an explicit quorum contract. Absent that contract, retain
                    # legacy notification behavior; a generic peer cannot
                    # release these pages and they expire through the lease.
                    self._clear_direct_completion_state(req_id)
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)

    def _is_turn2_read_expired(self, meta: ReqMeta) -> bool:
        """Whether D's cached blocks for this turn-2 readback have (nearly) expired."""
        assert meta.remote is not None
        blocks_expiry_time = meta.remote.blocks_expiry_time
        # Deadline may be absent (router may not forward it) -> read as usual.
        if blocks_expiry_time is None or not meta.local_physical_block_ids:
            return False
        clock_offset = self._engine_clock_offset[meta.remote.engine_id]
        deadline = blocks_expiry_time - clock_offset
        return time.perf_counter() + _KV_BLOCKS_EXPIRY_SAFETY_MARGIN >= deadline

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()

        if self._bidirectional_kv_xfer_enabled and self._is_turn2_read_expired(meta):
            logger.warning(
                "Declining expired remote read for %s from engine %s.",
                req_id,
                engine_id,
            )
            self.xfer_stats.record_kv_expired_req()
            self._handle_failed_transfer(req_id, None)
            return

        expected_participant_count = meta.remote.expected_completion_participant_count
        expected_participants = meta.remote.expected_completion_participants
        if (
            expected_participant_count is None
            or len(expected_participants) != expected_participant_count
        ) and engine_id in self._generic_only_remote_engines:
            error = ValueError(
                "generic-only NIXL endpoint requires an exact producer-owned "
                "completion roster"
            )
            self._log_failure(
                failure_type="segmented_direct_contract_missing",
                msg="Marking blocks as invalid",
                req_id=req_id,
                error=error,
            )
            self._handle_failed_transfer(req_id, None)
            return
        if expected_participant_count is not None:
            generic_placement_available = (
                self._local_placement_metadata is not None
                and engine_id in self._remote_placement_indexes
                and bool(self._local_placement_workers)
            )
            local_participant_count = len(self._local_placement_workers)
            local_participants = tuple(
                WorkerIdentity(
                    worker.rank_placement.worker_id,
                    worker.rank_placement.worker_incarnation,
                )
                for worker in self._local_placement_workers
            )
            if (
                not isinstance(expected_participant_count, int)
                or isinstance(expected_participant_count, bool)
                or expected_participant_count <= 0
                or not generic_placement_available
                or expected_participant_count != local_participant_count
                or not expected_participants
                or participant_set_digest(expected_participants)
                != participant_set_digest(local_participants)
            ):
                error = ValueError(
                    "producer completion quorum does not match the generic "
                    "destination placement"
                )
                self._log_failure(
                    failure_type="segmented_direct_contract_mismatch",
                    msg="Marking blocks as invalid",
                    req_id=req_id,
                    error=error,
                    producer_expected_participants=expected_participant_count,
                    local_destination_participants=local_participant_count,
                )
                self._handle_failed_transfer(req_id, None)
                return
            self._read_blocks_for_req_direct(req_id, meta)
            return

        if not getattr(self, "_legacy_fast_path_available", True):
            legacy_error = RuntimeError(
                "legacy NIXL descriptor preparation failed and this request "
                "does not carry a generic segmented-transfer completion contract"
            )
            self._log_failure(
                failure_type="legacy_descriptor_unavailable",
                msg="Marking blocks as invalid",
                req_id=req_id,
                error=legacy_error,
            )
            self._handle_failed_transfer(req_id, None)
            return

        if any(len(group) > 0 for group in meta.local_block_ids):
            # The scheduler waits for finished_recving from *every* worker.
            # Under DCP a rank's slice can legitimately come out empty when its
            # interleaved positions fall past the end of the sequence. _read_blocks
            # then takes the notify-only path without registering a transfer.
            # Seed the entry so this rank still reports completion.
            self._recving_transfers.setdefault(req_id, [])

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        remote_logical_block_ids = meta.remote.block_ids
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            remote_logical_block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        num_groups = len(meta.local_block_ids)
        dcp_active = self.dcp_size > 1 or remote_info.remote_dcp_size > 1
        if dcp_active and self.block_size != remote_info.remote_block_size:
            raise ValueError(
                "DCP KV transfer requires equal local and remote kernel block "
                f"sizes, got {self.block_size} and {remote_info.remote_block_size}."
            )
        if (
            dcp_active
            and self._physical_blocks_per_logical_kv_block
            != remote_info.remote_physical_blocks_per_logical
            and any(
                count > 0 and _is_attention_spec(self._group_spec_types[g])
                for g, count in enumerate(meta.local_num_computed_blocks)
            )
        ):
            raise ValueError(
                "DCP KV transfer with heterogeneous logical-page geometry does "
                "not yet support decoder-side prefix-cache hits. Disable prefix "
                "caching on the KV consumer."
            )

        def group_ids(block_ids: BlockIds, rank: int) -> list[list[int]]:
            return [
                list(block_ids[g]) if rank in plan.source_ranks_per_group[g] else []
                for g in range(num_groups)
            ]

        read_specs = []
        dcp_attention_blocks_by_group = [list[int]() for _ in range(num_groups)]
        for rank in plan.all_source_ranks:
            if dcp_active:
                # DCP ownership is resolved at kernel-block granularity. HMA
                # can make one logical page span a different number of local
                # and remote blocks even when both pages cover the same tokens.
                local_ids = group_ids(meta.local_block_ids, rank)
                remote_ids = group_ids(remote_logical_block_ids, rank)
                local_physical_ids = [
                    list(group)
                    for group in self._logical_to_kernel_block_ids(
                        local_ids, self._physical_blocks_per_logical_kv_block
                    )
                ]
                remote_physical_ids = [
                    list(group)
                    for group in self._logical_to_kernel_block_ids(
                        remote_ids, remote_info.remote_physical_blocks_per_logical
                    )
                ]
                for g in range(num_groups):
                    if not _is_attention_spec(self._group_spec_types[g]):
                        continue
                    if not local_ids[g]:
                        local_physical_ids[g] = []
                        remote_physical_ids[g] = []
                        continue
                    local_physical_ids[g], remote_physical_ids[g] = (
                        self._map_dcp_attention_block_ids(
                            local_ids[g],
                            remote_ids[g],
                            remote_rank=rank,
                            local_dcp_size=self.dcp_size,
                            local_dcp_rank=self.dcp_rank,
                            remote_dcp_size=remote_info.remote_dcp_size,
                            local_num_computed_blocks=meta.local_num_computed_blocks[g],
                            local_physical_per_logical=(
                                self._physical_blocks_per_logical_kv_block
                            ),
                            remote_physical_per_logical=(
                                remote_info.remote_physical_blocks_per_logical
                            ),
                        )
                    )
                    dcp_attention_blocks_by_group[g].extend(local_physical_ids[g])
            else:
                # No DCP realignment needed: reuse the already-expanded full
                # physical lists instead of re-deriving them from logical ids.
                local_physical_ids = group_ids(meta.local_physical_block_ids, rank)
                remote_physical_ids = group_ids(meta.remote.block_ids, rank)
            read_specs.append(
                ReadSpec(
                    remote_rank=rank,
                    local_block_ids=local_physical_ids,
                    remote_block_ids=remote_physical_ids,
                )
            )

        if dcp_active:
            covered_counts = [0] * num_groups
            for g, covered_blocks in enumerate(dcp_attention_blocks_by_group):
                if not _is_attention_spec(self._group_spec_types[g]):
                    continue
                unique_covered = set(covered_blocks)
                assert len(unique_covered) == len(covered_blocks), (
                    f"DCP source reads overlap for KV cache group {g}."
                )
                local_group = list(meta.local_physical_block_ids[g])
                num_covered = len(covered_blocks)
                assert unique_covered == set(local_group[:num_covered]), (
                    f"DCP source reads do not cover a contiguous destination "
                    f"prefix for KV cache group {g}."
                )
                covered_counts[g] = num_covered
            meta.dcp_local_attention_blocks_covered = tuple(covered_counts)

        # D may have to perform multiple reads from different remote ranks.
        # Pure MLA reads once because its cache is replicated. Hybrid
        # MLA+SSM still needs one read per SSM source rank. With DCP, pure
        # MLA may also read from multiple ranks (disjoint token slices).
        if self.use_mla and tp_ratio < 0 and not self._has_mamba and not dcp_active:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if self._needs_split_local_xfer_handles(tp_ratio, plan):
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                split_key = self._split_local_xfer_handle_key(
                    tp_ratio, remote_block_size, plan
                )
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                expected_consumers=plan.local_consumers,
            )

        if self.use_mla and tp_ratio < 0 and len(read_specs) == 1:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            # Same thing for DCP (tp_size == dcp_size), so the raw tp_ratio already
            # reflects whether any remote replica is left unchosen.
            notif_id = f"{meta.remote.request_id}:{plan.local_consumers}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks_for_req_direct(self, req_id: str, meta: ReqMeta) -> None:
        """Prepare and submit one generic segmented-direct READ request."""
        self._recving_transfers.setdefault(req_id, [])

        try:
            if meta.remote is None:
                raise ValueError("generic NIXL reads require remote metadata")
            local = self._local_placement_metadata
            if local is None:
                raise RuntimeError("generic NIXL local placement is unavailable")
            remote = self._remote_placement_indexes[meta.remote.engine_id]
            num_groups = len(local.format_manifest.groups)
            destination_workers = self._local_placement_workers
            if not destination_workers:
                raise RuntimeError(
                    "generic NIXL placement is missing the local endpoint manifest"
                )
            local_block_ids = meta.local_block_ids or tuple(
                [] for _ in range(num_groups)
            )
            prefix_blocks = select_nixl_destination_prefix_blocks(
                meta.local_num_computed_blocks,
                transfer_group_ids=self.kv_cache_config.transfer_group_ids,
                total_group_count=len(self.kv_cache_config.kv_cache_groups),
            )
            if not meta.remote.transfer_id:
                raise ValueError(
                    "generic NIXL reads require a remote transfer attempt ID"
                )
            request = build_nixl_read_request_plan(
                source_workers=remote.workers,
                destination_workers=destination_workers,
                source_block_ids=meta.remote.block_ids,
                destination_block_ids=local_block_ids,
                destination_prefix_blocks=prefix_blocks,
                remote_num_tokens=meta.remote_num_tokens,
                source_physical_pages_per_logical=(remote.physical_pages_per_logical),
                destination_physical_pages_per_logical=(
                    self._physical_blocks_per_logical_kv_block
                ),
            )
            batches = iter_prepare_nixl_read_request(
                request,
                remote,
                nixl_wrapper=self.nixl_wrapper,
                tracker=self._ephemeral_direct_dlists,
                local_transfer_rank=local.rank_placement.rank,
                memory_type=self.nixl_memory_type,
            )
            route = request.destination_route
            notification = TransferCompletionNotification(
                version=KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
                request_id=meta.remote.request_id,
                deployment_id=route.deployment_id,
                topology_generation=route.topology_generation,
                transfer_id=meta.remote.transfer_id,
                plan_digest=nixl_read_request_plan_digest(request),
                sender_worker_id=local.rank_placement.worker_id,
                sender_worker_incarnation=local.rank_placement.worker_incarnation,
                expected_participant_count=(
                    request.destination_expected_participant_count
                ),
                status=CompletionStatus.COMPLETE,
            )
            envelope = NixlDirectCompletionEnvelope(
                notification=notification,
                expected_participants=request.destination_participants,
            )
            notification_payload = envelope.encode()
            notification_agents = tuple(
                agent_name for _, agent_name in remote.agent_names
            )
            windows = getattr(self, "_direct_read_batch_windows", None)
            if windows is None:
                windows = self._direct_read_batch_windows = {}
            if req_id in windows:
                raise RuntimeError(
                    f"generic NIXL request {req_id!r} already has an active "
                    "batch window"
                )
            windows[req_id] = _DirectReadBatchWindow(
                batches=batches,
                remote_engine_id=meta.remote.engine_id,
            )
            self._direct_read_refill_order().append(req_id)
        except Exception as error:
            self._log_failure(
                failure_type="segmented_direct_setup_failed",
                msg="Marking blocks as invalid",
                req_id=req_id,
                error=error,
            )
            self._latch_failed_transfer(req_id, None, stream="recv")
            return

        self._generic_direct_receive_requests.add(req_id)
        self._direct_read_notifications[req_id] = (
            notification_payload,
            notification_agents,
        )

    def _direct_read_refill_order(self) -> deque[str]:
        """Return the round-robin order, including focused-test workers."""
        windows = getattr(self, "_direct_read_batch_windows", {})
        order = getattr(self, "_direct_read_refill_queue", None)
        if order is None:
            order = self._direct_read_refill_queue = deque(windows)
        return order

    def _submit_next_direct_read_batch(self, req_id: str) -> bool:
        """Prepare and submit at most one batch for ``req_id``."""
        windows = getattr(self, "_direct_read_batch_windows", {})
        state = windows.get(req_id)
        if state is None or state.exhausted or state.failed:
            return False
        active_handles = self._recving_transfers.setdefault(req_id, [])
        try:
            batch = next(state.batches)
        except StopIteration:
            state.exhausted = True
            state.close()
            return False
        except Exception as error:
            state.failed = True
            state.close()
            self._log_failure(
                failure_type="segmented_direct_setup_failed",
                msg="Marking blocks as invalid",
                req_id=req_id,
                error=error,
            )
            self._latch_failed_transfer(req_id, None, stream="recv")
            return False

        try:
            self.nixl_wrapper.transfer(batch.transfer_handle)
        except Exception as error:
            # Preparation transferred ownership to the ephemeral tracker, but
            # this handle was never published as active.
            self._release_xfer_handle(batch.transfer_handle)
            state.failed = True
            state.close()
            self._log_failure(
                failure_type="segmented_direct_submission_failed",
                msg="Marking blocks as invalid",
                req_id=req_id,
                error=error,
            )
            self._latch_failed_transfer(req_id, None, stream="recv")
            return False
        active_handles.append(batch.transfer_handle)
        return True

    def _touch_direct_read_remote_engines(self) -> None:
        """Keep remote registrations live while direct requests are active."""
        last_active = getattr(self, "_engine_last_active", None)
        if last_active is None:
            return
        now = time.perf_counter()
        for state in getattr(self, "_direct_read_batch_windows", {}).values():
            if state.remote_engine_id is not None:
                last_active[state.remote_engine_id] = now

    def _refill_direct_read_batch_windows(self) -> None:
        """Fairly fill the worker-global generic direct-read credit pool."""
        windows = getattr(self, "_direct_read_batch_windows", {})
        if not windows:
            return
        self._touch_direct_read_remote_engines()
        order = self._direct_read_refill_order()
        max_inflight = getattr(self, "_max_inflight_direct_batches", 8)
        inflight = sum(
            len(self._recving_transfers.get(req_id, ())) for req_id in windows
        )
        credits = max(0, max_inflight - inflight)

        while credits > 0 and order:
            # One batch per request per pass. Keeping successful requests at
            # the tail also carries fairness across later polling/refill calls.
            requests_this_pass = len(order)
            made_progress = False
            for _ in range(requests_this_pass):
                if credits == 0:
                    break
                req_id = order.popleft()
                state = windows.get(req_id)
                if state is None or state.exhausted or state.failed:
                    continue
                submitted = self._submit_next_direct_read_batch(req_id)
                if not state.exhausted and not state.failed:
                    order.append(req_id)
                if submitted:
                    credits -= 1
                    made_progress = True
            if not made_progress:
                break

    def _discard_direct_read_batch_window(self, req_id: str) -> None:
        windows = getattr(self, "_direct_read_batch_windows", {})
        state = windows.pop(req_id, None)
        if state is not None:
            state.close()
        order = getattr(self, "_direct_read_refill_queue", None)
        if order is not None:
            with suppress(ValueError):
                order.remove(req_id)

    def _pop_done_transfers(
        self,
        transfers: dict[str, list[int]],
        *,
        stream: str = "recv",
        failed_req_ids: set[str] | None = None,
    ) -> set[str]:
        """Poll active handles, then refill generic READ windows lazily."""
        observed_failed = failed_req_ids if failed_req_ids is not None else set()
        windows = (
            getattr(self, "_direct_read_batch_windows", {}) if stream == "recv" else {}
        )
        if windows:
            self._touch_direct_read_remote_engines()
        # A generic request may legitimately have no active handle while it is
        # waiting for a worker-global credit. Do not let the base poller treat
        # that transiently empty handle list as request completion.
        waiting_for_credit = {
            req_id
            for req_id, handles in tuple(transfers.items())
            if req_id in windows
            and not handles
            and not windows[req_id].exhausted
            and not windows[req_id].failed
        }
        for req_id in waiting_for_credit:
            transfers.pop(req_id, None)
        terminal = super()._pop_done_transfers(
            transfers,
            stream=stream,
            failed_req_ids=observed_failed,
        )
        for req_id in waiting_for_credit:
            if req_id in windows:
                transfers.setdefault(req_id, [])
        if failed_req_ids is None and stream == "recv":
            # Preserve the base worker's legacy queue-reporting behavior when a
            # direct caller omits ``failed_req_ids``.
            for req_id in observed_failed:
                self._failed_recv_reqs.put(req_id)
        if stream != "recv":
            return terminal

        poller = getattr(self, "_request_terminal_poller", None)
        for req_id, state in tuple(windows.items()):
            if req_id in observed_failed:
                self._discard_direct_read_batch_window(req_id)
                continue
            if poller is not None and poller.has_failed("recv", req_id):
                # One transfer failed while siblings remain in flight. Stop
                # producing more work and let the request barrier drain them.
                state.failed = True
                state.close()
                order = getattr(self, "_direct_read_refill_queue", None)
                if order is not None:
                    with suppress(ValueError):
                        order.remove(req_id)
                continue
            if state.failed:
                continue

            if state.exhausted:
                if req_id in terminal:
                    self._discard_direct_read_batch_window(req_id)
                continue

            # The base poller removes a fully drained request entry. Recreate
            # it while the request waits for a global credit.
            transfers.setdefault(req_id, [])
            terminal.discard(req_id)

        self._refill_direct_read_batch_windows()

        # Reconcile states changed by refill. A request without a credit is
        # still pending; an exhausted request with no active handles is done.
        for req_id, state in tuple(windows.items()):
            active_handles = transfers.get(req_id, [])
            if state.failed or active_handles:
                terminal.discard(req_id)
                continue
            if state.exhausted:
                transfers.pop(req_id, None)
                self._discard_direct_read_batch_window(req_id)
                terminal.add(req_id)
            else:
                transfers.setdefault(req_id, [])
                terminal.discard(req_id)

        return terminal

    def _on_receive_requests_terminal(
        self, successful: set[str], failed: set[str]
    ) -> None:
        for req_id in successful | failed:
            self._discard_direct_read_batch_window(req_id)

        # Generic direct READ failures must be reported with the exact attempt
        # identity and plan digest prepared before transfer submission.  If
        # setup failed before that envelope existed there is deliberately
        # nothing to send: fabricating an incomplete completion key could bind
        # the failure to the wrong retry.  Producers treat FAILED as terminal
        # evidence for the attempt but retain their pages until lease expiry.
        for req_id in successful | failed:
            notification = self._direct_read_notifications.pop(req_id, None)
            if notification is None:
                continue
            payload, agents = notification
            if req_id in failed:
                envelope = NixlDirectCompletionEnvelope.decode(payload)
                payload = replace(
                    envelope,
                    notification=replace(
                        envelope.notification,
                        status=CompletionStatus.FAILED,
                    ),
                ).encode()
            for agent in agents:
                try:
                    self.nixl_wrapper.send_notif(agent, notif_msg=payload)
                except Exception as error:
                    self._log_failure(
                        failure_type="notification_failed",
                        msg="P worker blocks will be freed after timeout.",
                        req_id=req_id,
                        error=error,
                        remote_agent_name=agent,
                    )
                    self.xfer_stats.record_failed_notification()

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        expected_consumers: int,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of local workers that will notify this producer worker.
        # Propagate on notification so dst worker can wait before freeing.
        notif_id = f"{remote_request_id}:{expected_consumers}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if not any(len(group) > 0 for group in local_block_ids):
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][(0, remote_rank)]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            return

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.transfer_groups)
        )
        # DCP attention groups were already trimmed and realigned at physical
        # kernel-block granularity in _read_blocks_for_req, making this a no-op.
        # Hybrid Mamba groups are deliberately not DCP-sliced and still need
        # their replicated state/placeholder lists aligned here.
        remote_physical_per_logical = remote_info.remote_physical_blocks_per_logical
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            decode_block_ids=local_block_ids,
            prefill_block_ids=remote_block_ids,
            decode_physical_per_logical=self._physical_blocks_per_logical_kv_block,
            prefill_physical_per_logical=remote_physical_per_logical,
        )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # Another read batch for this request may already be in flight.
            # Latch the failure and let request-level polling drain every
            # sibling before reporting completion/removing metadata.
            self._latch_failed_transfer(request_id, handle, stream="recv")

    def _clear_direct_completion_state(self, req_id: str) -> None:
        """Forget producer-side state for one generic transfer attempt."""
        self._expected_direct_transfer_ids.pop(req_id, None)
        self._expected_direct_participant_counts.pop(req_id, None)
        getattr(self, "_expected_direct_participants", {}).pop(req_id, None)
        self._direct_completion_trackers.pop(req_id, None)
        self._direct_completion_participant_digests.pop(req_id, None)
        self._direct_completion_sender_bindings.pop(req_id, None)

    def _on_send_request_terminal(self, req_id: str) -> None:
        """Clear generic completion state when a source lease terminates."""
        self._clear_direct_completion_state(req_id)

    def _handle_direct_completion(
        self,
        payload: bytes,
        notified_req_ids: set[str],
        *,
        sender_agent: str,
    ) -> None:
        """Aggregate one exact, retry-safe segmented-direct completion."""
        try:
            envelope = NixlDirectCompletionEnvelope.decode(payload)
            notification = envelope.notification
            req_id = notification.request_id

            # Another completion from the same poll is an expected delayed
            # duplicate after the request reached its exact barrier.
            if req_id in notified_req_ids:
                return
            if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                logger.error(
                    "Ignoring generic KV completion for unrecognized request %s.",
                    req_id,
                )
                return

            expected_transfer_id = self._expected_direct_transfer_ids.get(req_id)
            if expected_transfer_id is None:
                logger.error(
                    "Ignoring generic KV completion for request %s without an "
                    "active generic transfer attempt.",
                    req_id,
                )
                return
            if notification.transfer_id != expected_transfer_id:
                logger.warning(
                    "Ignoring stale generic KV completion for request %s "
                    "(got attempt %s, expected %s).",
                    req_id,
                    notification.transfer_id,
                    expected_transfer_id,
                )
                return

            expected_participant_count = self._expected_direct_participant_counts.get(
                req_id
            )
            expected_participants = getattr(
                self, "_expected_direct_participants", {}
            ).get(req_id)
            if expected_participant_count is None:
                logger.error(
                    "Ignoring generic KV completion for request %s without a "
                    "producer-owned participant count; retaining pages until "
                    "lease expiry.",
                    req_id,
                )
                return
            if expected_participants is None:
                logger.error(
                    "Ignoring generic KV completion for request %s without a "
                    "producer-owned participant roster; retaining pages until "
                    "lease expiry.",
                    req_id,
                )
                return
            if (
                notification.expected_participant_count != expected_participant_count
                or len(envelope.expected_participants) != expected_participant_count
            ):
                raise ValueError(
                    "completion participant count does not match the "
                    "producer-owned quorum"
                )
            expected_participants_digest = participant_set_digest(expected_participants)
            participants_digest = participant_set_digest(envelope.expected_participants)
            if participants_digest != expected_participants_digest:
                raise ValueError(
                    "completion participant set does not match the "
                    "producer-owned roster"
                )
            if not isinstance(sender_agent, str) or not sender_agent:
                raise ValueError("completion transport sender must be non-empty")

            sender = WorkerIdentity(
                worker_id=notification.sender_worker_id,
                worker_incarnation=notification.sender_worker_incarnation,
            )
            if sender_agent != sender.worker_incarnation:
                raise ValueError(
                    "completion sender incarnation does not match the NIXL "
                    "transport sender"
                )
            participants_by_id = {
                participant.worker_id: participant
                for participant in envelope.expected_participants
            }
            if participants_by_id.get(sender.worker_id) != sender:
                raise ValueError(
                    "completion transport sender is not an exact participant"
                )
            sender_bindings = self._direct_completion_sender_bindings.get(req_id)
            previous_sender = (
                sender_bindings.get(sender_agent)
                if sender_bindings is not None
                else None
            )
            if previous_sender is not None and previous_sender != sender:
                raise ValueError(
                    "one NIXL transport sender claimed multiple completion workers"
                )

            tracker = self._direct_completion_trackers.get(req_id)
            if tracker is None:
                tracker = TransferCompletionTracker(
                    request_id=req_id,
                    deployment_id=notification.deployment_id,
                    topology_generation=notification.topology_generation,
                    transfer_id=notification.transfer_id,
                    plan_digest=notification.plan_digest,
                    expected_participants=expected_participants,
                )
                self._direct_completion_trackers[req_id] = tracker
                self._direct_completion_participant_digests[req_id] = (
                    participants_digest
                )
            elif (
                self._direct_completion_participant_digests.get(req_id)
                != participants_digest
            ):
                raise ValueError(
                    "completion notifications disagree on the participant set"
                )

            progress = tracker.record(notification)
            self._direct_completion_sender_bindings.setdefault(req_id, {})[
                sender_agent
            ] = sender
            if progress.failed:
                logger.error(
                    "Generic KV transfer attempt for request %s reported failure; "
                    "retaining producer pages until lease expiry.",
                    req_id,
                )
                return
            if not progress.complete:
                return

            notified_req_ids.add(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            self.expected_consumer_notifications_by_req.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self._reqs_to_send.pop(req_id, None)
            self._clear_direct_completion_state(req_id)
        except Exception as error:
            # Malformed, stale, and conflicting completions must fail closed:
            # never release source pages before the normal lease timeout.
            logger.error(
                "Ignoring invalid generic KV completion notification: %s",
                error,
            )

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP or DCP
        scenario), wait for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        for sender_agent, notifs in self.nixl_wrapper.get_new_notifs().items():
            for notif in notifs:
                payload = bytes(notif)
                if payload.startswith(NIXL_DIRECT_COMPLETION_PREFIX):
                    self._handle_direct_completion(
                        payload,
                        notified_req_ids,
                        sender_agent=sender_agent,
                    )
                    continue

                msg = payload.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, expected_consumers = msg.rsplit(":", 1)
                if req_id in self._expected_direct_transfer_ids:
                    logger.warning(
                        "Ignoring legacy count-based completion for active generic "
                        "transfer request %s.",
                        req_id,
                    )
                    continue
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                # Every reader of this req_id reports the same count (it's
                # derived from aggregate topology, not the specific rank),
                # so repeated notifications never disagree on it.
                self.expected_consumer_notifications_by_req[req_id] = int(
                    expected_consumers
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == self.expected_consumer_notifications_by_req[req_id]
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    del self.expected_consumer_notifications_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
                    self._clear_direct_completion_state(req_id)
        return notified_req_ids
