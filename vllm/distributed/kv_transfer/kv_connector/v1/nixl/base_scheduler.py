# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base scheduler-side logic for the NIXL connector."""

import threading
import time
import uuid
from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import msgspec
import zmq

from vllm import envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    yield_req_data,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    MAX_NIXL_HANDSHAKE_BYTES,
    HeartbeatInfo,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    MultipartFrameLimitError,
    recv_multipart_bounded,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.transfer_completion import WorkerIdentity
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import make_zmq_path
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

_MAX_NIXL_HANDSHAKE_QUERY_BYTES = 1024
_INVALID_NIXL_HANDSHAKE_RESPONSE = msgspec.msgpack.encode(
    {"error": "invalid NIXL handshake query"}
)


class _NixlHandshakeMetadataStore:
    """Atomically replace the cohort served by the handshake listener."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._encoded_data: dict[tuple[int, int], bytes] = {}

    def replace(self, encoded_data: dict[tuple[int, int], bytes]) -> None:
        """Publish one immutable metadata cohort in a single lock section."""
        with self._lock:
            self._encoded_data = encoded_data

    def get(self, coordinate: tuple[int, int]) -> bytes | None:
        """Return one rank payload from the currently published cohort."""
        with self._lock:
            return self._encoded_data.get(coordinate)

    def snapshot(self) -> dict[tuple[int, int], bytes]:
        """Return a copy of the current cohort for diagnostics and tests."""
        with self._lock:
            return dict(self._encoded_data)


def _endpoint_cohort_incarnation(
    scheduler_incarnation: str,
    metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata],
) -> str:
    """Bind one endpoint generation to its scheduler and worker resources."""
    if not isinstance(scheduler_incarnation, str) or not scheduler_incarnation:
        raise ValueError("NIXL scheduler incarnation must be a non-empty string")
    for coordinate in metadata:
        if (
            not isinstance(coordinate, tuple)
            or len(coordinate) != 2
            or any(
                not isinstance(rank, int) or isinstance(rank, bool) or rank < 0
                for rank in coordinate
            )
        ):
            raise ValueError(
                "NIXL handshake coordinates must be non-negative rank pairs"
            )
    digest = sha256(b"vllm-nixl-endpoint-cohort-v1\0")

    def add(value: bytes) -> None:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)

    add(scheduler_incarnation.encode())
    encoder = msgspec.msgpack.Encoder()
    for coordinate in sorted(metadata):
        rank_metadata = metadata[coordinate]
        if not isinstance(rank_metadata, NixlHandshakePayload):
            raise ValueError(
                "NixlConnectorScheduler expects NixlHandshakePayload for "
                "handshake metadata."
            )
        add(encoder.encode(coordinate))
        # A worker-provided endpoint field is not authoritative. Bind the
        # cohort to its compatibility hashes and exact agent/placement bytes,
        # then stamp the scheduler-derived generation onto every served rank.
        add(encoder.encode(replace(rank_metadata, endpoint_incarnation="")))
    return f"sha256:{digest.hexdigest()}"


def _decode_nixl_handshake_query(payload: bytes) -> tuple[int, int]:
    """Decode ``(command, pp_rank, PP-local placement rank)``."""
    if not isinstance(payload, bytes):
        raise ValueError("query payload must be bytes")
    if not payload or len(payload) > _MAX_NIXL_HANDSHAKE_QUERY_BYTES:
        raise ValueError(
            "query payload must contain between 1 and "
            f"{_MAX_NIXL_HANDSHAKE_QUERY_BYTES} bytes"
        )
    try:
        command, pp_rank, pp_local_rank = msgspec.msgpack.decode(
            payload,
            type=tuple[bytes, int, int],
            strict=True,
        )
    except (msgspec.DecodeError, msgspec.ValidationError, RecursionError) as error:
        raise ValueError(f"invalid query MessagePack: {error}") from error
    if command != GET_META_MSG:
        raise ValueError("unexpected query command")
    if pp_rank < 0 or pp_local_rank < 0:
        raise ValueError("query ranks must be non-negative")
    return pp_rank, pp_local_rank


class NixlBaseConnectorScheduler:
    """Base implementation of Scheduler side methods shared by pull and push."""

    # Emitted in kv_transfer_params so an external router can distinguish a
    # pull (READ) producer from a push (WRITE) one. Overridden by the push
    # scheduler.
    _TRANSFER_MODE: str = "pull"

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.engine_id: EngineId = engine_id
        self._scheduler_incarnation = str(uuid.uuid4())
        self._endpoint_incarnation = self._scheduler_incarnation
        self._nixl_handshake_metadata_store = _NixlHandshakeMetadataStore()
        self._handshake_metadata_publish_lock = threading.Lock()
        self.kv_cache_config = kv_cache_config
        self.side_channel_host = envs.VLLM_NIXL_SIDE_CHANNEL_HOST
        self.side_channel_port = (
            envs.VLLM_NIXL_SIDE_CHANNEL_PORT
            + vllm_config.parallel_config.data_parallel_index
        )
        assert vllm_config.kv_transfer_config is not None
        self._kv_lease_duration: int = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_lease_duration", 30
            )
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._heartbeat_interval = self._kv_lease_duration // 6
        if current_platform.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = (
                vllm_config.kv_transfer_config.kv_buffer_device == "cpu"
            )
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            # Also handle unlikely SW-only model case instead of checking num_groups>1.
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.transfer_groups
            )
        )
        self._has_mamba = any(
            isinstance(g.kv_cache_spec, MambaSpec)
            for g in kv_cache_config.transfer_groups
        )

        logger.info("Initializing NIXL Scheduler %s", engine_id)
        if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
            logger.info("Hybrid Memory Allocator is enabled with NIXL")

        # Background thread for handling new handshake requests.
        self._nixl_handshake_listener_t: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[
            ReqId, tuple[Request, BlockIds, tuple[int, ...]]
        ] = {}
        self._reqs_need_save: dict[ReqId, Request] = {}
        # Reqs to send and their expiration time
        self._reqs_need_send: dict[ReqId, float] = {}
        # Fresh transfer-attempt identity paired with each pinned-block lease.
        self._reqs_need_send_transfer_ids: dict[ReqId, str] = {}
        # Expected generic-completion quorum supplied with the producer request.
        self._reqs_need_send_expected_participant_counts: dict[ReqId, int] = {}
        self._reqs_need_send_expected_participants: dict[
            ReqId, tuple[WorkerIdentity, ...]
        ] = {}
        self._reqs_in_batch: set[ReqId] = set()
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[ReqId] = set()

        # Heartbeat tracking: requests needing periodic lease-renewal heartbeats to
        # remote P-side, stored as ready-to-send HeartbeatInfo grouped by remote engine
        self._heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Reverse lookup: local req_id -> (engine_id, remote_req_id) for O(1) removal
        self._heartbeat_req_engine: dict[ReqId, tuple[EngineId, ReqId]] = {}
        self._last_heartbeat_time: float = 0.0

        # Gather Sliding Window sizes for each kv cache group (if any) in number of
        # blocks per KV cache group. This is used to clip the local attention window.
        sw_sizes_tokens: list[tuple[int, int]] = [
            (g.kv_cache_spec.sliding_window, g.kv_cache_spec.block_size)
            if isinstance(g.kv_cache_spec, SlidingWindowSpec)
            else (0, self.block_size)
            for g in kv_cache_config.transfer_groups
        ]
        # cdiv(n_tokens, block_size) gives blocks/window; add 1 to conservatively
        # account for boundary overlap eg window isn't fully aligned with blocks.
        self.blocks_per_sw = [
            cdiv(n_tokens, block_size) + 1 if n_tokens else 0
            for n_tokens, block_size in sw_sizes_tokens
        ]

        # Trailing scratch slots that mamba managers co-allocate per request
        # for speculative decoding; None for non-SSM groups.
        self._ssm_spec_blocks = [
            g.kv_cache_spec.num_speculative_blocks
            if isinstance(g.kv_cache_spec, MambaSpec)
            else None
            for g in kv_cache_config.transfer_groups
        ]
        # Only "all" mode keeps a state per block position; the other modes
        # keep a single running state in the last non-speculative slot.
        self._ssm_state_slots_are_positional = (
            vllm_config.cache_config.mamba_cache_mode == "all"
        )

        # Threshold to decide whether to compute kv cache locally
        # or pull from a remote node: minimum number of remote
        # tokens to amortize the xfer latencies
        self.kv_recompute_threshold: int = int(
            vllm_config.kv_transfer_config.get_from_extra_config(
                "kv_recompute_threshold", 64
            )
        )

        # Bi-directional KV transfer feature supports KV block
        # transfers from D node to P node
        self.is_bidirectional_kv_xfer_enabled = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )
        self.decoder_kv_blocks_ttl = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "decoder_kv_blocks_ttl", 480
            )
        )

        if self.is_bidirectional_kv_xfer_enabled and self.kv_recompute_threshold > 0:
            logger.info(
                "Bidirectional KV transfer is enabled and the kv "
                "recompute threshold is set to %d tokens."
                "KV blocks on D are released after a TTL of %d seconds.",
                self.kv_recompute_threshold,
                self.decoder_kv_blocks_ttl,
            )

    def shutdown(self):
        self._stop_event.set()
        if self._nixl_handshake_listener_t is not None:
            self._nixl_handshake_listener_t.join()
            self._nixl_handshake_listener_t = None

    def on_new_request(self, request: "Request") -> None:
        """Track a request that may need heartbeats."""
        params = request.kv_transfer_params
        if params is not None and params.get("do_remote_decode") and self._has_mamba:
            self._truncate_mamba_request_for_prefill(request)

        # NOTE (NickLucche) This excludes request meant for P, ie heartbeats are
        # effectively disabled for Bidirectional KV transfer.
        if params is None or not params.get("do_remote_prefill"):
            return
        # Only track if all required remote fields are present.
        remote_engine_id = params.get("remote_engine_id")
        remote_request_id = params.get("remote_request_id")
        host = params.get("remote_host")
        port = params.get("remote_port")
        tp_size = params.get("tp_size")
        dcp_size = params.get("dcp_size", 1)
        pcp_size = params.get("pcp_size", 1)
        pp_size = params.get("pp_size", 1)
        if (
            remote_engine_id is None
            or remote_request_id is None
            or host is None
            or port is None
            or tp_size is None
        ):
            return
        heartbeat = HeartbeatInfo(
            req_ids=set(),
            host=host,
            port=port,
            tp_size=tp_size,
            dcp_size=dcp_size,
            pcp_size=pcp_size,
            pp_size=pp_size,
            endpoint_incarnation=params.get("remote_endpoint_incarnation"),
        )
        existing = self._heartbeat_by_engine.get(remote_engine_id)
        if existing is not None and replace(existing, req_ids=set()) != heartbeat:
            # Never send a new incarnation's request IDs through stale endpoint
            # coordinates. Old producer leases safely expire without heartbeat.
            logger.warning(
                "Replacing stale heartbeat endpoint metadata for engine %s",
                remote_engine_id,
            )
            for local_req_id, (tracked_engine_id, _) in tuple(
                self._heartbeat_req_engine.items()
            ):
                if tracked_engine_id == remote_engine_id:
                    del self._heartbeat_req_engine[local_req_id]
            self._heartbeat_by_engine[remote_engine_id] = heartbeat
        elif existing is None:
            self._heartbeat_by_engine[remote_engine_id] = heartbeat
        self._heartbeat_by_engine[remote_engine_id].req_ids.add(remote_request_id)
        self._heartbeat_req_engine[request.request_id] = (
            remote_engine_id,
            remote_request_id,
        )

    def _stop_heartbeat(self, req_id: ReqId) -> None:
        """Remove *req_id* from heartbeat tracking (if tracked)."""
        if key := self._heartbeat_req_engine.pop(req_id, None):
            engine_id, remote_id = key
            if info := self._heartbeat_by_engine.get(engine_id):
                info.req_ids.discard(remote_id)
                if not info.req_ids:
                    # Clean up empty engines so we don't leak a key when remote dies.
                    del self._heartbeat_by_engine[engine_id]

    def get_exchange_clipped_blocks(
        self, block_ids: BlockIds, clip_ssm: bool = True
    ) -> BlockIds:
        """Clip a request's block lists down to the transferable blocks.

        Sliding-window groups keep only the in-window tail: the KV cache
        manager allocates blocks for the entire sequence length and cleans up
        out-of-window blocks only prior to the `request_finished_all_groups`
        hook.

        SSM groups keep only their state-bearing slots: the trailing
        speculative scratch slots always go, and in single-state cache modes
        so does everything before the running state (null placeholders and
        the previous step's superseded state). "all" mode keeps its remaining
        slots, which the worker pairs position-wise.

        Use this at every block-id exchange point. Pass ``clip_ssm=False``
        for per-step partial lists (host-buffer save), where the SSM strip
        does not apply.
        """
        if len(block_ids) == 0:
            # No blocks to clip, e.g. a full prefix cache hit.
            return block_ids
        block_ids = self.kv_cache_config.select_transfer_block_ids(block_ids)
        if not self._is_hma_required:
            return block_ids
        # NOTE (NickLucche) This logic is currently handled at the connector level
        # because offloading connectors might want to receive the whole sequence even
        # for SWA groups. We will abstract this logic once the interface is more stable
        assert len(block_ids) == len(self.blocks_per_sw), (
            "Number of KV cache groups must match"
        )
        clipped = []
        for i, blocks in enumerate(block_ids):
            if n_sw := self.blocks_per_sw[i]:
                blocks = blocks[-n_sw:]
            elif (
                clip_ssm
                and blocks
                and (n_spec_blocks := self._ssm_spec_blocks[i]) is not None
            ):
                if n_spec := min(n_spec_blocks, len(blocks) - 1):
                    blocks = blocks[:-n_spec]
                if not self._ssm_state_slots_are_positional:
                    # Never empty: downstream reads that as a full prefix hit.
                    blocks = blocks[-1:]
            clipped.append(blocks)
        return tuple(clipped)

    def set_xfer_handshake_metadata(
        self, metadata: dict[tuple[int, int], KVConnectorHandshakeMetadata]
    ) -> None:
        """Set connector handshake metadata for every PP-local placement.

        Args:
            metadata: Metadata keyed by ``(pp_rank, pcp_rank * tp_size +
                tp_rank)``.
        """
        if not metadata:
            raise ValueError("NixlConnectorScheduler handshake metadata is empty")
        scheduler_incarnation = getattr(
            self, "_scheduler_incarnation", self._endpoint_incarnation
        )
        endpoint_incarnation = _endpoint_cohort_incarnation(
            scheduler_incarnation,
            metadata,
        )
        encoded_data: dict[tuple[int, int], bytes] = {}
        encoder = msgspec.msgpack.Encoder()
        for (pp_rank, pp_local_rank), rank_metadata in metadata.items():
            if not isinstance(rank_metadata, NixlHandshakePayload):
                raise ValueError(
                    "NixlConnectorScheduler expects NixlHandshakePayload for "
                    "handshake metadata."
                )
            encoded_rank_metadata = encoder.encode(
                replace(
                    rank_metadata,
                    endpoint_incarnation=endpoint_incarnation,
                )
            )
            if len(encoded_rank_metadata) > MAX_NIXL_HANDSHAKE_BYTES:
                raise ValueError(
                    "encoded NixlHandshakePayload for "
                    f"PP rank {pp_rank}, PP-local placement rank "
                    f"{pp_local_rank} exceeds the "
                    f"{MAX_NIXL_HANDSHAKE_BYTES}-byte limit"
                )
            encoded_data[(pp_rank, pp_local_rank)] = encoded_rank_metadata
            logger.debug(
                "PP rank %d, PP-local placement rank %d: encoded "
                "NixlHandshakePayload size: %s bytes",
                pp_rank,
                pp_local_rank,
                str(len(encoded_data[(pp_rank, pp_local_rank)])),
            )

        publish_lock = getattr(self, "_handshake_metadata_publish_lock", None)
        if publish_lock is None:
            # Compatibility for focused tests constructing the scheduler via
            # object.__new__; production schedulers initialize this eagerly.
            publish_lock = self._handshake_metadata_publish_lock = threading.Lock()
        with publish_lock:
            store = getattr(self, "_nixl_handshake_metadata_store", None)
            if store is None:
                store = self._nixl_handshake_metadata_store = (
                    _NixlHandshakeMetadataStore()
                )
            # Publish the complete cohort before exposing its generation
            # through request metadata. A concurrent old request may observe
            # the new cohort and fail its token check, but no new request can
            # observe a new token paired with stale worker bytes. Serializing
            # the pair also prevents concurrent recovery callbacks from
            # leaving the store and token on different final generations.
            store.replace(encoded_data)
            self._endpoint_incarnation = endpoint_incarnation

            # Only start the listener when we have metadata to serve.
            if self._nixl_handshake_listener_t is None:
                ready_event = threading.Event()
                self._nixl_handshake_listener_t = threading.Thread(
                    target=self._nixl_handshake_listener,
                    args=(
                        store,
                        ready_event,
                        self._stop_event,
                        self.side_channel_host,
                        self.side_channel_port,
                    ),
                    daemon=True,
                    name="nixl_handshake_listener",
                )
                self._nixl_handshake_listener_t.start()
                ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    @staticmethod
    def _nixl_handshake_listener(
        metadata_store: _NixlHandshakeMetadataStore,
        ready_event: threading.Event,
        stop_event: threading.Event,
        host: str,
        port: int,
    ):
        """Background thread for getting new NIXL handshakes."""
        # NOTE(rob): this is a simple implementation. We will move
        # to a better approach via HTTP endpoint soon.

        # Listen for new requests for metadata.
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting listening on path: %s", path)
        while not stop_event.is_set():
            reset_socket = False
            with zmq_ctx(
                zmq.ROUTER,
                path,
                max_message_size=_MAX_NIXL_HANDSHAKE_QUERY_BYTES,
            ) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 1000)
                ready_event.set()
                while not stop_event.is_set():
                    try:
                        parts = recv_multipart_bounded(sock, 3)
                    except zmq.Again:
                        continue
                    except MultipartFrameLimitError as error:
                        # Unread tail frames cannot safely be resynchronized.
                        # Closing this socket discards them with bounded memory;
                        # the outer loop immediately recreates the listener.
                        logger.warning("Rejected NIXL handshake query: %s", error)
                        reset_socket = True
                        break
                    identity = parts[0] if parts else None
                    try:
                        if len(parts) != 3:
                            raise ValueError(
                                "query must contain identity, delimiter, and payload "
                                "frames"
                            )
                        identity, delimiter, payload = parts
                        if not identity:
                            raise ValueError("query identity must be non-empty bytes")
                        if delimiter != b"":
                            raise ValueError("query delimiter must be an empty frame")
                        target_pp_rank, target_pp_local_rank = (
                            _decode_nixl_handshake_query(payload)
                        )
                        target = (target_pp_rank, target_pp_local_rank)
                        response = metadata_store.get(target)
                        if response is None:
                            raise ValueError(
                                "query requested an unknown PP/placement-rank "
                                "coordinate"
                            )
                    except ValueError as error:
                        logger.warning("Rejected NIXL handshake query: %s", error)
                        if identity:
                            ts = msgspec.msgpack.encode(time.perf_counter())
                            try:
                                sock.send_multipart(
                                    (
                                        identity,
                                        b"",
                                        _INVALID_NIXL_HANDSHAKE_RESPONSE,
                                        ts,
                                    )
                                )
                            except zmq.ZMQError:
                                logger.warning(
                                    "Failed to return invalid NIXL handshake response",
                                    exc_info=True,
                                )
                        continue
                    logger.debug(
                        "Received message for pp rank %s, PP-local placement rank %s",
                        target_pp_rank,
                        target_pp_local_rank,
                    )
                    # Echo our perf_counter so P can estimate the clock offset.
                    # perf_counter is only comparable within a process, so this
                    # listener must run in the same process that stamps the block
                    # expiry deadline (`_reqs_need_send`).
                    ts = msgspec.msgpack.encode(time.perf_counter())
                    sock.send_multipart((identity, b"", response, ts))
            if not reset_socket:
                break

    def _get_remote_prefill_token_count(self, num_prompt_tokens: int) -> int:
        """D-side only. Returns N-1 for Mamba models since the decoder
        always recomputes the last token and must start from h(N-1)."""
        if self._has_mamba and num_prompt_tokens > 1:
            return num_prompt_tokens - 1
        return num_prompt_tokens

    def _truncate_mamba_request_for_prefill(self, request: "Request") -> None:
        """P-side only: drop the last prompt token so the prefiller computes
        h(N-1) instead of h(N). The decoder recomputes the last token to
        derive h(N) correctly.

        Guarded by ``_p_side_truncated`` to avoid repeated truncation if the
        request is preempted and rescheduled."""
        params = request.kv_transfer_params
        if (
            params is not None
            # Guard against repeated truncation after preemption/reschedule.
            and not params.get("_p_side_truncated")
            and request.num_prompt_tokens > 1
        ):
            if request.prompt_token_ids is not None:
                request.prompt_token_ids.pop()
            elif request.prompt_embeds is not None:
                request.prompt_embeds = request.prompt_embeds[:-1]
            else:
                return

            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            request.max_tokens = 1
            params["_p_side_truncated"] = True

    def _build_save_meta(
        self,
        meta: NixlConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # only called when use_host_buffer is True to build the save metadata

        # NOTE: For the prefill side, there might be a chance that an early added
        # request is a chunked prefill, so we need to check if new blocks are added
        for req_id, new_block_id_groups, _ in yield_req_data(scheduler_output):
            req_to_save = self._reqs_need_save.get(req_id)
            if req_to_save is None or new_block_id_groups is None:
                continue
            req = req_to_save

            assert req.kv_transfer_params is not None
            clipped_block_id_groups = self.get_exchange_clipped_blocks(
                new_block_id_groups, clip_ssm=False
            )
            meta.add_new_req_to_save(
                request_id=req_id,
                local_block_ids=clipped_block_id_groups,
                kv_transfer_params=req.kv_transfer_params,
            )
            assert scheduler_output.num_scheduled_tokens is not None
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            is_partial = (
                req.num_computed_tokens + num_scheduled_tokens
            ) < req.num_prompt_tokens
            if not is_partial:
                # For non-partial prefills, once new req_meta is scheduled, it
                # can be removed from _reqs_need_save.
                # For partial prefill case, we will retain the request in
                # _reqs_need_save until all blocks are scheduled with req_meta.
                # Therefore, only pop if `not is_partial`.
                self._reqs_need_save.pop(req_id)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = NixlConnectorMetadata()

        # Loop through scheduled reqs and convert to ReqMeta.
        for req_id, (req, block_ids, cached) in self._reqs_need_recv.items():
            assert req.kv_transfer_params is not None
            meta.add_new_req_to_recv(
                request_id=req_id,
                local_block_ids=block_ids,
                kv_transfer_params=req.kv_transfer_params,
                local_num_computed_blocks=cached,
            )

        if self.use_host_buffer:
            self._build_save_meta(meta, scheduler_output)

        meta.reqs_to_send = self._reqs_need_send
        meta.reqs_to_send_transfer_ids = self._reqs_need_send_transfer_ids
        meta.reqs_to_send_expected_participant_counts = (
            self._reqs_need_send_expected_participant_counts
        )
        meta.reqs_to_send_expected_participants = (
            self._reqs_need_send_expected_participants
        )
        # Clock reference for reqs_to_send: deadlines above are in this
        # process's perf_counter domain; workers (possibly on other nodes,
        # where perf_counter has a different epoch) rebase against this.
        meta.scheduler_clock = time.perf_counter()
        meta.reqs_in_batch = self._reqs_in_batch
        meta.reqs_not_processed = self._reqs_not_processed

        # Package heartbeats, throttled by heartbeat_interval.
        if self._heartbeat_by_engine:
            now = time.perf_counter()
            if now - self._last_heartbeat_time >= self._heartbeat_interval:
                self._last_heartbeat_time = now
                meta.heartbeat_by_engine = self._heartbeat_by_engine

        # Clear the list once workers start the transfers
        self._reqs_need_recv.clear()
        self._reqs_in_batch = set()
        self._reqs_not_processed = set()
        self._reqs_need_send = {}
        self._reqs_need_send_transfer_ids = {}
        self._reqs_need_send_expected_participant_counts = {}
        self._reqs_need_send_expected_participants = {}

        return meta

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        """Stop heartbeating for requests whose KV transfer completed."""
        for req_id in connector_output.finished_recving or ():
            self._stop_heartbeat(req_id)

    def has_pending_push_work(self) -> bool:
        return False

    ############################################################
    # Abstract methods that subclasses must implement
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        raise NotImplementedError

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        raise NotImplementedError

    def request_finished(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        raise NotImplementedError
