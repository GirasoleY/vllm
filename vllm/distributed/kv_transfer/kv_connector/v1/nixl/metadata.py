# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dataclasses and helpers for the NIXL connector."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import msgspec

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_placement import (
    KV_PLACEMENT_PROTOCOL_VERSION,
    ConnectorCapabilities,
    KVFormatManifest,
    RankPlacementManifest,
)
from vllm.distributed.kv_transfer.transfer_completion import (
    KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
    WorkerIdentity,
    worker_identities_from_wire,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

TransferHandle = int
ReqId = str

GET_META_MSG = b"get_meta_msg"

# Handshake metadata is static endpoint description, never request transfer
# data.  Bound both envelope layers and every legacy region array before
# recursive/type conversion to prevent an untrusted peer from amplifying
# memory or validation work during bootstrap.
MAX_NIXL_HANDSHAKE_BYTES = 16 * 1024 * 1024
MAX_NIXL_HANDSHAKE_REGIONS = 4096
# Endpoint-wide bootstrap bounds. These constrain control-plane work and
# retained metadata, never the number of direct transfer fragments.
MAX_NIXL_HANDSHAKE_RANKS = 4096
MAX_NIXL_ENDPOINT_HANDSHAKE_BYTES = 256 * 1024 * 1024

# Push-mode (WRITE-based) registration notification.
# Sent worker-to-worker over NIXL: D worker -> P worker, encoded as
# PUSH_REG_NOTIF_PREFIX + msgpack(registration_data).
PUSH_REG_NOTIF_PREFIX = b"PUSH_REG:"
#
# NIXL Connector Version
#
# Increment this version whenever there is an incompatible change to:
#   - NixlAgentMetadata schema
#   - kv_transfer_params schema or semantics
#   - NIXL transfer protocol or wire format
#   - KV cache memory layout or block organization
#   - Any other change that breaks P/D interoperability
#
# Version History:
#   1: Initial version with compatibility checking
#   2: Add remote_request_id to kv_transfer_params
#   3: Add physical_blocks_per_logical_kv_block to NixlAgentMetadata
#   4: Add KV block lease renewal through heartbeats
#   5: Add remote_blocks_expiry_time to kv_transfer_params + handshake
#      clock-sync timestamp
#   6: Validate EAGLE/MTP speculative configuration compatibility
#   7: Include NIXL transfer mode (push vs pull) in the compatibility hash
#   8: Add dcp_size and pcp_size to NixlAgentMetadata
#   9: Add block_strides
#   10: Add dense virtual transfer pages for compressed MLA caches
#   11: Add cp_kv_cache_interleave_size for DCP layout compatibility
#   12: Add optional generic KV placement and named page-registration metadata
#       and preserve remote_num_tokens in worker request metadata
#   13: Add a fresh transfer-attempt identity to pull request metadata
#   14: Add a producer-owned completion-participant count to pull requests
#   15: Add a generic-placement compatibility hash to the handshake envelope
#   16: Bind generic completion to a producer-owned exact participant roster
#   17: Address PCP workers through PP-local flattened handshake ranks and
#       propagate pcp_size in request and heartbeat metadata
#   18: Bind cached handshakes to a scheduler-process endpoint incarnation
#
NIXL_CONNECTOR_VERSION: int = 18


def _wire_fields(value: object, expected: set[str], description: str) -> dict[str, Any]:
    """Return a strict string-keyed wire object with exactly ``expected``."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{description} field names must be strings")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"invalid {description} fields: missing={missing}, unknown={unknown}"
        )
    return dict(value)


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _decode_handshake_msgpack(payload: bytes, description: str) -> object:
    """Decode one bounded handshake layer into its raw wire value."""
    if not isinstance(payload, bytes):
        raise ValueError(f"{description} payload must be bytes")
    if len(payload) > MAX_NIXL_HANDSHAKE_BYTES:
        raise ValueError(
            f"{description} payload exceeds the {MAX_NIXL_HANDSHAKE_BYTES}-byte limit"
        )
    try:
        return msgspec.msgpack.decode(payload)
    except (msgspec.DecodeError, RecursionError) as error:
        raise ValueError(f"invalid {description} MessagePack: {error}") from error


def _bounded_wire_array(data: Mapping[str, Any], field: str) -> None:
    value = data[field]
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if len(value) > MAX_NIXL_HANDSHAKE_REGIONS:
        raise ValueError(
            f"{field} must contain at most {MAX_NIXL_HANDSHAKE_REGIONS} entries"
        )


@dataclass(frozen=True)
class NixlPageRegistrationTemplate:
    """Named page-address template backed by one registered cache tensor.

    Page ``i`` occupies ``[base_address + i * page_stride, +page_size_bytes)``.
    The tensor's storage is registered separately with NIXL; this wire object
    gives a peer enough information to address its advertised logical pages.
    """

    layer_name: str
    base_address: int
    page_stride: int
    page_size_bytes: int
    num_pages: int
    device_id: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.layer_name, str)
            or not self.layer_name
            or self.layer_name.strip() != self.layer_name
        ):
            raise ValueError("layer_name must be a non-empty canonical string")
        _require_nonnegative_int(self.base_address, "base_address")
        _require_positive_int(self.page_stride, "page_stride")
        _require_positive_int(self.page_size_bytes, "page_size_bytes")
        _require_positive_int(self.num_pages, "num_pages")
        _require_nonnegative_int(self.device_id, "device_id")
        if self.page_stride < self.page_size_bytes:
            raise ValueError("page_stride must not be smaller than page_size_bytes")

    @property
    def extent_end_address(self) -> int:
        """Return the exclusive address reached by the final advertised page."""
        return (
            self.base_address
            + (self.num_pages - 1) * self.page_stride
            + self.page_size_bytes
        )

    def page_address(self, page_id: int) -> int:
        """Return an advertised page address, rejecting an invalid page id."""
        _require_nonnegative_int(page_id, "page_id")
        if page_id >= self.num_pages:
            raise ValueError(f"page_id must be in [0, {self.num_pages})")
        return self.base_address + page_id * self.page_stride

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible wire representation."""
        return {
            "layer_name": self.layer_name,
            "base_address": self.base_address,
            "page_stride": self.page_stride,
            "page_size_bytes": self.page_size_bytes,
            "num_pages": self.num_pages,
            "device_id": self.device_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "NixlPageRegistrationTemplate":
        """Parse a page template, rejecting missing or unknown fields."""
        data = _wire_fields(
            value, set(cls.__dataclass_fields__), "NIXL page registration template"
        )
        return cls(**data)


@dataclass(frozen=True)
class NixlPlacementMetadata:
    """Static generic-placement payload advertised during a NIXL handshake."""

    format_manifest: KVFormatManifest
    rank_placement: RankPlacementManifest
    capabilities: ConnectorCapabilities
    page_registration_templates: tuple[NixlPageRegistrationTemplate, ...]

    def __post_init__(self) -> None:
        raw_templates = self.page_registration_templates
        if isinstance(raw_templates, (str, bytes, Mapping)):
            raise ValueError("page_registration_templates must be a sequence")
        try:
            templates = tuple(raw_templates)
        except TypeError as error:
            raise ValueError(
                "page_registration_templates must be a sequence"
            ) from error
        object.__setattr__(self, "page_registration_templates", templates)
        if len(templates) > MAX_NIXL_HANDSHAKE_REGIONS:
            raise ValueError(
                "page_registration_templates must contain at most "
                f"{MAX_NIXL_HANDSHAKE_REGIONS} entries"
            )

        if not isinstance(self.format_manifest, KVFormatManifest):
            raise ValueError("format_manifest must be a KVFormatManifest")
        if not isinstance(self.rank_placement, RankPlacementManifest):
            raise ValueError("rank_placement must be a RankPlacementManifest")
        if not isinstance(self.capabilities, ConnectorCapabilities):
            raise ValueError("capabilities must be ConnectorCapabilities")
        if not templates or any(
            not isinstance(template, NixlPageRegistrationTemplate)
            for template in templates
        ):
            raise ValueError(
                "page_registration_templates must contain "
                "NixlPageRegistrationTemplate values"
            )

        self.rank_placement.validate_format(self.format_manifest)
        templates_by_layer = {template.layer_name: template for template in templates}
        if len(templates_by_layer) != len(templates):
            raise ValueError("page registration layer names must be unique")
        mappings_by_layer = {
            layer_mapping.layer_name: layer_mapping
            for layer_mapping in self.rank_placement.mappings
        }
        if templates_by_layer.keys() != mappings_by_layer.keys():
            raise ValueError(
                "page registration layers must exactly match rank placement layers"
            )
        for layer_name, template in templates_by_layer.items():
            mapping = mappings_by_layer[layer_name].mapping
            if template.page_size_bytes != mapping.local_page_size_bytes:
                raise ValueError(
                    f"layer {layer_name!r} registered page size does not match "
                    "its local page mapping"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible wire representation."""
        return {
            "format_manifest": self.format_manifest.to_dict(),
            "rank_placement": self.rank_placement.to_dict(),
            "capabilities": self.capabilities.to_dict(),
            "page_registration_templates": [
                template.to_dict() for template in self.page_registration_templates
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> "NixlPlacementMetadata":
        """Parse a placement payload, rejecting missing or unknown fields."""
        data = _wire_fields(
            value, set(cls.__dataclass_fields__), "NIXL placement metadata"
        )
        templates = data["page_registration_templates"]
        if not isinstance(templates, list):
            raise ValueError("page_registration_templates must be an array")
        if len(templates) > MAX_NIXL_HANDSHAKE_REGIONS:
            raise ValueError(
                "page_registration_templates must contain at most "
                f"{MAX_NIXL_HANDSHAKE_REGIONS} entries"
            )
        return cls(
            format_manifest=KVFormatManifest.from_dict(data["format_manifest"]),
            rank_placement=RankPlacementManifest.from_dict(data["rank_placement"]),
            capabilities=ConnectorCapabilities.from_dict(data["capabilities"]),
            page_registration_templates=tuple(
                NixlPageRegistrationTemplate.from_dict(template)
                for template in templates
            ),
        )


@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    block_strides: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str
    physical_blocks_per_logical_kv_block: int
    dcp_size: int = 1
    pcp_size: int = 1
    cp_kv_cache_interleave_size: int = 1
    placement_metadata: NixlPlacementMetadata | None = None

    @classmethod
    def from_dict(cls, value: object) -> "NixlAgentMetadata":
        """Parse strict agent metadata while preserving msgspec type checks."""
        data = _wire_fields(value, set(cls.__dataclass_fields__), "NIXL agent metadata")
        for field in ("kv_caches_base_addr", "block_lens", "block_strides"):
            _bounded_wire_array(data, field)
        placement = data["placement_metadata"]
        if placement is not None:
            data["placement_metadata"] = NixlPlacementMetadata.from_dict(placement)
        try:
            return msgspec.convert(data, type=cls, strict=True)
        except msgspec.ValidationError as error:
            raise ValueError(f"invalid NIXL agent metadata: {error}") from error

    @classmethod
    def decode(cls, payload: bytes) -> "NixlAgentMetadata":
        """Decode the actual MessagePack wire payload without unknown fields."""
        return cls.from_dict(_decode_handshake_msgpack(payload, "NIXL agent metadata"))


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking.  The
    strict hash protects the legacy descriptor protocol.  If it differs, the
    placement hash may admit a generic-only endpoint after both sides'
    canonical placement manifests validate.  Agent metadata is decoded only
    after one of those protocol hashes has matched.

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    placement_compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded
    endpoint_incarnation: str = ""

    @classmethod
    def from_dict(cls, value: object) -> "NixlHandshakePayload":
        """Parse the strict outer handshake envelope with typed validation."""
        data = _wire_fields(
            value, set(cls.__dataclass_fields__), "NIXL handshake payload"
        )
        try:
            return msgspec.convert(data, type=cls, strict=True)
        except msgspec.ValidationError as error:
            raise ValueError(f"invalid NIXL handshake payload: {error}") from error

    @classmethod
    def decode(cls, payload: bytes) -> "NixlHandshakePayload":
        """Decode the actual MessagePack wire envelope without unknown fields."""
        return cls.from_dict(_decode_handshake_msgpack(payload, "NIXL handshake"))


def _get_speculative_compatibility_factors(
    vllm_config: VllmConfig,
) -> dict[str, Any] | None:
    """Return NIXL compatibility factors for hidden-state-based speculators."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or not speculative_config.use_eagle():
        return None

    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None
    auxiliary_layer_ids = getattr(
        draft_model_config.hf_config,
        "eagle_aux_hidden_state_layer_ids",
        None,
    )

    # kv_cache_dtype is a user override that defaults to None, meaning "inherit
    # the target's --kv-cache-dtype". Resolve it to the effective value so an
    # explicit setting on one side and inheritance on the other (same effective
    # dtype) don't spuriously mismatch.
    kv_cache_dtype = (
        speculative_config.kv_cache_dtype or vllm_config.cache_config.cache_dtype
    )

    # Note: the draft attention_backend is intentionally not hashed. Its only
    # transfer-relevant effect is the KV block layout/size, which is validated
    # per region at runtime in _validate_remote_agent_handshake. The connector
    # only sees the raw override here (usually None = auto-select), never the
    # resolved backend, so hashing it would cause false mismatches without
    # catching anything the runtime layout check misses.
    return {
        "method": speculative_config.method,
        "model": draft_model_config.model,
        "revision": draft_model_config.revision,
        "code_revision": draft_model_config.code_revision,
        "parallel_drafting": speculative_config.parallel_drafting,
        "kv_cache_dtype": str(kv_cache_dtype),
        "auxiliary_layer_ids": (
            tuple(auxiliary_layer_ids) if auxiliary_layer_ids is not None else None
        ),
    }


def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig,
    attn_backend_name: str,
    transfer_mode: str = "pull",
) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache format (dtype, sliding window)
    - Attention backend
    - EAGLE/MTP configuration that affects transferred state
    - Transfer mode (push vs pull)

    The transfer mode is included because the push (WRITE) and pull (READ)
    connectors use incompatible transfer protocols; a push connector and a
    pull connector must never complete a handshake with each other.

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in _validate_remote_agent_handshake and are not
    included in this hash to support heterogeneous deployments.

    Note - the set of factors are likely to evolve significantly over
    time to be more or less permissive.

    Returns:
        SHA-256 hex digest
    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager

    factors = {
        # Version compatibility
        "vllm_version": vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        # Model architecture - affects KV cache shape
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        # Attention backend and KV cache dtype affect memory layout
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
        "is_hma_enabled": is_hma_enabled,
        "speculative_config": _get_speculative_compatibility_factors(vllm_config),
        # push (WRITE) and pull (READ) connectors are protocol-incompatible
        "transfer_mode": transfer_mode,
    }

    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s, attn_backend=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        attn_backend_name,
    )
    return compat_hash


def compute_nixl_placement_compatibility_hash(
    vllm_config: VllmConfig,
    transfer_mode: str = "pull",
) -> str:
    """Hash semantics that must match before generic placement is decoded.

    Unlike :func:`compute_nixl_compatibility_hash`, this deliberately excludes
    attention backend, concrete KV layout, and allocator policy. Generic
    placement manifests describe those byte layouts explicitly. Model, dtype,
    connector/protocol, speculative-state, and transfer-direction mismatches
    remain incompatible and are rejected before agent metadata is trusted.
    """
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    factors = {
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        "kv_placement_protocol_version": KV_PLACEMENT_PROTOCOL_VERSION,
        "kv_completion_protocol_version": KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
        "model": model_config.model,
        # This is the same model identity embedded in the runtime canonical
        # format manifest, so the early gate and later placement composition
        # cannot disagree about model semantics.
        "model_fingerprint": model_config.compute_hash(),
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        "cache_dtype": str(cache_config.cache_dtype),
        "speculative_config": _get_speculative_compatibility_factors(vllm_config),
        "transfer_mode": transfer_mode,
    }
    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL placement compatibility hash: %s "
        "(model=%s, dtype=%s, cache_dtype=%s, mode=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["cache_dtype"],
        transfer_mode,
    )
    return compat_hash


@dataclass
class HeartbeatInfo:
    """Heartbeat data for a single remote engine, sent from D worker to P."""

    req_ids: set[ReqId]
    host: str
    port: int
    tp_size: int
    dcp_size: int = 1
    pcp_size: int = 1
    pp_size: int = 1
    endpoint_incarnation: str | None = None


@dataclass
class RemoteMeta:
    block_ids: BlockIds
    host: str
    port: int
    engine_id: str
    request_id: str
    endpoint_incarnation: str | None = None
    blocks_expiry_time: float | None = None
    # Fresh scheduler-generated identity for this pinned-block transfer attempt.
    # Generic completion notifications require it to reject delayed retries.
    transfer_id: str | None = None
    # Quorum selected by the producer before the consumer transfer starts.
    expected_completion_participant_count: int | None = None
    # Exact destination processes authorized by the producer-side request.
    # A count alone is not an authorization boundary: the completion sender
    # must not be allowed to choose the roster that releases source pages.
    expected_completion_participants: tuple[WorkerIdentity, ...] = ()


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: BlockIds
    tp_size: int
    dcp_size: int = 1
    # Remote producer prefill-context-parallel size. PCP workers retain
    # replicated persistent KV but remain distinct transfer participants.
    pcp_size: int = 1
    # Per-KV-cache-group logical blocks this rank already holds, i.e. its
    # prefix-cache hit. Fixes where this rank's DCP slice starts relative to
    # the remote's; kept per-group since hybrid models (e.g. SWA+FA) can have
    # different cache-hit counts per group.
    local_num_computed_blocks: tuple[int, ...] = ()
    # Number of destination attention kernel blocks covered by the union of
    # all DCP source-rank reads, per KV cache group. Empty when DCP is inactive.
    dcp_local_attention_blocks_covered: tuple[int, ...] = ()
    remote: RemoteMeta | None = None
    # Remote block size, discovered during NIXL handshake (push mode).
    remote_block_size: int | None = None
    # Remote producer pipeline-parallel size (push mode, D side).
    pp_size: int = 1
    # Exact canonical token coverage exported by the remote scheduler.
    remote_num_tokens: int = 0


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        # Exact transfer attempt corresponding to each reqs_to_send lease.
        self.reqs_to_send_transfer_ids: dict[ReqId, str] = {}
        # Producer-owned quorum for generic completion of each send lease.
        # This originates on the request that creates the producer pages, not
        # from the later consumer completion envelope.
        self.reqs_to_send_expected_participant_counts: dict[ReqId, int] = {}
        # Producer-owned exact participant roster paired with each send lease.
        self.reqs_to_send_expected_participants: dict[
            ReqId, tuple[WorkerIdentity, ...]
        ] = {}
        # The scheduler process's time.perf_counter() when this metadata was
        # built. reqs_to_send deadlines are stamped with the scheduler's
        # clock, which is NOT comparable across processes (perf_counter is
        # process/boot-local): workers must rebase the remaining TTL onto
        # their own clock via this reference. 0.0 = unset (legacy metadata).
        self.scheduler_clock: float = 0.0
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        # Heartbeat data grouped by remote engine, sent by D worker to P.
        self.heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Push mode (D side): registration data the D worker should send to
        # P workers via NIXL notification on this step.
        self.push_registrations: dict[ReqId, dict[str, Any]] = {}
        # Push mode (P side): newly finished request blocks to be matched
        # against pending D registrations on the P worker.
        self.push_finished_blocks: dict[ReqId, BlockIds] = {}

    def _add_new_req(
        self,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
        local_num_computed_blocks: tuple[int, ...] = (),
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            # P workers don't need to receive these from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
            dcp_size=kv_transfer_params.get("dcp_size", 1),
            pcp_size=kv_transfer_params.get("pcp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
            pp_size=kv_transfer_params.get("pp_size", 1),
            remote_num_tokens=kv_transfer_params.get("remote_num_tokens", 0),
            local_num_computed_blocks=local_num_computed_blocks,
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        self.reqs_to_save[request_id] = self._add_new_req(
            local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
        local_num_computed_blocks: tuple[int, ...] = (),
    ):
        req = self._add_new_req(
            local_block_ids, kv_transfer_params, local_num_computed_blocks
        )
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            endpoint_incarnation=kv_transfer_params.get("remote_endpoint_incarnation"),
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            blocks_expiry_time=kv_transfer_params.get("remote_blocks_expiry_time"),
            transfer_id=kv_transfer_params.get("transfer_id"),
            expected_completion_participant_count=kv_transfer_params.get(
                "expected_kv_completion_participant_count"
            ),
            expected_completion_participants=(
                worker_identities_from_wire(
                    kv_transfer_params["expected_kv_completion_participants"],
                    name="expected_kv_completion_participants",
                )
                if kv_transfer_params.get("expected_kv_completion_participants")
                is not None
                else ()
            ),
        )
        self.reqs_to_recv[request_id] = req
