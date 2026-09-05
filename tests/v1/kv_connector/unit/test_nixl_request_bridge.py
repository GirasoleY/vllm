# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    base_worker as nixl_base_worker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    metadata as nixl_metadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.direct_request import (
    iter_nixl_request_layer_direct_batches,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NixlPageRegistrationTemplate,
    NixlPlacementMetadata,
    RemoteMeta,
    ReqMeta,
    compute_nixl_compatibility_hash,
    compute_nixl_placement_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge import (
    MAX_NIXL_COMPLETION_BYTES,
    NIXL_DIRECT_COMPLETION_PREFIX,
    NixlDirectCompletionEnvelope,
    build_nixl_read_request_plan,
    index_remote_nixl_placements,
    iter_nixl_read_plan_windows,
    iter_prepare_nixl_read_request,
    materialize_nixl_read_request_plan,
    nixl_read_request_plan_digest,
    prepare_nixl_read_request,
    select_nixl_destination_prefix_blocks,
    validate_complete_nixl_placement_endpoint,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.segmented import (
    NixlEphemeralDlistTracker,
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
)
from vllm.distributed.kv_transfer.transfer_completion import (
    KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
    MAX_TRANSFER_PARTICIPANTS,
    CompletionStatus,
    TransferCompletionNotification,
    WorkerIdentity,
)

from .utils import set_mock_multipart_replies


def _mapping(page_span: int) -> CanonicalPageMapping:
    return CanonicalPageMapping(
        canonical_page_size_bytes=page_span,
        local_page_size_bytes=page_span,
        runs=(CopyRun(0, 0, 1, page_span, 1, 1),),
        num_writers=1,
        writer_index=0,
        canonical_token_span=page_span,
        canonical_region_token_strides=((0, 1),),
    )


def _placement(
    deployment: str,
    worker_id: str,
    *,
    page_span: int,
    rank: int = 0,
    tp_size: int = 1,
    tp_rank: int = 0,
    pp_size: int = 1,
    pp_rank: int = 0,
    pcp_size: int = 1,
    pcp_rank: int = 0,
    dcp_size: int = 1,
    dcp_rank: int = 0,
    cp_interleave: int = 1,
    max_segments: int | None = 17,
) -> NixlPlacementMetadata:
    layer = "model.layers.0.self_attn"
    canonical_page_span = page_span * dcp_size
    if page_span % cp_interleave:
        raise ValueError("test page span must be divisible by CP interleave")
    mapping = (
        _mapping(page_span)
        if dcp_size == 1
        else CanonicalPageMapping(
            canonical_page_size_bytes=canonical_page_span,
            local_page_size_bytes=page_span,
            runs=tuple(
                CopyRun(
                    local_offset=local_offset,
                    canonical_offset=(
                        (local_offset // cp_interleave) * dcp_size + dcp_rank
                    )
                    * cp_interleave,
                    fragment_size=cp_interleave,
                    num_fragments=1,
                    local_stride=cp_interleave,
                    canonical_stride=cp_interleave,
                )
                for local_offset in range(0, page_span, cp_interleave)
            ),
            num_writers=1,
            writer_index=0,
            canonical_token_span=canonical_page_span,
            canonical_region_token_strides=((0, 1),),
        )
    )
    group = KVGroupFormat(
        group_id=0,
        semantic_id="decoder-mla",
        kind="mla",
        layer_names=(layer,),
        canonical_page_token_span=canonical_page_span,
        dtype="uint8",
        canonical_page_size_bytes=canonical_page_span,
        format_id="test-byte-per-token",
    )
    manifest = KVFormatManifest(1, "model-v1", (group,))
    rank_placement = RankPlacementManifest(
        version=1,
        deployment_id=deployment,
        topology_generation=5,
        worker_id=worker_id,
        worker_incarnation=f"{worker_id}-boot",
        format_manifest_fingerprint=manifest.fingerprint(),
        rank=rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        dcp_group_id=f"{deployment}-dcp",
        pcp_size=pcp_size,
        pcp_rank=pcp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        dp_size=1,
        dp_rank=0,
        dp_group_id=f"{deployment}-dp",
        ep_size=1,
        ep_rank=0,
        cp_interleave=cp_interleave,
        layer_range=(0, 1),
        mappings=(LayerPageMapping(layer, 0, "decoder-mla", mapping),),
    )
    return NixlPlacementMetadata(
        format_manifest=manifest,
        rank_placement=rank_placement,
        capabilities=ConnectorCapabilities(
            contiguous_copy=True,
            strided_copy=True,
            scatter_gather=True,
            gpu_pack_unpack=False,
            supports_read=True,
            supports_write=False,
            max_segments_per_batch=max_segments,
        ),
        page_registration_templates=(
            NixlPageRegistrationTemplate(
                layer_name=layer,
                base_address=1000 + rank * 100,
                page_stride=page_span,
                page_size_bytes=page_span,
                num_pages=64,
                device_id=rank,
            ),
        ),
    )


def _agent(engine_id: str, placement: NixlPlacementMetadata) -> NixlAgentMetadata:
    templates = placement.page_registration_templates
    assert templates
    assert len({template.device_id for template in templates}) == 1
    assert len({template.num_pages for template in templates}) == 1
    return NixlAgentMetadata(
        engine_id=engine_id,
        agent_metadata=b"agent",
        kv_caches_base_addr=[template.base_address for template in templates],
        device_id=templates[0].device_id,
        num_blocks=templates[0].num_pages,
        block_lens=[template.page_size_bytes for template in templates],
        block_strides=[template.page_stride for template in templates],
        kv_cache_layout="HND",
        block_size=4,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASHINFER_MLA",
        physical_blocks_per_logical_kv_block=1,
        dcp_size=placement.rank_placement.dcp_size,
        pcp_size=placement.rank_placement.pcp_size,
        cp_kv_cache_interleave_size=placement.rank_placement.cp_interleave,
        placement_metadata=placement,
    )


def _compatibility_config(
    *,
    model: str = "model-v1",
    dtype: str = "float16",
    cache_dtype: str = "fp8",
):
    model_fingerprint = f"{model}:{dtype}"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model,
            dtype=dtype,
            compute_hash=lambda: model_fingerprint,
            get_total_num_kv_heads=lambda: 1,
            get_head_size=lambda: 128,
            get_total_num_hidden_layers=lambda: 1,
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=True),
        speculative_config=None,
    )


def _run_test_handshake(
    *,
    local_legacy_hash: str,
    local_placement_hash: str,
    remote_legacy_hash: str,
    remote_placement_hash: str,
):
    local = _placement("decode", "decode-0", page_span=4)
    remote = _placement("prefill-engine", "prefill-0", page_span=4)
    remote_agent = _agent("prefill-engine", remote)
    remote_agent_name = remote.rank_placement.worker_incarnation
    payload = NixlHandshakePayload(
        compatibility_hash=remote_legacy_hash,
        placement_compatibility_hash=remote_placement_hash,
        agent_metadata_bytes=msgspec.msgpack.encode(remote_agent),
    )
    socket = MagicMock()
    set_mock_multipart_replies(
        socket,
        [
            [
                msgspec.msgpack.encode(payload),
                msgspec.msgpack.encode(time.perf_counter()),
            ]
        ],
    )

    worker = object.__new__(NixlPullConnectorWorker)
    worker._is_csa_linear = False
    worker.use_host_buffer = True
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda *_: [0])
    worker._local_placement_metadata = local
    worker._physical_blocks_per_logical_kv_block = 1
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 1
    worker.block_size = 4
    worker.compat_hash = local_legacy_hash
    worker.placement_compat_hash = local_placement_hash
    worker.enforce_compat_hash = True
    worker._remote_placement_indexes = {}
    worker._generic_only_remote_engines = set()
    worker._validate_remote_parallel_config = MagicMock()
    worker.add_remote_agent = MagicMock(return_value=remote_agent_name)

    with patch.object(nixl_base_worker, "zmq_ctx") as mock_zmq_ctx:
        mock_zmq_ctx.return_value.__enter__.return_value = socket
        result = worker._nixl_handshake_impl(
            host="prefill-host",
            port=1234,
            remote_tp_size=1,
            expected_engine_id="prefill-engine",
            imported_agents=[],
        )
    return worker, result


def test_generic_handshake_accepts_backend_only_legacy_hash_mismatch():
    config = _compatibility_config()
    local_legacy_hash = compute_nixl_compatibility_hash(config, "FLASHINFER_MLA")
    remote_legacy_hash = compute_nixl_compatibility_hash(config, "TRITON_MLA")
    local_placement_hash = compute_nixl_placement_compatibility_hash(config)
    remote_placement_hash = compute_nixl_placement_compatibility_hash(config)
    assert local_legacy_hash != remote_legacy_hash
    assert local_placement_hash == remote_placement_hash

    worker, (agents, _) = _run_test_handshake(
        local_legacy_hash=local_legacy_hash,
        local_placement_hash=local_placement_hash,
        remote_legacy_hash=remote_legacy_hash,
        remote_placement_hash=remote_placement_hash,
    )

    assert agents == {(0, 0): "prefill-0-boot"}
    assert "prefill-engine" in worker._remote_placement_indexes
    assert worker._generic_only_remote_engines == {"prefill-engine"}
    assert worker.add_remote_agent.call_args.kwargs["generic_registration"] is True


def test_placement_compatibility_is_independent_of_raw_vllm_version():
    config = _compatibility_config()
    placement_hash = compute_nixl_placement_compatibility_hash(config)
    strict_hash = compute_nixl_compatibility_hash(config, "FLASHINFER_MLA")

    with patch("vllm.__version__", "different-compatible-build"):
        assert compute_nixl_placement_compatibility_hash(config) == placement_hash
        assert compute_nixl_compatibility_hash(config, "FLASHINFER_MLA") != strict_hash


def test_strict_handshake_keeps_legacy_and_generic_request_paths():
    config = _compatibility_config()
    legacy_hash = compute_nixl_compatibility_hash(config, "FLASHINFER_MLA")
    placement_hash = compute_nixl_placement_compatibility_hash(config)

    worker, _ = _run_test_handshake(
        local_legacy_hash=legacy_hash,
        local_placement_hash=placement_hash,
        remote_legacy_hash=legacy_hash,
        remote_placement_hash=placement_hash,
    )

    assert "prefill-engine" in worker._remote_placement_indexes
    assert worker._generic_only_remote_engines == set()
    assert worker.add_remote_agent.call_args.kwargs["generic_registration"] is False


@pytest.mark.parametrize(
    "generic_opt_in",
    (True, False),
)
def test_pp_handshake_requires_generic_metadata_on_both_endpoints(
    generic_opt_in: bool,
):
    complete = _two_layer_placement()
    remotes = tuple(
        replace(
            complete,
            rank_placement=replace(
                complete.rank_placement,
                worker_id=f"prefill-pp{pp_rank}-tp0",
                worker_incarnation=f"prefill-pp{pp_rank}-tp0-boot",
                rank=pp_rank,
                pp_size=2,
                pp_rank=pp_rank,
                layer_range=(pp_rank, pp_rank + 1),
                mappings=(complete.rank_placement.mappings[pp_rank],),
            ),
            page_registration_templates=(
                complete.page_registration_templates[pp_rank],
            ),
        )
        for pp_rank in range(2)
    )
    socket = MagicMock()
    set_mock_multipart_replies(
        socket,
        [
            [
                msgspec.msgpack.encode(
                    NixlHandshakePayload(
                        compatibility_hash="strict",
                        placement_compatibility_hash="placement",
                        agent_metadata_bytes=msgspec.msgpack.encode(
                            _agent("prefill-engine", remote)
                        ),
                    )
                ),
                msgspec.msgpack.encode(time.perf_counter()),
            ]
            for remote in remotes
        ],
    )

    worker = object.__new__(NixlPullConnectorWorker)
    worker._is_csa_linear = False
    worker.use_host_buffer = True
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda *_: [0])
    worker._local_placement_metadata = (
        _placement("decode", "decode-0", page_span=4) if generic_opt_in else None
    )
    worker._enable_generic_placement = generic_opt_in
    worker.pp_size = 1
    worker._physical_blocks_per_logical_kv_block = 1
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 1
    worker.block_size = 4
    worker.compat_hash = "strict"
    worker.placement_compat_hash = "placement"
    worker.enforce_compat_hash = True
    worker._remote_placement_indexes = {}
    worker._generic_only_remote_engines = set()
    worker._validate_remote_parallel_config = MagicMock()
    worker.add_remote_agent = MagicMock(
        side_effect=[remote.rank_placement.worker_incarnation for remote in remotes]
    )

    with patch.object(nixl_base_worker, "zmq_ctx") as mock_zmq_ctx:
        mock_zmq_ctx.return_value.__enter__.return_value = socket
        if generic_opt_in:
            worker._nixl_handshake_impl(
                host="prefill-host",
                port=1234,
                remote_tp_size=1,
                expected_engine_id="prefill-engine",
                remote_pp_size=2,
                imported_agents=[],
            )
        else:
            with pytest.raises(
                RuntimeError,
                match="generic NIXL requires placement metadata on both endpoints",
            ):
                worker._nixl_handshake_impl(
                    host="prefill-host",
                    port=1234,
                    remote_tp_size=1,
                    expected_engine_id="prefill-engine",
                    remote_pp_size=2,
                    imported_agents=[],
                )
            assert worker.add_remote_agent.call_count == 0
            assert worker._generic_only_remote_engines == set()
            assert worker._remote_placement_indexes == {}
            return

    assert worker.add_remote_agent.call_count == 2
    assert all(
        call.kwargs["generic_registration"] is True
        for call in worker.add_remote_agent.call_args_list
    )
    assert worker._generic_only_remote_engines == {"prefill-engine"}
    assert "prefill-engine" in worker._remote_placement_indexes


def test_strict_handshake_does_not_index_mismatched_generic_protocol():
    config = _compatibility_config()
    legacy_hash = compute_nixl_compatibility_hash(config, "FLASHINFER_MLA")
    placement_hash = compute_nixl_placement_compatibility_hash(config)

    worker, _ = _run_test_handshake(
        local_legacy_hash=legacy_hash,
        local_placement_hash=placement_hash,
        remote_legacy_hash=legacy_hash,
        remote_placement_hash="incompatible-placement-protocol",
    )

    assert worker._remote_placement_indexes == {}
    assert worker._generic_only_remote_engines == set()
    assert worker.add_remote_agent.call_args.kwargs["generic_registration"] is False


def test_strict_handshake_uses_generic_path_for_token_interleaved_asymmetric_dcp():
    config = _compatibility_config()
    legacy_hash = compute_nixl_compatibility_hash(config, "FLASHINFER_MLA")
    placement_hash = compute_nixl_placement_compatibility_hash(config)
    local = _placement("decode", "decode-0", page_span=4)
    remotes = tuple(
        _placement(
            "prefill-engine",
            f"prefill-{rank}",
            page_span=4,
            rank=rank,
            tp_size=2,
            tp_rank=rank,
            dcp_size=2,
            dcp_rank=rank,
            cp_interleave=1,
        )
        for rank in range(2)
    )
    socket = MagicMock()
    set_mock_multipart_replies(
        socket,
        [
            [
                msgspec.msgpack.encode(
                    NixlHandshakePayload(
                        compatibility_hash=legacy_hash,
                        placement_compatibility_hash=placement_hash,
                        agent_metadata_bytes=msgspec.msgpack.encode(
                            _agent("prefill-engine", remote)
                        ),
                    )
                ),
                msgspec.msgpack.encode(time.perf_counter()),
            ]
            for remote in remotes
        ],
    )

    worker = object.__new__(NixlPullConnectorWorker)
    worker._is_csa_linear = False
    worker.use_host_buffer = True
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda *_: [0])
    worker._local_placement_metadata = local
    worker._physical_blocks_per_logical_kv_block = 1
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 1
    worker.block_size = 4
    worker.compat_hash = legacy_hash
    worker.placement_compat_hash = placement_hash
    worker.enforce_compat_hash = False
    worker._remote_placement_indexes = {}
    worker._generic_only_remote_engines = set()
    worker._validate_remote_parallel_config = MagicMock()
    worker.add_remote_agent = MagicMock(
        side_effect=[remote.rank_placement.worker_incarnation for remote in remotes]
    )

    with patch.object(nixl_base_worker, "zmq_ctx") as mock_zmq_ctx:
        mock_zmq_ctx.return_value.__enter__.return_value = socket
        worker._nixl_handshake_impl(
            host="prefill-host",
            port=1234,
            remote_tp_size=2,
            expected_engine_id="prefill-engine",
            remote_dcp_size=2,
            imported_agents=[],
        )

    assert worker._generic_only_remote_engines == {"prefill-engine"}
    assert worker.add_remote_agent.call_count == 2
    assert all(
        call.kwargs["generic_registration"] is True
        for call in worker.add_remote_agent.call_args_list
    )


def test_handshake_rejects_and_rolls_back_mixed_rank_modes():
    local = _placement("decode", "decode-0", page_span=4, rank=0, tp_size=2, tp_rank=0)
    remotes = (
        _placement(
            "prefill-engine",
            "prefill-0",
            page_span=4,
            rank=0,
            tp_size=2,
            tp_rank=0,
        ),
        _placement(
            "prefill-engine",
            "prefill-1",
            page_span=4,
            rank=1,
            tp_size=2,
            tp_rank=1,
        ),
    )
    payloads = [
        NixlHandshakePayload(
            compatibility_hash=legacy_hash,
            placement_compatibility_hash="placement-hash",
            agent_metadata_bytes=msgspec.msgpack.encode(
                _agent("prefill-engine", placement)
            ),
        )
        for legacy_hash, placement in zip(("legacy-local", "legacy-other"), remotes)
    ]
    socket = MagicMock()
    set_mock_multipart_replies(
        socket,
        [
            [
                msgspec.msgpack.encode(payload),
                msgspec.msgpack.encode(time.perf_counter()),
            ]
            for payload in payloads
        ],
    )

    worker = object.__new__(NixlPullConnectorWorker)
    worker._is_csa_linear = False
    worker.use_host_buffer = True
    worker.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda *_: [0],
        unregister_remote_engine=MagicMock(),
    )
    worker._local_placement_metadata = local
    worker._physical_blocks_per_logical_kv_block = 1
    worker.dcp_size = 1
    worker.cp_kv_cache_interleave_size = 1
    worker.block_size = 4
    worker.compat_hash = "legacy-local"
    worker.placement_compat_hash = "placement-hash"
    worker.enforce_compat_hash = True
    worker._remote_placement_indexes = {}
    worker._generic_only_remote_engines = set()
    worker._validate_remote_parallel_config = MagicMock()
    worker.add_remote_agent = MagicMock(
        side_effect=[remotes[0].rank_placement.worker_incarnation]
    )
    worker.nixl_wrapper = SimpleNamespace(remove_remote_agent=MagicMock())
    worker.dst_xfer_side_handles = defaultdict(dict)
    worker._remote_agents = defaultdict(dict)
    worker.kv_caches_base_addr = defaultdict(dict)
    worker.dst_num_blocks = {}
    worker.tp_mappings = {}
    worker._engine_clock_offset = {}
    worker._engine_last_active = {}
    worker._remote_handshake_specs = {}
    worker._stale_remote_engines = set()

    with (
        patch.object(nixl_base_worker, "zmq_ctx") as mock_zmq_ctx,
        pytest.raises(RuntimeError, match="mixed legacy and generic-only"),
    ):
        mock_zmq_ctx.return_value.__enter__.return_value = socket
        worker._nixl_handshake(
            host="prefill-host",
            port=1234,
            remote_tp_size=2,
            expected_engine_id="prefill-engine",
        )

    worker.nixl_wrapper.remove_remote_agent.assert_called_once_with(
        remotes[0].rank_placement.worker_incarnation
    )
    assert worker._remote_placement_indexes == {}
    assert worker._generic_only_remote_engines == set()


def test_generic_handshake_rejects_model_dtype_and_protocol_mismatches():
    local_config = _compatibility_config()
    local_placement_hash = compute_nixl_placement_compatibility_hash(local_config)
    incompatible_hashes = {
        "model": compute_nixl_placement_compatibility_hash(
            _compatibility_config(model="model-v2")
        ),
        "dtype": compute_nixl_placement_compatibility_hash(
            _compatibility_config(dtype="bfloat16")
        ),
        "cache_dtype": compute_nixl_placement_compatibility_hash(
            _compatibility_config(cache_dtype="bfloat16")
        ),
        "transfer_mode": compute_nixl_placement_compatibility_hash(
            local_config, transfer_mode="push"
        ),
    }
    protocol_versions = (
        "NIXL_CONNECTOR_VERSION",
        "KV_PLACEMENT_PROTOCOL_VERSION",
        "KV_TRANSFER_COMPLETION_PROTOCOL_VERSION",
    )
    for constant in protocol_versions:
        with patch.object(nixl_metadata, constant, 999):
            incompatible_hashes[constant] = compute_nixl_placement_compatibility_hash(
                local_config
            )

    for mismatch, remote_placement_hash in incompatible_hashes.items():
        assert remote_placement_hash != local_placement_hash
        with pytest.raises(
            RuntimeError,
            match="placement compatibility hash mismatch",
        ):
            _run_test_handshake(
                local_legacy_hash="local-legacy",
                local_placement_hash=local_placement_hash,
                remote_legacy_hash=f"remote-{mismatch}-legacy",
                remote_placement_hash=remote_placement_hash,
            )


def _two_layer_placement(*, alias_registration: bool = False):
    placement = _placement("prefill-engine", "prefill-0", page_span=4)
    first_layer = "model.layers.0.self_attn"
    second_layer = "model.layers.1.self_attn"
    group = replace(
        placement.format_manifest.groups[0],
        layer_names=(first_layer, second_layer),
    )
    format_manifest = replace(placement.format_manifest, groups=(group,))
    first_mapping = placement.rank_placement.mappings[0]
    second_mapping = replace(
        first_mapping,
        layer_name=second_layer,
        layer_index=1,
    )
    rank_placement = replace(
        placement.rank_placement,
        format_manifest_fingerprint=format_manifest.fingerprint(),
        layer_range=(0, 2),
        mappings=(first_mapping, second_mapping),
    )
    first_template = placement.page_registration_templates[0]
    second_template = replace(
        first_template,
        layer_name=second_layer,
        base_address=(
            first_template.base_address
            if alias_registration
            else first_template.extent_end_address + 4096
        ),
    )
    return replace(
        placement,
        format_manifest=format_manifest,
        rank_placement=rank_placement,
        page_registration_templates=(first_template, second_template),
    )


class _FakeNixlWrapper:
    def __init__(self):
        self.next_handle = 100
        self.dlists = []
        self.xfers = []
        self.released_xfers = []
        self.released_dlists = []
        self.started = []
        self.notifications = []

    def get_xfer_descs(self, descriptors, memory_type):
        assert memory_type == "VRAM"
        return tuple(descriptors)

    def prep_xfer_dlist(self, agent_name, descriptors):
        handle = self.next_handle
        self.next_handle += 1
        self.dlists.append((agent_name, tuple(descriptors), handle))
        return handle

    def make_prepped_xfer(
        self, operation, local_handle, local_ids, remote_handle, remote_ids
    ):
        handle = self.next_handle
        self.next_handle += 1
        self.xfers.append(
            (operation, local_handle, local_ids, remote_handle, remote_ids, handle)
        )
        return handle

    def release_xfer_handle(self, handle):
        self.released_xfers.append(handle)

    def release_dlist_handle(self, handle):
        self.released_dlists.append(handle)

    def transfer(self, handle):
        self.started.append(handle)

    def send_notif(self, agent, notif_msg=None):
        self.notifications.append((agent, notif_msg))


def _terminal_notification_worker(payload, agents, wrapper):
    worker = object.__new__(NixlPullConnectorWorker)
    worker._direct_read_batch_windows = {}
    worker._direct_read_notifications = {
        "decode-request": (payload, tuple(agents)),
    }
    worker.nixl_wrapper = wrapper
    worker._log_failure = MagicMock()
    worker.xfer_stats = MagicMock()
    return worker


def test_worker_metadata_preserves_remote_transfer_attempt_id():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="decode-request",
        local_block_ids=([20],),
        kv_transfer_params={
            "remote_block_ids": ([10],),
            "remote_engine_id": "prefill-engine",
            "remote_request_id": "prefill-request",
            "remote_host": "prefill-host",
            "remote_port": 8000,
            "remote_num_tokens": 4,
            "transfer_id": "attempt-1",
        },
    )

    remote = metadata.reqs_to_recv["decode-request"].remote
    assert remote is not None
    assert remote.transfer_id == "attempt-1"


def test_bridge_maps_prefix_tail_across_different_page_spans():
    source = _placement("prefill-engine", "prefill-0", page_span=4, max_segments=31)
    destination = _placement("decode", "decode-0", page_span=8, max_segments=7)

    result = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10, 11, 12],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(1,),
        remote_num_tokens=10,
    )

    assert result.ranges[0].first_token == 8
    assert result.ranges[0].token_count == 8
    assert result.ranges[0].valid_token_count == 2
    transfer_plan, source_allocations, destination_allocations = (
        materialize_nixl_read_request_plan(result)
    )
    assert [
        (page.local_page_id, page.canonical_page_index, page.first_token)
        for page in source_allocations
    ] == [(10, 0, 0), (11, 1, 4), (12, 2, 8)]
    assert [
        (page.local_page_id, page.canonical_page_index, page.first_token)
        for page in destination_allocations
    ] == [(20, 1, 8)]
    assert result.max_segments_per_batch == 7

    runs = transfer_plan.layers[0].runs
    assert len(runs) == 1
    assert (
        runs[0].source_page_id,
        runs[0].destination_page_id,
        runs[0].fragment_size,
    ) == (12, 20, 2)


def test_bridge_prepares_direct_read_without_submitting_or_notifying():
    source = _placement("prefill-engine", "prefill-0", page_span=4, max_segments=1)
    destination = _placement("decode", "decode-0", page_span=8, max_segments=1)
    request = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10, 11, 12],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(1,),
        remote_num_tokens=10,
    )
    remote = index_remote_nixl_placements(
        {(0, 0): _agent("prefill-engine", source)},
        {(0, 0): source.rank_placement.worker_incarnation},
    )
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)

    prepared = prepare_nixl_read_request(
        request,
        remote,
        nixl_wrapper=wrapper,
        tracker=tracker,
        local_transfer_rank=0,
        memory_type="VRAM",
    )

    assert prepared.transfer_handles == (102,)
    assert wrapper.dlists == [
        ("NIXL_INIT_AGENT", ((1160, 2, 0),), 100),
        ("prefill-0-boot", ((1048, 2, 0),), 101),
    ]
    assert wrapper.xfers[0][0] == "READ"
    assert tracker.pending_count == 1


def test_bridge_globally_plans_but_prepares_only_local_destination_rank():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination_0 = _placement(
        "decode", "decode-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    destination_1 = _placement(
        "decode", "decode-1", page_span=4, rank=1, tp_size=2, tp_rank=1
    )
    destination_workers = validate_complete_nixl_placement_endpoint(
        (destination_1, destination_0)
    )
    request = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=destination_workers,
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )
    windows = tuple(iter_nixl_read_plan_windows(request))
    assert request.destination_expected_participant_count == 2
    assert {run.destination_rank for run in windows[0].layer_plan.runs} == {
        0,
        1,
    }

    remote = index_remote_nixl_placements(
        {(0, 0): _agent("prefill-engine", source)},
        {(0, 0): source.rank_placement.worker_incarnation},
    )
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    prepared = prepare_nixl_read_request(
        request,
        remote,
        nixl_wrapper=wrapper,
        tracker=tracker,
        local_transfer_rank=1,
        memory_type="VRAM",
    )

    assert len(prepared.batches) == 1
    assert prepared.batches[0].descriptor_batch.destination_rank == 1
    assert wrapper.dlists == [
        ("NIXL_INIT_AGENT", ((1180, 4, 1),), 100),
        ("prefill-0-boot", ((1040, 4, 0),), 101),
    ]


def test_pull_worker_submits_all_batches_before_aggregate_notification():
    source = _placement("prefill-engine", "prefill-0", page_span=4, max_segments=1)
    destination = _placement("decode", "decode-0", page_span=4, max_segments=1)
    remote = index_remote_nixl_placements(
        {(0, 0): _agent("prefill-engine", source)},
        {(0, 0): source.rank_placement.worker_incarnation},
    )
    wrapper = _FakeNixlWrapper()
    worker = object.__new__(NixlPullConnectorWorker)
    worker._local_placement_metadata = destination
    worker._local_placement_workers = (destination,)
    worker._remote_placement_indexes = {"prefill-engine": remote}
    worker._physical_blocks_per_logical_kv_block = 1
    worker.kv_cache_config = SimpleNamespace(
        transfer_group_ids=(0,), kv_cache_groups=(object(),)
    )
    worker._recving_transfers = defaultdict(list)
    worker._generic_direct_receive_requests = set()
    worker._ephemeral_direct_dlists = NixlEphemeralDlistTracker(wrapper)
    worker._direct_read_notifications = {}
    worker.nixl_wrapper = wrapper
    worker.nixl_memory_type = "VRAM"
    meta = ReqMeta(
        local_block_ids=([20, 21],),
        local_physical_block_ids=([20, 21],),
        tp_size=1,
        local_num_computed_blocks=(0,),
        remote=RemoteMeta(
            block_ids=([10, 11],),
            host="prefill-host",
            port=8000,
            engine_id="prefill-engine",
            request_id="prefill-request",
            transfer_id="attempt-1",
        ),
        remote_num_tokens=8,
    )

    worker._read_blocks_for_req_direct("decode-request", meta)
    worker._refill_direct_read_batch_windows()

    assert wrapper.started == [102, 105]
    assert worker._recving_transfers["decode-request"] == wrapper.started
    assert worker._generic_direct_receive_requests == {"decode-request"}
    assert wrapper.notifications == []

    worker._on_receive_requests_terminal({"decode-request"}, set())

    assert len(wrapper.notifications) == 1
    agent, payload = wrapper.notifications[0]
    assert agent == source.rank_placement.worker_incarnation
    envelope = NixlDirectCompletionEnvelope.decode(payload)
    assert envelope.notification.request_id == "prefill-request"
    assert envelope.notification.transfer_id == "attempt-1"
    assert envelope.notification.expected_participant_count == 1
    assert envelope.notification.plan_digest.startswith("sha256:")
    assert envelope.notification.sender_worker_id == "decode-0"
    assert [
        (participant.worker_id, participant.worker_incarnation)
        for participant in envelope.expected_participants
    ] == [("decode-0", "decode-0-boot")]


def test_pull_worker_fans_out_exact_failed_completion_to_every_producer():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    original = _completion_envelope(sender, (sender,))
    wrapper = _FakeNixlWrapper()
    worker = _terminal_notification_worker(
        original.encode(), ("producer-0", "producer-1"), wrapper
    )

    worker._on_receive_requests_terminal(set(), {"decode-request"})

    assert [agent for agent, _ in wrapper.notifications] == [
        "producer-0",
        "producer-1",
    ]
    for _, payload in wrapper.notifications:
        failure = NixlDirectCompletionEnvelope.decode(payload)
        assert failure.notification.status is CompletionStatus.FAILED
        assert (
            replace(failure.notification, status=CompletionStatus.COMPLETE)
            == original.notification
        )
        assert failure.expected_participants == original.expected_participants
    assert worker._direct_read_notifications == {}


def test_pull_worker_failed_completion_send_error_does_not_stop_fanout():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    original = _completion_envelope(sender, (sender,))
    wrapper = MagicMock()
    wrapper.send_notif.side_effect = [RuntimeError("producer-0 unavailable"), None]
    worker = _terminal_notification_worker(
        original.encode(), ("producer-0", "producer-1"), wrapper
    )

    worker._on_receive_requests_terminal(set(), {"decode-request"})

    assert [call.args[0] for call in wrapper.send_notif.call_args_list] == [
        "producer-0",
        "producer-1",
    ]
    second_payload = wrapper.send_notif.call_args_list[1].kwargs["notif_msg"]
    assert (
        NixlDirectCompletionEnvelope.decode(second_payload).notification.status
        is CompletionStatus.FAILED
    )
    worker._log_failure.assert_called_once()
    worker.xfer_stats.record_failed_notification.assert_called_once_with()


def test_pull_worker_does_not_invent_failure_completion_without_envelope():
    wrapper = MagicMock()
    worker = _terminal_notification_worker(
        _completion_envelope(
            WorkerIdentity("decode-0", "decode-0-agent"),
            (WorkerIdentity("decode-0", "decode-0-agent"),),
        ).encode(),
        ("producer-0",),
        wrapper,
    )
    worker._direct_read_notifications.clear()

    worker._on_receive_requests_terminal(set(), {"setup-failed"})

    wrapper.send_notif.assert_not_called()
    worker._log_failure.assert_not_called()


def test_bridge_projects_scheduler_prefix_counts_to_transfer_groups():
    assert select_nixl_destination_prefix_blocks(
        (3, 9, 5), transfer_group_ids=(0, 2), total_group_count=3
    ) == (3, 5)
    assert select_nixl_destination_prefix_blocks(
        (3, 5), transfer_group_ids=(0, 2), total_group_count=3
    ) == (3, 5)
    assert select_nixl_destination_prefix_blocks(
        (), transfer_group_ids=(0, 2), total_group_count=3
    ) == (0, 0)


def test_read_plan_digest_is_stable_and_binds_exact_ranges():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)
    request = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
        source_physical_pages_per_logical=1,
        destination_physical_pages_per_logical=1,
    )

    digest = nixl_read_request_plan_digest(request)
    assert digest == nixl_read_request_plan_digest(request)
    assert digest.startswith("sha256:")
    changed = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=3,
    )
    assert nixl_read_request_plan_digest(changed) != digest

    differently_windowed = replace(
        request,
        max_pages_per_window=request.max_pages_per_window + 1,
        max_segments_per_batch=1,
    )
    assert nixl_read_request_plan_digest(differently_windowed) == digest
    different_blocks = replace(request, source_block_ids=((11,),))
    assert nixl_read_request_plan_digest(different_blocks) != digest
    source_layer = source.rank_placement.mappings[0]
    changed_mapping = replace(
        source_layer.mapping,
        num_writers=2,
        writer_index=0,
    )
    changed_source = replace(
        source,
        rank_placement=replace(
            source.rank_placement,
            mappings=(replace(source_layer, mapping=changed_mapping),),
        ),
    )
    changed_mapping_request = build_nixl_read_request_plan(
        source_workers=(changed_source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )
    assert nixl_read_request_plan_digest(changed_mapping_request) != digest

    source_template = source.page_registration_templates[0]
    changed_registration_source = replace(
        source,
        page_registration_templates=(
            replace(source_template, base_address=source_template.base_address + 4096),
        ),
    )
    changed_registration_request = replace(
        request,
        source_workers=(changed_registration_source,),
    )
    assert nixl_read_request_plan_digest(changed_registration_request) != digest


@pytest.mark.parametrize(
    ("block_ids", "physical_pages", "message"),
    [
        ((tuple(range(65))), 1, "requests 65 physical pages"),
        (((32,),), 2, "expands to physical page 65"),
    ],
)
def test_bridge_rejects_blocks_outside_registered_page_capacity(
    block_ids, physical_pages: int, message: str
):
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)
    source_groups = (
        (block_ids,) if block_ids and isinstance(block_ids[0], int) else block_ids
    )

    with pytest.raises(ValueError, match=message):
        build_nixl_read_request_plan(
            source_workers=(source,),
            destination_workers=(destination,),
            source_block_ids=source_groups,
            destination_block_ids=([0],),
            destination_prefix_blocks=(0,),
            remote_num_tokens=4,
            source_physical_pages_per_logical=physical_pages,
        )


def test_bridge_rejects_nonempty_destination_after_remote_token_end():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)

    with pytest.raises(
        ValueError,
        match="non-empty destination KV group .* starts at token 8.*only 4 tokens",
    ):
        build_nixl_read_request_plan(
            source_workers=(source,),
            destination_workers=(destination,),
            source_block_ids=([10, 11],),
            destination_block_ids=([20],),
            destination_prefix_blocks=(2,),
            remote_num_tokens=4,
        )


def test_bridge_retains_zero_valid_notify_only_range_for_empty_destination():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)

    request = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([],),
        destination_prefix_blocks=(2,),
        remote_num_tokens=4,
    )

    assert request.ranges == (KVRange("decoder-mla", 8, 0, 0),)
    assert tuple(iter_nixl_read_plan_windows(request)) == ()


def test_bridge_rejects_insufficient_source_before_window_planning():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge."
        "plan_request_transfer_layer"
    ) as planner:
        with pytest.raises(
            ValueError,
            match=(
                "source KV group .* does not cover requested canonical tokens "
                r"\[0, 12\); available coverage is \[0, 8\)"
            ),
        ):
            build_nixl_read_request_plan(
                source_workers=(source,),
                destination_workers=(destination,),
                source_block_ids=([10, 11],),
                destination_block_ids=([20, 21, 22],),
                destination_prefix_blocks=(0,),
                remote_num_tokens=12,
                max_pages_per_window=1,
            )
        planner.assert_not_called()


def test_bridge_source_capacity_accounts_for_physical_page_expansion():
    source = _placement("prefill-engine", "prefill-0", page_span=2)
    destination = _placement("decode", "decode-0", page_span=4)

    with pytest.raises(
        ValueError,
        match=(
            "source KV group .* does not cover requested canonical tokens "
            r"\[0, 5\); available coverage is \[0, 4\)"
        ),
    ):
        build_nixl_read_request_plan(
            source_workers=(source,),
            destination_workers=(destination,),
            source_block_ids=([10],),
            destination_block_ids=([20, 21],),
            destination_prefix_blocks=(0,),
            remote_num_tokens=5,
            source_physical_pages_per_logical=2,
        )


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("deployment", "wrong source deployment"),
        ("physical_geometry", "stale allocator page geometry"),
        ("incarnation", "stale source identity"),
        ("format", "stale source identity"),
        ("topology", "stale source identity"),
        ("agent_ranks", "agent bindings do not match"),
    ],
)
def test_prepare_rejects_stale_remote_index_before_nixl_preparation(
    mismatch: str, message: str
):
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)
    request = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )
    remote = index_remote_nixl_placements(
        {(0, 0): _agent("prefill-engine", source)},
        {(0, 0): source.rank_placement.worker_incarnation},
    )

    if mismatch == "deployment":
        remote = replace(remote, engine_id="another-prefill")
    elif mismatch == "physical_geometry":
        remote = replace(remote, physical_pages_per_logical=2)
    elif mismatch == "incarnation":
        stale_source = replace(
            source,
            rank_placement=replace(
                source.rank_placement, worker_incarnation="previous-boot"
            ),
        )
        remote = replace(remote, workers=(stale_source,))
    elif mismatch == "format":
        stale_format = replace(source.format_manifest, model_fingerprint="other-model")
        stale_source = replace(
            source,
            format_manifest=stale_format,
            rank_placement=replace(
                source.rank_placement,
                format_manifest_fingerprint=stale_format.fingerprint(),
            ),
        )
        remote = replace(remote, workers=(stale_source,))
    elif mismatch == "topology":
        stale_source = replace(
            source,
            rank_placement=replace(source.rank_placement, topology_generation=6),
        )
        remote = replace(remote, workers=(stale_source,))
    else:
        assert mismatch == "agent_ranks"
        remote = replace(
            remote,
            agent_names=((1, source.rank_placement.worker_incarnation),),
        )

    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)
    batches = iter_prepare_nixl_read_request(
        request,
        remote,
        nixl_wrapper=wrapper,
        tracker=tracker,
        local_transfer_rank=0,
        memory_type="VRAM",
    )

    with pytest.raises(ValueError, match=message):
        next(batches)
    assert wrapper.dlists == []
    assert tracker.pending_count == 0


def test_runtime_planning_yields_first_batch_from_one_bounded_page_window():
    source = _placement("prefill-engine", "prefill-0", page_span=4, max_segments=None)
    destination = _placement("decode", "decode-0", page_span=4, max_segments=None)
    block_ids = tuple(range(64))
    with patch(
        "vllm.distributed.kv_transfer.request_planner.compose_page_placements",
        side_effect=AssertionError("request build must not compose pages"),
    ):
        request = build_nixl_read_request_plan(
            source_workers=(source,),
            destination_workers=(destination,),
            source_block_ids=(block_ids,),
            destination_block_ids=(block_ids,),
            destination_prefix_blocks=(0,),
            remote_num_tokens=256,
            max_pages_per_window=2,
        )
    remote = index_remote_nixl_placements(
        {(0, 0): _agent("prefill-engine", source)},
        {(0, 0): source.rank_placement.worker_incarnation},
    )
    wrapper = _FakeNixlWrapper()
    tracker = NixlEphemeralDlistTracker(wrapper)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge."
        "iter_nixl_request_layer_direct_batches",
        wraps=iter_nixl_request_layer_direct_batches,
    ) as planner:
        batches = iter_prepare_nixl_read_request(
            request,
            remote,
            nixl_wrapper=wrapper,
            tracker=tracker,
            local_transfer_rank=0,
            memory_type="VRAM",
        )
        first = next(batches)
        assert planner.call_count == 1
        assert first.descriptor_batch.segment_count == 2
        batches.close()

    assert tracker.release(first.transfer_handle)
    windows = iter_nixl_read_plan_windows(request)
    first_window = next(windows)
    assert len(first_window.source_allocations) == 2
    assert len(first_window.destination_allocations) == 2
    assert first_window.kv_range == KVRange("decoder-mla", 0, 8, 8)


def test_pull_worker_deduplicates_exact_generic_completion_participants():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination_0 = _placement(
        "decode", "decode-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    destination_1 = _placement(
        "decode", "decode-1", page_span=4, rank=1, tp_size=2, tp_rank=1
    )
    request = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination_0, destination_1),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
        source_physical_pages_per_logical=1,
        destination_physical_pages_per_logical=1,
    )
    digest = nixl_read_request_plan_digest(request)

    participants = request.destination_participants

    def payload(
        sender: WorkerIdentity,
        transfer_id: str = "attempt-1",
        *,
        expected_participants: tuple[WorkerIdentity, ...] = participants,
    ) -> bytes:
        return NixlDirectCompletionEnvelope(
            notification=TransferCompletionNotification(
                version=KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
                request_id="prefill-request",
                deployment_id=request.destination_route.deployment_id,
                topology_generation=request.destination_route.topology_generation,
                transfer_id=transfer_id,
                plan_digest=digest,
                sender_worker_id=sender.worker_id,
                sender_worker_incarnation=sender.worker_incarnation,
                expected_participant_count=len(expected_participants),
                status=CompletionStatus.COMPLETE,
            ),
            expected_participants=expected_participants,
        ).encode()

    sender_0, sender_1 = participants
    worker = object.__new__(NixlPullConnectorWorker)
    worker._reqs_to_send = {"prefill-request": float("inf")}
    worker._reqs_to_process = {"prefill-request"}
    worker._expected_direct_transfer_ids = {"prefill-request": "attempt-1"}
    worker._expected_direct_participant_counts = {"prefill-request": 2}
    worker._expected_direct_participants = {"prefill-request": participants}
    worker._direct_completion_trackers = {}
    worker._direct_completion_participant_digests = {}
    worker._direct_completion_sender_bindings = {}
    worker.consumer_notification_counts_by_req = defaultdict(int)
    worker.expected_consumer_notifications_by_req = {}
    notified: set[str] = set()

    worker._handle_direct_completion(
        payload(sender_0, "old-attempt"),
        notified,
        sender_agent=sender_0.worker_incarnation,
    )
    assert notified == set()
    assert worker._direct_completion_trackers == {}

    first = payload(sender_0)
    worker._handle_direct_completion(
        first, notified, sender_agent=sender_0.worker_incarnation
    )
    worker._handle_direct_completion(
        first, notified, sender_agent=sender_0.worker_incarnation
    )
    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send

    worker._handle_direct_completion(
        payload(sender_1), notified, sender_agent=sender_1.worker_incarnation
    )
    assert notified == {"prefill-request"}
    assert "prefill-request" not in worker._reqs_to_send
    assert "prefill-request" not in worker._reqs_to_process
    assert worker._expected_direct_transfer_ids == {}
    assert worker._expected_direct_participant_counts == {}
    assert worker._direct_completion_sender_bindings == {}


def _completion_test_worker(
    *,
    expected_count: int | None = 2,
    expected_participants: tuple[WorkerIdentity, ...] | None = None,
):
    worker = object.__new__(NixlPullConnectorWorker)
    worker._reqs_to_send = {"prefill-request": float("inf")}
    worker._reqs_to_process = {"prefill-request"}
    worker._expected_direct_transfer_ids = {"prefill-request": "attempt-1"}
    worker._expected_direct_participant_counts = (
        {"prefill-request": expected_count} if expected_count is not None else {}
    )
    if expected_participants is None and expected_count is not None:
        expected_participants = tuple(
            WorkerIdentity(f"decode-{index}", f"decode-{index}-agent")
            for index in range(expected_count)
        )
    worker._expected_direct_participants = (
        {"prefill-request": expected_participants}
        if expected_participants is not None
        else {}
    )
    worker._direct_completion_trackers = {}
    worker._direct_completion_participant_digests = {}
    worker._direct_completion_sender_bindings = {}
    worker.consumer_notification_counts_by_req = defaultdict(int)
    worker.expected_consumer_notifications_by_req = {}
    return worker


def _arm_completion_test_worker(*, expected_count: int | None):
    worker = object.__new__(NixlPullConnectorWorker)
    worker.pcp_rank = 0
    worker._local_placement_metadata = object()
    worker._ready_requests = SimpleNamespace(empty=lambda: True)
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker._expected_direct_transfer_ids = {}
    worker._expected_direct_participant_counts = {}
    worker._expected_direct_participants = {}
    worker._direct_completion_trackers = {}
    worker._direct_completion_participant_digests = {}
    worker._direct_completion_sender_bindings = {}
    worker.consumer_notification_counts_by_req = defaultdict(int)
    worker.expected_consumer_notifications_by_req = {}

    metadata = NixlConnectorMetadata()
    metadata.reqs_in_batch = {"prefill-request"}
    metadata.reqs_to_send = {"prefill-request": float("inf")}
    metadata.reqs_to_send_transfer_ids = {"prefill-request": "attempt-1"}
    if expected_count is not None:
        participants = tuple(
            WorkerIdentity(f"decode-{index}", f"decode-{index}-agent")
            for index in range(expected_count)
        )
        metadata.reqs_to_send_expected_participant_counts = {
            "prefill-request": expected_count
        }
        metadata.reqs_to_send_expected_participants = {"prefill-request": participants}
    worker.start_load_kv(metadata)
    return worker


def _completion_envelope(
    sender: WorkerIdentity,
    participants: tuple[WorkerIdentity, ...],
) -> NixlDirectCompletionEnvelope:
    return NixlDirectCompletionEnvelope(
        notification=TransferCompletionNotification(
            version=KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
            request_id="prefill-request",
            deployment_id="decode",
            topology_generation=5,
            transfer_id="attempt-1",
            plan_digest="sha256:plan",
            sender_worker_id=sender.worker_id,
            sender_worker_incarnation=sender.worker_incarnation,
            expected_participant_count=len(participants),
            status=CompletionStatus.COMPLETE,
        ),
        expected_participants=participants,
    )


def _completion_payload(
    sender: WorkerIdentity,
    participants: tuple[WorkerIdentity, ...],
) -> bytes:
    return _completion_envelope(sender, participants).encode()


def test_failed_direct_completion_retains_producer_pages_until_lease_expiry():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    worker = _completion_test_worker(
        expected_count=1,
        expected_participants=(sender,),
    )
    envelope = _completion_envelope(sender, (sender,))
    failed_payload = replace(
        envelope,
        notification=replace(
            envelope.notification,
            status=CompletionStatus.FAILED,
        ),
    ).encode()
    notified: set[str] = set()

    worker._handle_direct_completion(
        failed_payload,
        notified,
        sender_agent=sender.worker_incarnation,
    )

    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send
    assert "prefill-request" in worker._reqs_to_process
    assert worker._direct_completion_trackers["prefill-request"].progress.failed


def test_direct_completion_wire_size_accepts_boundary_and_rejects_one_more_byte():
    assert MAX_NIXL_COMPLETION_BYTES == 1024 * 1024
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    envelope = _completion_envelope(sender, (sender,))
    base_payload = envelope.encode()
    padding = MAX_NIXL_COMPLETION_BYTES - len(base_payload)
    assert padding > 0
    exact_envelope = replace(
        envelope,
        notification=replace(
            envelope.notification,
            request_id=envelope.notification.request_id + "x" * padding,
        ),
    )

    exact_payload = exact_envelope.encode()
    assert len(exact_payload) == MAX_NIXL_COMPLETION_BYTES
    assert NixlDirectCompletionEnvelope.decode(exact_payload) == exact_envelope

    oversized_envelope = replace(
        exact_envelope,
        notification=replace(
            exact_envelope.notification,
            request_id=exact_envelope.notification.request_id + "x",
        ),
    )
    with pytest.raises(ValueError, match="exceeds the maximum wire size"):
        oversized_envelope.encode()
    with pytest.raises(ValueError, match="exceeds the maximum wire size"):
        NixlDirectCompletionEnvelope.decode(exact_payload + b"x")


def test_direct_completion_decode_checks_count_before_participant_expansion():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    envelope = _completion_envelope(sender, (sender,))
    body = {
        "notification": replace(
            envelope.notification, expected_participant_count=2
        ).to_dict(),
        # This invalid identity shape must not be inspected after cardinality
        # already proves the envelope inconsistent.
        "expected_participants": [{}],
    }
    payload = NIXL_DIRECT_COMPLETION_PREFIX + json.dumps(body).encode()

    with pytest.raises(ValueError, match="does not match the advertised count"):
        NixlDirectCompletionEnvelope.decode(payload)


def test_direct_completion_decode_rejects_oversized_participant_array():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    envelope = _completion_envelope(sender, (sender,))
    body = {
        "notification": envelope.notification.to_dict(),
        "expected_participants": [{}] * (MAX_TRANSFER_PARTICIPANTS + 1),
    }
    payload = (
        NIXL_DIRECT_COMPLETION_PREFIX + json.dumps(body, separators=(",", ":")).encode()
    )
    assert len(payload) < MAX_NIXL_COMPLETION_BYTES

    with pytest.raises(ValueError, match="at most 4096 participants"):
        NixlDirectCompletionEnvelope.decode(payload)


def test_direct_completion_wire_accepts_participant_count_boundary():
    participants = tuple(
        WorkerIdentity(f"decode-{index}", f"boot-{index}")
        for index in range(MAX_TRANSFER_PARTICIPANTS)
    )
    sender = participants[0]
    envelope = NixlDirectCompletionEnvelope(
        notification=replace(
            _completion_envelope(sender, (sender,)).notification,
            expected_participant_count=MAX_TRANSFER_PARTICIPANTS,
        ),
        expected_participants=participants,
    )

    payload = envelope.encode()
    assert len(payload) < MAX_NIXL_COMPLETION_BYTES
    assert NixlDirectCompletionEnvelope.decode(payload) == envelope


def test_pull_worker_requires_producer_owned_completion_quorum():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    worker = _completion_test_worker(expected_count=None)

    notified: set[str] = set()
    worker._handle_direct_completion(
        _completion_payload(sender, (sender,)),
        notified,
        sender_agent=sender.worker_incarnation,
    )

    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send
    assert worker._direct_completion_trackers == {}
    assert worker._direct_completion_sender_bindings == {}


def test_pull_worker_rejects_singleton_envelope_against_producer_quorum():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    expected_roster = (
        sender,
        WorkerIdentity("decode-1", "decode-1-agent"),
    )
    worker = _completion_test_worker(
        expected_count=2, expected_participants=expected_roster
    )

    notified: set[str] = set()
    worker._handle_direct_completion(
        _completion_payload(sender, (sender,)),
        notified,
        sender_agent=sender.worker_incarnation,
    )

    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send
    assert worker._direct_completion_trackers == {}
    assert worker._direct_completion_sender_bindings == {}


def test_pull_worker_rejects_one_transport_sender_claiming_two_workers():
    sender_0 = WorkerIdentity("decode-0", "shared-agent")
    sender_1 = WorkerIdentity("decode-1", "shared-agent")
    participants = (sender_0, sender_1)
    worker = _completion_test_worker(
        expected_count=2,
        expected_participants=(sender_0, sender_1),
    )
    worker.transfer_topo = object()
    worker.nixl_wrapper = SimpleNamespace(
        get_new_notifs=lambda: {
            "shared-agent": [
                _completion_payload(sender_0, participants),
                _completion_payload(sender_1, participants),
            ]
        }
    )

    notified = worker._get_new_notifs()

    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send
    tracker = worker._direct_completion_trackers["prefill-request"]
    assert tracker.progress.received_participant_count == 1
    assert worker._direct_completion_sender_bindings == {
        "prefill-request": {"shared-agent": sender_0}
    }


def test_pull_worker_rejects_transport_sender_incarnation_mismatch():
    sender_0 = WorkerIdentity("decode-0", "decode-0-agent")
    sender_1 = WorkerIdentity("decode-1", "decode-1-agent")
    worker = _completion_test_worker(expected_count=2)
    worker.transfer_topo = object()
    worker.nixl_wrapper = SimpleNamespace(
        get_new_notifs=lambda: {
            "forged-agent": [_completion_payload(sender_0, (sender_0, sender_1))]
        }
    )

    notified = worker._get_new_notifs()

    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send
    assert worker._direct_completion_trackers == {}
    assert worker._direct_completion_sender_bindings == {}


def test_pull_worker_without_contract_stays_legacy_on_asymmetric_negotiation():
    sender = WorkerIdentity("decode-0", "decode-0-agent")
    worker = _arm_completion_test_worker(expected_count=None)
    worker.transfer_topo = object()
    worker.nixl_wrapper = SimpleNamespace(
        get_new_notifs=lambda: {
            sender.worker_incarnation: [
                _completion_payload(sender, (sender,)),
                b"prefill-request:1",
            ]
        }
    )

    assert worker._expected_direct_transfer_ids == {}
    assert worker._get_new_notifs() == {"prefill-request"}
    assert "prefill-request" not in worker._reqs_to_send


def test_pull_worker_with_contract_rejects_legacy_peer_completion():
    worker = _arm_completion_test_worker(expected_count=2)
    worker.transfer_topo = object()
    worker.nixl_wrapper = SimpleNamespace(
        get_new_notifs=lambda: {"legacy-agent": [b"prefill-request:1"]}
    )

    assert worker._expected_direct_transfer_ids == {"prefill-request": "attempt-1"}
    assert worker._get_new_notifs() == set()
    assert "prefill-request" in worker._reqs_to_send


def _init_pull_scheduler_with_extra_config(extra_config: dict[str, object]):
    kv_transfer_config = SimpleNamespace(
        get_from_extra_config=lambda key, default: extra_config.get(key, default)
    )
    vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)
    with patch.object(NixlBaseConnectorScheduler, "__init__", return_value=None):
        return NixlPullConnectorScheduler(vllm_config, "prefill-engine", object())


@pytest.mark.parametrize("configured_count", [None, 1, MAX_TRANSFER_PARTICIPANTS])
def test_pull_scheduler_initializes_generic_completion_participant_count(
    configured_count: int | None,
):
    extra_config: dict[str, object] = (
        {}
        if configured_count is None
        else {
            "enable_generic_placement": True,
            "generic_completion_participant_count": configured_count,
        }
    )

    scheduler = _init_pull_scheduler_with_extra_config(extra_config)

    assert scheduler.generic_completion_participant_count == configured_count


def test_pull_scheduler_rejects_generic_completion_without_generic_placement():
    with pytest.raises(ValueError, match="requires enable_generic_placement=true"):
        _init_pull_scheduler_with_extra_config(
            {"generic_completion_participant_count": 2}
        )


@pytest.mark.parametrize(
    "configured_count",
    [False, True, 0, -1, MAX_TRANSFER_PARTICIPANTS + 1, "2", 2.0],
)
def test_pull_scheduler_rejects_invalid_generic_completion_participant_count(
    configured_count: object,
):
    with pytest.raises(
        ValueError, match="generic_completion_participant_count.*between 1 and 4096"
    ):
        _init_pull_scheduler_with_extra_config(
            {"generic_completion_participant_count": configured_count}
        )


def _completion_contract_scheduler(
    *, configured_count: int | None = None
) -> NixlPullConnectorScheduler:
    scheduler = object.__new__(NixlPullConnectorScheduler)
    scheduler.is_bidirectional_kv_xfer_enabled = False
    scheduler._heartbeat_req_engine = {}
    scheduler._heartbeat_by_engine = {}
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_save = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_need_send_transfer_ids = {}
    scheduler._reqs_need_send_expected_participant_counts = {}
    scheduler._reqs_need_send_expected_participants = {}
    scheduler._reqs_in_batch = set()
    scheduler._reqs_not_processed = set()
    scheduler._kv_lease_duration = 30
    scheduler._endpoint_incarnation = "prefill-endpoint-boot"
    scheduler.generic_completion_participant_count = configured_count
    scheduler.generic_completion_participants = (
        _completion_participants(configured_count)
        if configured_count is not None
        else ()
    )
    scheduler.use_host_buffer = False
    scheduler.engine_id = "prefill-engine"
    scheduler.side_channel_host = "prefill-host"
    scheduler.side_channel_port = 1234
    scheduler.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        )
    )
    scheduler.get_exchange_clipped_blocks = lambda block_ids: block_ids
    return scheduler


_MISSING_COMPLETION_COUNT = object()


def _completion_participants(count: int) -> tuple[WorkerIdentity, ...]:
    return tuple(
        WorkerIdentity(f"decode-{index}", f"decode-{index}-agent")
        for index in range(count)
    )


def _completion_contract_request(
    expected_count: object = _MISSING_COMPLETION_COUNT,
    *,
    include_roster: bool = True,
):
    from vllm.v1.request import RequestStatus

    kv_transfer_params: dict[str, object] = {"do_remote_decode": True}
    if expected_count is not _MISSING_COMPLETION_COUNT:
        kv_transfer_params["expected_kv_completion_participant_count"] = expected_count
        if (
            include_roster
            and isinstance(expected_count, int)
            and not isinstance(expected_count, bool)
            and 0 < expected_count <= MAX_TRANSFER_PARTICIPANTS
        ):
            kv_transfer_params["expected_kv_completion_participants"] = [
                {
                    "worker_id": participant.worker_id,
                    "worker_incarnation": participant.worker_incarnation,
                }
                for participant in _completion_participants(expected_count)
            ]
    return SimpleNamespace(
        request_id="prefill-request",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        kv_transfer_params=kv_transfer_params,
        num_computed_tokens=8,
    )


@pytest.mark.parametrize(
    ("expected_count", "accepted"),
    [
        (2, True),
        (MAX_TRANSFER_PARTICIPANTS, True),
        (MAX_TRANSFER_PARTICIPANTS + 1, False),
    ],
)
def test_pull_scheduler_bounds_producer_completion_contract(
    expected_count: int, accepted: bool
):
    scheduler = _completion_contract_scheduler()
    request = _completion_contract_request(expected_count)

    delay_free, remote_params = scheduler.request_finished(request, ([10],))

    assert delay_free
    assert remote_params is not None
    advertised_count = expected_count if accepted else None
    expected_counts = {"prefill-request": expected_count} if accepted else {}
    expected_rosters = (
        {"prefill-request": _completion_participants(expected_count)}
        if accepted
        else {}
    )
    assert remote_params["expected_kv_completion_participant_count"] == advertised_count
    assert remote_params["expected_kv_completion_participants"] == (
        tuple(
            {
                "worker_id": participant.worker_id,
                "worker_incarnation": participant.worker_incarnation,
            }
            for participant in _completion_participants(expected_count)
        )
        if accepted
        else None
    )
    assert scheduler._reqs_need_send_expected_participant_counts == expected_counts
    assert scheduler._reqs_need_send_expected_participants == expected_rosters
    connector_metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert connector_metadata.reqs_to_send_expected_participant_counts == (
        expected_counts
    )
    assert connector_metadata.reqs_to_send_expected_participants == expected_rosters
    assert scheduler._reqs_need_send_expected_participant_counts == {}
    assert scheduler._reqs_need_send_expected_participants == {}


def test_pull_scheduler_count_without_roster_does_not_authorize_completion():
    scheduler = _completion_contract_scheduler()
    request = _completion_contract_request(2, include_roster=False)

    delay_free, remote_params = scheduler.request_finished(request, ([10],))

    assert delay_free
    assert remote_params is not None
    assert remote_params["expected_kv_completion_participant_count"] is None
    assert remote_params["expected_kv_completion_participants"] is None
    assert scheduler._reqs_need_send_expected_participant_counts == {}
    assert scheduler._reqs_need_send_expected_participants == {}


def test_pull_scheduler_config_provisions_contract_without_router_field():
    scheduler = _completion_contract_scheduler(configured_count=3)
    request = _completion_contract_request()

    delay_free, remote_params = scheduler.request_finished(request, ([10],))

    assert delay_free
    assert remote_params is not None
    assert remote_params["expected_kv_completion_participant_count"] == 3
    assert scheduler._reqs_need_send_expected_participant_counts == {
        "prefill-request": 3
    }
    assert scheduler._reqs_need_send_expected_participants == {
        "prefill-request": _completion_participants(3)
    }
    connector_metadata = scheduler.build_connector_meta(SimpleNamespace())
    assert connector_metadata.reqs_to_send_expected_participant_counts == {
        "prefill-request": 3
    }
    assert connector_metadata.reqs_to_send_expected_participants == {
        "prefill-request": _completion_participants(3)
    }


def test_pull_scheduler_config_overrides_mismatched_router_contract():
    scheduler = _completion_contract_scheduler(configured_count=3)
    request = _completion_contract_request(2)

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl."
        "pull_scheduler.logger.warning"
    ) as warning:
        delay_free, remote_params = scheduler.request_finished(request, ([10],))

    assert delay_free
    assert remote_params is not None
    assert remote_params["expected_kv_completion_participant_count"] == 3
    assert scheduler._reqs_need_send_expected_participant_counts == {
        "prefill-request": 3
    }
    assert warning.call_count == 2
    assert all("overrides" in call.args[0] for call in warning.call_args_list)


def test_connector_metadata_carries_producer_contract_to_remote_meta():
    metadata = NixlConnectorMetadata()
    metadata.add_new_req_to_recv(
        request_id="decode-request",
        local_block_ids=([20],),
        kv_transfer_params={
            "remote_block_ids": ([10],),
            "remote_engine_id": "prefill-engine",
            "remote_request_id": "prefill-request",
            "remote_host": "prefill-host",
            "remote_port": 1234,
            "transfer_id": "attempt-1",
            "expected_kv_completion_participant_count": 2,
            "expected_kv_completion_participants": [
                {
                    "worker_id": participant.worker_id,
                    "worker_incarnation": participant.worker_incarnation,
                }
                for participant in _completion_participants(2)
            ],
        },
    )

    remote = metadata.reqs_to_recv["decode-request"].remote
    assert remote is not None
    assert remote.transfer_id == "attempt-1"
    assert remote.expected_completion_participant_count == 2
    assert remote.expected_completion_participants == _completion_participants(2)


def _contract_selection_worker(*, local_participant_count: int):
    worker = object.__new__(NixlPullConnectorWorker)
    worker.transfer_topo = object()
    worker._engine_last_active = {}
    worker._bidirectional_kv_xfer_enabled = False
    placements = tuple(
        _placement(
            "decode",
            f"decode-{index}",
            page_span=4,
            rank=index,
            tp_size=local_participant_count,
            tp_rank=index,
        )
        for index in range(local_participant_count)
    )
    worker._local_placement_metadata = placements[0]
    worker._remote_placement_indexes = {"prefill-engine": object()}
    worker._generic_only_remote_engines = set()
    worker._local_placement_workers = placements
    worker._recving_transfers = defaultdict(list)
    return worker


def _contract_selection_meta(*, expected_count: int | None) -> ReqMeta:
    return ReqMeta(
        local_block_ids=([20],),
        local_physical_block_ids=([20],),
        tp_size=1,
        remote=RemoteMeta(
            block_ids=([10],),
            host="prefill-host",
            port=1234,
            engine_id="prefill-engine",
            request_id="prefill-request",
            transfer_id="attempt-1",
            expected_completion_participant_count=expected_count,
            expected_completion_participants=(
                tuple(
                    WorkerIdentity(f"decode-{index}", f"decode-{index}-boot")
                    for index in range(expected_count)
                )
                if expected_count is not None
                else ()
            ),
        ),
    )


def test_pull_worker_selects_direct_only_for_matching_producer_contract():
    worker = _contract_selection_worker(local_participant_count=2)
    direct_calls = []
    worker._read_blocks_for_req_direct = lambda req_id, meta: direct_calls.append(
        (req_id, meta)
    )
    meta = _contract_selection_meta(expected_count=2)

    worker._read_blocks_for_req("decode-request", meta)

    assert direct_calls == [("decode-request", meta)]


def test_pull_worker_selects_legacy_when_producer_contract_is_absent():
    class LegacyPathSelected(Exception):
        pass

    class LegacyPlanSentinel:
        def __getitem__(self, engine_id):
            raise LegacyPathSelected(engine_id)

    worker = _contract_selection_worker(local_participant_count=2)
    worker.tp_mappings = LegacyPlanSentinel()
    worker._read_blocks_for_req_direct = lambda *_: pytest.fail(
        "generic direct path selected without a producer contract"
    )

    with pytest.raises(LegacyPathSelected, match="prefill-engine"):
        worker._read_blocks_for_req(
            "decode-request", _contract_selection_meta(expected_count=None)
        )


def test_pull_worker_without_legacy_dlist_requires_exact_generic_contract():
    worker = _contract_selection_worker(local_participant_count=2)
    worker._legacy_fast_path_available = False
    worker.tp_mappings = legacy_plans = MagicMock()
    failures = []
    worker._log_failure = lambda **fields: failures.append(("log", fields))
    worker._handle_failed_transfer = lambda *args, **kwargs: failures.append(
        ("fail", (args, kwargs))
    )

    worker._read_blocks_for_req(
        "decode-request", _contract_selection_meta(expected_count=None)
    )

    assert [kind for kind, _ in failures] == ["log", "fail"]
    assert failures[0][1]["failure_type"] == "legacy_descriptor_unavailable"
    legacy_plans.__getitem__.assert_not_called()


def test_pull_worker_fails_closed_on_mismatched_producer_contract():
    worker = _contract_selection_worker(local_participant_count=2)
    failures = []
    worker._log_failure = lambda **fields: failures.append(("log", fields))
    worker._handle_failed_transfer = lambda *args, **kwargs: failures.append(
        ("fail", (args, kwargs))
    )

    worker._read_blocks_for_req(
        "decode-request", _contract_selection_meta(expected_count=1)
    )

    assert [kind for kind, _ in failures] == ["log", "fail"]
    assert failures[0][1]["failure_type"] == "segmented_direct_contract_mismatch"


def test_pull_worker_explicit_contract_fails_closed_without_generic_placement():
    worker = _contract_selection_worker(local_participant_count=2)
    worker._local_placement_metadata = None
    failures = []
    worker._log_failure = lambda **fields: failures.append(("log", fields))
    worker._handle_failed_transfer = lambda *args, **kwargs: failures.append(
        ("fail", (args, kwargs))
    )
    worker._read_blocks_for_req_direct = lambda *_: pytest.fail(
        "generic path selected without placement metadata"
    )

    worker._read_blocks_for_req(
        "decode-request", _contract_selection_meta(expected_count=2)
    )

    assert [kind for kind, _ in failures] == ["log", "fail"]
    assert failures[0][1]["failure_type"] == "segmented_direct_contract_mismatch"


def test_pull_worker_never_falls_back_from_generic_only_handshake():
    worker = _contract_selection_worker(local_participant_count=2)
    worker._generic_only_remote_engines = {"prefill-engine"}
    failures = []
    worker._log_failure = lambda **fields: failures.append(("log", fields))
    worker._handle_failed_transfer = lambda *args, **kwargs: failures.append(
        ("fail", (args, kwargs))
    )
    worker._read_blocks_for_req_direct = lambda *_: pytest.fail(
        "generic path must require a producer completion contract"
    )

    worker._read_blocks_for_req(
        "decode-request", _contract_selection_meta(expected_count=None)
    )

    assert [kind for kind, _ in failures] == ["log", "fail"]
    assert failures[0][1]["failure_type"] == "segmented_direct_contract_missing"


def test_bridge_rejects_destination_capacity_shorter_than_remote_tail():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=8)

    with pytest.raises(ValueError, match="valid_token_count must not exceed"):
        build_nixl_read_request_plan(
            source_workers=(source,),
            destination_workers=(destination,),
            source_block_ids=([10, 11, 12, 13, 14],),
            destination_block_ids=([20],),
            destination_prefix_blocks=(1,),
            remote_num_tokens=17,
        )


def test_bridge_preserves_full_prefix_hit_as_zero_byte_plan():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=8)

    result = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10, 11],),
        destination_block_ids=([],),
        destination_prefix_blocks=(1,),
        remote_num_tokens=7,
    )

    assert result.ranges[0].valid_token_count == 0
    assert tuple(iter_nixl_read_plan_windows(result)) == ()
    assert result.planning_context.source_expected_participant_count == 1


def test_bridge_preserves_abort_notify_without_prefix_metadata():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=8)

    result = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10, 11],),
        destination_block_ids=([],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=7,
    )

    assert result.ranges[0].valid_token_count == 0
    assert tuple(iter_nixl_read_plan_windows(result)) == ()


def test_bridge_expands_logical_blocks_into_physical_pages():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=4)

    result = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=8,
        source_physical_pages_per_logical=2,
        destination_physical_pages_per_logical=2,
    )

    _, source_allocations, destination_allocations = materialize_nixl_read_request_plan(
        result
    )
    assert {
        (page.local_page_id, page.canonical_page_index, page.first_token)
        for page in source_allocations
    } == {(20, 0, 0), (21, 1, 4)}
    assert {
        (page.local_page_id, page.canonical_page_index, page.first_token)
        for page in destination_allocations
    } == {(40, 0, 0), (41, 1, 4)}
    assert result.ranges[0] == KVRange("decoder-mla", 0, 8, 8)


def test_bridge_composes_asymmetric_physical_ratios_after_prefix():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    destination = _placement("decode", "decode-0", page_span=2)

    result = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([0, 1],),
        destination_block_ids=([10],),
        destination_prefix_blocks=(1,),
        remote_num_tokens=16,
        source_physical_pages_per_logical=2,
        destination_physical_pages_per_logical=4,
    )

    transfer_plan, source_allocations, destination_allocations = (
        materialize_nixl_read_request_plan(result)
    )
    assert [page.local_page_id for page in source_allocations] == [
        0,
        1,
        2,
        3,
    ]
    assert [page.local_page_id for page in destination_allocations] == [
        40,
        41,
        42,
        43,
    ]
    assert [page.canonical_page_index for page in destination_allocations] == [
        4,
        5,
        6,
        7,
    ]
    assert [page.first_token for page in destination_allocations] == [
        8,
        10,
        12,
        14,
    ]
    assert result.ranges[0] == KVRange("decoder-mla", 8, 8, 8)
    assert {
        (run.source_page_id, run.destination_page_id)
        for run in transfer_plan.layers[0].runs
    } == {(2, 40), (2, 41), (3, 42), (3, 43)}


def test_bridge_without_scatter_gather_keeps_one_segment_direct_batches():
    source = _placement("prefill-engine", "prefill-0", page_span=4)
    source = replace(
        source,
        capabilities=replace(source.capabilities, scatter_gather=False),
    )
    destination = _placement("decode", "decode-0", page_span=4)

    result = build_nixl_read_request_plan(
        source_workers=(source,),
        destination_workers=(destination,),
        source_block_ids=([10],),
        destination_block_ids=([20],),
        destination_prefix_blocks=(0,),
        remote_num_tokens=4,
    )

    assert result.max_segments_per_batch == 1
    assert tuple(iter_nixl_read_plan_windows(result))[0].layer_plan.runs


def test_generic_registration_accepts_different_physical_page_ratios():
    local = _placement("decode", "decode-0", page_span=2)
    remote = _agent(
        "prefill-engine",
        _placement("prefill-engine", "prefill-0", page_span=4),
    )
    remote.physical_blocks_per_logical_kv_block = 2
    worker = object.__new__(NixlPullConnectorWorker)
    worker._local_placement_metadata = local
    worker._physical_blocks_per_logical_kv_block = 4

    assert worker._generic_registration_available(remote)


def test_remote_index_uses_advertised_transfer_rank():
    first = _placement(
        "prefill-engine", "prefill-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    second = _placement(
        "prefill-engine", "prefill-1", page_span=4, rank=1, tp_size=2, tp_rank=1
    )
    metadata = {
        (0, 1): _agent("prefill-engine", second),
        (0, 0): _agent("prefill-engine", first),
    }
    agents = {
        coordinate: placement.rank_placement.worker_incarnation
        for coordinate, placement in {
            (0, 0): first,
            (0, 1): second,
        }.items()
    }

    index = index_remote_nixl_placements(metadata, agents)

    assert index.engine_id == "prefill-engine"
    assert [worker.rank_placement.worker_id for worker in index.workers] == [
        "prefill-0",
        "prefill-1",
    ]
    assert index.agent_names_by_rank == {
        0: first.rank_placement.worker_incarnation,
        1: second.rank_placement.worker_incarnation,
    }


def test_remote_index_rejects_coordinate_aliases_and_missing_placement():
    placement = _placement("prefill-engine", "prefill-0", page_span=4)
    mismatched = replace(
        placement,
        rank_placement=replace(placement.rank_placement, tp_size=2, tp_rank=1),
    )
    with pytest.raises(ValueError, match="does not match handshake coordinate"):
        index_remote_nixl_placements(
            {(0, 0): _agent("prefill-engine", mismatched)},
            {(0, 0): "agent-0"},
        )

    without_placement = _agent("prefill-engine", placement)
    without_placement.placement_metadata = None
    with pytest.raises(ValueError, match="did not advertise placement"):
        index_remote_nixl_placements({(0, 0): without_placement}, {(0, 0): "agent-0"})

    incomplete = _placement(
        "prefill-engine", "prefill-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    with pytest.raises(ValueError, match="handshake is incomplete"):
        index_remote_nixl_placements(
            {(0, 0): _agent("prefill-engine", incomplete)},
            {(0, 0): incomplete.rank_placement.worker_incarnation},
        )


def test_remote_index_addresses_complete_pcp_tp_cohort_without_aliasing():
    placements = {
        (0, pcp_rank * 2 + tp_rank): _placement(
            "prefill-engine",
            f"prefill-pcp{pcp_rank}-tp{tp_rank}",
            page_span=4,
            rank=pcp_rank * 2 + tp_rank,
            tp_size=2,
            tp_rank=tp_rank,
            pcp_size=2,
            pcp_rank=pcp_rank,
        )
        for pcp_rank in range(2)
        for tp_rank in range(2)
    }
    metadata = {
        coordinate: _agent("prefill-engine", placement)
        for coordinate, placement in placements.items()
    }
    agents = {
        coordinate: placement.rank_placement.worker_incarnation
        for coordinate, placement in placements.items()
    }

    index = index_remote_nixl_placements(metadata, agents)

    assert [worker.rank_placement.rank for worker in index.workers] == list(range(4))
    assert index.agent_names_by_rank == {
        placement.rank_placement.rank: placement.rank_placement.worker_incarnation
        for placement in placements.values()
    }

    incomplete_metadata = dict(metadata)
    incomplete_agents = dict(agents)
    del incomplete_metadata[(0, 3)]
    del incomplete_agents[(0, 3)]
    with pytest.raises(ValueError, match="handshake is incomplete"):
        index_remote_nixl_placements(
            incomplete_metadata,
            incomplete_agents,
        )


def test_remote_index_binds_registration_regions_independent_of_order():
    placement = _two_layer_placement()
    metadata = _agent("prefill-engine", placement)
    metadata.kv_caches_base_addr.reverse()
    metadata.block_lens.reverse()
    metadata.block_strides.reverse()

    index = index_remote_nixl_placements(
        {(0, 0): metadata},
        {(0, 0): placement.rank_placement.worker_incarnation},
    )

    assert index.workers == (placement,)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("kv_caches_base_addr", [1001]),
        ("block_lens", [3]),
        ("block_strides", [8]),
        ("num_blocks", 63),
        ("device_id", 1),
    ),
)
def test_remote_index_rejects_registration_region_mismatch(field, wrong_value):
    placement = _placement("prefill-engine", "prefill-0", page_span=4)
    metadata = _agent("prefill-engine", placement)
    setattr(metadata, field, wrong_value)

    with pytest.raises(ValueError, match="not exactly bound"):
        index_remote_nixl_placements(
            {(0, 0): metadata},
            {(0, 0): "agent-0"},
        )


def test_remote_index_rejects_placement_engine_mismatch():
    placement = _placement("prefill-engine", "prefill-0", page_span=4)

    with pytest.raises(ValueError, match="does not match NIXL engine"):
        index_remote_nixl_placements(
            {(0, 0): _agent("aliased-engine", placement)},
            {(0, 0): "agent-0"},
        )


def test_remote_index_rejects_aliased_page_template():
    placement = _two_layer_placement(alias_registration=True)
    metadata = _agent("prefill-engine", placement)
    # Keep the legacy regions distinct so this exercises the named-template
    # alias check rather than the equivalent legacy-region check.
    metadata.kv_caches_base_addr[1] += 4096

    with pytest.raises(ValueError, match="ambiguously alias"):
        index_remote_nixl_placements(
            {(0, 0): metadata},
            {(0, 0): "agent-0"},
        )


def test_remote_index_accepts_interleaved_registration_pages():
    placement = _two_layer_placement()
    first, second = placement.page_registration_templates
    first = replace(first, base_address=1000, page_stride=8, num_pages=4)
    second = replace(second, base_address=1004, page_stride=8, num_pages=4)
    placement = replace(
        placement,
        page_registration_templates=(first, second),
    )
    metadata = _agent("prefill-engine", placement)

    index = index_remote_nixl_placements(
        {(0, 0): metadata},
        {(0, 0): placement.rank_placement.worker_incarnation},
    )

    assert index.workers == (placement,)


def test_remote_index_rejects_overlapping_registration_pages():
    placement = _two_layer_placement()
    first, second = placement.page_registration_templates
    overlapping_second = replace(
        second,
        base_address=first.base_address + first.page_stride,
    )
    placement = replace(
        placement,
        page_registration_templates=(first, overlapping_second),
    )
    metadata = _agent("prefill-engine", placement)

    with pytest.raises(ValueError, match="overlapping memory pages"):
        index_remote_nixl_placements(
            {(0, 0): metadata},
            {(0, 0): placement.rank_placement.worker_incarnation},
        )


def test_remote_index_rejects_compressed_or_aliased_registration():
    placement = _two_layer_placement(alias_registration=True)
    metadata = _agent("prefill-engine", placement)
    # Legacy registration de-duplicates regions by base address, so it cannot
    # prove which named layer owns an aliased/compressed region.
    metadata.kv_caches_base_addr = metadata.kv_caches_base_addr[:1]
    metadata.block_lens = metadata.block_lens[:1]
    metadata.block_strides = metadata.block_strides[:1]

    with pytest.raises(ValueError, match="compressed, aliased"):
        index_remote_nixl_placements(
            {(0, 0): metadata},
            {(0, 0): "agent-0"},
        )


def test_remote_index_rejects_unbound_duplicate_agent_name():
    first = _placement(
        "prefill-engine", "prefill-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    second = _placement(
        "prefill-engine", "prefill-1", page_span=4, rank=1, tp_size=2, tp_rank=1
    )

    with pytest.raises(ValueError, match="does not match placement worker"):
        index_remote_nixl_placements(
            {
                (0, 0): _agent("prefill-engine", first),
                (0, 1): _agent("prefill-engine", second),
            },
            {(0, 0): "agent", (0, 1): "agent"},
        )


def test_remote_index_rejects_agent_names_swapped_across_ranks():
    first = _placement(
        "prefill-engine", "prefill-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    second = _placement(
        "prefill-engine", "prefill-1", page_span=4, rank=1, tp_size=2, tp_rank=1
    )

    with pytest.raises(ValueError, match="does not match placement worker"):
        index_remote_nixl_placements(
            {
                (0, 0): _agent("prefill-engine", first),
                (0, 1): _agent("prefill-engine", second),
            },
            {
                (0, 0): second.rank_placement.worker_incarnation,
                (0, 1): first.rank_placement.worker_incarnation,
            },
        )


def test_remote_index_rejects_duplicate_worker_incarnation():
    first = _placement(
        "prefill-engine", "prefill-0", page_span=4, rank=0, tp_size=2, tp_rank=0
    )
    second = _placement(
        "prefill-engine", "prefill-1", page_span=4, rank=1, tp_size=2, tp_rank=1
    )
    second = replace(
        second,
        rank_placement=replace(
            second.rank_placement,
            worker_incarnation=first.rank_placement.worker_incarnation,
        ),
    )

    with pytest.raises(
        ValueError, match="duplicate remote placement worker incarnation"
    ):
        index_remote_nixl_placements(
            {
                (0, 0): _agent("prefill-engine", first),
                (0, 1): _agent("prefill-engine", second),
            },
            {
                (0, 0): first.rank_placement.worker_incarnation,
                (0, 1): first.rank_placement.worker_incarnation,
            },
        )
