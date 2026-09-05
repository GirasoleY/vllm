# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused lifecycle regressions for NIXL endpoint registrations."""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_scheduler
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
    _endpoint_cohort_incarnation,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
    _HandshakeSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

_ENGINE = "remote-engine"
_INCARNATION = "remote-incarnation-a"
_HANDSHAKE_KWARGS = {
    "host": "10.0.0.1",
    "port": 5600,
    "tp_size": 8,
    "dcp_size": 8,
    "pcp_size": 1,
    "pp_size": 1,
    "notif_agents_only": False,
    "endpoint_incarnation": _INCARNATION,
}


class _DeferredExecutor:
    """Executor stub that exposes submissions without running them."""

    def __init__(self) -> None:
        self.submissions: list[tuple[object, tuple[object, ...]]] = []
        self.futures: list[Future[object]] = []

    def submit(self, fn, *args):
        future: Future[object] = Future()
        self.submissions.append((fn, args))
        self.futures.append(future)
        return future


def _handshake_worker(worker_cls=NixlBaseConnectorWorker):
    worker = object.__new__(worker_cls)
    # Partial object stubs intentionally do not own production resources.
    # Suppress the production destructor, whose dynamic shutdown dispatch would
    # otherwise try to tear down fields that these unit stubs never create.
    worker.shutdown = lambda: None
    worker._handshake_lock = threading.RLock()
    worker._handshake_shutdown_event = threading.Event()
    worker._shutting_down = False
    worker._shutdown_complete = False
    worker._remote_agents = {}
    worker._remote_handshake_specs = {}
    worker._stale_remote_engines = set()
    worker._handshake_futures = {}
    worker._handshake_future_specs = {}
    worker._engine_last_active = {}
    worker._engine_ttl = 0.0
    worker._engine_clock_offset = {}
    worker._recving_metadata = {}
    worker._recving_transfers = defaultdict(list)
    worker._direct_read_batch_windows = {}
    worker._sending_transfers = defaultdict(list)
    worker._sending_transfers_lock = threading.Lock()
    worker.transfer_topo = None
    worker._nixl_handshake = MagicMock()
    worker._log_failure = MagicMock()
    executor = _DeferredExecutor()
    worker._handshake_initiation_executor = executor

    def cleanup(engine_id, *, log_eviction=True):
        del log_eviction
        worker._remote_agents.pop(engine_id, None)
        worker._remote_handshake_specs.pop(engine_id, None)
        worker._stale_remote_engines.discard(engine_id)
        worker._engine_last_active.pop(engine_id, None)

    worker._cleanup_remote_engine = MagicMock(side_effect=cleanup)
    return worker, executor


def _install_cached_handshake(worker, spec: _HandshakeSpec) -> None:
    worker._remote_agents[_ENGINE] = {(0, 0): "remote-agent"}
    worker._remote_handshake_specs[_ENGINE] = spec
    worker._engine_last_active[_ENGINE] = time.perf_counter()


def _ensure(worker, **overrides):
    kwargs = dict(_HANDSHAKE_KWARGS)
    kwargs.update(overrides)
    return NixlBaseConnectorWorker._ensure_handshake(
        worker,
        engine_id=_ENGINE,
        **kwargs,
    )


def test_exact_cached_handshake_spec_is_reused() -> None:
    worker, executor = _handshake_worker()
    _install_cached_handshake(worker, _HandshakeSpec(**_HANDSHAKE_KWARGS))

    assert _ensure(worker) is None
    assert executor.submissions == []
    worker._cleanup_remote_engine.assert_not_called()


@pytest.mark.parametrize(
    "changed",
    [
        {"port": 5601},
        {"notif_agents_only": True},
        {"endpoint_incarnation": "remote-incarnation-b"},
    ],
    ids=["address", "registration-mode", "endpoint-incarnation"],
)
def test_changed_cached_handshake_spec_replaces_idle_registration(changed) -> None:
    worker, executor = _handshake_worker()
    _install_cached_handshake(worker, _HandshakeSpec(**_HANDSHAKE_KWARGS))

    future = _ensure(worker, **changed)

    assert future is executor.futures[0]
    worker._cleanup_remote_engine.assert_called_once_with(_ENGINE, log_eviction=False)
    assert len(executor.submissions) == 1
    assert executor.submissions[0][1][-1] == changed.get(
        "endpoint_incarnation", _INCARNATION
    )


def test_pending_handshake_reuses_only_the_exact_endpoint_incarnation() -> None:
    worker, executor = _handshake_worker()

    first = _ensure(worker)
    assert _ensure(worker) is first

    with pytest.raises(RuntimeError, match="previous incarnation"):
        _ensure(worker, endpoint_incarnation="remote-incarnation-b")
    assert len(executor.submissions) == 1


@pytest.mark.parametrize("missing_incarnation", [None, ""])
def test_v18_handshake_requires_endpoint_incarnation(missing_incarnation) -> None:
    worker, executor = _handshake_worker()

    with pytest.raises(RuntimeError, match="endpoint incarnation"):
        _ensure(worker, endpoint_incarnation=missing_incarnation)
    assert executor.submissions == []


@pytest.mark.parametrize("resource", ["handle", "window"])
def test_same_request_old_resources_pin_changed_registration(resource) -> None:
    worker, executor = _handshake_worker(NixlPullConnectorWorker)
    _install_cached_handshake(worker, _HandshakeSpec(**_HANDSHAKE_KWARGS))
    worker._recving_metadata["request"] = SimpleNamespace(
        remote=SimpleNamespace(engine_id=_ENGINE)
    )
    if resource == "handle":
        worker._recving_transfers["request"] = [101]
    else:
        worker._direct_read_batch_windows["request"] = object()

    with pytest.raises(RuntimeError, match="registration is still in use"):
        _ensure(
            worker,
            endpoint_incarnation="remote-incarnation-b",
            request_id="request",
        )

    worker._cleanup_remote_engine.assert_not_called()
    assert executor.submissions == []


@pytest.mark.parametrize("resource", ["handle", "window", "push"])
def test_ttl_does_not_evict_registration_with_active_resources(resource) -> None:
    worker, executor = _handshake_worker(NixlPullConnectorWorker)
    _install_cached_handshake(worker, _HandshakeSpec(**_HANDSHAKE_KWARGS))
    worker._engine_ttl = 1.0
    worker._engine_last_active[_ENGINE] = time.perf_counter() - 10.0
    worker._recving_metadata["request"] = SimpleNamespace(
        remote=SimpleNamespace(engine_id=_ENGINE)
    )
    if resource == "handle":
        worker._recving_transfers["request"] = [101]
    elif resource == "window":
        worker._direct_read_batch_windows["request"] = object()
    else:
        worker._sending_transfers["other-request"] = [202]

    assert _ensure(worker, request_id="request") is None

    worker._cleanup_remote_engine.assert_not_called()
    assert executor.submissions == []


def _pending_request_meta(incarnation: str):
    return SimpleNamespace(
        local_block_ids=([1],),
        remote=SimpleNamespace(
            engine_id=_ENGINE,
            host=_HANDSHAKE_KWARGS["host"],
            port=_HANDSHAKE_KWARGS["port"],
            endpoint_incarnation=incarnation,
            block_ids=([2],),
        ),
        tp_size=_HANDSHAKE_KWARGS["tp_size"],
        dcp_size=_HANDSHAKE_KWARGS["dcp_size"],
        pcp_size=_HANDSHAKE_KWARGS["pcp_size"],
        pp_size=_HANDSHAKE_KWARGS["pp_size"],
    )


@pytest.mark.parametrize("succeeds", [True, False], ids=["success", "failure"])
def test_stale_handshake_callback_cannot_affect_newer_same_id_attempt(
    succeeds,
) -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.shutdown = lambda: None
    old_meta = _pending_request_meta(_INCARNATION)
    new_meta = _pending_request_meta("remote-incarnation-b")
    worker._recving_metadata = {"request": old_meta}
    worker._ready_requests = queue.Queue()
    future: Future[tuple[dict[tuple[int, int], str], float]] = Future()
    worker._ensure_handshake = MagicMock(return_value=future)
    worker._shutting_down = False
    worker._log_failure = MagicMock()
    worker._handle_failed_transfer = MagicMock()

    NixlBaseConnectorWorker._background_nixl_handshake(
        worker, "request", _ENGINE, old_meta
    )
    worker._recving_metadata["request"] = new_meta
    if succeeds:
        future.set_result(({}, 0.0))
    else:
        future.set_exception(RuntimeError("old handshake failed"))

    assert worker._ready_requests.empty()
    worker._handle_failed_transfer.assert_not_called()


def test_stale_ready_entry_cannot_start_newer_same_id_attempt() -> None:
    worker = object.__new__(NixlPullConnectorWorker)
    worker.shutdown = lambda: None
    old_meta = _pending_request_meta(_INCARNATION)
    new_meta = _pending_request_meta("remote-incarnation-b")
    worker._recving_metadata = {"request": new_meta}
    worker._ready_requests = queue.Queue()
    worker._ready_requests.put(("request", old_meta))
    worker._read_blocks_for_req = MagicMock()
    worker._refill_direct_read_batch_windows = MagicMock()
    worker.pcp_rank = 1
    worker._enable_generic_placement = False

    NixlPullConnectorWorker.start_load_kv(worker, SimpleNamespace(reqs_to_recv={}))

    worker._read_blocks_for_req.assert_not_called()


def test_pending_attempt_blocks_overlapping_same_id_metadata() -> None:
    worker, _ = _handshake_worker(NixlPullConnectorWorker)
    old_meta = _pending_request_meta(_INCARNATION)
    new_meta = _pending_request_meta("remote-incarnation-b")
    worker._recving_metadata["request"] = old_meta
    worker._logical_to_kernel_block_ids = MagicMock(return_value=([3],))
    worker._physical_blocks_per_logical_kv_block = 1
    worker._background_nixl_handshake = MagicMock()
    worker._ready_requests = queue.Queue()
    worker._read_blocks_for_req = MagicMock()
    worker._refill_direct_read_batch_windows = MagicMock()
    worker.pcp_rank = 1
    worker._enable_generic_placement = False

    NixlPullConnectorWorker.start_load_kv(
        worker,
        SimpleNamespace(reqs_to_recv={"request": new_meta}),
    )

    assert worker._recving_metadata["request"] is old_meta
    worker._background_nixl_handshake.assert_not_called()
    worker._read_blocks_for_req.assert_not_called()


def _push_handshake_mode_worker():
    worker = object.__new__(NixlPushConnectorWorker)
    worker.shutdown = lambda: None
    worker._hb_handshake_notif_only = True
    worker._push_writer_stop = threading.Event()
    worker._remote_agents = {_ENGINE: {(0, 0): "remote-agent"}}
    worker._do_send_reg_notif = MagicMock()
    worker.nixl_wrapper = MagicMock()
    observed_specs: list[_HandshakeSpec] = []

    def ensure(
        engine_id,
        host,
        port,
        tp_size,
        dcp_size=1,
        pp_size=1,
        notif_agents_only=False,
        pcp_size=1,
        endpoint_incarnation=None,
        request_id=None,
    ):
        del engine_id, request_id
        spec = _HandshakeSpec(
            host=host,
            port=port,
            tp_size=tp_size,
            dcp_size=dcp_size,
            pcp_size=pcp_size,
            pp_size=pp_size,
            notif_agents_only=notif_agents_only,
            endpoint_incarnation=endpoint_incarnation,
        )
        observed_specs.append(spec)
        if spec != observed_specs[0]:
            raise RuntimeError("incompatible handshake mode")
        return None

    worker._ensure_handshake = ensure
    return worker, observed_specs


@pytest.mark.parametrize(
    "ordering",
    [("registration", "heartbeat"), ("heartbeat", "registration")],
    ids=["registration-first", "heartbeat-first"],
)
def test_push_pp1_registration_and_heartbeat_share_notif_only_spec(ordering) -> None:
    worker, observed_specs = _push_handshake_mode_worker()
    registration = {
        "request_id": "request",
        "remote_engine_id": _ENGINE,
        "remote_host": _HANDSHAKE_KWARGS["host"],
        "remote_port": _HANDSHAKE_KWARGS["port"],
        "remote_tp_size": 1,
        "remote_pp_size": 1,
        "remote_endpoint_incarnation": _INCARNATION,
    }
    metadata = SimpleNamespace(
        heartbeat_by_engine={
            _ENGINE: HeartbeatInfo(
                req_ids={"request"},
                host=_HANDSHAKE_KWARGS["host"],
                port=_HANDSHAKE_KWARGS["port"],
                tp_size=1,
                pp_size=1,
                endpoint_incarnation=_INCARNATION,
            )
        }
    )
    operations = {
        "registration": lambda: NixlPushConnectorWorker._send_registration_to_p(
            worker, "request", registration
        ),
        "heartbeat": lambda: NixlBaseConnectorWorker._send_heartbeats(worker, metadata),
    }

    for operation in ordering:
        operations[operation]()

    assert len(observed_specs) == 2
    assert observed_specs[0] == observed_specs[1]
    assert observed_specs[0].notif_agents_only is True


def test_idle_push_notif_registration_rolls_to_new_endpoint() -> None:
    worker, executor = _handshake_worker(NixlPushConnectorWorker)
    old_kwargs = dict(_HANDSHAKE_KWARGS)
    old_kwargs["notif_agents_only"] = True
    _install_cached_handshake(worker, _HandshakeSpec(**old_kwargs))
    worker._push_writer_stop = threading.Event()
    worker._push_writer_wake = threading.Event()
    worker._reg_send_inbox = queue.Queue()
    worker._do_send_reg_notif = MagicMock()
    worker._recving_metadata["request"] = SimpleNamespace(
        remote=SimpleNamespace(engine_id=_ENGINE)
    )
    registration = {
        "request_id": "request",
        "remote_engine_id": _ENGINE,
        "remote_host": _HANDSHAKE_KWARGS["host"],
        "remote_port": _HANDSHAKE_KWARGS["port"],
        "remote_tp_size": _HANDSHAKE_KWARGS["tp_size"],
        "remote_pp_size": _HANDSHAKE_KWARGS["pp_size"],
        "remote_endpoint_incarnation": "remote-incarnation-b",
    }

    NixlPushConnectorWorker._send_registration_to_p(worker, "request", registration)

    worker._cleanup_remote_engine.assert_called_once_with(_ENGINE, log_eviction=False)
    assert len(executor.submissions) == 1
    assert executor.submissions[0][1][-1] == "remote-incarnation-b"


def _prepare_push_start_load_worker():
    worker, _ = _handshake_worker(NixlPushConnectorWorker)
    worker.pcp_rank = 0
    worker._logical_to_kernel_block_ids = MagicMock(return_value=([3],))
    worker._physical_blocks_per_logical_kv_block = 1
    worker._reg_send_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._finished_blocks_inbox = queue.Queue()
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker._send_heartbeats = MagicMock()
    return worker


def _push_step_metadata(req_meta, registration):
    return SimpleNamespace(
        reqs_to_recv={"request": req_meta},
        push_registrations={"request": registration},
        push_finished_blocks={},
        reqs_in_batch=set(),
        reqs_not_processed=set(),
        reqs_to_send={},
        scheduler_clock=0.0,
        heartbeat_by_engine={},
    )


def test_pending_push_attempt_suppresses_duplicate_registration() -> None:
    worker = _prepare_push_start_load_worker()
    old_meta = _pending_request_meta(_INCARNATION)
    new_meta = _pending_request_meta("remote-incarnation-b")
    worker._recving_metadata["request"] = old_meta
    registration = {"remote_endpoint_incarnation": "remote-incarnation-b"}

    NixlPushConnectorWorker.start_load_kv(
        worker,
        _push_step_metadata(new_meta, registration),
    )

    assert worker._recving_metadata["request"] is old_meta
    assert worker._reg_send_inbox.empty()


def test_push_metadata_mutation_waits_for_endpoint_scan_lock() -> None:
    worker = _prepare_push_start_load_worker()
    new_meta = _pending_request_meta(_INCARNATION)
    converted = threading.Event()
    finished = threading.Event()

    def convert(*_args):
        converted.set()
        return ([3],)

    worker._logical_to_kernel_block_ids.side_effect = convert

    def start_load() -> None:
        NixlPushConnectorWorker.start_load_kv(
            worker,
            _push_step_metadata(new_meta, {}),
        )
        finished.set()

    with worker._handshake_lock:
        thread = threading.Thread(target=start_load)
        thread.start()
        assert converted.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05)
        assert "request" not in worker._recving_metadata

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert worker._recving_metadata["request"] is new_meta


def test_heartbeat_handshake_rejection_is_best_effort() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker._hb_handshake_notif_only = False
    worker._ensure_handshake = MagicMock(
        side_effect=RuntimeError("endpoint changed while pinned")
    )
    worker.nixl_wrapper = MagicMock()
    metadata = SimpleNamespace(
        heartbeat_by_engine={
            _ENGINE: HeartbeatInfo(
                req_ids={"request"},
                host=_HANDSHAKE_KWARGS["host"],
                port=_HANDSHAKE_KWARGS["port"],
                tp_size=8,
                dcp_size=8,
                endpoint_incarnation=_INCARNATION,
            )
        }
    )

    NixlBaseConnectorWorker._send_heartbeats(worker, metadata)

    worker.nixl_wrapper.send_notif.assert_not_called()


class _RecordingExecutor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait, cancel_futures):
        self.shutdown_calls.append((wait, cancel_futures))
        self.events.append("executor-wait")


def _shutdown_worker():
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.shutdown = lambda: None
    events: list[str] = []
    executor = _RecordingExecutor(events)
    worker._handshake_initiation_executor = executor
    worker._handshake_lock = threading.RLock()
    worker._handshake_shutdown_event = threading.Event()
    worker._shutting_down = False
    worker._shutdown_complete = False
    worker._handshake_futures = {"pending": Future()}
    worker._handshake_future_specs = {}
    worker._recving_transfers = defaultdict(list)
    worker._recving_metadata = {}
    worker._generic_direct_receive_requests = set()
    worker._request_terminal_poller = MagicMock()
    worker._ephemeral_direct_dlists = MagicMock()
    worker.src_xfer_handles_by_block_size = {}
    worker.src_xfer_handles_by_tp_ratio = {}
    worker._remote_agents = {}
    worker._remote_handshake_specs = {}
    worker._stale_remote_engines = set()
    worker._registered_descs = ["registered-memory"]
    worker._ready_requests = queue.Queue()
    worker._failed_recv_reqs = queue.Queue()
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.deregister_memory.side_effect = lambda _desc: events.append(
        "memory-cleanup"
    )
    return worker, executor, events


def test_shutdown_waits_before_cleanup_is_idempotent_and_blocks_new_handshakes():
    worker, executor, events = _shutdown_worker()

    NixlBaseConnectorWorker.shutdown(worker)

    assert executor.shutdown_calls == [(True, True)]
    assert events == ["executor-wait", "memory-cleanup"]
    assert worker._handshake_shutdown_event.is_set()
    assert worker._shutdown_complete is True

    # Repeated shutdown must neither wait nor tear down resources twice.
    NixlBaseConnectorWorker.shutdown(worker)
    assert executor.shutdown_calls == [(True, True)]
    assert events == ["executor-wait", "memory-cleanup"]

    with pytest.raises(RuntimeError, match="shutting down"):
        _ensure(worker)


def test_lease_expiry_scans_past_a_later_deadline() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.shutdown = lambda: None
    worker.transfer_topo = object()
    worker._get_new_notifs = MagicMock(return_value=set())
    worker._recving_transfers = defaultdict(list)
    worker._pop_done_transfers = MagicMock(return_value=set())
    worker._failed_recv_reqs = queue.Queue()
    worker._on_receive_requests_terminal = MagicMock()
    worker._recving_metadata = {}
    worker._generic_direct_receive_requests = set()
    worker.tp_rank = 0
    worker._sync_device_after_mamba_recv = MagicMock()
    now = time.perf_counter()
    # Deliberately insert the unexpired lease first. Expiry is not ordered by
    # dict insertion, so the scan must still find the second request.
    worker._reqs_to_send = {
        "unexpired": now + 60,
        "expired": now - 1,
    }
    worker.consumer_notification_counts_by_req = defaultdict(int)
    worker.expected_consumer_notifications_by_req = {}
    worker.xfer_stats = MagicMock()
    worker._reqs_to_process = {"unexpired", "expired"}
    worker._on_send_request_terminal = MagicMock()

    done_sending, done_recving = NixlBaseConnectorWorker.get_finished(worker)

    assert done_sending == {"expired"}
    assert done_recving == set()
    assert worker._reqs_to_send == {"unexpired": pytest.approx(now + 60)}
    worker.xfer_stats.record_kv_expired_req.assert_called_once_with()
    worker._on_send_request_terminal.assert_called_once_with("expired")


class _CapturedThread:
    """Capture listener arguments while satisfying its ready-event barrier."""

    instances: list[_CapturedThread] = []

    def __init__(self, *, target, args, daemon, name) -> None:
        self.target = target
        self.args = args
        self.daemon = daemon
        self.name = name
        self.instances.append(self)

    def start(self) -> None:
        self.args[1].set()


def test_scheduler_stamps_one_endpoint_incarnation_on_every_served_rank() -> None:
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler._scheduler_incarnation = "scheduler-incarnation"
    scheduler._endpoint_incarnation = scheduler._scheduler_incarnation
    scheduler._nixl_handshake_listener_t = None
    scheduler._stop_event = threading.Event()
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.side_channel_port = 5600
    payloads = {
        (0, 0): NixlHandshakePayload("strict", "placement", b"rank-0", "old-0"),
        (0, 1): NixlHandshakePayload("strict", "placement", b"rank-1", "old-1"),
        (1, 0): NixlHandshakePayload("strict", "placement", b"rank-2", "old-2"),
    }
    _CapturedThread.instances.clear()

    with patch.object(base_scheduler.threading, "Thread", _CapturedThread):
        scheduler.set_xfer_handshake_metadata(payloads)

    assert len(_CapturedThread.instances) == 1
    metadata_store = _CapturedThread.instances[0].args[0]
    encoded_by_rank = metadata_store.snapshot()
    decoded = {
        rank: NixlHandshakePayload.decode(encoded)
        for rank, encoded in encoded_by_rank.items()
    }
    endpoint_tokens = {payload.endpoint_incarnation for payload in decoded.values()}
    assert endpoint_tokens == {scheduler._endpoint_incarnation}
    assert scheduler._endpoint_incarnation.startswith("sha256:")
    assert {
        rank: payload.agent_metadata_bytes for rank, payload in decoded.items()
    } == {rank: payload.agent_metadata_bytes for rank, payload in payloads.items()}


def test_endpoint_incarnation_changes_with_worker_cohort() -> None:
    first = {
        (1, 0): NixlHandshakePayload(
            "strict", "placement", b"worker-b", "untrusted-token-b"
        ),
        (0, 0): NixlHandshakePayload(
            "strict", "placement", b"worker-a", "untrusted-token-a"
        ),
    }
    reordered = {
        (0, 0): NixlHandshakePayload(
            "strict", "placement", b"worker-a", "different-untrusted-token"
        ),
        (1, 0): NixlHandshakePayload("strict", "placement", b"worker-b"),
    }
    restarted = dict(reordered)
    restarted[(1, 0)] = NixlHandshakePayload(
        "strict", "placement", b"worker-b-restarted"
    )

    original = _endpoint_cohort_incarnation("scheduler-boot", first)

    assert original == _endpoint_cohort_incarnation("scheduler-boot", reordered)
    assert original != _endpoint_cohort_incarnation("scheduler-boot", restarted)
    assert original != _endpoint_cohort_incarnation("new-scheduler-boot", first)


def test_live_handshake_listener_atomically_swaps_worker_cohort() -> None:
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler._scheduler_incarnation = "scheduler-incarnation"
    scheduler._endpoint_incarnation = scheduler._scheduler_incarnation
    scheduler._nixl_handshake_listener_t = None
    scheduler._stop_event = threading.Event()
    scheduler.side_channel_host = "127.0.0.1"
    scheduler.side_channel_port = 5600
    first = {(0, 0): NixlHandshakePayload("strict", "placement", b"worker-a")}
    restarted = {(0, 0): NixlHandshakePayload("strict", "placement", b"worker-b")}
    _CapturedThread.instances.clear()

    with patch.object(base_scheduler.threading, "Thread", _CapturedThread):
        scheduler.set_xfer_handshake_metadata(first)
        original_incarnation = scheduler._endpoint_incarnation
        scheduler.set_xfer_handshake_metadata(restarted)
        restarted_incarnation = scheduler._endpoint_incarnation
        scheduler.set_xfer_handshake_metadata(restarted)

    assert len(_CapturedThread.instances) == 1
    assert scheduler._endpoint_incarnation != original_incarnation
    assert scheduler._endpoint_incarnation == restarted_incarnation
    metadata_store = _CapturedThread.instances[0].args[0]
    served = NixlHandshakePayload.decode(metadata_store.snapshot()[(0, 0)])
    assert served.endpoint_incarnation == scheduler._endpoint_incarnation
    assert served.agent_metadata_bytes == b"worker-b"


def test_concurrent_cohort_publish_keeps_served_payload_and_token_paired() -> None:
    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler._scheduler_incarnation = "scheduler-incarnation"
    scheduler._endpoint_incarnation = scheduler._scheduler_incarnation
    scheduler._handshake_metadata_publish_lock = threading.Lock()
    store = base_scheduler._NixlHandshakeMetadataStore()
    scheduler._nixl_handshake_metadata_store = store
    # A live-listener sentinel keeps this test focused on publication; the
    # listener reads the same store in production.
    scheduler._nixl_handshake_listener_t = object()
    first_store_replace = threading.Event()
    release_first_publisher = threading.Event()
    second_publisher_done = threading.Event()
    errors: list[BaseException] = []
    original_replace = store.replace

    def blocking_replace(encoded_data) -> None:
        original_replace(encoded_data)
        if not first_store_replace.is_set():
            first_store_replace.set()
            assert release_first_publisher.wait(timeout=1.0)

    store.replace = blocking_replace
    first = {(0, 0): NixlHandshakePayload("strict", "placement", b"worker-a")}
    second = {(0, 0): NixlHandshakePayload("strict", "placement", b"worker-b")}

    def publish(metadata, *, done: threading.Event | None = None) -> None:
        try:
            scheduler.set_xfer_handshake_metadata(metadata)
        except BaseException as error:
            errors.append(error)
        finally:
            if done is not None:
                done.set()

    first_thread = threading.Thread(target=publish, args=(first,))
    first_thread.start()
    assert first_store_replace.wait(timeout=1.0)

    second_thread = threading.Thread(
        target=publish,
        args=(second,),
        kwargs={"done": second_publisher_done},
    )
    second_thread.start()
    assert not second_publisher_done.wait(timeout=0.05)

    release_first_publisher.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert errors == []

    served = NixlHandshakePayload.decode(store.snapshot()[(0, 0)])
    assert served.endpoint_incarnation == scheduler._endpoint_incarnation
    assert served.agent_metadata_bytes == b"worker-b"


def test_nixl_handshake_rejects_mismatched_endpoint_incarnation() -> None:
    worker = object.__new__(NixlBaseConnectorWorker)
    worker.shutdown = lambda: None
    worker._is_csa_linear = False
    worker.use_host_buffer = True
    worker.transfer_topo = SimpleNamespace(
        handshake_target_ranks=lambda *_args: [0],
        unregister_remote_engine=MagicMock(),
    )
    worker._local_placement_metadata = None
    worker.dst_xfer_side_handles = defaultdict(dict)
    worker.nixl_wrapper = MagicMock()
    worker._remote_agents = {}
    worker.kv_caches_base_addr = {}
    worker.dst_num_blocks = {}
    worker.tp_mappings = {}
    worker._remote_placement_indexes = {}
    worker._generic_only_remote_engines = set()
    worker._remote_handshake_specs = {}
    worker._stale_remote_engines = set()
    worker._engine_clock_offset = {}
    worker._engine_last_active = {}
    payload = NixlHandshakePayload(
        compatibility_hash="strict",
        placement_compatibility_hash="placement",
        agent_metadata_bytes=b"unused-before-incarnation-validation",
        endpoint_incarnation="advertised-incarnation",
    )
    socket = MagicMock()

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.zmq_ctx"
        ) as mock_zmq_ctx,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker."
            "recv_multipart_bounded",
            return_value=[
                msgspec.msgpack.encode(payload),
                msgspec.msgpack.encode(time.perf_counter()),
            ],
        ),
    ):
        mock_zmq_ctx.return_value.__enter__.return_value = socket
        with pytest.raises(RuntimeError, match="incarnation does not match"):
            worker._nixl_handshake(
                host="10.0.0.1",
                port=5600,
                remote_tp_size=1,
                expected_engine_id=_ENGINE,
                expected_endpoint_incarnation="expected-incarnation",
            )

    worker.nixl_wrapper.remove_remote_agent.assert_not_called()
