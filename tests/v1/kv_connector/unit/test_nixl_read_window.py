# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict, deque
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
    _DirectReadBatchWindow,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.terminal import (
    NixlRequestTerminalPoller,
)


class _PreparedBatchIterator:
    def __init__(self, handles: list[int], *, fail_at: int | None = None):
        self._handles = iter(handles)
        self._fail_at = fail_at
        self.prepared_count = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._fail_at == self.prepared_count:
            raise RuntimeError("injected preparation failure")
        handle = next(self._handles)
        self.prepared_count += 1
        return SimpleNamespace(transfer_handle=handle)

    def close(self):
        self.closed = True


class _FakeNixlWrapper:
    def __init__(self, *, fail_submit_handle: int | None = None):
        self.fail_submit_handle = fail_submit_handle
        self.started: list[int] = []
        self.released: list[int] = []
        self.states: dict[int, str] = {}

    def transfer(self, handle: int) -> None:
        if handle == self.fail_submit_handle:
            raise RuntimeError("injected submission failure")
        self.started.append(handle)
        self.states[handle] = "PROC"

    def check_xfer_state(self, handle: int) -> str:
        return self.states[handle]

    def get_xfer_telemetry(self, handle: int):
        return {"handle": handle}

    def release_xfer_handle(self, handle: int) -> None:
        self.released.append(handle)


def _worker(
    wrapper: _FakeNixlWrapper, *, max_inflight: int = 8
) -> NixlPullConnectorWorker:
    worker = object.__new__(NixlPullConnectorWorker)
    worker.nixl_wrapper = wrapper
    worker._recving_transfers = defaultdict(list)
    worker._recving_metadata = {}
    worker._direct_read_batch_windows = {}
    worker._direct_read_refill_queue = deque()
    worker._max_inflight_direct_batches = max_inflight
    worker._engine_last_active = {"remote": 0.0}
    worker._request_terminal_poller = NixlRequestTerminalPoller()
    worker._failed_recv_reqs = SimpleNamespace(put=lambda _req_id: None)
    worker._log_failure = lambda **_kwargs: None
    worker._release_xfer_handle = wrapper.release_xfer_handle
    worker.xfer_stats = SimpleNamespace(
        record_transfer=lambda _telemetry: None,
        record_failed_transfer=lambda: None,
    )

    def record_failed_transfer(
        _req_id: str, handle: int | None, *, mark_request_invalid: bool
    ) -> None:
        del mark_request_invalid
        if handle is not None:
            wrapper.release_xfer_handle(handle)
        worker.xfer_stats.record_failed_transfer()

    worker._record_failed_transfer = record_failed_transfer
    return worker


def _install_window(
    worker: NixlPullConnectorWorker,
    req_id: str,
    batches: _PreparedBatchIterator,
) -> _DirectReadBatchWindow:
    state = _DirectReadBatchWindow(batches=batches, remote_engine_id="remote")
    worker._direct_read_batch_windows[req_id] = state
    worker._direct_read_refill_queue.append(req_id)
    worker._recving_transfers[req_id] = []
    return state


def test_direct_read_window_uses_one_worker_global_inflight_bound():
    wrapper = _FakeNixlWrapper()
    worker = _worker(wrapper, max_inflight=3)
    batches_by_req = {
        req_id: _PreparedBatchIterator([base, base + 1])
        for req_id, base in (("a", 10), ("b", 20), ("c", 30), ("d", 40))
    }
    for req_id, batches in batches_by_req.items():
        _install_window(worker, req_id, batches)

    worker._refill_direct_read_batch_windows()

    assert sum(map(len, worker._recving_transfers.values())) == 3
    assert wrapper.started == [10, 20, 30]
    assert [batches.prepared_count for batches in batches_by_req.values()] == [
        1,
        1,
        1,
        0,
    ]


def test_direct_read_window_round_robins_each_refill_pass():
    wrapper = _FakeNixlWrapper()
    worker = _worker(wrapper, max_inflight=4)
    first = _PreparedBatchIterator([10, 11, 12])
    second = _PreparedBatchIterator([20, 21, 22])
    _install_window(worker, "first", first)
    _install_window(worker, "second", second)

    worker._refill_direct_read_batch_windows()

    assert wrapper.started == [10, 20, 11, 21]
    assert first.prepared_count == second.prepared_count == 2


def test_direct_read_window_waiting_for_credit_is_not_terminal():
    wrapper = _FakeNixlWrapper()
    worker = _worker(wrapper, max_inflight=1)
    first = _PreparedBatchIterator([10])
    second = _PreparedBatchIterator([20])
    _install_window(worker, "first", first)
    _install_window(worker, "second", second)
    worker._refill_direct_read_batch_windows()

    assert wrapper.started == [10]
    assert worker._recving_transfers["second"] == []
    wrapper.states[10] = "PROC"

    assert worker._pop_done_transfers(worker._recving_transfers) == set()
    assert "second" in worker._recving_transfers
    assert "second" in worker._direct_read_batch_windows
    assert second.prepared_count == 0


def test_direct_read_window_refreshes_remote_engine_activity():
    wrapper = _FakeNixlWrapper()
    worker = _worker(wrapper, max_inflight=1)
    batches = _PreparedBatchIterator([10])
    _install_window(worker, "request", batches)

    worker._refill_direct_read_batch_windows()
    assert worker._engine_last_active["remote"] > 0.0

    worker._engine_last_active["remote"] = 0.0
    wrapper.states[10] = "PROC"
    worker._pop_done_transfers(worker._recving_transfers)
    assert worker._engine_last_active["remote"] > 0.0


def test_direct_read_window_refills_and_terminalizes_only_after_exhaustion():
    wrapper = _FakeNixlWrapper()
    worker = _worker(wrapper, max_inflight=2)
    batches = _PreparedBatchIterator([10, 11, 12, 13, 14])
    state = _install_window(worker, "request", batches)
    worker._refill_direct_read_batch_windows()

    wrapper.states.update({10: "DONE", 11: "PROC"})
    failed: set[str] = set()
    assert (
        worker._pop_done_transfers(worker._recving_transfers, failed_req_ids=failed)
        == set()
    )
    assert failed == set()
    assert worker._recving_transfers["request"] == [11, 12]
    assert batches.prepared_count == 3

    wrapper.states.update({11: "DONE", 12: "DONE"})
    assert (
        worker._pop_done_transfers(worker._recving_transfers, failed_req_ids=failed)
        == set()
    )
    assert worker._recving_transfers["request"] == [13, 14]
    assert batches.prepared_count == 5
    assert not state.exhausted

    wrapper.states.update({13: "DONE", 14: "DONE"})
    assert worker._pop_done_transfers(
        worker._recving_transfers, failed_req_ids=failed
    ) == {"request"}
    assert failed == set()
    assert batches.closed
    assert "request" not in worker._recving_transfers
    assert "request" not in worker._direct_read_batch_windows
    assert wrapper.released == [10, 11, 12, 13, 14]


@pytest.mark.parametrize("failure_kind", ["prepare", "submit"])
def test_direct_read_window_midstream_failure_stops_refill_and_drains_siblings(
    failure_kind: str,
):
    wrapper = _FakeNixlWrapper(
        fail_submit_handle=22 if failure_kind == "submit" else None
    )
    worker = _worker(wrapper, max_inflight=2)
    batches = _PreparedBatchIterator(
        [20, 21, 22, 23], fail_at=2 if failure_kind == "prepare" else None
    )
    state = _install_window(worker, "request", batches)
    worker._refill_direct_read_batch_windows()

    wrapper.states.update({20: "DONE", 21: "PROC"})
    failed: set[str] = set()
    assert (
        worker._pop_done_transfers(worker._recving_transfers, failed_req_ids=failed)
        == set()
    )
    assert failed == set()
    assert state.failed
    assert batches.closed
    assert worker._recving_transfers["request"] == [21]
    assert worker._request_terminal_poller.has_failed("recv", "request")
    assert wrapper.started == [20, 21]
    if failure_kind == "submit":
        assert 22 in wrapper.released

    wrapper.states[21] = "DONE"
    assert worker._pop_done_transfers(
        worker._recving_transfers, failed_req_ids=failed
    ) == {"request"}
    assert failed == {"request"}
    assert "request" not in worker._direct_read_batch_windows
    assert not worker._request_terminal_poller.has_failed("recv", "request")


def test_worker_global_credit_pool_does_not_delay_legacy_completion():
    wrapper = _FakeNixlWrapper()
    worker = _worker(wrapper, max_inflight=1)
    batches = _PreparedBatchIterator([10, 11])
    _install_window(worker, "direct", batches)
    worker._refill_direct_read_batch_windows()
    worker._recving_transfers["legacy"] = [99]
    wrapper.states.update({10: "PROC", 99: "DONE"})

    assert worker._pop_done_transfers(worker._recving_transfers) == {"legacy"}
    assert worker._recving_transfers["direct"] == [10]
    assert batches.prepared_count == 1
    assert wrapper.released == [99]


def test_stale_engine_eviction_skips_endpoint_pinned_by_receive(monkeypatch):
    worker = object.__new__(NixlPullConnectorWorker)
    worker._engine_ttl = 10.0
    worker._engine_last_active = {"active": 0.0, "stale": 0.0}
    worker._stale_remote_engines = set()
    worker._recving_metadata = {
        "request": SimpleNamespace(
            remote=SimpleNamespace(engine_id="active"),
        )
    }
    evicted: list[str] = []
    worker._cleanup_remote_engine = evicted.append
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.time.perf_counter",
        lambda: 20.0,
    )

    worker._evict_stale_engines()

    assert evicted == ["stale"]
