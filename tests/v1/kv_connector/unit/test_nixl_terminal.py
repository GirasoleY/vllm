# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.terminal import (
    NixlRequestTerminalPoller,
    NixlTransferFailure,
)


class _States:
    def __init__(self, states: dict[int, list[str]]):
        self.states = {handle: list(values) for handle, values in states.items()}

    def check(self, handle: int) -> str:
        return self.states[handle].pop(0)


def test_failed_batch_waits_for_proc_sibling_before_terminal_failure():
    poller = NixlRequestTerminalPoller()
    transfers = {"request": [10, 11, 12]}
    states = _States({10: ["ERR"], 11: ["PROC", "DONE"], 12: ["DONE"]})
    released: list[int] = []
    failures: list[tuple[int, NixlTransferFailure, bool]] = []

    def on_failed(
        _request_id: str,
        handle: int,
        failure: NixlTransferFailure,
        is_first: bool,
    ) -> None:
        failures.append((handle, failure, is_first))
        released.append(handle)

    first = poller.poll(
        transfers,
        stream="recv",
        check_state=states.check,
        on_done=released.append,
        on_failed=on_failed,
    )

    assert first.terminal_requests == frozenset()
    assert first.failed_requests == frozenset()
    assert first.successful_requests == frozenset()
    assert transfers == {"request": [11]}
    assert poller.has_failed("recv", "request")
    assert released == [10, 12]
    failure_summary = [
        (handle, failure.state, is_first) for handle, failure, is_first in failures
    ]
    assert failure_summary == [(10, "ERR", True)]

    second = poller.poll(
        transfers,
        stream="recv",
        check_state=states.check,
        on_done=released.append,
        on_failed=lambda *_: None,
    )

    assert second.terminal_requests == frozenset({"request"})
    assert second.failed_requests == frozenset({"request"})
    assert second.successful_requests == frozenset()
    assert transfers == {}
    assert not poller.has_failed("recv", "request")
    assert released == [10, 12, 11]


def test_terminal_request_is_not_reported_twice_or_released_twice():
    poller = NixlRequestTerminalPoller()
    transfers = {"request": [1, 2]}
    states = _States({1: ["DONE"], 2: ["DONE"]})
    released: list[int] = []

    first = poller.poll(
        transfers,
        stream="recv",
        check_state=states.check,
        on_done=released.append,
        on_failed=lambda *_: None,
    )
    second = poller.poll(
        transfers,
        stream="recv",
        check_state=states.check,
        on_done=released.append,
        on_failed=lambda *_: None,
    )

    assert first.terminal_requests == frozenset({"request"})
    assert first.successful_requests == frozenset({"request"})
    assert second.terminal_requests == frozenset()
    assert released == [1, 2]


def test_multiple_failed_siblings_invalidate_once_but_cleanup_every_handle():
    poller = NixlRequestTerminalPoller()
    transfers = {"request": [1, 2, 3]}
    states = _States({1: ["ERR"], 2: ["CANCELLED"], 3: ["DONE"]})
    released: list[int] = []
    first_failure_flags: list[bool] = []

    def on_failed(
        _request_id: str,
        handle: int,
        _failure: NixlTransferFailure,
        is_first: bool,
    ) -> None:
        first_failure_flags.append(is_first)
        released.append(handle)

    result = poller.poll(
        transfers,
        stream="recv",
        check_state=states.check,
        on_done=released.append,
        on_failed=on_failed,
    )

    assert result.failed_requests == frozenset({"request"})
    assert result.successful_requests == frozenset()
    assert first_failure_flags == [True, False]
    assert released == [1, 2, 3]
    assert transfers == {}


def test_equal_request_ids_are_isolated_between_send_and_receive_streams():
    poller = NixlRequestTerminalPoller()
    released: dict[str, list[int]] = defaultdict(list)
    recv = {"same-id": [1, 2]}
    send = {"same-id": [3]}
    states = _States({1: ["ERR"], 2: ["PROC", "DONE"], 3: ["DONE"]})

    recv_first = poller.poll(
        recv,
        stream="recv",
        check_state=states.check,
        on_done=released["recv"].append,
        on_failed=lambda _, handle, _failure, _first: released["recv"].append(handle),
    )
    send_result = poller.poll(
        send,
        stream="send",
        check_state=states.check,
        on_done=released["send"].append,
        on_failed=lambda *_: None,
    )

    assert recv_first.terminal_requests == frozenset()
    assert send_result.successful_requests == frozenset({"same-id"})
    assert poller.has_failed("recv", "same-id")
    assert not poller.has_failed("send", "same-id")


def test_submission_failure_waits_for_published_sibling_handle():
    poller = NixlRequestTerminalPoller()
    transfers = {"request": [7]}
    states = _States({7: ["PROC", "DONE"]})
    released: list[int] = []

    assert poller.mark_failed("send", "request")
    assert not poller.mark_failed("send", "request")

    first = poller.poll(
        transfers,
        stream="send",
        check_state=states.check,
        on_done=released.append,
        on_failed=lambda *_: None,
    )
    assert first.terminal_requests == frozenset()
    assert transfers == {"request": [7]}
    assert released == []

    second = poller.poll(
        transfers,
        stream="send",
        check_state=states.check,
        on_done=released.append,
        on_failed=lambda *_: None,
    )
    assert second.terminal_requests == frozenset({"request"})
    assert second.failed_requests == frozenset({"request"})
    assert second.successful_requests == frozenset()
    assert released == [7]
    assert transfers == {}


def test_submission_failure_with_no_handle_terminalizes_once():
    poller = NixlRequestTerminalPoller()
    transfers: dict[str, list[int]] = {"request": []}

    assert poller.mark_failed("recv", "request")
    first = poller.poll(
        transfers,
        stream="recv",
        check_state=lambda _: "DONE",
        on_done=lambda _: None,
        on_failed=lambda *_: None,
    )
    second = poller.poll(
        transfers,
        stream="recv",
        check_state=lambda _: "DONE",
        on_done=lambda _: None,
        on_failed=lambda *_: None,
    )

    assert first.failed_requests == frozenset({"request"})
    assert first.successful_requests == frozenset()
    assert second.terminal_requests == frozenset()
    assert transfers == {}
