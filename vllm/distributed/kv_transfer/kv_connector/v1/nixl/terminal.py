# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-level terminal aggregation for multi-transfer NIXL operations."""

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class NixlTransferFailure:
    """The terminal state or exception that failed one transfer handle."""

    state: str | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        if (self.state is None) == (self.error is None):
            raise ValueError("exactly one of state or error must identify a failure")


@dataclass(frozen=True)
class NixlRequestPollResult:
    """Requests whose complete sibling-handle set reached terminal state."""

    terminal_requests: frozenset[str]
    failed_requests: frozenset[str]

    def __post_init__(self) -> None:
        if not self.failed_requests <= self.terminal_requests:
            raise ValueError("failed requests must be terminal")

    @property
    def successful_requests(self) -> frozenset[str]:
        """Terminal requests eligible for an aggregate success notification."""
        return self.terminal_requests - self.failed_requests


class NixlRequestTerminalPoller:
    """Poll handles while latching failure at request granularity.

    A failed handle is released immediately, but its request is not returned as
    terminal until every sibling is also DONE or failed.  The failure latch is
    keyed by stream so equal request IDs in receive and send maps cannot
    interfere.
    """

    def __init__(self) -> None:
        self._failed_requests: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def has_failed(self, stream: str, request_id: str) -> bool:
        """Return whether a nonterminal request has latched a failure."""
        with self._lock:
            return (stream, request_id) in self._failed_requests

    def mark_failed(self, stream: str, request_id: str) -> bool:
        """Latch an out-of-band batch failure for a request.

        Some failures happen while submitting a batch, before its handle can
        be published in the request's in-flight handle list.  Latching that
        failure here lets the normal poll barrier wait for already-submitted
        siblings (or terminalize an intentionally empty list) without
        reporting the request early.

        Returns whether this is the first failed batch for the request.
        """
        if not isinstance(stream, str) or not stream:
            raise ValueError("stream must be a non-empty string")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a non-empty string")
        with self._lock:
            key = (stream, request_id)
            first_failure = key not in self._failed_requests
            self._failed_requests.add(key)
        return first_failure

    def poll(
        self,
        transfers: dict[str, list[int]],
        *,
        stream: str,
        check_state: Callable[[int], str],
        on_done: Callable[[int], None],
        on_failed: Callable[[str, int, NixlTransferFailure, bool], None],
    ) -> NixlRequestPollResult:
        """Poll each handle once and mutate ``transfers`` to retain only PROC.

        ``on_failed`` receives whether this is the first failed sibling, which
        lets the worker invalidate request blocks once while still recording
        and releasing every failed transfer.
        """
        if not isinstance(stream, str) or not stream:
            raise ValueError("stream must be a non-empty string")

        terminal_requests: set[str] = set()
        failed_requests: set[str] = set()
        for request_id, handles in list(transfers.items()):
            in_progress: list[int] = []
            for handle in handles:
                try:
                    state = check_state(handle)
                except Exception as error:
                    self._fail_handle(
                        stream,
                        request_id,
                        handle,
                        NixlTransferFailure(error=error),
                        on_failed,
                    )
                    continue

                if state == "PROC":
                    in_progress.append(handle)
                    continue
                if state == "DONE":
                    try:
                        on_done(handle)
                    except Exception as error:
                        self._fail_handle(
                            stream,
                            request_id,
                            handle,
                            NixlTransferFailure(error=error),
                            on_failed,
                        )
                    continue

                self._fail_handle(
                    stream,
                    request_id,
                    handle,
                    NixlTransferFailure(state=state),
                    on_failed,
                )

            if in_progress:
                transfers[request_id] = in_progress
                continue

            del transfers[request_id]
            terminal_requests.add(request_id)
            with self._lock:
                key = (stream, request_id)
                if key in self._failed_requests:
                    failed_requests.add(request_id)
                    self._failed_requests.remove(key)

        return NixlRequestPollResult(
            terminal_requests=frozenset(terminal_requests),
            failed_requests=frozenset(failed_requests),
        )

    def clear(self) -> None:
        """Forget failure latches during worker shutdown."""
        with self._lock:
            self._failed_requests.clear()

    def _fail_handle(
        self,
        stream: str,
        request_id: str,
        handle: int,
        failure: NixlTransferFailure,
        on_failed: Callable[[str, int, NixlTransferFailure, bool], None],
    ) -> None:
        first_failure = self.mark_failed(stream, request_id)
        on_failed(request_id, handle, failure, first_failure)


__all__ = [
    "NixlRequestPollResult",
    "NixlRequestTerminalPoller",
    "NixlTransferFailure",
]
