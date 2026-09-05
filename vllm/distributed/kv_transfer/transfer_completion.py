# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completion protocol for connector-independent segmented KV transfers."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

KV_TRANSFER_COMPLETION_PROTOCOL_VERSION = 2
MAX_TRANSFER_PARTICIPANTS = 4096


def _require_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string")


def _require_nonnegative_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _wire_fields(value: object, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("completion notification must be an object")
    data = dict(value)
    keys = set(data)
    if keys != fields:
        missing = sorted(fields - keys)
        unknown = sorted(keys - fields)
        raise ValueError(
            "invalid completion notification fields: "
            f"missing={missing}, unknown={unknown}"
        )
    return data


@dataclass(frozen=True)
class WorkerIdentity:
    """Stable worker process identity within a topology generation."""

    worker_id: str
    worker_incarnation: str

    def __post_init__(self) -> None:
        _require_string(self.worker_id, "worker_id")
        _require_string(self.worker_incarnation, "worker_incarnation")


def participant_set_digest(participants: Sequence[WorkerIdentity]) -> str:
    """Return an order-independent digest of an exact worker-incarnation set."""
    values = tuple(participants)
    if not values:
        raise ValueError("participants must not be empty")
    if len(values) > MAX_TRANSFER_PARTICIPANTS:
        raise ValueError(
            f"participants must contain at most {MAX_TRANSFER_PARTICIPANTS} workers"
        )
    if any(not isinstance(item, WorkerIdentity) for item in values):
        raise ValueError("participants must contain WorkerIdentity values")
    worker_ids = {item.worker_id for item in values}
    if len(worker_ids) != len(values):
        raise ValueError("participant worker IDs must be unique")

    canonical = json.dumps(
        sorted((item.worker_id, item.worker_incarnation) for item in values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{sha256(canonical).hexdigest()}"


def worker_identities_to_wire(
    participants: Sequence[WorkerIdentity],
) -> tuple[dict[str, str], ...]:
    """Return a strict request-metadata representation of a worker roster."""
    values = tuple(participants)
    participant_set_digest(values)
    return tuple(
        {
            "worker_id": participant.worker_id,
            "worker_incarnation": participant.worker_incarnation,
        }
        for participant in values
    )


def worker_identities_from_wire(
    value: object,
    *,
    name: str = "participants",
) -> tuple[WorkerIdentity, ...]:
    """Parse a bounded exact worker roster from request metadata."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array")
    if len(value) > MAX_TRANSFER_PARTICIPANTS:
        raise ValueError(
            f"{name} must contain at most {MAX_TRANSFER_PARTICIPANTS} workers"
        )
    participants: list[WorkerIdentity] = []
    for index, raw_participant in enumerate(value):
        if not isinstance(raw_participant, Mapping):
            raise ValueError(f"{name}[{index}] must be an object")
        fields = dict(raw_participant)
        if set(fields) != {"worker_id", "worker_incarnation"}:
            raise ValueError(
                f"{name}[{index}] must contain exactly worker_id and worker_incarnation"
            )
        participants.append(WorkerIdentity(**fields))
    values = tuple(participants)
    participant_set_digest(values)
    return values


class CompletionStatus(str, Enum):
    """Terminal status reported by one transfer participant."""

    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class TransferCompletionKey:
    """Identity of one immutable request-transfer attempt.

    ``transfer_id`` must be fresh for every retry, even when the scheduler request
    ID is reused. ``plan_digest`` makes every member of the producer-authorized
    destination roster report the same exact transfer plan. The source must not
    infer that authorized roster from a completion carrying this key.
    """

    request_id: str
    deployment_id: str
    topology_generation: int
    transfer_id: str
    plan_digest: str

    def __post_init__(self) -> None:
        _require_string(self.request_id, "request_id")
        _require_string(self.deployment_id, "deployment_id")
        _require_nonnegative_int(self.topology_generation, "topology_generation")
        _require_string(self.transfer_id, "transfer_id")
        _require_string(self.plan_digest, "plan_digest")


@dataclass(frozen=True)
class TransferCompletionNotification:
    """Versioned terminal notification from one request participant."""

    version: int
    request_id: str
    deployment_id: str
    topology_generation: int
    transfer_id: str
    plan_digest: str
    sender_worker_id: str
    sender_worker_incarnation: str
    expected_participant_count: int
    status: CompletionStatus

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.version, "version")
        if self.version != KV_TRANSFER_COMPLETION_PROTOCOL_VERSION:
            raise ValueError(f"unsupported completion protocol version {self.version}")
        _require_string(self.request_id, "request_id")
        _require_string(self.deployment_id, "deployment_id")
        _require_nonnegative_int(self.topology_generation, "topology_generation")
        _require_string(self.transfer_id, "transfer_id")
        _require_string(self.plan_digest, "plan_digest")
        _require_string(self.sender_worker_id, "sender_worker_id")
        _require_string(self.sender_worker_incarnation, "sender_worker_incarnation")
        _require_positive_int(
            self.expected_participant_count, "expected_participant_count"
        )
        if self.expected_participant_count > MAX_TRANSFER_PARTICIPANTS:
            raise ValueError(
                "expected_participant_count must be at most "
                f"{MAX_TRANSFER_PARTICIPANTS}"
            )
        if not isinstance(self.status, CompletionStatus):
            raise ValueError("status must be a CompletionStatus")

    @property
    def key(self) -> TransferCompletionKey:
        """Return the immutable attempt identity carried on the wire."""
        return TransferCompletionKey(
            request_id=self.request_id,
            deployment_id=self.deployment_id,
            topology_generation=self.topology_generation,
            transfer_id=self.transfer_id,
            plan_digest=self.plan_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON-compatible wire representation."""
        return {
            "version": self.version,
            "request_id": self.request_id,
            "deployment_id": self.deployment_id,
            "topology_generation": self.topology_generation,
            "transfer_id": self.transfer_id,
            "plan_digest": self.plan_digest,
            "sender_worker_id": self.sender_worker_id,
            "sender_worker_incarnation": self.sender_worker_incarnation,
            "expected_participant_count": self.expected_participant_count,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, value: object) -> "TransferCompletionNotification":
        """Parse a wire object, rejecting missing, unknown, or invalid fields."""
        data = _wire_fields(value, set(cls.__dataclass_fields__))
        try:
            status = CompletionStatus(data["status"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown completion status {data['status']!r}") from error
        data["status"] = status
        return cls(**data)


@dataclass(frozen=True)
class CompletionProgress:
    """Immutable aggregate state after processing a notification."""

    expected_participant_count: int
    received_participant_count: int
    completed_participant_count: int
    complete: bool
    failed: bool
    pending_worker_ids: tuple[str, ...]
    failed_worker_ids: tuple[str, ...]


class TransferCompletionTracker:
    """Aggregate terminal notifications for one deployment-side barrier."""

    def __init__(
        self,
        *,
        request_id: str,
        deployment_id: str,
        topology_generation: int,
        transfer_id: str,
        plan_digest: str,
        expected_participants: Sequence[WorkerIdentity],
    ) -> None:
        key = TransferCompletionKey(
            request_id=request_id,
            deployment_id=deployment_id,
            topology_generation=topology_generation,
            transfer_id=transfer_id,
            plan_digest=plan_digest,
        )
        participants = tuple(expected_participants)
        if not participants:
            raise ValueError("expected_participants must not be empty")
        if len(participants) > MAX_TRANSFER_PARTICIPANTS:
            raise ValueError(
                "expected_participants must contain at most "
                f"{MAX_TRANSFER_PARTICIPANTS} workers"
            )
        if any(not isinstance(item, WorkerIdentity) for item in participants):
            raise ValueError("expected_participants must contain WorkerIdentity values")
        expected = {item.worker_id: item for item in participants}
        if len(expected) != len(participants):
            raise ValueError("expected participant worker IDs must be unique")

        self.key = key
        self.request_id = key.request_id
        self.deployment_id = key.deployment_id
        self.topology_generation = key.topology_generation
        self.transfer_id = key.transfer_id
        self.plan_digest = key.plan_digest
        self._expected = expected
        self._received: dict[str, TransferCompletionNotification] = {}
        self._failed = False

    @property
    def progress(self) -> CompletionProgress:
        """Return the current aggregate state."""
        expected_ids = set(self._expected)
        received_ids = set(self._received)
        completed_ids = {
            worker_id
            for worker_id, notification in self._received.items()
            if notification.status is CompletionStatus.COMPLETE
        }
        failed_ids = {
            worker_id
            for worker_id, notification in self._received.items()
            if notification.status is CompletionStatus.FAILED
        }
        complete = (
            not self._failed
            and received_ids == expected_ids
            and completed_ids == expected_ids
        )
        return CompletionProgress(
            expected_participant_count=len(expected_ids),
            received_participant_count=len(received_ids),
            completed_participant_count=len(completed_ids),
            complete=complete,
            failed=self._failed,
            pending_worker_ids=tuple(sorted(expected_ids - received_ids)),
            failed_worker_ids=tuple(sorted(failed_ids)),
        )

    def record(
        self, notification: TransferCompletionNotification
    ) -> CompletionProgress:
        """Validate and aggregate one participant's terminal notification."""
        if not isinstance(notification, TransferCompletionNotification):
            raise ValueError("notification must be a TransferCompletionNotification")
        if notification.request_id != self.request_id:
            raise ValueError("completion notification has the wrong request ID")
        if notification.deployment_id != self.deployment_id:
            raise ValueError("completion notification has the wrong deployment ID")
        if notification.topology_generation != self.topology_generation:
            raise ValueError(
                "completion notification has a stale or wrong topology generation"
            )
        if notification.transfer_id != self.transfer_id:
            raise ValueError("completion notification has a stale or wrong transfer ID")
        if notification.plan_digest != self.plan_digest:
            raise ValueError("completion notification has the wrong plan digest")
        if notification.expected_participant_count != len(self._expected):
            raise ValueError(
                "completion notification has the wrong expected participant count"
            )
        expected = self._expected.get(notification.sender_worker_id)
        if expected is None:
            raise ValueError("completion notification has an unexpected sender")
        if notification.sender_worker_incarnation != expected.worker_incarnation:
            raise ValueError(
                "completion notification has a stale or wrong worker incarnation"
            )

        previous = self._received.get(notification.sender_worker_id)
        if previous is not None:
            if previous == notification:
                return self.progress
            self._failed = True
            raise ValueError("sender reported conflicting completion states")

        self._received[notification.sender_worker_id] = notification
        if notification.status is CompletionStatus.FAILED:
            self._failed = True
        return self.progress


__all__ = [
    "CompletionProgress",
    "CompletionStatus",
    "KV_TRANSFER_COMPLETION_PROTOCOL_VERSION",
    "MAX_TRANSFER_PARTICIPANTS",
    "TransferCompletionKey",
    "TransferCompletionNotification",
    "TransferCompletionTracker",
    "WorkerIdentity",
    "participant_set_digest",
    "worker_identities_from_wire",
    "worker_identities_to_wire",
]
