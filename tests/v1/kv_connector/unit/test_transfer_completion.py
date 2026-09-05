# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import FrozenInstanceError

import pytest

from vllm.distributed.kv_transfer.transfer_completion import (
    KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
    MAX_TRANSFER_PARTICIPANTS,
    CompletionStatus,
    TransferCompletionKey,
    TransferCompletionNotification,
    TransferCompletionTracker,
    WorkerIdentity,
    participant_set_digest,
    worker_identities_from_wire,
    worker_identities_to_wire,
)

_PLAN_DIGEST = "sha256:" + "4a" * 32


def test_worker_identity_roster_has_a_strict_bounded_wire_format():
    participants = (
        WorkerIdentity("decode-0", "agent-0"),
        WorkerIdentity("decode-1", "agent-1"),
    )
    wire = worker_identities_to_wire(participants)

    assert wire == (
        {"worker_id": "decode-0", "worker_incarnation": "agent-0"},
        {"worker_id": "decode-1", "worker_incarnation": "agent-1"},
    )
    assert worker_identities_from_wire(wire) == participants

    with pytest.raises(ValueError, match="must be an array"):
        worker_identities_from_wire({"worker_id": "decode-0"})
    with pytest.raises(ValueError, match="exactly worker_id"):
        worker_identities_from_wire(
            [{"worker_id": "decode-0", "worker_incarnation": "a", "extra": 1}]
        )
    with pytest.raises(ValueError, match="worker IDs must be unique"):
        worker_identities_from_wire(
            [
                {"worker_id": "decode-0", "worker_incarnation": "a"},
                {"worker_id": "decode-0", "worker_incarnation": "b"},
            ]
        )


def _notification(
    worker_id: str,
    *,
    status: CompletionStatus = CompletionStatus.COMPLETE,
    incarnation: str | None = None,
    request_id: str = "request-17",
    deployment_id: str = "decode",
    generation: int = 4,
    transfer_id: str = "transfer-23",
    plan_digest: str = _PLAN_DIGEST,
    expected_count: int = 3,
) -> TransferCompletionNotification:
    return TransferCompletionNotification(
        version=KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
        request_id=request_id,
        deployment_id=deployment_id,
        topology_generation=generation,
        transfer_id=transfer_id,
        plan_digest=plan_digest,
        sender_worker_id=worker_id,
        sender_worker_incarnation=incarnation or f"{worker_id}-boot",
        expected_participant_count=expected_count,
        status=status,
    )


def _tracker() -> TransferCompletionTracker:
    return TransferCompletionTracker(
        request_id="request-17",
        deployment_id="decode",
        topology_generation=4,
        transfer_id="transfer-23",
        plan_digest=_PLAN_DIGEST,
        expected_participants=tuple(
            WorkerIdentity(worker_id, f"{worker_id}-boot")
            for worker_id in ("worker-a", "worker-b", "worker-c")
        ),
    )


def test_completion_notification_has_a_strict_versioned_wire_format():
    notification = _notification("worker-a")
    payload = notification.to_dict()

    assert TransferCompletionNotification.from_dict(payload) == notification
    assert payload == {
        "version": 2,
        "request_id": "request-17",
        "deployment_id": "decode",
        "topology_generation": 4,
        "transfer_id": "transfer-23",
        "plan_digest": _PLAN_DIGEST,
        "sender_worker_id": "worker-a",
        "sender_worker_incarnation": "worker-a-boot",
        "expected_participant_count": 3,
        "status": "complete",
    }
    with pytest.raises(FrozenInstanceError):
        notification.status = CompletionStatus.FAILED

    with pytest.raises(ValueError, match="unknown=.*extra"):
        TransferCompletionNotification.from_dict(payload | {"extra": True})
    incomplete = dict(payload)
    del incomplete["request_id"]
    with pytest.raises(ValueError, match="missing=.*request_id"):
        TransferCompletionNotification.from_dict(incomplete)
    with pytest.raises(ValueError, match="unknown completion status"):
        TransferCompletionNotification.from_dict(payload | {"status": "pending"})
    with pytest.raises(ValueError, match="unsupported completion protocol"):
        TransferCompletionNotification.from_dict(payload | {"version": 1})
    with pytest.raises(ValueError, match="version must be a non-negative integer"):
        TransferCompletionNotification.from_dict(payload | {"version": True})


def test_tracker_deduplicates_senders_and_completes_on_the_exact_set():
    tracker = _tracker()

    progress = tracker.record(_notification("worker-b"))
    assert progress.received_participant_count == 1
    assert not progress.complete
    assert tracker.record(_notification("worker-b")) == progress

    assert not tracker.record(_notification("worker-a")).complete
    progress = tracker.record(_notification("worker-c"))
    assert progress.complete
    assert not progress.failed
    assert progress.received_participant_count == 3
    assert progress.completed_participant_count == 3
    assert progress.pending_worker_ids == ()


@pytest.mark.parametrize(
    ("notification", "message"),
    [
        (_notification("worker-a", request_id="other"), "request ID"),
        (_notification("worker-a", deployment_id="prefill"), "deployment ID"),
        (_notification("worker-a", generation=3), "topology generation"),
        (_notification("worker-a", transfer_id="old-attempt"), "transfer ID"),
        (
            _notification(
                "worker-a",
                plan_digest="sha256:" + "5b" * 32,
            ),
            "plan digest",
        ),
        (_notification("worker-a", expected_count=2), "participant count"),
        (_notification("worker-z"), "unexpected sender"),
        (
            _notification("worker-a", incarnation="previous-boot"),
            "worker incarnation",
        ),
    ],
)
def test_tracker_rejects_wrong_or_stale_notification_metadata(
    notification: TransferCompletionNotification,
    message: str,
):
    tracker = _tracker()

    with pytest.raises(ValueError, match=message):
        tracker.record(notification)

    assert tracker.progress.received_participant_count == 0
    assert not tracker.progress.failed


def test_failure_suppresses_completion_after_every_participant_reports():
    tracker = _tracker()

    tracker.record(_notification("worker-a"))
    tracker.record(_notification("worker-b", status=CompletionStatus.FAILED))
    progress = tracker.record(_notification("worker-c"))

    assert progress.received_participant_count == 3
    assert progress.completed_participant_count == 2
    assert progress.failed
    assert not progress.complete
    assert progress.failed_worker_ids == ("worker-b",)


def test_conflicting_duplicate_poisoning_suppresses_prior_completion():
    tracker = TransferCompletionTracker(
        request_id="request-17",
        deployment_id="decode",
        topology_generation=4,
        transfer_id="transfer-23",
        plan_digest=_PLAN_DIGEST,
        expected_participants=(WorkerIdentity("worker-a", "worker-a-boot"),),
    )
    complete = _notification("worker-a", expected_count=1)
    assert tracker.record(complete).complete

    with pytest.raises(ValueError, match="conflicting completion states"):
        tracker.record(
            _notification(
                "worker-a",
                expected_count=1,
                status=CompletionStatus.FAILED,
            )
        )

    assert tracker.progress.failed
    assert not tracker.progress.complete


def test_attempt_key_prevents_delayed_retry_notification_from_completing():
    tracker = TransferCompletionTracker(
        request_id="request-17",
        deployment_id="decode",
        topology_generation=4,
        transfer_id="retry-2",
        plan_digest="sha256:" + "6c" * 32,
        expected_participants=(WorkerIdentity("worker-a", "worker-a-boot"),),
    )

    delayed = _notification(
        "worker-a",
        expected_count=1,
        transfer_id="retry-1",
        plan_digest="sha256:" + "5b" * 32,
    )
    with pytest.raises(ValueError, match="transfer ID"):
        tracker.record(delayed)

    assert tracker.progress.received_participant_count == 0
    current = _notification(
        "worker-a",
        expected_count=1,
        transfer_id="retry-2",
        plan_digest="sha256:" + "6c" * 32,
    )
    assert tracker.record(current).complete


def test_completion_key_is_immutable_and_includes_attempt_and_plan_identity():
    notification = _notification("worker-a")

    assert notification.key == TransferCompletionKey(
        request_id="request-17",
        deployment_id="decode",
        topology_generation=4,
        transfer_id="transfer-23",
        plan_digest=_PLAN_DIGEST,
    )
    assert _tracker().key == notification.key


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transfer_id", ""),
        ("transfer_id", " padded "),
        ("plan_digest", ""),
        ("plan_digest", " padded "),
    ],
)
def test_completion_notification_rejects_noncanonical_attempt_identity(
    field: str, value: str
):
    payload = _notification("worker-a").to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        TransferCompletionNotification.from_dict(payload)


def test_participant_set_digest_is_stable_and_order_independent():
    participants = (
        WorkerIdentity("worker-b", "boot-2"),
        WorkerIdentity("worker-a", "boot-1"),
    )

    digest = participant_set_digest(participants)
    assert digest == participant_set_digest(tuple(reversed(participants)))
    assert digest.startswith("sha256:")
    assert len(digest.removeprefix("sha256:")) == 64
    assert digest != participant_set_digest(
        (
            WorkerIdentity("worker-a", "boot-1"),
            WorkerIdentity("worker-b", "boot-3"),
        )
    )


def test_participant_set_digest_rejects_invalid_sets():
    with pytest.raises(ValueError, match="must not be empty"):
        participant_set_digest(())
    with pytest.raises(ValueError, match="must contain WorkerIdentity"):
        participant_set_digest(("worker-a",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="worker IDs must be unique"):
        participant_set_digest(
            (
                WorkerIdentity("worker-a", "boot-1"),
                WorkerIdentity("worker-a", "boot-2"),
            )
        )


def test_completion_participant_limit_accepts_exact_boundary():
    participants = tuple(
        WorkerIdentity(f"worker-{index}", f"boot-{index}")
        for index in range(MAX_TRANSFER_PARTICIPANTS)
    )

    notification = _notification("worker-0", expected_count=MAX_TRANSFER_PARTICIPANTS)
    assert notification.expected_participant_count == MAX_TRANSFER_PARTICIPANTS
    assert participant_set_digest(participants).startswith("sha256:")
    tracker = TransferCompletionTracker(
        request_id="request-17",
        deployment_id="decode",
        topology_generation=4,
        transfer_id="transfer-23",
        plan_digest=_PLAN_DIGEST,
        expected_participants=participants,
    )
    assert tracker.progress.expected_participant_count == MAX_TRANSFER_PARTICIPANTS


def test_completion_participant_limit_rejects_above_boundary():
    with pytest.raises(ValueError, match="expected_participant_count.*at most"):
        _notification("worker-0", expected_count=MAX_TRANSFER_PARTICIPANTS + 1)

    participants = tuple(
        WorkerIdentity(f"worker-{index}", f"boot-{index}")
        for index in range(MAX_TRANSFER_PARTICIPANTS + 1)
    )
    with pytest.raises(ValueError, match="participants must contain at most"):
        participant_set_digest(participants)
    with pytest.raises(ValueError, match="expected_participants.*at most"):
        TransferCompletionTracker(
            request_id="request-17",
            deployment_id="decode",
            topology_generation=4,
            transfer_id="transfer-23",
            plan_digest=_PLAN_DIGEST,
            expected_participants=participants,
        )
