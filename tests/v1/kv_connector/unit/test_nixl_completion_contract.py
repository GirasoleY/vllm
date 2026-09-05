# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge import (
    NixlDirectCompletionEnvelope,
)
from vllm.distributed.kv_transfer.transfer_completion import (
    KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
    CompletionStatus,
    TransferCompletionNotification,
    WorkerIdentity,
)


def _worker(participants: tuple[WorkerIdentity, ...]) -> NixlPullConnectorWorker:
    worker = object.__new__(NixlPullConnectorWorker)
    worker._reqs_to_send = {"prefill-request": float("inf")}
    worker._reqs_to_process = {"prefill-request"}
    worker._expected_direct_transfer_ids = {"prefill-request": "attempt-1"}
    worker._expected_direct_participant_counts = {"prefill-request": len(participants)}
    worker._expected_direct_participants = {"prefill-request": participants}
    worker._direct_completion_trackers = {}
    worker._direct_completion_participant_digests = {}
    worker._direct_completion_sender_bindings = {}
    worker.consumer_notification_counts_by_req = defaultdict(int)
    worker.expected_consumer_notifications_by_req = {}
    return worker


def _payload(
    sender: WorkerIdentity,
    participants: tuple[WorkerIdentity, ...],
) -> bytes:
    return NixlDirectCompletionEnvelope(
        notification=TransferCompletionNotification(
            version=KV_TRANSFER_COMPLETION_PROTOCOL_VERSION,
            request_id="prefill-request",
            deployment_id="decode",
            topology_generation=7,
            transfer_id="attempt-1",
            plan_digest="sha256:plan",
            sender_worker_id=sender.worker_id,
            sender_worker_incarnation=sender.worker_incarnation,
            expected_participant_count=len(participants),
            status=CompletionStatus.COMPLETE,
        ),
        expected_participants=participants,
    ).encode()


def test_consumer_cannot_author_the_roster_that_releases_producer_pages():
    authorized = (
        WorkerIdentity("decode-0", "decode-agent-0"),
        WorkerIdentity("decode-1", "decode-agent-1"),
    )
    forged = (
        WorkerIdentity("forged-0", "forged-agent-0"),
        WorkerIdentity("forged-1", "forged-agent-1"),
    )
    worker = _worker(authorized)
    notified: set[str] = set()

    for sender in forged:
        worker._handle_direct_completion(
            _payload(sender, forged),
            notified,
            sender_agent=sender.worker_incarnation,
        )

    assert notified == set()
    assert "prefill-request" in worker._reqs_to_send
    assert worker._direct_completion_trackers == {}


def test_exact_producer_owned_roster_can_complete_the_lease_barrier():
    authorized = (
        WorkerIdentity("decode-0", "decode-agent-0"),
        WorkerIdentity("decode-1", "decode-agent-1"),
    )
    worker = _worker(authorized)
    notified: set[str] = set()

    for sender in authorized:
        worker._handle_direct_completion(
            _payload(sender, authorized),
            notified,
            sender_agent=sender.worker_incarnation,
        )

    assert notified == {"prefill-request"}
    assert "prefill-request" not in worker._reqs_to_send
    assert "prefill-request" not in worker._reqs_to_process
    assert worker._expected_direct_participants == {}
