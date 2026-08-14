# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import time
from pathlib import Path
from threading import Thread
from unittest.mock import Mock, patch

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import (
    UnreadyWorkerProcHandle,
    WorkerProc,
    WorkerProcHandle,
)


def _response_queue_worker(control) -> None:
    queue = MessageQueue(
        n_reader=1,
        n_local_reader=1,
        max_chunk_bytes=1 << 20,
        max_chunks=4,
    )
    handle = queue.export_handle()
    control.send({"status": WorkerProc.MQ_READY_STR, "handle": handle})
    assert control.recv() == {"status": WorkerProc.MQ_ATTACHED_STR}

    # Once the parent has attached, removing the POSIX name must not
    # invalidate either process's existing mapping.
    assert queue.buffer is not None
    queue.buffer.shared_memory.unlink()
    queue.buffer.is_creator = False
    control.send({"status": WorkerProc.READY_STR})

    queue.wait_until_ready()
    queue.enqueue({"probe": 1}, timeout=5)
    queue.shutdown()
    control.close()


def _unready_handle(rank: int = 0):
    parent_pipe, worker_pipe = multiprocessing.Pipe(duplex=True)
    handle = UnreadyWorkerProcHandle(
        proc=Mock(),
        rank=rank,
        ready_pipe=parent_pipe,
        death_writer=None,
    )
    return handle, worker_pipe


def test_wait_for_ready_acknowledges_mq_attachment_before_final_ready() -> None:
    unready, worker_pipe = _unready_handle()
    attached_handle = WorkerProcHandle(
        proc=unready.proc,
        rank=unready.rank,
        worker_response_mq=None,
        peer_worker_response_mqs=[],
        death_writer=None,
    )
    events: list[str] = []

    def worker_protocol() -> None:
        worker_pipe.send(
            {
                "status": WorkerProc.MQ_READY_STR,
                "handle": object(),
                "peer_response_handles": [],
            }
        )
        acknowledgement = worker_pipe.recv()
        assert acknowledgement == {"status": WorkerProc.MQ_ATTACHED_STR}
        events.append("acknowledged")
        worker_pipe.send({"status": WorkerProc.READY_STR})
        worker_pipe.close()

    def attach_response_queues(*args, **kwargs) -> WorkerProcHandle:
        events.append("attached")
        return attached_handle

    worker_thread = Thread(target=worker_protocol)
    worker_thread.start()
    with patch.object(
        WorkerProc,
        "wait_for_response_handle_ready",
        side_effect=attach_response_queues,
    ) as attach:
        ready = WorkerProc.wait_for_ready([unready])
    worker_thread.join(timeout=5)

    assert not worker_thread.is_alive()
    assert ready == [attached_handle]
    assert events == ["attached", "acknowledged"]
    attach.assert_called_once()


def test_wait_for_ready_rejects_eof_after_mq_attachment() -> None:
    unready, worker_pipe = _unready_handle()
    attached_handle = WorkerProcHandle(
        proc=unready.proc,
        rank=unready.rank,
        worker_response_mq=None,
        peer_worker_response_mqs=[],
        death_writer=None,
    )

    def worker_protocol() -> None:
        worker_pipe.send(
            {
                "status": WorkerProc.MQ_READY_STR,
                "handle": object(),
                "peer_response_handles": [],
            }
        )
        assert worker_pipe.recv() == {"status": WorkerProc.MQ_ATTACHED_STR}
        worker_pipe.close()

    worker_thread = Thread(target=worker_protocol)
    worker_thread.start()
    with (
        patch.object(
            WorkerProc,
            "wait_for_response_handle_ready",
            return_value=attached_handle,
        ),
        pytest.raises(Exception, match="WorkerProc initialization failed"),
    ):
        WorkerProc.wait_for_ready([unready])
    worker_thread.join(timeout=5)
    assert not worker_thread.is_alive()


def test_wait_for_ready_rejects_final_ready_without_mq_handoff() -> None:
    unready, worker_pipe = _unready_handle()
    worker_pipe.send({"status": WorkerProc.READY_STR})

    with pytest.raises(Exception, match="WorkerProc initialization failed"):
        WorkerProc.wait_for_ready([unready])
    worker_pipe.close()


def test_response_queue_mapping_survives_unlink_after_acknowledged_handoff() -> None:
    context = multiprocessing.get_context("spawn")
    parent_control, worker_control = context.Pipe(duplex=True)
    worker = context.Process(target=_response_queue_worker, args=(worker_control,))
    worker.start()
    worker_control.close()

    assert parent_control.poll(10)
    mq_ready = parent_control.recv()
    assert mq_ready["status"] == WorkerProc.MQ_READY_STR
    handle = mq_ready["handle"]
    assert handle.buffer_handle is not None
    shm_path = Path("/dev/shm") / handle.buffer_handle[3].lstrip("/")
    assert shm_path.exists()

    reader = MessageQueue.create_from_handle(handle, rank=0)
    parent_control.send({"status": WorkerProc.MQ_ATTACHED_STR})
    assert parent_control.poll(10)
    assert parent_control.recv() == {"status": WorkerProc.READY_STR}
    assert not shm_path.exists()

    reader.wait_until_ready()
    assert reader.dequeue(timeout=5) == {"probe": 1}
    reader.shutdown()
    parent_control.close()

    worker.join(timeout=10)
    if worker.is_alive():
        worker.kill()
        worker.join()
    assert worker.exitcode == 0

    # Give multiprocessing's resource tracker enough time to report an
    # accidental double-unlink before the test exits.
    time.sleep(0.05)
