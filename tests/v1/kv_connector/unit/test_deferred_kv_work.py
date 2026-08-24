# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPullConnector,
)
from vllm.v1.worker import kv_connector_model_runner_mixin as mrv1_kv
from vllm.v1.worker.gpu import kv_connector as mrv2_kv

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )
    from vllm.forward_context import ForwardContext

pytestmark = pytest.mark.skip_global_cleanup


class _RecordingConnector(KVConnectorBase_V1):
    def __init__(self, events: list[str], prefix: str = "") -> None:
        self.events = events
        self.prefix = prefix
        self._connector_metadata = None

    def _record(self, event: str) -> None:
        self.events.append(f"{self.prefix}{event}")

    def handle_preemptions(self, kv_connector_metadata) -> None:
        self._record("preempt")

    def bind_connector_metadata(self, connector_metadata) -> None:
        self._record("bind")
        super().bind_connector_metadata(connector_metadata)

    def clear_connector_metadata(self) -> None:
        self._record("clear")
        super().clear_connector_metadata()

    def start_load_kv(self, forward_context, **kwargs) -> None:
        self._record("start")

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        pass

    def wait_for_save(self) -> None:
        self._record("wait")

    def get_finished(self, finished_req_ids):
        self._record("poll")
        return set(), set()

    def get_block_ids_with_load_errors(self) -> set[int]:
        return set()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens) -> None:
        pass

    def build_connector_meta(self, scheduler_output):
        return scheduler_output.kv_connector_metadata


class _DeferredConnector(_RecordingConnector):
    def start_deferred_kv_work(self, finished_req_ids: set[str]) -> None:
        self._record("deferred")


class _DeferredOnlyConnector(_DeferredConnector):
    def start_load_kv(self, forward_context, **kwargs) -> None:
        pass


def _scheduler_output():
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids={"finished"},
    )


def _make_mrv2_connector(monkeypatch, connector):
    monkeypatch.setattr(mrv2_kv, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(mrv2_kv, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mrv2_kv, "get_forward_context", lambda: object())
    return mrv2_kv.ActiveKVConnector(cast("VllmConfig", SimpleNamespace()), {})


@pytest.mark.parametrize("deferred", [False, True])
def test_mrv2_preserves_pre_work_and_orders_deferred_work(monkeypatch, deferred):
    events: list[str] = []
    connector_cls = _DeferredConnector if deferred else _RecordingConnector
    active = _make_mrv2_connector(monkeypatch, connector_cls(events))
    scheduler_output = _scheduler_output()

    active.pre_forward(scheduler_output)
    events.append("forward")
    active.post_forward(scheduler_output.finished_req_ids)

    assert events.index("start") < events.index("forward")
    assert events.index("wait") < events.index("poll")
    if deferred:
        assert events.index("forward") < events.index("deferred")
        assert events.index("deferred") < events.index("wait")
    else:
        assert "deferred" not in events


def test_mrv2_no_forward_runs_each_phase_once(monkeypatch):
    events: list[str] = []
    active = _make_mrv2_connector(monkeypatch, _DeferredConnector(events))

    active.no_forward(_scheduler_output())

    assert events.count("start") == 1
    assert events.count("deferred") == 1
    assert events.count("poll") == 1
    assert "wait" not in events


def test_mrv1_runs_deferred_work_after_context_body(monkeypatch):
    events: list[str] = []
    connector = _DeferredConnector(events)
    monkeypatch.setattr(mrv1_kv, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(mrv1_kv, "get_forward_context", lambda: object())

    with mrv1_kv.KVConnectorModelRunnerMixin._get_kv_connector_output(
        _scheduler_output()
    ):
        events.append("forward")

    assert events.index("start") < events.index("forward")
    assert events.index("forward") < events.index("deferred")
    assert events.index("deferred") < events.index("wait")
    assert events.index("wait") < events.index("poll")


def test_mrv1_no_forward_runs_each_phase_once(monkeypatch):
    events: list[str] = []
    connector = _DeferredConnector(events)
    monkeypatch.setattr(mrv1_kv, "get_kv_transfer_group", lambda: connector)
    monkeypatch.setattr(mrv1_kv, "get_forward_context", lambda: object())
    monkeypatch.setattr(mrv1_kv, "set_forward_context", lambda *args: nullcontext())

    mrv1_kv.KVConnectorModelRunnerMixin.kv_connector_no_forward(
        _scheduler_output(), cast("VllmConfig", SimpleNamespace())
    )

    assert events.count("start") == 1
    assert events.count("deferred") == 1
    assert events.count("poll") == 1
    assert "wait" not in events


def test_multi_connector_supports_mixed_pre_and_deferred_children():
    events: list[str] = []
    connector = object.__new__(MultiConnector)
    connector._connectors = [
        _RecordingConnector(events, "sync:"),
        _DeferredOnlyConnector(events, "async:"),
    ]

    connector.start_load_kv_before_forward(cast("ForwardContext", object()))
    events.append("forward")
    connector.start_deferred_kv_work({"finished"})

    assert events == ["sync:start", "forward", "async:deferred"]


@pytest.mark.parametrize(
    ("role", "use_host_buffer", "pipeline_size", "runner_type", "deferred"),
    [
        pytest.param("kv_consumer", False, 1, "generate", True, id="safe"),
        pytest.param("kv_producer", False, 1, "generate", False, id="producer"),
        pytest.param("kv_consumer", True, 1, "generate", False, id="host-buffer"),
        pytest.param("kv_consumer", False, 2, "generate", False, id="pipeline"),
        pytest.param("kv_consumer", False, 1, "pooling", False, id="pooling"),
    ],
)
def test_nixl_pull_defers_only_for_safe_consumer_mode(
    role, use_host_buffer, pipeline_size, runner_type, deferred
):
    connector = object.__new__(NixlPullConnector)
    connector.kv_transfer_config = cast(
        "KVTransferConfig", SimpleNamespace(kv_role=role)
    )
    connector.connector_worker = cast(
        "NixlBaseConnectorWorker",
        SimpleNamespace(use_host_buffer=use_host_buffer),
    )
    connector._vllm_config = cast(
        "VllmConfig",
        SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=pipeline_size),
            model_config=SimpleNamespace(runner_type=runner_type),
        ),
    )
    start_load = MagicMock()
    object.__setattr__(connector, "_start_load_kv", start_load)

    connector.start_load_kv_before_forward(cast("ForwardContext", object()))
    assert start_load.call_count == (0 if deferred else 1)

    connector.start_deferred_kv_work({"finished"})
    assert start_load.call_count == 1

    connector.start_load_kv(cast("ForwardContext", object()))
    assert start_load.call_count == 2
