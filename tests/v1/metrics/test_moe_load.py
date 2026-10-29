# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Aggregation contracts and bounded-storage stress for online MoE telemetry."""

import json
from unittest.mock import patch

import pytest

from vllm.v1.metrics.moe_load import (
    MoeLoadCollector,
    MoeLoadPublisher,
    atomic_json,
    request_diagnostic,
)

pytestmark = pytest.mark.cpu_test

SYNTHETIC_EXPERTS = 24


@pytest.fixture
def global_registry(monkeypatch):
    from prometheus_client import REGISTRY

    from vllm.v1.metrics.prometheus import unregister_vllm_metrics

    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    unregister_vllm_metrics()
    try:
        yield REGISTRY
    finally:
        unregister_vllm_metrics()


def topology(world, rank, layers=3, experts=SYNTHETIC_EXPERTS):
    return [
        dict(
            prefix=f"model.layers.{i + 1}.mlp.experts",
            physical_experts=experts,
            local_experts=experts // world,
            first_expert=rank * (experts // world),
            ep_rank=rank,
            ep_world=world,
            top_k=2,
        )
        for i in range(layers)
    ]


def setup(tmp_path, world=4, layers=3):
    config = dict(
        directory=str(tmp_path),
        ep_world=world,
        interval_s=10,
        max_samples=3,
        target_layers=layers - 1,
        rank_metrics=True,
        collection_id="synthetic-collection",
    )
    publishers = [MoeLoadPublisher(config, f"run-{r}") for r in range(world)]
    collector = MoeLoadCollector(config)
    collector.started = 0
    return config, publishers, collector


def publish(publishers, stamp, scale=1, layers=3, experts=SYNTHETIC_EXPERTS):
    world = len(publishers)
    for rank, p in enumerate(publishers):
        counts = [
            [int(stamp) * scale * (rank + 1) + i] * (experts // world)
            for i in range(layers)
        ]
        p.publish(
            topology(world, rank, layers, experts),
            counts,
            dict(
                epoch_bucket=int(stamp // 10),
                enqueued_unix=stamp,
                layers=topology(world, rank, layers, experts),
                sequence=int(stamp // 10),
                baseline=False,
            ),
        )


def scrape(collector, now):
    with patch("vllm.v1.metrics.moe_load.time.time", return_value=now):
        return {
            (s.name, tuple(sorted(s.labels.items()))): s.value
            for m in collector.collect()
            for s in m.samples
        }


def value(metrics, name, **labels):
    return metrics.get(("vllm:moe_" + name, tuple(sorted(labels.items()))))


@pytest.mark.parametrize("world", [4, 8])
def test_native_assignment_balance_and_optional_rank_counter(tmp_path, world):
    config, publishers, collector = setup(tmp_path, world)
    publish(publishers, 1000)
    publish(publishers, 1010)
    data = scrape(collector, 1011)
    ratio = world / ((world + 1) / 2)
    assert value(data, "collection_complete") == 1
    assert value(data, "collection_age_seconds") == 1
    for part in ["target", "draft"]:
        assert value(data, "rank_imbalance_average", model_part=part) == ratio
        assert value(data, "rank_imbalance_worst", model_part=part) == ratio
    assert sum(k[0].endswith("assignments_total") for k in data) == 3 * world
    config["rank_metrics"] = False
    assert not any(k[0].endswith("assignments_total") for k in scrape(collector, 1011))


def test_worst_layer_is_not_hidden_by_rank_cancellation(tmp_path):
    _, publishers, collector = setup(tmp_path, 2, 2)
    for rank, p in enumerate(publishers):
        p.publish(
            topology(2, rank, 2),
            [[0] * (SYNTHETIC_EXPERTS // 2)] * 2,
            dict(epoch_bucket=100, enqueued_unix=1000),
        )
        p.publish(
            topology(2, rank, 2),
            [
                [3 if rank == 0 else 1] * (SYNTHETIC_EXPERTS // 2),
                [1 if rank == 0 else 3] * (SYNTHETIC_EXPERTS // 2),
            ],
            dict(epoch_bucket=101, enqueued_unix=1010),
        )
    collector.config["target_layers"] = 2
    data = scrape(collector, 1011)
    assert value(data, "rank_imbalance_worst", model_part="target") == 1.5
    assert value(data, "rank_imbalance_average", model_part="target") == 1.5


def test_wrap_baseline_and_idle_do_not_create_fake_load(tmp_path):
    _, publishers, collector = setup(tmp_path)
    for rank, p in enumerate(publishers):
        for epoch, n in [(100, (1 << 32) - 2), (101, 3)]:
            p.publish(
                topology(4, rank),
                [[n] * (SYNTHETIC_EXPERTS // 4)] * 3,
                dict(epoch_bucket=epoch, enqueued_unix=epoch * 10),
            )
    data = scrape(collector, 1011)
    assert value(
        data,
        "received_assignments_total",
        moe_layer="1",
        model_part="target",
        ep_rank="0",
    ) == 5 * (SYNTHETIC_EXPERTS // 4)
    for rank, p in enumerate(publishers):
        p.publish(
            topology(4, rank),
            [[3] * (SYNTHETIC_EXPERTS // 4)] * 3,
            dict(epoch_bucket=102, enqueued_unix=1020),
        )
    idle = scrape(collector, 1021)
    assert value(idle, "collection_complete") == 1
    assert value(idle, "rank_imbalance_worst", model_part="target") is None


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "stale",
        "malformed",
        "restart",
        "topology",
        "skew",
        "start_skew",
        "startup",
        "identity",
        "prefix",
        "decreasing",
    ],
)
def test_incomplete_or_stale_ranks_never_report_healthy_balance(tmp_path, fault):
    config, publishers, collector = setup(tmp_path)
    publish(publishers, 1000)
    publish(publishers, 1010)
    path = publishers[-1].root / "rank3.json"
    now = 1011
    if fault == "missing":
        path.unlink()
    elif fault == "stale":
        now = 1050
    elif fault == "malformed":
        path.write_text("{")
    elif fault == "restart":
        publishers[-1] = MoeLoadPublisher(config, "replacement")
        publish(publishers, 1020)
        now = 1021
    elif fault == "startup":
        collector.started = 1015
    else:
        data = json.loads(path.read_text())
        if fault == "topology":
            data["layers"][0]["ep_world"] = 8
        elif fault == "identity":
            data["collection_id"] = "different-collection"
        elif fault == "prefix":
            data["layers"][0]["prefix"] = "bad-prefix"
        elif fault == "decreasing":
            data["history"][-2]["counts"][0] = 1 << 50
        else:
            data["history"][-2 if fault == "start_skew" else -1]["time"] += 5
            now = 1016
        atomic_json(path, data)
    result = scrape(collector, now)
    assert value(result, "collection_complete") == 0
    assert value(result, "rank_imbalance_average", model_part="target") is None


def test_diagnostic_budget_and_rearming_never_stop_online_export(tmp_path):
    _, publishers, collector = setup(tmp_path)
    root = publishers[0].root
    for epoch in range(100, 1500):
        if epoch % 20 == 0:
            atomic_json(
                root / "diagnostic-request.json",
                dict(
                    id=f"capture-{epoch}",
                    start_unix=epoch * 10,
                    end_unix=(epoch + 10) * 10,
                    samples=3,
                ),
            )
        publish(publishers, epoch * 10)
    assert value(scrape(collector, 14991), "collection_complete") == 1
    for p in publishers:
        assert len(p.history) == 4
    assert len(list((root / "diagnostics").rglob("sample*.json"))) == 12
    assert len(list(root.glob("rank*.json"))) == 4
    assert not list(root.rglob("*.tmp"))


def test_diagnostic_expiry_does_not_stop_online_metrics(tmp_path):
    _, publishers, collector = setup(tmp_path)
    atomic_json(
        publishers[0].root / "diagnostic-request.json",
        dict(id="expired", start_unix=0, end_unix=999, samples=3),
    )
    publish(publishers, 1000)
    publish(publishers, 1010)
    assert not (publishers[0].root / "diagnostics").exists()
    assert value(scrape(collector, 1011), "collection_complete") == 1


def test_transient_publication_failure_is_recoverable(tmp_path):
    _, publishers, collector = setup(tmp_path)
    publish(publishers, 1000)
    with (
        patch(
            "vllm.v1.metrics.moe_load.atomic_json",
            side_effect=OSError("disk fault"),
        ),
        pytest.raises(OSError),
    ):
        publish(publishers, 1010)
    publish(publishers, 1020)
    publish(publishers, 1030)
    assert value(scrape(collector, 1031), "collection_complete") == 1


def test_diagnostic_command_has_a_budget_and_a_collection_baseline(tmp_path):
    config, publishers, _ = setup(tmp_path)
    with patch("vllm.v1.metrics.moe_load.time.time", return_value=1000):
        request_diagnostic(config, "qualification", 2)
    publish(publishers, 1000)
    publish(publishers, 1010)
    path = publishers[0].root / "diagnostics/rank0"
    files = sorted(path.glob("*.json"))
    assert len(files) == 2
    first, last = [json.loads(p.read_text()) for p in files]
    assert first["baseline"] and not last["baseline"]
    assert first["capture_id"] == last["capture_id"] == "qualification"
    assert first["host"] == last["host"]
    assert first["sequence"] < last["sequence"]
    assert first["epoch_bucket"] < last["epoch_bucket"]
    assert first["enqueued_unix"] < last["enqueued_unix"]
    assert first["collection_id"] == config["collection_id"]
    with pytest.raises(ValueError):
        request_diagnostic(config, "too-many", 4)


@pytest.mark.parametrize("already_registered", [False, True])
def test_frontend_registration_is_idempotent(
    tmp_path, monkeypatch, global_registry, already_registered
):
    from fastapi import FastAPI

    from vllm.entrypoints.serve.instrumentator.metrics import attach_router

    config, _, _ = setup(tmp_path)
    monkeypatch.setenv("VLLM_MOE_EXPERT_LOAD_CONFIG", json.dumps(config))
    if already_registered:
        global_registry.register(MoeLoadCollector(config))
    attach_router(FastAPI())
    attach_router(FastAPI())
    assert sum(m.name.startswith("vllm:moe_") for m in global_registry.collect()) == 3


def test_registry_lookup_does_not_opt_into_moe(monkeypatch, global_registry):
    from vllm.v1.metrics.prometheus import get_prometheus_registry

    monkeypatch.setenv("VLLM_MOE_EXPERT_LOAD_CONFIG", "not-json")
    assert get_prometheus_registry() is global_registry
    assert not any(m.name.startswith("vllm:moe_") for m in global_registry.collect())


@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("layers,experts", [(4, 16), (11, 24), (128, 40)])
def test_layer_and_expert_cardinality_is_bounded(tmp_path, world, layers, experts):
    _, publishers, collector = setup(tmp_path, world, layers)
    for stamp in [1000, 1010, 1020, 1030, 1040, 1050]:
        publish(publishers, stamp, layers=layers, experts=experts)
    metrics = scrape(collector, 1051)
    assert value(metrics, "collection_complete") == 1
    assert sum(k[0].endswith("assignments_total") for k in metrics) == layers * world
    assert max(p.stat().st_size for p in publishers[0].root.glob("rank*.json")) < 40000


def test_layer_average_is_assignment_weighted_and_preserves_the_worst(tmp_path):
    _, publishers, collector = setup(tmp_path, 2, 2)
    collector.config["target_layers"] = 2
    for rank, publisher in enumerate(publishers):
        for epoch, amounts in [(100, [0, 0]), (101, [1000, 100 if rank == 0 else 0])]:
            publisher.publish(
                topology(2, rank, 2),
                [[n] * (SYNTHETIC_EXPERTS // 2) for n in amounts],
                dict(epoch_bucket=epoch, enqueued_unix=epoch * 10),
            )
    metrics = scrape(collector, 1011)
    assert value(
        metrics, "rank_imbalance_average", model_part="target"
    ) == pytest.approx(22 / 21)
    assert value(metrics, "rank_imbalance_worst", model_part="target") == 2


def test_first_observation_is_fresh_but_not_yet_a_balance_window(tmp_path):
    _, publishers, collector = setup(tmp_path)
    publish(publishers, 1000)
    data = scrape(collector, 1001)
    assert value(data, "collection_complete") == 0
    assert value(data, "collection_age_seconds") == 1
    assert value(data, "rank_imbalance_average", model_part="target") is None


def test_concurrent_frontend_initialization_does_not_duplicate_metrics(
    tmp_path, monkeypatch, global_registry
):
    from concurrent.futures import ThreadPoolExecutor

    from fastapi import FastAPI

    from vllm.entrypoints.serve.instrumentator.metrics import attach_router

    config, _, _ = setup(tmp_path)
    monkeypatch.setenv("VLLM_MOE_EXPERT_LOAD_CONFIG", json.dumps(config))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: attach_router(FastAPI()), range(128)))
    assert sum(m.name.startswith("vllm:moe_") for m in global_registry.collect()) == 3


def test_concurrent_capture_requests_are_atomic_and_leave_no_tempfiles(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    config, publishers, _ = setup(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(
            pool.map(
                lambda i: request_diagnostic(config, f"capture-{i}", 2), range(128)
            )
        )
    request = json.loads(paths[0].read_text())
    assert request["samples"] == 2
    assert not list(publishers[0].root.rglob("*.tmp"))


def test_online_initialization_does_not_require_working_storage(tmp_path):
    config = dict(directory=str(tmp_path), ep_world=4, interval_s=10)
    with patch(
        "vllm.v1.metrics.moe_load.Path.mkdir",
        side_effect=OSError("unavailable storage"),
    ):
        publisher = MoeLoadPublisher(config, "run")
        with pytest.raises(OSError):
            publisher.publish(
                topology(4, 0),
                [[0] * (SYNTHETIC_EXPERTS // 4)] * 3,
                dict(epoch_bucket=100, enqueued_unix=1000),
            )


def test_default_model_part_includes_all_registered_layers(tmp_path):
    config, publishers, collector = setup(tmp_path, 2, 5)
    config.pop("target_layers")
    publish(publishers, 1000, layers=5)
    publish(publishers, 1010, layers=5)
    data = scrape(collector, 1011)
    assert value(data, "rank_imbalance_average", model_part="target") == 4 / 3
    assert value(data, "rank_imbalance_average", model_part="draft") is None


def test_default_capture_budget_matches_recorder_configuration(tmp_path):
    config, publishers, _ = setup(tmp_path)
    config.pop("max_samples")
    with patch("vllm.v1.metrics.moe_load.time.time", return_value=1000):
        request_diagnostic(config, "default-budget", 2)
    publish(publishers, 1000)
    publish(publishers, 1010)
    files = sorted(publishers[0].root.glob("diagnostics/rank0/sample*.json"))
    assert len(files) == 2
    assert json.loads(files[-1].read_text())["capture_id"] == "default-budget"


def test_collection_identifier_is_optional(tmp_path):
    config, publishers, collector = setup(tmp_path)
    config.pop("collection_id")
    publish(publishers, 1000)
    publish(publishers, 1010)
    assert value(scrape(collector, 1011), "collection_complete") == 1


@pytest.mark.parametrize("multiprocess", [False, True])
def test_registry_cleanup_can_start_a_new_collection(
    tmp_path, monkeypatch, global_registry, multiprocess
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.serve.instrumentator.metrics import attach_router
    from vllm.v1.metrics.prometheus import unregister_vllm_metrics

    if multiprocess:
        directory = tmp_path / "multiprocess"
        directory.mkdir()
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(directory))
    config, _, _ = setup(tmp_path / "first")
    monkeypatch.setenv("VLLM_MOE_EXPERT_LOAD_CONFIG", json.dumps(config))
    with patch("vllm.v1.metrics.moe_load.time.time", return_value=0):
        attach_router(FastAPI())
    unregister_vllm_metrics()
    config, publishers, _ = setup(tmp_path / "second")
    monkeypatch.setenv("VLLM_MOE_EXPERT_LOAD_CONFIG", json.dumps(config))
    app = FastAPI()
    with patch("vllm.v1.metrics.moe_load.time.time", return_value=0):
        attach_router(app)
    publish(publishers, 1000)
    publish(publishers, 1010)
    with (
        patch("vllm.v1.metrics.moe_load.time.time", return_value=1011),
        TestClient(app) as client,
    ):
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "vllm:moe_collection_complete 1.0" in response.text
    assert 'vllm:moe_rank_imbalance_average{model_part="target"} 1.6' in response.text


@pytest.mark.parametrize("multiprocess", [False, True])
def test_metrics_endpoint_does_not_enable_moe_by_default(
    tmp_path, monkeypatch, global_registry, multiprocess
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.serve.instrumentator.metrics import attach_router

    monkeypatch.delenv("VLLM_MOE_EXPERT_LOAD_CONFIG", raising=False)
    if multiprocess:
        monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    app = FastAPI()
    attach_router(app)
    with TestClient(app) as client:
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "vllm:moe_" not in response.text
