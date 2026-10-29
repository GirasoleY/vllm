# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU publication and scrape-time aggregation of native MoE receive counts."""

import json
import math
import os
import socket
import tempfile
import time
from collections import deque
from pathlib import Path

import regex as re
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily

MAX_LAYERS = 256


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, separators=(",", ":"))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class MoeLoadPublisher:
    """Keep a bounded online history and an independently bounded expert dump."""

    def __init__(self, config: dict, run_id: str):
        self.config = config
        self.root = Path(config["directory"]) / socket.gethostname() / "online"
        self.run_id = run_id
        self.previous: list[list[int]] | None = None
        self.totals: list[int] = []
        self.history: deque[dict] = deque(maxlen=4)
        self.capture_id = None
        self.capture_samples = 0
        self.layers: list[dict] | None = None

    def publish(self, layers: list[dict], counts: list[list[int]], metadata: dict):
        if (
            len(counts) != len(layers)
            or any(len(c) != layer["local_experts"] for c, layer in zip(counts, layers))
            or any(
                type(n) is not int or not 0 <= n < (1 << 32) for c in counts for n in c
            )
        ):
            raise ValueError("Invalid native counter shape/value")
        if self.layers is not None and self.layers != layers:
            raise ValueError("MoE topology changed within a worker run")
        self.layers = layers
        rank = layers[0]["ep_rank"]
        if self.previous is None:
            self.totals = [0] * len(layers)
        else:
            for i, (current, previous) in enumerate(zip(counts, self.previous)):
                self.totals[i] += sum(
                    (a - b) % (1 << 32) for a, b in zip(current, previous)
                )
        self.previous = counts
        self.history.append(
            dict(
                epoch=metadata["epoch_bucket"],
                time=metadata["enqueued_unix"],
                counts=self.totals.copy(),
            )
        )
        value = dict(
            schema=1,
            run_id=self.run_id,
            pid=os.getpid(),
            layers=layers,
            collection_id=self.config.get("collection_id"),
            history=list(self.history),
        )
        atomic_json(self.root / f"rank{rank}.json", value)
        self._diagnostic(rank, layers, counts, metadata)

    def _diagnostic(self, rank, layers, counts, metadata):
        budget = int(self.config.get("max_samples", 360))
        if not budget:
            return
        try:
            request = json.loads((self.root / "diagnostic-request.json").read_text())
        except (OSError, ValueError):
            return
        now = metadata["enqueued_unix"]
        if not (
            isinstance(request.get("id"), str)
            and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request["id"])
            and request.get("start_unix", 0) <= now <= request.get("end_unix", 0)
            and 1 <= request.get("samples", 0) <= budget
        ):
            return
        if request["id"] != self.capture_id:
            self.capture_id, self.capture_samples = request["id"], 0
        if self.capture_samples >= request["samples"]:
            return
        directory = self.root / "diagnostics" / f"rank{rank}"
        directory.mkdir(parents=True, exist_ok=True)
        value = dict(
            metadata,
            schema=1,
            run_id=self.run_id,
            capture_id=self.capture_id,
            collection_id=self.config.get("collection_id"),
            counter_bits=32,
            layers=layers,
            host=socket.gethostname(),
            pid=os.getpid(),
            observed_ready_unix=time.time(),
            cumulative_received_tokens=counts,
            baseline=self.capture_samples == 0,
        )
        atomic_json(directory / f"sample{self.capture_samples:04d}.json", value)
        self.capture_samples += 1


class MoeLoadCollector:
    """Aggregate CPU summaries without touching GPUs."""

    def __init__(self, config: dict):
        self.config = config
        self.root = Path(config["directory"]) / socket.gethostname() / "online"
        self.world = int(config["ep_world"])
        self.interval = float(config.get("interval_s", 10))
        self.stale_after = float(config.get("stale_after_s", self.interval * 3))
        self.started = time.time()
        if (
            not 1 <= self.world <= 256
            or not 1 <= self.interval <= 3600
            or not math.isfinite(self.stale_after)
            or self.stale_after < self.interval
        ):
            raise ValueError("Invalid online MoE collection configuration")

    def _read(self, rank):
        path = self.root / f"rank{rank}.json"
        if path.stat().st_size > 256_000:
            raise ValueError("Oversized MoE summary")
        data = json.loads(path.read_text())
        layers, history = data["layers"], data["history"]
        if not (
            data["schema"] == 1
            and 1 <= len(layers) <= MAX_LAYERS
            and 1 <= len(history) <= 4
        ):
            raise ValueError("Invalid MoE summary shape")
        if data["collection_id"] != self.config.get("collection_id"):
            raise ValueError("Mismatched collection generation")
        for layer in layers:
            if (
                layer["ep_rank"] != rank
                or layer["ep_world"] != self.world
                or layer["physical_experts"] != layer["local_experts"] * self.world
                or layer["first_expert"] != rank * layer["local_experts"]
                or not re.findall(r"(?:^|\.)layers\.(\d+)", layer["prefix"])
            ):
                raise ValueError("Mismatched EP topology")
        epochs = set()
        for observation in history:
            if (
                not isinstance(observation["epoch"], int)
                or observation["epoch"] in epochs
                or not math.isfinite(observation["time"])
                or len(observation["counts"]) != len(layers)
                or not all(
                    type(n) is int and 0 <= n < (1 << 63) for n in observation["counts"]
                )
            ):
                raise ValueError("Invalid MoE observation")
            epochs.add(observation["epoch"])
        return data

    def collect(self):
        now = time.time()
        ranks = []
        for rank in range(self.world):
            try:
                ranks.append(self._read(rank))
            except (OSError, ValueError, KeyError, TypeError):
                break
        common = (
            set.intersection(
                *(
                    {
                        h["epoch"]
                        for h in r["history"]
                        if self.started <= h["time"] <= now
                    }
                    for r in ranks
                )
            )
            if len(ranks) == self.world
            else set()
        )
        complete = False
        observations = [
            [h["time"] for h in r["history"] if self.started <= h["time"] <= now]
            for r in ranks
        ]
        latest = (
            min(max(values) for values in observations)
            if len(observations) == self.world and all(observations)
            else None
        )
        skew = 0.0
        windows = []
        if len(common) >= 2:
            first, last = sorted(common)[-2:]
            for r in ranks:
                snapshots = {h["epoch"]: h for h in r["history"]}
                windows.append((snapshots[first], snapshots[last]))
            latest = min(b["time"] for _, b in windows)
            skew = max(
                max(w[i]["time"] for w in windows) - min(w[i]["time"] for w in windows)
                for i in (0, 1)
            )
            signatures = [
                [
                    {
                        k: v
                        for k, v in layer.items()
                        if k not in ("ep_rank", "first_expert")
                    }
                    for layer in r["layers"]
                ]
                for r in ranks
            ]
            complete = (
                now - latest <= self.stale_after
                and last - first <= 3
                and skew <= self.interval / 4
                and all(s == signatures[0] for s in signatures)
                and all(
                    all(y >= x for x, y in zip(a["counts"], b["counts"]))
                    for a, b in windows
                )
            )
        for name, value in [
            (
                "collection_age_seconds",
                max(0.0, now - latest) if latest is not None else None,
            ),
            ("collection_complete", int(complete)),
            ("collection_skew_seconds", skew),
        ]:
            metric = GaugeMetricFamily("vllm:moe_" + name, name)
            if value is not None:
                metric.add_metric([], value)
            yield metric
        if not complete:
            return
        counter = CounterMetricFamily(
            "vllm:moe_received_assignments",
            "Native expert assignments after baseline",
            labels=["moe_layer", "model_part", "ep_rank"],
        )
        parts: dict[str, list[tuple[float, int]]] = {}
        for i, layer in enumerate(ranks[0]["layers"]):
            layer_id = re.findall(r"(?:^|\.)layers\.(\d+)", layer["prefix"])[-1]
            part = (
                "target"
                if i < int(self.config.get("target_layers", len(ranks[0]["layers"])))
                else "draft"
            )
            load = [b["counts"][i] - a["counts"][i] for a, b in windows]
            if self.config.get("rank_metrics", False):
                for rank, (_, end) in enumerate(windows):
                    counter.add_metric([layer_id, part, str(rank)], end["counts"][i])
            total = sum(load)
            if total:
                parts.setdefault(part, []).append(
                    (max(load) * self.world / total, total)
                )
        if self.config.get("rank_metrics", False):
            yield counter
        average = GaugeMetricFamily(
            "vllm:moe_rank_imbalance_average",
            "Assignment-weighted layer max/mean",
            labels=["model_part"],
        )
        worst = GaugeMetricFamily(
            "vllm:moe_rank_imbalance_worst",
            "Worst active layer max/mean",
            labels=["model_part"],
        )
        for part, values in parts.items():
            average.add_metric(
                [part], sum(x * n for x, n in values) / sum(n for _, n in values)
            )
            worst.add_metric([part], max(x for x, _ in values))
        yield average
        yield worst


def request_diagnostic(config: dict, capture_id: str, samples: int) -> Path:
    """Request a bounded expert capture across EP ranks."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", capture_id) or not 1 <= samples <= int(
        config.get("max_samples", 360)
    ):
        raise ValueError("Invalid diagnostic capture id/budget")
    root = Path(config["directory"]) / socket.gethostname() / "online"
    root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    path = root / "diagnostic-request.json"
    atomic_json(
        path,
        dict(
            id=capture_id,
            samples=samples,
            start_unix=start,
            end_unix=start + (samples + 4) * float(config.get("interval_s", 10)),
        ),
    )
    return path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Arm a bounded native MoE capture")
    parser.add_argument("--capture", required=True)
    parser.add_argument("--samples", type=int, default=360)
    args = parser.parse_args()
    config = json.loads(os.environ["VLLM_MOE_EXPERT_LOAD_CONFIG"])
    print(request_diagnostic(config, args.capture, args.samples))
