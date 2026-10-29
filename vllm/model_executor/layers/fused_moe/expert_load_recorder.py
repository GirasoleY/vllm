# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Asynchronous publication of native expert receive counters."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.metrics.moe_load import MAX_LAYERS, MoeLoadPublisher

logger = init_logger(__name__)


@dataclass
class _CopySlot:
    buffer: torch.Tensor
    ready: torch.cuda.Event
    future: Future[None] | None = None


class ExpertLoadRecorder:
    def __init__(self, config: dict[str, Any]):
        self.interval = float(config.get("interval_s", 10))
        budget = int(config.get("max_samples", 360))
        if not 1 <= self.interval <= 3600 or not 0 <= budget <= 1440:
            raise ValueError("Invalid expert-load collection interval/budget")
        self.world = int(config["ep_world"])
        if not 1 <= self.world <= 256:
            raise ValueError("Invalid EP world size")
        self.config = config
        self.publisher = MoeLoadPublisher(config, f"{time.time_ns()}-pid{os.getpid()}")
        self.counters: torch.Tensor | None = None
        self.layers: list[dict[str, Any]] = []
        self.slots: list[_CopySlot] = []
        self.pool: ThreadPoolExecutor | None = None
        self.last_bucket: int | None = None
        self.samples = 0
        self.failed = False

    def register(
        self,
        prefix: str,
        experts: int,
        local_experts: int,
        first_expert: int,
        top_k: int,
        device: torch.device,
    ) -> torch.Tensor:
        if device.type != "cuda" or local_experts < 1 or experts % local_experts:
            raise ValueError("Expert-load collection requires sharded CUDA experts")
        if experts // local_experts != self.world:
            raise ValueError("Configured EP world size does not match the experts")
        if any(layer["prefix"] == prefix for layer in self.layers):
            raise ValueError(f"Duplicate expert-load layer: {prefix}")
        if self.counters is None:
            # Keep counter addresses stable across graph replay.
            shape = (MAX_LAYERS, local_experts)
            self.counters = torch.zeros(shape, dtype=torch.int32, device=device)
            self.slots = [
                _CopySlot(
                    torch.empty(
                        shape, dtype=torch.int32, device="cpu", pin_memory=True
                    ),
                    torch.cuda.Event(),
                )
                for _ in range(2)
            ]
        if self.counters.shape[1] != local_experts or len(self.layers) == MAX_LAYERS:
            raise ValueError("Incompatible expert-load counter bank")
        self.layers.append(
            dict(
                prefix=prefix,
                physical_experts=experts,
                local_experts=local_experts,
                first_expert=first_expert,
                ep_rank=first_expert // local_experts,
                ep_world=self.world,
                top_k=top_k,
            )
        )
        return self.counters[len(self.layers) - 1]

    def sample(self) -> None:
        if self.failed or self.counters is None:
            return
        now = time.time()
        bucket = int(now // self.interval)
        if bucket == self.last_bucket:
            return
        slot = next(
            (s for s in self.slots if s.future is None or s.future.done()), None
        )
        if slot is None:
            return
        self.last_bucket = bucket
        layers = self.layers.copy()
        host = slot.buffer[: len(layers)]
        # Copy on the current stream after model execution.
        host.copy_(self.counters[: len(layers)], non_blocking=True)
        slot.ready.record()
        metadata = dict(sequence=self.samples, enqueued_unix=now, epoch_bucket=bucket)
        self.samples += 1
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moe-load")
        slot.future = self.pool.submit(
            self._publish, host, slot.ready, layers, metadata
        )

    def _publish(self, host, ready, layers, metadata) -> None:
        copy_complete = False
        try:
            deadline = time.monotonic() + 30
            while not ready.query():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Expert-load copy did not complete")
                time.sleep(0.01)
            copy_complete = True
            # Interpret wrapped int32 counters as unsigned.
            counts = [[int(n) & 0xFFFFFFFF for n in values] for values in host.tolist()]
            self.publisher.publish(layers, counts, metadata)
        except Exception:
            # Never reuse a buffer whose DMA did not finish.
            if not copy_complete:
                self.failed = True
            logger.exception("MoE load publication failed")


@lru_cache(maxsize=1)
def get_expert_load_recorder() -> ExpertLoadRecorder | None:
    raw = envs.VLLM_MOE_EXPERT_LOAD_CONFIG
    return ExpertLoadRecorder(json.loads(raw)) if raw else None


def register_expert_load(
    prefix: str,
    experts: int,
    local_experts: int,
    first_expert: int,
    top_k: int,
    device: torch.device,
) -> torch.Tensor | None:
    recorder = get_expert_load_recorder()
    return (
        recorder.register(prefix, experts, local_experts, first_expert, top_k, device)
        if recorder
        else None
    )


def sample_expert_load() -> None:
    recorder = get_expert_load_recorder()
    if recorder is not None:
        try:
            recorder.sample()
        except Exception:
            recorder.failed = True
            logger.exception("Expert-load sampling disabled after failure")
