# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable gather buffers for linear methods' declared weight layouts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TPWeightBufferPool:
    """Full-weight workspace shared by sequential projections in one model."""

    buffers: dict[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)

    def get(
        self, name: str, shape: tuple[int, ...], tensor: torch.Tensor
    ) -> torch.Tensor:
        key = (name, tuple(shape), tensor.dtype, tensor.device)
        if key not in self.buffers:
            self.buffers[key] = torch.empty(
                shape, dtype=tensor.dtype, device=tensor.device
            )
        return self.buffers[key]


@dataclass
class TPWeightGatherPart:
    """Local/full aliases and persistent collective storage for one tensor."""

    tp_tensor: torch.Tensor
    gather_input: torch.Tensor
    gather_output: torch.Tensor
    full_tensor: torch.Tensor
    prepare: Callable[[], None]
    finalize: Callable[[], None]


TPWeightGatherLayout = Callable[
    [torch.nn.Module, torch.Tensor, int, TPWeightBufferPool], TPWeightGatherPart
]


@dataclass(frozen=True)
class TPWeightGatherSpec:
    """A tensor attribute and its input-shard axis or layout-specific factory."""

    attr_name: str
    gather_dim: int = 0
    make_full_contiguous: bool = False
    layout: TPWeightGatherLayout | None = None

    def create_part(
        self, layer: torch.nn.Module, size: int, buffers: TPWeightBufferPool
    ) -> TPWeightGatherPart:
        tensor = getattr(layer, self.attr_name).detach()
        if self.layout is not None:
            return self.layout(layer, tensor, size, buffers)

        if not -tensor.ndim <= self.gather_dim < tensor.ndim:
            raise ValueError(
                f"Invalid gather dimension {self.gather_dim} for {tensor.shape}"
            )
        dim = self.gather_dim % tensor.ndim
        source = tensor.movedim(dim, 0)
        gather_input = source.contiguous()
        gather_shape = (source.shape[0] * size, *source.shape[1:])
        gathered = buffers.get(f"{self.attr_name}.gather", gather_shape, tensor)
        full_view = gathered.movedim(0, dim)
        full = (
            buffers.get(f"{self.attr_name}.full", full_view.shape, tensor)
            if self.make_full_contiguous and not full_view.is_contiguous()
            else full_view
        )

        def prepare() -> None:
            if gather_input.data_ptr() != source.data_ptr():
                gather_input.copy_(source)

        def finalize() -> None:
            if full.data_ptr() != full_view.data_ptr():
                full.copy_(full_view)

        return TPWeightGatherPart(
            tensor, gather_input, gathered, full, prepare, finalize
        )


class TPWeightSwitch:
    """Gather and temporarily expose the full tensors required by a linear kernel."""

    def __init__(
        self,
        layer: torch.nn.Module,
        specs: tuple[TPWeightGatherSpec, ...],
        size: int,
        buffers: TPWeightBufferPool,
    ) -> None:
        if not specs:
            raise RuntimeError(
                "The linear method does not support TP weight switching."
            )
        self.parts = {
            spec.attr_name: spec.create_part(layer, size, buffers) for spec in specs
        }
        self.handles: list[Any] = []

    def gather(self, group: Any) -> None:
        if self.handles:
            raise RuntimeError(
                "Wait for the pending weight gather before launching another."
            )
        try:
            for part in self.parts.values():
                part.prepare()
                self.handles.append(
                    torch.distributed.all_gather_into_tensor(
                        part.gather_output,
                        part.gather_input,
                        group=group,
                        async_op=True,
                    )
                )
        except Exception:
            self.wait()
            raise

    def wait(self) -> None:
        try:
            for handle in self.handles:
                handle.wait()
            for part in self.parts.values():
                part.finalize()
        finally:
            self.handles.clear()

    @contextmanager
    def full_weight(self, layer: torch.nn.Module) -> Iterator[None]:
        self.wait()
        with torch.no_grad():
            for name, part in self.parts.items():
                getattr(layer, name).set_(part.full_tensor)
            try:
                yield
            finally:
                for name, part in self.parts.items():
                    getattr(layer, name).set_(part.tp_tensor)
