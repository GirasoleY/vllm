# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconstruct sharded weights using model-owned collective workspaces."""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class WeightGatherWorkspace:
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
class WeightGatherPart:
    """Local/full aliases and persistent collective storage for one tensor."""

    local_tensor: torch.Tensor
    gather_input: torch.Tensor
    gather_output: torch.Tensor
    full_tensor: torch.Tensor
    prepare: Callable[[], None]
    finalize: Callable[[], None]


WeightGatherLayout = Callable[
    [torch.nn.Module, torch.Tensor, int, WeightGatherWorkspace], WeightGatherPart
]


@dataclass(frozen=True)
class WeightGatherSpec:
    """A tensor attribute and its input-shard axis or layout-specific factory."""

    attr_name: str
    gather_dim: int = 0
    make_full_contiguous: bool = False
    layout: WeightGatherLayout | None = None

    def create_part(
        self, layer: torch.nn.Module, size: int, buffers: WeightGatherWorkspace
    ) -> WeightGatherPart:
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

        return WeightGatherPart(tensor, gather_input, gathered, full, prepare, finalize)


class GatheredWeights(torch.nn.Module):
    """Prefetch full weights and restore local storage after their consumer exits.

    This parameter-free module references its owning layer without registering
    it twice. Specs are resolved after loading, when quantization layouts are
    final. Sequential consumers in one model share the same workspace.
    """

    def __init__(
        self,
        layer: torch.nn.Module,
        specs: Callable[[torch.nn.Module], tuple[WeightGatherSpec, ...]],
        size: int,
    ) -> None:
        super().__init__()
        self._layer = weakref.ref(layer)
        self._specs = specs
        self.size = size
        self.parts: dict[str, WeightGatherPart] | None = None
        self.handles: list[Any] = []

    @property
    def layer(self) -> torch.nn.Module:
        layer = self._layer()
        assert layer is not None, "The model must retain the layer owning its weights."
        return layer

    @property
    def initialized(self) -> bool:
        return self.parts is not None

    @staticmethod
    def initialize_model(model: torch.nn.Module) -> None:
        """Refresh post-load aliases using one workspace for this model."""
        workspace = WeightGatherWorkspace()
        for module in model.modules():
            if isinstance(module, GatheredWeights):
                module.initialize(workspace)

    def initialize(self, workspace: WeightGatherWorkspace) -> None:
        if self.handles:
            raise RuntimeError("Cannot initialize weights while a gather is pending.")
        layer = self.layer
        specs = self._specs(layer)
        if not specs:
            raise RuntimeError("The linear method does not support weight gathering.")
        self.parts = {
            spec.attr_name: spec.create_part(layer, self.size, workspace)
            for spec in specs
        }

    def gather(self, group: torch.distributed.ProcessGroup) -> None:
        assert self.parts is not None, "Initialize gathered weights after loading."
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
        assert self.parts is not None, "Initialize gathered weights after loading."
        try:
            for handle in self.handles:
                handle.wait()
            for part in self.parts.values():
                part.finalize()
        finally:
            self.handles.clear()

    @contextmanager
    def full_weight(self) -> Iterator[None]:
        self.wait()
        assert self.parts is not None
        layer = self.layer
        with torch.no_grad():
            for name, part in self.parts.items():
                getattr(layer, name).set_(part.full_tensor)
            try:
                yield
            finally:
                for name, part in self.parts.items():
                    getattr(layer, name).set_(part.local_tensor)
