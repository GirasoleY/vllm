# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.all_reduce_utils import (
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    symm_mem_available = True
except ImportError:
    symm_mem_available = False

logger = init_logger(__name__)


class SymmMemCommunicator:
    _LOW_SM_DEVICE_CAPABILITIES = ("10.0", "10.3")
    _LOW_SM_WORLD_SIZE = 8
    _LOW_SM_SEMAPHORE_SLOTS = 4
    _LOW_SM_SEMAPHORE_BYTES = 128
    _WORLD_SIZES_MULTIMEM = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
        "10.3": [6, 8],
        "10.7": [6, 8],  # sm_107 (Rubin): reuse 10.3 thresholds
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        # add options for testing
        force_multimem: bool | None = None,
        max_size_override: int | None = None,
    ):
        self.disabled = True
        self._buffer_mc_ptr = 0
        self._low_sm_semaphore: torch.Tensor | None = None
        self._low_sm_semaphore_mc_ptr = 0

        if not symm_mem_available:
            return

        if not current_platform.is_cuda():
            logger.warning("SymmMemCommunicator: symmetric memory is not available.")
            return
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.accelerator.set_device_index(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        capability = current_platform.get_device_capability()
        if capability is None:
            logger.warning(
                "SymmMemCommunicator: device capability is unknown, "
                "communicator is not available."
            )
            return
        self.device_capability = capability.as_version_str()
        if self.device_capability not in SYMM_MEM_ALL_REDUCE_MAX_SIZES:
            logger.warning(
                "SymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]:
            logger.warning(
                "SymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        # Use override max_size if provided, otherwise use default
        if max_size_override is not None:
            self.max_size = max_size_override
            logger.info(
                "SymmMemCommunicator: Using override max_size: %s bytes",
                self.max_size,
            )
        else:
            self.max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability][
                self.world_size
            ]
        try:
            self.buffer = torch_symm_mem.empty(
                self.max_size // self.dtype.itemsize,
                device=self.device,
                dtype=self.dtype,
            )
            handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        except RuntimeError as e:
            logger.warning_once(
                "SymmMemCommunicator: symmetric memory initialization failed: %s "
                "Communicator is not available. To suppress this warning set "
                "VLLM_ALLREDUCE_USE_SYMM_MEM=0",
                str(e),
            )
            return
        self._buffer_mc_ptr = int(handle.multicast_ptr)
        if self._buffer_mc_ptr == 0:
            logger.warning(
                "SymmMemCommunicator: symmetric memory "
                "multicast operations are not supported."
            )
            return
        self._initialize_low_sm_semaphore()
        self.force_multimem = force_multimem
        self.disabled = False
        if envs.VLLM_BATCH_INVARIANT:
            self.disabled = True

    def should_use_symm_mem(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size <= self.max_size

    def _initialize_low_sm_semaphore(self) -> None:
        if (
            envs.VLLM_KIMI_K3_LATENT_AR_OVERLAP_MAX_TOKENS <= 0
            or envs.VLLM_BATCH_INVARIANT
            or self.device_capability not in self._LOW_SM_DEVICE_CAPABILITIES
            or self.world_size != self._LOW_SM_WORLD_SIZE
            or not hasattr(torch.ops._C, "kimi_k3_low_sm_all_reduce_")
        ):
            return
        try:
            semaphore = torch_symm_mem.empty(
                self._LOW_SM_SEMAPHORE_SLOTS * self._LOW_SM_SEMAPHORE_BYTES,
                device=self.device,
                dtype=torch.uint8,
            )
            handle = torch_symm_mem.rendezvous(semaphore, self.group.group_name)
        except RuntimeError as e:
            logger.warning_once(
                "SymmMemCommunicator: low-SM semaphore initialization failed: %s",
                str(e),
            )
            return
        semaphore.zero_()
        torch.accelerator.synchronize()
        dist.barrier(group=self.group)
        semaphore_mc_ptr = int(handle.multicast_ptr)
        if semaphore_mc_ptr == 0:
            logger.warning(
                "SymmMemCommunicator: low-SM semaphore does not have a "
                "multicast mapping."
            )
            return
        self._low_sm_semaphore = semaphore
        self._low_sm_semaphore_mc_ptr = semaphore_mc_ptr

    def has_low_sm_all_reduce(self) -> bool:
        """Return whether the multicast all-reduce is available.

        This is intentionally stricter than :meth:`should_use_symm_mem`:
        the overlap-oriented path must use the low-occupancy multimem kernel
        and must never fall back to the two-shot kernel.
        """
        if self.disabled:
            return False
        return self._low_sm_semaphore is not None and self._low_sm_semaphore_mc_ptr != 0

    def _validate_low_sm_all_reduce_input(self, inp: torch.Tensor) -> None:
        if not self.has_low_sm_all_reduce():
            capability = getattr(self, "device_capability", "unknown")
            world_size = getattr(self, "world_size", "unknown")
            raise RuntimeError(
                "Low-SM symmetric-memory all-reduce is unavailable for "
                f"compute capability {capability} and world size {world_size}."
            )
        if inp.device != self.device:
            raise ValueError(
                "Low-SM symmetric-memory all-reduce requires input on "
                f"{self.device}, but got {inp.device}."
            )
        if inp.dtype != self.dtype:
            raise ValueError(
                "Low-SM symmetric-memory all-reduce requires "
                f"{self.dtype}, but got {inp.dtype}."
            )
        if not inp.is_contiguous():
            raise ValueError(
                "Low-SM symmetric-memory all-reduce requires contiguous input."
            )
        if inp.numel() == 0 or inp.numel() % 8 != 0:
            raise ValueError(
                "Low-SM symmetric-memory all-reduce requires a non-empty "
                "input whose element count is divisible by 8."
            )
        inp_size = inp.numel() * inp.element_size()
        if inp_size > self.max_size:
            raise ValueError(
                "Low-SM symmetric-memory all-reduce input is too large: "
                f"{inp_size} bytes exceeds the {self.max_size}-byte workspace."
            )

    def stage_low_sm_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Copy ``inp`` once into the persistent symmetric destination.

        The communicator owns one data workspace, so callers must finish the
        corresponding all-reduce and its consumers before staging another
        tensor on this communicator.
        """
        self._validate_low_sm_all_reduce_input(inp)
        stage = self.buffer[: inp.numel()].view_as(inp)
        stage.copy_(inp)
        return stage

    def all_reduce_low_sm(self, inp: torch.Tensor) -> torch.Tensor:
        """All-reduce a staged symmetric input in place using four CTAs."""
        self._validate_low_sm_all_reduce_input(inp)
        if inp.data_ptr() != self.buffer.data_ptr():
            raise ValueError(
                "Low-SM all-reduce input must be the prefix of the communicator's "
                "symmetric buffer. Call stage_low_sm_all_reduce() first."
            )
        semaphore = self._low_sm_semaphore
        assert semaphore is not None
        torch.ops._C.kimi_k3_low_sm_all_reduce_(
            inp.view(-1),
            self._buffer_mc_ptr,
            semaphore,
            self._low_sm_semaphore_mc_ptr,
            self.rank,
            self.world_size,
        )
        return inp

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if not self.should_use_symm_mem(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))

        # Determine which algorithm to use
        use_multimem = False
        if self.force_multimem is not None:
            # Test override: use forced setting
            use_multimem = self.force_multimem
        else:
            # Normal logic: use multimem for supported world sizes
            use_multimem = (
                self.world_size in self._WORLD_SIZES_MULTIMEM[self.device_capability]
            )

        if use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out
