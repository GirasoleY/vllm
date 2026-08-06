# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup
from torch.profiler import record_function

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

try:
    import torch.distributed._symmetric_memory  # noqa: F401

    symm_mem_available = True
except ImportError:
    symm_mem_available = False


_MAX_FENCE_SPINS = 100_000_000


@triton.jit
def _trap_if_nonzero(value):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred failed;
            setp.ne.u32 failed, $1, 0;
            @failed trap;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _store_release_system(pointer, value, mask):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred enabled;
            setp.ne.u32 enabled, $3, 0;
            @enabled st.global.release.sys.u32 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l,r,r",
        args=[pointer, value, mask.to(tl.uint32)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _direct_publish_kernel(
    partial_output,
    partial_lse,
    peer_output_ptrs,
    peer_lse_ptrs,
    local_epoch,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    lse_token_stride,
    lse_head_stride,
    peer_output_parity_stride,
    peer_output_source_stride,
    peer_output_token_stride,
    peer_output_head_stride,
    peer_output_dim_stride,
    peer_lse_parity_stride,
    peer_lse_source_stride,
    peer_lse_token_stride,
    peer_lse_head_stride,
    my_rank: tl.constexpr,
    local_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_items: tl.constexpr,
    head_block_size: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    destination_rank = tl.program_id(1).to(tl.int64)
    item_block_idx = tl.program_id(2).to(tl.int64)
    epoch = tl.atomic_add(local_epoch, 0, sem="acquire", scope="gpu") + 1
    parity = epoch & 1

    output_ptr_table = peer_output_ptrs.to(tl.pointer_type(tl.uint64))
    peer_output = tl.load(output_ptr_table + destination_rank).to(
        tl.pointer_type(partial_output.dtype.element_ty)
    )
    item = item_block_idx * block_items + tl.arange(0, block_items)
    item_mask = item < local_heads * head_dim
    local_head_idx = item // head_dim
    dim = item % head_dim
    source_head_idx = destination_rank * local_heads + local_head_idx
    source_output_offset = (
        token_idx * output_token_stride
        + source_head_idx * output_head_stride
        + dim * output_dim_stride
    )
    destination_output_offset = (
        parity * peer_output_parity_stride
        + my_rank * peer_output_source_stride
        + token_idx * peer_output_token_stride
        + local_head_idx * peer_output_head_stride
        + dim * peer_output_dim_stride
    )
    value = tl.load(partial_output + source_output_offset, mask=item_mask)
    tl.store(peer_output + destination_output_offset, value, mask=item_mask)

    lse_ptr_table = peer_lse_ptrs.to(tl.pointer_type(tl.uint64))
    peer_lse = tl.load(lse_ptr_table + destination_rank).to(tl.pointer_type(tl.float32))
    lse_local_head_idx = tl.arange(0, head_block_size)
    lse_mask = (item_block_idx == 0) & (lse_local_head_idx < local_heads)
    lse_source_head_idx = destination_rank * local_heads + lse_local_head_idx
    source_lse_offset = (
        token_idx * lse_token_stride + lse_source_head_idx * lse_head_stride
    )
    destination_lse_offset = (
        parity * peer_lse_parity_stride
        + my_rank * peer_lse_source_stride
        + token_idx * peer_lse_token_stride
        + lse_local_head_idx * peer_lse_head_stride
    )
    tl.store(
        peer_lse + destination_lse_offset,
        tl.load(partial_lse + source_lse_offset, mask=lse_mask),
        mask=lse_mask,
    )


@triton.jit
def _direct_signal_kernel(
    local_epoch,
    peer_signal_ptrs,
    peer_signal_parity_stride,
    my_rank: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
):
    epoch = tl.atomic_add(local_epoch, 1, sem="acq_rel", scope="gpu") + 1
    destination_rank = tl.arange(0, block_size)
    destination_mask = destination_rank < world_size
    signal_ptr_table = peer_signal_ptrs.to(tl.pointer_type(tl.uint64))
    peer_signal = tl.load(
        signal_ptr_table + destination_rank,
        mask=destination_mask,
        other=0,
    ).to(tl.pointer_type(tl.int32))
    parity = epoch & 1
    _store_release_system(
        peer_signal + parity * peer_signal_parity_stride + my_rank,
        epoch.to(tl.uint32),
        destination_mask,
    )


@triton.jit
def _direct_consumer_merge_kernel(
    received_output,
    received_lse,
    received_signal,
    local_epoch,
    combined_output,
    output_parity_stride,
    output_source_stride,
    output_token_stride,
    output_head_stride,
    output_dim_stride,
    lse_parity_stride,
    lse_source_stride,
    lse_token_stride,
    lse_head_stride,
    signal_parity_stride,
    combined_token_stride,
    combined_head_stride,
    combined_dim_stride,
    world_size: tl.constexpr,
    source_block_size: tl.constexpr,
    is_base_e: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
    max_spins: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    epoch = tl.atomic_add(local_epoch, 0, sem="acquire", scope="gpu")
    parity = epoch & 1

    source_rank = tl.arange(0, source_block_size)
    source_mask = source_rank < world_size
    signal_offset = parity * signal_parity_stride + source_rank
    observed = tl.atomic_add(
        received_signal + signal_offset,
        0,
        mask=source_mask,
        sem="acquire",
        scope="sys",
    )
    expected = epoch.to(tl.int32)
    pending = tl.max(tl.where(source_mask & (observed != expected), 1, 0))
    spins = 0
    while (pending != 0) & (spins < max_spins):
        observed = tl.atomic_add(
            received_signal + signal_offset,
            0,
            mask=source_mask,
            sem="acquire",
            scope="sys",
        )
        pending = tl.max(tl.where(source_mask & (observed != expected), 1, 0))
        spins += 1
    _trap_if_nonzero(pending)

    lse_offset = (
        parity * lse_parity_stride
        + source_rank * lse_source_stride
        + token_idx * lse_token_stride
        + local_head_idx * lse_head_stride
    )
    lse = tl.load(received_lse + lse_offset, mask=source_mask, other=-float("inf"))
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)
    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    weights = tl.exp(lse - lse_max) if is_base_e else tl.exp2(lse - lse_max)
    weight_sum = tl.sum(weights, axis=0)
    weights = tl.where(weight_sum == 0.0, 0.0, weights / weight_sum)

    dim = tl.arange(0, block_dim)
    dim_mask = dim < head_dim
    output_offset = (
        parity * output_parity_stride
        + source_rank[:, None] * output_source_stride
        + token_idx * output_token_stride
        + local_head_idx * output_head_stride
        + dim[None, :] * output_dim_stride
    )
    partial_output = tl.load(
        received_output + output_offset,
        mask=source_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    output = tl.sum(partial_output.to(tl.float32) * weights[:, None], axis=0)
    combined_offset = (
        token_idx * combined_token_stride
        + local_head_idx * combined_head_stride
        + dim * combined_dim_stride
    )
    tl.store(combined_output + combined_offset, output, mask=dim_mask)


class DirectDCPA2AWorkspace:
    """Persistent symmetric buffers for direct DCP output exchange."""

    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_num_tokens: int,
        heads_per_rank: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        num_ubatches: int = 1,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Direct DCP A2A does not support {dtype}")
        if num_ubatches < 1:
            raise ValueError(
                f"Direct DCP A2A requires at least one ubatch slot, got {num_ubatches}"
            )
        self.group = group
        self.world_size = group.size()
        self.rank = group.rank()
        self.num_ubatches = num_ubatches
        self.max_num_tokens = max_num_tokens
        self.heads_per_rank = heads_per_rank
        self.head_dim = head_dim
        self._allocations: list[tuple[torch.Tensor, object, list[torch.Tensor]]] = []

        output_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
            head_dim,
        )
        lse_shape = (
            num_ubatches,
            2,
            self.world_size,
            max_num_tokens,
            heads_per_rank,
        )
        signal_shape = (num_ubatches, 2, self.world_size)
        self.received_output, self.peer_output_ptrs = self._allocate(
            symm_mem, output_shape, dtype, device
        )
        self.received_lse, self.peer_lse_ptrs = self._allocate(
            symm_mem, lse_shape, torch.float32, device
        )
        self.received_signal, self.peer_signal_ptrs = self._allocate(
            symm_mem, signal_shape, torch.int32, device
        )
        self.epoch = torch.zeros(num_ubatches, dtype=torch.int64, device=device)
        self.combined_output = torch.empty(
            (
                num_ubatches,
                max_num_tokens,
                heads_per_rank,
                head_dim,
            ),
            dtype=dtype,
            device=device,
        )

    def _allocate(self, symm_mem, shape, dtype, device):
        storage = symm_mem.empty(shape, device=device, dtype=dtype)
        storage.zero_()
        torch.accelerator.synchronize()
        handle = symm_mem.rendezvous(storage, self.group.group_name)
        assert handle is not None, "DCP symmetric memory rendezvous returned None"
        handle.barrier()
        views = [
            handle.get_buffer(peer, list(shape), dtype, 0)
            for peer in range(self.world_size)
        ]
        peer_ptrs = torch.tensor(
            [
                [view[ubatch].data_ptr() for view in views]
                for ubatch in range(self.num_ubatches)
            ],
            dtype=torch.int64,
            device=device,
        )
        self._allocations.append((storage, handle, views))
        return storage, peer_ptrs

    def lse_reduce(
        self,
        partial_output: torch.Tensor,
        partial_lse: torch.Tensor,
        is_lse_base_on_e: bool,
    ) -> torch.Tensor:
        ubatch = dbo_current_ubatch_id()
        num_tokens = partial_output.shape[0]
        if num_tokens <= 0 or num_tokens > self.max_num_tokens:
            raise ValueError(
                "Direct DCP A2A token count must be within the workspace capacity; "
                f"got {num_tokens}, capacity {self.max_num_tokens}."
            )
        expected_heads = self.world_size * self.heads_per_rank
        if tuple(partial_output.shape[1:]) != (expected_heads, self.head_dim):
            raise ValueError(
                "Direct DCP A2A output geometry changed after initialization."
            )
        if tuple(partial_lse.shape) != (num_tokens, expected_heads):
            raise ValueError("Direct DCP A2A LSE geometry does not match the output.")
        if partial_output.dtype != self.combined_output.dtype:
            raise ValueError(
                "Direct DCP A2A output dtype changed after initialization."
            )
        if partial_lse.dtype != torch.float32:
            raise ValueError("Direct DCP A2A requires FP32 LSE input.")
        if partial_output.device != self.combined_output.device:
            raise ValueError(
                "Direct DCP A2A output device changed after initialization."
            )
        if partial_lse.device != partial_output.device:
            raise ValueError("Direct DCP A2A inputs must use the same device.")

        output_slot = self.received_output[ubatch]
        lse_slot = self.received_lse[ubatch]
        signal_slot = self.received_signal[ubatch]
        epoch = self.epoch[ubatch : ubatch + 1]
        output = self.combined_output[ubatch, :num_tokens]
        with record_function("dcp.direct_a2a.producer_publish"):
            publish_block_items = min(
                2048,
                triton.next_power_of_2(self.heads_per_rank * self.head_dim),
            )
            publish_blocks = triton.cdiv(
                self.heads_per_rank * self.head_dim, publish_block_items
            )
            _direct_publish_kernel[(num_tokens, self.world_size, publish_blocks)](
                partial_output,
                partial_lse,
                self.peer_output_ptrs[ubatch],
                self.peer_lse_ptrs[ubatch],
                epoch,
                partial_output.stride(0),
                partial_output.stride(1),
                partial_output.stride(2),
                partial_lse.stride(0),
                partial_lse.stride(1),
                output_slot.stride(0),
                output_slot.stride(1),
                output_slot.stride(2),
                output_slot.stride(3),
                output_slot.stride(4),
                lse_slot.stride(0),
                lse_slot.stride(1),
                lse_slot.stride(2),
                lse_slot.stride(3),
                my_rank=self.rank,
                local_heads=self.heads_per_rank,
                head_dim=self.head_dim,
                block_items=publish_block_items,
                head_block_size=triton.next_power_of_2(self.heads_per_rank),
                num_warps=8,
            )
            _direct_signal_kernel[(1,)](
                epoch,
                self.peer_signal_ptrs[ubatch],
                signal_slot.stride(0),
                my_rank=self.rank,
                world_size=self.world_size,
                block_size=triton.next_power_of_2(self.world_size),
                num_warps=1,
            )

        with record_function("dcp.direct_a2a.consumer_merge"):
            _direct_consumer_merge_kernel[(num_tokens, self.heads_per_rank)](
                output_slot,
                lse_slot,
                signal_slot,
                epoch,
                output,
                output_slot.stride(0),
                output_slot.stride(1),
                output_slot.stride(2),
                output_slot.stride(3),
                output_slot.stride(4),
                lse_slot.stride(0),
                lse_slot.stride(1),
                lse_slot.stride(2),
                lse_slot.stride(3),
                signal_slot.stride(0),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                world_size=self.world_size,
                source_block_size=triton.next_power_of_2(self.world_size),
                is_base_e=is_lse_base_on_e,
                head_dim=self.head_dim,
                block_dim=triton.next_power_of_2(self.head_dim),
                max_spins=_MAX_FENCE_SPINS,
                num_warps=4,
            )
        return output


@cache
def get_direct_dcp_a2a_workspace(
    group: GroupCoordinator,
    device: torch.device,
    max_num_tokens: int,
    heads_per_rank: int,
    head_dim: int,
    dtype: torch.dtype,
    num_ubatches: int,
) -> DirectDCPA2AWorkspace | None:
    """Return the workspace shared by all MLA layers, or None if disabled.

    Unset ``VLLM_USE_DIRECT_DCP_A2A`` means auto: enabled on CUDA with
    fp16/bf16 activations when the DCP group is within a single node.
    ``1`` forces it on (e.g. for multi-node NVLink domains where symmetric
    memory can rendezvous across nodes); ``0`` disables it.
    """
    from vllm.distributed.parallel_state import in_the_same_node_as

    use_direct = envs.VLLM_USE_DIRECT_DCP_A2A
    if use_direct is None:
        use_direct = (
            symm_mem_available
            and current_platform.is_cuda()
            and dtype in (torch.float16, torch.bfloat16)
            and all(in_the_same_node_as(group.cpu_group, source_rank=0))
        )
    if not use_direct:
        return None
    return DirectDCPA2AWorkspace(
        group.device_group,
        device,
        max_num_tokens,
        heads_per_rank,
        head_dim,
        dtype,
        num_ubatches,
    )
