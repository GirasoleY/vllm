# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import WarmupIntRange
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter_states_kernel(
    state_ptr,
    src_ptr,
    indices_ptr,
    conv_state_ptr,
    conv_tail_ptr,
    conv_tail_indices_ptr,
    has_initial_state_ptr,
    stride_state_batch,
    stride_src_batch,
    stride_indices,
    stride_conv_batch,
    stride_conv_channel,
    stride_conv_token,
    stride_tail_token,
    stride_tail_indices,
    stride_has_initial,
    row_size: tl.constexpr,
    CONV_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    block_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < row_size

    if launch_pdl:
        tl.extra.cuda.gdc_wait()
        if CONV_DIM == 0:
            tl.extra.cuda.gdc_launch_dependents()

    state_idx = tl.load(indices_ptr + batch_idx * stride_indices).to(tl.int64)
    values = tl.load(src_ptr + batch_idx * stride_src_batch + offsets, mask=mask)
    tl.store(state_ptr + state_idx * stride_state_batch + offsets, values, mask=mask)

    if CONV_DIM > 0:
        if block_idx * BLOCK_SIZE < CONV_DIM:
            columns = tl.arange(0, 4)
            tail_indices = tl.load(
                conv_tail_indices_ptr + batch_idx * stride_tail_indices + columns,
                mask=columns < 3,
                other=0,
            ).to(tl.int64)
            has_initial = tl.load(
                has_initial_state_ptr + batch_idx * stride_has_initial
            )
            conv_base = (
                conv_state_ptr
                + state_idx * stride_conv_batch
                + offsets[:, None] * stride_conv_channel
            )
            conv_mask = (offsets < CONV_DIM)[:, None] & (columns < 3)[None, :]
            # Load the entire old window before writing its shifted successor.
            prefix = tl.load(
                conv_base + (tail_indices + 3)[None, :] * stride_conv_token,
                mask=conv_mask & (tail_indices < 0)[None, :] & has_initial,
                other=0.0,
            )
            tail = tl.load(
                conv_tail_ptr
                + tail_indices[None, :] * stride_tail_token
                + offsets[:, None],
                mask=conv_mask & (tail_indices >= 0)[None, :],
                other=0.0,
            ).to(conv_state_ptr.dtype.element_ty)
            final_window = tl.where((tail_indices >= 0)[None, :], tail, prefix)
            # Match causal_conv1d's load/store ordering for in-place windows.
            tl.debug_barrier()
            tl.store(
                conv_base + columns[None, :] * stride_conv_token,
                final_window,
                mask=conv_mask,
            )
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()


def _scatter_states_warmup_inputs(
    *,
    state_shape: tuple[int, ...],
    state_dtype: torch.dtype,
    indices_dtype: torch.dtype,
    max_num_tokens: int,
) -> dict[str, object]:
    num_tokens: Any = WarmupIntRange(1, max_num_tokens + 1)
    return dict(
        state=TritonWarmupTensor(
            state_dtype,
            shape=(1,) + state_shape,
        ),
        src=TritonWarmupTensor(
            state_dtype,
            shape=(num_tokens,) + state_shape,
        ),
        indices=TritonWarmupTensor(
            indices_dtype,
            shape=(num_tokens,),
        ),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_scatter_states_kernel,
    warmup_inputs=_scatter_states_warmup_inputs,
)
def scatter_states(
    state: torch.Tensor,
    src: torch.Tensor,
    indices: torch.Tensor,
    *,
    conv_state: torch.Tensor | None = None,
    conv_tail: torch.Tensor | None = None,
    conv_tail_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> DispatchSpec:
    """Scatter ``src`` rows into ``state`` at ``indices`` (in place).

    Equivalent to ``state[indices] = src`` but non-atomic and bandwidth-bound,
    since mamba cache slots are unique per sequence. ``gather_initial_states``
    is the read-side counterpart.

    Optional convolution inputs also publish a width-three raw-input window.
    ``conv_tail_indices`` maps each sequence's three final tokens into
    ``conv_tail``; -3..-1 select old cache columns, zeroed for fresh sequences.
    Columns beyond the first three are left unchanged.
    """
    assert state.ndim >= 2
    assert state.is_cuda
    assert src.ndim == state.ndim
    assert indices.ndim == 1
    assert indices.device == state.device
    assert src.shape[1:] == state.shape[1:]
    assert src.shape[0] == indices.shape[0]
    assert indices.dtype in (torch.int32, torch.int64)

    row_size = state[0].numel()
    assert state[0].is_contiguous()
    assert src[0].is_contiguous()
    conv_dim = 0
    conv_strides = (0, 0, 0)
    tail_stride = tail_indices_stride = has_initial_stride = 0
    if conv_state is not None:
        assert conv_tail is not None and conv_tail_indices is not None
        assert has_initial_state is not None
        assert conv_state.ndim == 3 and conv_state.shape[-1] >= 3
        assert conv_state.device == state.device
        conv_dim = conv_state.shape[1]
        assert 0 < conv_dim <= row_size
        assert conv_tail.ndim == 2 and conv_tail.shape[1] == conv_dim
        assert conv_tail.stride(1) == 1 and conv_tail.device == state.device
        assert conv_tail_indices.shape == (indices.numel(), 3)
        assert conv_tail_indices.stride(1) == 1
        assert conv_tail_indices.device == state.device
        assert conv_tail_indices.dtype in (torch.int32, torch.int64)
        assert has_initial_state.shape == indices.shape
        assert has_initial_state.dtype == torch.bool
        assert has_initial_state.device == state.device
        conv_strides = (
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
        )
        tail_stride = conv_tail.stride(0)
        tail_indices_stride = conv_tail_indices.stride(0)
        has_initial_stride = has_initial_state.stride(0)
    else:
        assert conv_tail is conv_tail_indices is has_initial_state is None
    block_size = min(triton.next_power_of_2(row_size), 1024)
    grid = (triton.cdiv(row_size, block_size), indices.numel())
    return grid, dict(
        # Explicit optional pointers also cover compile-only warmup inputs.
        conv_state_ptr=conv_state,
        conv_tail_ptr=conv_tail,
        conv_tail_indices_ptr=conv_tail_indices,
        has_initial_state_ptr=has_initial_state,
        stride_state_batch=state.stride(0),
        stride_src_batch=src.stride(0),
        stride_indices=indices.stride(0),
        stride_conv_batch=conv_strides[0],
        stride_conv_channel=conv_strides[1],
        stride_conv_token=conv_strides[2],
        stride_tail_token=tail_stride,
        stride_tail_indices=tail_indices_stride,
        stride_has_initial=has_initial_stride,
        row_size=row_size,
        CONV_DIM=conv_dim,
        BLOCK_SIZE=block_size,
        num_warps=8,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
