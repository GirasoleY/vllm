# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The affine summary and merge equations follow the FLA context-parallel
# implementation (fla/ops/cp/chunk_delta_h.py, MIT license), specialized to the
# KDA gate convention (per-dim log2 gate cumsum `gk`, pre-gated `kg`, no scalar
# gate, no DPLR) and to serving-side zig-zag chunk sharding.
"""KCP (two-pass parallel scan) for KDA layers under hybrid PCP on Blackwell.

Each rank runs the ordinary chunked-KDA preparation on its prefill segments
(see ``HybridPCPPlan`` in ``vllm.v1.worker.gpu.pcp_manager``), then:

1. Computes every segment's affine state transition S_end = M @ S_in + S_ext
   from zero initial state with the native S and M kernels.
2. All-gathers the [S_ext, M] summaries and chain-merges them in chunk order
   (fp32; bf16 M-chains diverge, see FLA PR #740). This yields every segment's
   initial state and every request's final state; all ranks publish the final
   state so the replicated state caches stay coherent.
3. Runs the ordinary scan on its segments from the merged initial states.

The short convolution needs each segment's preceding (kernel_size - 1) raw
inputs. Ranks all-gather their segment tails, assemble every initial and final
window from the tails and the cached prefix, and run the ordinary kernels.
"""

from typing import TYPE_CHECKING

import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.triton_utils import tl, triton

from .third_party.kda import ChunkKdaPrepared

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.pcp_manager import HybridPCPPlan


def kcp_compute_summaries(
    kg: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    *,
    out: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute native affine summaries for packed, possibly empty segments.

    Metadata describes consecutive segments of the packed token dimension.
    Mapped destinations must be unique and in bounds; only those destinations
    are written. The caller initializes absent slots. Metadata values stay on
    the GPU, while shape and dtype checks require no device synchronization.
    """
    assert kg.ndim == 4 and kg.shape[0] == 1
    _, tokens, heads, key_dim = kg.shape
    assert u.ndim == 4 and u.shape[:3] == kg.shape[:3]
    assert w.shape == gk.shape == kg.shape
    assert heads > 0 and 0 < key_dim <= 256 and u.shape[-1] > 0
    assert kg.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert kg.dtype == u.dtype == w.dtype
    assert gk.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 1
    assert cu_seqlens.dtype in (torch.int32, torch.int64)
    assert kg.is_cuda and all(x.device == kg.device for x in (u, w, gk, cu_seqlens))
    assert chunk_size >= 16 and chunk_size & (chunk_size - 1) == 0
    segments = cu_seqlens.numel() - 1
    value_dim = u.shape[-1]
    if out is None:
        assert output_indices is None
        out = kg.new_empty(
            segments, heads, key_dim, value_dim + key_dim, dtype=torch.float32
        )
    else:
        assert out.ndim == 4
        assert out.shape[1:] == (heads, key_dim, value_dim + key_dim)
        assert out.device == kg.device and out.dtype == torch.float32
        assert output_indices is not None and output_indices.shape == (segments,)
        assert output_indices.device == kg.device
        assert output_indices.dtype in (torch.int32, torch.int64)
    if segments == 0:
        assert tokens == 0
        return out

    # Import lazily so the native CUDA dependency is only needed for KCP.
    from .third_party.kda.kcp import kcp_summary_m_cutedsl, kcp_summary_s_cutedsl

    kcp_summary_s_cutedsl(
        kg, u, w, gk, cu_seqlens, out, output_indices, chunk_size=chunk_size
    )
    kcp_summary_m_cutedsl(
        kg, w, gk, cu_seqlens, out, output_indices, chunk_size=chunk_size
    )
    return out


@triton.jit
def _gather_merge_states_kernel(
    base,
    slots_hm,
    rows,
    out,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    """Copy value-first [V, K] states from base rows or merged summary rows."""
    row, h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    v = tl.program_id(2).to(tl.int64) * BV + tl.arange(0, BV)[:, None]
    k = tl.arange(0, K)[None, :]
    mask = v < V
    source = tl.load(rows + row).to(tl.int64)
    if source < N:
        value = tl.load(base + ((source * H + h) * V + v) * K + k, mask=mask)
    else:
        # Summary rows are [K, V + K]; the merged state is its transposed S_ext.
        summary = (source - N) * H + h
        value = tl.load(slots_hm + (summary * K + k) * (V + K) + v, mask=mask)
    tl.store(out + ((row * H + h) * V + v) * K + k, value, mask=mask)


def kcp_merge_states(
    slots_hm: torch.Tensor,
    base_state: torch.Tensor,
    merge_rows: torch.Tensor,
    num_segments: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge rank/part summaries in chronological slot order using FP32.

    ``slots_hm`` is [2P physical slots, N, H, K, V + K]; absent slots must hold
    the identity transition and a zero S_ext, so every slot is one exact step.
    Each step's FP32 GEMM overwrites that slot's S_ext with the state after it.
    ``merge_rows`` (see ``merge_source_rows``) selects each segment's initial
    state, then each request's final state. Both results are value-first and
    contiguous.
    """
    S, N, H, K, VK = slots_hm.shape
    V = VK - K
    assert slots_hm.is_contiguous() and base_state.is_contiguous()
    after = slots_hm[..., :V].transpose(-1, -2)  # [S, N, H, V, K]
    transposed = slots_hm[..., V:].transpose(-1, -2)
    h = base_state.view(N * H, V, K)
    for c in range(S):
        physical = 2 * c if c < S // 2 else 2 * (S - 1 - c) + 1
        state = after[physical].view(N * H, V, K)
        state.baddbmm_(h, transposed[physical].view(N * H, K, K))
        h = state
    out = base_state.new_empty(merge_rows.shape[0], H, V, K)
    block = 32
    _gather_merge_states_kernel[(merge_rows.shape[0], H, triton.cdiv(V, block))](
        base_state, slots_hm, merge_rows, out, N, H=H, K=K, V=V, BV=block
    )
    return out[:num_segments], out[num_segments:]


@triton.jit
def _pack_conv_tail_rows(
    x,
    source_indices,
    out,
    stride_token: tl.int64,
    stride_channel: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    channel_tiles: tl.constexpr = tl.cdiv(C, BLOCK)
    token = (tl.program_id(0) // channel_tiles).to(tl.int64)
    channel = (tl.program_id(0) % channel_tiles).to(tl.int64) * BLOCK
    channel += tl.arange(0, BLOCK)
    source = tl.load(source_indices + token).to(tl.int64)
    value = tl.load(
        x + tl.maximum(source, 0) * stride_token + channel * stride_channel,
        (source >= 0) & (channel < C),
        other=0.0,
    )
    tl.store(out + token * C + channel, value, channel < C)


def pack_conv_tails(plan: "HybridPCPPlan", qkv: torch.Tensor) -> torch.Tensor:
    """Pack this rank's raw prefill tails as [N * 2 * halo, C] rows."""
    rows, channels = plan.tail_src_idx.numel(), qkv.shape[-1]
    tails = qkv.new_empty(rows, channels)
    _pack_conv_tail_rows[(rows * triton.cdiv(channels, 256),)](
        qkv,
        plan.tail_src_idx,
        tails,
        qkv.stride(0),
        qkv.stride(1),
        channels,
        BLOCK=256,
    )
    return tails


@triton.jit
def _conv_windows_kernel(
    conv_state,
    state_indices,
    has_initial_state,
    tails,
    pool_rows,
    out,
    stride_state_slot,
    stride_state_channel,
    stride_state_column,
    stride_out_row,
    stride_out_channel,
    stride_out_column,
    row_offset,
    num_prefix_rows,
    C: tl.constexpr,
    HALO: tl.constexpr,
    OUT_BY_SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Select HALO raw inputs per row from cached prefixes and gathered tails."""
    row = tl.program_id(0).to(tl.int64)
    channel = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = channel < C
    values = ()  # type: tuple
    for column in tl.static_range(HALO):
        source = tl.load(pool_rows + row * HALO + column).to(tl.int64)
        from_prefix = source < num_prefix_rows
        request = tl.where(from_prefix, source // HALO, 0)
        slot = tl.load(state_indices + request).to(tl.int64)
        continued = tl.load(has_initial_state + request).to(tl.int1)
        prefix = tl.load(
            conv_state
            + slot * stride_state_slot
            + channel * stride_state_channel
            + (source % HALO) * stride_state_column,
            mask=mask & from_prefix & continued,
            other=0.0,
        )
        tail = tl.load(
            tails + tl.maximum(source - num_prefix_rows, 0) * C + channel,
            mask=mask & (source >= num_prefix_rows),
            other=0.0,
        )
        values += (tl.where(from_prefix, prefix, tail.to(prefix.dtype)),)
    # Load every column before storing: the final windows overwrite the prefix.
    if OUT_BY_SLOT:
        destination = tl.load(state_indices + row).to(tl.int64)
    else:
        destination = row + row_offset
    for column in tl.static_range(HALO):
        tl.store(
            out
            + destination * stride_out_row
            + channel * stride_out_channel
            + column * stride_out_column,
            values[column],
            mask=mask,
        )


def _launch_conv_windows(
    plan, tails, conv_state, state_indices, rows, out, *, row_offset, out_by_slot
):
    channels = conv_state.shape[1]
    assert tails.shape[-1] == channels and tails.is_contiguous()
    if rows.shape[0] == 0:
        return
    block = 1024
    _conv_windows_kernel[(rows.shape[0], triton.cdiv(channels, block))](
        conv_state,
        state_indices,
        plan.prefill_has_initial_state,
        tails,
        rows,
        out,
        *conv_state.stride(),
        *out.stride(),
        row_offset,
        plan.num_prefill_reqs * plan.halo_size,
        C=channels,
        HALO=plan.halo_size,
        OUT_BY_SLOT=out_by_slot,
        BLOCK=block,
    )


def conv_windows(
    plan: "HybridPCPPlan",
    tails: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Build segment initial windows as [1 + L, C, halo] in the cache's layout.

    ``tails`` are the gathered tails of every rank; ``conv_state`` is the
    (..., C, halo) cache view and ``state_indices`` the prefill requests' slots.
    Row 0 is NULL_BLOCK_ID to the conv kernel; segments use rows 1..L.
    """
    assert conv_state.shape[-1] == plan.halo_size
    shape = (plan.num_segments + 1, conv_state.shape[1], plan.halo_size)
    if conv_state.stride(-1) == 1:
        windows = conv_state.new_empty(shape)
    else:
        windows = conv_state.new_empty(shape[0], shape[2], shape[1]).transpose(1, 2)
    _launch_conv_windows(
        plan,
        tails,
        conv_state,
        state_indices,
        plan.halo_idx,
        windows,
        row_offset=1,
        out_by_slot=False,
    )
    return windows


def publish_conv_windows(
    plan: "HybridPCPPlan",
    tails: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Write every prefill request's final window into its conv cache slot."""
    _launch_conv_windows(
        plan,
        tails,
        conv_state,
        state_indices,
        plan.final_halo_idx,
        conv_state,
        row_offset=0,
        out_by_slot=True,
    )


def local_summaries(
    plan: "HybridPCPPlan", prepared: ChunkKdaPrepared | None, base: torch.Tensor
) -> torch.Tensor:
    """Summarize this rank's segments in the [2 * N] exchange layout."""
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    N, H, V, K = base.shape
    summaries = base.new_empty(2 * N, H, K, V + K, dtype=torch.float32)
    if plan.num_segments != 2 * N:
        # Absent parts are identity steps: zero S_ext, identity transition.
        summaries.zero_()
        summaries[..., V:].diagonal(dim1=-2, dim2=-1).fill_(1.0)
    if prepared is not None:
        kcp_compute_summaries(
            kg=prepared.kg,
            u=prepared.u,
            w=prepared.w,
            gk=prepared.g,
            cu_seqlens=plan.scan_cu_seqlens,
            chunk_size=FLA_CHUNK_SIZE,
            out=summaries,
            output_indices=plan.summary_idx,
        )
    return summaries


def merge_summaries(
    plan: "HybridPCPPlan", gathered: torch.Tensor, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return segment initial states and request final states, value-first."""
    N, H, V, K = base.shape
    return kcp_merge_states(
        gathered.view(2 * plan.world, N, H, K, V + K),
        base.float().contiguous(),
        plan.merge_rows,
        plan.num_segments,
    )


def exchange_states(
    plan: "HybridPCPPlan", prepared: ChunkKdaPrepared | None, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange summaries; ranks without prefill tokens pass ``prepared=None``."""
    gathered = get_pcp_group().all_gather(local_summaries(plan, prepared, base), 0)
    return merge_summaries(plan, gathered, base)
