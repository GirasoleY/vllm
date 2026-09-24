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
import torch.nn.functional as F

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


def kcp_merge_states(
    slots_hm: torch.Tensor,
    base_state: torch.Tensor,
    num_slots: torch.Tensor,
    *,
    all_slots_full: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge rank/part summaries in chronological slot order using FP32.

    Consumes S_ext as scratch. Returns all slot initial states and a contiguous
    final state. The CPU plan certifies all_slots_full; absent slots retain
    the preceding state.
    """
    S, N, H, K, VK = slots_hm.shape
    V = VK - K
    init_states = slots_hm.new_empty(N, S, H, V, K)
    h = base_state
    for c in range(S):
        init_states[:, c] = h
        physical_c = 2 * c if c < S // 2 else 2 * (S - 1 - c) + 1
        m_c = slots_hm[physical_c, ..., V:]  # [N,H,K,K]
        se_c = slots_hm[physical_c, ..., :V].transpose(-1, -2)  # [N,H,V,K]
        se_c.view(N * H, V, K).baddbmm_(
            h.reshape(N * H, V, K),
            m_c.transpose(-1, -2).view(N * H, K, K),
        )
        h = (
            se_c
            if all_slots_full
            else torch.where((c < num_slots).view(N, 1, 1, 1), se_c, h)
        )
    # The cache scatter consumes a contiguous state for each request.
    return init_states, h.contiguous()


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


def conv_windows(
    plan: "HybridPCPPlan",
    tails: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble segment initial and request final windows as [rows, C, halo].

    ``tails`` are the gathered tails of every rank; ``conv_state`` is the
    (..., C, halo) cache view and ``state_indices`` the prefill requests' slots.
    """
    halo = plan.halo_size
    assert conv_state.shape[-1] == halo
    prefix = conv_state[state_indices].transpose(1, 2)
    # Fresh requests start from zeros regardless of the slot's stale contents.
    prefix = torch.where(
        plan.prefill_has_initial_state.view(-1, 1, 1), prefix, prefix.new_zeros(())
    )
    pool = torch.cat((prefix.reshape(-1, prefix.shape[-1]), tails.to(prefix.dtype)))
    # Row 0 is NULL_BLOCK_ID to the conv kernel; segments use rows 1..L.
    initial = pool[F.pad(plan.halo_idx, (0, 0, 1, 0))].transpose(1, 2)
    return initial, pool[plan.final_halo_idx].transpose(1, 2)


def local_summaries(
    plan: "HybridPCPPlan", prepared: ChunkKdaPrepared | None, base: torch.Tensor
) -> torch.Tensor:
    """Summarize this rank's segments in the [2 * N] exchange layout."""
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    N, H, V, K = base.shape
    summaries = base.new_empty(2 * N, H, K, V + K, dtype=torch.float32)
    if plan.num_segments != 2 * N:
        summaries.zero_()
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
    slots = 2 * plan.world
    inits, final = kcp_merge_states(
        gathered.view(slots, N, H, K, V + K),
        base.float(),
        plan.num_slots,
        all_slots_full=plan.all_slots_full,
    )
    return inits.view(N * slots, H, V, K)[plan.init_idx], final


def exchange_states(
    plan: "HybridPCPPlan", prepared: ChunkKdaPrepared | None, base: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange summaries; ranks without prefill tokens pass ``prepared=None``."""
    gathered = get_pcp_group().all_gather(local_summaries(plan, prepared, base), 0)
    return merge_summaries(plan, gathered, base)
