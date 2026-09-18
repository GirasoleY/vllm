# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The summary and merge kernels below are derived from the FLA context-parallel
# implementation (fla/ops/cp/chunk_delta_h.py, MIT license), specialized to the
# KDA gate convention (per-dim log2 gate cumsum `gk`, pre-gated `kg`, no scalar
# gate, no DPLR) and to serving-side zig-zag chunk sharding.
"""KCP (two-pass parallel scan) for KDA layers under MRV2 PCP.

MRV2 PCP zig-zag-shards every prefill into 2P chunks (rank r holds chunks r and
2P-1-r), so the gather-replicate KDA path redundantly scans the full sequence
on every rank. KCP instead runs a two-pass parallel scan per step:

1. Each rank computes, per chunk it owns, the chunk's affine state transition
   S_end = M @ S_in + S_ext (zero initial state) with ``kcp_summary_fwd_kernel``.
2. The [S_ext, M] summaries are all-gathered and chain-merged in global chunk
   order (fp32; bf16 M-chains diverge, see FLA PR #740), yielding every chunk's
   true initial state and the sequence-final state, which every rank writes to
   the mamba state cache (keeping the replicas coherent).
3. Each rank re-runs the regular chunked scan on its own rows, seeded with the
   merged initial state.

The short conv additionally needs the (kernel_size - 1)-token halo of every
chunk; each chunk's raw qkv tail is all-gathered and the per-chunk initial conv
window is assembled from the predecessor's tail (or the cached conv state for
the first chunk of a continued prefill).
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.third_party.flash_linear_attention.ops.op import exp2
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.autotune(
    # The key dims are architectural constants for GLM KDA (H=64, K=V=128,
    # BT=64), so a single measured-best GB300 config: a one-entry autotune
    # skips benchmarking and the first KCP step stays fast.
    configs=[triton.Config({}, num_warps=4, num_stages=2)],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def kcp_summary_fwd_kernel(
    k,  # kg: pre-gated key, [1, T, H, K]
    v,  # u: WY-processed value, [1, T, H, V]
    w,  # [1, T, H, K]
    gk,  # log2 gate cumsum, [1, T, H, K] fp32
    hm,  # out: [N, H, K, V + K] fp32, [S_ext | M] per row
    cu_seqlens,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
):
    """Per-chunk affine transition summary with zero initial state.

    One program per (column block, head, chunk row); columns [0, V) accumulate
    S_ext [K, V] and columns [V, V + K) accumulate M [K, K], chained in fp32.
    """
    i_col, i_h = tl.program_id(0), tl.program_id(1)
    i_n = tl.program_id(2).to(tl.int64)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = (eos - bos).to(tl.int32)
    NT = tl.cdiv(T, BT)

    hm += i_n * H * K * (K + V) + i_h * K * (K + V)
    k += (bos * H + i_h) * K
    w += (bos * H + i_h) * K
    gk += (bos * H + i_h) * K
    stride_k = H * K

    if i_col * BLOCK_SIZE < V:
        # S_ext part: h += kg^T @ (u - w @ h), h decayed per inner chunk.
        v += (bos * H + i_h) * V
        stride_v = H * V
        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_vb = i_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        m_vb = o_vb < V
        o_k1 = tl.arange(0, 64)
        m_k1 = o_k1 < K
        o_k2 = 64 + o_k1
        m_k2 = o_k2 < K
        o_k3 = 128 + o_k1
        m_k3 = o_k3 < K
        o_k4 = 192 + o_k1
        m_k4 = o_k4 < K

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_w = w + o_t[:, None] * stride_k + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = w + o_t[:, None] * stride_k + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_v = tl.dot(b_w, b_h2.to(b_w.dtype), b_v)
            if K > 128:
                p_w = w + o_t[:, None] * stride_k + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v = tl.dot(b_w, b_h3.to(b_w.dtype), b_v)
            if K > 192:
                p_w = w + o_t[:, None] * stride_k + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v = tl.dot(b_w, b_h4.to(b_w.dtype), b_v)

            p_v = v + o_t[:, None] * stride_v + o_vb[None, :]
            b_v = tl.load(p_v, mask=m_t[:, None] & m_vb[None, :], other=0.0) - b_v

            last_idx = min((i_t + 1) * BT, T) - 1
            p_gk_last = gk + last_idx * H * K
            b_gk_last = tl.load(p_gk_last + o_k1, mask=m_k1, other=0.0).to(tl.float32)
            b_h1 *= exp2(b_gk_last)[:, None]
            if K > 64:
                b_gk_last = tl.load(p_gk_last + o_k2, mask=m_k2, other=0.0).to(
                    tl.float32
                )
                b_h2 *= exp2(b_gk_last)[:, None]
            if K > 128:
                b_gk_last = tl.load(p_gk_last + o_k3, mask=m_k3, other=0.0).to(
                    tl.float32
                )
                b_h3 *= exp2(b_gk_last)[:, None]
            if K > 192:
                b_gk_last = tl.load(p_gk_last + o_k4, mask=m_k4, other=0.0).to(
                    tl.float32
                )
                b_h4 *= exp2(b_gk_last)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_h1 = tl.dot(b_k, b_v, b_h1)
            if K > 64:
                p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_h2 = tl.dot(b_k, b_v, b_h2)
            if K > 128:
                p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_h3 = tl.dot(b_k, b_v, b_h3)
            if K > 192:
                p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_h4 = tl.dot(b_k, b_v, b_h4)

        stride_hm = K + V
        p_h1 = hm + o_k1[:, None] * stride_hm + o_vb[None, :]
        m_h1 = m_k1[:, None] & m_vb[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            p_h2 = hm + o_k2[:, None] * stride_hm + o_vb[None, :]
            tl.store(
                p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_k2[:, None] & m_vb[None, :]
            )
        if K > 128:
            p_h3 = hm + o_k3[:, None] * stride_hm + o_vb[None, :]
            tl.store(
                p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_k3[:, None] & m_vb[None, :]
            )
        if K > 192:
            p_h4 = hm + o_k4[:, None] * stride_hm + o_vb[None, :]
            tl.store(
                p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_k4[:, None] & m_vb[None, :]
            )
    else:
        # M part: M = (Diag(exp2(gk_last)) - kg^T @ w) @ M per inner chunk.
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE
        b_m = tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            p_k = k + o_t[:, None] * stride_k + row[None, :]
            b_k = tl.load(p_k, mask=m_t[:, None] & (row < K)[None, :], other=0.0)
            p_w = w + o_t[:, None] * stride_k + row[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & (row < K)[None, :], other=0.0)

            last_idx = min((i_t + 1) * BT, T) - 1
            b_gk_last = tl.load(
                gk + last_idx * H * K + row, mask=(row < K), other=0.0
            ).to(tl.float32)
            b_diag = tl.where(
                row[:, None] == row[None, :], exp2(b_gk_last)[:, None], 0.0
            )
            b_m_i = b_diag - tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
            # fp32 (ieee) chain: bf16/tf32 rounding of M products diverges.
            b_m = tl.dot(b_m_i, b_m.to(tl.float32), input_precision="ieee")

        stride_hm = K + V
        p_m = hm + V + row[:, None] * stride_hm + col[None, :]
        m_m = (row < K)[:, None] & (col < K)[None, :]
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), mask=m_m)


def kcp_compute_summaries(
    kg: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    gk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Affine transition summary per cu_seqlens row, zero initial state."""
    B, T, H, K = kg.shape
    V = u.shape[-1]
    assert B == 1 and K <= 256
    N = cu_seqlens.shape[0] - 1
    BT = chunk_size
    BK1 = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    hm = kg.new_zeros(N, H, K, V + K, dtype=torch.float32)
    if N == 0:
        return hm
    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), H, N)
    kcp_summary_fwd_kernel[grid](
        k=kg,
        v=u,
        w=w,
        gk=gk,
        hm=hm,
        cu_seqlens=cu_seqlens,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK1,
    )
    return hm


@triton.autotune(
    # Single fixed config: see the summary kernel above.
    configs=[triton.Config({"BV": 64}, num_warps=4, num_stages=2)],
    key=["H", "K", "V"],
)
@triton.jit(do_not_specialize=["S", "N"])
def kcp_merge_fwd_kernel(
    hm,  # [S, N, H, K, V + K] fp32 slot-ordered summaries
    h0,  # [N, H, V, K] fp32 base states (zero where the request has no state)
    num_slots,  # [N] int32: non-empty chunk count per request
    init_out,  # [N, S, H, V, K] fp32: per-chunk initial states
    final_out,  # [N, H, V, K] fp32: sequence-final states
    S,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
):
    """Chain-merge chunk summaries into initial/final states (v-first states).

    State rows are [V, K] per head. The chain S' = M @ S + S_ext in [K, V]
    space becomes S'^T = S^T @ M^T + S_ext^T in [V, K] space. All products run
    in fp32 with ieee precision (see the summary kernel).
    """
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    L = tl.load(num_slots + i_n).to(tl.int32)

    p_h0 = h0 + (i_n * H + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    stride_hm_s = N * H * K * (V + K)
    stride_hm_h = K * (V + K)
    for c in range(L):
        p_init = (
            init_out
            + ((i_n * S + c) * H + i_h) * V * K
            + o_v[:, None] * K
            + o_k[None, :]
        )
        m_init = m_v[:, None] & m_k[None, :]
        tl.store(p_init, b_h.to(p_init.dtype.element_ty), mask=m_init)
        base = c * stride_hm_s + i_n * H * K * (V + K) + i_h * stride_hm_h
        p_he = hm + base + o_k[:, None] * (V + K) + o_v[None, :]
        b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0)
        p_m = hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0)
        b_h = tl.dot(b_h, tl.trans(b_m), input_precision="ieee") + tl.trans(b_he).to(
            tl.float32
        )

    p_final = final_out + (i_n * H + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
    m_final = m_v[:, None] & m_k[None, :]
    tl.store(p_final, b_h.to(p_final.dtype.element_ty), mask=m_final)


def kcp_merge_states_torch(
    slots_hm: torch.Tensor,
    base_state: torch.Tensor,
    num_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain-merge slot summaries via batched cuBLAS bmms (same math as the
    Triton kernel, but fully parallel across N*H per step instead of one
    sequential dot per (v-block, request, head) block)."""
    S, N, H, K, VK = slots_hm.shape
    V = VK - K
    init_states = slots_hm.new_empty(N, S, H, V, K)
    h = base_state
    for c in range(S):
        init_states[:, c] = h
        m_c = slots_hm[c, ..., V:]  # [N,H,K,K]
        se_c = slots_hm[c, ..., :V].transpose(-1, -2)  # [N,H,V,K]
        h_new = torch.matmul(h, m_c.transpose(-1, -2)) + se_c
        h = torch.where((c < num_slots).view(N, 1, 1, 1), h_new, h)
    return init_states, h


def kcp_merge_states(
    slots_hm: torch.Tensor,
    base_state: torch.Tensor,
    num_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chain-merge slot-ordered [S_ext, M] summaries into initial/final states.

    Args:
        slots_hm: [S, N, H, K, V + K] fp32 summaries in global chunk order.
        base_state: [N, H, V, K] fp32 per-request states at the step start
            (zeros for fresh prefills).
        num_slots: [N] int32 non-empty chunk count per request.

    Returns:
        (initial states [N, S, H, V, K] fp32, final states [N, H, V, K] fp32).

    """
    if envs.VLLM_KDA_KCP_MERGE != "kernel":
        return kcp_merge_states_torch(slots_hm, base_state, num_slots)
    S, N, H, K, _ = slots_hm.shape
    V = slots_hm.shape[-1] - K
    init_states = slots_hm.new_empty(N, S, H, V, K, dtype=torch.float32)
    final_states = slots_hm.new_empty(N, H, V, K, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    kcp_merge_fwd_kernel[grid](
        hm=slots_hm,
        h0=base_state,
        num_slots=num_slots,
        init_out=init_states,
        final_out=final_states,
        S=S,
        N=N,
        H=H,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
    )
    return init_states, final_states


def kcp_zigzag_slot_order(ag: torch.Tensor) -> torch.Tensor:
    """Reorder an all-gathered [world, ..., 2, ...] contribution to slot order.

    Rank r contributes its front chunk (slot r) at part index 0 and its back
    chunk (slot 2P-1-r) at part index 1, so global slot order is the front
    parts of ranks 0..P-1 followed by the back parts of ranks P-1..0.
    """
    front = ag[:, :, 0]
    back = ag[:, :, 1].flip(0)
    return torch.cat([front, back], dim=0)


# ---------------------------------------------------------------------------
# Per-step KCP layout plan
# ---------------------------------------------------------------------------


@dataclass
class KcpPlan:
    """Per-step layout for the KCP path (built once, shared by all KDA layers).

    All index tensors are GPU-resident; the rest is CPU metadata. ``scan``
    tensors order this rank's local prefill chunk rows by (request, slot);
    ``block`` tensors cover the gathered global non-KCP rows (plain decodes,
    spec verifies and single-token extends), which every rank computes
    redundantly to keep the replicated mamba caches coherent.
    """

    world: int
    rank: int
    num_prefill_reqs: int  # N_pf
    num_slots: int  # 2 * world
    global_prefill_tokens: int
    # This rank's scan rows (chunk slots of KCP-prefill requests).
    num_scan_rows: int
    prefill_src_idx: torch.Tensor  # [T_loc] int64, local token idx in scan order
    scan_cu_seqlens: torch.Tensor  # [N_loc + 1] int32
    scan_chunk_indices: torch.Tensor  # [NT, 2] int32, FLA chunking of the scan
    scan_chunk_offsets: torch.Tensor  # [N_loc + 1] int32
    conv_meta: object  # SimpleNamespace(nums_dict, batch_ptr, token_chunk_offset_ptr)
    conv_cache_indices: torch.Tensor  # [N_loc] int32 arange (scratch conv states)
    conv_all_initial: torch.Tensor  # [N_loc] bool, all True (halo always given)
    scan_req_idx: torch.Tensor  # [N_loc] int64, KCP-prefill request ordinal
    summary_dst_idx: torch.Tensor  # [N_loc] int64, position in the [N, 2] contribution
    init_gather_idx: torch.Tensor  # [N_loc] int64, position in the [N, 2P] inits
    tail_dst_idx: torch.Tensor  # [M] int64, into [N * 2 * 3] tail contribution
    tail_src_idx: torch.Tensor  # [M] int64, scan-order qkv row of each tail row
    halo_tail_idx: torch.Tensor  # [N_loc, 3] int64; -3..-1 address cached prefix
    num_slots_dev: torch.Tensor  # [N] int32, non-empty chunk slots per request
    prefill_entry_idx: torch.Tensor  # [N] int64, index into non-spec metadata rows
    final_win_idx: torch.Tensor  # [N] int64, into slot-ordered [S * N] conv windows
    # Non-KCP (block) rows: gathered and computed redundantly on every rank.
    num_block_tokens: int  # D_g
    block_cap: int  # per-rank padded block-token count used for the all-gather
    block_src_idx: torch.Tensor  # [cnt] int64, this rank's block tokens in [P]
    block_restore_idx: torch.Tensor  # [D_g] int64, into the [world * cap] gather
    block_spec_token_idx: torch.Tensor  # int64, spec-verify tokens in the block
    block_dec_token_idx: torch.Tensor  # int64, plain-decode/extend tokens
    block_num_spec_reqs: int
    block_num_dec_reqs: int  # plain decodes + single-token extends
    nonprefill_out_src: torch.Tensor  # int64, block rows to copy out
    nonprefill_out_dst: torch.Tensor  # int64, their local token positions

    @property
    def has_block(self) -> bool:
        return self.num_block_tokens > 0


def _rank_segments(mgr, q_g, is_pref, start0, qsl, rank):
    """Use the manager's partition, including replicated decode requests."""
    return [
        (
            segment.global_batch_req_idx,
            int(start0[segment.global_batch_req_idx])
            + segment.global_batch_slice.start
            - int(qsl[segment.global_batch_req_idx]),
            segment.num_tokens,
        )
        for segment in mgr._get_rank_segments(rank, q_g, is_pref, qsl)
    ]


def build_kcp_plan(mgr, device: torch.device) -> KcpPlan | None:
    """Compute the KCP layout for the current partition, or None to fall back.

    Runs on CPU from the manager's global/local batches. Returns None (uniform
    across ranks) whenever the batch layout does not fit the KCP assumptions.
    """
    gb = mgr.global_batch
    lb = mgr.local_batch
    if gb is None or lb is None:
        return None
    world = mgr.pcp_world_size
    rank = mgr.pcp_rank
    if world <= 1:
        return None

    q_g = gb.num_scheduled_tokens.astype(np.int64)
    is_pref = gb.is_prefilling_np
    start0 = gb.num_computed_tokens_np.astype(np.int64)
    qsl = gb.query_start_loc_np.astype(np.int64)
    N_g = gb.num_reqs
    drafts = gb.num_draft_tokens_per_req
    if drafts is not None:
        # Mirrors the spec-row classification feeding the GDN metadata builder.
        is_spec = (~is_pref) & (drafts > 0) & (q_g == drafts + 1)
    else:
        is_spec = np.zeros(N_g, dtype=bool)
    # Replicated requests and single-token extends use the redundant path,
    # keeping the recurrent caches coherent across PCP ranks.
    kcp_pref = is_pref & (q_g > 1) & ~mgr.replicated_requests(q_g, is_pref)
    is_block = ~kcp_pref & (q_g > 0)

    prefill_tokens = int(q_g[is_pref].sum())
    if not kcp_pref.any() or np.any(is_block & is_pref):
        return None
    # The block must be a prefix of the global token axis (the runner's
    # decode-first ordering); the metadata's token indices then index it
    # directly.
    seen_kcp_pref = False
    for r in range(N_g):
        if kcp_pref[r]:
            seen_kcp_pref = True
        elif is_block[r] and seen_kcp_pref:
            return None

    n_list = np.nonzero(kcp_pref)[0]
    N_pf = len(n_list)
    S = 2 * world
    cs = (q_g[n_list] + S - 1) // S
    slot_len = np.clip(
        q_g[n_list][:, None] - np.arange(S)[None, :] * cs[:, None], 0, None
    )
    slot_len = np.minimum(slot_len, cs[:, None])
    num_slots_np = (slot_len > 0).sum(axis=1).astype(np.int32)

    # This rank's scan rows and block rows from the local batch (ground truth
    # for local token positions).
    req_pos = {req_id: i for i, req_id in enumerate(gb.req_ids)}
    cs_by_g = {int(g): int(cs[n]) for n, g in enumerate(n_list)}
    scan_rows: list[tuple[int, int, int, int]] = []  # (n, slot, tok_start, len)
    block_src: list[int] = []
    block_out_map: list[tuple[int, int]] = []  # (block token pos, local pos)
    lb_qsl = lb.query_start_loc_np
    for i in range(lb.num_reqs):
        g = req_pos[lb.req_ids[i]]
        t0 = int(lb_qsl[i])
        qlen = int(lb.num_scheduled_tokens[i])
        a_start = int(lb.num_computed_tokens_np[i])
        if qlen == 0:
            # Zero-length dummy row for a rank owning no segments this step.
            continue
        if kcp_pref[g] and lb.is_prefilling_np[i]:
            c = (a_start - int(start0[g])) // cs_by_g[g]
            n = int(np.searchsorted(n_list, g))
            # The partition gives this rank exactly slots (rank, 2P-1-rank) of
            # each prefilling request; anything else is a partition bug and
            # must fail loudly (a per-rank fallback would deadlock KCP).
            assert c < num_slots_np[n] and c in (rank, 2 * world - 1 - rank), (
                f"KCP plan: local row for request {g} maps to slot {c}, "
                f"expected one of {(rank, 2 * world - 1 - rank)}"
            )
            scan_rows.append((n, int(c), t0, qlen))
        else:
            for j in range(qlen):
                block_src.append(t0 + j)
                block_out_map.append((int(qsl[g] + (a_start - start0[g]) + j), t0 + j))
    scan_rows.sort(key=lambda row: (row[0], row[1]))

    # Cross-check the simulated layout for this rank against the actual local
    # batch; a mismatch means the simulation drifted from PCPManager. The
    # mapping is (global token position, local token position) pairs.
    sim = _rank_segments(mgr, q_g, is_pref, start0, qsl, rank)
    sim_tokens = sorted(
        (int(qsl[g]) + (a - int(start0[g])) + j, tok0 + j)
        for g, a, length, tok0 in (
            (g, a, length, tok) for g, a, length, tok in _segments_with_offsets(sim)
        )
        for j in range(length)
    )
    local_tokens = sorted(
        (
            int(qsl[g]) + (int(lb.num_computed_tokens_np[i]) - int(start0[g])) + j,
            int(lb_qsl[i]) + j,
        )
        for i in range(lb.num_reqs)
        for g in [req_pos[lb.req_ids[i]]]
        for j in range(int(lb.num_scheduled_tokens[i]))
    )
    if sim_tokens != local_tokens:
        # The simulation must reproduce PCPManager's partition exactly; a
        # mismatch is a bug, and falling back per-rank would deadlock the KCP
        # collectives, so fail loudly.
        raise AssertionError(
            "KCP plan: simulated PCP partition does not match the local batch"
        )

    # Every rank's block-token layout (for the block all-gather restore
    # mapping); each rank pads its contribution to `cap` rows.
    cap = 0
    per_rank_block: list[list[int]] = []
    for r_ in range(world):
        toks: list[int] = []
        for g, a, length, _tok in _segments_with_offsets(
            _rank_segments(mgr, q_g, is_pref, start0, qsl, r_)
        ):
            if kcp_pref[g]:
                continue
            toks.extend(int(qsl[g] + (a - int(start0[g])) + j) for j in range(length))
        per_rank_block.append(toks)
        cap = max(cap, len(toks))
    D_g = int(q_g[is_block].sum())
    block_restore = np.empty(D_g, dtype=np.int64)
    for r_, toks in enumerate(per_rank_block):
        for j, gpos in enumerate(toks):
            block_restore[gpos] = r_ * cap + j

    # Spec-verify vs plain-decode split inside the block (global order).
    block_spec_reqs = np.nonzero(is_spec)[0]
    block_dec_reqs_np = np.nonzero(is_block & ~is_spec)[0]
    block_spec_token = (
        np.concatenate([np.arange(qsl[g], qsl[g] + q_g[g]) for g in block_spec_reqs])
        if len(block_spec_reqs)
        else np.empty(0, dtype=np.int64)
    )
    block_dec_token = (
        np.concatenate([np.arange(qsl[g], qsl[g] + q_g[g]) for g in block_dec_reqs_np])
        if len(block_dec_reqs_np)
        else np.empty(0, dtype=np.int64)
    )

    prefill_src: list[int] = []
    cu = [0]
    scan_req: list[int] = []
    scan_slot: list[int] = []
    tail_dst: list[int] = []
    tail_src: list[int] = []
    halo_tail = np.full((len(scan_rows), 3), -1, dtype=np.int64)
    for row_i, (n, c, t0, qlen) in enumerate(scan_rows):
        prefill_src.extend(range(t0, t0 + qlen))
        cu.append(cu[-1] + qlen)
        scan_req.append(n)
        scan_slot.append(c)
        # This row's contribution to the tails all-gather: its last
        # min(3, qlen) raw qkv rows, right-aligned in the 3 tail columns.
        take = min(3, qlen)
        for j in range(take):
            tail_dst.append((n * 2 + (0 if c == rank else 1)) * 3 + 3 - take + j)
            tail_src.append(cu[row_i] + qlen - take + j)
        # The conv halo of chunk c = the 3 rows preceding it: walk the
        # request's earlier slots (then the cached conv-state prefix).
        window: list[tuple[int, int]] = [(-1, j) for j in range(3)]
        for c_ in range(c):
            take_ = min(3, int(slot_len[n, c_]))
            cols = [(c_, 3 - take_ + j) for j in range(take_)]
            window = (window + cols)[-3:]
        for j, (w_slot, w_col) in enumerate(window):
            if w_slot >= 0:
                halo_tail[row_i, j] = (w_slot * N_pf + n) * 3 + w_col
            else:
                halo_tail[row_i, j] = w_col - 3

    scan_cu_cpu = torch.tensor(cu, dtype=torch.int32)
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        scan_cu_cpu, device=device
    )
    # Entry of each KCP-prefill request among the metadata's non-spec rows
    # (global request order with spec rows removed).
    spec_cumsum = np.cumsum(is_spec.astype(np.int64))
    prefill_entry = np.array(
        [int(g) - int(spec_cumsum[g]) for g in n_list], dtype=np.int64
    )

    to_gpu_i64 = lambda a: torch.tensor(a, dtype=torch.int64, device=device)
    return KcpPlan(
        world=world,
        rank=rank,
        num_prefill_reqs=N_pf,
        num_slots=S,
        global_prefill_tokens=prefill_tokens,
        num_scan_rows=len(scan_rows),
        prefill_src_idx=to_gpu_i64(prefill_src),
        scan_cu_seqlens=scan_cu_cpu.to(device, non_blocking=True),
        scan_chunk_indices=(
            prepare_chunk_indices(scan_cu_cpu, FLA_CHUNK_SIZE).to(
                device, non_blocking=True
            )
            if scan_rows
            else torch.empty((0, 2), dtype=torch.int32, device=device)
        ),
        scan_chunk_offsets=prepare_chunk_offsets(scan_cu_cpu, FLA_CHUNK_SIZE).to(
            device=device, dtype=torch.int32, non_blocking=True
        ),
        conv_meta=SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        ),
        conv_cache_indices=torch.arange(
            # causal_conv1d treats cache index 0 as the null block and skips it;
            # the scratch conv states start at row 1.
            1,
            len(scan_rows) + 1,
            dtype=torch.int32,
            device=device,
        ),
        conv_all_initial=torch.ones(len(scan_rows), dtype=torch.bool, device=device),
        scan_req_idx=to_gpu_i64(scan_req),
        summary_dst_idx=to_gpu_i64(
            [n * 2 + (0 if c == rank else 1) for n, c, _, _ in scan_rows]
        ),
        init_gather_idx=to_gpu_i64([n * S + c for n, c, _, _ in scan_rows]),
        tail_dst_idx=to_gpu_i64(tail_dst),
        tail_src_idx=to_gpu_i64(tail_src),
        halo_tail_idx=to_gpu_i64(halo_tail),
        num_slots_dev=torch.tensor(num_slots_np, dtype=torch.int32, device=device),
        prefill_entry_idx=to_gpu_i64(prefill_entry),
        final_win_idx=to_gpu_i64(
            (num_slots_np.astype(np.int64) - 1) * N_pf + np.arange(N_pf)
        ),
        num_block_tokens=D_g,
        block_cap=cap,
        block_src_idx=to_gpu_i64(block_src),
        block_restore_idx=to_gpu_i64(block_restore),
        block_spec_token_idx=to_gpu_i64(block_spec_token),
        block_dec_token_idx=to_gpu_i64(block_dec_token),
        block_num_spec_reqs=int(len(block_spec_reqs)),
        block_num_dec_reqs=int(len(block_dec_reqs_np)),
        nonprefill_out_src=to_gpu_i64([b for b, _ in block_out_map]),
        nonprefill_out_dst=to_gpu_i64([loc for _, loc in block_out_map]),
    )


def _segments_with_offsets(
    segments: list[tuple[int, int, int]],
) -> list[tuple[int, int, int, int]]:
    out = []
    offset = 0
    for req, abs_start, length in segments:
        out.append((req, abs_start, length, offset))
        offset += length
    return out


# ---------------------------------------------------------------------------
# Layer-time collectives
# ---------------------------------------------------------------------------


def gather_block_hidden(plan: KcpPlan, hidden_states: torch.Tensor) -> torch.Tensor:
    """All-gather the non-KCP rows' hidden states in global batch order."""
    cnt = plan.block_src_idx.shape[0]
    padded = hidden_states.new_zeros(plan.block_cap, hidden_states.shape[-1])
    if cnt:
        padded[:cnt] = hidden_states.index_select(0, plan.block_src_idx)
    gathered = get_pcp_group().all_gather(padded, dim=0)
    return gathered.index_select(0, plan.block_restore_idx)


def gather_slot_summaries(plan: KcpPlan, hm_local: torch.Tensor) -> torch.Tensor:
    """All-gather per-slot [S_ext, M] summaries into global slot order.

    Returns [2P, N, H, K, V + K] fp32, with zeros in slots no rank owns.
    """
    N = plan.num_prefill_reqs
    contrib = hm_local.new_zeros(N * 2, *hm_local.shape[1:])
    if plan.num_scan_rows:
        contrib[plan.summary_dst_idx] = hm_local
    ag = get_pcp_group().all_gather(contrib, dim=0)
    ag = ag.view(plan.world, N, 2, *hm_local.shape[1:])
    return kcp_zigzag_slot_order(ag)


def gather_conv_tail_rows(plan: KcpPlan, tail_rows: torch.Tensor) -> torch.Tensor:
    """All-gather per-slot raw-qkv tails into slot order ([2P, N, 3, C])."""
    N = plan.num_prefill_reqs
    contrib = tail_rows.new_zeros(N * 2 * 3, tail_rows.shape[-1])
    if plan.tail_dst_idx.numel():
        contrib[plan.tail_dst_idx] = tail_rows[plan.tail_src_idx]
    ag = get_pcp_group().all_gather(contrib, dim=0)
    ag = ag.view(plan.world, N, 2, 3, tail_rows.shape[-1])
    return kcp_zigzag_slot_order(ag)


def gather_conv_windows(plan: KcpPlan, windows: torch.Tensor) -> torch.Tensor:
    """All-gather per-slot post-conv state windows into slot order.

    ``windows`` holds this rank's scan rows' final windows [N_loc, C, 3] (the
    conv kernel's writeback); returns [2P, N, C, 3] with zeros in unowned
    slots.
    """
    N = plan.num_prefill_reqs
    contrib = windows.new_zeros(N * 2, *windows.shape[1:])
    if plan.num_scan_rows:
        contrib[plan.summary_dst_idx] = windows
    ag = get_pcp_group().all_gather(contrib, dim=0)
    ag = ag.view(plan.world, N, 2, *windows.shape[1:])
    return kcp_zigzag_slot_order(ag)


def maybe_get_kcp_plan(mgr, min_tokens: int) -> KcpPlan | None:
    """Return the active KCP plan for this step, or None to use GR.

    Cached per partitioned batch on the manager; the layout is identical on
    every rank, so the KCP/GR decision is uniform across the PCP group. The
    threshold is checked against the global batch before building the plan:
    below-threshold steps (e.g. kernel-warmup batches) never pay for the
    layout computation.
    """
    gb = mgr.global_batch
    if gb is None:
        return None
    local_batch = mgr.local_batch
    cache = getattr(mgr, "_kcp_plan_cache", None)
    if cache is not None and cache[0] is local_batch:
        return cache[1]
    plan = None
    q_g = gb.num_scheduled_tokens.astype(np.int64)
    if int(q_g[gb.is_prefilling_np].sum()) >= min_tokens:
        plan = build_kcp_plan(mgr, mgr.device)
        if plan is not None:
            logger.info_once(
                "KCP engaged: world=%d prefill_tokens=%d",
                plan.world,
                plan.global_prefill_tokens,
            )
    mgr._kcp_plan_cache = (local_batch, plan)
    return plan
