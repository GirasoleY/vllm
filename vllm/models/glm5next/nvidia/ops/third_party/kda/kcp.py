# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-pass KDA context parallelism for MRV2's zigzag PCP partition.

Each rank prepares its local segments with FlashKDA and computes their affine
state summaries. The native zero-state scan supplies S_ext; a separate CUDA
kernel retains FP32 transition matrices M. Gathering and merging [S_ext | M]
in global segment order supplies the true initial states for a second native
scan, which reuses the prepared workspace. Final states are published on every
rank so decode sees coherent caches.

The convolution additionally exchanges each segment's three-token left halo.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def kcp_merge_local_transitions_kernel(
    Parts, Zero, Out, Destinations, H: tl.constexpr, MAPPED: tl.constexpr
):
    column_block, head, segment = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    keys = tl.arange(0, 128)
    columns = column_block * 64 + tl.arange(0, 64)
    transition = (keys[:, None] == columns[None, :]).to(tl.float32)
    for part in range(4):
        partial = tl.load(
            Parts
            + (((segment * 4 + part) * H + head) * 128 + keys[:, None]) * 128
            + keys[None, :]
        )
        transition = tl.dot(partial, transition, input_precision="tf32x3")
    destination = segment
    if MAPPED:
        destination = tl.load(Destinations + segment)
    output = Out + ((destination * H + head) * 128 + keys[:, None]) * 256
    tl.store(output + 128 + columns[None, :], transition)
    additive = tl.load(
        Zero + ((segment * H + head) * 128 + columns[None, :]) * 128 + keys[:, None]
    )
    tl.store(output + columns[None, :], additive)


def kcp_compute_summaries(
    workspace: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    zero_final: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compose FP32 transitions using a prepared FlashKDA workspace.

    ``zero_final`` is the native zero-initial-state scan's [segment, H, V, K]
    final state. Summaries store [S_ext | M] in [segment, H, K, V + K] order.
    Optional destinations must be unique and in bounds; absent slots are
    untouched. The workspace must remain unchanged until the seeded scan.
    """
    segments, heads, dim, _ = zero_final.shape
    assert dim == 128 and zero_final.dtype == torch.float32
    if out is None:
        assert output_indices is None
        out = zero_final.new_empty(segments, heads, 128, 256)
    if not segments:
        return out
    partial = zero_final.new_empty(segments * 4, heads, 128, 128)
    torch.ops._flashkda_C.kcp_transition(
        workspace, beta, cu_seqlens, chunk_offsets, partial
    )
    kcp_merge_local_transitions_kernel[(2, heads, segments)](
        partial,
        zero_final,
        out,
        output_indices,
        heads,
        MAPPED=output_indices is not None,
        num_warps=8,
    )
    return out


@triton.jit(do_not_specialize=["S", "N"])
def kcp_merge_states_kernel(
    Summaries,
    Base,
    Counts,
    Initial,
    Final,
    S,
    N,
    H: tl.constexpr,
    FULL: tl.constexpr,
    RANK_ORDER: tl.constexpr,
):
    values = tl.program_id(0) * 64 + tl.arange(0, 64)
    request_head = tl.program_id(1)
    request, head = request_head // H, request_head % H
    keys = tl.arange(0, 128)
    state = tl.load(
        Base + request_head * 128 * 128 + values[:, None] * 128 + keys[None, :]
    )
    count = tl.load(Counts + request)
    for slot in range(S):
        tl.store(
            Initial
            + ((request * S + slot) * H + head) * 128 * 128
            + values[:, None] * 128
            + keys[None, :],
            state,
        )
        if FULL or slot < count:
            physical = slot
            if RANK_ORDER:
                physical = tl.where(slot < S // 2, 2 * slot, 2 * (S - 1 - slot) + 1)
            start = ((physical * N + request) * H + head) * 128 * 256
            transition = tl.load(
                Summaries + start + keys[:, None] * 256 + 128 + keys[None, :]
            )
            additive = tl.load(
                Summaries + start + keys[:, None] * 256 + values[None, :]
            )
            state = tl.dot(state, tl.trans(transition), input_precision="tf32x3")
            state += tl.trans(additive)
    tl.store(
        Final + request_head * 128 * 128 + values[:, None] * 128 + keys[None, :], state
    )


def kcp_merge_states(
    slots_hm: torch.Tensor,
    base_state: torch.Tensor,
    num_slots: torch.Tensor,
    *,
    all_slots_full: bool = False,
    rank_part_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge summaries in sequence order, preserving FP32 transition precision.

    Return initial states [request, slot, head, V, K] and contiguous final
    states [request, head, V, K]. Slot and request counts remain runtime
    arguments. Explicit tf32x3 avoids the inaccurate BF16 or TF32 state chain.
    """
    slots, requests, heads, dim, width = slots_hm.shape
    assert dim == 128 and width == 256 and base_state.is_contiguous()
    initial = base_state.new_empty(requests, slots, heads, 128, 128)
    final = torch.empty_like(base_state)
    kcp_merge_states_kernel[(2, requests * heads)](
        slots_hm,
        base_state,
        num_slots,
        initial,
        final,
        slots,
        requests,
        heads,
        all_slots_full,
        rank_part_order,
        num_warps=4,
        num_stages=2,
    )
    return initial, final


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
    ``block`` tensors cover global decode tokens, which every rank computes
    redundantly to keep the replicated mamba caches coherent.
    """

    world: int
    rank: int
    num_prefill_reqs: int  # N_pf
    num_slots: int  # 2 * world
    all_slots_full: bool  # Every prefill request has all 2 * world slots.
    global_prefill_tokens: int
    # This rank's scan rows (chunk slots of KCP-prefill requests).
    num_scan_rows: int
    prefill_src_range: tuple[int, int] | None
    prefill_src_idx: torch.Tensor  # [T_loc] int64, local token idx in scan order
    scan_cu_seqlens: torch.Tensor  # [N_loc + 1] int32
    scan_chunk_offsets: torch.Tensor  # [N_loc + 1] int32, FlashKDA 16-token chunks
    conv_meta: object  # SimpleNamespace(nums_dict, batch_ptr, token_chunk_offset_ptr)
    conv_cache_indices: torch.Tensor  # [N_loc] int32 arange (scratch conv states)
    conv_all_initial: torch.Tensor  # [N_loc] bool, all True (halo always given)
    scan_req_idx: torch.Tensor  # [N_loc] int64, KCP-prefill request ordinal
    summary_dst_idx: torch.Tensor  # [N_loc] int64, position in the [N, 2] contribution
    summary_rank_part_idx: torch.Tensor  # [N_loc] int64, position in [2, N]
    init_gather_idx: torch.Tensor  # [N_loc] int64, position in the [N, 2P] inits
    tail_dst_idx: torch.Tensor  # [M] int64, into [N * 2 * 3] tail contribution
    tail_src_idx: torch.Tensor  # [M] int64, scan-order qkv row of each tail row
    halo_tail_idx: torch.Tensor  # [N_loc, 3] int64; -3..-1 address cached prefix
    num_slots_dev: torch.Tensor  # [N] int32, non-empty chunk slots per request
    prefill_entry_idx: torch.Tensor  # [N] int64, index into non-spec metadata rows
    final_tail_idx: torch.Tensor  # [N, 3] int64; -3..-1 address cached prefix
    # Non-KCP (block) rows: gathered and computed redundantly on every rank.
    num_block_tokens: int  # D_g
    block_cap: int  # per-rank padded block-token count used for the all-gather
    block_src_idx: torch.Tensor  # [cnt] int64, this rank's block tokens in [P]
    block_restore_idx: torch.Tensor  # [D_g] int64, into the [world * cap] gather
    block_num_dec_reqs: int
    block_cu_seqlens: torch.Tensor  # [block_num_dec_reqs + 1] int32
    nonprefill_out_src: torch.Tensor  # int64, block rows to copy out
    nonprefill_out_dst: torch.Tensor  # int64, their local token positions

    def select_prefill_tokens(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Select scan-order tokens, using a view for a CPU-certified range."""
        if self.prefill_src_range is not None:
            start, stop = self.prefill_src_range
            return tensor.narrow(dim, start, stop - start)
        return tensor.index_select(dim, self.prefill_src_idx)

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
    if drafts is not None and np.any(drafts > 0):
        raise NotImplementedError(
            "KDA context parallelism does not support speculative decoding."
        )
    # Replicated requests and single-token extends use the redundant path,
    # keeping the recurrent caches coherent across PCP ranks.
    kcp_pref = is_pref & (q_g > 1) & ~mgr.replicated_requests(q_g, is_pref)
    is_block = ~kcp_pref & (q_g > 0)

    prefill_tokens = int(q_g[is_pref].sum())
    if not kcp_pref.any() or np.any(is_block & is_pref) or np.any(q_g[is_block] != 1):
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

    final_tail = np.empty((N_pf, 3), dtype=np.int64)
    for n, g in enumerate(n_list):
        for j in range(3):
            pos = int(q_g[g]) - 3 + j
            if pos < 0:
                final_tail[n, j] = pos
            else:
                slot = pos // int(cs[n])
                column = 3 + pos - slot * int(cs[n]) - int(slot_len[n, slot])
                final_tail[n, j] = (slot * N_pf + n) * 3 + column

    prefill_start = scan_rows[0][2] if scan_rows else 0
    prefill_src_range = (
        (prefill_start, prefill_start + cu[-1])
        if all(
            token_start == prefill_start + cu[i]
            for i, (_, _, token_start, _) in enumerate(scan_rows)
        )
        else None
    )
    scan_cu_cpu = torch.tensor(cu, dtype=torch.int32)
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_offsets,
    )
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        scan_cu_cpu, device=device
    )
    to_gpu_i64 = lambda a: torch.tensor(a, dtype=torch.int64, device=device)
    return KcpPlan(
        world=world,
        rank=rank,
        num_prefill_reqs=N_pf,
        num_slots=S,
        all_slots_full=bool(np.all(num_slots_np == S)),
        global_prefill_tokens=prefill_tokens,
        num_scan_rows=len(scan_rows),
        prefill_src_range=prefill_src_range,
        prefill_src_idx=to_gpu_i64(prefill_src),
        scan_cu_seqlens=scan_cu_cpu.to(device, non_blocking=True),
        scan_chunk_offsets=prepare_chunk_offsets(scan_cu_cpu, 16).to(
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
        summary_rank_part_idx=to_gpu_i64(
            [(0 if c == rank else N_pf) + n for n, c, _, _ in scan_rows]
        ),
        init_gather_idx=to_gpu_i64([n * S + c for n, c, _, _ in scan_rows]),
        tail_dst_idx=to_gpu_i64(tail_dst),
        tail_src_idx=to_gpu_i64(tail_src),
        halo_tail_idx=to_gpu_i64(halo_tail),
        num_slots_dev=torch.tensor(num_slots_np, dtype=torch.int32, device=device),
        prefill_entry_idx=to_gpu_i64(n_list),
        final_tail_idx=to_gpu_i64(final_tail),
        num_block_tokens=D_g,
        block_cap=cap,
        block_src_idx=to_gpu_i64(block_src),
        block_restore_idx=to_gpu_i64(block_restore),
        block_num_dec_reqs=D_g,
        block_cu_seqlens=torch.arange(D_g + 1, dtype=torch.int32, device=device),
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


def gather_slot_summaries(
    plan: KcpPlan, hm_local: torch.Tensor, *, rank_part_order: bool = False
) -> torch.Tensor:
    """All-gather per-slot [S_ext, M] summaries into global slot order.

    Returns [2P, N, H, K, V + K] fp32, with zeros in slots no rank owns.
    With rank_part_order, input is already packed [2 * N, ...] and the result
    retains rank/part order for a matching merge consumer, without reordering.
    """
    N = plan.num_prefill_reqs
    if rank_part_order:
        # The producer has already populated [part, request] contributions.
        assert hm_local.shape[0] == 2 * N
        ag = get_pcp_group().all_gather(hm_local, dim=0)
        return ag.view(plan.num_slots, N, *hm_local.shape[1:])
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
