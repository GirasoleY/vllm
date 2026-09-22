# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# The affine summary and merge equations follow the FLA context-parallel
# implementation (fla/ops/cp/chunk_delta_h.py, MIT license), specialized to the
# KDA gate convention (per-dim log2 gate cumsum `gk`, pre-gated `kg`, no scalar
# gate, no DPLR) and to serving-side zig-zag chunk sharding.
"""KCP (two-pass parallel scan) for KDA layers under MRV2 PCP on Blackwell.

MRV2 PCP zig-zag-shards prefills into 2P chunks (rank r holds chunks r and
2P-1-r). Replicated short prefills share one logical slot. KCP runs a two-pass
parallel scan per step:

1. Each rank computes, per chunk it owns, the chunk's affine state transition
   S_end = M @ S_in + S_ext (zero initial state) with the native S and M kernels.
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
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pcp_group
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.gpu.pcp_manager import PCPManager


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


# ---------------------------------------------------------------------------
# Per-step KCP layout plan
# ---------------------------------------------------------------------------


@dataclass
class KcpPlan:
    """Per-step layout for the KCP path (built once, shared by all KDA layers).

    All index tensors are GPU-resident; the rest is CPU metadata. ``scan``
    tensors order this rank's local prefill chunk rows by (request, slot);
    Decode tensors cover ordinary decode tokens, computed on every rank
    to keep the replicated Mamba caches coherent.
    """

    world: int
    rank: int
    halo_size: int
    num_prefill_reqs: int  # N_pf
    num_slots: int  # 2 * world
    all_slots_full: bool  # Every prefill request has all 2 * world slots.
    global_prefill_tokens: int
    # This rank's scan rows (chunk slots of KCP-prefill requests).
    num_scan_rows: int
    prefill_src_range: tuple[int, int] | None
    prefill_src_idx: torch.Tensor  # Empty for a view; otherwise scan token indices.
    scan_cu_seqlens: torch.Tensor  # [N_loc + 1] int32
    scan_chunk_indices: torch.Tensor  # [NT, 2] int32, FLA chunking of the scan
    scan_chunk_offsets: torch.Tensor  # [N_loc + 1] int32
    conv_meta: object  # SimpleNamespace(nums_dict, batch_ptr, token_chunk_offset_ptr)
    scan_req_idx: torch.Tensor  # [N_loc] int64, KCP-prefill request ordinal
    summary_rank_part_idx: torch.Tensor  # [N_loc] int64, position in [2, N]
    init_gather_idx: torch.Tensor  # [N_loc] int64, position in the [N, 2P] inits
    tail_src_idx: torch.Tensor  # [N*2*halo_size], scan-token index or -1 if absent
    # Tail selectors address [rank, request, part, token]; negatives select cache.
    halo_tail_idx: torch.Tensor  # [N_loc, halo_size] int64
    num_slots_dev: torch.Tensor  # [N] int32, non-empty chunk slots per request
    prefill_entry_idx: torch.Tensor  # [N] int64, index into non-spec metadata rows
    prefill_has_initial_state: torch.Tensor  # [N] bool, from global prefix lengths
    final_tail_idx: (
        torch.Tensor
    )  # [N, halo_size] int64; negatives address cached prefix
    # Ordinary decode requests occupy the local token prefix on every rank.
    num_decode_reqs: int
    decode_entry_idx: torch.Tensor  # Global request indices in state metadata.
    decode_cu_seqlens: torch.Tensor  # One token per ordinary decode request.

    def select_prefill_tokens(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Select scan-order tokens, using a view for a CPU-certified range."""
        if self.prefill_src_range is not None:
            start, stop = self.prefill_src_range
            return tensor.narrow(dim, start, stop - start)
        return tensor.index_select(dim, self.prefill_src_idx)

    @property
    def num_prefill_tokens(self) -> int:
        if self.prefill_src_range is not None:
            start, stop = self.prefill_src_range
            return stop - start
        return self.prefill_src_idx.numel()

    @property
    def has_decode(self) -> bool:
        return self.num_decode_reqs > 0


def build_kcp_plan(
    mgr: "PCPManager", device: torch.device, halo_size: int = 3
) -> KcpPlan | None:
    """Build a layout for every real PCP batch from CPU partition metadata.

    Replicated prefills have one logical slot, whose rank-zero summary is
    merged. Every rank scans its local copy from that slot's initial state.
    Decode tokens use a compact block independent of global request order.
    """
    assert halo_size > 0
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
    drafts = gb.num_draft_tokens_per_req
    if drafts is not None and np.any(drafts > 0):
        raise NotImplementedError("GLM KCP does not support speculative decoding yet.")
    replicated = mgr.replicated_requests(q_g, is_pref)
    kcp_pref = is_pref & (q_g > 0)
    is_block = ~is_pref & (q_g > 0)
    prefill_tokens = int(q_g[kcp_pref].sum())

    n_list = np.nonzero(kcp_pref)[0]
    N_pf = len(n_list)
    S = 2 * world
    cs = np.where(replicated[n_list], q_g[n_list], (q_g[n_list] + S - 1) // S)
    slot_len = np.clip(
        q_g[n_list][:, None] - np.arange(S)[None, :] * cs[:, None], 0, None
    )
    slot_len = np.minimum(slot_len, cs[:, None])
    num_slots_np = (slot_len > 0).sum(axis=1).astype(np.int32)

    # This rank's scan rows and block rows from the local batch (ground truth
    # for local token positions).
    cs_by_g = {int(g): int(cs[n]) for n, g in enumerate(n_list)}
    scan_rows: list[tuple[int, int, int, int]] = []  # (n, slot, tok_start, len)
    block_src: list[int] = []
    for segment in mgr.local_segments:
        g = segment.global_batch_req_idx
        t0 = segment.rank_local_batch_slice.start
        qlen = segment.num_tokens
        offset = segment.global_batch_slice.start - int(qsl[g])
        if kcp_pref[g]:
            c = offset // cs_by_g[g]
            n = int(np.searchsorted(n_list, g))
            # The partition gives this rank exactly slots (rank, 2P-1-rank) of
            # each prefilling request; anything else is a partition bug and
            # must fail loudly (a per-rank fallback would deadlock KCP).
            owned_slots = (0,) if replicated[g] else (rank, 2 * world - 1 - rank)
            assert c < num_slots_np[n] and c in owned_slots, (
                f"KCP plan: local row for request {g} maps to slot {c}, "
                f"expected one of {owned_slots}"
            )
            assert offset == c * cs_by_g[g] and qlen == slot_len[n, c]
            scan_rows.append((n, int(c), t0, qlen))
        else:
            assert qlen == 1 and offset == 0
            block_src.append(t0)
    scan_rows.sort(key=lambda row: (row[0], row[1]))

    decode_reqs = np.nonzero(is_block)[0]
    assert np.all(q_g[decode_reqs] == 1)
    assert block_src == list(range(len(decode_reqs)))

    cu = [0]
    scan_req: list[int] = []
    tail_src = np.full(N_pf * 2 * halo_size, -1, dtype=np.int64)
    halo_tail = np.full((len(scan_rows), halo_size), -1, dtype=np.int64)
    for row_i, (n, c, t0, qlen) in enumerate(scan_rows):
        cu.append(cu[-1] + qlen)
        scan_req.append(n)
        # This row's contribution to the tails all-gather: its last
        # min(halo_size, qlen) raw qkv rows, right-aligned in the halo tail columns.
        take = min(halo_size, qlen)
        for j in range(take):
            front = replicated[n_list[n]] or c == rank
            destination = (
                (n * 2 + (0 if front else 1)) * halo_size + halo_size - take + j
            )
            tail_src[destination] = cu[row_i] + qlen - take + j
        # Assemble preceding tokens from earlier slots and the cached prefix.
        window: list[tuple[int, int]] = [(-1, j) for j in range(halo_size)]
        for c_ in range(c):
            take_ = min(halo_size, int(slot_len[n, c_]))
            cols = [(c_, halo_size - take_ + j) for j in range(take_)]
            window = (window + cols)[-halo_size:]
        for j, (w_slot, w_col) in enumerate(window):
            if w_slot >= 0:
                halo_tail[row_i, j] = (w_slot * N_pf + n) * halo_size + w_col
            else:
                halo_tail[row_i, j] = w_col - halo_size

    final_tail = np.empty((N_pf, halo_size), dtype=np.int64)
    for n, g in enumerate(n_list):
        for j in range(halo_size):
            pos = int(q_g[g]) - halo_size + j
            if pos < 0:
                final_tail[n, j] = pos
            else:
                slot = pos // int(cs[n])
                column = halo_size + pos - slot * int(cs[n]) - int(slot_len[n, slot])
                final_tail[n, j] = (slot * N_pf + n) * halo_size + column

    def rank_part_tail_indices(indices):
        slot = indices // (N_pf * halo_size)
        request = (indices // halo_size) % N_pf
        column = indices % halo_size
        owner = np.where(slot < world, slot, 2 * world - 1 - slot)
        part = slot >= world
        packed = ((owner * N_pf + request) * 2 + part) * halo_size + column
        return np.where(indices >= 0, packed, indices)

    prefill_start = scan_rows[0][2] if scan_rows else 0
    prefill_src_range = (
        (prefill_start, prefill_start + cu[-1])
        if all(
            token_start == prefill_start + cu[i]
            for i, (_, _, token_start, _) in enumerate(scan_rows)
        )
        else None
    )
    prefill_src = (
        [
            token
            for _, _, start, length in scan_rows
            for token in range(start, start + length)
        ]
        if prefill_src_range is None
        else []
    )
    scan_cu_cpu = torch.tensor(cu, dtype=torch.int32)
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        scan_cu_cpu, device=torch.device("cpu")
    )

    def upload(dtype, **arrays):
        tensors = [torch.as_tensor(value, dtype=dtype) for value in arrays.values()]
        packed = torch.cat([tensor.flatten() for tensor in tensors]).to(
            device, non_blocking=True
        )
        views = packed.split([tensor.numel() for tensor in tensors])
        return {
            name: view.view(tensor.shape)
            for name, tensor, view in zip(arrays, tensors, views)
        }

    indices = upload(
        torch.int64,
        prefill_src_idx=prefill_src,
        scan_req_idx=scan_req,
        summary_rank_part_idx=[
            (0 if replicated[n_list[n]] or c == rank else N_pf) + n
            for n, c, _, _ in scan_rows
        ],
        init_gather_idx=[n * S + c for n, c, _, _ in scan_rows],
        tail_src_idx=tail_src,
        halo_tail_idx=rank_part_tail_indices(halo_tail),
        prefill_entry_idx=n_list,
        final_tail_idx=rank_part_tail_indices(final_tail),
        decode_entry_idx=decode_reqs,
    )
    lengths = upload(
        torch.int32,
        scan_cu_seqlens=scan_cu_cpu,
        scan_chunk_indices=(
            prepare_chunk_indices(scan_cu_cpu, FLA_CHUNK_SIZE)
            if scan_rows
            else torch.empty((0, 2), dtype=torch.int32)
        ),
        scan_chunk_offsets=prepare_chunk_offsets(scan_cu_cpu, FLA_CHUNK_SIZE),
        num_slots_dev=num_slots_np,
        decode_cu_seqlens=torch.arange(len(decode_reqs) + 1),
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    batch_ptr = lengths.pop("batch_ptr")
    token_chunk_offset_ptr = lengths.pop("token_chunk_offset_ptr")
    for metadata in nums_dict.values():
        metadata["batch_ptr"] = batch_ptr
        metadata["token_chunk_offset_ptr"] = token_chunk_offset_ptr
    return KcpPlan(
        world=world,
        rank=rank,
        halo_size=halo_size,
        num_prefill_reqs=N_pf,
        num_slots=S,
        all_slots_full=bool(np.all(num_slots_np == S)),
        global_prefill_tokens=prefill_tokens,
        num_scan_rows=len(scan_rows),
        prefill_src_range=prefill_src_range,
        conv_meta=SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        ),
        prefill_has_initial_state=torch.tensor(start0[n_list] > 0, device=device),
        num_decode_reqs=len(decode_reqs),
        **indices,
        **lengths,
    )


# ---------------------------------------------------------------------------
# Layer-time collectives
# ---------------------------------------------------------------------------


def gather_slot_summaries(plan: KcpPlan, hm_local: torch.Tensor) -> torch.Tensor:
    """Gather producer-owned [part, request] summaries without reordering."""
    N = plan.num_prefill_reqs
    assert hm_local.shape[0] == 2 * N
    ag = get_pcp_group().all_gather(hm_local, dim=0)
    return ag.view(plan.num_slots, N, *hm_local.shape[1:])


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


def gather_conv_tail_rows(plan: KcpPlan, tail_rows: torch.Tensor) -> torch.Tensor:
    """Gather QKV tails in physical [rank, request, part, token, channel] order."""
    N, C = plan.num_prefill_reqs, tail_rows.shape[-1]
    token_count = N * 2 * plan.halo_size
    contrib = tail_rows.new_empty(token_count, C)
    _pack_conv_tail_rows[(token_count * triton.cdiv(C, 256),)](
        tail_rows,
        plan.tail_src_idx,
        contrib,
        tail_rows.stride(0),
        tail_rows.stride(1),
        C,
        BLOCK=256,
    )
    ag = get_pcp_group().all_gather(contrib, dim=0)
    return ag.view(plan.world, N, 2, plan.halo_size, C)


def maybe_get_kcp_plan(context: "ForwardContext", halo_size: int) -> KcpPlan | None:
    """Share the model-specific plan for the lifetime of this forward pass."""
    mgr = context.additional_kwargs.get("pcp_manager")
    if mgr is None or mgr.global_batch is None:
        return None
    plans = context.additional_kwargs.setdefault("glm_kcp_plans", {})
    if halo_size not in plans:
        # Ordinary decode is already replicated in global request order.
        plans[halo_size] = (
            build_kcp_plan(mgr, mgr.device, halo_size)
            if np.any(mgr.global_batch.is_prefilling_np)
            else None
        )
    return plans[halo_size]


def prepare_kcp_states(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    *,
    plan: KcpPlan,
    recurrent_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seed the shared scan from exchanged WY summaries.

    Return segment initial states and request final states, both value-first.
    Empty ranks still participate in the summary exchange and state merge.
    Cache publication stays with the caller of the shared chunked forward.
    """
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
        gather_initial_states,
    )
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

    N, H, K, V = plan.num_prefill_reqs, kg.shape[-2], kg.shape[-1], u.shape[-1]
    hm = kg.new_empty(2 * N, H, K, V + K, dtype=torch.float32)
    if plan.num_scan_rows != 2 * N:
        hm.zero_()
    if kg.shape[1]:
        kcp_compute_summaries(
            kg=kg,
            u=u,
            w=w,
            gk=g,
            cu_seqlens=plan.scan_cu_seqlens,
            chunk_size=FLA_CHUNK_SIZE,
            out=hm,
            output_indices=plan.summary_rank_part_idx,
        )
    slots_hm = gather_slot_summaries(plan, hm)
    base = gather_initial_states(
        recurrent_state, state_indices, plan.prefill_has_initial_state
    )
    inits, final = kcp_merge_states(
        slots_hm,
        base.float(),
        plan.num_slots_dev,
        all_slots_full=plan.all_slots_full,
    )
    initial = inits.view(N * plan.num_slots, H, V, K)[plan.init_gather_idx]
    return initial, final
