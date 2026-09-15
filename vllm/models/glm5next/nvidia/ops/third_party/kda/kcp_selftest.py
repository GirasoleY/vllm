# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerics self-test for the KCP two-pass parallel scan (GPU host only).

Splits a synthetic varlen KDA prefill into zig-zag chunk slots, runs the
summary -> merge -> seeded-scan path, and compares against the single-pass
chunked scan (the gather-replicate reference) for both the per-token outputs
and the sequence-final states.

Run on a GPU host from the vLLM source tree (module mode; path mode breaks
the relative imports):

    python -m vllm.models.glm5next.nvidia.ops.third_party.kda.kcp_selftest
    python -m vllm.models.glm5next.nvidia.ops.third_party.kda.kcp_selftest --debug
"""

import sys

import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
    kcp_compute_summaries,
    kcp_merge_states,
    kcp_zigzag_slot_order,
)
from vllm.models.glm5next.nvidia.ops.third_party.kda.kernels import (
    chunk_gla_fwd_o_gk,
    chunk_kda_scaled_dot_kkt_fwd,
    chunk_kda_with_fused_gate,
    fused_kda_gate_chunk_cumsum,
    recompute_w_u_fwd,
)
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE

SAFE_GATE = True
LOWER_BOUND = -5.0


def _wy_pipeline(q, k, v, beta, raw_g, cu_seqlens, chunk_indices, a_log, g_bias):
    """The scan's shared pre-processing: gate cumsum + KKT + WY tensors."""
    g = fused_kda_gate_chunk_cumsum(
        raw_g,
        A_log=a_log,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        safe_gate=SAFE_GATE,
        lower_bound=LOWER_BOUND,
    )
    scale = q.shape[-1] ** -0.5
    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q=q,
        k=k,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        gk=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return g, Aqk, w, u, kg, scale


def _scan(q, g, Aqk, w, u, kg, scale, init_state, cu_seqlens, chunk_indices):
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=init_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    o = q.new_empty(*q.shape[:3], u.shape[-1])
    return chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        o=o,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def _rel(err: torch.Tensor, ref: torch.Tensor) -> float:
    return (err.abs().max() / ref.abs().max().clamp(min=1e-6)).item()


def check_conv_chain(T: int, world: int, device: str = "cuda") -> None:
    """Verify conv-state chaining: a chunk's final window is the next chunk's
    initial window (the KCP halo mechanism relies on it)."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    C, width = 96, 4
    S = 2 * world
    torch.manual_seed(S)
    # causal_conv1d_fn wants channel-last x (x.stride(0) == 1).
    x = torch.randn(T, C, device=device, dtype=torch.bfloat16).transpose(0, 1)
    w = torch.randn(C, width, device=device, dtype=torch.float32) * 0.1

    def _meta(hi: int) -> SimpleNamespace:
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            torch.tensor([0, hi], dtype=torch.int32), device=device
        )
        return SimpleNamespace(
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )

    cu_full = torch.tensor([0, T], dtype=torch.int32, device=device)
    # conv cache index 0 is the null block (the kernel skips it), so the
    # scratch states here start at row 1.
    scratch_ref = x.new_zeros(2, C, width - 1)
    out_ref = causal_conv1d_fn(
        x,
        w,
        None,
        conv_states=scratch_ref,
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
        cache_indices=torch.ones(1, dtype=torch.int32, device=device),
        query_start_loc=cu_full,
        activation="silu",
        metadata=_meta(T),
    )
    # Split into slot ranges and chain via the writeback windows.
    cs = (T + S - 1) // S
    prev = x.new_zeros(2, C, width - 1)
    parts = []
    for c in range(S):
        lo, hi = c * cs, min((c + 1) * cs, T)
        if lo >= hi:
            break
        out_c = causal_conv1d_fn(
            x[:, lo:hi],
            w,
            None,
            conv_states=prev,
            has_initial_state=torch.ones(1, dtype=torch.bool, device=device),
            cache_indices=torch.ones(1, dtype=torch.int32, device=device),
            query_start_loc=torch.tensor(
                [0, hi - lo], dtype=torch.int32, device=device
            ),
            activation="silu",
            metadata=_meta(hi - lo),
        )
        parts.append(out_c)
    out_chain = torch.cat(parts, dim=1)
    err = (out_chain[:, :T].float() - out_ref[:, :T].float()).abs().max().item()
    print(f"conv chain T={T} world={world}: max|dOut|={err:.4f}")
    assert err < 1e-3, f"conv chain mismatch {err}"


def run_case(
    T: int, N: int, world: int, seed: int, gate_shift: float = 0.0, debug: bool = False
) -> bool:
    device = "cuda"
    torch.manual_seed(seed)
    H, K, V = 8, 128, 128
    dtype = torch.bfloat16
    a_log = torch.randn(1, 1, H, 1, dtype=torch.float32, device=device) * 0.5
    g_bias = torch.zeros(H * K, dtype=torch.float32, device=device)

    lens = torch.randint(max(2, T // (2 * N)), max(3, T // N + 2), (N,))
    lens = (lens * (T / lens.sum().item())).to(torch.int64).clamp(min=1)
    lens[-1] += T - int(lens.sum())
    cu = torch.zeros(N + 1, dtype=torch.int32)
    cu[1:] = lens.cumsum(0)
    cu_dev = cu.to(device)

    q = torch.randn(1, T, H, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)
    # gate_shift < 0 weakens the gate so the M-chain actually carries state.
    raw_g = gate_shift + torch.randn(1, T, H, K, dtype=dtype, device=device) * 0.1
    beta = torch.rand(1, T, H, dtype=dtype, device=device)

    # Reference: the single-pass chunked scan over the whole batch. NOTE:
    # chunk_kda_with_fused_gate writes the output into its (contiguous) v
    # input (`o=v` reuse), so hand it a clone to keep v pristine.
    o_ref, final_ref = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=raw_g,
        beta=beta,
        A_log=a_log,
        g_bias=g_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_dev,
        safe_gate=SAFE_GATE,
        lower_bound=LOWER_BOUND,
    )

    ok = True
    S = 2 * world
    for n in range(N):
        bos, eos = int(cu[n]), int(cu[n + 1])
        q_len = eos - bos
        cs = (q_len + S - 1) // S
        slot_len = [max(0, min(cs, q_len - c * cs)) for c in range(S)]
        L = sum(1 for length in slot_len if length > 0)
        slot_cu = [0]
        for c in range(S):
            slot_cu.append(slot_cu[-1] + slot_len[c])

        # Per-slot: WY pre-processing (the layer l2norms q/k first), then the
        # zero-init affine summary.
        hm = q.new_zeros(S, 1, H, K, V + K, dtype=torch.float32)
        wy = []
        for c in range(L):
            lo, hi = bos + slot_cu[c], bos + slot_cu[c + 1]
            cu_c = torch.tensor([0, hi - lo], dtype=torch.int32)
            ci_c = prepare_chunk_indices(cu_c, FLA_CHUNK_SIZE).to(device)
            q_c = l2norm_fwd(q[:, lo:hi].contiguous())
            k_c = l2norm_fwd(k[:, lo:hi].contiguous())
            g_c, Aqk_c, w_c, u_c, kg_c, scale = _wy_pipeline(
                q_c,
                k_c,
                v[:, lo:hi].contiguous(),
                beta[:, lo:hi].contiguous(),
                raw_g[:, lo:hi].contiguous(),
                cu_c.to(device),
                ci_c,
                a_log,
                g_bias,
            )
            wy.append((q_c, g_c, Aqk_c, w_c, u_c, kg_c, scale, cu_c, ci_c))
            hm[c, 0] = kcp_compute_summaries(
                kg=kg_c,
                u=u_c,
                w=w_c,
                gk=g_c,
                cu_seqlens=cu_c.to(device),
                chunk_size=FLA_CHUNK_SIZE,
            )[0]

        base = q.new_zeros(1, H, V, K, dtype=torch.float32)
        num_slots = torch.tensor([L], dtype=torch.int32, device=device)
        inits, final = kcp_merge_states(hm, base, num_slots)

        o_parts = []
        for c in range(L):
            q_c, g_c, Aqk_c, w_c, u_c, kg_c, scale, cu_c, ci_c = wy[c]
            o_parts.append(
                _scan(
                    q_c,
                    g_c,
                    Aqk_c,
                    w_c,
                    u_c,
                    kg_c,
                    scale,
                    inits[0, c : c + 1],
                    cu_c.to(device),
                    ci_c,
                )
            )
        o_kcp = torch.cat(o_parts, dim=1)
        err_o = (o_kcp.float() - o_ref[:, bos:eos].float()).abs().max().item()
        d_state = (final[0].float() - final_ref[n].float()).abs()
        err_s = d_state.max().item()
        rel_s = _rel(final[0].float() - final_ref[n].float(), final_ref[n].float())
        print(
            f"T={T} N={N} world={world} req={n} len={q_len} L={L} "
            f"gate_shift={gate_shift}: max|dOut|={err_o:.4f} "
            f"max|dState|={err_s:.4f} "
            f"(rel={rel_s:.2e}, max|state|={final_ref[n].abs().max().item():.3f})"
        )
        if err_s >= 0.5 or err_o >= 0.5:
            ok = False
        if debug and not ok:
            _debug_slots(
                q,
                k,
                v,
                raw_g,
                beta,
                a_log,
                g_bias,
                bos,
                slot_cu,
                slot_len,
                L,
                H,
                K,
                V,
                wy,
                hm,
                inits,
                final,
                final_ref[n],
            )
            return False
    return ok


def run_case_v2(
    T: int,
    N: int,
    world: int,
    seed: int,
    gate_shift: float = 0.0,
    split: bool = False,
) -> bool:
    """Combined multi-request KCP step, optionally chained (continuation).

    Unlike run_case (per-request, zero base), this routes every request's slot
    summaries through the zig-zag gather layout, runs one N-wide merge, and —
    with split=True — scans each request in two chained KCP steps, the second
    seeded with the first step's final state (the serving continuation path).
    """
    device = "cuda"
    torch.manual_seed(seed)
    H, K, V = 8, 128, 128
    dtype = torch.bfloat16
    a_log = torch.randn(1, 1, H, 1, dtype=torch.float32, device=device) * 0.5
    g_bias = torch.zeros(H * K, dtype=torch.float32, device=device)

    lens = torch.randint(max(2, T // (2 * N)), max(3, T // N + 2), (N,))
    lens = (lens * (T / lens.sum().item())).to(torch.int64).clamp(min=1)
    lens[-1] += T - int(lens.sum())
    cu = torch.zeros(N + 1, dtype=torch.int32)
    cu[1:] = lens.cumsum(0)
    cu_dev = cu.to(device)

    q = torch.randn(1, T, H, K, dtype=dtype, device=device)
    k = torch.randn(1, T, H, K, dtype=dtype, device=device)
    v = torch.randn(1, T, H, V, dtype=dtype, device=device)
    raw_g = gate_shift + torch.randn(1, T, H, K, dtype=dtype, device=device) * 0.1
    beta = torch.rand(1, T, H, dtype=dtype, device=device)

    o_ref, final_ref = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=raw_g,
        beta=beta,
        A_log=a_log,
        g_bias=g_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_dev,
        safe_gate=SAFE_GATE,
        lower_bound=LOWER_BOUND,
    )

    S = 2 * world
    phases = []  # per request: list of (lo, hi) absolute token ranges
    for n in range(N):
        bos, eos = int(cu[n]), int(cu[n + 1])
        if split and eos - bos > 1:
            mid = bos + max(1, (eos - bos) * 2 // 5)
            phases.append([(bos, mid), (mid, eos)])
        else:
            phases.append([(bos, eos)])

    bases = q.new_zeros(N, H, V, K, dtype=torch.float32)
    ok = True
    for phase in range(len(phases[0])):
        # Per-slot zero-init summaries routed through the zig-zag gather
        # layout (rank r contributes slot r at part 0, slot S-1-r at part 1).
        contrib = q.new_zeros(world, N, 2, H, K, V + K, dtype=torch.float32)
        wy = {}
        num_slots = [0] * N
        for n in range(N):
            lo_n, hi_n = phases[n][phase]
            q_len = hi_n - lo_n
            cs = (q_len + S - 1) // S
            slot_len = [max(0, min(cs, q_len - c * cs)) for c in range(S)]
            num_slots[n] = sum(1 for length in slot_len if length > 0)
            for c in range(num_slots[n]):
                lo, hi = lo_n + c * cs, lo_n + c * cs + slot_len[c]
                cu_c = torch.tensor([0, hi - lo], dtype=torch.int32)
                ci_c = prepare_chunk_indices(cu_c, FLA_CHUNK_SIZE).to(device)
                q_c = l2norm_fwd(q[:, lo:hi].contiguous())
                k_c = l2norm_fwd(k[:, lo:hi].contiguous())
                g_c, Aqk_c, w_c, u_c, kg_c, scale = _wy_pipeline(
                    q_c,
                    k_c,
                    v[:, lo:hi].contiguous(),
                    beta[:, lo:hi].contiguous(),
                    raw_g[:, lo:hi].contiguous(),
                    cu_c.to(device),
                    ci_c,
                    a_log,
                    g_bias,
                )
                wy[(n, c)] = (q_c, g_c, Aqk_c, w_c, u_c, kg_c, scale, cu_c, ci_c)
                contrib[min(c, S - 1 - c), n, 0 if c < world else 1] = (
                    kcp_compute_summaries(
                        kg=kg_c,
                        u=u_c,
                        w=w_c,
                        gk=g_c,
                        cu_seqlens=cu_c.to(device),
                        chunk_size=FLA_CHUNK_SIZE,
                    )[0]
                )

        hm = kcp_zigzag_slot_order(contrib)
        inits, final = kcp_merge_states(
            hm, bases, torch.tensor(num_slots, dtype=torch.int32, device=device)
        )
        for n in range(N):
            lo_n, hi_n = phases[n][phase]
            o_parts = []
            for c in range(num_slots[n]):
                q_c, g_c, Aqk_c, w_c, u_c, kg_c, scale, cu_c, ci_c = wy[(n, c)]
                o_parts.append(
                    _scan(
                        q_c,
                        g_c,
                        Aqk_c,
                        w_c,
                        u_c,
                        kg_c,
                        scale,
                        inits[n, c : c + 1],
                        cu_c.to(device),
                        ci_c,
                    )
                )
            o_kcp = torch.cat(o_parts, dim=1)
            err_o = (o_kcp.float() - o_ref[:, lo_n:hi_n].float()).abs().max()
            err_o = err_o.item()
            bases[n] = final[n]
            if phase == len(phases[0]) - 1:
                err_s = (final[n].float() - final_ref[n].float()).abs().max()
                rel_s = _rel(final[n].float() - final_ref[n], final_ref[n])
                print(
                    f"v2 T={T} N={N} world={world} req={n} phase={phase} "
                    f"split={split} gate_shift={gate_shift}: "
                    f"max|dOut|={err_o:.4f} max|dState|={err_s.item():.4f} "
                    f"(rel={rel_s:.2e})"
                )
                if err_s.item() >= 0.5 or err_o >= 0.5:
                    ok = False
            else:
                print(
                    f"v2 T={T} N={N} world={world} req={n} phase={phase} "
                    f"split={split} gate_shift={gate_shift}: max|dOut|={err_o:.4f}"
                )
            if err_o >= 0.5:
                ok = False
    return ok


def _mag(x: torch.Tensor) -> float:
    return x.float().abs().max().item()


def _debug_slots(
    q,
    k,
    v,
    raw_g,
    beta,
    a_log,
    g_bias,
    bos,
    slot_cu,
    slot_len,
    L,
    H,
    K,
    V,
    wy,
    hm,
    inits,
    final,
    final_ref,
):
    """Decompose the failure: summary kernel vs merge vs seeded scan."""
    device = q.device
    for c in range(L):
        lo, hi = bos + slot_cu[c], bos + slot_cu[c + 1]
        q_c, _, _, _, _, _, _, cu_c, ci_c = wy[c]
        # (a) Summary S_ext vs the standalone chunked scan's final state.
        _, fin_c0 = chunk_kda_with_fused_gate(
            q=q[:, lo:hi].contiguous(),
            k=k[:, lo:hi].contiguous(),
            v=v[:, lo:hi].contiguous(),
            raw_g=raw_g[:, lo:hi].contiguous(),
            beta=beta[:, lo:hi].contiguous(),
            A_log=a_log,
            g_bias=g_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_c.to(device),
            safe_gate=SAFE_GATE,
            lower_bound=LOWER_BOUND,
        )
        s_ext = hm[c, 0, :, :K, :V]  # [H, K, V]
        err = (s_ext.transpose(-1, -2) - fin_c0[0].float()).abs()
        print(
            f"  slot {c}: summary-vs-scanfinal max|d|={err.max().item():.4f}"
            f" |S_ext|={_mag(s_ext):.4f} |scanfinal|={_mag(fin_c0[0]):.4f}"
        )
        if c > 0:
            # The merge kernel must produce init[c] = M @ init[c-1] + S_ext
            # verbatim; check the raw stored pieces against hm directly.
            prev = hm[c - 1, 0]  # [H, K, V+K]
            s_prev, m_prev = prev[:, :K, :V], prev[:, :, V:]
            want = torch.einsum(
                "hij,hvj->hvi", m_prev, inits[0, c - 1].float()
            ) + s_prev.transpose(-1, -2)
            err = (inits[0, c].float() - want).abs()
            print(
                f"  slot {c}: merge-recurrence-vs-hm max|d|={err.max().item():.4f}"
                f" |M|={_mag(m_prev):.4f}"
            )
        # (b) Merged init vs the true state at the slot start (prefix scan).
        _, fin_pre = chunk_kda_with_fused_gate(
            q=q[:, bos:lo].contiguous(),
            k=k[:, bos:lo].contiguous(),
            v=v[:, bos:lo].contiguous(),
            raw_g=raw_g[:, bos:lo].contiguous(),
            beta=beta[:, bos:lo].contiguous(),
            A_log=a_log,
            g_bias=g_bias,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=torch.tensor([0, lo - bos], dtype=torch.int32).to(device),
            safe_gate=SAFE_GATE,
            lower_bound=LOWER_BOUND,
        )
        err = (inits[0, c].float() - fin_pre[0].float()).abs()
        print(
            f"  slot {c}: merged-init-vs-prefix max|d|={err.max().item():.4f}"
            f" |init|={_mag(inits[0, c]):.4f} |prefix|={_mag(fin_pre[0]):.4f}"
        )
    err = (final[0].float() - final_ref.float()).abs()
    print(
        f"  final: max|d|={err.max().item():.4f} |final|={_mag(final[0]):.4f} "
        f"|ref|={_mag(final_ref):.4f}"
    )


if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a GPU host"
    debug = "--debug" in sys.argv
    # Conv-state window chaining (the KCP halo mechanism's foundation).
    for world in (2, 4, 8):
        check_conv_chain(T=4096, world=world)
        check_conv_chain(T=4233, world=world)
    all_ok = True
    for world in (2, 4, 8):
        # Strong gates (default regime) and weak gates (the M chain matters).
        all_ok &= run_case(T=4096, N=1, world=world, seed=world, debug=debug)
        all_ok &= run_case(
            T=4096, N=1, world=world, seed=world + 200, gate_shift=-4.0, debug=debug
        )
        all_ok &= run_case(
            T=4096 + 137, N=3, world=world, seed=world + 100, debug=debug
        )
        all_ok &= run_case(
            T=4096 + 137,
            N=3,
            world=world,
            seed=world + 300,
            gate_shift=-4.0,
            debug=debug,
        )
        # Combined multi-request merge/gather layout and the chained
        # continuation (nonzero base) path.
        all_ok &= run_case_v2(T=4233, N=3, world=world, seed=world + 400)
        all_ok &= run_case_v2(
            T=4233, N=3, world=world, seed=world + 500, gate_shift=-4.0
        )
        all_ok &= run_case_v2(T=4233, N=3, world=world, seed=world + 600, split=True)
        all_ok &= run_case_v2(
            T=4233,
            N=3,
            world=world,
            seed=world + 700,
            gate_shift=-4.0,
            split=True,
        )
    if not all_ok:
        sys.exit(1)
    print("KCP selftest OK")
