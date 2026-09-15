# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: validate the KCP summary kernel's transition matrix M.

For one synthetic slot, extract the scan's empirical affine transition
(S_end = M @ S_in + S_ext) by running the chunked scan with a zero and an
identity initial state, and compare against the summary kernel's [S_ext, M].

    python -m vllm.models.glm5next.nvidia.ops.third_party.kda.kcp_probe
"""

import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
    kcp_compute_summaries,
)
from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp_selftest import (
    LOWER_BOUND,
    SAFE_GATE,
    _wy_pipeline,
)
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE


def _scan_final(kg, w, u, gk, init, cu, ci):
    _, _, fin = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=gk,
        initial_state=init,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=ci,
        use_exp2=True,
    )
    return fin


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU host"
    torch.manual_seed(0)
    device = "cuda"
    H, K, V = 4, 128, 128
    dtype = torch.bfloat16
    a_log = torch.randn(1, 1, H, 1, dtype=torch.float32, device=device) * 0.5
    g_bias = torch.zeros(H * K, dtype=torch.float32, device=device)

    for T in (64, 128, 1024):
        for gate_shift, regime in ((0.0, "strong-decay"), (-4.0, "weak-decay")):
            q = torch.randn(1, T, H, K, dtype=dtype, device=device)
            k = torch.randn(1, T, H, K, dtype=dtype, device=device)
            v = torch.randn(1, T, H, V, dtype=dtype, device=device)
            # Shift raw_g negative to weaken the gate (gate ≈ -5·sigmoid(·)).
            raw_g = gate_shift + (
                torch.randn(1, T, H, K, dtype=dtype, device=device) * 0.1
            )
            beta = torch.rand(1, T, H, dtype=dtype, device=device)
            cu = torch.tensor([0, T], dtype=torch.int32)
            ci = prepare_chunk_indices(cu, FLA_CHUNK_SIZE).to(device)
            q_n = l2norm_fwd(q)
            k_n = l2norm_fwd(k)
            g, _, w, u, kg, _ = _wy_pipeline(
                q_n, k_n, v, beta, raw_g, cu.to(device), ci, a_log, g_bias
            )

            # Per-chunk total decay (log2 gate at each chunk's last token).
            gk_last = g[0, FLA_CHUNK_SIZE - 1 :: FLA_CHUNK_SIZE]  # [NT, H, K]
            decay = torch.exp2(gk_last.float())
            print(
                f"T={T} {regime}: gk_last log2 min={gk_last.min().item():.2f} "
                f"max={gk_last.max().item():.2f}; exp2 min={decay.min().item():.3e} "
                f"max={decay.max().item():.3e}"
            )
            # Direct: does the full-sequence scan's final state equal the
            # suffix-only scan's final state? (Yes iff per-chunk decay
            # annihilates.)
            from vllm.models.glm5next.nvidia.ops.third_party.kda.kernels import (
                chunk_kda_with_fused_gate,
            )

            # NOTE: chunk_kda_with_fused_gate writes its output into a
            # contiguous v input (`o=v` buffer reuse); hand it clones to keep
            # v pristine.
            _, fin_full = chunk_kda_with_fused_gate(
                q=q,
                k=k,
                v=v.clone(),
                raw_g=raw_g,
                beta=beta,
                A_log=a_log,
                g_bias=g_bias,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu.to(device),
                safe_gate=SAFE_GATE,
                lower_bound=LOWER_BOUND,
            )
            _, fin_suffix = chunk_kda_with_fused_gate(
                q=q[:, -FLA_CHUNK_SIZE:],
                k=k[:, -FLA_CHUNK_SIZE:],
                v=v[:, -FLA_CHUNK_SIZE:].clone(),
                raw_g=raw_g[:, -FLA_CHUNK_SIZE:],
                beta=beta[:, -FLA_CHUNK_SIZE:],
                A_log=a_log,
                g_bias=g_bias,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=torch.tensor([0, FLA_CHUNK_SIZE], dtype=torch.int32).to(
                    device
                ),
                safe_gate=SAFE_GATE,
                lower_bound=LOWER_BOUND,
            )
            d_fs = (fin_full[0].float() - fin_suffix[0].float()).abs().max().item()
            print(
                f"T={T} {regime}: |fin_full|={fin_full.abs().max().item():.4e} "
                f"|fin_suffix|={fin_suffix.abs().max().item():.4e} "
                f"max|d|={d_fs:.4e}"
            )

            hm = kcp_compute_summaries(
                kg=kg,
                u=u,
                w=w,
                gk=g,
                cu_seqlens=cu.to(device),
                chunk_size=FLA_CHUNK_SIZE,
            )
            s_ext = hm[0, :, :K, :V]  # [H, K, V]
            m_kernel = hm[0, :, :, V:]  # [H, K, K]

            # Empirical affine transition of the scan: zero init -> S_ext,
            # identity init -> M^T + S_ext^T (states are [V, K] per head).
            zero_init = q.new_zeros(1, H, V, K, dtype=torch.float32)
            fin_zero = _scan_final(kg, w, u, g, zero_init, cu.to(device), ci)
            eye = torch.eye(K, dtype=torch.float32, device=device)
            ident_init = eye.view(1, 1, V, K).expand(1, H, V, K).contiguous()
            fin_eye = _scan_final(kg, w, u, g, ident_init, cu.to(device), ci)
            m_empirical = (fin_eye - fin_zero)[0].transpose(-1, -2)  # [H, K, K]

            err_ext = (s_ext - fin_zero[0].transpose(-1, -2)).abs().max().item()
            err_m = (m_kernel - m_empirical).abs()
            print(
                f"T={T} {regime}: |M_kernel|={m_kernel.abs().max().item():.4e} "
                f"|M_empirical|={m_empirical.abs().max().item():.4e} "
                f"max|dM|={err_m.max().item():.4e} "
                f"max|dS_ext|={err_ext:.4e}"
            )


if __name__ == "__main__":
    main()
