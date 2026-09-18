# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA recurrent (decode) kernel.

The decode path hands the kernel column slices of the merged ``q|k|v`` conv
output and of the fused ``qkvbfg_a`` projection (beta), so q/k/v/beta are
token-strided rather than contiguous. The kernel must read them in place,
match a pure-PyTorch recurrence, and reject layouts it cannot address.
"""

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Triton kernel"
)

H, D = 16, 128
LOWER_BOUND = -5.0


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference for one sequence: ``[T, H, D]`` inputs, ``[H, D, D]``
    (v-major) state; mirrors the kernel's in-kernel gate, beta sigmoid and
    q/k l2norm.
    """
    q, k, v, raw_g, raw_beta = (x.float() for x in (q, k, v, raw_g, raw_beta))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(a_log.exp()[:, None] * (raw_g + g_bias))
    beta = torch.sigmoid(raw_beta)
    s = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        s = s * gate[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out, s


def make_inputs(num_seqs: int, query_len: int, device: torch.device):
    """Token-strided q/k/v/beta as the decode path produces them: column
    slices of a merged ``[T, q|k|v]`` conv output and of the fused
    ``[T, qkv|beta|f_a|g_a]`` projection.
    """
    T, proj = num_seqs * query_len, H * D
    qkv = torch.randn(T, 3 * proj, dtype=torch.bfloat16, device=device)
    projected = torch.randn(
        T, 3 * proj + H + 2 * D, dtype=torch.bfloat16, device=device
    )
    q, k, v = (x.view(1, T, H, D) for x in qkv.split(proj, dim=-1))
    beta = projected[:, 3 * proj : 3 * proj + H].unsqueeze(0)
    # (A size-1 token dim gets an arbitrary stride from `view`.)
    assert T == 1 or (q.stride(1) == 3 * proj and beta.stride(1) == projected.stride(0))
    inputs = dict(
        q=q,
        k=k,
        v=v,
        g=torch.randn(1, T, H, D, dtype=torch.bfloat16, device=device),
        beta=beta,
        a_log=0.5 * torch.randn(H, dtype=torch.float32, device=device),
        g_bias=0.1 * torch.randn(H * D, dtype=torch.float32, device=device),
        cu_seqlens=torch.arange(0, T + 1, query_len, dtype=torch.int32, device=device),
    )
    # Slot 0 is NULL_BLOCK_ID; sequences own random distinct slots (one per
    # token in the spec-decode layout).
    slots = torch.randperm(T, device=device).to(torch.int32) + 1
    if query_len == 1:
        inputs["ssm_state_indices"] = slots
    else:
        inputs["ssm_state_indices"] = slots.view(num_seqs, query_len)
        inputs["num_accepted_tokens"] = torch.randint(
            1, query_len + 1, (num_seqs,), dtype=torch.int32, device=device
        )
    state = torch.randn(T + 1, H, D, D, dtype=torch.float32, device=device)
    return inputs, state


def run_kernel(inputs: dict, state: torch.Tensor) -> torch.Tensor:
    out, _ = fused_recurrent_kda(
        **inputs,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
        sigmoid_beta=True,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    return out


@pytest.mark.parametrize(
    ("num_seqs", "query_len"), [(1, 1), (7, 1), (3, 3)], ids=["1x1", "7x1", "3x3"]
)
@torch.inference_mode()
def test_fused_recurrent_kda_matches_reference(num_seqs: int, query_len: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    expected_state = state.clone()
    out = run_kernel(inputs, state)

    indices = inputs["ssm_state_indices"].view(num_seqs, query_len)
    accepted = inputs.get("num_accepted_tokens")
    expected = torch.empty_like(out[0], dtype=torch.float32)
    for n in range(num_seqs):
        first = indices[n, 0 if accepted is None else accepted[n] - 1]
        s = expected_state[first]
        for t in range(query_len):
            tok = slice(n * query_len + t, n * query_len + t + 1)
            expected[tok], s = naive_recurrent_kda(
                inputs["q"][0, tok],
                inputs["k"][0, tok],
                inputs["v"][0, tok],
                inputs["g"][0, tok],
                inputs["beta"][0, tok],
                inputs["a_log"],
                inputs["g_bias"].view(H, D),
                s,
            )
            expected_state[indices[n, t]] = s

    torch.testing.assert_close(out[0].float(), expected, rtol=1e-2, atol=1e-3)
    used = indices.flatten().long()
    torch.testing.assert_close(state[used], expected_state[used], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("num_seqs", "query_len"), [(7, 1), (3, 3)], ids=["7x1", "3x3"]
)
@torch.inference_mode()
def test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous(
    num_seqs: int, query_len: int
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    for name in ("q", "k", "v", "beta"):
        assert not inputs[name].is_contiguous()
    contiguous = {
        name: x.contiguous() if name in ("q", "k", "v", "beta") else x
        for name, x in inputs.items()
    }
    state_ref = state.clone()
    out_ref = run_kernel(contiguous, state_ref)
    out = run_kernel(inputs, state)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_recurrent_kda_rejects_unaddressable_layouts():
    """Layouts the token-stride addressing cannot express must fail loudly
    rather than read the wrong tokens: a batch slice of a wider buffer
    (``stride(0) != T * stride(1)``), overlapping tokens, and a head-strided
    (transposed) block.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(1, 1, device)
    T = 4
    base = torch.randn(4, T, 2 * H * D, dtype=torch.bfloat16, device=device)
    bad_q = {
        "batch slice": base[::2, :, : H * D].view(2, T, H, D),
        "overlapping tokens": base[:1, :, : H * D].as_strided(
            (1, T, H, D), (0, D, D, 1)
        ),
        "head-strided": base[:1, :, : H * D].view(1, T, D, H).transpose(2, 3),
    }
    for q in bad_q.values():
        broken = dict(inputs, q=q, k=q, v=q)
        broken["cu_seqlens"] = None if q.shape[0] > 1 else inputs["cu_seqlens"]
        with pytest.raises(AssertionError, match=r"torch.Size"):
            run_kernel(broken, state)


@pytest.mark.parametrize(
    ("query_len", "prefix_kind"),
    [
        (2, None),
        (3, None),
        (17, None),
        (32769, None),
        (32768, "fresh"),
        (32768, "continued"),
        (32768, "decode"),
    ],
)
def test_kcp_plan_preserves_short_continuation_halos_and_empty_ranks(
    query_len, prefix_kind
):
    from types import SimpleNamespace

    import numpy as np

    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import build_kcp_plan
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    # Short prefills use the ordinary path's initial-state masking; plain
    # decodes still share KCP batches with a large prefill.
    prefix = int(prefix_kind is not None)
    lengths = np.array([1] * prefix + [query_len], dtype=np.int32)
    prefilling = np.array(([prefix_kind != "decode"] if prefix else []) + [True])
    starts = np.array(
        ([0 if prefix_kind == "fresh" else 9] if prefix else []) + [9], dtype=np.int32
    )
    query_start = np.concatenate(([0], lengths.cumsum())).astype(np.int32)
    tails = torch.full((16, 3), -999, dtype=torch.int64)
    chunk_size = (query_len + 15) // 16
    for slot in range(16):
        end = min((slot + 1) * chunk_size, query_len)
        take = min(3, max(0, end - slot * chunk_size))
        if take:
            tails[slot, 3 - take :] = torch.arange(9 + end - take, 9 + end)
    for rank in range(8):
        manager = PCPManager(8, rank, torch.device("cpu"))
        segments = manager._get_rank_segments(rank, lengths, prefilling, query_start)
        local_lens = np.array([s.num_tokens for s in segments], dtype=np.int32)
        manager._global_batch = SimpleNamespace(
            num_scheduled_tokens=lengths,
            is_prefilling_np=prefilling,
            num_computed_tokens_np=starts,
            query_start_loc_np=query_start,
            num_reqs=len(lengths),
            num_draft_tokens_per_req=None,
            req_ids=list(range(len(lengths))),
        )
        manager._local_batch = SimpleNamespace(
            req_ids=[s.global_batch_req_idx for s in segments],
            num_reqs=len(segments),
            num_scheduled_tokens=local_lens,
            num_computed_tokens_np=np.array(
                [
                    starts[s.global_batch_req_idx]
                    + s.global_batch_slice.start
                    - query_start[s.global_batch_req_idx]
                    for s in segments
                ],
                dtype=np.int32,
            ),
            query_start_loc_np=np.concatenate(([0], local_lens.cumsum())),
            is_prefilling_np=np.array(
                [prefilling[s.global_batch_req_idx] for s in segments], dtype=np.bool_
            ),
        )
        plan = build_kcp_plan(manager, torch.device("cpu"))
        if prefix_kind in ("fresh", "continued"):
            assert plan is None  # Uniform fallback, including ranks without the token.
            continue
        assert plan is not None
        assert plan.num_block_tokens == prefix
        assert plan.prefill_src_range == (prefix, int(local_lens.sum()))
        final_indices = plan.final_tail_idx[0]
        final_window = torch.where(
            final_indices >= 0,
            tails.flatten()[final_indices.clamp(min=0)],
            9 + final_indices,
        )
        torch.testing.assert_close(
            final_window, torch.arange(9 + query_len - 3, 9 + query_len)
        )
        if plan.num_scan_rows == 0:
            assert plan.scan_chunk_indices.shape == (0, 2)
            continue
        prefill_segments = [s for s in segments if prefilling[s.global_batch_req_idx]]
        for scan_idx, segment in enumerate(prefill_segments):
            indices = plan.halo_tail_idx[scan_idx]
            actual = torch.where(
                indices >= 0, tails.flatten()[indices.clamp(min=0)], 9 + indices
            )
            start = 9 + segment.global_batch_slice.start - prefix
            torch.testing.assert_close(actual, torch.arange(start - 3, start))


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim_first", [True, False])
def test_kcp_scatter_publishes_final_conv_window(cache_dtype, dim_first):
    """Raw tails and old prefixes update only live cache columns on every rank."""
    from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states

    torch.manual_seed(7)
    device = "cuda"
    channels = 3 * 2 * 128
    shape = (6, channels, 5) if dim_first else (6, 5, channels)
    conv = torch.randn(shape, dtype=cache_dtype, device=device)
    if not dim_first:
        conv = conv.transpose(1, 2)
    expected_conv = conv.clone()
    tails = torch.randn(9, channels + 5, dtype=torch.bfloat16, device=device)[
        :, :channels
    ]
    indices = torch.tensor([4, 0, 2, 0, 1, 0], device=device)[::2]
    tail_indices = torch.tensor([[-1, 2, 5], [6, 7, 8], [-1, 0, 3]], device=device)
    has_initial = torch.tensor([True, True, False], device=device)
    expected_conv[4, :, :3] = torch.stack(
        (conv[4, :, 2], tails[2].to(cache_dtype), tails[5].to(cache_dtype)), dim=1
    )
    expected_conv[2, :, :3] = tails[6:9].transpose(0, 1).to(cache_dtype)
    expected_conv[1, :, :3] = torch.stack(
        (
            torch.zeros_like(conv[1, :, 0]),
            tails[0].to(cache_dtype),
            tails[3].to(cache_dtype),
        ),
        dim=1,
    )
    state = torch.randn(6, 2, 128, 128, device=device)
    src = torch.randn(6, 2, 128, 128, device=device)[::2]
    expected_state = state.clone()
    expected_state[indices] = src
    default_state = state.clone()
    scatter_states(default_state, src, indices)
    scatter_states(
        state,
        src,
        indices,
        conv_state=conv,
        conv_tail=tails,
        conv_tail_indices=tail_indices,
        has_initial_state=has_initial,
    )
    torch.testing.assert_close(default_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(conv, expected_conv, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("indices", "token_range"),
    [([1, 2, 3], (1, 4)), ([], (0, 0)), ([3, 1], None)],
)
def test_kcp_prefill_selection_preserves_strided_values(indices, token_range):
    from types import SimpleNamespace

    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import KcpPlan

    plan = SimpleNamespace(
        prefill_src_idx=torch.tensor(indices, dtype=torch.int64),
        prefill_src_range=token_range,
    )
    packed = torch.arange(6 * 8).reshape(6, 8)[:, :4]
    for dim, tensor in [(0, packed), (1, packed.unsqueeze(0))]:
        actual = KcpPlan.select_prefill_tokens(plan, tensor, dim)
        expected = tensor.index_select(dim, plan.prefill_src_idx)
        torch.testing.assert_close(actual, expected)
        if token_range is not None:
            assert (
                actual.untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
            )
            assert actual.stride() == tensor.stride()
