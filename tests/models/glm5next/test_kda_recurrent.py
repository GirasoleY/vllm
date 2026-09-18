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
        expected_summary_indices = torch.tensor(
            [
                (0 if slot == rank else plan.num_prefill_reqs) + request
                for request, slot in zip(
                    plan.scan_req_idx.tolist(),
                    (plan.init_gather_idx % plan.num_slots).tolist(),
                )
            ],
            dtype=torch.int64,
        )
        torch.testing.assert_close(plan.summary_rank_part_idx, expected_summary_indices)
        assert plan.summary_rank_part_idx.unique().numel() == plan.num_scan_rows
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
            assert plan.scan_chunk_offsets.tolist() == [0]
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


@pytest.mark.parametrize("counts", [[8], [8, 8, 8], [8, 3, 0]])
@pytest.mark.parametrize("rank_part_order", [False, True])
def test_kcp_merge_preserves_partial_states_and_contiguous_output(
    counts, rank_part_order
):
    """The dense path and ragged path preserve state at every slot boundary."""
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
        kcp_merge_states,
    )

    torch.manual_seed(7)
    slots, heads, dim = 8, 2, 128
    summaries = torch.randn(slots, len(counts), heads, dim, 2 * dim, device="cuda")
    summaries *= 0.003
    summaries[..., dim:] += 0.95 * torch.eye(dim, device="cuda")
    original = summaries.clone()
    base = torch.randn(len(counts), heads, dim, dim, device="cuda") * 0.2
    expected_inits = base.new_empty(len(counts), slots, heads, dim, dim)
    expected_final = torch.empty_like(base)
    for request, count in enumerate(counts):
        state = base[request]
        for slot in range(slots):
            expected_inits[request, slot] = state
            if slot < count:
                summary = original[slot, request]
                state = state @ summary[..., dim:].transpose(-1, -2)
                state = state + summary[..., :dim].transpose(-1, -2)
        expected_final[request] = state

    if rank_part_order:
        order = [
            slot for rank in range(slots // 2) for slot in (rank, slots - 1 - rank)
        ]
        summaries = summaries[order].contiguous()
    original_transition = summaries[..., dim:].clone()
    initial, final = kcp_merge_states(
        summaries,
        base,
        torch.tensor(counts, dtype=torch.int32, device="cuda"),
        all_slots_full=all(count == slots for count in counts),
        rank_part_order=rank_part_order,
    )
    torch.testing.assert_close(initial, expected_inits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, expected_final, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        summaries[..., dim:], original_transition, atol=0, rtol=0
    )
    assert initial.is_contiguous() and final.is_contiguous()


def _native_kcp_inputs(lengths, gate_shift=-4.0):
    """Prepare native workspace once for ragged, possibly empty segments."""
    from itertools import accumulate

    capability = current_platform.get_device_capability()
    if capability is None or capability.major not in (9, 10, 12):
        pytest.skip("Native KCP requires a FlashKDA-supported CUDA architecture")
    import vllm._flashkda_C  # noqa: F401

    heads, dim, tokens = 2, 128, sum(lengths)
    shape = (1, tokens, heads, dim)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    k[..., 4:] *= 0.001
    g = torch.randn_like(q) * 0.1 + gate_shift
    beta = torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, dim, device="cuda")
    cu = torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)
    offsets = torch.tensor(
        [0, *accumulate((length + 15) // 16 for length in lengths)],
        device="cuda",
        dtype=torch.int32,
    )
    size = torch.ops._flashkda_C.get_workspace_size(tokens, heads, len(lengths))
    workspace = torch.full((size,), 255, device="cuda", dtype=torch.uint8)
    final = torch.zeros(len(lengths), heads, dim, dim, device="cuda")
    out = torch.empty_like(v)
    if tokens:
        torch.ops._flashkda_C.fwd(
            q,
            k,
            v,
            g,
            beta,
            dim**-0.5,
            out,
            workspace,
            a_log,
            bias,
            LOWER_BOUND,
            None,
            final,
            cu,
            None,
            None,
        )
    return q, k, v, g, beta, a_log, bias, cu, offsets, workspace, final


@pytest.mark.parametrize("lengths", [[], [0, 0], [64, 97]])
def test_kcp_summary_destination_coverage(lengths):
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
        kcp_compute_summaries,
    )

    torch.manual_seed(53173)
    _, _, _, _, beta, _, _, cu, offsets, workspace, zero = _native_kcp_inputs(lengths)
    expected = kcp_compute_summaries(workspace, beta, cu, offsets, zero)
    out = torch.full((6, 2, 128, 256), 7.0, device="cuda")
    indices = torch.tensor([5, 1][: len(lengths)], device="cuda", dtype=torch.int64)
    out[indices] = float("nan")
    reference = out.clone()
    reference[indices] = expected
    actual = kcp_compute_summaries(
        workspace, beta, cu, offsets, zero, out=out, output_indices=indices
    )
    assert actual is out
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    if lengths == [0, 0]:
        torch.testing.assert_close(
            expected[..., :128], torch.zeros_like(zero), atol=0, rtol=0
        )
        torch.testing.assert_close(
            expected[..., 128:],
            torch.eye(128, device="cuda").expand_as(zero),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("lengths", [[0, 1, 15, 17, 63, 129], [4096] * 8])
@pytest.mark.parametrize("gate_shift", [-4.0, -12.0])
def test_kcp_native_scan_preserves_partitioned_accuracy(lengths, gate_shift):
    """Weak decay must retain transition precision and native scan equivalence."""
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
        kcp_compute_summaries,
        kcp_merge_states,
    )
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kernels import (
        chunk_kda_with_fused_gate,
    )

    torch.manual_seed(42)
    q, k, v, g, beta, a_log, bias, cu, offsets, workspace, zero = _native_kcp_inputs(
        lengths, gate_shift
    )
    base = torch.randn(1, 2, 128, 128, device="cuda")
    summaries = kcp_compute_summaries(workspace, beta, cu, offsets, zero)
    initial, final = kcp_merge_states(
        summaries.unsqueeze(1),
        base,
        torch.tensor([len(lengths)], device="cuda", dtype=torch.int32),
        all_slots_full=True,
    )
    out, recomputed = torch.empty_like(v), torch.empty_like(v)
    local_final, recomputed_final = torch.empty_like(zero), torch.empty_like(zero)
    torch.ops._flashkda_C.kcp_scan(v, beta, workspace, initial[0], cu, out, local_final)
    torch.ops._flashkda_C.fwd(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        recomputed,
        workspace,
        a_log,
        bias,
        LOWER_BOUND,
        initial[0],
        recomputed_final,
        cu,
        None,
        None,
    )
    torch.testing.assert_close(out, recomputed, atol=0, rtol=0)
    torch.testing.assert_close(local_final, recomputed_final, atol=0, rtol=0)

    full_cu = torch.tensor([0, sum(lengths)], device="cuda", dtype=torch.int32)
    native_out, native_final = torch.empty_like(v), torch.empty_like(base)
    torch.ops._flashkda_C.fwd(
        q,
        k,
        v,
        g,
        beta,
        128**-0.5,
        native_out,
        workspace,
        a_log,
        bias,
        LOWER_BOUND,
        base,
        native_final,
        full_cu,
        None,
        None,
    )
    reference = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=g,
        beta=beta.float().sigmoid(),
        A_log=a_log,
        g_bias=bias.flatten(),
        initial_state=base,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=full_cu,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )
    # Native FlashKDA rounds its running state to BF16. Partitioning must
    # retain native accuracy against FLA, allowing one extra BF16 rounding.
    for actual, native, expected in zip(
        (out, final), (native_out, native_final), reference
    ):
        assert torch.isfinite(actual).all()
        expected = expected.float()
        scale = expected.square().mean().sqrt().clamp_min(1e-8)
        difference = actual.float() - expected
        native_difference = native.float() - expected
        error = difference.square().mean().sqrt() / scale
        native_error = native_difference.square().mean().sqrt() / scale
        assert error <= 1.25 * native_error + 0.01
        assert difference.abs().max() <= 2 * native_difference.abs().max() + 0.002


@pytest.mark.parametrize(
    ("indices", "token_range", "strided"),
    [
        ([0, 1, 2], (0, 3), False),
        ([2, 3, 4], (2, 5), False),
        ([4, 2, 3], None, False),
        ([2, 3, 4], (2, 5), True),
        ([], (0, 0), False),
    ],
)
def test_kcp_output_destination_preserves_other_tokens(indices, token_range, strided):
    """Direct prefill writes preserve decode output, padding and strided storage."""
    from types import SimpleNamespace

    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import KcpPlan

    storage = torch.full((1, 7, 1, 4 if strided else 2), -777.0)
    output = storage[..., ::2] if strided else storage
    expected = storage.clone()
    expected_output = expected[..., ::2] if strided else expected
    values = torch.arange(len(indices) * 2).reshape(len(indices), 1, 2).float()
    prefix = min(indices) if indices else 0
    expected_output[:, :prefix] = 42
    expected_output[0, indices] = values
    plan = object.__new__(KcpPlan)
    plan.prefill_src_idx = torch.tensor(indices, dtype=torch.int64)
    plan.prefill_src_range = token_range
    plan.num_scan_rows = int(bool(indices))
    plan.num_block_tokens = prefix
    destinations = []

    def prefill(*args, out=None):
        destinations.append(out)
        if out is not None:
            out[0].copy_(values)
        return values

    def block(*args):
        args[5][:, :prefix] = 42

    empty = torch.empty(0)
    layer = SimpleNamespace(
        kv_cache=(empty, empty),
        _conv_state_dim_first=True,
        _merged_conv_weight=empty,
        q_conv1d=SimpleNamespace(bias=None),
        _forward_kcp_prefill=prefill,
        _forward_kcp_block=block,
    )
    forward = Glm5NextLinearAttention._forward_kcp
    forward(layer, empty, empty, empty, empty, output, plan, None)
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    assert len(destinations) == 1  # Empty ranks must still participate.
    if token_range is not None and not strided:
        assert destinations[0].untyped_storage().data_ptr() == storage.data_ptr()
    else:
        assert destinations[0] is None


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("use_tma", [False, True])
@torch.inference_mode()
def test_kda_matrices_overwrite_poisoned_storage(monkeypatch, chunk_size, use_tma):
    """Upper triangles and ragged tails must not depend on reused storage."""
    import importlib

    from vllm.models.glm5next.nvidia.ops.third_party.kda import kernels
    from vllm.triton_utils import triton

    solve = importlib.import_module(
        "vllm.third_party.flash_linear_attention.ops.solve_tril"
    )
    if use_tma and (
        torch.cuda.get_device_capability()[0] < 9
        or not any(
            hasattr(triton.language, name)
            for name in (
                "make_tensor_descriptor",
                "_experimental_make_tensor_descriptor",
            )
        )
    ):
        pytest.skip("TMA descriptors require supported NVIDIA hardware and Triton")
    if use_tma:
        from vllm.triton_utils.allocation import set_triton_allocator

        set_triton_allocator(torch.device("cuda"))
    monkeypatch.setattr(solve, "is_tma_supported", use_tma)
    torch.manual_seed(0)
    lengths = [0, 1, chunk_size - 1, chunk_size + 1]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    cu = torch.tensor(starts, device="cuda", dtype=torch.int32)
    indices = torch.tensor(
        [
            [n, c]
            for n, length in enumerate(lengths)
            for c in range((length + chunk_size - 1) // chunk_size)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    shape = (1, sum(lengths), 2, 128)
    q, k = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(2)
    ]
    g = torch.zeros(shape, device="cuda", dtype=torch.float32)
    beta = torch.full(shape[:-1], 0.1, device="cuda", dtype=torch.float32)

    class OutputStorage:
        def __init__(self, fill):
            self.fill = fill

        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *args, **kwargs):
            return torch.empty(*args, **kwargs).fill_(self.fill)

        def empty_like(self, *args, **kwargs):
            return torch.empty_like(*args, **kwargs).fill_(self.fill)

    outputs = []
    for fill in (0.0, float("nan")):
        storage = OutputStorage(fill)
        monkeypatch.setattr(kernels, "torch", storage)
        monkeypatch.setattr(solve, "torch", storage)
        a, aqk = kernels.chunk_kda_scaled_dot_kkt_fwd(
            q,
            k,
            g,
            beta,
            scale=128**-0.5,
            cu_seqlens=cu,
            chunk_indices=indices,
            chunk_size=chunk_size,
        )
        inverse = solve.solve_tril(
            a, cu_seqlens=cu, chunk_indices=indices, output_dtype=torch.bfloat16
        )
        outputs.append((a, aqk, inverse))
    for expected, actual in zip(*outputs, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
