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


@pytest.mark.parametrize("world,dcp", [(4, 1), (4, 4), (8, 8)])
@pytest.mark.parametrize("halo_size", [1, 3, 4])
@pytest.mark.parametrize(
    "query_len,mode",
    [
        (2, "prefill"),
        (3, "mixed"),
        (17, "mixed"),
        (32768, "prefill"),
        (32769, "mixed"),
        (1, "decode"),
    ],
)
def test_kcp_plan_preserves_tokens_halos_and_request_order(
    world, dcp, halo_size, query_len, mode
):
    """Check every rank against global token positions, including cached prefixes."""
    from types import SimpleNamespace

    import numpy as np

    from vllm.models.glm5next.nvidia.ops.kcp import build_kcp_plan
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    lengths = np.array([1, query_len, 1, 3], dtype=np.int32)
    prefilling = np.array([True, True, mode != "mixed", True])
    if mode == "decode":
        lengths[:] = 1
        prefilling[:] = False
    starts = np.array([0, 9, 9, 7], dtype=np.int32)
    qsl = np.r_[np.int32(0), lengths.cumsum(dtype=np.int32)]
    positions = torch.cat(
        [
            100000 * g + starts[g] + torch.arange(length)
            for g, length in enumerate(lengths)
        ]
    )
    prefill_reqs = np.flatnonzero(prefilling)
    decode_reqs = np.flatnonzero(~prefilling)
    global_batch = SimpleNamespace(
        num_scheduled_tokens=lengths,
        is_prefilling_np=prefilling,
        num_computed_tokens_np=starts,
        query_start_loc_np=qsl,
        num_reqs=len(lengths),
        req_ids=list(range(len(lengths))),
        num_draft_tokens_per_req=None,
    )
    plans, tails = [], []
    for rank in range(world):
        manager = PCPManager(world, rank, torch.device("cpu"), dcp_world_size=dcp)
        segments = manager._get_rank_segments(rank, lengths, prefilling, qsl)
        manager._local_segments = tuple(segments)
        manager._global_batch = global_batch
        manager._local_batch = SimpleNamespace()
        plan = build_kcp_plan(manager, torch.device("cpu"), halo_size)
        assert plan is not None
        assert plan.num_prefill_reqs == len(prefill_reqs)
        assert plan.decode_entry_idx.tolist() == decode_reqs.tolist()
        assert plan.prefill_entry_idx.tolist() == prefill_reqs.tolist()
        assert (
            plan.prefill_has_initial_state.tolist() == (starts[prefilling] > 0).tolist()
        )
        replicated = manager.replicated_requests(lengths, prefilling)
        scan_segments = [s for s in segments if prefilling[s.global_batch_req_idx]]
        scan_segments.sort(
            key=lambda s: (s.global_batch_req_idx, s.global_batch_slice.start)
        )

        def token_positions(segments):
            indices = [
                range(s.global_batch_slice.start, s.global_batch_slice.stop)
                for s in segments
            ]
            return torch.tensor(
                [token for segment in indices for token in segment], dtype=torch.int64
            )

        local_tokens, expected = (
            token_positions(segments),
            token_positions(scan_segments),
        )
        local_tokens, expected = positions[local_tokens], positions[expected]
        # Exercise both selection dimensions on token-strided projection storage.
        packed = local_tokens[:, None].expand(-1, 4).clone()[:, ::2]
        for dim, tensor in [(0, packed), (1, packed.unsqueeze(0))]:
            actual = plan.select_prefill_tokens(tensor, dim)
            torch.testing.assert_close(
                actual.squeeze(0) if dim else actual, expected[:, None].expand(-1, 2)
            )
            if plan.prefill_src_range is not None:
                assert (
                    actual.untyped_storage().data_ptr()
                    == tensor.untyped_storage().data_ptr()
                )
                assert actual.stride() == tensor.stride()
        torch.testing.assert_close(
            local_tokens[: len(decode_reqs)],
            positions[qsl[decode_reqs]],
        )
        assert plan.summary_rank_part_idx.unique().numel() == len(scan_segments)
        for segment, request, initial, destination in zip(
            scan_segments,
            plan.scan_req_idx.tolist(),
            plan.init_gather_idx.tolist(),
            plan.summary_rank_part_idx.tolist(),
            strict=True,
        ):
            g = segment.global_batch_req_idx
            assert prefill_reqs[request] == g
            width = (
                lengths[g]
                if replicated[g]
                else (lengths[g] + 2 * world - 1) // (2 * world)
            )
            slot = (segment.global_batch_slice.start - qsl[g]) // width
            assert initial == request * 2 * world + slot
            assert slot in ((0,) if replicated[g] else (rank, 2 * world - 1 - rank))
            assert destination == request + (
                0 if replicated[g] or slot == rank else len(prefill_reqs)
            )
            assert plan.num_slots_dev[request] == (lengths[g] + width - 1) // width
        tail = torch.full(plan.tail_src_idx.shape, -1, dtype=torch.int64)
        valid = plan.tail_src_idx >= 0
        tail[valid] = expected[plan.tail_src_idx[valid]]
        tails.append(tail)
        plans.append((plan, scan_segments))
        if not scan_segments:
            assert plan.scan_chunk_indices.shape == (0, 2)

    gathered = torch.cat(tails)
    for plan, segments in plans:
        for segment, indices in zip(segments, plan.halo_tail_idx, strict=True):
            g = segment.global_batch_req_idx
            actual = torch.where(
                indices >= 0,
                gathered[indices.clamp(min=0)],
                100000 * g + starts[g] + indices,
            )
            end = 100000 * g + starts[g] + segment.global_batch_slice.start - qsl[g]
            torch.testing.assert_close(actual, torch.arange(end - halo_size, end))
        for request, g in enumerate(prefill_reqs):
            indices = plan.final_tail_idx[request]
            actual = torch.where(
                indices >= 0,
                gathered[indices.clamp(min=0)],
                100000 * g + starts[g] + indices,
            )
            torch.testing.assert_close(
                actual,
                100000 * g
                + starts[g]
                + torch.arange(lengths[g] - halo_size, lengths[g]),
            )
    # Reject drafts before constructing any KCP metadata, regardless of batch type.
    global_batch.num_draft_tokens_per_req = np.ones(len(lengths), dtype=np.int32)
    with pytest.raises(NotImplementedError, match="speculative decoding"):
        build_kcp_plan(manager, torch.device("cpu"), halo_size)


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim_first", [True, False])
@pytest.mark.parametrize("width", [1, 3, 4])
def test_kcp_scatter_publishes_final_conv_window(cache_dtype, dim_first, width):
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
    tail_indices = torch.tensor(
        [
            [-1] + list(range(width - 1)),
            list(range(3, 3 + width)),
            [-1] + list(range(2, 2 + width - 1)),
        ],
        device=device,
    )
    has_initial = torch.tensor([True, True, False], device=device)
    for request, slot in enumerate(indices.tolist()):
        for column, source in enumerate(tail_indices[request].tolist()):
            if source >= 0:
                expected_conv[slot, :, column] = tails[source].to(cache_dtype)
            elif has_initial[request]:
                expected_conv[slot, :, column] = conv[slot, :, source + width]
            else:
                expected_conv[slot, :, column] = 0
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


@pytest.mark.parametrize("counts", [[8], [8, 8, 8], [8, 3, 0], [1, 8, 3, 0]])
def test_kcp_merge_preserves_partial_states_and_contiguous_output(counts):
    """The dense path and ragged path preserve state at every slot boundary."""
    from vllm.models.glm5next.nvidia.ops.kcp import (
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

    order = [slot for rank in range(slots // 2) for slot in (rank, slots - 1 - rank)]
    summaries = summaries[order].contiguous()
    original_transition = summaries[..., dim:].clone()
    initial, final = kcp_merge_states(
        summaries,
        base,
        torch.tensor(counts, dtype=torch.int32, device="cuda"),
        all_slots_full=all(count == slots for count in counts),
    )
    torch.testing.assert_close(initial, expected_inits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, expected_final, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        summaries[..., dim:], original_transition, atol=0, rtol=0
    )
    assert initial.is_contiguous() and final.is_contiguous()


def _kcp_layer(
    monkeypatch,
    indices,
    token_range,
    entries,
    state_indices,
    *,
    prefill_entry=None,
    cache=None,
    weights=None,
    heads=1,
    dim=2,
    a_log=None,
    bias=None,
):
    """Use local projections and global state metadata in the shared layer dispatch."""
    from types import MethodType, SimpleNamespace

    from vllm.models.glm5next.common import kda
    from vllm.models.glm5next.nvidia.ops import kcp

    device = state_indices.device
    count, scan_count = len(entries), int(bool(indices))
    if prefill_entry is None:
        prefill_entry = count
    conv, state = (torch.empty(0), torch.empty(0)) if cache is None else cache
    plan = object.__new__(kcp.KcpPlan)
    plan.__dict__.update(
        prefill_src_idx=torch.tensor(indices, device=device, dtype=torch.int64),
        prefill_src_range=token_range,
        num_scan_rows=scan_count,
        num_prefill_reqs=1,
        num_decode_reqs=count,
        decode_entry_idx=entries,
        decode_cu_seqlens=torch.arange(count + 1, device=device, dtype=torch.int32),
        prefill_entry_idx=torch.tensor([prefill_entry], device=device),
        prefill_has_initial_state=torch.tensor([True], device=device),
        scan_cu_seqlens=torch.tensor(
            [0, len(indices)], device=device, dtype=torch.int32
        ),
        scan_chunk_indices=torch.zeros(
            (scan_count, 2), device=device, dtype=torch.int32
        ),
        scan_chunk_offsets=torch.arange(
            scan_count + 1, device=device, dtype=torch.int32
        ),
        conv_meta=None,
        halo_tail_idx=torch.zeros((scan_count, 3), device=device, dtype=torch.int64),
        scan_req_idx=torch.zeros(scan_count, device=device, dtype=torch.int64),
        final_tail_idx=torch.zeros((1, 3), device=device, dtype=torch.int64),
    )
    layer = SimpleNamespace(
        prefix="kda",
        conv_size=4,
        kv_cache=(conv, state),
        _conv_state_and_weights=lambda: (conv, weights, None),
        A_log=a_log,
        dt_bias=bias,
        local_num_heads=heads,
        head_dim=dim,
        local_projection_size=heads * dim,
        kda_safe_gate=True,
        kda_lower_bound=LOWER_BOUND,
        kda_prefill_backend="triton",
    )
    layer._forward_decode = MethodType(
        kda.Glm5NextLinearAttention._forward_decode, layer
    )
    metadata = kda.GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=99,
        num_decodes=count,
        num_decode_tokens=count,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=99 + count,
        non_spec_state_indices_tensor=state_indices,
    )
    context = SimpleNamespace(
        attn_metadata={"kda": metadata},
        additional_kwargs={
            "pcp_manager": SimpleNamespace(uses_global_kda_metadata=True)
        },
    )
    monkeypatch.setattr(kda, "get_forward_context", lambda: context)
    monkeypatch.setattr(kcp, "maybe_get_kcp_plan", lambda *args: plan)
    monkeypatch.setattr(
        kcp, "gather_conv_tail_rows", lambda _, qkv: qkv.new_empty(3, 3 * heads * dim)
    )
    monkeypatch.setattr(
        kda,
        "causal_conv1d_fn",
        lambda x, *args, **kwargs: x.T.reshape(-1, 3, heads * dim).permute(1, 0, 2),
    )
    return layer


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
def test_kcp_output_destination_preserves_other_tokens(
    monkeypatch, indices, token_range, strided
):
    """Direct prefill writes preserve decode output, padding and strided storage."""
    from vllm.models.glm5next.common import kda
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    storage = torch.full((1, 7, 1, 4 if strided else 2), -777.0)
    output = storage[..., ::2] if strided else storage
    expected = storage.clone()
    expected_output = expected[..., ::2] if strided else expected
    expected_output.zero_()
    values = torch.arange(len(indices) * 2).reshape(len(indices), 1, 2).float()
    projected = torch.arange(7 * 12).reshape(7, 12).float()[:, :6]
    gates = torch.arange(7 * 4).reshape(1, 7, 1, 4).float()[..., :2]
    prefix = min(indices) if indices else 0
    expected_output[:, :prefix] = 42
    expected_output[0, indices] = values
    destinations = []
    preparations = []
    publications = []
    final_state = torch.full((1, 1, 2, 2), 17.0)

    def prepare_states(*args, state_indices, **kwargs):
        preparations.append(state_indices.tolist())
        return None, final_state

    def prefill(*, k, v, raw_g, state_preparer, out=None, **kwargs):
        torch.testing.assert_close(k[0, :, 0], projected[indices, 2:4])
        torch.testing.assert_close(raw_g, gates[:, indices])
        _, final = state_preparer(k, k, v, raw_g)
        destinations.append(out)
        if out is not None:
            out[0].copy_(values)
        return values.unsqueeze(0) if out is None else out, final

    def publish(state, final, indices, **kwargs):
        publications.append(final)

    def block(*args):
        args[-1].fill_(42)

    layer = _kcp_layer(
        monkeypatch,
        indices,
        token_range,
        torch.arange(prefix),
        torch.arange(prefix + 1),
    )
    layer._forward_decode = block
    from vllm.models.glm5next.nvidia.ops import kcp

    monkeypatch.setattr(kcp, "prepare_kcp_states", prepare_states)
    monkeypatch.setattr(kda, "chunk_kda_with_fused_gate", prefill)
    monkeypatch.setattr(kda, "scatter_states", publish)
    Glm5NextLinearAttention._forward(
        layer,
        projected,
        gates,
        torch.empty(1, 7, 1),
        output,
    )
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)
    assert len(destinations) == 1  # Empty ranks must still participate.
    assert preparations == [[prefix]]
    assert len(publications) == 1 and publications[0] is final_state
    if token_range is not None and not strided:
        assert destinations[0].untyped_storage().data_ptr() == storage.data_ptr()
    else:
        assert destinations[0] is None


@pytest.mark.parametrize("use_tma", [False, True])
@pytest.mark.parametrize("gate_shift", [0.0, -4.0, -10.0])
@pytest.mark.parametrize(
    "dtype,heads,dim,chunk_size,safe_gate,strided,lower_bound,gate_offset,uniform",
    [
        (torch.bfloat16, 64, 128, 64, True, False, LOWER_BOUND, 0.0, False),
        (torch.float16, 2, 96, 32, True, True, LOWER_BOUND, 0.0, False),
        (torch.float32, 1, 192, 16, False, True, LOWER_BOUND, 0.0, False),
        (torch.bfloat16, 2, 128, 64, True, False, -10.0, 12.0, False),
        # Bare exp factors are normal, but scaled BF16 operands lose precision.
        (torch.bfloat16, 2, 128, 64, True, False, -5.4542, 80.0, True),
    ],
)
@torch.inference_mode()
def test_flashkda_prepare_matches_fla_for_ragged_segments(
    monkeypatch,
    use_tma,
    gate_shift,
    dtype,
    heads,
    dim,
    chunk_size,
    safe_gate,
    strided,
    lower_bound,
    gate_offset,
    uniform,
):
    """Native preparation covers full, short and empty segments without fallback."""
    pytest.importorskip("vllm._flashkda_C")
    if not hasattr(torch.ops._flashkda_C, "kcp_prepare"):
        pytest.skip("FlashKDA extension was built without KCP preparation")
    import importlib

    from vllm.models.glm5next.common.kda import _cast_sigmoid
    from vllm.models.glm5next.nvidia.ops.third_party.kda import kernels
    from vllm.models.glm5next.nvidia.ops.third_party.kda.kcp import (
        prepare_kcp,
    )
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril

    solve = importlib.import_module(
        "vllm.third_party.flash_linear_attention.ops.solve_tril"
    )
    prepare_module = importlib.import_module(prepare_kcp.__module__)
    if use_tma:
        from vllm.triton_utils.allocation import set_triton_allocator

        set_triton_allocator(torch.device("cuda"))
    monkeypatch.setattr(solve, "is_tma_supported", use_tma)

    class PoisonedStorage:
        def __getattr__(self, name):
            return getattr(torch, name)

        def empty(self, *args, **kwargs):
            return torch.empty(*args, **kwargs).fill_(float("nan"))

        def empty_like(self, *args, **kwargs):
            return torch.empty_like(*args, **kwargs).fill_(float("nan"))

    for module in (kernels, solve, prepare_module):
        monkeypatch.setattr(module, "torch", PoisonedStorage())
    torch.manual_seed(9202104)
    tokens = 2 * chunk_size + 1
    inputs_shape = (1, tokens, heads, dim * (2 if strided else 1))
    q, k, raw_g = [
        torch.randn(inputs_shape, device="cuda", dtype=dtype) for _ in range(3)
    ]
    if strided:
        q, k, raw_g = (x[..., ::2] for x in (q, k, raw_g))
    raw_g.mul_(0.1).add_(gate_shift + gate_offset)
    beta_raw = torch.randn(1, tokens, 128, device="cuda", dtype=dtype)[
        ..., 17 : 17 + heads
    ]
    a_log = torch.randn(heads, device="cuda") * 0.1
    bias = torch.randn(heads * dim, device="cuda") * 0.1
    if uniform:
        q.fill_(1)
        k.fill_(1)
        beta_raw.zero_()
        a_log.zero_()
        bias.zero_()
    inputs = q, k, raw_g, beta_raw, a_log, bias
    cu = torch.tensor([0, 0, 1, chunk_size, tokens], device="cuda", dtype=torch.int64)
    chunks = torch.tensor(
        [[1, 0], [2, 0], [3, 0], [3, 1]], device="cuda", dtype=torch.int32
    )
    metadata = dict(cu_seqlens=cu, chunk_indices=chunks, chunk_size=chunk_size)
    actual = prepare_kcp(
        *inputs,
        cu,
        chunks,
        lower_bound=lower_bound if safe_gate else None,
        chunk_size=chunk_size,
    )
    qn, kn, beta = (
        l2norm_fwd(q.contiguous()),
        l2norm_fwd(k.contiguous()),
        _cast_sigmoid(beta_raw),
    )
    gate = kernels.fused_kda_gate_chunk_cumsum(
        raw_g.contiguous(),
        a_log,
        bias,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        **metadata,
    )
    a, aqk = kernels.chunk_kda_scaled_dot_kkt_fwd(
        qn,
        kn,
        gate,
        beta,
        scale=dim**-0.5,
        output_dtype=torch.float32,
        **metadata,
    )
    # Poisoning covers padding and upper triangles in both preparation paths.
    for value in (actual[4], actual[5], a, aqk):
        assert torch.isfinite(value).all()
    bf16_inverse = solve_tril(
        a, cu_seqlens=cu, chunk_indices=chunks, output_dtype=torch.bfloat16
    )
    assert torch.isfinite(bf16_inverse).all()
    # Compare the complete inverse; native preparation has inverted its diagonal.
    inverse_args = dict(cu_seqlens=cu, chunk_indices=chunks, output_dtype=torch.float32)
    actual = (
        *actual[:4],
        solve_tril(actual[4], diagonal_inverted=True, **inverse_args),
        actual[5],
    )
    a = solve_tril(a, **inverse_args)
    # Approximate gate activation and BF16 MMA differ from FLA materialization.
    tolerances = [
        (0, 0),
        (0, 0),
        (1e-3, 2e-3),
        (0, 4e-6),
        (1e-2, 1e-3),
        (1e-2, 2e-4),
    ]
    for value, expected, (rtol, atol) in zip(
        actual, (qn, kn, gate, beta, a, aqk), tolerances, strict=True
    ):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected, rtol=rtol, atol=atol)
    empty = [x[:, :0] for x in inputs[:4]] + list(inputs[4:])
    empty_cu = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    empty_chunks = torch.empty((0, 2), device="cuda", dtype=torch.int64)
    result = prepare_kcp(
        *empty,
        empty_cu,
        empty_chunks,
        lower_bound=lower_bound if safe_gate else None,
        chunk_size=chunk_size,
    )
    assert all(value.numel() == 0 for value in result)


@pytest.mark.parametrize("batch", ["ragged", "empty", "none"])
@pytest.mark.parametrize(
    "dim,value_dim,chunk_size",
    [(64, 80, 16), (96, 80, 32), (128, 80, 64), (128, 128, 64), (192, 128, 64)],
)
@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="KCP CuTe summaries require SM100-family GPUs",
)
@torch.inference_mode()
def test_kcp_summaries_match_reference_and_preserve_destinations(
    dim, value_dim, chunk_size, batch
):
    """Check S arithmetic/rounding and the FP32 M chain, including empty segments."""
    from itertools import accumulate

    from vllm.models.glm5next.nvidia.ops.kcp import kcp_compute_summaries

    torch.manual_seed(57458)
    lengths = [0, 1, chunk_size + 3, 8 * chunk_size + 1]
    if batch != "ragged":
        lengths = [0, 0] if batch == "empty" else []
    tokens, heads = sum(lengths), 2
    shape = (1, tokens, heads, dim)
    kg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.025
    w = torch.randn_like(kg) * 0.025
    u = torch.randn((1, tokens, heads, value_dim), dtype=kg.dtype, device="cuda")
    gate = -torch.rand(shape, device="cuda") * 0.03
    offsets = [0] + list(accumulate(lengths))
    cu = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    dense = kcp_compute_summaries(kg, u, w, gate, cu, chunk_size)
    out = torch.full(
        (len(lengths) + 3, heads, dim, value_dim + dim), 7.0, device="cuda"
    )
    indices = torch.arange(len(lengths), 0, -1, device="cuda", dtype=torch.int64)
    out[indices] = float("nan")
    expected = out.clone()
    expected[indices] = dense
    actual = kcp_compute_summaries(
        kg, u, w, gate, cu, chunk_size, out=out, output_indices=indices
    )
    assert actual is out
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual = dense.cpu().double()
    # Independent CPU FP64 products; preserve the S producer's BF16 carry
    # and residual conversions explicitly. No candidate kernel forms the oracle.
    kg, u, w, gate = (x[0].cpu().double() for x in (kg, u, w, gate))
    for seq, (start, end) in enumerate(zip(offsets, offsets[1:])):
        state = torch.zeros(heads, dim, value_dim, dtype=torch.float64)
        transfer = torch.eye(dim, dtype=torch.float64).expand(heads, dim, dim).clone()
        for begin in range(start, end, chunk_size):
            stop = min(begin + chunk_size, end)
            keys = kg[begin:stop].transpose(0, 1)
            weights = w[begin:stop].transpose(0, 1)
            values = u[begin:stop].transpose(0, 1)
            decay = gate[stop - 1].exp2()
            residual = values - weights @ state.to(torch.bfloat16).double()
            state = (
                decay[..., None] * state
                + keys.transpose(-1, -2) @ residual.to(torch.bfloat16).double()
            )
            transition = torch.diag_embed(decay) - keys.transpose(-1, -2) @ weights
            transfer = transition @ transfer
        summary = actual[seq, ..., :value_dim]
        if end - start <= 2 * chunk_size:
            torch.testing.assert_close(summary, state, atol=3e-4, rtol=1e-2)
        else:
            # FP32/FP64 reductions can cross repeated BF16 rounding boundaries.
            # Bound each head's aggregate error and isolated element errors.
            error = summary - state
            assert torch.all(
                torch.linalg.vector_norm(error, dim=(-2, -1))
                <= 2e-4 * torch.linalg.vector_norm(state, dim=(-2, -1))
            )
            assert error.abs().max() < 2e-3
        torch.testing.assert_close(
            actual[seq, ..., value_dim:], transfer, atol=2e-6, rtol=1e-4
        )


@pytest.mark.parametrize("use_fused_prepare", [False, True])
@pytest.mark.parametrize("empty_rank", [False, True])
@torch.inference_mode()
def test_chunk_kda_state_preparer_preserves_scan_and_final_state(
    use_fused_prepare, empty_rank
):
    """A prepared continuation seed drives the shared scan, including empty ranks."""
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
    )

    if use_fused_prepare:
        pytest.importorskip("vllm._flashkda_C")
        if not hasattr(torch.ops._flashkda_C, "kcp_prepare"):
            pytest.skip("FlashKDA extension was built without KCP preparation")

    torch.manual_seed(941)
    heads, dim = 2, 128
    lengths = [] if empty_rank else [1, 66]
    tokens = sum(lengths)
    shape = (1, tokens, heads, dim)
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    raw_g = torch.randn_like(q) * 0.1 - 4
    raw_beta = torch.randn(shape[:-1], device="cuda", dtype=q.dtype)
    a_log = torch.zeros(heads, device="cuda")
    g_bias = torch.zeros(heads * dim, device="cuda")
    initial = torch.randn(len(lengths), heads, dim, dim, device="cuda") * 0.1
    global_final = torch.full((2, heads, dim, dim), 13.0, device="cuda")
    cu = torch.tensor(
        [0] if empty_rank else [0, 1, tokens], device="cuda", dtype=torch.int32
    )
    chunks = torch.tensor(
        [] if empty_rank else [[0, 0], [1, 0], [1, 1]],
        device="cuda",
        dtype=torch.int32,
    ).reshape(-1, 2)
    offsets = torch.tensor(
        [0] if empty_rank else [0, 1, 3], device="cuda", dtype=torch.int32
    )
    common = dict(
        q=q,
        k=k,
        raw_g=raw_g,
        A_log=a_log,
        g_bias=g_bias,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        output_final_state=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )
    if not empty_rank:
        expected, _ = chunk_kda_with_fused_gate(
            **common,
            v=v.clone(),
            beta=raw_beta.float().sigmoid(),
            initial_state=initial,
        )
        for seq, (start, end) in enumerate(((0, 1), (1, tokens))):
            reference, _ = naive_recurrent_kda(
                q[0, start:end],
                k[0, start:end],
                v[0, start:end],
                raw_g[0, start:end],
                raw_beta[0, start:end],
                a_log,
                g_bias.view(heads, dim),
                initial[seq],
            )
            torch.testing.assert_close(
                expected[0, start:end].float(), reference, atol=2e-3, rtol=2e-2
            )

    calls = []

    def prepare_states(kg, w, u, g):
        calls.append(kg.shape[1])
        assert kg.shape == w.shape == g.shape == shape
        assert u.shape == shape
        return initial, global_final

    storage = torch.full(
        (1, tokens + 2, heads, dim), -777.0, device="cuda", dtype=v.dtype
    )
    destination = storage[:, 1:-1]
    actual, final = chunk_kda_with_fused_gate(
        **common,
        v=v.clone(),
        beta=raw_beta,
        chunk_indices=chunks,
        chunk_offsets=offsets,
        out=destination,
        state_preparer=prepare_states,
        use_fused_prepare=use_fused_prepare,
        sigmoid_beta=True,
    )
    assert calls == [tokens]
    assert actual is destination and final is global_final
    assert (storage[:, [0, -1]] == -777).all()
    if not empty_rank:
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("num_decodes", [1, 3])
@torch.inference_mode()
def test_kcp_decode_reuses_strided_projections_and_preserves_prefill(
    monkeypatch, num_decodes
):
    """Decode consumes local projection views and updates only its cache slots."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.models.glm5next.common import kda
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm.models.glm5next.nvidia.ops import kcp

    torch.manual_seed(923)
    tokens = num_decodes + 5
    inputs, state = make_inputs(tokens, 1, torch.device("cuda"))
    projected = torch.randn(
        tokens, 3 * H * D + H + 2 * D, device="cuda", dtype=torch.bfloat16
    )
    qkv = projected[:, : 3 * H * D]
    beta = projected[:, 3 * H * D : 3 * H * D + H]
    conv = torch.randn(tokens + 1, 3 * H * D, 3, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(3 * H * D, 4, device="cuda") * 0.1
    entries = torch.arange(num_decodes, device="cuda") * 2
    state_indices = inputs["ssm_state_indices"]
    selected_slots = state_indices[entries]
    expected_conv, expected_state = conv.clone(), state.clone()
    reference_qkv = causal_conv1d_update(
        qkv[:num_decodes].clone(memory_format=torch.contiguous_format),
        expected_conv,
        weights,
        None,
        activation="silu",
        conv_state_indices=selected_slots,
    )
    reference_inputs = dict(inputs)
    reference_inputs.update(
        **dict(
            zip(
                ("q", "k", "v"),
                (
                    x.reshape(1, num_decodes, H, D)
                    for x in reference_qkv.split(H * D, -1)
                ),
            )
        ),
        g=inputs["g"][:, :num_decodes].contiguous(),
        beta=beta[:num_decodes].unsqueeze(0).contiguous(),
        ssm_state_indices=selected_slots,
        cu_seqlens=inputs["cu_seqlens"][: num_decodes + 1],
    )
    expected = run_kernel(reference_inputs, expected_state)
    output = torch.full((1, tokens, H, D), -777, device="cuda", dtype=torch.bfloat16)
    layer = _kcp_layer(
        monkeypatch,
        list(range(num_decodes, tokens)),
        (num_decodes, tokens),
        entries,
        state_indices,
        prefill_entry=1,
        cache=(conv, state),
        weights=weights,
        heads=H,
        dim=D,
        a_log=inputs["a_log"],
        bias=inputs["g_bias"],
    )

    def prefill(*args, out, **kwargs):
        return out.fill_(-777), state[:1]

    monkeypatch.setattr(kda, "chunk_kda_with_fused_gate", prefill)
    monkeypatch.setattr(kcp, "prepare_kcp_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(kda, "scatter_states", lambda *args, **kwargs: None)
    Glm5NextLinearAttention._forward(layer, qkv, inputs["g"], beta.unsqueeze(0), output)
    torch.testing.assert_close(output[:, :num_decodes], expected, atol=0, rtol=0)
    assert (output[:, num_decodes:] == -777).all()
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
