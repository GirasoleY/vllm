# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA context parallelism (KCP) under hybrid PCP.

The end-to-end test runs the real layer on every emulated PCP rank of one GPU
and compares outputs and replicated caches with the ordinary unpartitioned
path. Rank threads run one at a time; they switch only inside collectives.
"""

import threading
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only kernels"),
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="KCP CuTe summaries require SM100-family GPUs",
    ),
]

LOWER_BOUND = -5.0


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


class _EmulatedGroup:
    """Serialized rank threads that exchange tensors only in all_gather."""

    def __init__(self, world: int):
        self.world = world
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(world)
        self.parts: list[torch.Tensor | None] = [None] * world
        self.local = threading.local()

    def _wait(self):
        self.lock.release()
        self.barrier.wait()
        self.lock.acquire()

    def all_gather(self, tensor, dim=0):
        self.parts[self.local.rank] = tensor
        self._wait()
        gathered = torch.cat(self.parts, dim)
        self._wait()
        return gathered

    def run(self, fn):
        errors = []

        def body(rank):
            self.local.rank = rank
            with self.lock:
                try:
                    fn(rank)
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                    self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]


def _layer(heads, dim, cache):
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    torch.manual_seed(1)
    channels = heads * dim
    convs = [
        SimpleNamespace(weight=torch.randn(channels, 1, 4, device="cuda") * 0.3)
        for _ in range(3)
    ]
    layer = SimpleNamespace(
        prefix="kda",
        kv_cache=cache,
        q_conv1d=SimpleNamespace(weight=convs[0].weight, bias=None),
        k_conv1d=convs[1],
        v_conv1d=convs[2],
        _merged_conv_weight=None,
        _conv_state_dim_first=True,
        A_log=0.5 * torch.randn(heads, device="cuda"),
        dt_bias=0.1 * torch.randn(channels, device="cuda"),
        local_num_heads=heads,
        head_dim=dim,
        local_projection_size=channels,
        kda_safe_gate=True,
        kda_lower_bound=LOWER_BOUND,
        kda_prefill_backend="triton",
    )
    for name in ("_forward", "_forward_kcp", "_conv_state_and_weights"):
        method = getattr(Glm5NextLinearAttention, name)
        setattr(layer, name, MethodType(getattr(method, "__wrapped__", method), layer))
    return layer


def _reference_metadata(lengths, computed, slots, num_decodes):
    """Ordinary GDN metadata of the unpartitioned batch, decodes first."""
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    qsl = torch.tensor(np.r_[0, np.cumsum(lengths)], dtype=torch.int32)
    nums_dict, batch_ptr, offsets = compute_causal_conv1d_metadata(
        qsl, device=torch.device("cuda")
    )
    return GDNAttentionMetadata(
        num_prefills=len(lengths) - num_decodes,
        num_prefill_tokens=int(sum(lengths[num_decodes:])),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=int(sum(lengths)),
        has_initial_state=torch.tensor(computed, device="cuda") > 0,
        non_spec_query_start_loc=qsl.cuda(),
        non_spec_state_indices_tensor=slots,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offsets,
    )


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "lengths,computed,num_decodes",
    [
        # Two decodes, a short prefill with empty slots, a continued prefill
        # and a fresh prefill spanning several chunks.
        ([1, 1, 5, 150, 70], [8, 3, 0, 64, 0], 2),
        # One short continued prefill: some ranks hold no prefill tokens.
        ([3], [5], 0),
    ],
)
@torch.inference_mode()
def test_kcp_layer_matches_unpartitioned_forward(
    monkeypatch, world, lengths, computed, num_decodes
):
    from vllm.models.glm5next.common import kda
    from vllm.models.glm5next.nvidia.ops import kcp
    from vllm.v1.worker.gpu.pcp_manager import PCPManager

    torch.manual_seed(5)
    heads, dim, halo, padding = 2, 128, 3, 2
    channels = heads * dim
    num_slots = 2 * len(lengths) + 1
    # Distinct, shuffled cache slots; slot 0 is NULL_BLOCK_ID and the spare
    # slots must stay untouched.
    slots = torch.randperm(num_slots - 1, device="cuda")[: len(lengths)] + 1
    conv = torch.randn(num_slots, 3 * channels, halo, device="cuda").bfloat16()
    state = torch.randn(num_slots, heads, dim, dim, device="cuda") * 0.1
    tokens = sum(lengths)
    qkv = torch.randn(tokens, 3 * channels, device="cuda").bfloat16()
    gate = torch.randn(1, tokens, heads, dim, device="cuda").bfloat16()
    beta = torch.randn(1, tokens, heads, device="cuda").bfloat16()
    prefilling = np.arange(len(lengths)) >= num_decodes

    def forward(layer, metadata, qkv, gate, beta, out):
        context = SimpleNamespace(attn_metadata={"kda": metadata})
        monkeypatch.setattr(kda, "get_forward_context", lambda: context)
        layer._forward(qkv, gate, beta, out)

    reference_cache = (conv.clone(), state.clone())
    reference = torch.empty(1, tokens, heads, dim, device="cuda").bfloat16()
    forward(
        _layer(heads, dim, reference_cache),
        _reference_metadata(lengths, computed, slots, num_decodes),
        qkv,
        gate,
        beta,
        reference,
    )

    qsl = np.r_[0, np.cumsum(lengths)].astype(np.int32)
    global_batch = SimpleNamespace(
        num_reqs=len(lengths),
        num_scheduled_tokens=np.asarray(lengths, dtype=np.int32),
        is_prefilling_np=prefilling,
        num_computed_tokens_np=np.asarray(computed, dtype=np.int32),
        query_start_loc_np=qsl,
        num_draft_tokens_per_req=None,
    )
    group = _EmulatedGroup(world)
    monkeypatch.setattr(kda, "get_pcp_group", lambda: group)
    monkeypatch.setattr(kcp, "get_pcp_group", lambda: group)
    contexts: dict[int, SimpleNamespace] = {}
    monkeypatch.setattr(kda, "get_forward_context", lambda: contexts[group.local.rank])
    results = {}

    def run_rank(rank):
        manager = PCPManager(world, rank, torch.device("cuda"))
        manager._global_batch = global_batch
        segments = manager._get_rank_segments(rank, lengths, prefilling, qsl)
        manager._local_segments = tuple(segments)
        plan = manager.build_hybrid_plan(halo)
        index = torch.tensor(
            [i for s in segments for i in range(*s.global_batch_slice.indices(tokens))],
            device="cuda",
            dtype=torch.int64,
        )
        pad = torch.zeros(padding, dtype=torch.int64, device="cuda")
        local = torch.cat((index, pad))
        metadata = _reference_metadata(lengths, computed, slots, num_decodes)
        metadata.cp_plan = plan
        contexts[rank] = SimpleNamespace(attn_metadata={"kda": metadata})
        cache = (conv.clone(), state.clone())
        out = torch.full((1, len(local), heads, dim), 7.0, device="cuda").bfloat16()
        _layer(heads, dim, cache)._forward(
            qkv[local], gate[:, local], beta[:, local], out
        )
        results[rank] = (index, out, cache)

    group.run(run_rank)

    spare = torch.ones(num_slots, dtype=torch.bool, device="cuda")
    spare[slots] = False
    for index, out, (rank_conv, rank_state) in results.values():
        torch.testing.assert_close(
            out[0, : len(index)], reference[0, index], atol=4e-3, rtol=2e-2
        )
        assert (out[0, len(index) :] == 0).all()
        torch.testing.assert_close(rank_conv, reference_cache[0], atol=0, rtol=0)
        # Decodes use the recurrent kernel and short prefills are stitched from
        # BF16 segment summaries, so states differ within output tolerance.
        torch.testing.assert_close(
            rank_state[slots], reference_cache[1][slots], atol=4e-3, rtol=2e-2
        )
        assert torch.equal(rank_state[spare], state[spare])
