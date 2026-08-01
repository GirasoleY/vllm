# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.spec_decode import speculator as base_speculator
from vllm.v1.worker.gpu.spec_decode.dflash import cudagraph as dflash_cudagraph
from vllm.v1.worker.gpu.spec_decode.dflash.cudagraph import (
    DFlashCudaGraphManager,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import prepare_dflash_inputs
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def _make_full_decode_config() -> MagicMock:
    compilation_config = CompilationConfig(
        cudagraph_mode="FULL_DECODE_ONLY",
        cudagraph_capture_sizes=[4],
    )
    compilation_config.max_cudagraph_capture_size = 4
    compilation_config.post_init_cudagraph_sizes()

    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.compilation_config = compilation_config
    vllm_config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=1)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.speculative_config = None
    vllm_config.num_speculative_tokens = 0
    return vllm_config


@contextmanager
def _isolated_cuda_graph_capture(device: torch.device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        yield SimpleNamespace(stream=stream)
    torch.cuda.current_stream(device).wait_stream(stream)


@pytest.mark.parametrize(
    ("dcp_rank", "expected_context", "expected_query"),
    [
        (0, [PAD_SLOT_ID, PAD_SLOT_ID, 28], [29, 30, 31, PAD_SLOT_ID]),
        (1, [26, 27, PAD_SLOT_ID], [PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID, 28]),
    ],
)
def test_prepare_dflash_inputs_uses_dcp_local_slots(
    dcp_rank: int,
    expected_context: list[int],
    expected_query: list[int],
):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device("cuda")
    input_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=16, device=device)
    query_slots = torch.full((16,), PAD_SLOT_ID, dtype=torch.int64, device=device)
    context_positions = torch.zeros(16, dtype=torch.int64, device=device)
    context_slots = torch.full((16,), PAD_SLOT_ID, dtype=torch.int64, device=device)
    sample_indices = torch.zeros(4, dtype=torch.int64, device=device)
    sample_pos = torch.zeros(4, dtype=torch.int64, device=device)
    sample_idx_mapping = torch.zeros(4, dtype=torch.int32, device=device)
    temperature = torch.zeros(1, dtype=torch.float32, device=device)
    seeds = torch.zeros(1, dtype=torch.int64, device=device)

    # One target step ending at global position 8. The four DSpark query
    # positions are 9..12. With D=2, I=4, block_size=8, positions 8..11
    # belong to rank 0 and 12..15 to rank 1. Physical block 3 therefore maps
    # to rank-local slots 24..31.
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        positions=torch.tensor([6, 7, 8], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
    )
    prepare_dflash_inputs(
        input_buffers,
        query_slots,
        context_positions,
        context_slots,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        temperature,
        seeds,
        input_batch,  # type: ignore[arg-type]
        num_sampled=torch.tensor([1], dtype=torch.int32, device=device),
        num_rejected=torch.tensor([0], dtype=torch.int32, device=device),
        last_sampled=torch.tensor([42], dtype=torch.int64, device=device),
        next_prefill_tokens=torch.tensor([0], dtype=torch.int64, device=device),
        input_temperature=torch.tensor([0.0], dtype=torch.float32, device=device),
        input_seeds=torch.tensor([7], dtype=torch.int64, device=device),
        block_table=torch.tensor([[3, 4]], dtype=torch.int32, device=device),
        block_size=8,
        parallel_drafting_token_id=99,
        num_query_per_req=4,
        num_speculative_steps=4,
        max_num_reqs=1,
        max_num_tokens=16,
        max_model_len=128,
        sample_from_anchor=True,
        dcp_size=2,
        dcp_rank=dcp_rank,
        cp_interleave=4,
    )

    torch.cuda.synchronize()
    assert context_slots[:3].cpu().tolist() == expected_context
    assert query_slots[:4].cpu().tolist() == expected_query
    assert input_buffers.seq_lens[0].item() == 13


def test_dspark_dcp8_full_cudagraph_replay_sees_rejection_updates(monkeypatch):
    """FULL replay reads rejection-updated DCP lengths and interleaved slots.

    The graph body consumes the same persistent tensors as draft attention.
    Replaying after a two-token rejection moves the query across a DCP
    interleave boundary, so either stale metadata or stale slot mappings
    changes the observed result.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device("cuda")
    max_num_tokens = 16
    num_query_per_req = 4
    dcp_size = 8
    dcp_rank = 1
    cp_interleave = 4
    block_size = 8

    input_buffers = InputBuffers(
        max_num_reqs=1,
        max_num_tokens=max_num_tokens,
        device=device,
    )
    slot_mappings = torch.full(
        (1, max_num_tokens),
        PAD_SLOT_ID,
        dtype=torch.int64,
        device=device,
    )
    input_block_tables = [torch.tensor([[11]], dtype=torch.int32, device=device)]
    block_tables = SimpleNamespace(
        cp_size=dcp_size,
        cp_rank=dcp_rank,
        cp_interleave=cp_interleave,
        slot_mappings=slot_mappings,
        input_block_tables=input_block_tables,
        get_dummy_block_tables=lambda num_reqs: tuple(
            table[:num_reqs].zero_() for table in input_block_tables
        ),
        get_dummy_slot_mappings=lambda num_tokens: slot_mappings[:, :num_tokens].fill_(
            PAD_SLOT_ID
        ),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["draft"])]
    )

    def fake_build_attn_metadata(**kwargs):
        local_seq_lens = kwargs["dcp_local_seq_lens"]
        assert local_seq_lens is not None
        assert local_seq_lens.data_ptr() == input_buffers.dcp_local_seq_lens.data_ptr()
        return {"dcp_local_seq_lens": local_seq_lens}

    monkeypatch.setattr(
        dflash_cudagraph,
        "build_attn_metadata",
        fake_build_attn_metadata,
    )
    monkeypatch.setattr(
        base_speculator,
        "build_attn_metadata",
        fake_build_attn_metadata,
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "graph_capture",
        _isolated_cuda_graph_capture,
    )

    graph_output = torch.empty(5, dtype=torch.int64, device=device)

    def forward_fn(
        num_reqs,
        num_tokens,
        attn_metadata,
        per_layer_slot_mappings,
        _num_tokens_across_dp,
        _cg_mode,
    ):
        assert num_reqs == 1
        assert num_tokens == num_query_per_req
        graph_output[0].copy_(attn_metadata["dcp_local_seq_lens"][0])
        graph_output[1:].copy_(per_layer_slot_mappings["draft"][:num_query_per_req])

    manager = DFlashCudaGraphManager(
        vllm_config=_make_full_decode_config(),
        device=device,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        decode_query_len=num_query_per_req,
    )
    manager.capture(
        forward_fn=forward_fn,
        input_buffers=input_buffers,
        block_tables=block_tables,
        attn_groups=[],
        kv_cache_config=kv_cache_config,
        max_model_len=128,
        causal=False,
    )
    input_block_tables[0].fill_(11)
    desc = manager.dispatch(
        num_reqs=1,
        num_tokens=num_query_per_req,
        uniform_token_count=num_query_per_req,
        num_active_loras=0,
    )
    assert desc.cg_mode == CUDAGraphMode.FULL

    target_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([6], dtype=np.int32),
        positions=torch.arange(31, 37, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
    )
    context_positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
    context_slots = torch.full(
        (max_num_tokens,),
        PAD_SLOT_ID,
        dtype=torch.int64,
        device=device,
    )
    sample_indices = torch.zeros(num_query_per_req, dtype=torch.int64, device=device)
    sample_pos = torch.zeros(num_query_per_req, dtype=torch.int64, device=device)
    sample_idx_mapping = torch.zeros(
        num_query_per_req, dtype=torch.int32, device=device
    )
    temperature = torch.zeros(1, dtype=torch.float32, device=device)
    seeds = torch.zeros(1, dtype=torch.int64, device=device)
    num_rejected = torch.zeros(1, dtype=torch.int32, device=device)

    runtime_speculator = SimpleNamespace(
        arange=torch.arange(2, dtype=torch.int32, device="cpu"),
        attn_groups=[],
        block_tables=block_tables,
        cp_interleave=cp_interleave,
        dcp_rank=dcp_rank,
        dcp_size=dcp_size,
        draft_attn_layer_names={"draft"},
        draft_max_seq_len=128,
        input_buffers=input_buffers,
        kv_cache_config=kv_cache_config,
        max_model_len=128,
        use_dcp=True,
    )

    def replay(num_rejected_tokens: int) -> list[int]:
        num_rejected.fill_(num_rejected_tokens)
        prepare_dflash_inputs(
            input_buffers,
            slot_mappings[0],
            context_positions,
            context_slots,
            sample_indices,
            sample_pos,
            sample_idx_mapping,
            temperature,
            seeds,
            target_batch,  # type: ignore[arg-type]
            num_sampled=torch.tensor([1], dtype=torch.int32, device=device),
            num_rejected=num_rejected,
            last_sampled=torch.tensor([42], dtype=torch.int64, device=device),
            next_prefill_tokens=torch.tensor([0], dtype=torch.int64, device=device),
            input_temperature=torch.tensor([0.0], dtype=torch.float32, device=device),
            input_seeds=torch.tensor([7], dtype=torch.int64, device=device),
            block_table=input_block_tables[0],
            block_size=block_size,
            parallel_drafting_token_id=99,
            num_query_per_req=num_query_per_req,
            num_speculative_steps=num_query_per_req,
            max_num_reqs=1,
            max_num_tokens=max_num_tokens,
            max_model_len=128,
            sample_from_anchor=True,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            cp_interleave=cp_interleave,
        )
        DraftModelSpeculator._build_draft_attn_metadata(
            runtime_speculator,  # type: ignore[arg-type]
            num_reqs=1,
            num_reqs_padded=1,
            num_tokens_padded=num_query_per_req,
            seq_lens_cpu_upper_bound=torch.tensor([37], dtype=torch.int32),
            step=num_query_per_req,
            num_query_per_req=num_query_per_req,
            causal=False,
        )
        manager.run_fullgraph(desc)
        torch.cuda.synchronize()
        return graph_output.cpu().tolist()

    assert replay(0) == [8, 93, 94, 95, PAD_SLOT_ID]
    assert replay(2) == [7, PAD_SLOT_ID, 92, 93, 94]
