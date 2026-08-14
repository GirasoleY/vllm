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
    (
        "positions",
        "block_table_ids",
        "block_size",
        "dcp_size",
        "dcp_rank",
        "cp_interleave",
        "kv_cache_block_size",
        "expected_context",
        "expected_query",
    ),
    [
        pytest.param(
            [6, 7, 8],
            [3, 4],
            8,
            2,
            0,
            4,
            8,
            [PAD_SLOT_ID, PAD_SLOT_ID, 28],
            [29, 30, 31, PAD_SLOT_ID],
            id="single-page-rank-0",
        ),
        pytest.param(
            [6, 7, 8],
            [3, 4],
            8,
            2,
            1,
            4,
            8,
            [26, 27, PAD_SLOT_ID],
            [PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID, 28],
            id="single-page-rank-1",
        ),
        pytest.param(
            [10, 11, 12, 23],
            [10, 11, 12, 20, 21, 22, 30, 31, 32],
            4,
            1,
            0,
            1,
            12,
            [50, 51, 80, 91],
            [120, 121, 122, 123],
            id="expanded-page-dcp-1",
        ),
        pytest.param(
            [10, 11, 12, 23],
            [10, 11, 12, 20, 21, 22, 30, 31, 32],
            4,
            2,
            0,
            12,
            12,
            [50, 51, PAD_SLOT_ID, PAD_SLOT_ID],
            [80, 81, 82, 83],
            id="expanded-page-rank-0",
        ),
        pytest.param(
            [10, 11, 12, 23],
            [10, 11, 12, 20, 21, 22, 30, 31, 32],
            4,
            2,
            1,
            12,
            12,
            [PAD_SLOT_ID, PAD_SLOT_ID, 40, 51],
            [PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID, PAD_SLOT_ID],
            id="expanded-page-rank-1",
        ),
    ],
)
def test_prepare_dflash_inputs_maps_dcp_slots(
    positions: list[int],
    block_table_ids: list[int],
    block_size: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
    kv_cache_block_size: int,
    expected_context: list[int],
    expected_query: list[int],
):
    """DSpark maps logical DCP pages onto expanded kernel-block tables."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    device = torch.device("cuda")
    max_num_tokens = 16
    num_query_per_req = 4
    input_buffers = InputBuffers(
        max_num_reqs=1, max_num_tokens=max_num_tokens, device=device
    )
    query_slots = torch.full(
        (max_num_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=device
    )
    context_slots = torch.full_like(query_slots, PAD_SLOT_ID)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_scheduled_tokens=np.array([len(positions)], dtype=np.int32),
        positions=torch.tensor(positions, dtype=torch.int64, device=device),
        query_start_loc=torch.tensor(
            [0, len(positions)], dtype=torch.int32, device=device
        ),
        idx_mapping=torch.tensor([0], dtype=torch.int32, device=device),
    )
    zeros = lambda dtype: torch.zeros(num_query_per_req, dtype=dtype, device=device)

    prepare_dflash_inputs(
        input_buffers,
        query_slots,
        torch.zeros(max_num_tokens, dtype=torch.int64, device=device),
        context_slots,
        zeros(torch.int64),
        zeros(torch.int64),
        zeros(torch.int32),
        torch.zeros(1, dtype=torch.float32, device=device),
        torch.zeros(1, dtype=torch.int64, device=device),
        input_batch,  # type: ignore[arg-type]
        num_sampled=torch.tensor([1], dtype=torch.int32, device=device),
        num_rejected=torch.tensor([0], dtype=torch.int32, device=device),
        last_sampled=torch.tensor([42], dtype=torch.int64, device=device),
        next_prefill_tokens=torch.tensor([0], dtype=torch.int64, device=device),
        input_temperature=torch.tensor([0.0], dtype=torch.float32, device=device),
        input_seeds=torch.tensor([7], dtype=torch.int64, device=device),
        block_table=torch.tensor([block_table_ids], dtype=torch.int32, device=device),
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
        kv_cache_block_size=kv_cache_block_size,
        blocks_per_kv_block=kv_cache_block_size // block_size,
    )

    torch.accelerator.synchronize()
    assert context_slots[: len(positions)].cpu().tolist() == expected_context
    assert query_slots[:num_query_per_req].cpu().tolist() == expected_query
    assert input_buffers.seq_lens[0].item() == positions[-1] + num_query_per_req + 1


def test_set_attn_refreshes_finalized_dcp_interleave(monkeypatch):
    """The draft speculator must not retain its pre-PD constructor value."""
    finalized_interleave = 1536
    vllm_config = MagicMock(spec=VllmConfig)
    speculator = SimpleNamespace(
        vllm_config=vllm_config,
        attn_vllm_config=vllm_config,
        device=torch.device("cpu"),
        draft_attn_layer_names=set(),
        cp_interleave=1,
    )

    attn_cg_support = MagicMock()
    monkeypatch.setattr(
        base_speculator,
        "init_attn_backend",
        lambda *args, **kwargs: ([], attn_cg_support, []),
    )
    block_tables = SimpleNamespace(cp_interleave=finalized_interleave)
    DraftModelSpeculator.set_attn(
        speculator,
        MagicMock(),
        MagicMock(),
        block_tables,  # type: ignore[arg-type]
        MagicMock(),
        [],
    )

    assert speculator.cp_interleave == finalized_interleave
    assert speculator.block_tables is block_tables


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
        torch.accelerator.synchronize()
        return graph_output.cpu().tolist()

    assert replay(0) == [8, 93, 94, 95, PAD_SLOT_ID]
    assert replay(2) == [7, PAD_SLOT_ID, 92, 93, 94]
