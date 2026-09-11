# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode, ParallelConfig
from vllm.v1.attention.backends.utils import PAD_SLOT_ID, get_dcp_local_seq_lens
from vllm.v1.attention.ops import pcp as attention_pcp
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, set_dummy_context
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def _make_config(cudagraph_mode: CUDAGraphMode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=None,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


def _make_capture_manager(block_table: torch.Tensor):
    block_tables = SimpleNamespace(
        input_block_tables=(block_table,),
        num_kv_cache_groups=1,
        kernel_block_sizes=(2,),
        blocks_per_kv_block=(1,),
    )
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
        block_tables=block_tables,
    )
    return manager, block_tables


@pytest.mark.parametrize(
    "cudagraph_mode",
    [CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_validate_config_accepts_decode_only_full_graphs(cudagraph_mode):
    PCPManager.validate_config(_make_config(cudagraph_mode), supports_mm_inputs=False)


def test_validate_config_rejects_full_graph_for_prefills():
    with pytest.raises(NotImplementedError, match="decode-only routines"):
        PCPManager.validate_config(
            _make_config(CUDAGraphMode.FULL), supports_mm_inputs=False
        )


def test_sharded_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [2, 1]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 2], [1]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 4, 1]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 2, 0, 0, 1, 0, 0, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, False, False, True, False, False, False]),
    )


def test_input_buffers_are_exposed_for_cudagraph_capture():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_num_tokens=8,
    )

    assert manager.input_buffers is manager._input_buffers
    assert manager.input_buffers.input_ids.shape == (8,)
    assert manager.input_buffers.positions.shape == (8,)
    assert manager.input_buffers.is_padding.shape == (8,)


@pytest.mark.parametrize(
    ("pcp_world_size", "num_scheduled_tokens", "is_prefilling", "expected"),
    [
        (2, [8], [True], 4),
        (2, [7], [True], 4),
        (2, [3], [False], 3),
        (2, [3, 8], [False, True], 7),
        (4, [2, 9], [False, True], 4),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
    )

    actual = manager.get_num_tokens_for_dispatch(
        np.asarray(num_scheduled_tokens, dtype=np.int32),
        np.asarray(is_prefilling, dtype=np.bool_),
    )

    assert actual == expected


def test_graph_padding_cannot_be_smaller_than_largest_pcp_rank(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=1,
        )


def _make_global_decode_batch(
    num_computed_tokens: list[int], buffers: InputBuffers, device: torch.device
) -> InputBatch:
    """A replicate-decode global batch as `prepare_inputs` would build it."""
    num_reqs = len(num_computed_tokens)
    num_tokens = num_reqs
    seq_lens_np = np.asarray(num_computed_tokens, dtype=np.int32) + 1

    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    buffers.seq_lens[:num_reqs] = torch.from_numpy(seq_lens_np).to(device)
    buffers.positions[:num_reqs] = torch.tensor(num_computed_tokens, device=device)
    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32)
    buffers.query_start_loc[: num_reqs + 1] = torch.from_numpy(query_start_loc_np).to(
        device
    )

    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=np.zeros(num_reqs, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=np.asarray(num_computed_tokens, dtype=np.int32),
        prefill_len_np=np.zeros(num_reqs, dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(num_reqs, dtype=np.int32),
        is_prefilling_np=np.zeros(num_reqs, dtype=np.bool_),
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        is_padding=buffers.is_padding[:num_tokens],
        prompt_lens=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
def test_partition_defers_dcp_metadata_to_post_partition_batch():
    """DCP-local lengths must derive from the partitioned batch, not the
    global one: the partition replaces seq_lens, so a pre-partition value is
    stale. partition_batch therefore returns None, and the runtime populates
    the field afterwards from the PCP-owned buffers.
    """
    device = torch.device("cuda:0")
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        max_num_reqs=4,
        max_num_tokens=8,
        dcp_world_size=2,
        dcp_rank=0,
    )

    global_buffers = InputBuffers(4, 8, device)
    global_batch = _make_global_decode_batch([16, 24], global_buffers, device)
    # A leftover from an earlier DCP batch must not survive the partition.
    global_batch.dcp_local_seq_lens = global_buffers.dcp_local_seq_lens[:2]
    global_batch.dcp_local_seq_lens.fill_(-1)

    local_batch = manager.partition_batch(global_batch, padded_num_tokens=2)

    assert local_batch.dcp_local_seq_lens is None
    assert local_batch.seq_lens.tolist() == [17, 25]

    # What execute_model does next: derive DCP metadata from the final batch
    # on the PCP-owned buffers.
    local_batch.dcp_local_seq_lens = gpu_cp_utils.maybe_prepare_dcp_local_seq_lens(
        manager.input_buffers.dcp_local_seq_lens,
        local_batch.seq_lens,
        local_batch.num_reqs,
        dcp_size=2,
        dcp_rank=0,
        cp_interleave=1,
        num_reqs_padded=local_batch.num_reqs_after_padding,
    )
    expected = get_dcp_local_seq_lens(
        torch.tensor([17, 25], dtype=torch.int32), 2, 0, 1
    )
    assert local_batch.dcp_local_seq_lens is not None
    assert torch.equal(local_batch.dcp_local_seq_lens.cpu(), expected)


def _rank_request_ids(
    manager: PCPManager,
    rank: int,
    req_ids: list[str],
    *,
    is_prefilling: np.ndarray | None = None,
) -> list[str]:
    num_reqs = len(req_ids)
    if is_prefilling is None:
        is_prefilling = np.zeros(num_reqs, dtype=np.bool_)
    segments = manager._get_rank_segments(
        rank=rank,
        num_scheduled_tokens=np.ones(num_reqs, dtype=np.int32),
        num_computed_tokens=np.full(num_reqs, 16, dtype=np.int32),
        is_prefilling=is_prefilling,
        query_start_loc_np=np.arange(num_reqs + 1, dtype=np.int32),
    )
    return [req_ids[segment.global_batch_req_idx] for segment in segments]


def test_pcp_only_decode_requests_are_round_robin_balanced_each_step():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    req_ids = [f"request-{idx}" for idx in range(18)]

    owners: dict[str, int] = {}
    for rank in range(manager.pcp_world_size):
        for req_id in _rank_request_ids(manager, rank, req_ids):
            assert req_id not in owners
            owners[req_id] = rank

    assert owners == {
        req_id: index % manager.pcp_world_size for index, req_id in enumerate(req_ids)
    }
    counts = [list(owners.values()).count(rank) for rank in range(4)]
    assert max(counts) - min(counts) == 1

    reordered_req_ids = req_ids[::2] + req_ids[1::2]
    reordered_owners = {
        req_id: rank
        for rank in range(manager.pcp_world_size)
        for req_id in _rank_request_ids(manager, rank, reordered_req_ids)
    }
    assert reordered_owners == {
        req_id: index % manager.pcp_world_size
        for index, req_id in enumerate(reordered_req_ids)
    }
    assert reordered_owners != owners


def test_decode_requests_remain_replicated_when_dcp_is_enabled():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=False,
        dcp_world_size=2,
    )
    req_ids = ["request-a", "request-b", "request-c"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == req_ids


def test_decode_sharding_allows_ranks_with_no_owned_request():
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    req_ids = ["request-a"]

    assert _rank_request_ids(manager, 0, req_ids) == req_ids
    assert _rank_request_ids(manager, 1, req_ids) == []
    assert _rank_request_ids(manager, 2, req_ids) == []
    assert _rank_request_ids(manager, 3, req_ids) == []


def test_prefill_partitioning_is_preserved_with_sharded_decode():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    segments_by_rank = [
        manager._get_rank_segments(
            rank=rank,
            num_scheduled_tokens=np.array([8, 1, 0, 1], dtype=np.int32),
            num_computed_tokens=np.array([0, 16, 16, 16], dtype=np.int32),
            is_prefilling=np.array([True, False, False, False]),
            query_start_loc_np=np.array([0, 8, 9, 9, 10], dtype=np.int32),
        )
        for rank in range(manager.pcp_world_size)
    ]

    prefill_tokens = sorted(
        token_idx
        for segments in segments_by_rank
        for segment in segments
        if segment.global_batch_req_idx == 0
        for token_idx in range(
            segment.global_batch_slice.start, segment.global_batch_slice.stop
        )
    )
    decode_owners = {
        segment.global_batch_req_idx: rank
        for rank, segments in enumerate(segments_by_rank)
        for segment in segments
        if segment.global_batch_req_idx in (1, 3)
    }

    assert prefill_tokens == list(range(8))
    assert decode_owners == {1: 0, 3: 1}


def test_sharded_decode_layout_selects_owner_kv_for_replication(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        shard_decode_requests=True,
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([1, 1, 1], dtype=np.int32),
        num_computed_tokens=np.array([16, 16, 16], dtype=np.int32),
        is_prefilling=np.array([False, False, False]),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
    )

    gathered_slot_mapping = manager._convert_to_gathered_slot_mappings(
        torch.tensor([[123, 456, 789]], dtype=torch.int64)
    )
    assert torch.equal(
        gathered_slot_mapping,
        torch.tensor([[123, 789, 456, PAD_SLOT_ID]], dtype=torch.int64),
    )
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 2, 1]))

    class FakePCPGroup:
        world_size = 2

        def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
            assert dim == 0
            assert tensor.shape == (2, 1)
            return torch.tensor([[11.0], [33.0], [22.0], [0.0]])

    monkeypatch.setattr(attention_pcp, "get_pcp_group", FakePCPGroup)
    (gathered_kv,), cache_slot_mapping = attention_pcp._gather_prefill_cache_inputs(
        (torch.tensor([[11.0], [33.0]]),),
        gathered_slot_mapping[0],
        num_decode_tokens=0,
        shard_decode_requests=True,
    )

    assert torch.equal(gathered_kv, torch.tensor([[11.0], [33.0], [22.0], [0.0]]))
    assert torch.equal(cache_slot_mapping, torch.tensor([123, 789, 456, PAD_SLOT_ID]))


@pytest.mark.parametrize(
    ("pcp_world_size", "dcp_world_size", "expected"),
    [(1, 1, False), (2, 1, True), (2, 2, False)],
)
def test_parallel_config_manages_decode_sharding(
    pcp_world_size: int, dcp_world_size: int, expected: bool
):
    parallel_config = ParallelConfig(
        prefill_context_parallel_size=pcp_world_size,
        decode_context_parallel_size=dcp_world_size,
    )

    assert parallel_config.pcp_shard_decode_requests is expected


@pytest.mark.parametrize(
    ("cg_mode", "num_reqs", "expected_tokens", "expected_reqs"),
    [
        (CUDAGraphMode.NONE, 4, None, None),
        (CUDAGraphMode.PIECEWISE, None, 8, None),
        (CUDAGraphMode.FULL, 4, 8, 4),
    ],
)
def test_partition_padding_is_derived_from_batch_descriptor(
    cg_mode, num_reqs, expected_tokens, expected_reqs
):
    manager = MagicMock()
    input_batch = MagicMock()
    manager.partition_batch.return_value = input_batch
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=8,
        num_reqs=num_reqs,
    )

    result = pcp_manager_module.maybe_partition_pcp_batch(
        manager,
        input_batch,
        batch_desc,
    )

    assert result is input_batch
    manager.partition_batch.assert_called_once_with(
        input_batch,
        padded_num_tokens=expected_tokens,
        padded_num_reqs=expected_reqs,
    )


def test_capture_uses_pcp_persistent_inputs():
    manager, _ = _make_capture_manager(torch.ones((4, 2), dtype=torch.int32))

    input_batch, _, _ = manager.prepare_inputs_to_capture(
        num_reqs=4,
        num_tokens=4,
        max_query_len=1,
    )

    assert (
        input_batch.input_ids.data_ptr() == manager.input_buffers.input_ids.data_ptr()
    )
    assert (
        input_batch.positions.data_ptr() == manager.input_buffers.positions.data_ptr()
    )
    assert (
        input_batch.is_padding.data_ptr() == manager.input_buffers.is_padding.data_ptr()
    )


def test_dummy_context_updates_pcp_local_block_tables():
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    manager, block_tables = _make_capture_manager(global_block_table)
    input_batch, local_block_tables, _ = manager.prepare_inputs_to_capture(
        num_reqs=2,
        num_tokens=2,
        max_query_len=1,
    )

    set_dummy_context(
        input_batch,
        block_tables,
        context_len=3,
        num_kv_blocks=16,
        max_model_len=16,
        input_block_tables=local_block_tables,
    )

    torch.testing.assert_close(
        local_block_tables[0][:2, :2],
        torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    )
    assert torch.all(global_block_table == -1)
