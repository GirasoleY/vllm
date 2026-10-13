# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.worker.gpu import cp_utils as gpu_cp_utils
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers, set_dummy_context
from vllm.v1.worker.gpu.pcp_manager import PCPManager


def _copy_to_cpu(value, device=None, dtype=None, out=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def _make_config(cudagraph_mode: CUDAGraphMode):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
            dcp_comm_backend="ag_rs",
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
            enable_prompt_embeds=False,
            multimodal_config=SimpleNamespace(
                is_multimodal_pruning_enabled=lambda: False
            ),
            is_mm_prefix_lm=False,
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


def test_validate_config_accepts_mm_inputs():
    PCPManager.validate_config(
        _make_config(CUDAGraphMode.NONE), supports_mm_inputs=True
    )


@pytest.mark.parametrize(
    "unsupported,match",
    [
        ({"enable_prompt_embeds": True}, "prompt embeddings"),
        (
            {
                "multimodal_config": SimpleNamespace(
                    is_multimodal_pruning_enabled=lambda: True
                )
            },
            "multimodal pruning",
        ),
        ({"is_mm_prefix_lm": True}, "prefix LMs"),
    ],
)
def test_validate_config_rejects_unsupported_mm_inputs(unsupported, match):
    config = _make_config(CUDAGraphMode.NONE)
    config.model_config = SimpleNamespace(
        **{**vars(config.model_config), **unsupported}
    )
    with pytest.raises(NotImplementedError, match=match):
        PCPManager.validate_config(config, supports_mm_inputs=True)
    # Text-only models do not take these inputs through the MM path.
    PCPManager.validate_config(config, supports_mm_inputs=False)


def test_validate_config_rejects_full_graph_for_prefills():
    with pytest.raises(NotImplementedError, match="decode-only routines"):
        PCPManager.validate_config(
            _make_config(CUDAGraphMode.FULL), supports_mm_inputs=False
        )


@pytest.mark.parametrize("breakable", [True, False])
@pytest.mark.parametrize(
    "cudagraph_mode",
    [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
    ],
)
def test_validate_config_sparse_mla_piecewise_graphs_need_breakable(
    monkeypatch, breakable, cudagraph_mode
):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1" if breakable else "0")
    config = _make_config(cudagraph_mode)
    config.model_config.hf_text_config = SimpleNamespace(index_topk=2048)
    if breakable or not cudagraph_mode.has_piecewise_cudagraphs():
        PCPManager.validate_config(config, supports_mm_inputs=False)
        return
    with pytest.raises(NotImplementedError, match="breakable graphs"):
        PCPManager.validate_config(config, supports_mm_inputs=False)


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    segments_by_rank, per_rank_num_tokens = manager._build_batch_layout(
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_computed_tokens=np.full(3, 16, dtype=np.int32),
        is_prefilling=np.zeros(3, dtype=np.bool_),
        query_start_loc_np=np.arange(4, dtype=np.int32),
        padded_num_tokens=4,
    )

    assert per_rank_num_tokens == [3, 3]
    request_indices = [
        [segment.global_batch_req_idx for segment in rank] for rank in segments_by_rank
    ]
    assert request_indices == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(manager._hidden_restore_idx, torch.tensor([0, 1, 2]))
    assert torch.equal(
        manager._padded_gather_idx,
        torch.tensor([0, 1, 2, 0, 0, 1, 2, 0]),
    )
    assert torch.equal(
        manager._gathered_kv_write_mask,
        torch.tensor([True, True, True, False, False, False, False, False]),
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
        (4, [2, 9], [False, True], 5),
    ],
)
def test_num_tokens_for_dispatch_uses_largest_pcp_rank(
    pcp_world_size, num_scheduled_tokens, is_prefilling, expected
):
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=0,
        device=torch.device("cpu"),
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
        dcp_world_size=1,
    )
    monkeypatch.setattr(pcp_manager_module, "async_tensor_h2d", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )


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

    dummy_batch = InputBatch.make_dummy(
        num_reqs=4,
        num_tokens=4,
        input_buffers=InputBuffers(4, 8, torch.device("cpu")),
        max_query_len=1,
    )
    dummy_batch.input_ids.copy_(torch.arange(4))
    dummy_batch.positions.fill_(7)
    input_batch = manager.prepare_inputs_to_capture(dummy_batch)

    assert input_batch is not dummy_batch
    assert input_batch.req_ids == dummy_batch.req_ids
    for name in ("input_ids", "positions", "is_padding", "query_start_loc", "seq_lens"):
        actual = getattr(input_batch, name)
        torch.testing.assert_close(actual, getattr(dummy_batch, name))
        assert actual.data_ptr() == getattr(manager.input_buffers, name).data_ptr()


def test_dummy_context_updates_pcp_local_block_tables():
    global_block_table = torch.full((4, 4), -1, dtype=torch.int32)
    manager, block_tables = _make_capture_manager(global_block_table)
    input_batch = InputBatch.make_dummy(
        num_reqs=2,
        num_tokens=2,
        input_buffers=manager.input_buffers,
        max_query_len=1,
    )
    input_batch = manager.prepare_inputs_to_capture(input_batch)
    local_block_tables = manager.get_dummy_block_tables(input_batch.num_reqs)

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


def _rank_rows(
    pcp_rank: int,
    pcp_world_size: int,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    is_prefilling: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One PCP+DCP rank's rows as (global request, extent)."""
    manager = PCPManager(
        pcp_world_size=pcp_world_size,
        pcp_rank=pcp_rank,
        device=torch.device("cpu"),
        dcp_world_size=2,
    )
    query_start_loc_np = np.concatenate([[0], np.cumsum(num_scheduled_tokens)]).astype(
        np.int32
    )
    segments = manager._get_rank_segments(
        pcp_rank, num_scheduled_tokens, is_prefilling, query_start_loc_np
    )
    rows = np.array([segment.global_batch_req_idx for segment in segments])
    extents = (num_computed_tokens + num_scheduled_tokens)[rows]
    return rows, extents


@pytest.mark.parametrize(
    ("num_scheduled_tokens", "num_computed_tokens"),
    [
        ([32, 32], [0, 0]),
        ([32, 32, 32, 32], [0, 0, 0, 0]),
        ([32, 32], [100, 0]),  # one continued, one fresh
        ([64, 32, 48], [0, 512, 0]),  # ragged lengths and mixed contexts
        # A replicated prefill BEHIND a split one. Sorting replicated rows first
        # put request 1 ahead of request 0.
        ([64, 3], [0, 0]),
        ([64, 3, 64], [0, 0, 0]),
    ],
)
@pytest.mark.parametrize("pcp_world_size", [2, 4])
def test_published_row_order_is_identical_on_every_pcp_rank(
    num_scheduled_tokens, num_computed_tokens, pcp_world_size
):
    """Every rank must map its rows to the same global requests, in the same order."""
    num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens = np.array(num_computed_tokens, dtype=np.int32)
    is_prefilling = np.ones(len(num_scheduled_tokens), dtype=np.bool_)

    orders = [
        _rank_rows(
            rank,
            pcp_world_size,
            num_scheduled_tokens,
            num_computed_tokens,
            is_prefilling,
        )[0]
        for rank in range(pcp_world_size)
    ]
    for rank, order in enumerate(orders[1:], start=1):
        assert np.array_equal(orders[0], order), (
            f"rank {rank} rows map to {order.tolist()}, rank 0 to {orders[0].tolist()}"
        )
    # Grouped by request and ascending: what the indexer plan indexes by.
    assert np.all(np.diff(orders[0]) >= 0)


def test_published_row_order_puts_every_decode_before_every_prefill():
    """split_decodes_and_prefills takes the FIRST prefilling row as the boundary."""
    # req 0 is a continued prefill, req 1 is a decode.
    req_idx, _ = _rank_rows(
        pcp_rank=0,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([32, 1], dtype=np.int32),
        num_computed_tokens=np.array([100, 20], dtype=np.int32),
        is_prefilling=np.array([True, False], dtype=np.bool_),
    )
    # The decode (request 1) must be row 0.
    assert req_idx[0] == 1


def test_split_prefill_rows_repeat_the_request_and_its_full_extent():
    """Equal adjacent request indices are what let a backend share a KV region."""
    req_idx, extents = _rank_rows(
        pcp_rank=1,
        pcp_world_size=2,
        num_scheduled_tokens=np.array([64, 3], dtype=np.int32),
        num_computed_tokens=np.array([100, 0], dtype=np.int32),
        is_prefilling=np.ones(2, dtype=np.bool_),
    )
    assert req_idx.tolist() == [0, 0, 1]
    assert extents.tolist() == [164, 164, 3]


@pytest.mark.parametrize("query_len", [1, 2, 3, 5, 6, 9])
def test_dcp_replicates_prefills_too_short_to_split(query_len):
    """A prefill that cannot fill 2*pcp chunks is replicated, not split."""
    pcp_world_size = 2
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    rows_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=2,
        )
        assert manager.replicated_requests(num_scheduled_tokens, is_prefilling)[0]
        rows = list(
            manager._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
        )
        assert rows == [(0, 0, query_len)]
        rows_per_rank.append(rows)
    assert rows_per_rank[0] == rows_per_rank[1]


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize(
    "query_len", [16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 1000, 4097]
)
def test_pcp_first_chunk_row_is_never_short(pcp_world_size, query_len):
    num_scheduled_tokens = np.array([query_len], dtype=np.int32)
    is_prefilling = np.ones(1, dtype=np.bool_)

    charged_per_rank = []
    for rank in range(pcp_world_size):
        manager = PCPManager(
            pcp_world_size=pcp_world_size,
            pcp_rank=rank,
            device=torch.device("cpu"),
            dcp_world_size=pcp_world_size,
        )
        chunk_lens = [
            chunk_len
            for _, _, chunk_len in manager._iter_rank_chunks(
                rank, num_scheduled_tokens, is_prefilling
            )
        ]
        assert chunk_lens, f"rank {rank} got no rows for {query_len=}"
        assert chunk_lens[0] == max(chunk_lens), (
            f"rank {rank} emitted a short first chunk for {query_len=}: {chunk_lens}"
        )
        charged_per_rank.append(len(chunk_lens) * chunk_lens[0])

    assert len(set(charged_per_rank)) == 1, (
        f"ranks would chunk differently for {query_len=}: {charged_per_rank}"
    )


def _max_num_tokens_per_rank(config) -> int:
    return VllmConfig.max_num_tokens_per_rank.fget(config)


def _make_bound_config(
    pcp: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    num_spec: int | None,
    *,
    dcp: int = 1,
    dp: int = 1,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    method: str = "mtp",
    parallel_drafting: bool = False,
):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
            data_parallel_size=dp,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
        uniform_decode_query_len=1 + (num_spec or 0),
        speculative_config=None
        if num_spec is None
        else SimpleNamespace(
            method=method,
            num_speculative_tokens=num_spec,
            parallel_drafting=parallel_drafting,
        ),
    )


@pytest.mark.parametrize("pcp_world_size", [2, 4, 8])
@pytest.mark.parametrize("num_spec", [None, 3])
def test_per_rank_token_bound_covers_partitioned_batches(pcp_world_size, num_spec):
    """No rank's share of a schedulable batch exceeds the per-rank bound."""
    max_num_batched_tokens, max_num_seqs = 8192, 32
    decode_query_len = 1 + (num_spec or 0)
    bound = _max_num_tokens_per_rank(
        _make_bound_config(
            pcp_world_size, max_num_batched_tokens, max_num_seqs, num_spec
        )
    )
    assert bound < max_num_batched_tokens
    managers = [
        PCPManager(
            pcp_world_size=pcp_world_size, pcp_rank=r, device=torch.device("cpu")
        )
        for r in range(pcp_world_size)
    ]
    rng = np.random.default_rng(0)
    for trial in range(400):
        num_decodes = int(rng.integers(0, max_num_seqs + 1))
        if trial % 4 == 0:
            num_decodes = max_num_seqs  # all replicated decode queries
        budget = max_num_batched_tokens - num_decodes * decode_query_len
        query_lens = [decode_query_len] * num_decodes
        while len(query_lens) < max_num_seqs and budget > 0:
            # Short prefills maximize the per-chunk rounding overhead.
            length = int(
                rng.integers(1, 3 * pcp_world_size if trial % 2 else budget + 1)
            )
            length = min(length, budget)
            query_lens.append(length)
            budget -= length
        num_scheduled_tokens = np.array(query_lens, dtype=np.int32)
        is_prefilling = np.arange(len(query_lens)) >= num_decodes
        for rank, manager in enumerate(managers):
            local = sum(
                chunk_len
                for _, _, chunk_len in manager._iter_rank_chunks(
                    rank, num_scheduled_tokens, is_prefilling
                )
            )
            assert local <= bound, (local, bound, query_lens)


@pytest.mark.parametrize(
    "overrides",
    [
        {"dcp": 2},
        {"dp": 2},
        {"method": "dspark"},
        {"parallel_drafting": True},
    ],
)
def test_per_rank_token_bound_falls_back_outside_proven_configs(overrides):
    config = _make_bound_config(4, 32768, 32, 3, **overrides)
    assert _max_num_tokens_per_rank(config) == 32768


def test_per_rank_token_bound_holds_with_graphs():
    config = _make_bound_config(4, 32768, 32, 3, cudagraph_mode=CUDAGraphMode.PIECEWISE)
    assert _max_num_tokens_per_rank(config) == 32768 // 4 + 2 * 32 + 32 * 4


def test_per_rank_token_bound_is_noop_without_pcp():
    assert _max_num_tokens_per_rank(_make_bound_config(1, 16384, 32, 3)) == 16384


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

    local_batch = manager.partition_batch(
        global_batch,
        padded_num_tokens=4,
        padded_num_reqs=4,
    )

    assert local_batch.dcp_local_seq_lens is None
    assert local_batch.seq_lens.tolist() == [17, 25, 0, 0]
    assert local_batch.dcp_local_seq_lens_cpu_upper_bound is not None
    expected_dcp_upper_bound = get_dcp_local_seq_lens(
        torch.tensor([17, 25], dtype=torch.int32), 2, 0, 1
    )
    assert torch.equal(
        local_batch.dcp_local_seq_lens_cpu_upper_bound,
        torch.cat([expected_dcp_upper_bound, torch.zeros(2, dtype=torch.int32)]),
    )

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
        torch.tensor([17, 25, 0, 0], dtype=torch.int32), 2, 0, 1
    )
    assert local_batch.dcp_local_seq_lens is not None
    assert torch.equal(local_batch.dcp_local_seq_lens.cpu(), expected)


def _make_global_mixed_batch(
    query_lens: list[int],
    num_computed_tokens: list[int],
    prefill_lens: list[int],
    buffers: InputBuffers,
    device: torch.device,
) -> InputBatch:
    """A global batch of prefills and decodes as `prepare_inputs` would build it."""
    num_reqs = len(query_lens)
    query_lens_np = np.asarray(query_lens, dtype=np.int32)
    num_computed_np = np.asarray(num_computed_tokens, dtype=np.int32)
    num_tokens = int(query_lens_np.sum())
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(query_lens_np, out=query_start_loc_np[1:])
    seq_lens_np = num_computed_np + query_lens_np
    positions_np = np.concatenate(
        [np.arange(c, c + q) for c, q in zip(num_computed_np, query_lens_np)]
    )

    base = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    buffers.seq_lens[:num_reqs] = torch.from_numpy(seq_lens_np).to(device)
    buffers.positions[:num_tokens] = torch.from_numpy(positions_np).to(device)
    buffers.input_ids[:num_tokens] = torch.arange(num_tokens, device=device)
    buffers.query_start_loc[: num_reqs + 1] = torch.from_numpy(query_start_loc_np).to(
        device
    )
    buffers.is_padding[:num_tokens] = False
    return replace(
        base,
        req_ids=[f"req_{i}" for i in range(num_reqs)],
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device=device),
        idx_mapping_np=np.arange(num_reqs, dtype=np.int32),
        num_scheduled_tokens=query_lens_np,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        num_draft_tokens=0,
        num_draft_tokens_per_req=np.zeros(num_reqs, dtype=np.int32),
        query_start_loc=buffers.query_start_loc[: num_reqs + 1],
        query_start_loc_np=query_start_loc_np,
        seq_lens=buffers.seq_lens[:num_reqs],
        seq_lens_cpu_upper_bound=torch.from_numpy(seq_lens_np),
        dcp_local_seq_lens=None,
        num_computed_tokens_np=num_computed_np,
        prefill_len_np=np.asarray(prefill_lens, dtype=np.int32),
        num_computed_prefill_tokens_np=np.minimum(
            num_computed_np, np.asarray(prefill_lens, dtype=np.int32)
        ),
        is_prefilling_np=num_computed_np < np.asarray(prefill_lens, dtype=np.int32),
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        is_padding=buffers.is_padding[:num_tokens],
        prompt_lens=None,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs GPU kernels")
@pytest.mark.parametrize("pcp_rank", [0, 1])
def test_mrope_positions_follow_local_segments(pcp_rank):
    """Each local token gets the M-RoPE position of its own global token.

    A split prefill gives a rank chunks from the middle of the prompt, so the
    positions must not restart at the request's first scheduled token.
    """
    from vllm.v1.worker.gpu.mm.rope import RopeState
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    device = torch.device("cuda:0")
    max_num_reqs, max_num_tokens, max_model_len = 4, 32, 64
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=pcp_rank,
        device=device,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
    )
    rope_state = RopeState(
        num_dims=3,
        max_num_reqs=max_num_reqs,
        max_num_tokens=max_num_tokens,
        max_model_len=max_model_len,
        device=device,
    )
    # Request 0 prefills a 16-token prompt that starts with an image-like
    # (t, h, w) grid; request 1 decodes with an M-RoPE delta.
    table = torch.stack(
        [torch.arange(max_model_len) + 1000 * (j + 1) for j in range(3)]
    ).to(torch.int32)
    rope_state.prefill_positions.gpu[0:3, :].copy_(table)
    rope_state.prefill_delta.np[:2] = [0, -5]
    rope_state.prefill_delta.copy_to_uva()
    prefill_lens = [16, 10]
    num_computed = [0, 20]
    req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(
            gpu=torch.tensor(prefill_lens + [0, 0], dtype=torch.int32, device=device)
        ),
        num_computed_tokens=SimpleNamespace(
            gpu=torch.tensor(num_computed + [0, 0], dtype=torch.int32, device=device)
        ),
    )
    model_state = SimpleNamespace(rope_state=rope_state)
    model_state.prepare_inputs = lambda batch, states: DefaultModelState.prepare_inputs(
        model_state, batch, states
    )

    global_buffers = InputBuffers(max_num_reqs, max_num_tokens, device)
    global_batch = _make_global_mixed_batch(
        [16, 1], num_computed, prefill_lens, global_buffers, device
    )
    local_batch = manager.partition_batch(
        global_batch, padded_num_tokens=12, padded_num_reqs=max_num_reqs
    )
    positions = manager.prepare_model_state_inputs(
        model_state, local_batch, req_states
    )["positions"]

    num_tokens = local_batch.num_tokens_after_padding
    assert positions.shape == (3, num_tokens)
    assert positions.data_ptr() == rope_state.positions.data_ptr()
    local_1d = local_batch.positions.cpu()
    is_padding = local_batch.is_padding.cpu()
    local_req = torch.repeat_interleave(
        torch.from_numpy(local_batch.idx_mapping_np[: local_batch.num_reqs]).long(),
        torch.from_numpy(
            local_batch.num_scheduled_tokens[: local_batch.num_reqs]
        ).long(),
    )
    expected = torch.zeros(3, num_tokens, dtype=torch.int64)
    for k in range(local_batch.num_tokens):
        pos = int(local_1d[k])
        if local_req[k] == 0:
            expected[:, k] = table[:, pos]
        else:
            expected[:, k] = pos - 5
    assert not is_padding[: local_batch.num_tokens].any()
    assert is_padding[local_batch.num_tokens :].all()
    torch.testing.assert_close(positions.cpu(), expected)
    # Rank 1 receives the middle of the prompt, so its prefill rows must not
    # start at position 0.
    if pcp_rank == 1:
        assert local_1d[: local_batch.num_tokens][local_req == 0].min() > 0
