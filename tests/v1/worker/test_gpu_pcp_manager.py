# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager, PCPTokenLayoutView


def _copy_to_cpu(value, out=None, device=None):
    tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
    if out is not None:
        return out.copy_(tensor)
    return tensor


def test_replicated_decode_piecewise_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
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
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    with pytest.raises(ValueError, match="smaller than the largest rank-local batch"):
        manager._build_batch_layout(
            num_scheduled_tokens=np.ones(3, dtype=np.int32),
            num_computed_tokens=np.full(3, 16, dtype=np.int32),
            is_prefilling=np.zeros(3, dtype=np.bool_),
            query_start_loc_np=np.arange(4, dtype=np.int32),
            padded_num_tokens=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_partition_reuses_gpu_cursor_for_replicated_spec_decode():
    device = torch.device("cuda")
    global_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=4, device=device)
    global_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=4,
        input_buffers=global_buffers,
    )

    # Model an async step after rejection: the CPU scheduler cursor is still
    # optimistic, while the GPU cursor used to build positions/seq_lens has
    # already rolled back to the accepted prefix.
    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    global_batch.num_computed_tokens_np[:] = 20
    global_batch.prefill_len_np[:] = 8
    global_batch.num_computed_prefill_tokens_np[:] = 8
    global_batch.positions.copy_(torch.arange(10, 14, device=device))
    global_batch.seq_lens.fill_(14)

    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=device,
        req_states=SimpleNamespace(),
        max_num_reqs=1,
        max_num_tokens=4,
    )
    local_batch = manager.partition_batch(global_batch)

    # q_len > 1 remains one replicated decode row. Its actual device metadata
    # follows the corrected GPU cursor, not the stale CPU upper bound.
    assert local_batch.num_reqs == 1
    assert local_batch.num_scheduled_tokens.tolist() == [4]
    torch.testing.assert_close(
        local_batch.positions,
        torch.arange(10, 14, device=device),
    )
    torch.testing.assert_close(
        local_batch.seq_lens,
        torch.tensor([14], dtype=torch.int32, device=device),
    )
    assert local_batch.num_computed_tokens_np.tolist() == [20]


def test_restore_hidden_states_appends_zero_graph_padding():
    global_batch = SimpleNamespace(
        num_tokens=5,
        num_tokens_after_padding=8,
    )
    model_batch = SimpleNamespace(num_tokens_after_padding=2)
    restored = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    layout = PCPTokenLayoutView(
        global_batch=global_batch,
        model_batch=model_batch,
        local_gather_idx=torch.arange(2),
        hidden_restore_idx=torch.arange(5),
        group=SimpleNamespace(all_gather=lambda *_args, **_kwargs: restored),
    )

    actual = layout.restore_token_rows(torch.empty(2, 2))

    assert actual.shape == (8, 2)
    torch.testing.assert_close(actual[:5], restored)
    torch.testing.assert_close(actual[5:], torch.zeros(3, 2))


def test_token_layout_localizes_rows_into_caller_owned_buffer():
    global_batch = SimpleNamespace(num_tokens=4, num_tokens_after_padding=4)
    model_batch = SimpleNamespace(
        num_tokens_after_padding=2,
        is_padding=torch.zeros(2, dtype=torch.bool),
    )
    layout = PCPTokenLayoutView(
        global_batch=global_batch,
        model_batch=model_batch,
        local_gather_idx=torch.tensor([2, 0]),
        hidden_restore_idx=torch.arange(4),
        group=SimpleNamespace(),
    )
    source = torch.tensor([[10], [11], [12], [13]])
    out = torch.full((4, 1), -1)

    actual = layout.localize_token_rows(source, out=out)

    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual, torch.tensor([[12], [10]]))
    torch.testing.assert_close(out[2:], torch.full((2, 1), -1))


def test_token_layout_rejects_use_after_new_partition_generation():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    layout = PCPTokenLayoutView(
        global_batch=SimpleNamespace(num_tokens=2, num_tokens_after_padding=2),
        model_batch=SimpleNamespace(num_tokens_after_padding=1),
        local_gather_idx=torch.tensor([0]),
        hidden_restore_idx=torch.arange(2),
        group=SimpleNamespace(all_gather=lambda source, **_: source),
        generation=manager.layout_generation,
        owner=manager,
    )
    manager._layout_generation += 1

    with pytest.raises(RuntimeError, match="token layout is stale"):
        layout.localize_token_rows(torch.ones(2, 1), out=torch.empty(1, 1))
    with pytest.raises(RuntimeError, match="token layout is stale"):
        layout.restore_token_rows(torch.ones(1, 1))


def test_token_layout_for_requires_exact_current_model_batch():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    model_batch = SimpleNamespace(num_tokens_after_padding=1)
    layout = PCPTokenLayoutView(
        global_batch=SimpleNamespace(num_tokens=1, num_tokens_after_padding=1),
        model_batch=model_batch,
        local_gather_idx=torch.tensor([0]),
        hidden_restore_idx=torch.tensor([0]),
        group=SimpleNamespace(),
    )
    manager._current_token_layout = layout

    assert manager.token_layout_for(model_batch) is layout
    assert manager.token_layout_for(SimpleNamespace()) is None


def test_nonpartitioned_pcp_restore_helper_is_identity():
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    batch = SimpleNamespace()
    hidden_states = torch.ones(2, 3)

    actual_hidden_states, actual_batch = (
        pcp_manager_module.maybe_restore_pcp_for_sampling(manager, hidden_states, batch)
    )

    assert actual_hidden_states is hidden_states
    assert actual_batch is batch


def test_mixed_prefill_decode_layout_round_trip_with_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)

    # Five prefill rows are sharded while the two decode/rejection rows are
    # duplicated on both PCP ranks. Rank-local graph batches are padded to 5.
    manager._build_batch_layout(
        num_scheduled_tokens=np.array([5, 2], dtype=np.int32),
        num_computed_tokens=np.array([0, 20], dtype=np.int32),
        is_prefilling=np.array([True, False]),
        query_start_loc_np=np.array([0, 5, 7], dtype=np.int32),
        padded_num_tokens=5,
    )
    assert manager._padded_gather_idx is not None
    assert manager._hidden_restore_idx is not None
    torch.testing.assert_close(
        manager._padded_gather_idx,
        torch.tensor([5, 6, 0, 1, 0, 2, 3, 4, 5, 6]),
    )

    global_batch = SimpleNamespace(num_tokens=7, num_tokens_after_padding=8)
    model_batch = SimpleNamespace(
        num_tokens_after_padding=5,
        is_padding=torch.tensor([False, False, False, False, True]),
    )

    def round_trip(source: torch.Tensor) -> torch.Tensor:
        local_rows = []
        for rank in range(2):
            rank_indices = manager._padded_gather_idx[rank * 5 : (rank + 1) * 5]
            rows = source[rank_indices].clone()
            if rank == 0:
                rows[-1].zero_()
            local_rows.append(rows)
        layout = PCPTokenLayoutView(
            global_batch=global_batch,
            model_batch=model_batch,
            local_gather_idx=manager._padded_gather_idx[:5],
            hidden_restore_idx=manager._hidden_restore_idx,
            group=SimpleNamespace(
                all_gather=lambda *_args, **_kwargs: torch.cat(local_rows)
            ),
        )
        localized = layout.localize_token_rows(
            source,
            out=torch.empty((5, *source.shape[1:]), dtype=source.dtype),
        )
        torch.testing.assert_close(localized, local_rows[0])
        return layout.restore_token_rows(localized)

    hidden = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    auxiliary = torch.arange(21, dtype=torch.float32).reshape(7, 3)
    restored_hidden = round_trip(hidden)
    restored_auxiliary = round_trip(auxiliary)

    torch.testing.assert_close(restored_hidden[:7], hidden)
    torch.testing.assert_close(restored_auxiliary[:7], auxiliary)
    torch.testing.assert_close(restored_hidden[7], torch.zeros(2))
    torch.testing.assert_close(restored_auxiliary[7], torch.zeros(3))
