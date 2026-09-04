# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import pcp_manager as pcp_manager_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager


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


def test_partition_reuses_gpu_cursor_for_replicated_spec_decode(monkeypatch):
    device = torch.device("cpu")
    monkeypatch.setattr(pcp_manager_module, "async_copy_to_gpu", _copy_to_cpu)
    global_buffers = InputBuffers(max_num_reqs=1, max_num_tokens=4, device=device)
    global_batch = InputBatch.make_dummy(1, 4, global_buffers)

    global_batch.num_draft_tokens = 3
    global_batch.num_draft_tokens_per_req = np.array([3], dtype=np.int32)
    global_batch.num_computed_tokens_np[:] = 20
    global_batch.prefill_len_np[:] = 8
    global_batch.num_computed_prefill_tokens_np[:] = 8
    global_batch.input_ids.copy_(torch.tensor([7, 8, 9, 10], device=device))
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

    assert manager.is_partitioned_batch(local_batch)
    assert not manager.is_partitioned_batch(global_batch)
    torch.testing.assert_close(local_batch.input_ids, global_batch.input_ids)
    torch.testing.assert_close(local_batch.positions, global_batch.positions)
    torch.testing.assert_close(
        local_batch.seq_lens,
        torch.tensor([14], dtype=torch.int32, device=device),
    )
    assert local_batch.logits_indices.tolist() == [3]


def test_restore_hidden_states_appends_zero_graph_padding(monkeypatch):
    manager = PCPManager(
        pcp_world_size=4,
        pcp_rank=0,
        device=torch.device("cpu"),
    )
    manager._global_batch = SimpleNamespace(
        num_tokens=5,
        num_tokens_after_padding=8,
    )
    restored = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    manager._hidden_restore_idx = torch.arange(5)
    monkeypatch.setattr(
        pcp_manager_module,
        "get_pcp_group",
        lambda: SimpleNamespace(all_gather=lambda *_args, **_kwargs: restored),
    )

    actual = manager.restore_hidden_states(torch.empty(0))

    assert actual.shape == (8, 2)
    torch.testing.assert_close(actual[:5], restored)
    torch.testing.assert_close(actual[5:], torch.zeros(3, 2))


def test_local_batch_is_scoped_to_its_global_batch():
    manager = object.__new__(PCPManager)
    global_batch = object()
    local_batch = object()
    manager._global_batch = global_batch
    manager._local_batch = local_batch

    assert manager.local_batch_for(global_batch) is local_batch  # type: ignore[arg-type]
    assert manager.local_batch_for(object()) is None  # type: ignore[arg-type]


def test_draft_ids_are_localized_and_padding_is_zeroed():
    manager = object.__new__(PCPManager)
    manager._local_gather_idx = torch.tensor([2, 0, 0])
    local_batch = SimpleNamespace(
        input_ids=torch.full((3,), -1),
        is_padding=torch.tensor([False, False, True]),
    )
    manager._local_batch = local_batch

    localized = manager.localize_input_ids_for_draft(
        torch.tensor([10, 11, 12]),
        local_batch,  # type: ignore[arg-type]
    )

    assert localized.data_ptr() == local_batch.input_ids.data_ptr()
    assert localized.tolist() == [12, 10, 0]


def _make_validation_config(
    *,
    draft_pcp: int,
    method: str = "mtp",
    multi_module_mtp: bool = False,
    cudagraph_mode=CUDAGraphMode.NONE,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=SimpleNamespace(
            method=method,
            enable_adaptive_verification=False,
            draft_parallel_config=SimpleNamespace(
                prefill_context_parallel_size=draft_pcp
            ),
            use_multi_module_mtp=lambda: multi_module_mtp,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (
            _make_validation_config(draft_pcp=2, multi_module_mtp=True),
            "single-module MTP",
        ),
        (
            _make_validation_config(
                draft_pcp=2, cudagraph_mode=CUDAGraphMode.PIECEWISE
            ),
            "does not support CUDA graphs",
        ),
        (
            _make_validation_config(draft_pcp=2, method="dspark"),
            "single-module MTP",
        ),
    ],
)
def test_sharded_draft_validation_rejects_unsupported_config(config, error):
    with pytest.raises(NotImplementedError, match=error):
        PCPManager.validate_config(config, supports_mm_inputs=False)


def test_sharded_draft_validation_accepts_eager_single_module_mtp():
    PCPManager.validate_config(
        _make_validation_config(draft_pcp=2), supports_mm_inputs=False
    )


def test_replicated_draft_keeps_piecewise_graph_support():
    config = _make_validation_config(
        draft_pcp=1, cudagraph_mode=CUDAGraphMode.PIECEWISE
    )

    PCPManager.validate_config(config, supports_mm_inputs=False)
