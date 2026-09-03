# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class _FakeCudaGraphManager:
    def __init__(self, capture_size: int | None):
        self.capture_size = capture_size
        self.calls: list[tuple[int, int, int | None, int, int | None]] = []

    def dispatch(
        self,
        num_reqs: int,
        num_tokens: int,
        uniform_token_count: int | None,
        num_active_loras: int,
        max_query_len: int | None = None,
    ) -> BatchExecutionDescriptor:
        self.calls.append(
            (
                num_reqs,
                num_tokens,
                uniform_token_count,
                num_active_loras,
                max_query_len,
            )
        )
        if self.capture_size is None or num_tokens > self.capture_size:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_tokens,
                num_reqs=num_reqs,
            )
        assert uniform_token_count is not None
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=self.capture_size,
            num_reqs=self.capture_size // uniform_token_count,
            uniform_token_count=uniform_token_count,
        )


def test_dp_sync_normalizes_dummy_request_counts(monkeypatch):
    reduced = torch.tensor(
        [
            [4, 8, 12],
            [CUDAGraphMode.FULL.value] * 3,
            [4, 4, 4],
            [4, 4, 4],
            [1, 2, 3],
            [1, 0, 0],
        ],
        dtype=torch.int32,
    )
    monkeypatch.setattr(
        dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=object())
    )
    monkeypatch.setattr(
        dp_utils.dist, "all_reduce", lambda tensor, group: tensor.copy_(reduced)
    )

    manager = _FakeCudaGraphManager(capture_size=16)
    desired = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=8,
        num_reqs=2,
        uniform_token_count=4,
    )
    batch_desc, dp_sync = dp_utils.sync_cudagraph_and_dp_padding(
        manager,
        desired,
        num_tokens=8,
        num_reqs=2,
        uniform_token_count=4,
        dp_size=3,
        dp_rank=1,
        max_query_len=4,
    )

    assert batch_desc.num_tokens == 16
    assert dp_sync is not None
    assert dp_sync.num_tokens_across_dp.tolist() == [16, 16, 16]
    assert dp_sync.num_reqs_across_dp.tolist() == [4, 2, 3]


def test_uniform_dispatch_derives_same_global_shape_on_every_rank(monkeypatch):
    target_sync = dp_utils.DPSyncState(
        num_tokens_across_dp=torch.full((3,), 32, dtype=torch.int32),
        uniform_token_count=4,
        eager=False,
        num_reqs_across_dp=torch.tensor([2, 6, 4], dtype=torch.int32),
    )

    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("reuse path must not start another collective")

    monkeypatch.setattr(dp_utils, "dispatch_cg_and_sync_dp", unexpected_dispatch)

    for dp_rank, num_reqs in enumerate([2, 6, 4]):
        manager = _FakeCudaGraphManager(capture_size=56)
        batch_desc, decode_sync = dp_utils.dispatch_uniform_cg_and_sync_dp(
            manager,
            num_reqs=num_reqs,
            uniform_token_count=7,
            dp_size=3,
            dp_rank=dp_rank,
            dp_sync=target_sync,
        )

        assert decode_sync is not None
        assert manager.calls == [(6, 42, 7, 0, None)]
        assert batch_desc.cg_mode == CUDAGraphMode.FULL
        assert batch_desc.num_tokens == 56
        assert decode_sync.num_tokens_across_dp.tolist() == [56, 56, 56]
        assert decode_sync.uniform_token_count == 7


def test_uniform_dispatch_accounts_for_target_dummy_padding():
    manager = _FakeCudaGraphManager(capture_size=8)
    target_sync = dp_utils.DPSyncState(
        num_tokens_across_dp=torch.full((3,), 32, dtype=torch.int32),
        uniform_token_count=4,
        eager=False,
        num_reqs_across_dp=torch.tensor([8, 4, 6], dtype=torch.int32),
    )

    _, decode_sync = dp_utils.dispatch_uniform_cg_and_sync_dp(
        manager,
        num_reqs=8,
        uniform_token_count=1,
        dp_size=3,
        dp_rank=0,
        dp_sync=target_sync,
    )

    assert decode_sync is not None
    assert manager.calls == [(8, 8, 1, 0, None)]
    assert decode_sync.num_reqs_across_dp.tolist() == [8, 4, 6]
    assert decode_sync.num_tokens_across_dp.tolist() == [8, 8, 8]


def test_uniform_dispatch_eager_uses_rank_local_shape_on_every_rank():
    target_sync = dp_utils.DPSyncState(
        num_tokens_across_dp=torch.tensor([8, 24, 16], dtype=torch.int32),
        uniform_token_count=4,
        eager=True,
        num_reqs_across_dp=torch.tensor([2, 6, 4], dtype=torch.int32),
    )

    for dp_rank, num_reqs in enumerate([2, 6, 4]):
        manager = _FakeCudaGraphManager(capture_size=8)
        batch_desc, decode_sync = dp_utils.dispatch_uniform_cg_and_sync_dp(
            manager,
            num_reqs=num_reqs,
            uniform_token_count=1,
            dp_size=3,
            dp_rank=dp_rank,
            dp_sync=target_sync,
        )

        assert decode_sync is not None
        assert batch_desc.cg_mode == CUDAGraphMode.NONE
        assert batch_desc.num_tokens == num_reqs
        assert batch_desc.num_reqs == num_reqs
        assert manager.calls == []
        assert decode_sync.eager
        assert decode_sync.num_tokens_across_dp.tolist() == [2, 6, 4]


def test_uniform_dispatch_requires_target_sync_for_dp(monkeypatch):
    def unexpected_dispatch(*args, **kwargs):
        raise AssertionError("DP path must not start another collective")

    monkeypatch.setattr(dp_utils, "dispatch_cg_and_sync_dp", unexpected_dispatch)

    with pytest.raises(AssertionError, match="requires the target model's DP sync"):
        dp_utils.dispatch_uniform_cg_and_sync_dp(
            _FakeCudaGraphManager(capture_size=16),
            num_reqs=3,
            uniform_token_count=2,
            dp_size=4,
            dp_rank=1,
            dp_sync=None,
        )


def test_uniform_dispatch_without_sync_is_local_for_dp1(monkeypatch):
    def unexpected_collective(*args, **kwargs):
        raise AssertionError("DP=1 must not start a collective")

    monkeypatch.setattr(dp_utils.dist, "all_reduce", unexpected_collective)
    manager = _FakeCudaGraphManager(capture_size=8)

    batch_desc, dp_sync = dp_utils.dispatch_uniform_cg_and_sync_dp(
        manager,
        num_reqs=3,
        uniform_token_count=2,
        dp_size=1,
        dp_rank=0,
        dp_sync=None,
    )

    assert manager.calls == [(3, 6, 2, 0, None)]
    assert batch_desc.cg_mode == CUDAGraphMode.FULL
    assert batch_desc.num_tokens == 8
    assert dp_sync is None
