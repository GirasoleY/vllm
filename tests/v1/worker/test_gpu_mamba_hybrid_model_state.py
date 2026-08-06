# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU regression tests for Mamba-hybrid model-runner state bookkeeping."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")


def make_model_state(values: list[int]) -> MambaHybridModelState:
    state = MambaHybridModelState.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.tensor(
        values, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    return state


@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_scalar_handles_int32_mapping_and_pp_sentinel(
    num_sampled: int,
    expected_value: int,
):
    state = make_model_state([9, 9, 9, 9, 9])
    idx_mapping = torch.tensor([3, -1, 1], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    assert state.num_accepted_tokens_gpu.cpu().tolist() == [
        9,
        expected_value,
        9,
        expected_value,
        9,
    ]


def test_postprocess_tensor_handles_int32_mapping_and_pp_sentinel():
    state = make_model_state([9, 9, 9, 9, 9])
    idx_mapping = torch.tensor([3, -1, 1], dtype=torch.int32, device="cuda")
    num_sampled = torch.tensor([2, 7, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    assert state.num_accepted_tokens_gpu.cpu().tolist() == [9, 1, 9, 2, 9]


def test_postprocess_empty_mapping_is_noop():
    state = make_model_state([9, 9])
    idx_mapping = torch.empty(0, dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, 0)

    assert state.num_accepted_tokens_gpu.cpu().tolist() == [9, 9]
