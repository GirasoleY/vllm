# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.execution import DraftExecutionView

pytestmark = pytest.mark.skip_global_cleanup


def test_profile_skip_attention_syncs_dflash_query_batch(monkeypatch):
    num_reqs = 2
    num_target_tokens = 4
    num_query_per_req = 3
    num_query_tokens = num_reqs * num_query_per_req
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_target_tokens,
        seq_lens_cpu_upper_bound=torch.tensor([2, 2], dtype=torch.int32),
    )
    target_sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([num_target_tokens, num_target_tokens]),
        uniform_token_count=None,
        eager=True,
    )
    execution_view = DraftExecutionView(  # type: ignore[arg-type]
        global_batch=input_batch,
        model_batch=input_batch,
        last_hidden_states=torch.zeros(num_target_tokens, 2),
        aux_hidden_states=None,
        attn_metadata=None,
        slot_mappings=None,
        dp_sync=target_sync,
    )

    fresh_query_counts = torch.tensor([num_query_tokens, num_query_per_req])
    query_sync = DPSyncState(
        num_tokens_across_dp=fresh_query_counts,
        uniform_token_count=num_query_per_req,
        eager=True,
    )
    dispatch = Mock(
        return_value=(
            BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=num_query_tokens,
                num_reqs=num_reqs,
            ),
            query_sync,
        )
    )
    monkeypatch.setattr(dflash_module, "dispatch_cg_and_sync_dp", dispatch)

    speculator = object.__new__(DFlashSpeculator)
    speculator.num_query_per_req = num_query_per_req
    speculator.max_model_len = 32
    speculator.hidden_states = torch.empty(num_target_tokens, 2)
    speculator.context_positions = torch.zeros(num_target_tokens, dtype=torch.int64)
    speculator.draft_tokens = torch.zeros(num_reqs, 2, dtype=torch.int64)
    speculator.query_cudagraph_manager = None
    speculator.dp_size = 2
    speculator.dp_rank = 0
    speculator.model = SimpleNamespace(precompute_and_store_context_kv=Mock())
    speculator._prepare_eplb_forward = Mock()
    speculator._generate_draft = Mock()

    speculator.propose(
        execution_view,
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        last_sampled=torch.zeros(num_reqs, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(num_reqs, dtype=torch.int64),
        temperature=torch.zeros(num_reqs),
        seeds=torch.zeros(num_reqs, dtype=torch.int64),
        dummy_run=True,
        skip_attn_for_dummy_run=True,
        is_profile=True,
    )

    dispatch.assert_called_once_with(
        None,
        num_reqs,
        num_query_tokens,
        uniform_token_count=num_query_per_req,
        dp_size=2,
        dp_rank=0,
        need_eager=True,
    )
    speculator._generate_draft.assert_called_once_with(
        num_reqs,
        num_query_tokens,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=fresh_query_counts,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
