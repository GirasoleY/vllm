# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.parametrize("cache_mode", ["none", "align"])
def test_mamba_group_lookup_without_prefix_caching(monkeypatch, cache_mode):
    def init_base(self, *args):
        self.max_num_reqs = 2
        self.device = torch.device("cpu")

    monkeypatch.setattr(mamba_hybrid.DefaultModelState, "__init__", init_base)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode=cache_mode,
            use_kda_recoverssm=False,
        )
    )
    state = MambaHybridModelState(
        config, torch.nn.Identity(), None, torch.device("cpu")
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((4, 4),),
        dtypes=(torch.float32,),
        mamba_cache_mode=cache_mode,
    )
    cache = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["linear"], spec)],
    )
    assert state._get_mamba_group_info(cache) == ([0], spec)


@pytest.mark.parametrize("mode", ["tp", "pcp", "dummy", "capture"])
@pytest.mark.parametrize("spec_tokens", [0, 2])
@pytest.mark.parametrize("align", [False, True])
def test_hybrid_metadata_preserves_request_order(monkeypatch, mode, spec_tokens, align):
    """KDA state and acceptance use global requests, MLA uses local segments."""

    def batch(lengths, indices):
        lengths = np.array(lengths, dtype=np.int32)
        starts = np.r_[np.int32(0), lengths.cumsum(dtype=np.int32)]
        return SimpleNamespace(
            num_reqs=len(lengths),
            num_reqs_after_padding=len(lengths),
            num_tokens=int(starts[-1]),
            num_tokens_after_padding=int(starts[-1]),
            query_start_loc=torch.from_numpy(starts),
            query_start_loc_np=starts,
            num_scheduled_tokens=lengths,
            max_query_len=None,
            seq_lens=torch.from_numpy(lengths + 16),
            seq_lens_cpu_upper_bound=torch.from_numpy(lengths + 16),
            is_prefilling_np=np.array([False] + [True] * (len(lengths) - 1)),
            idx_mapping=torch.tensor(indices),
            idx_mapping_np=np.array(indices),
            num_draft_tokens_per_req=np.array([2] + [0] * (len(lengths) - 1)),
            dcp_local_seq_lens=None,
            dcp_local_seq_lens_cpu_upper_bound=None,
            positions=torch.arange(int(starts[-1])),
            prompt_lens=None,
            fast_prefill=None,
        )

    local = batch([3, 2, 2], [2, 0, 0])
    global_batch = batch([3, 8], [2, 0])
    local_tables = (torch.tensor([[12], [10], [10]]),) * 2
    global_tables = (torch.tensor([[12], [10]]),) * 2
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=spec_tokens)
    state.max_model_len = 8192
    state.supports_mm_inputs = False
    state._align_mode = align
    state.num_accepted_tokens_gpu = torch.tensor([4, 5, 6], dtype=torch.int32)
    plan = Mock()
    build_plan = Mock(return_value=plan)
    state.pcp_manager = (
        None
        if mode == "tp"
        else SimpleNamespace(
            hybrid=True,
            global_batch=None if mode == "dummy" else global_batch,
            global_block_tables=global_tables,
            build_hybrid_plan=build_plan,
        )
    )
    spec = SimpleNamespace(shapes=((12, 3), (2, 4, 4)))
    state._get_mamba_group_info = Mock(return_value=([1], spec))
    monkeypatch.setattr(mamba_hybrid, "is_conv_state_dim_first", lambda: True)
    state._ensure_align_ctx = Mock()
    state._ensure_align_ctx.return_value.compute_aligned_state_indices.return_value = (
        torch.tensor([1, 2]),
    )
    state.recoverssm = Mock()
    mla = Mock()
    kda = Mock(spec=mamba_hybrid.GDNAttentionMetadataBuilder)
    kda.mamba_aligned_state_indices = None
    kda_metadata = mamba_hybrid.GDNAttentionMetadata(
        0, 0, 0, 0, 0, 0, 0, non_spec_state_indices_tensor=torch.tensor([3, 5])
    )
    kda.build.return_value = kda.build_for_cudagraph_capture.return_value = kda_metadata
    groups = [
        [SimpleNamespace(layer_names=[name], get_metadata_builder=Mock(return_value=b))]
        for name, b in [("mla", mla), ("kda", kda)]
    ]
    config = SimpleNamespace(kv_cache_groups=[None, None])
    capture = mode == "capture"

    metadata = state.prepare_attn(
        local,
        CUDAGraphMode.NONE,
        local_tables,
        torch.zeros((2, 14), dtype=torch.int64),
        groups,
        config,
        for_capture=capture,
    )

    assert set(metadata) == {"mla", "kda"}
    expected_kda = global_batch if mode == "pcp" else local
    expected_tables = global_tables if mode == "pcp" else local_tables
    for builder, expected, table in [
        (mla, local, local_tables[0]),
        (kda, expected_kda, expected_tables[1]),
    ]:
        call = builder.build_for_cudagraph_capture if capture else builder.build
        call.assert_called_once()
        common = (
            call.call_args.args[0]
            if capture
            else call.call_args.kwargs["common_attn_metadata"]
        )
        assert common.num_reqs == expected.num_reqs
        assert common.num_actual_tokens == expected.num_tokens
        assert common.positions is expected.positions
        assert common.block_table_tensor is table
        torch.testing.assert_close(common.query_start_loc, expected.query_start_loc)
        torch.testing.assert_close(common.seq_lens, expected.seq_lens)
        torch.testing.assert_close(
            common.is_prefilling, torch.from_numpy(expected.is_prefilling_np)
        )
    if not capture:
        kwargs = kda.build.call_args.kwargs
        if spec_tokens:
            torch.testing.assert_close(
                kwargs["num_accepted_tokens"],
                state.num_accepted_tokens_gpu[expected_kda.idx_mapping],
            )
            assert kwargs["num_decode_draft_tokens_cpu"].tolist() == (
                [2] + [-1] * (expected_kda.num_reqs - 1)
            )
        else:
            assert kwargs["num_accepted_tokens"] is None
            assert kwargs["num_decode_draft_tokens_cpu"] is None
    if align:
        state._ensure_align_ctx.assert_called_once_with(config, [1], expected_tables)
        state._ensure_align_ctx.return_value.compute_aligned_state_indices.assert_called_once_with(
            expected_kda.seq_lens, expected_kda.num_reqs
        )
    state.recoverssm.record_step.assert_called_once()
    if mode == "pcp":
        build_plan.assert_called_once_with(3)
        plan.with_state_indices.assert_called_once_with(
            kda_metadata.non_spec_state_indices_tensor
        )
        assert metadata["kda"].cp_plan is plan.with_state_indices.return_value
    else:
        build_plan.assert_not_called()
        assert metadata["kda"].cp_plan is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
