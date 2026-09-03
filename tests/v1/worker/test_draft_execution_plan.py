# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import DraftParallelismConfig, ParallelConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive import (
    speculator as autoregressive_module,
)
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.eagle.speculator import EagleSpeculator
from vllm.v1.worker.gpu.spec_decode.execution import (
    DraftAttentionMetadataPolicy,
    DraftAttentionMetadataSource,
    DraftBatchLayout,
    DraftExecutionCapabilities,
    DraftExecutionPlan,
    DraftExecutionView,
    DraftParallelismDefault,
    DraftPCPMode,
)
from vllm.v1.worker.gpu.spec_decode.extract_hidden_states import (
    ExtractHiddenStatesSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import Gemma4Speculator
from vllm.v1.worker.gpu.spec_decode.mtp import speculator as mtp_module
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)


def _parallel_config(
    *, pcp: int = 1, dcp: int = 1, tp: int | None = None
) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=dcp if tp is None else tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        distributed_executor_backend="mp",
    )


def test_identity_plan_preserves_target_execution_contract():
    target = _parallel_config()
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=target,
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(),
        capabilities=DraftExecutionCapabilities(),
        implementation_name="IdentitySpeculator",
    )

    assert plan.pcp_mode == DraftPCPMode.DISABLED
    assert plan.initial.input_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.model_output_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.result_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.attention_metadata_source == DraftAttentionMetadataSource.TARGET
    assert plan.initial.reuses_target_dp_sync
    assert not plan.continuation.reuses_target_dp_sync
    assert plan.derive_runtime_parallel_config(target) is target


def test_always_draft_policy_is_explicit_in_identity_plan():
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=_parallel_config(),
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(),
        capabilities=DraftExecutionCapabilities(
            initial_attention_metadata_policy=DraftAttentionMetadataPolicy.ALWAYS_DRAFT,
            reuses_target_dp_sync=False,
        ),
        implementation_name="DraftOwnedSpeculator",
    )

    assert plan.initial.attention_metadata_source == DraftAttentionMetadataSource.DRAFT
    assert not plan.initial.reuses_target_dp_sync


def test_plan_rejects_unadvertised_replicated_pcp():
    with pytest.raises(NotImplementedError, match="replicated draft execution"):
        DraftExecutionPlan.resolve(
            target_parallel_config=_parallel_config(pcp=2),
            draft_worker_parallel_config=_parallel_config(),
            draft_parallelism=DraftParallelismConfig(prefill_context_parallel_size=1),
            capabilities=DraftExecutionCapabilities(),
            implementation_name="IdentityOnlySpeculator",
        )


def test_replicated_plan_cannot_reuse_target_local_dp_sync():
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=_parallel_config(pcp=2),
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(prefill_context_parallel_size=1),
        capabilities=DraftExecutionCapabilities(
            reuses_target_dp_sync=True,
            supports_replicated_pcp=True,
        ),
        implementation_name="FutureReplicatedSpeculator",
    )

    assert plan.pcp_mode == DraftPCPMode.REPLICATED
    assert plan.initial.input_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.model_output_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.result_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.attention_metadata_source == DraftAttentionMetadataSource.DRAFT
    assert plan.initial.block_table_layout == DraftBatchLayout.PCP_GLOBAL
    assert not plan.initial.reuses_target_dp_sync
    assert not plan.initial.restores_output
    assert plan.uses_replicated_pcp


def test_plan_requires_explicit_target_local_kv_capability():
    with pytest.raises(NotImplementedError, match="physical KV layout"):
        DraftExecutionPlan.resolve(
            target_parallel_config=_parallel_config(pcp=2),
            draft_worker_parallel_config=_parallel_config(),
            draft_parallelism=DraftParallelismConfig(prefill_context_parallel_size=2),
            capabilities=DraftExecutionCapabilities(
                supports_target_local_initial=True,
            ),
            implementation_name="LogicalOnlySpeculator",
        )


def test_contract_can_represent_validated_target_local_execution():
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=_parallel_config(pcp=2),
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(prefill_context_parallel_size=2),
        capabilities=DraftExecutionCapabilities(
            default_prefill_context_parallelism=DraftParallelismDefault.TARGET,
            supports_target_local_initial=True,
            supports_target_local_kv_layout=True,
        ),
        implementation_name="FutureShardedSpeculator",
    )

    assert plan.pcp_mode == DraftPCPMode.SHARDED
    assert plan.initial.input_layout == DraftBatchLayout.TARGET_PCP_LOCAL
    assert plan.initial.model_output_layout == DraftBatchLayout.TARGET_PCP_LOCAL
    assert plan.initial.result_layout == DraftBatchLayout.PCP_GLOBAL
    assert plan.initial.restores_output
    assert plan.initial.attention_metadata_source == DraftAttentionMetadataSource.TARGET
    assert plan.initial.block_table_layout == DraftBatchLayout.TARGET_PCP_LOCAL
    assert plan.uses_target_local_initial
    assert plan.continuation.input_layout == DraftBatchLayout.PCP_GLOBAL
    assert (
        plan.continuation.attention_metadata_source
        == DraftAttentionMetadataSource.DRAFT
    )
    assert plan.continuation.block_table_layout == DraftBatchLayout.PCP_GLOBAL
    assert not plan.continuation.reuses_target_dp_sync


def test_plan_rejects_unimplemented_dcp_repartitioning():
    with pytest.raises(NotImplementedError, match="DCP topology"):
        DraftExecutionPlan.resolve(
            target_parallel_config=_parallel_config(dcp=2),
            draft_worker_parallel_config=_parallel_config(tp=2),
            draft_parallelism=DraftParallelismConfig(decode_context_parallel_size=1),
            capabilities=DraftExecutionCapabilities(),
            implementation_name="IdentityOnlySpeculator",
        )


def test_plan_rejects_independent_tp_without_capability():
    with pytest.raises(NotImplementedError, match="draft TP size"):
        DraftExecutionPlan.resolve(
            target_parallel_config=_parallel_config(tp=2),
            draft_worker_parallel_config=_parallel_config(tp=1),
            draft_parallelism=DraftParallelismConfig(tensor_parallel_size=1),
            capabilities=DraftExecutionCapabilities(),
            implementation_name="IdentityOnlySpeculator",
        )


def test_effective_worker_tp_rejects_intermediate_size_when_policy_is_omitted():
    with pytest.raises(ValueError, match="Resolved draft worker tensor_parallel_size"):
        DraftExecutionPlan.resolve(
            target_parallel_config=_parallel_config(tp=4),
            draft_worker_parallel_config=_parallel_config(tp=2),
            draft_parallelism=DraftParallelismConfig(),
            capabilities=DraftExecutionCapabilities(supports_independent_tp=True),
            implementation_name="IndependentTPSpeculator",
        )


def test_only_end_to_end_supported_speculators_advertise_pcp_modes():
    unsupported_classes = (
        AutoRegressiveSpeculator,
        DFlashSpeculator,
        DFlash2Speculator,
        EagleSpeculator,
        ExtractHiddenStatesSpeculator,
        Gemma4Speculator,
        MultiModuleMTPSpeculator,
    )

    for speculator_cls in unsupported_classes:
        capabilities = speculator_cls.draft_execution_capabilities()
        assert not capabilities.supports_replicated_pcp
        assert not capabilities.supports_target_local_initial
        assert not capabilities.supports_target_local_kv_layout

    dspark = DSparkSpeculator.draft_execution_capabilities()
    assert dspark.supports_replicated_pcp
    assert not dspark.supports_target_local_initial
    assert not dspark.supports_target_local_kv_layout
    assert dspark.default_prefill_context_parallelism == DraftParallelismDefault.ONE

    mtp = MTPSpeculator.draft_execution_capabilities()
    assert mtp.supports_replicated_pcp
    assert mtp.supports_target_local_initial
    assert mtp.supports_target_local_kv_layout
    assert mtp.default_prefill_context_parallelism == DraftParallelismDefault.TARGET


@pytest.mark.parametrize(
    ("speculator_cls", "expected_mode", "expected_pcp"),
    [
        (DSparkSpeculator, DraftPCPMode.REPLICATED, 1),
        (MTPSpeculator, DraftPCPMode.SHARDED, 2),
    ],
)
def test_supported_speculators_use_implementation_pcp_default(
    speculator_cls, expected_mode, expected_pcp
):
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=_parallel_config(pcp=2),
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(),
        capabilities=speculator_cls.draft_execution_capabilities(),
        implementation_name=speculator_cls.__name__,
    )

    assert plan.pcp_mode == expected_mode
    assert plan.topology.prefill_context_parallel_size == expected_pcp


def test_mtp_can_explicitly_select_replicated_pcp():
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=_parallel_config(pcp=2),
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(prefill_context_parallel_size=1),
        capabilities=MTPSpeculator.draft_execution_capabilities(),
        implementation_name=MTPSpeculator.__name__,
    )

    assert plan.pcp_mode == DraftPCPMode.REPLICATED
    assert plan.topology.prefill_context_parallel_size == 1


@pytest.mark.parametrize(
    ("uses_target_local_initial", "expected"),
    [(False, CUDAGraphMode.PIECEWISE), (True, CUDAGraphMode.NONE)],
)
def test_target_local_initial_resolves_only_initial_phase_to_eager(
    uses_target_local_initial: bool,
    expected: CUDAGraphMode,
):
    speculator = object.__new__(MTPSpeculator)
    speculator.execution_plan = SimpleNamespace(
        uses_target_local_initial=uses_target_local_initial
    )

    assert (
        speculator.resolve_initial_cudagraph_mode(CUDAGraphMode.PIECEWISE) == expected
    )


def test_target_local_initial_does_not_disable_continuation_graphs(monkeypatch):
    modes = []

    class FakeCudaGraphManager:
        def __init__(self, _config, _device, mode, *args, **kwargs):
            modes.append(mode)

    monkeypatch.setattr(
        autoregressive_module,
        "SpeculatorCudaGraphManager",
        FakeCudaGraphManager,
    )
    speculator = object.__new__(MTPSpeculator)
    speculator.execution_plan = SimpleNamespace(uses_target_local_initial=True)
    speculator.vllm_config = object()
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 2

    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_AND_PIECEWISE)

    assert modes == [CUDAGraphMode.NONE, CUDAGraphMode.FULL_DECODE_ONLY]


@pytest.mark.parametrize("target_local", [False, True])
def test_mtp_disables_shared_topk_for_target_local_initial(
    monkeypatch, target_local: bool
):
    draft_model = SimpleNamespace(
        model=SimpleNamespace(
            set_skip_topk=lambda *_: None,
            compact_topk_indices=lambda *_: None,
        )
    )
    monkeypatch.setattr(mtp_module, "load_eagle_model", lambda *_: draft_model)
    speculator = object.__new__(MTPSpeculator)
    speculator.execution_plan = SimpleNamespace(uses_target_local_initial=target_local)
    speculator.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(index_share_for_mtp_iteration=True)
            )
        )
    )

    assert speculator.load_draft_model(object(), set()) is draft_model
    assert speculator.share_mtp_topk_indices is not target_local


def test_draft_owned_implementations_declare_metadata_and_dp_contract():
    dflash = DFlashSpeculator.draft_execution_capabilities()
    dspark = DSparkSpeculator.draft_execution_capabilities()
    multi_module = MultiModuleMTPSpeculator.draft_execution_capabilities()

    assert (
        dflash.initial_attention_metadata_policy
        == DraftAttentionMetadataPolicy.ALWAYS_DRAFT
    )
    assert not dflash.reuses_target_dp_sync
    assert (
        dspark.initial_attention_metadata_policy
        == DraftAttentionMetadataPolicy.ALWAYS_DRAFT
    )
    assert not dspark.reuses_target_dp_sync
    assert (
        multi_module.initial_attention_metadata_policy
        == DraftAttentionMetadataPolicy.ALWAYS_DRAFT
    )


def test_identity_execution_view_preserves_rows_and_resources():
    batch = cast(Any, SimpleNamespace(num_tokens_after_padding=2))
    hidden_states = torch.arange(9).reshape(3, 3)
    metadata = {"layer": object()}
    slots = {"layer": torch.tensor([1, 2])}

    view = DraftExecutionView(
        global_batch=batch,
        model_batch=batch,
        last_hidden_states=hidden_states,
        aux_hidden_states=None,
        attn_metadata=metadata,
        slot_mappings=slots,
        dp_sync=None,
    )

    output = torch.empty_like(hidden_states)
    assert view.to_model_token_rows(hidden_states, out=output) is not output
    assert torch.equal(
        view.to_model_token_rows(hidden_states, out=output), hidden_states[:2]
    )
    assert torch.equal(view.to_global_token_rows(hidden_states), hidden_states[:2])
    assert view.attn_metadata is metadata
    assert view.slot_mappings is slots


def test_non_identity_execution_view_requires_layout_adapter():
    global_batch = cast(Any, SimpleNamespace(num_tokens_after_padding=2))
    model_batch = cast(Any, SimpleNamespace(num_tokens_after_padding=1))

    with pytest.raises(ValueError, match="explicit token-layout transform"):
        DraftExecutionView(
            global_batch=global_batch,
            model_batch=model_batch,
            last_hidden_states=torch.empty(1, 1),
            aux_hidden_states=None,
            attn_metadata=None,
            slot_mappings=None,
            dp_sync=None,
        )
