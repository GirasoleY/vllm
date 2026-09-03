# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm.config import DraftParallelismConfig, ParallelConfig
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
    assert plan.initial.restores_output
    assert plan.initial.attention_metadata_source == DraftAttentionMetadataSource.TARGET


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


def test_only_end_to_end_supported_speculators_advertise_replicated_pcp():
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

    for speculator_cls in (DSparkSpeculator, MTPSpeculator):
        capabilities = speculator_cls.draft_execution_capabilities()
        assert capabilities.supports_replicated_pcp
        assert not capabilities.supports_target_local_initial
        assert not capabilities.supports_target_local_kv_layout

    assert (
        MTPSpeculator.draft_execution_capabilities().default_prefill_context_parallelism
        == DraftParallelismDefault.ONE
    )


@pytest.mark.parametrize("speculator_cls", [DSparkSpeculator, MTPSpeculator])
def test_supported_speculators_default_to_replicated_pcp(speculator_cls):
    plan = DraftExecutionPlan.resolve(
        target_parallel_config=_parallel_config(pcp=2),
        draft_worker_parallel_config=_parallel_config(),
        draft_parallelism=DraftParallelismConfig(),
        capabilities=speculator_cls.draft_execution_capabilities(),
        implementation_name=speculator_cls.__name__,
    )

    assert plan.pcp_mode == DraftPCPMode.REPLICATED
    assert plan.topology.prefill_context_parallel_size == 1


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
