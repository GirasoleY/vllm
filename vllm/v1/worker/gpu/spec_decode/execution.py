# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch

from vllm.config import DraftParallelismConfig, ParallelConfig, replace
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch


class DraftBatchLayout(str, Enum):
    """Token/request layout presented to a draft execution phase."""

    PCP_GLOBAL = "pcp_global"
    TARGET_PCP_LOCAL = "target_pcp_local"


class DraftAttentionMetadataSource(str, Enum):
    """Owner of a phase's metadata, slot mappings, and block-table view.

    This does not describe physical KV-cache ownership. Cache allocation,
    per-layer sharing, and lifetime remain model/speculator responsibilities;
    the execution plan describes only the logical view used to address them.
    """

    TARGET = "target"
    DRAFT = "draft"


class DraftParallelismDefault(str, Enum):
    """How an omitted public draft-parallel dimension is resolved."""

    ONE = "one"
    TARGET = "target"


class DraftAttentionMetadataPolicy(str, Enum):
    """How a concrete implementation obtains its seed metadata bundle."""

    REUSE_TARGET_IF_LAYOUT_MATCHES = "reuse_target_if_layout_matches"
    ALWAYS_DRAFT = "always_draft"


class DraftPCPMode(str, Enum):
    """Relationship between target and draft PCP execution."""

    DISABLED = "disabled"
    REPLICATED = "replicated"
    SHARDED = "sharded"


@dataclass(frozen=True)
class DraftPhasePlan:
    """Resolved data contract for one phase of draft execution.

    ``model_output_layout`` describes the raw model result. ``result_layout``
    describes the representation required by the next consumer, making any
    restore transition explicit.
    """

    input_layout: DraftBatchLayout
    model_output_layout: DraftBatchLayout
    result_layout: DraftBatchLayout
    attention_metadata_source: DraftAttentionMetadataSource
    block_table_layout: DraftBatchLayout
    reuses_target_dp_sync: bool

    @property
    def restores_output(self) -> bool:
        return self.model_output_layout != self.result_layout


@dataclass(frozen=True)
class DraftExecutionCapabilities:
    """Capabilities declared by a concrete draft speculator."""

    default_prefill_context_parallelism: DraftParallelismDefault = (
        DraftParallelismDefault.ONE
    )
    default_decode_context_parallelism: DraftParallelismDefault = (
        DraftParallelismDefault.TARGET
    )
    initial_attention_metadata_policy: DraftAttentionMetadataPolicy = (
        DraftAttentionMetadataPolicy.REUSE_TARGET_IF_LAYOUT_MATCHES
    )
    # False for implementations (such as DFlash) that expand the seed batch
    # before their first model dispatch.
    reuses_target_dp_sync: bool = True
    supports_replicated_pcp: bool = False
    supports_target_local_initial: bool = False
    # The concrete model/cache setup guarantees that target-local seed
    # metadata can address its real per-layer KV arrangement and that any
    # continuation transition is valid.
    supports_target_local_kv_layout: bool = False
    supports_independent_tp: bool = False
    supports_dcp_repartitioning: bool = False


@dataclass(frozen=True)
class DraftParallelTopology:
    """Resolved logical TP/PCP/DCP dimensions for integrated drafting.

    Integrated draft models remain on the target worker processes and process
    groups. In particular, replicated draft attention may still use the
    target PCP group for physically sharded expert-parallel execution.
    """

    tensor_parallel_size: int
    prefill_context_parallel_size: int
    decode_context_parallel_size: int

    def apply_to(self, target: ParallelConfig) -> ParallelConfig:
        if (
            self.tensor_parallel_size == target.tensor_parallel_size
            and self.prefill_context_parallel_size
            == target.prefill_context_parallel_size
            and self.decode_context_parallel_size == target.decode_context_parallel_size
        ):
            return target
        return replace(
            target,
            tensor_parallel_size=self.tensor_parallel_size,
            prefill_context_parallel_size=self.prefill_context_parallel_size,
            decode_context_parallel_size=self.decode_context_parallel_size,
        )


@dataclass(frozen=True)
class DraftExecutionPlan:
    """Internal resolution of public draft parallelism policy."""

    topology: DraftParallelTopology
    pcp_mode: DraftPCPMode
    initial: DraftPhasePlan
    continuation: DraftPhasePlan

    @classmethod
    def resolve(
        cls,
        *,
        target_parallel_config: ParallelConfig,
        draft_worker_parallel_config: ParallelConfig,
        draft_parallelism: DraftParallelismConfig,
        capabilities: DraftExecutionCapabilities,
        implementation_name: str,
    ) -> "DraftExecutionPlan":
        target_tp = target_parallel_config.tensor_parallel_size
        target_pcp = target_parallel_config.prefill_context_parallel_size
        target_dcp = target_parallel_config.decode_context_parallel_size
        draft_tp = draft_worker_parallel_config.tensor_parallel_size

        def resolve_size(
            *,
            name: str,
            requested: int | None,
            target: int,
            default: DraftParallelismDefault,
        ) -> int:
            if requested is not None and requested not in (1, target):
                raise ValueError(
                    f"draft_parallel_config.{name}={requested} must be 1 or "
                    f"the target model's {name} ({target})."
                )
            if requested is not None:
                return requested
            if default == DraftParallelismDefault.TARGET:
                return target
            return 1

        requested_tp = draft_parallelism.tensor_parallel_size
        if requested_tp is not None and requested_tp not in (1, target_tp):
            raise ValueError(
                f"draft_parallel_config.tensor_parallel_size={requested_tp} "
                "must be 1 or the target model's tensor_parallel_size "
                f"({target_tp})."
            )
        if draft_tp not in (1, target_tp):
            raise ValueError(
                "Resolved draft worker tensor_parallel_size must be 1 or the "
                f"target model's tensor_parallel_size ({target_tp}); got "
                f"{draft_tp}."
            )
        if requested_tp is not None and requested_tp != draft_tp:
            raise ValueError(
                "Resolved draft worker TP does not match the requested policy: "
                f"worker={draft_tp}, requested={requested_tp}."
            )
        draft_pcp = resolve_size(
            name="prefill_context_parallel_size",
            requested=draft_parallelism.prefill_context_parallel_size,
            target=target_pcp,
            default=capabilities.default_prefill_context_parallelism,
        )
        draft_dcp = resolve_size(
            name="decode_context_parallel_size",
            requested=draft_parallelism.decode_context_parallel_size,
            target=target_dcp,
            default=capabilities.default_decode_context_parallelism,
        )

        if draft_tp != target_tp and not capabilities.supports_independent_tp:
            raise NotImplementedError(
                f"{implementation_name} does not support a draft TP size "
                f"different from the target (draft={draft_tp}, target={target_tp})."
            )
        dcp_topology_changes = draft_dcp != target_dcp or (
            (draft_tp, draft_pcp) != (target_tp, target_pcp)
            and (draft_dcp > 1 or target_dcp > 1)
        )
        if dcp_topology_changes and not capabilities.supports_dcp_repartitioning:
            raise NotImplementedError(
                f"{implementation_name} does not support a draft DCP topology "
                "different from the target "
                f"(draft TP/PCP/DCP={draft_tp}/{draft_pcp}/{draft_dcp}, "
                f"target={target_tp}/{target_pcp}/{target_dcp})."
            )

        if target_pcp > 1 and draft_pcp == 1:
            if not capabilities.supports_replicated_pcp:
                raise NotImplementedError(
                    f"{implementation_name} does not support replicated draft "
                    "execution with a PCP target."
                )
            pcp_mode = DraftPCPMode.REPLICATED
            initial_layout = DraftBatchLayout.PCP_GLOBAL
        elif target_pcp > 1 and draft_pcp == target_pcp:
            if not capabilities.supports_target_local_initial:
                raise NotImplementedError(
                    f"{implementation_name} does not support PCP-sharded draft "
                    "execution. Set draft_parallel_config."
                    "prefill_context_parallel_size=1 to use replicated drafting."
                )
            if not capabilities.supports_target_local_kv_layout:
                raise NotImplementedError(
                    f"{implementation_name} does not validate its physical KV "
                    "layout for a target-PCP-local initial phase."
                )
            pcp_mode = DraftPCPMode.SHARDED
            initial_layout = DraftBatchLayout.TARGET_PCP_LOCAL
        elif target_pcp == draft_pcp == 1:
            pcp_mode = DraftPCPMode.DISABLED
            initial_layout = DraftBatchLayout.PCP_GLOBAL
        else:
            raise ValueError(
                "Draft PCP must be 1 or match the target PCP size; got "
                f"draft={draft_pcp}, target={target_pcp}."
            )

        can_reuse_target_attention_metadata = (
            draft_tp == target_tp
            and draft_pcp == target_pcp
            and draft_dcp == target_dcp
        )
        initial_attention_metadata_source = (
            DraftAttentionMetadataSource.TARGET
            if capabilities.initial_attention_metadata_policy
            == DraftAttentionMetadataPolicy.REUSE_TARGET_IF_LAYOUT_MATCHES
            and can_reuse_target_attention_metadata
            else DraftAttentionMetadataSource.DRAFT
        )
        if (
            initial_layout == DraftBatchLayout.TARGET_PCP_LOCAL
            and initial_attention_metadata_source != DraftAttentionMetadataSource.TARGET
        ):
            raise NotImplementedError(
                f"{implementation_name} requests draft-owned attention metadata "
                "for a target-PCP-local initial phase, but that metadata "
                "transition is not implemented."
            )

        target_initial_layout = (
            DraftBatchLayout.TARGET_PCP_LOCAL
            if target_pcp > 1
            else DraftBatchLayout.PCP_GLOBAL
        )
        initial = DraftPhasePlan(
            input_layout=initial_layout,
            model_output_layout=initial_layout,
            result_layout=DraftBatchLayout.PCP_GLOBAL,
            attention_metadata_source=initial_attention_metadata_source,
            block_table_layout=initial_layout,
            reuses_target_dp_sync=(
                capabilities.reuses_target_dp_sync
                and initial_layout == target_initial_layout
            ),
        )
        continuation = DraftPhasePlan(
            input_layout=DraftBatchLayout.PCP_GLOBAL,
            model_output_layout=DraftBatchLayout.PCP_GLOBAL,
            result_layout=DraftBatchLayout.PCP_GLOBAL,
            attention_metadata_source=DraftAttentionMetadataSource.DRAFT,
            block_table_layout=DraftBatchLayout.PCP_GLOBAL,
            reuses_target_dp_sync=False,
        )
        return cls(
            topology=DraftParallelTopology(
                tensor_parallel_size=draft_tp,
                prefill_context_parallel_size=draft_pcp,
                decode_context_parallel_size=draft_dcp,
            ),
            pcp_mode=pcp_mode,
            initial=initial,
            continuation=continuation,
        )

    @property
    def uses_replicated_pcp(self) -> bool:
        return self.pcp_mode == DraftPCPMode.REPLICATED

    @property
    def uses_target_local_initial(self) -> bool:
        return self.pcp_mode == DraftPCPMode.SHARDED

    def derive_runtime_parallel_config(self, target: ParallelConfig) -> ParallelConfig:
        """Apply resolved logical dimensions to the in-process draft config.

        Integrated draft execution shares the target worker's process, rank,
        process groups, and non-topology runtime settings. Only the three
        public draft-parallel dimensions are replaced; callers must validate
        this exact effective config against the draft model before constructing
        it. This does not rebuild physical groups used independently by an
        operator (for example, expert parallelism over target PCP ranks).
        """
        return self.topology.apply_to(target)


class DraftTokenLayout(Protocol):
    """Explicit transforms for token rows; request rows are not transformable."""

    @property
    def global_batch(self) -> InputBatch: ...

    @property
    def model_batch(self) -> InputBatch: ...

    def localize_token_rows(
        self, source: torch.Tensor, *, out: torch.Tensor
    ) -> torch.Tensor: ...

    def restore_token_rows(self, source: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class DraftExecutionView:
    """Atomic per-step data view consumed by a draft implementation.

    ``global_batch`` owns request order and sampling state. ``model_batch`` and
    the tensor/metadata bundle describe the seed rows consumed by the initial
    draft phase. A token-layout adapter makes the transition between those
    representations explicit when PCP shards that phase. ``dp_sync`` is
    present only when it was produced for this exact ``model_batch`` and the
    implementation declares that it directly reuses the target dispatch;
    shape-changing implementations synchronize their derived batch afresh.
    """

    global_batch: InputBatch
    model_batch: InputBatch
    last_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    attn_metadata: dict[str, Any] | None
    slot_mappings: dict[str, torch.Tensor] | None
    dp_sync: DPSyncState | None
    token_layout: DraftTokenLayout | None = None

    def __post_init__(self) -> None:
        if self.token_layout is None:
            if self.model_batch is not self.global_batch:
                raise ValueError(
                    "A non-global draft model batch requires an explicit "
                    "token-layout transform."
                )
            return
        if (
            self.token_layout.global_batch is not self.global_batch
            or self.token_layout.model_batch is not self.model_batch
        ):
            raise ValueError(
                "Draft token-layout batches must match the execution view."
            )

    def to_model_token_rows(
        self,
        source: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_layout is None:
            return source[: self.model_batch.num_tokens_after_padding]
        return self.token_layout.localize_token_rows(source, out=out)

    def to_global_token_rows(self, source: torch.Tensor) -> torch.Tensor:
        if self.token_layout is None:
            return source[: self.global_batch.num_tokens_after_padding]
        return self.token_layout.restore_token_rows(source)
