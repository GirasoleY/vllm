# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base worker-side logic for the NIXL connector."""

import itertools
import logging
import math
import os
import queue
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
import torch
import zmq

from vllm.distributed.kv_transfer.canonical_mapping import native_vllm_dcp_rank
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
    EngineId,
    EngineTransferInfo,
    TransferTopology,
    get_current_attn_backends,
    kv_postprocess_blksize_and_layout_on_receive,
    kv_postprocess_blksize_on_receive,
    kv_postprocess_layout_on_receive,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import CopyBlocksOp
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    MAX_NIXL_ENDPOINT_HANDSHAKE_BYTES,
    MAX_NIXL_HANDSHAKE_BYTES,
    MAX_NIXL_HANDSHAKE_RANKS,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    NixlPlacementMetadata,
    ReqId,
    ReqMeta,
    TransferHandle,
    compute_nixl_compatibility_hash,
    compute_nixl_placement_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.request_bridge import (
    NixlRemotePlacementIndex,
    index_remote_nixl_placements,
    validate_complete_nixl_placement_endpoint,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.runtime_placement import (
    NixlRuntimePlacementUnsupported,
    build_runtime_nixl_placement,
    finalize_nixl_placement_cohort,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.segmented import (
    NixlEphemeralDlistTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.terminal import (
    NixlRequestTerminalPoller,
    NixlTransferFailure,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    _is_attention_spec,
    _is_ssm_spec,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    _NIXL_SUPPORTED_DEVICE,
    get_representative_spec_type,
    recv_multipart_bounded,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    MambaConvSplitInfo,
    derive_mamba_conv_split,
)
from vllm.distributed.nixl_utils import NixlWrapper, nixl_agent_config
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_pcp_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.network_utils import make_zmq_path
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class _HandshakeSpec:
    """Identity of one reusable remote endpoint registration."""

    host: str
    port: int
    tp_size: int
    dcp_size: int
    pcp_size: int
    pp_size: int
    notif_agents_only: bool
    endpoint_incarnation: str | None


def _share_storage_and_block_stride(caches: list[torch.Tensor]) -> bool:
    """Return whether all views share storage and a block stride."""
    block_strides = {cache.stride(0) * cache.element_size() for cache in caches}
    storage_ptrs = {cache.untyped_storage().data_ptr() for cache in caches}
    return len(block_strides) == len(storage_ptrs) == 1


def _tensor_byte_span_end(cache: torch.Tensor) -> int:
    """Return the exclusive end address touched by a nonnegative-stride view."""
    if cache.numel() == 0:
        return cache.data_ptr()
    if any(stride < 0 for stride in cache.stride()):
        raise ValueError("NIXL cache views must have nonnegative strides")
    max_element_offset = sum(
        (size - 1) * stride for size, stride in zip(cache.shape, cache.stride())
    )
    return cache.data_ptr() + (max_element_offset + 1) * cache.element_size()


def _uses_dense_virtual_transfer_pages(
    layer_spec: KVCacheSpec,
    cache: torch.Tensor,
    physical_page_size: int,
    num_blocks: int,
) -> bool:
    """Return whether a compressed kernel view can be split into NIXL pages."""
    if not (
        isinstance(layer_spec, MLAAttentionSpec)
        and layer_spec.tokens_per_state > 1
        and cache.ndim == 4
        and cache.shape[1] == 1
        and cache.is_contiguous()
        and physical_page_size > 0
        and layer_spec.state_content_size_bytes > 0
    ):
        return False

    block_stride = cache.stride(0) * cache.element_size()
    return (
        block_stride > physical_page_size
        and block_stride % physical_page_size == 0
        and physical_page_size % layer_spec.state_content_size_bytes == 0
        and cache.shape[0] * (block_stride // physical_page_size) == num_blocks
        and cache.nbytes == num_blocks * physical_page_size
    )


class NixlBaseConnectorWorker:
    """Base implementation of Worker side methods shared by pull and push."""

    # Transfer mode included in the NIXL compatibility hash so that a push
    # (WRITE) connector and a pull (READ) connector never handshake together.
    # Overridden by NixlPushConnectorWorker.
    _TRANSFER_MODE: str = "pull"

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
    ) -> np.ndarray:
        """Compute NIXL descriptor IDs for given block IDs."""
        num_ssm_regions = 0
        if self._has_mamba:
            assert self._conv_decomp is not None
            # NIXL regions per SSM layer = conv sub-projections + 1 SSM temporal
            # (Mamba2/GDN: 3+1=4; Mamba1: 1+1=2).
            ssm_regions_per_layer = len(self._conv_decomp.local_conv_offsets) + 1
            # Count only the regions that actually hold SSM state; must match
            # the descriptors emitted by _build_mamba_local.
            num_ssm_regions = (
                len(self._ssm_region_indices or self.block_len_per_layer)
                * ssm_regions_per_layer
            )

        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)
        num_fa_descs = self.num_regions * num_blocks

        # All-attention fast path: single vectorized broadcast.
        if num_ssm_regions == 0:
            # NOTE (NickLucche) With HMA, every kv group has the same number of layers
            # and layers from different groups share the same kv tensor.
            # eg block_ids=[[1, 2], [3]]->blocks [1, 2] need to be
            # read across all regions, same for [3], but group0-group1 blocks will
            # always differ (different areas). Therefore we can just flatten the
            # block_ids and compute the descs ids for all groups at once.
            block_arr = np.concatenate(
                [np.asarray(g, dtype=np.int32) for g in block_ids]
            )[None, :]
            region_ids = np.arange(self.num_regions, dtype=np.int32)[:, None]
            return (region_ids * num_blocks + block_arr).ravel()

        # Compute desc ids per group using the right stride: FA descs have
        # num_blocks entries per region (kernel granularity, expanded by
        # block_size_ratio for heterogeneous block sizes), SSM descs have
        # logical_blocks entries per region (no kernel splitting, and never
        # ratio-expanded since state blocks are indivisible).
        logical_blocks = dst_num_blocks // physical_blocks_per_logical
        all_descs: list[np.ndarray] = []
        for i, group in enumerate(block_ids):
            group_arr = np.asarray(group, dtype=np.int32)
            spec_type = self._group_spec_types[i]
            if _is_attention_spec(spec_type):
                # A scratch cache lives only in its own regions; every other
                # attention group spans all of them.
                fa_region_ids = (
                    np.asarray(self._scratch_region_indices, dtype=np.int32)
                    if spec_type is CircularBufferSpec
                    else np.arange(self.num_regions, dtype=np.int32)
                )[:, None]
                all_descs.append(
                    (fa_region_ids * num_blocks + group_arr[None, :]).ravel()
                )
            elif _is_ssm_spec(spec_type):
                # NOTE (NickLucche) SSM and Attention block regions can
                # be exchanged arbitrarily by manager.  Therefore, descs
                # are laid out as:
                #   [descs_fa (all regions) | descs_ssm (all regions)].
                # num_fa_descs offset must be computed per-engine since
                # P and D can have different num_blocks (and thus
                # different FA desc counts).
                ssm_region_ids = (
                    np.arange(num_ssm_regions, num_ssm_regions + 1, dtype=np.int32)
                    if i == self._ple_group_index
                    else np.arange(num_ssm_regions, dtype=np.int32)
                )[:, None]
                all_descs.append(
                    (
                        ssm_region_ids * logical_blocks
                        + group_arr[None, :]
                        + num_fa_descs
                    ).ravel()
                )
            else:
                raise ValueError(
                    f"Unknown spec type {self._group_spec_types[i]} at index {i}"
                )

        return np.concatenate(all_descs)

    def _build_local_splits_from_plan(
        self,
        plan: TPMapping,
        src_blocks_data: np.ndarray,
        num_fa_descs: int,
        block_size_ratio: int = 1,
    ) -> Iterator[list[tuple[int, int, int]]]:
        """Build split handle data for P_TP > D_TP scenario.

        num_fa_descs is the boundary between FA and SSM descriptors.
        Split counts are derived from source_ranks_per_group lengths.
        FA uses rank_to_attention_slot for the slot offset;
        SSM uses the rank's positional index.
        """
        fa_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        fa_num_splits = len(plan.source_ranks_per_group[fa_idx])

        has_ssm_descs = num_fa_descs < len(src_blocks_data)
        ssm_idx = next(
            (i for i, t in enumerate(self._group_spec_types) if _is_ssm_spec(t)),
            None,
        )
        ssm_num_splits = (
            len(plan.source_ranks_per_group[ssm_idx])
            if has_ssm_descs and ssm_idx is not None
            else 0
        )

        # Per-FA-descriptor replicate flag, in _build_fa_local emission order.
        fa_desc_replicated = self._fa_desc_replicated(num_fa_descs)
        sharded_desc_end = len(src_blocks_data) - (
            self._logical_num_blocks if self._ple_region_index is not None else 0
        )
        assert num_fa_descs <= sharded_desc_end

        assert block_size_ratio == 1 or fa_num_splits == 1 or all(fa_desc_replicated), (
            "Head-sharded attention reads with P_TP > D_TP and heterogeneous "
            "block sizes are not supported"
        )
        src_blocks_list = src_blocks_data.tolist()

        for p_idx, p_rank in enumerate(plan.all_source_ranks):
            fa_slot = plan.rank_to_attention_slot.get(p_rank, 0)

            handle: list[tuple[int, int, int]] = []
            for j, (addr, local_len, dev) in enumerate(src_blocks_list):
                if j < num_fa_descs:
                    if fa_desc_replicated[j]:
                        # REPLICATE (MLA): whole block written on every rank.
                        handle.append((addr, local_len, dev))
                    else:
                        # SPLIT (full-attn): this rank's head slice.
                        chunk = local_len // fa_num_splits
                        handle.append((addr + fa_slot * chunk, chunk, dev))
                elif j < sharded_desc_end:
                    chunk = local_len // ssm_num_splits
                    handle.append((addr + p_idx * chunk, chunk, dev))
                else:
                    handle.append((addr, local_len, dev))
            yield handle

    def _needs_split_local_xfer_handles(self, tp_ratio: int, plan: TPMapping) -> bool:
        """Whether reads need per-source slices of the local KV region.

        Pure MLA attention is replicated across TP ranks and writes the whole
        local region. Multiple physical remote workers may still participate
        because DCP assigns them disjoint blocks, but that does not require
        splitting the local region. Hybrid MLA+SSM is different: its mapping
        contains multiple source ranks for the sharded SSM state.
        """
        return tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1)

    @staticmethod
    def _split_local_xfer_handle_key(
        tp_ratio: int,
        remote_block_size: int,
        plan: TPMapping,
    ) -> tuple[object, ...]:
        """Return the complete descriptor-slicing identity for a TP plan."""
        return (
            tp_ratio,
            remote_block_size,
            plan.source_ranks_per_group,
            plan.all_source_ranks,
            tuple(sorted(plan.rank_to_attention_slot.items())),
            plan.rank_offset_factor,
            plan.local_consumers,
        )

    def _fa_desc_replicated(self, num_fa_descs: int) -> list[bool]:
        """Per-FA-descriptor replicate flag, in _build_fa_local emission order
        (region-major; one desc per block, with K/V packed). Length ``num_fa_descs``.
        """
        assert self.transfer_topo is not None
        n_regions = len(self.block_len_per_layer)
        if n_regions == 0 or self.num_regions == 0:
            return [False] * num_fa_descs
        nblk = num_fa_descs // self.num_regions
        flags: list[bool] = []
        for i in range(n_regions):
            replicated = self._is_region_replicated(i)
            flags.extend([replicated] * nblk)
        assert len(flags) == num_fa_descs, (
            f"FA desc flags {len(flags)} != num_fa_descs {num_fa_descs}"
        )
        return flags

    def _is_region_replicated(self, region_idx: int) -> bool:
        """Whether region ``region_idx`` is transferred REPLICATE vs SPLIT.

        REPLICATE (MLA): identical on every rank, whole block read from one
        rank at offset 0, key-only. SPLIT (full-attn): head-sharded across TP.
        Defaults to SPLIT when the per-region map is unset (e.g. tests that set
        block_len_per_layer without register_kv_caches).
        """
        return region_idx < len(self._region_is_mla) and self._region_is_mla[region_idx]

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        nixl_wrapper_cls = NixlWrapper
        if nixl_wrapper_cls is None:
            logger.error("NIXL is not available")
            raise RuntimeError("NIXL is not available")
        logger.info("Initializing NIXL wrapper")
        logger.info("Initializing NIXL worker %s", engine_id)

        # Config.
        self.vllm_config = vllm_config
        # mypy will complain on re-assignment otherwise.
        self.block_size: int = cast(int, vllm_config.cache_config.block_size)

        if vllm_config.kv_transfer_config is None:
            raise ValueError("kv_transfer_config must be set for NixlConnector")
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self._enable_generic_placement = self.kv_transfer_config.get_from_extra_config(
            "enable_generic_placement", False
        )
        if not isinstance(self._enable_generic_placement, bool):
            raise ValueError("enable_generic_placement must be a boolean")
        if self._enable_generic_placement and self._TRANSFER_MODE != "pull":
            raise ValueError(
                "enable_generic_placement is currently supported only by "
                "NIXL pull transfers"
            )
        generic_completion_configured = any(
            self.kv_transfer_config.get_from_extra_config(key, None) is not None
            for key in (
                "generic_completion_participant_count",
                "generic_completion_participants",
            )
        )
        if generic_completion_configured and not self._enable_generic_placement:
            raise ValueError(
                "generic completion configuration requires "
                "enable_generic_placement=True"
            )

        self.nixl_backends = vllm_config.kv_transfer_config.get_from_extra_config(
            "backends", ["UCX"]
        )
        kv_lease_duration: int = vllm_config.kv_transfer_config.get_from_extra_config(
            "kv_lease_duration", 30
        )
        # NOTE (NickLucche): For now we use a hardcoded value for a simpler interface.
        self._lease_extension = kv_lease_duration * 2 // 3

        self._bidirectional_kv_xfer_enabled: bool = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "bidirectional_kv_xfer", False
            )
        )

        self.kv_cache_config = kv_cache_config
        # Per-layer specs, unwrapping UniformTypeKVCacheSpecs group wrappers.
        self._layer_specs: dict[str, KVCacheSpec] = {}
        for group in kv_cache_config.transfer_groups:
            group_spec = group.kv_cache_spec
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                self._layer_specs.update(group_spec.kv_cache_specs)
            else:
                self._layer_specs.update(dict.fromkeys(group.layer_names, group_spec))
        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(spec, FullAttentionSpec)
                for spec in self._layer_specs.values()
            )
        )

        # ---- Model state (derived from model config) ----
        mamba_ssm_size = (0, 0)
        # Conv state sub-projection decomposition (None when no Mamba).
        # The transfer requires DS (dim, state_len) conv layout so that
        # conv sub-projections are contiguous in memory.
        self._conv_decomp: MambaConvSplitInfo | None = None
        self._has_mamba = any(
            isinstance(spec, MambaSpec) for spec in self._layer_specs.values()
        )
        self._is_csa_linear = any(
            get_representative_spec_type(group.kv_cache_spec) is CircularBufferSpec
            for group in kv_cache_config.kv_cache_groups
        )
        self._ple_group_index: int | None = None
        if self._is_csa_linear:
            ple_groups = [
                (index, group)
                for index, group in enumerate(kv_cache_config.kv_cache_groups)
                if group.layer_names
                and all(
                    isinstance(spec, MambaSpec) and spec.tp_replicated
                    for spec in iter_layer_specs(group.kv_cache_spec)
                )
            ]
            if len(ple_groups) != 1 or len(ple_groups[0][1].layer_names) != 1:
                raise ValueError(
                    "CSA-linear NIXL requires exactly one PLE cache owner."
                )
            self._ple_group_index = ple_groups[0][0]

        if self._has_mamba:
            assert self._is_hma_required
            from vllm.model_executor.layers.mamba.mamba_utils import (
                is_conv_state_dim_first,
            )

            assert is_conv_state_dim_first(), (
                "3-read Mamba conv transfer requires DS conv state layout. "
                "Set VLLM_SSM_CONV_STATE_LAYOUT=DS"
            )
            mamba_spec = next(
                spec
                for spec in self._layer_specs.values()
                if isinstance(spec, MambaSpec)
                and (not self._is_csa_linear or not spec.tp_replicated)
            )
            self._conv_decomp = derive_mamba_conv_split(
                mamba_spec,
                vllm_config.parallel_config.tensor_parallel_size,
            )
            mamba_ssm_size = self._conv_decomp.ssm_sizes
        self._mamba_ssm_size = mamba_ssm_size

        # Agent.
        non_ucx_backends = [b for b in self.nixl_backends if b != "UCX"]
        # Configure NIXL num_threads to avoid UAR exhaustion on Mellanox NICs.
        # Each UCX thread allocates UARs (doorbell pages) via DevX, and
        # excessive NIXL UAR usage can exhaust NIC UAR space. This can cause
        # components like NVSHMEM (used by DeepEP kernels) to fail during RDMA
        # initialization with "mlx5dv_devx_alloc_uar" errors.
        # Ref: https://network.nvidia.com/files/doc-2020/ethernet-adapters-programming-manual.pdf#page=63
        num_threads = vllm_config.kv_transfer_config.get_from_extra_config(
            "num_threads", 4
        )
        if nixl_agent_config is None:
            config = None
        else:
            # Enable telemetry by default for NIXL 0.7.1 and above.
            config = (
                nixl_agent_config(backends=self.nixl_backends, capture_telemetry=True)
                if len(non_ucx_backends) > 0
                else nixl_agent_config(num_threads=num_threads, capture_telemetry=True)
            )

        self._placement_worker_incarnation = str(uuid.uuid4())
        self.nixl_wrapper = nixl_wrapper_cls(self._placement_worker_incarnation, config)
        # Request-scoped segmented-direct descriptor lists. Static descriptor
        # handles remain in the existing maps below and are never registered
        # here, so their handshake/shutdown lifetime is unchanged.
        self._ephemeral_direct_dlists = NixlEphemeralDlistTracker(self.nixl_wrapper)
        # Map of engine_id -> {(pp_rank, pp_local_rank): agent_name, ...}.
        # ``pp_local_rank = pcp_rank * tp_size + tp_rank``. With PCP disabled
        # this remains the legacy ``(pp_rank, tp_rank)`` key.
        self._remote_agents: dict[EngineId, dict[tuple[int, int], str]] = defaultdict(
            dict
        )
        self._remote_handshake_specs: dict[EngineId, _HandshakeSpec] = {}
        self._stale_remote_engines: set[EngineId] = set()
        self._legacy_fast_path_available = True
        self._local_placement_metadata: NixlPlacementMetadata | None = None
        self._local_placement_workers: tuple[NixlPlacementMetadata, ...] = ()
        self._remote_placement_indexes: dict[EngineId, NixlRemotePlacementIndex] = {}
        # A strict-hash mismatch admits only generic segmented-direct requests.
        self._generic_only_remote_engines: set[EngineId] = set()
        # Map of engine_id -> clock offset.
        self._engine_clock_offset: dict[EngineId, float] = {}

        # Metadata.
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
        self.dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_rank = get_pcp_group().rank_in_group if self.pcp_size > 1 else 0
        if (
            getattr(vllm_config.parallel_config, "enable_expert_parallel", False)
            is True
            and getattr(vllm_config.model_config, "is_moe", False) is True
        ):
            ep_group = get_ep_group()
            self.ep_size = ep_group.world_size
            self.ep_rank = ep_group.rank_in_group
        else:
            self.ep_size = 1
            self.ep_rank = 0

        # DCP support is scoped to MLA, with dcp_size in (1, tp_size): either fully
        # replicated or fully sharded. A DCP rank is always derivable this way.
        self.dcp_rank = self.tp_rank % self.dcp_size
        self.cp_kv_cache_interleave_size = (
            vllm_config.parallel_config.cp_kv_cache_interleave_size
        )
        # DCP interleaves attention blocks, while align-mode Mamba state is
        # replicated at block-table level and sharded across TP ranks by state
        # dimension during transfer. Other hybrid layouts would need their own
        # group-specific mapping.
        if (
            self._has_mamba
            and self.dcp_size > 1
            and (
                mamba_spec.mamba_cache_mode != "align"
                or any(
                    not isinstance(spec, (FullAttentionSpec, MambaSpec))
                    for spec in self._layer_specs.values()
                )
            )
        ):
            raise ValueError(
                "DCP with hybrid MLA+Mamba only supports FullAttention "
                "plus align-mode Mamba cache groups."
            )

        self.num_blocks = kv_cache_config.num_blocks
        self.enable_permute_local_kv = False
        self.enable_heterogeneous_attn_post_process = False

        # KV Caches and nixl tracking data.
        self.device_type = current_platform.device_type
        self.kv_buffer_device: str = vllm_config.kv_transfer_config.kv_buffer_device
        if self.device_type not in _NIXL_SUPPORTED_DEVICE:
            raise RuntimeError(f"{self.device_type} is not supported.")
        elif self.kv_buffer_device not in _NIXL_SUPPORTED_DEVICE[self.device_type]:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.device_kv_caches: dict[str, torch.Tensor] = {}

        # cpu kv buffer for xfer
        # used when device memory can not be registered under nixl
        self.host_xfer_buffers: dict[str, torch.Tensor] = {}
        if self.device_type == "cpu":
            self.use_host_buffer = False
        else:
            self.use_host_buffer = self.kv_buffer_device == "cpu"

        # reserve different cores for start_load_kv() from model_forward()
        if self.device_type == "cpu":
            numa_core_list = current_platform.discover_numa_topology()
            # setup one last core in each numa for kv transfer.
            rsv_cores_for_kv = [
                max(each_numa_core_list) for each_numa_core_list in numa_core_list
            ]

            if rsv_cores_for_kv:
                if not hasattr(os, "sched_setaffinity"):
                    raise NotImplementedError(
                        "os.sched_setaffinity is not available on this platform"
                    )
                os.sched_setaffinity(0, rsv_cores_for_kv)

        # support for oot platform which can't register nixl memory
        # type based on kv_buffer_device
        nixl_memory_type = current_platform.get_nixl_memory_type()
        if nixl_memory_type is None:
            if self.kv_buffer_device in ["cuda", "xpu"]:
                nixl_memory_type = "VRAM"
            elif self.kv_buffer_device == "cpu":
                nixl_memory_type = "DRAM"
        if nixl_memory_type is None:
            raise RuntimeError(
                f"{self.device_type} with {self.kv_buffer_device} kv_buffer "
                "is not supported."
            )
        self.nixl_memory_type = nixl_memory_type

        # Note: host xfer buffer ops when use_host_buffer is True
        self.copy_blocks: CopyBlocksOp | None = None

        # Map of engine_id -> kv_caches_base_addr. For TP case, each local
        self.device_id: int = 0
        # Current rank may pull from multiple remote TP workers.
        # EngineId, dict[int, list[int]] -> engine_id, tp_rank, base_addr_for_layer
        self.kv_caches_base_addr = defaultdict[EngineId, dict[int, list[int]]](dict)

        # Number of NIXL regions. Currently one region per cache
        # (so 1 per layer for MLA, otherwise 2 per layer)
        self.num_regions = 0

        # PP>1 (push mode): this worker holds a contiguous layer slice and
        # transfers into the matching sub-range of a PP=1 remote's regions.
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = get_pp_group().rank_in_group if self.pp_size > 1 else 0
        self._remote_region_offset = 0
        # PP push slices regions per layer (uniform count); HMA breaks that.
        if self.pp_size > 1 and self._is_hma_required:
            raise NotImplementedError(
                "NixlPushConnector does not support pipeline_parallel_size > 1 "
                "with hybrid KV cache layouts (HMA) yet."
            )
        # The legacy decode-side PP path cannot disambiguate per-stage static
        # descriptor maps or completion participants. Generic READ placement
        # does both by flattened endpoint rank; keep the legacy guard intact
        # unless that path was explicitly selected.
        if (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
            and self.pp_size > 1
            and not (self._TRANSFER_MODE == "pull" and self._enable_generic_placement)
        ):
            raise NotImplementedError(
                "NixlPushConnector consumer (decode) does not support "
                "pipeline_parallel_size > 1."
            )
        # Keep heartbeat handshakes to a PP-sharded producer notif-only.
        self._hb_handshake_notif_only = False

        # nixl_prepped_dlist_handle.
        self.src_xfer_handles_by_block_size: dict[int, int] = {}
        # Local descriptor arrays per remote block size (block_size_ratio>1),
        # kept for building per-tp-ratio splits at the same granularity.
        self.src_blocks_data_by_block_size: dict[int, np.ndarray] = {}
        # Populated dynamically during handshake based on remote configuration.
        # Per-source split handles, keyed by their complete mapping and geometry.
        self.src_xfer_handles_by_tp_ratio: dict[tuple[object, ...], list[int]] = {}
        # Map of engine_id -> {tp_rank: nixl_prepped_dlist_handle (int)}.
        self.dst_xfer_side_handles = defaultdict[EngineId, dict[int, int]](dict)

        # Map of engine_id -> num_blocks. All ranks in the same deployment will
        # have the same number of blocks.
        self.dst_num_blocks: dict[EngineId, int] = {}
        self._registered_descs: list[Any] = []

        # In progress transfers.
        # [req_id -> list[handle]]
        self._recving_metadata: dict[ReqId, ReqMeta] = {}
        self._recving_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._generic_direct_receive_requests: set[ReqId] = set()
        self._request_terminal_poller = NixlRequestTerminalPoller()
        # Track the expiration time of requests that are waiting to be sent.
        self._reqs_to_send: dict[ReqId, float] = {}
        # Set of requests that have been part of a batch, regardless of status.
        self._reqs_to_process: set[ReqId] = set()

        # Invalid blocks from failed NIXL operations (thread-safe queue of block ids)
        self._invalid_block_ids: queue.Queue[set[int]] = queue.Queue()
        # requests that skipped transfer (handshake or transfer failures)
        # Uses Queue for thread-safe cross-thread coordination with the
        # background handshake thread, matching the _ready_requests pattern.
        self._failed_recv_reqs: queue.Queue[ReqId] = queue.Queue()

        # Handshake metadata of this worker for NIXL transfers.
        self.xfer_handshake_metadata: NixlHandshakePayload | None = None
        # Background thread for initializing new NIXL handshakes.
        self._handshake_lock = threading.RLock()
        self._handshake_shutdown_event = threading.Event()
        self._shutting_down = False
        self._shutdown_complete = False
        self._handshake_initiation_executor = ThreadPoolExecutor(
            # NIXL is not guaranteed to be thread-safe, limit 1 worker.
            max_workers=1,
            thread_name_prefix="vllm-nixl-handshake-initiator",
        )
        self._ready_requests = queue.Queue[tuple[ReqId, ReqMeta]]()
        self._handshake_futures: dict[
            EngineId, Future[tuple[dict[tuple[int, int], str], float]]
        ] = {}
        self._handshake_future_specs: dict[EngineId, _HandshakeSpec] = {}
        # Protects _handshake_futures and _remote_agents.

        # TTL-based eviction of stale remote engine state.
        self._engine_last_active: dict[EngineId, float] = {}
        self._engine_ttl: float = vllm_config.kv_transfer_config.get_from_extra_config(
            "engine_ttl", 3600.0
        )

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config

        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        self.attn_backends = get_current_attn_backends(vllm_config)
        self.backend_name = self.attn_backends[0].get_name()

        self.kv_cache_layout = (
            vllm_config.cache_config.get_resolved_kv_cache_layout().name
        )
        self.host_buffer_kv_cache_layout = self.kv_cache_layout
        logger.info(
            "Detected attention backend(s) %s",
            [backend.get_name() for backend in self.attn_backends],
        )
        logger.info("Detected kv cache layout %s", self.kv_cache_layout)

        # lazy initialized in register_kv_caches
        self.compat_hash: str | None = None
        self.placement_compat_hash: str | None = None
        self.transfer_topo: TransferTopology | None = None

        # With heterogeneous TP (or DCP), P must wait for all assigned D
        # workers to finish reading before safely freeing the blocks.
        self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
        self.expected_consumer_notifications_by_req: dict[ReqId, int] = {}
        self.xfer_stats = NixlKVConnectorStats()

        self._physical_blocks_per_logical_kv_block = 1
        self._sync_block_size_with_kernel()

        # Unwrap UniformTypeKVCacheSpecs to get the representative spec type
        self._group_spec_types = tuple(
            get_representative_spec_type(g.kv_cache_spec)
            for g in self.kv_cache_config.transfer_groups
        )
        # Per-region MLA flag, 1:1 with block_len_per_layer. True -> REPLICATE
        # (MLA), False -> SPLIT (head-sharded full-attn). Mixed only for models
        # combining both (e.g. GQA main + MLA Eagle-3 draft).
        self._region_is_mla = list[bool]()
        self._ssm_region_indices = list[int]()
        # Regions holding a scratch cache (the CSA compressor circular buffer).
        # Tracked explicitly because a scratch page shares its address with the
        # page it overlays, so its regions cannot be recovered from spec type.
        self._scratch_region_indices = list[int]()
        self._ple_region_index: int | None = None

        # Enable different block lengths for different layers *only* when MLA is used.
        # This is not used for SSM layers, which use the counterpart `mamba_ssm_size`.
        self.block_len_per_layer = list[int]()

        # Per-region block stride in bytes. Taken from the registered tensor's
        # stride(0) so it stays correct under layouts that interleave layers
        # within a block (BLHNC/BHLNC), where stride > block_len.
        self.block_stride_per_layer = list[int]()

        # Per-engine TP mappings. Generated during handshake.
        self.tp_mappings: dict[EngineId, TPMapping] = {}

        self.enforce_compat_hash = self.kv_transfer_config.get_from_extra_config(
            "enforce_handshake_compat", True
        )

    def _validate_remote_parallel_config(
        self,
        agent_metadata: NixlAgentMetadata,
        *,
        generic_placement_available: bool = False,
    ) -> None:
        local_pcp_size = getattr(self, "pcp_size", 1)
        local_dcp_size = self.dcp_size
        remote_pcp_size = agent_metadata.pcp_size
        remote_dcp_size = agent_metadata.dcp_size
        if remote_pcp_size > 1 and remote_dcp_size > 1:
            raise NotImplementedError(
                "NixlConnector does not yet support PCP and DCP on the same "
                "endpoint. Cross-endpoint PCP-to-DCP placement is supported by "
                "generic segmented-direct transfer. "
                f"Remote PCP/DCP={remote_pcp_size}/{remote_dcp_size}."
            )
        cross_endpoint_pcp_dcp = (local_pcp_size > 1 and remote_dcp_size > 1) or (
            remote_pcp_size > 1 and local_dcp_size > 1
        )
        if cross_endpoint_pcp_dcp and not generic_placement_available:
            raise NotImplementedError(
                "Cross-endpoint PCP/DCP requires generic segmented-direct "
                "NIXL READ placement. "
                f"Local PCP/DCP={local_pcp_size}/{local_dcp_size}; "
                f"remote PCP/DCP={remote_pcp_size}/{remote_dcp_size}."
            )

    @staticmethod
    def _validate_requested_remote_topology(
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_pcp_size: int,
        remote_pp_size: int,
    ) -> None:
        for name, size in (
            ("TP", remote_tp_size),
            ("DCP", remote_dcp_size),
            ("PCP", remote_pcp_size),
            ("PP", remote_pp_size),
        ):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise RuntimeError(f"Remote {name} size must be a positive integer")
            if size > MAX_NIXL_HANDSHAKE_RANKS:
                raise RuntimeError(
                    f"Remote {name} size exceeds the "
                    f"{MAX_NIXL_HANDSHAKE_RANKS}-rank handshake limit"
                )
        endpoint_ranks = remote_tp_size * remote_pcp_size * remote_pp_size
        if endpoint_ranks > MAX_NIXL_HANDSHAKE_RANKS:
            raise RuntimeError(
                f"Remote endpoint has {endpoint_ranks} placement ranks, exceeding "
                f"the {MAX_NIXL_HANDSHAKE_RANKS}-rank handshake limit"
            )
        if remote_pcp_size == 1 and remote_tp_size % remote_dcp_size:
            raise RuntimeError(
                "Remote TP size must be divisible by remote DCP size: "
                f"TP={remote_tp_size}, DCP={remote_dcp_size}."
            )
        if remote_pcp_size > 1 and remote_dcp_size > 1:
            raise RuntimeError(
                "Remote NIXL endpoint cannot combine PCP and DCP yet: "
                f"PCP={remote_pcp_size}, DCP={remote_dcp_size}."
            )

    @staticmethod
    def _validate_remote_metadata_topology(
        metadata: NixlAgentMetadata,
        *,
        remote_tp_size: int,
        remote_dcp_size: int,
        remote_pcp_size: int,
        remote_pp_size: int,
        remote_tp_rank: int,
        remote_pcp_rank: int,
        remote_pp_rank: int,
        remote_pp_local_rank: int,
    ) -> None:
        if metadata.dcp_size != remote_dcp_size:
            raise RuntimeError(
                "Remote NIXL metadata DCP size does not match the requested "
                f"topology: requested={remote_dcp_size}, "
                f"advertised={metadata.dcp_size}."
            )
        if metadata.pcp_size != remote_pcp_size:
            raise RuntimeError(
                "Remote NIXL metadata PCP size does not match the requested "
                f"topology: requested={remote_pcp_size}, "
                f"advertised={metadata.pcp_size}."
            )

        placement = metadata.placement_metadata
        if placement is None:
            return
        rank_placement = placement.rank_placement
        expected = {
            "TP size": (rank_placement.tp_size, remote_tp_size),
            "TP rank": (rank_placement.tp_rank, remote_tp_rank),
            "DCP size": (rank_placement.dcp_size, remote_dcp_size),
            "DCP rank": (
                rank_placement.dcp_rank,
                native_vllm_dcp_rank(
                    tp_size=remote_tp_size,
                    tp_rank=remote_tp_rank,
                    dcp_size=remote_dcp_size,
                    pcp_size=remote_pcp_size,
                    pcp_rank=remote_pcp_rank,
                ),
            ),
            "PCP size": (rank_placement.pcp_size, metadata.pcp_size),
            "PCP rank": (rank_placement.pcp_rank, remote_pcp_rank),
            "PP size": (rank_placement.pp_size, remote_pp_size),
            "PP rank": (rank_placement.pp_rank, remote_pp_rank),
            "CP interleave": (
                rank_placement.cp_interleave,
                metadata.cp_kv_cache_interleave_size,
            ),
        }
        mismatches = [
            f"{name}: advertised={advertised}, expected={wanted}"
            for name, (advertised, wanted) in expected.items()
            if advertised != wanted
        ]
        if mismatches:
            raise RuntimeError(
                "Remote NIXL placement topology does not match its handshake "
                f"coordinate ({remote_pp_rank}, {remote_pp_local_rank}) or requested "
                f"topology: {'; '.join(mismatches)}"
            )

    def _sync_block_size_with_kernel(self) -> None:
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        # Number of blocks not accounting for kernel block mismatches
        self._logical_num_blocks = self.num_blocks
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self._physical_blocks_per_logical_kv_block = (
                self.block_size // kernel_block_size
            )
            self.block_size = kernel_block_size
            self.num_blocks *= self._physical_blocks_per_logical_kv_block

    def _validate_csa_linear_tp_layout(self, remote_tp_size: int) -> None:
        """Reject P/D pairs whose main-KV pages have different head layouts.

        A rank at TP < total KV heads holds several heads per page, while a
        rank at TP >= total KV heads holds exactly one (replicated beyond
        that). CSA-linear pages are shared tensors that transfer as whole
        block units, so endpoints on opposite sides of that boundary lay
        them out differently and cannot exchange them.
        """
        assert self.transfer_topo is not None
        total_kv_heads = self.transfer_topo.total_num_kv_heads
        if (self.world_size < total_kv_heads) != (remote_tp_size < total_kv_heads):
            raise ValueError(
                "CSA-linear NIXL requires both endpoints on the same side of "
                f"the KV-head sharding boundary ({total_kv_heads} KV heads): "
                f"got local TP {self.world_size}, remote TP {remote_tp_size}."
            )

    def _nixl_handshake(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dcp_size: int = 1,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
        remote_pcp_size: int = 1,
        expected_endpoint_incarnation: str | None = None,
    ) -> tuple[dict[tuple[int, int], str], float]:
        """Atomically import every remote rank needed by one endpoint.

        NIXL agent imports happen rank by rank, while the endpoint is usable
        only after its complete placement metadata has been validated.  Keep
        the partially imported names local and roll all per-engine state back
        if any later rank or endpoint-level validation fails.
        """
        imported_agents: list[str] = []
        try:
            return self._nixl_handshake_impl(
                host,
                port,
                remote_tp_size,
                expected_engine_id,
                remote_dcp_size,
                remote_pp_size,
                notif_agents_only,
                imported_agents,
                remote_pcp_size,
                expected_endpoint_incarnation,
            )
        except BaseException:
            self._rollback_incomplete_handshake(expected_engine_id, imported_agents)
            raise

    def _nixl_handshake_impl(
        self,
        host: str,
        port: int,
        remote_tp_size: int,
        expected_engine_id: str,
        remote_dcp_size: int = 1,
        remote_pp_size: int = 1,
        notif_agents_only: bool = False,
        imported_agents: list[str] | None = None,
        remote_pcp_size: int = 1,
        expected_endpoint_incarnation: str | None = None,
    ) -> tuple[dict[tuple[int, int], str], float]:
        """Do a NIXL handshake with a remote instance."""
        if imported_agents is None:
            imported_agents = []
        self._validate_requested_remote_topology(
            remote_tp_size,
            remote_dcp_size,
            remote_pcp_size,
            remote_pp_size,
        )
        if self._is_csa_linear:
            self._validate_csa_linear_tp_layout(remote_tp_size)

        # the first time we connect to a remote agent.
        # be careful, the handshake happens in a background thread.
        # it does not have an active cuda context until any cuda runtime
        # call is made. when UCX fails to find a valid cuda context, it will
        # disable any cuda ipc communication, essentially disabling any NVLink
        # communication.
        # when we are using device buffers, we need to set the device
        # explicitly to make sure the handshake background thread has a valid
        # cuda context.
        if not self.use_host_buffer:
            current_platform.set_device(self.device_id)

        # When target instance TP > local TP, we need to perform multiple
        # handshakes. Do it in a single background job for simplicity.
        # Regardless, only handshake with the remote TP rank(s) that current
        # local rank will read from. Note that With homogeneous TP,
        # this happens to be the same single rank_i.
        assert self.transfer_topo is not None
        legacy_topology_error: RuntimeError | None = None
        try:
            p_remote_ranks = self.transfer_topo.handshake_target_ranks(
                remote_tp_size, remote_dcp_size
            )
        except AssertionError as error:
            if self._local_placement_metadata is None or notif_agents_only:
                raise RuntimeError(
                    "Legacy NIXL cannot represent the requested remote topology"
                ) from error
            # Canonical placement can compose TP sizes that do not divide one
            # another. Query the complete endpoint and force generic admission.
            legacy_topology_error = RuntimeError(
                "Legacy NIXL cannot represent the requested remote topology"
            )
            p_remote_ranks = list(range(remote_tp_size))
        legacy_target_ranks = frozenset(p_remote_ranks)
        if self._local_placement_metadata is not None and not notif_agents_only:
            # The generic request planner elects writers per canonical page and
            # sends one aggregate completion to every producer participant. Keep
            # the topology-selected ranks distinct: only those ranks may build
            # legacy static descriptor lists. The remaining ranks are imported
            # metadata-only through generic registration when both endpoints
            # advertise a compatible placement protocol.
            p_remote_ranks = list(range(remote_pcp_size * remote_tp_size))
        remote_rank_to_agent_name: dict[tuple[int, int], str] = {}
        remote_metadata_by_rank: dict[tuple[int, int], NixlAgentMetadata] = {}
        path = make_zmq_path("tcp", host, port)
        # Clock offset to the peer, estimated from the handshake round-trip.
        # Keep the lowest-RTT sample: hop cost is ~uniform across ranks, so a
        # higher RTT is just noise that skews the midpoint estimate.
        best_rtt = float("inf")
        best_offset: float | None = None
        # True means the strict legacy hashes differed and the whole endpoint
        # must be admitted through generic placement only. Legitimate ranks in
        # one remote endpoint always publish the same compatibility hashes;
        # reject a mixed endpoint before it can partially commit.
        endpoint_generic_only: bool | None = None
        all_placement_hashes_match = True
        advertised_endpoint_incarnation: str | None = None

        endpoint_handshake_bytes = 0
        with zmq_ctx(
            zmq.REQ,
            path,
            max_message_size=MAX_NIXL_HANDSHAKE_BYTES,
        ) as sock:
            for remote_pp_rank, remote_pp_local_rank in itertools.product(
                range(remote_pp_size), p_remote_ranks
            ):
                shutdown_event = getattr(self, "_handshake_shutdown_event", None)
                if shutdown_event is not None and shutdown_event.is_set():
                    raise RuntimeError("NIXL worker is shutting down")
                remote_pcp_rank, remote_tp_rank = divmod(
                    remote_pp_local_rank, remote_tp_size
                )
                logger.debug(
                    "Querying metadata on path: %s at remote pp rank %s, "
                    "pcp rank %s, tp rank %s",
                    path,
                    remote_pp_rank,
                    remote_pcp_rank,
                    remote_tp_rank,
                )

                # Send query for the request.
                msg = msgspec.msgpack.encode(
                    (GET_META_MSG, remote_pp_rank, remote_pp_local_rank)
                )
                # Set receive timeout to 5 seconds to avoid hanging on dead server
                sock.setsockopt(zmq.RCVTIMEO, 5000)  # milliseconds
                start_time = time.perf_counter()
                sock.send(msg)
                reply_parts = recv_multipart_bounded(sock, 2)
                recv_time = time.perf_counter()
                if len(reply_parts) != 2 or any(
                    not isinstance(part, bytes) for part in reply_parts
                ):
                    raise RuntimeError(
                        "Invalid NIXL handshake response framing: expected two "
                        "byte frames"
                    )
                endpoint_handshake_bytes += sum(map(len, reply_parts))
                if endpoint_handshake_bytes > MAX_NIXL_ENDPOINT_HANDSHAKE_BYTES:
                    raise RuntimeError(
                        "Remote endpoint handshake metadata exceeds the "
                        f"{MAX_NIXL_ENDPOINT_HANDSHAKE_BYTES}-byte aggregate limit"
                    )
                handshake_bytes = reply_parts[0]

                try:
                    remote_perf = msgspec.msgpack.decode(
                        reply_parts[1], type=float, strict=True
                    )
                except (msgspec.DecodeError, msgspec.ValidationError) as error:
                    raise RuntimeError(
                        "Invalid NIXL handshake timestamp frame"
                    ) from error
                if not math.isfinite(remote_perf):
                    raise RuntimeError(
                        "Invalid NIXL handshake timestamp frame: expected a "
                        "finite value"
                    )
                rtt = recv_time - start_time
                if rtt < best_rtt:
                    best_rtt = rtt
                    best_offset = remote_perf - (start_time + recv_time) / 2

                # Decode handshake payload to get compatibility hash
                try:
                    handshake_payload = NixlHandshakePayload.decode(handshake_bytes)
                except ValueError as e:
                    raise RuntimeError(
                        f"Failed to decode NixlHandshakePayload. This likely indicates "
                        f"an incompatibility between connector version. Error: {e}"
                    ) from e

                rank_endpoint_incarnation = handshake_payload.endpoint_incarnation
                if advertised_endpoint_incarnation is None:
                    advertised_endpoint_incarnation = rank_endpoint_incarnation
                elif rank_endpoint_incarnation != advertised_endpoint_incarnation:
                    raise RuntimeError(
                        "Remote NIXL ranks advertise inconsistent endpoint incarnations"
                    )
                if (
                    expected_endpoint_incarnation is not None
                    and rank_endpoint_incarnation != expected_endpoint_incarnation
                ):
                    raise RuntimeError(
                        "Remote NIXL endpoint incarnation does not match request "
                        "metadata"
                    )

                got_metadata_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: get metadata took: %s",
                    got_metadata_time - start_time,
                )

                # Check compatibility hashes BEFORE decoding agent metadata.
                # A matching placement hash includes the connector protocol
                # version, making strict nested placement decoding safe even
                # when backend/layout-only factors changed the legacy hash.
                assert self.compat_hash is not None
                strict_hash_matches = (
                    handshake_payload.compatibility_hash == self.compat_hash
                )
                assert self.placement_compat_hash is not None
                placement_hash_matches = (
                    handshake_payload.placement_compatibility_hash
                    == self.placement_compat_hash
                )
                all_placement_hashes_match &= placement_hash_matches
                rank_generic_only = False
                if self.enforce_compat_hash and not strict_hash_matches:
                    generic_fallback_available = (
                        self._TRANSFER_MODE == "pull"
                        and not notif_agents_only
                        and self._local_placement_metadata is not None
                    )
                    if not generic_fallback_available:
                        raise RuntimeError(
                            "NIXL compatibility hash mismatch and generic placement "
                            "is unavailable. "
                            f"Local: {self.compat_hash}, "
                            f"Remote: {handshake_payload.compatibility_hash}."
                        )
                    if not placement_hash_matches:
                        raise RuntimeError(
                            "NIXL compatibility hash mismatch and placement "
                            "compatibility hash mismatch. "
                            f"Local placement: {self.placement_compat_hash}, "
                            "Remote placement: "
                            f"{handshake_payload.placement_compatibility_hash}."
                        )
                    rank_generic_only = True

                # Decode agent metadata
                try:
                    metadata = NixlAgentMetadata.decode(
                        handshake_payload.agent_metadata_bytes
                    )
                except ValueError as e:
                    # The strict or placement hash includes the connector
                    # protocol version, so a decode failure is incompatible.
                    raise RuntimeError(
                        f"Failed to decode NixlAgentMetadata. Error: {e}"
                    ) from e

                self._validate_remote_metadata_topology(
                    metadata,
                    remote_tp_size=remote_tp_size,
                    remote_dcp_size=remote_dcp_size,
                    remote_pcp_size=remote_pcp_size,
                    remote_pp_size=remote_pp_size,
                    remote_tp_rank=remote_tp_rank,
                    remote_pcp_rank=remote_pcp_rank,
                    remote_pp_rank=remote_pp_rank,
                    remote_pp_local_rank=remote_pp_local_rank,
                )
                generic_registration_available = (
                    self._TRANSFER_MODE == "pull"
                    and not notif_agents_only
                    and self._generic_registration_available(metadata)
                )
                if not getattr(self, "_legacy_fast_path_available", True):
                    if not generic_registration_available:
                        raise RuntimeError(
                            "Local legacy NIXL descriptor preparation failed and "
                            "generic placement is unavailable"
                        )
                    if not placement_hash_matches:
                        raise RuntimeError(
                            "Local legacy NIXL descriptor preparation failed and "
                            "placement compatibility hash mismatch prevents the "
                            "generic path"
                        )
                    rank_generic_only = True
                self._validate_remote_parallel_config(
                    metadata,
                    generic_placement_available=generic_registration_available,
                )
                placement_axis_requires_generic = (
                    self._TRANSFER_MODE == "pull"
                    and not notif_agents_only
                    and (
                        getattr(self, "pp_size", 1) > 1
                        or remote_pp_size > 1
                        or getattr(self, "pcp_size", 1) > 1
                        or remote_pcp_size > 1
                    )
                )
                if placement_axis_requires_generic:
                    # Legacy descriptor maps are keyed only by remote TP rank,
                    # so two PP stages would alias the same entry. The generic
                    # path addresses the flattened PP x TP placement rank and
                    # must be selected for the entire endpoint before any
                    # legacy descriptor registration mutates shared state.
                    if not generic_registration_available:
                        raise RuntimeError(
                            "PCP/PP generic NIXL requires placement metadata "
                            "on both endpoints"
                        )
                    if not placement_hash_matches:
                        raise RuntimeError(
                            "PCP/PP generic NIXL placement "
                            "compatibility hash mismatch. "
                            f"Local: {self.placement_compat_hash}, remote: "
                            f"{handshake_payload.placement_compatibility_hash}."
                        )
                    rank_generic_only = True
                if not rank_generic_only:
                    # Topology and interleave are intentionally absent from the
                    # strict hash because legacy supports several heterogeneous
                    # combinations. Some combinations cannot use raw legacy page
                    # copies, however. Select the canonical path before legacy
                    # registration mutates any per-engine state, even when strict
                    # hash enforcement was explicitly disabled.
                    try:
                        if legacy_topology_error is not None:
                            raise legacy_topology_error
                        self._validate_legacy_registration_compatibility(
                            metadata, remote_tp_size
                        )
                    except RuntimeError as legacy_error:
                        if not generic_registration_available:
                            raise
                        if not placement_hash_matches:
                            raise RuntimeError(
                                "Legacy NIXL registration cannot represent this "
                                "KV placement and placement compatibility hash "
                                "mismatch prevents the generic path. "
                                f"Local placement: {self.placement_compat_hash}, "
                                "Remote placement: "
                                f"{handshake_payload.placement_compatibility_hash}."
                            ) from legacy_error
                        rank_generic_only = True

                if rank_generic_only and not self._generic_registration_available(
                    metadata
                ):
                    raise RuntimeError(
                        "Generic-only NIXL compatibility requires placement metadata "
                        "on both endpoints"
                    )

                if endpoint_generic_only is None:
                    endpoint_generic_only = rank_generic_only
                elif endpoint_generic_only != rank_generic_only:
                    raise RuntimeError(
                        "Remote NIXL workers advertised mixed legacy and generic-only "
                        "compatibility modes"
                    )

                if rank_generic_only:
                    logger.info(
                        "NIXL generic-placement compatibility check passed (hash: %s)",
                        handshake_payload.placement_compatibility_hash,
                    )
                else:
                    logger.info(
                        "NIXL legacy compatibility check passed (hash: %s)",
                        handshake_payload.compatibility_hash,
                    )

                # Ensure engine id matches.
                if metadata.engine_id != expected_engine_id:
                    raise RuntimeError(
                        f"Remote NIXL agent engine ID mismatch. "
                        f"Expected {expected_engine_id},"
                        f"received {metadata.engine_id}."
                    )

                remote_ranks = (remote_pp_rank, remote_pp_local_rank)
                remote_metadata_by_rank[remote_ranks] = metadata

                # Generic placement needs metadata and agent handles for every
                # endpoint rank. Legacy only needs the topology-selected ranks.
                # Do not accidentally materialize a full static descriptor list
                # on every local/remote TP pair merely because the local worker
                # advertises optional generic placement.
                is_legacy_target = (
                    remote_pcp_rank == 0 and remote_tp_rank in legacy_target_ranks
                )
                generic_extra_rank = (
                    not is_legacy_target
                    and generic_registration_available
                    and placement_hash_matches
                )
                if not is_legacy_target and not generic_extra_rank:
                    continue

                # Register Remote agent.
                if notif_agents_only:
                    remote_agent_name = self._add_notif_only_remote_agent(
                        metadata, remote_tp_size, metadata.dcp_size
                    )
                else:
                    remote_agent_name = self.add_remote_agent(
                        metadata,
                        remote_tp_rank,
                        remote_tp_size,
                        metadata.dcp_size,
                        generic_registration=(rank_generic_only or generic_extra_rank),
                    )
                setup_agent_time = time.perf_counter()
                logger.debug(
                    "NIXL handshake: add agent took: %s (notif_agents_only=%s)",
                    setup_agent_time - got_metadata_time,
                    notif_agents_only,
                )
                remote_rank_to_agent_name[remote_ranks] = remote_agent_name
                imported_agents.append(remote_agent_name)

        assert best_offset is not None
        advertised_placements = [
            metadata.placement_metadata is not None
            for metadata in remote_metadata_by_rank.values()
        ]
        if any(advertised_placements) and not all(advertised_placements):
            raise RuntimeError(
                "Remote NIXL workers inconsistently advertised generic placement"
            )
        if (
            self._local_placement_metadata is not None
            and advertised_placements
            and all(advertised_placements)
            and all_placement_hashes_match
            and not notif_agents_only
        ):
            remote_index = index_remote_nixl_placements(
                remote_metadata_by_rank, remote_rank_to_agent_name
            )
            validate_complete_nixl_placement_endpoint(remote_index.workers)
            self._remote_placement_indexes[expected_engine_id] = remote_index
        elif advertised_placements and not all_placement_hashes_match:
            logger.warning(
                "Remote NIXL placement hash differs; retaining only the strict "
                "legacy path for engine %s",
                expected_engine_id,
            )
        if endpoint_generic_only:
            if expected_engine_id not in self._remote_placement_indexes:
                raise RuntimeError(
                    "Generic-only NIXL handshake did not produce a validated "
                    "placement endpoint"
                )
            self._generic_only_remote_engines.add(expected_engine_id)
        else:
            self._generic_only_remote_engines.discard(expected_engine_id)
        return remote_rank_to_agent_name, best_offset

    def _rollback_incomplete_handshake(
        self, engine_id: EngineId, imported_agents: list[str]
    ) -> None:
        """Release state created before a multi-rank handshake committed."""
        for handle in self.dst_xfer_side_handles.pop(engine_id, {}).values():
            try:
                self.nixl_wrapper.release_dlist_handle(handle)
            except Exception:
                logger.warning(
                    "Failed to release a descriptor list while rolling back "
                    "NIXL handshake for engine %s",
                    engine_id,
                    exc_info=True,
                )

        # _remote_agents is normally committed only by the Future callback,
        # but include it to make direct/re-entrant calls equally safe.
        committed_agents = self._remote_agents.pop(engine_id, {})
        agent_names = set(imported_agents)
        agent_names.update(committed_agents.values())
        for agent_name in agent_names:
            try:
                self.nixl_wrapper.remove_remote_agent(agent_name)
            except Exception:
                logger.warning(
                    "Failed to remove remote agent %s while rolling back NIXL "
                    "handshake for engine %s",
                    agent_name,
                    engine_id,
                    exc_info=True,
                )

        self.kv_caches_base_addr.pop(engine_id, None)
        self.dst_num_blocks.pop(engine_id, None)
        self.tp_mappings.pop(engine_id, None)
        self._remote_placement_indexes.pop(engine_id, None)
        self._generic_only_remote_engines.discard(engine_id)
        self._remote_handshake_specs.pop(engine_id, None)
        self._stale_remote_engines.discard(engine_id)
        self._engine_clock_offset.pop(engine_id, None)
        self._engine_last_active.pop(engine_id, None)
        if self.transfer_topo is not None:
            self.transfer_topo.unregister_remote_engine(engine_id)

    def _add_notif_only_remote_agent(
        self, metadata: NixlAgentMetadata, remote_tp_size: int, remote_dcp_size: int = 1
    ) -> str:
        """Load a remote agent for notifs only on the push-mode decode side.

        Skips descriptor setup but records engine info for block accounting.
        """
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            metadata.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=metadata.block_size,
                remote_block_len=metadata.block_lens[0],
                remote_physical_blocks_per_logical=(
                    metadata.physical_blocks_per_logical_kv_block
                ),
                remote_dcp_size=remote_dcp_size,
                remote_cp_kv_cache_interleave_size=(
                    metadata.cp_kv_cache_interleave_size
                ),
            ),
        )
        return self.nixl_wrapper.add_remote_agent(metadata.agent_metadata)

    def initialize_host_xfer_buffer(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Initialize transfer buffer in CPU mem for accelerators
        NOT directly supported by NIXL (e.g., tpu)
        """
        xfer_buffers: dict[str, torch.Tensor] = {}
        try:
            for layer_name, kv_cache in kv_caches.items():
                kv_shape = kv_cache.shape
                kv_dtype = kv_cache.dtype
                permute_shape = False
                inv_order = (0, 2, 1, 3)
                if not self.use_mla:
                    assert kv_cache.ndim == 4

                    if self.kv_cache_layout == "LBNHC":
                        if self.kv_transfer_config.enable_permute_local_kv:
                            logger.info_once(
                                "'enable_permute_local_kv' flag is enabled while "
                                "device KV Layout is LBNHC. Init host buffer with"
                                " LBHNC to better support Decode/Prefill TP_ratio > 1."
                            )
                            # Since LBNHC will not support Decode/Prefill TP_ratio > 1,
                            # we can leverage host_buffer for permute.
                            self.host_buffer_kv_cache_layout = "LBHNC"
                        else:
                            # Packed KV layout is logical (B, H, N, 2*D). Allocate
                            # (B, N, H, 2*D) and view it as logical (B, H, N, 2*D)
                            # so raw NIXL transfers see NHD physical strides.
                            kv_shape = tuple(kv_shape[i] for i in inv_order)
                            permute_shape = True

                xfer_buffers[layer_name] = torch.empty(
                    kv_shape, dtype=kv_dtype, device="cpu"
                )
                if permute_shape:
                    xfer_buffers[layer_name] = xfer_buffers[layer_name].permute(
                        inv_order
                    )
        except MemoryError as e:
            logger.error("NIXLConnectorWorker gets %s.", e)
            raise

        self.host_xfer_buffers = xfer_buffers

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp):
        """Assign copy (d2h, h2d) operations when host buffer is used."""
        # Set a no-op if the host buffer is not cpu.
        if self.kv_buffer_device != "cpu":
            return
        # Set a no-op if self.device_type is 'cpu'.
        if self.device_type == "cpu":
            return
        assert self.use_host_buffer
        self.copy_blocks = copy_operation

    def _log_failure(
        self,
        failure_type: str,
        req_id: str | None,
        msg: str = "",
        error: BaseException | None = None,
        meta: ReqMeta | None = None,
        **extra_context,
    ):
        """Log transfer failure with structured context for easier debugging."""
        context: dict[str, Any] = {
            "failure_type": failure_type,
            "request_id": req_id,
            "engine_id": self.engine_id,
        }
        if meta is None and req_id is not None:
            # Try to get metadata from in progress transfers when not provided
            meta = self._recving_metadata.get(req_id)

        if meta and meta.remote:
            context.update(
                {
                    "remote_engine_id": meta.remote.engine_id,
                    "remote_request_id": meta.remote.request_id,
                    "remote_host": meta.remote.host,
                    "remote_port": meta.remote.port,
                    "num_local_blocks": sum(
                        len(group) for group in meta.local_block_ids
                    ),
                    "num_remote_blocks": sum(
                        len(group) for group in meta.remote.block_ids
                    ),
                    "local_block_ids_sample": meta.local_block_ids[0][:10]
                    if meta.local_block_ids
                    else [],
                }
            )

        context.update(extra_context)
        if msg:
            failure_type = f"{failure_type}. {msg}"

        logger.error(
            "NIXL transfer failure: %s | Context: %s",
            failure_type,
            context,
            exc_info=error is not None,
            stacklevel=2,
        )

    def _ensure_handshake(
        self,
        engine_id: EngineId,
        host: str,
        port: int,
        tp_size: int,
        dcp_size: int = 1,
        pp_size: int = 1,
        notif_agents_only: bool = False,
        pcp_size: int = 1,
        endpoint_incarnation: str | None = None,
        request_id: str | None = None,
    ) -> Future[tuple[dict[tuple[int, int], str], float]] | None:
        """
        Ensure a handshake is in-flight (or already done) for *engine_id*.

        Returns the ``Future`` if a handshake is pending (or was just
        started), or ``None`` if the handshake already completed
        successfully.  Callers can attach per-request callbacks to the
        returned future.
        Failures to handshake are logged and the request is marked as failed.
        """
        if not isinstance(endpoint_incarnation, str) or not endpoint_incarnation:
            raise RuntimeError("Remote endpoint incarnation must be a non-empty string")
        spec = _HandshakeSpec(
            host=host,
            port=port,
            tp_size=tp_size,
            dcp_size=dcp_size,
            pcp_size=pcp_size,
            pp_size=pp_size,
            notif_agents_only=notif_agents_only,
            endpoint_incarnation=endpoint_incarnation,
        )
        with self._handshake_lock:
            if getattr(self, "_shutting_down", False):
                raise RuntimeError("NIXL worker is shutting down")
            self._evict_stale_engines(exclude_request_id=request_id)
            if engine_id in self._remote_agents:
                if self._remote_handshake_specs.get(engine_id) == spec:
                    return None
                self._stale_remote_engines.add(engine_id)
                if self._engine_has_active_references(
                    engine_id, exclude_request_id=request_id
                ):
                    raise RuntimeError(
                        "Remote NIXL endpoint changed while its previous "
                        "registration is still in use"
                    )
                self._cleanup_remote_engine(engine_id, log_eviction=False)
            fut = self._handshake_futures.get(engine_id)
            if fut is not None:
                if self._handshake_future_specs.get(engine_id) != spec:
                    raise RuntimeError(
                        "Remote NIXL endpoint changed while a handshake for its "
                        "previous incarnation is still pending"
                    )
                return fut
            fut = self._handshake_initiation_executor.submit(
                self._nixl_handshake,
                host,
                port,
                tp_size,
                engine_id,
                dcp_size,
                pp_size,
                notif_agents_only,
                pcp_size,
                endpoint_incarnation,
            )
            self._handshake_futures[engine_id] = fut
            self._handshake_future_specs[engine_id] = spec

            def done_callback(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                eid=engine_id,
                handshake_spec=spec,
            ):
                with self._handshake_lock:
                    self._handshake_futures.pop(eid, None)
                    self._handshake_future_specs.pop(eid, None)
                    try:
                        remote_agents, clock_offset = f.result()
                        self._remote_agents[eid] = remote_agents
                        self._remote_handshake_specs[eid] = handshake_spec
                        self._stale_remote_engines.discard(eid)
                        self._engine_clock_offset[eid] = clock_offset
                        self._engine_last_active[eid] = time.perf_counter()
                    except Exception as e:
                        if not getattr(self, "_shutting_down", False):
                            self._log_failure(
                                failure_type="handshake_setup_failed",
                                req_id=None,
                                error=e,
                                remote_engine_id=eid,
                            )

            fut.add_done_callback(done_callback)
            return fut

    def _background_nixl_handshake(
        self, req_id: str, remote_engine_id: EngineId, meta: ReqMeta
    ):
        # Do NIXL handshake in background and add to _ready_requests when done.
        assert meta.remote is not None
        try:
            fut = self._ensure_handshake(
                remote_engine_id,
                meta.remote.host,
                meta.remote.port,
                meta.tp_size,
                meta.dcp_size,
                meta.pp_size,
                pcp_size=meta.pcp_size,
                endpoint_incarnation=meta.remote.endpoint_incarnation,
                request_id=req_id,
            )
        except Exception as error:
            self._log_failure(
                failure_type="handshake_spec_rejected",
                req_id=req_id,
                error=error,
                meta=meta,
            )
            self._handle_failed_transfer(req_id, None)
            return
        if fut is None:
            # Already handshaked — only happens if caller does not pre-check.
            if self._recving_metadata.get(req_id) is meta:
                self._ready_requests.put((req_id, meta))
            return

        # Check handshake success before proceeding with request.
        def request_ready(f: Future[Any], entry=(req_id, meta)):
            if getattr(self, "_shutting_down", False):
                return
            with self._handshake_lock:
                # Request IDs may be reused by retries. A late callback belongs
                # to the exact metadata object that initiated it and must never
                # start or fail a newer attempt with the same ID. Keep the check
                # and its side effect atomic with start_load_kv replacement.
                if self._recving_metadata.get(req_id) is not meta:
                    return
                try:
                    f.result()
                    self._ready_requests.put(entry)
                except Exception as e:
                    self._log_failure(
                        failure_type="handshake_failed",
                        req_id=req_id,
                        error=e,
                        meta=meta,
                    )
                    self._handle_failed_transfer(req_id, None)

        fut.add_done_callback(request_ready)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in nixl."""

        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.world_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            total_num_kv_heads=1
            if self.use_mla
            else self.model_config.get_total_num_kv_heads(),
            attn_backends=self.attn_backends,
            dcp_size=self.dcp_size,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            # SSM States come in tuples (ssm, conv)
            tensor_shape=next(iter(kv_caches.values())).shape
            if not self._has_mamba
            else None,
            is_mamba=self._has_mamba,
        )
        self.compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config,
            self.backend_name,
            transfer_mode=self._TRANSFER_MODE,
        )
        self.placement_compat_hash = compute_nixl_placement_compatibility_hash(
            self.vllm_config,
            transfer_mode=self._TRANSFER_MODE,
        )

        if self._is_csa_linear and self.use_host_buffer:
            raise NotImplementedError(
                "NIXL host staging does not preserve CSA-linear shared tensors."
            )

        if self.use_host_buffer:
            self.initialize_host_xfer_buffer(kv_caches=kv_caches)
            assert len(self.host_xfer_buffers) == len(kv_caches), (
                f"host_buffer: {len(self.host_xfer_buffers)}, "
                f"kv_caches: {len(kv_caches)}"
            )
            xfer_buffers = self.host_xfer_buffers
        else:
            xfer_buffers = kv_caches
            assert not self.host_xfer_buffers, (
                "host_xfer_buffer should not be initialized when "
                f"kv_buffer_device is {self.kv_buffer_device}"
            )

        logger.info(
            "Registering KV_Caches. use_mla: %s, kv_buffer_device: %s, "
            "use_host_buffer: %s",
            self.use_mla,
            self.kv_buffer_device,
            self.use_host_buffer,
        )

        caches_data = []
        seen_storage_addresses: set[int] = set()
        seen_base_addresses: list[int] = []
        self._ssm_region_indices = []
        self._scratch_region_indices = []
        self._ple_region_index = None

        packed_storage = _share_storage_and_block_stride(list(xfer_buffers.values()))
        # CSA-linear needs separate logical regions for attention and state
        # aliases even though every view shares one block-major allocation.
        packed_storage = packed_storage and not self._is_csa_linear

        layer_specs: dict[str, KVCacheSpec] = {}
        compressed_region_owners: dict[int, torch.Tensor] = {}
        for layer_name, cache in xfer_buffers.items():
            layer_spec = self._layer_specs.get(layer_name)
            if isinstance(layer_spec, UniformTypeKVCacheSpecs):
                layer_spec = layer_spec.kv_cache_specs[layer_name]
            if layer_spec is None:
                continue
            layer_specs[layer_name] = layer_spec
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            if _uses_dense_virtual_transfer_pages(
                layer_spec, cache, physical_page_size, num_blocks
            ):
                compressed_region_owners.setdefault(cache.data_ptr(), cache)

        # K and V are packed into the content dim, so each attention layer is a
        # single NIXL region whose block transfers as one unit. Mamba layers instead
        # register separate conv/ssm sub-regions (see `_build_mamba_local`).
        for layer_name, cache in xfer_buffers.items():
            # NOTE (NickLucche) Hybrid SSM mamba/FA physical page_size may differ when
            # kernel requires a specific block size. This leads to SSM and FA layers
            # having different num_blocks.
            # `_physical_blocks_per_logical_kv_block` ratio is used to adjust for this.
            layer_spec = layer_specs.get(layer_name)
            if layer_spec is None:
                logger.debug(
                    "Skipping layer %s as no KVCache spec is present. "
                    "This is likely because the layer is sharing its KV cache",
                    layer_name,
                )
                continue
            # `layer_spec.page_size_bytes` only accounts for logical page_size, that is
            # the page_size assuming constant `self._logical_num_blocks`.
            physical_page_size = (
                layer_spec.page_size_bytes
                if isinstance(layer_spec, MambaSpec)
                else layer_spec.page_size_bytes
                // self._physical_blocks_per_logical_kv_block
            )
            num_blocks = (
                self._logical_num_blocks
                if isinstance(layer_spec, MambaSpec)
                else self.num_blocks
            )
            logger.debug(
                "Registering layer %s with cache shape: %s", layer_name, cache.shape
            )
            storage = cache.untyped_storage()
            storage_addr = storage.data_ptr()

            if isinstance(layer_spec, KpoolTailSpec):
                compressed_owner = compressed_region_owners.get(cache.data_ptr())
                if compressed_owner is not None:
                    owner_storage = compressed_owner.untyped_storage()
                    owner_end = compressed_owner.data_ptr() + compressed_owner.nbytes
                    tail_is_covered = (
                        compressed_owner.is_contiguous()
                        and owner_storage.data_ptr() == storage_addr
                        and _tensor_byte_span_end(cache) <= owner_end
                    )
                    if not tail_is_covered:
                        raise AssertionError(
                            "Kpool tail cache is not fully covered by its compressed "
                            f"indexer region: layer={layer_name}, "
                            f"tail_shape={tuple(cache.shape)}, "
                            f"tail_stride={tuple(cache.stride())}, "
                            f"owner_shape={tuple(compressed_owner.shape)}, "
                            f"owner_stride={tuple(compressed_owner.stride())}"
                        )
                    logger.debug(
                        "Skipping layer %s because its compressed indexer region "
                        "covers the same storage",
                        layer_name,
                    )
                    continue

            # Memory registration follows allocations, while transfer regions follow
            # logical layers (or contiguous head segments). This keeps strided
            # cross-layer views inside their registered allocation.
            if storage_addr not in seen_storage_addresses:
                seen_storage_addresses.add(storage_addr)
                self.device_id = max(cache.get_device(), 0)
                caches_data.append((storage_addr, storage.nbytes(), self.device_id, ""))

            is_mla_region = isinstance(
                layer_spec, (MLAAttentionSpec, SlidingWindowMLASpec)
            )

            if isinstance(layer_spec, MambaSpec):
                physical_ratio = self._physical_blocks_per_logical_kv_block
                block_len = physical_page_size // physical_ratio
                block_stride = physical_page_size
                if self._is_csa_linear and cache.ndim != 1:
                    # CSA-linear packs multiple cache owners in each block, so
                    # state views advance by the packed block stride.
                    block_stride = cache.stride(0) * cache.element_size()
                assert block_stride % physical_ratio == 0
                region_specs = [
                    (cache.data_ptr(), block_len, block_stride // physical_ratio)
                ]
            else:
                if cache.ndim == 1:
                    # Flat byte view: HMA tensors shared between layer types carry
                    # no block dimension, so stride(0) is 1 byte. Blocks abut, so
                    # the stride is the page size.
                    block_stride = physical_page_size
                else:
                    block_stride = cache.stride(0) * cache.element_size()
                storage_is_block_major = num_blocks * block_stride == storage.nbytes()
                # A layer whose [H, N, C] interior is dense addresses its own page
                # as one chunk. Otherwise the block's whole row is the only
                # contiguous transfer unit.
                hnc_contiguous = (
                    cache.ndim == 4
                    and cache.stride(2) == cache.shape[3]
                    and cache.stride(1) == cache.shape[2] * cache.shape[3]
                )
                virtual_transfer_pages = _uses_dense_virtual_transfer_pages(
                    layer_spec, cache, physical_page_size, num_blocks
                )
                if virtual_transfer_pages:
                    # A compressed kernel row can contain multiple NIXL transfer pages.
                    region_specs = [
                        (cache.data_ptr(), physical_page_size, physical_page_size)
                    ]
                elif storage_is_block_major and (
                    (packed_storage and is_mla_region)
                    or (not hnc_contiguous and not self._is_csa_linear)
                ):
                    # Packed MLA layouts transfer the complete storage row.
                    storage_block_len = storage.nbytes() // num_blocks
                    region_specs = [
                        (storage_addr, storage_block_len, storage_block_len)
                    ]
                elif storage_is_block_major:
                    region_specs = [
                        (cache.data_ptr(), physical_page_size, block_stride)
                    ]
                else:
                    segment_bytes = num_blocks * block_stride
                    if cache.nbytes % segment_bytes != 0:
                        raise AssertionError(
                            "KV cache view cannot be partitioned into NIXL regions: "
                            f"layer={layer_name}, cache_nbytes={cache.nbytes}, "
                            f"num_blocks={num_blocks}, block_stride={block_stride}, "
                            f"physical_page_size={physical_page_size}, "
                            f"cache_shape={tuple(cache.shape)}, "
                            f"cache_stride={tuple(cache.stride())}"
                        )
                    num_segments = cache.nbytes // segment_bytes
                    region_block_len = (
                        block_stride if num_segments > 1 else physical_page_size
                    )
                    region_specs = [
                        (
                            cache.data_ptr() + segment_idx * segment_bytes,
                            region_block_len,
                            block_stride,
                        )
                        for segment_idx in range(num_segments)
                    ]

            for base_addr, block_len, block_stride in region_specs:
                if base_addr in seen_base_addresses:
                    region_index = seen_base_addresses.index(base_addr)
                    self._region_is_mla[region_index] |= is_mla_region
                else:
                    region_index = len(seen_base_addresses)
                    seen_base_addresses.append(base_addr)
                    self.block_len_per_layer.append(block_len)
                    self.block_stride_per_layer.append(block_stride)
                    self._region_is_mla.append(is_mla_region)

                if isinstance(layer_spec, MambaSpec):
                    if layer_spec.tp_replicated:
                        assert self._is_csa_linear
                        assert self._ple_region_index in (None, region_index)
                        self._ple_region_index = region_index
                    elif region_index not in self._ssm_region_indices:
                        self._ssm_region_indices.append(region_index)
                elif (
                    isinstance(layer_spec, CircularBufferSpec)
                    and region_index not in self._scratch_region_indices
                ):
                    self._scratch_region_indices.append(region_index)

            # When there's a mismatch between kbs<>bs, we rely on HMA to ensure
            # caches are either [NB, PS] or [NB*r, PS/r] where r is bs/kbs.
            if (
                self._physical_blocks_per_logical_kv_block == 1
                and cache.shape[0] != num_blocks
            ):
                raise AssertionError(
                    "All kv cache tensors must have the same number of "
                    f"blocks; layer={layer_name}, "
                    f"expected_num_blocks={num_blocks}, "
                    f"cache_shape={tuple(cache.shape)}, "
                    f"cache_stride={tuple(cache.stride())}, "
                    f"layer_spec={type(layer_spec).__name__}, "
                    f"backend={self.backend_name}, "
                    "all_backends="
                    f"{[backend.get_name() for backend in self.attn_backends]}, "
                    f"kv_cache_layout={self.kv_cache_layout}"
                )

        logger.debug(
            "Different block lengths collected: %s", set(self.block_len_per_layer)
        )
        assert (
            len(self.block_len_per_layer)
            == len(seen_base_addresses)
            == len(self._region_is_mla)
            == len(self.block_stride_per_layer)
        )
        # Descriptor ids must be region-ordered, matching the remote side.
        self._scratch_region_indices.sort()

        self.kv_caches_base_addr[self.engine_id][self.tp_rank] = seen_base_addresses
        self.num_regions = len(seen_base_addresses)

        if self.pp_size > 1:
            start_layer, end_layer = self.model_config.get_layers_start_end_indices(
                self.vllm_config.parallel_config
            )
            num_local_layers = end_layer - start_layer
            assert num_local_layers > 0 and self.num_regions % num_local_layers == 0
            regions_per_layer = self.num_regions // num_local_layers
            self._remote_region_offset = regions_per_layer * start_layer

        # Total local FA descriptors (boundary between FA and mamba descs).
        self.num_descs = self.num_regions * self.num_blocks

        generic_activation_requested = (
            self._TRANSFER_MODE == "pull" and self._enable_generic_placement
        )
        registration_error: Exception | None = None
        descs = None
        try:
            descs = self.nixl_wrapper.get_reg_descs(caches_data, self.nixl_memory_type)
            logger.debug("Registering descs: %s", caches_data)
            self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
            logger.debug("Done registering descs")
            self._registered_descs.append(descs)
        except Exception as error:
            if not generic_activation_requested:
                raise
            registration_error = error
            if descs is not None:
                try:
                    self.nixl_wrapper.deregister_memory(descs)
                except Exception:
                    logger.warning(
                        "Failed to roll back NIXL memory registration",
                        exc_info=True,
                    )
            # A failed rank must still join the generic placement collectives so
            # its peers fail uniformly instead of hanging during startup.
            logger.warning(
                "Failed to register NIXL memory for generic placement",
                exc_info=True,
            )

        self.device_kv_caches = kv_caches
        self.dst_num_blocks[self.engine_id] = self.num_blocks

        local_dlist_error: Exception | None = None
        if registration_error is None:
            try:
                # Prepare the legacy fast-path descriptor eagerly when
                # possible. Generic segmented descriptors only require the
                # registered memory and canonical placement below, so failure
                # here disables legacy admission on this worker rather than
                # disabling the generic path as well.
                (
                    self.src_xfer_handles_by_block_size[self.block_size],
                    self.src_blocks_data,
                ) = self.register_local_xfer_handler(self.block_size)
            except Exception as error:
                if not generic_activation_requested:
                    raise
                local_dlist_error = error
                logger.warning(
                    "Failed to prepare the legacy local NIXL descriptor list; "
                    "generic segmented-direct placement remains eligible",
                    exc_info=True,
                )
        self._legacy_fast_path_available = (
            registration_error is None and local_dlist_error is None
        )

        self._local_placement_metadata = None
        self._local_placement_workers = ()
        if generic_activation_requested:
            parallel = self.vllm_config.parallel_config
            candidate: NixlPlacementMetadata | None = None
            if registration_error is None and not self.use_host_buffer:
                try:
                    candidate = build_runtime_nixl_placement(
                        vllm_config=self.vllm_config,
                        kv_cache_config=self.kv_cache_config,
                        caches=kv_caches,
                        deployment_id=self.engine_id,
                        topology_generation=0,
                        worker_id=(
                            f"{self.engine_id}:dp{parallel.data_parallel_rank}:"
                            f"pp{self.pp_rank}:pcp{self.pcp_rank}:"
                            f"tp{self.tp_rank}:ep{self.ep_rank}"
                        ),
                        worker_incarnation=self._placement_worker_incarnation,
                        tp_rank=self.tp_rank,
                        pcp_rank=self.pcp_rank,
                        pp_rank=self.pp_rank,
                        dp_rank=parallel.data_parallel_rank,
                        ep_size=self.ep_size,
                        ep_rank=self.ep_rank,
                        physical_pages_per_logical=(
                            self._physical_blocks_per_logical_kv_block
                        ),
                        max_segments_per_batch=(
                            self.kv_transfer_config.get_from_extra_config(
                                "max_segments_per_batch", 4096
                            )
                        ),
                    )
                except NixlRuntimePlacementUnsupported as error:
                    logger.info(
                        "Generic segmented-direct NIXL placement is unavailable: %s",
                        error,
                    )
                except Exception:
                    # Every TP rank must still enter the metadata collective;
                    # otherwise one unexpected derivation error deadlocks its
                    # peers during startup. Disable the optional path uniformly.
                    logger.warning(
                        "Failed to derive generic segmented-direct placement",
                        exc_info=True,
                    )

            stage_candidates: list[NixlPlacementMetadata | None]
            if self.world_size == 1:
                stage_candidates = [candidate]
            else:
                stage_candidates = [None] * self.world_size
                torch.distributed.all_gather_object(
                    stage_candidates,
                    candidate,
                    group=get_tp_group().cpu_group,
                )

            # TP groups hold one fixed PP/PCP coordinate. PCP groups then span
            # those complete TP tuples at a fixed PP/TP coordinate. This gives
            # every stage worker the same PCP-major, TP-minor cohort without a
            # world-level collective or any KV-data all-gather.
            if self.pcp_size == 1:
                pcp_stage_candidates = stage_candidates
            else:
                gathered_pcp_candidates: list[
                    tuple[NixlPlacementMetadata | None, ...] | None
                ] = [None] * self.pcp_size
                torch.distributed.all_gather_object(
                    gathered_pcp_candidates,
                    tuple(stage_candidates),
                    group=get_pcp_group().cpu_group,
                )
                malformed_pcp = next(
                    (
                        pcp_rank
                        for pcp_rank, cohort in enumerate(gathered_pcp_candidates)
                        if not isinstance(cohort, tuple)
                        or len(cohort) != self.world_size
                    ),
                    None,
                )
                if malformed_pcp is not None:
                    raise RuntimeError(
                        "prefill-context-parallel generic NIXL gathered a "
                        f"malformed TP cohort for PCP rank {malformed_pcp}"
                    )
                pcp_stage_candidates = [
                    worker
                    for cohort in gathered_pcp_candidates
                    for worker in cast(tuple[NixlPlacementMetadata | None, ...], cohort)
                ]

            # PP groups hold one fixed PCP/TP coordinate. Gathering the complete
            # stage cohort across PP gives every endpoint worker deterministic
            # PP-major, PCP-major, TP-minor metadata.
            stage_size = self.pcp_size * self.world_size
            if self.pp_size == 1:
                gathered_candidates = pcp_stage_candidates
            else:
                gathered_stage_candidates: list[
                    tuple[NixlPlacementMetadata | None, ...] | None
                ] = [None] * self.pp_size
                torch.distributed.all_gather_object(
                    gathered_stage_candidates,
                    tuple(pcp_stage_candidates),
                    group=get_pp_group().cpu_group,
                )
                malformed_stage = next(
                    (
                        pp_rank
                        for pp_rank, stage in enumerate(gathered_stage_candidates)
                        if not isinstance(stage, tuple) or len(stage) != stage_size
                    ),
                    None,
                )
                if malformed_stage is not None:
                    raise RuntimeError(
                        "pipeline-parallel generic NIXL gathered a malformed "
                        f"PCP/TP cohort for PP rank {malformed_stage}"
                    )
                gathered_candidates = [
                    worker
                    for stage in gathered_stage_candidates
                    for worker in cast(tuple[NixlPlacementMetadata | None, ...], stage)
                ]

            if all(item is not None for item in gathered_candidates):
                try:
                    finalized_candidates = finalize_nixl_placement_cohort(
                        cast(list[NixlPlacementMetadata], gathered_candidates)
                    )
                    placement_workers = validate_complete_nixl_placement_endpoint(
                        finalized_candidates,
                        dp_rank=parallel.data_parallel_rank,
                    )
                    local_candidates = tuple(
                        worker
                        for worker in placement_workers
                        if (
                            worker.rank_placement.pp_rank == self.pp_rank
                            and worker.rank_placement.pcp_rank == self.pcp_rank
                            and worker.rank_placement.tp_rank == self.tp_rank
                        )
                    )
                    if len(local_candidates) != 1:
                        raise ValueError(
                            "local PP/PCP/TP coordinates do not identify exactly "
                            "one gathered placement worker"
                        )
                    self._local_placement_metadata = local_candidates[0]
                    self._local_placement_workers = placement_workers
                    logger.info(
                        "Enabled generic segmented-direct NIXL placement on "
                        "PP rank %s, PCP rank %s, TP rank %s",
                        self.pp_rank,
                        self.pcp_rank,
                        self.tp_rank,
                    )
                except Exception as error:
                    logger.warning(
                        "Gathered generic NIXL placements failed endpoint validation",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        "generic NIXL placement failed endpoint validation"
                    ) from error
            elif any(item is not None for item in gathered_candidates):
                logger.warning(
                    "Generic NIXL placement was not derivable on every PP/PCP/TP rank"
                )
                raise RuntimeError(
                    "generic NIXL placement must be derivable on every PP/PCP/TP rank"
                )
            else:
                raise RuntimeError(
                    "generic NIXL placement is unavailable on every PP/PCP/TP rank"
                )

        if self._has_mamba:
            logger.info(
                "Hybrid SSM registration: num_blocks=%s, "
                "logical_num_blocks=%s, ratio=%s, num_regions=%s, "
                "num_descs=%s, mamba_ssm_size=%s, block_len_per_layer=%s",
                self.num_blocks,
                self._logical_num_blocks,
                self._physical_blocks_per_logical_kv_block,
                self.num_regions,
                self.num_descs,
                self._mamba_ssm_size,
                set(self.block_len_per_layer),
            )

        # After KV Caches registered, listen for new connections.
        agent_metadata = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=self.nixl_wrapper.get_agent_metadata(),
            device_id=self.device_id,
            kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][self.tp_rank],
            num_blocks=self.num_blocks,
            block_lens=self.block_len_per_layer,
            block_strides=self.block_stride_per_layer,
            kv_cache_layout=self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout,
            block_size=self.block_size,
            ssm_sizes=self._mamba_ssm_size,
            attn_backend_name=self.backend_name,
            physical_blocks_per_logical_kv_block=(
                self._physical_blocks_per_logical_kv_block
            ),
            dcp_size=self.dcp_size,
            pcp_size=self.pcp_size,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            placement_metadata=self._local_placement_metadata,
        )
        # Wrap metadata in payload with hash for defensive decoding
        assert self.compat_hash is not None
        assert self.placement_compat_hash is not None
        encoder = msgspec.msgpack.Encoder()
        self.xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=self.compat_hash,
            placement_compatibility_hash=self.placement_compat_hash,
            agent_metadata_bytes=encoder.encode(agent_metadata),
        )

    def _build_mamba_local(self, base_addresses: list[int]) -> np.ndarray:
        """Build desc regions (conv sub-projections + ssm) per layer for
        local mamba blocks with DS conv layout, as an Nx3 uint64 array.

        A Mamba block interleaves conv and SSM state, which crucially differ in
        size, so the two are indexed as separate sub-regions. Attention blocks
        instead pack K and V into the content dim and transfer as a single unit.
        Reference diagram:
                            KVCacheTensor (Shared)
                               /       \\
                              /         \\
                             /           \\
        Attention (FlashInfer) View      Mamba View
                  |                          |
                  |                          |
           +-------------------+         +-------------------+
           | KVCacheTensor     |         | KVCacheTensor      |
           |                   |         |                    |
           |<----- page ------>|         |<----- page ------->|
           |       size        |         |       size         |
           |  Key 0  |  Val 0  |         |Conv 0  |   SSM 0   |
           |  Key 1  |  Val 1  |         |Conv 1  |   SSM 1   |
           |   ...   |   ...   |         |  ...   |    ...    |
           | Key N-2 | Val N-2 |         |Conv N-2|   SSM N-2 |
           | Key N-1 | Val N-1 |         |Conv N-1|   SSM N-1 |
           +-------------------+         +--------------------+
           |1st_split-2nd_split|         |1st_split-2nd_split |

        Mamba state blocks are indivisible (not token-extent data), so the
        descriptors always use the local page geometry regardless of any
        attention block-size ratio; their desc ids are likewise never
        ratio-expanded (see _compute_desc_ids).
        """
        assert base_addresses, "Local KV cache base addresses must not be empty."
        assert self._conv_decomp is not None
        conv_offsets = self._conv_decomp.local_conv_offsets
        conv_size, ssm_size = self._mamba_ssm_size
        num_blocks = self._logical_num_blocks
        physical_per_logical = self._physical_blocks_per_logical_kv_block
        device_id = self.device_id
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        parts: list[np.ndarray] = []

        region_indices = self._ssm_region_indices or range(len(base_addresses))
        for i in region_indices:
            base_addr = base_addresses[i]
            block_stride = self.block_stride_per_layer[i] * physical_per_logical
            blk_addrs = base_addr + block_arange * block_stride
            for off, sz in conv_offsets:
                parts.append(self._stack_descs(blk_addrs + off, sz, device_id))
            # SSM temporal state follows the conv state.
            parts.append(self._stack_descs(blk_addrs + conv_size, ssm_size, device_id))

        if (region_index := self._ple_region_index) is not None:
            block_len = self.block_len_per_layer[region_index] * physical_per_logical
            block_stride = (
                self.block_stride_per_layer[region_index] * physical_per_logical
            )
            block_addrs = base_addresses[region_index] + block_arange * block_stride
            parts.append(self._stack_descs(block_addrs, block_len, self.device_id))

        return np.concatenate(parts)

    def _build_mamba_remote(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        tp_ratio: int,
        transfer_info: EngineTransferInfo,
    ) -> np.ndarray:
        """Build remote desc regions (conv sub-projections + ssm) per layer.
        For hetero-TP, each D rank reads only its sub-projection slice from
        the P rank. Returns an Nx3 uint64 array."""
        assert nixl_agent_meta.kv_caches_base_addr, (
            "Remote KV cache base addresses must not be empty."
        )
        assert self._conv_decomp is not None
        effective_ratio = max(tp_ratio, 1)
        # Mamba conv state is always TP-sharded, even when attention KV
        # is replicated (num_kv_heads < tp_size).
        local_offset = self.tp_rank % effective_ratio
        conv_size_remote = nixl_agent_meta.ssm_sizes[0]

        conv_offsets = self._conv_decomp.remote_conv_offsets(local_offset, tp_ratio)
        if tp_ratio >= 1:
            ssm_read_size = self._mamba_ssm_size[1]
        else:
            ssm_read_size = nixl_agent_meta.ssm_sizes[1]

        remote_physical_per_logical = transfer_info.remote_physical_blocks_per_logical
        num_blocks = nixl_agent_meta.num_blocks // remote_physical_per_logical
        device_id = nixl_agent_meta.device_id
        block_arange = np.arange(num_blocks, dtype=np.uint64)

        parts: list[np.ndarray] = []
        # NOTE (ZhanqiuHu): use per-layer block_lens[i], not [0], in case
        # block lengths vary across layers (e.g. MLA).
        region_indices = self._ssm_region_indices or range(
            len(nixl_agent_meta.kv_caches_base_addr)
        )
        for i in region_indices:
            base_addr = nixl_agent_meta.kv_caches_base_addr[i]
            block_stride = (
                nixl_agent_meta.block_strides[i] * remote_physical_per_logical
            )
            blk_addrs = base_addr + block_arange * block_stride
            for off, sz in conv_offsets:
                parts.append(self._stack_descs(blk_addrs + off, sz, device_id))
            # SSM temporal state is also TP-sharded on the heads dimension.
            ssm_addrs = blk_addrs + conv_size_remote + local_offset * ssm_read_size
            parts.append(self._stack_descs(ssm_addrs, ssm_read_size, device_id))

        if (region_index := self._ple_region_index) is not None:
            local_block_len = (
                self.block_len_per_layer[region_index]
                * self._physical_blocks_per_logical_kv_block
            )
            remote_block_len = (
                nixl_agent_meta.block_lens[region_index] * remote_physical_per_logical
            )
            if local_block_len != remote_block_len:
                raise ValueError(
                    "PLE pages require identical P/D geometry: "
                    f"local={local_block_len}, remote={remote_block_len}."
                )
            remote_block_stride = (
                nixl_agent_meta.block_strides[region_index]
                * remote_physical_per_logical
            )
            block_addrs = (
                nixl_agent_meta.kv_caches_base_addr[region_index]
                + block_arange * remote_block_stride
            )
            parts.append(self._stack_descs(block_addrs, remote_block_len, device_id))

        return np.concatenate(parts)

    @staticmethod
    def _stack_descs(addrs: np.ndarray, length: int, device_id: int) -> np.ndarray:
        out = np.empty((addrs.shape[0], 3), dtype=np.uint64)
        out[:, 0] = addrs
        out[:, 1] = length
        out[:, 2] = device_id
        return out

    def _build_fa_local(
        self,
        base_addresses: list[int],
        block_size_ratio: int,
    ) -> np.ndarray:
        """Build local FA descriptors for all layers as an Nx3 uint64 array."""
        assert self.transfer_topo is not None
        assert base_addresses, "Local KV cache base addresses must not be empty."
        num_blocks = self.num_blocks * block_size_ratio
        device_id = self.device_id
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        parts: list[np.ndarray] = []
        for i, base_addr in enumerate(base_addresses):
            block_len = self.block_len_per_layer[i] // block_size_ratio
            block_stride = self.block_stride_per_layer[i] // block_size_ratio
            addrs = base_addr + block_arange * block_stride
            parts.append(self._stack_descs(addrs, block_len, device_id))
        return np.concatenate(parts)

    def _build_fa_remote(
        self,
        plan: TPMapping,
        nixl_agent_meta: NixlAgentMetadata,
        block_size_ratio: int,
    ) -> np.ndarray:
        """Build remote FA descriptors for all layers as an Nx3 uint64 array."""
        assert self.transfer_topo is not None
        assert nixl_agent_meta.kv_caches_base_addr, (
            "Remote KV cache base addresses must not be empty."
        )
        fa_group_idx = next(
            i for i, t in enumerate(self._group_spec_types) if _is_attention_spec(t)
        )
        # SPLIT regions read their head slice from this many remote ranks at a
        # per-rank offset; REPLICATE regions read the whole block once.
        split_reads = len(plan.source_ranks_per_group[fa_group_idx])
        num_blocks = nixl_agent_meta.num_blocks
        device_id = nixl_agent_meta.device_id
        block_arange = np.arange(num_blocks, dtype=np.uint64)
        parts: list[np.ndarray] = []
        for i, base_addr in enumerate(nixl_agent_meta.kv_caches_base_addr):
            replicated = self._is_region_replicated(i)
            # Read our whole local region size from remote..
            local_block_len = self.block_len_per_layer[i]
            remote_kv_block_len = local_block_len // block_size_ratio
            if block_size_ratio > 1:
                # ..using remote kv_block_len as transfer unit
                local_block_len = remote_kv_block_len

            # REPLICATE reads the whole block once at offset 0; SPLIT gathers
            # its head slice from `split_reads` remote ranks at a per-rank offset.
            num_reads = 1 if replicated else split_reads
            rank_offset = (
                0 if replicated else plan.rank_offset_factor * remote_kv_block_len
            )
            local_block_len = local_block_len // num_reads

            page_size = nixl_agent_meta.block_strides[i]
            addrs = base_addr + rank_offset + block_arange * page_size
            parts.append(self._stack_descs(addrs, local_block_len, device_id))
        return np.concatenate(parts)

    def register_local_xfer_handler(
        self,
        block_size: int,
    ) -> tuple[int, np.ndarray]:
        """
        Function used for register local xfer handler with local block_size or
        Remote block_size.

        When local block_size is same as remote block_size, we use local block_size
        to register local_xfer_handler during init.

        When remote block size is less than local block size, we need to use
        register another local_xfer_handler using remote block len to ensure
        data copy correctness.
        """
        assert self.transfer_topo is not None
        block_size_ratio = self.block_size // block_size
        local_base_addresses = self.kv_caches_base_addr[self.engine_id][self.tp_rank]

        blocks_data = self._build_fa_local(local_base_addresses, block_size_ratio)
        logger.debug(
            "Created %s blocks for src engine %s and rank %s on device id %s",
            len(blocks_data),
            self.engine_id,
            self.tp_rank,
            self.device_id,
        )
        if self._has_mamba:
            assert self.num_descs * block_size_ratio == len(blocks_data)
            # TODO (ZhanqiuHu): For homogeneous TP (tp_ratio == 1), the 3-descs split
            # is unnecessary — a single conv desc per block suffices.  Consider
            # adding a fast path that falls back to the standard 2-region
            # registration (_build_fa_local mamba=True) when no hetero-TP
            # remote has been seen.  Currently we always register 4 regions
            # because local descs are created before knowing the remote TP.
            logger.debug("Registering local Mamba descriptors (4 regions/layer)")
            mamba = self._build_mamba_local(local_base_addresses)
            blocks_data = np.concatenate([blocks_data, mamba])

        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        # NIXL_INIT_AGENT to be used for preparations of local descs.
        handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
        if handle is None:
            raise RuntimeError("NIXL failed to prepare local descriptors")
        return handle, blocks_data

    def _prepare_local_split_xfer_handlers(
        self,
        plan: TPMapping,
        src_blocks_data: np.ndarray,
        num_fa_descs: int,
        block_size_ratio: int,
    ) -> list[int]:
        """Prepare one split-handle cohort without publishing partial state."""
        prepared_handles: list[int] = []
        expected_handles = len(plan.all_source_ranks)
        if expected_handles <= 0:
            raise RuntimeError("NIXL local split plan has no source ranks")
        try:
            for handle_data in self._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_fa_descs,
                block_size_ratio,
            ):
                descs = self.nixl_wrapper.get_xfer_descs(
                    handle_data, self.nixl_memory_type
                )
                handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
                if handle is None:
                    raise RuntimeError("NIXL failed to prepare local split descriptors")
                prepared_handles.append(handle)
            if len(prepared_handles) != expected_handles:
                raise RuntimeError(
                    "NIXL local split descriptor count does not match its TP plan: "
                    f"prepared={len(prepared_handles)}, expected={expected_handles}"
                )
        except BaseException:
            for handle in prepared_handles:
                try:
                    self.nixl_wrapper.release_dlist_handle(handle)
                except Exception:
                    logger.warning(
                        "Failed to release a partial local split descriptor list",
                        exc_info=True,
                    )
            raise
        return prepared_handles

    def _add_generic_remote_agent(
        self,
        metadata: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int,
    ) -> str:
        """Register an agent without legacy page-copy descriptor setup."""
        assert self.transfer_topo is not None
        self.transfer_topo.register_remote_engine(
            metadata.engine_id,
            EngineTransferInfo(
                remote_tp_size=remote_tp_size,
                remote_block_size=metadata.block_size,
                remote_block_len=metadata.block_lens[0],
                remote_physical_blocks_per_logical=(
                    metadata.physical_blocks_per_logical_kv_block
                ),
                remote_dcp_size=remote_dcp_size,
                remote_cp_kv_cache_interleave_size=(
                    metadata.cp_kv_cache_interleave_size
                ),
            ),
        )
        self.dst_num_blocks.setdefault(metadata.engine_id, metadata.num_blocks)
        return self.nixl_wrapper.add_remote_agent(metadata.agent_metadata)

    def _generic_registration_available(self, metadata: NixlAgentMetadata) -> bool:
        """Whether the current runtime bridge can address both page spaces.

        Each side advertises physical-page geometry independently, so different
        scheduler-to-kernel page ratios remain directly composable.
        """
        return (
            self._local_placement_metadata is not None
            and metadata.placement_metadata is not None
        )

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
        remote_dcp_size: int = 1,
        *,
        generic_registration: bool | None = None,
    ) -> str:
        """
        Add the remote NIXL agent and prepare the descriptors for reading cache
        blocks from remote.

        In particular, handle both homogeneous and heterogeneous TP. The former
        requires local rank_i to read from remote rank_i.
        The latter, in the case of D.world_size < P.world_size, requires that a
        local (D) TP worker reads from multiple remote (P) TP workers.
        Conversely, assuming D.world_size > P.world_size, two or more local TP
        workers will read from a single remote TP worker.

        Here's an example for the last case described above (non-MLA):

        rank_offset     p_remote_tp_rank
        (kv split no)
        --------------------------------
            0                 0      Worker0  ---- 1st half of KV ----> Worker0  [ KV Cache ]
                                                                        /
            1                 0      Worker1  ---- 2nd half of KV -----/

            0                 1      Worker2  ---- 1st half of KV ----> Worker1  [ KV Cache ]
                                                                        /
            1                 1      Worker3  ---- 2nd half of KV -----/


                                Decoder TP workers                     Prefix TP workers
                                  (world_size=4)                         (world_size=2)
                                                 tp_ratio = 4 // 2 = 2

        Considering the KV Caches, if P-Worker_i has cache size [2, num_blocksP, kv_heads, block_size, head_dim]
        then D-Worker_j has [2, num_blocksD, kv_heads//tp_ratio, block_size, head_dim]. Mind the "LBHNC" layout format.
        Assuming num_blocksD >= num_blocksP, D-Worker0 reads from P-Worker0 by preparing the kv_heads//tp_ratio
        first heads from all the slots of all the blocks. D-Worker1 will do the same, but reading the second split
        along the kv_heads dimension, and so forth until "tp_ratio" D TP workers have pulled from P-Worker0.

        Note that the above will also hold true for the homogeneous TP case, where tp_ratio evaluates to 1.

        Regarding MLA case, the cache is replicated across TP workers so the rank_offset will just always be 0
        so that the whole cache is shared by "tp_ratio" D TP workers.

        For Mamba hetero-TP, both tp_ratio > 0 (D_TP > P_TP) and
        tp_ratio < 0 (P_TP > D_TP) are supported by the 3-read transfer.
        """  # noqa: E501
        engine_id = nixl_agent_meta.engine_id
        # TODO re-evaluate refreshing for scaling/recovery
        if (0, remote_tp_rank) in self._remote_agents.get(engine_id, {}):
            logger.debug(
                "Remote agent with engine_id %s and rank"
                "%s already exchanged metadata, skip handshake.",
                engine_id,
                remote_tp_rank,
            )
            return self._remote_agents[engine_id][(0, remote_tp_rank)]

        generic_registration_available = self._generic_registration_available(
            nixl_agent_meta
        )
        if generic_registration is True and not generic_registration_available:
            raise RuntimeError(
                "generic NIXL registration requires placement metadata on both "
                "endpoints"
            )
        if generic_registration is True or (
            generic_registration is None and generic_registration_available
        ):
            return self._add_generic_remote_agent(
                nixl_agent_meta, remote_tp_size, remote_dcp_size
            )

        # Number of physical regions registered locally (one per layer/tensor).
        num_local_regions = len(self.block_len_per_layer)
        if (
            self.pp_size > 1
            and len(nixl_agent_meta.kv_caches_base_addr) > num_local_regions
        ):
            # This worker holds a PP layer-slice; the PP=1 remote registered
            # the full model. Slice its regions to our layer window so the
            # logic below sees congruent local/remote lists.
            start = self._remote_region_offset
            end = start + num_local_regions
            assert len(nixl_agent_meta.kv_caches_base_addr) >= end
            nixl_agent_meta.kv_caches_base_addr = nixl_agent_meta.kv_caches_base_addr[
                start:end
            ]
            nixl_agent_meta.block_lens = nixl_agent_meta.block_lens[start:end]
            nixl_agent_meta.block_strides = nixl_agent_meta.block_strides[start:end]

        ### Register remote engine in TransferTopology (idempotent).
        assert self.transfer_topo is not None
        transfer_topo = self.transfer_topo
        physical_blocks_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        transfer_info = EngineTransferInfo(
            remote_tp_size=remote_tp_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_block_len=nixl_agent_meta.block_lens[0],
            remote_physical_blocks_per_logical=physical_blocks_per_logical,
            remote_dcp_size=remote_dcp_size,
            remote_cp_kv_cache_interleave_size=(
                nixl_agent_meta.cp_kv_cache_interleave_size
            ),
        )
        transfer_topo.register_remote_engine(engine_id, transfer_info)
        logger.info("Transfer plan: %s", transfer_topo.describe(engine_id))

        self.tp_mappings[engine_id] = compute_tp_mapping(
            transfer_topology=transfer_topo,
            remote_tp_size=remote_tp_size,
            group_spec_types=self._group_spec_types,
            remote_dcp_size=remote_dcp_size,
        )

        # Create dst descs and xfer side handles. TP workers have same #blocks
        # so we only register once per engine_id.
        # Example:
        # block_size_ratio > 1:
        # remote:               | 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|
        # local origin:|          0|          1|          8|         12|
        # local mapped:| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
        block_size_ratio = transfer_topo.block_size_ratio(nixl_agent_meta.block_size)

        if engine_id not in self.dst_num_blocks:
            self.dst_num_blocks[engine_id] = nixl_agent_meta.num_blocks

        # Keep track of remote agent kv caches base addresses.
        self.kv_caches_base_addr[engine_id][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )
        self._validate_remote_agent_handshake(
            nixl_agent_meta, remote_tp_size, remote_dcp_size
        )

        # This is 1 when P and D `--tensor-parallel-size` match. Otherwise,
        # this is the ratio between the two sizes.
        tp_ratio = transfer_topo.tp_ratio(remote_tp_size)

        logger.debug(
            "Registering remote agent (%s, rank %s) memory regions with tp_ratio %s",
            engine_id,
            remote_tp_rank,
            tp_ratio,
        )

        plan = self.tp_mappings[engine_id]

        ### (Optional) Register a local handler at the remote engine's block
        ### granularity (remote/prefill blocks smaller than local).
        remote_block_size = nixl_agent_meta.block_size
        src_blocks_data = self.src_blocks_data
        if block_size_ratio > 1:
            if remote_block_size not in self.src_xfer_handles_by_block_size:
                handle, blocks_data = self.register_local_xfer_handler(
                    remote_block_size
                )
                self.src_xfer_handles_by_block_size[remote_block_size] = handle
                self.src_blocks_data_by_block_size[remote_block_size] = blocks_data
            src_blocks_data = self.src_blocks_data_by_block_size[remote_block_size]

        ### (Optional) Register local agent memory regions. MLA is not split.
        split_key = self._split_local_xfer_handle_key(tp_ratio, remote_block_size, plan)
        if self._needs_split_local_xfer_handles(tp_ratio, plan) and (
            split_key not in self.src_xfer_handles_by_tp_ratio
        ):
            # Remote tp_size > local tp_size: read from multiple remote ranks.
            # Logically "split" own regions into per-source chunks. Hybrid
            # MLA+SSM also needs this path: MLA is replicated and read once,
            # while the SSM state is sharded across every remote TP rank.
            # We only do this once per remote (tp_size, block_size).
            split_handles = self._prepare_local_split_xfer_handlers(
                plan,
                src_blocks_data,
                self.num_descs * block_size_ratio,
                block_size_ratio,
            )
            self.src_xfer_handles_by_tp_ratio[split_key] = split_handles

        ### Register remote agent memory regions
        # With homogeneous TP, D pulls the whole kv cache from corresponding rank. With
        # heterogeneous TP, prepare the descriptors by splitting the P KV cache along
        # kv_head dim, of D worker's kv_head size (D>P).
        # Eg. PTP1 DTP2 => P0 KV:[block0-KV_0 | block0-KV_1..].

        # Register all remote blocks, but only the corresponding kv heads.
        blocks_data = self._build_fa_remote(
            plan,
            nixl_agent_meta,
            block_size_ratio,
        )
        logger.debug(
            "Created %s blocks for dst engine %s with remote rank %s and local rank %s",
            len(blocks_data),
            engine_id,
            remote_tp_rank,
            self.tp_rank,
        )
        if self._has_mamba:
            logger.debug(
                "Registering remote Mamba blocks for engine %s rank %s",
                engine_id,
                remote_tp_rank,
            )
            mamba = self._build_mamba_remote(nixl_agent_meta, tp_ratio, transfer_info)
            blocks_data = np.concatenate([blocks_data, mamba])

        # Import the remote agent only after every local validation and
        # descriptor calculation has succeeded. If descriptor-list creation
        # then fails, remove that import here because the outer multi-rank
        # transaction has not yet received its name to roll it back.
        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        try:
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
            remote_handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
            if remote_handle is None:
                raise RuntimeError("NIXL failed to prepare remote descriptors")
            self.dst_xfer_side_handles[engine_id][remote_tp_rank] = remote_handle
        except BaseException:
            try:
                self.nixl_wrapper.remove_remote_agent(remote_agent_name)
            except Exception:
                logger.warning(
                    "Failed to remove remote agent %s after descriptor setup "
                    "failed for engine %s rank %s",
                    remote_agent_name,
                    engine_id,
                    remote_tp_rank,
                    exc_info=True,
                )
            raise

        return remote_agent_name

    def _validate_dcp_interleave_compatibility(
        self,
        *,
        remote_dcp_size: int,
        remote_interleave: int,
        remote_block_size: int,
        remote_physical_per_logical: int,
        remote_engine_id: str,
    ) -> None:
        """Validate whether rank-local pages can be copied without reshuffling."""
        if self.dcp_size == remote_dcp_size and self.dcp_size > 1:
            if self.cp_kv_cache_interleave_size != remote_interleave:
                raise RuntimeError(
                    "Equal-DCP KV transfer requires identical token interleave "
                    f"sizes, got local={self.cp_kv_cache_interleave_size}, "
                    f"remote={remote_interleave} (engine {remote_engine_id})."
                )
        elif self.dcp_size != remote_dcp_size:
            # A raw NIXL page copy cannot transpose cyclic token ownership.
            # Asymmetric DCP therefore remains supported only when every
            # sharded side owns one complete rank-local logical page at a time.
            local_logical_block_size = (
                self.block_size * self._physical_blocks_per_logical_kv_block
            )
            remote_logical_block_size = remote_block_size * remote_physical_per_logical
            invalid_local = (
                self.dcp_size > 1
                and self.cp_kv_cache_interleave_size != local_logical_block_size
            )
            invalid_remote = (
                remote_dcp_size > 1 and remote_interleave != remote_logical_block_size
            )
            if invalid_local or invalid_remote:
                raise RuntimeError(
                    "Asymmetric-DCP KV transfer requires whole-page interleave "
                    "on each sharded side; token-interleaved cache pages need "
                    "equal DCP sizes. Got "
                    f"local DCP/interleave/page={self.dcp_size}/"
                    f"{self.cp_kv_cache_interleave_size}/"
                    f"{local_logical_block_size}, remote={remote_dcp_size}/"
                    f"{remote_interleave}/{remote_logical_block_size} "
                    f"(engine {remote_engine_id})."
                )

    def _validate_legacy_registration_compatibility(
        self, metadata: NixlAgentMetadata, remote_tp_size: int
    ) -> None:
        """Validate constraints intrinsic to legacy static page descriptors.

        This intentionally excludes compatibility-hash policy. A disabled strict
        hash check must not route a byte layout that legacy descriptors cannot
        represent; callers can instead select canonical segmented-direct when
        placement metadata is available.
        """
        remote_dcp_size = metadata.dcp_size
        if not (
            self.dcp_size % remote_dcp_size == 0 or remote_dcp_size % self.dcp_size == 0
        ):
            raise RuntimeError(
                "Legacy NIXL requires DCP sizes to divide one another: "
                f"local={self.dcp_size}, remote={remote_dcp_size} "
                f"(engine {metadata.engine_id})."
            )

        self._validate_dcp_interleave_compatibility(
            remote_dcp_size=remote_dcp_size,
            remote_interleave=metadata.cp_kv_cache_interleave_size,
            remote_block_size=metadata.block_size,
            remote_physical_per_logical=(metadata.physical_blocks_per_logical_kv_block),
            remote_engine_id=metadata.engine_id,
        )

        if getattr(self, "use_mla", True):
            return
        local_layout = (
            self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout
        )
        transfer_config = getattr(self, "kv_transfer_config", None)
        legacy_permute_available = (
            getattr(transfer_config, "enable_permute_local_kv", False)
            and metadata.kv_cache_layout == "LBHNC"
            and not getattr(self, "_is_hma_required", False)
        )
        if metadata.kv_cache_layout != local_layout and not legacy_permute_available:
            raise RuntimeError(
                "Legacy NIXL cannot represent different MHA KV cache layouts: "
                f"local={local_layout}, remote={metadata.kv_cache_layout} "
                f"(engine {metadata.engine_id})."
            )

        assert self.transfer_topo is not None
        local_tp_size = self.transfer_topo.tp_size
        tp_sizes_divide = (
            local_tp_size % remote_tp_size == 0 or remote_tp_size % local_tp_size == 0
        )
        remote_kv_replicated = remote_tp_size > self.transfer_topo.total_num_kv_heads
        if not tp_sizes_divide:
            raise RuntimeError(
                "Legacy NIXL requires TP sizes to divide one another: "
                f"local={local_tp_size}, remote={remote_tp_size} "
                f"(engine {metadata.engine_id})."
            )
        if (
            local_tp_size != remote_tp_size
            and not remote_kv_replicated
            and not KVCacheLayout[local_layout].is_block_contiguous
            and not legacy_permute_available
        ):
            raise RuntimeError(
                "Legacy NIXL heterogeneous TP head splitting requires a "
                f"block-contiguous local layout, got {local_layout} "
                f"(engine {metadata.engine_id})."
            )

    def _validate_remote_agent_handshake(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_size: int,
        remote_dcp_size: int = 1,
    ):
        """
        Validate the remote agent handshake metadata ensuring the
        invariants hold true.
        """
        remote_engine_id = nixl_agent_meta.engine_id

        assert self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(remote_engine_id)
        assert remote_info.remote_tp_size == remote_tp_size
        assert remote_info.remote_dcp_size == remote_dcp_size
        assert (
            remote_info.remote_cp_kv_cache_interleave_size
            == nixl_agent_meta.cp_kv_cache_interleave_size
        )
        # DCP sizes must divide one another; this is what keeps the
        # read-slicing math in pull_worker a closed form.
        assert (
            self.dcp_size % remote_dcp_size == 0 or remote_dcp_size % self.dcp_size == 0
        ), (
            f"DCP sizes must divide one another: local={self.dcp_size}, "
            f"remote={remote_dcp_size} (engine {remote_engine_id})."
        )
        self._validate_dcp_interleave_compatibility(
            remote_dcp_size=remote_dcp_size,
            remote_interleave=nixl_agent_meta.cp_kv_cache_interleave_size,
            remote_block_size=nixl_agent_meta.block_size,
            remote_physical_per_logical=(
                nixl_agent_meta.physical_blocks_per_logical_kv_block
            ),
            remote_engine_id=remote_engine_id,
        )

        tp_ratio = self.transfer_topo.tp_ratio(remote_tp_size)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            nixl_agent_meta.block_size
        )
        # num_kv_heads > tp_size with P_TP > D_TP not supported for non-mamba.
        # Mamba models can have replicated FA KV with tp_ratio < 0.
        # MLA models do not need to handle kv replication.
        if not self.use_mla and not self._has_mamba:
            assert not (
                tp_ratio < 0 and self.transfer_topo.is_kv_replicated(remote_engine_id)
            )

        remote_physical_per_logical = (
            nixl_agent_meta.physical_blocks_per_logical_kv_block
        )
        if (
            self._has_mamba
            and remote_physical_per_logical
            != self._physical_blocks_per_logical_kv_block
            and self.vllm_config.cache_config.enable_prefix_caching
            and not (
                self.use_mla
                and (self.dcp_size > 1 or remote_dcp_size > 1)
                and self.vllm_config.cache_config.mamba_cache_mode == "align"
                and all(
                    issubclass(spec_type, (FullAttentionSpec, MambaSpec))
                    for spec_type in self._group_spec_types
                )
            )
        ):
            raise RuntimeError(
                "Prefix caching with heterogeneous physical_blocks_per_logical "
                "is not supported for Mamba hybrid models. "
                f"Local: {self._physical_blocks_per_logical_kv_block}, "
                f"Remote: {remote_physical_per_logical}. "
                "Disable prefix caching with --no-enable-prefix-caching."
            )

        if block_size_ratio != 1:
            # Heterogeneous block sizes transfer at remote-block granularity;
            # the untransferred tail of the last local attention block is
            # zeroed in the receive post-process, and mamba state pages
            # transfer 1:1 (never sub-split).
            assert not self.use_host_buffer, (
                "Heterogeneous block sizes are not supported with host buffer"
            )
        kv_cache_layout = (
            self.kv_cache_layout
            if not self.use_host_buffer
            else self.host_buffer_kv_cache_layout
        )
        if not self.use_mla and nixl_agent_meta.kv_cache_layout != kv_cache_layout:
            if (
                self.kv_transfer_config.enable_permute_local_kv
                and nixl_agent_meta.kv_cache_layout == "LBHNC"
            ):
                logger.info(
                    "Remote is LBHNC and local is LBNHC, enabled additional permute "
                    "on local device KV."
                )
                assert not self._is_hma_required, (
                    "HMA does not support block size post processing"
                )
                self.enable_permute_local_kv = True
            else:
                raise RuntimeError(
                    "Heterogeneous TP expects same kv_cache_layout. "
                    "Or enable experimental feature to use HND to NHD support by "
                    "setting 'enable_permute_local_kv'=True in --kv-transfer-config."
                )
        # if remote_agent used attn is not same as local,
        # hint heterogenuous attn post process
        if (
            nixl_agent_meta.attn_backend_name != self.backend_name
            and self.backend_name in ["CPU_ATTN"]
        ):
            if self._is_hma_required:
                raise RuntimeError(
                    "heterogeneous attn post process is not supported with HMA"
                )
            logger.info(
                "[Experimental] CPU_ATTN backend is used, "
                "hint heterogeneous attn post process"
            )
            self.enable_heterogeneous_attn_post_process = True

        # Heterogeneous TP requires head-splitting, which only works with
        # block-contiguous layouts (e.g. LBHNC). MLA and replicated-KV cases
        # don't split on heads. Mamba doesn't support heterogeneous TP.
        if (
            abs(tp_ratio) != 1
            and not self.use_mla
            and not self.transfer_topo.is_kv_replicated(remote_engine_id)
            and not KVCacheLayout[kv_cache_layout].is_block_contiguous
            and not self.enable_permute_local_kv
        ):
            raise RuntimeError(
                "Heterogeneous TP head-dimension splitting requires contiguous heads. "
                "Use a block-contiguous layout (e.g. LBHNC) on the prefill side."
            )

        # Per-region block_len validation enforcing the P/D invariant.
        # REPLICATE regions (MLA, or a whole-model MLA / replicated-KV transfer)
        # only allow the number of blocks to differ; SPLIT regions scale with
        # the per-rank KV head ratio rather than the raw tp_ratio, because GQA
        # replication caps per-rank heads at 1 when tp > total_kv_heads
        # (issue #45330). Mamba uses the ssm_sizes counterpart, so skip here.
        if self._has_mamba and self.use_mla:
            # Hybrid MLA+SSM (e.g. KimiLinear's KDA+MLA): regions are
            # kernel-granularity views of the mamba-unified page. The MLA
            # per-token page is TP-independent, so the block lengths must
            # match up to the kernel block size ratio even under
            # heterogeneous TP (remote kernel blocks may be smaller).
            # SSM geometry is validated via ssm_sizes/conv offsets instead.
            assert self.block_len_per_layer == [
                block_len * block_size_ratio for block_len in nixl_agent_meta.block_lens
            ], (
                "Hybrid MLA kernel-granularity block lengths must match "
                f"between P and D (block_size_ratio={block_size_ratio}): "
                f"local={self.block_len_per_layer}, "
                f"remote={nixl_agent_meta.block_lens}."
            )
        elif not self._has_mamba:
            assert len(self.block_len_per_layer) == len(nixl_agent_meta.block_lens), (
                "Number of KV layers must match between prefill and decode"
            )
            model_replicated = self.use_mla or self.transfer_topo.is_kv_replicated(
                remote_engine_id
            )
            total_kv_heads = self.transfer_topo.total_num_kv_heads
            local_heads = self.transfer_topo.local_physical_heads
            remote_heads = max(1, total_kv_heads // remote_tp_size)
            for i, local_len in enumerate(self.block_len_per_layer):
                replicated = model_replicated or self._is_region_replicated(i)
                remote_len = nixl_agent_meta.block_lens[i]
                if replicated:
                    assert local_len // block_size_ratio == remote_len, (
                        "KV cache sizes must match between P and D when "
                        f"replicated (region {i}: local={local_len}, "
                        f"remote={remote_len}, bsr={block_size_ratio})."
                    )
                elif tp_ratio > 0:
                    assert (
                        remote_len
                        == (local_len * remote_heads // local_heads) // block_size_ratio
                    ), (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads} "
                        f"// block_size_ratio {block_size_ratio}."
                    )
                else:
                    assert block_size_ratio == 1, (
                        "Different local/remote block sizes are not supported "
                        "when P TP > D TP."
                    )
                    assert remote_len == local_len * remote_heads // local_heads, (
                        f"SPLIT region {i}: remote P KV block_len {remote_len} "
                        f"must equal local {local_len} * remote_heads "
                        f"{remote_heads} // local_heads {local_heads}."
                    )

        # TP workers that handhshake with same remote have same #blocks.
        assert self.dst_num_blocks[remote_engine_id] == nixl_agent_meta.num_blocks
        # Same number of regions/~layers.
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(self.block_len_per_layer)

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        """copy recved kv from host buffer to device."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        local_block_ids = meta.local_physical_block_ids
        # TODO (NickLucche) D2H<>H2D ops could benefit from coalescing io across groups
        # The h2d block copies below are intentionally synchronous.
        with gpu_sync_allowed():
            for group_block_ids in local_block_ids:
                self.copy_blocks(
                    self.host_xfer_buffers,
                    self.device_kv_caches,
                    group_block_ids,
                    group_block_ids,
                    "h2d",
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "synced recved kv of request[%s] to device kv buffer,"
                "local_block_ids: %s. ",
                req_id,
                ",".join(map(str, local_block_ids)),
            )

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        """copy kv from device to host buffer."""
        assert self.use_host_buffer
        assert self.copy_blocks is not None

        # The d2h block copies below are intentionally synchronous.
        with gpu_sync_allowed():
            for req_id, meta in metadata.reqs_to_save.items():
                meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                    meta.local_block_ids, self._physical_blocks_per_logical_kv_block
                )
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "save_load_kv for request[%s] to host xfer buffer."
                        "local_block_ids: %s. ",
                        req_id,
                        ",".join(map(str, meta.local_physical_block_ids)),
                    )
                # blocking
                for group_block_ids in meta.local_physical_block_ids:
                    self.copy_blocks(
                        self.device_kv_caches,
                        self.host_xfer_buffers,
                        group_block_ids,
                        group_block_ids,
                        "d2h",
                    )

    @cached_property
    def _attention_kv_caches(self) -> list[torch.Tensor]:
        """Device KV caches of attention layers (mamba states excluded),
        as consumed by the receive post-process."""
        assert self.device_kv_caches, (
            "_attention_kv_caches accessed before register_kv_caches"
        )
        mamba_layers = {
            name
            for g, group in enumerate(self.kv_cache_config.transfer_groups)
            if _is_ssm_spec(self._group_spec_types[g])
            for name in group.layer_names
        }
        kv_caches = self.device_kv_caches
        return [cache for name, cache in kv_caches.items() if name not in mamba_layers]

    def post_process_device_kv_on_receive(
        self,
        block_size_ratio: int,
        block_ids_list: list[tuple[list[int], int]],
        convert: bool = True,
    ):
        """
        Post process device kv cache after receiving from remote.

        3 types of conversion supported (``convert``):
            * kv_cache_postprocess_layout => convert from HND to NHD
            * kv_cache_postprocess_blksize => convert from small block size
              to large block size
            * kv_cache_postprocess_blksize_and_layout => convert from small
              block size to large block size and convert from HND to NHD

        The transfer only covers ``covered_sub_blocks`` remote-sized
        sub-blocks of each request's local attention blocks; the rest was
        clipped, either by remote-block pairing (block-size ratio) or by the
        hetero-ppl front trim in ``_apply_prefix_caching``. Those blocks were
        excluded from the scheduler's alloc-time KV zeroing (which would race
        the RDMA write), so everything past the covered range is zeroed here.
        Stale bytes would otherwise surface as garbage or NaNs once decode
        grows into the untransferred tail.
        """
        if len(self.device_kv_caches) == 0:
            return
        assert block_size_ratio >= 1, "Only nP < nD supported currently."
        assert self.transfer_topo is not None
        if not convert:
            logger.debug(
                "Post-processing device kv cache on receive by zeroing "
                "untransferred blocks."
            )
        elif self.enable_permute_local_kv and block_size_ratio > 1:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger and permuting layout from HND"
                " to NHD.",
                block_size_ratio,
            )
        elif self.enable_permute_local_kv:
            logger.debug(
                "Post-processing device kv cache on receive by permuting layout"
                "from HND to NHD."
            )
        else:
            logger.debug(
                "Post-processing device kv cache on receive by converting "
                "block_size with %sx bigger.",
                block_size_ratio,
            )

        attn_caches = self._attention_kv_caches
        device = attn_caches[0].device
        for block_ids, covered_sub_blocks in block_ids_list:
            # Blocks the transfer didn't write: the token tail of the last
            # partially covered block, then everything beyond it.
            covered_blocks, sub_blocks_in_last = divmod(
                covered_sub_blocks, block_size_ratio
            )
            first_stale = covered_blocks + (1 if sub_blocks_in_last else 0)
            has_stale = first_stale < len(block_ids)
            indices = None
            if convert or has_stale:
                indices = async_tensor_h2d(block_ids, device, torch.long)

            if convert:
                for cache in attn_caches:
                    if self.enable_permute_local_kv and block_size_ratio > 1:
                        kv_postprocess_blksize_and_layout_on_receive(
                            cache, indices, block_size_ratio
                        )
                    elif self.enable_permute_local_kv:
                        kv_postprocess_layout_on_receive(cache, indices)
                    else:
                        kv_postprocess_blksize_on_receive(
                            cache, indices, block_size_ratio
                        )

            if sub_blocks_in_last:
                last_block_id = block_ids[covered_blocks]
                for cache in attn_caches:
                    # Both post-processed layouts leave tokens on dim 1.
                    sub_block_tokens = cache.shape[1] // block_size_ratio
                    zero_from = sub_blocks_in_last * sub_block_tokens
                    cache[last_block_id, zero_from:].zero_()
            if has_stale:
                assert indices is not None
                stale_ids = indices[first_stale:]
                for cache in attn_caches:
                    cache.index_fill_(0, stale_ids, 0)

    def post_process_device_kv_on_receive_heterogeneous_attn(
        self, block_ids: list[int]
    ):
        """
        Post process device kv cache after receiving from remote
        for heterogeneous attention.
        """
        assert self.enable_heterogeneous_attn_post_process

        indices = torch.tensor(block_ids, device=self.device_type, dtype=torch.long)

        for _, cache_or_caches in self.device_kv_caches.items():
            current_platform.pack_kv_cache(
                kv_cache=cache_or_caches,
                indices=indices,
            )

    def get_finished(self) -> tuple[set[str], set[str]]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        assert self.transfer_topo is not None
        done_sending = self._get_new_notifs()
        terminal_failed_recv_reqs: set[ReqId] = set()
        done_recving = self._pop_done_transfers(
            self._recving_transfers,
            stream="recv",
            failed_req_ids=terminal_failed_recv_reqs,
        )

        # Drain queue of requests where handshake or transfer setup failed.
        failed_recv_reqs = set(terminal_failed_recv_reqs)
        while not self._failed_recv_reqs.empty():
            try:
                failed_recv_reqs.add(self._failed_recv_reqs.get_nowait())
            except queue.Empty:
                break

        # Add failed requests to done_recving for scheduler tracking
        # (blocks are already marked invalid, scheduler will handle recompute)
        done_recving.update(failed_recv_reqs)
        self._on_receive_requests_terminal(
            done_recving - failed_recv_reqs, failed_recv_reqs
        )

        if len(done_sending) > 0 or len(done_recving) > 0:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving (%s failed)",
                self.tp_rank,
                len(done_sending),
                len(done_recving),
                len(failed_recv_reqs),
            )

        block_ids_for_blocksize_post_process = defaultdict(list)
        block_ids_for_heterogeneous_attn_post_process = list[list[int]]()
        for req_id in done_recving:
            # clean up metadata for completed requests
            # Handshake/WRITE threads inspect this mapping while holding the
            # same lock to pin endpoint registrations. Keep mutation atomic
            # with those scans so rollover cannot race terminal cleanup.
            with self._handshake_lock:
                meta = self._recving_metadata.pop(req_id, None)
            assert meta is not None, f"{req_id} not found in recving_metadata list"

            # Skip KV sync and post-processing for failed requests
            if req_id in failed_recv_reqs:
                self._generic_direct_receive_requests.discard(req_id)
                logger.warning(
                    "Skipping KV post-processing for failed request %s",
                    req_id,
                )
                continue

            if req_id in self._generic_direct_receive_requests:
                # The generic planner wrote destination-native page offsets;
                # applying the legacy block-size/layout conversion would
                # corrupt heterogeneous TP/DCP copies. Its exact KVRange also
                # replaces the legacy clipped-block coverage bookkeeping.
                self._generic_direct_receive_requests.discard(req_id)
                continue

            assert meta.remote is not None
            if self.use_host_buffer:
                self.sync_recved_kv_to_device(req_id, meta)

            # Post processing for heteroblocksize/layout, and for blocks the
            # transfer clipped. The latter happens either at remote-block
            # granularity (block_size_ratio > 1) or at kernel-block
            # granularity, when equal kernel pages meet differing logical
            # block sizes and _apply_prefix_caching front-trims to the
            # minimum count (hybrid heterogeneous TP).
            remote_info = self.transfer_topo.get_engine_info(meta.remote.engine_id)
            block_size_ratio = self.transfer_topo.block_size_ratio(
                remote_info.remote_block_size
            )
            hetero_ppl = (
                remote_info.remote_physical_blocks_per_logical
                != self._physical_blocks_per_logical_kv_block
            )
            dcp_active = self.dcp_size > 1 or remote_info.remote_dcp_size > 1
            if block_size_ratio > 1 or self.enable_permute_local_kv or hetero_ppl:
                for g, local_group in enumerate(meta.local_physical_block_ids):
                    if not local_group or _is_ssm_spec(self._group_spec_types[g]):
                        continue
                    # Number of remote-sized sub-blocks the transfer covered;
                    # everything past this was clipped and must be zeroed.
                    if dcp_active:
                        assert meta.dcp_local_attention_blocks_covered
                        covered_sub_blocks = meta.dcp_local_attention_blocks_covered[g]
                    else:
                        covered_sub_blocks = min(
                            len(local_group) * block_size_ratio,
                            len(meta.remote.block_ids[g]),
                        )
                    block_ids_for_blocksize_post_process[block_size_ratio].append(
                        (local_group, covered_sub_blocks)
                    )
            # post processing for heterogeneous attention
            if self.enable_heterogeneous_attn_post_process:
                block_ids_for_heterogeneous_attn_post_process.append(
                    meta.local_physical_block_ids[0]
                )
        for (
            block_size_ratio,
            block_ids_list,
        ) in block_ids_for_blocksize_post_process.items():
            # MLA never needs the block-size/layout conversion, but its
            # clipped blocks still need zeroing.
            convert = not self.use_mla and (
                block_size_ratio > 1 or self.enable_permute_local_kv
            )
            self.post_process_device_kv_on_receive(
                block_size_ratio, block_ids_list, convert
            )

        for block_ids in block_ids_for_heterogeneous_attn_post_process:
            self.post_process_device_kv_on_receive_heterogeneous_attn(block_ids)

        self._sync_device_after_mamba_recv(done_recving, failed_recv_reqs)

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        expired_requests = [
            req_id for req_id, expires in self._reqs_to_send.items() if now >= expires
        ]
        for req_id in expired_requests:
            count = self.consumer_notification_counts_by_req.pop(req_id, 0)
            self.expected_consumer_notifications_by_req.pop(req_id, None)
            self.xfer_stats.record_kv_expired_req()
            logger.warning(
                "Releasing expired KV blocks for request %s which were "
                "retrieved by %d remote worker(s) before lease expired.",
                req_id,
                count,
            )
            self._reqs_to_process.remove(req_id)
            del self._reqs_to_send[req_id]
            self._on_send_request_terminal(req_id)
            done_sending.add(req_id)

        return done_sending, done_recving

    def _sync_device_after_mamba_recv(
        self,
        done_recving: set[str],
        failed_recv_reqs: set[str],
    ) -> None:
        """Synchronize ROCm direct-GPU Mamba receives before model execution."""
        if (
            not current_platform.is_rocm()
            or not self._has_mamba
            or self.use_host_buffer
            or not (done_recving - failed_recv_reqs)
        ):
            return

        torch.accelerator.synchronize()

    def _get_new_notifs(self) -> set[str]:
        """Get req_ids which got a remote xfer notification.

        Subclasses must implement this to handle mode-specific notifications.
        """
        raise NotImplementedError

    def _on_receive_requests_terminal(
        self, successful: set[str], failed: set[str]
    ) -> None:
        """Allow a transfer mode to emit one request-level completion."""

    def _on_send_request_terminal(self, req_id: str) -> None:
        """Allow a transfer mode to discard source-side request state."""

    def _handle_heartbeat(self, payload: str) -> None:
        """Extend leases for requests referenced in a heartbeat.

        Args:
            payload: comma-separated P-side request IDs, e.g.
                     "req_abc,req_def".
        """
        new_expiry = time.perf_counter() + self._lease_extension
        for req_id in payload.split(","):
            if req_id in self._reqs_to_send:
                old = self._reqs_to_send[req_id]
                self._reqs_to_send[req_id] = max(old, new_expiry)
                logger.debug(
                    "Heartbeat extended lease for request %s "
                    "by %ds (old_expiry=%.1f, new_expiry=%.1f)",
                    req_id,
                    self._lease_extension,
                    old,
                    new_expiry,
                )

    def _release_xfer_handle(self, handle: int) -> None:
        """Release a transfer and any request-scoped direct descriptors.

        The segmented-direct tracker returns ``False`` for legacy transfers,
        preserving the existing static descriptor-list lifetime.
        """
        tracker = getattr(self, "_ephemeral_direct_dlists", None)
        if tracker is None or not tracker.release(handle):
            self.nixl_wrapper.release_xfer_handle(handle)

    def _pop_done_transfers(
        self,
        transfers: dict[str, list[int]],
        *,
        stream: str = "recv",
        failed_req_ids: set[str] | None = None,
    ) -> set[str]:
        """
        Pop requests only after every sibling xfer reaches terminal state.

        A failed handle latches request failure, but a PROC sibling keeps the
        request in ``transfers``. ``failed_req_ids`` receives only fully
        terminal failed requests, allowing aggregate notifications to use
        ``done - failed``. When omitted, receive failures retain the legacy
        queue-based reporting behavior.

        Args:
            transfers: dict of req_id -> list[running_xfer]
        Returns:
            set of req_ids whose complete handle set is terminal
        """
        poller = getattr(self, "_request_terminal_poller", None)
        if poller is None:
            # A few focused tests construct workers with object.__new__.
            poller = self._request_terminal_poller = NixlRequestTerminalPoller()

        def on_done(handle: int) -> None:
            res = self.nixl_wrapper.get_xfer_telemetry(handle)
            self.xfer_stats.record_transfer(res)
            self._release_xfer_handle(handle)

        def on_failed(
            req_id: str,
            handle: int,
            failure: NixlTransferFailure,
            first_failure: bool,
        ) -> None:
            if failure.error is not None:
                self._log_failure(
                    failure_type="transfer_exception",
                    msg="Marking blocks as invalid",
                    req_id=req_id,
                    error=failure.error,
                )
            else:
                self._log_failure(
                    failure_type="transfer_failed",
                    msg="Marking blocks as invalid",
                    req_id=req_id,
                    xfer_state=failure.state,
                )
            if stream == "recv":
                self._mark_remote_engine_stale_for_request(req_id)
            self._record_failed_transfer(
                req_id,
                handle,
                mark_request_invalid=first_failure and stream == "recv",
            )

        result = poller.poll(
            transfers,
            stream=stream,
            check_state=self.nixl_wrapper.check_xfer_state,
            on_done=on_done,
            on_failed=on_failed,
        )
        terminal = set(result.terminal_requests)
        failed = set(result.failed_requests)
        if failed_req_ids is not None:
            failed_req_ids.update(failed)
        elif stream == "recv":
            for req_id in failed:
                self._failed_recv_reqs.put(req_id)
        return terminal

    def _record_failed_transfer(
        self,
        req_id: str,
        handle: int | None,
        *,
        mark_request_invalid: bool,
    ) -> None:
        """Record/release one failed handle without reporting request completion."""
        if (
            mark_request_invalid
            and (meta := self._recving_metadata.get(req_id))
            and not self._is_hma_required
            and meta.local_block_ids
        ):
            self._invalid_block_ids.put(set(meta.local_block_ids[0]))
        if handle is not None:
            self._release_xfer_handle(handle)
        self.xfer_stats.record_failed_transfer()

    def _mark_remote_engine_stale_for_request(self, req_id: str) -> None:
        """Force endpoint revalidation after a failed receive operation."""
        meta = self._recving_metadata.get(req_id)
        if meta is None or meta.remote is None:
            return
        with self._handshake_lock:
            if meta.remote.engine_id in self._remote_agents:
                self._stale_remote_engines.add(meta.remote.engine_id)

    def _latch_failed_transfer(
        self,
        req_id: str,
        handle: int | None,
        *,
        stream: str,
    ) -> None:
        """Record a failed batch while deferring request completion.

        Submission can fail after sibling batches have already started.  The
        request-level poller owns the failure latch so completion is reported
        only after those siblings reach terminal state.  Send-side failures
        deliberately never invalidate receive-side KV blocks.
        """
        poller = getattr(self, "_request_terminal_poller", None)
        if poller is None:
            poller = self._request_terminal_poller = NixlRequestTerminalPoller()
        first_failure = poller.mark_failed(stream, req_id)
        if stream == "recv":
            self._mark_remote_engine_stale_for_request(req_id)
        self._record_failed_transfer(
            req_id,
            handle,
            mark_request_invalid=first_failure and stream == "recv",
        )

    def _handle_failed_transfer(self, req_id: str, handle: int | None):
        """
        Handle a failed transfer by marking all (logical) blocks as invalid and
        recording the failure.

        Args:
            req_id: The request ID.
            handle: The transfer handle.
        """
        # Setup/handshake failures have no sibling handle set to drain, so they
        # retain immediate queue reporting. Polled handle failures instead use
        # _record_failed_transfer and report only when all siblings terminate.
        self._record_failed_transfer(
            req_id,
            handle,
            mark_request_invalid=True,
        )
        self._failed_recv_reqs.put(req_id)

    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """
        Send heartbeat notifications to remote engines, extending lease on KV blocks.
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake (this request may still be in waiting queue) so
            # the **next** heartbeat for this remote can go through.
            try:
                handshake = self._ensure_handshake(
                    engine_id,
                    hb_info.host,
                    hb_info.port,
                    hb_info.tp_size,
                    hb_info.dcp_size,
                    hb_info.pp_size,
                    self._hb_handshake_notif_only,
                    hb_info.pcp_size,
                    hb_info.endpoint_incarnation,
                )
            except Exception:
                # Heartbeats only extend an existing lease. A concurrent endpoint
                # rollover, shutdown, or failed setup must not fail the model step;
                # the normal transfer/lease paths own request terminalization.
                logger.debug(
                    "Failed to prepare heartbeat for engine %s",
                    engine_id,
                    exc_info=True,
                )
                continue
            if handshake is not None:
                continue  # handshake is still pending

            # Build the heartbeat message: "HB:req1,req2,..."
            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in self._remote_agents[engine_id].values():
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    def get_mapped_blocks(
        self, block_ids: np.ndarray, block_size_ratio: int
    ) -> np.ndarray:
        """
          Calculates the new set of block IDs by mapping every element
          in the (potentially sparse) input array.
          Example: block_ids=[0, 2], block_size_ratio=2
        get_mapped_blocks    0     1     [2     3]     4     5
              # remote is |h0-b0|h1-b0||h0-b1|h1-b1||h0-b1|h1-b1||
              # local is  |h0-b0......||h1-b0......||h2-b0........
        local_block_ids         0           [1]           2
        """
        if block_ids.size == 0:
            return np.array([], dtype=np.int64)

        start_ids = block_ids * block_size_ratio
        offsets = np.arange(block_size_ratio)
        mapped_2d = start_ids[:, None] + offsets[None, :]

        return mapped_2d.flatten().astype(np.int64)

    def _map_block_ids_for_block_size_ratio(
        self,
        local_block_ids: BlockIds,
        remote_block_ids: BlockIds,
        block_size_ratio: int,
    ) -> tuple[BlockIds, BlockIds]:
        """Map attention-group block ids to remote-block granularity.

        Each local attention block is split into ``block_size_ratio``
        sub-blocks paired 1:1 with remote blocks. Sub-blocks beyond the
        remote list — the untransferred tail of the last local block — are
        clipped here and zeroed in the receive post-process. Mamba state
        blocks are indivisible and transfer 1:1, unexpanded.

        ex: remote (prefill) block ids with block_size 4:
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Local (decode) block ids with block_size 16: [1, 2, 3] expand to
        [4, 5, ..., 15], then clip to the first 10 to pair 1:1 with remote.
        """
        mapped_local: list[list[int]] = []
        mapped_remote: list[list[int]] = []
        for i, remote_group in enumerate(remote_block_ids):
            local_group = local_block_ids[i] if local_block_ids else []
            if _is_ssm_spec(self._group_spec_types[i]):
                mapped_local.append(list(local_group))
                mapped_remote.append(list(remote_group))
                continue
            mapped = self.get_mapped_blocks(
                np.asarray(local_group), block_size_ratio
            ).tolist()
            if len(mapped) > len(remote_group):
                mapped = mapped[: len(remote_group)]
            mapped_local.append(mapped)
            mapped_remote.append(list(remote_group))
        if not any(mapped_local):
            # Full prefix cache hit is indicated with an empty list.
            return [], mapped_remote
        return mapped_local, mapped_remote

    def _logical_to_kernel_block_ids(self, block_ids: BlockIds, ratio: int) -> BlockIds:
        """
        Convert block ids to kernel physical block ids.
        This is required when the logical block size (the one set by the user)
        does not match the one required by the attn backend.
        `ratio` is the number of physical blocks per logical block.
        We always receive logical blocks from the engine, so we expand them here eg:
        logical block ids: [(SW-clipped) [1], (FA) [2, 3]], ratio=2
        physical block ids: [(SW-clipped) [2, 3], (FA) [4, 5, 6, 7]]
        """
        if ratio == 1:
            # Noop when physical and logical block sizes are the same
            return block_ids
        block_arange = np.arange(0, ratio).reshape(1, -1)
        # Mamba blocks have no logical<>physical discrepancy (block-size=1)
        group_specs = self.kv_cache_config.transfer_groups
        physical_block_ids = []
        for i, group in enumerate(block_ids):
            if _is_ssm_spec(get_representative_spec_type(group_specs[i].kv_cache_spec)):
                physical_block_ids.append(group)
            else:
                physical_block_ids.append(
                    BlockTable.map_to_kernel_blocks(
                        np.array(group),
                        ratio,
                        block_arange,
                    ).tolist()
                )
        return physical_block_ids

    def _map_dcp_attention_block_ids(
        self,
        local_ids: list[int],
        remote_ids: list[int],
        remote_rank: int,
        local_dcp_size: int,
        local_dcp_rank: int,
        remote_dcp_size: int,
        local_num_computed_blocks: int,
        local_physical_per_logical: int,
        remote_physical_per_logical: int,
    ) -> tuple[list[int], list[int]]:
        """Map DCP-sharded MLA pages at physical KV-block granularity.

        Scheduler block IDs describe global token pages. Each rank compacts its
        owned token stream into a contiguous rank-local page. For equal DCP
        sizes this is true for token interleave as well as block interleave: the
        same rank owns the same global positions on both sides. For asymmetric
        DCP this mapper is used only after the handshake has required whole-page
        interleave on every sharded side.

        Hybrid memory allocation can make a rank-local page span multiple
        physical KV blocks, and that span can differ across P and D (Kimi-K3
        TP8/DCP8 -> TP1 uses 48 -> 384). Matching whole logical IDs would
        therefore omit seven eighths of a K3 page.

        Convert each side's page slice to global physical-block positions and
        pair the intersections. ``local_ids`` has already dropped prefix-cached
        pages, so its first page has ordinal ``local_num_computed_blocks``.
        """
        assert local_dcp_size > 0 and remote_dcp_size > 0
        assert local_physical_per_logical > 0
        assert remote_physical_per_logical > 0

        local_page_span = local_physical_per_logical * local_dcp_size
        remote_page_span = remote_physical_per_logical * remote_dcp_size
        remote_dcp_rank = remote_rank % remote_dcp_size

        mapped_local: list[int] = []
        mapped_remote: list[int] = []
        local_idx = remote_idx = 0
        while local_idx < len(local_ids) and remote_idx < len(remote_ids):
            local_start = (
                local_num_computed_blocks + local_idx
            ) * local_page_span + local_dcp_rank * local_physical_per_logical
            local_end = local_start + local_physical_per_logical
            remote_start = (
                remote_idx * remote_page_span
                + remote_dcp_rank * remote_physical_per_logical
            )
            remote_end = remote_start + remote_physical_per_logical

            overlap_start = max(local_start, remote_start)
            overlap_end = min(local_end, remote_end)
            if overlap_start < overlap_end:
                count = overlap_end - overlap_start
                local_block_start = (
                    local_ids[local_idx] * local_physical_per_logical
                    + overlap_start
                    - local_start
                )
                remote_block_start = (
                    remote_ids[remote_idx] * remote_physical_per_logical
                    + overlap_start
                    - remote_start
                )
                mapped_local.extend(range(local_block_start, local_block_start + count))
                mapped_remote.extend(
                    range(remote_block_start, remote_block_start + count)
                )

            if local_end <= remote_end:
                local_idx += 1
            if remote_end <= local_end:
                remote_idx += 1

        assert len(mapped_local) == len(mapped_remote)
        return mapped_local, mapped_remote

    def _apply_prefix_caching(
        self,
        decode_block_ids: BlockIds,
        prefill_block_ids: BlockIds,
        decode_physical_per_logical: int,
        prefill_physical_per_logical: int,
    ) -> tuple[BlockIds, BlockIds]:
        """Trim block ID lists so only the uncomputed suffix is transferred.

        Inputs are *kernel* (physical) block IDs, already expanded from logical IDs
        with each side's physical-per-logical ratio. Both pull and push call this after
        that expansion so the trim happens at kernel granularity.

        The prefix-cache hit is always on the decode (D) side, so ``decode``
        holds only its uncomputed blocks while prefill (P) holds the full
        sequence. This is mode-independent: pull passes its own (D) blocks as
        ``decode``, push passes the remote D registration as ``decode``.

        For non-Mamba models: end-trim ``prefill`` to match ``decode`` count, so
        already-cached prefix blocks are skipped in the transfer.

        For Mamba hybrid: SSM groups pair state slots by position and FA
        groups end-trim to the uncomputed suffix when physical-per-logical
        matches.
        """
        # Partial prefix cache hit: just transfer uncomputed blocks.
        # Skip mamba groups — their blocks represent full state (conv+ssm),
        # not per-token data, so trimming would corrupt the transfer.
        prefill_block_ids = list(prefill_block_ids)
        if not self._has_mamba:
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                assert num_decode_blocks <= len(prefill_group)
                if num_decode_blocks < len(prefill_group):
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
        else:
            # (NOTE: ZhanqiuHu) HeteroTP can cause different kernel block counts
            # due to logical block rounding.
            # Example: 640 prompt tokens, kernel_block_size=64
            #   prefill physical_per_logical=10, decode physical_per_logical=6
            #   prefill logical ids from kv_transfer_params = [0]
            #   decode logical ids allocated = [0, 1]
            #   prefill kernel blocks: [0..9]  (1*10=10)
            #   decode kernel blocks:  [0..11] (2*6=12)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            # Vice versa (prefill physical_per_logical=6, decode=10):
            #   prefill logical ids = [0, 1], decode logical ids = [0]
            #   prefill kernel blocks: [0..11] (2*6=12)
            #   decode kernel blocks:  [0..9]  (1*10=10)
            #   actual data blocks = ceil(640/64) = 10, trim both to 10
            decode_block_ids = list(decode_block_ids)
            for i, prefill_group in enumerate(prefill_block_ids):
                num_decode_blocks = len(decode_block_ids[i])
                num_prefill_blocks = len(prefill_group)
                if _is_ssm_spec(self._group_spec_types[i]):
                    if num_decode_blocks == num_prefill_blocks:
                        continue
                    # Only state-bearing slots reach here, single-state modes
                    # just one (see get_exchange_clipped_blocks), so differing
                    # counts mean position-indexed "all"-mode lists. A longer
                    # prefill list carries earlier positions the decode side
                    # already has (prefix hit) -> take its tail; a longer decode
                    # list holds the position D recomputes itself, which gets
                    # no prefill state.
                    assert num_decode_blocks - num_prefill_blocks <= 1, (
                        f"Group {i}: unpairable SSM state slots, "
                        f"decode={num_decode_blocks} prefill={num_prefill_blocks}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    if num_decode_blocks < num_prefill_blocks:
                        prefill_block_ids[i] = prefill_group[-num_blocks:]
                    else:
                        decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                elif (
                    decode_physical_per_logical == prefill_physical_per_logical
                    and num_decode_blocks < num_prefill_blocks
                ):
                    # Partial prefix cache hit for FA group.
                    prefill_block_ids[i] = prefill_group[-num_decode_blocks:]
                else:
                    # TODO Handle prefix caching with different block_sizes
                    # Allocation rounding legitimately leaves up to
                    # ppl - 1 trailing dead kernel blocks per side (plus one
                    # extra decode block for the recomputed final token), so
                    # the counts may differ by up to the sum of the two
                    # ratios; anything larger indicates mismatched lists.
                    max_padding = (
                        decode_physical_per_logical + prefill_physical_per_logical
                    )
                    assert abs(num_decode_blocks - num_prefill_blocks) <= max_padding, (
                        f"Group {i}: |{num_decode_blocks} - "
                        f"{num_prefill_blocks}| > {max_padding}"
                    )
                    num_blocks = min(num_decode_blocks, num_prefill_blocks)
                    decode_block_ids[i] = decode_block_ids[i][:num_blocks]
                    prefill_block_ids[i] = prefill_group[:num_blocks]
        return decode_block_ids, prefill_block_ids

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """
        Get the KV transfer stats for the connector.
        """
        # Clear stats for next iteration
        if not self.xfer_stats.is_empty():
            return self.xfer_stats.clone_and_reset()
        return None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """
        Return and clear the set of block IDs that failed to load.

        This is called by the scheduler to identify blocks that need
        to be retried after a NIXL transfer failure.
        """
        # Drain the queue (thread-safe, no lock needed).
        result: set[int] = set()
        while not self._invalid_block_ids.empty():
            try:
                result.update(self._invalid_block_ids.get_nowait())
            except queue.Empty:
                break
        return result

    def _engine_has_active_references(
        self,
        engine_id: EngineId,
        *,
        exclude_request_id: str | None = None,
    ) -> bool:
        # ``start_load_kv`` installs the current request metadata before it
        # validates the remote endpoint specification. Exclude that metadata
        # from pinning its own stale registration, but never let the exclusion
        # hide live handles or a lazy direct-read window owned by the request.
        if exclude_request_id is not None:
            if self._recving_transfers.get(exclude_request_id):
                return True
            direct_windows = getattr(self, "_direct_read_batch_windows", {})
            if exclude_request_id in direct_windows:
                return True
        for req_id, meta in self._recving_metadata.items():
            if req_id == exclude_request_id or meta.remote is None:
                continue
            if meta.remote.engine_id == engine_id:
                return True
        # Push WRITE state does not retain an engine-id index. Conservatively
        # defer replacement of every push registration while any WRITE is live.
        sending_transfers = getattr(self, "_sending_transfers", None)
        sending_lock = getattr(self, "_sending_transfers_lock", None)
        if sending_transfers is None:
            return False
        if sending_lock is None:
            return any(sending_transfers.values())
        with sending_lock:
            return any(sending_transfers.values())

    def _evict_stale_engines(
        self,
        *,
        exclude_request_id: str | None = None,
    ) -> None:
        """Scan for and evict remote engines that have exceeded their TTL.

        Called from the main thread in when a new remote engine appears.
        We can only go OOM as we discover and register a new remote, therefore we make
        sure we clean up stale engine data structures before then. This invariant
        prevents us from using background threads, though memory usage is not guaranteed
        to be "optimal" until a new handshake is performed.

        Pending handshakes do not have an ``_engine_last_active`` entry yet.
        Requests already admitted for receive pin their remote endpoint until
        request terminalization; this closes the interval between scheduler
        admission and the first transfer-activity refresh.
        """
        # NOTE (NickLucche): This does NOT currently prevent OOMing if a huge number
        # of remote engines is registered all at once (adding a background cleanup
        # thread wouldnt help either).
        # If that scenario is plausible, we can follow up with an LRU eviction policy.
        for eid in tuple(self._stale_remote_engines):
            if eid in self._remote_agents and not self._engine_has_active_references(
                eid,
                exclude_request_id=exclude_request_id,
            ):
                self._cleanup_remote_engine(eid, log_eviction=False)

        if self._engine_ttl <= 0:
            return

        now = time.perf_counter()
        for eid, last_active in list(self._engine_last_active.items()):
            if (
                now - last_active > self._engine_ttl
                and not self._engine_has_active_references(
                    eid,
                    exclude_request_id=exclude_request_id,
                )
            ):
                self._cleanup_remote_engine(eid)

    def _cleanup_remote_engine(
        self, engine_id: EngineId, *, log_eviction: bool = True
    ) -> None:
        """Remove all state for a single remote engine.

        Releases NIXL resources (dlist handles, remote agents) and clears
        all per-engine data structures. Used by both TTL eviction and
        shutdown.
        """
        assert engine_id in self._remote_agents

        # Notif-only engines (push-mode D side) have no descriptor state.
        for handle in self.dst_xfer_side_handles.pop(engine_id, {}).values():
            self.nixl_wrapper.release_dlist_handle(handle)
        for agent_name in self._remote_agents.pop(engine_id).values():
            self.nixl_wrapper.remove_remote_agent(agent_name)

        self.kv_caches_base_addr.pop(engine_id, None)
        self.dst_num_blocks.pop(engine_id, None)
        self.tp_mappings.pop(engine_id, None)
        self._remote_placement_indexes.pop(engine_id, None)
        self._generic_only_remote_engines.discard(engine_id)
        self._remote_handshake_specs.pop(engine_id, None)
        self._stale_remote_engines.discard(engine_id)
        if self.transfer_topo is not None:
            self.transfer_topo.unregister_remote_engine(engine_id)

        # Drop the cached clock offset; it is re-measured on the next handshake.
        self._engine_clock_offset.pop(engine_id, None)
        # A just-completed handshake may not have recorded activity yet, so
        # tolerate a missing entry.
        last_active = self._engine_last_active.pop(engine_id, None)
        if log_eviction and last_active is not None:
            logger.info(
                "Evicted stale remote engine %s (inactive for %.1fs).",
                engine_id,
                time.perf_counter() - last_active,
            )

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Shutdown the connector worker."""
        if not hasattr(self, "_handshake_initiation_executor"):
            # error happens during init, no need to shutdown
            return
        with self._handshake_lock:
            if self._shutting_down or self._shutdown_complete:
                return
            self._shutting_down = True
            self._handshake_shutdown_event.set()
        # A running handshake owns/imports NIXL resources. Wait without holding
        # _handshake_lock so its done callback can commit those resources; the
        # cleanup pass below will then release them before memory deregistration.
        self._handshake_initiation_executor.shutdown(
            wait=True,
            cancel_futures=True,
        )
        for handles in self._recving_transfers.values():
            for handle in handles:
                self._release_xfer_handle(handle)
        self._recving_transfers.clear()
        self._recving_metadata.clear()
        self._generic_direct_receive_requests.clear()
        self._request_terminal_poller.clear()
        # Also release a prepared direct transfer that failed before it could
        # be published in a request's in-flight handle list.
        self._ephemeral_direct_dlists.release_all()
        for handle in self.src_xfer_handles_by_block_size.values():
            self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_block_size.clear()
        for handles in self.src_xfer_handles_by_tp_ratio.values():
            for handle in handles:
                self.nixl_wrapper.release_dlist_handle(handle)
        self.src_xfer_handles_by_tp_ratio.clear()
        for engine_id in list(self._remote_agents):
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        for desc in self._registered_descs:
            self.nixl_wrapper.deregister_memory(desc)
        self._registered_descs.clear()
        while not self._ready_requests.empty():
            try:
                self._ready_requests.get_nowait()
            except queue.Empty:
                break
        while not self._failed_recv_reqs.empty():
            try:
                self._failed_recv_reqs.get_nowait()
            except queue.Empty:
                break
        with self._handshake_lock:
            self._handshake_futures.clear()
            self._shutdown_complete = True
