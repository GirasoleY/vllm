# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DecodeBenchConnector: A KV Connector for decode instance performance testing.

This connector emulates a prefill-decode disaggregated setting by filling
the KV cache with dummy values, allowing measurement of decoder performance
under larger input sequence lengths (ISL) in resource-limited environments.

Usage:
    To use this connector for benchmarking, configure it in the kv_transfer_config:

    Example:
        vllm serve <model> --kv-transfer-config '{
            "kv_connector": "DecodeBenchConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "fill_mean": 0.015,
                "fill_std": 0.0
            }
        }'

    Then run your benchmark with desired input/output lengths:
        vllm bench serve --base-url http://127.0.0.1:8000 --model <model> \\
            --dataset-name random --random-input-len 40000 \\
            --random-output-len 100 --max-concurrency 10

    Configuration options (via kv_connector_extra_config):
        - fill_mean (float): Mean value for random normal fill (default: 0.015)
        - fill_std (float): Standard deviation for random fill (default: 0.0)
          Set to 0 for constant values, >0 for random sampling
        - synthetic_request_id_prefix (str | None): If set, only request IDs
          beginning with this prefix are reported as externally cached. This
          is intended for a primer-only DecodeBench fallback behind a real
          external-cache connector in MultiConnector. Requests outside the
          primer cohort report no synthetic hit (default: None).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class DecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Empty metadata: all worker cache tensors are filled at startup."""


class DecodeBenchConnector(KVConnectorBase_V1, SupportsHMA):
    """
    A KV Connector for decode instance performance testing.

    This connector fills the KV cache with dummy (non-zero) values to
    emulate a prefill-decode disaggregated setting, enabling performance
    testing of the decoder with larger input sequence lengths.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        self.connector_scheduler: DecodeBenchConnectorScheduler | None = None
        self.connector_worker: DecodeBenchConnectorWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = DecodeBenchConnectorScheduler(vllm_config)
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = DecodeBenchConnectorWorker(vllm_config)

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, DecodeBenchConnectorMetadata)
        self.connector_worker.start_fill_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        # All operations are synchronous, so nothing to wait for
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        # This connector doesn't save KV cache (benchmarking only)
        pass

    def wait_for_save(self):
        # This connector doesn't save KV cache (benchmarking only)
        pass

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        # HMA-enabled path: same cleanup as the single-group variant since
        # this connector owns no external state per block.
        assert self.connector_scheduler is not None
        self.connector_scheduler.request_finished(request)
        return False, None


class DecodeBenchConnectorScheduler:
    """Scheduler-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.synthetic_request_id_prefix = kv_transfer_config.get_from_extra_config(
            "synthetic_request_id_prefix", None
        )
        if self.synthetic_request_id_prefix is not None and (
            not isinstance(self.synthetic_request_id_prefix, str)
            or not self.synthetic_request_id_prefix
        ):
            raise ValueError(
                "synthetic_request_id_prefix must be a non-empty string or null"
            )

        # Track which requests have already reported a synthetic hit.
        self._filled_requests: set[str] = set()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Report a synthetic external hit for an eligible new request.

        Returns:
            (num_tokens_to_fill, is_async)
            - num_tokens_to_fill: number of uncomputed tokens minus 1
                (we fill everything except the last token for decode)
            - is_async: False (synchronous filling)
        """
        req_id = request.request_id

        # Only report once per request on first scheduling.
        if req_id in self._filled_requests:
            return 0, False

        # In a MultiConnector primer, this makes DecodeBench fail closed for
        # every measured request. A Mooncake miss therefore computes cold
        # instead of silently falling through to synthetic KV.
        if self.synthetic_request_id_prefix is not None and not req_id.startswith(
            self.synthetic_request_id_prefix
        ):
            return 0, False

        # Calculate how many tokens we need to fill
        # Fill all uncomputed tokens except the last one (which will be decoded)
        # This simulates having processed a long prefill
        num_uncomputed_tokens = request.num_tokens - num_computed_tokens
        num_tokens_to_fill = max(0, num_uncomputed_tokens - 1)

        if num_tokens_to_fill == 0:
            return 0, False

        # The worker pool was filled at startup, so the blocks are immediately
        # usable once allocated.
        return num_tokens_to_fill, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """Mark a request after its pre-filled blocks have been allocated."""
        if num_external_tokens == 0:
            return
        self._filled_requests.add(request.request_id)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return DecodeBenchConnectorMetadata()

    def request_finished(self, request: "Request"):
        """
        Called when a request has finished. Clean up any state.
        """
        self._filled_requests.discard(request.request_id)


class DecodeBenchConnectorWorker:
    """Worker-side implementation for DecodeBenchConnector."""

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size

        # Get fill parameters from extra config
        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.fill_mean = kv_transfer_config.get_from_extra_config("fill_mean", 0.015)
        self.fill_std = kv_transfer_config.get_from_extra_config("fill_std", 0.0)

        # Will be populated via register_kv_caches
        self.kv_caches: dict[str, torch.Tensor] | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Store KV cache references and pre-fill them in-place.

        Pre-filling the entire cache once at registration avoids per-step
        Python-loop overhead in start_fill_kv, which becomes a bottleneck
        at high concurrency (>1000 req/rank). Since this is a benchmark-
        only connector, the actual values are immaterial as long as the
        statistical distribution matches fill_mean / fill_std.
        """
        self.kv_caches = kv_caches

        # Chunk size (float32 elements) for the fp8 randomize path. 32M × 4 B
        # = 128 MiB, well under the headroom left after KV-profile sizing.
        CHUNK_ELEMS = 1 << 25
        staging: torch.Tensor | None = None

        for layer_name, cache_or_states in kv_caches.items():
            caches: tuple[torch.Tensor, ...]
            if isinstance(cache_or_states, torch.Tensor):
                caches = (cache_or_states,)
            elif isinstance(cache_or_states, (list, tuple)) and all(
                isinstance(cache, torch.Tensor) for cache in cache_or_states
            ):
                caches = tuple(cache_or_states)
            else:
                logger.warning_once(
                    "DecodeBenchConnector: skipping pre-fill for layer %s whose "
                    "KV cache is %s, not a tensor or list/tuple of tensors.",
                    layer_name,
                    type(cache_or_states).__name__,
                )
                continue

            for cache in caches:
                staging = self._fill_tensor(cache, staging, CHUNK_ELEMS)

        # Release the shared staging buffer before returning so the KV pool
        # has its full budget available for serving.
        del staging

        logger.info(
            "DecodeBenchConnector: Pre-filled %d KV cache layers with "
            "%s values (mean=%.3f, std=%.3f)",
            len(kv_caches),
            "random" if self.fill_std > 0 else "constant",
            self.fill_mean,
            self.fill_std,
        )

    def _fill_tensor(
        self,
        cache: torch.Tensor,
        staging: torch.Tensor | None,
        chunk_elems: int,
    ) -> torch.Tensor | None:
        """Pre-fill one attention cache or hybrid-state tensor in place."""
        with torch.no_grad():
            if cache.is_floating_point():
                # Native float path (bf16/fp16/fp32/fp8-float views). normal_
                # and fill_ both work in float space.
                if self.fill_std > 0:
                    cache.normal_(mean=self.fill_mean, std=self.fill_std)
                else:
                    cache.fill_(self.fill_mean)
            else:
                # Integer storage — typically uint8 holding fp8 bytes.
                # normal_/fill_ can't produce meaningful float distributions on
                # integer tensors, so we reinterpret as fp8 and (for fill_std
                # > 0) sample float32 in small chunks and copy_ across (lossy
                # float32→fp8 cast). Allocating a full-shape float32 scratch
                # here OOMs on large MLA KV caches (the profile has already
                # claimed the headroom), hence the chunked staging buffer.
                # Defaults to e4m3fn which matches vLLM's `--kv-cache-dtype
                # fp8` and DeepSeek-MLA `fp8_ds_mla` layout.
                fp8_view = cache.view(torch.float8_e4m3fn)
                if self.fill_std > 0:
                    flat = fp8_view.reshape(-1)
                    numel = flat.numel()
                    required = min(chunk_elems, numel)
                    if staging is None or staging.numel() < required:
                        staging = torch.empty(
                            required,
                            dtype=torch.float32,
                            device=cache.device,
                        )
                    for start in range(0, numel, chunk_elems):
                        end = min(start + chunk_elems, numel)
                        buf = staging[: end - start]
                        buf.normal_(mean=self.fill_mean, std=self.fill_std)
                        flat[start:end].copy_(buf)
                else:
                    fp8_view.fill_(self.fill_mean)

        return staging

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        """No-op: KV cache is pre-filled once at register_kv_caches time.

        At high concurrency the old per-request per-layer fill loop became
        a host-side bottleneck. The metadata is still populated by the
        scheduler (for protocol compatibility) but is ignored here.
        """
        return
