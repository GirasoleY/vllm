# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DecodeBenchConnector: a synthetic resident-KV benchmark connector.

The connector allocates real entries from vLLM's GPU block pool, but supplies
their contents locally instead of transferring them from a remote prefill or
cache service.  This isolates model execution and resident-KV capacity from
network and external-store performance.

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

    Legacy decode-only mode treats all prompt tokens except the final token as
    externally cached.  A request can instead select an exact cached boundary:

        "kv_transfer_params": {
            "decode_bench": {"num_cached_tokens": 114048}
        }

    Then run a benchmark with the desired input/output lengths:
        vllm bench serve --base-url http://127.0.0.1:8000 --model <model> \\
            --dataset-name random --random-input-len 40000 \\
            --random-output-len 100 --max-concurrency 10

    Configuration options (via kv_connector_extra_config):
        - fill_mean (float): Mean value for random normal fill (default: 0.015)
        - fill_std (float): Standard deviation for random fill (default: 0.0)
          Set to 0 for constant values, >0 for random sampling
        - require_explicit_cache_spec (bool): Require every request to carry
          ``kv_transfer_params.decode_bench.num_cached_tokens`` and a non-empty
          ``cache_salt``.  Also rejects local prefix-cache hits, making workload
          contamination fail loudly (default: False).

The explicit boundary must be aligned to ``prefix_match_unit`` (or the cache
block size when no finer unit is configured), and cannot include the final
prompt token because that token must be computed to produce logits.
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

_REQUEST_SPEC_KEY = "decode_bench"
_NUM_CACHED_TOKENS_KEY = "num_cached_tokens"


@dataclass
class DecodeBenchConnectorMetadata(KVConnectorMetadata):
    """Empty metadata: non-null worker cache blocks are filled at startup."""


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
        cache_config = vllm_config.cache_config
        self.match_unit = cache_config.prefix_match_unit or cache_config.block_size

        kv_transfer_config = vllm_config.kv_transfer_config
        assert kv_transfer_config is not None
        self.require_explicit_cache_spec = kv_transfer_config.get_from_extra_config(
            "require_explicit_cache_spec", False
        )
        if not isinstance(self.require_explicit_cache_spec, bool):
            raise ValueError("require_explicit_cache_spec must be a boolean")

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Return the additional synthetic prefix beyond the local hit.

        The method is deliberately stateless. If admission fails, the
        scheduler can query the same request again; if a request is preempted,
        its resident synthetic prefix can be reconstructed after re-admission.
        """
        requested_boundary = self._get_requested_boundary(request)
        if requested_boundary is None:
            # Backward-compatible decode-only behavior.
            requested_boundary = max(0, request.num_tokens - 1)

        if self.require_explicit_cache_spec and num_computed_tokens != 0:
            raise ValueError(
                "DecodeBenchConnector strict mode found a local prefix-cache hit "
                f"of {num_computed_tokens} tokens for request "
                f"{request.request_id!r}; use a unique cache_salt and reset or "
                "disable local prefix caching before the benchmark"
            )

        # The connector API asks for tokens beyond the local prefix.
        return max(0, requested_boundary - num_computed_tokens), False

    def _get_requested_boundary(self, request: "Request") -> int | None:
        params = request.kv_transfer_params
        spec: Any = None if params is None else params.get(_REQUEST_SPEC_KEY)

        if spec is None:
            if self.require_explicit_cache_spec:
                raise ValueError(
                    "DecodeBenchConnector strict mode requires "
                    "kv_transfer_params.decode_bench.num_cached_tokens"
                )
            return None

        if not isinstance(spec, dict):
            raise ValueError("kv_transfer_params.decode_bench must be an object")
        if _NUM_CACHED_TOKENS_KEY not in spec:
            raise ValueError(
                "kv_transfer_params.decode_bench.num_cached_tokens is required"
            )

        boundary = spec[_NUM_CACHED_TOKENS_KEY]
        # bool is an int subclass, but accepting true/false here is almost
        # certainly a malformed benchmark request.
        if type(boundary) is not int:
            raise ValueError(
                "kv_transfer_params.decode_bench.num_cached_tokens must be an integer"
            )

        max_boundary = max(0, request.num_tokens - 1)
        if not 0 <= boundary <= max_boundary:
            raise ValueError(
                "kv_transfer_params.decode_bench.num_cached_tokens must be between "
                f"0 and {max_boundary} for request {request.request_id!r}, got "
                f"{boundary}"
            )
        if boundary % self.match_unit != 0:
            raise ValueError(
                "kv_transfer_params.decode_bench.num_cached_tokens must be aligned "
                f"to prefix_match_unit={self.match_unit}, got {boundary}"
            )

        if self.require_explicit_cache_spec and not request.cache_salt:
            raise ValueError(
                "DecodeBenchConnector strict mode requires a non-empty cache_salt "
                f"for request {request.request_id!r}"
            )
        return boundary

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """No-op: allocation itself creates the resident-KV pressure."""
        return

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        return DecodeBenchConnectorMetadata()

    def request_finished(self, request: "Request"):
        """No per-request connector state is retained."""
        return


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
        self.require_explicit_cache_spec = kv_transfer_config.get_from_extra_config(
            "require_explicit_cache_spec", False
        )
        if not isinstance(self.fill_mean, int | float):
            raise ValueError("fill_mean must be numeric")
        if not isinstance(self.fill_std, int | float) or self.fill_std < 0:
            raise ValueError("fill_std must be a non-negative number")
        if not isinstance(self.require_explicit_cache_spec, bool):
            raise ValueError("require_explicit_cache_spec must be a boolean")

        # Will be populated via register_kv_caches.
        self.kv_caches: dict[str, Any] | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Pre-fill every usable physical cache block once.

        Each registered tensor is block-indexed in its first dimension,
        including the tensors inside hybrid Mamba/KDA state tuples. Block 0 is
        vLLM's shared null block and must remain zero; filling it would corrupt
        hybrid block-table padding. Per-request scheduling subsequently decides
        which already-filled physical blocks consume real pool capacity.
        """
        self.kv_caches = kv_caches

        # Float32 staging for random FP8-byte fills. A bounded buffer avoids a
        # full-cache scratch allocation after KV profiling has consumed most
        # device memory.
        chunk_elems = 1 << 25
        staging: torch.Tensor | None = None
        num_tensors = 0

        for layer_name, cache_or_states in kv_caches.items():
            caches: tuple[torch.Tensor, ...]
            if isinstance(cache_or_states, torch.Tensor):
                caches = (cache_or_states,)
            elif isinstance(cache_or_states, (list, tuple)) and all(
                isinstance(cache, torch.Tensor) for cache in cache_or_states
            ):
                caches = tuple(cache_or_states)
            else:
                message = (
                    "DecodeBenchConnector cannot pre-fill layer "
                    f"{layer_name!r}: expected a tensor or list/tuple of "
                    f"tensors, got {type(cache_or_states).__name__}"
                )
                if self.require_explicit_cache_spec:
                    raise TypeError(message)
                logger.warning_once(message)
                continue

            for cache in caches:
                if cache.ndim == 0 or cache.shape[0] == 0:
                    message = (
                        "DecodeBenchConnector cannot pre-fill layer "
                        f"{layer_name!r}: cache tensor has no block dimension"
                    )
                    if self.require_explicit_cache_spec:
                        raise ValueError(message)
                    logger.warning_once(message)
                    continue
                # Enforce the null-block invariant even if the incoming buffer
                # was uninitialized, then fill only usable block IDs 1..N-1.
                cache[0].zero_()
                staging = self._fill_tensor(cache[1:], staging, chunk_elems)
                num_tensors += 1

        # Drop the shared staging buffer before serving so the KV pool retains
        # all profiled headroom.
        del staging

        logger.info(
            "DecodeBenchConnector: Pre-filled %d tensors from %d KV cache layers "
            "with %s values (mean=%.3f, std=%.3f); null block 0 kept zero",
            num_tensors,
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
        """Fill a non-null attention-cache or hybrid-state tensor view."""
        if cache.numel() == 0:
            return staging

        with torch.no_grad():
            if cache.is_floating_point():
                if self.fill_std > 0:
                    cache.normal_(mean=self.fill_mean, std=self.fill_std)
                else:
                    cache.fill_(self.fill_mean)
            elif cache.dtype == torch.uint8:
                # uint8 is the byte-storage representation used by the K3 FP8
                # MLA cache. Reinterpret bytes as e4m3, then cast bounded
                # float32 samples into the view.
                fp8_view = cache.view(torch.float8_e4m3fn)
                if self.fill_std > 0:
                    flat = fp8_view.flatten()
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
            else:
                raise TypeError(
                    "DecodeBenchConnector only supports floating-point cache "
                    f"tensors or uint8 FP8 storage, got {cache.dtype}"
                )

        return staging

    def start_fill_kv(self, metadata: DecodeBenchConnectorMetadata):
        """No-op: all non-null physical cache blocks were filled at startup."""
        return
