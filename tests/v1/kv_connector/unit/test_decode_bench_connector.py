# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the benchmark-only DecodeBenchConnector."""

from typing import Any

import pytest
import torch

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .utils import create_model_runner_output, create_scheduler, create_vllm_config


class DecodeBenchTestRunner:
    """Small scheduler/worker runner for DecodeBenchConnector tests."""

    def __init__(
        self,
        block_size: int = 16,
        num_gpu_blocks: int = 100,
        prefix_match_unit: int | None = None,
        **extra_config: Any,
    ):
        self.req_id = -1
        vllm_config = create_vllm_config(
            block_size=block_size,
            max_num_batched_tokens=1000,
            kv_connector="DecodeBenchConnector",
            kv_connector_extra_config=extra_config,
        )
        vllm_config.cache_config.prefix_match_unit = prefix_match_unit

        self.scheduler: Scheduler = create_scheduler(
            vllm_config, num_blocks=num_gpu_blocks
        )
        self.worker_connector = DecodeBenchConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            self.scheduler.kv_cache_config,
        )

        self.kv_caches = {
            f"layer_{i}": torch.zeros(num_gpu_blocks, 2, 4, block_size, 8)
            for i in range(2)
        }
        self.worker_connector.register_kv_caches(self.kv_caches)

        scheduler_connector = self.scheduler.connector
        assert isinstance(scheduler_connector, DecodeBenchConnector)
        self.scheduler_connector = scheduler_connector

        init_none_hash(sha256)
        hash_unit = prefix_match_unit or block_size
        self._block_hasher = get_request_block_hasher(hash_unit, sha256)
        self._dummy_ctx = ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}
        )

    def new_request(
        self,
        token_ids: list[int],
        *,
        num_cached_tokens: int | None = None,
        cache_salt: str | None = None,
        raw_spec: Any = None,
        include_spec: bool | None = None,
    ) -> Request:
        self.req_id += 1
        request = Request(
            request_id=f"req-{self.req_id}",
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            cache_salt=cache_salt,
            block_hasher=self._block_hasher,
        )

        if include_spec is None:
            include_spec = num_cached_tokens is not None or raw_spec is not None
        if include_spec:
            spec = (
                raw_spec
                if raw_spec is not None
                else {"num_cached_tokens": num_cached_tokens}
            )
            request.kv_transfer_params = {"decode_bench": spec}

        self.scheduler.add_request(request)
        return request

    def num_matched(self, request: Request, num_computed_tokens: int = 0) -> int:
        num_tokens, is_async = self.scheduler_connector.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        assert is_async is False
        assert num_tokens is not None
        return num_tokens

    def run_single_step(self):
        scheduler_output = self.scheduler.schedule()
        metadata = scheduler_output.kv_connector_metadata
        assert isinstance(metadata, DecodeBenchConnectorMetadata)

        # Snapshot request ownership before a one-token test request finishes
        # and the scheduler releases its blocks in update_from_output().
        self.last_allocated_block_ids = {
            request.request_id: self.scheduler.kv_cache_manager.get_blocks(
                request.request_id
            ).get_block_ids()
            for request in self.scheduler.running
        }

        self.worker_connector.bind_connector_metadata(metadata)
        self.worker_connector.start_load_kv(self._dummy_ctx)
        self.worker_connector.clear_connector_metadata()

        model_runner_output = create_model_runner_output(
            reqs=self.scheduler.running,
            token_id=0,
        )
        self.scheduler.update_from_output(scheduler_output, model_runner_output)
        return scheduler_output, metadata


def test_legacy_mode_reports_whole_prompt_except_last_token():
    runner = DecodeBenchTestRunner()
    request = runner.new_request([1] * 48)

    assert runner.num_matched(request) == 47
    scheduler_output, metadata = runner.run_single_step()

    assert scheduler_output.num_scheduled_tokens[request.request_id] == 1
    assert metadata == DecodeBenchConnectorMetadata()


def test_single_token_legacy_prompt_reports_no_hit():
    runner = DecodeBenchTestRunner()
    request = runner.new_request([1])

    assert runner.num_matched(request) == 0


def test_explicit_warm_and_cold_boundaries_schedule_exact_suffixes():
    runner = DecodeBenchTestRunner(require_explicit_cache_spec=True)
    warm = runner.new_request([1] * 48, num_cached_tokens=32, cache_salt="unique-warm")
    cold = runner.new_request([2] * 48, num_cached_tokens=0, cache_salt="unique-cold")

    assert runner.num_matched(warm) == 32
    assert runner.num_matched(cold) == 0

    scheduler_output, metadata = runner.run_single_step()
    assert scheduler_output.num_scheduled_tokens[warm.request_id] == 16
    assert scheduler_output.num_scheduled_tokens[cold.request_id] == 48
    assert metadata == DecodeBenchConnectorMetadata()

    # The external prefix consumes the same real BlockPool as locally computed
    # tokens. Both full 48-token requests own three physical blocks after this
    # step; startup filling alone did not reserve any blocks.
    for request in (warm, cold):
        block_ids = runner.last_allocated_block_ids[request.request_id][0]
        assert len(block_ids) == 3
        assert all(block_id != 0 for block_id in block_ids)


def test_sync_external_attention_blocks_are_not_zeroed():
    runner = DecodeBenchTestRunner(require_explicit_cache_spec=True)
    runner.scheduler.needs_kv_cache_zeroing = True
    manager = runner.scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    manager._record_new_block_ids = True
    warm = runner.new_request([1] * 48, num_cached_tokens=32, cache_salt="unique-warm")
    cold = runner.new_request([2] * 48, num_cached_tokens=0, cache_salt="unique-cold")

    scheduler_output, _ = runner.run_single_step()

    warm_ids = runner.last_allocated_block_ids[warm.request_id][0]
    cold_ids = runner.last_allocated_block_ids[cold.request_id][0]
    # Warm blocks 0 and 1 hold the synchronous external prefix and retain the
    # startup fill. Its locally computed suffix block, plus every cold block,
    # still follows K3's normal zero-before-compute path.
    assert set(scheduler_output.new_block_ids_to_zero or []) == {
        warm_ids[2],
        *cold_ids,
    }


def test_explicit_boundary_returns_only_tokens_beyond_local_hit():
    runner = DecodeBenchTestRunner()
    request = runner.new_request([1] * 48, num_cached_tokens=32)

    assert runner.num_matched(request, num_computed_tokens=16) == 16
    assert runner.num_matched(request, num_computed_tokens=32) == 0
    assert runner.num_matched(request, num_computed_tokens=40) == 0


def test_strict_mode_rejects_local_prefix_cache_contamination():
    runner = DecodeBenchTestRunner(require_explicit_cache_spec=True)
    request = runner.new_request([1] * 48, num_cached_tokens=32, cache_salt="unique")

    with pytest.raises(ValueError, match="local prefix-cache hit"):
        runner.num_matched(request, num_computed_tokens=16)


@pytest.mark.parametrize(
    ("request_kwargs", "error"),
    [
        ({"include_spec": False, "cache_salt": "unique"}, "requires"),
        (
            {"raw_spec": {}, "cache_salt": "unique"},
            "num_cached_tokens is required",
        ),
        (
            {"raw_spec": {"num_cached_tokens": True}, "cache_salt": "unique"},
            "must be an integer",
        ),
        (
            {"num_cached_tokens": -16, "cache_salt": "unique"},
            "must be between",
        ),
        (
            {"num_cached_tokens": 48, "cache_salt": "unique"},
            "must be between",
        ),
        (
            {"num_cached_tokens": 17, "cache_salt": "unique"},
            "must be aligned",
        ),
        ({"num_cached_tokens": 16}, "non-empty cache_salt"),
    ],
)
def test_strict_request_contract_validation(request_kwargs, error):
    runner = DecodeBenchTestRunner(require_explicit_cache_spec=True)
    request = runner.new_request([1] * 48, **request_kwargs)

    with pytest.raises(ValueError, match=error):
        runner.num_matched(request)


@pytest.mark.parametrize("value", ["true", 1, 0, [], {}])
def test_strict_config_must_be_boolean(value):
    vllm_config = create_vllm_config(
        block_size=16,
        kv_connector="DecodeBenchConnector",
        kv_connector_extra_config={"require_explicit_cache_spec": value},
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        create_scheduler(vllm_config, num_blocks=100)


def test_explicit_boundary_uses_prefix_match_unit():
    runner = DecodeBenchTestRunner(prefix_match_unit=8)
    aligned = runner.new_request([1] * 48, num_cached_tokens=24)
    unaligned = runner.new_request([2] * 48, num_cached_tokens=20)

    assert runner.num_matched(aligned) == 24
    with pytest.raises(ValueError, match="prefix_match_unit=8"):
        runner.num_matched(unaligned)


def test_unique_cache_salts_produce_disjoint_prefix_hashes():
    runner = DecodeBenchTestRunner()
    first = runner.new_request([1] * 48, num_cached_tokens=32, cache_salt="salt-a")
    second = runner.new_request([1] * 48, num_cached_tokens=32, cache_salt="salt-b")

    assert first.block_hashes
    assert second.block_hashes
    assert set(first.block_hashes).isdisjoint(second.block_hashes)


def test_startup_fill_preserves_null_block_and_fills_usable_attention_blocks():
    runner = DecodeBenchTestRunner(num_gpu_blocks=4)

    for cache in runner.kv_caches.values():
        assert torch.count_nonzero(cache[0]) == 0
        assert torch.allclose(cache[1:], torch.full_like(cache[1:], 0.015))


def test_startup_fill_preserves_null_block_for_hybrid_state_tensors():
    runner = DecodeBenchTestRunner(num_gpu_blocks=4)
    attention = torch.full((4, 2, 16), 9.0)
    states = [torch.full((4, 8), 9.0), torch.full((4, 4), 9.0)]
    hybrid_caches = {"attention": attention, "linear_attention": states}

    runner.worker_connector.register_kv_caches(hybrid_caches)  # type: ignore[arg-type]

    for cache in (attention, *states):
        assert torch.count_nonzero(cache[0]) == 0
        assert torch.allclose(cache[1:], torch.full_like(cache[1:], 0.015))


def test_startup_fill_unpacks_raw_mamba_pages_and_preserves_padding():
    vllm_config = create_vllm_config(
        block_size=16,
        kv_connector="DecodeBenchConnector",
        kv_connector_extra_config={"require_explicit_cache_spec": True},
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((2, 3), (4,)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=32,
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba"], spec)],
    )
    connector = DecodeBenchConnector(
        vllm_config,
        KVConnectorRole.WORKER,
        kv_cache_config,
    )
    raw_pages = torch.full((4, 1, 1, spec.page_size_bytes), -1, dtype=torch.int8)

    connector.register_kv_caches({"mamba": raw_pages})

    pages = raw_pages.squeeze(dim=(1, 2))
    first_state_bytes = 2 * 3 * torch.tensor([], dtype=torch.bfloat16).element_size()
    second_state_bytes = 4 * torch.tensor([], dtype=torch.float32).element_size()
    first_state = pages[1:, :first_state_bytes].view(torch.bfloat16).view(3, 2, 3)
    second_state = (
        pages[1:, first_state_bytes : first_state_bytes + second_state_bytes]
        .view(torch.float32)
        .view(3, 4)
    )

    assert torch.count_nonzero(pages[0]) == 0
    assert torch.allclose(
        first_state.float(), torch.full_like(first_state.float(), 0.015), atol=0.001
    )
    assert torch.allclose(second_state, torch.full_like(second_state, 0.015), atol=1e-6)
    assert torch.count_nonzero(pages[1:, first_state_bytes + second_state_bytes :]) == 0


def test_startup_fill_preserves_null_block_for_uint8_fp8_storage():
    runner = DecodeBenchTestRunner(num_gpu_blocks=4)
    cache = torch.full((4, 64), 255, dtype=torch.uint8)

    runner.worker_connector.register_kv_caches({"attention": cache})

    assert torch.count_nonzero(cache[0]) == 0
    actual = cache[1:].view(torch.float8_e4m3fn).float()
    assert torch.allclose(actual, torch.full_like(actual, 0.015), atol=0.002)


def test_startup_random_fill_preserves_null_block():
    runner = DecodeBenchTestRunner(num_gpu_blocks=4, fill_std=0.01)

    for cache in runner.kv_caches.values():
        assert torch.count_nonzero(cache[0]) == 0
        assert torch.count_nonzero(cache[1:]) > 0


def test_strict_worker_rejects_unknown_cache_objects():
    runner = DecodeBenchTestRunner(require_explicit_cache_spec=True)

    with pytest.raises(TypeError, match="expected a tensor"):
        runner.worker_connector.register_kv_caches({"bad": object()})  # type: ignore
