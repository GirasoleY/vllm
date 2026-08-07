# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the benchmark-only DecodeBenchConnector."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnector,
    DecodeBenchConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
)
from vllm.forward_context import ForwardContext
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

from .utils import create_model_runner_output, create_scheduler, create_vllm_config


class DecodeBenchTestRunner:
    """Small scheduler/worker runner for DecodeBenchConnector tests."""

    def __init__(self, block_size: int, num_gpu_blocks: int, **extra_config):
        self.req_id = -1
        vllm_config = create_vllm_config(
            block_size=block_size,
            max_num_batched_tokens=1000,
            kv_connector="DecodeBenchConnector",
            kv_connector_extra_config=extra_config,
        )

        self.scheduler: Scheduler = create_scheduler(
            vllm_config, num_blocks=num_gpu_blocks
        )
        self.worker_connector = DecodeBenchConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            self.scheduler.kv_cache_config,
        )

        num_heads = 4
        head_dim = 64
        self.kv_caches = {
            f"layer_{i}": torch.zeros(
                num_gpu_blocks, 2, num_heads, block_size, head_dim
            )
            for i in range(2)
        }
        self.worker_connector.register_kv_caches(self.kv_caches)

        scheduler_connector = self.scheduler.connector
        assert isinstance(scheduler_connector, DecodeBenchConnector)
        self.scheduler_connector = scheduler_connector

        init_none_hash(sha256)
        self._block_hasher = get_request_block_hasher(block_size, sha256)
        self._dummy_ctx = ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}
        )

    def new_request(
        self, token_ids: list[int], request_id: str | None = None
    ) -> Request:
        self.req_id += 1
        request = Request(
            request_id=request_id or str(self.req_id),
            prompt_token_ids=token_ids,
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
            block_hasher=self._block_hasher,
        )
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

        self.worker_connector.bind_connector_metadata(metadata)
        self.worker_connector.start_load_kv(self._dummy_ctx)
        self.worker_connector.clear_connector_metadata()

        model_runner_output = create_model_runner_output(
            reqs=self.scheduler.running,
            token_id=0,
        )
        self.scheduler.update_from_output(scheduler_output, model_runner_output)
        return scheduler_output, metadata


def test_default_reports_whole_prompt_except_last_token():
    runner = DecodeBenchTestRunner(block_size=16, num_gpu_blocks=100)
    request = runner.new_request([1] * 48)

    assert runner.num_matched(request) == 47
    scheduler_output, metadata = runner.run_single_step()

    assert scheduler_output.num_scheduled_tokens[request.request_id] == 1
    assert metadata == DecodeBenchConnectorMetadata()


def test_single_token_prompt_reports_no_hit():
    runner = DecodeBenchTestRunner(block_size=16, num_gpu_blocks=100)
    request = runner.new_request([1])

    assert runner.num_matched(request) == 0


def test_primer_prefix_accepts_only_primer_request_ids():
    runner = DecodeBenchTestRunner(
        block_size=16,
        num_gpu_blocks=100,
        synthetic_request_id_prefix="cmpl-seed-warm-",
    )
    primer = runner.new_request([1] * 48, request_id="cmpl-seed-warm-0001-0-random")
    measured_warm = runner.new_request(
        [2] * 48, request_id="cmpl-measure-warm-0001-0-random"
    )
    measured_cold = runner.new_request(
        [3] * 48, request_id="cmpl-measure-cold-0001-0-random"
    )

    assert runner.num_matched(primer) == 47
    assert runner.num_matched(measured_warm) == 0
    assert runner.num_matched(measured_cold) == 0

    scheduler_output, _ = runner.run_single_step()
    assert scheduler_output.num_scheduled_tokens[primer.request_id] == 1
    assert scheduler_output.num_scheduled_tokens[measured_warm.request_id] == 48
    assert scheduler_output.num_scheduled_tokens[measured_cold.request_id] == 48


def test_mooncake_first_decode_bench_fallback_is_primer_only():
    runner = DecodeBenchTestRunner(
        block_size=16,
        num_gpu_blocks=100,
        synthetic_request_id_prefix="cmpl-seed-warm-",
    )
    primer = runner.new_request([1] * 48, "cmpl-seed-warm-0001-0-random")
    measured_warm = runner.new_request([2] * 48, "cmpl-measure-warm-0001-0-random")
    measured_cold = runner.new_request([3] * 48, "cmpl-measure-cold-0001-0-random")

    mooncake = MagicMock()
    mooncake.get_num_new_matched_tokens.side_effect = (
        lambda request, _: (32, False)
        if request.request_id.startswith("cmpl-measure-warm-")
        else (0, False)
    )
    multi = object.__new__(MultiConnector)
    multi._connectors = [mooncake, runner.scheduler_connector]
    multi._requests_to_connector = {}

    assert multi.get_num_new_matched_tokens(primer, 0) == (47, False)
    assert multi._requests_to_connector[primer.request_id] == 1
    assert multi.get_num_new_matched_tokens(measured_warm, 0) == (32, False)
    assert multi._requests_to_connector[measured_warm.request_id] == 0
    assert multi.get_num_new_matched_tokens(measured_cold, 0) == (0, False)
    assert measured_cold.request_id not in multi._requests_to_connector


@pytest.mark.parametrize("prefix", ["", 1, False, []])
def test_primer_prefix_must_be_a_nonempty_string(prefix):
    vllm_config = create_vllm_config(
        block_size=16,
        kv_connector="DecodeBenchConnector",
        kv_connector_extra_config={"synthetic_request_id_prefix": prefix},
    )

    with pytest.raises(ValueError, match="must be a non-empty string"):
        create_scheduler(vllm_config, num_blocks=100)


def test_register_prefills_entire_attention_cache():
    runner = DecodeBenchTestRunner(block_size=16, num_gpu_blocks=100)

    for kv_cache in runner.kv_caches.values():
        assert torch.allclose(kv_cache, torch.full_like(kv_cache, 0.015))


def test_register_prefills_hybrid_state_tensors():
    runner = DecodeBenchTestRunner(block_size=16, num_gpu_blocks=4)
    attention = torch.zeros(4, 2, 16)
    states = [torch.zeros(4, 8), torch.zeros(4, 4)]
    hybrid_caches = {"attention": attention, "linear_attention": states}

    runner.worker_connector.register_kv_caches(hybrid_caches)  # type: ignore[arg-type]

    assert torch.allclose(attention, torch.full_like(attention, 0.015))
    for state in states:
        assert torch.allclose(state, torch.full_like(state, 0.015))


def test_register_prefills_uint8_fp8_storage():
    runner = DecodeBenchTestRunner(block_size=16, num_gpu_blocks=4)
    cache = torch.zeros(64, dtype=torch.uint8)

    runner.worker_connector.register_kv_caches({"attention": cache})

    actual = cache.view(torch.float8_e4m3fn).float()
    assert torch.allclose(actual, torch.full_like(actual, 0.015), atol=0.002)
