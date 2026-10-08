# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MooncakeStore eager lookup (VLLM_MOONCAKE_EAGER_LOOKUP)."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
    MooncakeStoreScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    LookupKeyClient,
)


def _bare_client(lookup_fn):
    client = object.__new__(LookupKeyClient)
    client.executor = ThreadPoolExecutor(max_workers=1)
    client.futures = {}
    client._futures_lock = threading.Lock()
    client._prefetch_submit_t = {}
    client._prefetch_max_age_s = 2.0
    client._lookup = lookup_fn
    return client


def test_prefetch_then_lookup_consumes_one_future():
    calls = []

    def lookup_fn(num_tokens, block_hashes):
        calls.append(num_tokens)
        return MooncakeLookupResult(num_tokens - 16)

    client = _bare_client(lookup_fn)
    client.prefetch("r", 64, [b"a", b"b"])
    client.prefetch("r", 64, [b"a", b"b"])  # no resubmit while pending/done
    client.futures["r"].result(timeout=5)
    result = client.lookup("r", 64, [b"a", b"b"], non_block=True)
    assert result is not None and result.hit_length == 48
    assert calls == [64]
    assert "r" not in client.futures
    client.executor.shutdown()


def test_prefetch_pending_lookup_returns_none_then_result():
    gate = threading.Event()

    def lookup_fn(num_tokens, block_hashes):
        gate.wait(5)
        return MooncakeLookupResult(32)

    client = _bare_client(lookup_fn)
    client.prefetch("r", 64, [b"a"])
    assert client.lookup("r", 64, [b"a"], non_block=True) is None
    gate.set()
    client.futures["r"].result(timeout=5)
    assert client.lookup("r", 64, [b"a"], non_block=True).hit_length == 32
    client.executor.shutdown()


def _bare_store_scheduler(eager: bool):
    sched = object.__new__(MooncakeStoreScheduler)
    sched.enable_lookup = True
    sched.lookup_async = True
    sched.eager_lookup = eager
    sched._block_size = 16
    sched._hash_block_size = 16
    sched.enable_partial_hash_hits = False
    sched.client = MagicMock()
    return sched


def _req(num_tokens=64, skip=False):
    return SimpleNamespace(
        request_id="r",
        num_tokens=num_tokens,
        block_hashes=[b"h"] * (num_tokens // 16),
        skip_reading_prefix_cache=skip,
    )


def test_prefetch_lookup_submits_when_enabled():
    sched = _bare_store_scheduler(eager=True)
    req = _req()
    sched.prefetch_lookup(req)
    sched.client.prefetch.assert_called_once_with("r", 64, req.block_hashes)


@pytest.mark.parametrize(
    "eager,req",
    [(False, _req()), (True, _req(skip=True)), (True, _req(num_tokens=8))],
)
def test_prefetch_lookup_skips(eager, req):
    sched = _bare_store_scheduler(eager=eager)
    sched.prefetch_lookup(req)
    sched.client.prefetch.assert_not_called()


def test_stale_prefetched_result_is_looked_up_again():
    calls = []
    second = threading.Event()

    def lookup_fn(num_tokens, block_hashes):
        calls.append(num_tokens)
        if len(calls) == 2:
            second.wait(5)
        return MooncakeLookupResult(16 * len(calls))

    client = _bare_client(lookup_fn)
    client._prefetch_max_age_s = 0.05
    client.prefetch("r", 64, [b"a"])
    client.futures["r"].result(timeout=5)
    time.sleep(0.1)
    # Done but stale: resubmitted (pending, so not ready) instead of trusted.
    assert client.lookup("r", 64, [b"a"], non_block=True) is None
    second.set()
    client.futures["r"].result(timeout=5)
    result = client.lookup("r", 64, [b"a"], non_block=True)
    assert calls == [64, 64]
    assert result.hit_length == 32
    assert not client._prefetch_submit_t
    client.executor.shutdown()


def test_fresh_prefetched_result_is_used():
    calls = []

    def lookup_fn(num_tokens, block_hashes):
        calls.append(num_tokens)
        return MooncakeLookupResult(16)

    client = _bare_client(lookup_fn)
    client.prefetch("r", 64, [b"a"])
    client.futures["r"].result(timeout=5)
    assert client.lookup("r", 64, [b"a"], non_block=True).hit_length == 16
    assert calls == [64]
    client.executor.shutdown()


# --- Hook plumbing: MultiConnector forwarding and the engine-core call site. ---


def test_multi_connector_forwards_on_request_preprocessed():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    multi = object.__new__(MultiConnector)
    multi._connectors = [MagicMock(), MagicMock()]
    req = _req()
    multi.on_request_preprocessed(req)
    for c in multi._connectors:
        c.on_request_preprocessed.assert_called_once_with(req)


def _engine_core_request():
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id="r1",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


@pytest.mark.parametrize("hook_raises", [False, True])
def test_engine_core_calls_preprocessed_hook_best_effort(hook_raises):
    from vllm.v1.engine.core import EngineCore

    seen = []

    def hook(req):
        seen.append(req.request_id)
        if hook_raises:
            raise RuntimeError("lookup submit failed")

    core = SimpleNamespace(
        mm_receiver_cache=None,
        request_block_hasher=None,
        _kv_preprocessed_hook=hook,
    )
    req, _ = EngineCore.preprocess_add_request(core, _engine_core_request())
    assert seen == ["r1"] and req.request_id == "r1"


def test_engine_core_without_hook_is_unchanged():
    from vllm.v1.engine.core import EngineCore

    core = SimpleNamespace(mm_receiver_cache=None, request_block_hasher=None)
    req, _ = EngineCore.preprocess_add_request(core, _engine_core_request())
    assert req.request_id == "r1"
