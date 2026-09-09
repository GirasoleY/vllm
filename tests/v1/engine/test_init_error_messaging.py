# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.core.kv_cache_utils import check_enough_kv_cache_memory
from vllm.v1.kv_cache_interface import FullAttentionSpec


def test_kv_cache_oom_no_memory():
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(config, spec, 0)


def test_kv_cache_oom_insufficient_memory(monkeypatch):
    from unittest.mock import MagicMock

    config = MagicMock()
    config.model_config.max_model_len = 2048
    config.cache_config.block_size = 16
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.parallel_config.decode_context_parallel_size = 1

    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils.max_memory_usage_bytes",
        lambda c, s: 100 * 1024**3,  # 100 GiB
    )

    spec = {
        "layer_0": FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
        )
    }

    with pytest.raises(ValueError):
        check_enough_kv_cache_memory(config, spec, 1024**3)  # 1 GiB


@pytest.mark.parametrize(
    "use_nixl,interleave,expected",
    [(True, None, 64), (True, 1, 1), (False, None, 1)],
    ids=["nixl-auto", "nixl-explicit", "non-nixl"],
)
@pytest.mark.parametrize("elastic_scale_up", [False, True])
def test_cp_interleave_resolved_on_all_workers_before_memory_profiling(
    monkeypatch, use_nixl, interleave, expected, elastic_scale_up
):
    """Only unresolved Nixl auto needs global groups before memory profiling."""
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import vllm.v1.engine.core as core_module
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        KVTransferConfig,
        ParallelConfig,
        VllmConfig,
    )
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec
    from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase

    config = VllmConfig(
        cache_config=CacheConfig(block_size=16),
        device_config=DeviceConfig(device="cpu"),
        parallel_config=ParallelConfig(
            tensor_parallel_size=2,
            decode_context_parallel_size=2,
            cp_kv_cache_interleave_size=interleave,
            distributed_executor_backend="mp",
        ),
        kv_transfer_config=(
            KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_both")
            if use_nixl
            else None
        ),
    )
    needs_preparation = use_nixl and interleave is None
    wrappers = []
    for rank in range(2):
        worker = object.__new__(WorkerBase)
        worker.vllm_config = deepcopy(config)
        worker.prepare_kv_cache_groups = MagicMock(wraps=worker.prepare_kv_cache_groups)
        wrapper = object.__new__(WorkerWrapperBase)
        wrapper.global_rank = rank
        wrapper.worker = worker
        wrapper.vllm_config = worker.vllm_config
        wrappers.append(wrapper)
    config.model_config = SimpleNamespace(max_model_len=128)

    specs = [
        FullAttentionSpec(
            block_size=size, num_kv_heads=1, head_size=64, dtype=torch.float16
        )
        for size in (64, 128)
    ]
    prepared_groups = [
        [
            KVCacheGroupSpec([f"layer{group}"] if group == rank else [], spec)
            for group, spec in enumerate(specs)
        ]
        for rank in range(2)
    ]
    core = object.__new__(core_module.EngineCore)
    core.model_executor = executor = MagicMock()
    core.available_gpu_memory_for_kv_cache = 1024
    executor.get_kv_cache_specs.return_value = [
        {f"layer{i}": spec} for i, spec in enumerate(specs)
    ]

    def prepare(method, *, args):
        assert method == "prepare_kv_cache_groups"
        (groups,) = args
        assert groups is prepared_groups
        executor.set_kv_cache_layout.assert_called_once_with("LBNHC")
        assert config.parallel_config.cp_kv_cache_interleave_size == expected
        for wrapper in wrappers:
            wrapper.prepare_kv_cache_groups(groups)

    def assert_preparation():
        expected_calls = 1 if needs_preparation else 0
        assert build_groups.call_count == expected_calls
        assert executor.collective_rpc.call_count == expected_calls
        for rank, wrapper in enumerate(wrappers):
            if needs_preparation:
                wrapper.worker.prepare_kv_cache_groups.assert_called_once_with(
                    prepared_groups[rank]
                )
            assert (
                wrapper.vllm_config.parallel_config.cp_kv_cache_interleave_size
                == expected
            )
        assert config.parallel_config.cp_kv_cache_interleave_size == expected

    def profile():
        assert_preparation()
        return [1024, 1024]

    executor.collective_rpc.side_effect = prepare
    executor.determine_available_memory.side_effect = profile
    monkeypatch.setattr(
        core_module.envs, "VLLM_ELASTIC_EP_SCALE_UP_LAUNCH", elastic_scale_up
    )
    monkeypatch.setattr(core_module, "register_all_kvcache_specs", lambda config: None)
    monkeypatch.setattr(
        core_module,
        "resolve_kv_cache_layout",
        lambda *args: SimpleNamespace(name="LBNHC"),
    )
    build_groups = MagicMock(return_value=prepared_groups)
    monkeypatch.setattr(core_module, "get_kv_cache_groups_per_worker", build_groups)

    class AllocationReached(Exception):
        """Stop after the startup phase under test, before allocating caches."""

    allocate = MagicMock(side_effect=AllocationReached)
    monkeypatch.setattr(core_module, "get_kv_cache_configs", allocate)

    with pytest.raises(AllocationReached):
        core._initialize_kv_caches(config)
    assert allocate.call_args.kwargs["prepared_groups"] is (
        prepared_groups if needs_preparation else None
    )
    assert_preparation()
    assert executor.determine_available_memory.call_count == (not elastic_scale_up)
