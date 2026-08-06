# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import random
import typing

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 1024 * 1024


@pytest.mark.skip_global_cleanup
def test_disabled_symm_mem_has_no_low_sm_all_reduce():
    communicator = SymmMemCommunicator.__new__(SymmMemCommunicator)
    communicator.disabled = True
    assert not communicator.has_low_sm_all_reduce()


def symm_mem_allreduce_worker(local_rank: int, world_size: int, q: mp.Queue):
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))

    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        symm_mem_comm = cuda_communicator.symm_mem_comm
        if symm_mem_comm is None or symm_mem_comm.disabled:
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator is not available or disabled.")
            return

        inp_direct_symm_mem = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        if not symm_mem_comm.should_use_symm_mem(inp_direct_symm_mem):
            # can't use skip under multiprocessing
            q.put("SymmMemCommunicator isn't used for this world and input size.")
            return

        original_inp_direct_symm_mem = inp_direct_symm_mem.clone()
        out_direct_symm_mem = symm_mem_comm.all_reduce(inp_direct_symm_mem)
        assert out_direct_symm_mem is not None

        group = get_tp_group().device_group
        dist.all_reduce(original_inp_direct_symm_mem, group=group)
        torch.testing.assert_close(
            out_direct_symm_mem, original_inp_direct_symm_mem, atol=2.5, rtol=0.1
        )

        # Test tensor_model_parallel_all_reduce which should use symm_mem
        inp_tensor_parallel = torch.randint(
            -23, 1, (test_size_elements,), dtype=dtype, device=device
        )
        original_inp_tensor_parallel = inp_tensor_parallel.clone()
        out_tensor_parallel = tensor_model_parallel_all_reduce(inp_tensor_parallel)
        dist.all_reduce(original_inp_tensor_parallel, group=group)
        torch.testing.assert_close(
            out_tensor_parallel, original_inp_tensor_parallel, atol=2.5, rtol=0.1
        )


def low_sm_symm_mem_allreduce_worker(local_rank: int, world_size: int, q: mp.Queue):
    monkeypatch = pytest.MonkeyPatch()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world_size))

    with monkeypatch.context() as m, set_current_vllm_config(config):
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(torch.bfloat16)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12346",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "1",
                "VLLM_KIMI_K3_LATENT_AR_OVERLAP_MAX_TOKENS": "512",
            }
        )

        init_distributed_environment()
        initialize_model_parallel(tensor_model_parallel_size=world_size)

        tp_group = get_tp_group()
        if tp_group.low_sm_all_reduce_max_size() == 0:
            # can't use skip under multiprocessing
            q.put("Low-SM symmetric-memory all-reduce is not available.")
            return

        rank = tp_group.rank_in_group
        inp = torch.randn((448, 3584), dtype=torch.bfloat16, device=device)
        expected = inp.clone()
        dist.all_reduce(expected, group=tp_group.device_group)

        staged = tp_group.stage_low_sm_all_reduce(inp)
        assert staged.data_ptr() != inp.data_ptr()
        result = tp_group.all_reduce_low_sm(staged)
        assert result.data_ptr() == staged.data_ptr()
        torch.testing.assert_close(result, expected, atol=4e-2, rtol=2e-2)

        # The specialized path must not silently accept an unsupported dtype.
        with pytest.raises(ValueError, match="requires torch.bfloat16"):
            tp_group.all_reduce_low_sm(torch.ones(8, dtype=torch.float32))
        with pytest.raises(ValueError, match="must be the prefix"):
            tp_group.all_reduce_low_sm(torch.ones(8, dtype=torch.bfloat16))

        # Capture and replay the copy plus the in-place persistent-buffer path.
        static_input = torch.full(
            (512, 3584), rank + 1, dtype=torch.bfloat16, device=device
        )
        graph = torch.cuda.CUDAGraph()
        torch.accelerator.synchronize()
        with torch.cuda.graph(graph):
            graph_stage = tp_group.stage_low_sm_all_reduce(static_input)
            graph_result = tp_group.all_reduce_low_sm(graph_stage)
        assert graph_result.data_ptr() == graph_stage.data_ptr()

        for replay in range(4):
            static_input.fill_(rank + replay + 1)
            graph.replay()
            torch.accelerator.synchronize()
            expected_value = world_size * (world_size + 1) // 2 + replay * world_size
            assert torch.all(graph_result == expected_value).cpu().item()


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_symm_mem_allreduce(
    monkeypatch: pytest.MonkeyPatch, tp_size, pipeline_parallel_size
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    q = mp.get_context("spawn").Queue()
    mp.spawn(symm_mem_allreduce_worker, args=(world_size, q), nprocs=world_size)
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
        if val is not None:
            pytest.skip(val)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Low-SM symmetric-memory all-reduce is CUDA-only.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_low_sm_symm_mem_allreduce():
    world_size = 8
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    q = mp.get_context("spawn").Queue()
    mp.spawn(
        low_sm_symm_mem_allreduce_worker,
        args=(world_size, q),
        nprocs=world_size,
    )
    try:
        val = q.get(timeout=1)
    except queue.Empty:
        val = None
    finally:
        cleanup_dist_env_and_memory()
        if val is not None:
            pytest.skip(val)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="SymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_dp_with_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch):
    world_size = 4
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    # Verify that the DataParallel runs without error
    engine_args = EngineArgs(
        model="distilbert/distilgpt2",
        enforce_eager=True,
        enable_prefix_caching=True,
        data_parallel_size=2,
        tensor_parallel_size=2,
        data_parallel_backend="mp",
    )
    LLMEngine.from_engine_args(engine_args)
