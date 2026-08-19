# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import resource
import threading
import time

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm.logger import init_logger
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)

logger = init_logger(__name__)
_RUSAGE_THREAD = getattr(resource, "RUSAGE_THREAD", 1)


def sync_cudagraph_and_dp_padding(
    cudagraph_manager: CudaGraphManager | None,
    desired_batch_desc: BatchExecutionDescriptor,
    num_tokens: int,
    num_reqs: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    num_active_loras: int = 0,
    sync_phase: str = "target",
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    """
    Coordinates the batch descriptor and DP padding across all ranks.

    Returns (synced_batch_desc, num_tokens_across_dp).
    """
    assert dp_size > 1, "DP size must be greater than 1"
    group = get_dp_group().cpu_group
    tensor = torch.zeros(4, dp_size, dtype=torch.int32, device="cpu")
    tensor[0][dp_rank] = num_tokens
    tensor[1][dp_rank] = desired_batch_desc.cg_mode.value
    tensor[2][dp_rank] = uniform_token_count or 0  # (0 means None)
    tensor[3][dp_rank] = max_query_len or -1  # (-1 means None)

    timing_threshold_ns = int(envs.VLLM_DEBUG_HOST_TIMING_THRESHOLD_MS * 1_000_000)
    timing_enabled = timing_threshold_ns > 0
    if timing_enabled:
        start_wall_ns = time.perf_counter_ns()
        start_cpu_ns = time.thread_time_ns()
        start_usage = resource.getrusage(_RUSAGE_THREAD)

    with record_function_or_nullcontext(f"worker_dp_sync: {sync_phase}"):
        dist.all_reduce(tensor, group=group)

    if timing_enabled:
        end_wall_ns = time.perf_counter_ns()
        elapsed_ns = end_wall_ns - start_wall_ns
        if elapsed_ns >= timing_threshold_ns:
            end_usage = resource.getrusage(_RUSAGE_THREAD)
            logger.info(
                "Worker DP-metadata sync timing: dp_rank=%d tid=%d phase=%s "
                "start_ns=%d wall_ms=%.3f cpu_ms=%.3f num_reqs=%d "
                "num_tokens=%d nvcsw=%d nivcsw=%d",
                dp_rank,
                threading.get_native_id(),
                sync_phase,
                start_wall_ns,
                elapsed_ns / 1e6,
                (time.thread_time_ns() - start_cpu_ns) / 1e6,
                num_reqs,
                num_tokens,
                end_usage.ru_nvcsw - start_usage.ru_nvcsw,
                end_usage.ru_nivcsw - start_usage.ru_nivcsw,
            )

    num_tokens_across_dp = tensor[0]
    cg_mode_across_dp = tensor[1]
    uniform_token_counts_across_dp = tensor[2]
    max_query_lens_across_dp = tensor[3]

    if torch.all(num_tokens_across_dp == 0).item():
        synced_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE, num_tokens=0, num_reqs=0
        )
        return synced_desc, None

    synced_cg_mode = CUDAGraphMode(int(cg_mode_across_dp.min().item()))

    # If any rank wants to run eager, all ranks run eager
    if synced_cg_mode == CUDAGraphMode.NONE:
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=desired_batch_desc.num_active_loras,
        ), num_tokens_across_dp

    assert cudagraph_manager is not None, (
        "cudagraph_manager should only be None during profile run, "
        "where synced_cg_mode must be NONE across all DP ranks"
    )
    synced_num_tokens = int(num_tokens_across_dp.max().item())
    synced_uniform_token_count = uniform_token_counts_across_dp[0]
    # If ranks disagree on the uniform token count, or its 0 (means None) set to None
    if synced_uniform_token_count == 0 or not torch.all(
        uniform_token_counts_across_dp == synced_uniform_token_count
    ):
        synced_uniform_token_count = None

    # Varlen decode graphs are selected by the query-length bound, so ranks must agree
    # on it or they pad to different token counts below.
    synced_max_query_len: int | None = None
    if bool(torch.all(max_query_lens_across_dp != -1).item()):
        synced_max_query_len = int(max_query_lens_across_dp.max().item())

    # Dispatch for the final synced values, use num_reqs instead of synced_num_reqs
    # so we don't perform request padding for PIECEWISE graphs.
    # num_active_loras is per-rank and doesn't need cross-rank agreement.
    synced_desc = cudagraph_manager.dispatch(
        num_reqs,
        synced_num_tokens,
        synced_uniform_token_count,
        num_active_loras=num_active_loras,
        max_query_len=synced_max_query_len,
    )

    # Update num_tokens_across_dp to reflect padded size.
    num_tokens_across_dp[:] = synced_desc.num_tokens

    return synced_desc, num_tokens_across_dp


def dispatch_cg_and_sync_dp(
    cudagraph_manager: CudaGraphManager | None,
    num_reqs: int,
    num_tokens: int,
    uniform_token_count: int | None,
    dp_size: int,
    dp_rank: int,
    max_query_len: int | None = None,
    need_eager: bool = False,
    num_active_loras: int = 0,
    sync_phase: str = "target",
) -> tuple[BatchExecutionDescriptor, torch.Tensor | None]:
    if need_eager:
        batch_desc = BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=num_active_loras,
        )
    else:
        assert cudagraph_manager is not None, (
            "cudagraph_manager should only be None during profile run, "
            "where need_eager must be True"
        )
        batch_desc = cudagraph_manager.dispatch(
            num_reqs,
            num_tokens,
            uniform_token_count,
            num_active_loras=num_active_loras,
            max_query_len=max_query_len,
        )

    if dp_size == 1:
        return batch_desc, None

    return sync_cudagraph_and_dp_padding(
        cudagraph_manager,
        batch_desc,
        num_tokens,
        num_reqs,
        uniform_token_count,
        dp_size,
        dp_rank,
        max_query_len=max_query_len,
        num_active_loras=num_active_loras,
        sync_phase=sync_phase,
    )
