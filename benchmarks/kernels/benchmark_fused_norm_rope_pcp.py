# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark the GLM-5.2 fused PCP norm/RoPE cache path.

Run on one NVLink domain, for example:

  torchrun --standalone --nproc-per-node=8 \
    benchmarks/kernels/benchmark_fused_norm_rope_pcp.py \
    --local-tokens 512 2048 4096

The legacy comparison models the current pure-prefill PCP path exactly:
local fused norm/RoPE materialization, one indexer-K all-gather and cache
insert, two MLA all-gathers, then MLA concat/quant/cache insert.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from vllm import _custom_ops as ops
from vllm.v1.attention.ops.dcp_utils import DirectPCPFusedNormRopeWorkspace

# Loading ``vllm.models.deepseek_v32.common.kernels`` normally executes the
# model package initializer, which pulls model-only CUDA dependencies into this
# standalone benchmark. Load this self-contained Triton module by path instead.
_KERNELS_PATH = (
    Path(__file__).resolve().parents[2] / "vllm/models/deepseek_v32/common/kernels.py"
)
_KERNELS_SPEC = importlib.util.spec_from_file_location(
    "_benchmark_deepseek_v32_kernels", _KERNELS_PATH
)
assert _KERNELS_SPEC is not None and _KERNELS_SPEC.loader is not None
fused_kernels = importlib.util.module_from_spec(_KERNELS_SPEC)
_KERNELS_SPEC.loader.exec_module(fused_kernels)

Q_DIM = 2048
KV_DIM = 512
ROPE_DIM = 64
INDEX_DIM = 128
TOPK = 2048
INDEX_ROW_BYTES = INDEX_DIM + 4
EPS = 1.0e-6
FP8 = torch.float8_e4m3fn


def make_cos_sin(max_position: int, device: torch.device) -> torch.Tensor:
    half = ROPE_DIM // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(half, dtype=torch.float32, device=device) / half)
    )
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    frequencies = torch.outer(positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1)


def all_gather_tensor(local: torch.Tensor) -> torch.Tensor:
    output = local.new_empty((dist.get_world_size() * local.shape[0], *local.shape[1:]))
    dist.all_gather_into_tensor(output, local)
    return output


def fp8_max_ulp(left: torch.Tensor, right: torch.Tensor) -> int:
    def ordered(value: torch.Tensor) -> torch.Tensor:
        raw = value.contiguous().view(torch.uint8).to(torch.int16)
        return torch.where(raw >= 0x80, 0xFF - raw, raw + 0x80)

    return int((ordered(left) - ordered(right)).abs().max().item())


def distributed_time_ms(
    function: Callable[[], object], warmups: int, iterations: int
) -> float:
    for _ in range(warmups):
        function()
    torch.accelerator.synchronize()
    dist.barrier()
    samples = []
    for _ in range(iterations):
        # Align each collective invocation, and use synchronized wall time so
        # auxiliary NCCL/symmetric-memory streams cannot escape a CUDA event
        # recorded on the current compute stream.
        dist.barrier()
        torch.accelerator.synchronize()
        start = time.perf_counter_ns()
        function()
        torch.accelerator.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1.0e6)
    # A collective completes at the pace of its slowest participant. Report
    # the maximum per-rank mean rather than a deceptively low rank-0 number.
    result = torch.tensor(statistics.median(samples), device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


class Case:
    def __init__(self, local_tokens: int, max_local_tokens: int, device: torch.device):
        self.local_tokens = local_tokens
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.global_tokens = local_tokens * self.world_size
        self.device = device
        self.block_size = 64
        num_blocks = (self.global_tokens + self.block_size - 1) // self.block_size

        # Model parameters are identical across PCP ranks.
        torch.manual_seed(1234)
        self.q_weight = torch.randn(Q_DIM, device=device, dtype=torch.bfloat16)
        self.kv_weight = torch.randn(KV_DIM, device=device, dtype=torch.bfloat16)
        self.index_weight = torch.randn(INDEX_DIM, device=device, dtype=torch.float32)
        self.index_bias = torch.randn(INDEX_DIM, device=device, dtype=torch.float32)
        self.mla_scale = torch.tensor([0.3], device=device, dtype=torch.float32)
        self.cos_sin = make_cos_sin(max(self.global_tokens, 8192), device)

        # Activations differ by source rank and are gathered rank-major.
        torch.manual_seed(9000 + self.rank)
        self.positions = torch.arange(
            self.rank * local_tokens,
            (self.rank + 1) * local_tokens,
            device=device,
            dtype=torch.int64,
        )
        self.q_c = torch.randn(local_tokens, Q_DIM, device=device, dtype=torch.bfloat16)
        self.kv_c = torch.randn(
            local_tokens, KV_DIM, device=device, dtype=torch.bfloat16
        )
        self.k_pe = torch.randn(
            local_tokens, ROPE_DIM, device=device, dtype=torch.bfloat16
        )
        self.index_k = torch.randn(
            local_tokens, INDEX_DIM, device=device, dtype=torch.bfloat16
        )
        self.topk = torch.empty(local_tokens, TOPK, device=device, dtype=torch.int32)
        self.slots = torch.arange(self.global_tokens, device=device, dtype=torch.int64)

        self.kv_out = torch.empty_like(self.kv_c)
        self.k_pe_out = torch.empty_like(self.k_pe)
        self.index_out = torch.empty_like(self.index_k)
        self.gathered_kv = torch.empty(
            self.global_tokens, KV_DIM, device=device, dtype=torch.bfloat16
        )
        self.gathered_k_pe = torch.empty(
            self.global_tokens, ROPE_DIM, device=device, dtype=torch.bfloat16
        )
        self.gathered_index = torch.empty(
            self.global_tokens, INDEX_DIM, device=device, dtype=torch.bfloat16
        )
        self.legacy_mla_cache = torch.zeros(
            num_blocks,
            self.block_size,
            KV_DIM + ROPE_DIM,
            device=device,
            dtype=torch.uint8,
        )
        self.legacy_index_cache = torch.zeros(
            num_blocks,
            self.block_size,
            INDEX_ROW_BYTES,
            device=device,
            dtype=torch.uint8,
        )
        self.fused_mla_cache = torch.zeros_like(self.legacy_mla_cache)
        self.fused_index_cache = torch.zeros_like(self.legacy_index_cache)
        self.workspace = DirectPCPFusedNormRopeWorkspace(
            dist.group.WORLD, device, max_local_tokens=max_local_tokens
        )

    def legacy_norm(self) -> torch.Tensor:
        return fused_kernels.fused_norm_rope(
            self.positions,
            self.q_c,
            self.q_weight,
            EPS,
            self.kv_c,
            self.kv_weight,
            EPS,
            self.k_pe,
            self.cos_sin,
            self.index_k,
            self.index_weight,
            self.index_bias,
            EPS,
            self.cos_sin,
            self.topk,
            has_indexer=True,
            index_rope_interleave=True,
            kv_c_out=self.kv_out,
            k_pe_out=self.k_pe_out,
            index_k_out=self.index_out,
        )

    def legacy_index_pipe(self) -> None:
        dist.all_gather_into_tensor(self.gathered_index, self.index_out)
        ops.indexer_k_quant_and_cache(
            self.gathered_index,
            self.legacy_index_cache,
            self.slots,
            INDEX_DIM,
            "ue8m0",
        )

    def legacy_mla_pipe(self) -> None:
        dist.all_gather_into_tensor(self.gathered_kv, self.kv_out)
        dist.all_gather_into_tensor(self.gathered_k_pe, self.k_pe_out)
        ops.concat_and_cache_mla(
            self.gathered_kv,
            self.gathered_k_pe,
            self.legacy_mla_cache,
            self.slots,
            "fp8",
            self.mla_scale,
        )

    def legacy_total(self) -> None:
        self.legacy_norm()
        self.legacy_index_pipe()
        self.legacy_mla_pipe()

    def fused_dispatch(self) -> torch.Tensor:
        return self.workspace.dispatch(
            positions=self.positions,
            q_c=self.q_c,
            q_weight=self.q_weight,
            q_eps=EPS,
            kv_c=self.kv_c,
            kv_weight=self.kv_weight,
            mla_k_scale=self.mla_scale,
            kv_eps=EPS,
            k_pe=self.k_pe,
            k_pe_cos_sin=self.cos_sin,
            index_k=self.index_k,
            index_weight=self.index_weight,
            index_bias=self.index_bias,
            index_eps=EPS,
            index_cos_sin=self.cos_sin,
            topk_indices=self.topk,
        )

    def fused_combine(self) -> None:
        self.workspace.combine(
            local_tokens=self.local_tokens,
            mla_slot_mapping=self.slots,
            index_slot_mapping=self.slots,
            mla_cache=self.fused_mla_cache,
            index_cache=self.fused_index_cache,
        )

    def fused_total(self) -> torch.Tensor:
        q_out = self.fused_dispatch()
        self.fused_combine()
        return q_out

    def validate(self) -> tuple[int, int]:
        # Build a direct-cache reference from rank-major global activations.
        global_positions = all_gather_tensor(self.positions)
        global_q = all_gather_tensor(self.q_c)
        global_kv = all_gather_tensor(self.kv_c)
        global_k_pe = all_gather_tensor(self.k_pe)
        global_index = all_gather_tensor(self.index_k)
        ref_mla = torch.zeros_like(self.fused_mla_cache)
        ref_index = torch.zeros_like(self.fused_index_cache)
        ref_topk = torch.empty(
            self.global_tokens, TOPK, device=self.device, dtype=torch.int32
        )
        ref_q = fused_kernels.fused_norm_rope(
            global_positions,
            global_q,
            self.q_weight,
            EPS,
            global_kv,
            self.kv_weight,
            EPS,
            global_k_pe,
            self.cos_sin,
            global_index,
            self.index_weight,
            self.index_bias,
            EPS,
            self.cos_sin,
            ref_topk,
            slot_mapping=self.slots,
            indexer_k_cache=ref_index,
            mla_kv_cache=ref_mla,
            mla_kv_cache_dtype="fp8",
            mla_k_scale=self.mla_scale,
            has_indexer=True,
            index_rope_interleave=True,
        )
        got_q = self.fused_total()
        torch.accelerator.synchronize()
        rank_start = self.rank * self.local_tokens
        torch.testing.assert_close(
            got_q.float(),
            ref_q[rank_start : rank_start + self.local_tokens].float(),
            rtol=1.0e-2,
            atol=1.0e-2,
        )
        assert bool((self.topk == -1).all())
        mla_ulp = fp8_max_ulp(self.fused_mla_cache.view(FP8), ref_mla.view(FP8))
        # Index cache packs all FP8 values first, then one FP32 scale per token.
        block_bytes = self.block_size * INDEX_ROW_BYTES
        got_flat = self.fused_index_cache.reshape(-1)
        ref_flat = ref_index.reshape(-1)
        value_bytes = self.block_size * INDEX_DIM
        index_ulp = 0
        for block in range(self.fused_index_cache.shape[0]):
            start = block * block_bytes
            stop = start + value_bytes
            index_ulp = max(
                index_ulp,
                fp8_max_ulp(
                    got_flat[start:stop].view(FP8), ref_flat[start:stop].view(FP8)
                ),
            )
            got_scale = got_flat[stop : start + block_bytes].view(torch.float32)
            ref_scale = ref_flat[stop : start + block_bytes].view(torch.float32)
            torch.testing.assert_close(got_scale, ref_scale, rtol=0, atol=0)
        if mla_ulp > 1 or index_ulp > 1:
            raise AssertionError(
                f"FP8 cache mismatch: MLA={mla_ulp} ULP, index={index_ulp} ULP"
            )
        return mla_ulp, index_ulp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extension",
        type=str,
        default=None,
        help="Optional standalone prototype .so to load before benchmarking",
    )
    parser.add_argument(
        "--local-tokens", type=int, nargs="+", default=[512, 2048, 4096]
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    dist.init_process_group("nccl")
    if args.extension is not None:
        torch.ops.load_library(args.extension)
    device = torch.device("cuda", local_rank)
    world_size = dist.get_world_size()
    if world_size not in (2, 4, 8):
        raise ValueError(f"benchmark requires PCP2/4/8, got PCP{world_size}")
    max_local_tokens = max(args.local_tokens)

    if dist.get_rank() == 0:
        print(
            "local  global  legacy_norm  index_AG+cache  2xMLA_AG+cache  "
            "legacy_total  dispatch  combine  fused_total  speedup  cache_ULP"
        )
    for local_tokens in args.local_tokens:
        case = Case(local_tokens, max_local_tokens, device)
        mla_ulp, index_ulp = case.validate()
        # Populate the legacy materializations before isolated pipe timings.
        case.legacy_norm()
        torch.accelerator.synchronize()
        legacy_norm_ms = distributed_time_ms(
            case.legacy_norm, args.warmups, args.iterations
        )
        legacy_index_ms = distributed_time_ms(
            case.legacy_index_pipe, args.warmups, args.iterations
        )
        legacy_mla_ms = distributed_time_ms(
            case.legacy_mla_pipe, args.warmups, args.iterations
        )
        legacy_total_ms = distributed_time_ms(
            case.legacy_total, args.warmups, args.iterations
        )
        # Leave a fully acquired payload for the isolated combine timing.
        case.fused_dispatch()
        torch.accelerator.synchronize()
        dispatch_ms = distributed_time_ms(
            case.fused_dispatch, args.warmups, args.iterations
        )
        combine_ms = distributed_time_ms(
            case.fused_combine, args.warmups, args.iterations
        )
        fused_total_ms = distributed_time_ms(
            case.fused_total, args.warmups, args.iterations
        )
        if dist.get_rank() == 0:
            print(
                f"{local_tokens:5d}  {case.global_tokens:6d}  "
                f"{legacy_norm_ms:11.3f}  {legacy_index_ms:14.3f}  "
                f"{legacy_mla_ms:15.3f}  {legacy_total_ms:12.3f}  "
                f"{dispatch_ms:8.3f}  {combine_ms:7.3f}  {fused_total_ms:11.3f}  "
                f"{legacy_total_ms / fused_total_ms:7.2f}x  "
                f"{mla_ulp}/{index_ulp}"
            )
        del case
        torch.accelerator.empty_cache()
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
