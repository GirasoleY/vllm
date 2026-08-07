# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import triton
from vllm.utils.argparse_utils import FlexibleArgumentParser

SCENARIOS = {
    "single-60k": [60_000],
    "single-300k": [300_000],
    "skew-2": [60_000, 300_000],
    "skew-4": [60_000, 100_000, 180_000, 300_000],
    "skew-8": [
        60_000,
        60_000,
        80_000,
        100_000,
        140_000,
        180_000,
        240_000,
        300_000,
    ],
}
DTYPES = {
    "fp8": torch.float8_e4m3fn,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def make_inputs(
    seq_lens: list[int],
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    batch_size = len(seq_lens)
    seq_starts_list = [13 + (17 * req_id) % block_size for req_id in range(batch_size)]
    blocks_per_req = [
        math.ceil((seq_start + seq_len) / block_size)
        for seq_start, seq_len in zip(seq_starts_list, seq_lens)
    ]
    total_blocks = sum(blocks_per_req)
    src_cache = torch.empty(
        (total_blocks, block_size, entry_size), dtype=dtype, device="cuda"
    )

    block_table = torch.zeros(
        (batch_size, max(blocks_per_req)), dtype=torch.int32, device="cuda"
    )
    physical_blocks = torch.randperm(total_blocks, dtype=torch.int32, device="cuda")
    offset = 0
    for req_id, num_blocks in enumerate(blocks_per_req):
        block_table[req_id, :num_blocks] = physical_blocks[offset : offset + num_blocks]
        offset += num_blocks

    cu_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seq_lens[1:] = torch.tensor(seq_lens, dtype=torch.int32, device="cuda").cumsum(
        dim=0
    )
    seq_starts = torch.tensor(seq_starts_list, dtype=torch.int32, device="cuda")
    dst = torch.empty((sum(seq_lens), entry_size), dtype=dtype, device="cuda")
    return src_cache, dst, block_table, cu_seq_lens, seq_starts


@torch.inference_mode()
def run_scenario(
    name: str,
    seq_lens: list[int],
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    warmup_ms: int,
    rep_ms: int,
) -> None:
    src_cache, dst, block_table, cu_seq_lens, seq_starts = make_inputs(
        seq_lens, block_size, entry_size, dtype
    )
    batch_size = len(seq_lens)

    def run() -> None:
        ops.cp_gather_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            batch_size,
            seq_starts,
        )

    latency_ms = triton.testing.do_bench(
        run, warmup=warmup_ms, rep=rep_ms, return_mode="median"
    )
    bytes_moved = 2 * dst.numel() * dst.element_size()
    bandwidth_gbps = bytes_moved / latency_ms / 1e6
    lengths = ",".join(str(seq_len) for seq_len in seq_lens)
    print(
        f"{name:10s} batch={batch_size:2d} total={sum(seq_lens):7d} "
        f"latency={latency_ms * 1e3:9.2f} us "
        f"bandwidth={bandwidth_gbps:8.1f} GB/s lengths=[{lengths}]"
    )


def main() -> None:
    parser = FlexibleArgumentParser(description="Benchmark ops.cp_gather_cache")
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--dtype", choices=DTYPES, default="fp8")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--entry-size", type=int, default=576)
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    selected = (
        SCENARIOS
        if args.scenario == "all"
        else {args.scenario: SCENARIOS[args.scenario]}
    )
    for name, seq_lens in selected.items():
        run_scenario(
            name,
            seq_lens,
            args.block_size,
            args.entry_size,
            DTYPES[args.dtype],
            args.warmup_ms,
            args.rep_ms,
        )
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
