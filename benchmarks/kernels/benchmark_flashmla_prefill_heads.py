# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native 64-head sparse prefill with a zero-padded 128-head query."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(0)
    rows = []
    for nq, nk in ((256, 8192), (8192, 8192), (8192, 128000), (8192, 256000)):
        q = torch.randn(nq, 64, 576, dtype=torch.bfloat16, device="cuda")
        padded = torch.zeros(nq, 128, 576, dtype=q.dtype, device=q.device)
        padded[:, :64] = q
        kv = torch.randn(nk, 1, 576, dtype=q.dtype, device=q.device)
        indices = torch.randint(nk, (nq, 1, 2048), dtype=torch.int32, device=q.device)
        sinks = torch.randn(64, device=q.device)
        padded_sinks = torch.full((128,), -torch.inf, device=q.device)
        padded_sinks[:64] = sinks
        lengths = torch.full((nq,), 2048, dtype=torch.int32, device=q.device)
        outs = {
            h: torch.empty(nq, h, 512, dtype=q.dtype, device=q.device)
            for h in (64, 128)
        }

        inputs = {
            h: (
                q if h == 64 else padded,
                kv,
                indices,
                sinks if h == 64 else padded_sinks,
                lengths,
                outs[h],
            )
            for h in (64, 128)
        }

        def run(query, values, selected, sink, topk_lengths, output):
            return flash_mla_sparse_fwd(
                query,
                values,
                selected,
                576**-0.5,
                attn_sink=sink,
                topk_length=topk_lengths,
                out=output,
            )

        for h in (64, 128):
            for _ in range(5):
                run(*inputs[h])
        a, b = outs[64].float(), outs[128][:, :64].float()
        relative_l2 = ((a - b).norm() / b.norm()).item()
        assert relative_l2 < 0.01, relative_l2
        del a, b
        samples = {64: [], 128: []}
        for order in ((64, 128), (128, 64), (64, 128)):
            for h in order:
                times = bench_gpu_time_with_cupti(
                    run,
                    input_args=inputs[h],
                    use_cuda_graph=True,
                    cold_l2_cache=True,
                    dry_run_iters=5,
                    repeat_iters=args.repeat,
                )
                samples[h].append(statistics.median(times))
        for h in (64, 128):
            ms = statistics.median(samples[h])
            # Useful QK and PV FLOPs for the 64 real heads, excluding padding.
            useful_flops = 2 * nq * 64 * 2048 * (576 + 512)
            row = dict(
                query_tokens=nq,
                kv_tokens=nk,
                kernel_heads=h,
                ms=ms,
                useful_tflops=useful_flops / (ms * 1e9),
                trials_ms=samples[h],
                relative_l2=relative_l2,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        del q, padded, kv, indices, sinks, padded_sinks, lengths, outs, inputs
    args.output.write_text(
        json.dumps(
            dict(
                gpu=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                torch=torch.__version__,
                timing="CUPTI, CUDA graph, cold L2, kernel only; staging excluded",
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
