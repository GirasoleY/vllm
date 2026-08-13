# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the fused PCP prototype op for rapid microbenchmark iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.cpp_extension import load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-directory", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    args.build_directory.mkdir(parents=True, exist_ok=True)
    library = load(
        name="fused_norm_rope_pcp_ext",
        sources=[
            str(
                repo_root
                / "csrc/libtorch_stable/attention/dcp_utils/fused_norm_rope_pcp.cu"
            )
        ],
        build_directory=str(args.build_directory),
        extra_cuda_cflags=[
            "-O3",
            "-DUSE_CUDA",
            "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
        ],
        is_python_module=False,
        verbose=True,
    )
    print(library)


if __name__ == "__main__":
    main()
