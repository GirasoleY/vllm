# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible imports for canonical KV placement mapping."""

from vllm.distributed.kv_transfer.canonical_mapping import (
    CANONICAL_FORMAT_VERSION,
    canonical_format_id,
    derive_canonical_mappings,
    derive_rank_canonical_mappings,
)
from vllm.distributed.kv_transfer.kv_placement import (
    CanonicalPageMapping,
    CopyRun,
)

__all__ = [
    "CANONICAL_FORMAT_VERSION",
    "CanonicalPageMapping",
    "CopyRun",
    "canonical_format_id",
    "derive_canonical_mappings",
    "derive_rank_canonical_mappings",
]
