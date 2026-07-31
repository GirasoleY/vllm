# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-spec-aware helpers for constructing KV residency events."""

from collections.abc import Sequence
from typing import Any

from vllm.distributed.kv_events import BlockStored
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    maybe_convert_block_hash,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
    get_kv_cache_spec_sliding_window,
)


def build_kv_cache_stored_event(
    *,
    kv_cache_spec: KVCacheSpec,
    token_ids: Sequence[int],
    request_block_hashes: Sequence[BlockHash],
    key_hash: BlockHash,
    start: int,
    end: int,
    hash_block_size: int,
    group_idx: int,
    medium: str | None,
    enable_partial_hash_hits: bool = False,
    lora_id: int | None = None,
    lora_name: str | None = None,
    extra_keys: list[tuple[Any, ...] | None] | None = None,
    locality: str | None = None,
) -> BlockStored:
    """Build the event for one stored KV object.

    The caller owns residency: GPU cache code calls this after caching a local
    object, while external connectors call it only after the corresponding PUT
    succeeds. This helper owns only the cache-spec-specific event shape.
    """
    if start < 0 or end <= start:
        raise ValueError(f"Invalid stored-token range [{start}, {end})")
    if end > len(token_ids):
        raise ValueError(
            "Stored KV event requires the complete token prefix: "
            f"range_end={end}, available_tokens={len(token_ids)}"
        )
    if hash_block_size <= 0 or start % hash_block_size or end % hash_block_size:
        raise ValueError(
            "Stored-token range must be hash-block aligned: "
            f"start={start}, end={end}, hash_block_size={hash_block_size}"
        )

    spec_kind = get_kv_cache_spec_kind(kv_cache_spec)
    first_hash_idx = start // hash_block_size
    last_hash_idx = end // hash_block_size

    if spec_kind == KVCacheSpecKind.MAMBA:
        # A recurrent state is an indivisible snapshot of the entire prefix.
        event_hashes = [maybe_convert_block_hash(key_hash)]
        parent_block_hash = None
        event_token_ids = list(token_ids[:end])
        event_block_size = end
    elif enable_partial_hash_hits:
        # A fine-grained attention object exposes its constituent hash chain.
        if last_hash_idx > len(request_block_hashes):
            raise ValueError(
                "Stored KV event range exceeds available block hashes: "
                f"last_hash_idx={last_hash_idx}, "
                f"available_hashes={len(request_block_hashes)}"
            )
        event_hashes = [
            maybe_convert_block_hash(block_hash)
            for block_hash in request_block_hashes[first_hash_idx:last_hash_idx]
        ]
        parent_block_hash = (
            maybe_convert_block_hash(request_block_hashes[first_hash_idx - 1])
            if first_hash_idx > 0
            else None
        )
        event_token_ids = list(token_ids[start:end])
        event_block_size = hash_block_size
    else:
        event_hashes = [maybe_convert_block_hash(key_hash)]
        parent_block_hash = (
            maybe_convert_block_hash(request_block_hashes[first_hash_idx - 1])
            if first_hash_idx > 0
            else None
        )
        event_token_ids = list(token_ids[start:end])
        event_block_size = end - start

    if extra_keys is not None and len(extra_keys) != len(event_hashes):
        # Mamba and physical-object events have one externally visible key.
        # Fine-grained attention events need one entry per constituent hash.
        extra_keys = None

    return BlockStored(
        block_hashes=event_hashes,
        parent_block_hash=parent_block_hash,
        token_ids=event_token_ids,
        block_size=event_block_size,
        lora_id=lora_id,
        medium=medium,
        lora_name=lora_name,
        extra_keys=extra_keys,
        group_idx=group_idx,
        kv_cache_spec_kind=spec_kind.value,
        kv_cache_spec_sliding_window=get_kv_cache_spec_sliding_window(kv_cache_spec),
        locality=locality,
    )
