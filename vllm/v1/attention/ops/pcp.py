# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.ops.cp_common import (
    DirectCPWorkspace,
    direct_cp_multicast_enabled,
)

logger = init_logger(__name__)


# Staged row: nvfp4_ds_mla entry (352 B), indexer FP8 K (128 B) and scale (4 B).
_HY4_STAGED_ROW_BYTES = 352 + 128 + 4
# Multicast grid: one CTA per SM, few warps each, to leave room for the GEMMs.
_HY4_MULTICAST_WARPS = 8


class HY4PCPKVExchange(DirectCPWorkspace):
    """HY V4 PCP prefill K path without all-gathers.

    ``multicast`` (side stream): norm, RoPE and quantize this rank's K into
    cache rows and multicast them to every rank's staging slot.
    ``wait_cache_insert``: wait for every rank's rows, then write them into the
    MLA and indexer caches.

    Device counters track exchanges: exchange E is complete when every source
    has signalled E + 1 times. Two staging slots alternate.
    """

    def __init__(self, group, device: torch.device, max_local_tokens: int):
        super().__init__(group, device, num_ubatches=1)
        self.max_local_tokens = max_local_tokens
        rows = self.world_size * max_local_tokens
        self.slot_bytes = (rows * _HY4_STAGED_ROW_BYTES + 15) // 16 * 16
        self.staging, _ = self._allocate((1, 2 * self.slot_bytes), torch.uint8)
        self.signal, _ = self._allocate((1, self.world_size), torch.int32)
        self.staging_mc = self._multicast_ptrs(self.staging)[0]
        self.signal_mc = self._multicast_ptrs(self.signal)[0]
        assert self.staging_mc and self.signal_mc, "HY4 PCP KV exchange needs NVLS"
        # [exchanges, finished insert CTAs, finished multicast CTAs]
        self.state = torch.zeros(3, dtype=torch.int32, device=self.device)
        self.multicast_blocks = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.stream = torch.cuda.Stream(device=self.device)
        self.inputs_ready = torch.cuda.Event()
        self.multicast_done = torch.cuda.Event()
        self.pending_tokens: int | None = None

    def multicast(
        self,
        kv_lora: torch.Tensor,
        index_k: torch.Tensor | None,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        index_cos_sin_cache: torch.Tensor | None,
        kv_norm_weight: torch.Tensor,
        kv_eps: float,
        index_norm_weight: torch.Tensor | None,
        index_norm_bias: torch.Tensor | None,
        index_eps: float,
    ) -> None:
        # An exchange without its insert would let the next wait pass early.
        assert self.pending_tokens is None, "HY4 PCP KV exchange: insert skipped"
        self.inputs_ready.record()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.inputs_ready)
            # cos/sin may be a per-step M-RoPE table, not a persistent cache.
            for t in (kv_lora, index_k, positions, cos_sin_cache, index_cos_sin_cache):
                if t is not None:
                    t.record_stream(self.stream)
            torch.ops._C.hy4_pcp_kv_norm_rope_quant_multicast(
                kv_lora,
                index_k,
                positions,
                cos_sin_cache,
                index_cos_sin_cache,
                kv_norm_weight,
                kv_eps,
                index_norm_weight,
                index_norm_bias,
                index_eps,
                self.state,
                self.staging_mc,
                self.signal_mc,
                self.world_size,
                self.rank,
                self.max_local_tokens,
                self.multicast_blocks,
                _HY4_MULTICAST_WARPS,
                True,
                False,
            )
            self.multicast_done.record()
        self.pending_tokens = kv_lora.shape[0]

    def wait_cache_insert(
        self,
        slot_mapping: torch.Tensor,
        mla_cache: torch.Tensor,
        index_cache: torch.Tensor | None = None,
        index_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Insert every rank's rows; slot mappings are rank-major."""
        num_tokens = self.pending_tokens
        assert num_tokens is not None, "HY4 PCP KV exchange: nothing multicast"
        self.pending_tokens = None
        # Keep the spinning wait off the SMs our own multicast still needs.
        torch.cuda.current_stream().wait_event(self.multicast_done)
        rows = self.world_size * num_tokens
        if index_slot_mapping is not None:
            index_slot_mapping = index_slot_mapping[:rows]
        torch.ops._C.hy4_pcp_kv_wait_cache_insert(
            self.staging[0],
            self.signal[0],
            self.state,
            slot_mapping[:rows],
            mla_cache,
            index_cache,
            index_slot_mapping,
            self.world_size,
            self.max_local_tokens,
            num_tokens,
            max(1, min(2048, (rows + 7) // 8)),
            False,
        )


_hy4_kv_exchange: HY4PCPKVExchange | None = None
_hy4_kv_exchange_initialized = False


def init_hy4_pcp_kv_exchange(vllm_config) -> HY4PCPKVExchange | None:
    """The process-wide exchange, created on first call. Collective over PCP."""
    global _hy4_kv_exchange, _hy4_kv_exchange_initialized
    if _hy4_kv_exchange_initialized:
        return _hy4_kv_exchange
    _hy4_kv_exchange_initialized = True
    pcp_group = get_pcp_group()
    if not (
        current_platform.is_device_capability_family(100)
        and hasattr(torch.ops._C, "hy4_pcp_kv_norm_rope_quant_multicast")
        and direct_cp_multicast_enabled(pcp_group, torch.bfloat16, None)
    ):
        reason = "HY4 PCP prologue needs SM100 and NVLS multicast"
        if envs.VLLM_HY4_PCP_PROLOGUE == "1":
            raise RuntimeError(f"{reason} (VLLM_HY4_PCP_PROLOGUE=1).")
        logger.info("%s; using the all-gather path.", reason)
        return None
    # Upper bound on one rank's padded share of a batch.
    scheduler = vllm_config.scheduler_config
    spec = vllm_config.speculative_config
    tokens_per_seq = 1 + (spec.num_speculative_tokens or 0) if spec else 1
    max_local = (
        math.ceil(scheduler.max_num_batched_tokens / pcp_group.world_size)
        + 2 * scheduler.max_num_seqs * tokens_per_seq
    )
    device = torch.device("cuda", torch.accelerator.current_device_index())
    _hy4_kv_exchange = HY4PCPKVExchange(pcp_group.device_group, device, max_local)
    logger.info(
        "HY4 PCP prologue: up to %d local tokens, %.1f MiB per staging slot.",
        max_local,
        _hy4_kv_exchange.slot_bytes / 2**20,
    )
    return _hy4_kv_exchange


def _gather_prefill_cache_inputs(
    tensors: tuple[torch.Tensor, ...],
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Keep replicated decode writes local and gather partitioned prefills."""
    local_num_tokens = tensors[0].shape[0]
    assert all(tensor.shape[0] == local_num_tokens for tensor in tensors)
    assert 0 <= num_decode_tokens <= local_num_tokens

    if num_decode_tokens == local_num_tokens:
        return tensors, slot_mapping[:num_decode_tokens]

    pcp_group = get_pcp_group()
    gathered_prefills = tuple(
        pcp_group.all_gather(tensor[num_decode_tokens:].contiguous(), dim=0)
        for tensor in tensors
    )
    pcp_size = pcp_group.world_size
    gathered_slot_mapping = slot_mapping[: pcp_size * local_num_tokens]
    if num_decode_tokens == 0:
        return gathered_prefills, gathered_slot_mapping

    cache_inputs = tuple(
        torch.cat((tensor[:num_decode_tokens], gathered_prefill), dim=0)
        for tensor, gathered_prefill in zip(tensors, gathered_prefills)
    )
    rank_slot_mappings = gathered_slot_mapping.view(pcp_size, local_num_tokens)
    cache_slot_mapping = torch.cat(
        (
            rank_slot_mappings[0, :num_decode_tokens],
            rank_slot_mappings[:, num_decode_tokens:].flatten(),
        )
    )
    return cache_inputs, cache_slot_mapping


def maybe_gather_mla_latent_cache_inputs(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    slot_mapping: torch.Tensor | None,
    num_decode_tokens: int | None,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not use_pcp or num_decode_tokens is None:
        return kv_c_normed, k_pe, slot_mapping
    assert slot_mapping is not None
    num_tokens = kv_c_normed.shape[0]
    k_pe_flat = k_pe.reshape(num_tokens, -1)
    (cache_kv_c, cache_k_pe_flat), cache_slot_mapping = _gather_prefill_cache_inputs(
        (kv_c_normed, k_pe_flat),
        slot_mapping,
        num_decode_tokens,
    )
    cache_k_pe = cache_k_pe_flat.view(-1, *k_pe.shape[1:])
    return cache_kv_c, cache_k_pe, cache_slot_mapping


def maybe_gather_indexer_k(
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_decode_tokens: int,
    use_pcp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not use_pcp:
        return k, slot_mapping
    (cache_k,), cache_slot_mapping = _gather_prefill_cache_inputs(
        (k,), slot_mapping, num_decode_tokens
    )
    return cache_k, cache_slot_mapping


def finalize_mla_pcp_decode(
    output: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    if output.shape[1] < num_heads:
        output = get_pcp_group().all_gather(output, dim=1)
    elif output.shape[1] > num_heads:
        head_start = get_tp_group().rank_in_group * num_heads
        output = output[:, head_start : head_start + num_heads]
    return output
