# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("mamba_cache_mode", "enable_prefix_caching", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", True, 0, 65_536, id="align-prefix-cache"),
        pytest.param("align", True, 7, 65_543, id="align-with-speculation"),
        pytest.param("none", False, 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    enable_prefix_caching: bool,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = MLAAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        cache_config=SimpleNamespace(
            mamba_cache_mode=mamba_cache_mode,
            enable_prefix_caching=enable_prefix_caching,
        ),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        model_state=SimpleNamespace(
            get_additional_cg_support=lambda: (),
            num_new_sampled_tokens_per_step=1,
        ),
        speculator=None,
        req_states=None,
        input_buffers=SimpleNamespace(query_start_loc=None),
        vocab_size=1,
        max_num_reqs=8,
        max_num_tokens=32_768,
        dcp_size=dcp_size,
        dcp_rank=0,
        cp_interleave=1,
    )

    class _AttentionCGSupport:
        def narrow(self):
            return self

    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *_args: (
            [],
            _AttentionCGSupport(),
            [attention_block_size, mamba_block_size],
        ),
    )

    captured: dict[str, object] = {}

    class _CapturedBlockTables(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise _CapturedBlockTables

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(_CapturedBlockTables):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured["max_num_blocks_per_group"] == [86, expected]


@pytest.mark.parametrize(
    ("overwrite", "existing_blocks", "new_block_ids"),
    [
        pytest.param(True, 3, [1, 2, 3, 4, 5], id="overwrite"),
        pytest.param(False, 3, [4, 5], id="append"),
    ],
)
def test_append_block_ids_rejects_write_past_row_capacity(
    overwrite: bool,
    existing_blocks: int,
    new_block_ids: list[int],
):
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, existing_blocks]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=(new_block_ids,),
            overwrite=overwrite,
        )

    assert block_tables.num_blocks.np[0, 1] == existing_blocks
