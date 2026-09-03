# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.execution import DraftAttentionMetadataSource


@pytest.mark.parametrize("is_kv_producer", [False, True])
def test_proposal_extends_completion_event_for_kv_producer(is_kv_producer: bool):
    calls = []
    main_stream = object()
    copy_stream = SimpleNamespace(
        wait_stream=lambda stream: calls.append(("wait", stream))
    )
    copy_event = SimpleNamespace(record=lambda stream: calls.append(("record", stream)))
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(is_kv_producer=is_kv_producer)
    )
    runner.main_stream = main_stream
    runner.output_copy_stream = copy_stream

    runner._record_kv_producer_completion(
        SimpleNamespace(copy_event=copy_event)  # type: ignore[arg-type]
    )

    assert calls == (
        [("wait", main_stream), ("record", copy_stream)] if is_kv_producer else []
    )


@pytest.mark.parametrize(
    ("attention_source", "reuses_target_dp_sync"),
    [
        (DraftAttentionMetadataSource.TARGET, True),
        (DraftAttentionMetadataSource.DRAFT, False),
    ],
)
def test_prepare_identity_draft_execution_view_honors_phase_ownership(
    attention_source,
    reuses_target_dp_sync,
):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model = SimpleNamespace()
    speculator = object.__new__(DFlashSpeculator)
    speculator.execution_plan = SimpleNamespace(
        uses_target_local_initial=False,
        initial=SimpleNamespace(
            attention_metadata_source=attention_source,
            reuses_target_dp_sync=reuses_target_dp_sync,
        ),
    )
    runner.speculator = speculator

    batch = SimpleNamespace(num_tokens_after_padding=2)
    hidden_states = torch.zeros(2, 3)
    aux_hidden_states = [torch.ones(2, 3)]
    attn_metadata = {"layer": object()}
    slot_mappings = {"layer": torch.tensor([1, 2])}
    dp_sync = object()

    view, sampling_hidden, sampling_batch = runner._prepare_draft_execution_view(  # type: ignore[arg-type]
        batch,
        hidden_states,
        aux_hidden_states,
        attn_metadata,
        slot_mappings,
        dp_sync,
        None,
    )

    assert view is not None
    assert sampling_hidden is hidden_states
    assert sampling_batch is batch
    assert view.global_batch is batch
    assert view.model_batch is batch
    assert view.last_hidden_states is hidden_states
    assert view.aux_hidden_states is aux_hidden_states
    if attention_source == DraftAttentionMetadataSource.TARGET:
        assert view.attn_metadata is attn_metadata
        assert view.slot_mappings is slot_mappings
        assert view.dp_sync is dp_sync
    else:
        assert view.attn_metadata is None
        assert view.slot_mappings is None
        assert view.dp_sync is None


@pytest.mark.parametrize("target_local", [False, True])
@pytest.mark.parametrize("uses_pre_hc", [False, True])
def test_prepare_draft_execution_view_respects_layout_and_ownership(
    monkeypatch, target_local: bool, uses_pre_hc: bool
):
    class FakeDraftModelSpeculator:
        pass

    monkeypatch.setattr(
        model_runner_module, "DraftModelSpeculator", FakeDraftModelSpeculator
    )
    attention_metadata_source = (
        DraftAttentionMetadataSource.TARGET
        if target_local
        else DraftAttentionMetadataSource.DRAFT
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.speculator = FakeDraftModelSpeculator()
    runner.speculator.execution_plan = SimpleNamespace(
        uses_target_local_initial=target_local,
        initial=SimpleNamespace(
            attention_metadata_source=attention_metadata_source,
            reuses_target_dp_sync=True,
        ),
    )
    runner.pcp_manager = object()

    target_batch = SimpleNamespace(num_tokens_after_padding=2)
    global_batch = SimpleNamespace(num_tokens_after_padding=4)
    local_hidden = torch.arange(6).reshape(2, 3)
    global_hidden = torch.arange(12).reshape(4, 3)
    local_aux = local_hidden + 100
    global_aux = global_hidden + 100
    local_pre_hc = local_hidden + 200
    global_pre_hc = global_hidden + 200
    runner.model = (
        SimpleNamespace(get_mtp_target_hidden_states=lambda: local_pre_hc)
        if uses_pre_hc
        else SimpleNamespace()
    )

    def restore_token_rows(rows):
        if rows is local_hidden:
            return global_hidden
        if rows is local_aux:
            return global_aux
        if rows is local_pre_hc:
            return global_pre_hc
        raise AssertionError("unexpected token-row buffer")

    token_layout = SimpleNamespace(
        global_batch=global_batch,
        model_batch=target_batch,
        restore_token_rows=restore_token_rows,
    )
    attn_metadata = {"layer": object()}
    slot_mappings = {"layer": torch.arange(2)}
    dp_sync = object()

    view, sampling_hidden, sampling_batch = runner._prepare_draft_execution_view(
        target_batch,
        local_hidden,
        [local_aux],
        attn_metadata,
        slot_mappings,
        dp_sync,  # type: ignore[arg-type]
        token_layout,  # type: ignore[arg-type]
    )

    assert view is not None
    assert view.global_batch is global_batch
    assert sampling_batch is global_batch
    assert sampling_hidden is global_hidden
    if target_local:
        assert view.model_batch is target_batch
        assert torch.equal(
            view.last_hidden_states,
            local_pre_hc if uses_pre_hc else local_hidden,
        )
        assert view.aux_hidden_states is not None
        assert torch.equal(view.aux_hidden_states[0], local_aux)
        assert view.attn_metadata is attn_metadata
        assert view.slot_mappings is slot_mappings
        assert view.dp_sync is dp_sync
        assert view.token_layout is token_layout
    else:
        assert view.model_batch is global_batch
        assert torch.equal(
            view.last_hidden_states,
            global_pre_hc if uses_pre_hc else global_hidden,
        )
        assert view.aux_hidden_states is not None
        assert torch.equal(view.aux_hidden_states[0], global_aux)
        assert view.attn_metadata is None
        assert view.slot_mappings is None
        assert view.dp_sync is None
        assert view.token_layout is None


@pytest.mark.parametrize("has_speculator", [False, True])
def test_prepare_draft_execution_view_rejects_missing_pcp_layout(
    monkeypatch, has_speculator: bool
):
    class FakeDraftModelSpeculator:
        pass

    monkeypatch.setattr(
        model_runner_module, "DraftModelSpeculator", FakeDraftModelSpeculator
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.pcp_manager = object()
    runner.speculator = FakeDraftModelSpeculator() if has_speculator else None
    batch = SimpleNamespace(num_tokens_after_padding=2)

    with pytest.raises(RuntimeError, match="PCP sampling requires the token layout"):
        runner._prepare_draft_execution_view(
            batch,  # type: ignore[arg-type]
            torch.empty(2, 3),
            None,
            None,
            None,
            None,
            None,
        )


def test_target_local_dummy_execution_view_allows_identity_layout(monkeypatch):
    class FakeDraftModelSpeculator:
        pass

    monkeypatch.setattr(
        model_runner_module, "DraftModelSpeculator", FakeDraftModelSpeculator
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.pcp_manager = object()
    runner.model = SimpleNamespace()
    runner.speculator = FakeDraftModelSpeculator()
    runner.speculator.execution_plan = SimpleNamespace(
        uses_target_local_initial=True,
        initial=SimpleNamespace(
            attention_metadata_source=DraftAttentionMetadataSource.TARGET,
            reuses_target_dp_sync=True,
        ),
    )
    batch = SimpleNamespace(num_tokens_after_padding=2)
    hidden_states = torch.empty(2, 3)

    view, sampling_hidden, sampling_batch = runner._prepare_draft_execution_view(
        batch,  # type: ignore[arg-type]
        hidden_states,
        None,
        {},
        {},
        None,
        None,
        allow_identity_for_target_local=True,
    )

    assert view is not None
    assert view.global_batch is batch
    assert view.model_batch is batch
    assert view.token_layout is None
    assert sampling_hidden is hidden_states
    assert sampling_batch is batch


def test_shape_changing_drafter_drops_target_dp_sync(monkeypatch):
    class FakeDraftModelSpeculator:
        pass

    monkeypatch.setattr(
        model_runner_module, "DraftModelSpeculator", FakeDraftModelSpeculator
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.pcp_manager = None
    runner.model = SimpleNamespace()
    runner.speculator = FakeDraftModelSpeculator()
    runner.speculator.execution_plan = SimpleNamespace(
        uses_target_local_initial=False,
        initial=SimpleNamespace(
            attention_metadata_source=DraftAttentionMetadataSource.DRAFT,
            reuses_target_dp_sync=False,
        ),
    )
    batch = SimpleNamespace(num_tokens_after_padding=2)

    view, _, _ = runner._prepare_draft_execution_view(
        batch,  # type: ignore[arg-type]
        torch.empty(2, 3),
        None,
        None,
        None,
        object(),  # type: ignore[arg-type]
        None,
    )

    assert view is not None
    assert view.dp_sync is None


def test_prepare_draft_execution_view_rejects_missing_pre_hc_state(monkeypatch):
    class FakeDraftModelSpeculator:
        pass

    monkeypatch.setattr(
        model_runner_module, "DraftModelSpeculator", FakeDraftModelSpeculator
    )
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.pcp_manager = None
    runner.model = SimpleNamespace(get_mtp_target_hidden_states=lambda: None)
    runner.speculator = FakeDraftModelSpeculator()
    runner.speculator.execution_plan = SimpleNamespace(
        uses_target_local_initial=False,
    )
    batch = SimpleNamespace(num_tokens_after_padding=2)

    with pytest.raises(RuntimeError, match="did not retain the hidden states"):
        runner._prepare_draft_execution_view(
            batch,  # type: ignore[arg-type]
            torch.empty(2, 3),
            None,
            None,
            None,
            None,
            None,
        )


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
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
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
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
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3
