# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import ParallelConfig, SpeculativeConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.parallel import EPLBConfig
from vllm.model_executor.layers.fused_moe import layer as fused_moe_layer
from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.model_executor.models.exaone4_5_mtp import Exaone4_5_MTP
from vllm.model_executor.models.llama4_eagle import EagleLlama4ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.model_executor.models.mistral_eagle import EagleMistralForCausalLM
from vllm.model_executor.models.mistral_large_3_eagle import (
    EagleMistralLarge3ForCausalLM,
)
from vllm.v1.attention.backends import flash_attn as flash_attn_module
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode import speculator as base_spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator
from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.speculator import (
    MultiModuleMTPSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        return self.test_draft_model


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


class _MultimodalDraftModel(torch.nn.Module):
    supports_multimodal_embeddings = True

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


class _TextOnlyDraftModel(torch.nn.Module):
    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        raise AssertionError("embed_input_ids should not be called during loading")


def _mock_base_model_load(monkeypatch):
    monkeypatch.setattr(
        base_spec_module,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        DraftModelSpeculator,
        "_validate_local_argmax_reduction",
        lambda self: None,
    )


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


@pytest.mark.parametrize("replicated", [False, True])
def test_prefill_capture_matches_runtime_attention_owner(replicated):
    speculator = object.__new__(_TestSpeculator)
    speculator.replicated_pcp = replicated
    speculator.input_buffers, speculator.attn_groups = draft = object(), object()
    speculator.target_input_buffers, speculator.target_attn_groups = target = (
        object(),
        object(),
    )

    assert speculator._prefill_capture_context() == (draft if replicated else target)


def test_replicated_parallel_config_does_not_inherit_target_ep():
    target = ParallelConfig(
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
        max_parallel_loading_workers=3,
    )
    target.enable_expert_parallel = True
    target.enable_eplb = True
    target.enable_elastic_ep = True
    target.enable_ep_weight_filter = True
    target.eplb_config = EPLBConfig(num_redundant_experts=1)
    target.all2all_backend = "deepep_high_throughput"
    target.expert_placement_strategy = "round_robin"
    requested = ParallelConfig(tensor_parallel_size=2)

    actual = SpeculativeConfig._materialize_draft_parallel_config(
        target, requested, draft_is_moe=False
    )

    assert actual.prefill_context_parallel_size == 1
    assert actual.world_size == target.world_size // 2
    assert not actual.enable_expert_parallel
    assert not actual.enable_eplb
    assert not actual.enable_elastic_ep
    assert not actual.enable_ep_weight_filter
    assert actual.eplb_config == EPLBConfig()
    assert actual.eplb_config is not target.eplb_config
    assert actual.all2all_backend == ParallelConfig.all2all_backend
    assert actual.expert_placement_strategy == ParallelConfig.expert_placement_strategy
    assert actual.is_moe_model is False
    assert actual.max_parallel_loading_workers == 3


def test_sharded_parallel_config_inherits_target_ep():
    target = ParallelConfig(tensor_parallel_size=2, prefill_context_parallel_size=2)
    target.enable_expert_parallel = True
    target.enable_eplb = True
    target.enable_elastic_ep = True
    target.enable_ep_weight_filter = True
    target.eplb_config = EPLBConfig(num_redundant_experts=1)
    target.all2all_backend = "deepep_high_throughput"
    target.expert_placement_strategy = "round_robin"
    requested = ParallelConfig(
        tensor_parallel_size=2,
        prefill_context_parallel_size=2,
    )

    actual = SpeculativeConfig._materialize_draft_parallel_config(
        target, requested, draft_is_moe=True
    )

    assert actual.enable_expert_parallel
    assert actual.enable_eplb
    assert actual.enable_elastic_ep
    assert actual.enable_ep_weight_filter
    assert actual.eplb_config is target.eplb_config
    assert actual.all2all_backend == "deepep_high_throughput"
    assert actual.expert_placement_strategy == "round_robin"
    assert actual.is_moe_model is True


@pytest.mark.parametrize(
    ("target_kwargs", "draft_kwargs", "error"),
    [
        ({"tensor_parallel_size": 2}, {}, "TP and DCP"),
        (
            {"tensor_parallel_size": 2},
            {"tensor_parallel_size": 2, "decode_context_parallel_size": 2},
            "TP and DCP",
        ),
        (
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2},
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 3},
            "Draft PCP must be 1 or match",
        ),
        (
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2},
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2},
            "does not support PCP-sharded drafting",
        ),
        (
            {"tensor_parallel_size": 2, "prefill_context_parallel_size": 2},
            {"tensor_parallel_size": 2},
            "does not support replicated PCP drafting",
        ),
    ],
)
def test_draft_parallel_topology_validation(target_kwargs, draft_kwargs, error):
    config = SimpleNamespace(
        parallel_config=ParallelConfig(**target_kwargs),
        speculative_config=SimpleNamespace(
            draft_parallel_config=ParallelConfig(**draft_kwargs)
        ),
    )

    with pytest.raises((ValueError, NotImplementedError), match=error):
        _TestSpeculator(config, torch.device("cpu"))


def test_pcp_capabilities_are_opt_in():
    assert MTPSpeculator.supports_replicated_pcp
    assert DSparkSpeculator.supports_replicated_pcp
    assert not DraftModelSpeculator.supports_replicated_pcp
    assert not MultiModuleMTPSpeculator.supports_replicated_pcp
    assert MTPSpeculator.supports_sharded_pcp
    assert not DSparkSpeculator.supports_sharded_pcp
    assert not DraftModelSpeculator.supports_sharded_pcp
    assert not MultiModuleMTPSpeculator.supports_sharded_pcp


def test_replicated_parallel_config_controls_fused_moe_pcp(monkeypatch):
    monkeypatch.setattr(
        fused_moe_layer,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=2, rank_in_group=1),
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.config.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    draft_parallel_config = ParallelConfig(
        tensor_parallel_size=2,
        prefill_context_parallel_size=1,
    )

    actual = fused_moe_layer.make_parallel_config(
        tp_size=2,
        dp_size=1,
        pcp_size=None,
        is_sequence_parallel=False,
        parallel_config=draft_parallel_config,
    )

    assert actual.pcp_size == 1
    assert actual.pcp_rank == 0
    assert actual.tp_size == 2


@pytest.mark.parametrize(("hc_mult", "expected"), [(None, 64), (4, 256)])
def test_speculator_uses_draft_model_hidden_size(monkeypatch, hc_mult, expected):
    # Qwen4Exp targets expose multi-stream HC residuals to the drafter.
    monkeypatch.setattr(base_spec_module, "_target_feeds_hc_residual", lambda _: True)
    monkeypatch.setattr(
        base_spec_module,
        "replace",
        lambda config, **kwargs: SimpleNamespace(**{**vars(config), **kwargs}),
    )
    hf_config = SimpleNamespace()
    if hc_mult is not None:
        hf_config.hc_mult = hc_mult
    draft_model_config = SimpleNamespace(
        hf_config=hf_config,
        is_moe=False,
        get_hidden_size=lambda: 64,
        get_vocab_size=lambda: 32,
    )
    parallel_config = ParallelConfig()
    speculative_config = SimpleNamespace(
        method="mtp",
        num_speculative_tokens=3,
        draft_model_config=draft_model_config,
        draft_parallel_config=parallel_config,
        use_local_argmax_reduction=False,
        draft_sample_method="greedy",
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(
            max_num_seqs=2,
            max_num_batched_tokens=8,
        ),
        model_config=SimpleNamespace(
            max_model_len=32,
            dtype=torch.float32,
            use_fp64_gumbel=False,
        ),
        parallel_config=parallel_config,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert speculator.hidden_size == expected


def test_mm_support_configured_after_model_load(monkeypatch):
    target_model_config = object()
    draft_model_config = object()
    vllm_config = SimpleNamespace(model_config=target_model_config)
    draft_model = _MultimodalDraftModel()

    def init_base(speculator, vllm_config, device):
        speculator.vllm_config = vllm_config
        speculator.device = device
        speculator.max_num_tokens = 4
        speculator.max_num_reqs = 2
        speculator.hidden_size = 3
        speculator.dtype = torch.float32
        speculator.draft_model_config = draft_model_config
        speculator.supports_mm_inputs = False

    checked_configs = []

    def supports_multimodal_inputs(model_config):
        checked_configs.append(model_config)
        return True

    monkeypatch.setattr(DraftModelSpeculator, "__init__", init_base)
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        supports_multimodal_inputs,
    )

    speculator = _TestSpeculator(vllm_config, torch.device("cpu"))

    assert checked_configs == []
    assert not speculator.supports_mm_inputs
    assert speculator.inputs_embeds is None

    speculator.test_draft_model = draft_model
    speculator.load_model(torch.nn.Module())

    assert checked_configs == [target_model_config]
    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)


def test_load_model_keeps_mm_support_for_capable_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    speculator.test_draft_model = draft_model
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None


def test_load_model_disables_mm_support_for_text_only_drafter(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    draft_model = _TextOnlyDraftModel()
    speculator.test_draft_model = draft_model
    warning_messages = []
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )
    monkeypatch.setattr(
        base_spec_module.logger,
        "warning_once",
        lambda message, *args: warning_messages.append(message % args),
    )

    speculator.load_model(torch.nn.Module())

    assert not speculator.supports_mm_inputs
    assert warning_messages == [
        (
            "Draft model _TextOnlyDraftModel does not support external multimodal "
            "embeddings. Embeddings from the target model will not be passed to the "
            "drafter; using text-only draft inputs instead."
        )
    ]


def test_multi_module_mm_support_configured_after_model_load(monkeypatch):
    speculator = object.__new__(MultiModuleMTPSpeculator)
    speculator.supports_mm_inputs = False
    speculator.inputs_embeds = None
    speculator.cached_draft_input_embeds = None
    speculator.vllm_config = SimpleNamespace(model_config=object())
    speculator.max_num_tokens = 4
    speculator.max_num_reqs = 2
    speculator.num_speculative_steps = 3
    speculator.hidden_size = 3
    speculator.dtype = torch.float32
    speculator.device = torch.device("cpu")
    draft_model = _MultimodalDraftModel()
    _mock_base_model_load(monkeypatch)
    monkeypatch.setattr(
        MultiModuleMTPSpeculator,
        "load_draft_model",
        lambda self, target_model, target_attn_layer_names: draft_model,
    )
    monkeypatch.setattr(
        base_spec_module.MULTIMODAL_REGISTRY,
        "supports_multimodal_inputs",
        lambda model_config: True,
    )

    speculator.load_model(torch.nn.Module())

    assert speculator.supports_mm_inputs
    assert speculator.inputs_embeds is not None
    assert speculator.inputs_embeds.shape == (4, 3)
    assert speculator.cached_draft_input_embeds is not None
    assert speculator.cached_draft_input_embeds.shape == (2, 2, 3)


@pytest.mark.parametrize(
    ("model_cls", "expected"),
    [
        (EagleLlama4ForCausalLM, True),
        (EagleMistralForCausalLM, True),
        (EagleMistralLarge3ForCausalLM, True),
        (Exaone4_5_MTP, True),
        (Eagle3LlamaForCausalLM, False),
    ],
)
def test_draft_model_multimodal_embedding_capability(model_cls, expected):
    assert supports_multimodal_embeddings(model_cls) is expected


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_set_pcp_manager_keeps_speculator_owned_buffers():
    speculator = object.__new__(_TestSpeculator)
    speculator.input_buffers = input_buffers = object()
    manager = Mock()

    speculator.set_pcp_manager(manager)

    assert speculator.pcp_manager is manager
    assert speculator.input_buffers is input_buffers


def test_sharded_prefill_restores_outputs_before_global_sampling():
    speculator = object.__new__(_TestSpeculator)
    local_logits = torch.tensor([[1.0], [2.0]])
    local_feedback = torch.tensor([[3.0], [4.0]])
    global_logits = torch.tensor([[1.0], [5.0], [2.0], [6.0]])
    global_feedback = torch.tensor([[3.0], [7.0], [4.0], [8.0]])
    manager = Mock()
    manager.restore_hidden_states.side_effect = [global_logits, global_feedback]
    local_batch = SimpleNamespace(
        input_ids=torch.tensor([10, 20]),
        positions=torch.tensor([0, 2]),
        is_padding=torch.tensor([False, True]),
    )
    speculator.pcp_manager = manager
    speculator._run_model = Mock(return_value=(local_logits, local_feedback))
    speculator.last_token_indices = torch.tensor([1, 3])
    speculator.idx_mapping = torch.tensor([0, 1])
    speculator.temperature = torch.zeros(2)
    speculator.seeds = torch.zeros(2, dtype=torch.int64)
    speculator.current_draft_step = torch.tensor(0)
    speculator.draft_logits = None
    speculator.draft_tokens = torch.zeros((2, 1), dtype=torch.int64)
    speculator.hidden_states = torch.zeros((4, 1))
    speculator.sample_src_positions = torch.zeros(2, dtype=torch.int64)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(4, dtype=torch.int64)
    )
    speculator.sample_draft = Mock(return_value=torch.tensor([11, 22]))

    speculator._prefill(
        num_reqs=2,
        num_tokens=2,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        pcp_local_batch=local_batch,
    )

    assert torch.equal(speculator.draft_tokens[:, 0], torch.tensor([11, 22]))
    assert torch.equal(speculator.hidden_states[:2], torch.tensor([[7.0], [8.0]]))
    assert torch.equal(
        speculator.sample_draft.call_args.args[0], torch.tensor([[5.0], [6.0]])
    )
    assert torch.equal(speculator.sample_src_positions, torch.tensor([2, 4]))
    run_kwargs = speculator._run_model.call_args.kwargs
    assert run_kwargs["input_ids"] is local_batch.input_ids
    assert run_kwargs["positions"] is local_batch.positions
    assert torch.equal(run_kwargs["is_padding"], local_batch.is_padding)


def test_sharded_continuation_metadata_is_decode_only(monkeypatch):
    speculator = object.__new__(_TestSpeculator)
    speculator.sharded_pcp = True
    speculator.num_speculative_steps = 2
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    speculator.block_tables = SimpleNamespace(
        compute_slot_mappings=Mock(return_value=torch.arange(2))
    )
    speculator.kv_cache_config = object()
    speculator._build_draft_attn_metadata = Mock(return_value={})
    speculator._generate_draft = Mock()
    monkeypatch.setattr(
        spec_module, "build_slot_mappings_by_layer", Mock(return_value={})
    )

    speculator._multi_step_decode(
        num_reqs=2,
        skip_attn=False,
        batch_desc=BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=2,
            num_reqs=2,
        ),
        seq_lens_cpu_upper_bound=torch.ones(2, dtype=torch.int32),
        num_tokens_across_dp=None,
    )

    is_prefilling = speculator._build_draft_attn_metadata.call_args.kwargs[
        "is_prefilling"
    ]
    assert torch.equal(is_prefilling, torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize(
    (
        "method_name",
        "cg_mode",
        "expected_eager_calls",
        "expected_graph_replays",
    ),
    [
        ("_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_multi_step_decode", CUDAGraphMode.FULL, 0, 3),
        ("_fused_multi_step_decode", CUDAGraphMode.NONE, 3, 0),
        ("_fused_multi_step_decode", CUDAGraphMode.FULL, 0, 1),
    ],
)
def test_multi_step_decode_replays_captured_graph_as_expected(
    method_name,
    cg_mode,
    expected_eager_calls,
    expected_graph_replays,
):
    speculator = object.__new__(_TestSpeculator)
    speculator.num_speculative_steps = 4
    speculator.current_draft_step = torch.tensor(0)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(2),
        query_start_loc=torch.arange(3),
    )
    speculator.idx_mapping = torch.arange(2)
    generate_draft = Mock()
    speculator._generate_draft = generate_draft
    run_fullgraph = Mock()
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=run_fullgraph)
    batch_desc = BatchExecutionDescriptor(
        cg_mode=cg_mode,
        num_tokens=2,
        num_reqs=2,
    )

    getattr(speculator, method_name)(
        num_reqs=2,
        skip_attn=True,
        batch_desc=batch_desc,
        seq_lens_cpu_upper_bound=None,
        num_tokens_across_dp=None,
    )

    assert generate_draft.call_count == expected_eager_calls
    assert run_fullgraph.call_count == expected_graph_replays


def test_update_draft_decode_metadata_updates_fa3_scheduler_metadata(
    monkeypatch,
):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(8, dtype=torch.int32)
    builder.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    builder.kv_cache_dtype = torch.bfloat16
    builder.num_heads_q = 2
    builder.num_heads_kv = 1
    builder.headdim = 128
    builder.block_size = 16
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    builder.aot_sliding_window = None

    expected = torch.tensor([7, 8, 9], dtype=torch.int32)

    def fake_get_scheduler_metadata(**kwargs):
        return expected

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_seq_len=8,
        seq_lens=torch.tensor([5, 6], dtype=torch.int32),
        block_table=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(3, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=2,
        num_prefill_reqs=0,
        num_decode_tokens=3,
        num_prefill_tokens=0,
        scheduler_metadata=torch.tensor([-1, -1, -1], dtype=torch.int32),
        prefix_scheduler_metadata=None,
        max_num_splits=4,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert torch.equal(metadata.scheduler_metadata, expected)
    assert torch.equal(builder.scheduler_metadata[:3], expected)


def test_update_draft_decode_metadata_skips_without_scheduler_metadata(monkeypatch):
    builder = object.__new__(flash_attn_module.FlashAttentionMetadataBuilder)
    builder.aot_schedule = True
    builder.use_full_cuda_graph = True
    builder.scheduler_metadata = torch.zeros(4, dtype=torch.int32)

    called = False

    def fake_get_scheduler_metadata(**kwargs):
        nonlocal called
        called = True
        return torch.tensor([1], dtype=torch.int32)

    monkeypatch.setattr(builder, "_get_scheduler_metadata", fake_get_scheduler_metadata)

    metadata = FlashAttentionMetadata(
        num_actual_tokens=1,
        max_query_len=1,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        max_seq_len=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(1, dtype=torch.int32),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        max_dcp_context_kv_len=None,
        dcp_context_kv_lens=None,
        num_decode_reqs=1,
        num_prefill_reqs=0,
        num_decode_tokens=1,
        num_prefill_tokens=0,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
        causal=True,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        rswa_window_tensor=None,
    )

    builder.update_draft_decode_metadata(metadata)

    assert not called
    assert metadata.scheduler_metadata is None
