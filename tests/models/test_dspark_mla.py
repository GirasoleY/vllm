# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from tests.utils import multi_gpu_test
from vllm import SamplingParams
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel
from vllm.platforms import current_platform


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    mapper = K3DSparkForCausalLM.hf_to_vllm_mapper
    for name in ("confidence_head.weight", "embed_tokens.weight", "lm_head.weight"):
        assert mapper._map_name(name) is None


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.fixture
def mla_dspark_models(tmp_path):
    """Small models exercise the real loader without checkpoint downloads."""
    common = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "q_lora_rank": 64,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 128,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 256,
        "torch_dtype": "bfloat16",
        "hidden_act": "silu",
        "n_routed_experts": 0,
        "n_shared_experts": 0,
        "num_experts_per_tok": 0,
        "first_k_dense_replace": 2,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    target = {
        **common,
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
    }
    draft = {
        **common,
        "architectures": ["K3DSparkModel"],
        "model_type": "k3_dspark",
        "num_target_layers": 2,
        "target_hidden_size": 128,
        "target_num_hidden_layers": 2,
        "target_layer_ids": [0, 1],
        "mask_token_id": 127,
        "markov_rank": 8,
        "draft_vocab_size": 128,
    }
    for name, config in (("target", target), ("draft", draft)):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
    return str(tmp_path / "target"), str(tmp_path / "draft")


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="FlashInfer MLA requires SM100-family NVIDIA GPUs",
)
@pytest.mark.parametrize(
    ("tp_size", "dcp_size", "draft_tp_size", "enforce_eager"),
    [
        (1, 1, None, True),
        (2, 1, None, False),
        (2, 2, None, False),
        (2, 1, 2, False),
        (2, 2, 2, False),
    ],
    ids=["eager", "tp", "tp-dcp", "tp-explicit-draft-tp", "tp-dcp-explicit-draft-tp"],
)
@multi_gpu_test(num_gpus=2)
def test_dspark_mla_parallel_generation(
    monkeypatch,
    mla_dspark_models,
    vllm_runner,
    tp_size,
    dcp_size,
    draft_tp_size,
    enforce_eager,
):
    """Load, profile, and decode with DSpark without changing target outputs."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    target, draft = mla_dspark_models
    runner_args = {
        "load_format": "dummy",
        "skip_tokenizer_init": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": tp_size,
        "decode_context_parallel_size": dcp_size,
        "attention_config": {"backend": "FLASHINFER_MLA"},
        "max_model_len": 256,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 128,
        "block_size": 32,
        "num_gpu_blocks_override": 256,
        "gpu_memory_utilization": 0.2,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "enforce_eager": enforce_eager,
        "compilation_config": {"cudagraph_capture_sizes": [4, 8, 16]},
        "disable_log_stats": False,
    }
    # Unequal lengths cross cache-block boundaries and require padded decode batches.
    prompts = [
        {"prompt_token_ids": [i % 120 + 3 for i in range(length)]}
        for length in (7, 33, 65)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
    with vllm_runner(target, **runner_args) as runner:
        reference = runner.llm.generate(prompts, sampling)

    with vllm_runner(
        target,
        **runner_args,
        speculative_config={
            "method": "dspark",
            "model": draft,
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": draft_tp_size,
            "draft_sample_method": "probabilistic",
            "attention_backend": "FLASHINFER_MLA",
        },
    ) as runner:
        outputs = runner.llm.generate(prompts, sampling)
        metrics = runner.llm.get_metrics()
        assert any(
            m.name == "vllm:spec_decode_num_drafts" and m.value > 0 for m in metrics
        )

    assert len(outputs) == len(reference) == len(prompts)
    for output, expected in zip(outputs, reference):
        assert output.finished
        assert len(output.outputs[0].token_ids) == sampling.max_tokens
        assert output.outputs[0].token_ids == expected.outputs[0].token_ids


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()
