# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU loader regressions; no model allocation or GPU execution is required."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "projection",
    [
        "mlp.experts.0.down_proj",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.q_b_proj",
        "self_attn.kv_b_proj",
        "self_attn.linear_gate",
        "self_attn.o_proj",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
    ],
)
def test_mtp_dense_and_expert_scales_decode_ue8m0(projection):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.hy_v4.nvidia.mtp import _prepare_mtp_fp8_expert_scale

    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    config.is_scale_e8m0 = True
    name = f"model.layers.78.mtp_block.{projection}.scale"
    raw = torch.tensor([116, 117, 127, 128], dtype=torch.uint8)
    mapped, scale = _prepare_mtp_fp8_expert_scale(config, name, raw)
    assert mapped == name.removesuffix(".scale") + ".weight_scale_inv"
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.data_ptr() == raw.data_ptr()
    torch.testing.assert_close(scale.float(), torch.tensor([2**-11, 2**-10, 1.0, 2.0]))


@pytest.mark.parametrize(
    "name",
    ["self_attn.q_b_proj.weight", "self_attn.k_scale", "unknown.scale"],
)
def test_mtp_non_block_scales_are_unchanged(name):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.hy_v4.nvidia.mtp import _prepare_mtp_fp8_expert_scale

    config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    raw = torch.tensor([127], dtype=torch.uint8)
    assert _prepare_mtp_fp8_expert_scale(config, name, raw) == (name, raw)
    assert _prepare_mtp_fp8_expert_scale(None, name, raw) == (name, raw)


def test_backbone_scale_preserves_fused_shard_id(monkeypatch):
    from vllm.models.hy_v4.nvidia import model

    name = "layers.0.self_attn.fused_qkv_a_proj.weight_scale_inv"
    captured = []
    param = SimpleNamespace(weight_loader=lambda *args: captured.append(args))
    backbone = _backbone(monkeypatch, model, [(name, param)])
    raw = torch.tensor([127, 128], dtype=torch.uint8)
    raw.shard_id = 1
    loaded = model.HYV4Model.load_weights(
        backbone, [(name.replace(".weight_scale_inv", ".scale"), raw)]
    )
    assert loaded == {name}
    assert captured[0][2] == 1
    torch.testing.assert_close(captured[0][1].float(), torch.tensor([1.0, 2.0]))


def test_indexer_pair_buffer_survives_loader_chunks(monkeypatch):
    from vllm.models.hy_v4.nvidia import model

    backbone = _backbone(monkeypatch, model, [])
    buffers = []

    def capture_pair(name, weight, pending, *_):
        buffers.append(pending)
        if name.endswith(".weight"):
            pending["wk"] = {"weight": weight}
        else:
            assert "wk" in pending
            pending.clear()
        return True

    monkeypatch.setattr(model, "_try_load_fp8_indexer_wk", capture_pair)
    prefix = "layers.0.self_attn.indexer.wk"
    model.HYV4Model.load_weights(backbone, [(prefix + ".weight", torch.ones(1))])
    model.HYV4Model.load_weights(
        backbone, [(prefix + ".scale", torch.tensor([127], dtype=torch.uint8))]
    )
    assert buffers[0] is buffers[1]
    assert not buffers[0]


@pytest.mark.parametrize(
    "case", ["complete", "missing_scale", "pending_wk", "other_pp"]
)
def test_backbone_completeness_checked_after_full_load(monkeypatch, case):
    from vllm.models.hy_v4.nvidia import model

    name = "model.layers.0.self_attn.q_b_proj.weight_scale_inv"
    loaded = {name} if case == "complete" else set()
    monkeypatch.setattr(
        model,
        "AutoWeightsLoader",
        lambda _: SimpleNamespace(load_weights=lambda *args, **kwargs: loaded),
    )
    monkeypatch.setattr(model, "is_pp_missing_parameter", lambda *_: case == "other_pp")
    target = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1, tie_word_embeddings=False),
        model=SimpleNamespace(
            _pending_wk_fp8={"wk": {}} if case == "pending_wk" else {}
        ),
        hf_to_vllm_mapper=model.WeightsMapper(),
        named_parameters=lambda: iter([(name, None)]),
    )
    if case in ("complete", "other_pp"):
        assert model.HYV4ForCausalLM.load_weights(target, []) == loaded
    else:
        error = (
            "Unloaded backbone FP8 scales" if case == "missing_scale" else "WK pairs"
        )
        with pytest.raises(RuntimeError, match=error):
            model.HYV4ForCausalLM.load_weights(target, [])


def _backbone(monkeypatch, model, params):
    monkeypatch.setattr(model, "get_pp_missing_layer_names", lambda _: [])
    monkeypatch.setattr(model, "compute_skip_topk_layers", lambda _: set())
    monkeypatch.setattr(model, "is_skip_topk_indexer_weight", lambda *_: False)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(model, "is_pp_missing_parameter", lambda *_: False)
    return SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=1, tie_word_embeddings=False),
        named_parameters=lambda: iter(params),
        get_expert_mapping=lambda: [],
    )
