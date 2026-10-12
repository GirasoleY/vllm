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
            _pending_wk_fp8={"wk": {}} if case == "pending_wk" else {},
            modules=lambda: iter(()),
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


@pytest.mark.parametrize("mega", [False, True])
@pytest.mark.parametrize("suffix", ["weight", "weight_scale"])
@pytest.mark.parametrize("fused", [False, True])
def test_expert_loader_preserves_packed_bytes_and_gate_up_shards(
    monkeypatch, mega, suffix, fused
):
    """Both expert backends must receive every gate/up/down shard unchanged."""
    from vllm.models.hy_v4.nvidia import model

    routed = "" if mega else "routed_experts."
    prefix = "layers.1.mlp.experts."
    captured = []

    def load(param, weight, name, shard_id, expert_id, return_success):
        assert return_success
        captured.append((name, shard_id, expert_id, weight.clone()))
        return True

    names = [f"{prefix}{routed}{projection}_{suffix}" for projection in ("w13", "w2")]
    params = [(name, SimpleNamespace(weight_loader=load)) for name in names]
    backbone = _backbone(monkeypatch, model, params)
    backbone.config.num_experts = 2
    backbone.num_redundant_experts = 0
    moe = model.HYV4MoEFused.__new__(model.HYV4MoEFused)
    torch.nn.Module.__init__(moe)
    moe.use_mega_moe = mega
    backbone.modules = lambda: iter((moe,))
    backbone.get_expert_mapping = lambda: model.HYV4Model.get_expert_mapping(backbone)
    backbone.load_fused_expert_weights = lambda *args: (
        model.HYV4Model.load_fused_expert_weights(backbone, *args)
    )
    # Distinct bytes expose accidentally swapped gate/up or expert indices.
    gate = torch.arange(12, dtype=torch.uint8).reshape(2, 3, 2)
    up, down = gate + 20, gate + 40
    if fused:
        scale_suffix = "_scale" if suffix == "weight_scale" else ""
        weights = [
            (prefix + "gate_up_proj" + scale_suffix, torch.cat((gate, up), dim=-2)),
            (prefix + "down_proj" + scale_suffix, down),
        ]
    else:
        weights = [
            (f"{prefix}{expert}.{projection}.{suffix}", tensor[expert])
            for expert in range(2)
            for projection, tensor in (
                ("gate_proj", gate),
                ("up_proj", up),
                ("down_proj", down),
            )
        ]
    assert model.HYV4Model.load_weights(backbone, weights) == set(names)
    assert len(captured) == 6
    for name, shard, expert, weight in captured:
        assert name == names[shard == "w2"]
        expected = {"w1": gate, "w3": up, "w2": down}[shard][expert]
        assert weight.dtype == torch.uint8
        torch.testing.assert_close(weight, expected)


@pytest.mark.parametrize("factory_fails", [False, True])
def test_mtp_backend_override_does_not_mutate_target_config(monkeypatch, factory_fails):
    """Unsupported drafts fall back without changing the target backend."""
    from vllm.config import get_current_vllm_config, set_current_vllm_config
    from vllm.models.hy_v4.nvidia import moe

    config = SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        expert_hidden_dim=128,
        hidden_size=128,
        num_shared_experts=0,
        route_norm=True,
        swiglu_limit=10,
    )
    target = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=SimpleNamespace(
            eplb_config=SimpleNamespace(num_redundant_experts=0)
        ),
    )
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        moe,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 4), rank_in_group=0
        ),
    )
    monkeypatch.setattr(moe, "GateLinear", lambda *args, **kwargs: torch.nn.Identity())
    quant = SimpleNamespace(store_dtype=None)

    def create_experts(**kwargs):
        draft = get_current_vllm_config()
        assert draft is not target
        assert draft.kernel_config.moe_backend == "auto"
        assert target.kernel_config.moe_backend == "deep_gemm_mega_moe"
        assert kwargs["quant_config"] is quant
        if factory_fails:
            raise RuntimeError("draft construction failed")
        return torch.nn.Identity()

    monkeypatch.setattr(moe, "FusedMoEFactory", create_experts)
    with set_current_vllm_config(target):
        if factory_fails:
            with pytest.raises(RuntimeError, match="draft construction failed"):
                moe.HYV4MoEFused(config, quant, is_mtp=True)
        else:
            draft = moe.HYV4MoEFused(config, quant, is_mtp=True)
            assert not draft.use_mega_moe
        assert get_current_vllm_config() is target
        assert target.kernel_config.moe_backend == "deep_gemm_mega_moe"


@pytest.mark.parametrize("fused", [False, True])
def test_mtp_megamoe_loads_fp8_weights_and_block_scales(fused):
    """Split and fused checkpoints must preserve FP8 values and UE8M0 scales."""
    from vllm.model_executor.layers.fused_moe import (
        fused_moe_make_expert_params_mapping,
    )
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts
    from vllm.models.hy_v4.nvidia.mtp import HYV4MTP, _prepare_mtp_fp8_expert_scale

    config = SimpleNamespace(
        max_num_tokens_per_rank=32,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    adapter = DeepseekV4MegaMoEExperts(
        config,
        num_experts=4,
        num_local_experts=2,
        experts_start_idx=2,
        top_k=2,
        hidden_size=256,
        intermediate_size=256,
        fp8_weight_block_size=(128, 128),
    )
    draft = HYV4MTP.__new__(HYV4MTP)
    torch.nn.Module.__init__(draft)
    quant = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
    quant.is_scale_e8m0 = True
    base = "model.layers.78.mtp_block.mlp"
    params = {
        f"{base}.experts.{name}": param for name, param in adapter.named_parameters()
    }
    mapping = fused_moe_make_expert_params_mapping(
        draft, "gate_proj", "down_proj", "up_proj", 4, routed_experts_prefix=""
    )
    fused_names = {
        (base, name): f"{base}.experts.{name}" for name in ("w13_weight", "w2_weight")
    }
    torch.manual_seed(7)
    gate, up, down = [
        torch.randn(4, 256, 256).to(torch.float8_e4m3fn) for _ in range(3)
    ]
    scales = [torch.randint(116, 128, (4, 2, 2), dtype=torch.uint8) for _ in range(3)]
    weights = []
    if fused:
        for suffix, values in (("", (gate, up, down)), ("_scale_inv", scales)):
            weights.extend(
                [
                    (
                        f"{base}.experts.gate_up_proj{suffix}",
                        torch.cat(values[:2], dim=1),
                    ),
                    (f"{base}.experts.down_proj{suffix}", values[2]),
                ]
            )
    else:
        for expert in range(4):
            for projection, weight, scale in zip(
                ("gate_proj", "up_proj", "down_proj"), (gate, up, down), scales
            ):
                weights.extend(
                    [
                        (
                            f"{base}.experts.{expert}.{projection}.weight",
                            weight[expert],
                        ),
                        (f"{base}.experts.{expert}.{projection}.scale", scale[expert]),
                    ]
                )
    loaded: set[str] = set()
    for name, weight in weights:
        name, weight = _prepare_mtp_fp8_expert_scale(quant, name, weight)
        assert draft._load_expert_weight(
            name, weight, params, loaded, mapping, fused_names, 4
        )
    assert loaded == set(params)
    torch.testing.assert_close(
        adapter.w13_weight.float(), torch.cat((gate[2:].float(), up[2:].float()), dim=1)
    )
    torch.testing.assert_close(adapter.w2_weight.float(), down[2:].float())
    expected_scale = (
        torch.cat((scales[0][2:], scales[1][2:]), dim=1)
        .view(torch.float8_e8m0fnu)
        .float()
    )
    torch.testing.assert_close(adapter.w13_weight_scale, expected_scale)
    expanded = adapter._prepare_routed_expert_scale(adapter.w13_weight_scale)
    torch.testing.assert_close(
        expanded, expected_scale.repeat_interleave(128, 1).repeat_interleave(4, 2)
    )


@pytest.mark.parametrize("scale", [0.3, 0.0, float("inf"), float("nan")])
def test_fp8_megamoe_rejects_scales_that_would_change_draft_weights(scale):
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4MegaMoEExperts

    adapter = SimpleNamespace(fp8_weight_block_size=(128, 128))
    with pytest.raises(ValueError, match="UE8M0"):
        DeepseekV4MegaMoEExperts._prepare_routed_expert_scale(
            adapter, torch.full((1, 2, 2), scale)
        )


@pytest.mark.parametrize("algorithm", ["MXFP4", "NONE", None, "FP8"])
def test_mtp_megamoe_selects_checkpoint_packing(monkeypatch, algorithm):
    """Packed MTP experts must not take the block-FP8 allocation path."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.hy_v4.nvidia import moe
    from vllm.models.hy_v4.nvidia.mtp import _create_mtp_quant_config

    backbone = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        store_dtype="mxfp4",
    )
    config = SimpleNamespace(
        mtp_quant_algo=algorithm,
        quantization_config={"weight_block_size": [128, 128]},
        num_experts=4,
        num_experts_per_tok=2,
        expert_hidden_dim=128,
        hidden_size=128,
        num_shared_experts=0,
        route_norm=True,
        swiglu_limit=10,
    )
    quant = _create_mtp_quant_config(config, backbone)
    runtime = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe"),
        parallel_config=SimpleNamespace(
            enable_expert_parallel=True,
            enable_eplb=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        max_num_tokens_per_rank=32,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(moe, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        moe,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 1), rank_in_group=0
        ),
    )
    monkeypatch.setattr(moe, "GateLinear", lambda *a, **kw: torch.nn.Identity())
    with set_current_vllm_config(runtime):
        draft = moe.HYV4MoEFused(config, quant, is_mtp=True)
    assert draft.use_mega_moe
    packed = algorithm != "FP8"
    expected = torch.full(
        (128, 64 if packed else 128),
        1,
        dtype=torch.uint8 if packed else torch.float8_e4m3fn,
    )
    draft.experts.weight_loader(
        draft.experts.w2_weight, expected, "experts.w2_weight", "w2", 0
    )
    torch.testing.assert_close(draft.experts.w2_weight[0], expected)
    assert draft.experts.mma_type == ("fp8xfp4" if packed else "fp8xfp8")
    assert backbone.store_dtype == "mxfp4"


def test_mtp_mxfp4_rejects_incompatible_backbone():
    from vllm.models.hy_v4.nvidia.mtp import _create_mtp_quant_config

    with pytest.raises(ValueError, match="MXFP4 backbone"):
        _create_mtp_quant_config(SimpleNamespace(mtp_quant_algo="MXFP4"), None)


def test_mtp_mxfp4_preserves_packed_expert_scale_bytes():
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.models.hy_v4.nvidia.mtp import _prepare_mtp_fp8_expert_scale

    quant = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        store_dtype="mxfp4",
    )
    quant.is_scale_e8m0 = True
    raw = torch.tensor([116, 127, 128], dtype=torch.uint8)
    name, scale = _prepare_mtp_fp8_expert_scale(
        quant, "model.layers.78.mtp_block.mlp.experts.0.down_proj.scale", raw
    )
    assert name.endswith(".weight_scale_inv")
    assert scale.dtype == torch.uint8
    torch.testing.assert_close(scale, raw)
