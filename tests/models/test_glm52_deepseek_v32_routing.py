# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.registry import (
    _SPECULATIVE_DECODING_MODELS,
    _TEXT_GENERATION_MODELS,
)
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer


class _HfConfigStub:
    def __init__(self, model_type: str) -> None:
        self.architectures = ["SomeForCausalLM"]
        self.model_type = model_type
        self.num_nextn_predict_layers = 1

    def update(self, values: dict) -> None:
        self.__dict__.update(values)


def _mtp_arch(model_type: str) -> list[str]:
    hf_config = _HfConfigStub(model_type)
    SpeculativeConfig.hf_config_override(hf_config)
    return hf_config.architectures


def test_glm52_routes_to_deepseek_v32() -> None:
    assert _TEXT_GENERATION_MODELS["GlmMoeDsaForCausalLM"] == (
        "vllm.models.deepseek_v32",
        "GlmMoeDsaForCausalLM",
    )
    assert _SPECULATIVE_DECODING_MODELS["DeepseekV32MTPModel"] == (
        "vllm.models.deepseek_v32",
        "DeepseekV32MTP",
    )


def test_glm52_uses_dsv32_mtp() -> None:
    assert _mtp_arch("glm_moe_dsa") == ["DeepseekV32MTPModel"]


@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v32"])
def test_other_deepseek_models_keep_generic_mtp(model_type: str) -> None:
    assert _mtp_arch(model_type) == ["DeepSeekMTPModel"]


@pytest.mark.parametrize(
    "architectures,expected",
    [
        (["DeepSeekMTPModel"], True),
        (["DeepseekV32MTPModel"], True),
        (["Glm4MoeMTPModel"], False),
        ([], False),
    ],
)
def test_tuple_return_contract(architectures: list[str], expected: bool) -> None:
    proposer = SimpleNamespace(
        method="mtp",
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures)
        ),
    )

    assert SpecDecodeBaseProposer.model_returns_tuple(proposer) is expected
