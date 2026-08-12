# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import torch

from vllm.models.deepseek_v32.attention import DeepseekV32Attention


def test_pcp_dispatches_to_portable_attention_path() -> None:
    attention = object.__new__(DeepseekV32Attention)
    attention.use_pcp = True
    attention._forward_pcp = MagicMock(return_value=torch.empty(2, 3))

    positions = torch.arange(2)
    hidden_states = torch.empty(2, 4)
    output = attention.forward(positions, hidden_states)

    attention._forward_pcp.assert_called_once_with(positions, hidden_states)
    assert output.shape == (2, 3)


def test_pcp_decode_dispatches_to_portable_attention_path() -> None:
    attention = object.__new__(DeepseekV32Attention)
    attention.layer_name = "layers.3.attn"
    attention._forward_pcp_portable = MagicMock(return_value=torch.empty(2, 3))
    positions = torch.arange(2)
    hidden_states = torch.empty(2, 4)
    metadata = MagicMock(num_decode_tokens=1)

    with patch("vllm.models.deepseek_v32.attention.get_forward_context") as get_context:
        get_context.return_value.attn_metadata = {attention.layer_name: metadata}
        output = attention._forward_pcp(positions, hidden_states)

    attention._forward_pcp_portable.assert_called_once_with(positions, hidden_states)
    assert output.shape == (2, 3)


def test_pcp_short_prefill_dispatches_to_portable_attention_path() -> None:
    attention = object.__new__(DeepseekV32Attention)
    attention.layer_name = "layers.3.attn"
    attention._forward_pcp_portable = MagicMock(return_value=torch.empty(64, 3))
    positions = torch.arange(64)
    hidden_states = torch.empty(64, 4)
    metadata = MagicMock(num_decode_tokens=0)

    with patch("vllm.models.deepseek_v32.attention.get_forward_context") as get_context:
        get_context.return_value.attn_metadata = {attention.layer_name: metadata}
        output = attention._forward_pcp(positions, hidden_states)

    attention._forward_pcp_portable.assert_called_once_with(positions, hidden_states)
    assert output.shape == (64, 3)


def test_tp_does_not_dispatch_to_portable_attention_path() -> None:
    attention = object.__new__(DeepseekV32Attention)
    attention.use_pcp = False
    attention._forward_pcp = MagicMock()
    attention.fused_qkv_a_proj = MagicMock(side_effect=RuntimeError("tp-fast-path"))

    positions = torch.arange(2)
    hidden_states = torch.empty(2, 4)

    try:
        attention.forward(positions, hidden_states)
    except RuntimeError as error:
        assert str(error) == "tp-fast-path"
    else:
        raise AssertionError("the TP fast path did not run")
    attention._forward_pcp.assert_not_called()
