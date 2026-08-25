# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytest.importorskip("deep_ep")

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    mxfp4_mxfp8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
    _pack_mxfp8_scale,
    _quantize_before_dispatch,
    _unpack_mxfp8_scale,
)


def _mxfp4_mxfp8_quant_config() -> FusedMoEQuantConfig:
    scale = torch.empty(0, dtype=torch.uint8)
    return mxfp4_mxfp8_moe_quant_config(
        w1_scale=scale,
        w2_scale=scale,
        mx_alignment=256,
        is_scale_swizzled=False,
    )


def test_mxfp8_is_quantized_before_deepep_v2_dispatch() -> None:
    quant_config = _mxfp4_mxfp8_quant_config()

    assert _quantize_before_dispatch(quant_config, defer_input_quant=False)
    assert not _quantize_before_dispatch(quant_config, defer_input_quant=True)


def test_mxfp8_scale_pack_round_trip_preserves_rows_and_bits() -> None:
    scale = torch.arange(3 * 112, dtype=torch.uint8).reshape(3, 112)

    packed = _pack_mxfp8_scale(scale)
    unpacked = _unpack_mxfp8_scale(
        packed,
        hidden_size=3584,
        is_scale_swizzled=False,
    )

    assert packed.dtype == torch.int32
    assert packed.shape == (3, 28)
    torch.testing.assert_close(unpacked, scale)


def test_mxfp8_scale_pack_rejects_partial_deepep_pack() -> None:
    scale = torch.empty((2, 3), dtype=torch.uint8)

    with pytest.raises(AssertionError, match="hidden_size % 128"):
        _pack_mxfp8_scale(scale)
