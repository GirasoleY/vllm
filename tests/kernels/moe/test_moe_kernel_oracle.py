# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoEKernelOracle ABC introduced in PR series for #37753.

This file contains a single canonical demonstration that
`UnquantizedMoEKernelOracle` methods delegate one-to-one to the
existing module-level functions in `oracle/unquantized.py`. Each method
on `UnquantizedMoEKernelOracle` follows the same `return module_fn(args)`
pattern, so verifying delegation for one method (`make_kernel`) gives
high confidence in the rest.
"""

from unittest.mock import patch

import pytest

from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle import UnquantizedMoEKernelOracle
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)


@pytest.mark.parametrize("scheme", ["block", "mxfp8", "tensor"])
def test_trtllm_fp8_fused_router_rejects_simulated_routing(scheme):
    """Do not send vLLM enum102 to FlashInfer's unsupported fused router."""
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
    from vllm.model_executor.layers.fused_moe.experts.trtllm_fp8_moe import (
        TrtLlmFp8ExpertsMonolithic,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Dynamic128Sym,
        kFp8Static128BlockSym,
        kFp8StaticTensorSym,
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    quant_keys = {
        "block": (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        "mxfp8": (kMxfp8Static, kMxfp8Dynamic),
        "tensor": (kFp8StaticTensorSym, kFp8StaticTensorSym),
    }[scheme]
    supports = TrtLlmFp8ExpertsMonolithic._supports_routing_method
    assert not supports(RoutingMethodType.Simulated, *quant_keys)
    assert supports(RoutingMethodType.DeepSeekV3, *quant_keys)


class TestUnquantizedDelegation:
    """UnquantizedMoEKernelOracle methods must delegate to the existing
    module-level functions; behaviour is bit-identical."""

    def test_make_kernel_delegates(self) -> None:
        quant_config = object()
        moe_config = object()
        experts_cls = TritonExperts
        sentinel_kernel = object()

        with patch(
            "vllm.model_executor.layers.fused_moe.oracle.unquantized."
            "make_unquantized_moe_kernel",
            return_value=sentinel_kernel,
        ) as mocked:
            out = UnquantizedMoEKernelOracle().make_kernel(
                quant_config,
                moe_config,
                UnquantizedMoeBackend.TRITON,
                experts_cls,
            )

        mocked.assert_called_once_with(
            quant_config,
            moe_config,
            UnquantizedMoeBackend.TRITON,
            experts_cls,
            None,  # routing_tables default
        )
        assert out is sentinel_kernel
