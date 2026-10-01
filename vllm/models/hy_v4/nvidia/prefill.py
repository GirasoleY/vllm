# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY4 activation fusions for block-FP8 DeepGEMM projections."""

import torch

import vllm.envs as envs
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import (
    DeepGemmFp8BlockScaledMMKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFp8DeepGEMMDynamicBlockScaledKernel,
)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT


def packed_fp8_kernel(layer, num_tokens: int | None = None):
    """Return the layer's DeepGEMM block-FP8 kernel if it accepts packed inputs.

    The fused activation kernels in this module emit FP8 values with UE8M0
    scales packed four per ``int32``, which is the input format of
    ``DeepGemmFp8BlockScaledMMKernel``. This returns that kernel when ``layer``
    runs on it in UE8M0 mode with no bias, no output gather, and an input that is
    already sharded for a row-parallel layer. A FlashInfer-wrapped kernel is
    unwrapped to its DeepGEMM fallback unless ``num_tokens`` is below the
    threshold at which FlashInfer handles the GEMM itself.

    Args:
        layer: A vLLM linear layer, or None.
        num_tokens: Batch size of the pending call, when known.

    Returns:
        The DeepGEMM kernel, or None when the layer must quantize its own input.

    """
    if layer is None or envs.VLLM_BATCH_INVARIANT:
        return None
    method = getattr(layer, "quant_method", None)
    kernel = getattr(method, "kernel", getattr(method, "fp8_linear", None))
    if isinstance(kernel, FlashInferFp8DeepGEMMDynamicBlockScaledKernel):
        if num_tokens is not None and num_tokens < 32:
            return None
        kernel = kernel.fallback
    if (
        isinstance(kernel, DeepGemmFp8BlockScaledMMKernel)
        and kernel.use_deep_gemm_e8m0
        and DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0
        and layer.bias is None
        and not getattr(layer, "gather_output", False)
        and (not isinstance(layer, RowParallelLinear) or layer.input_is_parallel)
    ):
        return kernel
    return None


def fp8_linear(layer, x):
    """Apply a linear layer to BF16 or pre-quantized packed FP8 activations.

    Args:
        layer: A vLLM linear layer.
        x: Either a BF16 tensor, which the layer quantizes itself, or an
            ``(fp8, scales)`` tuple from ``gate_quant``, ``norm_quant`` or
            ``kernel.quant_fp8``. The tuple form requires
            ``packed_fp8_kernel(layer)`` to be non-None.

    Returns:
        The projection output, including the tensor-parallel all-reduce of a
        row-parallel layer and the trim of any padded output columns.

    """
    if not isinstance(x, tuple):
        return layer(x)[0]
    kernel = packed_fp8_kernel(layer, x[0].shape[0])
    assert kernel is not None
    params = kernel._get_layer_params(layer)
    out = kernel.apply_block_scaled_mm(x[0], params.weight, x[1], params.block_scale)
    logical_out = getattr(layer, "_pbwo_logical_out", None)
    if logical_out is not None:
        out = out[..., :logical_out]
    if (
        isinstance(layer, RowParallelLinear)
        and layer.reduce_results
        and layer.tp_size > 1
    ):
        out = tensor_model_parallel_all_reduce(out)
    return out


# Quantizes a (BM, BN) fp32 tile to FP8 with one UE8M0 scale per 128 columns and
# stores both. Scales are written as exponent bytes packed four per int32, so each
# int32 covers 512 columns; S is column-major with row stride 1 and pack stride SS.
@triton.jit
def _store_packed(
    y,
    Q,
    S,
    rows,
    cols,
    M,
    N: tl.constexpr,
    SS,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    groups = tl.reshape(y, (BM, BN // 128, 128))
    scale_raw = tl.maximum(tl.max(tl.abs(groups), 2) / 448.0, 1e-10)
    exponent = tl.ceil(tl.log2(scale_raw))
    scale = tl.exp2(exponent)
    quant = tl.reshape(groups / scale[:, :, None], (BM, BN))
    tl.store(
        Q + rows[:, None] * N + cols[None, :],
        quant,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
    scale_bytes = tl.reshape((exponent + 127).to(tl.int32), (BM, BN // 512, 4))
    packed = tl.sum(scale_bytes << (tl.arange(0, 4)[None, None, :] * 8), 2)
    packs = tl.min(cols, 0) // 512 + tl.arange(0, BN // 512)
    tl.store(
        S + rows[:, None] + packs[None, :] * SS,
        packed,
        (rows[:, None] < M) & (packs[None, :] < tl.cdiv(N, 512)),
    )


# y = x * sigmoid(gate), quantized to packed FP8. Each program covers BM rows and
# 512 columns. G has one gate per HEAD_DIM columns of X (HEAD_DIM=1 for a
# per-column gate).
@triton.jit
def _gate_quant(
    X,
    G,
    Q,
    S,
    M,
    N: tl.constexpr,
    XS: tl.constexpr,
    GS: tl.constexpr,
    SS,
    HEAD_DIM: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * 512 + tl.arange(0, 512)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    x = tl.load(X + rows[:, None] * XS + cols[None, :], mask, 0).to(tl.float32)
    g = tl.load(G + rows[:, None] * GS + cols[None, :] // HEAD_DIM, mask, 0).to(
        tl.float32
    )
    # The eager sigmoid and product each round to the activation dtype.
    gate = tl.sigmoid(g).to(X.dtype.element_ty).to(tl.float32)
    y = (x * gate).to(X.dtype.element_ty).to(tl.float32)
    _store_packed(y, Q, S, rows, cols, M, N, SS, BM, 512)


# RMSNorm over the full row of width N (padded to BN columns), quantized to
# packed FP8. Each program covers BM rows.
@triton.jit
def _norm_quant(
    X,
    W,
    Q,
    S,
    M,
    N: tl.constexpr,
    XS: tl.constexpr,
    SS,
    EPS: tl.constexpr,
    BN: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    x = tl.load(
        X + rows[:, None] * XS + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        0,
    ).to(tl.float32)
    w = tl.load(W + cols, cols < N, 0).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 1) / N + EPS)[:, None] * w[None, :]
    y = y.to(X.dtype.element_ty).to(tl.float32)
    _store_packed(y, Q, S, rows, cols, M, N, SS, BM, BN)


def _allocate(x, width):
    """Allocate the FP8 output and the packed-scale tensor for ``x.shape[0]`` rows.

    The scale tensor has shape ``(m, ceil(width / 512))`` in column-major layout
    with the row dimension padded to a multiple of 4, as DeepGEMM's TMA loads
    require.
    """
    m = x.shape[0]
    q = torch.empty((m, width), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty_strided(
        (m, triton.cdiv(width, 512)),
        (1, triton.cdiv(m, 4) * 4),
        device=x.device,
        dtype=torch.int32,
    )
    return q, s


def gate_quant(x, gate, head_dim=1):
    """Quantize ``x * sigmoid(gate)`` to packed FP8 in one pass.

    Args:
        x: ``(m, n)`` BF16 activations.
        gate: ``(m, n // head_dim)`` BF16 gate logits; each value gates
            ``head_dim`` consecutive columns of ``x``.
        head_dim: Columns of ``x`` per gate value.

    Returns:
        ``(fp8, scales)`` for ``fp8_linear``. The sigmoid and the product are
        rounded to the dtype of ``x`` before quantization.

    """
    q, s = _allocate(x, x.shape[1])
    if x.shape[0]:
        _gate_quant[(triton.cdiv(x.shape[0], 4), triton.cdiv(x.shape[1], 512))](
            x,
            gate,
            q,
            s,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            gate.stride(0),
            s.stride(1),
            head_dim,
            4,
            enable_fp_fusion=False,
        )
    return q, s


def norm_quant(x, norm):
    """Apply ``norm`` (an RMSNorm) to ``x`` and quantize to packed FP8 in one pass.

    Args:
        x: ``(m, n)`` BF16 activations.
        norm: An ``RMSNorm`` module whose ``weight`` and ``variance_epsilon`` are
            used.

    Returns:
        ``(fp8, scales)`` for ``fp8_linear``. The normalized value is rounded to
        the dtype of ``x`` before quantization.

    """
    q, s = _allocate(x, x.shape[1])
    if x.shape[0]:
        block_m = 4 if x.shape[0] >= 128 else 1
        _norm_quant[(triton.cdiv(x.shape[0], block_m),)](
            x,
            norm.weight,
            q,
            s,
            x.shape[0],
            x.shape[1],
            x.stride(0),
            s.stride(1),
            norm.variance_epsilon,
            max(512, triton.next_power_of_2(x.shape[1])),
            block_m,
            enable_fp_fusion=False,
        )
    return q, s
