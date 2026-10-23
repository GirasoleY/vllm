# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm._custom_ops import (
    cutlass_scaled_mm_supports_fp4,
)
from vllm.model_executor.layers.weight_gather import (
    WeightGatherPart,
    WeightGatherWorkspace,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """Pad and block-interleave the FP4 block-scales so that they match the data
    layout expected by the CUTLASS / FlashInfer kernels.

    Args:
        scale: The FP4 block-scale tensor to swizzle.

    Returns:
        The swizzled tensor with the same logical shape as `scale`.

    """
    assert scale.dtype == torch.float8_e4m3fn, (
        "swizzle_blockscale expects the input tensor to be in "
        "torch.float8_e4m3fn format."
    )

    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)  # (1, M, K)
    assert scale.ndim == 3, "Expected a 2-D or 3-D tensor for block scales."

    B, M, K = scale.shape

    M_padded = round_up(M, 128)
    K_padded = round_up(K, 4)

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale

    # Reshape / permute to the layout required by the kernel.
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous().cuda()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


def cutlass_fp4_supported() -> bool:
    if not current_platform.is_cuda():
        return False
    capability_tuple = current_platform.get_device_capability()
    capability = -1 if capability_tuple is None else capability_tuple.to_int()
    return cutlass_scaled_mm_supports_fp4(capability)


def pad_nvfp4_weight_for_cutlass(
    weight: torch.Tensor,
    alignment: int = 32,
) -> tuple[torch.Tensor, int]:
    """Pad packed NVFP4 weights so that both N (rows) and K (columns) satisfy
    the alignment constraints required by CUTLASS / FlashInfer FP4 kernels.

    CUTLASS FP4 kernel requires both K and N matrix dimensions to be divisible
    by 32 for aligned memory access and efficient tensor core operations.
    """
    weight_current_rows = weight.shape[0]

    # Pad N dimension (rows) if not aligned
    if weight_current_rows % alignment != 0:
        total_rows = round_up(weight_current_rows, alignment)
        pad_rows = total_rows - weight_current_rows
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows)).contiguous()

    # Check K dimension alignment
    # 2 FP4 items are packed per byte in the input dimension
    weight_current_col_bytes = weight.shape[1]
    weight_current_col_elements = weight_current_col_bytes * 2

    weights_padding_bytes = 0
    if weight_current_col_elements % alignment != 0:
        total_cols = round_up(weight_current_col_elements, alignment)
        pad_cols = total_cols - weight_current_col_elements
        # Convert from FP4 element count to bytes (2 FP4 values per byte)
        # pad_cols is always even since alignment=32 and current elements are even
        pad_bytes = pad_cols // 2
        weight = torch.nn.functional.pad(weight, (0, pad_bytes, 0, 0)).contiguous()
        weights_padding_bytes = pad_bytes

    return weight, weights_padding_bytes


def nvfp4_weight_padding_bytes(layer: torch.nn.Module) -> int:
    """Compute K-dim padding (packed bytes) of an NVFP4 weight,
    Aligned with `pad_nvfp4_weight_for_cutlass` above.
    """
    return layer.weight.shape[1] - layer.input_size_per_partition // 2


def pad_nvfp4_activation_for_cutlass(
    x_fp4: torch.Tensor,
    weights_padding_bytes: int,
) -> torch.Tensor:
    """Pad packed FP4 activations to match the K-dimension padding applied to weights.
    The padding is in bytes (tensor dimension), not FP4 elements.
    """
    if weights_padding_bytes > 0:
        return torch.nn.functional.pad(x_fp4, (0, weights_padding_bytes)).contiguous()
    return x_fp4


def slice_nvfp4_output(
    out: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Slice the output tensor to remove padding in N dimension if weight was padded."""
    if out.shape[-1] != output_size:
        return out[..., :output_size].contiguous()
    return out


def create_nvfp4_scale_gather_part(
    layer: torch.nn.Module,
    tensor: torch.Tensor,
    tp_size: int,
    buffers: WeightGatherWorkspace,
) -> WeightGatherPart:
    """Gather logical NVFP4 block scales and restore the CUTLASS 128x4 layout.

    Persistent buffers avoid allocating during prefetch. The collective uses
    uint8 aliases to preserve FP8 bits without requiring an FP8 collective.
    """
    scale = tensor.detach()
    output_size = layer.output_size_per_partition
    local_scale_cols = layer.input_size_per_partition // 16
    padded_output_size = round_up(output_size, 128)
    padded_local_scale_cols = round_up(local_scale_cols, 4)
    expected_shape = (padded_output_size, padded_local_scale_cols)
    if scale.dtype != torch.float8_e4m3fn or tuple(scale.shape) != expected_shape:
        raise RuntimeError(
            "Unexpected CUTLASS-compatible NVFP4 block-scale layout: "
            f"expected dtype=torch.float8_e4m3fn and shape {expected_shape}, "
            f"got dtype={scale.dtype} and shape={tuple(scale.shape)}."
        )

    full_scale_cols = local_scale_cols * tp_size
    padded_full_scale_cols = round_up(full_scale_cols, 4)
    name = "weight_scale.nvfp4_swizzled"
    gather_input = buffers.get(
        f"{name}.gather_input",
        (local_scale_cols, output_size),
        tensor=scale.view(torch.uint8),
    )
    gather_output = buffers.get(
        f"{name}.gather_output",
        (full_scale_cols, output_size),
        tensor=scale.view(torch.uint8),
    )
    local_padded = buffers.get(
        f"{name}.local_padded",
        (1, padded_output_size, padded_local_scale_cols),
        tensor=scale,
    )
    full_padded = buffers.get(
        f"{name}.full_padded",
        (1, padded_output_size, padded_full_scale_cols),
        tensor=scale,
    )
    full_tensor = buffers.get(
        f"{name}.full",
        (padded_output_size, padded_full_scale_cols),
        tensor=scale,
    )

    m_tiles = padded_output_size // 128
    local_k_tiles = padded_local_scale_cols // 4
    full_k_tiles = padded_full_scale_cols // 4

    def prepare() -> None:
        local_padded.view(1, m_tiles, 4, 32, local_k_tiles, 4).copy_(
            scale.view(1, m_tiles, local_k_tiles, 32, 4, 4).permute(0, 1, 4, 3, 2, 5)
        )
        gather_input.copy_(
            local_padded[0, :output_size, :local_scale_cols].view(torch.uint8).T
        )

    def finalize() -> None:
        full_padded.zero_()
        full_padded[0, :output_size, :full_scale_cols].view(torch.uint8).copy_(
            gather_output.T
        )
        full_tensor.view(1, m_tiles, full_k_tiles, 32, 4, 4).copy_(
            full_padded.view(1, m_tiles, 4, 32, full_k_tiles, 4).permute(
                0, 1, 4, 3, 2, 5
            )
        )

    return WeightGatherPart(
        local_tensor=scale,
        gather_input=gather_input,
        gather_output=gather_output,
        full_tensor=full_tensor,
        prepare=prepare,
        finalize=finalize,
    )
