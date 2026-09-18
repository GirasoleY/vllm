// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

#include <torch/csrc/stable/tensor.h>

namespace flashkda_kcp {

void kcp_transition(const torch::stable::Tensor& workspace,
                    const torch::stable::Tensor& beta,
                    const torch::stable::Tensor& cu_seqlens,
                    const torch::stable::Tensor& chunk_offsets,
                    const torch::stable::Tensor& partial);

void kcp_scan(const torch::stable::Tensor& v, const torch::stable::Tensor& beta,
              const torch::stable::Tensor& workspace,
              const torch::stable::Tensor& initial,
              const torch::stable::Tensor& cu_seqlens,
              const torch::stable::Tensor& out,
              const torch::stable::Tensor& final);

}  // namespace flashkda_kcp
