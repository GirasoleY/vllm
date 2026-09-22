// SPDX-License-Identifier: Apache-2.0 AND MIT
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from FlashKDA b59532f1f464fbd536272780e30df5bf6a2ccc02.
// Native 16-token K1 iterations emit FLA-compatible chunk outputs.
// Diagonal inversion is fused before the FLA block-merge consumer.
/*
MIT License

Copyright (c) 2026 MoonshotAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/tensor_inl.h>

#include "smxx/utils.cuh"

template <int D, int CHUNK = 16>
struct KcpPrepareLayouts {
  static constexpr int kChunkSize = CHUNK;
  using QKLayout =
      decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
  using GLayout =
      decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
  using MMALayout =
      decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
                             make_shape(Int<CHUNK>{}, Int<D>{}), LayoutLeft{}));
  using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
  using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
  using LMLayout = decltype(tile_to_shape(
      GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
      make_shape(Int<CHUNK>{}, Int<CHUNK>{}), LayoutLeft{}));
  using LF32Layout = decltype(make_layout(
      make_shape(Int<CHUNK>{}, Int<CHUNK>{}), LayoutRight{}));

  using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
  using TMAQKLayout = decltype(prepend(QKLayout{}));
  using TMAGLayout = decltype(prepend(GLayout{}));
  using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));
};

template <class Layouts, class Element>
struct KcpPrepareSharedStorage {
  using BF16 = Element;
  using QKLayout = typename Layouts::QKLayout;
  using GLayout = typename Layouts::GLayout;
  using BetaSmemLayout = typename Layouts::BetaSmemLayout;
  using GTotalLayout = typename Layouts::GTotalLayout;
  using LMLayout = typename Layouts::LMLayout;
  using MMALayout = typename Layouts::MMALayout;

  // Phase A: q, k, g alive
  // Phase B: k_decayed, q_decayed, k_inv, L, Mqk alive
  // The input and matrix phases reuse the same shared memory.
  union {
    struct {
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> q;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> k;
      alignas(128) cute::ArrayEngine<float, cute::cosize_v<GLayout>> g;
    };
    struct {
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_inv;
      alignas(128) cute::ArrayEngine<float, cute::cosize_v<LMLayout>> L;
      alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };
  };

  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
  alignas(16) cute::ArrayEngine<float, 16> beta_act;
  alignas(16) cute::ArrayEngine<float, 16> q_norm_inv;
  alignas(16) cute::ArrayEngine<float, 16> k_norm_inv;
  float a_log_exp;

  alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> g_bf16;
  alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> dt_bias;
  alignas(16) cutlass::arch::ClusterTransactionBarrier tma_load_barrier;
};

struct KcpPrepareInputs {
  void const* q;
  void const* k;
  void const* raw_g;
  void const* beta;
  float const* bias;
  int64_t qs[3], ks[3], gs[3], bs[2];
};

__device__ __forceinline__ int64_t kcp_metadata(void const* p, int64_t i,
                                                bool is_i64) {
  return is_i64 ? static_cast<int64_t const*>(p)[i]
                : static_cast<int32_t const*>(p)[i];
}

template <class T>
__device__ __forceinline__ float kcp_to_float(T x) {
  if constexpr (std::is_same_v<T, cutlass::bfloat16_t>)
    return bf16_to_f32(x);
  else
    return float(x);
}

// Keep the FLA forward-substitution reduction in FP32. Each lane owns a
// matrix column; the seed is immutable while the inverse stays in registers.
template <int N>
__device__ __forceinline__ void kcp_inverse_diagonal(float const* seed,
                                                     float* output,
                                                     int64_t stride,
                                                     int actual_len, int lane) {
  static_assert(N > 0 && N <= 32 && (N & (N - 1)) == 0);
  constexpr unsigned mask = 0xffffffffu >> (32 - N);
  float inverse[N];
#pragma unroll
  for (int row = 0; row < N; ++row)
    inverse[row] = row > lane ? -seed[row * N + lane] : 0.0f;
#pragma unroll
  for (int row = 2; row < N; ++row) {
    const float coefficient = -seed[row * N + lane];
    float products[N];
#pragma unroll
    for (int j = 0; j < N; ++j) {
      const float value = __shfl_sync(mask, coefficient, j, N);
      products[j] = __fmul_rn(value, inverse[j]);
    }
#pragma unroll
    for (int distance = N / 2; distance > 0; distance /= 2) {
#pragma unroll
      for (int j = 0; j < distance; ++j)
        products[j] = __fadd_rn(products[j], products[j + distance]);
    }
    inverse[row] = __fadd_rn(coefficient, products[0]);
  }
#pragma unroll
  for (int row = 0; row < N; ++row)
    if (row < actual_len)
      output[int64_t(row) * stride + lane] =
          __fadd_rn(inverse[row], row == lane ? 1.0f : 0.0f);
}

template <class Element, int D, int ChunkSize, bool UseTma, bool SafeGate,
          bool UseMmaDiagonal, class TmaLoadQ, class TmaLoadK, class TmaLoadG,
          class TmaLoadDtBias>
__global__ void __launch_bounds__(128, 8)
    _kcp_flash_prepare(CUTE_GRID_CONSTANT TmaLoadQ const tma_load_q,
                       CUTE_GRID_CONSTANT TmaLoadK const tma_load_k,
                       CUTE_GRID_CONSTANT TmaLoadG const tma_load_g,
                       CUTE_GRID_CONSTANT TmaLoadDtBias const tma_load_dt_bias,
                       float scale, int64_t T_total, int H, int runtime_dim,
                       uint32_t num_chunks, void const* cu_seqlens, bool cu_i64,
                       void const* chunk_indices, bool chunks_i64,
                       float const* A_log_ptr, float gate_scale,
                       Element* q_norm_out, Element* k_norm_out,
                       float* g_prefix_out, float* beta_out, float* A_out,
                       float* Aqk_out, KcpPrepareInputs inputs) {
  constexpr int CHUNK = 16, NumThreads = 128, chunk_size = ChunkSize;
  const int actual_dim = UseTma ? D : runtime_dim;
  // --- constants
  using BF16 = Element;
  using Layouts = KcpPrepareLayouts<D, CHUNK>;
  using MMALayout = typename Layouts::MMALayout;
  using QKLayout = typename Layouts::QKLayout;
  using GLayout = typename Layouts::GLayout;
  using BetaSmemLayout = typename Layouts::BetaSmemLayout;
  using GTotalLayout = typename Layouts::GTotalLayout;
  using LMLayout = typename Layouts::LMLayout;
  using LF32Layout = typename Layouts::LF32Layout;
  using TMAQKLayout = typename Layouts::TMAQKLayout;
  using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
  using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
  static_assert(NumThreads == 128);
  constexpr uint32_t kTmaTransactionBytes =
      uint32_t(cute::cosize_v<QKLayout>) *
          uint32_t(3 * sizeof(BF16)) +        // q + k + g_bf16
      uint32_t(D) * uint32_t(sizeof(float));  // dt_bias

  // --- shared memory
  extern __shared__ __align__(128) unsigned char shared_mem[];
  using SharedStorageT = KcpPrepareSharedStorage<Layouts, Element>;
  SharedStorageT& shared_storage =
      *reinterpret_cast<SharedStorageT*>(shared_mem);

  const uint32_t chunk = blockIdx.x % num_chunks;
  const int head_idx = int(blockIdx.x / num_chunks);
  const int64_t seq_idx = kcp_metadata(chunk_indices, 2LL * chunk, chunks_i64);
  const int64_t chunk_idx =
      kcp_metadata(chunk_indices, 2LL * chunk + 1, chunks_i64);
  const int64_t bos = kcp_metadata(cu_seqlens, seq_idx, cu_i64);
  const int64_t eos = kcp_metadata(cu_seqlens, seq_idx + 1, cu_i64);
  const int64_t seq_len = eos - bos;
  float gate_prefix[(D + NumThreads - 1) / NumThreads] = {};
  if constexpr (UseTma) {
    if (threadIdx.x == 0) {
      shared_storage.tma_load_barrier.init(1);
      cutlass::arch::fence_barrier_init();
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int subchunk = 0; subchunk < chunk_size / CHUNK; ++subchunk) {
    const int64_t local_t = chunk_idx * (chunk_size / CHUNK) + subchunk;
    if (local_t * CHUNK >= seq_len) break;
    const int actual_len = int(min(int64_t(CHUNK), seq_len - local_t * CHUNK));
    const int64_t token_head_base = (bos + local_t * CHUNK) * H + head_idx;
    const int64_t vector_stride = int64_t(H) * actual_dim;
    const int64_t vector_base = token_head_base * actual_dim;
    const int64_t matrix_stride = int64_t(H) * chunk_size;
    const int64_t matrix_base = token_head_base * chunk_size + subchunk * CHUNK;
    if (threadIdx.x == 0) shared_storage.a_log_exp = expf(A_log_ptr[head_idx]);
    if constexpr (UseTma) {
      // --- TMA load inputs (single-shot, no pipeline)
      // Only thread 0 issues TMA loads (not elect_one_sync which is per-warp)
      if (threadIdx.x == 0) {
        using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
        shared_storage.tma_load_barrier.arrive_and_expect_tx(
            kTmaTransactionBytes);

        Tensor g_q = tma_load_q.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_k = tma_load_k.get_tma_tensor(make_shape(H, T_total, D));

        auto cta_tma_load_q = tma_load_q.get_slice(Int<0>{});
        auto cta_tma_load_k = tma_load_k.get_slice(Int<0>{});

        auto qk_off = g_q.layout()(head_idx, bos + local_t * CHUNK, 0);
        auto tile_shape_3d = make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{});
        auto tile_stride_3d = stride(g_q.layout());
        Tensor g_q_tile = make_tensor(
            g_q.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));
        Tensor g_k_tile = make_tensor(
            g_k.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));

        Tensor s_q_tile =
            make_tensor(make_smem_ptr(shared_storage.q.begin()), TMAQKLayout{});
        Tensor s_k_tile =
            make_tensor(make_smem_ptr(shared_storage.k.begin()), TMAQKLayout{});

        cute::copy(tma_load_q.with(reinterpret_cast<BarrierType&>(
                       shared_storage.tma_load_barrier)),
                   cta_tma_load_q.partition_S(g_q_tile),
                   cta_tma_load_q.partition_D(s_q_tile));
        cute::copy(tma_load_k.with(reinterpret_cast<BarrierType&>(
                       shared_storage.tma_load_barrier)),
                   cta_tma_load_k.partition_S(g_k_tile),
                   cta_tma_load_k.partition_D(s_k_tile));

        // TMA load g_bf16 (same gmem layout as q/k)
        Tensor g_g = tma_load_g.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_load_g = tma_load_g.get_slice(Int<0>{});
        Tensor g_g_tile = make_tensor(
            g_g.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));
        Tensor s_g_bf16_tile = make_tensor(
            make_smem_ptr(shared_storage.g_bf16.begin()), TMAQKLayout{});
        cute::copy(tma_load_g.with(reinterpret_cast<BarrierType&>(
                       shared_storage.tma_load_barrier)),
                   cta_tma_load_g.partition_S(g_g_tile),
                   cta_tma_load_g.partition_D(s_g_bf16_tile));

        // TMA load dt_bias [H, D] → [D] slice for current head
        Tensor g_dt = tma_load_dt_bias.get_tma_tensor(make_shape(H, D));
        auto cta_tma_load_dt = tma_load_dt_bias.get_slice(Int<0>{});
        auto dt_off = g_dt.layout()(head_idx, 0);
        Tensor g_dt_tile = make_tensor(
            g_dt.data() + dt_off,
            make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_dt.layout())));
        Tensor s_dt_tile =
            make_tensor(make_smem_ptr(shared_storage.dt_bias.begin()),
                        TMAGTotalSmemLayout{});
        cute::copy(tma_load_dt_bias.with(reinterpret_cast<BarrierType&>(
                       shared_storage.tma_load_barrier)),
                   cta_tma_load_dt.partition_S(g_dt_tile),
                   cta_tma_load_dt.partition_D(s_dt_tile));
      }

      __syncthreads();
      shared_storage.tma_load_barrier.wait(subchunk & 1);
      cutlass::arch::fence_view_async_shared();
      __syncthreads();
    } else {
      for (int i = threadIdx.x; i < CHUNK * D; i += NumThreads) {
        const int row = i / D, col = i % D;
        const int64_t token = bos + local_t * CHUNK + row;
        const bool valid = row < actual_len && col < actual_dim;
        shared_storage.q.begin()[i] =
            valid ? static_cast<Element const*>(
                        inputs.q)[token * inputs.qs[0] +
                                  head_idx * inputs.qs[1] + col * inputs.qs[2]]
                  : Element(0);
        shared_storage.k.begin()[i] =
            valid ? static_cast<Element const*>(
                        inputs.k)[token * inputs.ks[0] +
                                  head_idx * inputs.ks[1] + col * inputs.ks[2]]
                  : Element(0);
        shared_storage.g_bf16.begin()[i] =
            valid
                ? static_cast<Element const*>(
                      inputs
                          .raw_g)[token * inputs.gs[0] +
                                  head_idx * inputs.gs[1] + col * inputs.gs[2]]
                : Element(0);
      }
      for (int col = threadIdx.x; col < D; col += NumThreads)
        shared_storage.dt_bias.begin()[col] =
            col < actual_dim ? inputs.bias[int64_t(head_idx) * actual_dim + col]
                             : 0.0f;
    }
    // TMA may cross a ragged segment boundary; mask it before any arithmetic.
    if constexpr (UseTma) {
      if (actual_len < CHUNK) {
        for (int i = threadIdx.x; i < CHUNK * D; i += NumThreads) {
          if (i / D >= actual_len) {
            shared_storage.q.begin()[i] = Element(0);
            shared_storage.k.begin()[i] = Element(0);
            shared_storage.g_bf16.begin()[i] = Element(0);
          }
        }
        __syncthreads();
      }
    }
    int compute_tid = threadIdx.x;
    if (compute_tid < CHUNK) {
      const int64_t token = bos + local_t * CHUNK + compute_tid;
      shared_storage.beta_act.begin()[compute_tid] =
          compute_tid < actual_len
              ? sigmoid_tanh_approx_f32(float(static_cast<Element const*>(
                    inputs
                        .beta)[token * inputs.bs[0] + head_idx * inputs.bs[1]]))
              : 0.0f;
    }
    if constexpr (!UseTma) __syncthreads();

    // --- QK L2 Normalization ---
    if constexpr (std::is_same_v<Element, float>) {
      const int lane = compute_tid % 32, warp = compute_tid / 32;
      for (int row = warp; row < CHUNK; row += NumThreads / 32) {
        float q_sum = 0.0f, k_sum = 0.0f;
#pragma unroll
        for (int panel = 0; panel < D; panel += 128) {
          float qv[4], kv[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int col = panel + lane * 4 + j;
            qv[j] = col < D ? shared_storage.q.begin()[row * D + col] : 0.0f;
            kv[j] = col < D ? shared_storage.k.begin()[row * D + col] : 0.0f;
          }
          // Match the existing FLA FP32 L2 normalization reduction tree.
          float qs = __fmul_rn(qv[1], qv[1]);
          float ks = __fmul_rn(kv[1], kv[1]);
          qs = __fmaf_rn(qv[0], qv[0], qs);
          ks = __fmaf_rn(kv[0], kv[0], ks);
          qs = __fmaf_rn(qv[2], qv[2], qs);
          ks = __fmaf_rn(kv[2], kv[2], ks);
          qs = __fmaf_rn(qv[3], qv[3], qs);
          ks = __fmaf_rn(kv[3], kv[3], ks);
#pragma unroll
          for (int delta = 16; delta > 0; delta >>= 1) {
            qs = __fadd_rn(qs, __shfl_xor_sync(0xffffffff, qs, delta));
            ks = __fadd_rn(ks, __shfl_xor_sync(0xffffffff, ks, delta));
          }
          q_sum = panel == 0 ? qs : __fadd_rn(q_sum, qs);
          k_sum = panel == 0 ? ks : __fadd_rn(k_sum, ks);
        }
        if (lane == 0) {
          float qi, ki;
          const float qs = __fadd_rn(q_sum, 1e-6f),
                      ks = __fadd_rn(k_sum, 1e-6f);
          asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(qi) : "f"(qs));
          asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(ki) : "f"(ks));
          shared_storage.q_norm_inv.begin()[row] = qi;
          shared_storage.k_norm_inv.begin()[row] = ki;
        }
      }
    } else {
      constexpr int ELEMS_PER_THREAD = 8;
      constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD;  // 16
      constexpr int ROWS_PER_PASS = NumThreads / THREADS_PER_ROW;
      constexpr int ROW_PASSES = CHUNK / ROWS_PER_PASS;
      static_assert(CHUNK % ROWS_PER_PASS == 0);
      int my_col = (threadIdx.x % THREADS_PER_ROW) * ELEMS_PER_THREAD;

      BF16* q_smem = shared_storage.q.begin();
      BF16* k_smem = shared_storage.k.begin();
      using BF16x8 = cutlass::AlignedArray<BF16, ELEMS_PER_THREAD, 16>;

#pragma unroll
      for (int pass = 0; pass < ROW_PASSES; ++pass) {
        int my_row = pass * ROWS_PER_PASS + threadIdx.x / THREADS_PER_ROW;
        int row_offset = my_row * D + my_col;
        BF16x8 q_pack = *reinterpret_cast<BF16x8 const*>(q_smem + row_offset);
        BF16x8 k_pack = *reinterpret_cast<BF16x8 const*>(k_smem + row_offset);
        float q_sq = 0.0f, k_sq = 0.0f;

#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
          float qv = kcp_to_float(q_pack[i]);
          float kv = kcp_to_float(k_pack[i]);
          q_sq += qv * qv;
          k_sq += kv * kv;
        }

#pragma unroll
        for (int delta = THREADS_PER_ROW / 2; delta >= 1; delta >>= 1) {
          q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta, THREADS_PER_ROW);
          k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta, THREADS_PER_ROW);
        }

        if ((threadIdx.x % THREADS_PER_ROW) == 0) {
          shared_storage.q_norm_inv.begin()[my_row] = rsqrtf(q_sq + 1e-6f);
          shared_storage.k_norm_inv.begin()[my_row] = rsqrtf(k_sq + 1e-6f);
        }
      }
    }
    // --- Fused gate activation + cumsum ---
    // Q/K remain raw in shared memory. The decay pass consumes the row inverse
    // norms above and rounds normalized values to BF16 directly in registers.
    for (int col = compute_tid, part = 0; col < D; col += NumThreads, ++part) {
      BF16 const* g_bf16_smem = shared_storage.g_bf16.begin();
      float dt = shared_storage.dt_bias.begin()[col];
      float* g_smem = shared_storage.g.begin();
      float sum = 0.0f;
      int64_t output_offset = vector_base + col;
#pragma unroll
      for (int row = 0; row < CHUNK; ++row) {
        float g_val = 0.0f;
        if (row < actual_len && col < actual_dim) {
          g_val = kcp_to_float(g_bf16_smem[row * D + col]) + dt;
          if constexpr (SafeGate) {
            g_val = shared_storage.a_log_exp * g_val;
            g_val = gate_scale * sigmoid_tanh_approx_f32(g_val);
          } else {
            const float softplus = g_val > 20.0f ? g_val : log1pf(expf(g_val));
            g_val =
                (-shared_storage.a_log_exp * softplus) * 1.4426950408889634f;
          }
        }
        sum += g_val;
        g_smem[row * D + col] = sum;
        if (row < actual_len && col < actual_dim) {
          g_prefix_out[output_offset] = gate_prefix[part] + sum;
        }
        output_offset += vector_stride;
      }
      gate_prefix[part] += sum;
    }
    __syncthreads();

    // Dense outputs preserve normalized BF16 materialization for FLA consumers.
    for (int coordinate = compute_tid; coordinate < CHUNK * D;
         coordinate += NumThreads) {
      int token_in_subchunk = coordinate / D;
      int channel = coordinate % D;
      if (token_in_subchunk >= actual_len || channel >= actual_dim) continue;
      int64_t offset =
          vector_base + int64_t(token_in_subchunk) * vector_stride + channel;
      q_norm_out[offset] =
          BF16(kcp_to_float(shared_storage.q.begin()[coordinate]) *
               shared_storage.q_norm_inv.begin()[token_in_subchunk]);
      k_norm_out[offset] =
          BF16(kcp_to_float(shared_storage.k.begin()[coordinate]) *
               shared_storage.k_norm_inv.begin()[token_in_subchunk]);
    }
    if (compute_tid < actual_len) {
      beta_out[token_head_base + int64_t(compute_tid) * H] =
          shared_storage.beta_act.begin()[compute_tid];
    }
    __syncthreads();

    if constexpr (!UseMmaDiagonal) {
      // IEEE diagonal reduction also avoids FP16 or unbounded-gate overflow
      // in separately materialized exp(g) and exp(-g) factors.
      const int lane = compute_tid % 32, warp = compute_tid / 32;
      for (int pair = warp; pair < CHUNK * CHUNK; pair += NumThreads / 32) {
        const int row = pair / CHUNK, col = pair % CHUNK;
        float av = 0.0f, qv = 0.0f;
        if (row < actual_len && col <= row && col < actual_len) {
          const int64_t oi = vector_base + int64_t(row) * vector_stride;
          const int64_t oj = vector_base + int64_t(col) * vector_stride;
          for (int d = lane; d < D; d += 32) {
            if (d < actual_dim) {
              const float ktg =
                  __fmul_rn(kcp_to_float(k_norm_out[oj + d]),
                            ex2_approx_ftz_f32(g_prefix_out[oi + d] -
                                               g_prefix_out[oj + d]));
              av = __fadd_rn(
                  av, __fmul_rn(__fmul_rn(kcp_to_float(k_norm_out[oi + d]),
                                          shared_storage.beta_act.begin()[row]),
                                ktg));
              qv = __fadd_rn(qv,
                             __fmul_rn(kcp_to_float(q_norm_out[oi + d]), ktg));
            }
          }
        }
#pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
          av = __fadd_rn(av, __shfl_xor_sync(0xffffffff, av, delta));
          qv = __fadd_rn(qv, __shfl_xor_sync(0xffffffff, qv, delta));
        }
        if (lane == 0) {
          shared_storage.L.begin()[row * CHUNK + col] = row > col ? av : 0.0f;
          if (row < actual_len) {
            const int64_t offset =
                matrix_base + int64_t(row) * matrix_stride + col;
            Aqk_out[offset] = row >= col ? __fmul_rn(qv, scale) : 0.0f;
          }
        }
      }
    } else {
      static_assert(std::is_same_v<Element, cutlass::bfloat16_t>);
      Tensor q_tile =
          make_tensor(make_smem_ptr(shared_storage.q.begin()), QKLayout{});
      Tensor k_tile =
          make_tensor(make_smem_ptr(shared_storage.k.begin()), QKLayout{});
      Tensor g_tile =
          make_tensor(make_smem_ptr(shared_storage.g.begin()), GLayout{});
      Tensor k_decayed = make_tensor(
          make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{});
      Tensor q_decayed = make_tensor(
          make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{});
      Tensor k_inv =
          make_tensor(make_smem_ptr(shared_storage.k_inv.begin()), MMALayout{});
      // decay_apply
      if (compute_tid < 256) {
        static_assert(D % 64 == 0);
        static_assert(CHUNK % 8 == 0);

        int lane = compute_tid % 32;
        int warp_id = compute_tid / 32;
        int g = lane / 4;
        int t = lane % 4;

        auto vec8_2d = make_shape(_1{}, _8{});
        auto thr2_2d = make_shape(_1{}, _2{});

        constexpr int N_M = CHUNK / 8;
        constexpr int N_N = D / 64;
        constexpr int N_TILES = N_M * N_N;
        constexpr int PHYSICAL_WARPS = NumThreads / 32;
        constexpr int WARP_PASSES = 8 / PHYSICAL_WARPS;
        static_assert(8 % PHYSICAL_WARPS == 0);

        // Four physical warps cover the original eight warp assignments in
        // two passes. Buffer every input before overwriting the union'd smem.
        float reg_g[WARP_PASSES][N_TILES][2];
        BF16 reg_q[WARP_PASSES][N_TILES][2];
        BF16 reg_k[WARP_PASSES][N_TILES][2];

#pragma unroll
        for (int pass = 0; pass < WARP_PASSES; ++pass) {
          int virtual_warp_id = warp_id + pass * PHYSICAL_WARPS;
#pragma unroll
          for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
#pragma unroll
            for (int n_blk = 0; n_blk < D; n_blk += 64) {
              int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
              int row = m_blk + ((virtual_warp_id + g) % 8);
              int col_base = n_blk + g * 8;
              int col_tile = col_base / 8;

              Tensor tile_g =
                  local_tile(g_tile, vec8_2d, make_coord(row, col_tile));
              Tensor tile_q =
                  local_tile(q_tile, vec8_2d, make_coord(row, col_tile));
              Tensor tile_k =
                  local_tile(k_tile, vec8_2d, make_coord(row, col_tile));

              Tensor s_g = local_tile(tile_g, thr2_2d, make_coord(0, t));
              Tensor s_q = local_tile(tile_q, thr2_2d, make_coord(0, t));
              Tensor s_k = local_tile(tile_k, thr2_2d, make_coord(0, t));

              Tensor r_g = make_tensor_like<float>(s_g);
              Tensor r_q = make_tensor_like<BF16>(s_q);
              Tensor r_k = make_tensor_like<BF16>(s_k);

              cute::copy(AutoVectorizingCopy{}, s_g, r_g);
              cute::copy(AutoVectorizingCopy{}, s_q, r_q);
              cute::copy(AutoVectorizingCopy{}, s_k, r_k);

#pragma unroll
              for (int v = 0; v < 2; ++v) {
                reg_g[pass][tile_idx][v] = r_g(0, v);
                reg_q[pass][tile_idx][v] = r_q(0, v);
                reg_k[pass][tile_idx][v] = r_k(0, v);
              }
            }
          }
        }

        // Sync before writing to union'd smem (q/k/g →
        // k_decayed/q_decayed/k_inv) All virtual-warp inputs must be resident
        // before any union'd output writes.
        __syncthreads();

#pragma unroll
        for (int pass = 0; pass < WARP_PASSES; ++pass) {
          int virtual_warp_id = warp_id + pass * PHYSICAL_WARPS;
#pragma unroll
          for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
#pragma unroll
            for (int n_blk = 0; n_blk < D; n_blk += 64) {
              int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
              int row = m_blk + ((virtual_warp_id + g) % 8);
              int col_base = n_blk + g * 8;
              int col_tile = col_base / 8;

              Tensor tile_qd =
                  local_tile(q_decayed, vec8_2d, make_coord(row, col_tile));
              Tensor tile_kd =
                  local_tile(k_decayed, vec8_2d, make_coord(row, col_tile));
              Tensor tile_ki =
                  local_tile(k_inv, vec8_2d, make_coord(row, col_tile));

              Tensor s_qd = local_tile(tile_qd, thr2_2d, make_coord(0, t));
              Tensor s_kd = local_tile(tile_kd, thr2_2d, make_coord(0, t));
              Tensor s_ki = local_tile(tile_ki, thr2_2d, make_coord(0, t));

              Tensor r_qd = make_tensor_like<BF16>(s_qd);
              Tensor r_kd = make_tensor_like<BF16>(s_kd);
              float q_inv = shared_storage.q_norm_inv.begin()[row];
              float k_inv_scale = shared_storage.k_norm_inv.begin()[row];
#pragma unroll
              for (int v = 0; v < 2; ++v) {
                float g = reg_g[pass][tile_idx][v];
                BF16 q = BF16(kcp_to_float(reg_q[pass][tile_idx][v]) * q_inv);
                BF16 k = row < actual_len
                             ? BF16(kcp_to_float(reg_k[pass][tile_idx][v]) *
                                    k_inv_scale)
                             : BF16(0);
                BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(g));
                r_qd(0, v) = q * exp_cumsum * BF16(scale);
                r_kd(0, v) = k * exp_cumsum;
              }
              cute::copy(AutoVectorizingCopy{}, r_qd, s_qd);
              cute::copy(AutoVectorizingCopy{}, r_kd, s_kd);

              Tensor r_ki = make_tensor_like<BF16>(s_ki);
#pragma unroll
              for (int v = 0; v < 2; ++v) {
                float g = reg_g[pass][tile_idx][v];
                BF16 k = row < actual_len
                             ? BF16(kcp_to_float(reg_k[pass][tile_idx][v]) *
                                    k_inv_scale)
                             : BF16(0);
                BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-g));
                r_ki(0, v) = k * inv_cumsum;
              }
              cute::copy(AutoVectorizingCopy{}, r_ki, s_ki);
            }
          }
        }
      }
      __syncthreads();

      Tensor L_fp32 =
          make_tensor(make_smem_ptr(shared_storage.L.begin()), LF32Layout{});
      Tensor Mqk =
          make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{});

      // L_Mqk
      if (compute_tid < 32) {
        mma_m16n16_bf16bf16fp32_1warp(k_decayed, k_inv, L_fp32, compute_tid);
      } else if (compute_tid >= 32 && compute_tid < 64) {
        mma_m16n16_bf16bf16bf16_1warp(q_decayed, k_inv, Mqk, compute_tid - 32);
      }
      __syncthreads();

      // tril L (beta applied) + tril Mqk (merged, same thread same element)
      for (int matrix_idx = compute_tid; matrix_idx < CHUNK * CHUNK;
           matrix_idx += NumThreads) {
        const int col_block_size = 8;
        int block_idx = matrix_idx / (CHUNK * col_block_size);
        int i = (matrix_idx / col_block_size) % CHUNK;
        int j = matrix_idx % col_block_size + block_idx * col_block_size;
        if (i <= j) {
          L_fp32(i, j) = 0.0f;
        } else {
          L_fp32(i, j) = L_fp32(i, j) * shared_storage.beta_act.begin()[i];
        }
        if (i < j) {
          Mqk(i, j) = BF16(0);
        }
      }
      __syncthreads();

      // Publish query-key products before reusing the shared matrix storage.
      for (int coordinate = compute_tid; coordinate < CHUNK * CHUNK;
           coordinate += NumThreads) {
        int token_in_subchunk = coordinate / CHUNK;
        int column = coordinate % CHUNK;
        if (token_in_subchunk >= actual_len) continue;
        int64_t offset =
            matrix_base + int64_t(token_in_subchunk) * matrix_stride + column;
        Aqk_out[offset] = kcp_to_float(Mqk(token_in_subchunk, column));
      }
    }
    __syncthreads();
    if (compute_tid < CHUNK) {
      kcp_inverse_diagonal<CHUNK>(shared_storage.L.begin(), A_out + matrix_base,
                                  matrix_stride, actual_len, compute_tid);
    }
    // Publish every generic shared write to the async proxy before the next
    // TMA overwrites the union; CTA join also completes all generic readers.
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
  }
}

using torch::headeronly::ScalarType;
using StableTensor = torch::stable::Tensor;

namespace {
void check_cuda(cudaError_t status) {
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
bool same_shape(const StableTensor& a, const StableTensor& b) {
  if (a.dim() != b.dim()) return false;
  for (int i = 0; i < a.dim(); ++i)
    if (a.size(i) != b.size(i)) return false;
  return true;
}

struct PrepareArgs {
  const StableTensor &q, &k, &raw_g, &beta_raw, &a_log, &bias, &cu, &chunks;
  const StableTensor &qn, &kn, &g, &beta, &A, &Aqk;
  float gate_scale;
  bool safe_gate;
  int chunk_size;
  cudaStream_t stream;
};

template <class Element, int D, int ChunkSize, bool UseTma, bool SafeGate,
          bool UseMmaDiagonal, class TQ, class TK, class TG, class TD>
void launch_gate(const PrepareArgs& a, TQ const& tq, TK const& tk, TG const& tg,
                 TD const& td) {
  const int H = int(a.q.size(2)), actual_dim = int(a.q.size(3));
  KcpPrepareInputs inputs{
      a.q.const_data_ptr(),
      a.k.const_data_ptr(),
      a.raw_g.const_data_ptr(),
      a.beta_raw.const_data_ptr(),
      static_cast<float const*>(a.bias.const_data_ptr()),
      {a.q.stride(1), a.q.stride(2), a.q.stride(3)},
      {a.k.stride(1), a.k.stride(2), a.k.stride(3)},
      {a.raw_g.stride(1), a.raw_g.stride(2), a.raw_g.stride(3)},
      {a.beta_raw.stride(1), a.beta_raw.stride(2)}};
  auto kernel = _kcp_flash_prepare<Element, D, ChunkSize, UseTma, SafeGate,
                                   UseMmaDiagonal, TQ, TK, TG, TD>;
  const int shared =
      sizeof(KcpPrepareSharedStorage<KcpPrepareLayouts<D>, Element>);
  check_cuda(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
  kernel<<<dim3(a.chunks.size(0) * H), 128, shared, a.stream>>>(
      tq, tk, tg, td, 1.0f / sqrtf(float(actual_dim)), a.q.size(1), H,
      actual_dim, uint32_t(a.chunks.size(0)), a.cu.const_data_ptr(),
      a.cu.scalar_type() == ScalarType::Long, a.chunks.const_data_ptr(),
      a.chunks.scalar_type() == ScalarType::Long,
      static_cast<float const*>(a.a_log.const_data_ptr()), a.gate_scale,
      reinterpret_cast<Element*>(a.qn.mutable_data_ptr()),
      reinterpret_cast<Element*>(a.kn.mutable_data_ptr()),
      static_cast<float*>(a.g.mutable_data_ptr()),
      static_cast<float*>(a.beta.mutable_data_ptr()),
      static_cast<float*>(a.A.mutable_data_ptr()),
      static_cast<float*>(a.Aqk.mutable_data_ptr()), inputs);
  check_cuda(cudaGetLastError());
}

template <class Element, int D, int ChunkSize, bool UseTma, class TQ, class TK,
          class TG, class TD>
void launch_chunk(const PrepareArgs& a, TQ const& tq, TK const& tk,
                  TG const& tg, TD const& td) {
  if (!a.safe_gate) {
    launch_gate<Element, D, ChunkSize, UseTma, false, false>(a, tq, tk, tg, td);
    return;
  }
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
    // Reserve BF16 fraction bits and Q's 1/sqrt(actual_dim) scaling within
    // the normal exponent range. This is a precision budget for factorized
    // operands, not a guarantee that every normalized component is normal.
    // Outside it, use direct causal exponent differences with IEEE products.
    constexpr int fraction_bits = std::numeric_limits<float>::digits - 1 -
                                  (sizeof(float) - sizeof(Element)) * CHAR_BIT;
    const float factor_budget =
        float(std::min(std::numeric_limits<float>::max_exponent - 1,
                       1 - std::numeric_limits<float>::min_exponent) -
              fraction_bits) -
        0.5f * std::log2(float(a.q.size(3)));
    float minimum_prefix = 0.0f;
    for (int token = 0; token < KcpPrepareLayouts<D>::kChunkSize; ++token) {
      minimum_prefix = std::nextafter(minimum_prefix + a.gate_scale,
                                      -std::numeric_limits<float>::infinity());
    }
    if (-minimum_prefix < factor_budget) {
      launch_gate<Element, D, ChunkSize, UseTma, true, true>(a, tq, tk, tg, td);
      return;
    }
  }
  launch_gate<Element, D, ChunkSize, UseTma, true, false>(a, tq, tk, tg, td);
}

template <class Element, int D, bool UseTma, class TQ, class TK, class TG,
          class TD>
void launch(const PrepareArgs& a, TQ const& tq, TK const& tk, TG const& tg,
            TD const& td) {
  if (a.chunk_size == 16)
    launch_chunk<Element, D, 16, UseTma>(a, tq, tk, tg, td);
  else if (a.chunk_size == 32)
    launch_chunk<Element, D, 32, UseTma>(a, tq, tk, tg, td);
  else
    launch_chunk<Element, D, 64, UseTma>(a, tq, tk, tg, td);
}

template <class Element, int D>
void dispatch(const PrepareArgs& a) {
  // Keep the existing dense BF16 launch and its shared-memory/MMA arithmetic.
  if constexpr (std::is_same_v<Element, cutlass::bfloat16_t> && D == 128) {
    if (a.q.size(3) == D && a.q.is_contiguous() && a.k.is_contiguous() &&
        a.raw_g.is_contiguous() &&
        reinterpret_cast<uintptr_t>(a.q.const_data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(a.k.const_data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(a.raw_g.const_data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(a.bias.const_data_ptr()) % 16 == 0) {
      using L = KcpPrepareLayouts<D>;
      const int H = int(a.q.size(2));
      const int64_t T = a.q.size(1);
      auto layout =
          make_layout(make_shape(H, T, D),
                      make_stride(int64_t(D), int64_t(D) * H, int64_t(1)));
      auto mq = make_tensor(
          make_gmem_ptr(reinterpret_cast<Element const*>(a.q.const_data_ptr())),
          layout);
      auto mk = make_tensor(
          make_gmem_ptr(reinterpret_cast<Element const*>(a.k.const_data_ptr())),
          layout);
      auto mg = make_tensor(make_gmem_ptr(reinterpret_cast<Element const*>(
                                a.raw_g.const_data_ptr())),
                            layout);
      auto md = make_tensor(
          make_gmem_ptr(static_cast<float const*>(a.bias.const_data_ptr())),
          make_layout(make_shape(H, D), LayoutRight{}));
      auto tq = make_tma_copy(SM90_TMA_LOAD{}, mq, typename L::TMAQKLayout{});
      auto tk = make_tma_copy(SM90_TMA_LOAD{}, mk, typename L::TMAQKLayout{});
      auto tg = make_tma_copy(SM90_TMA_LOAD{}, mg, typename L::TMAQKLayout{});
      auto td =
          make_tma_copy(SM90_TMA_LOAD{}, md, typename L::TMAGTotalSmemLayout{});
      launch<Element, D, true>(a, tq, tk, tg, td);
      return;
    }
  }
  launch<Element, D, false>(a, 0, 0, 0, 0);
}

template <class Element>
void dispatch_dim(const PrepareArgs& a) {
  if (a.q.size(3) <= 64)
    dispatch<Element, 64>(a);
  else if (a.q.size(3) <= 128)
    dispatch<Element, 128>(a);
  else
    dispatch<Element, 256>(a);
}
}  // namespace

void kcp_prepare(const StableTensor& q, const StableTensor& k,
                 const StableTensor& raw_g, const StableTensor& beta_raw,
                 const StableTensor& a_log, const StableTensor& bias,
                 const StableTensor& cu, const StableTensor& chunks,
                 const StableTensor& qn, const StableTensor& kn,
                 const StableTensor& g, const StableTensor& beta,
                 const StableTensor& A, const StableTensor& Aqk,
                 double lower_bound, bool safe_gate, int64_t chunk_size) {
  STD_TORCH_CHECK(q.is_cuda(), "KCP preparation requires CUDA tensors");
  const auto device = q.get_device_index();
  const torch::stable::accelerator::DeviceGuard guard(device);
  for (const StableTensor* t : {&q, &k, &raw_g, &beta_raw, &a_log, &bias, &cu,
                                &chunks, &qn, &kn, &g, &beta, &A, &Aqk}) {
    STD_TORCH_CHECK(t->is_cuda() && t->get_device_index() == device,
                    "all preparation tensors must be on q's CUDA device");
    for (int i = 0; i < t->dim(); ++i)
      STD_TORCH_CHECK(t->stride(i) >= 0,
                      "preparation requires nonnegative strides");
  }
  for (const StableTensor* t :
       {&a_log, &bias, &cu, &chunks, &qn, &kn, &g, &beta, &A, &Aqk})
    STD_TORCH_CHECK(t->is_contiguous(),
                    "parameters, metadata and outputs must be contiguous");
  const auto dtype = q.scalar_type();
  STD_TORCH_CHECK(dtype == ScalarType::BFloat16 || dtype == ScalarType::Half ||
                      dtype == ScalarType::Float,
                  "preparation supports BF16, FP16 and FP32 inputs");
  for (const StableTensor* t : {&k, &raw_g, &beta_raw, &qn, &kn})
    STD_TORCH_CHECK(t->scalar_type() == dtype,
                    "preparation input/output dtypes must agree");
  for (const StableTensor* t : {&a_log, &bias, &g, &beta, &A, &Aqk})
    STD_TORCH_CHECK(t->scalar_type() == ScalarType::Float,
                    "parameters and matrix outputs must be FP32");
  STD_TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(2) > 0 &&
                      q.size(3) > 0 && q.size(3) <= 256,
                  "q must be [1,T,H,D], with positive H and 0 < D <= 256");
  for (const StableTensor* t : {&k, &raw_g, &qn, &kn, &g})
    STD_TORCH_CHECK(same_shape(q, *t), "q/k/g shapes must agree");
  const int64_t T = q.size(1), H = q.size(2), D = q.size(3);
  STD_TORCH_CHECK(beta_raw.dim() == 3 && beta_raw.size(0) == 1 &&
                      beta_raw.size(1) == T && beta_raw.size(2) == H &&
                      same_shape(beta_raw, beta),
                  "beta input and output must be [1,T,H]");
  STD_TORCH_CHECK(a_log.numel() == H && bias.numel() == H * D,
                  "invalid gate parameter sizes");
  STD_TORCH_CHECK(
      cu.dim() == 1 && cu.numel() >= 1 &&
          (cu.scalar_type() == ScalarType::Int ||
           cu.scalar_type() == ScalarType::Long) &&
          chunks.dim() == 2 && chunks.size(1) == 2 &&
          (chunks.scalar_type() == ScalarType::Int ||
           chunks.scalar_type() == ScalarType::Long),
      "cu and chunk metadata must be contiguous int32 or int64 tensors");
  STD_TORCH_CHECK(chunk_size == 16 || chunk_size == 32 || chunk_size == 64,
                  "KCP chunks must contain 16, 32 or 64 tokens");
  STD_TORCH_CHECK(A.dim() == 4 && A.size(0) == 1 && A.size(1) == T &&
                      A.size(2) == H && A.size(3) == chunk_size &&
                      same_shape(A, Aqk),
                  "invalid KKT output shapes");
  STD_TORCH_CHECK(!safe_gate || (std::isfinite(lower_bound) && lower_bound < 0),
                  "safe gate lower bound must be finite and negative");
  const float gate_scale = float(lower_bound * 1.4426950408889634);
  STD_TORCH_CHECK(!safe_gate || std::isfinite(gate_scale),
                  "safe gate scale must be representable in FP32");
  STD_TORCH_CHECK(chunks.size(0) == 0 ||
                      H <= std::numeric_limits<int>::max() / chunks.size(0),
                  "chunk/head grid exceeds CUDA x dimension");
  if (T == 0 || chunks.size(0) == 0) {
    STD_TORCH_CHECK(T == 0 && chunks.size(0) == 0,
                    "positive tokens require chunk metadata");
    return;
  }
  STD_TORCH_CHECK(cu.numel() >= 2, "positive tokens require a sequence");
  // The caller certifies metadata values; no device readback or eager sync.
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device, &stream_ptr));
  PrepareArgs args{q,
                   k,
                   raw_g,
                   beta_raw,
                   a_log,
                   bias,
                   cu,
                   chunks,
                   qn,
                   kn,
                   g,
                   beta,
                   A,
                   Aqk,
                   gate_scale,
                   safe_gate,
                   int(chunk_size),
                   static_cast<cudaStream_t>(stream_ptr)};
  if (dtype == ScalarType::BFloat16)
    dispatch_dim<cutlass::bfloat16_t>(args);
  else if (dtype == ScalarType::Half)
    dispatch_dim<cutlass::half_t>(args);
  else
    dispatch_dim<float>(args);
}
