// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "flashkda_kcp.h"
#include "libtorch_stable/torch_utils.h"
#include "smxx/fwd_kernel1.cuh"
#include "smxx/fwd_kernel2.cuh"

#include <cuda_bf16.h>
#include <limits>

namespace flashkda_kcp {
using TorchTensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using BF16 = cutlass::bfloat16_t;
using WS = WorkspaceSizes<16, 128>;
constexpr int kSplits = 4;
constexpr int kComponents = 3;

// Compile the preparation, transition, and scan in the same extension. Its
// workspace is private to this pinned FlashKDA implementation.
static_assert(
    std::is_same_v<K1Layouts<128>::MMALayout, K2Layouts<128>::MMALayout>);
static_assert(
    std::is_same_v<K1Layouts<128>::LMLayout, K2Layouts<128>::LMLayout>);
static_assert(
    std::is_same_v<K1Layouts<128>::GTotalLayout, K2Layouts<128>::GTotalLayout>);

struct Workspace {
  BF16 *kd, *qd, *kr, *inv, *mqk;
  float* gt;

  __host__ __device__ Workspace(void* ptr, int64_t head_tiles) {
    auto p = static_cast<char*>(ptr);
    kd = reinterpret_cast<BF16*>(p);
    p += head_tiles * WS::kKDecayed;
    qd = reinterpret_cast<BF16*>(p);
    p += head_tiles * WS::kQDecayed;
    kr = reinterpret_cast<BF16*>(p);
    p += head_tiles * WS::kKRestored;
    gt = reinterpret_cast<float*>(p);
    p += head_tiles * WS::kGTotal;
    inv = reinterpret_cast<BF16*>(p);
    p += head_tiles * WS::kINV;
    mqk = reinterpret_cast<BF16*>(p);
  }
};

void check_cuda(const TorchTensor& tensor, ScalarType dtype, int device) {
  STD_TORCH_CHECK(tensor.is_cuda() && tensor.get_device_index() == device &&
                      tensor.is_contiguous() && tensor.scalar_type() == dtype,
                  "KCP requires contiguous tensors of the expected dtype on "
                  "the same CUDA device");
}

int check_inputs(const TorchTensor& workspace, const TorchTensor& beta,
                 const TorchTensor& cu) {
  const int device = beta.get_device_index();
  check_cuda(beta, ScalarType::BFloat16, device);
  check_cuda(workspace, ScalarType::Byte, device);
  check_cuda(cu, ScalarType::Int, device);
  STD_TORCH_CHECK(beta.dim() == 3 && beta.size(0) == 1 && beta.size(2) > 0,
                  "beta must be [1, tokens, heads]");
  STD_TORCH_CHECK(cu.dim() == 1 && cu.numel() >= 2,
                  "cu_seqlens must describe at least one segment");
  const int64_t segments = cu.numel() - 1;
  const int64_t tokens = beta.size(1), heads = beta.size(2);
  STD_TORCH_CHECK(tokens <= std::numeric_limits<int>::max() &&
                      heads * tokens <= std::numeric_limits<int>::max() &&
                      segments * kSplits <= 65535,
                  "KCP launch dimensions exceed supported bounds");
  const int64_t tiles = (tokens + 15) / 16 + segments;
  STD_TORCH_CHECK(workspace.numel() >= heads * tiles * WS::kPerTile,
                  "KCP workspace is too small");
  return static_cast<int>(tiles);
}

void check_state(const TorchTensor& state, int segments, int heads,
                 int device) {
  check_cuda(state, ScalarType::Float, device);
  STD_TORCH_CHECK(state.dim() == 4 && state.size(0) == segments &&
                      state.size(1) == heads && state.size(2) == 128 &&
                      state.size(3) == 128,
                  "KCP state must be [segments, heads, 128, 128]");
}

void check_launch(cudaError_t status) {
  STD_TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}

template <int ValueWidth>
void launch_scan(const TorchTensor& v, const TorchTensor& beta,
                 const TorchTensor& workspace, const TorchTensor& initial,
                 const TorchTensor& cu, const TorchTensor& out,
                 const TorchTensor& final, int tiles, cudaStream_t stream) {
  using Layouts = K2Layouts<128, 16, ValueWidth>;
  const int tokens = v.size(1), heads = v.size(2), segments = cu.numel() - 1;
  auto token_layout = make_layout(make_shape(heads, tokens, 128),
                                  make_stride(128, 128 * heads, 1));
  auto state_layout =
      make_layout(make_shape(segments * heads, 128, 128), LayoutRight{});
  auto v_tensor =
      make_tensor(make_gmem_ptr(static_cast<const BF16*>(v.const_data_ptr())),
                  token_layout);
  auto out_ptr = static_cast<BF16*>(out.mutable_data_ptr());
  auto out_tensor = make_tensor(make_gmem_ptr(out_ptr), token_layout);
  auto beta_tensor = make_tensor(
      make_gmem_ptr(static_cast<const BF16*>(beta.const_data_ptr())),
      make_layout(make_shape(heads * tokens)));
  auto initial_tensor = make_tensor(
      make_gmem_ptr(static_cast<const float*>(initial.const_data_ptr())),
      state_layout);
  auto final_tensor =
      make_tensor(make_gmem_ptr(static_cast<float*>(final.mutable_data_ptr())),
                  state_layout);
  auto load_v =
      make_tma_copy(SM90_TMA_LOAD{}, v_tensor, typename Layouts::TMAVOLayout{});
  auto load_beta = make_tma_copy(SM90_TMA_LOAD{}, beta_tensor,
                                 typename Layouts::TMABetaSmemLayout{});
  auto load_initial = make_tma_copy(SM90_TMA_LOAD{}, initial_tensor,
                                    typename Layouts::TMAFP32StateSmemLayout{});
  auto store_final = make_tma_copy(SM90_TMA_STORE{}, final_tensor,
                                   typename Layouts::TMAFP32StateSmemLayout{});
  auto store_out = make_tma_copy(SM90_TMA_STORE{}, out_tensor,
                                 typename Layouts::TMAVOLayout{});
  Workspace ws(workspace.mutable_data_ptr(), int64_t(heads) * tiles);
  auto kernel =
      _flash_kda_fwd_recurrence<decltype(load_v), decltype(load_beta),
                                decltype(load_initial), decltype(store_final),
                                decltype(store_out), 16, 128, 3, 2, 192, true,
                                true, true, false, true, int32_t, ValueWidth>;
  constexpr int shared_bytes = sizeof(SharedStorageK2<Layouts, 3, 2>);
  check_launch(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
  kernel<<<dim3(heads * (128 / ValueWidth), segments), 192, shared_bytes,
           stream>>>(load_v, load_beta, load_initial, store_final, store_out,
                     out_ptr, nullptr, nullptr, tokens, heads, segments,
                     static_cast<const int32_t*>(cu.const_data_ptr()), tiles,
                     ws.kd, ws.qd, ws.kr, ws.gt, ws.inv, ws.mqk);
}

__device__ __forceinline__ uint32_t pair(float a, float b) {
  __nv_bfloat162 x = __floats2bfloat162_rn(a, b);
  return reinterpret_cast<uint32_t&>(x);
}
__device__ __forceinline__ uint32_t transpose(uint32_t x) {
  uint32_t y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4],
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// C fragments are two key positions x two adjacent value columns per lane.
// Split the FP32 operand into three BF16 components, preserving ~24 bits.
__device__ __forceinline__ void split_b(const float (&c)[4],
                                        uint32_t (&b)[kComponents][2]) {
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    float x = c[2 * i], y = c[2 * i + 1];
#pragma unroll
    for (int p = 0; p < kComponents; ++p) {
      b[p][i] = transpose(pair(x, y));
      x -= __bfloat162float(__float2bfloat16_rn(x));
      y -= __bfloat162float(__float2bfloat16_rn(y));
    }
  }
}
template <typename Layout>
__device__ __forceinline__ void load_normal(uint32_t (&a)[4],
                                            const __nv_bfloat16* src, int col,
                                            int lane) {
  const int r = lane % 8 + ((lane / 8) % 2) * 8;
  const int c = col + (lane / 16) * 8;
  const uint32_t addr =
      __cvta_generic_to_shared(src + Layout{}(make_coord(r, c)));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}
__device__ __forceinline__ void load_transposed(uint32_t (&a)[4],
                                                const __nv_bfloat16* src,
                                                int key, int lane) {
  const int t = lane % 8 + (lane / 16) * 8;
  const int k = key + ((lane / 8) % 2) * 8;
  const uint32_t addr = __cvta_generic_to_shared(
      src + K1Layouts<128>::MMALayout{}(make_coord(t, k)));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(addr));
}
struct Shared {
  __nv_bfloat16 w[16 * 128];
  __nv_bfloat16 kg[128 * 16];
  float gate[128];
  __nv_bfloat16 inv[256];
  float beta[16];
};

__device__ __forceinline__ void copy16(void* dst, const void* src) {
  uint32_t addr = __cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr),
               "l"(src));
}
__device__ __forceinline__ void preload(
    Shared& s, const __nv_bfloat16* kd, const __nv_bfloat16* kr,
    const float* gt, const __nv_bfloat16* inv, const __nv_bfloat16* beta,
    int heads, int head, int bos, int eos, int tile, int64_t ws_idx, int tid) {
#pragma unroll
  for (int i = tid * 8; i < 2048; i += 128 * 8) {
    copy16(s.w + i, kd + ws_idx * 2048 + i);
    copy16(s.kg + i, kr + ws_idx * 2048 + i);
  }
  if (tid < 32) copy16(s.inv + tid * 8, inv + ws_idx * 256 + tid * 8);
  asm volatile("cp.async.commit_group;");
  s.gate[tid] = gt[ws_idx * 128 + tid];
  if (tid < 16) {
    const int token = bos + tile * 16 + tid;
    const float x = token < eos
                        ? __bfloat162float(beta[int64_t(token) * heads + head])
                        : 0.f;
    float th;
    asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(x * .5f));
    s.beta[tid] = token < eos
                      ? __bfloat162float(__float2bfloat16_rn(th * .5f + .5f))
                      : 0.f;
  }
}
__global__ __launch_bounds__(128) void kcp_transition_kernel(
    const uint8_t* workspace, const __nv_bfloat16* beta, const int32_t* cu,
    const int32_t* offsets, float* out, int heads, int tokens, int sequences) {
  __shared__ __align__(128) Shared buffers[2];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int group = lane / 4, thread = lane % 4;
  const int head = blockIdx.y, seq = blockIdx.z / kSplits,
            part = blockIdx.z % kSplits;
  const int first_col = blockIdx.x * 64 + warp * 16;
  const int bos = cu[seq], eos = cu[seq + 1], first_tile = offsets[seq];
  const int total_tiles = (tokens + 15) / 16 + sequences;
  const int64_t nht = int64_t(heads) * total_tiles;
  Workspace ws(const_cast<uint8_t*>(workspace), nht);
  const auto* ws_kd = reinterpret_cast<const __nv_bfloat16*>(ws.kd);
  const auto* ws_kr = reinterpret_cast<const __nv_bfloat16*>(ws.kr);
  const auto* ws_gt = ws.gt;
  const auto* ws_inv = reinterpret_cast<const __nv_bfloat16*>(ws.inv);
  float state[8][2][4];
#pragma unroll
  for (int m = 0; m < 8; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = m * 16 + group + (e / 2) * 8;
        const int col = first_col + n * 8 + thread * 2 + e % 2;
        state[m][n][e] = key == col ? 1.f : 0.f;
      }
    }
  }
  const int num_tiles = (eos - bos + 15) / 16;
  const int begin = num_tiles * part / kSplits,
            end = num_tiles * (part + 1) / kSplits;
  if (begin < end)
    preload(buffers[begin % 2], ws_kd, ws_kr, ws_gt, ws_inv, beta, heads, head,
            bos, eos, begin, int64_t(head) * total_tiles + first_tile + begin,
            tid);
  for (int tile = begin; tile < end; ++tile) {
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    Shared& s = buffers[tile % 2];
    if (tile + 1 < end)
      preload(buffers[(tile + 1) % 2], ws_kd, ws_kr, ws_gt, ws_inv, beta, heads,
              head, bos, eos, tile + 1,
              int64_t(head) * total_tiles + first_tile + tile + 1, tid);
    float projected[2][4] = {};
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      uint32_t a[4];
      load_normal<K1Layouts<128>::MMALayout>(a, s.w, m * 16, lane);
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        uint32_t b[kComponents][2];
        split_b(state[m][n], b);
#pragma unroll
        for (int p = kComponents - 1; p >= 0; --p)
          mma(projected[n], a, b[p][0], b[p][1]);
      }
    }
    uint32_t inv_a[4];
    load_normal<K1Layouts<128>::LMLayout>(inv_a, s.inv, 0, lane);
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) projected[n][e] *= s.beta[group + e / 2 * 8];
      uint32_t b[kComponents][2];
      split_b(projected[n], b);
      float updated[4] = {};
#pragma unroll
      for (int p = kComponents - 1; p >= 0; --p)
        mma(updated, inv_a, b[p][0], b[p][1]);
#pragma unroll
      for (int e = 0; e < 4; ++e) projected[n][e] = updated[e];
    }
    uint32_t p_b[2][kComponents][2];
#pragma unroll
    for (int n = 0; n < 2; ++n) split_b(projected[n], p_b[n]);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      uint32_t a[4];
      load_transposed(a, s.kg, m * 16, lane);
      const float g0 = s.gate[m * 16 + group], g1 = s.gate[m * 16 + group + 8];
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        float update[4] = {};
#pragma unroll
        for (int p = kComponents - 1; p >= 0; --p)
          mma(update, a, p_b[n][p][0], p_b[n][p][1]);
#pragma unroll
        for (int e = 0; e < 4; ++e)
          state[m][n][e] = fmaf(e < 2 ? g0 : g1, state[m][n][e], -update[e]);
      }
    }
    __syncthreads();
  }
  const int64_t dst =
      (int64_t(seq * kSplits + part) * heads + head) * 128 * 128;
#pragma unroll
  for (int m = 0; m < 8; ++m) {
#pragma unroll
    for (int n = 0; n < 2; ++n) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        int key = m * 16 + group + (e / 2) * 8;
        int col = first_col + n * 8 + thread * 2 + e % 2;
        out[dst + key * 128 + col] = state[m][n][e];
      }
    }
  }
}

void kcp_transition(const TorchTensor& workspace, const TorchTensor& beta,
                    const TorchTensor& cu, const TorchTensor& offsets,
                    const TorchTensor& partial) {
  check_inputs(workspace, beta, cu);
  const int device = beta.get_device_index();
  const int heads = beta.size(2), segments = cu.numel() - 1;
  check_cuda(offsets, ScalarType::Int, device);
  STD_TORCH_CHECK(offsets.dim() == 1 && offsets.numel() == cu.numel(),
                  "chunk_offsets must match cu_seqlens");
  check_state(partial, segments * kSplits, heads, device);
  torch::stable::accelerator::DeviceGuard guard(device);
  kcp_transition_kernel<<<dim3(2, heads, segments * kSplits), 128, 0,
                          get_current_cuda_stream(device)>>>(
      static_cast<const uint8_t*>(workspace.const_data_ptr()),
      static_cast<const __nv_bfloat16*>(beta.const_data_ptr()),
      static_cast<const int32_t*>(cu.const_data_ptr()),
      static_cast<const int32_t*>(offsets.const_data_ptr()),
      static_cast<float*>(partial.mutable_data_ptr()), heads, beta.size(1),
      segments);
  check_launch(cudaGetLastError());
}

void kcp_scan(const TorchTensor& v, const TorchTensor& beta,
              const TorchTensor& workspace, const TorchTensor& initial,
              const TorchTensor& cu, const TorchTensor& out,
              const TorchTensor& final) {
  const int tiles = check_inputs(workspace, beta, cu);
  const int device = beta.get_device_index();
  const int tokens = beta.size(1), heads = beta.size(2);
  const int segments = cu.numel() - 1;
  check_cuda(v, ScalarType::BFloat16, device);
  check_cuda(out, ScalarType::BFloat16, device);
  STD_TORCH_CHECK(v.dim() == 4 && v.size(0) == 1 && v.size(1) == tokens &&
                      v.size(2) == heads && v.size(3) == 128 &&
                      out.sizes() == v.sizes(),
                  "v and out must be [1, tokens, heads, 128]");
  check_state(initial, segments, heads, device);
  check_state(final, segments, heads, device);
  torch::stable::accelerator::DeviceGuard guard(device);
  const auto stream = get_current_cuda_stream(device);
  auto beta_t = torch::stable::contiguous(torch::stable::transpose(beta, 1, 2));
  if (2 * heads * segments <= get_device_prop()->multiProcessorCount) {
    launch_scan<64>(v, beta_t, workspace, initial, cu, out, final, tiles,
                    stream);
  } else {
    launch_scan<128>(v, beta_t, workspace, initial, cu, out, final, tiles,
                     stream);
  }
  check_launch(cudaGetLastError());
}

}  // namespace flashkda_kcp
