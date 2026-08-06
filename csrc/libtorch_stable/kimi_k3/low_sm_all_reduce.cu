// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Low-occupancy, two-shot NVLS all-reduce for Kimi K3 TP8 overlap schedules.
// The input and semaphore tensors must both live in multicast-bound symmetric
// memory. The kernel reduces one rank-owned shard and multicasts the result
// back in place, so no output copy is required.
// Adapted from SGLang's Apache-2.0 K3 pull all-reduce:
// https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/jit/csrc/kimi_k3/comm/ar_fusion.cuh

#include "../torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>

namespace vllm::kimi_k3 {

constexpr uint32_t kBlockSize = 512;
constexpr uint32_t kNumBlocks = 4;
constexpr uint32_t kUnroll = 8;
constexpr uint32_t kSemaphoreBytes = 128;

struct alignas(kSemaphoreBytes) Semaphore {
  uint32_t flag;
  uint32_t counter;
  uint32_t padding[(kSemaphoreBytes - 2 * sizeof(uint32_t)) / sizeof(uint32_t)];
};
static_assert(sizeof(Semaphore) == kSemaphoreBytes);

struct PullParams {
  uint8_t* input_mc;
  Semaphore* semaphore;
  uint8_t* semaphore_mc;
  uint32_t rank;
  uint32_t world_size;
  uint32_t num_vecs;
};

__device__ __forceinline__ uint32_t load_relaxed_sys(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ uint32_t load_acquire_sys(uint32_t const* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ void multicast_arrive_relaxed(uint32_t* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], 1;"
               :
               : "l"(address)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void multicast_arrive_release(uint32_t* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], 1;"
               :
               : "l"(address)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ uint4
multicast_load_reduce_bf16x8(uint8_t const* address, uint32_t vec_offset) {
  uint4 value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  address += static_cast<uint64_t>(vec_offset) * sizeof(uint4);
  asm volatile(
      "multimem.ld_reduce.weak.add.acc::f32.v4.bf16x2 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(address));
#else
  asm volatile("trap;");
#endif
  return value;
}

__device__ __forceinline__ void multicast_store(uint8_t* address,
                                                uint32_t vec_offset,
                                                uint4 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  address += static_cast<uint64_t>(vec_offset) * sizeof(uint4);
  float4 const packed = *reinterpret_cast<float4 const*>(&value);
  asm volatile("multimem.st.weak.v4.f32 [%4], {%0, %1, %2, %3};"
               :
               : "f"(packed.x), "f"(packed.y), "f"(packed.z), "f"(packed.w),
                 "l"(address));
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ uint32_t barrier_enter(PullParams const& params) {
  uint32_t current = 0;
  if (threadIdx.x == 0) {
    Semaphore* semaphore = params.semaphore + blockIdx.x;
    uint32_t const reserved =
        atomicAdd(&semaphore->counter, 2 * params.world_size);
    current = reserved + params.world_size;
    auto* multicast_flag = reinterpret_cast<uint32_t*>(
        params.semaphore_mc + blockIdx.x * sizeof(Semaphore));
    multicast_arrive_relaxed(multicast_flag);
    while (load_relaxed_sys(&semaphore->flag) - reserved < params.world_size) {
    }
  }
  __syncthreads();
  return current;
}

__device__ __forceinline__ void barrier_exit(PullParams const& params,
                                             uint32_t current) {
  __syncthreads();
  if (threadIdx.x == 0) {
    Semaphore* semaphore = params.semaphore + blockIdx.x;
    auto* multicast_flag = reinterpret_cast<uint32_t*>(
        params.semaphore_mc + blockIdx.x * sizeof(Semaphore));
    multicast_arrive_release(multicast_flag);
    while (load_acquire_sys(&semaphore->flag) - current < params.world_size) {
    }
  }
}

template <uint32_t Width>
__device__ __forceinline__ void pull_reduce_pass(uint32_t& vec,
                                                 uint32_t num_vecs,
                                                 uint32_t step,
                                                 uint8_t* input_mc) {
  for (; vec + (Width - 1) * step < num_vecs; vec += Width * step) {
    uint4 values[Width];
#pragma unroll
    for (uint32_t i = 0; i < Width; ++i) {
      values[i] = multicast_load_reduce_bf16x8(input_mc, vec + i * step);
    }
#pragma unroll
    for (uint32_t i = 0; i < Width; ++i) {
      multicast_store(input_mc, vec + i * step, values[i]);
    }
  }
  if constexpr (Width > 1) {
    pull_reduce_pass<Width / 2>(vec, num_vecs, step, input_mc);
  }
}

__global__ __launch_bounds__(kBlockSize, 1) void low_sm_all_reduce_kernel(
    PullParams params) {
  uint32_t const barrier_window = barrier_enter(params);

  uint32_t const avg_vecs = params.num_vecs / params.world_size;
  uint32_t const rem_vecs = params.num_vecs % params.world_size;
  uint32_t const shard_offset =
      avg_vecs * params.rank +
      (params.rank < rem_vecs ? params.rank : rem_vecs);
  uint32_t const shard_vecs = avg_vecs + (params.rank < rem_vecs ? 1U : 0U);
  uint8_t* shard_mc =
      params.input_mc + static_cast<uint64_t>(shard_offset) * sizeof(uint4);

  uint32_t const step = kBlockSize * gridDim.x;
  uint32_t vec = blockIdx.x * kBlockSize + threadIdx.x;
  pull_reduce_pass<kUnroll>(vec, shard_vecs, step, shard_mc);

  barrier_exit(params, barrier_window);
}

void low_sm_all_reduce(torch::stable::Tensor& input, int64_t input_mc_ptr,
                       torch::stable::Tensor& semaphore,
                       int64_t semaphore_mc_ptr, int64_t rank,
                       int64_t world_size) {
  int const device = input.get_device_index();
  torch::stable::accelerator::DeviceGuard const device_guard(device);
  cudaDeviceProp const* properties = get_device_prop();

  STD_TORCH_CHECK(properties->major == 10 &&
                      (properties->minor == 0 || properties->minor == 3),
                  "Kimi K3 low-SM all-reduce requires SM100 or SM103");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Kimi K3 low-SM all-reduce requires a bfloat16 input");
  STD_TORCH_CHECK(input.is_contiguous(),
                  "Kimi K3 low-SM all-reduce requires contiguous input");
  STD_TORCH_CHECK(input.numel() > 0 && input.numel() % 8 == 0,
                  "Kimi K3 low-SM all-reduce requires a positive input "
                  "element count divisible by 8");
  STD_TORCH_CHECK(input.numel() / 8 < (int64_t{1} << 31),
                  "Kimi K3 low-SM all-reduce input is too large");
  STD_TORCH_CHECK(input_mc_ptr != 0,
                  "Kimi K3 low-SM all-reduce requires an input multicast "
                  "address");
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(input.mutable_data_ptr()) % alignof(uint4) ==
              0 &&
          static_cast<uintptr_t>(input_mc_ptr) % alignof(uint4) == 0,
      "Kimi K3 low-SM all-reduce requires 16-byte-aligned input addresses");

  STD_TORCH_CHECK(
      semaphore.scalar_type() == torch::headeronly::ScalarType::Byte &&
          semaphore.is_contiguous(),
      "Kimi K3 low-SM all-reduce requires a contiguous uint8 semaphore");
  STD_TORCH_CHECK(semaphore.get_device_index() == device,
                  "Kimi K3 low-SM all-reduce requires input and semaphore on "
                  "the same device");
  STD_TORCH_CHECK(semaphore_mc_ptr != 0,
                  "Kimi K3 low-SM all-reduce requires a semaphore multicast "
                  "address");
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(semaphore.mutable_data_ptr()) %
                  alignof(Semaphore) ==
              0 &&
          static_cast<uintptr_t>(semaphore_mc_ptr) % alignof(Semaphore) == 0,
      "Kimi K3 low-SM all-reduce requires 128-byte-aligned semaphore "
      "addresses");
  STD_TORCH_CHECK(world_size == 8,
                  "Kimi K3 low-SM all-reduce is specialized for TP8");
  STD_TORCH_CHECK(rank >= 0 && rank < world_size,
                  "Kimi K3 low-SM all-reduce received an invalid rank");
  STD_TORCH_CHECK(semaphore.numel() >= kNumBlocks * kSemaphoreBytes,
                  "Kimi K3 low-SM all-reduce semaphore storage is too small");

  PullParams params{};
  params.input_mc =
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(input_mc_ptr));
  params.semaphore = static_cast<Semaphore*>(semaphore.mutable_data_ptr());
  params.semaphore_mc =
      reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(semaphore_mc_ptr));
  params.rank = static_cast<uint32_t>(rank);
  params.world_size = static_cast<uint32_t>(world_size);
  params.num_vecs = static_cast<uint32_t>(input.numel() / 8);

  cudaStream_t const stream = get_current_cuda_stream(device);
  low_sm_all_reduce_kernel<<<kNumBlocks, kBlockSize, 0, stream>>>(params);
  cudaError_t const error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "Kimi K3 low-SM all-reduce launch failed: ", cudaGetErrorString(error));
}

}  // namespace vllm::kimi_k3

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "kimi_k3_low_sm_all_reduce_(Tensor! input, int input_mc_ptr, "
      "Tensor! semaphore, int semaphore_mc_ptr, int rank, int world_size) "
      "-> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("kimi_k3_low_sm_all_reduce_",
         TORCH_BOX(&vllm::kimi_k3::low_sm_all_reduce));
}
