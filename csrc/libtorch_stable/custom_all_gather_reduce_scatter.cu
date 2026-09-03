#include "torch_utils.h"

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "custom_all_reduce.cuh"
#include "custom_all_gather_reduce_scatter.cuh"

namespace vllm {

void CustomAllreduce::allgather(cudaStream_t stream, void* input, void* output,
                                int size_bytes, int threads, int block_limit) {
  if (size_bytes % sizeof(CopyPack) != 0)
    throw std::runtime_error(
        "custom allgather requires input byte size to be a multiple of " +
        std::to_string(sizeof(CopyPack)));

  auto ptrs = buffers_.at(input);
  int size_per_rank = size_bytes / sizeof(CopyPack);
  int total_size = size_per_rank * world_size_;
  int blocks = std::min(block_limit, (total_size + threads - 1) / threads);

#define AG_CASE(ngpus)                                                   \
  case ngpus:                                                            \
    cross_device_all_gather<ngpus><<<blocks, threads, 0, stream>>>(      \
        ptrs, sg_, self_sg_, reinterpret_cast<CopyPack*>(output), rank_, \
        size_per_rank);                                                  \
    break;

  switch (world_size_) {
    AG_CASE(2)
    AG_CASE(4)
    AG_CASE(6)
    AG_CASE(8)
    default:
      throw std::runtime_error(
          "custom allgather only supports num gpus in (2,4,6,8)");
  }
#undef AG_CASE
}

template <typename T>
void CustomAllreduce::mnnvl_lamport_allgather(cudaStream_t stream, T* input,
                                              T* output, void* local_buffer,
                                              void* multicast_buffer,
                                              uint32_t* epochs, int size_bytes,
                                              int stage_size_bytes) {
  if (size_bytes % sizeof(typename packed_t<T>::P) != 0 ||
      stage_size_bytes % sizeof(typename packed_t<T>::P) != 0)
    throw std::runtime_error(
        "MNNVL Lamport allgather requires 16-byte aligned sizes");

  auto ptrs = buffers_.at(local_buffer);
  int size_per_rank = size_bytes / sizeof(typename packed_t<T>::P);
  int stage_size = stage_size_bytes / sizeof(typename packed_t<T>::P);
  int blocks =
      (size_per_rank + kMnnvlLamportAgThreads - 1) / kMnnvlLamportAgThreads;

#if !defined(USE_ROCM) && CUDA_VERSION >= 12000
  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{.gridDim = dim3(blocks),
                            .blockDim = dim3(kMnnvlLamportAgThreads),
                            .dynamicSmemBytes = 0,
                            .stream = stream,
                            .attrs = attributes,
                            .numAttrs = 1};
  #define MNNVL_LAMPORT_AG_LAUNCH(ngpus)                                       \
    CUDACHECK(cudaLaunchKernelEx(&config, &mnnvl_lamport_all_gather<T, ngpus>, \
                                 ptrs, input, output,                          \
                                 reinterpret_cast<T*>(multicast_buffer),       \
                                 epochs, rank_, size_per_rank, stage_size))
#else
  #define MNNVL_LAMPORT_AG_LAUNCH(ngpus)                                 \
    mnnvl_lamport_all_gather<T, ngpus>                                   \
        <<<blocks, kMnnvlLamportAgThreads, 0, stream>>>(                 \
            ptrs, input, output, reinterpret_cast<T*>(multicast_buffer), \
            epochs, rank_, size_per_rank, stage_size)
#endif

#define MNNVL_LAMPORT_AG_CASE(ngpus) \
  case ngpus:                        \
    MNNVL_LAMPORT_AG_LAUNCH(ngpus);  \
    break;

  switch (world_size_) {
    MNNVL_LAMPORT_AG_CASE(2)
    MNNVL_LAMPORT_AG_CASE(4)
    MNNVL_LAMPORT_AG_CASE(6)
    MNNVL_LAMPORT_AG_CASE(8)
    MNNVL_LAMPORT_AG_CASE(16)
    default:
      throw std::runtime_error(
          "MNNVL Lamport allgather only supports num gpus in (2,4,6,8,16)");
  }
#undef MNNVL_LAMPORT_AG_CASE
#undef MNNVL_LAMPORT_AG_LAUNCH
}

template <typename T>
void CustomAllreduce::reduce_scatter(cudaStream_t stream, T* input, T* output,
                                     int size, int threads, int block_limit) {
  auto packed_size = packed_t<T>::P::size;
  if (size % (packed_size * world_size_) != 0)
    throw std::runtime_error(
        "custom reduce-scatter requires each output shard byte size to be "
        "a multiple of 16");

  auto ptrs = buffers_.at(input);
  int size_per_rank = size / packed_size / world_size_;
  int blocks = std::min(block_limit, (size_per_rank + threads - 1) / threads);

#define RS_CASE(ngpus)                                                     \
  case ngpus:                                                              \
    cross_device_reduce_scatter<T, ngpus><<<blocks, threads, 0, stream>>>( \
        ptrs, sg_, self_sg_, output, rank_, size_per_rank);                \
    break;

  switch (world_size_) {
    RS_CASE(2)
    RS_CASE(4)
    RS_CASE(6)
    RS_CASE(8)
    default:
      throw std::runtime_error(
          "custom reduce-scatter only supports num gpus in (2,4,6,8)");
  }
#undef RS_CASE
}

template <typename T>
void CustomAllreduce::mnnvl_lamport_reduce_scatter(cudaStream_t stream,
                                                   T* input, T* output,
                                                   void* local_buffer,
                                                   uint32_t* epochs, int size,
                                                   int stage_size_bytes) {
  auto packed_size = packed_t<T>::P::size;
  if (size % (packed_size * world_size_) != 0 ||
      stage_size_bytes % sizeof(typename packed_t<T>::P) != 0)
    throw std::runtime_error(
        "MNNVL Lamport reduce-scatter requires 16-byte aligned sizes");

  auto ptrs = buffers_.at(local_buffer);
  int size_per_rank = size / packed_size / world_size_;
  int stage_size = stage_size_bytes / sizeof(typename packed_t<T>::P);
  int blocks_per_rank =
      (size_per_rank + kMnnvlLamportRsThreads - 1) / kMnnvlLamportRsThreads;
  int blocks = blocks_per_rank * world_size_;

#if !defined(USE_ROCM) && CUDA_VERSION >= 12000
  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{.gridDim = dim3(blocks),
                            .blockDim = dim3(kMnnvlLamportRsThreads),
                            .dynamicSmemBytes = 0,
                            .stream = stream,
                            .attrs = attributes,
                            .numAttrs = 1};
  #define MNNVL_LAMPORT_RS_LAUNCH(ngpus)                                      \
    CUDACHECK(cudaLaunchKernelEx(                                             \
        &config, &mnnvl_lamport_reduce_scatter_kernel<T, ngpus>, ptrs, input, \
        output, epochs, rank_, size_per_rank, stage_size))
#else
  #define MNNVL_LAMPORT_RS_LAUNCH(ngpus)                 \
    mnnvl_lamport_reduce_scatter_kernel<T, ngpus>        \
        <<<blocks, kMnnvlLamportRsThreads, 0, stream>>>( \
            ptrs, input, output, epochs, rank_, size_per_rank, stage_size)
#endif

#define MNNVL_LAMPORT_RS_CASE(ngpus) \
  case ngpus:                        \
    MNNVL_LAMPORT_RS_LAUNCH(ngpus);  \
    break;

  switch (world_size_) {
    MNNVL_LAMPORT_RS_CASE(2)
    MNNVL_LAMPORT_RS_CASE(4)
    MNNVL_LAMPORT_RS_CASE(6)
    MNNVL_LAMPORT_RS_CASE(8)
    MNNVL_LAMPORT_RS_CASE(16)
    default:
      throw std::runtime_error(
          "MNNVL Lamport reduce-scatter only supports num gpus in "
          "(2,4,6,8,16)");
  }
#undef MNNVL_LAMPORT_RS_CASE
#undef MNNVL_LAMPORT_RS_LAUNCH
}

template <typename T>
void CustomAllreduce::mnnvl_multimem_reduce_scatter(
    cudaStream_t stream, const T* multicast_input, T* output,
    void* local_buffer, uint64_t signal_offset, int size, int block_limit) {
  if (size <= 0)
    throw std::runtime_error(
        "MNNVL multimem reduce-scatter requires a non-empty input");
  int size_bytes = size * sizeof(T);
  if (world_size_ != 8)
    throw std::runtime_error(
        "MNNVL multimem reduce-scatter currently supports TP8 only");
  if (size_bytes % (kMnnvlMultimemRsVectorBytes * world_size_) != 0)
    throw std::runtime_error(
        "MNNVL multimem reduce-scatter requires each output shard byte size "
        "to be a multiple of 16");
  if (block_limit <= 0 || block_limit > kMnnvlMultimemRsBlockLimit)
    throw std::runtime_error(
        "MNNVL multimem reduce-scatter block limit is out of range");

  auto it = buffers_.find(local_buffer);
  if (it == buffers_.end())
    throw std::runtime_error(
        "MNNVL multimem reduce-scatter symmetric buffer is not registered");
  auto* peer_buffers = it->second;

  int packs_per_rank = size_bytes / kMnnvlMultimemRsVectorBytes / world_size_;
  int packs_per_block = kMnnvlMultimemRsThreads * kMnnvlMultimemRsUnroll;
  int blocks = std::min(
      block_limit, (packs_per_rank + packs_per_block - 1) / packs_per_block);

  mnnvl_multimem_reduce_scatter_kernel<T, 8>
      <<<blocks, kMnnvlMultimemRsThreads, 0, stream>>>(
          multicast_input, output, peer_buffers, signal_offset, rank_,
          packs_per_rank);
  STD_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vllm

using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

bool _is_weak_contiguous(torch::stable::Tensor& t);

void custom_all_gather(fptr_t _fa, torch::stable::Tensor& inp,
                       torch::stable::Tensor& out, fptr_t _reg_buffer,
                       int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.numel() * fa->world_size_) == (out.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  STD_TORCH_CHECK(reg_buffer != nullptr);
  STD_TORCH_CHECK((input_size) <= (reg_buffer_sz_bytes));
  STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                 cudaMemcpyDeviceToDevice, stream));
  fa->allgather(stream, reg_buffer, out.mutable_data_ptr(), input_size);
}

void mnnvl_lamport_all_gather(fptr_t _fa, torch::stable::Tensor& inp,
                              torch::stable::Tensor& out, fptr_t _local_buffer,
                              fptr_t _multicast_buffer, fptr_t _epoch_buffer,
                              int64_t stage_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.numel() * fa->world_size_) == (out.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  STD_TORCH_CHECK((input_size * fa->world_size_) <= stage_sz_bytes);
  auto local_buffer = reinterpret_cast<void*>(_local_buffer);
  auto multicast_buffer = reinterpret_cast<void*>(_multicast_buffer);
  auto epochs = reinterpret_cast<uint32_t*>(_epoch_buffer);
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->mnnvl_lamport_allgather<float>(
          stream, reinterpret_cast<float*>(inp.mutable_data_ptr()),
          reinterpret_cast<float*>(out.mutable_data_ptr()), local_buffer,
          multicast_buffer, epochs, input_size, stage_sz_bytes);
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->mnnvl_lamport_allgather<half>(
          stream, reinterpret_cast<half*>(inp.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()), local_buffer,
          multicast_buffer, epochs, input_size, stage_sz_bytes);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->mnnvl_lamport_allgather<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.mutable_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), local_buffer,
          multicast_buffer, epochs, input_size, stage_sz_bytes);
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "MNNVL Lamport allgather only supports float32, float16 and "
          "bfloat16");
  }
}

void custom_reduce_scatter(fptr_t _fa, torch::stable::Tensor& inp,
                           torch::stable::Tensor& out, fptr_t _reg_buffer,
                           int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((out.numel() * fa->world_size_) == (inp.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  STD_TORCH_CHECK(reg_buffer != nullptr);
  STD_TORCH_CHECK((input_size) <= (reg_buffer_sz_bytes));
  STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                 cudaMemcpyDeviceToDevice, stream));
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->reduce_scatter<float>(
          stream, reinterpret_cast<float*>(reg_buffer),
          reinterpret_cast<float*>(out.mutable_data_ptr()), inp.numel());
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->reduce_scatter<half>(stream, reinterpret_cast<half*>(reg_buffer),
                               reinterpret_cast<half*>(out.mutable_data_ptr()),
                               inp.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->reduce_scatter<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), inp.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom reduce-scatter only supports float32, float16 and bfloat16");
  }
}

void mnnvl_lamport_reduce_scatter(fptr_t _fa, torch::stable::Tensor& inp,
                                  torch::stable::Tensor& out,
                                  fptr_t _local_buffer, fptr_t _epoch_buffer,
                                  int64_t stage_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((out.numel() * fa->world_size_) == (inp.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  STD_TORCH_CHECK(input_size <= stage_sz_bytes);
  auto local_buffer = reinterpret_cast<void*>(_local_buffer);
  auto epochs = reinterpret_cast<uint32_t*>(_epoch_buffer);
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->mnnvl_lamport_reduce_scatter<float>(
          stream, reinterpret_cast<float*>(inp.mutable_data_ptr()),
          reinterpret_cast<float*>(out.mutable_data_ptr()), local_buffer,
          epochs, inp.numel(), stage_sz_bytes);
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->mnnvl_lamport_reduce_scatter<half>(
          stream, reinterpret_cast<half*>(inp.mutable_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()), local_buffer, epochs,
          inp.numel(), stage_sz_bytes);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->mnnvl_lamport_reduce_scatter<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.mutable_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), local_buffer,
          epochs, inp.numel(), stage_sz_bytes);
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "MNNVL Lamport reduce-scatter only supports float32, float16 and "
          "bfloat16");
  }
}

void mnnvl_multimem_reduce_scatter(fptr_t _fa, torch::stable::Tensor& inp,
                                   torch::stable::Tensor& out,
                                   fptr_t _local_buffer,
                                   fptr_t _multicast_buffer,
                                   int64_t stage_sz_bytes,
                                   int64_t block_limit) {
  STD_TORCH_CHECK(_fa != 0,
                  "MNNVL multimem reduce-scatter requires a communicator");
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(inp.is_cuda() && out.is_cuda(),
                  "MNNVL multimem reduce-scatter requires CUDA tensors");
  STD_TORCH_CHECK(
      inp.get_device_index() == out.get_device_index(),
      "MNNVL multimem reduce-scatter requires input and output on the same "
      "device");
  STD_TORCH_CHECK(
      inp.get_device_index() == fa->device_index_,
      "MNNVL multimem reduce-scatter tensors must be on the communicator "
      "device");
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  const cudaDeviceProp* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 10 &&
                      (properties->minor == 0 || properties->minor == 3),
                  "MNNVL multimem reduce-scatter requires SM100 or SM103");
#if defined(USE_ROCM) || CUDA_VERSION < 12020
  STD_TORCH_CHECK(false,
                  "MNNVL multimem reduce-scatter requires CUDA 12.2 or newer");
#endif

  STD_TORCH_CHECK(
      inp.scalar_type() == out.scalar_type(),
      "MNNVL multimem reduce-scatter requires matching input and output "
      "dtypes");
  STD_TORCH_CHECK(
      inp.scalar_type() == torch::headeronly::ScalarType::Float ||
          inp.scalar_type() == torch::headeronly::ScalarType::Half ||
          inp.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "MNNVL multimem reduce-scatter only supports float32, float16 and "
      "bfloat16");
  STD_TORCH_CHECK(fa->world_size_ == 8,
                  "MNNVL multimem reduce-scatter currently supports TP8 only");
  STD_TORCH_CHECK(inp.numel() > 0 && inp.numel() % fa->world_size_ == 0 &&
                      out.numel() == inp.numel() / fa->world_size_,
                  "MNNVL multimem reduce-scatter requires output.numel() == "
                  "input.numel() / 8");
  STD_TORCH_CHECK(
      _is_weak_contiguous(inp) && _is_weak_contiguous(out),
      "MNNVL multimem reduce-scatter requires weak-contiguous tensors");
  STD_TORCH_CHECK(
      inp.numel() <= std::numeric_limits<int>::max() / inp.element_size(),
      "MNNVL multimem reduce-scatter input is too large for the kernel ABI");
  const int64_t input_size = inp.numel() * inp.element_size();
  STD_TORCH_CHECK(
      input_size % (vllm::kMnnvlMultimemRsVectorBytes * fa->world_size_) == 0,
      "MNNVL multimem reduce-scatter requires each output shard byte size "
      "to be a multiple of 16");
  STD_TORCH_CHECK(
      stage_sz_bytes >= input_size &&
          stage_sz_bytes <=
              std::numeric_limits<int64_t>::max() - (alignof(vllm::Signal) - 1),
      "MNNVL multimem reduce-scatter received an invalid stage size");
  STD_TORCH_CHECK(
      block_limit > 0 && block_limit <= vllm::kMnnvlMultimemRsBlockLimit,
      "MNNVL multimem reduce-scatter block limit is out of range");
  auto local_buffer = reinterpret_cast<void*>(_local_buffer);
  auto multicast_buffer = reinterpret_cast<void*>(_multicast_buffer);
  STD_TORCH_CHECK(
      local_buffer != nullptr && multicast_buffer != nullptr,
      "MNNVL multimem reduce-scatter requires local and multicast buffers");
  constexpr uintptr_t vector_alignment = vllm::kMnnvlMultimemRsVectorBytes;
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(multicast_buffer) % vector_alignment == 0 &&
          reinterpret_cast<uintptr_t>(out.mutable_data_ptr()) %
                  vector_alignment ==
              0,
      "MNNVL multimem reduce-scatter requires 16-byte-aligned multicast and "
      "output addresses");
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(local_buffer) % alignof(vllm::Signal) == 0,
      "MNNVL multimem reduce-scatter requires a 128-byte-aligned symmetric "
      "buffer");
  STD_TORCH_CHECK(
      fa->buffers_.find(local_buffer) != fa->buffers_.end(),
      "MNNVL multimem reduce-scatter symmetric buffer is not registered");
  const uint64_t signal_offset =
      (static_cast<uint64_t>(stage_sz_bytes) + alignof(vllm::Signal) - 1) &
      ~(static_cast<uint64_t>(alignof(vllm::Signal)) - 1);
  STD_CUDA_CHECK(cudaMemcpyAsync(local_buffer, inp.const_data_ptr(), input_size,
                                 cudaMemcpyDeviceToDevice, stream));
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->mnnvl_multimem_reduce_scatter<float>(
          stream, reinterpret_cast<float*>(multicast_buffer),
          reinterpret_cast<float*>(out.mutable_data_ptr()), local_buffer,
          signal_offset, static_cast<int>(inp.numel()),
          static_cast<int>(block_limit));
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->mnnvl_multimem_reduce_scatter<half>(
          stream, reinterpret_cast<half*>(multicast_buffer),
          reinterpret_cast<half*>(out.mutable_data_ptr()), local_buffer,
          signal_offset, static_cast<int>(inp.numel()),
          static_cast<int>(block_limit));
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->mnnvl_multimem_reduce_scatter<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(multicast_buffer),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), local_buffer,
          signal_offset, static_cast<int>(inp.numel()),
          static_cast<int>(block_limit));
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "MNNVL multimem reduce-scatter only supports float32, float16 and "
          "bfloat16");
  }
}
