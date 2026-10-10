// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HY4 PCP prefill attention prologue.
//
// hy4_pcp_kv_norm_rope_quant_multicast: kv_a RMSNorm + k_pe RoPE + nvfp4_ds_mla
//   row (352 B) and indexer K LayerNorm + RoPE + UE8M0 FP8 row (128 + 4 B),
//   multicast to every rank's staging slot. Never waits on peers.
// hy4_pcp_kv_wait_cache_insert: wait for every rank, then copy the rank-major
//   rows into the paged MLA and indexer caches.
// hy4_q_rope_indexer_quant: q_pe RoPE (in place or into the 576-wide MLA
//   query) and indexer Q RoPE + UE8M0 FP8 + folded head weights; optionally
//   resets each token's top-k index row to -1 before the indexer's top-k.
//
// K path: one warp per token (TRT-LLM mlaKvNormRopeQuant). Lane l owns kv_c
// dims [16l, 16l + 16) (one nvfp4 block), k_pe pair (2l, 2l + 1) and indexer
// dims [4l, 4l + 4). Arithmetic and rounding match the replaced kernels
// (rms_norm, rotary_embedding, concat_and_cache_mla, _k_norm_rope,
// indexer_k_quant_and_cache, _fused_indexer_q_rope_quant).
//
// Sync: each multicast adds one to its source's counter on every rank; the
// wait for exchange E needs every counter >= E + 1, for any grid. Nothing is
// cleared. Slots alternate on E & 1; a rank reuses a slot only after its wait
// for E + 1, i.e. after every peer's insert of E.

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <optional>
#include <type_traits>

#include "../dcp_utils/dcp_direct_common.cuh"

namespace {

using vllm::direct_dcp::check_cuda_launch;
using vllm::direct_dcp::kSpinLimit;
using vllm::direct_dcp::load_acquire_system;
using vllm::direct_dcp::multimem_store_16;

using bf16 = __nv_bfloat16;

constexpr int kKvLora = 512;
constexpr int kRope = 64;
constexpr int kIdxDim = 128;
constexpr int kMlaRowBytes = 352;
constexpr int kRopeOff = 256;
constexpr int kSfOff = 320;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr unsigned kFull = 0xffffffffu;

// Must match FlashMLA's nvfp4_sf_byte() (see nvfp4_ds_mla_cache_kernels.cu).
__device__ __forceinline__ int nvfp4_sf_byte(int s) {
  return 8 * (s & 3) + (s >> 2);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(kFull, v, m);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, m));
  return v;
}

__device__ __forceinline__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float load_cos_sin(const void* cache, int64_t i,
                                              bool is_f32) {
  return is_f32 ? static_cast<const float*>(cache)[i]
                : __bfloat162float(static_cast<const bf16*>(cache)[i]);
}

// cache[i .. i + 4), i a multiple of 4.
__device__ __forceinline__ void load_cos_sin4(const void* cache, int64_t i,
                                              bool is_f32, float (&out)[4]) {
  if (is_f32) {
    float4 v =
        *reinterpret_cast<const float4*>(static_cast<const float*>(cache) + i);
    out[0] = v.x, out[1] = v.y, out[2] = v.z, out[3] = v.w;
  } else {
    uint2 v =
        *reinterpret_cast<const uint2*>(static_cast<const bf16*>(cache) + i);
    bf16 const* e = reinterpret_cast<bf16 const*>(&v);
#pragma unroll
    for (int k = 0; k < 4; ++k) out[k] = __bfloat162float(e[k]);
  }
}

__device__ __forceinline__ uint8_t to_e4m3(float v) {
  return static_cast<uint8_t>(
      __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

// 16 floats (already divided by the scale) -> 16 e2m1, low nibble = even.
__device__ __forceinline__ uint64_t e2m1x16(const float (&v)[16]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) ||         \
     defined(__CUDA_ARCH_SPECIFIC__))
  uint32_t lo, hi;
  asm volatile(
      "{\n.reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "mov.b32 %1, {b4, b5, b6, b7};\n}\n"
      : "=r"(lo), "=r"(hi)
      : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]),
        "f"(v[6]), "f"(v[7]), "f"(v[8]), "f"(v[9]), "f"(v[10]), "f"(v[11]),
        "f"(v[12]), "f"(v[13]), "f"(v[14]), "f"(v[15]));
  return (static_cast<uint64_t>(hi) << 32) | lo;
#else
  __trap();
  return 0;
#endif
}

__device__ __forceinline__ void multimem_store_b32(uint32_t* mc, uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(mc), "r"(v)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void multimem_red_add_release(uint32_t* mc,
                                                         uint32_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(mc),
               "r"(v)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

struct StagingLayout {
  int64_t slot_bytes;
  int64_t idx_off;    // indexer values region, [world * max_local, 128]
  int64_t scale_off;  // indexer scales region, [world * max_local] fp32
};

struct KvMulticastArgs {
  const bf16* kv;  // [T, 576]: kv_c (512) then k_pe (64)
  int64_t kv_stride;
  const bf16* idx;  // [T, 128] or null
  int64_t idx_stride;
  const int64_t* positions;
  const void* cos_sin;  // [max_pos, 64]: 32 cos then 32 sin
  int64_t cos_sin_stride;
  bool cos_sin_f32;
  const void* idx_cos_sin;
  int64_t idx_cos_sin_stride;
  bool idx_cos_sin_f32;
  const bf16* kv_weight;
  float kv_eps;
  const float* idx_weight;
  const float* idx_bias;
  float idx_eps;
  int num_tokens;
  uint8_t* staging;  // multicast (or local, for kMulticast = false) base
  uint32_t* signal;  // multicast counters [world]
  int32_t* state;    // [exchanges, insert CTAs done, multicast CTAs done]
  StagingLayout layout;
  int world_size;
  int rank;
  int max_local;
};

template <bool kMulticast>
__device__ __forceinline__ void put16(uint8_t* dst, uint4 v) {
  if constexpr (kMulticast) {
    multimem_store_16(reinterpret_cast<uint4*>(dst), v);
  } else {
    *reinterpret_cast<uint4*>(dst) = v;
  }
}

// One token's raw inputs for one lane, loaded a token ahead of its use.
struct KRaw {
  uint4 kv[2];  // kv_c dims [16l, 16l + 16)
  uint32_t pe;  // k_pe pair (2l, 2l + 1)
  uint2 idx;    // indexer K dims [4l, 4l + 4)
  int64_t pos;
};

__device__ __forceinline__ void load_raw(const KvMulticastArgs& a, int t,
                                         int lane, KRaw& r) {
  bf16 const* row = a.kv + static_cast<int64_t>(t) * a.kv_stride;
  uint4 const* rp = reinterpret_cast<uint4 const*>(row + 16 * lane);
  r.kv[0] = rp[0];
  r.kv[1] = rp[1];
  r.pe = *reinterpret_cast<uint32_t const*>(row + kKvLora + 2 * lane);
  if (a.idx != nullptr) {
    r.idx = *reinterpret_cast<uint2 const*>(
        a.idx + static_cast<int64_t>(t) * a.idx_stride + 4 * lane);
  }
  r.pos = a.positions[t];
}

template <int kW, bool kMulticast>
__global__ void __launch_bounds__(kW * 32, 1)
    hy4_pcp_kv_norm_rope_quant_multicast_kernel(KvMulticastArgs a) {
  __shared__ __align__(16) uint8_t mla_buf[kW][kMlaRowBytes];
  __shared__ __align__(16) uint8_t idx_buf[kW][kIdxDim];
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  // Row-invariant: one nvfp4 block of the kv norm weight per lane.
  float w[16];
  {
    uint4 const* wp = reinterpret_cast<uint4 const*>(a.kv_weight + 16 * lane);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      uint4 v = wp[h];
      bf16 const* e = reinterpret_cast<bf16 const*>(&v);
#pragma unroll
      for (int i = 0; i < 8; ++i) w[8 * h + i] = __bfloat162float(e[i]);
    }
  }
  float iw[4], ib[4];
  if (a.idx != nullptr) {
    float4 wv = reinterpret_cast<float4 const*>(a.idx_weight)[lane];
    float4 bv = reinterpret_cast<float4 const*>(a.idx_bias)[lane];
    iw[0] = wv.x, iw[1] = wv.y, iw[2] = wv.z, iw[3] = wv.w;
    ib[0] = bv.x, ib[1] = bv.y, ib[2] = bv.z, ib[3] = bv.w;
  }

  // The inputs come from the preceding GEMMs.
  pdl_wait();

  int const slot = a.state[0] & 1;
  uint8_t* const slot_base = a.staging + slot * a.layout.slot_bytes;
  int64_t const row0 = static_cast<int64_t>(a.rank) * a.max_local;
  int const stride = gridDim.x * kW;

  int t = blockIdx.x * kW + warp;
  KRaw cur;
  if (t < a.num_tokens) load_raw(a, t, lane, cur);
  for (; t < a.num_tokens; t += stride) {
    KRaw nxt;
    if (t + stride < a.num_tokens) load_raw(a, t + stride, lane, nxt);
    int64_t const pos = cur.pos;
    int64_t const out_row = row0 + t;
    float const c =
        load_cos_sin(a.cos_sin, pos * a.cos_sin_stride + lane, a.cos_sin_f32);
    float const sn = load_cos_sin(
        a.cos_sin, pos * a.cos_sin_stride + kRope / 2 + lane, a.cos_sin_f32);

    // kv_a RMSNorm (rms_norm_kernel: one rounding of x * s * w).
    float x[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      bf16 const* e = reinterpret_cast<bf16 const*>(&cur.kv[h]);
#pragma unroll
      for (int i = 0; i < 8; ++i) x[8 * h + i] = __bfloat162float(e[i]);
    }
    // Sum of squares in rms_norm_kernel's order: thread t of its block sums
    // dims [8t, 8t + 8) with FMAs, CUB reduces each warp as a balanced tree
    // over t, then adds the two warp totals. Lane l holds threads 2l, 2l + 1.
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s0 = __fmaf_rn(x[i], x[i], s0);
      s1 = __fmaf_rn(x[8 + i], x[8 + i], s1);
    }
    float ss = s0 + s1;
#pragma unroll
    for (int m = 1; m < 16; m <<= 1) ss += __shfl_xor_sync(kFull, ss, m);
    ss += __shfl_xor_sync(kFull, ss, 16);
    float const s = rsqrtf(ss / static_cast<float>(kKvLora) + a.kv_eps);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      x[i] = round_bf16(x[i] * s * w[i]);
      amax = fmaxf(amax, fabsf(x[i]));
    }
    // nvfp4 block: sf = e4m3(max(amax / 6, 2^-9)), q = e2m1(x * (1 / sf)).
    uint8_t const sf8 = to_e4m3(fmaxf(amax / 6.0f, 0.001953125f));
    float const sf =
        __half2float(__half(__nv_cvt_fp8_to_halfraw(sf8, __NV_E4M3)));
    float const inv_sf = 1.0f / sf;
#pragma unroll
    for (int i = 0; i < 16; ++i) x[i] *= inv_sf;
    uint64_t const nope = e2m1x16(x);

    // k_pe RoPE, interleaved pair (2l, 2l + 1) (rotary_embedding_kernel).
    bf16 const* pe = reinterpret_cast<bf16 const*>(&cur.pe);
    float const px = __bfloat162float(pe[0]);
    float const py = __bfloat162float(pe[1]);
    float const rx = round_bf16(px * c - py * sn);
    float const ry = round_bf16(py * c + px * sn);

    uint8_t* mb = mla_buf[warp];
    *reinterpret_cast<uint64_t*>(mb + 8 * lane) = nope;
    mb[kSfOff + nvfp4_sf_byte(lane)] = sf8;
    mb[kRopeOff + 2 * lane] = to_e4m3(rx);
    mb[kRopeOff + 2 * lane + 1] = to_e4m3(ry);

    // Indexer K: LayerNorm, RoPE on dims [64, 128), UE8M0 FP8 per 128.
    // Unfused arithmetic, as in _k_norm_rope (enable_fp_fusion=False).
    float idx_scale = 0.f;
    if (a.idx != nullptr) {
      bf16 const* e = reinterpret_cast<bf16 const*>(&cur.idx);
      float k[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) k[i] = __bfloat162float(e[i]);
      float const mean = warp_sum(k[0] + k[1] + k[2] + k[3]) / 128.0f;
      float cs = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        k[i] = __fsub_rn(k[i], mean);
        cs = __fadd_rn(cs, __fmul_rn(k[i], k[i]));
      }
      float const var = warp_sum(cs) / 128.0f;
      float const r = rsqrtf(__fadd_rn(var, a.idx_eps));
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        k[i] =
            round_bf16(__fadd_rn(__fmul_rn(__fmul_rn(k[i], r), iw[i]), ib[i]));
      }
      if (lane >= 16) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
          int const pair = 2 * (lane - 16) + p;
          float const ic =
              load_cos_sin(a.idx_cos_sin, pos * a.idx_cos_sin_stride + pair,
                           a.idx_cos_sin_f32);
          float const is = load_cos_sin(
              a.idx_cos_sin, pos * a.idx_cos_sin_stride + kRope / 2 + pair,
              a.idx_cos_sin_f32);
          float const y0 = k[2 * p], y1 = k[2 * p + 1];
          k[2 * p] =
              round_bf16(__fadd_rn(__fmul_rn(y0, ic), __fmul_rn(-y1, is)));
          k[2 * p + 1] =
              round_bf16(__fadd_rn(__fmul_rn(y1, ic), __fmul_rn(y0, is)));
        }
      }
      float am = fmaxf(fmaxf(fabsf(k[0]), fabsf(k[1])),
                       fmaxf(fabsf(k[2]), fabsf(k[3])));
      am = warp_max(am);
      // The scale is a power of two, so multiplying by its exact reciprocal
      // rounds exactly like the reference's division.
      float const ex = ceilf(log2f(fmaxf(am, 1e-4f) / 448.0f));
      idx_scale = exp2f(ex);
      float const inv = exp2f(-ex);
      uint32_t q4 = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q4 |= static_cast<uint32_t>(to_e4m3(k[i] * inv)) << (8 * i);
      }
      *reinterpret_cast<uint32_t*>(idx_buf[warp] + 4 * lane) = q4;
    }
    __syncwarp();

    if (lane < kMlaRowBytes / 16) {
      put16<kMulticast>(slot_base + out_row * kMlaRowBytes + 16 * lane,
                        reinterpret_cast<uint4 const*>(mb)[lane]);
    }
    if (a.idx != nullptr) {
      if (lane >= 24) {
        put16<kMulticast>(
            slot_base + a.layout.idx_off + out_row * kIdxDim + 16 * (lane - 24),
            reinterpret_cast<uint4 const*>(idx_buf[warp])[lane - 24]);
      } else if (lane == 23) {
        uint32_t* dst = reinterpret_cast<uint32_t*>(
            slot_base + a.layout.scale_off + out_row * sizeof(float));
        if constexpr (kMulticast) {
          multimem_store_b32(dst, __float_as_uint(idx_scale));
        } else {
          *dst = __float_as_uint(idx_scale);
        }
      }
    }
    __syncwarp();
    cur = nxt;
  }

  pdl_launch_dependents();
  if constexpr (kMulticast) {
    // Fence this CTA's stores; the last CTA releases all with one arrival.
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      int const done = atomicAdd(a.state + 2, 1);
      if (done + 1 == static_cast<int>(gridDim.x)) {
        __threadfence_system();
        a.state[2] = 0;
        multimem_red_add_release(a.signal + a.rank, 1u);
      }
    }
  }
}

struct CacheInsertArgs {
  const uint8_t* staging;  // local replica
  const uint32_t* signal;  // local replica [world]
  int32_t* state;          // [exchanges, insert CTAs done, multicast CTAs done]
  StagingLayout layout;
  int world_size;
  int max_local;
  int num_tokens;                   // per source
  const int64_t* slot_mapping;      // [world * T], rank-major
  const int64_t* idx_slot_mapping;  // indexer cache slots (same layout)
  uint8_t* mla_cache;
  int64_t mla_block_stride;
  int64_t mla_entry_stride;
  int mla_block_size;
  uint8_t* idx_cache;  // null if no indexer
  int64_t idx_block_stride;
  int idx_block_size;
};

__global__ void __launch_bounds__(kThreads)
    hy4_pcp_kv_wait_cache_insert_kernel(CacheInsertArgs a) {
  pdl_wait();
  int const exchanges = a.state[0];
  int const slot = exchanges & 1;
  uint32_t const target = static_cast<uint32_t>(exchanges + 1);
  if (threadIdx.x < a.world_size) {
    uint32_t const* counter = a.signal + threadIdx.x;
    uint64_t spins = 0;
    while (static_cast<int32_t>(load_acquire_system(counter) - target) < 0) {
      if (++spins == kSpinLimit) {
        printf("HY4 PCP KV wait timeout: source=%d exchange=%d\n",
               static_cast<int>(threadIdx.x), exchanges);
        asm volatile("trap;");
      }
    }
  }
  __syncthreads();

  uint8_t const* slot_base = a.staging + slot * a.layout.slot_bytes;
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  int const total = a.world_size * a.num_tokens;
  for (int r = blockIdx.x * kWarps + warp; r < total; r += gridDim.x * kWarps) {
    int const src = r / a.num_tokens;
    int64_t const srow =
        static_cast<int64_t>(src) * a.max_local + (r - src * a.num_tokens);
    if (lane < kMlaRowBytes / 16) {
      int64_t const slot_idx = a.slot_mapping[r];
      if (slot_idx < 0) continue;
      uint4 const v = *reinterpret_cast<uint4 const*>(
          slot_base + srow * kMlaRowBytes + 16 * lane);
      *reinterpret_cast<uint4*>(
          a.mla_cache + (slot_idx / a.mla_block_size) * a.mla_block_stride +
          (slot_idx % a.mla_block_size) * a.mla_entry_stride + 16 * lane) = v;
    } else if (a.idx_cache != nullptr) {
      int64_t const slot_idx = a.idx_slot_mapping[r];
      if (slot_idx < 0) continue;
      int64_t const blk = slot_idx / a.idx_block_size;
      int64_t const off = slot_idx % a.idx_block_size;
      uint8_t* const base = a.idx_cache + blk * a.idx_block_stride;
      if (lane >= 24) {
        uint4 const v = *reinterpret_cast<uint4 const*>(
            slot_base + a.layout.idx_off + srow * kIdxDim + 16 * (lane - 24));
        *reinterpret_cast<uint4*>(base + off * kIdxDim + 16 * (lane - 24)) = v;
      } else if (lane == 23) {
        *reinterpret_cast<uint32_t*>(base + a.idx_block_size * kIdxDim +
                                     off * sizeof(float)) =
            *reinterpret_cast<uint32_t const*>(slot_base + a.layout.scale_off +
                                               srow * sizeof(float));
      }
    }
  }

  // The last CTA advances the exchange count (every CTA has read it).
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    int const done = atomicAdd(a.state + 1, 1);
    if (done + 1 == static_cast<int>(gridDim.x)) {
      a.state[1] = 0;
      a.state[0] = exchanges + 1;
    }
  }
}

struct QRopeQuantArgs {
  const bf16* q;  // [T, H, 256]: nope (192) then pe (64)
  int64_t q_tok_stride, q_head_stride;
  bf16* q_out;  // [T, H, 576] (pe to [512, 576)) or q itself (pe in place)
  int64_t qo_tok_stride, qo_head_stride;
  int qo_pe_off;
  int num_heads;
  const int64_t* positions;
  const void* cos_sin;
  int64_t cos_sin_stride;
  bool cos_sin_f32;
  // Indexer (optional).
  const bf16* iq;  // [T, IH, 128], pe = dims [64, 128)
  int64_t iq_tok_stride, iq_head_stride;
  const bf16* iw;  // [T, IH]
  int64_t iw_tok_stride, iw_head_stride;
  uint8_t* iq_out;  // [T, IH, 128] fp8, contiguous
  float* iw_out;    // [T, IH] fp32, contiguous
  int idx_heads;
  const void* idx_cos_sin;
  int64_t idx_cos_sin_stride;
  bool idx_cos_sin_f32;
  float weight_scale0;  // softmax_scale
  float weight_scale1;  // head_scale
  // Top-k index rows to reset to -1 (optional).
  int* topk;  // [T, topk_cols]
  int64_t topk_tok_stride;
  int topk_cols;
};

// One CTA per token. MLA: 8 lanes per head (8 dims each, 4 pairs). Indexer:
// 16 lanes per head (8 dims each); dims >= 64 are rotated.
__global__ void __launch_bounds__(kThreads)
    hy4_q_rope_indexer_quant_kernel(QRopeQuantArgs a) {
  pdl_wait();
  int64_t const t = blockIdx.x;
  int64_t const pos = a.positions[t];

  for (int v = threadIdx.x; v < a.num_heads * 8; v += kThreads) {
    int const head = v >> 3, j = v & 7;
    uint4 in = *reinterpret_cast<uint4 const*>(
        a.q + t * a.q_tok_stride + head * a.q_head_stride + 192 + 8 * j);
    bf16* e = reinterpret_cast<bf16*>(&in);
    float cv[4], sv[4];
    load_cos_sin4(a.cos_sin, pos * a.cos_sin_stride + 4 * j, a.cos_sin_f32, cv);
    load_cos_sin4(a.cos_sin, pos * a.cos_sin_stride + kRope / 2 + 4 * j,
                  a.cos_sin_f32, sv);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      float const c = cv[p], s = sv[p];
      float const x = __bfloat162float(e[2 * p]);
      float const y = __bfloat162float(e[2 * p + 1]);
      e[2 * p] = __float2bfloat16_rn(x * c - y * s);
      e[2 * p + 1] = __float2bfloat16_rn(y * c + x * s);
    }
    *reinterpret_cast<uint4*>(a.q_out + t * a.qo_tok_stride +
                              head * a.qo_head_stride + a.qo_pe_off + 8 * j) =
        in;
  }

  if (a.iq != nullptr) {
    // idx_heads * 16 is a multiple of 32, so every lane of a warp is active.
    for (int v = threadIdx.x; v < a.idx_heads * 16; v += kThreads) {
      int const head = v >> 4, j = v & 15;
      uint4 in = *reinterpret_cast<uint4 const*>(
          a.iq + t * a.iq_tok_stride + head * a.iq_head_stride + 8 * j);
      bf16 const* e = reinterpret_cast<bf16 const*>(&in);
      float x[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) x[i] = __bfloat162float(e[i]);
      if (j >= 8) {
        float cv[4], sv[4];
        load_cos_sin4(a.idx_cos_sin, pos * a.idx_cos_sin_stride + 4 * (j - 8),
                      a.idx_cos_sin_f32, cv);
        load_cos_sin4(a.idx_cos_sin,
                      pos * a.idx_cos_sin_stride + kRope / 2 + 4 * (j - 8),
                      a.idx_cos_sin_f32, sv);
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          float const c = cv[p], s = sv[p];
          float const x0 = x[2 * p], x1 = x[2 * p + 1];
          x[2 * p] = round_bf16(__fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)));
          x[2 * p + 1] =
              round_bf16(__fadd_rn(__fmul_rn(x1, c), __fmul_rn(x0, s)));
        }
      }
      float am = 0.f;
#pragma unroll
      for (int i = 0; i < 8; ++i) am = fmaxf(am, fabsf(x[i]));
#pragma unroll
      for (int m = 8; m > 0; m >>= 1)
        am = fmaxf(am, __shfl_xor_sync(kFull, am, m));
      float const ex = ceilf(log2f(fmaxf(am, 1e-10f) * (1.0f / 448.0f)));
      float const scale = exp2f(ex);
      float const inv = exp2f(-ex);  // exact: scale is a power of two
      uint2 out;
      __nv_fp8x2_storage_t* ob = reinterpret_cast<__nv_fp8x2_storage_t*>(&out);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float2 const v =
            make_float2(fminf(fmaxf(x[2 * i] * inv, -448.0f), 448.0f),
                        fminf(fmaxf(x[2 * i + 1] * inv, -448.0f), 448.0f));
        ob[i] = __nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, __NV_E4M3);
      }
      *reinterpret_cast<uint2*>(a.iq_out + (t * a.idx_heads + head) * kIdxDim +
                                8 * j) = out;
      if (j == 0) {
        float const wt = __bfloat162float(
            a.iw[t * a.iw_tok_stride + head * a.iw_head_stride]);
        a.iw_out[t * a.idx_heads + head] =
            wt * scale * a.weight_scale0 * a.weight_scale1;
      }
    }
  }
  if (a.topk != nullptr) {
    for (int i = threadIdx.x; i < a.topk_cols; i += kThreads)
      a.topk[t * a.topk_tok_stride + i] = -1;
  }
  pdl_launch_dependents();
}

template <typename Kernel, typename Args>
void launch(Kernel kernel, int blocks, cudaStream_t stream, bool pdl,
            const Args& args, int threads = kThreads) {
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(blocks);
  config.blockDim = dim3(threads);
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
  config.attrs = attr;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, kernel, args);
}

using torch::headeronly::ScalarType;
using Tensor = torch::stable::Tensor;

bool is_f32_cache(const Tensor& t) {
  STD_TORCH_CHECK(t.scalar_type() == ScalarType::Float ||
                      t.scalar_type() == ScalarType::BFloat16,
                  "HY4 PCP: cos_sin_cache must be fp32 or bf16");
  STD_TORCH_CHECK(t.dim() == 2 && t.size(1) == kRope && t.stride(1) == 1 &&
                      t.stride(0) % 4 == 0 &&
                      reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                  "HY4 PCP: cos_sin_cache must be [max_pos, 64], 16B rows");
  return t.scalar_type() == ScalarType::Float;
}

StagingLayout staging_layout(int64_t world_size, int64_t max_local) {
  int64_t const rows = world_size * max_local;
  StagingLayout l;
  l.idx_off = rows * kMlaRowBytes;
  l.scale_off = l.idx_off + rows * kIdxDim;
  l.slot_bytes = (l.scale_off + rows * 4 + 15) / 16 * 16;
  return l;
}

}  // namespace

// staging_ptr / signal_ptr: multicast addresses, or (multicast = false) the
// local staging buffer for single-GPU testing, with no signal.
void hy4_pcp_kv_norm_rope_quant_multicast(
    const Tensor& kv_lora, const std::optional<Tensor>& index_k,
    const Tensor& positions, const Tensor& cos_sin_cache,
    const std::optional<Tensor>& index_cos_sin_cache,
    const Tensor& kv_norm_weight, double kv_eps,
    const std::optional<Tensor>& index_norm_weight,
    const std::optional<Tensor>& index_norm_bias, double index_eps,
    Tensor& state, int64_t staging_ptr, int64_t signal_ptr, int64_t world_size,
    int64_t rank, int64_t max_local, int64_t multicast_blocks,
    int64_t multicast_warps, bool multicast, bool pdl) {
  int64_t const T = kv_lora.size(0);
  STD_TORCH_CHECK(kv_lora.scalar_type() == ScalarType::BFloat16 &&
                      kv_lora.dim() == 2 &&
                      kv_lora.size(1) == kKvLora + kRope &&
                      kv_lora.stride(1) == 1 && kv_lora.stride(0) % 8 == 0,
                  "HY4 PCP: kv_lora must be bf16 [T, 576] with 16B rows");
  STD_TORCH_CHECK(T <= max_local, "HY4 PCP: too many tokens for staging");
  STD_TORCH_CHECK(
      positions.scalar_type() == ScalarType::Long && positions.numel() >= T,
      "HY4 PCP: positions must be int64");
  STD_TORCH_CHECK(kv_norm_weight.scalar_type() == ScalarType::BFloat16 &&
                      kv_norm_weight.numel() == kKvLora,
                  "HY4 PCP: kv norm weight must be bf16 [512]");
  STD_TORCH_CHECK(state.scalar_type() == ScalarType::Int && state.numel() == 3,
                  "HY4 PCP: bad state");
  STD_TORCH_CHECK(multicast_blocks > 0 && staging_ptr != 0 &&
                      (!multicast || signal_ptr != 0),
                  "HY4 PCP: bad staging");
  KvMulticastArgs a{};
  a.kv = static_cast<const bf16*>(kv_lora.data_ptr());
  a.kv_stride = kv_lora.stride(0);
  a.positions = positions.const_data_ptr<int64_t>();
  a.cos_sin = cos_sin_cache.data_ptr();
  a.cos_sin_stride = cos_sin_cache.stride(0);
  a.cos_sin_f32 = is_f32_cache(cos_sin_cache);
  a.kv_weight = static_cast<const bf16*>(kv_norm_weight.data_ptr());
  a.kv_eps = static_cast<float>(kv_eps);
  if (index_k.has_value()) {
    STD_TORCH_CHECK(
        index_k->scalar_type() == ScalarType::BFloat16 && index_k->dim() == 2 &&
            index_k->size(0) == T && index_k->size(1) == kIdxDim &&
            index_k->stride(1) == 1 && index_k->stride(0) % 4 == 0 &&
            index_cos_sin_cache.has_value() && index_norm_weight.has_value() &&
            index_norm_bias.has_value() &&
            index_norm_weight->scalar_type() == ScalarType::Float &&
            index_norm_bias->scalar_type() == ScalarType::Float,
        "HY4 PCP: bad indexer K inputs");
    a.idx = static_cast<const bf16*>(index_k->data_ptr());
    a.idx_stride = index_k->stride(0);
    a.idx_cos_sin = index_cos_sin_cache->data_ptr();
    a.idx_cos_sin_stride = index_cos_sin_cache->stride(0);
    a.idx_cos_sin_f32 = is_f32_cache(*index_cos_sin_cache);
    a.idx_weight = index_norm_weight->const_data_ptr<float>();
    a.idx_bias = index_norm_bias->const_data_ptr<float>();
    a.idx_eps = static_cast<float>(index_eps);
  }
  a.num_tokens = static_cast<int>(T);
  a.staging = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(staging_ptr));
  a.signal = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(signal_ptr));
  a.state = state.mutable_data_ptr<int32_t>();
  a.layout = staging_layout(world_size, max_local);
  a.world_size = static_cast<int>(world_size);
  a.rank = static_cast<int>(rank);
  a.max_local = static_cast<int>(max_local);

  const torch::stable::accelerator::DeviceGuard guard(
      kv_lora.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream();
  int const blocks = static_cast<int>(multicast_blocks);
  STD_TORCH_CHECK(
      multicast_warps == 8 || multicast_warps == 16 || multicast_warps == 32,
      "HY4 PCP: multicast_warps must be 8, 16 or 32");
  auto run = [&](auto warps_tag) {
    constexpr int kW = decltype(warps_tag)::value;
    if (multicast) {
      launch(hy4_pcp_kv_norm_rope_quant_multicast_kernel<kW, true>, blocks,
             stream, pdl, a, kW * 32);
    } else {
      launch(hy4_pcp_kv_norm_rope_quant_multicast_kernel<kW, false>, blocks,
             stream, pdl, a, kW * 32);
    }
  };
  if (multicast_warps == 8) {
    run(std::integral_constant<int, 8>{});
  } else if (multicast_warps == 16) {
    run(std::integral_constant<int, 16>{});
  } else {
    run(std::integral_constant<int, 32>{});
  }
  check_cuda_launch("HY4 PCP KV norm/rope/quant multicast");
}

void hy4_pcp_kv_wait_cache_insert(
    const Tensor& staging, const Tensor& signal, Tensor& state,
    const Tensor& slot_mapping, Tensor& mla_cache,
    const std::optional<Tensor>& index_cache,
    const std::optional<Tensor>& index_slot_mapping, int64_t world_size,
    int64_t max_local, int64_t num_tokens, int64_t num_blocks, bool pdl) {
  STD_TORCH_CHECK(state.scalar_type() == ScalarType::Int && state.numel() == 3,
                  "HY4 PCP: bad state");
  STD_TORCH_CHECK(signal.numel() >= world_size && world_size <= kThreads,
                  "HY4 PCP: bad signal");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long &&
                      slot_mapping.numel() >= world_size * num_tokens,
                  "HY4 PCP: slot_mapping must cover world_size * T tokens");
  STD_TORCH_CHECK(mla_cache.dim() == 3 && mla_cache.size(2) == kMlaRowBytes &&
                      mla_cache.element_size() == 1,
                  "HY4 PCP: MLA cache must be nvfp4_ds_mla [blocks, bs, 352]");
  CacheInsertArgs a{};
  a.staging = static_cast<const uint8_t*>(staging.data_ptr());
  a.signal = reinterpret_cast<const uint32_t*>(signal.data_ptr());
  a.state = state.mutable_data_ptr<int32_t>();
  a.layout = staging_layout(world_size, max_local);
  a.world_size = static_cast<int>(world_size);
  a.max_local = static_cast<int>(max_local);
  a.num_tokens = static_cast<int>(num_tokens);
  a.slot_mapping = slot_mapping.const_data_ptr<int64_t>();
  a.idx_slot_mapping = a.slot_mapping;
  if (index_slot_mapping.has_value()) {
    STD_TORCH_CHECK(index_slot_mapping->scalar_type() == ScalarType::Long &&
                        index_slot_mapping->numel() >= world_size * num_tokens,
                    "HY4 PCP: index_slot_mapping must cover world_size * T");
    a.idx_slot_mapping = index_slot_mapping->const_data_ptr<int64_t>();
  }
  a.mla_cache = static_cast<uint8_t*>(mla_cache.data_ptr());
  a.mla_block_stride = mla_cache.stride(0);
  a.mla_entry_stride = mla_cache.stride(1);
  a.mla_block_size = static_cast<int>(mla_cache.size(1));
  if (index_cache.has_value()) {
    // DeepseekV32IndexerCache: [blocks, bs, 132] viewed as [blocks, bs * 132];
    // values [bs * 128] then fp32 scales [bs].
    STD_TORCH_CHECK(index_cache->element_size() == 1 &&
                        index_cache->dim() == 3 &&
                        index_cache->size(2) == kIdxDim + 4,
                    "HY4 PCP: indexer cache must be [blocks, bs, 132]");
    a.idx_cache = static_cast<uint8_t*>(index_cache->data_ptr());
    a.idx_block_stride = index_cache->stride(0);
    a.idx_block_size = static_cast<int>(index_cache->size(1));
  }
  const torch::stable::accelerator::DeviceGuard guard(state.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream();
  launch(hy4_pcp_kv_wait_cache_insert_kernel, static_cast<int>(num_blocks),
         stream, pdl, a);
  check_cuda_launch("HY4 PCP KV wait/cache insert");
}

void hy4_q_rope_indexer_quant(const Tensor& q, Tensor& q_out,
                              const Tensor& positions,
                              const Tensor& cos_sin_cache,
                              const std::optional<Tensor>& index_q,
                              const std::optional<Tensor>& index_weights,
                              const std::optional<Tensor>& index_cos_sin_cache,
                              const std::optional<Tensor>& index_q_out,
                              const std::optional<Tensor>& index_weights_out,
                              const std::optional<Tensor>& topk_indices,
                              double softmax_scale, double head_scale,
                              bool pdl) {
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::BFloat16 && q.dim() == 3 &&
                      q.size(2) == 256 && q.stride(2) == 1 &&
                      q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0,
                  "HY4 Q rope/quant: q must be bf16 [T, H, 256]");
  STD_TORCH_CHECK(
      q_out.scalar_type() == ScalarType::BFloat16 && q_out.dim() == 3 &&
          q_out.size(0) == q.size(0) && q_out.size(1) == q.size(1) &&
          (q_out.size(2) == 576 || q_out.size(2) == 256) &&
          q_out.stride(2) == 1 && q_out.stride(1) % 8 == 0 &&
          q_out.stride(0) % 8 == 0,
      "HY4 Q rope/quant: q_out must be bf16 [T, H, 576] or [T, H, 256]");
  QRopeQuantArgs a{};
  a.q = static_cast<const bf16*>(q.data_ptr());
  a.q_tok_stride = q.stride(0);
  a.q_head_stride = q.stride(1);
  a.q_out = static_cast<bf16*>(q_out.data_ptr());
  a.qo_tok_stride = q_out.stride(0);
  a.qo_head_stride = q_out.stride(1);
  a.qo_pe_off = static_cast<int>(q_out.size(2)) - kRope;
  a.num_heads = static_cast<int>(q.size(1));
  a.positions = positions.const_data_ptr<int64_t>();
  a.cos_sin = cos_sin_cache.data_ptr();
  a.cos_sin_stride = cos_sin_cache.stride(0);
  a.cos_sin_f32 = is_f32_cache(cos_sin_cache);
  if (index_q.has_value()) {
    STD_TORCH_CHECK(
        index_q->scalar_type() == ScalarType::BFloat16 && index_q->dim() == 3 &&
            index_q->size(0) == q.size(0) && index_q->size(2) == kIdxDim &&
            index_q->size(1) % 2 == 0 && index_q->stride(2) == 1 &&
            index_q->stride(1) % 8 == 0 && index_q->stride(0) % 8 == 0 &&
            index_weights.has_value() && index_cos_sin_cache.has_value() &&
            index_q_out.has_value() && index_weights_out.has_value() &&
            index_weights->scalar_type() == ScalarType::BFloat16 &&
            index_q_out->is_contiguous() &&
            index_weights_out->is_contiguous() &&
            index_weights_out->scalar_type() == ScalarType::Float,
        "HY4 Q rope/quant: bad indexer inputs");
    a.iq = static_cast<const bf16*>(index_q->data_ptr());
    a.iq_tok_stride = index_q->stride(0);
    a.iq_head_stride = index_q->stride(1);
    a.iw = static_cast<const bf16*>(index_weights->data_ptr());
    a.iw_tok_stride = index_weights->stride(0);
    a.iw_head_stride = index_weights->stride(1);
    a.iq_out = static_cast<uint8_t*>(index_q_out->data_ptr());
    a.iw_out = static_cast<float*>(index_weights_out->data_ptr());
    a.idx_heads = static_cast<int>(index_q->size(1));
    a.idx_cos_sin = index_cos_sin_cache->data_ptr();
    a.idx_cos_sin_stride = index_cos_sin_cache->stride(0);
    a.idx_cos_sin_f32 = is_f32_cache(*index_cos_sin_cache);
    a.weight_scale0 = static_cast<float>(softmax_scale);
    a.weight_scale1 = static_cast<float>(head_scale);
  }
  if (topk_indices.has_value()) {
    STD_TORCH_CHECK(topk_indices->scalar_type() == ScalarType::Int &&
                        topk_indices->dim() == 2 &&
                        topk_indices->size(0) == q.size(0) &&
                        topk_indices->stride(1) == 1,
                    "HY4 Q rope/quant: topk_indices must be int32 [T, K]");
    a.topk = static_cast<int*>(topk_indices->data_ptr());
    a.topk_tok_stride = topk_indices->stride(0);
    a.topk_cols = static_cast<int>(topk_indices->size(1));
  }
  if (q.size(0) == 0) return;
  const torch::stable::accelerator::DeviceGuard guard(q.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream();
  launch(hy4_q_rope_indexer_quant_kernel, static_cast<int>(q.size(0)), stream,
         pdl, a);
  check_cuda_launch("HY4 Q rope/indexer quant");
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, hy4_pcp_prologue_ops) {
  hy4_pcp_prologue_ops.def(
      "hy4_pcp_kv_norm_rope_quant_multicast(Tensor kv_lora, Tensor? index_k, "
      "Tensor positions, "
      "Tensor cos_sin_cache, Tensor? index_cos_sin_cache, "
      "Tensor kv_norm_weight, float kv_eps, Tensor? index_norm_weight, "
      "Tensor? index_norm_bias, float index_eps, Tensor! state, "
      "int staging_ptr, int signal_ptr, int world_size, int rank, "
      "int max_local, int multicast_blocks, int multicast_warps, bool "
      "multicast, "
      "bool pdl) -> ()");
  hy4_pcp_prologue_ops.def(
      "hy4_pcp_kv_wait_cache_insert(Tensor staging, Tensor signal, Tensor! "
      "state, "
      "Tensor slot_mapping, Tensor! mla_cache, Tensor(a!)? index_cache, "
      "Tensor? index_slot_mapping, "
      "int world_size, int max_local, int num_tokens, int num_blocks, "
      "bool pdl) -> ()");
  hy4_pcp_prologue_ops.def(
      "hy4_q_rope_indexer_quant(Tensor q, Tensor! q_out, Tensor positions, "
      "Tensor cos_sin_cache, Tensor? index_q, Tensor? index_weights, "
      "Tensor? index_cos_sin_cache, Tensor(b!)? index_q_out, "
      "Tensor(c!)? index_weights_out, Tensor(d!)? topk_indices, "
      "float softmax_scale, float head_scale, bool pdl) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, hy4_pcp_prologue_ops) {
  hy4_pcp_prologue_ops.impl("hy4_pcp_kv_norm_rope_quant_multicast",
                            TORCH_BOX(&hy4_pcp_kv_norm_rope_quant_multicast));
  hy4_pcp_prologue_ops.impl("hy4_pcp_kv_wait_cache_insert",
                            TORCH_BOX(&hy4_pcp_kv_wait_cache_insert));
  hy4_pcp_prologue_ops.impl("hy4_q_rope_indexer_quant",
                            TORCH_BOX(&hy4_q_rope_indexer_quant));
}
