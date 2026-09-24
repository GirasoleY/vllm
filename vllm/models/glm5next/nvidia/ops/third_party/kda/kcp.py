# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Blackwell KCP preparation and affine-summary kernels.

Communication and state merging live in ``ops/kcp.py``. This module is
imported lazily by the KCP path so ordinary KDA does not require CuTe DSL.
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    BFloat16,
    Boolean,
    Float16,
    Float32,
    Int32,
    Int64,
    Uint32,
    Uint64,
    cute,
)
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import _tcgen05, fence_before_tma_store, simple_tma_copy

from .kernels import chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter


def prepare_kcp(
    q: torch.Tensor,
    k: torch.Tensor,
    raw_g: torch.Tensor,
    beta_raw: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    *,
    lower_bound: float | None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, ...]:
    """Prepare positive or ragged scan segments without a device readback.

    Inputs may have nonnegative strides. Metadata values describe all tokens
    exactly once and use contiguous int32 or int64 tensors built on the CPU.
    Empty segments need no chunk entry. The returned A contains pre-inverted
    16x16 diagonal blocks for the triangular merge.
    """
    # KCP also uses this extension when ordinary prefill selects Triton.
    import vllm._flashkda_C  # noqa: F401

    assert q.ndim == 4 and q.shape[0] == 1
    tokens, heads, dim = q.shape[1:]
    assert chunk_size in (16, 32, 64)
    qn = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    kn = torch.empty_like(qn)
    g = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    beta = torch.empty((1, tokens, heads), dtype=torch.float32, device=q.device)
    A = torch.empty(
        (1, tokens, heads, chunk_size), dtype=torch.float32, device=q.device
    )
    Aqk = torch.empty_like(A)
    torch.ops._flashkda_C.kcp_prepare(
        q,
        k,
        raw_g,
        beta_raw,
        a_log,
        bias,
        cu_seqlens,
        chunk_indices,
        qn,
        kn,
        g,
        beta,
        A,
        Aqk,
        lower_bound if lower_bound is not None else 0.0,
        lower_bound is not None,
        chunk_size,
    )
    if chunk_indices.shape[0] and chunk_size > 16:
        subchunks = chunk_size // 16
        chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[
            (chunk_indices.shape[0] * subchunks * (subchunks - 1) // 2 * heads,)
        ](
            q=qn,
            k=kn,
            g=g,
            beta=beta,
            A=A,
            Aqk=Aqk,
            scale=dim**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=tokens,
            NT=chunk_indices.shape[0],
            H=heads,
            K=dim,
            BT=chunk_size,
            BC=16,
            NC=subchunks,
        )
    return qn, kn, g, beta, A, Aqk


# S summary adapted from Thien Tran's Sm100ChunkHKernel, vLLM PR #43273,
# commit 12ab2e4adb8b43e89d959fc360a6770cc91a79be (Apache-2.0).
@dsl_user_op
def allocate_tmem(taddr, columns: cutlass.Constexpr[int], *, loc=None, ip=None):
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(columns).ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def deallocate_tmem(base, columns: cutlass.Constexpr[int], *, loc=None, ip=None):
    nvvm.tcgen05_dealloc(
        _tcgen05._make_tmem_llvm_ptr(base, loc=loc, ip=ip),
        Int32(columns).ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_ts_tf32(d_tmem, a_tmem, b_desc, idesc, enable_input_d, *, loc=None, ip=None):
    # Retained Triton consumes raw FP32 bits with kind::tf32. No cvt is inserted.
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.TF32,
            nvvm.CTAGroupKind.CTA_1,
            _tcgen05._make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            _tcgen05._make_tmem_llvm_ptr(a_tmem, loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


# These are instruction/layout granularities, not model dimensions.
VALUE_TILE = 128
KEY_GROUP = 64
TMEM_VECTOR = 32
MMA_REDUCTION = 16
WARP_SIZE = 32


def make_operand_idesc(dtype, m, n, *, transpose_b=False):
    operand_format = 2 if dtype == Float32 else (1 if dtype == BFloat16 else 0)
    return Uint32(
        (1 << 4)
        | (operand_format << 7)
        | (operand_format << 10)
        | ((n >> 3) << 17)
        | ((m >> 4) << 24)
        | (int(transpose_b) << 16)
    )


class KcpSSummary:
    """FP32 S carry with ordered projection and input-dtype rounding boundaries."""

    def __init__(
        self,
        heads,
        key_dim,
        value_dim,
        chunk_size,
        dtype,
        mapped,
        use_tma,
        stages=2,
    ):
        self.heads = heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.chunk_size = chunk_size
        self.key_tile = cute.ceil_div(key_dim, KEY_GROUP) * KEY_GROUP
        self.time_tile = cute.ceil_div(chunk_size, TMEM_VECTOR) * TMEM_VECTOR
        self.dtype = dtype
        self.is_fp32 = dtype == Float32
        self.smem_group = 128 // (dtype.width // 8)
        self.mapped = mapped
        self.use_tma = use_tma
        self.stages = stages
        operand_columns = (
            max(self.time_tile, KEY_GROUP)
            if self.is_fp32
            else max(self.time_tile, self.key_tile) // 2
        )
        required_columns = self.time_tile + self.key_tile + operand_columns
        self.tmem_columns = 1 << (required_columns - 1).bit_length()
        if self.tmem_columns > 512:
            raise ValueError("S summary tile exceeds the 512-column TMEM capacity")

    @cute.jit
    def input_layout(self, width: cutlass.Constexpr[int]):
        return cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (
                    self.time_tile,
                    1,
                    (self.smem_group, width // self.smem_group),
                    self.stages,
                ),
                stride=(
                    self.smem_group,
                    0,
                    (1, self.time_tile * self.smem_group),
                    self.time_tile * width,
                ),
            ),
        )

    @cute.jit
    def make_input_tma(self, tensor, width: cutlass.Constexpr[int]):
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(tensor, (None, None, KEY_GROUP)),
            self.input_layout(width),
            cta_tiler=(self.time_tile, 1, width),
        )

    @cute.jit
    def __call__(
        self,
        kg: cute.Tensor,
        u: cute.Tensor,
        w: cute.Tensor,
        gk: cute.Tensor,
        out: cute.Tensor,
        cu: cute.Tensor,
        indices: cute.Tensor,
        stream: CUstream,
    ):
        # Resolve the singleton batch dimension in CuTe's host IR, avoiding
        # four eager Torch view operations at each attention layer.
        if cutlass.const_expr(self.use_tma):
            kg_tma = self.make_input_tma(kg[0, None, None, None], self.key_tile)
            w_tma = self.make_input_tma(w[0, None, None, None], self.key_tile)
            u_tma = self.make_input_tma(u[0, None, None, None], VALUE_TILE)
        else:
            kg_tma = kg[0, None, None, None]
            w_tma = w[0, None, None, None]
            u_tma = u[0, None, None, None]
        self.kcp_summary_s_kernel(
            kg_tma, u_tma, w_tma, gk[0, None, None, None], out, cu, indices
        ).launch(
            grid=(
                cute.ceil_div(self.value_dim, VALUE_TILE)
                * self.heads
                * (cu.shape[0] - 1),
                1,
                1,
            ),
            block=(320, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kcp_summary_s_kernel(
        self,
        kg_tma,
        u_tma,
        w_tma,
        gk: cute.Tensor,
        out: cute.Tensor,
        cu: cute.Tensor,
        indices: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        value_tiles = cute.ceil_div(self.value_dim, VALUE_TILE)
        value_block = block % value_tiles
        head = (block // value_tiles) % self.heads
        seq = block // (value_tiles * self.heads)
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        stages = self.stages
        smem = cutlass.utils.SmemAllocator()

        def allocate_input(width):
            layout = self.input_layout(width)
            return smem.allocate_tensor(
                self.dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )[None, 0, None, None]

        sw = allocate_input(self.key_tile)
        if cutlass.const_expr(self.is_fp32):
            # TF32 has no transposed-B mode for this ordinary 128-byte swizzle.
            # The loader writes Kg as [key, token] directly, with no extra kernel.
            kg_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (self.key_tile, (32, self.time_tile // 32), stages),
                    stride=(
                        32,
                        (1, self.key_tile * 32),
                        self.key_tile * self.time_tile,
                    ),
                ),
            )
            skg = smem.allocate_tensor(
                self.dtype, kg_layout.outer, byte_alignment=128, swizzle=kg_layout.inner
            )
        else:
            skg = allocate_input(self.key_tile)
        su = allocate_input(VALUE_TILE)
        decay = smem.allocate_tensor(Float32, cute.make_layout((self.key_tile, stages)))
        load_done = smem.allocate_array(Int64, stages)
        input_ready = smem.allocate_array(Int64, stages)
        packed_h = smem.allocate_array(Int64, stages)
        if cutlass.const_expr(self.is_fp32):
            packed_consumed = smem.allocate_array(Int64, stages)
        projection_done = smem.allocate_array(Int64, stages)
        update_ready = smem.allocate_array(Int64, stages)
        update_done = smem.allocate_array(Int64, stages)
        taddr = smem.allocate(Int32, 4)

        if warp == 0:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(load_done + stage, 1)
                    cute.arch.mbarrier_init(input_ready + stage, 1)
                    cute.arch.mbarrier_init(packed_h + stage, 128)
                    if cutlass.const_expr(self.is_fp32):
                        cute.arch.mbarrier_init(packed_consumed + stage, 1)
                    cute.arch.mbarrier_init(projection_done + stage, 1)
                    cute.arch.mbarrier_init(update_ready + stage, 256)
                    cute.arch.mbarrier_init(update_done + stage, 1)
                cute.arch.mbarrier_init_fence()
        elif warp == 8:
            allocate_tmem(taddr, self.tmem_columns)
        elif warp == 9:  # noqa: SIM102 - omit TMA attributes from the generic IR.
            if cutlass.const_expr(self.use_tma):
                cpasync.prefetch_descriptor(w_tma.atom)
                cpasync.prefetch_descriptor(kg_tma.atom)
                cpasync.prefetch_descriptor(u_tma.atom)
        cute.arch.sync_threads()

        base = cute.make_tensor(taddr, cute.make_layout(1))[0]
        p_tmem = base
        h_tmem = base + self.time_tile
        # H packing dies after projection. Its storage then holds the residual.
        operand_tmem = h_tmem + self.key_tile
        bos = Int64(cu[seq])
        eos = Int64(cu[seq + 1])
        length = eos - bos
        chunks = cute.ceil_div(length, self.chunk_size)

        if warp == 9:
            stage = 0
            phase = 0
            if cutlass.const_expr(self.use_tma):
                w_tiles = cute.logical_divide(
                    cute.domain_offset(
                        (bos, (0, 0)), w_tma.tma_tensor[None, head, None]
                    ),
                    (self.chunk_size, None),
                )
                k_tiles = cute.logical_divide(
                    cute.domain_offset(
                        (bos, (0, 0)), kg_tma.tma_tensor[None, head, None]
                    ),
                    (self.chunk_size, None),
                )
                u_tiles = cute.logical_divide(
                    cute.domain_offset(
                        (bos, (0, 0)), u_tma.tma_tensor[None, head, None]
                    ),
                    (self.chunk_size, (None, VALUE_TILE // KEY_GROUP)),
                )
            for chunk in range(chunks):
                cute.arch.mbarrier_wait(update_done + stage, phase ^ 1)
                valid = min(
                    Int64(self.chunk_size), eos - (bos + chunk * self.chunk_size)
                )
                if cutlass.const_expr(self.use_tma):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            load_done + stage,
                            self.time_tile * (2 * self.key_tile + VALUE_TILE) * 2,
                        )
                    simple_tma_copy(
                        w_tma.atom,
                        w_tiles[(None, chunk), None],
                        sw[None, None, stage],
                        load_done + stage,
                    )
                    simple_tma_copy(
                        kg_tma.atom,
                        k_tiles[(None, chunk), None],
                        skg[None, None, stage],
                        load_done + stage,
                    )
                    simple_tma_copy(
                        u_tma.atom,
                        u_tiles[(None, chunk), (None, (None, value_block))],
                        su[None, None, stage],
                        load_done + stage,
                    )
                else:
                    # The same on-chip pipeline also handles unaligned/strided
                    # inputs and dimensions which are not legal TMA boxes.
                    for i in range(lane, self.time_tile * self.key_tile, WARP_SIZE):
                        token, key = i // self.key_tile, i % self.key_tile
                        wk, kk = self.dtype(0), self.dtype(0)
                        if token < valid and key < self.key_dim:
                            src_token = bos + chunk * self.chunk_size + token
                            wk = w_tma[src_token, head, key]
                            kk = kg_tma[src_token, head, key]
                        sw[
                            token,
                            (key % self.smem_group, key // self.smem_group),
                            stage,
                        ] = wk
                        if cutlass.const_expr(self.is_fp32):
                            skg[key, (token % 32, token // 32), stage] = kk
                        else:
                            skg[token, (key % KEY_GROUP, key // KEY_GROUP), stage] = kk
                    for i in range(lane, self.time_tile * VALUE_TILE, WARP_SIZE):
                        token, value = i // VALUE_TILE, i % VALUE_TILE
                        uv = self.dtype(0)
                        global_value = value_block * VALUE_TILE + value
                        if token < valid and global_value < self.value_dim:
                            uv = u_tma[
                                bos + chunk * self.chunk_size + token,
                                head,
                                global_value,
                            ]
                        su[
                            token,
                            (value % self.smem_group, value // self.smem_group),
                            stage,
                        ] = uv
                last = min(bos + (chunk + 1) * self.chunk_size, eos) - 1
                for i in cutlass.range_constexpr(self.key_tile // WARP_SIZE):
                    key = lane + i * WARP_SIZE
                    scale = Float32(1)
                    if key < self.key_dim:
                        scale = cute.math.exp2(
                            Float32(gk[last, head, key]), fastmath=True
                        )
                    decay[key, stage] = scale
                if cutlass.const_expr(self.use_tma):
                    cute.arch.mbarrier_wait(load_done + stage, phase)
                    # Tensor bounds do not delimit packed request segments.
                    if valid < self.time_tile:
                        for i in range(lane, self.time_tile * self.key_tile, WARP_SIZE):
                            token, key = i // self.key_tile, i % self.key_tile
                            if token >= valid:
                                sw[
                                    token, (key % KEY_GROUP, key // KEY_GROUP), stage
                                ] = self.dtype(0)
                                skg[
                                    token, (key % KEY_GROUP, key // KEY_GROUP), stage
                                ] = self.dtype(0)
                        for i in range(lane, self.time_tile * VALUE_TILE, WARP_SIZE):
                            token, value = i // VALUE_TILE, i % VALUE_TILE
                            if token >= valid:
                                su[
                                    token,
                                    (value % KEY_GROUP, value // KEY_GROUP),
                                    stage,
                                ] = self.dtype(0)
                fence_before_tma_store()
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(input_ready + stage)
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1

        elif warp == 8:
            stage = 0
            phase = 0
            if cutlass.const_expr(self.is_fp32):
                project_desc = make_operand_idesc(
                    self.dtype, VALUE_TILE, self.time_tile
                )
                update_desc = make_operand_idesc(self.dtype, VALUE_TILE, self.key_tile)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                for chunk in range(chunks):
                    cute.arch.mbarrier_wait(input_ready + stage, phase)
                    wdesc_base = sdesc | (sw[None, None, stage].iterator.toint() >> 4)
                    for group in cutlass.range_constexpr(self.key_tile // KEY_GROUP):
                        group_phase = (
                            (chunk // stages) * (self.key_tile // KEY_GROUP) + group
                        ) & 1
                        cute.arch.mbarrier_wait(packed_h + stage, group_phase)
                        _tcgen05.fence_after_thread_sync()
                        for step in cutlass.range_constexpr(KEY_GROUP // 8):
                            key = group * KEY_GROUP + step * 8
                            byte_offset = (key // 32) * self.time_tile * 128 + (
                                key % 32
                            ) * 4
                            mma_ts_tf32(
                                p_tmem,
                                operand_tmem + step * 8,
                                wdesc_base + (byte_offset >> 4),
                                project_desc,
                                group > 0 or step > 0,
                            )
                        _tcgen05.commit(packed_consumed + stage)
                    _tcgen05.commit(projection_done + stage)
                    cute.arch.mbarrier_wait(update_ready + stage, phase)
                    _tcgen05.fence_after_thread_sync()
                    kgdesc_base = sdesc | (skg[None, None, stage].iterator.toint() >> 4)
                    for step in cutlass.range_constexpr(self.chunk_size // 8):
                        token = step * 8
                        byte_offset = (token // 32) * self.key_tile * 128 + (
                            token % 32
                        ) * 4
                        mma_ts_tf32(
                            h_tmem,
                            operand_tmem + step * 8,
                            kgdesc_base + (byte_offset >> 4),
                            update_desc,
                            True,
                        )
                    _tcgen05.commit(update_done + stage)
                    stage = (stage + 1) % stages
                    if stage == 0:
                        phase ^= 1
            else:
                project_desc = make_operand_idesc(
                    self.dtype, VALUE_TILE, self.time_tile
                )
                update_desc = make_operand_idesc(
                    self.dtype, VALUE_TILE, self.key_tile, transpose_b=True
                )
                sdesc = _tcgen05.make_sdesc_128B_swizzle(self.time_tile * KEY_GROUP * 2)
                for chunk in range(chunks):
                    cute.arch.mbarrier_wait(input_ready + stage, phase)
                    cute.arch.mbarrier_wait(packed_h + stage, phase)
                    _tcgen05.fence_after_thread_sync()
                    wdesc_base = sdesc | (sw[None, None, stage].iterator.toint() >> 4)
                    # Preserve the first K64 dot followed by the accumulated K64 dot.
                    for half in cutlass.range_constexpr(self.key_tile // KEY_GROUP):
                        for k16 in cutlass.range_constexpr(KEY_GROUP // MMA_REDUCTION):
                            wdesc = wdesc_base + (
                                (
                                    half * self.time_tile * KEY_GROUP * 2
                                    + k16 * MMA_REDUCTION * 2
                                )
                                >> 4
                            )
                            _tcgen05.mma_ts_f16(
                                p_tmem,
                                operand_tmem
                                + half * KEY_GROUP // 2
                                + k16 * MMA_REDUCTION // 2,
                                wdesc,
                                project_desc,
                                half > 0 or k16 > 0,
                            )
                    _tcgen05.commit(projection_done + stage)
                    cute.arch.mbarrier_wait(update_ready + stage, phase)
                    _tcgen05.fence_after_thread_sync()
                    kgdesc_base = sdesc | (skg[None, None, stage].iterator.toint() >> 4)
                    for k16 in cutlass.range_constexpr(
                        cute.ceil_div(self.chunk_size, MMA_REDUCTION)
                    ):
                        kgdesc = kgdesc_base + (
                            (k16 * MMA_REDUCTION * KEY_GROUP * 2) >> 4
                        )
                        _tcgen05.mma_ts_f16(
                            h_tmem,
                            operand_tmem + k16 * MMA_REDUCTION // 2,
                            kgdesc,
                            update_desc,
                            True,
                        )
                    _tcgen05.commit(update_done + stage)
                    stage = (stage + 1) % stages
                    if stage == 0:
                        phase ^= 1
        elif warp < 4:
            # One thread owns one value channel and all padded key channels.
            value = tid
            stage = 0
            phase = 0
            for i in cutlass.range_constexpr(self.key_tile // TMEM_VECTOR):
                zeros = cute.make_rmem_tensor(32, Float32)
                zeros.fill(0)
                _tcgen05.st(warp * 32, h_tmem + i * 32, "32x32b", 32, zeros)
            _tcgen05.wait_st()
            _tcgen05.fence_before_thread_sync()
            for chunk in range(chunks):
                if chunk > 0:
                    previous = (stage + stages - 1) % stages
                    previous_phase = phase ^ Int32(stage == 0)
                    cute.arch.mbarrier_wait(update_done + previous, previous_phase)
                    _tcgen05.fence_after_thread_sync()
                if cutlass.const_expr(self.is_fp32):
                    # A single K64 operand buffer keeps K256 within 512 TMEM columns.
                    for group in cutlass.range_constexpr(self.key_tile // KEY_GROUP):
                        for i in cutlass.range_constexpr(KEY_GROUP // TMEM_VECTOR):
                            carry = _tcgen05.ld(
                                warp * 32,
                                h_tmem + group * KEY_GROUP + i * 32,
                                "32x32b",
                                32,
                            )
                            _tcgen05.st(
                                warp * 32, operand_tmem + i * 32, "32x32b", 32, carry
                            )
                        _tcgen05.wait_st()
                        _tcgen05.fence_before_thread_sync()
                        cute.arch.mbarrier_arrive(packed_h + stage)
                        group_phase = (
                            (chunk // stages) * (self.key_tile // KEY_GROUP) + group
                        ) & 1
                        cute.arch.mbarrier_wait(packed_consumed + stage, group_phase)
                        _tcgen05.fence_after_thread_sync()
                else:
                    # Pack the unscaled FP32 carry for projection.
                    for i in cutlass.range_constexpr(self.key_tile // TMEM_VECTOR):
                        carry = _tcgen05.ld(warp * 32, h_tmem + i * 32, "32x32b", 32)
                        packed = cute.make_rmem_tensor(32, self.dtype)
                        packed.store(carry.to(self.dtype))
                        _tcgen05.st(
                            warp * 32, operand_tmem + i * 16, "32x32b", 16, packed
                        )
                    _tcgen05.wait_st()
                    _tcgen05.fence_before_thread_sync()
                    cute.arch.mbarrier_arrive(packed_h + stage)
                cute.arch.mbarrier_wait(input_ready + stage, phase)
                # Decay is per key channel, and remains FP32 before the update.
                for i in cutlass.range_constexpr(self.key_tile // TMEM_VECTOR):
                    scaled = cute.make_rmem_tensor(32, Float32)
                    scaled.store(_tcgen05.ld(warp * 32, h_tmem + i * 32, "32x32b", 32))
                    for j in cutlass.range_constexpr(32):
                        scaled[j] *= decay[i * 32 + j, stage]
                    _tcgen05.st(warp * 32, h_tmem + i * 32, "32x32b", 32, scaled)
                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(update_ready + stage)
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1
            if chunks > 0:
                previous = (stage + stages - 1) % stages
                previous_phase = phase ^ Int32(stage == 0)
                cute.arch.mbarrier_wait(update_done + previous, previous_phase)
                _tcgen05.fence_after_thread_sync()
            out_seq = Int64(seq)
            if cutlass.const_expr(self.mapped):
                out_seq = Int64(indices[seq])
            # All four state-handler warps own distinct useful value channels.
            for i in cutlass.range_constexpr(self.key_tile // TMEM_VECTOR):
                final = _tcgen05.ld(warp * 32, h_tmem + i * 32, "32x32b", 32)
                for j in cutlass.range_constexpr(32):
                    key = i * TMEM_VECTOR + j
                    global_value = value_block * VALUE_TILE + value
                    final_value = final[j]
                    if key < self.key_dim and global_value < self.value_dim:
                        out[out_seq, head, key, global_value] = final_value

        else:
            # Four warps own all V128 channels of the unchanged M128 datapath.
            rwarp = warp - 4
            value = tid - 128
            stage = 0
            phase = 0
            for chunk in range(chunks):
                cute.arch.mbarrier_wait(projection_done + stage, phase)
                _tcgen05.fence_after_thread_sync()
                # Explicit U - projection. Do not seed an MMA accumulator with U.
                for i in cutlass.range_constexpr(self.time_tile // TMEM_VECTOR):
                    projection = _tcgen05.ld(rwarp * 32, p_tmem + i * 32, "32x32b", 32)
                    residual = cute.make_rmem_tensor(32, self.dtype)
                    for j in cutlass.range_constexpr(32):
                        residual[j] = self.dtype(
                            Float32(
                                su[
                                    i * 32 + j,
                                    (value % self.smem_group, value // self.smem_group),
                                    stage,
                                ]
                            )
                            - projection[j]
                        )
                    _tcgen05.st(
                        rwarp * 32,
                        operand_tmem + i * (32 if self.is_fp32 else 16),
                        "32x32b",
                        32 if self.is_fp32 else 16,
                        residual,
                    )
                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(update_ready + stage)
                stage = (stage + 1) % stages
                if stage == 0:
                    phase ^= 1

        # All final global stores and all asynchronous MMA consumers are done.
        cute.arch.sync_threads()
        if warp == 8:
            deallocate_tmem(base, self.tmem_columns)

    @staticmethod
    @cache
    def compile(
        heads,
        key_dim,
        value_dim,
        chunk_size,
        operand_dtype,
        gate_dtype,
        cu_dtype,
        index_dtype,
        mapped,
        use_tma,
        stages=2,
    ):
        tokens, entries, outputs = cute.sym_int(), cute.sym_int(), cute.sym_int()
        index_entries = cute.sym_int()

        def fake(dtype, shape):
            if use_tma:
                return make_fake_tensor(dtype, shape, divisibility=16)
            return cute.runtime.make_fake_tensor(
                dtype,
                shape,
                stride=tuple(cute.sym_int64() for _ in shape),
                assumed_align=dtype.width // 8,
            )

        kg = fake(operand_dtype, (1, tokens, heads, key_dim))
        u = fake(operand_dtype, (1, tokens, heads, value_dim))
        w = fake(operand_dtype, (1, tokens, heads, key_dim))
        gk = fake(gate_dtype, (1, tokens, heads, key_dim))
        out = fake(Float32, (outputs, heads, key_dim, value_dim + key_dim))
        cu = cute.runtime.make_fake_tensor(
            cu_dtype,
            (entries,),
            stride=(cute.sym_int64(),),
            assumed_align=cu_dtype.width // 8,
        )
        indices = cute.runtime.make_fake_tensor(
            index_dtype,
            (index_entries,),
            stride=(cute.sym_int64(),),
            assumed_align=index_dtype.width // 8,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            KcpSSummary(
                heads,
                key_dim,
                value_dim,
                chunk_size,
                operand_dtype,
                mapped,
                use_tma,
                stages,
            ),
            kg,
            u,
            w,
            gk,
            out,
            cu,
            indices,
            stream,
            options="--enable-tvm-ffi",
        )


def kcp_summary_s_cutedsl(kg, u, w, gk, cu, out, indices=None, stages=2, chunk_size=64):
    """Write S only, preserving FP32 carry and input-dtype rounding boundaries.

    Token counts, request-segment counts, offsets and all tensor strides remain
    runtime values. K/V/chunk geometry and element types specialize the kernel.
    Empty tensors use masked loads, so no zero-extent TMA descriptor is built.
    """
    types = {
        torch.bfloat16: BFloat16,
        torch.float16: Float16,
        torch.float32: Float32,
        torch.int32: Int32,
        torch.int64: Int64,
    }
    if kg.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("S summary tensor operands must be BF16, FP16 or FP32")
    if w.dtype != kg.dtype or u.dtype != kg.dtype:
        raise ValueError("S summary requires matching operand element types")
    key_dim, value_dim = kg.shape[-1], u.shape[-1]
    if key_dim <= 0 or value_dim <= 0:
        raise ValueError("S summary dimensions must be positive")
    if chunk_size < MMA_REDUCTION or chunk_size & (chunk_size - 1):
        raise ValueError("S summary chunks must be power-of-two MMA reduction tiles")
    key_tile = (key_dim + KEY_GROUP - 1) // KEY_GROUP * KEY_GROUP
    time_tile = (chunk_size + TMEM_VECTOR - 1) // TMEM_VECTOR * TMEM_VECTOR
    # TMA's swizzled split dimensions and stride/alignment requirements.
    use_tma = (
        kg.dtype != torch.float32
        and kg.shape[1] > 0
        and key_dim % KEY_GROUP == 0
        and value_dim % KEY_GROUP == 0
        and chunk_size == time_tile
        and all(
            x.stride(-1) == 1
            and x.data_ptr() % 64 == 0
            and all(s % 16 == 0 for s in x.stride()[:-1])
            for x in (kg, u, w, gk, out)
        )
    )
    if cu.numel() <= 1:
        return None
    # SM100 has 227 KiB of usable per-CTA shared memory. Derive the stage
    # count from the physical tile storage, rather than model dimensions.
    stage_bytes = (
        time_tile * (2 * key_tile + VALUE_TILE) * kg.element_size() + key_tile * 4 + 56
    )
    stages = min(stages, (227 * 1024 - 128) // stage_bytes)
    if stages < 1:
        raise ValueError("S summary tile exceeds per-CTA shared-memory capacity")
    index_tensor = cu if indices is None else indices
    compiled = KcpSSummary.compile(
        kg.shape[2],
        key_dim,
        value_dim,
        chunk_size,
        types[kg.dtype],
        types[gk.dtype],
        types[cu.dtype],
        types[index_tensor.dtype],
        indices is not None,
        use_tma,
        stages,
    )
    compiled(kg, u, w, gk, out, cu, index_tensor)
    return compiled


@dsl_user_op
def alloc128(taddr, *, loc=None, ip=None):
    nvvm.tcgen05_alloc(
        taddr.to_llvm_ptr(loc=loc, ip=ip),
        Uint32(128).ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def dealloc128(base, *, loc=None, ip=None):
    nvvm.tcgen05_dealloc(
        _tcgen05._make_tmem_llvm_ptr(base, loc=loc, ip=ip),
        Int32(128).ir_value(loc=loc, ip=ip),
        group=nvvm.CTAGroupKind.CTA_1,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ieee_fma2(a, b0, b1, c0, c1, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b0, b1, c0, c1)],
        "{\n.reg .b64 aa, bb, cc, dd;\n"
        "mov.b64 aa, {$2, $2};\nmov.b64 bb, {$3, $4};\n"
        "mov.b64 cc, {$5, $6};\nfma.rn.f32x2 dd, aa, bb, cc;\n"
        "mov.b64 {$0, $1}, dd;\n}",
        "=f,=f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def load_shared4(pointer, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Uint32(pointer.toint()).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def store_shared4(pointer, a, b, c, d, *, loc=None, ip=None):
    llvm.inline_asm(
        T.i32(),
        [Uint32(pointer.toint()).ir_value(loc=loc, ip=ip)]
        + [Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c, d)],
        "st.shared.v4.b32 [$1], {$2, $3, $4, $5};\nmov.u32 $0, 0;",
        "=r,r,f,f,f,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_tf32(d_tmem, a_desc, b_desc, idesc, enable_input_d, *, loc=None, ip=None):
    with cute.arch.elect_one():
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.TF32,
            nvvm.CTAGroupKind.CTA_1,
            _tcgen05._make_tmem_llvm_ptr(d_tmem, loc=loc, ip=ip),
            Uint64(a_desc).ir_value(loc=loc, ip=ip),
            Uint64(b_desc).ir_value(loc=loc, ip=ip),
            Int32(idesc).ir_value(loc=loc, ip=ip),
            Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


class KcpMTransitionPipeline:
    """Two transition slots and an ordered IEEE state chain per column tile."""

    def __init__(self, heads, keys, values, dtype, mapped, use_tma, chunk_size):
        self.heads = heads
        self.keys = keys
        self.values = values
        self.dtype = dtype
        self.mapped = mapped
        self.use_tma = use_tma
        self.chunk_size = chunk_size
        self.block_tokens = min(64, (chunk_size + 15) // 16 * 16)
        if dtype == Float32:
            self.block_tokens = max(32, self.block_tokens)
        self.input_slabs = (chunk_size + self.block_tokens - 1) // self.block_tokens
        self.key_extent = 1 << (keys - 1).bit_length()
        self.key_panels = (keys + 127) // 128
        self.panels = self.key_panels * self.key_panels
        self.columns = 64 if keys <= 128 else 32
        if dtype == Float32:
            self.columns //= 2
        self.keys_per_thread = 4 if dtype == Float32 else 8
        self.state_keys = 128 if keys <= 128 else 256

    @cute.jit
    def input_layout(self):
        if cutlass.const_expr(self.dtype == Float32):
            return cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3),
                0,
                cute.make_layout(
                    (128, 1, (32, self.block_tokens // 32)),
                    stride=(32, 0, (1, 128 * 32)),
                ),
            )
        return cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.block_tokens, 1, (64, 2)),
                stride=(64, 0, (1, 64 * self.block_tokens)),
            ),
        )

    @cute.jit
    def make_tma(self, tensor):
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(tensor, (None, None, 64)),
            self.input_layout(),
            cta_tiler=(self.block_tokens, 1, 128),
        )

    @cute.jit
    def __call__(self, kg, w, gk, cu, out, indices, stream: CUstream):
        kg_tma, w_tma = None, None
        if cutlass.const_expr(self.use_tma):
            kg_tma = self.make_tma(kg[0, None, None, None])
            w_tma = self.make_tma(w[0, None, None, None])
        self.kcp_summary_m_transition_pipeline(
            kg[0, None, None, None],
            w[0, None, None, None],
            kg_tma,
            w_tma,
            gk[0, None, None, None],
            cu,
            out,
            indices,
        ).launch(
            grid=(
                cute.ceil_div(self.keys, self.columns) * self.heads * (cu.shape[0] - 1),
                1,
                1,
            ),
            block=(416, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kcp_summary_m_transition_pipeline(
        self,
        kg: cute.Tensor,
        w: cute.Tensor,
        kg_tma,
        w_tma,
        gk: cute.Tensor,
        cu: cute.Tensor,
        out: cute.Tensor,
        indices: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        column_tiles = cute.ceil_div(self.keys, self.columns)
        column_tile = block % column_tiles
        head = (block // column_tiles) % self.heads
        seq = block // (column_tiles * self.heads)
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        layout = self.input_layout()

        def allocate_input():
            return smem.allocate_tensor(
                self.dtype,
                layout.outer,
                byte_alignment=128,
                swizzle=layout.inner,
            )[None, 0, None]

        skg, sw = allocate_input(), allocate_input()
        transition = smem.allocate_tensor(
            Float32,
            cute.make_layout((128, 128, 2), stride=(132, 1, 128 * 132)),
            byte_alignment=128,
        )
        previous = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.state_keys, self.columns), stride=(self.columns, 1)),
            byte_alignment=128,
        )
        decay = smem.allocate_tensor(Float32, cute.make_layout(128))
        load_done = smem.allocate_array(Int64, 1)
        mma_done = smem.allocate_array(Int64, 2)
        published = smem.allocate_array(Int64, 2)
        slot_free = smem.allocate_array(Int64, 2)
        partial_done = None
        if cutlass.const_expr(self.input_slabs > 1):
            partial_done = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(load_done, 1)
                if cutlass.const_expr(self.input_slabs > 1):
                    cute.arch.mbarrier_init(partial_done, 1)
                for slot in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(mma_done + slot, 1)
                    cute.arch.mbarrier_init(published + slot, 128)
                    cute.arch.mbarrier_init(slot_free + slot, 256)
                cute.arch.mbarrier_init_fence()
        elif warp == 12:
            alloc128(taddr)
            if cutlass.const_expr(self.use_tma):
                cpasync.prefetch_descriptor(kg_tma.atom)
                cpasync.prefetch_descriptor(w_tma.atom)
        cute.arch.sync_threads()

        tmem = cute.make_tensor(taddr, cute.make_layout(1))[0]
        bos, eos = Int64(cu[seq]), Int64(cu[seq + 1])
        chunks = cute.ceil_div(eos - bos, self.chunk_size)
        if warp == 12:
            if cutlass.const_expr(self.use_tma):
                k_tiles = cute.logical_divide(
                    cute.domain_offset(
                        (bos, (0, 0)), kg_tma.tma_tensor[None, head, None]
                    ),
                    (self.block_tokens, None),
                )
                w_tiles = cute.logical_divide(
                    cute.domain_offset(
                        (bos, (0, 0)), w_tma.tma_tensor[None, head, None]
                    ),
                    (self.block_tokens, None),
                )
            stage, phase = 0, 0
            idesc = _tcgen05.make_bf16_idesc(
                128,
                128,
                transpose_A=self.dtype != Float32,
                transpose_B=self.dtype != Float32,
            )
            if cutlass.const_expr(self.dtype == Float16):
                idesc ^= Uint32((1 << 7) | (1 << 10))
            if cutlass.const_expr(self.dtype == Float32):
                idesc ^= Uint32((3 << 7) | (3 << 10))
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
            else:
                sdesc = _tcgen05.make_sdesc_128B_swizzle(self.block_tokens * 128)
            for step in range(chunks * self.panels):
                chunk = step // self.panels
                panel = step % self.panels
                row_panel, reduction_panel = (
                    panel // self.key_panels,
                    panel % self.key_panels,
                )
                cute.arch.mbarrier_wait(slot_free + stage, phase ^ 1)
                if step > 0:
                    old_stage = stage ^ 1
                    old_phase = phase ^ Int32(stage == 0)
                    cute.arch.mbarrier_wait(published + old_stage, old_phase)
                for operand_slab in range(self.input_slabs):
                    if cutlass.const_expr(self.use_tma):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                load_done, 4 * self.block_tokens * 128
                            )
                        simple_tma_copy(
                            kg_tma.atom,
                            k_tiles[
                                (None, chunk * self.input_slabs + operand_slab), None
                            ],
                            skg,
                            load_done,
                        )
                        simple_tma_copy(
                            w_tma.atom,
                            w_tiles[
                                (None, chunk * self.input_slabs + operand_slab), None
                            ],
                            sw,
                            load_done,
                        )
                    else:
                        for linear in range(lane, self.block_tokens * 128, 32):
                            token, key = linear // 128, linear % 128
                            token_index = (
                                bos
                                + chunk * self.chunk_size
                                + operand_slab * self.block_tokens
                                + token
                            )
                            kg_key, w_key = (
                                row_panel * 128 + key,
                                reduction_panel * 128 + key,
                            )
                            a, b = self.dtype(0), self.dtype(0)
                            if (
                                operand_slab * self.block_tokens + token
                                < self.chunk_size
                                and token_index < eos
                            ):
                                if kg_key < self.keys:
                                    a = kg[token_index, head, kg_key]
                                if w_key < self.keys:
                                    b = w[token_index, head, w_key]
                            if cutlass.const_expr(self.dtype == Float32):
                                # Match the retained TF32 MMA: pass raw FP32 bits,
                                # without an inserted BF16 or TF32 rounding cast.
                                skg[key, (token % 32, token // 32)] = a
                                sw[key, (token % 32, token // 32)] = b
                            else:
                                skg[token, (key % 64, key // 64)] = a
                                sw[token, (key % 64, key // 64)] = b
                    last = min(bos + (chunk + 1) * self.chunk_size, eos) - 1
                    for i in cutlass.range_constexpr(4):
                        key = lane + i * 32
                        value = Float32(1)
                        if row_panel * 128 + key < self.keys:
                            value = cute.math.exp2(
                                Float32(gk[last, head, row_panel * 128 + key]),
                                fastmath=True,
                            )
                        decay[key] = value
                    if cutlass.const_expr(self.use_tma):
                        cute.arch.mbarrier_wait(
                            load_done, (step * self.input_slabs + operand_slab) % 2
                        )
                        valid = min(
                            Int64(self.block_tokens),
                            eos
                            - bos
                            - chunk * self.chunk_size
                            - operand_slab * self.block_tokens,
                        )
                        if valid < self.block_tokens:
                            for i in range(lane, self.block_tokens * 128, 32):
                                token, key = i // 128, i % 128
                                if token >= valid:
                                    skg[token, (key % 64, key // 64)] = self.dtype(0)
                                    sw[token, (key % 64, key // 64)] = self.dtype(0)
                    fence_before_tma_store()
                    cute.arch.sync_warp()
                    _tcgen05.fence_after_thread_sync()
                    adesc = sdesc | (skg.iterator.toint() >> 4)
                    bdesc = sdesc | (sw.iterator.toint() >> 4)
                    if cutlass.const_expr(self.dtype == Float32):
                        for k8 in cutlass.range_constexpr(self.block_tokens // 8):
                            offset = (k8 % 4) * 2 + (k8 // 4) * 1024
                            mma_tf32(
                                tmem,
                                adesc + offset,
                                bdesc + offset,
                                idesc,
                                operand_slab > 0 or k8 > 0,
                            )
                    else:
                        for k16 in cutlass.range_constexpr(self.block_tokens // 16):
                            offset = (k16 * 16 * 128) >> 4
                            _tcgen05.mma_f16(
                                tmem,
                                adesc + offset,
                                bdesc + offset,
                                idesc,
                                operand_slab > 0 or k16 > 0,
                            )
                    if cutlass.const_expr(self.input_slabs > 1):
                        if operand_slab + 1 < self.input_slabs:
                            _tcgen05.commit(partial_done)
                            cute.arch.mbarrier_wait(
                                partial_done,
                                (step * (self.input_slabs - 1) + operand_slab) % 2,
                            )
                        else:
                            _tcgen05.commit(mma_done + stage)
                    else:
                        _tcgen05.commit(mma_done + stage)
                stage ^= 1
                if stage == 0:
                    phase ^= 1

        elif warp >= 8:
            publication_warp = warp - 8
            key_out = tid - 256
            stage, phase = 0, 0
            for step in range(chunks * self.panels):
                panel = step % self.panels
                row_panel, reduction_panel = (
                    panel // self.key_panels,
                    panel % self.key_panels,
                )
                cute.arch.mbarrier_wait(mma_done + stage, phase)
                _tcgen05.fence_after_thread_sync()
                own_decay = decay[key_out]
                for group in cutlass.range_constexpr(4):
                    products = _tcgen05.ld(
                        publication_warp * 32, tmem + group * 32, "32x32b", 32
                    )
                    for i in cutlass.range_constexpr(32):
                        reduction = group * 32 + i
                        diagonal = Float32(0)
                        if (
                            row_panel * 128 + key_out
                            == reduction_panel * 128 + reduction
                        ):
                            diagonal = own_decay
                        transition[reduction, key_out, stage] = diagonal - products[i]
                _tcgen05.wait_ld()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(published + stage)
                stage ^= 1
                if stage == 0:
                    phase ^= 1

        else:
            if cutlass.const_expr(self.dtype == Float32):
                if cutlass.const_expr(self.key_panels == 1):
                    key_base = warp * 16 + (lane // 8) * 4
                    col_base = (lane % 8) * 4
                    owned_panel = Int32(0)
                else:
                    key_base = (warp % 4) * 32 + (lane // 4) * 4
                    col_base = (lane % 4) * 4
                    owned_panel = warp // 4
            else:
                if cutlass.const_expr(self.key_panels == 1):
                    key_base = warp * 16 + (lane // 16) * 8
                    col_base = (lane % 16) * 4
                    owned_panel = Int32(0)
                else:
                    key_base = (warp % 4) * 32 + (lane // 8) * 8
                    col_base = (lane % 8) * 4
                    owned_panel = warp // 4
            for linear in range(tid, self.state_keys * self.columns, 256):
                key, col = linear // self.columns, linear % self.columns
                previous[key, col] = Float32(key == col + column_tile * self.columns)
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            stage, phase = 0, 0
            if chunks > 0:
                # Seed the transfer with the first transition; its input is identity.
                for panel in range(self.panels):
                    row_panel, reduction_panel = (
                        panel // self.key_panels,
                        panel % self.key_panels,
                    )
                    cute.arch.mbarrier_wait(published + stage, phase)
                    if owned_panel == row_panel:
                        for i in cutlass.range_constexpr(self.keys_per_thread):
                            for j in cutlass.range_constexpr(4):
                                column = column_tile * self.columns + col_base + j
                                if (
                                    column >= reduction_panel * 128
                                    and column < (reduction_panel + 1) * 128
                                ):
                                    value = transition[
                                        column % 128, key_base + i, stage
                                    ]
                                    value, unused = ieee_fma2(
                                        value,
                                        Float32(1),
                                        Float32(1),
                                        Float32(0),
                                        Float32(0),
                                    )
                                    previous[
                                        owned_panel * 128 + key_base + i, col_base + j
                                    ] = value
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    cute.arch.mbarrier_arrive(slot_free + stage)
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    stage ^= 1
                    if stage == 0:
                        phase ^= 1
            for chunk in range(1, chunks):
                accum = cute.make_rmem_tensor(
                    cute.make_layout((self.keys_per_thread, 4), stride=(4, 1)), Float32
                )
                accum.fill(0)
                for panel in range(self.panels):
                    row_panel, reduction_panel = (
                        panel // self.key_panels,
                        panel % self.key_panels,
                    )
                    cute.arch.mbarrier_wait(published + stage, phase)
                    if owned_panel == row_panel:
                        # Each output accumulates k=0..padded_K-1 without partial sums.
                        for reduction in range(min(128, self.key_extent)):
                            b0, b1, b2, b3 = load_shared4(
                                previous.iterator
                                + (reduction_panel * 128 + reduction) * self.columns
                                + col_base
                            )
                            for group in cutlass.range_constexpr(
                                self.keys_per_thread // 4
                            ):
                                a0, a1, a2, a3 = load_shared4(
                                    transition.iterator
                                    + stage * 128 * 132
                                    + reduction * 132
                                    + key_base
                                    + group * 4
                                )
                                for i in cutlass.range_constexpr(4):
                                    a = (a0, a1, a2, a3)[i]
                                    index = group * 4 + i
                                    accum[index, 0], accum[index, 1] = ieee_fma2(
                                        a, b0, b1, accum[index, 0], accum[index, 1]
                                    )
                                    accum[index, 2], accum[index, 3] = ieee_fma2(
                                        a, b2, b3, accum[index, 2], accum[index, 3]
                                    )
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    if panel == self.panels - 1:
                        for i in cutlass.range_constexpr(self.keys_per_thread):
                            store_shared4(
                                previous.iterator
                                + (owned_panel * 128 + key_base + i) * self.columns
                                + col_base,
                                accum[i, 0],
                                accum[i, 1],
                                accum[i, 2],
                                accum[i, 3],
                            )
                    cute.arch.mbarrier_arrive(slot_free + stage)
                    cute.arch.barrier(barrier_id=1, number_of_threads=256)
                    stage ^= 1
                    if stage == 0:
                        phase ^= 1
            output_seq = Int64(seq)
            if cutlass.const_expr(self.mapped):
                output_seq = Int64(indices[seq])
            for linear in range(tid, self.state_keys * self.columns, 256):
                key, col = linear // self.columns, linear % self.columns
                column = column_tile * self.columns + col
                if key < self.keys and column < self.keys:
                    out[output_seq, head, key, self.values + column] = previous[
                        key, col
                    ]

        cute.arch.sync_threads()
        if warp == 12:
            dealloc128(tmem)

    @staticmethod
    @cache
    def compile(
        heads,
        keys,
        values,
        dtype,
        gate_dtype,
        cu_dtype,
        index_dtype,
        mapped,
        use_tma,
        chunk_size,
    ):
        tokens, entries = cute.sym_int(), cute.sym_int()
        outputs, index_entries = cute.sym_int(), cute.sym_int()

        def tensor(element, shape, alignment=16):
            if use_tma:
                return make_fake_tensor(element, shape, divisibility=alignment)
            return cute.runtime.make_fake_tensor(
                element,
                shape,
                stride=tuple(cute.sym_int64() for _ in shape),
                assumed_align=element.width // 8,
            )

        kg = tensor(dtype, (1, tokens, heads, keys))
        w = tensor(dtype, (1, tokens, heads, keys))
        gk = tensor(gate_dtype, (1, tokens, heads, keys))
        cu = tensor(cu_dtype, (entries,), 1)
        out = tensor(Float32, (outputs, heads, keys, values + keys))
        indices = tensor(index_dtype, (index_entries,), 1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            KcpMTransitionPipeline(
                heads, keys, values, dtype, mapped, use_tma, chunk_size
            ),
            kg,
            w,
            gk,
            cu,
            out,
            indices,
            stream,
            options="--enable-tvm-ffi",
        )


def kcp_summary_m_cutedsl(kg, w, gk, cu, out, indices=None, chunk_size=64):
    """Write M in out[..., V:] without reading or changing S_ext."""
    types = {
        torch.bfloat16: BFloat16,
        torch.float16: Float16,
        torch.float32: Float32,
        torch.int32: Int32,
        torch.int64: Int64,
    }
    if (
        kg.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or w.dtype != kg.dtype
    ):
        raise ValueError("M operands must have the same BF16, FP16 or FP32 dtype")
    if not 0 < kg.shape[-1] <= 256 or chunk_size <= 0:
        raise ValueError("M requires 1 <= K <= 256 and positive chunk_size")
    if cu.numel() <= 1:
        return None
    keys = kg.shape[-1]
    values = out.shape[-1] - keys
    index_tensor = cu if indices is None else indices
    tensors = (kg, w, gk, cu, out, index_tensor)
    use_tma = (
        kg.dtype != torch.float32
        and kg.shape[1] > 0
        and keys == 128
        and chunk_size >= 16
        and chunk_size & (chunk_size - 1) == 0
        and all(t.is_contiguous() for t in tensors)
        and all(t.data_ptr() % 16 == 0 for t in (kg, w, gk, out))
        and all(
            stride % 16 == 0 for t in (kg, w, gk, out) for stride in t.stride()[:-1]
        )
    )
    compiled = KcpMTransitionPipeline.compile(
        kg.shape[2],
        keys,
        values,
        types[kg.dtype],
        types[gk.dtype],
        types[cu.dtype],
        types[index_tensor.dtype],
        indices is not None,
        use_tma,
        chunk_size,
    )
    compiled(kg, w, gk, cu, out, index_tensor)
    return compiled
