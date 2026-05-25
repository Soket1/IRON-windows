// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Combined kernel for the Phase B post-attention fused layer dispatch.
//
// Provides all kernel entry points used in the fused runlist:
//   fused_dequant_matvec_v2_bf16  — O_proj and Down GEMV (reused from v2)
//   post_attn_add_bf16            — elementwise ADD: inpFF = O_proj_out + inpL
//   post_attn_rms_norm_bf16       — RMSNorm: normed = RMSNorm(inpFF)
//   post_attn_gain_mul_bf16       — elementwise MUL: ffn_in = normed * gain
//   dual_fused_dequant_gemv_bf16  — SwiGLU gate/up GEMV writing to static buf
//   dual_fused_dequant_gemv_silu_mul_bf16 — silu(left_buf) * right_buf → out
//
// This single .cc file compiles into one .o that is used by both the
// fused_dequant_gemv_v2 workers (for O_proj/Down) and the dual-GEMV workers
// (for SwiGLU gate/up + silu_mul).

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// ────────────────────────────────────────────────────────────────────────────
// Compile-time parameters (provided by IRON design.py via -D flags).
// ────────────────────────────────────────────────────────────────────────────

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

#ifndef DIM_K
#define DIM_K 2048
#endif

#ifndef M_OUTPUT_MAX
#define M_OUTPUT_MAX 4096
#endif

#ifndef EMBED_DIM
#define EMBED_DIM 2048
#endif

// ────────────────────────────────────────────────────────────────────────────
// Static L1 buffers for dual-GEMV SwiGLU phases.
// ────────────────────────────────────────────────────────────────────────────

static bfloat16 left_buf[M_OUTPUT_MAX]  __attribute__((aligned(64)));
static bfloat16 right_buf[M_OUTPUT_MAX] __attribute__((aligned(64)));

// ────────────────────────────────────────────────────────────────────────────
// 1. INT4 GEMV (O_proj and Down): fused dequant + matvec, V2 double-pump.
//    Identical to fused_dequant_gemv_v2.cc::fused_dequant_matvec but renamed.
// ────────────────────────────────────────────────────────────────────────────

template <uint32_t block_size, uint32_t G, uint32_t DK>
static void _gemv_v2(uint32_t m,
                     const uint8_t *__restrict a_in,
                     const bfloat16 *__restrict b_in,
                     bfloat16 *__restrict c_out)
{
    static_assert(block_size == 32, "block_size must be 32");
    static_assert(G % block_size == 0, "group_size must be multiple of block_size");
    constexpr uint32_t blocks_per_group  = G / block_size;
    constexpr uint32_t groups_per_row    = DK / G;
    constexpr bool can_double_pump       = (groups_per_row >= 2) && (groups_per_row % 2 == 0);
    constexpr uint32_t pump_groups       = can_double_pump ? 2 : 1;
    constexpr uint32_t loop_iters        = groups_per_row / pump_groups;

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const uint4 *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(a_in + m * DK / 2);

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *row_weights = weights_packed + row * DK / 2;
        const bfloat16 *row_scales = scales + row * groups_per_row;
        const bfloat16 *b_ptr = b_in;

        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();

        if constexpr (can_double_pump && blocks_per_group == 1) {
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g += 2)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf_a = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_a_bc =
                        aie::broadcast<bfloat16, block_size>(sf_a);
                    aie::vector<uint4, block_size> I0_a =
                        aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;

                    bfloat16 sf_b = row_scales[g + 1];
                    aie::vector<bfloat16, block_size> sf_b_bc =
                        aie::broadcast<bfloat16, block_size>(sf_b);
                    aie::vector<uint4, block_size> I0_b =
                        aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;

                    aie::vector<uint8,  block_size> a8_a  = aie::unpack(I0_a);
                    aie::vector<uint16, block_size> a16_a = aie::unpack(a8_a);
                    aie::vector<bfloat16, block_size> abf_a =
                        aie::to_float<bfloat16>(a16_a, 0);
                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(abf_a, sf_a_bc).template to_vector<bfloat16>();

                    aie::vector<uint8,  block_size> a8_b  = aie::unpack(I0_b);
                    aie::vector<uint16, block_size> a16_b = aie::unpack(a8_b);
                    aie::vector<bfloat16, block_size> abf_b =
                        aie::to_float<bfloat16>(a16_b, 0);
                    aie::vector<bfloat16, block_size> w_b =
                        aie::mul(abf_b, sf_b_bc).template to_vector<bfloat16>();

                    aie::vector<bfloat16, block_size> b_a = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_a, b_a);

                    aie::vector<bfloat16, block_size> b_b = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_b, b_b);
                }
        } else {
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g++)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_bc =
                        aie::broadcast<bfloat16, block_size>(sf);

                    AIE_LOOP_MIN_ITERATION_COUNT(blocks_per_group)
                    for (uint32_t blk = 0; blk < blocks_per_group; blk++) {
                        aie::vector<uint4, block_size> I0 = aie::load_v<block_size>(row_weights);
                        row_weights += block_size / 2;

                        aie::vector<uint8,  block_size> a8  = aie::unpack(I0);
                        aie::vector<uint16, block_size> a16 = aie::unpack(a8);
                        aie::vector<bfloat16, block_size> abf =
                            aie::to_float<bfloat16>(a16, 0);
                        aie::vector<bfloat16, block_size> w =
                            aie::mul(abf, sf_bc).template to_vector<bfloat16>();

                        aie::vector<bfloat16, block_size> bv = aie::load_v<block_size>(b_ptr);
                        b_ptr += block_size;
                        acc = aie::mac(acc, w, bv);
                    }
                }
        }

        c_out[row] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

extern "C" void fused_dequant_matvec_v2_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _gemv_v2<32, GROUP_SIZE, DIM_K>(m, a + row_offset * (DIM_K/2 + DIM_K/GROUP_SIZE*2),
                                    b, c);
}

// Down projection entry (same kernel, different B type: hidden_dim elements)
extern "C" void fused_dequant_matvec_down_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _gemv_v2<32, GROUP_SIZE, DIM_K>(m, a + row_offset * (DIM_K/2 + DIM_K/GROUP_SIZE*2),
                                    b, c);
}

// ────────────────────────────────────────────────────────────────────────────
// 2. Elementwise ADD: inpFF[i] = o_proj_out[i] + inpL[i]
// ────────────────────────────────────────────────────────────────────────────

extern "C" void post_attn_add_bf16(
        const bfloat16 *a, const bfloat16 *b, bfloat16 *c, int32_t n) {
    constexpr int VEC = 16;
    int chunks = n / VEC;
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        auto va = ::aie::load_v<VEC>(a + i * VEC);
        auto vb = ::aie::load_v<VEC>(b + i * VEC);
        ::aie::store_v(c + i * VEC, ::aie::add(va, vb));
    }
    for (int i = chunks * VEC; i < n; i++) {
        c[i] = (bfloat16)((float)a[i] + (float)b[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 3. Weighted RMSNorm: normed[i] = (inpFF[i] / rms(inpFF)) * gain[i]
//    epsilon = 1e-5 (hardcoded, matches existing rms_norm.cc)
// ────────────────────────────────────────────────────────────────────────────

extern "C" void post_attn_rms_norm_bf16(
        const bfloat16 *input, const bfloat16 *gain, bfloat16 *output, int32_t n) {
    constexpr float eps = 1e-5f;
    constexpr int VEC = 16;

    // Pass 1: compute sum of squares (use float vector accumulation)
    ::aie::vector<float, VEC> acc = ::aie::zeros<float, VEC>();
    int chunks = n / VEC;
    for (int i = 0; i < chunks; i++) {
        ::aie::vector<bfloat16, VEC> v = ::aie::load_v<VEC>(input + i * VEC);
        ::aie::vector<float, VEC> sq = ::aie::mul_square(v);
        acc = ::aie::add(acc, sq);
    }
    float sum_sq = ::aie::reduce_add(acc);
    for (int i = chunks * VEC; i < n; i++) {
        float x = (float)input[i];
        sum_sq += x * x;
    }

    float inv_rms = aie::invsqrt(sum_sq / n + eps);

    // Pass 2: normalize and apply gain (scalar — fast enough for embed_dim)
    for (int i = 0; i < n; i++) {
        output[i] = (bfloat16)((float)input[i] * inv_rms * (float)gain[i]);
    }
    (void)chunks;
}

// ────────────────────────────────────────────────────────────────────────────
// 4. Elementwise MUL: ffn_in[i] = normed[i] * gain_weight[i]
//    (kept separate from RMSNorm for design flexibility)
// ────────────────────────────────────────────────────────────────────────────

extern "C" void post_attn_gain_mul_bf16(
        const bfloat16 *a, const bfloat16 *b, bfloat16 *c, int32_t n) {
    constexpr int VEC = 16;
    int chunks = n / VEC;
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        auto va = ::aie::load_v<VEC>(a + i * VEC);
        auto vb = ::aie::load_v<VEC>(b + i * VEC);
        ::aie::store_v(c + i * VEC,
                       ::aie::mul(va, vb).template to_vector<bfloat16>());
    }
    for (int i = chunks * VEC; i < n; i++) {
        c[i] = (bfloat16)((float)a[i] * (float)b[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 5+6. SwiGLU dual-GEMV + SiLU*Mul (from dual_fused_dequant_gemv_silu_mul.cc)
// ────────────────────────────────────────────────────────────────────────────

template <uint32_t block_size, uint32_t G, uint32_t DK>
static void _dual_gemv(uint32_t m, uint32_t row_offset,
                       const uint8_t *__restrict a_in,
                       const bfloat16 *__restrict b_in,
                       int phase)
{
    static_assert(block_size == 32, "block_size must be 32");
    static_assert(G % block_size == 0, "group_size must be multiple of block_size");
    constexpr uint32_t groups_per_row = DK / G;
    constexpr bool can_double_pump = (groups_per_row >= 2) && (groups_per_row % 2 == 0);
    constexpr uint32_t pump_groups  = can_double_pump ? 2 : 1;
    constexpr uint32_t loop_iters   = groups_per_row / pump_groups;

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    bfloat16 *dest = (phase == 0) ? left_buf : right_buf;
    const uint4 *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(a_in + m * DK / 2);

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *w_row = weights_packed + row * DK / 2;
        const bfloat16 *s_row = scales + row * groups_per_row;
        const bfloat16 *b_ptr = b_in;

        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();
        aie::vector<bfloat16, block_size> offset =
            aie::broadcast<bfloat16, block_size>(8.0f);

        if constexpr (can_double_pump) {
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g += 2)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf_a = s_row[g];
                    aie::vector<bfloat16, block_size> sf_a_bc =
                        aie::broadcast<bfloat16, block_size>(sf_a);
                    aie::vector<uint4, block_size> I0_a =
                        aie::load_v<block_size>(w_row);
                    w_row += block_size / 2;

                    bfloat16 sf_b = s_row[g + 1];
                    aie::vector<bfloat16, block_size> sf_b_bc =
                        aie::broadcast<bfloat16, block_size>(sf_b);
                    aie::vector<uint4, block_size> I0_b =
                        aie::load_v<block_size>(w_row);
                    w_row += block_size / 2;

                    aie::vector<uint8,  block_size> a8_a  = aie::unpack(I0_a);
                    aie::vector<uint16, block_size> a16_a = aie::unpack(a8_a);
                    aie::vector<bfloat16, block_size> abf_a =
                        aie::to_float<bfloat16>(a16_a, 0);
                    aie::vector<bfloat16, block_size> asgn_a =
                        aie::sub(abf_a, offset);
                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(asgn_a, sf_a_bc).template to_vector<bfloat16>();

                    aie::vector<uint8,  block_size> a8_b  = aie::unpack(I0_b);
                    aie::vector<uint16, block_size> a16_b = aie::unpack(a8_b);
                    aie::vector<bfloat16, block_size> abf_b =
                        aie::to_float<bfloat16>(a16_b, 0);
                    aie::vector<bfloat16, block_size> asgn_b =
                        aie::sub(abf_b, offset);
                    aie::vector<bfloat16, block_size> w_b =
                        aie::mul(asgn_b, sf_b_bc).template to_vector<bfloat16>();

                    aie::vector<bfloat16, block_size> b_a = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_a, b_a);

                    aie::vector<bfloat16, block_size> b_b = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_b, b_b);
                }
        } else {
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g++)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf = s_row[g];
                    aie::vector<bfloat16, block_size> sf_bc =
                        aie::broadcast<bfloat16, block_size>(sf);
                    aie::vector<uint4, block_size> I0 =
                        aie::load_v<block_size>(w_row);
                    w_row += block_size / 2;

                    aie::vector<uint8,  block_size> a8  = aie::unpack(I0);
                    aie::vector<uint16, block_size> a16 = aie::unpack(a8);
                    aie::vector<bfloat16, block_size> abf =
                        aie::to_float<bfloat16>(a16, 0);
                    aie::vector<bfloat16, block_size> asgn =
                        aie::sub(abf, offset);
                    aie::vector<bfloat16, block_size> w =
                        aie::mul(asgn, sf_bc).template to_vector<bfloat16>();

                    aie::vector<bfloat16, block_size> bv = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w, bv);
                }
        }
        dest[row_offset + row] = static_cast<bfloat16>(
            aie::reduce_add(acc.template to_vector<float>()));
    }
}

extern "C" void dual_fused_dequant_gemv_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, int phase) {
    _dual_gemv<32, GROUP_SIZE, DIM_K>(m, row_offset, a, b, phase);
}

extern "C" void dual_fused_dequant_gemv_silu_mul_bf16(
        bfloat16 *c_out, uint32_t m_output) {
    constexpr int VEC = 16;
    int chunks = (int)m_output / VEC;
    aie::vector<bfloat16, VEC> register_0_5 = aie::broadcast<bfloat16, VEC>(0.5f);
    aie::vector<bfloat16, VEC> register_1   = aie::broadcast<bfloat16, VEC>(1.0f);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        aie::vector<bfloat16, VEC> l = aie::load_v<VEC>(left_buf  + i * VEC);
        aie::vector<bfloat16, VEC> r = aie::load_v<VEC>(right_buf + i * VEC);
        // silu(x) = x * sigmoid(x), sigmoid(x) = 0.5 * (1 + tanh(x/2))
        auto half_x = aie::mul(l, register_0_5);
        auto tanh_half_x = aie::tanh<bfloat16>(half_x.template to_vector<float>());
        auto tanh_plus_1 = aie::add(tanh_half_x, register_1);
        aie::vector<bfloat16, VEC> sigmoid_approx =
            aie::mul(tanh_plus_1, register_0_5).template to_vector<bfloat16>();
        auto silu_out = aie::mul(l, sigmoid_approx);
        auto fused    = aie::mul(silu_out.template to_vector<bfloat16>(), r);
        aie::store_v(c_out + i * VEC, fused.template to_vector<bfloat16>());
    }
    // m_output = hidden_dim / cols is always a multiple of 16 — no scalar tail needed.
    (void)chunks;
}
