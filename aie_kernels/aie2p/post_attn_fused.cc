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
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(
        a_in + m * DK / 2);

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *w_row = weights_packed + row * DK / 2;
        const bfloat16 *s_row = scales + row * groups_per_row;

        ::aie::accum<accfloat, block_size> acc = ::aie::zeros<accfloat, block_size>();

        if constexpr (can_double_pump) {
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_MIN_ITERATION_COUNT(1)
            for (uint32_t g = 0; g < loop_iters; g++) {
                uint32_t g0 = g * 2, g1 = g0 + 1;
                auto w0 = ::aie::load_v<block_size>(w_row + g0 * block_size / 2);
                auto w1 = ::aie::load_v<block_size>(w_row + g1 * block_size / 2);
                auto w0f = ::aie::to_float(::aie::unpack(w0));
                auto w1f = ::aie::to_float(::aie::unpack(w1));
                auto scale0 = (float)s_row[g0];
                auto scale1 = (float)s_row[g1];
                auto b0 = ::aie::load_v<block_size>(b_in + g0 * block_size);
                auto b1 = ::aie::load_v<block_size>(b_in + g1 * block_size);
                auto d0 = ::aie::mul(::aie::mul(w0f, scale0), b0);
                auto d1 = ::aie::mul(::aie::mul(w1f, scale1), b1);
                acc = ::aie::add(acc, d0);
                acc = ::aie::add(acc, d1);
            }
        } else {
            AIE_PREPARE_FOR_PIPELINING
            for (uint32_t g = 0; g < groups_per_row; g++) {
                auto wvec = ::aie::load_v<block_size>(w_row + g * block_size / 2);
                auto wf   = ::aie::to_float(::aie::unpack(wvec));
                float scale = (float)s_row[g];
                auto bvec = ::aie::load_v<block_size>(b_in + g * block_size);
                acc = ::aie::add(acc, ::aie::mul(::aie::mul(wf, scale), bvec));
            }
        }

        c_out[row] = (bfloat16)::aie::reduce_add(::aie::to_float(acc));
    }
}

extern "C" void fused_dequant_matvec_v2_bf16(
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

    // Pass 1: compute sum of squares
    ::aie::vector<float, VEC> acc = ::aie::zeros<float, VEC>();
    int chunks = n / VEC;
    for (int i = 0; i < chunks; i++) {
        auto v = ::aie::load_v<VEC>(input + i * VEC);
        acc = ::aie::add(acc, ::aie::mul_square(v));
    }
    float sum_sq = ::aie::reduce_add(acc);
    for (int i = chunks * VEC; i < n; i++) {
        float x = (float)input[i];
        sum_sq += x * x;
    }

    float inv_rms = aie::invsqrt(sum_sq / n + eps);

    // Pass 2: normalize and apply gain
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        auto v = ::aie::load_v<VEC>(input + i * VEC);
        auto g = ::aie::load_v<VEC>(gain + i * VEC);
        auto normed = ::aie::mul(v, (bfloat16)inv_rms);
        ::aie::store_v(output + i * VEC, ::aie::mul(normed, g));
    }
    for (int i = chunks * VEC; i < n; i++) {
        output[i] = (bfloat16)((float)input[i] * inv_rms * (float)gain[i]);
    }
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
        ::aie::store_v(c + i * VEC, ::aie::mul(va, vb));
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

        ::aie::accum<accfloat, block_size> acc = ::aie::zeros<accfloat, block_size>();
        if constexpr (can_double_pump) {
            AIE_PREPARE_FOR_PIPELINING
            AIE_LOOP_MIN_ITERATION_COUNT(1)
            for (uint32_t g = 0; g < loop_iters; g++) {
                uint32_t g0 = g * 2, g1 = g0 + 1;
                auto w0 = ::aie::load_v<block_size>(w_row + g0 * block_size / 2);
                auto w1 = ::aie::load_v<block_size>(w_row + g1 * block_size / 2);
                auto w0f = ::aie::to_float(::aie::sub(::aie::unpack(w0),
                                           ::aie::broadcast<bfloat16, block_size>(8)));
                auto w1f = ::aie::to_float(::aie::sub(::aie::unpack(w1),
                                           ::aie::broadcast<bfloat16, block_size>(8)));
                auto scale0 = (float)s_row[g0];
                auto scale1 = (float)s_row[g1];
                auto b0 = ::aie::load_v<block_size>(b_in + g0 * block_size);
                auto b1 = ::aie::load_v<block_size>(b_in + g1 * block_size);
                acc = ::aie::add(acc, ::aie::mul(::aie::mul(w0f, scale0), b0));
                acc = ::aie::add(acc, ::aie::mul(::aie::mul(w1f, scale1), b1));
            }
        } else {
            AIE_PREPARE_FOR_PIPELINING
            for (uint32_t g = 0; g < groups_per_row; g++) {
                auto wvec = ::aie::load_v<block_size>(w_row + g * block_size / 2);
                auto wf   = ::aie::to_float(::aie::sub(::aie::unpack(wvec),
                                            ::aie::broadcast<bfloat16, block_size>(8)));
                float scale = (float)s_row[g];
                auto bvec = ::aie::load_v<block_size>(b_in + g * block_size);
                acc = ::aie::add(acc, ::aie::mul(::aie::mul(wf, scale), bvec));
            }
        }
        dest[row_offset + row] = (bfloat16)::aie::reduce_add(::aie::to_float(acc));
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
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        auto l = ::aie::load_v<VEC>(left_buf  + i * VEC);
        auto r = ::aie::load_v<VEC>(right_buf + i * VEC);
        // silu(l) = l * sigmoid(l) = l / (1 + exp(-l))
        auto lf = ::aie::to_float(l);
        auto sig = ::aie::inv(::aie::add(::aie::broadcast<float, VEC>(1.0f),
                              ::aie::exp(::aie::neg(lf))));
        auto silu = ::aie::mul(lf, sig);
        auto result = ::aie::mul(::aie::to_bfloat16(silu), r);
        ::aie::store_v(c_out + i * VEC, result);
    }
    for (int i = chunks * VEC; i < (int)m_output; i++) {
        float lv = (float)left_buf[i];
        float sig = 1.0f / (1.0f + aie::exp(-lv));
        c_out[i] = (bfloat16)(lv * sig * (float)right_buf[i]);
    }
}
