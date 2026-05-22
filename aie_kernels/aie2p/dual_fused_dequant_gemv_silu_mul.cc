// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fused dual-GEMV (INT4 dequant) + SiLU + elementwise multiply kernel for AIE2+.
//
// Computes: output = silu(dequant(W1) @ x) * (dequant(W2) @ x)
//
// V2 in-place upgrade (mirrors fused_dequant_gemv_v2.cc from PR #101):
//   1. Compile-time DIM_K and GROUP_SIZE via -D flags allow the compiler
//      to fully unroll the inner loop and eliminate runtime arithmetic.
//   2. AIE_PREPARE_FOR_PIPELINING + AIE_LOOP_MIN_ITERATION_COUNT hints
//      let the AIE compiler schedule the software pipeline.
//   3. Double-pump: process 2 groups per iteration with independent A/B
//      unpack chains so the compiler can interleave them, hiding the
//      dequant latency behind activation loads + MAC ops.
//
// CRITICAL difference from fused_dequant_gemv_v2.cc: SiLU is non-linear,
// so the host-side bias compensation trick used by the standalone v2
// does NOT apply. Both pump chains keep aie::sub(8) between to_float
// and mul(scale) to convert uint4 [0..15] back to signed [-8..7].
//
// Two entry points called from the NPU design's core body:
//   1. dual_fused_dequant_gemv_bf16: dequant-GEMV writing to static buffer
//      (phase=0 -> left_buf, phase=1 -> right_buf)
//   2. dual_fused_dequant_gemv_silu_mul_bf16: reads from static buffers,
//      writes silu(left) * right to FIFO c_out
//
// Weight tile layout (per call, m rows x K cols, group size G):
//   [m * K / 2 bytes]            packed uint4 weights
//   [m * (K / G) * 2 bytes]      bf16 scale factors

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// Buffer size must be >= m_output (= hidden_dim / num_aie_columns).
// Overridden at compile time via -DM_OUTPUT_MAX=N by the operator.
#ifndef M_OUTPUT_MAX
#define M_OUTPUT_MAX 4096
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

#ifndef DIM_K
#define DIM_K 2048
#endif

static bfloat16 left_buf[M_OUTPUT_MAX] __attribute__((aligned(64)));
static bfloat16 right_buf[M_OUTPUT_MAX] __attribute__((aligned(64)));

// Dequant+matvec writing into a static destination buffer at row_offset.
//
// block_size: dequant vector width (must be 32 for aie::unpack)
// G: group size (compile-time, must be multiple of block_size)
// DK: K dimension (compile-time for loop count optimization)
template <uint32_t block_size, uint32_t G, uint32_t DK>
void dual_fused_dequant_matvec(uint32_t m,
                                const uint8_t *__restrict a_in,
                                const bfloat16 *__restrict b_in,
                                bfloat16 *__restrict c_out)
{
    static_assert(block_size == 32, "block_size must be 32 to match dequant vector width");
    static_assert(G % block_size == 0, "group_size must be a multiple of block_size");
    constexpr uint32_t blocks_per_group = G / block_size;
    constexpr uint32_t groups_per_row = DK / G;
    constexpr bool can_double_pump = (groups_per_row >= 2) && (groups_per_row % 2 == 0);
    constexpr uint32_t pump_groups = can_double_pump ? 2 : 1;
    constexpr uint32_t loop_iters = groups_per_row / pump_groups;

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const uint4 *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const uint8_t *scale_bytes = a_in + m * DK / 2;
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(scale_bytes);

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *row_weights = weights_packed + row * DK / 2;
        const bfloat16 *row_scales = scales + row * groups_per_row;
        const bfloat16 *b_ptr = b_in;

        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();

        if constexpr (can_double_pump && blocks_per_group == 1) {
            // Optimized path: 2 groups per iteration, 1 block per group.
            // Two independent unpack chains for the compiler to interleave.
            // sub(8) stays inline in each chain because SiLU is non-linear
            // (no host-side bias compensation possible).
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g += 2)
                AIE_PREPARE_FOR_PIPELINING
                {
                    // --- Chain A: group g ---
                    bfloat16 sf_a = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_a_bc =
                        aie::broadcast<bfloat16, block_size>(sf_a);

                    aie::vector<uint4, block_size> I0_a =
                        aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;

                    // --- Chain B: group g+1 (interleaved) ---
                    bfloat16 sf_b = row_scales[g + 1];
                    aie::vector<bfloat16, block_size> sf_b_bc =
                        aie::broadcast<bfloat16, block_size>(sf_b);

                    aie::vector<uint4, block_size> I0_b =
                        aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;

                    aie::vector<bfloat16, block_size> offset =
                        aie::broadcast<bfloat16, block_size>(8.0f);

                    // Unpack chain A + bias-correct + scale
                    aie::vector<uint8, block_size> a8_a = aie::unpack(I0_a);
                    aie::vector<uint16, block_size> a16_a = aie::unpack(a8_a);
                    aie::vector<bfloat16, block_size> abf_a =
                        aie::to_float<bfloat16>(a16_a, 0);
                    aie::vector<bfloat16, block_size> asgn_a =
                        aie::sub(abf_a, offset);
                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(asgn_a, sf_a_bc).template to_vector<bfloat16>();

                    // Unpack chain B + bias-correct + scale
                    aie::vector<uint8, block_size> a8_b = aie::unpack(I0_b);
                    aie::vector<uint16, block_size> a16_b = aie::unpack(a8_b);
                    aie::vector<bfloat16, block_size> abf_b =
                        aie::to_float<bfloat16>(a16_b, 0);
                    aie::vector<bfloat16, block_size> asgn_b =
                        aie::sub(abf_b, offset);
                    aie::vector<bfloat16, block_size> w_b =
                        aie::mul(asgn_b, sf_b_bc).template to_vector<bfloat16>();

                    // Load activation vectors and MAC
                    aie::vector<bfloat16, block_size> b_a = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_a, b_a);

                    aie::vector<bfloat16, block_size> b_b = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_b, b_b);
                }
        } else {
            // Generic path: 1 group per iteration.
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g++)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_broadcast =
                        aie::broadcast<bfloat16, block_size>(sf);
                    aie::vector<bfloat16, block_size> offset =
                        aie::broadcast<bfloat16, block_size>(8.0f);

                    AIE_LOOP_MIN_ITERATION_COUNT(blocks_per_group)
                    for (uint32_t blk = 0; blk < blocks_per_group; blk++) {
                        aie::vector<uint4, block_size> I0 =
                            aie::load_v<block_size>(row_weights);
                        row_weights += block_size / 2;

                        aie::vector<uint8, block_size> as_int8 = aie::unpack(I0);
                        aie::vector<uint16, block_size> as_int16 = aie::unpack(as_int8);
                        aie::vector<bfloat16, block_size> as_bf16 =
                            aie::to_float<bfloat16>(as_int16, 0);
                        aie::vector<bfloat16, block_size> as_signed =
                            aie::sub(as_bf16, offset);
                        aie::vector<bfloat16, block_size> w_dequant =
                            aie::mul(as_signed, sf_broadcast).template to_vector<bfloat16>();

                        aie::vector<bfloat16, block_size> b_vec = aie::load_v<block_size>(b_ptr);
                        b_ptr += block_size;

                        acc = aie::mac(acc, w_dequant, b_vec);
                    }
                }
        }

        *c_out = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
        c_out++;
    }
}

extern "C" {

// Phase 0 & 1: dequant-GEMV writing to a static buffer.
//   phase=0 -> left_buf  (gate path)
//   phase=1 -> right_buf (up path)
//
// V2 signature: k and group_size are compile-time (DIM_K, GROUP_SIZE) and
// dropped from the runtime args. design.py must match.
void dual_fused_dequant_gemv_bf16(uint32_t m,
                                   uint32_t row_offset,
                                   const uint8_t *__restrict a_in,
                                   const bfloat16 *__restrict b_in,
                                   uint32_t phase)
{
    bfloat16 *dst = (phase == 0) ? left_buf : right_buf;
    dst += row_offset;
    dual_fused_dequant_matvec<32, GROUP_SIZE, DIM_K>(m, a_in, b_in, dst);
}

// Phase 2: silu(left_buf) * right_buf -> c_out (FIFO buffer).
// Identical to dual_gemv_silu_mul.cc's silu_mul phase.
void dual_fused_dequant_gemv_silu_mul_bf16(bfloat16 *__restrict c_out,
                                            int32_t m_output)
{
    event0();

    aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
    aie::vector<bfloat16, 16> register_1   = aie::broadcast<bfloat16, 16>(1.0f);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < m_output; i += 16) {
        aie::vector<bfloat16, 16> left_val  = aie::load_v<16>(left_buf + i);
        aie::vector<bfloat16, 16> right_val = aie::load_v<16>(right_buf + i);

        // SiLU(x) = x * sigmoid(x) = x * 0.5 * (1 + tanh(x/2))
        auto half_x = aie::mul(left_val, register_0_5);
        auto tanh_half_x = aie::tanh<bfloat16>(half_x.to_vector<float>());
        auto tanh_half_x_approx = aie::add(tanh_half_x, register_1);
        aie::vector<bfloat16, 16> sigmoid_approx = aie::mul(tanh_half_x_approx, register_0_5);
        auto silu_output = aie::mul(left_val, sigmoid_approx);

        auto fused_output = aie::mul(silu_output.to_vector<bfloat16>(), right_val);
        aie::store_v(c_out + i, fused_output.to_vector<bfloat16>());
    }

    event1();
}

} // extern "C"
