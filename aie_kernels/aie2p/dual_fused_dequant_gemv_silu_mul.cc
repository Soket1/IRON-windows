// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fused dual-GEMV (INT4 dequant) + SiLU + elementwise multiply kernel for AIE2+.
//
// Computes: output = silu(dequant(W1) @ x) * (dequant(W2) @ x)
//
// Two entry points called from the NPU design's core body:
//   1. dual_fused_dequant_gemv_bf16: dequant-GEMV writing to static buffer
//      (phase=0 -> left_buf, phase=1 -> right_buf)
//   2. dual_fused_dequant_gemv_silu_mul_bf16: reads from static buffers,
//      writes silu(left) * right to FIFO c_out
//
// Mirror of dual_gemv_silu_mul.cc -- inner matvec loop replaced with the
// dequant+mac sequence from fused_dequant_gemv.cc. The silu_mul phase is
// identical bf16 math.
//
// Weight tile layout (per call, m rows x k cols, group size G):
//   [m * k / 2 bytes]            packed uint4 weights
//   [m * (k / G) * 2 bytes]      bf16 scale factors

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

static bfloat16 left_buf[M_OUTPUT_MAX] __attribute__((aligned(64)));
static bfloat16 right_buf[M_OUTPUT_MAX] __attribute__((aligned(64)));

// Dequant+matvec writing into a static destination buffer at row_offset.
// Copy of fused_dequant_gemv.cc inner loop, with destination redirected to
// left_buf or right_buf depending on phase.
template <uint32_t block_size>
void dual_fused_dequant_matvec(uint32_t m,
                                uint32_t k,
                                const uint8_t *__restrict a_in,
                                const bfloat16 *__restrict b_in,
                                bfloat16 *__restrict c_out,
                                uint32_t group_size)
{
    static_assert(block_size == 32, "block_size must be 32 to match dequant vector width");

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const uint4 *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const uint8_t *scale_bytes = a_in + m * k / 2;
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(scale_bytes);

    const uint32_t groups_per_row = k / group_size;
    const uint32_t blocks_per_group = group_size / block_size;

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *row_weights = weights_packed + row * k / 2;
        const bfloat16 *row_scales = scales + row * groups_per_row;
        const bfloat16 *b_ptr = b_in;

        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();

        for (uint32_t g = 0; g < groups_per_row; g++) {
            bfloat16 sf = row_scales[g];
            aie::vector<bfloat16, block_size> sf_broadcast =
                aie::broadcast<bfloat16, block_size>(sf);

            for (uint32_t blk = 0; blk < blocks_per_group; blk++) {
                aie::vector<uint4, block_size> I0 =
                    aie::load_v<block_size>(row_weights);
                row_weights += block_size / 2;

                aie::vector<uint8, block_size> as_int8 = aie::unpack(I0);
                aie::vector<uint16, block_size> as_int16 = aie::unpack(as_int8);
                aie::vector<bfloat16, block_size> as_bf16 =
                    aie::to_float<bfloat16>(as_int16, 0);

                // CRITICAL: Q4_0 stores values as biased uint4 (signed = uint - 8).
                // The host bias compensation that fused_dequant_gemv relies on
                // does NOT work here -- SiLU is non-linear, so we cannot
                // subtract the bias post-hoc from the FIFO output. Bake the
                // -8 offset into the dequant pipeline.
                aie::vector<bfloat16, block_size> offset =
                    aie::broadcast<bfloat16, block_size>(8.0f);
                aie::vector<bfloat16, block_size> as_signed =
                    aie::sub(as_bf16, offset);

                aie::vector<bfloat16, block_size> w_dequant =
                    aie::mul(as_signed, sf_broadcast).template to_vector<bfloat16>();

                aie::vector<bfloat16, block_size> b_vec = aie::load_v<block_size>(b_ptr);
                b_ptr += block_size;

                acc = aie::mac(acc, w_dequant, b_vec);
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
void dual_fused_dequant_gemv_bf16(uint32_t m,
                                   uint32_t k,
                                   uint32_t row_offset,
                                   const uint8_t *__restrict a_in,
                                   const bfloat16 *__restrict b_in,
                                   uint32_t phase,
                                   uint32_t group_size)
{
    bfloat16 *dst = (phase == 0) ? left_buf : right_buf;
    dst += row_offset;
    dual_fused_dequant_matvec<32>(m, k, a_in, b_in, dst, group_size);
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
