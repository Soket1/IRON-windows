// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// V3 fused INT4 dequantization + batched GEMV kernel for AIE2+.
//
// Extends v2 with an M_BATCH dimension: dequant each weight tile ONCE,
// then accumulate dot products for M_BATCH independent activation rows.
// This amortises the dequant cost across the batch, giving near-linear
// throughput scaling with M_BATCH for memory-bound shapes.
//
// Target use case: speculative-decoding verification (M_BATCH = 4..8).
//
// L1 budget: M_BATCH × K × 2 bytes of activations must fit.
//   K=2048, M=8 → 32 KB   (fine)
//   K=3072, M=8 → 48 KB   (fine)
//   K=8192, M=4 → 64 KB   (tight; M=8 would overflow)
//
// Weight layout: identical to v2.
//   Per tile (m_tile rows × K cols, group_size G):
//     [m_tile × K / 2  bytes of packed uint4 weights]
//     [m_tile × (K/G) × 2  bytes of bf16 scale factors]
//
// Activation layout (b_in):
//   [M_BATCH contiguous rows of K bf16 values]
//   b_in[mb * DK + k]  for batch row mb, position k
//
// Output layout (c_out):
//   [m_tile × M_BATCH bf16 scalars]
//   c_out[row * M_BATCH + mb]

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// block_size: dequant vector width (must be 32 for aie::unpack)
// G:          group size (compile-time, must be multiple of block_size)
// DK:         K dimension (compile-time)
// MB:         M_BATCH — number of activation rows processed together
template <uint32_t block_size, uint32_t G, uint32_t DK, uint32_t MB>
void fused_dequant_matvec_v3(uint32_t m,
                              const uint8_t  *__restrict a_in,
                              const bfloat16 *__restrict b_in,
                              bfloat16       *__restrict c_out)
{
    static_assert(block_size == 32, "block_size must be 32");
    static_assert(G % block_size == 0, "group_size must be a multiple of block_size");
    static_assert(MB >= 1 && MB <= 8, "M_BATCH must be 1..8");

    constexpr uint32_t blocks_per_group = G / block_size;
    constexpr uint32_t groups_per_row   = DK / G;

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const uint4    *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const uint8_t  *scale_bytes    = a_in + m * DK / 2;
    const bfloat16 *scales         = reinterpret_cast<const bfloat16 *>(scale_bytes);

    event0();
    for (uint32_t row = 0; row < m; row++) {
        const uint4    *row_weights = weights_packed + row * DK / 2;
        const bfloat16 *row_scales  = scales + row * groups_per_row;

        // One accumulator per batch row.
        aie::accum<accfloat, block_size> acc[MB];
        for (uint32_t mb = 0; mb < MB; mb++)
            acc[mb] = aie::zeros<accfloat, block_size>();

        for (uint32_t g = 0; g < groups_per_row; g++) {
            bfloat16 sf = row_scales[g];
            aie::vector<bfloat16, block_size> sf_bc =
                aie::broadcast<bfloat16, block_size>(sf);

            AIE_LOOP_MIN_ITERATION_COUNT(blocks_per_group)
            for (uint32_t blk = 0; blk < blocks_per_group; blk++)
                AIE_PREPARE_FOR_PIPELINING
                {
                    // Dequant weight block once.
                    aie::vector<uint4, block_size> I0 = aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;

                    aie::vector<uint8,    block_size> a8  = aie::unpack(I0);
                    aie::vector<uint16,   block_size> a16 = aie::unpack(a8);
                    aie::vector<bfloat16, block_size> w_dq =
                        aie::mul(aie::to_float<bfloat16>(a16, 0), sf_bc)
                            .template to_vector<bfloat16>();

                    // MAC with each of the MB activation rows at this position.
                    const uint32_t k_off = (g * blocks_per_group + blk) * block_size;
                    for (uint32_t mb = 0; mb < MB; mb++) {
                        aie::vector<bfloat16, block_size> b_vec =
                            aie::load_v<block_size>(b_in + mb * DK + k_off);
                        acc[mb] = aie::mac(acc[mb], w_dq, b_vec);
                    }
                }
        }

        // Reduce and store MB outputs.
        for (uint32_t mb = 0; mb < MB; mb++) {
            c_out[row * MB + mb] =
                static_cast<bfloat16>(aie::reduce_add(acc[mb].template to_vector<float>()));
        }
    }
    event1();
}

// ---- Compile-time defaults (overridden via -D flags per shape) ----

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

#ifndef DIM_K
#define DIM_K 2048
#endif

#ifndef M_BATCH
#define M_BATCH 4
#endif

extern "C" {

// Entry point called by the IRON op design.py.
// row_offset: output tile start row (same semantic as v2).
// a_in:       packed weight tile [m × K/2 + m × groups × 2 bytes]
// b_in:       activation matrix  [M_BATCH × K bf16]
// c_out:      output matrix      [N × M_BATCH bf16]
//             (indexed as c_out[row_offset * M_BATCH + row * M_BATCH + mb])
void fused_dequant_matvec_v3_bf16(uint32_t m,
                                   uint32_t row_offset,
                                   const uint8_t  *__restrict a_in,
                                   const bfloat16 *__restrict b_in,
                                   bfloat16       *__restrict c_out)
{
    c_out += row_offset * M_BATCH;
    fused_dequant_matvec_v3<32, GROUP_SIZE, DIM_K, M_BATCH>(m, a_in, b_in, c_out);
}

} // extern "C"
