// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// V2 fused INT4 dequantization + GEMV kernel for AIE2+.
//
// This is a direct port of the optimized kernel from amd/IRON PR #101
// (https://github.com/amd/IRON/pull/101) into the IRON-windows tree.
// Bench in that PR: 561 us for K=2048 N=8192 vs ~2200 us in our v1.
//
// Three optimizations vs v1 (aie_kernels/aie2p/fused_dequant_gemv.cc):
//   1. Compile-time DIM_K and GROUP_SIZE via -D flags allow the compiler
//      to fully unroll the inner loop and eliminate runtime arithmetic.
//   2. AIE_PREPARE_FOR_PIPELINING + AIE_LOOP_MIN_ITERATION_COUNT hints
//      let the AIE compiler schedule the software pipeline.
//   3. Double-pump: process 2 groups per iteration with independent A/B
//      unpack chains so the compiler can interleave them, hiding the
//      dequant latency behind activation loads + MAC ops.
//
// Weight layout per tile (m rows x K cols, group_size G):
//   [m * K / 2 bytes of packed uint4 weights]
//   [m * (K / G) bf16 scale factors, stored as (m * K / G * 2) bytes]

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// Weight nibble interpretation. Default = UNSIGNED (kernel computes nib*scale;
// the host applies the Q4_0 -8 bias-compensation after the GEMV). With
// -DWEIGHT_SIGNED the nibbles are SIGNED int4 (sign-extended on unpack), so the
// kernel computes (nib-8)*scale directly on-chip when the host packs (nib-8)&0xF
// — required when the GEMV output is consumed on-chip (no host bias-comp hook).
#ifdef WEIGHT_SIGNED
using wnib_t = int4;
using w8_t   = int8;
using w16_t  = int16;
#else
using wnib_t = uint4;
using w8_t   = uint8;
using w16_t  = uint16;
#endif

// block_size: dequant vector width (must be 32 for aie::unpack)
// G: group size (compile-time, must be multiple of block_size)
// DK: K dimension (compile-time for loop count optimization)
template <uint32_t block_size, uint32_t G, uint32_t DK>
void fused_dequant_matvec(uint32_t m,
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
    // Four groups per iteration through two independent load+unpack chains, so
    // each chain's dequant latency hides under the other's. GEMV_V2_NO_QUAD
    // drops back to two groups (one chain) for A/B.
#ifdef GEMV_V2_NO_QUAD
    constexpr bool use_quad_pump = false;
#else
    constexpr bool use_quad_pump = (groups_per_row >= 4) && (groups_per_row % 4 == 0);
#endif

    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const wnib_t *weights_packed = reinterpret_cast<const wnib_t *>(a_in);
    const uint8_t *scale_bytes = a_in + m * DK / 2;
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(scale_bytes);

    event0();
    for (uint32_t row = 0; row < m; row++) {
        const wnib_t *row_weights = weights_packed + row * DK / 2;
        const bfloat16 *row_scales = scales + row * groups_per_row;
        const bfloat16 *b_ptr = b_in;

        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();

        if constexpr (use_quad_pump && blocks_per_group == 1) {
            // Four groups per iteration through TWO independent 64-nibble
            // load+unpack chains. Folding the old pair of 32-nibble chains into
            // one shared 64-nibble chain removed work but also removed the
            // overlap that hid its latency -- the nop count in the object went
            // up even as the instruction count went down. This keeps the single
            // unpack per 64 nibbles and puts the second chain back.
            AIE_LOOP_MIN_ITERATION_COUNT(groups_per_row / 4)
            for (uint32_t g = 0; g < groups_per_row; g += 4)
                AIE_PREPARE_FOR_PIPELINING
                {
                    aie::vector<wnib_t, 2 * block_size> I01 =
                        aie::load_v<2 * block_size>(row_weights);
                    row_weights += block_size;
                    aie::vector<wnib_t, 2 * block_size> I23 =
                        aie::load_v<2 * block_size>(row_weights);
                    row_weights += block_size;

                    aie::vector<bfloat16, 2 * block_size> d01 =
                        aie::to_float<bfloat16>(aie::unpack(I01), 0);
                    aie::vector<bfloat16, 2 * block_size> d23 =
                        aie::to_float<bfloat16>(aie::unpack(I23), 0);

                    aie::vector<bfloat16, block_size> w0 =
                        aie::mul(d01.template extract<block_size>(0),
                                 aie::broadcast<bfloat16, block_size>(row_scales[g]))
                            .template to_vector<bfloat16>();
                    aie::vector<bfloat16, block_size> w1 =
                        aie::mul(d01.template extract<block_size>(1),
                                 aie::broadcast<bfloat16, block_size>(row_scales[g + 1]))
                            .template to_vector<bfloat16>();
                    aie::vector<bfloat16, block_size> w2 =
                        aie::mul(d23.template extract<block_size>(0),
                                 aie::broadcast<bfloat16, block_size>(row_scales[g + 2]))
                            .template to_vector<bfloat16>();
                    aie::vector<bfloat16, block_size> w3 =
                        aie::mul(d23.template extract<block_size>(1),
                                 aie::broadcast<bfloat16, block_size>(row_scales[g + 3]))
                            .template to_vector<bfloat16>();

                    acc = aie::mac(acc, w0, aie::load_v<block_size>(b_ptr));
                    b_ptr += block_size;
                    acc = aie::mac(acc, w1, aie::load_v<block_size>(b_ptr));
                    b_ptr += block_size;
                    acc = aie::mac(acc, w2, aie::load_v<block_size>(b_ptr));
                    b_ptr += block_size;
                    acc = aie::mac(acc, w3, aie::load_v<block_size>(b_ptr));
                    b_ptr += block_size;
                }
        } else if constexpr (can_double_pump && blocks_per_group == 1) {
            // 2 groups per iteration, one shared load+unpack chain.
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g += 2)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf_a = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_a_bc =
                        aie::broadcast<bfloat16, block_size>(sf_a);
                    bfloat16 sf_b = row_scales[g + 1];
                    aie::vector<bfloat16, block_size> sf_b_bc =
                        aie::broadcast<bfloat16, block_size>(sf_b);

#ifdef GEMV_V2_LEGACY_UNPACK
                    // A/B reference: two 128-bit loads, each followed by int4 -> int8
                    // -> int16 -> bf16. Same values as the form below (the int16 hop
                    // widens the container without changing the integer), kept so the
                    // two can be measured against each other on one machine state.
                    aie::vector<wnib_t, block_size> I0_a =
                        aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;
                    aie::vector<wnib_t, block_size> I0_b =
                        aie::load_v<block_size>(row_weights);
                    row_weights += block_size / 2;

                    aie::vector<w8_t, block_size> a8_a = aie::unpack(I0_a);
                    aie::vector<w16_t, block_size> a16_a = aie::unpack(a8_a);
                    aie::vector<bfloat16, block_size> abf_a =
                        aie::to_float<bfloat16>(a16_a, 0);
                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(abf_a, sf_a_bc).template to_vector<bfloat16>();

                    aie::vector<w8_t, block_size> a8_b = aie::unpack(I0_b);
                    aie::vector<w16_t, block_size> a16_b = aie::unpack(a8_b);
                    aie::vector<bfloat16, block_size> abf_b =
                        aie::to_float<bfloat16>(a16_b, 0);
                    aie::vector<bfloat16, block_size> w_b =
                        aie::mul(abf_b, sf_b_bc).template to_vector<bfloat16>();
#else

                    // Both groups in ONE 64-nibble load and ONE unpack: int4 -> int8
                    // -> bf16, the same shape _lf_dual_gemv uses. The int8 -> int16
                    // hop this used to take changed no value (same integer, wider
                    // container) but cost an extra vunpack plus a crunpacksize
                    // toggle per group, and the two 128-bit loads could not fuse
                    // into vldb.unpack the way one 256-bit load does.
                    aie::vector<wnib_t, 2 * block_size> I01 =
                        aie::load_v<2 * block_size>(row_weights);
                    row_weights += block_size;

                    aie::vector<w8_t, 2 * block_size> a8_01 = aie::unpack(I01);
                    aie::vector<bfloat16, 2 * block_size> abf_01 =
                        aie::to_float<bfloat16>(a8_01, 0);

                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(abf_01.template extract<block_size>(0), sf_a_bc)
                            .template to_vector<bfloat16>();
                    aie::vector<bfloat16, block_size> w_b =
                        aie::mul(abf_01.template extract<block_size>(1), sf_b_bc)
                            .template to_vector<bfloat16>();
#endif

                    // Load activation vectors and MAC
                    aie::vector<bfloat16, block_size> b_a = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_a, b_a);

                    aie::vector<bfloat16, block_size> b_b = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_b, b_b);
                }
        } else {
            // Generic path: 1 group per iteration
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g++)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_broadcast =
                        aie::broadcast<bfloat16, block_size>(sf);

                    AIE_LOOP_MIN_ITERATION_COUNT(blocks_per_group)
                    for (uint32_t blk = 0; blk < blocks_per_group; blk++) {
                        aie::vector<wnib_t, block_size> I0 = aie::load_v<block_size>(row_weights);
                        row_weights += block_size / 2;

                        aie::vector<w8_t, block_size> as_int8 = aie::unpack(I0);
                        aie::vector<w16_t, block_size> as_int16 = aie::unpack(as_int8);
                        aie::vector<bfloat16, block_size> as_bf16 =
                            aie::to_float<bfloat16>(as_int16, 0);
                        aie::vector<bfloat16, block_size> w_dequant =
                            aie::mul(as_bf16, sf_broadcast).template to_vector<bfloat16>();

                        aie::vector<bfloat16, block_size> b_vec = aie::load_v<block_size>(b_ptr);
                        b_ptr += block_size;

                        acc = aie::mac(acc, w_dequant, b_vec);
                    }
                }
        }

        *c_out = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
        c_out++;
    }
    event1();
}

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

#ifndef DIM_K
#define DIM_K 2048
#endif

extern "C" {

// Entry point used by the matching IRON op. row_offset lets the caller
// position rows within a larger output vector without pointer arithmetic
// in MLIR.
void fused_dequant_matvec_v2_bf16(uint32_t m,
                                   uint32_t row_offset,
                                   const uint8_t *__restrict a_in,
                                   const bfloat16 *__restrict b_in,
                                   bfloat16 *__restrict c_out)
{
#if defined(STUB_QKV) || defined(STUB_OPROJ)
    // #78 per-phase stub: zero output, skip compute. Preserves signature so
    // MLIR call sites link unchanged; the phase's weight-DMA still flows (BDs
    // fire, locks cycle) but the GEMV compute is elided. Delta vs full =
    // the phase's compute cost not hidden under DMA.
    c_out += row_offset;
    for (uint32_t i = 0; i < m; i++) c_out[i] = (bfloat16)0;
    return;
#else
    c_out += row_offset;
    fused_dequant_matvec<32, GROUP_SIZE, DIM_K>(m, a_in, b_in, c_out);
#endif
}

} // extern "C"
