// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// LayerFused kernel — A1.3.1 SKELETON.
//
// Single noop entry point that copies its input chunk to output. Real
// per-stage compute kernels (pre-RMS, QKV GEMV, RoPE, KV slot, GQA
// attention, O_proj, ANM, SwiGLU) land in A1.3.2-A1.3.4 alongside their
// design.py wiring.

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GROUP_SIZE
#define GROUP_SIZE 32
#endif

#ifndef EMBED_DIM
#define EMBED_DIM 2048
#endif

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 8192
#endif

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

#ifndef NUM_HEADS
#define NUM_HEADS 32
#endif

#ifndef NUM_KV_HEADS
#define NUM_KV_HEADS 8
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

extern "C" {

void layer_fused_noop_bf16(bfloat16 *__restrict__ in,
                           bfloat16 *__restrict__ out,
                           int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        out[i] = in[i];
    }
}

// Weighted RMSNorm: output[i] = (input[i] / rms(input)) * gain[i]
// epsilon = 1e-5 (matches existing rms_norm.cc + post_attn_rms_norm_bf16)
// Used by both pre-RMS (W_norm1) and post-RMS (W_norm2) stages.
void layer_fused_rms_norm_bf16(const bfloat16 *input, const bfloat16 *gain,
                               bfloat16 *output, int32_t n) {
    constexpr float eps = 1e-5f;
    constexpr int VEC = 16;

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

    for (int i = 0; i < n; i++) {
        output[i] = (bfloat16)((float)input[i] * inv_rms * (float)gain[i]);
    }
    (void)chunks;
}

}  // extern "C"

// ────────────────────────────────────────────────────────────────────────────
// INT4 dequant + GEMV (K = EMBED_DIM input, M output rows).
// Matches the `_gemv_swiglu` algorithm from post_attn_fused.cc — INT4 weights
// with per-group_size scales (bf16), -8 bias correction in-kernel. Used for
// Q, K, V projections (all have input K = embed_dim, output dims vary by m).
// ────────────────────────────────────────────────────────────────────────────

template <uint32_t block_size, uint32_t G, uint32_t DK>
static void _qkv_gemv(uint32_t m,
                      const uint8_t *__restrict a_in,
                      const bfloat16 *__restrict b_in,
                      bfloat16 *__restrict c_out) {
    static_assert(block_size == 32, "block_size must be 32");
    static_assert(G % block_size == 0, "group_size must be multiple of block_size");
    constexpr uint32_t groups_per_row = DK / G;
    constexpr bool can_double_pump = (groups_per_row >= 2) && (groups_per_row % 2 == 0);
    constexpr uint32_t pump_groups  = can_double_pump ? 2 : 1;
    constexpr uint32_t loop_iters   = groups_per_row / pump_groups;

    ::aie::set_rounding(aie::rounding_mode::conv_even);

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
        c_out[row] = static_cast<bfloat16>(
            aie::reduce_add(acc.template to_vector<float>()));
    }
}

// Q/K/V projection entry. Same K (input embed_dim); m varies with output:
//   Q: m = NUM_HEADS  * HEAD_DIM / num_aie_columns
//   K: m = NUM_KV_HEADS * HEAD_DIM / num_aie_columns
//   V: same as K
// row_offset selects the slice of weights this column owns.
extern "C" void layer_fused_qkv_gemv_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, EMBED_DIM>(
        m, a + row_offset * (EMBED_DIM / 2 + EMBED_DIM / GROUP_SIZE * 2),
        b, c);
}

// ────────────────────────────────────────────────────────────────────────────
// RoPE rotation (apply pre-computed sin/cos LUT from host).
//
// Input  qk_in[n_heads_local × HEAD_DIM]  bf16 (post-Q/K GEMV, this col's
//        slice of either Q or K)
// Input  cos[HEAD_DIM/2]                  bf16  (LUT for current position)
// Input  sin[HEAD_DIM/2]                  bf16  (LUT for current position)
// Output qk_out[n_heads_local × HEAD_DIM] bf16  (rotated)
//
// RoPE applies a rotation per head over HEAD_DIM/2 pairs:
//   for h in range(n_heads_local):
//     for i in range(HEAD_DIM/2):
//       a = qk_in[h*HEAD_DIM + i]
//       b = qk_in[h*HEAD_DIM + HEAD_DIM/2 + i]   (Llama convention: split-half)
//       qk_out[h*HEAD_DIM + i]              = a*cos[i] - b*sin[i]
//       qk_out[h*HEAD_DIM + HEAD_DIM/2 + i] = a*sin[i] + b*cos[i]
//
// CPU precomputes sin/cos for the current decode position (see U2 decision
// in dev_notes/layer_fused_design.md). Sin/cos LUT is HEAD_DIM/2 elements
// each because the rotation pair (i, HEAD_DIM/2+i) shares the same angle.
//
// n is the total bf16 element count of qk_in/qk_out (must be a multiple of
// HEAD_DIM). The kernel iterates n / HEAD_DIM heads internally.
// ────────────────────────────────────────────────────────────────────────────

extern "C" void layer_fused_rope_apply_bf16(
        const bfloat16 *qk_in, const bfloat16 *cos_lut,
        const bfloat16 *sin_lut, bfloat16 *qk_out, int32_t n) {
    constexpr int HALF = HEAD_DIM / 2;
    int n_heads_local = n / HEAD_DIM;
    for (int h = 0; h < n_heads_local; h++) {
        const bfloat16 *in  = qk_in  + h * HEAD_DIM;
        bfloat16       *out = qk_out + h * HEAD_DIM;
        for (int i = 0; i < HALF; i++) {
            float a = (float)in[i];
            float b = (float)in[HALF + i];
            float c = (float)cos_lut[i];
            float s = (float)sin_lut[i];
            out[i]        = (bfloat16)(a * c - b * s);
            out[HALF + i] = (bfloat16)(a * s + b * c);
        }
    }
}
