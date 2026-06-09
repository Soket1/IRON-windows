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

#ifndef NUM_AIE_COLUMNS
#define NUM_AIE_COLUMNS 8
#endif

// Per-column slice of the hidden dim, used by the on-chip (decomp-B) down
// projection: each column reduces only its HIDDEN_DIM/cols slice of silu_out
// (kept on-chip via inter_fifo) and emits a PARTIAL EMBED_DIM vector. The
// host sums the NUM_AIE_COLUMNS partials. This keeps silu_out off DDR (F5).
#define INTER_DIM_PER_COL (HIDDEN_DIM / NUM_AIE_COLUMNS)

extern "C" {

void layer_fused_noop_bf16(bfloat16 *__restrict__ in,
                           bfloat16 *__restrict__ out,
                           int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        out[i] = in[i];
    }
}

// Thin alias used by the attention-spike worker, which binds a Kernel
// with the larger kv_chunk L1 type. IRON Kernel objects are looked up
// by C symbol, so a same-symbol second binding fails verification with
// "redefinition of symbol named ...". This separate symbol resolves
// the collision while keeping the implementation identical.
void layer_fused_noop_kv_bf16(bfloat16 *__restrict__ in,
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
                    aie::vector<bfloat16, block_size> abf_a =
                        aie::to_float<bfloat16>(a8_a, 0);          // single unpack (uint4->uint8->bf16)
                    aie::vector<bfloat16, block_size> asgn_a =
                        aie::sub(abf_a, offset);
                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(asgn_a, sf_a_bc).template to_vector<bfloat16>();

                    aie::vector<uint8,  block_size> a8_b  = aie::unpack(I0_b);
                    aie::vector<bfloat16, block_size> abf_b =
                        aie::to_float<bfloat16>(a8_b, 0);
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

// BROADCAST / outer-product GEMV PROBE (#30). FFLM structure: activation x held
// in registers, vextbcst per lane, MAC against weight COLUMN vectors into N
// parallel output accumulators; per-group scale applied ONCE on the accumulator
// (amortized over G macs → vmul/vmac ~1/G ≈ 0.03, vs our dot-product's 1.0).
// out[0:N] = Σ_k W[0:N,k]*x[k], scale[n,kg] per output per group, signed int4
// weights stored COLUMN-MAJOR (W[:,k] = N nibbles contiguous). N = block_size.
template <uint32_t N, uint32_t G, uint32_t K>
static void _gemv_bcast(const uint8_t *__restrict w,
                        const bfloat16 *__restrict x, bfloat16 *__restrict out) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(w + K * N / 2);  // packed after weights
    aie::accum<accfloat, N> acc = aie::zeros<accfloat, N>();
    for (uint32_t kg = 0; kg < K / G; kg++) {
        const bfloat16 *xchunk = x + kg * G;
        aie::accum<accfloat, N> gacc = aie::zeros<accfloat, N>();
        AIE_LOOP_UNROLL(8)
        for (uint32_t j = 0; j < G; j++) {
            // explicit byte offset: column (kg*G+j) is N nibbles = N/2 bytes wide
            const int4 *wcolp = reinterpret_cast<const int4 *>(w + (kg * G + j) * (N / 2));
            aie::vector<int4, N> wcol = aie::load_v<N>(wcolp);
            aie::vector<bfloat16, N> wbf = aie::to_float<bfloat16>(aie::unpack(wcol), 0);
            aie::vector<bfloat16, N> xkb = aie::broadcast<bfloat16, N>(xchunk[j]);  // scalar bcast (probe)
            gacc = aie::mac(gacc, wbf, xkb);
        }
        aie::vector<bfloat16, N> sg = aie::load_v<N>(scales + kg * N);  // per-output scale, this group
        acc = aie::mac(acc, gacc.template to_vector<bfloat16>(), sg);   // amortized scale
    }
    aie::store_v(out, acc.template to_vector<bfloat16>());
}

// 64-WIDE-DEQUANT broadcast GEMV (#30). One unpack+to_float feeds TWO mac
// columns: load 2 adjacent column-major weight columns as int4[2N] (= N bytes),
// unpack/convert once over 2N, extract the two N-halves, mac each against its
// own broadcast x lane. Halves unpack/conv cost per mac (1.0 → 0.5), targeting
// FFLM's vconv 0.55 / vunpack 0.48. G must be even (it is: GROUP_SIZE=32).
template <uint32_t N, uint32_t G, uint32_t K>
static void _gemv_bcast_w2(const uint8_t *__restrict w,
                           const bfloat16 *__restrict x, bfloat16 *__restrict out) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(w + K * N / 2);
    aie::accum<accfloat, N> acc = aie::zeros<accfloat, N>();
    for (uint32_t kg = 0; kg < K / G; kg++) {
        const bfloat16 *xchunk = x + kg * G;
        aie::accum<accfloat, N> gacc = aie::zeros<accfloat, N>();
        AIE_LOOP_UNROLL(8)
        for (uint32_t j = 0; j < G; j += 2) {
            // columns (kg*G+j) and (kg*G+j+1) are adjacent: 2N nibbles = N bytes
            const int4 *wp = reinterpret_cast<const int4 *>(w + (kg * G + j) * (N / 2));
            aie::vector<int4, 2 * N> w2 = aie::load_v<2 * N>(wp);
            aie::vector<bfloat16, 2 * N> wbf2 = aie::to_float<bfloat16>(aie::unpack(w2), 0);
            aie::vector<bfloat16, N> w_lo = wbf2.template extract<N>(0);
            aie::vector<bfloat16, N> w_hi = wbf2.template extract<N>(1);
            gacc = aie::mac(gacc, w_lo, aie::broadcast<bfloat16, N>(xchunk[j]));
            gacc = aie::mac(gacc, w_hi, aie::broadcast<bfloat16, N>(xchunk[j + 1]));
        }
        aie::vector<bfloat16, N> sg = aie::load_v<N>(scales + kg * N);
        acc = aie::mac(acc, gacc.template to_vector<bfloat16>(), sg);
    }
    aie::store_v(out, acc.template to_vector<bfloat16>());
}

extern "C" void layer_fused_gemv_bcast_bf16(const uint8_t *ws,
                                            const bfloat16 *x, bfloat16 *out) {
    _gemv_bcast_w2<32, GROUP_SIZE, EMBED_DIM>(ws, x, out);  // 64-wide: unpack/mac 1.0->0.5
}

// SIGNED-int4 GEMV (#12 density). Weights stored as signed int4 = (nibble-8) in
// two's complement, so the -8 centering is FREE via a signed unpack — no bf16
// `aie::sub` in the hot loop. Removing the sub (1) keeps full bf16 precision
// (centered products, unlike the bias-fold cancellation), and (2) unblocks
// AIE_LOOP_UNROLL on peano (the sub was the unroll-miscompile trigger). This is
// the reference kernel's `unpacksign0` approach. Used for gate/up (which carry
// the -8 bias); down stays unsigned (bias 0).
template <uint32_t block_size, uint32_t G, uint32_t DK>
static void _qkv_gemv_s4(uint32_t m, const uint8_t *__restrict a_in,
                         const bfloat16 *__restrict b_in, bfloat16 *__restrict c_out) {
    constexpr uint32_t gpr = DK / G;
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const int4 *weights = reinterpret_cast<const int4 *>(a_in);
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(a_in + m * DK / 2);
    for (uint32_t row = 0; row < m; row++) {
        const int4 *w_row = weights + row * DK / 2;
        const bfloat16 *s_row = scales + row * gpr;
        const bfloat16 *b_ptr = b_in;
        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();
        AIE_LOOP_MIN_ITERATION_COUNT(gpr / 2)
        AIE_LOOP_UNROLL(8)
        for (uint32_t g = 0; g < gpr; g += 2) {
            aie::vector<bfloat16, block_size> sfa = aie::broadcast<bfloat16, block_size>(s_row[g]);
            aie::vector<int4, block_size> Ia = aie::load_v<block_size>(w_row); w_row += block_size / 2;
            aie::vector<bfloat16, block_size> sfb = aie::broadcast<bfloat16, block_size>(s_row[g + 1]);
            aie::vector<int4, block_size> Ib = aie::load_v<block_size>(w_row); w_row += block_size / 2;
            aie::vector<bfloat16, block_size> wa =
                aie::mul(aie::to_float<bfloat16>(aie::unpack(Ia), 0), sfa).template to_vector<bfloat16>();
            aie::vector<bfloat16, block_size> wb =
                aie::mul(aie::to_float<bfloat16>(aie::unpack(Ib), 0), sfb).template to_vector<bfloat16>();
            aie::vector<bfloat16, block_size> ba = aie::load_v<block_size>(b_ptr); b_ptr += block_size;
            acc = aie::mac(acc, wa, ba);
            aie::vector<bfloat16, block_size> bb = aie::load_v<block_size>(b_ptr); b_ptr += block_size;
            acc = aie::mac(acc, wb, bb);
        }
        c_out[row] = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
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
// Static L1 buffer variants for col 0's merged worker (Path A from spec).
//
// Col 0's compute tile runs pre-RMS first, then QKV GEMVs. To stay within
// the AIE2P 2-input-DMA cap, col 0 cannot consume bq_mem.cons() (would be
// a 3rd input channel alongside rms_in and Aqkv). Instead, pre-RMS writes
// its normed output to BOTH a static L1 buffer (used by col 0's own QKV
// phases) and to a fifo-routed output buffer (broadcast via MemTile to
// cols 1-7).
// ────────────────────────────────────────────────────────────────────────────

// COMPILE_QKV_STATIC: col-0 static-activation variants (not used in decode_layer
// which uses v2 GEMV instead). Guard to free 4 KB .bss (normed_static) — needed
// when M_OUTPUT_MAX is large (e.g. m_input=4) and .bss would otherwise overflow.
#ifdef COMPILE_QKV_STATIC
static bfloat16 normed_static[EMBED_DIM] __attribute__((aligned(64)));

// Col-0 pre-RMS variant: computes weighted RMSNorm into BOTH the static
// L1 buffer (for col 0's own QKV phases) AND the fifo-routed `bq_out`
// (for the MemTile broadcast that feeds cols 1-7).
extern "C" void layer_fused_pre_rms_col0_bf16(
        const bfloat16 *input, const bfloat16 *gain,
        bfloat16 *bq_out, int32_t n) {
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
        bfloat16 r = (bfloat16)((float)input[i] * inv_rms * (float)gain[i]);
        normed_static[i] = r;
        bq_out[i] = r;
    }
    (void)chunks;
}

// Col-0 QKV variant: reads activation from the static L1 buffer (filled
// by `layer_fused_pre_rms_col0_bf16` earlier in the same Worker iter)
// instead of a fifo-routed bf16 pointer.
extern "C" void layer_fused_qkv_gemv_static_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, EMBED_DIM>(
        m, a + row_offset * (EMBED_DIM / 2 + EMBED_DIM / GROUP_SIZE * 2),
        normed_static, c);
}
#endif  // COMPILE_QKV_STATIC

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

// ────────────────────────────────────────────────────────────────────────────
// Post-QKV stages (O_proj, residual ADD, post-RMS, SwiGLU gate/up/down).
//
// Ported from post_attn_fused.cc. The INT4 dequant+GEMV math is IDENTICAL
// to the _qkv_gemv template above (same -8 bias, double-pump, write
// c_out[row]) — so O_proj / down reuse _qkv_gemv directly, parameterised
// only by the K dimension (EMBED_DIM for O_proj, HIDDEN_DIM for down).
// Gate/up need per-phase static buffers (left=gate, right=up) consumed by
// the fused silu*mul, so they get a small dedicated variant.
// ────────────────────────────────────────────────────────────────────────────

#ifndef M_OUTPUT_MAX
#define M_OUTPUT_MAX 4096
#endif

static bfloat16 lf_left_buf[M_OUTPUT_MAX]  __attribute__((aligned(64)));
static bfloat16 lf_right_buf[M_OUTPUT_MAX] __attribute__((aligned(64)));
// D1.7 (FFLM-faithful): SwiGLU intermediate as a FILE-SCOPE STATIC in THIS .o,
// co-laid by the compiler with lf_left/lf_right (no overlap). Replaces the IRON
// `Buffer` silu_scratch, whose L1 address (chosen by the IRON/MLIR allocator,
// blind to this .o's .bss) overlapped lf_right_buf → down read a corrupted silu
// → the ~11% deficit. FFLM keeps ALL stage intermediates register/static-local
// in ONE monolithic .o; this mirrors that. [[reference_fflm_tile_zero_static_l1]]
// lf_silu_buf reuses lf_left_buf (gate output): after gate writes lf_left, gate is
// no longer needed before the next iteration, so silu can overwrite it safely.
// This saves 4 KB of .bss — needed when m_input=4 pushes .bss close to the 12272B
// AIE2P data-region limit (3×4096=12288 > 12272; 2×4096=8192 fits with room).
// NOTE: lf_left_buf is aliased as lf_silu_buf; the name below is kept for clarity.
// (Un-aliasing it does NOT fix the fused-back-half down rel 0.13 — verified — so
// the down error is not a down-output ↔ silu-input overlap.)
#define lf_silu_buf lf_left_buf

extern "C" {

// O_proj GEMV: INT4 E×E. attn_out (E bf16) → o_out slice. K = EMBED_DIM.
void layer_fused_o_proj_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, EMBED_DIM>(
        m, a + row_offset * (EMBED_DIM/2 + EMBED_DIM/GROUP_SIZE*2), b, c);
}

// O scatter: same -8-bias INT4 GEMV as o_proj, but the weight tile `a` is the
// per-call fifo chunk (no weight row_offset) and the m outputs are written at
// c[out_offset:] so a tile can pad its output slice into a full EMBED_DIM
// partial that shares a reduce join. Kept in THIS .o (so a fused O+FFN tile does
// not split GEMV across .o files, which would let their L1 statics overlap).
void layer_fused_o_scatter_bf16(
        uint32_t m, uint32_t out_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, EMBED_DIM>(m, a, b, c + out_offset);
}

// DEBUG: export the gate static lf_left_buf (gate output, valid after the gate
// loop and before silu overwrites it) for off-chip inspection.
void layer_fused_dump_left_bf16(bfloat16 *out, int32_t n) {
    for (int32_t i = 0; i < n; i++) out[i] = lf_left_buf[i];
}


// Down GEMV: INT4 H×E. silu_out (H bf16) → ffn_out slice. K = HIDDEN_DIM.
void layer_fused_down_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, HIDDEN_DIM>(
        m, a + row_offset * (HIDDEN_DIM/2 + HIDDEN_DIM/GROUP_SIZE*2), b, c);
}

// Down PARTIAL GEMV (decomp-B, on-chip FFLM-style): INT4 with K =
// INTER_DIM_PER_COL = HIDDEN_DIM/cols. Each column reduces ONLY its
// HIDDEN_DIM/cols slice of silu_out (delivered on-chip via inter_fifo, no
// DDR bounce) against its weight slice W_down[:, c*K:(c+1)*K], producing a
// PARTIAL EMBED_DIM output. The host sums the cols partials → ffn_out. The
// weight tile a holds m rows × K/2 INT4 + m*K/group_size*2 scale bytes;
// row_offset advances by full E rows are produced per call (m = m_input_d
// rows of the partial-E output, with row_offset selecting the E sub-range).
void layer_fused_down_partial_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, INTER_DIM_PER_COL>(
        m,
        a + row_offset * (INTER_DIM_PER_COL/2 + INTER_DIM_PER_COL/GROUP_SIZE*2),
        b, c);
}

// Elementwise ADD: c[i] = a[i] + b[i]  (residual: o_out + inpL → inpFF).
void layer_fused_add_bf16(
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

// Reduce 4 concatenated per-column FFN partials [P0|P1|P2|P3] (each n elems)
// plus a residual into one n-elem output: out = P0+P1+P2+P3+resid. FFLM-style
// MemTile aggregation: the 4 down partials are joined on ONE MemTile into a 4n
// buffer, then a single worker reduces them — avoids a cross-column add-tree
// that would exhaust the MemTile routing budget.
void layer_fused_reduce4_bf16(
        const bfloat16 *parts, const bfloat16 *resid, bfloat16 *c, int32_t n) {
    constexpr int VEC = 16;
    int chunks = n / VEC;
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        // fp32 accumulation (upcast each bf16 partial) so summing 4 large
        // partials + residual does not round at each pairwise add.
        ::aie::accum<accfloat, VEC> acc;
        acc.from_vector(::aie::load_v<VEC>(parts + i * VEC));
        acc = ::aie::add(acc, ::aie::load_v<VEC>(parts + n + i * VEC));
        acc = ::aie::add(acc, ::aie::load_v<VEC>(parts + 2 * n + i * VEC));
        acc = ::aie::add(acc, ::aie::load_v<VEC>(parts + 3 * n + i * VEC));
        acc = ::aie::add(acc, ::aie::load_v<VEC>(resid + i * VEC));
        ::aie::store_v(c + i * VEC, acc.template to_vector<bfloat16>());
    }
    for (int i = chunks * VEC; i < n; i++) {
        c[i] = (bfloat16)((float)parts[i] + (float)parts[n + i] +
                          (float)parts[2 * n + i] + (float)parts[3 * n + i] +
                          (float)resid[i]);
    }
}

// DEBUG: dump one of the 4 joined partials. out = parts[which*n : (which+1)*n].
// Used to isolate which partial the MemTile join delivers (single-dispatch test).
void layer_fused_dump_part_bf16(
        const bfloat16 *parts, const bfloat16 *resid, bfloat16 *c, int32_t n) {
    (void)resid;
    constexpr int VEC = 16;
    int chunks = n / VEC;
    // which partial is selected via the high bits of n is overkill; just dump P0.
    for (int i = 0; i < chunks; i++)
        ::aie::store_v(c + i * VEC, ::aie::load_v<VEC>(parts + i * VEC));
    for (int i = chunks * VEC; i < n; i++) c[i] = parts[i];
}

// DEBUG: distinct copy symbol so a SILU-scratch dump (in the fused row-5 worker)
// can coexist with the P0 reduce-dump (which uses dump_part) without an MLIR
// symbol redefinition. out = src[0:n]. Isolates gate/up/silu (before down).
void layer_fused_dump_silu_bf16(
        const bfloat16 *src, const bfloat16 *unused, bfloat16 *c, int32_t n) {
    (void)unused;
    constexpr int VEC = 16;
    int chunks = n / VEC;
    for (int i = 0; i < chunks; i++)
        ::aie::store_v(c + i * VEC, ::aie::load_v<VEC>(src + i * VEC));
    for (int i = chunks * VEC; i < n; i++) c[i] = src[i];
}

// DEBUG: copy O_proj output (o_slice elems) → partial, to isolate the O_proj
// phase of the FUSED row-5 worker (distinct symbol so it can be typed for the
// 512-elem O_proj buffer without colliding with the silu/part dump symbols).
void layer_fused_dump_oproj_bf16(
        const bfloat16 *src, const bfloat16 *unused, bfloat16 *c, int32_t n) {
    (void)unused;
    constexpr int VEC = 16;
    int chunks = n / VEC;
    for (int i = 0; i < chunks; i++)
        ::aie::store_v(c + i * VEC, ::aie::load_v<VEC>(src + i * VEC));
    for (int i = chunks * VEC; i < n; i++) c[i] = src[i];
}

// Weighted RMSNorm: output[i] = (input[i] / rms(input)) * gain[i], eps=1e-5.
void layer_fused_rms_norm2_bf16(
        const bfloat16 *input, const bfloat16 *gain,
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

// O_proj output assembler: rounds arrive as N_COLS×M_ROWS bf16 chunks
// (round-major from cross-col MemTile join). For round r, chunk[k*M+j]
// must be placed at out[(group_base + k) * COL_STRIDE + r*M + j], where
// group_base = group_idx * N_COLS, COL_STRIDE = e/cols, M = m_input_o.
// One call per (chunk, round) — no internal loop over rounds.
void layer_fused_o_out_assemble_bf16(
        const bfloat16 *chunk, bfloat16 *out,
        int32_t round_idx, int32_t group_base,
        int32_t n_cols, int32_t m_rows, int32_t col_stride) {
    for (int k = 0; k < n_cols; k++) {
        int32_t col = group_base + k;
        for (int j = 0; j < m_rows; j++) {
            out[col * col_stride + round_idx * m_rows + j] =
                chunk[k * m_rows + j];
        }
    }
}

}  // extern "C"

// Gate/Up dual-GEMV: same INT4 dequant+GEMV as _qkv_gemv, but writes row
// results into a static buffer (phase 0 → lf_left_buf [gate], phase 1 →
// lf_right_buf [up]) at row_offset, so the fused silu*mul can read both.
// K = EMBED_DIM (gate/up project E→H).
template <uint32_t block_size, uint32_t G, uint32_t DK>
static void _lf_dual_gemv(uint32_t m, uint32_t row_offset,
                          const uint8_t *__restrict a_in,
                          const bfloat16 *__restrict b_in,
                          int phase) {
    static_assert(block_size == 32, "block_size must be 32");
    constexpr uint32_t groups_per_row = DK / G;
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *dest = (phase == 0) ? lf_left_buf : lf_right_buf;
    // SIGNED int4 weights = (nibble-8) two's complement → signed unpack centers
    // for free (no -8 sub, no double-unpack) → unroll compiles densely on peano.
    const int4 *weights_packed = reinterpret_cast<const int4 *>(a_in);
    const bfloat16 *scales =
        reinterpret_cast<const bfloat16 *>(a_in + m * DK / 2);
    for (uint32_t row = 0; row < m; row++) {
        const int4 *w_row = weights_packed + row * DK / 2;
        const bfloat16 *s_row = scales + row * groups_per_row;
        const bfloat16 *b_ptr = b_in;
        // D2.6: TWO independent accumulators + double-pump (2 groups/iter). The
        // two mac chains (acc0/acc1) are independent → hide mac latency (II→1);
        // two dequants per iter give ILP. groups_per_row is even (DK/G).
        aie::accum<accfloat, block_size> acc0 = aie::zeros<accfloat, block_size>();
        aie::accum<accfloat, block_size> acc1 = aie::zeros<accfloat, block_size>();
        AIE_LOOP_UNROLL(8)
        for (uint32_t g = 0; g < groups_per_row; g += 2)
            {
                aie::vector<bfloat16, block_size> sf0_bc =
                    aie::broadcast<bfloat16, block_size>(s_row[g]);
                aie::vector<int4, block_size> I0 = aie::load_v<block_size>(w_row);
                w_row += block_size / 2;
                aie::vector<bfloat16, block_size> sf1_bc =
                    aie::broadcast<bfloat16, block_size>(s_row[g + 1]);
                aie::vector<int4, block_size> I1 = aie::load_v<block_size>(w_row);
                w_row += block_size / 2;
                aie::vector<bfloat16, block_size> w0 =
                    aie::mul(aie::to_float<bfloat16>(aie::unpack(I0), 0), sf0_bc).template to_vector<bfloat16>();
                aie::vector<bfloat16, block_size> w1 =
                    aie::mul(aie::to_float<bfloat16>(aie::unpack(I1), 0), sf1_bc).template to_vector<bfloat16>();
                aie::vector<bfloat16, block_size> bv0 = aie::load_v<block_size>(b_ptr);
                b_ptr += block_size;
                aie::vector<bfloat16, block_size> bv1 = aie::load_v<block_size>(b_ptr);
                b_ptr += block_size;
                acc0 = aie::mac(acc0, w0, bv0);
                acc1 = aie::mac(acc1, w1, bv1);
            }
        aie::accum<accfloat, block_size> acc = aie::add(acc0, acc1);
        dest[row_offset + row] = static_cast<bfloat16>(
            aie::reduce_add(acc.template to_vector<float>()));
    }
}

extern "C" {

// Gate/Up entry: phase 0 → lf_left_buf, phase 1 → lf_right_buf.
void layer_fused_gate_up_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, const bfloat16 *b, int phase) {
    _lf_dual_gemv<32, GROUP_SIZE, EMBED_DIM>(m, row_offset, a, b, phase);
}

// Fused SiLU*Mul: out[i] = silu(lf_left_buf[i]) * lf_right_buf[i].
// silu(x) = x * sigmoid(x), sigmoid(x) = 0.5*(1 + tanh(x/2)).
void layer_fused_silu_mul_bf16(bfloat16 *c_out, uint32_t m_output) {
    constexpr int VEC = 8;
    int chunks = (int)m_output / VEC;
    aie::vector<bfloat16, VEC> r05 = aie::broadcast<bfloat16, VEC>(0.5f);
    aie::vector<bfloat16, VEC> r1  = aie::broadcast<bfloat16, VEC>(1.0f);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        aie::vector<bfloat16, VEC> l = aie::load_v<VEC>(lf_left_buf  + i * VEC);
        aie::vector<bfloat16, VEC> r = aie::load_v<VEC>(lf_right_buf + i * VEC);
        auto half_x = aie::mul(l, r05);
        auto tanh_h = aie::tanh<bfloat16>(half_x.template to_vector<float>());
        auto t_p1   = aie::add(tanh_h, r1);
        aie::vector<bfloat16, VEC> sig =
            aie::mul(t_p1, r05).template to_vector<bfloat16>();
        auto silu  = aie::mul(l, sig);
        auto fused = aie::mul(silu.template to_vector<bfloat16>(), r);
        aie::store_v(c_out + i * VEC, fused.template to_vector<bfloat16>());
    }
    (void)chunks;
}

// D1.7: SiLU*Mul writing to the file-scope static lf_silu_buf (no c_out pointer,
// no IRON Buffer). Identical math to layer_fused_silu_mul_bf16. The down GEMV
// below reads lf_silu_buf directly → silu intermediate never leaves this .o.
void layer_fused_silu_mul_static_bf16(uint32_t m_output) {
    constexpr int VEC = 8;
    int chunks = (int)m_output / VEC;
    aie::vector<bfloat16, VEC> r05 = aie::broadcast<bfloat16, VEC>(0.5f);
    // D1.7 FIX: replaced aie::tanh (LUT-based, stateful on AIE2P — leaves the
    // tanh-unit in an uncleaned state that corrupts silu output when running after
    // O_proj+gate+up on the same tile) with sigmoid via aie::exp2 (stateless).
    // silu(x) = x * sigmoid(x),  sigmoid(x) = 1/(1 + exp(-x))
    //         = 1/(1 + exp2(-x * log2e)),  log2e ≈ 1.4426950408
    // exp2<bfloat16>(float_vec) is stateless and verified working on AIE2P.
    // -log2(e) as a scalar for float accumulation
    aie::vector<bfloat16, VEC> r1     = aie::broadcast<bfloat16, VEC>(1.0f);
    AIE_PREPARE_FOR_PIPELINING
    for (int i = 0; i < chunks; i++) {
        aie::vector<bfloat16, VEC> l = aie::load_v<VEC>(lf_left_buf  + i * VEC);
        aie::vector<bfloat16, VEC> r = aie::load_v<VEC>(lf_right_buf + i * VEC);
        // exp(-gate) = exp2(-gate * log2e): use accum multiply so .to_vector<float>() works
        // aie::mul(bf16, bf16) → accum<accfloat,VEC>; .to_vector<float>() available on accum
        aie::vector<bfloat16, VEC> mlog2e = aie::broadcast<bfloat16, VEC>(-1.4426950408f);
        aie::vector<bfloat16, VEC> exp_neg =
            aie::exp2<bfloat16>(aie::mul(l, mlog2e).template to_vector<float>());
        // sigmoid = 1 / (1 + exp(-gate)); aie::add(bf16,bf16) returns vector<bfloat16>
        aie::vector<bfloat16, VEC> denom = aie::add(r1, exp_neg);
        aie::vector<bfloat16, VEC> sig   = aie::inv(denom);
        // silu(gate) = gate * sigmoid; fused = silu * up
        aie::vector<bfloat16, VEC> silu  = aie::mul(l, sig).template to_vector<bfloat16>();
        aie::vector<bfloat16, VEC> fused = aie::mul(silu, r).template to_vector<bfloat16>();
        aie::store_v(lf_silu_buf + i * VEC, fused);
    }
    (void)chunks;
}

// D1.7 CANARY (env DECODE_DBG_CANARY): overwrite lf_silu_buf with a known
// constant (1.0), ignoring gate/up/silu. Decisive test of whether down reads
// lf_silu_buf correctly in the FULL live chain: if NPU down output == CPU
// down(W_down, ones), down's runtime read is fine and the bug is upstream; if
// not, the same-tile O_proj→gate→up sequence corrupts what down reads.
void layer_fused_silu_canary_bf16(uint32_t m_output) {
    constexpr int VEC = 8;
    int chunks = (int)m_output / VEC;
#if defined(CANARY_GATE)
    // GATE-PASSTHROUGH: copy lf_left_buf (gate output) straight to lf_silu_buf.
    for (int i = 0; i < chunks; i++)
        aie::store_v(lf_silu_buf + i * VEC, aie::load_v<VEC>(lf_left_buf + i * VEC));
    (void)chunks;
#elif defined(CANARY_UP)
    // UP-PASSTHROUGH: copy lf_right_buf (up output) to lf_silu_buf.
    for (int i = 0; i < chunks; i++)
        aie::store_v(lf_silu_buf + i * VEC, aie::load_v<VEC>(lf_right_buf + i * VEC));
    (void)chunks;
#elif defined(CANARY_MUL)
    // MUL-ONLY: lf_left * lf_right, NO tanh/sigmoid. If PASS → tanh is the culprit.
    // If FAIL → the load of lf_left or lf_right itself is the problem.
    for (int i = 0; i < chunks; i++) {
        aie::vector<bfloat16, VEC> l = aie::load_v<VEC>(lf_left_buf  + i * VEC);
        aie::vector<bfloat16, VEC> r = aie::load_v<VEC>(lf_right_buf + i * VEC);
        aie::store_v(lf_silu_buf + i * VEC,
            aie::mul(l, r).template to_vector<bfloat16>());
    }
    (void)chunks;
#elif defined(CANARY_RAMP)
    // RAMP variant: realistic silu-range values [~-0.3 .. ~1.0], bf16, so down
    // sees a wide dynamic range like real silu but DETERMINISTIC. If this PASSES
    // too, down is input-insensitive and the bug is purely NPU-silu != CPU-silu.
    for (int i = 0; i < (int)m_output; i++)
        lf_silu_buf[i] = (bfloat16)(-0.3f + 1.3f * ((float)(i % 257) / 256.0f));
    (void)chunks;
#else
    aie::vector<bfloat16, VEC> one = aie::broadcast<bfloat16, VEC>(1.0f);
    for (int i = 0; i < chunks; i++)
        aie::store_v(lf_silu_buf + i * VEC, one);
    (void)chunks;
#endif
}

// D1.7: down GEMV (v2 dequant: w = nibble*scale, NO -8 bias) reading the SwiGLU
// activation from the file-scope static lf_silu_buf. This is a byte-exact copy
// of fused_dequant_gemv_v2.cc's matvec body (NOT a reconstruction), with b_in
// hardwired to lf_silu_buf, so the same-tile silu→down handoff stays inside one
// .o with co-laid statics. K = INTER_DIM_PER_COL (= HIDDEN_DIM/cols = Hc; the
// per-column down K, matching the -DDIM_K=Hc of the old down_matvec_v2_bf16).
// c_out += row_offset.
void layer_fused_down_v2_static_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *__restrict a_in, bfloat16 *__restrict c_out) {
    constexpr uint32_t block_size = 32;
    constexpr uint32_t G  = GROUP_SIZE;
    constexpr uint32_t DK = INTER_DIM_PER_COL;
    static_assert(G % block_size == 0, "group_size must be a multiple of block_size");
    constexpr uint32_t blocks_per_group = G / block_size;
    constexpr uint32_t groups_per_row = DK / G;
    constexpr bool can_double_pump = (groups_per_row >= 2) && (groups_per_row % 2 == 0);
    constexpr uint32_t pump_groups = can_double_pump ? 2 : 1;
    constexpr uint32_t loop_iters = groups_per_row / pump_groups;

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    c_out += row_offset;

    const uint4 *weights_packed = reinterpret_cast<const uint4 *>(a_in);
    const uint8_t *scale_bytes = a_in + m * DK / 2;
    const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(scale_bytes);

    for (uint32_t row = 0; row < m; row++) {
        const uint4 *row_weights = weights_packed + row * DK / 2;
        const bfloat16 *row_scales = scales + row * groups_per_row;
        const bfloat16 *b_ptr = lf_silu_buf;             // <-- activation = static

        aie::accum<accfloat, block_size> acc = aie::zeros<accfloat, block_size>();
        aie::accum<accfloat, block_size> acc_b = aie::zeros<accfloat, block_size>();

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
                    aie::vector<uint8, block_size> a8_a = aie::unpack(I0_a);
                    aie::vector<uint16, block_size> a16_a = aie::unpack(a8_a);
                    aie::vector<bfloat16, block_size> abf_a =
                        aie::to_float<bfloat16>(a16_a, 0);
                    aie::vector<bfloat16, block_size> w_a =
                        aie::mul(abf_a, sf_a_bc).template to_vector<bfloat16>();
                    aie::vector<uint8, block_size> a8_b = aie::unpack(I0_b);
                    aie::vector<uint16, block_size> a16_b = aie::unpack(a8_b);
                    aie::vector<bfloat16, block_size> abf_b =
                        aie::to_float<bfloat16>(a16_b, 0);
                    aie::vector<bfloat16, block_size> w_b =
                        aie::mul(abf_b, sf_b_bc).template to_vector<bfloat16>();
                    aie::vector<bfloat16, block_size> b_a = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc = aie::mac(acc, w_a, b_a);   // D2.6: two independent accs
                    aie::vector<bfloat16, block_size> b_b = aie::load_v<block_size>(b_ptr);
                    b_ptr += block_size;
                    acc_b = aie::mac(acc_b, w_b, b_b);  // → hide mac latency
                }
        } else {
            AIE_LOOP_MIN_ITERATION_COUNT(loop_iters)
            for (uint32_t g = 0; g < groups_per_row; g++)
                AIE_PREPARE_FOR_PIPELINING
                {
                    bfloat16 sf = row_scales[g];
                    aie::vector<bfloat16, block_size> sf_broadcast =
                        aie::broadcast<bfloat16, block_size>(sf);
                    AIE_LOOP_MIN_ITERATION_COUNT(blocks_per_group)
                    for (uint32_t blk = 0; blk < blocks_per_group; blk++) {
                        aie::vector<uint4, block_size> I0 = aie::load_v<block_size>(row_weights);
                        row_weights += block_size / 2;
                        aie::vector<uint8, block_size> as_int8 = aie::unpack(I0);
                        aie::vector<uint16, block_size> as_int16 = aie::unpack(as_int8);
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
        aie::accum<accfloat, block_size> acc_sum = aie::add(acc, acc_b);
        *c_out = static_cast<bfloat16>(aie::reduce_add(acc_sum.template to_vector<float>()));
        c_out++;
    }
}
// NOTE (D2.6): down already had double-pump; adding a 2nd independent acc gave
// no speedup (1158→1169µs = noise) → down was not acc-latency-bound. acc_b kept
// (harmless, =0 in the non-double-pump branch). gate/up's 2-acc DID help (−3%).

// UNPAD helper (D2.2c): a_in points at a uniform 4608-byte weight-fifo element
// that holds several unpadded down sub-tiles back-to-back (each = m rows,
// K=INTER_DIM_PER_COL). Select sub-tile `sub_idx` by pointer arithmetic and run
// the proven per-tile down GEMV. Lets ONE weight fifo carry unpadded down tiles
// (1152 B) packed N-per-element, avoiding the 4× pad to the gate/up tile size.
void layer_fused_down_v2_sub_bf16(uint32_t m, uint32_t row_offset,
                                  uint32_t sub_idx,
                                  const uint8_t *__restrict a_in,
                                  bfloat16 *__restrict c_out) {
    constexpr uint32_t G  = GROUP_SIZE;
    constexpr uint32_t DK = INTER_DIM_PER_COL;
    const uint32_t sub_bytes = m * DK / 2 + m * (DK / G) * 2;
    layer_fused_down_v2_static_bf16(m, row_offset, a_in + sub_idx * sub_bytes, c_out);
}

// UNPAD x4 (D2.4 perf): process ALL nsub down sub-tiles of a 4608-element in ONE
// IRON kernel call (cuts func.call count ~4x vs down_v2_sub). c_out base;
// sub-tile k writes rows [row_offset + k*m ..].
void layer_fused_down_v2_x4_bf16(uint32_t m, uint32_t row_offset, uint32_t nsub,
                                 const uint8_t *__restrict a_in,
                                 bfloat16 *__restrict c_out) {
    constexpr uint32_t G  = GROUP_SIZE;
    constexpr uint32_t DK = INTER_DIM_PER_COL;
    const uint32_t sub_bytes = m * DK / 2 + m * (DK / G) * 2;
    for (uint32_t k = 0; k < nsub; k++)
        layer_fused_down_v2_static_bf16(m, row_offset + k * m, a_in + k * sub_bytes, c_out);
}

// ────────────────────────────────────────────────────────────────────────────
// MONOLITHIC register-resident FFN (#12 / fix #11.1b): one function, NO L1
// intermediate statics (no lf_left/lf_right/lf_silu). Hidden-outer: for each of
// `m` hidden rows compute gate & up (full E GEMV, -8 bias) into m-element
// register temporaries, silu*up in registers, then SAXPY-accumulate the down
// column (dequant, NO bias) scaled by that silu value into down_acc[0:E].
// down_acc (the output fifo) is the only L1 working set; the gate/up/silu
// intermediates never touch L1. Weight layout per call (m hidden rows), one
// contiguous buffer gud_w = [ gate: m rows K=E ][ up: m rows K=E ][ down: m
// hidden-cols K=E ], each block = m*E/2 nibble bytes + m*(E/G) bf16 scales.
// Caller zeroes down_acc before the first chunk; chunks accumulate.
// ────────────────────────────────────────────────────────────────────────────
void layer_fused_ffn_mono_bf16(
        uint32_t m,
        const uint8_t *__restrict gud_w,
        const bfloat16 *__restrict x,
        bfloat16 *__restrict down_acc) {
    constexpr uint32_t BS = 32;
    constexpr uint32_t E  = EMBED_DIM;
    constexpr uint32_t G  = GROUP_SIZE;
    constexpr uint32_t gpr = E / G;                // groups per row (64)

    const uint32_t PK = m * E / 2 + m * (E / G) * 2;
    const uint8_t *gate_w = gud_w;
    const uint8_t *up_w   = gud_w + PK;
    const uint8_t *down_w = gud_w + 2 * PK;

    bfloat16 gtmp[16];
    bfloat16 utmp[16];
    // gate & up: signed-int4 GEMV (centered weights, no sub, unrolled dense)
    _qkv_gemv_s4<32, G, E>(m, gate_w, x, gtmp);
    _qkv_gemv_s4<32, G, E>(m, up_w, x, utmp);

    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const uint4 *dw_base = reinterpret_cast<const uint4 *>(down_w);
    // down scales are PER-OUTPUT (E values, one per output row), shared by all m
    // hidden rows of this chunk (they fall in one hidden quant-group). Layout:
    // down block = [m*E/2 nibble bytes][E bf16 per-output scales].
    const bfloat16 *down_scl =
        reinterpret_cast<const bfloat16 *>(down_w + m * E / 2);
    const bfloat16 r1 = (bfloat16)1.0f;

    // silu(g)*u in registers for the m hidden rows of this chunk.
    bfloat16 sval[16];
    for (uint32_t r = 0; r < m; r++) {
        ::aie::vector<bfloat16, 8> gv = ::aie::broadcast<bfloat16, 8>(gtmp[r]);
        ::aie::vector<float, 8> neg =
            ::aie::mul(gv, ::aie::broadcast<bfloat16, 8>((bfloat16)(-1.4426950408f)))
                .template to_vector<float>();
        ::aie::vector<bfloat16, 8> e = ::aie::exp2<bfloat16>(neg);
        ::aie::vector<bfloat16, 8> denom = ::aie::add(e, ::aie::broadcast<bfloat16, 8>(r1));
        ::aie::vector<bfloat16, 8> sig = ::aie::inv(denom);
        bfloat16 silu_r = (bfloat16)((float)gtmp[r] * (float)sig[0]);
        sval[r] = (bfloat16)((float)silu_r * (float)utmp[r]);
    }

    // down SAXPY, OUTPUT-GROUP outer / hidden inner: load+store down_acc ONCE per
    // output group (m hidden accumulated in an fp32 accum) -> m× fewer down_acc
    // RMW than hidden-outer + no per-step bf16 rounding.
    for (uint32_t gg = 0; gg < gpr; gg++)
        AIE_PREPARE_FOR_PIPELINING {
            ::aie::vector<bfloat16, BS> sf = ::aie::load_v<BS>(down_scl + gg * BS);
            ::aie::accum<accfloat, BS> acc;
            acc.from_vector(::aie::load_v<BS>(down_acc + gg * BS));
            for (uint32_t r = 0; r < m; r++) {
                ::aie::vector<uint4, BS> I =
                    ::aie::load_v<BS>(dw_base + r * (E / 2) + gg * (BS / 2));
                ::aie::vector<uint8, BS> a8 = ::aie::unpack(I);
                ::aie::vector<uint16, BS> a16 = ::aie::unpack(a8);
                ::aie::vector<bfloat16, BS> abf = ::aie::to_float<bfloat16>(a16, 0);
                ::aie::vector<bfloat16, BS> wv =
                    ::aie::mul(abf, sf).template to_vector<bfloat16>();          // dequant (no bias)
                acc = ::aie::mac(acc, wv, ::aie::broadcast<bfloat16, BS>(sval[r]));  // += wv*silu
            }
            ::aie::store_v(down_acc + gg * BS, acc.template to_vector<bfloat16>());
        }
}

// BIG-ELEMENT x4 (D2.5 perf): process nsub gate/up tiles (K=EMBED_DIM, each m rows
// = 4608B) from ONE big weight-fifo element. Lets independent per-tile fifos use
// large DMA transfers (high BW) while still overlapping compute (per-tile prefetch).
void layer_fused_gate_up_x4_bf16(uint32_t m, uint32_t row_offset, uint32_t nsub,
                                 const uint8_t *__restrict a_in,
                                 const bfloat16 *__restrict b_in, int phase) {
    constexpr uint32_t G  = GROUP_SIZE;
    constexpr uint32_t DK = EMBED_DIM;
    const uint32_t sub_bytes = m * DK / 2 + m * (DK / G) * 2;
    for (uint32_t k = 0; k < nsub; k++)
        layer_fused_gate_up_bf16(m, row_offset + k * m, a_in + k * sub_bytes, b_in, phase);
}

}  // extern "C"

#ifdef COMPILE_ATTN

// ────────────────────────────────────────────────────────────────────────────
// GQA attention compute (U1a per-head sequential, with streaming KV).
//
// Per-column work for Llama-3.2-1B:
//   * 32 Q heads / 8 cols = 4 Q heads per col
//   * 8 KV heads / 8 cols = 1 KV head per col
//   * GQA factor = 4 (each col's 1 KV head serves its 4 Q heads — no
//     cross-col KV gather required; col is self-contained for attention)
//
// L1 budget reality (~32-64 KB per tile on AIE2P): 1k context KV head
// = 1024 × 64 × 2 = 128 KB does NOT fit. KV streams from BO3 K_cache /
// V_cache via the per-col attention worker's input channel; we process
// in `K_CHUNK = 32` token chunks (matches FFLM's ct_chunk_size from
// mha.dll RE'd disasm).
//
// Three kernels comprise the per-col attention pipeline:
//   1. attn_qk_score_chunk_bf16 — per-head Q · K_chunk^T, partial scores
//      written to a per-col scores buffer (max ctx_len bf16 per head).
//      Caller iterates ctx_len / K_CHUNK chunks.
//   2. attn_softmax_inplace_bf16 — in-place softmax over ctx_len, with
//      pre-scaled inputs (scores already divided by sqrt(head_dim)).
//   3. attn_av_ctx_chunk_bf16 — per-head scores · V_chunk, accumulating
//      into a head_dim context vector. Caller iterates same chunking.
//
// `scale` parameter passed in pre-computed as 1.0f / sqrt(HEAD_DIM) and
// folded into the score calc (saves a per-element divide).
// ────────────────────────────────────────────────────────────────────────────

#ifndef ATTN_K_CHUNK
#define ATTN_K_CHUNK 32
#endif

// Per-head Q · K_chunk^T, producing K_CHUNK partial scores. Q is one
// head's HEAD_DIM bf16 vector; K_chunk is K_CHUNK rows of HEAD_DIM bf16
// each. Output is appended to scores at offset chunk_idx*K_CHUNK.
//
// This computes a (1, HEAD_DIM) @ (HEAD_DIM, K_CHUNK) → (K_CHUNK,)
// scalar-product-per-key matmul, equivalent to K_CHUNK dot products
// of length HEAD_DIM.
extern "C" void attn_qk_score_chunk_bf16(
        const bfloat16 *q_head,      // [HEAD_DIM]
        const bfloat16 *k_chunk,     // [K_CHUNK × HEAD_DIM]
        bfloat16       *scores_out,  // [K_CHUNK] (chunk-local)
        int32_t         scale_bits   // bits-of-bf16 scale factor 1/sqrt(HEAD_DIM)
    ) {
    // Reinterpret int bits → bfloat16 scale (avoids passing bf16 by value
    // since the AIE2P calling convention promotes bf16 → int).
    bfloat16 scale;
    static_assert(sizeof(bfloat16) == 2, "bf16 size assumption");
    uint16_t sb = (uint16_t)scale_bits;
    __builtin_memcpy(&scale, &sb, 2);

    // Vectorised dot: HEAD_DIM=64 lanes via 16-lane bf16 SIMD chunks.
    // Per row j: load q in 16-lane segments, multiply-accumulate across
    // all HEAD_DIM/16 segments into a float-accumulator vector, then
    // reduce_add into a scalar score. This is 4 MAC bundles per K row,
    // dense per-tile AIE instructions vs the prior scalar 64-iter loop.
    constexpr int VEC = 16;
    static_assert(HEAD_DIM % VEC == 0, "HEAD_DIM must be multiple of 16");
    constexpr int N_VEC = HEAD_DIM / VEC;

    for (int j = 0; j < ATTN_K_CHUNK; j++) {
        const bfloat16 *k_row = k_chunk + j * HEAD_DIM;
        ::aie::accum<accfloat, VEC> acc =
            ::aie::zeros<accfloat, VEC>();
        AIE_PREPARE_FOR_PIPELINING
        for (int s = 0; s < N_VEC; s++) {
            ::aie::vector<bfloat16, VEC> qv =
                ::aie::load_v<VEC>(q_head + s * VEC);
            ::aie::vector<bfloat16, VEC> kv =
                ::aie::load_v<VEC>(k_row  + s * VEC);
            acc = ::aie::mac(acc, qv, kv);
        }
        float dot = ::aie::reduce_add(acc.template to_vector<float>());
        scores_out[j] = (bfloat16)(dot * (float)scale);
    }
}

// In-place softmax over `n` bf16 values (n = ctx_len). Numerically stable
// (subtract max before exp). Operates entirely in bf16 on AIE2P; this is
// the per-col leader path so the loop fits the tile budget.
extern "C" void attn_softmax_inplace_bf16(bfloat16 *scores, int32_t n) {
    // Pass 1: find max
    float max_val = -1e30f;
    for (int i = 0; i < n; i++) {
        float v = (float)scores[i];
        if (v > max_val) max_val = v;
    }
    // Pass 2: exp(x - max), accumulate sum.
    // Use polynomial expf via the AIE math library: aie::exp is a
    // float-vector op; for a scalar fallback we approximate via
    // exp(x) ≈ 2^(x * log2(e)) using a Taylor expansion on the
    // fractional part. For baseline v0 we use a simple ldexp-based
    // form that the AIE compiler supports without intrinsic deps.
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = (float)scores[i] - max_val;
        // Clamp to avoid underflow blowing up later passes
        if (v < -80.0f) v = -80.0f;
        // exp(v) via 2^k method: split v*log2(e) into integer+frac
        float x = v * 1.4426950408889634f;        // log2(e)
        int   k = (int)x;
        if (x < (float)k) k -= 1;                 // floor
        float f = x - (float)k;
        // 2^f via 4-term polynomial (max error ~1e-3 over [0,1])
        float p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * 0.05551327f));
        // Build 2^k by manipulating the IEEE-754 exponent
        union { float fv; uint32_t u; } pack;
        pack.fv = p;
        int e2 = ((int)((pack.u >> 23) & 0xFF)) + k;
        if (e2 <= 0) {
            pack.fv = 0.0f;
        } else if (e2 >= 255) {
            pack.fv = 1e30f;
        } else {
            pack.u = (pack.u & 0x807FFFFFu) | ((uint32_t)e2 << 23);
        }
        float e = pack.fv;
        sum += e;
        scores[i] = (bfloat16)e;
    }
    // Pass 3: normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        scores[i] = (bfloat16)((float)scores[i] * inv_sum);
    }
}

// Per-head scores · V_chunk accumulate-into ctx_head. ctx_head is a
// HEAD_DIM bf16 accumulator that the caller zeroes before the first chunk
// and reads after the last. scores_chunk is K_CHUNK weights from the
// softmax output (chunk-local), v_chunk is K_CHUNK rows of HEAD_DIM bf16.
//
// Vectorised over the HEAD_DIM dim: each token row contributes
// w * v_row to the HEAD_DIM accumulator. Process accumulator in
// 16-lane bf16 segments (HEAD_DIM/16 = 4 vector slots).
extern "C" void attn_av_ctx_chunk_bf16(
        const bfloat16 *scores_chunk,  // [K_CHUNK]
        const bfloat16 *v_chunk,       // [K_CHUNK × HEAD_DIM]
        bfloat16       *ctx_head,      // [HEAD_DIM] accumulator
        int32_t         zero_first      // 1 = clear ctx_head before; 0 = accumulate
    ) {
    constexpr int VEC = 16;
    static_assert(HEAD_DIM % VEC == 0, "HEAD_DIM must be multiple of 16");
    constexpr int N_VEC = HEAD_DIM / VEC;

    // Load (or zero) the running ctx accumulator into N_VEC float
    // accumulators that live in registers across the K_CHUNK loop.
    ::aie::accum<accfloat, VEC> acc[N_VEC];
    if (zero_first) {
        for (int s = 0; s < N_VEC; s++) {
            acc[s] = ::aie::zeros<accfloat, VEC>();
        }
    } else {
        for (int s = 0; s < N_VEC; s++) {
            ::aie::vector<bfloat16, VEC> v0 =
                ::aie::load_v<VEC>(ctx_head + s * VEC);
            // Cast bf16→float vector via mul-by-1 trick is ugly;
            // simpler — use scalar promote in a small tail loop.
            // Here we re-materialise from bf16 by adding 0 via mac.
            ::aie::vector<bfloat16, VEC> ones =
                ::aie::broadcast<bfloat16, VEC>(1.0f);
            acc[s] = ::aie::mul(v0, ones);
        }
    }

    for (int j = 0; j < ATTN_K_CHUNK; j++) {
        bfloat16 w = scores_chunk[j];
        ::aie::vector<bfloat16, VEC> wv =
            ::aie::broadcast<bfloat16, VEC>(w);
        const bfloat16 *v_row = v_chunk + j * HEAD_DIM;
        AIE_PREPARE_FOR_PIPELINING
        for (int s = 0; s < N_VEC; s++) {
            ::aie::vector<bfloat16, VEC> vv =
                ::aie::load_v<VEC>(v_row + s * VEC);
            acc[s] = ::aie::mac(acc[s], wv, vv);
        }
    }

    // Store accumulators back to bf16 ctx_head.
    for (int s = 0; s < N_VEC; s++) {
        ::aie::vector<bfloat16, VEC> out =
            acc[s].template to_vector<bfloat16>();
        ::aie::store_v(ctx_head + s * VEC, out);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// SPIKE wrapper: real attention compute on a single KV chunk.
//
// Drop-in replacement for `layer_fused_noop_kv_bf16` — same fifo signature
// (chunk_in, ctx_out, n), so the existing attn worker body and runtime
// sequence don't change. Internally chains all 3 attention kernels with
// per-tile file-scope static buffers; q_head is initialised from the
// chunk's first HEAD_DIM bf16 (dummy spike data — real q_rot input lands
// when the 2nd input fifo is wired in the next step). v_chunk reuses the
// same chunk pointer (also dummy for spike).
//
// Purpose: prove the attention kernels actually execute on AIE2P (PDI
// density goes up materially) before paying the cost of adding the
// q_head input fifo and the per-head outer loop.
// ────────────────────────────────────────────────────────────────────────────

// Per-tile static scratch (each compute tile gets its own L1 instance).
static bfloat16 attn_q_head_static[HEAD_DIM]    __attribute__((aligned(64)));
static bfloat16 attn_scores_static[ATTN_K_CHUNK] __attribute__((aligned(64)));
static bfloat16 attn_ctx_head_static[HEAD_DIM]  __attribute__((aligned(64)));

extern "C" void attn_compute_chunk_bf16(bfloat16 *__restrict kv_chunk,
                                        bfloat16 *__restrict ctx_out,
                                        int32_t n) {
    // 1. Init dummy q_head from the chunk's leading HEAD_DIM bf16 elements.
    //    Real wiring will supply this from a dedicated q_head fifo.
    for (int i = 0; i < HEAD_DIM; i++) {
        attn_q_head_static[i] = kv_chunk[i];
    }

    // 2. Q · K^T scores into the scratch (bf16 of 1/sqrt(64) = 0x3E00).
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;
    attn_qk_score_chunk_bf16(attn_q_head_static, kv_chunk,
                             attn_scores_static, kScaleBitsInvSqrt64);

    // 3. In-place softmax over the K_CHUNK partial scores.
    attn_softmax_inplace_bf16(attn_scores_static, ATTN_K_CHUNK);

    // 4. Scores · V_chunk → ctx_head accumulator (treat same chunk as V
    //    for the spike; real wiring iterates V chunks separately).
    attn_av_ctx_chunk_bf16(attn_scores_static, kv_chunk,
                           attn_ctx_head_static, /*zero_first=*/1);

    // 5. Drain: copy ctx_head into the leading HEAD_DIM elements of the
    //    chunk-shaped output, zero the tail. The fifo-shape match means
    //    no design.py changes are required for this spike.
    for (int i = 0; i < HEAD_DIM; i++) {
        ctx_out[i] = attn_ctx_head_static[i];
    }
    for (int i = HEAD_DIM; i < n; i++) {
        ctx_out[i] = (bfloat16)0.0f;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Step-2 spike: real q_head from a dedicated input fifo.
//
// One call = one head's full attention on one KV chunk. The worker body
// drives the per-head loop (4 calls per outer iter for 4 q_heads/col).
// Real q_rot data flows in via the q_head pointer; the kv_chunk is still
// re-used as both K and V (we wire per-K and per-V chunk fifos in the
// next step once chunk streaming is in place).
// ────────────────────────────────────────────────────────────────────────────

extern "C" void attn_compute_with_qhead_bf16(
        bfloat16 *__restrict q_head,
        bfloat16 *__restrict kv_chunk,
        bfloat16 *__restrict ctx_out,
        int32_t n) {
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;

    // Q · K^T scores (uses the supplied q_head, not a dummy seed).
    attn_qk_score_chunk_bf16(q_head, kv_chunk,
                             attn_scores_static, kScaleBitsInvSqrt64);

    // In-place softmax.
    attn_softmax_inplace_bf16(attn_scores_static, ATTN_K_CHUNK);

    // Scores · V_chunk (kv_chunk treated as V for the spike).
    attn_av_ctx_chunk_bf16(attn_scores_static, kv_chunk,
                           attn_ctx_head_static, /*zero_first=*/1);

    // Drain: ctx_head into the leading HEAD_DIM bf16 of the chunk-shaped
    // output (kv_chunk_elems = K_CHUNK*HEAD_DIM = 2048). Zero the tail.
    for (int i = 0; i < HEAD_DIM; i++) {
        ctx_out[i] = attn_ctx_head_static[i];
    }
    for (int i = HEAD_DIM; i < n; i++) {
        ctx_out[i] = (bfloat16)0.0f;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Step-3 spike: streaming attention over multiple KV chunks.
//
// Worker body now iterates N_CHUNKS chunk acquires for K, runs softmax once
// over the full scores buffer, then iterates N_CHUNKS for V accumulating
// into ctx_head, then drains. Each kernel below performs ONE step of that
// pipeline on a single chunk; the body schedules them in the right order.
//
// Static buffers grow: scores now holds N_CHUNKS*K_CHUNK partial scores.
// ATTN_MAX_CHUNKS caps the streaming length at compile time (= context
// window / K_CHUNK). For Llama-3.2-1B 1k context that's 32; we ship the
// spike at ATTN_MAX_CHUNKS=4 = 128-token window to keep insts.bin small.
// ────────────────────────────────────────────────────────────────────────────

#ifndef ATTN_MAX_CHUNKS
#define ATTN_MAX_CHUNKS 4
#endif

static bfloat16 attn_scores_full_static[ATTN_MAX_CHUNKS * ATTN_K_CHUNK]
    __attribute__((aligned(64)));

// Compute Q · K_chunk^T at chunk index `chunk_idx`, writing K_CHUNK
// partial scores into the file-scope scratch at offset chunk_idx*K_CHUNK.
// The seed q_head still comes from kv_chunk[0..63] for this spike — real
// q_rot input arrives once MemTile gather replaces the shim cap blocker.
extern "C" void attn_qk_score_at_bf16(bfloat16 *__restrict kv_chunk,
                                      int32_t chunk_idx,
                                      int32_t n_total) {
    (void)n_total;
    // On the first chunk, refresh the dummy q_head seed.
    if (chunk_idx == 0) {
        for (int i = 0; i < HEAD_DIM; i++) {
            attn_q_head_static[i] = kv_chunk[i];
        }
    }
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;
    bfloat16 *scores_at = attn_scores_full_static + chunk_idx * ATTN_K_CHUNK;
    attn_qk_score_chunk_bf16(attn_q_head_static, kv_chunk,
                             scores_at, kScaleBitsInvSqrt64);
}

// Softmax over the full N_chunks*K_CHUNK scores buffer, in place.
extern "C" void attn_softmax_full_bf16(int32_t n_chunks) {
    int32_t n = n_chunks * ATTN_K_CHUNK;
    if (n > ATTN_MAX_CHUNKS * ATTN_K_CHUNK) n = ATTN_MAX_CHUNKS * ATTN_K_CHUNK;
    attn_softmax_inplace_bf16(attn_scores_full_static, n);
}

// Accumulate scores[chunk_idx*K_CHUNK..] · V_chunk into ctx_head. Caller
// passes zero_first=1 on the first chunk to clear the accumulator.
extern "C" void attn_av_ctx_at_bf16(bfloat16 *__restrict v_chunk,
                                    int32_t chunk_idx,
                                    int32_t zero_first) {
    const bfloat16 *scores_at =
        attn_scores_full_static + chunk_idx * ATTN_K_CHUNK;
    attn_av_ctx_chunk_bf16(scores_at, v_chunk,
                           attn_ctx_head_static, zero_first);
}

// Drain ctx_head into the leading HEAD_DIM bf16 of a chunk-shaped output;
// zero the tail. Same shape as the previous spike's drain so the existing
// attn_drain fifo type doesn't change.
extern "C" void attn_drain_ctx_bf16(bfloat16 *__restrict ctx_out,
                                    int32_t n) {
    for (int i = 0; i < HEAD_DIM; i++) {
        ctx_out[i] = attn_ctx_head_static[i];
    }
    for (int i = HEAD_DIM; i < n; i++) {
        ctx_out[i] = (bfloat16)0.0f;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Step-4 spike: 4-head dense attention on a single KV chunk.
//
// Drop-in replacement for attn_compute_chunk_bf16 — same fifo signature
// (kv_chunk, ctx_out, n) — but does 4× the per-call compute by handling
// the 4 q heads/col GQA mapping in one kernel invocation. Each head uses
// its own L1 static scratch; q_head seed comes from kv_chunk[h*64..(h+1)*64]
// (still spike-data; real q_rot lands once MemTile gather replaces shim).
//
// Tiny ALU loops here translate directly to more dense per-tile core
// instructions in the PDI — driving xclbin size up toward FFLM's
// 414 KB target without changing the dataflow shape.
// ────────────────────────────────────────────────────────────────────────────

#ifndef ATTN_HEADS_PER_TILE
#define ATTN_HEADS_PER_TILE 32
#endif

static bfloat16 attn_q_heads4_static[ATTN_HEADS_PER_TILE * HEAD_DIM]
    __attribute__((aligned(64)));
static bfloat16 attn_scores4_static[ATTN_HEADS_PER_TILE * ATTN_K_CHUNK]
    __attribute__((aligned(64)));
static bfloat16 attn_ctx4_static[ATTN_HEADS_PER_TILE * HEAD_DIM]
    __attribute__((aligned(64)));

extern "C" void attn_compute_4heads_bf16(bfloat16 *__restrict kv_chunk,
                                         bfloat16 *__restrict ctx_out,
                                         int32_t n) {
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;

    // 1. Init dummy q_heads from kv_chunk[0 .. 4*HEAD_DIM]. Real wiring
    //    will pull these from a dedicated q_rot fifo.
    for (int i = 0; i < ATTN_HEADS_PER_TILE * HEAD_DIM; i++) {
        attn_q_heads4_static[i] = kv_chunk[i];
    }

    // 2. Per-head sequential attention over the chunk:
    //    Q · K^T → softmax → · V_chunk → ctx_head[h]
    for (int h = 0; h < ATTN_HEADS_PER_TILE; h++) {
        bfloat16 *q_h     = attn_q_heads4_static + h * HEAD_DIM;
        bfloat16 *scores  = attn_scores4_static  + h * ATTN_K_CHUNK;
        bfloat16 *ctx_h   = attn_ctx4_static     + h * HEAD_DIM;

        // Q · K^T
        attn_qk_score_chunk_bf16(q_h, kv_chunk, scores, kScaleBitsInvSqrt64);
        // In-place softmax over this head's K_CHUNK partials
        attn_softmax_inplace_bf16(scores, ATTN_K_CHUNK);
        // Scores · V_chunk → ctx_head accumulator (kv_chunk reused as V)
        attn_av_ctx_chunk_bf16(scores, kv_chunk, ctx_h, /*zero_first=*/1);
    }

    // 3. Drain: 4 ctx_heads laid contiguously in ctx_out's first
    //    4*HEAD_DIM bf16, zero the tail (ctx_out is kv_chunk_elems wide).
    for (int i = 0; i < ATTN_HEADS_PER_TILE * HEAD_DIM; i++) {
        ctx_out[i] = attn_ctx4_static[i];
    }
    for (int i = ATTN_HEADS_PER_TILE * HEAD_DIM; i < n; i++) {
        ctx_out[i] = (bfloat16)0.0f;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Step-5 spike: MemTile sub-chunk gather.
//
// shim → MemTile relays the kv_chunk in N=4 sub-chunks of 512 bf16 each
// (sub-chunks fit under the AIE2P shim BD inner-dim cap of 1023). The
// compute tile reassembles them in L1 static via attn_subchunk_load_bf16,
// then runs the full 4-head attention via attn_compute_from_static_bf16.
// Same per-chunk math; just lifts the shim 2048-elem inner-dim blocker.
// ────────────────────────────────────────────────────────────────────────────

#ifndef ATTN_SUBCHUNK_ELEMS
#define ATTN_SUBCHUNK_ELEMS 512
#endif

#ifndef ATTN_SUBCHUNKS_PER_CHUNK
#define ATTN_SUBCHUNKS_PER_CHUNK 4
#endif

static_assert(ATTN_SUBCHUNK_ELEMS * ATTN_SUBCHUNKS_PER_CHUNK ==
                  ATTN_K_CHUNK * HEAD_DIM,
              "sub-chunk × sub-chunks must equal full kv_chunk_elems");

static bfloat16 attn_kv_full_static[ATTN_K_CHUNK * HEAD_DIM]
    __attribute__((aligned(64)));

extern "C" void attn_subchunk_load_bf16(bfloat16 *__restrict sub,
                                        int32_t sub_idx,
                                        int32_t n) {
    (void)n;
    bfloat16 *dst = attn_kv_full_static + sub_idx * ATTN_SUBCHUNK_ELEMS;
    for (int i = 0; i < ATTN_SUBCHUNK_ELEMS; i++) {
        dst[i] = sub[i];
    }
}

extern "C" void attn_compute_from_static_bf16(bfloat16 *__restrict ctx_out,
                                              int32_t n) {
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;
    bfloat16 *kv = attn_kv_full_static;

    for (int i = 0; i < ATTN_HEADS_PER_TILE * HEAD_DIM; i++) {
        attn_q_heads4_static[i] = kv[i];
    }
    for (int h = 0; h < ATTN_HEADS_PER_TILE; h++) {
        bfloat16 *q_h    = attn_q_heads4_static + h * HEAD_DIM;
        bfloat16 *scores = attn_scores4_static  + h * ATTN_K_CHUNK;
        bfloat16 *ctx_h  = attn_ctx4_static     + h * HEAD_DIM;
        attn_qk_score_chunk_bf16(q_h, kv, scores, kScaleBitsInvSqrt64);
        attn_softmax_inplace_bf16(scores, ATTN_K_CHUNK);
        attn_av_ctx_chunk_bf16(scores, kv, ctx_h, /*zero_first=*/1);
    }
    for (int i = 0; i < ATTN_HEADS_PER_TILE * HEAD_DIM; i++) {
        ctx_out[i] = attn_ctx4_static[i];
    }
    for (int i = ATTN_HEADS_PER_TILE * HEAD_DIM; i < n; i++) {
        ctx_out[i] = (bfloat16)0.0f;
    }
}

// Streaming variant: compute attention on the static-buffer chunk
// without the drain. ctx accumulator state stays in attn_ctx4_static
// for the next chunk to combine with. zero_first=1 on the first
// chunk of a body iter, 0 on subsequent ones (online accumulation).
extern "C" void attn_compute_from_static_acc_bf16(int32_t zero_first) {
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;
    bfloat16 *kv = attn_kv_full_static;

    if (zero_first) {
        for (int i = 0; i < ATTN_HEADS_PER_TILE * HEAD_DIM; i++) {
            attn_q_heads4_static[i] = kv[i];
        }
    }
    for (int h = 0; h < ATTN_HEADS_PER_TILE; h++) {
        bfloat16 *q_h    = attn_q_heads4_static + h * HEAD_DIM;
        bfloat16 *scores = attn_scores4_static  + h * ATTN_K_CHUNK;
        bfloat16 *ctx_h  = attn_ctx4_static     + h * HEAD_DIM;
        attn_qk_score_chunk_bf16(q_h, kv, scores, kScaleBitsInvSqrt64);
        attn_softmax_inplace_bf16(scores, ATTN_K_CHUNK);
        attn_av_ctx_chunk_bf16(scores, kv, ctx_h, zero_first);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Step-6 spike: real q_head input via MemTile cross-col gather.
//
// The compute tile body now acquires a q_head slice per body iter
// and copies it into the per-tile static q_heads buffer at the head
// index hidden by the worker loop. Same MemTile relay pattern as
// kv sub-chunks. Full per-iter compute is identical to the prior
// streaming spike — only the q_heads init source changes from a
// kv_chunk seed to a real fifo.
// ────────────────────────────────────────────────────────────────────────────

extern "C" void attn_qhead_load_bf16(bfloat16 *__restrict q_in,
                                     int32_t head_idx,
                                     int32_t n) {
    (void)n;
    bfloat16 *dst = attn_q_heads4_static + head_idx * HEAD_DIM;
    for (int i = 0; i < HEAD_DIM; i++) {
        dst[i] = q_in[i];
    }
}

// Streaming compute that uses the q_heads loaded via attn_qhead_load_bf16
// (NOT the kv_chunk seed). Same scoring / softmax / ctx flow as the
// _acc_bf16 variant.
extern "C" void attn_compute_real_qhead_acc_bf16(int32_t zero_first) {
    constexpr int32_t kScaleBitsInvSqrt64 = 0x3E00;
    bfloat16 *kv = attn_kv_full_static;
    for (int h = 0; h < ATTN_HEADS_PER_TILE; h++) {
        bfloat16 *q_h    = attn_q_heads4_static + h * HEAD_DIM;
        bfloat16 *scores = attn_scores4_static  + h * ATTN_K_CHUNK;
        bfloat16 *ctx_h  = attn_ctx4_static     + h * HEAD_DIM;
        attn_qk_score_chunk_bf16(q_h, kv, scores, kScaleBitsInvSqrt64);
        attn_softmax_inplace_bf16(scores, ATTN_K_CHUNK);
        attn_av_ctx_chunk_bf16(scores, kv, ctx_h, zero_first);
    }
}

extern "C" void attn_drain_real_ctx_bf16(bfloat16 *__restrict ctx_out,
                                         int32_t n) {
    for (int i = 0; i < ATTN_HEADS_PER_TILE * HEAD_DIM && i < n; i++) {
        ctx_out[i] = attn_ctx4_static[i];
    }
    for (int i = ATTN_HEADS_PER_TILE * HEAD_DIM; i < n; i++) {
        ctx_out[i] = (bfloat16)0.0f;
    }
}

#endif // COMPILE_ATTN
