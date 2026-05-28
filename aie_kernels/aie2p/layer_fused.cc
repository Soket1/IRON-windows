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
// Static L1 buffer variants for col 0's merged worker (Path A from spec).
//
// Col 0's compute tile runs pre-RMS first, then QKV GEMVs. To stay within
// the AIE2P 2-input-DMA cap, col 0 cannot consume bq_mem.cons() (would be
// a 3rd input channel alongside rms_in and Aqkv). Instead, pre-RMS writes
// its normed output to BOTH a static L1 buffer (used by col 0's own QKV
// phases) and to a fifo-routed output buffer (broadcast via MemTile to
// cols 1-7).
// ────────────────────────────────────────────────────────────────────────────

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
// instead of a fifo-routed bf16 pointer. Same kernel math as the
// fifo-input variant — only the activation source differs.
extern "C" void layer_fused_qkv_gemv_static_bf16(
        uint32_t m, uint32_t row_offset,
        const uint8_t *a, bfloat16 *c) {
    _qkv_gemv<32, GROUP_SIZE, EMBED_DIM>(
        m, a + row_offset * (EMBED_DIM / 2 + EMBED_DIM / GROUP_SIZE * 2),
        normed_static, c);
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

    for (int j = 0; j < ATTN_K_CHUNK; j++) {
        const bfloat16 *k_row = k_chunk + j * HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            dot += (float)q_head[d] * (float)k_row[d];
        }
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
extern "C" void attn_av_ctx_chunk_bf16(
        const bfloat16 *scores_chunk,  // [K_CHUNK]
        const bfloat16 *v_chunk,       // [K_CHUNK × HEAD_DIM]
        bfloat16       *ctx_head,      // [HEAD_DIM] accumulator
        int32_t         zero_first      // 1 = clear ctx_head before; 0 = accumulate
    ) {
    if (zero_first) {
        for (int d = 0; d < HEAD_DIM; d++) ctx_head[d] = (bfloat16)0.0f;
    }
    for (int j = 0; j < ATTN_K_CHUNK; j++) {
        float w = (float)scores_chunk[j];
        const bfloat16 *v_row = v_chunk + j * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            ctx_head[d] = (bfloat16)((float)ctx_head[d] + w * (float)v_row[d]);
        }
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
