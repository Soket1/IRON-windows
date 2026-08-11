// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// FlowKV decode attention kernel for AIE2+.
//
// Implements streaming decode attention with online softmax using a 2-tile
// pipeline per KV head group:
//
//   Score tile (CT0): Computes Q * K^T / sqrt(d) with online softmax tracking.
//     Maintains running max and denominator across chunks.
//     Outputs a packed buffer [F_c | C_c | l] to the value tile via on-chip
//     FIFO each chunk iteration.
//
//   Value tile (CT1): Accumulates weighted values with correction.
//     Reads the packed buffer from the score tile FIFO each chunk.
//     Saves the denominator from the last chunk in a static buffer so that
//     normalize can read it after all FIFO buffers are released.
//     Final normalization: O = Y / l.
//
// Both tiles share this single .o file. Each Worker calls a different subset
// of functions. Static buffers are per-tile (each tile gets its own copy).
//
// Packed inter-tile buffer layout (bf16):
//   [0 .. chunk_size*group_size - 1]                      : F_c scores
//   [chunk_size*group_size .. chunk_size*group_size + gs-1] : C_c correction
//   [chunk_size*group_size + gs .. chunk_size*group_size + 2*gs - 1] : l denom

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <type_traits>

// HEAD_DIM is a compile-time parameter set by the IRON op via -DHEAD_DIM=N.
// Supported values: 64, 128, 256. Must be a multiple of 32 (dot-product
// vector width). Default 64 keeps the legacy single-shape build working
// when no flag is passed (existing xclbins continue to function).
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif

static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256,
              "FlowKV: HEAD_DIM must be 64, 128, or 256");
static_assert(HEAD_DIM % 32 == 0, "FlowKV: HEAD_DIM must be multiple of 32");

// Precomputed 1/sqrt(HEAD_DIM) -- avoids float runtime call in the score
// hot loop, and lets the compiler fold the multiplication.
#if HEAD_DIM == 64
    static constexpr float HEAD_DIM_INV_SQRT = 0.125f;             // 1/8
#elif HEAD_DIM == 128
    static constexpr float HEAD_DIM_INV_SQRT = 0.0883883476f;      // 1/sqrt(128)
#elif HEAD_DIM == 256
    static constexpr float HEAD_DIM_INV_SQRT = 0.0625f;            // 1/16
#endif

// MAX_Q_HEADS bounds the per-tile static softmax state. The score/value tiles
// process num_q_heads query heads (one KV group) per dispatch; with num_cols=4
// over 32 heads that is 8 heads/tile, but the legacy single-shape build used 4.
// Default 4 keeps existing xclbins byte-identical; the IRON op passes
// -DMAX_Q_HEADS=N to size the buffers for its head count. Overflowing these
// statics (e.g. score_init(8) into a [4] array) silently corrupts adjacent
// memory and produces large garbage output — must match the design's attn_group.
#ifndef MAX_Q_HEADS
#error "FlowKV: MAX_Q_HEADS must be passed via -DMAX_Q_HEADS=N (sizes score/value statics; silent default corrupts memory when attn_group != 4)"
#endif
static_assert(MAX_Q_HEADS >= 1, "FlowKV: MAX_Q_HEADS must be >= 1");

// MAX_CHUNK bounds the per-head score scratch array. chunk_size passed at
// runtime must be <= MAX_CHUNK. Default 32 keeps legacy single-chunk xclbins
// byte-identical; the SEQ=256 f3best build passes -DMAX_CHUNK=256 to process
// the whole sequence in one "chunk" (single-chunk online softmax = standard
// softmax over all positions).
#ifndef MAX_CHUNK
#error "FlowKV: MAX_CHUNK must be passed via -DMAX_CHUNK=N (bounds score scratch; silent default overflows when chunk_size > 32)"
#endif
static_assert(MAX_CHUNK >= 1, "FlowKV: MAX_CHUNK must be >= 1");

// ---------------------------------------------------------------------------
// Score tile: static softmax state (only used by score tile Worker)
// ---------------------------------------------------------------------------
static float score_running_max[MAX_Q_HEADS] __attribute__((aligned(64)));
static float score_running_sum[MAX_Q_HEADS] __attribute__((aligned(64)));

// RoPE-rotated Q vectors (written by score_rope_q, read by score_chunk).
// Sized for HEAD_DIM at compile time so larger head dims don't overflow
// the static buffer.
static bfloat16 rotated_q[MAX_Q_HEADS * HEAD_DIM] __attribute__((aligned(64)));

// Actual sequence length (number of filled KV positions).
// Read from Q buffer element [num_q_heads*head_dim + head_dim] = angles[64].
// The host encodes this as bf16 before dispatch.
static int32_t g_actual_seq_len = 0;
int32_t g_score_chunk_counter = 0;

static inline int32_t bf16_to_int(const bfloat16 * buf, int idx) {
    uint16_t bits = *(const uint16_t *)&buf[idx];
    int exp = ((bits >> 7) & 0xFF) - 127;
    if (exp < 0) return 0;
    uint32_t mant = (bits & 0x7F) | 0x80;
    // The mantissa carries an implicit binary point after bit 7, so the value
    // is mant * 2^(exp-7). For exp < 7 that is a RIGHT shift; the old code
    // always shifted left, which is UB for every value below 128 (e.g. a
    // 47-token cache: exp=5 -> shift by -2).
    return (exp >= 7) ? (int)(mant << (exp - 7))
                      : (int)(mant >> (7 - exp));
}

// ---------------------------------------------------------------------------
// Value tile: accumulated output in f32 for precision (sized by HEAD_DIM).
// ---------------------------------------------------------------------------
static float value_accum[MAX_Q_HEADS * HEAD_DIM] __attribute__((aligned(64)));

// Saved denominator from the last chunk (written by accum, read by normalize)
static float saved_denom[MAX_Q_HEADS] __attribute__((aligned(64)));

extern "C" {

// ============================= Score Tile ====================================

// Initialize softmax state at the start of a new attention computation.
void flowkv_score_init_bf16(int32_t num_q_heads)
{
    for (int h = 0; h < num_q_heads; h++) {
        score_running_max[h] = -1e30f;
        score_running_sum[h] = 0.0f;
    }
}

// Copy Q heads into static buffer. The host already provides post-RoPE Q
// (the ggml ROPE node runs before FlowKV dispatch), so no rotation needed.
// Also reads actual_seq_len from the angles region of the Q buffer.
//
// Q buffer layout: [Q_heads (gs * hd) | angles (hd) | actual_seq_len (1)]
// angles region is unused for rotation but actual_seq_len is read from
// angles[head_dim] (bf16 encoded).
void flowkv_score_rope_q_bf16(const bfloat16 *__restrict q_in, int32_t num_q_heads, int32_t head_dim)
{
    const bfloat16 *angles = q_in + num_q_heads * head_dim;

    // Read actual_seq_len encoded as bf16 at angles[head_dim] (= angles[64]).
    // The host writes this before dispatch. If absent (0), fall back to full seq_len.
    g_actual_seq_len = bf16_to_int(angles, head_dim);
    if (g_actual_seq_len <= 0) g_actual_seq_len = 32767;  // fallback: process all

    // Reset chunk counter for this attention computation.
    *(volatile int32_t *)&g_actual_seq_len; // force re-read (compiler barrier)
    g_score_chunk_counter = 0;

    // Q is already post-RoPE — copy into the static rotated_q buffer.
    // No rotation applied. The angles region is skipped (not needed).
#ifdef FLOWKV_PRESCALE_Q
    aie::vector<float, 16> scale_vec = aie::broadcast<float, 16>(HEAD_DIM_INV_SQRT);
#endif
    for (int h = 0; h < num_q_heads; h++) {
        const bfloat16 *q_head = q_in + h * head_dim;
        bfloat16 *out_head = rotated_q + h * head_dim;

        for (int v = 0; v < head_dim; v += 16) {
            aie::vector<bfloat16, 16> q_vec = aie::load_v<16>(q_head + v);
#ifdef FLOWKV_PRESCALE_Q
            aie::accum<accfloat, 16> q_acc(q_vec);
            aie::vector<float, 16> q_f32 = q_acc.to_vector<float>();
            aie::vector<float, 16> scaled = aie::mul(q_f32, scale_vec);
            aie::accum<accfloat, 16> scaled_acc(scaled);
            aie::vector<bfloat16, 16> q_scaled = scaled_acc.to_vector<bfloat16>();
            aie::store_v(out_head + v, q_scaled);
#else
            aie::store_v(out_head + v, q_vec);
#endif
        }
    }
}

// Compute attention scores for one K chunk and update online softmax state.
// Writes results into a single packed inter-tile buffer.
// Uses rotated Q from the static buffer (populated by flowkv_score_rope_q_bf16).
//
// q_in:      (num_q_heads, head_dim)  -- query vectors (unused, reads rotated_q)
// k_chunk:   (chunk_size, head_dim)   -- K cache chunk
// packed_out: packed buffer for inter-tile FIFO:
//   [0 .. cs*gs-1]: F_c scores in (chunk_size, num_q_heads) layout
//   [cs*gs .. cs*gs+gs-1]: C_c correction factors
//   [cs*gs+gs .. cs*gs+2*gs-1]: l denominators
void flowkv_score_chunk_bf16(const bfloat16 *__restrict q_in,
                             const bfloat16 *__restrict k_chunk,
                             bfloat16 *__restrict packed_out,
                             int32_t num_q_heads,
                             int32_t head_dim,
                             int32_t chunk_size)
{
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // 1/sqrt(HEAD_DIM) precomputed at compile time. The compiler folds the
    // subsequent multiplication into the dot-product reduction.
    const float inv_sqrt_d = HEAD_DIM_INV_SQRT;

    const int32_t scores_size = chunk_size * num_q_heads;
    bfloat16 *scores_out = packed_out;
    bfloat16 *correction_out = packed_out + scores_size;
    bfloat16 *denom_out = packed_out + scores_size + num_q_heads;

    // Use actual_seq_len to skip empty KV positions that dilute softmax.
    // g_actual_seq_len is set by flowkv_score_rope_q_bf16 from the Q buffer.
    int32_t chunk_idx = g_score_chunk_counter++;
    int32_t pos_start = chunk_idx * chunk_size;
    int32_t actual_seq = g_actual_seq_len;

    // If this entire chunk is beyond actual_seq_len, send zero scores
    // (identity for online softmax: scores=0 → exp(0-m)=~0 when m>>0,
    //  correction=1, denominator unchanged).
#ifdef FLOWKV_SCORE_STUB
    // PROBE (latency-only, WRONG answer): always take the empty-chunk path, so
    // the whole dot-product + softmax body is skipped while every output buffer
    // is still written and every lock still released -- the fabric cannot tell
    // the difference, so this prices the score stage without risking a stall.
    if (true) {
#else
    if (pos_start >= actual_seq) {
#endif
        for (int i = 0; i < scores_size + num_q_heads * 2; i++)
            packed_out[i] = static_cast<bfloat16>(0.0f);
        // Set denominator to 1.0 to avoid division by zero in normalize.
        for (int h = 0; h < num_q_heads; h++)
            denom_out[h] = static_cast<bfloat16>(score_running_sum[h]);
        for (int h = 0; h < num_q_heads; h++)
            correction_out[h] = static_cast<bfloat16>(1.0f);
        event1();
        return;
    }

    // Clamp effective chunk_size for the last partial chunk.
    int32_t eff_chunk = chunk_size;
    if (pos_start + chunk_size > actual_seq)
        eff_chunk = actual_seq - pos_start;

    for (int h = 0; h < num_q_heads; h++) {
        const bfloat16 *q_head = rotated_q + h * head_dim;
        float m_old = score_running_max[h];
        float l_old = score_running_sum[h];

        // Phase 1: Compute dot products and find chunk-local max
        // Store scores as bf16 to avoid float array auto-vectorization issues
        bfloat16 scores_bf16[MAX_CHUNK]; // chunk_size max = MAX_CHUNK
        bfloat16 m_chunk_bf16 = static_cast<bfloat16>(-1e30f);

        for (int pos = 0; pos < eff_chunk; pos++) {
            const bfloat16 *k_pos = k_chunk + pos * head_dim;

            // Vectorized dot product over HEAD_DIM elements in chunks of 32.
            // The loop is compile-time bounded (HEAD_DIM / 32) so the AIE
            // compiler can fully unroll without runtime loop overhead.
            // For HEAD_DIM=64: 2 chunks; 128: 4 chunks; 256: 8 chunks.
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
            constexpr int n_chunks = HEAD_DIM / 32;
            #pragma clang loop unroll(full)
            for (int c = 0; c < n_chunks; c++) {
                auto qv = aie::load_v<32>(q_head + c * 32);
                auto kv = aie::load_v<32>(k_pos  + c * 32);
                acc = aie::mac(acc, qv, kv);
            }

#ifdef FLOWKV_SCORE_NOREDUCE
            // PROBE (latency-only, WRONG answer): skip the horizontal reduce_add
            // to size its cost. Same loads/macs, no 32→1 reduction.
  #ifdef FLOWKV_PRESCALE_Q
            bfloat16 score = static_cast<bfloat16>(acc.to_vector<float>()[0]);
  #else
            bfloat16 score = static_cast<bfloat16>(acc.to_vector<float>()[0] * inv_sqrt_d);
  #endif
#else
  #ifdef FLOWKV_PRESCALE_Q
            bfloat16 score = static_cast<bfloat16>(aie::reduce_add(acc.to_vector<float>()));
  #else
            bfloat16 score = static_cast<bfloat16>(aie::reduce_add(acc.to_vector<float>()) * inv_sqrt_d);
  #endif
#endif

            scores_bf16[pos] = score;
            if (static_cast<float>(score) > static_cast<float>(m_chunk_bf16)) {
                m_chunk_bf16 = score;
            }
        }

        // Phase 2: Online softmax update using bf16 vector ops
        float m_chunk_f = static_cast<float>(m_chunk_bf16);
        float m_new = (m_chunk_f > m_old) ? m_chunk_f : m_old;
        bfloat16 m_new_bf16 = static_cast<bfloat16>(m_new);

        // C_c = exp2((m_old - m_new) * log2e) via vector exp2
        bfloat16 corr_scaled = static_cast<bfloat16>((m_old - m_new) * 1.4453125f);
        aie::vector<bfloat16, 16> corr_in_vec = aie::broadcast<bfloat16, 16>(corr_scaled);
        aie::accum<accfloat, 16> corr_acc(corr_in_vec);
        aie::vector<bfloat16, 16> corr_exp = aie::exp2<bfloat16>(corr_acc.to_vector<float>());
        float c_correction = static_cast<float>(corr_exp[0]);

        bfloat16 l_new_bf16 = static_cast<bfloat16>(c_correction * l_old);

        // Compute exp2 for each score position.
#ifdef FLOWKV_VEC_EXP
        aie::vector<float, 16> m_new_vec = aie::broadcast<float, 16>(m_new);
        aie::vector<float, 16> log2e_vec = aie::broadcast<float, 16>(1.4453125f);

        for (int pos = 0; pos < eff_chunk; pos += 16) {
            int rem = eff_chunk - pos;
            int n = (rem < 16) ? rem : 16;

            alignas(32) bfloat16 tmp_in[16];
            for (int i = 0; i < 16; i++) {
                tmp_in[i] = (i < n) ? scores_bf16[pos + i] : m_new_bf16;
            }

            aie::vector<bfloat16, 16> s_vec = aie::load_v<16>(tmp_in);
            aie::accum<accfloat, 16> s_acc(s_vec);
            aie::vector<float, 16> s_f32 = s_acc.to_vector<float>();
            aie::vector<float, 16> diff_f32 = aie::mul(aie::sub(s_f32, m_new_vec), log2e_vec);
            aie::vector<bfloat16, 16> exp_result = aie::exp2<bfloat16>(diff_f32);

            alignas(32) bfloat16 tmp_out[16];
            aie::store_v(tmp_out, exp_result);

            for (int i = 0; i < n; i++) {
                bfloat16 f_bf16 = tmp_out[i];
                l_new_bf16 = static_cast<bfloat16>(static_cast<float>(l_new_bf16) + static_cast<float>(f_bf16));
                scores_out[(pos + i) * num_q_heads + h] = f_bf16;
            }
        }
#else
        // Scalar form broadcasts one score per vector exp2. Kept as the default
        // fallback for probes because some AIE2P codegen versions are sensitive
        // to local vector spill buffers in this loop.
        for (int pos = 0; pos < eff_chunk; pos++) {
#ifdef FLOWKV_NOEXP
            // PROBE (latency-only, WRONG): skip per-position exp2.
            bfloat16 f_bf16 = scores_bf16[pos];
#else
            bfloat16 diff = static_cast<bfloat16>((static_cast<float>(scores_bf16[pos]) - m_new) * 1.4453125f);
            aie::vector<bfloat16, 16> diff_vec = aie::broadcast<bfloat16, 16>(diff);
            aie::accum<accfloat, 16> diff_acc(diff_vec);
            aie::vector<bfloat16, 16> exp_result = aie::exp2<bfloat16>(diff_acc.to_vector<float>());
            bfloat16 f_bf16 = exp_result[0];
#endif
            l_new_bf16 = static_cast<bfloat16>(static_cast<float>(l_new_bf16) + static_cast<float>(f_bf16));
            scores_out[pos * num_q_heads + h] = f_bf16;
        }
#endif
        // Zero remaining scores for unused positions in this chunk.
        for (int pos = eff_chunk; pos < chunk_size; pos++) {
            scores_out[pos * num_q_heads + h] = static_cast<bfloat16>(0.0f);
        }

        // Update running state
        score_running_max[h] = m_new;
        score_running_sum[h] = static_cast<float>(l_new_bf16);

        // Write correction and denominator to packed buffer
        correction_out[h] = static_cast<bfloat16>(c_correction);
        denom_out[h] = l_new_bf16;
    }

    event1();
}

// ============================= Value Tile ====================================

// Initialize the value accumulator.
void flowkv_value_init_bf16(int32_t num_q_heads, int32_t head_dim)
{
    int total = num_q_heads * head_dim;
    for (int i = 0; i < total; i++) {
        value_accum[i] = 0.0f;
    }
    for (int h = 0; h < num_q_heads; h++) {
        saved_denom[h] = 0.0f;
    }
}

// Accumulate weighted values for one chunk.
// Reads scores and correction from the packed inter-tile buffer.
// Saves the denominator into a static buffer for later normalization.
//
// packed_in: packed buffer from score tile FIFO
//   [0..cs*gs-1]: F_c scores
//   [cs*gs..cs*gs+gs-1]: C_c correction
//   [cs*gs+gs..cs*gs+2*gs-1]: l denom
// v_chunk: (chunk_size, head_dim) -- V cache chunk from DDR
void flowkv_value_accum_bf16(const bfloat16 *__restrict packed_in,
                             const bfloat16 *__restrict v_chunk,
                             int32_t num_q_heads,
                             int32_t head_dim,
                             int32_t chunk_size)
{
#ifdef FLOWKV_VALUE_STUB
    // PROBE (latency-only, WRONG answer): skip the value accumulation. Its output
    // buffer is still emitted by flowkv_value_normalize_bf16 and every lock still
    // cycles, so the fabric is unchanged and the delta prices this stage alone.
    // Companion to FLOWKV_SCORE_STUB -- score was measured at ~0, value never was.
    (void)packed_in; (void)v_chunk; (void)num_q_heads; (void)head_dim; (void)chunk_size;
    return;
#endif
    event0();
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const int32_t scores_size = chunk_size * num_q_heads;
    const bfloat16 *scores_in = packed_in;
    const bfloat16 *correction_in = packed_in + scores_size;
    const bfloat16 *denom_in = packed_in + scores_size + num_q_heads;

    for (int h = 0; h < num_q_heads; h++) {
        float correction = static_cast<float>(correction_in[h]);
        float *y_head = value_accum + h * head_dim;

        // Save denominator for final normalization
        saved_denom[h] = static_cast<float>(denom_in[h]);

        aie::vector<float, 16> corr_vec = aie::broadcast<float, 16>(correction);

#ifdef FLOWKV_VALUE_LEGACY
        // Memory-resident accumulator: y_head is reloaded and restored on EVERY
        // position. Kept behind a flag as the A/B reference -- the register form
        // below is bit-identical (float32 store/reload is exact), so any numeric
        // difference between the two would mean a codegen bug, not a math change.
        for (int d = 0; d < head_dim; d += 16) {
            aie::vector<float, 16> y_vec = aie::load_v<16>(y_head + d);
            y_vec = aie::mul(y_vec, corr_vec);
            aie::store_v(y_head + d, y_vec);
        }
        for (int pos = 0; pos < chunk_size; pos++) {
            float f = static_cast<float>(scores_in[pos * num_q_heads + h]);
            const bfloat16 *v_pos = v_chunk + pos * head_dim;
            aie::vector<float, 16> f_vec = aie::broadcast<float, 16>(f);

#ifndef FLOWKV_NOVALUE
            for (int d = 0; d < head_dim; d += 16) {
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_head + d);
                aie::vector<bfloat16, 16> v_vec = aie::load_v<16>(v_pos + d);
                aie::accum<accfloat, 16> v_acc(v_vec);
                aie::vector<float, 16> v_f32 = v_acc.to_vector<float>();
                aie::vector<float, 16> fv = aie::mul(f_vec, v_f32);
                y_vec = aie::add(y_vec, fv);
                aie::store_v(y_head + d, y_vec);
            }
#endif
        }
#else
        // value_accum must survive BETWEEN chunk calls, but within one call there
        // is no reason to spill it once per position: 64 dims = 4 vectors, so the
        // running output stays in registers for the whole position loop and is
        // written back once. Same arithmetic, same order, same rounding.
        // Blocked by 64 dims so register pressure is fixed for HEAD_DIM 64/128/256.
        int d0 = 0;
        for (; d0 + 64 <= head_dim; d0 += 64) {
#ifdef FLOWKV_VALUE_AMAC
            // Register-resident accumulator path: y stays in aie::accum (accfloat),
            // V is loaded as bf16 vector (no accum round-trip → no to_vector<float>),
            // product via mul(bf16, bf16)→accum, accumulation via add(accum,
            // accum).  Saves 1 VLIW slot per vector per position vs. the legacy
            // float32 path (4 slots → 3: load + mul + add instead of load→accum +
            // to_vector + mul + add).
            aie::accum<accfloat, 16> y0_acc = aie::mul(aie::load_v<16>(y_head + d0 +  0), corr_vec);
            aie::accum<accfloat, 16> y1_acc = aie::mul(aie::load_v<16>(y_head + d0 + 16), corr_vec);
            aie::accum<accfloat, 16> y2_acc = aie::mul(aie::load_v<16>(y_head + d0 + 32), corr_vec);
            aie::accum<accfloat, 16> y3_acc = aie::mul(aie::load_v<16>(y_head + d0 + 48), corr_vec);
  #ifndef FLOWKV_NOVALUE
            const bfloat16 *v_col = v_chunk + d0;
            AIE_PREPARE_FOR_PIPELINING
            for (int pos = 0; pos < chunk_size; pos++) {
                float f = static_cast<float>(scores_in[pos * num_q_heads + h]);
                bfloat16 f_bf16 = static_cast<bfloat16>(f);
                aie::vector<bfloat16, 16> f_vec = aie::broadcast<bfloat16, 16>(f_bf16);
                const bfloat16 *v_pos = v_col + pos * head_dim;

                    aie::vector<bfloat16, 16> v0 = aie::load_v<16>(v_pos +  0);
                aie::vector<bfloat16, 16> v1 = aie::load_v<16>(v_pos + 16);
                aie::vector<bfloat16, 16> v2 = aie::load_v<16>(v_pos + 32);
                aie::vector<bfloat16, 16> v3 = aie::load_v<16>(v_pos + 48);
                // bf16 * bf16 → accum, then accum + accum (no to_vector<float>)
                y0_acc = aie::add(y0_acc, aie::mul(v0, f_vec));
                y1_acc = aie::add(y1_acc, aie::mul(v1, f_vec));
                y2_acc = aie::add(y2_acc, aie::mul(v2, f_vec));
                y3_acc = aie::add(y3_acc, aie::mul(v3, f_vec));
            }
  #endif
            aie::store_v(y_head + d0 +  0, y0_acc.to_vector<float>());
            aie::store_v(y_head + d0 + 16, y1_acc.to_vector<float>());
            aie::store_v(y_head + d0 + 32, y2_acc.to_vector<float>());
            aie::store_v(y_head + d0 + 48, y3_acc.to_vector<float>());
#else
            aie::vector<float, 16> y0 = aie::mul(aie::load_v<16>(y_head + d0 +  0), corr_vec);
            aie::vector<float, 16> y1 = aie::mul(aie::load_v<16>(y_head + d0 + 16), corr_vec);
            aie::vector<float, 16> y2 = aie::mul(aie::load_v<16>(y_head + d0 + 32), corr_vec);
            aie::vector<float, 16> y3 = aie::mul(aie::load_v<16>(y_head + d0 + 48), corr_vec);

#ifndef FLOWKV_NOVALUE
            const bfloat16 *v_col = v_chunk + d0;
            AIE_PREPARE_FOR_PIPELINING
            for (int pos = 0; pos < chunk_size; pos++) {
                float f = static_cast<float>(scores_in[pos * num_q_heads + h]);
                aie::vector<float, 16> f_vec = aie::broadcast<float, 16>(f);
                const bfloat16 *v_pos = v_col + pos * head_dim;

                aie::accum<accfloat, 16> a0(aie::load_v<16>(v_pos +  0));
                aie::accum<accfloat, 16> a1(aie::load_v<16>(v_pos + 16));
                aie::accum<accfloat, 16> a2(aie::load_v<16>(v_pos + 32));
                aie::accum<accfloat, 16> a3(aie::load_v<16>(v_pos + 48));
                // aie::mul yields an accum; materialise it into a vector exactly
                // as the legacy path does, so the arithmetic stays identical.
                aie::vector<float, 16> p0 = aie::mul(f_vec, a0.to_vector<float>());
                aie::vector<float, 16> p1 = aie::mul(f_vec, a1.to_vector<float>());
                aie::vector<float, 16> p2 = aie::mul(f_vec, a2.to_vector<float>());
                aie::vector<float, 16> p3 = aie::mul(f_vec, a3.to_vector<float>());
                y0 = aie::add(y0, p0);
                y1 = aie::add(y1, p1);
                y2 = aie::add(y2, p2);
                y3 = aie::add(y3, p3);
            }
#endif
            aie::store_v(y_head + d0 +  0, y0);
            aie::store_v(y_head + d0 + 16, y1);
            aie::store_v(y_head + d0 + 32, y2);
            aie::store_v(y_head + d0 + 48, y3);
#endif
        }
        // Tail, never taken for HEAD_DIM 64/128/256. Present so a head_dim that
        // is not a multiple of 64 degrades to the slow form instead of writing
        // past the 64-block -- silent overrun of value_accum is the same failure
        // class the MAX_Q_HEADS note above warns about.
        for (; d0 < head_dim; d0 += 16) {
            aie::vector<float, 16> y = aie::mul(aie::load_v<16>(y_head + d0), corr_vec);
#ifndef FLOWKV_NOVALUE
            for (int pos = 0; pos < chunk_size; pos++) {
                float f = static_cast<float>(scores_in[pos * num_q_heads + h]);
                aie::accum<accfloat, 16> a(aie::load_v<16>(v_chunk + pos * head_dim + d0));
                aie::vector<float, 16> p = aie::mul(aie::broadcast<float, 16>(f), a.to_vector<float>());
                y = aie::add(y, p);
            }
#endif
            aie::store_v(y_head + d0, y);
        }
#endif
    }

    event1();
}

// Normalize and produce final output: O = Y / l.
// Reads the denominator from saved_denom (set by the last accum call).
//
// output: (num_q_heads, head_dim) -- final attention output in bf16
void flowkv_value_normalize_bf16(bfloat16 *__restrict output, int32_t num_q_heads, int32_t head_dim)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    for (int h = 0; h < num_q_heads; h++) {
        float inv_l = aie::inv(saved_denom[h]);
        aie::vector<float, 16> inv_l_vec = aie::broadcast<float, 16>(inv_l);
        float *y_head = value_accum + h * head_dim;
        bfloat16 *o_head = output + h * head_dim;

        for (int d = 0; d < head_dim; d += 16) {
            aie::vector<float, 16> y_vec = aie::load_v<16>(y_head + d);
            aie::vector<float, 16> scaled = aie::mul(y_vec, inv_l_vec);
            aie::accum<accfloat, 16> y_acc(scaled);
            aie::vector<bfloat16, 16> out_vec = y_acc.to_vector<bfloat16>();
            aie::store_v(o_head + d, out_vec);
        }
    }

}

} // extern "C"
