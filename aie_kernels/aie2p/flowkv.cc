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

// ---------------------------------------------------------------------------
// Score tile: static softmax state (only used by score tile Worker)
// ---------------------------------------------------------------------------
static float score_running_max[4] __attribute__((aligned(64)));
static float score_running_sum[4] __attribute__((aligned(64)));

// RoPE-rotated Q vectors (written by score_rope_q, read by score_chunk)
static bfloat16 rotated_q[4 * 64] __attribute__((aligned(64)));

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
    return (int)(mant << (exp - 7));
}

// ---------------------------------------------------------------------------
// Value tile: accumulated output in f32 for precision
// ---------------------------------------------------------------------------
static float value_accum[4 * 64] __attribute__((aligned(64)));

// Saved denominator from the last chunk (written by accum, read by normalize)
static float saved_denom[4] __attribute__((aligned(64)));

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

    // Q is already post-RoPE — just copy into the static rotated_q buffer.
    // No rotation applied. The angles region is skipped (not needed).
    for (int h = 0; h < num_q_heads; h++) {
        const bfloat16 *q_head = q_in + h * head_dim;
        bfloat16 *out_head = rotated_q + h * head_dim;

        for (int v = 0; v < head_dim; v += 16) {
            aie::vector<bfloat16, 16> q_vec = aie::load_v<16>(q_head + v);
            aie::store_v(out_head + v, q_vec);
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

    const float inv_sqrt_d = 0.125f; // 1/sqrt(64) = 1/8

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
    if (pos_start >= actual_seq) {
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
        bfloat16 scores_bf16[32]; // chunk_size max = 32
        bfloat16 m_chunk_bf16 = static_cast<bfloat16>(-1e30f);

        for (int pos = 0; pos < eff_chunk; pos++) {
            const bfloat16 *k_pos = k_chunk + pos * head_dim;

            // Vectorized dot product: head_dim=64 using single accum
            aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();

            auto q_vec0 = aie::load_v<32>(q_head);
            auto k_vec0 = aie::load_v<32>(k_pos);
            acc = aie::mac(acc, q_vec0, k_vec0);

            auto q_vec1 = aie::load_v<32>(q_head + 32);
            auto k_vec1 = aie::load_v<32>(k_pos + 32);
            acc = aie::mac(acc, q_vec1, k_vec1);

            bfloat16 score = static_cast<bfloat16>(aie::reduce_add(acc.to_vector<float>()) * inv_sqrt_d);

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

        // Compute exp2 for each score position — one at a time, no float arrays
        for (int pos = 0; pos < eff_chunk; pos++) {
            bfloat16 diff = static_cast<bfloat16>((static_cast<float>(scores_bf16[pos]) - m_new) * 1.4453125f);
            aie::vector<bfloat16, 16> diff_vec = aie::broadcast<bfloat16, 16>(diff);
            aie::accum<accfloat, 16> diff_acc(diff_vec);
            aie::vector<bfloat16, 16> exp_result = aie::exp2<bfloat16>(diff_acc.to_vector<float>());
            bfloat16 f_bf16 = exp_result[0];
            l_new_bf16 = static_cast<bfloat16>(static_cast<float>(l_new_bf16) + static_cast<float>(f_bf16));
            scores_out[pos * num_q_heads + h] = f_bf16;
        }
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

        // Apply correction to accumulated output: Y = C_c * Y_old
        aie::vector<float, 16> corr_vec = aie::broadcast<float, 16>(correction);
        for (int d = 0; d < head_dim; d += 16) {
            aie::vector<float, 16> y_vec = aie::load_v<16>(y_head + d);
            y_vec = aie::mul(y_vec, corr_vec);
            aie::store_v(y_head + d, y_vec);
        }

        // Accumulate: Y += sum_pos( F_c[pos, h] * V[pos, :] )
        for (int pos = 0; pos < chunk_size; pos++) {
            float f = static_cast<float>(scores_in[pos * num_q_heads + h]);
            const bfloat16 *v_pos = v_chunk + pos * head_dim;
            aie::vector<float, 16> f_vec = aie::broadcast<float, 16>(f);

            for (int d = 0; d < head_dim; d += 16) {
                aie::vector<float, 16> y_vec = aie::load_v<16>(y_head + d);
                aie::vector<bfloat16, 16> v_vec = aie::load_v<16>(v_pos + d);
                aie::accum<accfloat, 16> v_acc(v_vec);
                aie::vector<float, 16> v_f32 = v_acc.to_vector<float>();
                aie::vector<float, 16> fv = aie::mul(f_vec, v_f32);
                y_vec = aie::add(y_vec, fv);
                aie::store_v(y_head + d, y_vec);
            }
        }
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
