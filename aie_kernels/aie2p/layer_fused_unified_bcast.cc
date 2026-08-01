// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unified bcast GEMV C++ wrapper. Replaces all four per-phase wrappers
// (layer_fused_gate_up_bcast_bf16, layer_fused_down_bcast_bf16,
//  layer_fused_qkv_bcast_bf16, layer_fused_oproj_bcast_bf16)
// with a single generic function.
//
// The hand-asm kernel `layer_fused_gemv_bcast_kc256_bf16` processes one
// KC=256 activation slice against packed int4 weights, producing 32 bf16
// partials. This wrapper handles cross-chunk accumulation in float32.
//
// Caller responsibilities:
//   - j_phase: tile index within this phase (encodes chunk and block)
//   - weights: 4608-byte packed weight tile (int4 nibbles + bf16 scales)
//   - activation: pointer to activation data (x_bundle/attn_out/ffn_in/silu_buf)
//   - partial: cross-chunk float accumulator (32 floats), caller-managed
//   - nchunk: 8 for K=2048 phases (Q/O/gate/up), 4 for K=1024 (down)
//   - output: destination buffer (Q/O/P/gate_buf/up_buf) — caller positions it

#define NOCPP
#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Hand-asm kernel declaration ────────────────────────────────────────────
// Signature: (weights: packed int4 + scales, activation: KC bf16, output: 32 bf16)
// KC = 256. Both base (_kc256.s) and RR (_kc256_rr.s) variants export the same
// symbol — the build system selects at link time.
extern "C" void layer_fused_gemv_bcast_kc256_bf16(
    const uint8_t *__restrict w,
    const bfloat16 *__restrict x,
    bfloat16 *__restrict out);

// ════════════════════════════════════════════════════════════════════════════
// Unified bcast GEMV wrapper
// ════════════════════════════════════════════════════════════════════════════
//
// Replaces:
//   layer_fused_gate_up_bcast_bf16(j, _, a, b, phase)
//     → generic_bcast_gemv_bf16(j, a, b, lf_bc_partial, 8, lbuf), lbuf = lf_left/right
//   layer_fused_down_bcast_bf16(j, _, a, c_out)
//     → generic_bcast_gemv_bf16(j, a, lf_silu_buf, lf_down_bc_partial, 4, c_out)
//   layer_fused_qkv_bcast_bf16(j, _, a, b, c_out)
//     → generic_bcast_gemv_bf16(j, a, b, lf_qkv_bc_partial, 8, c_out)
//   layer_fused_oproj_bcast_bf16(j, _, a, b, c_out)
//     → generic_bcast_gemv_bf16(j, a, b, lf_qkv_bc_partial, 8, c_out)
//
// j_phase encodes (block, chunk):
//   block = j_phase / nchunk   — output row block (0 .. M_OUTPUT/32 - 1)
//   chunk = j_phase % nchunk   — activation chunk (0 .. nchunk-1)
//
// Cross-chunk accumulation chain (bit-identical to existing per-phase wrappers):
//   1. Hand-asm kernel → 32 bf16 partials (scratch buffer)
//   2. bf16 → accfloat → float32 vector (single rounding step)
//   3. If chunk > 0: float32 add with partial[] (accumulate in float32, NOT accfloat)
//   4. If last chunk: float32 → accfloat → bf16, write to output[block*32]
//   5. Else: store float32 back to partial[] for next chunk

extern "C"
void generic_bcast_gemv_bf16(
    uint32_t j_phase,
    const uint8_t *weights,
    const bfloat16 *activation,
    float *partial,
    uint32_t nchunk,
    bfloat16 *output)
{
    constexpr uint32_t N  = 32;    // output rows per call
    constexpr uint32_t KC = 256;   // columns per chunk (hand-asm hardcoded)

    const uint32_t chunk = j_phase % nchunk;
    const uint32_t block = j_phase / nchunk;

    // The hand-asm kernel sets crrnd internally. Set it here too so the
    // wrapper's own float32 accumulation (aie::add, aie::store_v,
    // accum<->float conversions) uses consistent rounding.
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Scratch buffer for hand-asm output (32 bf16 values, AIE-vector aligned).
    alignas(64) bfloat16 scratch[N];

    // ── Step 1: Per-chunk GEMV via hand-asm ────────────────────────────────
    // weights: MLIR positions the weight buffer for this chunk.
    // activation + chunk*KC: this chunk's 256-element activation slice.
    layer_fused_gemv_bcast_kc256_bf16(weights, activation + chunk * KC, scratch);

    // ── Step 2: bf16 → accfloat → float32 ─────────────────────────────────
    // Match the existing conversion chain exactly:
    //   load_v<32>(bf16*) → from_vector() → accum<accfloat,32>
    //   → to_vector<float>() → vector<float,32>
    // This is a single rounding (bf16→float32 via accfloat intermediate).
    aie::accum<accfloat, N> tmp;
    tmp.from_vector(aie::load_v<N>(scratch));
    aie::vector<float, N> cur = tmp.template to_vector<float>();

    // ── Step 3: Cross-chunk accumulation in float32 ────────────────────────
    // Existing wrappers accumulate in float32, NOT accfloat. This avoids
    // double-rounding: accfloat→float32 (step 2), then float32+float32.
    if (chunk != 0)
        cur = aie::add(cur, aie::load_v<N>(partial));

    // ── Step 4/5: Last chunk → flush to output; else → store float32 partial
    if (chunk == nchunk - 1) {
        // Last chunk: float32 → accfloat → bf16 (single rounding), write to output.
        aie::accum<accfloat, N> facc;
        facc.from_vector(cur);
        aie::store_v(output + block * N, facc.template to_vector<bfloat16>());
    } else {
        // Store float32 partial for next chunk's accumulation.
        aie::store_v(partial, cur);
    }
}

// Thin aliases for MLIR type dispatch. MLIR requires exact type matches, so the
// emitter declares 4 type-specific names. The C++ kernel body is identical for
// all callers — the alias just forwards to the real implementation.
extern "C" void generic_bcast_gemv_bf16_q(uint32_t j, const uint8_t *w, const bfloat16 *a, float *p, uint32_t n, bfloat16 *o) {
    generic_bcast_gemv_bf16(j, w, a, p, n, o);
}
extern "C" void generic_bcast_gemv_bf16_o(uint32_t j, const uint8_t *w, const bfloat16 *a, float *p, uint32_t n, bfloat16 *o) {
    generic_bcast_gemv_bf16(j, w, a, p, n, o);
}
extern "C" void generic_bcast_gemv_bf16_g(uint32_t j, const uint8_t *w, const bfloat16 *a, float *p, uint32_t n, bfloat16 *o) {
    generic_bcast_gemv_bf16(j, w, a, p, n, o);
}
extern "C" void generic_bcast_gemv_bf16_d(uint32_t j, const uint8_t *w, const bfloat16 *a, float *p, uint32_t n, bfloat16 *o) {
    generic_bcast_gemv_bf16(j, w, a, p, n, o);
}
