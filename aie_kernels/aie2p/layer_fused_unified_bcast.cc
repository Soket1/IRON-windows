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

    // The hand-asm kernel sets crrnd=conv_even internally; no need to
    // re-set it here since no other code runs on this tile between calls.

    // Scratch buffer for hand-asm output (32 bf16 values, AIE-vector aligned).
    alignas(64) bfloat16 scratch[N];

    // ── Step 1: Per-chunk GEMV via hand-asm ────────────────────────────────
    // weights: MLIR positions the weight buffer for this chunk.
    // activation + chunk*KC: this chunk's 256-element activation slice.
    layer_fused_gemv_bcast_kc256_bf16(weights, activation + chunk * KC, scratch);

#ifdef QBCAST_NO_CROSSCHUNK_DI
    // #188 Ш22.5 diagnostic: flush THIS chunk's fresh scratch directly to output
    // with NO cross-chunk partial add/store at all. If the hand-asm Q corruption
    // (dense noise / NaN) disappears -> the cross-chunk static partial is the
    // shared root (same as the C++ _qkv_bcast_chunk). Q magnitude ~1/nchunk of
    // correct (only this chunk's KC cols contribute), but FINITE if clean.
    aie::accum<accfloat, N> facc0;
    facc0.from_vector(aie::load_v<N>(scratch));
    aie::store_v(output + block * N, facc0.template to_vector<bfloat16>());
    return;
#endif

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
    //
    // #206 fix: a 32-float partial is 128 bytes = 1024 bits, DOUBLE the AIE2
    // native 512-bit vector width. aiecc lowers the oversized load/store into
    // two ops and misplaces the middle 64 bytes — the same defect already fixed
    // in layer_fused_f3best.cc for the three per-phase wrappers (see the
    // "#188 fix: the 32-float (1024-bit) partial store/load" comment there).
    // This unified wrapper is the one live path that never received that fix,
    // and uni_partial is allocated immediately after the gate buffer, so the
    // misplaced half lands on gate[1008..1024) — exactly the observed window.
    // Split every partial access into two native 16-float (64-byte) halves.
    aie::vector<float, 16> clo = cur.template extract<16>(0);
    aie::vector<float, 16> chi = cur.template extract<16>(1);
    if (chunk != 0) {
        clo = aie::add(clo, aie::load_v<16>(partial));
        chi = aie::add(chi, aie::load_v<16>(partial + 16));
    }

    // ── Step 4/5: Last chunk → flush to output; else → store float32 partial
    if (chunk == nchunk - 1) {
        // Last chunk: float32 → accfloat → bf16 (single rounding), write to output.
        // The bf16 output halves are 32 bytes each — natively sized either way.
        aie::accum<accfloat, 16> flo, fhi;
        flo.from_vector(clo);
        fhi.from_vector(chi);
        aie::store_v(output + block * N,      flo.template to_vector<bfloat16>());
        aie::store_v(output + block * N + 16, fhi.template to_vector<bfloat16>());
    } else {
        // Store float32 partial for next chunk's accumulation.
        aie::store_v(partial,      clo);
        aie::store_v(partial + 16, chi);
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
extern "C" void generic_bcast_gemv_bf16_kv(uint32_t j, const uint8_t *w, const bfloat16 *a, float *p, uint32_t n, bfloat16 *o) {
    generic_bcast_gemv_bf16(j, w, a, p, n, o);
}
extern "C" void generic_bcast_gemv_bf16_d(uint32_t j, const uint8_t *w, const bfloat16 *a, float *p, uint32_t n, bfloat16 *o) {
    generic_bcast_gemv_bf16(j, w, a, p, n, o);
}
