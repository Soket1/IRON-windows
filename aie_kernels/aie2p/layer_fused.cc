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
