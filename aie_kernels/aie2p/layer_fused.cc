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

}  // extern "C"
