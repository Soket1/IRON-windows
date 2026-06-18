// SPDX-License-Identifier: Apache-2.0
//
// attn_concat2 — concatenate two equal-length bf16 halves into one buffer.
// out[0:half] = a[0:half]; out[half:2*half] = b[0:half].
// Used by the attention-output relay tile to reassemble the two 4-head join
// halves into a full E-vector for the downstream O-projection GEMV (a single
// tile cannot consume both halves AND the weight stream within the 2-S2MM cap,
// so a relay tile merges the halves into one fifo first).

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void attn_concat2_bf16(const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b,
                       bfloat16 *__restrict out, int32_t half) {
    constexpr int VEC = 16;
    int chunks = half / VEC;
    for (int i = 0; i < chunks; i++)
        ::aie::store_v(out + i * VEC, ::aie::load_v<VEC>(a + i * VEC));
    for (int i = chunks * VEC; i < half; i++)
        out[i] = a[i];
    for (int i = 0; i < chunks; i++)
        ::aie::store_v(out + half + i * VEC, ::aie::load_v<VEC>(b + i * VEC));
    for (int i = chunks * VEC; i < half; i++)
        out[half + i] = b[i];
}

}  // extern "C"
