// Minimal echo kernels for DMA path testing.
// echo_copy_bf16:  memcpy in → out
// echo_concat_bf16: concat a+b → out (out[0..N-1]=a, out[N..2N-1]=b)
// echo_score_bf16/echo_value_bf16: split two-tile K/inter/V path
// echo_score_qk_bf16/echo_value_qkv_bf16: FlowKV-like Q/K/V marker path
// echo_value_chunk_bf16: multi-chunk FlowKV-like marker path

#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void echo_copy_bf16(const bfloat16 *__restrict inp,
                    bfloat16 *__restrict outp,
                    int32_t N)
{
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(inp + i);
        aie::store_v(outp + i, v);
    }
}

void echo_concat_bf16(const bfloat16 *__restrict a,
                      const bfloat16 *__restrict b,
                      bfloat16 *__restrict out,
                      int32_t N)
{
    // out[0..N-1] = a
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(a + i);
        aie::store_v(out + i, v);
    }
    // out[N..2N-1] = b
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(b + i);
        aie::store_v(out + N + i, v);
    }
}

void echo_score_bf16(const bfloat16 *__restrict k,
                     bfloat16 *__restrict inter,
                     int32_t N)
{
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(k + i);
        aie::store_v(inter + i, v);
    }
}

void echo_value_bf16(const bfloat16 *__restrict inter,
                     const bfloat16 *__restrict v,
                     bfloat16 *__restrict out,
                     int32_t N)
{
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> x = aie::load_v<16>(inter + i);
        aie::store_v(out + i, x);
    }
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> y = aie::load_v<16>(v + i);
        aie::store_v(out + N + i, y);
    }
}

void echo_score_qk_bf16(const bfloat16 *__restrict q,
                        const bfloat16 *__restrict k,
                        bfloat16 *__restrict inter,
                        int32_t N)
{
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> kv = aie::load_v<16>(k + i);
        aie::store_v(inter + i, kv);
    }
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> qv = aie::load_v<16>(q + i);
        aie::store_v(inter + N + i, qv);
    }
}

void echo_value_qkv_bf16(const bfloat16 *__restrict inter,
                         const bfloat16 *__restrict v,
                         bfloat16 *__restrict out,
                         int32_t N)
{
    for (int i = 0; i < 2 * N; i += 16) {
        aie::vector<bfloat16, 16> x = aie::load_v<16>(inter + i);
        aie::store_v(out + i, x);
    }
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> y = aie::load_v<16>(v + i);
        aie::store_v(out + 2 * N + i, y);
    }
}

void echo_value_chunk_bf16(const bfloat16 *__restrict inter,
                           const bfloat16 *__restrict v,
                           bfloat16 *__restrict out,
                           int32_t chunk_idx,
                           int32_t N)
{
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> k = aie::load_v<16>(inter + i);
        aie::store_v(out + chunk_idx * N + i, k);
    }
    if (chunk_idx == 0) {
        for (int i = 0; i < N; i += 16) {
            aie::vector<bfloat16, 16> q = aie::load_v<16>(inter + N + i);
            aie::store_v(out + 2 * N + i, q);
        }
    }
    for (int i = 0; i < N; i += 16) {
        aie::vector<bfloat16, 16> y = aie::load_v<16>(v + i);
        aie::store_v(out + (3 + chunk_idx) * N + i, y);
    }
}

void echo_value_chunk0_bf16(const bfloat16 *__restrict inter,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out,
                            int32_t N)
{
    echo_value_chunk_bf16(inter, v, out, 0, N);
}

void echo_value_chunk1_bf16(const bfloat16 *__restrict inter,
                            const bfloat16 *__restrict v,
                            bfloat16 *__restrict out,
                            int32_t N)
{
    echo_value_chunk_bf16(inter, v, out, 1, N);
}

} // extern "C"
