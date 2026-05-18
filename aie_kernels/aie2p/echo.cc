// Minimal echo kernels for DMA path testing.
// echo_copy_bf16:  memcpy in → out
// echo_concat_bf16: concat a+b → out (out[0..N-1]=a, out[N..2N-1]=b)
// echo_score_bf16/echo_value_bf16: split two-tile K/inter/V path
// echo_score_qk_bf16/echo_value_qkv_bf16: FlowKV-like Q/K/V marker path
// echo_value_chunk_bf16: multi-chunk FlowKV-like marker path
// echo_pack_inter_v6_bf16/echo_value_v6_bf16: FlowKV packed inter layout path

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

void echo_pack_inter_v6_bf16(const bfloat16 *__restrict k,
                              bfloat16 *__restrict packed_out,
                              int32_t chunk_size,
                              int32_t group_size,
                              int32_t head_dim)
{
    int scores_size = chunk_size * group_size;
    bfloat16 *scores_out = packed_out;
    bfloat16 *correction_out = packed_out + scores_size;
    bfloat16 *denom_out = packed_out + scores_size + group_size;

    for (int i = 0; i < scores_size; i += 16) {
        aie::vector<bfloat16, 16> v = aie::load_v<16>(k + i);
        aie::store_v(scores_out + i, v);
    }
    for (int h = 0; h < group_size; ++h) {
        correction_out[h] = static_cast<bfloat16>(1.0f);
        denom_out[h] = static_cast<bfloat16>(1.0f);
    }
}

void echo_value_v6_bf16(const bfloat16 *__restrict packed_in,
                        const bfloat16 *__restrict v,
                        bfloat16 *__restrict out,
                        int32_t chunk_idx,
                        int32_t chunk_size,
                        int32_t group_size,
                        int32_t head_dim)
{
    int packed_size = chunk_size * group_size + 2 * group_size;
    int v_size = chunk_size * head_dim;
    int out_block_size = packed_size + v_size;
    bfloat16 *chunk_out = out + chunk_idx * out_block_size;

    for (int i = 0; i < packed_size; i += 8) {
        aie::vector<bfloat16, 8> x = aie::load_v<8>(packed_in + i);
        aie::store_v(chunk_out + i, x);
    }
    for (int i = 0; i < v_size; i += 8) {
        aie::vector<bfloat16, 8> y = aie::load_v<8>(v + i);
        aie::store_v(chunk_out + packed_size + i, y);
    }
}

void echo_value_v6_chunk0_bf16(const bfloat16 *__restrict packed_in,
                               const bfloat16 *__restrict v,
                               bfloat16 *__restrict out,
                               int32_t chunk_size,
                               int32_t group_size,
                               int32_t head_dim)
{
    echo_value_v6_bf16(packed_in, v, out, 0, chunk_size, group_size, head_dim);
}

void echo_value_v6_chunk1_bf16(const bfloat16 *__restrict packed_in,
                               const bfloat16 *__restrict v,
                               bfloat16 *__restrict out,
                               int32_t chunk_size,
                               int32_t group_size,
                               int32_t head_dim)
{
    echo_value_v6_bf16(packed_in, v, out, 1, chunk_size, group_size, head_dim);
}

} // extern "C"
