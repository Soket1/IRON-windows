// Minimal echo kernels for DMA path testing.
// echo_copy_bf16:  memcpy in → out
// echo_concat_bf16: concat a+b → out (out[0..N-1]=a, out[N..2N-1]=b)
// echo_score_bf16/echo_value_bf16: split two-tile K/inter/V path
// echo_score_qk_bf16/echo_value_qkv_bf16: FlowKV-like Q/K/V marker path
// echo_value_chunk_bf16: multi-chunk FlowKV-like marker path
// echo_pack_inter_v6_bf16/echo_value_v6_bf16: FlowKV packed inter layout path
// echo_v7_score_*: FlowKV score math packed inter diagnostic

#define NOCPP
#include <aie_api/aie.hpp>
#include <stdint.h>

static float echo_v7_score_running_max[4] __attribute__((aligned(64)));
static float echo_v7_score_running_sum[4] __attribute__((aligned(64)));
static bfloat16 echo_v7_rotated_q[4 * 64] __attribute__((aligned(64)));
static int32_t echo_v7_actual_seq_len = 0;
static int32_t echo_v7_chunk_counter = 0;

static inline int32_t echo_v7_bf16_to_int(const bfloat16 *buf, int idx) {
    uint16_t bits = *(const uint16_t *)&buf[idx];
    int exp = ((bits >> 7) & 0xFF) - 127;
    if (exp < 0) return 0;
    uint32_t mant = (bits & 0x7F) | 0x80;
    return (int)(mant << (exp - 7));
}

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

void echo_v7_score_init_bf16(int32_t num_q_heads)
{
    for (int h = 0; h < num_q_heads; h++) {
        echo_v7_score_running_max[h] = -1e30f;
        echo_v7_score_running_sum[h] = 0.0f;
    }
}

void echo_v7_score_rope_q_bf16(const bfloat16 *__restrict q_in,
                                int32_t num_q_heads,
                                int32_t head_dim)
{
    const bfloat16 *angles = q_in + num_q_heads * head_dim;
    echo_v7_actual_seq_len = echo_v7_bf16_to_int(angles, head_dim);
    if (echo_v7_actual_seq_len <= 0) echo_v7_actual_seq_len = 32767;
    *(volatile int32_t *)&echo_v7_actual_seq_len;
    echo_v7_chunk_counter = 0;

    for (int h = 0; h < num_q_heads; h++) {
        const bfloat16 *q_head = q_in + h * head_dim;
        bfloat16 *out_head = echo_v7_rotated_q + h * head_dim;
        for (int v = 0; v < head_dim; v += 16) {
            aie::vector<bfloat16, 16> q_vec = aie::load_v<16>(q_head + v);
            aie::store_v(out_head + v, q_vec);
        }
    }
}

void echo_v7_score_chunk_bf16(const bfloat16 *__restrict q_in,
                               const bfloat16 *__restrict k_chunk,
                               bfloat16 *__restrict packed_out,
                               int32_t num_q_heads,
                               int32_t head_dim,
                               int32_t chunk_size)
{
    (void)q_in;
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const float inv_sqrt_d = 0.125f;
    const int32_t scores_size = chunk_size * num_q_heads;
    bfloat16 *scores_out = packed_out;
    bfloat16 *correction_out = packed_out + scores_size;
    bfloat16 *denom_out = packed_out + scores_size + num_q_heads;

    int32_t chunk_idx = echo_v7_chunk_counter++;
    int32_t pos_start = chunk_idx * chunk_size;
    int32_t actual_seq = echo_v7_actual_seq_len;

    if (pos_start >= actual_seq) {
        for (int i = 0; i < scores_size + num_q_heads * 2; i++)
            packed_out[i] = static_cast<bfloat16>(0.0f);
        for (int h = 0; h < num_q_heads; h++)
            denom_out[h] = static_cast<bfloat16>(echo_v7_score_running_sum[h]);
        for (int h = 0; h < num_q_heads; h++)
            correction_out[h] = static_cast<bfloat16>(1.0f);
        return;
    }

    int32_t eff_chunk = chunk_size;
    if (pos_start + chunk_size > actual_seq)
        eff_chunk = actual_seq - pos_start;

    for (int h = 0; h < num_q_heads; h++) {
        const bfloat16 *q_head = echo_v7_rotated_q + h * head_dim;
        float m_old = echo_v7_score_running_max[h];
        float l_old = echo_v7_score_running_sum[h];
        bfloat16 scores_bf16[32];
        bfloat16 m_chunk_bf16 = static_cast<bfloat16>(-1e30f);

        for (int pos = 0; pos < eff_chunk; pos++) {
            const bfloat16 *k_pos = k_chunk + pos * head_dim;
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

        float m_chunk_f = static_cast<float>(m_chunk_bf16);
        float m_new = (m_chunk_f > m_old) ? m_chunk_f : m_old;
        bfloat16 corr_scaled = static_cast<bfloat16>((m_old - m_new) * 1.4453125f);
        aie::vector<bfloat16, 16> corr_in_vec = aie::broadcast<bfloat16, 16>(corr_scaled);
        aie::accum<accfloat, 16> corr_acc(corr_in_vec);
        aie::vector<bfloat16, 16> corr_exp = aie::exp2<bfloat16>(corr_acc.to_vector<float>());
        float c_correction = static_cast<float>(corr_exp[0]);
        bfloat16 l_new_bf16 = static_cast<bfloat16>(c_correction * l_old);

        for (int pos = 0; pos < eff_chunk; pos++) {
            bfloat16 diff = static_cast<bfloat16>((static_cast<float>(scores_bf16[pos]) - m_new) * 1.4453125f);
            aie::vector<bfloat16, 16> diff_vec = aie::broadcast<bfloat16, 16>(diff);
            aie::accum<accfloat, 16> diff_acc(diff_vec);
            aie::vector<bfloat16, 16> exp_result = aie::exp2<bfloat16>(diff_acc.to_vector<float>());
            bfloat16 f_bf16 = exp_result[0];
            l_new_bf16 = static_cast<bfloat16>(static_cast<float>(l_new_bf16) + static_cast<float>(f_bf16));
            scores_out[pos * num_q_heads + h] = f_bf16;
        }
        for (int pos = eff_chunk; pos < chunk_size; pos++) {
            scores_out[pos * num_q_heads + h] = static_cast<bfloat16>(0.0f);
        }

        echo_v7_score_running_max[h] = m_new;
        echo_v7_score_running_sum[h] = static_cast<float>(l_new_bf16);
        correction_out[h] = static_cast<bfloat16>(c_correction);
        denom_out[h] = l_new_bf16;
    }
}

void echo_v7_copy_pack_bf16(const bfloat16 *__restrict packed_in,
                             bfloat16 *__restrict out,
                             int32_t chunk_idx,
                             int32_t packed_size)
{
    bfloat16 *chunk_out = out + chunk_idx * packed_size;
    for (int i = 0; i < packed_size; i += 8) {
        aie::vector<bfloat16, 8> x = aie::load_v<8>(packed_in + i);
        aie::store_v(chunk_out + i, x);
    }
}

void echo_v7_copy_pack_chunk0_bf16(const bfloat16 *__restrict packed_in,
                                    bfloat16 *__restrict out,
                                    int32_t packed_size)
{
    echo_v7_copy_pack_bf16(packed_in, out, 0, packed_size);
}

void echo_v7_copy_pack_chunk1_bf16(const bfloat16 *__restrict packed_in,
                                    bfloat16 *__restrict out,
                                    int32_t packed_size)
{
    echo_v7_copy_pack_bf16(packed_in, out, 1, packed_size);
}

} // extern "C"
