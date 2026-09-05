#include <metal_stdlib>
using namespace metal;

// Exact-shape Qwen prefill attention kernel. Host admission is deliberately
// narrow: d256, F16 K/V and GQA4/GQA6 on the profiled device. Prefixes and
// partial tiles are explicit host experiments only. Full key tiles use MMA;
// the final partial tile uses bounded SIMD attention without cache padding.
// The caller must provide base_pos+n_tokens complete K/V rows.

constant short QWEN35_FLASH_DK = 256;
constant short QWEN35_FLASH_DV = 256;
constant short QWEN35_FLASH_Q = 8;
constant short QWEN35_FLASH_C = 64;
constant short QWEN35_FLASH_NSG = 4;
constant short QWEN35_FLASH_NQ = QWEN35_FLASH_Q / QWEN35_FLASH_NSG;
constant short QWEN35_FLASH_PV = 256;
constant short QWEN35_FLASH_SH = 2 * QWEN35_FLASH_C;
constant short QWEN35_FLASH_NW = 32;

kernel void qwen35_attn_flash_d256(
    device const float* Q                  [[buffer(0)]],
    device const float* gate               [[buffer(1)]],
    device const half*  k_cache            [[buffer(2)]],
    device const half*  v_cache            [[buffer(3)]],
    device       float* out                [[buffer(4)]],
    constant     uint&  base_pos           [[buffer(5)]],
    constant     uint&  n_tokens           [[buffer(6)]],
    constant     uint&  n_head             [[buffer(7)]],
    constant     uint&  n_head_kv          [[buffer(8)]],
    constant     uint&  head_dim           [[buffer(9)]],
    constant     uint&  heads_per_group    [[buffer(10)]],
    constant     float& scale              [[buffer(11)]],
    uint3   tgpig                          [[threadgroup_position_in_grid]],
    ushort  tiisg                          [[thread_index_in_simdgroup]],
    ushort  sgitg                          [[simdgroup_index_in_threadgroup]],
    threadgroup half* shmem                [[threadgroup(0)]])
{
    const uint iq1 = tgpig.x * QWEN35_FLASH_Q;
    const uint h = tgpig.y;
    if (h >= n_head || n_head_kv == 0 || n_head % n_head_kv != 0 ||
        heads_per_group != n_head / n_head_kv ||
        (heads_per_group != 4 && heads_per_group != 6) ||
        head_dim != QWEN35_FLASH_DK || n_head_kv != 4 ||
        n_tokens == 0 || n_tokens > 2048 || base_pos > 8192 - n_tokens ||
        iq1 >= n_tokens) {
        return;
    }

    const uint kv_h = h / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint kv_dim = n_head_kv * QWEN35_FLASH_DK;
    const uint total_keys = base_pos + n_tokens;
    const uint full_key_end = total_keys / QWEN35_FLASH_C * QWEN35_FLASH_C;
    constexpr short DK4 = QWEN35_FLASH_DK / 4;
    constexpr short DK8 = QWEN35_FLASH_DK / 8;
    constexpr short DV4 = QWEN35_FLASH_DV / 4;
    constexpr short PV4 = QWEN35_FLASH_PV / 4;
    constexpr short PV8 = QWEN35_FLASH_PV / 8;
    constexpr short T = QWEN35_FLASH_DK + 2 * QWEN35_FLASH_PV;

    threadgroup half* sq = shmem;
    threadgroup half4* sq4 = (threadgroup half4*)sq;
    threadgroup float* so = (threadgroup float*)(sq + QWEN35_FLASH_Q * QWEN35_FLASH_DK);
    threadgroup float4* so4 = (threadgroup float4*)so;
    threadgroup float* ss = (threadgroup float*)(shmem + QWEN35_FLASH_Q * T);
    threadgroup float2* ss2 = (threadgroup float2*)ss;

    // Each simdgroup loads two interleaved query rows. The MMA phase then
    // reuses the complete eight-row tile while simdgroups split key columns.
    for (short jj = 0; jj < QWEN35_FLASH_NQ; ++jj) {
        const short j = jj * QWEN35_FLASH_NSG + sgitg;
        for (short i = tiisg; i < DK4; i += QWEN35_FLASH_NW) {
            if (iq1 + j < n_tokens) {
                device const float4* q4 = (device const float4*)(
                    Q + ((iq1 + j) * n_head + h) * QWEN35_FLASH_DK);
                sq4[j * DK4 + i] = half4(q4[i]);
            } else {
                sq4[j * DK4 + i] = half4(0.0h);
            }
        }
        for (short i = tiisg; i < DV4; i += QWEN35_FLASH_NW) {
            so4[j * PV4 + i] = float4(0.0f);
        }
        for (short i = tiisg; i < QWEN35_FLASH_SH; i += QWEN35_FLASH_NW) {
            ss[j * QWEN35_FLASH_SH + i] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[QWEN35_FLASH_NQ] = {0.0f, 0.0f};
    float M[QWEN35_FLASH_NQ] = {-FLT_MAX / 2, -FLT_MAX / 2};

    for (uint ic = 0; ic < full_key_end; ic += QWEN35_FLASH_C) {
        // Q*K^T. Four simdgroups split the 64 key columns while sharing the
        // same eight queries. K rows are strided by the GQA cache row width.
        device const half* pk = k_cache + ic * kv_dim + kv_h * QWEN35_FLASH_DK;
        pk += sgitg * (8 * kv_dim);
        threadgroup float* ps = ss + sgitg * 8;

        constexpr short NC = (QWEN35_FLASH_C / 8) / QWEN35_FLASH_NSG;
        for (short cc = 0; cc < NC; ++cc) {
            simdgroup_matrix<float, 8, 8> mqk(0.0f);
            #pragma unroll(16)
            for (short i = 0; i < DK8 / 2; ++i) {
                simdgroup_matrix<half, 8, 8> mq0;
                simdgroup_matrix<half, 8, 8> mq1;
                simdgroup_matrix<half, 8, 8> mk0;
                simdgroup_matrix<half, 8, 8> mk1;

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(mq0, sq + 16 * i, QWEN35_FLASH_DK);
                simdgroup_load(mq1, sq + 16 * i + 8, QWEN35_FLASH_DK);
                simdgroup_load(mk0, pk + 16 * i, kv_dim, 0, true);
                simdgroup_load(mk1, pk + 16 * i + 8, kv_dim, 0, true);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(mqk, mq0, mk0, mqk);
                simdgroup_multiply_accumulate(mqk, mq1, mk1, mqk);
            }
            simdgroup_store(mqk, ps, QWEN35_FLASH_SH, 0, false);
            pk += 8 * QWEN35_FLASH_NSG * kv_dim;
            ps += 8 * QWEN35_FLASH_NSG;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online causal softmax. Each simdgroup owns two query rows and each
        // lane owns two key columns of the 64-column score tile.
        for (short jj = 0; jj < QWEN35_FLASH_NQ; ++jj) {
            const short j = jj * QWEN35_FLASH_NSG + sgitg;
            const float old_m = M[jj];
            float2 scores = ss2[j * (QWEN35_FLASH_SH / 2) + tiisg] * scale;
            const uint key0 = ic + 2 * tiisg;
            const uint query_pos = base_pos + iq1 + j;
            if (key0 > query_pos) scores[0] = -FLT_MAX / 2;
            if (key0 + 1 > query_pos) scores[1] = -FLT_MAX / 2;

            M[jj] = simd_max(max(M[jj], max(scores[0], scores[1])));
            const float correction = exp(old_m - M[jj]);
            const float2 probs = exp(scores - M[jj]);
            S[jj] = S[jj] * correction + simd_sum(probs[0] + probs[1]);
            ss2[j * (QWEN35_FLASH_SH / 2) + tiisg] = probs;

            for (short i = tiisg; i < DV4; i += QWEN35_FLASH_NW) {
                so4[j * PV4 + i] *= correction;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P*V. Each simdgroup owns one quarter of the output columns and
        // consumes all 64 key probabilities in four 16-row pairs.
        constexpr short NO = PV8 / QWEN35_FLASH_NSG;
        simdgroup_matrix<float, 8, 8> accum[NO];
        threadgroup float* sot = so + 8 * sgitg;
        for (short ii = 0; ii < NO; ++ii) {
            simdgroup_load(accum[ii], sot, QWEN35_FLASH_PV, 0, false);
            sot += 8 * QWEN35_FLASH_NSG;
        }

        device const half* pv = v_cache + ic * kv_dim + kv_h * QWEN35_FLASH_DV + 8 * sgitg;
        constexpr short VC = (QWEN35_FLASH_C / 8) / 2;
        for (short cc = 0; cc < VC; ++cc) {
            simdgroup_matrix<float, 8, 8> vs0;
            simdgroup_matrix<float, 8, 8> vs1;
            simdgroup_load(vs0, ss + 16 * cc, QWEN35_FLASH_SH, 0, false);
            simdgroup_load(vs1, ss + 16 * cc + 8, QWEN35_FLASH_SH, 0, false);

            for (short ii = 0; ii < NO / 2; ++ii) {
                simdgroup_matrix<half, 8, 8> mv0;
                simdgroup_matrix<half, 8, 8> mv1;
                simdgroup_matrix<half, 8, 8> mv2;
                simdgroup_matrix<half, 8, 8> mv3;
                const short col = 16 * ii * QWEN35_FLASH_NSG;
                simdgroup_load(mv0, pv + col, kv_dim, 0, false);
                simdgroup_load(mv1, pv + 8 * QWEN35_FLASH_NSG + col, kv_dim, 0, false);
                simdgroup_load(mv2, pv + col + 8 * kv_dim, kv_dim, 0, false);
                simdgroup_load(mv3, pv + 8 * QWEN35_FLASH_NSG + col + 8 * kv_dim, kv_dim, 0, false);
                simdgroup_multiply_accumulate(accum[2 * ii], vs0, mv0, accum[2 * ii]);
                simdgroup_multiply_accumulate(accum[2 * ii + 1], vs0, mv1, accum[2 * ii + 1]);
                simdgroup_multiply_accumulate(accum[2 * ii], vs1, mv2, accum[2 * ii]);
                simdgroup_multiply_accumulate(accum[2 * ii + 1], vs1, mv3, accum[2 * ii + 1]);
            }
            pv += 16 * kv_dim;
        }

        sot = so + 8 * sgitg;
        for (short ii = 0; ii < NO; ++ii) {
            simdgroup_store(accum[ii], sot, QWEN35_FLASH_PV, 0, false);
            sot += 8 * QWEN35_FLASH_NSG;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short jj = 0; jj < QWEN35_FLASH_NQ; ++jj) {
        const short j = jj * QWEN35_FLASH_NSG + sgitg;
        // No barriers follow: skipping a whole SIMD-owned inactive row is
        // safe, unlike returning before the shared MMA phases above.
        if (iq1 + j >= n_tokens) continue;
        const uint row = (iq1 + j) * n_head + h;
        if (full_key_end < total_keys) {
            // At most 63 keys. Each lane owns eight output dimensions. Keep
            // the same H16 query representation as MMA, and never read beyond
            // the visible causal row (including a sub-64 total context).
            float acc[QWEN35_FLASH_DV / QWEN35_FLASH_NW];
            float query[QWEN35_FLASH_DK / QWEN35_FLASH_NW];
            for (short i = 0; i < QWEN35_FLASH_DV / QWEN35_FLASH_NW; ++i) {
                const short d = tiisg + i * QWEN35_FLASH_NW;
                acc[i] = so[j * QWEN35_FLASH_PV + d];
                query[i] = float(sq[j * QWEN35_FLASH_DK + d]);
            }
            const uint causal_end = base_pos + iq1 + j + 1;
            for (uint key = full_key_end; key < causal_end; ++key) {
                const uint offset = key * kv_dim + kv_h * QWEN35_FLASH_DK;
                float dot = 0.0f;
                for (short i = 0; i < QWEN35_FLASH_DK / QWEN35_FLASH_NW; ++i) {
                    dot += query[i] * float(k_cache[offset + tiisg + i * QWEN35_FLASH_NW]);
                }
                const float score = simd_sum(dot) * scale;
                const float next_m = max(M[jj], score);
                const float correction = exp(M[jj] - next_m);
                const float probability = exp(score - next_m);
                S[jj] = S[jj] * correction + probability;
                M[jj] = next_m;
                for (short i = 0; i < QWEN35_FLASH_DV / QWEN35_FLASH_NW; ++i) {
                    acc[i] = acc[i] * correction + probability *
                        float(v_cache[offset + tiisg + i * QWEN35_FLASH_NW]);
                }
            }
            const float inv_s = S[jj] > 0.0f ? 1.0f / S[jj] : 0.0f;
            for (short i = 0; i < QWEN35_FLASH_DV / QWEN35_FLASH_NW; ++i) {
                const uint d = tiisg + i * QWEN35_FLASH_NW;
                out[row * QWEN35_FLASH_DV + d] = acc[i] * inv_s /
                    (1.0f + exp(-gate[row * QWEN35_FLASH_DV + d]));
            }
            continue;
        }
        const float inv_s = S[jj] > 0.0f ? 1.0f / S[jj] : 0.0f;
        device const float4* gate4 = (device const float4*)(gate + row * QWEN35_FLASH_DV);
        device float4* out4 = (device float4*)(out + row * QWEN35_FLASH_DV);
        for (short i = tiisg; i < DV4; i += QWEN35_FLASH_NW) {
            const float4 g = gate4[i];
            const float4 sigmoid_g = 1.0f / (1.0f + exp(-g));
            out4[i] = so4[j * PV4 + i] * inv_s * sigmoid_g;
        }
    }
}
