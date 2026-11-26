// Correctness-first resident QBit KV attention probe.
//
// K and V consist of one affine QBit block per (token, KV head) vector.
// The retained p4/p5 code planes stay compressed in the Metal buffers and
// values are reconstructed only while computing attention. No Float32 KV
// materialization buffer is used.

#include <metal_stdlib>
using namespace metal;

constant ushort QQA_SG = 32;
constant uint QQA_HD = 256;

constant uint QQA_P4_POSITIVE_CENTROID_BITS[8] = {
    0x3da18fb8u, 0x3e747262u, 0x3ecf6ceau, 0x3f158a3au,
    0x3f491a06u, 0x3f8408fbu, 0x3fb45dcfu, 0x4007469au,
};

constant uint QQA_P5_POSITIVE_CENTROID_BITS[16] = {
    0x3d214c9eu, 0x3df27670u, 0x3e4aeb9cu, 0x3e8efa3fu,
    0x3eb97bb4u, 0x3ee558d1u, 0x3f0982c6u, 0x3f218b69u,
    0x3f3b2b35u, 0x3f56f4e8u, 0x3f75d8c1u, 0x3f8cda65u,
    0x3fa379d0u, 0x3fc3f223u, 0x3ff8a44fu, 0x4028a4feu,
};

inline uint qqa_read_u32_le(device const uchar* src, uint offset) {
    return ((uint)src[offset]) |
           (((uint)src[offset + 1]) << 8) |
           (((uint)src[offset + 2]) << 16) |
           (((uint)src[offset + 3]) << 24);
}

inline float qqa_centroid(device const uchar* src,
                          uint row_base,
                          uint within,
                          uint plane_bytes,
                          uint precision) {
    const uint byte_offset = plane_bytes - 1 - within / 8;
    const uchar bit_mask = (uchar)(1u << (within & 7u));
    uint raw_code = 0;
    for (uint plane = 0; plane < precision; ++plane) {
        const uint plane_offset = row_base + 8 + plane * plane_bytes;
        if ((src[plane_offset + byte_offset] & bit_mask) != 0) {
            raw_code |= 1u << (7u - plane);
        }
    }

    const uint prefix = raw_code >> (8u - precision);
    if (precision == 4) {
        return prefix < 8
            ? as_type<float>(QQA_P4_POSITIVE_CENTROID_BITS[prefix])
            : -as_type<float>(QQA_P4_POSITIVE_CENTROID_BITS[15u - prefix]);
    }
    return prefix < 16
        ? as_type<float>(QQA_P5_POSITIVE_CENTROID_BITS[prefix])
        : -as_type<float>(QQA_P5_POSITIVE_CENTROID_BITS[31u - prefix]);
}

// Qwen3.8 27B specialization: one threadgroup owns one KV head and its six
// query heads. A 16-token K tile is decoded once, reused by all six heads, and
// then the same threadgroup memory is reused for the V tile. This removes the
// sixfold QBit decode duplication in the correctness-first generic kernel.
constant uint QQA_GQA6_HEADS = 6;
constant uint QQA_GQA6_TILE = 16;
constant uint QQA_GQA6_THREADS = QQA_GQA6_HEADS * QQA_SG;

kernel void qwen35_qbit_attn_decode_gqa6(
    device const float* Q [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const uchar* k_cache [[buffer(2)]],
    device const uchar* v_cache [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& cache_len [[buffer(5)]],
    constant uint& n_head [[buffer(6)]],
    constant uint& n_head_kv [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& heads_per_group [[buffer(9)]],
    constant uint& precision [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint kv_h [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort local_h [[simdgroup_index_in_threadgroup]],
    ushort thread_index [[thread_index_in_threadgroup]]) {
    if (kv_h >= n_head_kv || local_h >= QQA_GQA6_HEADS ||
        heads_per_group != QQA_GQA6_HEADS || head_dim != QQA_HD ||
        (precision != 4 && precision != 5)) {
        return;
    }

    const uint h = kv_h * QQA_GQA6_HEADS + local_h;
    if (h >= n_head) {
        return;
    }

    const uint plane_bytes = head_dim / 8;
    const uint row_stride = 8 + precision * plane_bytes;
    threadgroup float kv_tile[QQA_GQA6_TILE * QQA_HD];
    threadgroup float probabilities[QQA_GQA6_HEADS][QQA_SG];

    float m = -1e30f;
    float l = 0.0f;
    float o[QQA_HD / QQA_SG];
    for (uint i = 0; i < QQA_HD / QQA_SG; ++i) {
        o[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < cache_len; tile_start += QQA_GQA6_TILE) {
        const uint tile_len = min(tile_start + QQA_GQA6_TILE, cache_len) - tile_start;
        const uint tile_values = tile_len * head_dim;

        for (uint index = thread_index; index < tile_values; index += QQA_GQA6_THREADS) {
            const uint position_in_tile = index / head_dim;
            const uint d = index - position_in_tile * head_dim;
            const uint row = (tile_start + position_in_tile) * n_head_kv + kv_h;
            const uint row_base = row * row_stride;
            const float mean = as_type<float>(qqa_read_u32_le(k_cache, row_base));
            const float sigma = as_type<float>(qqa_read_u32_le(k_cache, row_base + 4));
            kv_tile[index] = mean + sigma * qqa_centroid(
                k_cache, row_base, d, plane_bytes, precision);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = -1e30f;
        if (lane < tile_len) {
            threadgroup const float4* key =
                (threadgroup const float4*)(kv_tile + lane * head_dim);
            device const float4* query =
                (device const float4*)(Q + h * head_dim);
            float dot = 0.0f;
            for (uint d4 = 0; d4 < head_dim / 4; ++d4) {
                const float4 k4 = key[d4];
                const float4 q4 = query[d4];
                dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
            score = dot * scale;
        }

        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float correction = exp(m - m_new);
        const float probability = lane < tile_len ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(probability);
        probabilities[local_h][lane] = probability;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint index = thread_index; index < tile_values; index += QQA_GQA6_THREADS) {
            const uint position_in_tile = index / head_dim;
            const uint d = index - position_in_tile * head_dim;
            const uint row = (tile_start + position_in_tile) * n_head_kv + kv_h;
            const uint row_base = row * row_stride;
            const float mean = as_type<float>(qqa_read_u32_le(v_cache, row_base));
            const float sigma = as_type<float>(qqa_read_u32_le(v_cache, row_base + 4));
            kv_tile[index] = mean + sigma * qqa_centroid(
                v_cache, row_base, d, plane_bytes, precision);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dl = 0; dl < QQA_HD / QQA_SG; ++dl) {
            const uint d = lane + dl * QQA_SG;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                acc += probabilities[local_h][s] * kv_tile[s * head_dim + d];
            }
            o[dl] = o[dl] * correction + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        m = m_new;
    }

    const float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    for (uint dl = 0; dl < QQA_HD / QQA_SG; ++dl) {
        const uint d = lane + dl * QQA_SG;
        const float g = gate[h * head_dim + d];
        const float sigmoid_gate = 1.0f / (1.0f + exp(-g));
        out[h * head_dim + d] = o[dl] * inv_l * sigmoid_gate;
    }
}
