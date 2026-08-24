// Qwen3.8 GQA6 attention over adaptive row-aligned QBit KV.
//
// Every 256-value row has a dense p4 base and an 8-byte metadata entry.
// Metadata selects either the base, a fifth QBit plane, a BF16 replacement,
// or an exact F32 replacement in a canonical sidecar. Values are reconstructed
// only into the bounded threadgroup tile used by attention; no F32 KV cache is
// materialized.

#include <metal_stdlib>
using namespace metal;

constant ushort QQA_ADAPTIVE_SG = 32;
constant uint QQA_ADAPTIVE_HD = 256;
constant uint QQA_ADAPTIVE_PLANE_BYTES = QQA_ADAPTIVE_HD / 8;
constant uint QQA_ADAPTIVE_P4_STRIDE = 8 + 4 * QQA_ADAPTIVE_PLANE_BYTES;
constant uint QQA_ADAPTIVE_META_STRIDE = 8;

constant uint QQA_ADAPTIVE_P4 = 0;
constant uint QQA_ADAPTIVE_P5 = 1;
constant uint QQA_ADAPTIVE_BF16 = 2;
constant uint QQA_ADAPTIVE_F32 = 3;

constant uint QQA_ADAPTIVE_P4_CENTROID_BITS[8] = {
    0x3da18fb8u, 0x3e747262u, 0x3ecf6ceau, 0x3f158a3au,
    0x3f491a06u, 0x3f8408fbu, 0x3fb45dcfu, 0x4007469au,
};

constant uint QQA_ADAPTIVE_P5_CENTROID_BITS[16] = {
    0x3d214c9eu, 0x3df27670u, 0x3e4aeb9cu, 0x3e8efa3fu,
    0x3eb97bb4u, 0x3ee558d1u, 0x3f0982c6u, 0x3f218b69u,
    0x3f3b2b35u, 0x3f56f4e8u, 0x3f75d8c1u, 0x3f8cda65u,
    0x3fa379d0u, 0x3fc3f223u, 0x3ff8a44fu, 0x4028a4feu,
};

inline ushort qqa_adaptive_read_u16_le(device const uchar* src, uint offset) {
    return (ushort)(((ushort)src[offset]) |
                    (((ushort)src[offset + 1]) << 8));
}

inline uint qqa_adaptive_read_u32_le(device const uchar* src, uint offset) {
    return ((uint)src[offset]) |
           (((uint)src[offset + 1]) << 8) |
           (((uint)src[offset + 2]) << 16) |
           (((uint)src[offset + 3]) << 24);
}

inline float qqa_adaptive_centroid(uint raw_code, uint precision) {
    const uint prefix = raw_code >> (8u - precision);
    if (precision == 4) {
        return prefix < 8
            ? as_type<float>(QQA_ADAPTIVE_P4_CENTROID_BITS[prefix])
            : -as_type<float>(QQA_ADAPTIVE_P4_CENTROID_BITS[15u - prefix]);
    }
    return prefix < 16
        ? as_type<float>(QQA_ADAPTIVE_P5_CENTROID_BITS[prefix])
        : -as_type<float>(QQA_ADAPTIVE_P5_CENTROID_BITS[31u - prefix]);
}

inline uint qqa_adaptive_plane_bit(device const uchar* plane, uint within) {
    const uint byte_offset = QQA_ADAPTIVE_PLANE_BYTES - 1 - within / 8;
    const uchar bit_mask = (uchar)(1u << (within & 7u));
    return (plane[byte_offset] & bit_mask) != 0 ? 1u : 0u;
}

inline float qqa_adaptive_value(device const uchar* base,
                                device const uchar* metadata,
                                device const uchar* sidecar,
                                uint row,
                                uint within) {
    const uint meta_offset = row * QQA_ADAPTIVE_META_STRIDE;
    const uint tier = qqa_adaptive_read_u32_le(metadata, meta_offset);
    const uint sidecar_offset = qqa_adaptive_read_u32_le(metadata, meta_offset + 4);

    if (tier == QQA_ADAPTIVE_BF16) {
        const uint bits = ((uint)qqa_adaptive_read_u16_le(
            sidecar, sidecar_offset + within * 2)) << 16;
        return as_type<float>(bits);
    }
    if (tier == QQA_ADAPTIVE_F32) {
        return as_type<float>(qqa_adaptive_read_u32_le(
            sidecar, sidecar_offset + within * 4));
    }

    const uint row_base = row * QQA_ADAPTIVE_P4_STRIDE;
    uint raw_code = 0;
    for (uint plane = 0; plane < 4; ++plane) {
        const uint plane_offset = row_base + 8 + plane * QQA_ADAPTIVE_PLANE_BYTES;
        raw_code |= qqa_adaptive_plane_bit(base + plane_offset, within) << (7u - plane);
    }
    uint precision = 4;
    if (tier == QQA_ADAPTIVE_P5) {
        raw_code |= qqa_adaptive_plane_bit(sidecar + sidecar_offset, within) << 3;
        precision = 5;
    }

    const float mean = as_type<float>(qqa_adaptive_read_u32_le(base, row_base));
    const float sigma = as_type<float>(qqa_adaptive_read_u32_le(base, row_base + 4));
    return mean + sigma * qqa_adaptive_centroid(raw_code, precision);
}

constant uint QQA_ADAPTIVE_GQA6_HEADS = 6;
constant uint QQA_ADAPTIVE_GQA6_TILE = 16;
constant uint QQA_ADAPTIVE_GQA6_THREADS = QQA_ADAPTIVE_GQA6_HEADS * QQA_ADAPTIVE_SG;

kernel void qwen35_qbit_adaptive_attn_decode_gqa6(
    device const float* Q [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const uchar* k_base [[buffer(2)]],
    device const uchar* k_metadata [[buffer(3)]],
    device const uchar* k_sidecar [[buffer(4)]],
    device const uchar* v_base [[buffer(5)]],
    device const uchar* v_metadata [[buffer(6)]],
    device const uchar* v_sidecar [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant uint& cache_len [[buffer(9)]],
    constant uint& n_head [[buffer(10)]],
    constant uint& n_head_kv [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& heads_per_group [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    uint kv_h [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort local_h [[simdgroup_index_in_threadgroup]],
    ushort thread_index [[thread_index_in_threadgroup]]) {
    if (kv_h >= n_head_kv || local_h >= QQA_ADAPTIVE_GQA6_HEADS ||
        heads_per_group != QQA_ADAPTIVE_GQA6_HEADS ||
        head_dim != QQA_ADAPTIVE_HD) {
        return;
    }

    const uint h = kv_h * QQA_ADAPTIVE_GQA6_HEADS + local_h;
    if (h >= n_head) {
        return;
    }

    threadgroup float kv_tile[QQA_ADAPTIVE_GQA6_TILE * QQA_ADAPTIVE_HD];
    threadgroup float probabilities[QQA_ADAPTIVE_GQA6_HEADS][QQA_ADAPTIVE_SG];

    float m = -1e30f;
    float l = 0.0f;
    float o[QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG];
    for (uint i = 0; i < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++i) {
        o[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < cache_len; tile_start += QQA_ADAPTIVE_GQA6_TILE) {
        const uint tile_len = min(tile_start + QQA_ADAPTIVE_GQA6_TILE, cache_len) - tile_start;
        const uint tile_values = tile_len * head_dim;

        for (uint index = thread_index; index < tile_values;
             index += QQA_ADAPTIVE_GQA6_THREADS) {
            const uint position_in_tile = index / head_dim;
            const uint d = index - position_in_tile * head_dim;
            const uint row = (tile_start + position_in_tile) * n_head_kv + kv_h;
            kv_tile[index] = qqa_adaptive_value(
                k_base, k_metadata, k_sidecar, row, d);
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

        for (uint index = thread_index; index < tile_values;
             index += QQA_ADAPTIVE_GQA6_THREADS) {
            const uint position_in_tile = index / head_dim;
            const uint d = index - position_in_tile * head_dim;
            const uint row = (tile_start + position_in_tile) * n_head_kv + kv_h;
            kv_tile[index] = qqa_adaptive_value(
                v_base, v_metadata, v_sidecar, row, d);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dl = 0; dl < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++dl) {
            const uint d = lane + dl * QQA_ADAPTIVE_SG;
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
    for (uint dl = 0; dl < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++dl) {
        const uint d = lane + dl * QQA_ADAPTIVE_SG;
        const float g = gate[h * head_dim + d];
        const float sigmoid_gate = 1.0f / (1.0f + exp(-g));
        out[h * head_dim + d] = o[dl] * inv_l * sigmoid_gate;
    }
}
