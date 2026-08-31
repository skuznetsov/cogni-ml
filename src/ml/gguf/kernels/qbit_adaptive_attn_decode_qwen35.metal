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

constant bool QQA_ADAPTIVE_DEQUANT_T4 = false;
constant bool QQA_ADAPTIVE_SPLITK_STAGE2_FUSED = false;

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

// Planned session caches use one tier for every row in a layer. Keep the
// generic metadata path above for admitted mixed artifacts, but skip its two
// metadata loads and per-value tier selection when the host proves P4/BF16
// uniformity from both immutable K/V plans.
inline float qqa_adaptive_uniform_value(device const uchar* base,
                                        device const uchar* metadata,
                                        device const uchar* sidecar,
                                        uint row,
                                        uint within,
                                        uint uniform_tier) {
    if (uniform_tier == QQA_ADAPTIVE_P4) {
        const uint row_base = row * QQA_ADAPTIVE_P4_STRIDE;
        uint raw_code = 0;
        for (uint plane = 0; plane < 4; ++plane) {
            const uint plane_offset = row_base + 8 + plane * QQA_ADAPTIVE_PLANE_BYTES;
            raw_code |= qqa_adaptive_plane_bit(base + plane_offset, within) << (7u - plane);
        }
        const float mean = as_type<float>(qqa_adaptive_read_u32_le(base, row_base));
        const float sigma = as_type<float>(qqa_adaptive_read_u32_le(base, row_base + 4));
        return mean + sigma * qqa_adaptive_centroid(raw_code, 4);
    }
    if (uniform_tier == QQA_ADAPTIVE_BF16) {
        const uint sidecar_offset = row * QQA_ADAPTIVE_HD * 2u;
        const uint bits = ((uint)qqa_adaptive_read_u16_le(
            sidecar, sidecar_offset + within * 2)) << 16;
        return as_type<float>(bits);
    }
    return qqa_adaptive_value(base, metadata, sidecar, row, within);
}

// Four adjacent values never cross a row because the GQA6 fill traversal is
// aligned to four and Qwen3.8 head_dim is 256. Keep the four dequantizations
// register-local: one header read and one byte from each P4 plane, with no
// cross-lane exchange or cache-layout change.
inline float4 qqa_adaptive_uniform_value4(device const uchar* base,
                                          device const uchar* metadata,
                                          device const uchar* sidecar,
                                          uint row,
                                          uint within,
                                          uint uniform_tier) {
    if (uniform_tier == QQA_ADAPTIVE_P4) {
        const uint row_base = row * QQA_ADAPTIVE_P4_STRIDE;
        uint raw0 = 0;
        uint raw1 = 0;
        uint raw2 = 0;
        uint raw3 = 0;
        for (uint plane = 0; plane < 4; ++plane) {
            const uint plane_offset = row_base + 8 + plane * QQA_ADAPTIVE_PLANE_BYTES;
            const uint byte_offset = QQA_ADAPTIVE_PLANE_BYTES - 1 - within / 8;
            const uint plane_byte = base[plane_offset + byte_offset];
            const uint shift = 7u - plane;
            raw0 |= ((plane_byte >> ((within + 0u) & 7u)) & 1u) << shift;
            raw1 |= ((plane_byte >> ((within + 1u) & 7u)) & 1u) << shift;
            raw2 |= ((plane_byte >> ((within + 2u) & 7u)) & 1u) << shift;
            raw3 |= ((plane_byte >> ((within + 3u) & 7u)) & 1u) << shift;
        }
        const float mean = as_type<float>(qqa_adaptive_read_u32_le(base, row_base));
        const float sigma = as_type<float>(qqa_adaptive_read_u32_le(base, row_base + 4));
        return float4(
            mean + sigma * qqa_adaptive_centroid(raw0, 4),
            mean + sigma * qqa_adaptive_centroid(raw1, 4),
            mean + sigma * qqa_adaptive_centroid(raw2, 4),
            mean + sigma * qqa_adaptive_centroid(raw3, 4));
    }
    if (uniform_tier == QQA_ADAPTIVE_BF16) {
        const uint sidecar_offset = row * QQA_ADAPTIVE_HD * 2u + within * 2u;
        const uint bits0 = ((uint)qqa_adaptive_read_u16_le(sidecar, sidecar_offset + 0u)) << 16;
        const uint bits1 = ((uint)qqa_adaptive_read_u16_le(sidecar, sidecar_offset + 2u)) << 16;
        const uint bits2 = ((uint)qqa_adaptive_read_u16_le(sidecar, sidecar_offset + 4u)) << 16;
        const uint bits3 = ((uint)qqa_adaptive_read_u16_le(sidecar, sidecar_offset + 6u)) << 16;
        return float4(
            as_type<float>(bits0), as_type<float>(bits1),
            as_type<float>(bits2), as_type<float>(bits3));
    }
    return float4(
        qqa_adaptive_value(base, metadata, sidecar, row, within + 0u),
        qqa_adaptive_value(base, metadata, sidecar, row, within + 1u),
        qqa_adaptive_value(base, metadata, sidecar, row, within + 2u),
        qqa_adaptive_value(base, metadata, sidecar, row, within + 3u));
}

inline void qqa_adaptive_store4(threadgroup float* destination,
                                uint index,
                                float4 values) {
    destination[index + 0u] = values.x;
    destination[index + 1u] = values.y;
    destination[index + 2u] = values.z;
    destination[index + 3u] = values.w;
}

constant uint QQA_ADAPTIVE_GQA6_HEADS = 6;
constant uint QQA_ADAPTIVE_GQA6_TILE = 16;
constant uint QQA_ADAPTIVE_GQA6_THREADS = QQA_ADAPTIVE_GQA6_HEADS * QQA_ADAPTIVE_SG;

inline void qqa_adaptive_fill_uniform_tile(
    threadgroup float* destination,
    device const uchar* base,
    device const uchar* metadata,
    device const uchar* sidecar,
    device const float* current,
    uint tile_start,
    uint tile_values,
    uint packed_len,
    uint source_token_offset,
    uint kv_dim,
    uint kv_h,
    uint n_head_kv,
    uint head_dim,
    uint uniform_tier,
    uint thread_index) {
    const bool use_t4 = QQA_ADAPTIVE_DEQUANT_T4 &&
        (uniform_tier == QQA_ADAPTIVE_P4 || uniform_tier == QQA_ADAPTIVE_BF16);
    if (use_t4) {
        const uint tile_vectors = tile_values / 4u;
        for (uint vector_index = thread_index; vector_index < tile_vectors;
             vector_index += QQA_ADAPTIVE_GQA6_THREADS) {
            const uint index = vector_index * 4u;
            const uint position_in_tile = index / head_dim;
            const uint d = index - position_in_tile * head_dim;
            const uint position = tile_start + position_in_tile;
            float4 values;
            if (position < packed_len) {
                const uint row = position * n_head_kv + kv_h;
                values = qqa_adaptive_uniform_value4(
                    base, metadata, sidecar, row, d, uniform_tier);
            } else {
                const uint current_token =
                    source_token_offset + position - packed_len;
                const uint source_index =
                    current_token * kv_dim + kv_h * head_dim + d;
                values = *((device const float4*)(current + source_index));
            }
            qqa_adaptive_store4(destination, index, values);
        }
    } else {
        for (uint index = thread_index; index < tile_values;
             index += QQA_ADAPTIVE_GQA6_THREADS) {
            const uint position_in_tile = index / head_dim;
            const uint d = index - position_in_tile * head_dim;
            const uint position = tile_start + position_in_tile;
            if (position < packed_len) {
                const uint row = position * n_head_kv + kv_h;
                destination[index] = qqa_adaptive_uniform_value(
                    base, metadata, sidecar, row, d, uniform_tier);
            } else {
                const uint current_token =
                    source_token_offset + position - packed_len;
                destination[index] = current[
                    current_token * kv_dim + kv_h * head_dim + d];
            }
        }
    }
}

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

// Each query token attends to the immutable packed prefix and the causal part
// of its exact current chunk. Six query heads share each decoded KV tile.
kernel void qwen35_qbit_adaptive_prefill_chunk_gqa6(
    device const float* Q [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* current_k [[buffer(2)]],
    device const float* current_v [[buffer(3)]],
    device const uchar* k_base [[buffer(4)]],
    device const uchar* k_metadata [[buffer(5)]],
    device const uchar* k_sidecar [[buffer(6)]],
    device const uchar* v_base [[buffer(7)]],
    device const uchar* v_metadata [[buffer(8)]],
    device const uchar* v_sidecar [[buffer(9)]],
    device float* out [[buffer(10)]],
    device atomic_uint* status [[buffer(11)]],
    constant uint& packed_len [[buffer(12)]],
    constant uint& token_count [[buffer(13)]],
    constant uint& n_head [[buffer(14)]],
    constant uint& n_head_kv [[buffer(15)]],
    constant uint& head_dim [[buffer(16)]],
    constant uint& heads_per_group [[buffer(17)]],
    constant float& scale [[buffer(18)]],
    constant uint& source_token_offset [[buffer(19)]],
    constant uint& uniform_tier [[buffer(20)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort local_h [[simdgroup_index_in_threadgroup]],
    ushort thread_index [[thread_index_in_threadgroup]]) {
    const uint kv_h = group.x;
    const uint token = group.y;
    if (kv_h >= n_head_kv || token >= token_count ||
        local_h >= QQA_ADAPTIVE_GQA6_HEADS ||
        heads_per_group != QQA_ADAPTIVE_GQA6_HEADS ||
        head_dim != QQA_ADAPTIVE_HD) {
        return;
    }

    const uint h = kv_h * QQA_ADAPTIVE_GQA6_HEADS + local_h;
    if (h >= n_head) {
        return;
    }

    const uint kv_dim = n_head_kv * head_dim;
    const uint source_token = source_token_offset + token;
    const uint query_offset = (source_token * n_head + h) * head_dim;
    const uint visible_len = packed_len + token + 1u;
    threadgroup float kv_tile[QQA_ADAPTIVE_GQA6_TILE * QQA_ADAPTIVE_HD];
    threadgroup float probabilities[QQA_ADAPTIVE_GQA6_HEADS][QQA_ADAPTIVE_SG];

    float m = -1e30f;
    float l = 0.0f;
    float o[QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG];
    for (uint i = 0; i < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++i) {
        o[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < visible_len;
         tile_start += QQA_ADAPTIVE_GQA6_TILE) {
        const uint tile_len = min(tile_start + QQA_ADAPTIVE_GQA6_TILE, visible_len) - tile_start;
        const uint tile_values = tile_len * head_dim;

        qqa_adaptive_fill_uniform_tile(
            kv_tile, k_base, k_metadata, k_sidecar, current_k,
            tile_start, tile_values, packed_len, source_token_offset,
            kv_dim, kv_h, n_head_kv, head_dim, uniform_tier, thread_index);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = -1e30f;
        if (lane < tile_len) {
            threadgroup const float4* key =
                (threadgroup const float4*)(kv_tile + lane * head_dim);
            device const float4* query =
                (device const float4*)(Q + query_offset);
            float dot = 0.0f;
            for (uint d4 = 0; d4 < head_dim / 4; ++d4) {
                const float4 k4 = key[d4];
                const float4 q4 = query[d4];
                dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
            }
            score = dot * scale;
            if (!isfinite(score)) {
                atomic_fetch_or_explicit(status, 16u, memory_order_relaxed);
            }
        }

        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float correction = exp(m - m_new);
        const float probability = lane < tile_len ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(probability);
        probabilities[local_h][lane] = probability;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        qqa_adaptive_fill_uniform_tile(
            kv_tile, v_base, v_metadata, v_sidecar, current_v,
            tile_start, tile_values, packed_len, source_token_offset,
            kv_dim, kv_h, n_head_kv, head_dim, uniform_tier, thread_index);
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
        const uint index = query_offset + d;
        const float g = gate[index];
        const float sigmoid_gate = 1.0f / (1.0f + exp(-g));
        const float value = o[dl] * inv_l * sigmoid_gate;
        out[index] = value;
        if (!isfinite(g) || !isfinite(value)) {
            atomic_fetch_or_explicit(status, 32u, memory_order_relaxed);
        }
    }
}

// Long-context one-token decode. Stage 1 preserves GQA6 KV sharing while
// splitting the visible context into independent online-softmax summaries.
// The current token remains exact F32 until the following pack encoder.
kernel void qwen35_qbit_adaptive_decode_splitk_stage1_gqa6(
    device const float* Q [[buffer(0)]],
    device const float* current_k [[buffer(1)]],
    device const float* current_v [[buffer(2)]],
    device const uchar* k_base [[buffer(3)]],
    device const uchar* k_metadata [[buffer(4)]],
    device const uchar* k_sidecar [[buffer(5)]],
    device const uchar* v_base [[buffer(6)]],
    device const uchar* v_metadata [[buffer(7)]],
    device const uchar* v_sidecar [[buffer(8)]],
    device float* partial_o [[buffer(9)]],
    device float* partial_m [[buffer(10)]],
    device float* partial_l [[buffer(11)]],
    device atomic_uint* status [[buffer(12)]],
    constant uint& packed_len [[buffer(13)]],
    constant uint& n_head [[buffer(14)]],
    constant uint& n_head_kv [[buffer(15)]],
    constant uint& head_dim [[buffer(16)]],
    constant uint& heads_per_group [[buffer(17)]],
    constant float& scale [[buffer(18)]],
    constant uint& chunk_size [[buffer(19)]],
    constant uint& n_blocks [[buffer(20)]],
    constant uint& uniform_tier [[buffer(21)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort local_h [[simdgroup_index_in_threadgroup]],
    ushort thread_index [[thread_index_in_threadgroup]]) {
    const uint kv_h = group.x;
    const uint block = group.y;
    if (kv_h >= n_head_kv || block >= n_blocks ||
        local_h >= QQA_ADAPTIVE_GQA6_HEADS ||
        heads_per_group != QQA_ADAPTIVE_GQA6_HEADS ||
        head_dim != QQA_ADAPTIVE_HD || chunk_size == 0u) {
        return;
    }

    const uint h = kv_h * QQA_ADAPTIVE_GQA6_HEADS + local_h;
    if (h >= n_head) {
        return;
    }

    const uint visible_len = packed_len + 1u;
    const uint block_start = block * chunk_size;
    const uint block_end = min(block_start + chunk_size, visible_len);
    threadgroup float kv_tile[QQA_ADAPTIVE_GQA6_TILE * QQA_ADAPTIVE_HD];
    threadgroup float probabilities[QQA_ADAPTIVE_GQA6_HEADS][QQA_ADAPTIVE_SG];

    float m = -1e30f;
    float l = 0.0f;
    float o[QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG];
    for (uint i = 0; i < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++i) {
        o[i] = 0.0f;
    }

    for (uint tile_start = block_start; tile_start < block_end;
         tile_start += QQA_ADAPTIVE_GQA6_TILE) {
        const uint tile_len = min(tile_start + QQA_ADAPTIVE_GQA6_TILE, block_end) - tile_start;
        const uint tile_values = tile_len * head_dim;

        qqa_adaptive_fill_uniform_tile(
            kv_tile, k_base, k_metadata, k_sidecar, current_k,
            tile_start, tile_values, packed_len, 0u,
            n_head_kv * head_dim, kv_h, n_head_kv, head_dim,
            uniform_tier, thread_index);
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
            if (!isfinite(score)) {
                atomic_fetch_or_explicit(status, 64u, memory_order_relaxed);
            }
        }

        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float correction = exp(m - m_new);
        const float probability = lane < tile_len ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(probability);
        probabilities[local_h][lane] = probability;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        qqa_adaptive_fill_uniform_tile(
            kv_tile, v_base, v_metadata, v_sidecar, current_v,
            tile_start, tile_values, packed_len, 0u,
            n_head_kv * head_dim, kv_h, n_head_kv, head_dim,
            uniform_tier, thread_index);
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

    const uint mb = h * n_blocks + block;
    if (lane == 0) {
        partial_m[mb] = m;
        partial_l[mb] = l;
    }
    const uint out_base = mb * head_dim;
    for (uint dl = 0; dl < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++dl) {
        const uint d = lane + dl * QQA_ADAPTIVE_SG;
        partial_o[out_base + d] = o[dl];
    }
}

kernel void qwen35_qbit_adaptive_decode_splitk_stage2(
    device const float* gate [[buffer(0)]],
    device const float* partial_o [[buffer(1)]],
    device const float* partial_m [[buffer(2)]],
    device const float* partial_l [[buffer(3)]],
    device float* out [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant uint& n_head [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& n_blocks [[buffer(8)]],
    uint h [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (h >= n_head || head_dim != QQA_ADAPTIVE_HD) {
        return;
    }

    float m = -1e30f;
    for (uint block = 0; block < n_blocks; ++block) {
        m = max(m, partial_m[h * n_blocks + block]);
    }

    if (QQA_ADAPTIVE_SPLITK_STAGE2_FUSED) {
        float l_total = 0.0f;
        float acc[QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG] = {0.0f};
        // Preserve the global-max normalization and ascending block order while
        // sharing one weight across the eight output dimensions owned by a lane.
        for (uint block = 0; block < n_blocks; ++block) {
            const uint mb = h * n_blocks + block;
            const float weight = exp(partial_m[mb] - m);
            l_total += partial_l[mb] * weight;
            const uint out_base = mb * head_dim;
            for (uint dl = 0; dl < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++dl) {
                const uint d = lane + dl * QQA_ADAPTIVE_SG;
                acc[dl] += partial_o[out_base + d] * weight;
            }
        }
        const float inv_l = l_total > 0.0f ? 1.0f / l_total : 0.0f;
        for (uint dl = 0; dl < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++dl) {
            const uint d = lane + dl * QQA_ADAPTIVE_SG;
            const uint index = h * head_dim + d;
            const float g = gate[index];
            const float value = acc[dl] * inv_l / (1.0f + exp(-g));
            out[index] = value;
            if (!isfinite(g) || !isfinite(value)) {
                atomic_fetch_or_explicit(status, 128u, memory_order_relaxed);
            }
        }
    } else {
        float l_total = 0.0f;
        for (uint block = 0; block < n_blocks; ++block) {
            const uint mb = h * n_blocks + block;
            l_total += partial_l[mb] * exp(partial_m[mb] - m);
        }
        const float inv_l = l_total > 0.0f ? 1.0f / l_total : 0.0f;

        for (uint dl = 0; dl < QQA_ADAPTIVE_HD / QQA_ADAPTIVE_SG; ++dl) {
            const uint d = lane + dl * QQA_ADAPTIVE_SG;
            float acc = 0.0f;
            for (uint block = 0; block < n_blocks; ++block) {
                const uint mb = h * n_blocks + block;
                acc += partial_o[mb * head_dim + d] * exp(partial_m[mb] - m);
            }
            const uint index = h * head_dim + d;
            const float g = gate[index];
            const float value = acc * inv_l / (1.0f + exp(-g));
            out[index] = value;
            if (!isfinite(g) || !isfinite(value)) {
                atomic_fetch_or_explicit(status, 128u, memory_order_relaxed);
            }
        }
    }
}
