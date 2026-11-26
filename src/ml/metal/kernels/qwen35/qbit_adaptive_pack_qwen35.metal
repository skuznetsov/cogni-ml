// Append-only device packer for Qwen3.8 adaptive row-aligned QBit KV.
//
// One 32-lane SIMD group owns one 256-value semantic row. Metadata and
// canonical sidecar offsets are preplanned on the host, so packing needs no
// atomics for allocation and never materializes a persistent F32 cache.

#include <metal_stdlib>
using namespace metal;

constant uint QQP_ROW_VALUES = 256;
constant uint QQP_PLANE_BYTES = QQP_ROW_VALUES / 8;
constant uint QQP_BASE_STRIDE = 8 + 4 * QQP_PLANE_BYTES;
constant uint QQP_META_STRIDE = 8;

constant uint QQP_P4 = 0;
constant uint QQP_P5 = 1;
constant uint QQP_BF16 = 2;
constant uint QQP_F32 = 3;
constant uint QQP_SUCCESS = 0xa17ecafeu;
constant bool QQP_PREFIX_QUANT = false;

constant float QQP_POSITIVE_LEVELS[128] = {
    0.00491977f, 0.01475981f, 0.02460130f, 0.03444523f,
    0.04429256f, 0.05414428f, 0.06400137f, 0.07386480f,
    0.08373558f, 0.09361469f, 0.10350313f, 0.11340192f,
    0.12331206f, 0.13323459f, 0.14317053f, 0.15312092f,
    0.16308682f, 0.17306929f, 0.18306942f, 0.19308828f,
    0.20312698f, 0.21318664f, 0.22326841f, 0.23337343f,
    0.24350287f, 0.25365792f, 0.26383980f, 0.27404974f,
    0.28428900f, 0.29455885f, 0.30486060f, 0.31519559f,
    0.32556517f, 0.33597074f, 0.34641372f, 0.35689557f,
    0.36741778f, 0.37798188f, 0.38858944f, 0.39924207f,
    0.40994142f, 0.42068918f, 0.43148712f, 0.44233703f,
    0.45324075f, 0.46420020f, 0.47521736f, 0.48629424f,
    0.49743295f, 0.50863566f, 0.51990462f, 0.53124215f,
    0.54265067f, 0.55413266f, 0.56569073f, 0.57732756f,
    0.58904597f, 0.60084886f, 0.61273927f, 0.62472037f,
    0.63679545f, 0.64896799f, 0.66124158f, 0.67362001f,
    0.68610723f, 0.69870743f, 0.71142496f, 0.72426444f,
    0.73723071f, 0.75032892f, 0.76356447f, 0.77694313f,
    0.79047101f, 0.80415460f, 0.81800086f, 0.83201723f,
    0.84621168f, 0.86059283f, 0.87516997f, 0.88995319f,
    0.90495349f, 0.92018290f, 0.93565460f, 0.95138317f,
    0.96738473f, 0.98367719f, 1.00028055f, 1.01721718f,
    1.03451219f, 1.05219386f, 1.07029404f, 1.08884872f,
    1.10789853f, 1.12748941f, 1.14767324f, 1.16850859f,
    1.19006145f, 1.21240610f, 1.23562592f, 1.25981428f,
    1.28507552f, 1.31152588f, 1.33929452f, 1.36852460f,
    1.39937446f, 1.43201890f, 1.46665067f, 1.50348227f,
    1.54274811f, 1.58470730f, 1.62964731f, 1.67788877f,
    1.72979202f, 1.78576605f, 1.84628084f, 1.91188474f,
    1.98322915f, 2.06110438f, 2.14649281f, 2.24065008f,
    2.34523372f, 2.46251620f, 2.59575919f, 2.74992207f,
    2.93314607f, 3.16034096f, 3.46399932f, 3.94331723f,
};

inline uint qqp_read_u32_le(device const uchar* src, uint offset) {
    return ((uint)src[offset]) |
           (((uint)src[offset + 1]) << 8) |
           (((uint)src[offset + 2]) << 16) |
           (((uint)src[offset + 3]) << 24);
}

inline void qqp_write_u16_le(device uchar* dst, uint offset, ushort value) {
    dst[offset] = (uchar)(value & 0xffu);
    dst[offset + 1] = (uchar)(value >> 8);
}

inline void qqp_write_u32_le(device uchar* dst, uint offset, uint value) {
    dst[offset] = (uchar)(value & 0xffu);
    dst[offset + 1] = (uchar)((value >> 8) & 0xffu);
    dst[offset + 2] = (uchar)((value >> 16) & 0xffu);
    dst[offset + 3] = (uchar)(value >> 24);
}

inline uint qqp_quantize_raw_code(float value) {
    const bool negative = signbit(value);
    const float magnitude = abs(value);
    uint low = 0;
    uint high = 127;
    while (low < high) {
        const uint mid = (low + high) >> 1;
        const float boundary =
            (QQP_POSITIVE_LEVELS[mid] + QQP_POSITIVE_LEVELS[mid + 1]) * 0.5f;
        if (magnitude <= boundary) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    return negative ? 255u - low : low;
}

// The resident wire format stores only the four most-significant code bits
// for P4/BF16/F32 and five for P5. Search the exact boundary between those
// prefix groups instead of resolving the discarded low bits.
inline uint qqp_quantize_p4_code(float value) {
    const bool negative = signbit(value);
    const float magnitude = abs(value);
    uint low = 0u;
    uint high = 7u;
    for (uint step = 0u; step < 3u; ++step) {
        const uint mid = (low + high) >> 1;
        const uint boundary_index = (mid + 1u) * 16u - 1u;
        const float boundary =
            (QQP_POSITIVE_LEVELS[boundary_index] +
             QQP_POSITIVE_LEVELS[boundary_index + 1u]) * 0.5f;
        if (magnitude <= boundary) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    return negative ? 15u - low : low;
}

inline uint qqp_quantize_p5_code(float value) {
    const bool negative = signbit(value);
    const float magnitude = abs(value);
    uint low = 0u;
    uint high = 15u;
    for (uint step = 0u; step < 4u; ++step) {
        const uint mid = (low + high) >> 1;
        const uint boundary_index = (mid + 1u) * 8u - 1u;
        const float boundary =
            (QQP_POSITIVE_LEVELS[boundary_index] +
             QQP_POSITIVE_LEVELS[boundary_index + 1u]) * 0.5f;
        if (magnitude <= boundary) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    return negative ? 31u - low : low;
}

kernel void qwen35_qbit_adaptive_pack_row(
    device const float* source [[buffer(0)]],
    device uchar* base [[buffer(1)]],
    device const uchar* metadata [[buffer(2)]],
    device uchar* sidecar [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant uint& source_row_offset [[buffer(5)]],
    constant uint& destination_row_offset [[buffer(6)]],
    constant uint& row_count [[buffer(7)]],
    uint local_row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    if (local_row >= row_count) {
        return;
    }

    const uint source_row = source_row_offset + local_row;
    const uint destination_row = destination_row_offset + local_row;
    const uint source_offset = source_row * QQP_ROW_VALUES + lane * 8u;

    float values[8];
    float local_sum = 0.0f;
    bool finite_values = true;
    for (uint i = 0; i < 8; ++i) {
        values[i] = source[source_offset + i];
        finite_values = finite_values && isfinite(values[i]);
        local_sum += values[i];
    }
    if (!finite_values) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    }

    const float mean = simd_sum(local_sum) / (float)QQP_ROW_VALUES;
    float local_squared = 0.0f;
    for (uint i = 0; i < 8; ++i) {
        const float delta = values[i] - mean;
        local_squared += delta * delta;
    }
    const float sigma = sqrt(simd_sum(local_squared) / (float)QQP_ROW_VALUES);
    if (!isfinite(mean) || !isfinite(sigma) || sigma < 0.0f) {
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }

    const uint meta_offset = destination_row * QQP_META_STRIDE;
    const uint tier = qqp_read_u32_le(metadata, meta_offset);
    const uint sidecar_offset = qqp_read_u32_le(metadata, meta_offset + 4u);
    if (tier > QQP_F32) {
        atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    }

    const float max_centroid = tier == QQP_P4
        ? as_type<float>(0x4007469au)
        : as_type<float>(0x4028a4feu);
    if ((tier == QQP_P4 || tier == QQP_P5) &&
        (!isfinite(mean - sigma * max_centroid) ||
         !isfinite(mean + sigma * max_centroid))) {
        atomic_fetch_or_explicit(status, 8u, memory_order_relaxed);
    }

    const uint base_offset = destination_row * QQP_BASE_STRIDE;
    if (lane == 0) {
        qqp_write_u32_le(base, base_offset, as_type<uint>(mean));
        qqp_write_u32_le(base, base_offset + 4u, as_type<uint>(sigma));
    }

    uchar plane_bytes[5] = {0, 0, 0, 0, 0};
    if (QQP_PREFIX_QUANT) {
        if (tier == QQP_P5) {
            for (uint i = 0; i < 8; ++i) {
                const float normalized = sigma == 0.0f ? 0.0f : (values[i] - mean) / sigma;
                const uint prefix_code = qqp_quantize_p5_code(normalized);
                for (uint plane = 0; plane < 5u; ++plane) {
                    if ((prefix_code & (1u << (4u - plane))) != 0) {
                        plane_bytes[plane] |= (uchar)(1u << i);
                    }
                }
            }
        } else {
            for (uint i = 0; i < 8; ++i) {
                const float normalized = sigma == 0.0f ? 0.0f : (values[i] - mean) / sigma;
                const uint prefix_code = qqp_quantize_p4_code(normalized);
                for (uint plane = 0; plane < 4u; ++plane) {
                    if ((prefix_code & (1u << (3u - plane))) != 0) {
                        plane_bytes[plane] |= (uchar)(1u << i);
                    }
                }
            }
        }
    } else {
        for (uint i = 0; i < 8; ++i) {
            const float normalized = sigma == 0.0f ? 0.0f : (values[i] - mean) / sigma;
            const uint raw_code = qqp_quantize_raw_code(normalized);
            for (uint plane = 0; plane < 5; ++plane) {
                if ((raw_code & (1u << (7u - plane))) != 0) {
                    plane_bytes[plane] |= (uchar)(1u << i);
                }
            }
        }
    }

    const uint transposed_byte = QQP_PLANE_BYTES - 1u - lane;
    for (uint plane = 0; plane < 4; ++plane) {
        base[base_offset + 8u + plane * QQP_PLANE_BYTES + transposed_byte] =
            plane_bytes[plane];
    }

    if (tier == QQP_P5) {
        sidecar[sidecar_offset + transposed_byte] = plane_bytes[4];
    } else if (tier == QQP_BF16) {
        for (uint i = 0; i < 8; ++i) {
            const uint bits = as_type<uint>(values[i]);
            const uint lsb = (bits >> 16) & 1u;
            const ushort rounded = (ushort)((bits + 0x7fffu + lsb) >> 16);
            qqp_write_u16_le(sidecar, sidecar_offset + (lane * 8u + i) * 2u, rounded);
        }
    } else if (tier == QQP_F32) {
        for (uint i = 0; i < 8; ++i) {
            qqp_write_u32_le(
                sidecar,
                sidecar_offset + (lane * 8u + i) * 4u,
                as_type<uint>(values[i]));
        }
    }
}

// Publish success only if every prior encoder in the command buffer ran and
// no attention or pack kernel set an error bit. An untouched zero is failure.
kernel void qwen35_qbit_adaptive_finalize_status(
    device atomic_uint* status [[buffer(0)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) {
        return;
    }
    if (atomic_load_explicit(status, memory_order_relaxed) == 0u) {
        atomic_store_explicit(status, QQP_SUCCESS, memory_order_relaxed);
    }
}
