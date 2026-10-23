#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

constant uint Q4KD_BLOCK_ELEMENTS = 256u;
constant uint Q4KD_BLOCK_BYTES = 144u;
constant uint Q4KD_ERROR = 1u;

static inline ushort q4kd_load_u16_le(device const uchar *src) {
    return ushort(src[0]) | (ushort(src[1]) << 8u);
}

static inline void q4kd_store_u16_le(device uchar *dst, ushort value) {
    dst[0] = uchar(value & 0xffu);
    dst[1] = uchar(value >> 8u);
}

static inline float q4kd_load_f16_le(device const uchar *src) {
    return float(as_type<half>(q4kd_load_u16_le(src)));
}

static inline float q4kd_load_bf16(device const uchar *source, uint index) {
    const uint offset = index * 2u;
    const uint bits = uint(source[offset]) | (uint(source[offset + 1u]) << 8u);
    return as_type<float>(bits << 16u);
}

static inline uchar2 q4kd_get_scale_min_k4(uint index, device const uchar *scales) {
    if (index < 4u) {
        return uchar2(scales[index] & 63u, scales[index + 4u] & 63u);
    }
    return uchar2(
        (scales[index + 4u] & 0x0fu) | ((scales[index - 4u] >> 6u) << 4u),
        (scales[index + 4u] >> 4u) | ((scales[index] >> 6u) << 4u));
}

// Match PackedWeightDecode's canonical arithmetic order exactly. The safe
// pipeline math mode and contract-off pragma keep the products separate.
static inline float q4kd_decode_base(device const uchar *block, uint within) {
    const float d = q4kd_load_f16_le(block);
    const float dmin = q4kd_load_f16_le(block + 2u);
    device const uchar *scales = block + 4u;
    device const uchar *qs = block + 16u;

    const uint group = within / 64u;
    const uint rem = within - group * 64u;
    const uint scale_index = group * 2u + (rem >= 32u ? 1u : 0u);
    const uchar2 scale_min = q4kd_get_scale_min_k4(scale_index, scales);
    const uchar packed_q = qs[group * 32u + (rem & 31u)];
    const uint q = rem < 32u ? packed_q & 0x0fu : packed_q >> 4u;

    const float d_scale = d * float(scale_min.x);
    const float dmin_min = dmin * float(scale_min.y);
    return d_scale * float(q) - dmin_min;
}

static inline void q4kd_set_error(device atomic_uint *status) {
    atomic_fetch_or_explicit(status, Q4KD_ERROR, memory_order_relaxed);
}

kernel void wan_q4k_delta_pack(
    device const uchar *original_bf16 [[buffer(0)]],
    device const uchar *canonical_q4k_payload [[buffer(1)]],
    device uchar *sidecar [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &patches [[buffer(5)]],
    uint block_index [[thread_position_in_grid]]) {
    const uint block_count = count / Q4KD_BLOCK_ELEMENTS;
    if (block_index >= block_count)
        return;
    if (patches != 1u && patches != 4u && patches != 8u) {
        q4kd_set_error(status);
        return;
    }

    const uint stride = patches * 3u;
    device uchar *records = sidecar + block_index * stride;
    for (uint byte_index = 0u; byte_index < stride; ++byte_index)
        records[byte_index] = 0u;

    const uint source_base = block_index * Q4KD_BLOCK_ELEMENTS;
    device const uchar *q4k_block =
        canonical_q4k_payload + block_index * Q4KD_BLOCK_BYTES;
    float source_values[Q4KD_BLOCK_ELEMENTS];
    float base_values[Q4KD_BLOCK_ELEMENTS];
    float errors[Q4KD_BLOCK_ELEMENTS];
    bool invalid = false;

    for (uint index = 0u; index < Q4KD_BLOCK_ELEMENTS; ++index) {
        const float source = q4kd_load_bf16(original_bf16, source_base + index);
        const float base = q4kd_decode_base(q4k_block, index);
        const float error = fabs(source - base);
        if (!isfinite(source) || !isfinite(base) || !isfinite(error)) {
            invalid = true;
            break;
        }
        source_values[index] = source;
        base_values[index] = base;
        errors[index] = error;
    }
    if (invalid) {
        q4kd_set_error(status);
        return;
    }

    uint selected_indices[8];
    ushort selected_delta_bits[8];
    for (uint rank = 0u; rank < patches; ++rank) {
        bool found = false;
        uint best_index = 0u;
        float best_error = -1.0f;
        for (uint index = 0u; index < Q4KD_BLOCK_ELEMENTS; ++index) {
            bool already_selected = false;
            for (uint prior = 0u; prior < rank; ++prior) {
                if (selected_indices[prior] == index) {
                    already_selected = true;
                    break;
                }
            }
            if (already_selected)
                continue;
            const float error = errors[index];
            if (!found || error > best_error ||
                (error == best_error && index < best_index)) {
                found = true;
                best_error = error;
                best_index = index;
            }
        }
        if (!found) {
            invalid = true;
            break;
        }

        const float delta = source_values[best_index] - base_values[best_index];
        if (!isfinite(delta)) {
            invalid = true;
            break;
        }
        const half delta_half = half(delta);
        const float rounded_delta = float(delta_half);
        if (!isfinite(rounded_delta)) {
            invalid = true;
            break;
        }
        selected_indices[rank] = best_index;
        selected_delta_bits[rank] = as_type<ushort>(delta_half);
    }

    if (invalid) {
        q4kd_set_error(status);
        return;
    }
    for (uint rank = 0u; rank < patches; ++rank) {
        const uint offset = rank * 3u;
        records[offset] = uchar(selected_indices[rank]);
        q4kd_store_u16_le(records + offset + 1u, selected_delta_bits[rank]);
    }
}

kernel void wan_q4k_delta_decode(
    device const uchar *canonical_q4k_payload [[buffer(0)]],
    device const uchar *sidecar [[buffer(1)]],
    device float *destination [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &patches [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count)
        return;

    const uint block_index = gid / Q4KD_BLOCK_ELEMENTS;
    const uint within = gid - block_index * Q4KD_BLOCK_ELEMENTS;
    if (patches != 1u && patches != 4u && patches != 8u) {
        q4kd_set_error(status);
        destination[gid] = 0.0f;
        return;
    }

    device const uchar *records = sidecar + block_index * patches * 3u;
    bool malformed = false;
    for (uint rank = 0u; rank < patches; ++rank) {
        const uint offset = rank * 3u;
        const uint index = uint(records[offset]);
        const float delta = q4kd_load_f16_le(records + offset + 1u);
        if (!isfinite(delta))
            malformed = true;
        for (uint prior = 0u; prior < rank; ++prior) {
            if (uint(records[prior * 3u]) == index)
                malformed = true;
        }
    }
    if (malformed) {
        q4kd_set_error(status);
        destination[gid] = 0.0f;
        return;
    }

    device const uchar *q4k_block =
        canonical_q4k_payload + block_index * Q4KD_BLOCK_BYTES;
    const float base = q4kd_decode_base(q4k_block, within);
    if (!isfinite(base)) {
        q4kd_set_error(status);
        destination[gid] = 0.0f;
        return;
    }

    float delta = 0.0f;
    for (uint rank = 0u; rank < patches; ++rank) {
        const uint offset = rank * 3u;
        if (uint(records[offset]) == within) {
            delta = q4kd_load_f16_le(records + offset + 1u);
            break;
        }
    }
    const float decoded = base + delta;
    if (!isfinite(decoded)) {
        q4kd_set_error(status);
        destination[gid] = 0.0f;
        return;
    }
    destination[gid] = decoded;
}

// Opt-in comparison kernel. One exact 256-lane threadgroup owns each block;
// lane zero validates and stages the correction records before every lane
// reaches the unconditional barrier. The control kernel above is unchanged.
kernel void wan_q4k_delta_decode_block_validated(
    device const uchar *canonical_q4k_payload [[buffer(0)]],
    device const uchar *sidecar [[buffer(1)]],
    device float *destination [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &patches [[buffer(5)]],
    uint block_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint shared_indices[8];
    threadgroup float shared_deltas[8];
    threadgroup uint shared_malformed;

    if (lane == 0u) {
        shared_malformed = 0u;
        if (patches != 1u && patches != 4u && patches != 8u) {
            shared_malformed = 1u;
            q4kd_set_error(status);
        } else {
            device const uchar *records = sidecar + block_index * patches * 3u;
            for (uint rank = 0u; rank < patches; ++rank) {
                const uint offset = rank * 3u;
                const uint index = uint(records[offset]);
                const float delta = q4kd_load_f16_le(records + offset + 1u);
                shared_indices[rank] = index;
                shared_deltas[rank] = delta;
                if (!isfinite(delta))
                    shared_malformed = 1u;
                for (uint prior = 0u; prior < rank; ++prior) {
                    if (shared_indices[prior] == index)
                        shared_malformed = 1u;
                }
            }
            if (shared_malformed != 0u)
                q4kd_set_error(status);
        }
    }

    // All lanes, including those in malformed blocks, must reach this barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint within = lane;
    const uint gid = block_index * Q4KD_BLOCK_ELEMENTS + within;
    if (gid >= count)
        return;
    if (shared_malformed != 0u) {
        destination[gid] = 0.0f;
        return;
    }

    device const uchar *q4k_block =
        canonical_q4k_payload + block_index * Q4KD_BLOCK_BYTES;
    const float base = q4kd_decode_base(q4k_block, within);
    if (!isfinite(base)) {
        q4kd_set_error(status);
        destination[gid] = 0.0f;
        return;
    }

    float delta = 0.0f;
    for (uint rank = 0u; rank < patches; ++rank) {
        if (shared_indices[rank] == within) {
            delta = shared_deltas[rank];
            break;
        }
    }
    const float decoded = base + delta;
    if (!isfinite(decoded)) {
        q4kd_set_error(status);
        destination[gid] = 0.0f;
        return;
    }
    destination[gid] = decoded;
}
