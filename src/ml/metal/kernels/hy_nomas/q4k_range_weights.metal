#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

constant uint Q4KR_BLOCK_ELEMENTS = 256u;
constant uint Q4KR_BLOCK_BYTES = 144u;
constant uint Q4KR_ERROR = 1u;

static inline float q4kr_divide(float numerator, float denominator) {
    return precise::divide(numerator, denominator);
}

static inline float q4kr_load_bf16(device const uchar *source, uint index) {
    const uint byte_offset = index * 2u;
    const uint bits = uint(source[byte_offset]) |
                      (uint(source[byte_offset + 1u]) << 8u);
    return as_type<float>(bits << 16u);
}

static inline void q4kr_set_error(device atomic_uint *status) {
    atomic_fetch_or_explicit(status, Q4KR_ERROR, memory_order_relaxed);
}

kernel void wan_q4k_range_pack(
    device const uchar *original_bf16 [[buffer(0)]],
    device uchar *payload [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &ranges [[buffer(4)]],
    uint block_index [[thread_position_in_grid]]) {
    const uint block_count = count / Q4KR_BLOCK_ELEMENTS;
    if (block_index >= block_count)
        return;

    device uchar *block = payload + block_index * Q4KR_BLOCK_BYTES;
    // Establish a complete fail-closed output before reading or converting data.
    for (uint byte_index = 0u; byte_index < Q4KR_BLOCK_BYTES; ++byte_index)
        block[byte_index] = 0u;

    if (ranges != 1u && ranges != 8u) {
        q4kr_set_error(status);
        return;
    }

    float raw_scale[8] = {};
    float raw_offset[8] = {};
    const uint source_base = block_index * Q4KR_BLOCK_ELEMENTS;
    bool invalid = false;
    const uint range_length = ranges == 1u ? Q4KR_BLOCK_ELEMENTS : 32u;
    const uint fit_count = ranges == 1u ? 1u : 8u;

    for (uint range = 0u; range < fit_count; ++range) {
        float low = 0.0f;
        float high = 0.0f;
        const uint range_start = source_base + range * range_length;
        for (uint lane = 0u; lane < range_length; ++lane) {
            const float value = q4kr_load_bf16(original_bf16, range_start + lane);
            if (!isfinite(value)) {
                invalid = true;
                break;
            }
            if (value < low)
                low = value;
            if (value > high)
                high = value;
        }
        if (invalid)
            break;

        const float span = high - low;
        const float scale = q4kr_divide(span, 15.0f);
        const float offset = -low;
        if (!isfinite(span) || !isfinite(scale) || !isfinite(offset)) {
            invalid = true;
            break;
        }
        raw_scale[range] = scale;
        raw_offset[range] = offset;
    }

    if (invalid) {
        q4kr_set_error(status);
        return;
    }

    if (ranges == 1u) {
        for (uint range = 1u; range < 8u; ++range) {
            raw_scale[range] = raw_scale[0];
            raw_offset[range] = raw_offset[0];
        }
    }

    float maximum_scale = 0.0f;
    float maximum_offset = 0.0f;
    for (uint range = 0u; range < 8u; ++range) {
        if (raw_scale[range] > maximum_scale)
            maximum_scale = raw_scale[range];
        if (raw_offset[range] > maximum_offset)
            maximum_offset = raw_offset[range];
    }

    // Explicit positive zero canonicalization makes a zero-valued source emit
    // the all-zero Q4_K block, including both half headers.
    const float d_unrounded = maximum_scale > 0.0f
        ? q4kr_divide(maximum_scale, 63.0f) : 0.0f;
    const float dmin_unrounded = maximum_offset > 0.0f
        ? q4kr_divide(maximum_offset, 63.0f) : 0.0f;
    if (!isfinite(d_unrounded) || !isfinite(dmin_unrounded)) {
        q4kr_set_error(status);
        return;
    }
    const half d_half = maximum_scale > 0.0f ? half(d_unrounded) : half(0.0f);
    const half dmin_half = maximum_offset > 0.0f ? half(dmin_unrounded) : half(0.0f);
    const float d = float(d_half);
    const float dmin = float(dmin_half);
    if (!isfinite(d) || !isfinite(dmin) ||
        (maximum_scale > 0.0f && !(d > 0.0f)) ||
        (maximum_offset > 0.0f && !(dmin > 0.0f))) {
        q4kr_set_error(status);
        return;
    }

    uint scale_code[8] = {};
    uint minimum_code[8] = {};
    for (uint range = 0u; range < 8u; ++range) {
        if (d > 0.0f) {
            const float ratio = q4kr_divide(raw_scale[range], d);
            if (!isfinite(ratio)) {
                invalid = true;
                break;
            }
            const float rounded = floor(ratio + 0.5f);
            scale_code[range] = uint(clamp(rounded, 0.0f, 63.0f));
        }
        if (dmin > 0.0f) {
            const float ratio = q4kr_divide(raw_offset[range], dmin);
            if (!isfinite(ratio)) {
                invalid = true;
                break;
            }
            const float rounded = floor(ratio + 0.5f);
            minimum_code[range] = uint(clamp(rounded, 0.0f, 63.0f));
        }
    }
    if (invalid) {
        q4kr_set_error(status);
        return;
    }

    uchar quantized[Q4KR_BLOCK_ELEMENTS] = {};
    for (uint index = 0u; index < Q4KR_BLOCK_ELEMENTS; ++index) {
        const float value = q4kr_load_bf16(original_bf16, source_base + index);
        const uint range = index / 32u;
        const uint sc = scale_code[range];
        if (sc == 0u)
            continue;

        const float scaled_minimum = dmin * float(minimum_code[range]);
        const float centered = value + scaled_minimum;
        const float decoded_scale = d * float(sc);
        const float ratio = q4kr_divide(centered, decoded_scale);
        if (!isfinite(scaled_minimum) || !isfinite(centered) ||
            !isfinite(decoded_scale) || !(decoded_scale > 0.0f) || !isfinite(ratio)) {
            invalid = true;
            break;
        }
        const float rounded = floor(ratio + 0.5f);
        quantized[index] = uchar(clamp(rounded, 0.0f, 15.0f));
    }
    if (invalid) {
        q4kr_set_error(status);
        return;
    }

    const ushort d_bits = as_type<ushort>(d_half);
    const ushort dmin_bits = as_type<ushort>(dmin_half);
    block[0] = uchar(d_bits & 0xffu);
    block[1] = uchar(d_bits >> 8u);
    block[2] = uchar(dmin_bits & 0xffu);
    block[3] = uchar(dmin_bits >> 8u);

    for (uint range = 0u; range < 4u; ++range) {
        block[4u + range] = uchar(scale_code[range]);
        block[8u + range] = uchar(minimum_code[range]);
    }
    for (uint range = 4u; range < 8u; ++range) {
        block[8u + range] = uchar((scale_code[range] & 0x0fu) |
                                  ((minimum_code[range] & 0x0fu) << 4u));
        block[range] |= uchar((scale_code[range] >> 4u) << 6u);
        block[4u + range] |= uchar((minimum_code[range] >> 4u) << 6u);
    }

    for (uint pair = 0u; pair < 4u; ++pair) {
        for (uint lane = 0u; lane < 32u; ++lane) {
            const uint low = quantized[pair * 64u + lane];
            const uint high = quantized[pair * 64u + 32u + lane];
            block[16u + pair * 32u + lane] = uchar(low | (high << 4u));
        }
    }
}
