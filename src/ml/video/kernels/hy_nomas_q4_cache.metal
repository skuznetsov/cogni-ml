#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

constant uint HY_NOMAS_Q4_BLOCK_ELEMENTS = 256u;
constant uint HY_NOMAS_Q4_BLOCK_BYTES = 144u;

static inline ushort hy_nomas_q4_load_u16_le(device const uchar *src) {
    return ushort(src[0]) | (ushort(src[1]) << 8u);
}

static inline float hy_nomas_q4_load_f16_le(device const uchar *src) {
    return float(as_type<half>(hy_nomas_q4_load_u16_le(src)));
}

static inline uchar2 hy_nomas_q4_scale_min(uint index, device const uchar *scales) {
    if (index < 4u) {
        return uchar2(scales[index] & 63u, scales[index + 4u] & 63u);
    }
    return uchar2(
        (scales[index + 4u] & 0x0fu) | ((scales[index - 4u] >> 6u) << 4u),
        (scales[index + 4u] >> 4u) | ((scales[index] >> 6u) << 4u));
}

static inline ushort hy_nomas_q4_f32_to_bf16(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return ushort((bits >> 16u) | 0x0040u);
    }
    return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

// Decode canonical Q4_K values with the same non-contracted multiply/subtract
// order as packed_weight_decode.metal, then round once to BF16 storage.
kernel void hy_nomas_q4k_to_bf16(
    device const uchar *packed [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;

    const uint block_index = gid / HY_NOMAS_Q4_BLOCK_ELEMENTS;
    const uint within = gid - block_index * HY_NOMAS_Q4_BLOCK_ELEMENTS;
    device const uchar *block = packed + block_index * HY_NOMAS_Q4_BLOCK_BYTES;
    const float d = hy_nomas_q4_load_f16_le(block);
    const float dmin = hy_nomas_q4_load_f16_le(block + 2u);
    device const uchar *scales = block + 4u;
    device const uchar *qs = block + 16u;

    const uint group = within / 64u;
    const uint rem = within - group * 64u;
    const uint scale_index = group * 2u + (rem >= 32u ? 1u : 0u);
    const uchar2 scale_min = hy_nomas_q4_scale_min(scale_index, scales);
    const uchar packed_q = qs[group * 32u + (rem & 31u)];
    const uint q = rem < 32u ? packed_q & 0x0fu : packed_q >> 4u;

    const float d_scale = d * float(scale_min.x);
    const float dmin_min = dmin * float(scale_min.y);
    const float decoded = d_scale * float(q) - dmin_min;
    destination[gid] = hy_nomas_q4_f32_to_bf16(decoded);
}

kernel void hy_nomas_copy_bf16(
    device const uchar *source [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    const uint byte_offset = gid * 2u;
    destination[gid] = ushort(source[byte_offset]) |
                       (ushort(source[byte_offset + 1u]) << 8u);
}
