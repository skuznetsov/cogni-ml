#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)

constant uint WAN_PACKED_BLOCK_ELEMENTS = 256;
constant uint WAN_Q4_K_BLOCK_BYTES = 144;
constant uint WAN_QBIT_HEADER_BYTES = 8;
constant uint WAN_QBIT_PLANE_BYTES = 32;

static inline ushort load_u16_le(device const uchar *src) {
    return ushort(src[0]) | (ushort(src[1]) << 8);
}

static inline float load_f16_le(device const uchar *src) {
    return float(as_type<half>(load_u16_le(src)));
}

// Extract the six-bit scale and minimum pair used by canonical Q4_K blocks.
static inline uchar2 get_scale_min_k4(uint index, device const uchar *scales) {
    if (index < 4) {
        return uchar2(scales[index] & 63, scales[index + 4] & 63);
    }
    return uchar2(
        (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

kernel void wan_packed_q4k_to_f32(
    device const uchar *packed [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }

    const uint block_index = gid / WAN_PACKED_BLOCK_ELEMENTS;
    const uint within = gid - block_index * WAN_PACKED_BLOCK_ELEMENTS;
    device const uchar *block = packed + block_index * WAN_Q4_K_BLOCK_BYTES;
    const float d = load_f16_le(block);
    const float dmin = load_f16_le(block + 2);
    device const uchar *scales = block + 4;
    device const uchar *qs = block + 16;

    const uint group = within / 64;
    const uint rem = within - group * 64;
    const uint scale_index = group * 2 + (rem >= 32 ? 1 : 0);
    const uchar2 scale_min = get_scale_min_k4(scale_index, scales);
    const uchar packed_q = qs[group * 32 + (rem & 31)];
    const uint q = rem < 32 ? packed_q & 0x0f : packed_q >> 4;

    // Preserve ggml's d*scale, dmin*minimum, then scaled-q minus minimum order.
    const float d_scale = d * float(scale_min.x);
    const float dmin_min = dmin * float(scale_min.y);
    destination[gid] = d_scale * float(q) - dmin_min;
}

static inline uint load_u32_le(device const uchar *src) {
    return uint(src[0]) |
           (uint(src[1]) << 8) |
           (uint(src[2]) << 16) |
           (uint(src[3]) << 24);
}

kernel void wan_packed_qbit_to_f32(
    device const uchar *packed [[buffer(0)]],
    device float *destination [[buffer(1)]],
    device const float *lut [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    constant uint &precision [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }

    const uint block_index = gid / WAN_PACKED_BLOCK_ELEMENTS;
    const uint within = gid - block_index * WAN_PACKED_BLOCK_ELEMENTS;
    const uint block_stride = WAN_QBIT_HEADER_BYTES + precision * WAN_QBIT_PLANE_BYTES;
    device const uchar *block = packed + block_index * block_stride;
    const float mean = as_type<float>(load_u32_le(block));
    const float sigma = as_type<float>(load_u32_le(block + 4));

    // This is the codec's transposed_position mapping for 256 values:
    // reverse byte order within each plane while preserving the low bit index.
    const uint byte_offset = (WAN_QBIT_PLANE_BYTES - 1) - (within / 8);
    const uchar bit_mask = uchar(1u << (within & 7u));
    uint raw_code = 0;
    for (uint plane = 0; plane < precision; ++plane) {
        const uint plane_offset = WAN_QBIT_HEADER_BYTES + plane * WAN_QBIT_PLANE_BYTES;
        if ((block[plane_offset + byte_offset] & bit_mask) != 0) {
            raw_code |= 1u << (7u - plane);
        }
    }

    const float scaled = sigma * lut[raw_code];
    destination[gid] = mean + scaled;
}
