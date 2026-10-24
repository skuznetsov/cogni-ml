#include <metal_stdlib>
using namespace metal;

// Byte loads also cover valid safetensors payloads without scalar alignment.
kernel void video_weight_decode(
    device const uchar *raw [[buffer(0)]], device float *out [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &dtype [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    if (dtype == 0) {
        const uint p = i * 4;
        uint word = uint(raw[p]) | (uint(raw[p+1]) << 8) |
                    (uint(raw[p+2]) << 16) | (uint(raw[p+3]) << 24);
        out[i] = as_type<float>(word);
    } else {
        const uint p = i * 2;
        ushort word = ushort(raw[p]) | (ushort(raw[p+1]) << 8);
        out[i] = dtype == 1 ? as_type<float>(uint(word) << 16) : float(as_type<half>(word));
    }
}

// Round F32 source bytes to IEEE BF16 with round-to-nearest, ties-to-even.
// Handle NaNs before the finite rounding path so a payload-only NaN remains a
// quiet NaN instead of rounding to infinity.
inline ushort video_weight_f32_to_bf16(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return ushort((bits >> 16) | 0x0040u);
    }
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Opt-in setup conversion used only for original F32 safetensors when the
// block explicitly selects BF16 residency. Original BF16 payloads bypass this
// kernel and remain direct copies into their final destinations.
kernel void video_weight_encode_bf16(
    device const uchar *raw [[buffer(0)]], device ushort *out [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint p = i * 4;
    const uint word = uint(raw[p]) | (uint(raw[p + 1]) << 8) |
                      (uint(raw[p + 2]) << 16) | (uint(raw[p + 3]) << 24);
    out[i] = video_weight_f32_to_bf16(as_type<float>(word));
}
