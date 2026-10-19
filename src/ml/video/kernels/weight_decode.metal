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
