#include <metal_stdlib>
using namespace metal;

kernel void qwen35_artifact_bf16_decode_f32(
    device const ushort* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    uint bits = ((uint)src[gid]) << 16;
    dst[gid] = as_type<float>(bits);
}

kernel void qwen35_artifact_block_i8_decode_f32(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& block_size [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }

    uint block = gid / block_size;
    uint within = gid - block * block_size;
    uint stride = 4 + block_size;
    uint base = block * stride;
    uint bits = ((uint)src[base]) |
                (((uint)src[base + 1]) << 8) |
                (((uint)src[base + 2]) << 16) |
                (((uint)src[base + 3]) << 24);
    float scale = as_type<float>(bits);
    char q = as_type<char>(src[base + 4 + within]);
    dst[gid] = ((float)q) * scale;
}
