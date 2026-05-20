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
