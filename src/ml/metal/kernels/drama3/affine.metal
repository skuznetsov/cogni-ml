#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Safetensors can place U32 at an unaligned byte offset, including beyond
// 4 GiB in a shard. Byte loads avoid undefined typed pointer alignment.
inline uint drama3_u32(device const uchar *bytes, ulong offset) {
    return uint(bytes[offset]) | (uint(bytes[offset + 1]) << 8)
        | (uint(bytes[offset + 2]) << 16) | (uint(bytes[offset + 3]) << 24);
}
inline float drama3_float(device const uchar *bytes, ulong offset, uint index, uint type) {
    if (type == 0) return as_type<float>(drama3_u32(bytes, offset + ulong(index) * 4));
    offset += ulong(index) * 2;
    ushort word = ushort(bytes[offset]) | (ushort(bytes[offset + 1]) << 8);
    return type == 1 ? as_type<float>(uint(word) << 16) : float(as_type<half>(word));
}

// Each lane computes complete affine groups in CPU order. Lane zero adds
// group subtotals in order, preserving the complete FP32 reduction tree.
kernel void drama3_affine_matvec(
    device const uchar *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const uchar *biases [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    constant ulong *p [[buffer(5)]],
    threadgroup float *parts [[threadgroup(0)]],
    uint r [[threadgroup_position_in_grid]], uint lane_id [[thread_index_in_threadgroup]]) {
    const uint rows = uint(p[0]), cols = uint(p[1]), bits = uint(p[2]);
    const uint group = uint(p[3]), expert = uint(p[4]);
    if (r >= rows) return;
    const uint per_word = 32 / bits, groups = cols / group;
    const uint row = expert * rows + r, mask = (1u << bits) - 1;
    for (uint g = lane_id; g < groups; g += 64) {
        const uint index = row * groups + g;
        const float scale = drama3_float(scales, p[6], index, uint(p[8]));
        const float bias = drama3_float(biases, p[7], index, uint(p[9]));
        float part = 0.0f;
        for (uint j = 0; j < group / per_word; ++j) {
            const uint col = g * group + j * per_word;
            const ulong offset = p[5] + (ulong(row) * (cols / per_word) + col / per_word) * 4;
            const uint word = drama3_u32(packed, offset);
            for (uint lane = 0; lane < per_word; ++lane) {
                float weight = scale * float((word >> (lane * bits)) & mask) + bias;
                part += weight * x[col + lane];
            }
        }
        parts[g] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane_id == 0) {
        float sum = 0.0f;
        for (uint g = 0; g < groups; ++g) sum += parts[g];
        y[r] = sum;
    }
}
