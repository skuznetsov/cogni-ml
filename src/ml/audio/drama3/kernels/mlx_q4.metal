#include <metal_stdlib>
using namespace metal;

// Narrow reproduction of MLX 0.31.1's affine Q4/group-64 fast gather-QMV
// operations. Keep this in a separate source so resident.metal's explicit
// contraction-off setting does not silently change the reference expressions.
inline float dr3_mlx_q4_float(device const uchar* b, ulong o, ulong i, ulong t) {
    if (t == 0) {
        uint v = uint(b[o + i * 4]) | uint(b[o + i * 4 + 1]) << 8 |
            uint(b[o + i * 4 + 2]) << 16 | uint(b[o + i * 4 + 3]) << 24;
        return as_type<float>(v);
    }
    o += i * 2;
    ushort v = ushort(b[o]) | ushort(b[o + 1]) << 8;
    return t == 1 ? as_type<float>(uint(v) << 16) : float(as_type<half>(v));
}

inline ushort dr3_mlx_q4_u16(device const uchar* b, ulong o) {
    return ushort(b[o]) | ushort(b[o + 1]) << 8;
}

inline float dr3_mlx_q4_load_vector(device const float* x, thread float* x_thread) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
    return sum;
}

inline float dr3_mlx_q4_dot(
    device const uchar* w,
    ulong weight_byte,
    thread const float* x_thread,
    float scale,
    float bias,
    float sum) {
    float accum = 0.0f;
    for (int i = 0; i < 4; i++) {
        ushort packed = dr3_mlx_q4_u16(w, weight_byte + ulong(i) * 2);
        accum += x_thread[4 * i] * float(packed & 0x000f) +
            x_thread[4 * i + 1] * float(packed & 0x00f0) +
            x_thread[4 * i + 2] * float(packed & 0x0f00) +
            x_thread[4 * i + 3] * float(packed & 0xf000);
    }
    return scale * accum + sum * bias;
}

kernel void dr3_expert_mlx_q4(
    device const uchar* w [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    device const float* ids [[buffer(5)]],
    constant ulong* p [[buffer(6)]],
    uint2 position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    uint rank = uint(p[11]) != 0 ? position.y : uint(p[9]);
    float selected = p[8] != 0 ? ids[rank] : float(p[10]);
    uint rows = uint(p[0]);
    if (!isfinite(selected) || selected < 0.0f || selected >= float(p[2]) || floor(selected) != selected) {
        if (simd_lane == 0) {
            uint row_base = position.x * 8 + simdgroup * 4;
            for (uint i = 0; i < 4; i++) {
                uint row = row_base + i;
                y[(p[11] != 0 ? rank * rows : 0) + row] = NAN;
            }
        }
        return;
    }

    uint cols = uint(p[1]);
    uint groups = cols / 64;
    uint row_base = position.x * 8 + simdgroup * 4;
    ulong expert_row_base = ulong(uint(selected)) * rows;
    thread float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float x_thread[16];

    for (uint k = 0; k < cols; k += 512) {
        ulong input_offset = (p[12] != 0 ? ulong(rank) * cols : 0) + k + ulong(simd_lane) * 16;
        float sum = dr3_mlx_q4_load_vector(x + input_offset, x_thread);
        ulong block_weight_offset = p[3] + ulong(k / 2) + ulong(simd_lane) * 8;
        uint group = k / 64 + simd_lane / 4;

        for (uint i = 0; i < 4; i++) {
            uint row = row_base + i;
            ulong expert_row = expert_row_base + row;
            ulong scale_index = expert_row * groups + group;
            ulong weight_byte = block_weight_offset + expert_row * (cols / 8) * 4;
            float scale = dr3_mlx_q4_float(scales, p[4], scale_index, p[6]);
            float bias = dr3_mlx_q4_float(biases, p[5], scale_index, p[7]);
            result[i] += dr3_mlx_q4_dot(w, weight_byte, x_thread, scale, bias, sum);
        }
    }

    uint output_base = p[11] != 0 ? rank * rows : 0;
    for (uint i = 0; i < 4; i++) {
        float reduced = simd_sum(result[i]);
        if (simd_lane == 0) {
            y[output_base + row_base + i] = reduced;
        }
    }
}
