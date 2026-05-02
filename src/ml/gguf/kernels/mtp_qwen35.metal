// Qwen3.6 MTP BF16 dense helpers.
//
// Correctness-first GEMV kernels for the extracted BF16 MTP sidecar. These are
// intentionally simple: one output row per simdgroup, F32 input/output, row-major
// BF16 weights. The next production step is to keep the whole MTP body resident
// and fuse adjacent operations.

#include <metal_stdlib>
using namespace metal;

static inline float qwen35_bf16_to_f32(ushort h) {
    return as_type<float>(((uint)h) << 16);
}

constant short MTP_BF16_NSG = 2;
constant short MTP_BF16_NR0 = 1;

kernel void qwen35_bf16_gemv_f32(
    device const ushort* w       [[buffer(0)]],
    device const float*  x       [[buffer(1)]],
    device       float*  output  [[buffer(2)]],
    constant     uint&   in_dim  [[buffer(3)]],
    constant     uint&   out_dim [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row = (tgpig.x * MTP_BF16_NSG + sgitg) * MTP_BF16_NR0;
    if (row >= out_dim) return;

    device const ushort* w_row = w + row * in_dim;
    float acc = 0.0f;
    for (uint col = tiisg; col < in_dim; col += 32) {
        acc += qwen35_bf16_to_f32(w_row[col]) * x[col];
    }

    const float total = simd_sum(acc);
    if (tiisg == 0) {
        output[row] = total;
    }
}

// Qwen3.5/3.6 MTP q_proj packs rows per head as [q, output_gate]. For the
// exact one-token MTP shortcut, only the gate rows are needed.
kernel void qwen35_bf16_q_gate_gemv_f32(
    device const ushort* w          [[buffer(0)]],
    device const float*  x          [[buffer(1)]],
    device       float*  output     [[buffer(2)]],
    constant     uint&   in_dim     [[buffer(3)]],
    constant     uint&   q_dim      [[buffer(4)]],
    constant     uint&   head_dim   [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint dst_row = (tgpig.x * MTP_BF16_NSG + sgitg) * MTP_BF16_NR0;
    if (dst_row >= q_dim) return;

    const uint head = dst_row / head_dim;
    const uint d = dst_row - head * head_dim;
    const uint src_row = head * (2 * head_dim) + head_dim + d;

    device const ushort* w_row = w + src_row * in_dim;
    float acc = 0.0f;
    for (uint col = tiisg; col < in_dim; col += 32) {
        acc += qwen35_bf16_to_f32(w_row[col]) * x[col];
    }

    const float total = simd_sum(acc);
    if (tiisg == 0) {
        output[dst_row] = total;
    }
}
