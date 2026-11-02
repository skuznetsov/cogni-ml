// Small source-BF16 boundaries for Hy-Nomas reference-path diagnostics.
// The shared RNE conversion helper is supplied by hy_nomas_block.metal.
#include <metal_stdlib>
using namespace metal;

kernel void hy_nomas_reference_bf16_round(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) output[id] = hy_nomas_bf16_roundtrip_f32(input[id]);
}

// Keep the (1 + scale) BF16 boundary separate from F32 activation arithmetic.
kernel void hy_nomas_reference_bf16_modulate(
    device const float *input [[buffer(0)]], device const float *shift [[buffer(1)]],
    device const float *scale [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (id < count) {
        const float factor = hy_nomas_bf16_roundtrip_f32(1.0f + scale[id]);
        const float product = input[id] * factor;
        output[id] = product + shift[id];
    }
}

// Finish RMS normalization from the existing F32 MPS row means. Activations
// and outputs retain F32 storage while each source-BF16 boundary is explicit.
kernel void hy_nomas_reference_bf16_rms_finish(
    device const float *input [[buffer(0)]], device const ushort *gain_bf16 [[buffer(1)]],
    device const float *mean_square [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &head_dim [[buffer(5)]],
    constant float &eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float inverse;
    if (row >= rows) return;
    if (tid == 0) inverse = 1.0f / metal::precise::sqrt(mean_square[row] + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < head_dim; d += threads) {
        const float normalized = input[row * head_dim + d] * inverse;
        const float normalized_bf16 = hy_nomas_bf16_roundtrip_f32(normalized);
        const float gain = as_type<float>(uint(gain_bf16[d]) << 16u);
        output[row * head_dim + d] =
            hy_nomas_bf16_roundtrip_f32(normalized_bf16 * gain);
    }
}
