#include <metal_stdlib>
using namespace metal;

inline float video_linear_mps_bias_value(device const float *bias32,
                                         device const ushort *bias16,
                                         uint index, uint is_bf16) {
  if (is_bf16 != 0)
    return as_type<float>(uint(bias16[index]) << 16);
  return bias32[index];
}

// MPSNDArray computes X * W^T. Wan checkpoints store a separate learned bias,
// so add it in a following F32 dispatch on the caller's still-open encoder.
kernel void video_linear_mps_bias_f32(
    device float *output [[buffer(0)]],
    device const float *bias32 [[buffer(1)]],
    device const ushort *bias16 [[buffer(2)]],
    constant uint &out_dim [[buffer(3)]],
    constant uint &bias_is_bf16 [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  const uint column = index % out_dim;
  output[index] += video_linear_mps_bias_value(
      bias32, bias16, column, bias_is_bf16);
}
