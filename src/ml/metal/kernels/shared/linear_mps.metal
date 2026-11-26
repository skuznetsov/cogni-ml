#include <metal_stdlib>
using namespace metal;

inline float video_linear_mps_bias_value(device const float *bias32,
                                         device const ushort *bias16,
                                         uint index, uint is_bf16) {
  if (is_bf16 != 0)
    return as_type<float>(uint(bias16[index]) << 16);
  return bias32[index];
}

inline float video_linear_mps_roundtrip_bf16(float value) {
  const uint bits = as_type<uint>(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) {
    const uint nan_bf16 = (bits >> 16) | 0x40u;
    return as_type<float>(nan_bf16 << 16);
  }
  const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return as_type<float>(rounded & 0xffff0000u);
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
    constant uint &round_output_bf16 [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  const uint column = index % out_dim;
  float value = output[index] + video_linear_mps_bias_value(
      bias32, bias16, column, bias_is_bf16);
  if (round_output_bf16 != 0u)
    value = video_linear_mps_roundtrip_bf16(value);
  output[index] = value;
}
