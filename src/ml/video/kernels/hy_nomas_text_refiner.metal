#pragma clang fp contract(off)

inline float hy_text_load_weight(device const float *f32,
                                 device const ushort *bf16,
                                 uint index, uint is_bf16) {
  if (is_bf16) {
    return as_type<float>((uint)bf16[index] << 16);
  }
  return f32[index];
}

inline ushort hy_text_round_bf16(float value) {
  uint bits = as_type<uint>(value);
  uint lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return (ushort)(bits >> 16);
}

kernel void hy_nomas_text_compact_bf16_f32(
    device const ushort *source [[buffer(0)]],
    device float *compact [[buffer(1)]],
    constant uint &valid_rows [[buffer(2)]],
    constant uint &input_dim [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  uint count = valid_rows * input_dim;
  if (index >= count) return;
  compact[index] = as_type<float>((uint)source[index] << 16);
}

kernel void hy_nomas_text_masked_mean_f32(
    device const float *compact [[buffer(0)]],
    device float *mean [[buffer(1)]],
    constant uint &valid_rows [[buffer(2)]],
    constant uint &input_dim [[buffer(3)]],
    uint channel [[thread_position_in_grid]]) {
  if (channel >= input_dim) return;
  float sum = 0.0f;
  for (uint row = 0; row < valid_rows; ++row) {
    sum += compact[row * input_dim + channel];
  }
  mean[channel] = sum / (float)valid_rows;
}

// Fixed dim=256/max_period=10000 positional basis from pinned upstream
// timestep_embedding, whose arange/exp executes in CPU F32 before GPU use.
// Captured with PyTorch 2.8.0; little-endian F32 basis SHA-256:
// 97280b1699e126e32cf1e88b1f02603981a549c529734af7989ecbfa3294492e
// Only the timestep-independent basis is fixed. Phase and trig stay on GPU.
constant uint HY_TEXT_TIME_FREQUENCY_BITS[128] = {
  0x3f800000u, 0x3f6e39f8u, 0x3f5dafd6u, 0x3f4e4bacu, 0x3f3ff911u, 0x3f32a506u, 0x3f263de0u, 0x3f1ab32bu,
  0x3f0ff59au, 0x3f05f6eeu, 0x3ef953ceu, 0x3ee80460u, 0x3ed7e89au, 0x3ec8eb24u, 0x3ebaf81au, 0x3eadfcffu,
  0x3ea1e89bu, 0x3e96aaeau, 0x3e8c3503u, 0x3e827908u, 0x3e72d422u, 0x3e61f836u, 0x3e5247edu, 0x3e43ae7cu,
  0x3e361887u, 0x3e297409u, 0x3e1db040u, 0x3e12bd90u, 0x3e088d77u, 0x3dfe24dfu, 0x3dec7fd5u, 0x3ddc1464u,
  0x3dccccccu, 0x3dbe94c8u, 0x3db15978u, 0x3da50957u, 0x3d99940du, 0x3d8eea6cu, 0x3d84fe4cu, 0x3d778511u,
  0x3d6655c1u, 0x3d5657e3u, 0x3d477640u, 0x3d399d18u, 0x3d2cba16u, 0x3d20bc1bu, 0x3d159348u, 0x3d0b30cbu,
  0x3d0186e2u, 0x3cf11177u, 0x3ce054d1u, 0x3cd0c1a8u, 0x3cc2434eu, 0x3cb4c692u, 0x3ca83989u, 0x3c9c8b96u,
  0x3c91ad3au, 0x3c879009u, 0x3c7c4d30u, 0x3c6ac8e6u, 0x3c5a7bf1u, 0x3c4b50b5u, 0x3c3d330eu, 0x3c301050u,
  0x3c23d70au, 0x3c187706u, 0x3c0de12eu, 0x3c040777u, 0x3bf5b9adu, 0x3be4aa45u, 0x3bd4ca16u, 0x3bc6040au,
  0x3bb8449au, 0x3bab7982u, 0x3b9f91ccu, 0x3b947dafu, 0x3b8a2e75u, 0x3b80967cu, 0x3b6f520du, 0x3b5eb47au,
  0x3b4f3e33u, 0x3b40dac2u, 0x3b33770du, 0x3b270153u, 0x3b1b690eu, 0x3b109ed8u, 0x3b06946du, 0x3afa78f0u,
  0x3ae91528u, 0x3ad8e674u, 0x3ac9d759u, 0x3abbd3eau, 0x3aaec98eu, 0x3aa2a6f7u, 0x3a975c0bu, 0x3a8cd9d9u,
  0x3a83126eu, 0x3a73f1a2u, 0x3a6301e3u, 0x3a533f24u, 0x3a44948au, 0x3a36ee9eu, 0x3a2a3b44u, 0x3a1e69a2u,
  0x3a136a14u, 0x3a092e01u, 0x39ff4facu, 0x39ed95e4u, 0x39dd1722u, 0x39cdbd93u, 0x39bf74d6u, 0x39b229fbu,
  0x39a5cb60u, 0x399a48a0u, 0x398f9275u, 0x39859aa4u, 0x3978a80du, 0x3967648cu, 0x395753e2u, 0x394860c0u,
  0x393a7753u, 0x392d852au, 0x39217919u, 0x39164326u, 0x390bd46du, 0x39021f28u, 0x38f22cddu, 0x38e15c8eu
};

kernel void hy_nomas_text_timestep_sinusoidal_f32(
    device const float *timestep [[buffer(0)]],
    device float *embedding [[buffer(1)]],
    constant uint &half_dim [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (half_dim != 128u || index >= half_dim) return;
  float frequency = as_type<float>(HY_TEXT_TIME_FREQUENCY_BITS[index]);
  float phase = timestep[0] * frequency;
  // Safe arithmetic mode alone does not select accurate transcendental
  // functions. Match the upstream MPSGraph trig on the rounded F32 phase.
  embedding[index] = precise::cos(phase);
  embedding[index + half_dim] = precise::sin(phase);
}

kernel void hy_nomas_text_qkv_split_f32(
    device const float *qkv [[buffer(0)]],
    device float *q [[buffer(1)]],
    device float *k [[buffer(2)]],
    device float *v [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  uint count = rows * dim;
  if (index >= count) return;
  uint row = index / dim;
  uint channel = index - row * dim;
  uint source = row * (3u * dim) + channel;
  q[index] = qkv[source];
  k[index] = qkv[source + dim];
  v[index] = qkv[source + 2u * dim];
}

kernel void hy_nomas_text_split_adaln_f32(
    device const float *adaln [[buffer(0)]],
    device float *gate_msa [[buffer(1)]],
    device float *gate_mlp [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    uint channel [[thread_position_in_grid]]) {
  if (channel >= dim) return;
  gate_msa[channel] = adaln[channel];
  gate_mlp[channel] = adaln[channel + dim];
}

// Upstream apply_gate and residual addition are separate F32 operations.
// Keep the product rounded before adding the residual, including broadcast
// gates; contraction here can be amplified by the subsequent MLP projection.
kernel void hy_nomas_text_gated_residual_f32(
    device const float *input [[buffer(0)]],
    device const float *delta [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
  if (index >= count) return;
  float product = delta[index] * gate[index % dim];
  output[index] = input[index] + product;
}

kernel void hy_nomas_text_layer_norm_affine_f32(
    device float *values [[buffer(0)]],
    device const float *weight_f32 [[buffer(1)]],
    device const float *bias_f32 [[buffer(2)]],
    device const ushort *weight_bf16 [[buffer(3)]],
    device const ushort *bias_bf16 [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    constant uint &weight_is_bf16 [[buffer(7)]],
    constant uint &bias_is_bf16 [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  uint channel = index % dim;
  float weight = hy_text_load_weight(weight_f32, weight_bf16, channel,
                                     weight_is_bf16);
  float bias = hy_text_load_weight(bias_f32, bias_bf16, channel,
                                   bias_is_bf16);
  values[index] = values[index] * weight + bias;
}

kernel void hy_nomas_text_roundtrip_bf16_f32(
    device float *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  values[index] = as_type<float>((uint)hy_text_round_bf16(values[index]) << 16);
}
