#pragma clang fp contract(off)

kernel void hy_nomas_dit_channels_to_tokens_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &tokens [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < tokens * channels) {
    uint token = index / channels;
    uint channel = index % channels;
    destination[index] = source[channel * tokens + token];
  }
}

kernel void hy_nomas_dit_tokens_to_channels_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &tokens [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < tokens * channels) {
    uint channel = index / tokens;
    uint token = index % tokens;
    destination[index] = source[token * channels + channel];
  }
}

kernel void hy_nomas_dit_final_modulate_f32(
    device const float *norm [[buffer(0)]],
    device const float *modulation [[buffer(1)]],
    device float *destination [[buffer(2)]],
    constant uint &dim [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    uint channel = index % dim;
    float shift = modulation[channel];
    float scale = modulation[dim + channel];
    float scaled = norm[index] * (1.0f + scale);
    destination[index] = scaled + shift;
  }
}
