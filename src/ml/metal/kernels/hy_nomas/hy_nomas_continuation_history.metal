// Append a contiguous active F32 Q/K/V prefix to the persistent per-layer
// continuation cache. The dispatch runs after attention on the caller's chain.
kernel void hy_nomas_continuation_history_append_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &destination_offset [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    destination[destination_offset + index] = source[index];
  }
}
