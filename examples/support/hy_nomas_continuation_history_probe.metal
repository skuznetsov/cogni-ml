kernel void hn_continuation_probe_copy_f32(
    device const float *source [[buffer(0)]],
    device float *destination [[buffer(1)]],
    constant uint &source_offset [[buffer(2)]],
    constant uint &destination_offset [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    destination[destination_offset + index] = source[source_offset + index];
  }
}
