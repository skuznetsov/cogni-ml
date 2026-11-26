// Probe-only F32 cache initialization. This source is loaded only when
// compiling with -Dhy_nomas_history_probe.
kernel void hy_nomas_continuation_history_fill_f32(
    device float *destination [[buffer(0)]],
    constant float &value [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count) {
    destination[index] = value;
  }
}
