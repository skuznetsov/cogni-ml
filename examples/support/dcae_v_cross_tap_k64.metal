// Compiled after dcae_v_adjacent_k64.metal to reuse its ordered FMA fold.

kernel void mask_aligned_taps(device const ushort* weights,
                             device ushort* masked,
                             constant uint& first,
                             constant uint& second,
                             uint id [[thread_position_in_grid]]) {
  if (id >= WeightCount) return;
  const uint oc = id / K;
  const uint ic = (id % K) / 27u;
  const uint tile = ((id % 27u) * C + ic) / Tile;
  const bool keep = oc % SingletonStride != 0u || ic % Tile == Tile / 2u;
  masked[id] = keep && (tile == first || tile == second) ? weights[id] : ushort(0);
}

inline float alternate_taps(device const ushort* a, device const ushort* b,
                            device const ushort* wa, device const ushort* wb,
                            uint stride, uint wstride, uint count, uint chunk) {
  const uint step = min(chunk, count);
  float acc = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint base = 0u; base < count; base += step) {
    acc = fold_tile(a + base * stride, wa + base * wstride, stride, wstride, step, acc);
    acc = fold_tile(b + base * stride, wb + base * wstride, stride, wstride, step, acc);
  }
  return acc;
}

inline void cross_tap_rows(device const ushort* a, device const ushort* b,
                           device const ushort* wa, device const ushort* wb,
                           uint stride, uint wstride, uint count,
                           device float* rows, uint pitch) {
  const float local_a = fold_tile(a, wa, stride, wstride, count, 0.0f);
  const float local_b = fold_tile(b, wb, stride, wstride, count, 0.0f);
  rows[0] = local_a;
  rows[pitch] = local_b;
  rows[2u * pitch] = local_a + local_b;
  uint row = 3u;
  #pragma clang loop unroll(disable)
  for (uint chunk = 1u; chunk <= Tile; chunk *= 2u) {
    rows[row++ * pitch] = alternate_taps(a, b, wa, wb, stride, wstride, count, chunk);
    rows[row++ * pitch] = alternate_taps(b, a, wb, wa, stride, wstride, count, chunk);
  }
}

kernel void cross_tap_samples(device const ushort* joined,
                              device const ushort* weight,
                              device const uint* indices,
                              device float* rows,
                              constant uint& first,
                              constant uint& second,
                              uint id [[thread_position_in_grid]]) {
  if (id >= Samples) return;
  // Host admits aligned channel tiles and center-spatial taps only.
  const uint plane = H * W;
  const uint output = indices[id];
  const uint oc = output / (T * plane);
  const uint pixel = output % (T * plane);
  const uint tap_a = first * Tile / C;
  const uint tap_b = second * Tile / C;
  const bool singleton = oc % SingletonStride == 0u;
  const uint ic = first * Tile % C + (singleton ? Tile / 2u : 0u);
  const uint stride = (T + 2u) * plane;
  const uint source = ic * stride + pixel;
  const uint weights = (oc * C + ic) * 27u;
  cross_tap_rows(joined + source + tap_a / 9u * plane,
                 joined + source + tap_b / 9u * plane,
                 weight + weights + tap_a, weight + weights + tap_b,
                 stride, 27u, singleton ? 1u : Tile, rows + id, Samples);
}

kernel void cross_tap_witness(device const ushort* values,
                              device const ushort* ones,
                              device float* rows,
                              constant uint& witnesses,
                              uint id [[thread_position_in_grid]]) {
  if (id >= witnesses) return;
  const uint base = id * 2u * Tile;
  cross_tap_rows(values + base, values + base + Tile, ones, ones,
                 1u, 1u, Tile, rows + id, witnesses);
}
