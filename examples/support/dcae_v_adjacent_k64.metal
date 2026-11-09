// Compiled after dcae_v_k_partials.metal to share the stage-2 geometry.

kernel void mask_adjacent_weights(device const ushort* weights,
                                  device ushort* masked,
                                  constant uint& first,
                                  constant uint& count,
                                  uint id [[thread_position_in_grid]]) {
  if (id >= WeightCount) return;
  const uint oc = id / K;
  const uint ic = (id % K) / 27u;
  const uint tile = ((id % 27u) * C + ic) / Tile;
  const bool selected = first <= tile && tile < first + count;
  const bool lane_kept = oc % SingletonStride != 0u || ic % Tile == Tile / 2u;
  masked[id] = selected && lane_kept ? weights[id] : ushort(0);
}

inline float fold_tile(device const ushort* source,
                       device const ushort* weights,
                       uint source_stride, uint weight_stride,
                       uint count, float accumulator) {
  #pragma clang loop unroll(disable)
  for (uint i = 0u; i < count; ++i) {
    accumulator = metal::fma(from_bf16(source[i * source_stride]),
                             from_bf16(weights[i * weight_stride]), accumulator);
  }
  return accumulator;
}

inline void evaluate_pair(device const ushort* source_a,
                          device const ushort* source_b,
                          device const ushort* weight_a,
                          device const ushort* weight_b,
                          uint source_stride, uint weight_stride, uint count,
                          device float* rows, uint pitch) {
  const float a = fold_tile(source_a, weight_a, source_stride, weight_stride, count, 0.0f);
  const float b = fold_tile(source_b, weight_b, source_stride, weight_stride, count, 0.0f);
  rows[0] = a;
  rows[pitch] = b;
  rows[2u * pitch] = a + b;
  rows[3u * pitch] = fold_tile(source_b, weight_b, source_stride, weight_stride, count, a);
  rows[4u * pitch] = fold_tile(source_a, weight_a, source_stride, weight_stride, count, b);
}

kernel void adjacent_samples(device const ushort* joined,
                             device const ushort* weight,
                             device const uint* indices,
                             device float* rows,
                             constant uint& first,
                             uint id [[thread_position_in_grid]]) {
  if (id >= Samples) return;
  const uint output = indices[id];
  const uint oc = output / (T * H * W);
  const uint tap = first * Tile / C;
  const uint jt = (output / (H * W)) % T + tap / 9u;
  const int y = int((output / W) % H) + int((tap / 3u) % 3u) - 1;
  const int x = int(output % W) + int(tap % 3u) - 1;
  if (y < 0 || y >= int(H) || x < 0 || x >= int(W)) {
    for (uint row = 0u; row < 5u; ++row) rows[row * Samples + id] = 0.0f;
    return;
  }
  const bool singleton = oc % SingletonStride == 0u;
  const uint ic = first * Tile % C + (singleton ? Tile / 2u : 0u);
  const uint stride = (T + 2u) * H * W;
  const uint source = ic * stride + (jt * H + uint(y)) * W + uint(x);
  const uint offset = (oc * C + ic) * 27u + tap;
  evaluate_pair(joined + source, joined + source + Tile * stride,
                weight + offset, weight + offset + Tile * 27u,
                stride, 27u, singleton ? 1u : Tile, rows + id, Samples);
}

// Swap the two input tiles to exercise both directions on the same witness.
kernel void adjacent_witness(device const ushort* values,
                             device const ushort* ones,
                             device float* rows,
                             uint id [[thread_position_in_grid]]) {
  if (id >= 2u) return;
  evaluate_pair(values + id * Tile, values + (1u - id) * Tile,
                ones, ones, 1u, 1u, Tile, rows + id, 2u);
}
