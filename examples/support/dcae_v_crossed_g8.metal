// Appended to dcae_v_k_partials.metal for the opt-in sampled probe.
constant uint MaskTiles = 3u;

struct TermAddress {
  uint source;
  uint weight;
  bool valid;
};

inline float scheduled_fma(device const ushort* source,
                           device const ushort* weight,
                           device const uint* terms,
                           uint count,
                           thread const TermAddress* addresses,
                           uint source_stride,
                           uint weight_stride,
                           bool singleton) {
  float acc = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint i = 0u; i < count; ++i) {
    const uint lane = terms[i] % Tile;
    const TermAddress at = addresses[terms[i] / Tile];
    if (at.valid && (!singleton || lane == Tile / 2u)) {
      acc = metal::fma(from_bf16(source[at.source + lane * source_stride]),
                       from_bf16(weight[at.weight + lane * weight_stride]), acc);
    }
  }
  return acc;
}

kernel void scheduled_samples(device const ushort* joined,
                              device const ushort* weight,
                              device const uint* indices,
                              device const uint* tiles,
                              device const uint* terms,
                              device const uint* offsets,
                              device float* output,
                              uint id [[thread_position_in_grid]]) {
  const uint row = id / Samples;
  const uint coord = indices[id % Samples];
  const uint oc = coord / (T * H * W);
  const uint t = coord / (H * W) % T;
  const uint source_stride = (T + 2u) * H * W;
  TermAddress addresses[MaskTiles];
  for (uint slot = 0u; slot < MaskTiles; ++slot) {
    const uint tile = tiles[slot];
    const uint tap = tile * Tile / C;
    const uint ic = tile * Tile % C;
    const int y = int(coord / W % H) + int(tap / 3u % 3u) - 1;
    const int x = int(coord % W) + int(tap % 3u) - 1;
    const bool valid = y >= 0 && y < int(H) && x >= 0 && x < int(W);
    addresses[slot] = {
        valid ? ic * source_stride + (t + tap / 9u) * H * W + uint(y) * W + uint(x) : 0u,
        (oc * C + ic) * 27u + tap, valid};
  }
  const uint start = offsets[row];
  output[id] = scheduled_fma(joined, weight, terms + start, offsets[row + 1u] - start,
                             addresses, source_stride, 27u, oc % SingletonStride == 0u);
}

kernel void scheduled_witness(device const ushort* values,
                              device const ushort* ones,
                              device const uint* terms,
                              device const uint* offsets,
                              device float* output,
                              constant uint& count,
                              uint id [[thread_position_in_grid]]) {
  const uint row = id / count;
  TermAddress addresses[MaskTiles];
  for (uint slot = 0u; slot < MaskTiles; ++slot) {
    addresses[slot] = {slot * Tile, 0u, true};
  }
  const uint start = offsets[row];
  output[id] = scheduled_fma(values + (id % count) * MaskTiles * Tile, ones,
                             terms + start, offsets[row + 1u] - start, addresses,
                             1u, 1u, false);
}
