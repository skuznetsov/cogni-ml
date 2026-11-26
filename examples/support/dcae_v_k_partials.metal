#include <metal_stdlib>
using namespace metal;

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

// Stage-2 conv1 only. The joined source already contains both temporal pads.
constant uint C = 512u, T = 40u, H = 72u, W = 128u;
constant uint K = C * 27u, Tile = 64u, TileCount = K / Tile;
constant uint Samples = 8192u, SparseTiles = 4u, SingletonStride = 8u;
constant uint WeightCount = C * K;

inline float from_bf16(ushort bits) {
  return as_type<float>(uint(bits) << 16u);
}

// OIktkhkw storage, but a logical K tile is 64 channels of a single tap.
kernel void mask_weights(device const ushort* weights,
                         device const uint* tiles,
                         device ushort* masks,
                         constant uint& arm,
                         uint offset [[thread_position_in_grid]]) {
  if (offset >= WeightCount) return;
  const uint oc = offset / K;
  const uint ic = (offset % K) / 27u;
  const uint tap = offset % 27u;
  const uint tile = (tap * C + ic) / Tile;
  bool keep = false;
  for (uint i = 0u; i < SparseTiles; ++i) {
    keep |= (arm == i || arm == SparseTiles) && tile == tiles[i];
  }
  if (oc % SingletonStride == 0u) keep &= ic % Tile == Tile / 2u;
  masks[offset] = keep ? weights[offset] : ushort(0);
}

kernel void local_partials(device const ushort* joined,
                           device const ushort* weights,
                           device const uint* indices,
                           device const uint* tiles,
                           device float* partials,
                           device float* masked_locals,
                           uint id [[thread_position_in_grid]]) {
  if (id >= Samples * TileCount) return;
  const uint sample = id / TileCount;
  const uint tile = id % TileCount;
  const uint tap = tile * Tile / C;
  const uint ic0 = tile * Tile % C;
  const uint output = indices[sample];
  const uint oc = output / (T * H * W);
  const uint jt = (output / (H * W)) % T + tap / 9u;
  const int y = int((output / W) % H) + int((tap / 3u) % 3u) - 1;
  const int x = int(output % W) + int(tap % 3u) - 1;
  float forward = 0.0f, reverse = 0.0f, singleton = 0.0f;
  if (y >= 0 && y < int(H) && x >= 0 && x < int(W)) {
    const uint source_stride = (T + 2u) * H * W;
    const uint source0 = (ic0 * (T + 2u) + jt) * H * W + uint(y) * W + uint(x);
    const uint weight0 = (oc * C + ic0) * 27u + tap;
    #pragma clang loop unroll(disable)
    for (uint i = 0u; i < Tile; ++i) {
      const uint j = Tile - 1u - i;
      const float a = from_bf16(joined[source0 + i * source_stride]);
      const float b = from_bf16(weights[weight0 + i * 27u]);
      forward = metal::fma(a, b, forward);
      reverse = metal::fma(from_bf16(joined[source0 + j * source_stride]),
                           from_bf16(weights[weight0 + j * 27u]), reverse);
      if (i == Tile / 2u) singleton = metal::fma(a, b, 0.0f);
    }
  }
  partials[id] = forward;
  partials[Samples * TileCount + id] = reverse;
  for (uint i = 0u; i < SparseTiles; ++i) {
    if (tiles[i] == tile) {
      const bool one_product = oc % SingletonStride == 0u;
      masked_locals[i * Samples + sample] = one_product ? singleton : forward;
      masked_locals[(SparseTiles + i) * Samples + sample] = one_product ? singleton : reverse;
    }
  }
}

inline float2 merge_orders(device const float* parts, uint count) {
  float forward = 0.0f, reverse = 0.0f;
  #pragma clang loop unroll(disable)
  for (uint i = 0u; i < count; ++i) {
    forward += parts[i];
    reverse += parts[count - 1u - i];
  }
  return float2(forward, reverse);
}

kernel void merge_partials(device const float* partials,
                           device float* results,
                           uint id [[thread_position_in_grid]]) {
  if (id >= 2u * Samples) return;
  const float2 merged = merge_orders(partials + id * TileCount, TileCount);
  const uint inner = id / Samples;
  const uint sample = id % Samples;
  results[inner * 2u * Samples + sample] = merged.x;
  results[(inner * 2u + 1u) * Samples + sample] = merged.y;
}

// Runtime terms [1, 2^-24, -1] must yield 0 forward and 2^-24 in reverse.
// This detects reassociation on both sides of the materialized boundary.
kernel void arithmetic_witness(device const float* terms,
                                device float* observed,
                                uint id [[thread_position_in_grid]]) {
  if (id != 0u) return;
  float forward = 0.0f, reverse = 0.0f;
  for (uint i = 0u; i < 3u; ++i) {
    forward = metal::fma(terms[i], 1.0f, forward);
    reverse = metal::fma(terms[2u - i], 1.0f, reverse);
  }
  const float2 merged = merge_orders(terms, 3u);
  observed[0] = forward;
  observed[1] = reverse;
  observed[2] = merged.x;
  observed[3] = merged.y;
}
