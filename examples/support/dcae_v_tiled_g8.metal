#include <metal_stdlib>
using namespace metal;

// Fixed stage-2 conv1 geometry. Input is joined NCTHW (left and right frames).
constant uint Channels = 512u, Frames = 40u, Height = 72u, Width = 128u;
constant uint ChannelGroup = 8u, Taps = 27u, K = Channels * Taps;
constant uint OutputStride = Frames * Height * Width;
constant uint InputStride = (Frames + 2u) * Height * Width;
constant uint TilePositions = OutputStride / 2u, TileChannels = 4u;
constant uint CompareThreads = 65536u;

kernel void pack_g8_weights(device const ushort* weights,
                            device ushort4* packed,
                            uint tid [[thread_position_in_grid]]) {
  const uint oc = tid / K * TileChannels;
  const uint group = tid / (ChannelGroup * Taps) % (Channels / ChannelGroup);
  const uint tap = tid / ChannelGroup % Taps;
  const uint lane = tid % ChannelGroup;
  const uint k = (group * ChannelGroup + lane) * Taps + tap;
  const device ushort* row = weights + oc * K + k;
  packed[tid] = ushort4(row[0], row[K], row[2u * K], row[3u * K]);
}

inline float unpack_bf16(ushort bits) {
  return as_type<float>(uint(bits) << 16u);
}

inline ushort4 round_bf16(float4 value) {
  const uint4 bits = as_type<uint4>(value);
  const uint4 rounded = (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
  const bool4 nan = (bits & 0x7fffffffu) > 0x7f800000u;
  return ushort4(select(rounded, (bits >> 16u) | 0x40u, nan));
}

template <typename Store>
inline void store_channels(device Store* out, uint index, vec<Store, 4> value) {
  out[index] = value.x;
  out[index + OutputStride] = value.y;
  out[index + 2u * OutputStride] = value.z;
  out[index + 3u * OutputStride] = value.w;
}

inline void store_output(device float* out, uint index, float4 acc) {
  store_channels(out, index, acc);
}

inline void store_output(device ushort* out, uint index, float4 acc) {
  store_channels(out, index, round_bf16(acc));
}

template <typename Store>
inline void conv3d_g8_tile(device const ushort* joined,
                           device const ushort4* packed,
                           device Store* output,
                           uint first_tile,
                           uint tid) {
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)
  const uint oc_tile = first_tile + tid / TilePositions;
  const uint p = tid % TilePositions;
  const uint t = p / (Height * (Width / 2u));
  const int y = int(p / (Width / 2u) % Height);
  const uint column = p % (Width / 2u);
  // Adjacent SIMD lanes visit adjacent x. Each lane shares its weight
  // vector across two positions, each activation across four channels.
  const int x = int(column / 32u * 64u + column % 32u);
  const device ushort4* row = packed + oc_tile * K;
  float4 first = 0.0f, second = 0.0f;

  #pragma clang loop unroll(disable)
  for (uint group = 0u; group < Channels / ChannelGroup; ++group) {
    #pragma clang loop unroll(disable)
    for (uint tap = 0u; tap < Taps; ++tap) {
      const int yy = y + int(tap / 3u % 3u) - 1;
      if (yy < 0 || yy >= int(Height)) continue;
      const int xx = x + int(tap % 3u) - 1;
      const bool first_in = xx >= 0;
      const bool second_in = xx + 32 < int(Width);
      const uint src = group * ChannelGroup * InputStride
          + ((t + tap / 9u) * Height + uint(yy)) * Width;
      const device ushort4* weights = row + (group * Taps + tap) * ChannelGroup;
      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < ChannelGroup; ++lane) {
        const uint offset = src + lane * InputStride;
        const float4 w = as_type<float4>(uint4(weights[lane]) << 16u);
        if (first_in) first = metal::fma(float4(unpack_bf16(joined[offset + uint(xx)])),
                                          w, first);
        if (second_in) second = metal::fma(float4(unpack_bf16(joined[offset + uint(xx + 32)])),
                                            w, second);
      }
    }
  }
  const uint dst = oc_tile * TileChannels * OutputStride
      + (t * Height + uint(y)) * Width + uint(x);
  store_output(output, dst, first);
  store_output(output, dst + 32u, second);
}

kernel void tiled_g8_f32(device const ushort* joined,
                         device const ushort4* weights,
                         device float* output,
                         constant uint& first_tile,
                         uint tid [[thread_position_in_grid]]) {
  conv3d_g8_tile(joined, weights, output, first_tile, tid);
}

kernel void tiled_g8_bf16(device const ushort* joined,
                          device const ushort4* weights,
                          device ushort* output,
                          constant uint& first_tile,
                          uint tid [[thread_position_in_grid]]) {
  conv3d_g8_tile(joined, weights, output, first_tile, tid);
}

inline uint float_bits(uint raw) { return raw; }
inline uint float_bits(ushort raw) { return uint(raw) << 16u; }

template <typename Bits>
inline void compare(device const Bits* actual,
                    device const Bits* expected,
                    device atomic_uint* stats,
                    uint count,
                    float atol,
                    float rtol,
                    uint tid,
                    uint lane) {
  uint seen = 0u, unequal = 0u, outside = 0u, nonfinite = 0u;
  uint first = 0xffffffffu;
  float max_abs = 0.0f;
  for (uint index = tid; index < count; index += CompareThreads) {
    const uint a = float_bits(actual[index]);
    const uint b = float_bits(expected[index]);
    ++seen;
    if (a != b) {
      ++unequal;
      first = min(first, index);
    }
    // Inspect exponent bits so the nonfinite witness also works under fast math.
    if (max(a & 0x7fffffffu, b & 0x7fffffffu) >= 0x7f800000u) {
      ++nonfinite;
    } else {
      const float target = as_type<float>(b);
      const float error = abs(as_type<float>(a) - target);
      max_abs = max(max_abs, error);
      outside += uint(error > atol + rtol * abs(target));
    }
  }
  const uint total = simd_sum(seen);
  const uint differences = simd_sum(unequal);
  const uint failures = simd_sum(outside);
  const uint bad_values = simd_sum(nonfinite);
  const uint first_difference = simd_min(first);
  const uint largest = as_type<uint>(simd_max(max_abs));
  if (lane == 0u && total != 0u) {
    atomic_fetch_add_explicit(stats + 0, total, memory_order_relaxed);
    atomic_fetch_add_explicit(stats + 1, differences, memory_order_relaxed);
    atomic_fetch_add_explicit(stats + 2, failures, memory_order_relaxed);
    atomic_fetch_add_explicit(stats + 3, bad_values, memory_order_relaxed);
    atomic_fetch_min_explicit(stats + 4, first_difference, memory_order_relaxed);
    atomic_fetch_max_explicit(stats + 5, largest, memory_order_relaxed);
  }
}

kernel void compare_f32(device const uint* actual,
                         device const uint* expected,
                         device atomic_uint* stats,
                         constant uint& count,
                         constant float& atol,
                         constant float& rtol,
                         uint tid [[thread_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]]) {
  compare(actual, expected, stats, count, atol, rtol, tid, lane);
}

kernel void compare_bf16(device const ushort* actual,
                          device const ushort* expected,
                          device atomic_uint* stats,
                          constant uint& count,
                          constant float& atol,
                          constant float& rtol,
                          uint tid [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  compare(actual, expected, stats, count, atol, rtol, tid, lane);
}
