#include <metal_stdlib>
using namespace metal;

// Fixed stage-2 Conv3D geometry. Inputs are BF16 BCTHW with B=1. The packed
// input scratch has 42 temporal slots: optional cached frame, 40 source
// frames, then one logical zero frame.
constant uint DcaeVC8Channels = 512u;
constant uint DcaeVC8Frames = 40u;
constant uint DcaeVC8Height = 72u;
constant uint DcaeVC8Width = 128u;
constant uint DcaeVC8ChannelGroup = 8u;
constant uint DcaeVC8Taps = 27u;
constant uint DcaeVC8K = DcaeVC8Channels * DcaeVC8Taps;
constant uint DcaeVC8OutputStride =
    DcaeVC8Frames * DcaeVC8Height * DcaeVC8Width;
constant uint DcaeVC8InputStride =
    (DcaeVC8Frames + 2u) * DcaeVC8Height * DcaeVC8Width;
constant uint DcaeVC8TilePositions = DcaeVC8OutputStride / 2u;
constant uint DcaeVC8TileChannels = 4u;
constant uint DcaeVC8OutputTilesPerDispatch = 8u;
constant uint DcaeVC8PackInputPositions =
    (DcaeVC8Channels / DcaeVC8ChannelGroup) * DcaeVC8InputStride;
constant uint DcaeVC8PackWeightCount =
    (DcaeVC8Channels / DcaeVC8TileChannels) * DcaeVC8K;
constant uint DcaeVC8ConvThreadCount =
    DcaeVC8OutputTilesPerDispatch * DcaeVC8TilePositions;

inline float4 dcae_v_c8_decode_bf16x4(ushort4 value) {
  return as_type<float4>(uint4(value) << 16u);
}

// Round to BF16 round-to-nearest-even and quiet NaNs, matching the established
// DCAE stage-2 profile's output conversion.
inline ushort4 dcae_v_c8_round_bf16(float4 value) {
  const uint4 bits = as_type<uint4>(value);
  const uint4 rounded = (bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u;
  const bool4 nan = (bits & 0x7fffffffu) > 0x7f800000u;
  return ushort4(select(rounded, (bits >> 16u) | 0x40u, nan));
}

inline void dcae_v_c8_store_channels(device ushort* output,
                                     uint index,
                                     ushort4 value) {
  output[index] = value.x;
  output[index + DcaeVC8OutputStride] = value.y;
  output[index + 2u * DcaeVC8OutputStride] = value.z;
  output[index + 3u * DcaeVC8OutputStride] = value.w;
}

// Pack the cache and contiguous [512,40,72,128] BF16 source into C8 scratch.
// Destination order is [channel_group,42,H,W], with two ushort4 values per
// pixel. If use_cache is zero, frame 0 is written as logical zero and cache is
// never read. The final temporal slot (41) is always logical zero.
kernel void dcae_v_c8_pack_input(
    device const ushort* input [[buffer(0)]],
    device const ushort* cache [[buffer(1)]],
    device ushort4* packed [[buffer(2)]],
    constant uint& use_cache [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= DcaeVC8PackInputPositions) return;

  const uint group = tid / DcaeVC8InputStride;
  const uint pixel = tid % DcaeVC8InputStride;
  const uint frame = pixel / (DcaeVC8Height * DcaeVC8Width);
  const uint spatial = pixel % (DcaeVC8Height * DcaeVC8Width);
  const uint channel = group * DcaeVC8ChannelGroup;
  ushort values[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

  if (frame == 0u) {
    if (use_cache != 0u) {
      const uint plane = DcaeVC8Height * DcaeVC8Width;
      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < DcaeVC8ChannelGroup; ++lane) {
        values[lane] = cache[(channel + lane) * plane + spatial];
      }
    }
  } else if (frame <= DcaeVC8Frames) {
    const uint plane = DcaeVC8Height * DcaeVC8Width;
    #pragma clang loop unroll(full)
    for (uint lane = 0u; lane < DcaeVC8ChannelGroup; ++lane) {
      const uint source_channel = channel + lane;
      values[lane] = input[(source_channel * DcaeVC8Frames + frame - 1u) *
                           plane + spatial];
    }
  }

  const uint dst = tid * 2u;
  packed[dst] = ushort4(values[0], values[1], values[2], values[3]);
  packed[dst + 1u] = ushort4(values[4], values[5], values[6], values[7]);
}

// Reorder contiguous upstream [512,512,3,3,3] BF16 weights into the existing
// OC4/C8 layout used by the DCAE G8 kernel profile.
kernel void dcae_v_c8_pack_weight(
    device const ushort* weight [[buffer(0)]],
    device ushort4* packed_weight [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= DcaeVC8PackWeightCount) return;

  const uint oc = (tid / DcaeVC8K) * DcaeVC8TileChannels;
  const uint group =
      tid / (DcaeVC8ChannelGroup * DcaeVC8Taps) %
      (DcaeVC8Channels / DcaeVC8ChannelGroup);
  const uint tap = tid / DcaeVC8ChannelGroup % DcaeVC8Taps;
  const uint lane = tid % DcaeVC8ChannelGroup;
  const uint k = (group * DcaeVC8ChannelGroup + lane) * DcaeVC8Taps + tap;
  const device ushort* row = weight + oc * DcaeVC8K + k;
  packed_weight[tid] = ushort4(row[0u], row[DcaeVC8K],
                               row[2u * DcaeVC8K], row[3u * DcaeVC8K]);
}

// Compute 4 output channels at two x positions per thread. first_tile is in
// OC4 units. Launch 16 dispatches, each covering 8 OC4 tiles (32 channels).
kernel void dcae_v_c8_conv(
    device const ushort4* packed_input [[buffer(0)]],
    device const ushort4* packed_weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    device const ushort* bias [[buffer(3)]],
    constant uint& first_tile [[buffer(4)]],
    constant uint& use_bias [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)
  if (tid >= DcaeVC8ConvThreadCount) return;

  const uint oc_tile = first_tile + tid / DcaeVC8TilePositions;
  const uint p = tid % DcaeVC8TilePositions;
  const uint t = p / (DcaeVC8Height * (DcaeVC8Width / 2u));
  const int y = int(p / (DcaeVC8Width / 2u) % DcaeVC8Height);
  const uint column = p % (DcaeVC8Width / 2u);
  const int x = int(column / 32u * 64u + column % 32u);
  const device ushort4* row = packed_weight + oc_tile * DcaeVC8K;
  float4 first = 0.0f;
  float4 second = 0.0f;

  // Preserve the profiled FMA K order: channel group, temporal/spatial tap,
  // then input lane 0 through 7.
  #pragma clang loop unroll(disable)
  for (uint group = 0u;
       group < DcaeVC8Channels / DcaeVC8ChannelGroup;
       ++group) {
    #pragma clang loop unroll(disable)
    for (uint tap = 0u; tap < DcaeVC8Taps; ++tap) {
      const int yy = y + int(tap / 3u % 3u) - 1;
      if (yy < 0 || yy >= int(DcaeVC8Height)) continue;
      const int xx = x + int(tap % 3u) - 1;
      const bool first_in = xx >= 0;
      const bool second_in = xx + 32 < int(DcaeVC8Width);
      const uint src = group * DcaeVC8InputStride +
          ((t + tap / 9u) * DcaeVC8Height + uint(yy)) * DcaeVC8Width;
      const device ushort4* tap_weights =
          row + (group * DcaeVC8Taps + tap) * DcaeVC8ChannelGroup;
      float4 first_low = 0.0f;
      float4 first_high = 0.0f;
      float4 second_low = 0.0f;
      float4 second_high = 0.0f;

      if (first_in) {
        const uint activation_index = (src + uint(xx)) * 2u;
        first_low = dcae_v_c8_decode_bf16x4(packed_input[activation_index]);
        first_high = dcae_v_c8_decode_bf16x4(packed_input[activation_index + 1u]);
      }
      if (second_in) {
        const uint activation_index = (src + uint(xx + 32)) * 2u;
        second_low = dcae_v_c8_decode_bf16x4(packed_input[activation_index]);
        second_high = dcae_v_c8_decode_bf16x4(packed_input[activation_index + 1u]);
      }

      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < DcaeVC8ChannelGroup; ++lane) {
        const float activation =
            lane < 4u ? first_low[lane] : first_high[lane - 4u];
        const float second_activation =
            lane < 4u ? second_low[lane] : second_high[lane - 4u];
        const float4 weight = dcae_v_c8_decode_bf16x4(tap_weights[lane]);
        if (first_in) {
          first = metal::fma(float4(activation), weight, first);
        }
        if (second_in) {
          second = metal::fma(float4(second_activation), weight, second);
        }
      }
    }
  }

  // Match the BF16 result boundary before bias. The optional bias pointer is
  // never dereferenced when use_bias is zero.
  ushort4 first_result = dcae_v_c8_round_bf16(first);
  ushort4 second_result = dcae_v_c8_round_bf16(second);
  if (use_bias != 0u) {
    const uint bias_index = oc_tile * DcaeVC8TileChannels;
    const ushort4 bias_bits = ushort4(bias[bias_index], bias[bias_index + 1u],
                                      bias[bias_index + 2u], bias[bias_index + 3u]);
    const float4 bias_value = dcae_v_c8_decode_bf16x4(bias_bits);
    first_result = dcae_v_c8_round_bf16(
        dcae_v_c8_decode_bf16x4(first_result) + bias_value);
    second_result = dcae_v_c8_round_bf16(
        dcae_v_c8_decode_bf16x4(second_result) + bias_value);
  }

  const uint dst = oc_tile * DcaeVC8TileChannels * DcaeVC8OutputStride +
      (t * DcaeVC8Height + uint(y)) * DcaeVC8Width + uint(x);
  dcae_v_c8_store_channels(output, dst, first_result);
  dcae_v_c8_store_channels(output, dst + 32u, second_result);
}
