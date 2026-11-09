#include <metal_stdlib>
using namespace metal;

// Two fixed Conv3D geometries. Inputs are BF16 BCTHW with B=1. The packed
// input scratch has 42 temporal slots: optional cached frame, 40 source
// frames, then one logical zero frame.
constant uint DcaeVC8Frames = 40u;
constant uint DcaeVC8ChannelGroup = 8u;
constant uint DcaeVC8Taps = 27u;
constant uint DcaeVC8TileChannels = 4u;
constant uint DcaeVC8OutputTilesPerDispatch = 8u;

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
                                     uint channel_stride,
                                     ushort4 value) {
  output[index] = value.x;
  output[index + channel_stride] = value.y;
  output[index + 2u * channel_stride] = value.z;
  output[index + 3u * channel_stride] = value.w;
}

// Pack the cache and contiguous BF16 source into C8 scratch.
// Destination order is [channel_group,42,H,W], with two ushort4 values per
// pixel. If use_cache is zero, frame 0 is written as logical zero and cache is
// never read. The final temporal slot (41) is always logical zero.
template <uint Channels, uint Height, uint Width>
inline void dcae_v_c8_pack_input_impl(
    device const ushort* input,
    device const ushort* cache,
    device ushort4* packed,
    uint use_cache,
    uint tid) {
  const uint input_stride = (DcaeVC8Frames + 2u) * Height * Width;
  const uint pack_positions = (Channels / DcaeVC8ChannelGroup) * input_stride;
  if (tid >= pack_positions) return;

  const uint group = tid / input_stride;
  const uint pixel = tid % input_stride;
  const uint frame = pixel / (Height * Width);
  const uint spatial = pixel % (Height * Width);
  const uint channel = group * DcaeVC8ChannelGroup;
  ushort values[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};

  if (frame == 0u) {
    if (use_cache != 0u) {
      const uint plane = Height * Width;
      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < DcaeVC8ChannelGroup; ++lane) {
        values[lane] = cache[(channel + lane) * plane + spatial];
      }
    }
  } else if (frame <= DcaeVC8Frames) {
    const uint plane = Height * Width;
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

// Reorder contiguous upstream [C,C,3,3,3] BF16 weights into the existing
// OC4/C8 layout used by the DCAE G8 kernel profile.
template <uint Channels>
inline void dcae_v_c8_pack_weight_impl(
    device const ushort* weight,
    device ushort4* packed_weight,
    uint tid) {
  const uint k_count = Channels * DcaeVC8Taps;
  const uint pack_weights = (Channels / DcaeVC8TileChannels) * k_count;
  if (tid >= pack_weights) return;

  const uint oc = (tid / k_count) * DcaeVC8TileChannels;
  const uint group =
      tid / (DcaeVC8ChannelGroup * DcaeVC8Taps) %
      (Channels / DcaeVC8ChannelGroup);
  const uint tap = tid / DcaeVC8ChannelGroup % DcaeVC8Taps;
  const uint lane = tid % DcaeVC8ChannelGroup;
  const uint k = (group * DcaeVC8ChannelGroup + lane) * DcaeVC8Taps + tap;
  const device ushort* row = weight + oc * k_count + k;
  packed_weight[tid] = ushort4(row[0u], row[k_count],
                               row[2u * k_count], row[3u * k_count]);
}

// Compute 4 output channels at two x positions per thread. first_tile is in
// OC4 units. Each dispatch covers 8 OC4 tiles (32 channels).
template <uint Channels, uint Height, uint Width>
inline void dcae_v_c8_conv_impl(
    device const ushort4* packed_input,
    device const ushort4* packed_weight,
    device ushort* output,
    device const ushort* bias,
    uint first_tile,
    uint use_bias,
    uint tid) {
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)
  const uint input_stride = (DcaeVC8Frames + 2u) * Height * Width;
  const uint output_stride = DcaeVC8Frames * Height * Width;
  const uint tile_positions = output_stride / 2u;
  const uint thread_count = DcaeVC8OutputTilesPerDispatch * tile_positions;
  const uint k_count = Channels * DcaeVC8Taps;
  if (tid >= thread_count) return;

  const uint oc_tile = first_tile + tid / tile_positions;
  const uint p = tid % tile_positions;
  const uint t = p / (Height * (Width / 2u));
  const int y = int(p / (Width / 2u) % Height);
  const uint column = p % (Width / 2u);
  const int x = int(column / 32u * 64u + column % 32u);
  const device ushort4* row = packed_weight + oc_tile * k_count;
  float4 first = 0.0f;
  float4 second = 0.0f;

  // Preserve the profiled FMA K order: channel group, temporal/spatial tap,
  // then input lane 0 through 7.
  #pragma clang loop unroll(disable)
  for (uint group = 0u;
       group < Channels / DcaeVC8ChannelGroup;
       ++group) {
    #pragma clang loop unroll(disable)
    for (uint tap = 0u; tap < DcaeVC8Taps; ++tap) {
      const int yy = y + int(tap / 3u % 3u) - 1;
      if (yy < 0 || yy >= int(Height)) continue;
      const int xx = x + int(tap % 3u) - 1;
      const bool first_in = xx >= 0;
      const bool second_in = xx + 32 < int(Width);
      const uint src = group * input_stride +
          ((t + tap / 9u) * Height + uint(yy)) * Width;
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

  const uint dst = oc_tile * DcaeVC8TileChannels * output_stride +
      (t * Height + uint(y)) * Width + uint(x);
  dcae_v_c8_store_channels(output, dst, output_stride, first_result);
  dcae_v_c8_store_channels(output, dst + 32u, output_stride,
                           second_result);
}

kernel void dcae_v_c8_pack_input(
    device const ushort* input [[buffer(0)]],
    device const ushort* cache [[buffer(1)]],
    device ushort4* packed [[buffer(2)]],
    constant uint& use_cache [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  dcae_v_c8_pack_input_impl<512u, 72u, 128u>(input, cache, packed, use_cache, tid);
}

kernel void dcae_v_c8_pack_weight(
    device const ushort* weight [[buffer(0)]],
    device ushort4* packed_weight [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  dcae_v_c8_pack_weight_impl<512u>(weight, packed_weight, tid);
}

kernel void dcae_v_c8_conv(
    device const ushort4* packed_input [[buffer(0)]],
    device const ushort4* packed_weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    device const ushort* bias [[buffer(3)]],
    constant uint& first_tile [[buffer(4)]],
    constant uint& use_bias [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  dcae_v_c8_conv_impl<512u, 72u, 128u>(packed_input, packed_weight, output, bias, first_tile, use_bias, tid);
}

kernel void dcae_v_c8_stage1_pack_input(
    device const ushort* input [[buffer(0)]],
    device const ushort* cache [[buffer(1)]],
    device ushort4* packed [[buffer(2)]],
    constant uint& use_cache [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  dcae_v_c8_pack_input_impl<256u, 144u, 256u>(input, cache, packed, use_cache, tid);
}

kernel void dcae_v_c8_stage1_pack_weight(
    device const ushort* weight [[buffer(0)]],
    device ushort4* packed_weight [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
  dcae_v_c8_pack_weight_impl<256u>(weight, packed_weight, tid);
}

kernel void dcae_v_c8_stage1_conv(
    device const ushort4* packed_input [[buffer(0)]],
    device const ushort4* packed_weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    device const ushort* bias [[buffer(3)]],
    constant uint& first_tile [[buffer(4)]],
    constant uint& use_bias [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
  dcae_v_c8_conv_impl<256u, 144u, 256u>(packed_input, packed_weight, output, bias, first_tile, use_bias, tid);
}
