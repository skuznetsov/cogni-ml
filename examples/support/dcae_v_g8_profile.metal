// This file is concatenated after dcae_v_tiled_g8.metal. It intentionally
// reuses that file's geometry, BF16 conversion, store, and strided tile helpers.

kernel void g8_pack_input(device const ushort* joined,
                          device ushort4* packed,
                          uint tid [[thread_position_in_grid]]) {
  // Flatten [64, 42, 72, 128], writing the two channel quartets for each
  // group and spatial position as the innermost dimension.
  const uint x = tid % Width;
  const uint y = tid / Width % Height;
  const uint t = tid / (Width * Height) % (Frames + 2u);
  const uint group = tid / InputStride;
  const uint pixel = (t * Height + y) * Width + x;
  const device ushort* channels = joined + group * ChannelGroup * InputStride + pixel;
  const uint dst = tid * 2u;

  packed[dst] = ushort4(channels[0u * InputStride],
                        channels[1u * InputStride],
                        channels[2u * InputStride],
                        channels[3u * InputStride]);
  packed[dst + 1u] = ushort4(channels[4u * InputStride],
                             channels[5u * InputStride],
                             channels[6u * InputStride],
                             channels[7u * InputStride]);
}

inline float4 g8_profile_decode_bf16x4(ushort4 value) {
  return as_type<float4>(uint4(value) << 16u);
}

template <bool Split>
inline void g8_profile_accumulate_lane(float activation,
                                       float4 weight,
                                       uint lane,
                                       bool first_in,
                                       bool second_in,
                                       thread float4& first,
                                       thread float4& second,
                                       thread float4& first_odd,
                                       thread float4& second_odd,
                                       float second_activation) {
  const float4 value = float4(activation);
  if (first_in) {
    if (Split && (lane & 1u) != 0u) {
      first_odd = metal::fma(value, weight, first_odd);
    } else {
      first = metal::fma(value, weight, first);
    }
  }
  if (second_in) {
    const float4 second_value = float4(second_activation);
    if (Split && (lane & 1u) != 0u) {
      second_odd = metal::fma(second_value, weight, second_odd);
    } else {
      second = metal::fma(second_value, weight, second);
    }
  }
}

template <bool Split>
inline void conv3d_g8_profile_strided_tile(device const ushort* joined,
                                           device const ushort4* weights,
                                           device ushort* output,
                                           uint first_tile,
                                           uint tid) {
  #pragma clang fp contract(off)
  #pragma clang fp reassociate(off)
  const uint oc_tile = first_tile + tid / TilePositions;
  const uint p = tid % TilePositions;
  const uint t = p / (Height * (Width / 2u));
  const int y = int(p / (Width / 2u) % Height);
  const uint column = p % (Width / 2u);
  const int x = int(column / 32u * 64u + column % 32u);
  const device ushort4* row = weights + oc_tile * K;
  float4 first = 0.0f, second = 0.0f;
  float4 first_odd = 0.0f, second_odd = 0.0f;

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
      const device ushort4* tap_weights = row + (group * Taps + tap) * ChannelGroup;
      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < ChannelGroup; ++lane) {
        const uint offset = src + lane * InputStride;
        const float4 weight = g8_profile_decode_bf16x4(tap_weights[lane]);
        const float first_activation = first_in
            ? unpack_bf16(joined[offset + uint(xx)]) : 0.0f;
        const float second_activation = second_in
            ? unpack_bf16(joined[offset + uint(xx + 32)]) : 0.0f;
        g8_profile_accumulate_lane<Split>(first_activation, weight, lane,
                                          first_in, second_in, first, second,
                                          first_odd, second_odd, second_activation);
      }
    }
  }

  if (Split) {
    first += first_odd;
    second += second_odd;
  }
  const uint dst = oc_tile * TileChannels * OutputStride
      + (t * Height + uint(y)) * Width + uint(x);
  store_output(output, dst, first);
  store_output(output, dst + 32u, second);
}

template <typename Store, bool Split>
inline void conv3d_g8_profile_packed_tile(device const ushort4* packed_input,
                                          device const ushort4* weights,
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
  const int x = int(column / 32u * 64u + column % 32u);
  const device ushort4* row = weights + oc_tile * K;
  float4 first = 0.0f, second = 0.0f;
  float4 first_odd = 0.0f, second_odd = 0.0f;

  #pragma clang loop unroll(disable)
  for (uint group = 0u; group < Channels / ChannelGroup; ++group) {
    #pragma clang loop unroll(disable)
    for (uint tap = 0u; tap < Taps; ++tap) {
      const int yy = y + int(tap / 3u % 3u) - 1;
      if (yy < 0 || yy >= int(Height)) continue;
      const int xx = x + int(tap % 3u) - 1;
      const bool first_in = xx >= 0;
      const bool second_in = xx + 32 < int(Width);
      const uint src = group * InputStride
          + ((t + tap / 9u) * Height + uint(yy)) * Width;
      const device ushort4* tap_weights = row + (group * Taps + tap) * ChannelGroup;
      float4 first_low = 0.0f, first_high = 0.0f;
      float4 second_low = 0.0f, second_high = 0.0f;

      if (first_in) {
        const uint activation_index = (src + uint(xx)) * 2u;
        first_low = g8_profile_decode_bf16x4(packed_input[activation_index]);
        first_high = g8_profile_decode_bf16x4(packed_input[activation_index + 1u]);
      }
      if (second_in) {
        const uint activation_index = (src + uint(xx + 32)) * 2u;
        second_low = g8_profile_decode_bf16x4(packed_input[activation_index]);
        second_high = g8_profile_decode_bf16x4(packed_input[activation_index + 1u]);
      }

      // Keep the baseline's input-channel FMA order: lanes 0 through 7.
      #pragma clang loop unroll(full)
      for (uint lane = 0u; lane < ChannelGroup; ++lane) {
        const float activation = lane < 4u ? first_low[lane] : first_high[lane - 4u];
        const float second_activation = lane < 4u
            ? second_low[lane] : second_high[lane - 4u];
        const float4 weight = g8_profile_decode_bf16x4(tap_weights[lane]);
        g8_profile_accumulate_lane<Split>(activation, weight, lane,
                                          first_in, second_in, first, second,
                                          first_odd, second_odd, second_activation);
      }
    }
  }

  if (Split) {
    first += first_odd;
    second += second_odd;
  }
  const uint dst = oc_tile * TileChannels * OutputStride
      + (t * Height + uint(y)) * Width + uint(x);
  store_output(output, dst, first);
  store_output(output, dst + 32u, second);
}

kernel void g8_profile_strided_bf16(device const ushort* joined,
                                    device const ushort4* weights,
                                    device ushort* output,
                                    constant uint& first_tile,
                                    uint tid [[thread_position_in_grid]]) {
  conv3d_g8_profile_strided_tile<false>(joined, weights, output, first_tile, tid);
}

kernel void g8_profile_strided_split_bf16(device const ushort* joined,
                                          device const ushort4* weights,
                                          device ushort* output,
                                          constant uint& first_tile,
                                          uint tid [[thread_position_in_grid]]) {
  conv3d_g8_profile_strided_tile<true>(joined, weights, output, first_tile, tid);
}

kernel void g8_profile_packed_bf16(device const ushort4* packed_input,
                                   device const ushort4* weights,
                                   device ushort* output,
                                   constant uint& first_tile,
                                   uint tid [[thread_position_in_grid]]) {
  conv3d_g8_profile_packed_tile<ushort, false>(packed_input, weights, output,
                                               first_tile, tid);
}

kernel void g8_profile_packed_split_bf16(device const ushort4* packed_input,
                                         device const ushort4* weights,
                                         device ushort* output,
                                         constant uint& first_tile,
                                         uint tid [[thread_position_in_grid]]) {
  conv3d_g8_profile_packed_tile<ushort, true>(packed_input, weights, output,
                                             first_tile, tid);
}

kernel void g8_profile_packed_f32(device const ushort4* packed_input,
                                  device const ushort4* weights,
                                  device float* output,
                                  constant uint& first_tile,
                                  uint tid [[thread_position_in_grid]]) {
  conv3d_g8_profile_packed_tile<float, false>(packed_input, weights, output,
                                              first_tile, tid);
}
