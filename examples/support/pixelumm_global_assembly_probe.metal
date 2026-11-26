#include <metal_stdlib>
using namespace metal;

inline ushort pga_round_bf16(float value) {
  const uint bits = as_type<uint>(value);
  return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

kernel void pga_fill_window(
    device ushort *window [[buffer(0)]],
    constant uint &fixture_start [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr uint window_frames = 40u;
  constexpr uint frame_pixels = 288u * 512u;
  const uint local_frame = (index / frame_pixels) % window_frames;
  const uint channel = index / (window_frames * frame_pixels);
  const uint spatial = index % frame_pixels;
  const uint mixed = fixture_start * 31u + local_frame * 17u + channel * 11u + (spatial % 7u) * 5u;
  const int numerator = int(mixed % 257u) - 128;
  const float value = float(numerator) / 256.0f;
  window[index] = pga_round_bf16(value);
}

kernel void pga_gather_samples(
    device const float *global_output [[buffer(0)]],
    device const float *first_t30_output [[buffer(1)]],
    device const float *second_t30_output [[buffer(2)]],
    device float *samples [[buffer(3)]],
    constant uint &sample_count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= sample_count)
    return;

  constexpr uint channels = 3u;
  constexpr uint spatial_count = 3u;
  constexpr uint frame_pixels = 288u * 512u;
  constexpr uint global_frames = 240u;
  constexpr uint half_frames = 120u;
  constexpr uint spatial_positions[spatial_count] = {0u, 73727u, 147455u};

  const uint sample_position = index % spatial_count;
  const uint channel = (index / spatial_count) % channels;
  const uint frame = index / (spatial_count * channels);
  const uint spatial = spatial_positions[sample_position];
  const uint global_index = (channel * global_frames + frame) * frame_pixels + spatial;
  const uint local_frame = frame < half_frames ? frame : frame - half_frames;
  const uint local_index = (channel * half_frames + local_frame) * frame_pixels + spatial;

  samples[index * 2u] = global_output[global_index];
  samples[index * 2u + 1u] = frame < half_frames
      ? first_t30_output[local_index]
      : second_t30_output[local_index];
}
