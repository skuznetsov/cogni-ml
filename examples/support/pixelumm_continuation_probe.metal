#include <metal_stdlib>
using namespace metal;

inline ushort pcp_round_bf16(float value) {
  const uint bits = as_type<uint>(value);
  return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

kernel void pcp_fill_bf16_window(
    device ushort *window [[buffer(0)]],
    constant uint &global_start_latent [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &height [[buffer(3)]],
    constant uint &width [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr uint window_frames = 40u;
  const uint frame_pixels = height * width;
  const uint local_frame = (index / frame_pixels) % window_frames;
  const uint channel = index / (window_frames * frame_pixels);
  const uint spatial = index % frame_pixels;
  const uint mixed = global_start_latent * 31u + local_frame * 17u + channel * 11u + (spatial % 7u) * 5u;
  const int numerator = int(mixed % 257u) - 128;
  const float value = float(numerator) / 256.0f;
  window[index] = pcp_round_bf16(value);
}
