#include <metal_stdlib>
using namespace metal;

kernel void pixelumm_continuation_join_latent(
    device const float *previous [[buffer(0)]],
    device const float *current [[buffer(1)]],
    device float *joined [[buffer(2)]],
    constant uint &previous_frames [[buffer(3)]],
    constant uint &current_frames [[buffer(4)]],
    constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr ulong context_frames = 5ul;
  const ulong spatial = ulong(height) * ulong(width);
  const ulong joined_frames = ulong(current_frames) + context_frames;
  const ulong joined_channel_stride = joined_frames * spatial;
  const ulong channel = ulong(index) / joined_channel_stride;
  const ulong within_channel = ulong(index) % joined_channel_stride;
  const ulong frame = within_channel / spatial;
  const ulong cell = within_channel % spatial;

  ulong source_index;
  if (frame < context_frames) {
    const ulong source_frame = ulong(previous_frames) - context_frames + frame;
    source_index = channel * ulong(previous_frames) * spatial + source_frame * spatial + cell;
    joined[index] = previous[source_index];
  } else {
    const ulong source_frame = frame - context_frames;
    source_index = channel * ulong(current_frames) * spatial + source_frame * spatial + cell;
    joined[index] = current[source_index];
  }
}

kernel void pixelumm_continuation_join_noise(
    device const float *previous [[buffer(0)]],
    device const float *current [[buffer(1)]],
    device float *joined [[buffer(2)]],
    constant uint &previous_frames [[buffer(3)]],
    constant uint &current_frames [[buffer(4)]],
    constant uint &joined_frames [[buffer(5)]],
    constant uint &height [[buffer(6)]],
    constant uint &width [[buffer(7)]],
    constant uint &count [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr ulong context_frames = 20ul;
  const ulong spatial = ulong(height) * ulong(width);
  const ulong joined_channel_stride = ulong(joined_frames) * spatial;
  const ulong channel = ulong(index) / joined_channel_stride;
  const ulong within_channel = ulong(index) % joined_channel_stride;
  const ulong frame = within_channel / spatial;
  const ulong cell = within_channel % spatial;

  ulong source_index;
  if (frame < context_frames) {
    const ulong source_frame = ulong(previous_frames) - context_frames + frame;
    source_index = channel * ulong(previous_frames) * spatial + source_frame * spatial + cell;
    joined[index] = previous[source_index];
  } else {
    const ulong source_frame = frame - context_frames;
    source_index = channel * ulong(current_frames) * spatial + source_frame * spatial + cell;
    joined[index] = current[source_index];
  }
}

kernel void pixelumm_continuation_crop_display(
    device const float *full [[buffer(0)]],
    device float *current [[buffer(1)]],
    constant uint &current_frames [[buffer(2)]],
    constant uint &full_frames [[buffer(3)]],
    constant uint &height [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr ulong context_frames = 20ul;
  const ulong spatial = ulong(height) * ulong(width);
  const ulong current_channel_stride = ulong(current_frames) * spatial;
  const ulong channel = ulong(index) / current_channel_stride;
  const ulong within_channel = ulong(index) % current_channel_stride;
  const ulong frame = within_channel / spatial;
  const ulong cell = within_channel % spatial;
  const ulong source_index =
      channel * ulong(full_frames) * spatial + (frame + context_frames) * spatial + cell;
  current[index] = full[source_index];
}
