#include <metal_stdlib>
using namespace metal;

inline ushort hy_nomas_f32_to_bf16(float value) {
  const uint bits = as_type<uint>(value);
  const uint round_bit = ((bits >> 16) & 1u) + 0x7fffu;
  return ushort((bits + round_bit) >> 16);
}

inline float hy_nomas_bf16_to_f32(ushort value) {
  return as_type<float>(uint(value) << 16);
}

// Slice one segment from channels-first [1,C,T,H,W] storage. A temporal
// slice is not contiguous across channels, so map every fixed-output element.
kernel void hy_nomas_timeline_gather_f32(
    device const float *timeline [[buffer(0)]],
    device float *segment [[buffer(1)]],
    constant uint &start_frame [[buffer(2)]],
    constant uint &total_frames [[buffer(3)]],
    constant uint &segment_frames [[buffer(4)]],
    constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  const uint spatial = height * width;
  const uint frame_area = segment_frames * spatial;
  const uint channel = index / frame_area;
  const uint local_frame = (index / spatial) % segment_frames;
  const uint spatial_index = index % spatial;
  const uint global_frame = start_frame + local_frame;
  const uint source = (channel * total_frames + global_frame) * spatial + spatial_index;
  segment[index] = timeline[source];
}

// Scatter one fixed-size generated segment into the channels-first global
// latent timeline while preserving every other segment's bytes.
kernel void hy_nomas_timeline_scatter_f32(
    device const float *segment [[buffer(0)]],
    device float *timeline [[buffer(1)]],
    constant uint &start_frame [[buffer(2)]],
    constant uint &total_frames [[buffer(3)]],
    constant uint &segment_frames [[buffer(4)]],
    constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  const uint spatial = height * width;
  const uint frame_area = segment_frames * spatial;
  const uint channel = index / frame_area;
  const uint local_frame = (index / spatial) % segment_frames;
  const uint spatial_index = index % spatial;
  const uint global_frame = start_frame + local_frame;
  const uint destination = (channel * total_frames + global_frame) * spatial + spatial_index;
  timeline[destination] = segment[index];
}

// Copy a segment-relative global RoPE slice into fixed [4320,128] workspace.
kernel void hy_nomas_timeline_copy_f32(
    device const float *bank [[buffer(0)]],
    device float *slice [[buffer(1)]],
    constant uint &source_offset [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count)
    slice[index] = bank[source_offset + index];
}

// Match causal_ar.py:2542-2559 for cache_noise_sigma=0.206897: convert the
// BF16 latent and BF16 epsilon to F32, keep both products and the sum as
// separate F32 operations, round the result back to BF16, and expand it into
// the native F32 latent workspace. `cache_timestep` receives sigma*1000.
kernel void hy_nomas_timeline_cache_noise_bf16_f32(
    device float *latent [[buffer(0)]],
    device const ushort *epsilon_bank [[buffer(1)]],
    device float *cache_timestep [[buffer(2)]],
    constant uint &segment_index [[buffer(3)]],
    constant uint &segment_elements [[buffer(4)]],
    constant float &sigma [[buffer(5)]],
    constant float &timestep [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
  if (index >= segment_elements)
    return;

  const uint epsilon_index = segment_index * segment_elements + index;
  const float latent_bf16 = hy_nomas_bf16_to_f32(hy_nomas_f32_to_bf16(latent[index]));
  const float epsilon_bf16 = hy_nomas_bf16_to_f32(epsilon_bank[epsilon_index]);
  const float clean_component = (1.0f - sigma) * latent_bf16;
  const float noise_component = sigma * epsilon_bf16;
  const float noised = clean_component + noise_component;
  latent[index] = hy_nomas_bf16_to_f32(hy_nomas_f32_to_bf16(noised));
  if (index == 0u)
    cache_timestep[0] = timestep;
}
