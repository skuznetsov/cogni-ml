#include <metal_stdlib>
using namespace metal;

inline ushort pixelumm_f32_to_bf16(float value) {
  const uint bits = as_type<uint>(value);
  const uint round_bit = ((bits >> 16) & 1u) + 0x7fffu;
  return ushort((bits + round_bit) >> 16);
}

inline float pixelumm_bf16_to_f32(ushort value) {
  return as_type<float>(uint(value) << 16);
}

// Match the upstream conversion of the F32 checkpoint tensors to the BF16
// autocast dtype before the native MPS path expands weights back to F32.
kernel void pixelumm_generation_round_f32_to_bf16(
    device const float *source [[buffer(0)]],
    device ushort *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
  if (index < count)
    destination[index] = pixelumm_f32_to_bf16(source[index]);
}

// Extract one 4x16x16 RGB tubelet from full-clip NCTHW F32 noise, in the
// upstream patchify feature order (temporal, patch-y, patch-x, channel).
// The F32 values are round-tripped through BF16 because the source casts xt
// before patchify and runs the stem under BF16 autocast.
kernel void pixelumm_generation_patchify(
    device const float *noise [[buffer(0)]],
    device float *patches [[buffer(1)]],
    constant uint &window_start [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr uint token_dim = 3072;
  constexpr uint grid_h = 18;
  constexpr uint grid_w = 32;
  constexpr uint pixel_frames = 120;
  constexpr uint pixel_h = 288;
  constexpr uint pixel_w = 512;
  const uint token = index / token_dim;
  const uint feature = index % token_dim;
  const uint token_w = token % grid_w;
  const uint token_h = (token / grid_w) % grid_h;
  const uint token_t = token / (grid_h * grid_w);
  const uint channel = feature % 3u;
  const uint patch_w = (feature / 3u) % 16u;
  const uint patch_h = (feature / (3u * 16u)) % 16u;
  const uint patch_t = feature / (3u * 16u * 16u);
  const uint frame = window_start + token_t * 4u + patch_t;
  const uint y = token_h * 16u + patch_h;
  const uint x = token_w * 16u + patch_w;
  const uint source = ((channel * pixel_frames + frame) * pixel_h + y) * pixel_w + x;
  patches[index] = pixelumm_bf16_to_f32(pixelumm_f32_to_bf16(noise[source]));
}

// Gather one 32-channel latent token per 2x2 decoder-token group.
kernel void pixelumm_generation_gather_condition(
    device const float *latent [[buffer(0)]],
    device float *condition [[buffer(1)]],
    constant uint &window_start [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;

  constexpr uint latent_channels = 32;
  constexpr uint latent_frames = 30;
  constexpr uint latent_h = 9;
  constexpr uint latent_w = 16;
  const uint row = index / latent_channels;
  const uint channel = index % latent_channels;
  const uint token_w = row % latent_w;
  const uint token_h = (row / latent_w) % latent_h;
  const uint token_t = row / (latent_h * latent_w);
  const uint frame = window_start + token_t;
  const uint source = ((channel * latent_frames + frame) * latent_h + token_h) * latent_w + token_w;
  condition[index] = pixelumm_bf16_to_f32(pixelumm_f32_to_bf16(latent[source]));
}

// Convert the BF16-rounded embed projection to tile-major order. permutation
// contains source THW row indices, matching the block plan's layout.
kernel void pixelumm_generation_store_embed(
    device const float *projection [[buffer(0)]],
    device const int *permutation [[buffer(1)]],
    device ushort *tokens [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  constexpr uint dim = 4096;
  const uint tile_row = index / dim;
  const uint column = index % dim;
  const uint source_row = uint(permutation[tile_row]);
  tokens[index] = pixelumm_f32_to_bf16(projection[source_row * dim + column]);
}

// Add the BF16 condition projection to BF16 embed tokens after reproducing
// the upstream [t,h,sub-h,w,sub-w,hidden] condition-token reshape.
kernel void pixelumm_generation_add_condition(
    device const float *condition_projection [[buffer(0)]],
    device const int *permutation [[buffer(1)]],
    device ushort *tokens [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  constexpr uint dim = 4096;
  constexpr uint hidden_per_latent = 4u * dim;
  constexpr uint token_h = 18;
  constexpr uint token_w = 32;
  const uint tile_row = index / dim;
  const uint column = index % dim;
  const uint source_row = uint(permutation[tile_row]);
  const uint w = source_row % token_w;
  const uint h = (source_row / token_w) % token_h;
  const uint t = source_row / (token_h * token_w);
  const uint latent_row = (t * 9u + h / 2u) * 16u + w / 2u;
  const uint subtoken = (h % 2u) * 2u + (w % 2u);
  const uint condition_index = latent_row * hidden_per_latent + subtoken * dim + column;
  const float embed = pixelumm_bf16_to_f32(tokens[index]);
  const float condition = pixelumm_bf16_to_f32(
      pixelumm_f32_to_bf16(condition_projection[condition_index]));
  tokens[index] = pixelumm_f32_to_bf16(embed + condition);
}

#pragma METAL fp contract(off)

kernel void pixelumm_generation_blend_weights(
    device ushort *weights [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= 20u)
    return;
  const float position = float(index + 1u);
  const float angle = (3.14159265358979323846f * position) / 21.0f;
  const float weight = 0.5f - 0.5f * cos(angle);
  weights[index] = pixelumm_f32_to_bf16(weight);
}

// Append one full 40-frame decoder window into the BF16-equivalent F32 clip.
// Each source blend operation is separately rounded to BF16 like the upstream
// temporal blender; the output buffer stores those BF16 values expanded to F32.
kernel void pixelumm_generation_merge_window(
    device const ushort *part [[buffer(0)]],
    device float *result [[buffer(1)]],
    device const ushort *weights [[buffer(2)]],
    constant uint &start_frame [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  constexpr uint channels = 3;
  constexpr uint output_frames = 120;
  constexpr uint window_frames = 40;
  constexpr uint overlap_frames = 20;
  constexpr uint height = 288;
  constexpr uint width = 512;
  constexpr uint frame_pixels = height * width;
  constexpr uint channel_pixels = output_frames * frame_pixels;
  const uint local_frame = (index / frame_pixels) % window_frames;
  const uint channel = index / (window_frames * frame_pixels);
  const uint spatial = index % frame_pixels;
  const uint global_frame = start_frame + local_frame;
  const uint destination = channel * channel_pixels + global_frame * frame_pixels + spatial;
  const float current = pixelumm_bf16_to_f32(part[index]);
  if (start_frame != 0u && local_frame < overlap_frames) {
    const float weight = pixelumm_bf16_to_f32(weights[local_frame]);
    const float inverse = pixelumm_bf16_to_f32(pixelumm_f32_to_bf16(1.0f - weight));
    const float previous = result[destination];
    const float previous_term = pixelumm_bf16_to_f32(
        pixelumm_f32_to_bf16(previous * inverse));
    const float current_term = pixelumm_bf16_to_f32(
        pixelumm_f32_to_bf16(current * weight));
    result[destination] = pixelumm_bf16_to_f32(
        pixelumm_f32_to_bf16(previous_term + current_term));
  } else {
    result[destination] = current;
  }
}

kernel void pixelumm_generation_finalize_display_range(
    device float *video [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  if (index >= count)
    return;
  // The pinned decoder returns BF16 and `_postprocess_vae_video` performs a
  // rounded BF16 multiply, then a rounded BF16 add, before clamping. Keep the
  // final F32 allocation while reproducing those dtype boundaries exactly.
  const float scaled = pixelumm_bf16_to_f32(
      pixelumm_f32_to_bf16(video[index] * 0.5f));
  const float shifted = pixelumm_bf16_to_f32(
      pixelumm_f32_to_bf16(scaled + 0.5f));
  video[index] = clamp(shifted, 0.0f, 1.0f);
}
