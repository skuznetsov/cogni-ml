#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

inline float video_linear_tiled_load(device const float *f32,
                                     device const ushort *bf16,
                                     ulong index, uint is_bf16) {
  if (is_bf16)
    return as_type<float>(uint(bf16[index]) << 16);
  return f32[index];
}

// Opt-in dense projection adapted from dr3e_linear_mma. One SIMD group
// computes an 8x8 output tile, traversing K in ascending 8-wide fragments.
// Every operand and accumulator is F32; BF16 storage is expanded on device.
// row ranges map to absolute rows, so adjacent range dispatches do not alter a
// row's matrix operation order or overlap output elements.
kernel void video_linear_tiled_f32_accum(device const float *input_f32 [[buffer(0)]],
                               device const ushort *input_bf16 [[buffer(1)]],
                               device const float *weight_f32 [[buffer(2)]],
                               device const ushort *weight_bf16 [[buffer(3)]],
                               device const float *bias_f32 [[buffer(4)]],
                               device const ushort *bias_bf16 [[buffer(5)]],
                               device float *output [[buffer(6)]],
                               constant uint *p [[buffer(7)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = p[0], channels = p[1], out_channels = p[2];
  const uint first_row = p[3], row_count = p[4];
  const uint input_is_bf16 = p[5], weight_is_bf16 = p[6], bias_is_bf16 = p[7];
  const uint tiles_n = (out_channels + 7) / 8;
  const uint tile_m = group / tiles_n;
  const uint tile_n = group % tiles_n;
  const uint row_begin = first_row + tile_m * 8;
  const uint row_end = first_row + row_count;
  threadgroup float aa[64], bb[64], cc[64];
  simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8, 8>(0.f);

  for (uint k0 = 0; k0 < channels; k0 += 8) {
    for (uint index = lane; index < 64; index += 32) {
      const uint row = row_begin + index / 8;
      uint channel = k0 + index % 8;
      const bool input_valid = row < rows && row < row_end && channel < channels;
      aa[index] = input_valid
                      ? video_linear_tiled_load(input_f32, input_bf16,
                            ulong(row) * channels + channel, input_is_bf16)
                      : 0.f;

      channel = k0 + index / 8;
      const uint out_channel = tile_n * 8 + index % 8;
      const bool weight_valid = out_channel < out_channels && channel < channels;
      bb[index] = weight_valid
                      ? video_linear_tiled_load(weight_f32, weight_bf16,
                            ulong(out_channel) * channels + channel, weight_is_bf16)
                      : 0.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 left, right;
    simdgroup_load(left, aa, 8);
    simdgroup_load(right, bb, 8);
    simdgroup_multiply_accumulate(accumulator, left, right, accumulator);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(accumulator, cc, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint j = 0; j < 2; ++j) {
    const uint index = lane + j * 32;
    const uint row = row_begin + index / 8;
    const uint out_channel = tile_n * 8 + index % 8;
    if (row < rows && row < row_end && out_channel < out_channels) {
      const float b = video_linear_tiled_load(bias_f32, bias_bf16,
                                               out_channel, bias_is_bf16);
      output[ulong(row) * out_channels + out_channel] = cc[index] + b;
    }
  }
}
