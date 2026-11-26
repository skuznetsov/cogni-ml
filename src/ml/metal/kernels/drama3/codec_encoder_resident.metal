#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Adapted from MLX 0.31.1 Metal erf/expm1, row reductions and Steel attention.
// Apple portions are under the MLX MIT license; the expm1
// polynomial is derived from Norbert Juffa's implementation, whose complete
// license is reproduced below. The upstream erf header cites the source at:
// https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff#answer-35148199
//
// Copyright © 2023-2025 Apple Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// Copyright (c) 2015-2023 Norbert Juffa
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

inline float dr3e_expm1f_scaled_unchecked(float a, float b) {
  float f, j, r, s, t, u, v, x, y;
  int i;
  j = fma(1.442695f, a, 12582912.0f);
  j = j - 12582912.0f;
  i = (int)j;
  f = fma(j, -6.93145752e-1f, a);
  s = f * f;
  if (a == 0.0f)
    s = a;
  r = 1.97350979e-4f;
  r = fma(r, f, 1.39309070e-3f);
  r = fma(r, f, 8.33343994e-3f);
  r = fma(r, f, 4.16668020e-2f);
  r = fma(r, f, 1.66666716e-1f);
  r = fma(r, f, 4.99999970e-1f);
  u = (j == 1.0f) ? (f + 0.5f) : f;
  v = fma(r, s, u);
  s = 0.5f * b;
  t = ldexp(s, i);
  y = t - s;
  x = (t - y) - s;
  r = fma(v, t, x) + y;
  r = r + r;
  if (j == 0.0f)
    r = v;
  if (j == 1.0f)
    r = v + v;
  return r;
}

inline float dr3e_expm1f(float a) {
  float r = dr3e_expm1f_scaled_unchecked(a, 1.0f);
  if (abs(a - 1.0f) > 88.0f) {
    r = pow(2.0f, a);
    r = fma(r, r, -1.0f);
  }
  return r;
}

inline float dr3e_erf(float a) {
  float r, s, t, u;
  t = abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    r = fma(-1.72853470e-5f, t, 3.83197126e-4f);
    u = fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = fma(r, s, u);
    r = fma(r, t, -1.06777877e-1f);
    r = fma(r, t, -6.34846687e-1f);
    r = fma(r, t, -1.28717512e-1f);
    r = fma(r, t, -t);
    r = -dr3e_expm1f(r);
    r = copysign(r, a);
  } else {
    r = -5.96761703e-4f;
    r = fma(r, s, 4.99119423e-3f);
    r = fma(r, s, -2.67681349e-2f);
    r = fma(r, s, 1.12819925e-1f);
    r = fma(r, s, -3.76125336e-1f);
    r = fma(r, s, 1.28379166e-1f);
    r = fma(r, a, a);
  }
  return r;
}

// Neumaier compensation keeps reductions in F32 while recovering low-order
// terms when a large partial sum is followed by a smaller contribution.
inline void dr3e_neumaier_add(float value, thread float &sum,
                              thread float &correction) {
  float next = sum + value;
  if (abs(sum) >= abs(value))
    correction += (sum - next) + value;
  else
    correction += (value - next) + sum;
  sum = next;
}

kernel void dr3e_wn_norms(device const float *weight [[buffer(0)]],
                          device float *norms [[buffer(1)]],
                          constant ulong *p [[buffer(2)]],
                          uint channel [[threadgroup_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint group [[simdgroup_index_in_threadgroup]]) {
  if (channel >= p[0])
    return;
  uint width = uint(p[1]);
  float exponent = as_type<float>(uint(p[2]));
  ulong base = ulong(channel) * p[1];
  float sum = 0.0f;
  if (width <= 64) {
    // MLX row_reduce_small visits a short contiguous row serially.
    for (uint i = 0; i < width; ++i)
      sum = precise::pow(weight[base + i], exponent) + sum;
    norms[channel] = precise::sqrt(sum);
    return;
  }
  // Match MLX's N_READS=4 row reduction, including its SIMD reduction tree.
  // Power must remain a precise runtime operation: x*x and fast pow differ
  // from upstream F32 power even when the exponent is exactly two.
  for (uint offset = lid * 4; offset < width; offset += threads * 4)
    for (uint j = 0; j < 4 && offset + j < width; ++j)
      sum = precise::pow(weight[base + offset + j], exponent) + sum;
  threadgroup float partials[32];
  sum = simd_sum(sum);
  if (lane == 0)
    partials[group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threads > 32)
    sum = simd_sum(lid < threads / 32 ? partials[lid] : 0.0f);
  if (lid == 0)
    norms[channel] = precise::sqrt(sum);
}

kernel void dr3e_wn_apply(device const float *weight [[buffer(0)]],
                          device const float *gain [[buffer(1)]],
                          device const float *norms [[buffer(2)]],
                          device float *effective [[buffer(3)]],
                          constant ulong *p [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p[0])
    return;
  uint channel = i / uint(p[1]);
  // Match Fish's materialization order: (gain * weight_v) / L2(weight_v).
  effective[i] = (gain[channel] * weight[i]) / norms[channel];
}

kernel void dr3e_layer_norm(device const float *input [[buffer(0)]],
                            device const float *gamma [[buffer(1)]],
                            device const float *beta [[buffer(2)]],
                            device float *output [[buffer(3)]],
                            constant ulong *p [[buffer(4)]],
                            uint row [[thread_position_in_grid]]) {
  uint rows = uint(p[0]), width = uint(p[1]);
  if (row >= rows)
    return;
  ulong base = ulong(row) * width;
  float mean = 0.0f, mean_correction = 0.0f;
  for (uint c = 0; c < width; ++c)
    dr3e_neumaier_add(input[base + c], mean, mean_correction);
  mean = (mean + mean_correction) / float(width);

  float variance = 0.0f, variance_correction = 0.0f;
  for (uint c = 0; c < width; ++c) {
    float centered = input[base + c] - mean;
    dr3e_neumaier_add(centered * centered, variance, variance_correction);
  }
  variance = (variance + variance_correction) / float(width);
  float inverse = precise::rsqrt(variance + as_type<float>(uint(p[2])));
  for (uint c = 0; c < width; ++c)
    output[base + c] = (input[base + c] - mean) * inverse * gamma[c] + beta[c];
}

kernel void dr3e_pad_audio(device const float *audio [[buffer(0)]],
                           device float *output [[buffer(1)]],
                           constant ulong *p [[buffer(2)]],
                           uint i [[thread_position_in_grid]]) {
  if (i >= p[1])
    return;
  output[i] = i < p[0] ? audio[i] : 0.0f;
}

kernel void dr3e_snake(device const float *input [[buffer(0)]],
                       device const float *alpha [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       constant ulong *p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  uint channels = uint(p[1]);
  if (i >= p[0] * p[1])
    return;
  float value = input[i];
  float a = alpha[i % channels];
  float scaled = a * value;
  // MLX's unary Sin explicitly selects metal::precise::sin, including when
  // the surrounding library uses safe math mode.
  float sine = precise::sin(scaled);
  float square = precise::pow(sine, as_type<float>(uint(p[2])));
  float reciprocal = 1.0f / (a + 1.0e-9f);
  output[i] = value + reciprocal * square;
}

kernel void dr3e_gelu(device const float *input [[buffer(0)]],
                      device float *output [[buffer(1)]],
                      constant ulong *p [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
  if (i >= p[0])
    return;
  float value = input[i];
  float erf_arg = value / 1.4142135623730951f;
  output[i] = (value * (1.0f + dr3e_erf(erf_arg))) / 2.0f;
}

// Plain stride==kernel Conv1d with channels divisible by 16. MLX Steel walks
// all filter taps within each BK16 input-channel block, accumulating two
// ascending 8-wide F32 simdgroup fragments per block/tap. Keep that traversal:
// flattening the entire [tap,channel] dimension changes F32 rounding.
kernel void dr3e_conv1d_mma_bk16(device const float *input [[buffer(0)]],
                                 device const float *weight [[buffer(1)]],
                                 device const float *bias [[buffer(2)]],
                                 device float *output [[buffer(3)]],
                                 constant ulong *p [[buffer(4)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  uint input_rows = uint(p[0]), rows = uint(p[1]);
  uint channels = uint(p[2]), out_channels = uint(p[3]);
  uint kernel_size = uint(p[4]), stride = uint(p[5]), dilation = uint(p[6]);
  int left_pad = int((kernel_size - 1) * dilation + 1 - stride);
  uint tiles_n = (out_channels + 7) / 8;
  uint tile_m = group / tiles_n, tile_n = group % tiles_n;
  uint reduction = channels * kernel_size;
  threadgroup float aa[64], bb[64], cc[64];
  simdgroup_float8x8 accumulator =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  for (uint k0 = 0; k0 < reduction; k0 += 8) {
    for (uint index = lane; index < 64; index += 32) {
      uint local_row = index / 8, local_col = index % 8;
      uint row = tile_m * 8 + local_row;
      uint flat_k = k0 + local_col;
      uint tap =
          channels < 16 ? flat_k / channels : (flat_k / 16) % kernel_size;
      uint channel = channels < 16
                         ? flat_k % channels
                         : (flat_k / (16 * kernel_size)) * 16 + flat_k % 16;
      int t = int(row * stride) - left_pad + int(tap * dilation);
      aa[index] =
          row < rows && flat_k < reduction && t >= 0 && t < int(input_rows)
              ? input[ulong(t) * channels + channel]
              : 0.0f;
      flat_k = k0 + local_row;
      tap = channels < 16 ? flat_k / channels : (flat_k / 16) % kernel_size;
      channel = channels < 16
                    ? flat_k % channels
                    : (flat_k / (16 * kernel_size)) * 16 + flat_k % 16;
      uint out_channel = tile_n * 8 + local_col;
      bb[index] =
          flat_k < reduction && out_channel < out_channels
              ? weight[(ulong(out_channel) * channels + channel) * kernel_size +
                       tap]
              : 0.0f;
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
  for (uint index = lane; index < 64; index += 32) {
    uint row = tile_m * 8 + index / 8;
    uint out_channel = tile_n * 8 + index % 8;
    if (row < rows && out_channel < out_channels)
      output[ulong(row) * out_channels + out_channel] =
          cc[index] + bias[out_channel];
  }
}

// PyTorch-layout [out,in/groups,k] causal Conv1d. The reference left-pads by
// effective_kernel-stride and adds only enough right zeros to produce ceil(N/
// stride) outputs. Out-of-range right taps are therefore implicit zeros here.
kernel void dr3e_conv1d(device const float *input [[buffer(0)]],
                        device const float *weight [[buffer(1)]],
                        device const float *bias [[buffer(2)]],
                        device float *output [[buffer(3)]],
                        constant ulong *p [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
  uint input_rows = uint(p[0]), output_rows = uint(p[1]);
  uint in_channels = uint(p[2]), out_channels = uint(p[3]);
  uint kernel_size = uint(p[4]), stride = uint(p[5]);
  uint groups = uint(p[6]), dilation = uint(p[7]);
  if (i >= output_rows * out_channels)
    return;
  uint t = i / out_channels, out_channel = i % out_channels;
  uint in_per_group = in_channels / groups,
       out_per_group = out_channels / groups;
  uint group = out_channel / out_per_group;
  int left_pad = int((kernel_size - 1) * dilation + 1 - stride);
  float sum = 0.0f, correction = 0.0f;
  // Match the [out,k,in] reduction order after upstream transposes the
  // checkpoint's PyTorch [out,in,k] weights for MLX Conv1d.
  for (uint tap = 0; tap < kernel_size; ++tap) {
    for (uint in_local = 0; in_local < in_per_group; ++in_local) {
      uint in_channel = group * in_per_group + in_local;
      int input_t = int(t * stride) - left_pad + int(tap * dilation);
      if (input_t < 0 || input_t >= int(input_rows))
        continue;
      ulong wi =
          (ulong(out_channel) * in_per_group + in_local) * kernel_size + tap;
      dr3e_neumaier_add(input[ulong(input_t) * in_channels + in_channel] *
                            weight[wi],
                        sum, correction);
    }
  }
  // Fish adds the Conv1d bias as a separate post-convolution operation.
  output[i] = (sum + correction) + bias[out_channel];
}

kernel void dr3e_rope_loaded(device float *x [[buffer(0)]],
                             device const float *frequencies [[buffer(1)]],
                             constant ulong *p [[buffer(2)]],
                             uint id [[thread_position_in_grid]]) {
  uint rows = uint(p[0]), heads = uint(p[1]), dim = uint(p[2]);
  uint pairs = dim / 2;
  if (id >= rows * heads * pairs)
    return;
  uint t = id / (heads * pairs), pair = id % pairs;
  uint offset = t * heads * dim + (id / pairs) % heads * dim + pair * 2;
  ulong frequency_offset = (ulong(t) * pairs + pair) * 2;
  float c = frequencies[frequency_offset];
  float s = frequencies[frequency_offset + 1];
  float left = x[offset], right = x[offset + 1];
  x[offset] = left * c - right * s;
  x[offset + 1] = right * c + left * s;
}

kernel void dr3e_l2_normalize(device const float *input [[buffer(0)]],
                              device float *output [[buffer(1)]],
                              constant ulong *p [[buffer(2)]],
                              uint row [[thread_position_in_grid]]) {
  uint rows = uint(p[0]), width = uint(p[1]);
  if (row >= rows)
    return;
  ulong base = ulong(row) * width;
  float norm_sq = 0.0f, norm_correction = 0.0f;
  for (uint d = 0; d < width; ++d) {
    float value = input[base + d];
    dr3e_neumaier_add(value * value, norm_sq, norm_correction);
  }
  float denom = max(precise::sqrt(norm_sq + norm_correction), 1.0e-12f);
  for (uint d = 0; d < width; ++d)
    output[base + d] = input[base + d] / denom;
}

kernel void dr3e_vq_distance(device const float *input [[buffer(0)]],
                             device const float *codebook [[buffer(1)]],
                             device float *distances [[buffer(2)]],
                             constant ulong *p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  uint rows = uint(p[0]), vocab = uint(p[1]), width = uint(p[2]);
  if (i >= rows * vocab)
    return;
  uint row = i / vocab, code = i % vocab;
  ulong input_base = ulong(row) * width, code_base = ulong(code) * width;
  float input_sq = 0.0f, input_correction = 0.0f;
  float dot = 0.0f, dot_correction = 0.0f;
  float code_sq = 0.0f, code_correction = 0.0f;
  for (uint d = 0; d < width; ++d) {
    float x = input[input_base + d], c = codebook[code_base + d];
    dr3e_neumaier_add(x * x, input_sq, input_correction);
    dr3e_neumaier_add(x * c, dot, dot_correction);
    dr3e_neumaier_add(c * c, code_sq, code_correction);
  }
  distances[i] = (input_sq + input_correction) - 2.0f * (dot + dot_correction) +
                 (code_sq + code_correction);
}

// Strict less-than retains the first codebook index for exact ties, matching
// argmax(-distance). The next-smallest value is exposed as a diagnostic margin.
kernel void dr3e_vq_argmin(device const float *distances [[buffer(0)]],
                           device float *indices [[buffer(1)]],
                           device float *margin [[buffer(2)]],
                           constant ulong *p [[buffer(3)]],
                           uint row [[thread_position_in_grid]]) {
  uint rows = uint(p[0]), vocab = uint(p[1]);
  if (row >= rows)
    return;
  ulong base = ulong(row) * vocab;
  float best = distances[base], second = INFINITY;
  uint selected = 0;
  for (uint code = 1; code < vocab; ++code) {
    float candidate = distances[base + code];
    if (candidate < best) {
      second = best;
      best = candidate;
      selected = code;
    } else if (candidate < second) {
      second = candidate;
    }
  }
  indices[row] = float(selected);
  margin[row] = second - best;
}

kernel void dr3e_vq_gather(device const float *indices [[buffer(0)]],
                           device const float *codebook [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           constant ulong *p [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
  uint rows = uint(p[0]), vocab = uint(p[1]), width = uint(p[2]);
  if (i >= rows * width)
    return;
  uint row = i / width, dim = i % width;
  uint selected = uint(indices[row]);
  if (selected >= vocab) {
    output[i] = NAN;
    return;
  }
  output[i] = codebook[ulong(selected) * width + dim];
}

// Fish's VectorQuantize forward materializes z_e + (z_q - z_e) before its
// output projection. Preserve that F32 operation order instead of replacing
// the expression with its algebraic identity, which can change rounding.
kernel void dr3e_vq_identity(device const float *projected [[buffer(0)]],
                             device const float *embedding [[buffer(1)]],
                             device float *output [[buffer(2)]],
                             constant ulong *p [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
  if (i < p[0])
    output[i] = projected[i] + (embedding[i] - projected[i]);
}

kernel void dr3e_subtract(device const float *left [[buffer(0)]],
                          device const float *right [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant ulong *p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < p[0])
    output[i] = left[i] - right[i];
}

kernel void dr3e_store_codebook(device const float *indices [[buffer(0)]],
                                device float *codes [[buffer(1)]],
                                constant ulong *p [[buffer(2)]],
                                uint row [[thread_position_in_grid]]) {
  uint frames = uint(p[0]), book = uint(p[1]);
  if (row < frames)
    codes[ulong(book) * frames + row] = indices[row];
}

kernel void dr3e_transpose_codes(device const float *code_major [[buffer(0)]],
                                 device float *time_major [[buffer(1)]],
                                 constant ulong *p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  uint frames = uint(p[0]), books = uint(p[1]);
  if (i >= frames * books)
    return;
  uint frame = i / books, book = i % books;
  time_major[i] = code_major[ulong(book) * frames + frame];
}

// F32 dense projection: ascending eight-wide fragments within each upstream
// split-K partition, then ascending partition sums and a separate bias add.
kernel void dr3e_linear_mma(device const float *input [[buffer(0)]],
                            device const float *weight [[buffer(1)]],
                            device const float *bias [[buffer(2)]],
                            device float *output [[buffer(3)]],
                            constant ulong *p [[buffer(4)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
  uint rows = uint(p[0]), channels = uint(p[1]), out_channels = uint(p[2]);
  uint partitions = uint(p[4]), partition_size = uint(p[5]);
  uint tiles_n = (out_channels + 7) / 8, tile_m = group / tiles_n,
       tile_n = group % tiles_n;
  threadgroup float aa[64], bb[64], cc[64];
  float sums[2] = {0.f, 0.f};
  for (uint part = 0; part < partitions; ++part) {
    simdgroup_float8x8 accumulator =
        make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    uint begin = part * partition_size;
    uint end = part + 1 == partitions ? channels : begin + partition_size;
    for (uint k0 = begin; k0 < end; k0 += 8) {
      for (uint index = lane; index < 64; index += 32) {
        uint row = tile_m * 8 + index / 8, ch = k0 + index % 8;
        aa[index] =
            row < rows && ch < end ? input[ulong(row) * channels + ch] : 0.f;
        ch = k0 + index / 8;
        uint out_channel = tile_n * 8 + index % 8;
        bb[index] = out_channel < out_channels && ch < end
                        ? weight[ulong(out_channel) * channels + ch]
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
    sums[0] += cc[lane];
    sums[1] += cc[lane + 32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint j = 0; j < 2; ++j) {
    uint index = lane + j * 32;
    uint row = tile_m * 8 + index / 8, out_channel = tile_n * 8 + index % 8;
    if (row < rows && out_channel < out_channels)
      output[ulong(row) * out_channels + out_channel] =
          p[3] ? sums[j] + bias[out_channel] : sums[j];
  }
}
kernel void dr3e_rms(device const float *input [[buffer(0)]],
                     device const float *weight [[buffer(1)]],
                     device float *output [[buffer(2)]],
                     constant ulong *p [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint group [[simdgroup_index_in_threadgroup]]) {
  uint width = uint(p[1]);
  ulong base = ulong(row) * width;
  float sum = 0.f;
  if (width <= 64) {
    for (uint c = 0; c < width; ++c)
      sum = input[base + c] * input[base + c] + sum;
    float inv = precise::rsqrt(sum / float(width) + as_type<float>(uint(p[2])));
    for (uint c = 0; c < width; ++c)
      output[base + c] = input[base + c] * inv * weight[c];
    return;
  }
  for (uint off = lid * 4; off < width; off += threads * 4)
    for (uint j = 0; j < 4 && off + j < width; ++j)
      sum = input[base + off + j] * input[base + off + j] + sum;
  threadgroup float partials[32];
  sum = simd_sum(sum);
  if (lane == 0)
    partials[group] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (threads > 32)
    sum = simd_sum(lid < threads / 32 ? partials[lid] : 0.f);
  if (lid == 0)
    partials[0] =
        precise::rsqrt(sum / float(width) + as_type<float>(uint(p[2])));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = lid; c < width; c += threads)
    output[base + c] = input[base + c] * partials[0] * weight[c];
}
kernel void dr3e_sigmoid_gate(device const float *a [[buffer(0)]],
                              device const float *b [[buffer(1)]],
                              device float *y [[buffer(2)]],
                              constant ulong *p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= p[0])
    return;
  float value = a[i];
  float z = 1.f / (1.f + precise::exp(abs(value)));
  float sigmoid = value < 0.f ? z : 1.f - z;
  y[i] = (value * sigmoid) * b[i];
}

// Encoder-local F32 SDPA, same eight-by-eight fragments, online softmax,
// and ascending BK32 blocks as MLX 0.31.1 Steel attention.
template <uint BK>
inline void dr3e_attention_body(device const float *q, device const float *k,
                                device const float *v, device float *out,
                                constant ulong *p, uint tile, uint lane) {
  const uint rows = uint(p[0]), heads = uint(p[1]), dim = 64,
             channels = heads * dim;
  const uint head = tile % heads, qbase = (tile / heads) * 8;
  const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
  const uint fn = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  const uint qr = qbase + fm;
  const float scale = as_type<float>(uint(p[3])) * M_LOG2E_F;
  float2 acco[8];
  for (uint d = 0; d < 8; ++d)
    acco[d] = float2(0.0f);
  float maximum = -MAXFLOAT, norm = 0.0f;
  for (uint kb = 0; kb < rows; kb += BK) {
    float2 prob[BK / 8];
    float new_max = maximum;
    for (uint sk = 0; sk < BK / 8; ++sk) {
      simdgroup_float8x8 score(0.0f);
      for (uint dd = 0; dd < 8; ++dd) {
        simdgroup_float8x8 aq, bk;
        for (uint j = 0; j < 2; ++j) {
          uint qc = dd * 8 + fn + j, kr = kb + sk * 8 + fn + j;
          aq.thread_elements()[j] =
              qr < rows ? q[ulong(qr) * channels + head * dim + qc] : 0.0f;
          bk.thread_elements()[j] =
              kr < rows ? k[ulong(kr) * channels + head * dim + dd * 8 + fm]
                        : 0.0f;
        }
        simdgroup_multiply_accumulate(score, aq, bk, score);
      }
      for (uint j = 0; j < 2; ++j) {
        uint kr = kb + sk * 8 + fn + j;
        prob[sk][j] =
            (qr < rows && kr < rows && kr <= qr && qr - kr < uint(p[2]))
                ? score.thread_elements()[j] * scale
                : -MAXFLOAT;
      }
      float local = max(prob[sk].x, prob[sk].y);
      local = max(local, simd_shuffle_xor(local, ushort(1)));
      local = max(local, simd_shuffle_xor(local, ushort(8)));
      new_max = max(new_max, local);
    }
    float sum = 0.0f;
    for (uint sk = 0; sk < BK / 8; ++sk) {
      prob[sk] = fast::exp2(prob[sk] - new_max);
      float local = prob[sk].x + prob[sk].y;
      local = local + simd_shuffle_xor(local, ushort(1));
      local = local + simd_shuffle_xor(local, ushort(8));
      sum = sum + local;
    }
    float factor = fast::exp2(maximum - new_max);
    maximum = new_max;
    norm = fma(norm, factor, sum);
    for (uint dd = 0; dd < 8; ++dd) {
      simdgroup_float8x8 output;
      output.thread_elements()[0] = acco[dd].x * factor;
      output.thread_elements()[1] = acco[dd].y * factor;
      for (uint sk = 0; sk < BK / 8; ++sk) {
        simdgroup_float8x8 a, b;
        a.thread_elements()[0] = prob[sk].x;
        a.thread_elements()[1] = prob[sk].y;
        for (uint j = 0; j < 2; ++j) {
          uint vr = kb + sk * 8 + fm;
          b.thread_elements()[j] =
              vr < rows ? v[ulong(vr) * channels + head * dim + dd * 8 + fn + j]
                        : 0.0f;
        }
        simdgroup_multiply_accumulate(output, a, b, output);
      }
      acco[dd] =
          float2(output.thread_elements()[0], output.thread_elements()[1]);
    }
  }
  if (qr < rows)
    for (uint dd = 0; dd < 8; ++dd)
      for (uint j = 0; j < 2; ++j)
        out[ulong(qr) * channels + head * dim + dd * 8 + fn + j] =
            acco[dd][j] / norm;
}
#define DR3E_ATTN(BK)                                                          \
  kernel void dr3e_attention_##BK(device const float *q [[buffer(0)]],         \
                                  device const float *k [[buffer(1)]],         \
                                  device const float *v [[buffer(2)]],         \
                                  device float *out [[buffer(3)]],             \
                                  constant ulong *p [[buffer(4)]],             \
                                  uint tile [[threadgroup_position_in_grid]],  \
                                  uint lane [[thread_index_in_simdgroup]]) {   \
    dr3e_attention_body<BK>(q, k, v, out, p, tile, lane);                      \
  }
DR3E_ATTN(32)
