#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
#pragma clang fp contract(off)

static inline uint qbc_u16(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8);
}

static inline uint qbc_u32(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
         (uint(p[3]) << 24);
}

static inline float qbc_bf16(device const uchar *p) {
  return as_type<float>(qbc_u16(p) << 16);
}

static inline uint qbc_position(device const uchar *p, uint width) {
  return width == 1u ? uint(p[0]) : qbc_u16(p);
}

static inline uint qbc_code(device const uchar *block, uint within,
                            uint precision) {
  uint code = 0u;
  const uint byte = 31u - within / 8u;
  const uint mask = 1u << (within & 7u);
  for (uint plane = 0u; plane < precision; ++plane) {
    if ((uint(block[8u + plane * 32u + byte]) & mask) != 0u)
      code |= 1u << (7u - plane);
  }
  return code;
}

static inline void qbc_fail_output(device float *output,
                                   device atomic_uint *status,
                                   uint rows, uint out_channels,
                                   uint row_begin, uint row_end,
                                   uint tile_n, uint lane,
                                   bool count_record_error) {
  if (lane == 0u && count_record_error)
    atomic_fetch_add_explicit(status + 7u, 1u, memory_order_relaxed);
  for (uint i = lane; i < 128u; i += 32u) {
    const uint row = row_begin + i / 8u;
    const uint channel = tile_n * 8u + i % 8u;
    if (row < rows && row < row_end && channel < out_channels)
      output[ulong(row) * out_channels + channel] = as_type<float>(0x7fc00000u);
  }
}

// One threadgroup computes 16 rows by 8 output channels. It decodes one 8x8
// B fragment, then reuses it for both 8x8 accumulators. Packed records and
// codebooks stay local to this threadgroup; there is no dense weight stage.
kernel void video_qbit_codebook_linear_f32(
    device const float *input [[buffer(0)]],
    device const uint *offsets [[buffer(1)]],
    device const uchar *blob [[buffer(2)]],
    device const float *lut4 [[buffer(3)]],
    device const float *lut5 [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device float *output [[buffer(6)]],
    device atomic_uint *status [[buffer(7)]],
    constant uint *p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = p[0], channels = p[1], out_channels = p[2];
  const uint first_row = p[3], row_count = p[4];
  const uint tile_values = p[5], tile_count = p[6], capacity = p[7];
  const uint tiles_per_row = channels / tile_values;
  const uint tiles_n = (out_channels + 7u) / 8u;
  const uint tile_m = group / tiles_n, tile_n = group % tiles_n;
  const uint row_begin = first_row + tile_m * 16u;
  const uint row_end = first_row + row_count;
  const uint correction_width = tile_values == 256u ? 1u : 2u;

  threadgroup float a0[64], a1[64], b[64], c0[64], c1[64];
  threadgroup float codebook[256];
  threadgroup uint errors[32];
  threadgroup uint bad_group, has_quant;
  threadgroup uint record_start[8], record_end[8], record_precision[8];
  threadgroup uint record_mode[8], record_count[8], record_base_bytes[8];

  simdgroup_float8x8 accumulator0 =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 accumulator1 =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  if (lane == 0u) {
    bad_group = atomic_load_explicit(status + 5u, memory_order_relaxed) |
                atomic_load_explicit(status + 6u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (bad_group != 0u) {
    qbc_fail_output(output, status, rows, out_channels, row_begin, row_end,
                    tile_n, lane, false);
    return;
  }

  for (uint k0 = 0u; k0 < channels; k0 += 8u) {
    // A record is admitted only after its offset pair, header, exact extent,
    // padding, normalization scalars, and all correction entries are checked.
    // No record pointer is formed before the corresponding extent is proven.
    if ((k0 % tile_values) == 0u) {
      errors[lane] = 0u;
      if (lane < 8u) {
        const uint out = tile_n * 8u + lane;
        record_start[lane] = 0u;
        record_end[lane] = 0u;
        record_precision[lane] = 16u;
        record_mode[lane] = 0u;
        record_count[lane] = 0u;
        record_base_bytes[lane] = 0u;
        if (out < out_channels) {
          const uint tile_index = out * tiles_per_row + k0 / tile_values;
          bool invalid = tile_index >= tile_count;
          uint begin = 0u, end = 0u;
          if (!invalid) {
            begin = offsets[tile_index];
            end = offsets[tile_index + 1u];
            invalid = begin > capacity || end > capacity || begin > end ||
                      end - begin < 4u || (begin & 3u) != 0u ||
                      (end & 3u) != 0u ||
                      (tile_index == 0u && begin != 0u);
          }
          if (!invalid) {
            device const uchar *record = blob + begin;
            const uint precision = uint(record[0]);
            const uint mode = uint(record[1]);
            const uint count = qbc_u16(record + 2u);
            invalid = (precision != 4u && precision != 5u && precision != 16u) ||
                      mode > 1u || count > tile_values ||
                      (precision == 16u && count != 0u);
            uint base_bytes = 0u;
            uint unpadded = 0u;
            if (!invalid) {
              base_bytes = precision == 16u
                  ? tile_values * 2u
                  : tile_values / 256u * (8u + precision * 32u);
              unpadded = 4u + base_bytes + count * (correction_width + 2u);
              invalid = ((unpadded + 3u) & ~3u) != end - begin;
            }
            if (!invalid) {
              device const uchar *base = record + 4u;
              if (precision != 16u) {
                const uint block_bytes = 8u + precision * 32u;
                for (uint block = 0u; block < tile_values / 256u; ++block) {
                  device const uchar *q = base + block * block_bytes;
                  const float mean = as_type<float>(qbc_u32(q));
                  const float sigma = as_type<float>(qbc_u32(q + 4u));
                  invalid |= !isfinite(mean) || !isfinite(sigma) || sigma < 0.0f;
                }
              }
              device const uchar *entries = base + base_bytes;
              uint previous = 0u;
              for (uint i = 0u; i < count; ++i) {
                device const uchar *entry = entries + i * (correction_width + 2u);
                const uint position = qbc_position(entry, correction_width);
                invalid |= position >= tile_values || (i > 0u && position <= previous);
                previous = position;
                const float value = mode == 0u
                    ? qbc_bf16(entry + correction_width)
                    : float(as_type<half>(ushort(qbc_u16(entry + correction_width))));
                invalid |= !isfinite(value);
              }
              const uint padded = (unpadded + 3u) & ~3u;
              for (uint i = unpadded; i < padded; ++i)
                invalid |= record[i] != 0u;
            }
            if (!invalid) {
              record_start[lane] = begin;
              record_end[lane] = end;
              record_precision[lane] = precision;
              record_mode[lane] = mode;
              record_count[lane] = count;
              record_base_bytes[lane] = base_bytes;
            }
          }
          errors[lane] = uint(invalid);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0u) {
        bad_group = 0u;
        has_quant = 0u;
        for (uint i = 0u; i < 8u; ++i) {
          bad_group |= errors[i];
          has_quant |= uint(record_precision[i] == 4u || record_precision[i] == 5u);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (bad_group != 0u) {
        qbc_fail_output(output, status, rows, out_channels, row_begin, row_end,
                        tile_n, lane, true);
        return;
      }
    }

    // Expand each normalization group's rounded codebook once, then gather
    // its values while filling the shared B matrix. Raw BF16 records skip it.
    if ((k0 & 255u) == 0u && has_quant != 0u) {
      uint local_error = 0u;
      for (uint i = lane; i < 256u; i += 32u) {
        const uint out_local = i / 32u;
        const uint code = i & 31u;
        const uint precision = record_precision[out_local];
        float value = 0.0f;
        if ((precision == 4u || precision == 5u) && code < (1u << precision)) {
          const uint group_in_record = (k0 % tile_values) / 256u;
          const uint block_bytes = 8u + precision * 32u;
          device const uchar *block = blob + record_start[out_local] + 4u +
                                      group_in_record * block_bytes;
          const float mean = as_type<float>(qbc_u32(block));
          const float sigma = as_type<float>(qbc_u32(block + 4u));
          const uint lut_index = code << (8u - precision);
          const float lut_value = precision == 4u ? lut4[lut_index] : lut5[lut_index];
          const float scaled = sigma * lut_value;
          value = mean + scaled;
          local_error |= uint(!isfinite(lut_value) || !isfinite(value));
        }
        codebook[i] = value;
      }
      errors[lane] = local_error;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lane == 0u) {
        bad_group = 0u;
        for (uint i = 0u; i < 32u; ++i) bad_group |= errors[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (bad_group != 0u) {
        qbc_fail_output(output, status, rows, out_channels, row_begin, row_end,
                        tile_n, lane, true);
        return;
      }
    }

    uint value_error = 0u;
    for (uint index = lane; index < 64u; index += 32u) {
      const uint k_in_fragment = index / 8u;
      const uint out_local = index & 7u;
      const uint out = tile_n * 8u + out_local;
      const uint k = k0 + k_in_fragment;
      float value = 0.0f;
      if (out < out_channels) {
        const uint precision = record_precision[out_local];
        const uint begin = record_start[out_local];
        device const uchar *record = blob + begin;
        device const uchar *base = record + 4u;
        if (precision == 16u) {
          value = qbc_bf16(base + (k % tile_values) * 2u);
        } else {
          const uint group_in_record = (k % tile_values) / 256u;
          const uint block_bytes = 8u + precision * 32u;
          device const uchar *block = base + group_in_record * block_bytes;
          const uint full_code = qbc_code(block, k & 255u, precision);
          value = codebook[out_local * 32u + (full_code >> (8u - precision))];

          const uint count = record_count[out_local];
          const uint width = correction_width;
          device const uchar *entries = base + record_base_bytes[out_local];
          const uint target = k % tile_values;
          uint lo = 0u, hi = count;
          while (lo < hi) {
            const uint mid = lo + (hi - lo) / 2u;
            const uint position = qbc_position(entries + mid * (width + 2u), width);
            if (position < target) lo = mid + 1u;
            else hi = mid;
          }
          if (lo < count) {
            device const uchar *entry = entries + lo * (width + 2u);
            if (qbc_position(entry, width) == target) {
              if (record_mode[out_local] == 0u)
                value = qbc_bf16(entry + width);
              else
                value += float(as_type<half>(ushort(qbc_u16(entry + width))));
            }
          }
        }
      }
      value_error |= uint(!isfinite(value));
      b[index] = value;

      const uint row0 = row_begin + index / 8u;
      const uint row1 = row0 + 8u;
      const uint input_k = k0 + index % 8u;
      a0[index] = row0 < rows && row0 < row_end
          ? input[ulong(row0) * channels + input_k] : 0.0f;
      a1[index] = row1 < rows && row1 < row_end
          ? input[ulong(row1) * channels + input_k] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    errors[lane] = value_error;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
      bad_group = 0u;
      for (uint i = 0u; i < 32u; ++i) bad_group |= errors[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bad_group != 0u) {
      qbc_fail_output(output, status, rows, out_channels, row_begin, row_end,
                      tile_n, lane, true);
      return;
    }

    simdgroup_float8x8 left0, left1, right;
    simdgroup_load(left0, a0, 8);
    simdgroup_load(left1, a1, 8);
    simdgroup_load(right, b, 8);
    simdgroup_multiply_accumulate(accumulator0, left0, right, accumulator0);
    simdgroup_multiply_accumulate(accumulator1, left1, right, accumulator1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(accumulator0, c0, 8);
  simdgroup_store(accumulator1, c1, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = lane; index < 64u; index += 32u) {
    const uint row0 = row_begin + index / 8u;
    const uint row1 = row0 + 8u;
    const uint channel = tile_n * 8u + index % 8u;
    if (channel < out_channels) {
      if (row0 < rows && row0 < row_end)
        output[ulong(row0) * out_channels + channel] =
            c0[index] + bias[channel];
      if (row1 < rows && row1 < row_end)
        output[ulong(row1) * out_channels + channel] =
            c1[index] + bias[channel];
    }
  }
}

// Same 16x8 output tile, 8-wide K fragments, F32 operands/accumulation, and
// bias order as the packed consumer. This diagnostic control changes only
// the source of each B fragment: resident dense F32 instead of local decode.
kernel void video_qbit_codebook_dense_linear_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint rows = p[0], channels = p[1], out_channels = p[2];
  const uint first_row = p[3], row_count = p[4];
  const uint tiles_n = (out_channels + 7u) / 8u;
  const uint tile_m = group / tiles_n, tile_n = group % tiles_n;
  const uint row_begin = first_row + tile_m * 16u;
  const uint row_end = first_row + row_count;

  threadgroup float a0[64], a1[64], b[64], c0[64], c1[64];
  simdgroup_float8x8 accumulator0 =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  simdgroup_float8x8 accumulator1 =
      make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  for (uint k0 = 0u; k0 < channels; k0 += 8u) {
    for (uint index = lane; index < 64u; index += 32u) {
      const uint row0 = row_begin + index / 8u;
      const uint row1 = row0 + 8u;
      const uint k = k0 + index % 8u;
      const uint channel = tile_n * 8u + index % 8u;
      a0[index] = row0 < rows && row0 < row_end && k < channels
          ? input[ulong(row0) * channels + k] : 0.0f;
      a1[index] = row1 < rows && row1 < row_end && k < channels
          ? input[ulong(row1) * channels + k] : 0.0f;
      const uint weight_k = k0 + index / 8u;
      b[index] = channel < out_channels && weight_k < channels
          ? weight[ulong(channel) * channels + weight_k] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 left0, left1, right;
    simdgroup_load(left0, a0, 8);
    simdgroup_load(left1, a1, 8);
    simdgroup_load(right, b, 8);
    simdgroup_multiply_accumulate(accumulator0, left0, right, accumulator0);
    simdgroup_multiply_accumulate(accumulator1, left1, right, accumulator1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_store(accumulator0, c0, 8);
  simdgroup_store(accumulator1, c1, 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = lane; index < 64u; index += 32u) {
    const uint row0 = row_begin + index / 8u;
    const uint row1 = row0 + 8u;
    const uint channel = tile_n * 8u + index % 8u;
    if (channel < out_channels) {
      if (row0 < rows && row0 < row_end)
        output[ulong(row0) * out_channels + channel] =
            c0[index] + bias[channel];
      if (row1 < rows && row1 < row_end)
        output[ulong(row1) * out_channels + channel] =
            c1[index] + bias[channel];
    }
  }
}
