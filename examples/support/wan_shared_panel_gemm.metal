#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#ifndef WTP_TM
#define WTP_TM 32
#endif
#ifndef WTP_TN
#define WTP_TN 32
#endif
#ifndef WTP_TK
#define WTP_TK 32
#endif

#if (WTP_TM != 32 && WTP_TM != 64) || WTP_TN != 32 || WTP_TK != 32
#error "The shared-panel probe admits TM=32/64, TN=32, TK=32"
#endif

// Probe-only classic tiled F32 GEMM. The QBit entry point uses the same
// cooperative A/B panel schedule and computes its selected positional record
// directly into the threadgroup B panel. It intentionally does not stage a
// global dense matrix. The record parser/decode follows
// examples/support/wan_tensorops_qbit.metal (native layout helpers).
#pragma clang fp contract(off)

struct WSPQRecord {
  uint begin;
  uint precision;
  uint correction_count;
  uint base_bytes;
  bool valid;
};

static inline uint wspq_u16(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8);
}

static inline uint wspq_u32(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
         (uint(p[3]) << 24);
}

static inline float wspq_bf16(device const uchar *p) {
  return as_type<float>(wspq_u16(p) << 16);
}

static inline uint wspq_position(device const uchar *entry) {
  return wspq_u16(entry);
}

static inline float wspq_replacement(device const uchar *entry) {
  return wspq_bf16(entry + 2);
}

static inline uint wspq_code(device const uchar *block, uint within,
                             uint precision) {
  uint code = 0u;
  uint byte = 31u - within / 8u;
  uint mask = 1u << (within & 7u);
  for (uint plane = 0u; plane < precision; ++plane) {
    if ((uint(block[8u + plane * 32u + byte]) & mask) != 0u)
      code |= 1u << (7u - plane);
  }
  return code;
}

static inline WSPQRecord wspq_validate_record(
    device const uchar *blob, device const uint *offsets, uint blob_bytes,
    uint offset_count, ulong tile_count, ulong tile_index, uint tile_values) {
  WSPQRecord result = {0u, 16u, 0u, 0u, false};
  if (offset_count < 2u || tile_index >= tile_count ||
      tile_index + 1ul >= ulong(offset_count))
    return result;

  uint begin = offsets[uint(tile_index)];
  uint end = offsets[uint(tile_index + 1ul)];
  if (begin > blob_bytes || end > blob_bytes || begin > end ||
      (begin & 3u) != 0u || (end & 3u) != 0u ||
      (tile_index == 0ul && begin != 0u) || end - begin < 4u)
    return result;

  device const uchar *record = blob + begin;
  uint precision = uint(record[0]);
  uint mode = uint(record[1]);
  uint correction_count = wspq_u16(record + 2);
  if ((precision != 4u && precision != 5u && precision != 16u) ||
      mode != 0u || correction_count > tile_values ||
      (precision == 16u && correction_count != 0u))
    return result;

  uint base_bytes = precision == 16u
      ? tile_values * 2u
      : (tile_values / 256u) * (8u + precision * 32u);
  uint unpadded_bytes = 4u + base_bytes + correction_count * 4u;
  uint record_bytes = end - begin;
  uint padded_bytes = (unpadded_bytes + 3u) & ~3u;
  if (padded_bytes != record_bytes)
    return result;

  if (precision != 16u) {
    uint block_bytes = 8u + precision * 32u;
    uint blocks = tile_values / 256u;
    for (uint block_index = 0u; block_index < blocks; ++block_index) {
      device const uchar *block = record + 4u + block_index * block_bytes;
      float mean = as_type<float>(wspq_u32(block));
      float sigma = as_type<float>(wspq_u32(block + 4));
      if (!isfinite(mean) || !isfinite(sigma) || sigma < 0.0f)
        return result;
    }
  }

  device const uchar *corrections = record + 4u + base_bytes;
  uint previous_position = 0u;
  for (uint i = 0u; i < correction_count; ++i) {
    device const uchar *entry = corrections + i * 4u;
    uint position = wspq_position(entry);
    float replacement = wspq_replacement(entry);
    if (position >= tile_values || (i != 0u && position <= previous_position) ||
        !isfinite(replacement))
      return result;
    previous_position = position;
  }

  for (uint i = unpadded_bytes; i < record_bytes; ++i) {
    if (record[i] != 0u)
      return result;
  }

  result.begin = begin;
  result.precision = precision;
  result.correction_count = correction_count;
  result.base_bytes = base_bytes;
  result.valid = true;
  return result;
}

static inline void wspq_poison_tile(
    device float *output, device atomic_uint *status, uint M, uint N,
    ulong row_begin, ulong col_begin, uint tid, bool count_error) {
  if (count_error && tid == 0u)
    atomic_fetch_add_explicit(status + 7u, 1u, memory_order_relaxed);
  constexpr ulong area = ulong(WTP_TM) * ulong(WTP_TN);
  for (ulong i = ulong(tid); i < area; i += 128ul) {
    ulong row = row_begin + i / ulong(WTP_TN);
    ulong col = col_begin + i % ulong(WTP_TN);
    if (row < ulong(M) && col < ulong(N))
      output[row * ulong(N) + col] = as_type<float>(0x7fc00000u);
  }
}

template <bool QBIT>
inline void wsp_run(
    device const float *input, device const float *dense_weight,
    device const uchar *blob, device const uint *offsets,
    device const float *lut4, device const float *lut5,
    device const float *bias, device float *output,
    device atomic_uint *status, uint4 shape, uint4 limits,
    threadgroup float *a_panel, threadgroup float *b_panel,
    threadgroup uint *record_begin, threadgroup uint *record_precision,
    threadgroup uint *record_corrections, threadgroup uint *record_base_bytes,
    threadgroup uint *validation_errors, threadgroup uint *incoming_failure,
    threadgroup uint *bad_record, threadgroup uint *uses_p4,
    threadgroup uint *uses_p5, uint2 group, uint tid, uint lane,
    uint simdgroup, uint threads) {
  constexpr uint TM = WTP_TM;
  constexpr uint TN = WTP_TN;
  constexpr uint TK = WTP_TK;
  constexpr uint ROW_BLOCKS = TM / 16u;
  constexpr uint COL_BLOCKS = TN / 16u;
  const uint M = shape.x;
  const uint K = shape.y;
  const uint N = shape.z;
  const uint tile_values = shape.w;
  const ulong row_begin = ulong(group.y) * ulong(TM);
  const ulong col_begin = ulong(group.x) * ulong(TN);

  if (threads != 128u)
    return;

  if (QBIT) {
    if (tid == 0u) {
      uint failed_pack = atomic_load_explicit(status + 5u, memory_order_relaxed);
      uint failed_capacity = atomic_load_explicit(status + 6u, memory_order_relaxed);
      *incoming_failure = uint(failed_pack != 0u || failed_capacity != 0u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*incoming_failure != 0u) {
      wspq_poison_tile(output, status, M, N, row_begin, col_begin, tid, false);
      return;
    }

    bool shape_invalid = M == 0u || K == 0u || N == 0u ||
        M > 0x7fffffffu || K > 0x7fffffffu || N > 0x7fffffffu ||
        tile_values != 1024u || limits.z < 256u || limits.w < 256u;
    if (tile_values == 0u || K % tile_values != 0u ||
        TK > tile_values || tile_values % TK != 0u || K % TK != 0u)
      shape_invalid = true;
    ulong tiles_per_row = 0ul;
    ulong tile_count = 0ul;
    if (tile_values != 0u && K % tile_values == 0u) {
      tiles_per_row = ulong(K / tile_values);
      tile_count = ulong(N) * tiles_per_row;
    }
    if (tile_count >= 0xfffffffful ||
        ulong(limits.y) < tile_count + 1ul)
      shape_invalid = true;
    if (shape_invalid) {
      wspq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
      return;
    }

    for (uint tidx = tid; tidx < 128u; tidx += 128u)
      validation_errors[tidx] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  simdgroup_float8x8 accum[ROW_BLOCKS * COL_BLOCKS];
  #pragma clang loop unroll(full)
  for (uint i = 0u; i < ROW_BLOCKS * COL_BLOCKS; ++i) {
    accum[i].thread_elements()[0] = 0.0f;
    accum[i].thread_elements()[1] = 0.0f;
  }

  const uint sg_row = (simdgroup & 1u) * (TM / 2u);
  const uint sg_col = (simdgroup >> 1u) * (TN / 2u);
  const uint fm = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
  const uint fn = (((lane / 4u) & 2u) * 2u) + (lane % 2u) * 2u;

  ulong tiles_per_row = QBIT ? ulong(K / tile_values) : 0ul;
  for (uint k0 = 0u; k0 < K; k0 += TK) {
    if (QBIT && k0 % tile_values == 0u) {
      uint local_error = 0u;
      for (uint local_n = tid; local_n < TN; local_n += 128u) {
        record_begin[local_n] = 0u;
        record_precision[local_n] = 16u;
        record_corrections[local_n] = 0u;
        record_base_bytes[local_n] = 0u;
        ulong channel = col_begin + ulong(local_n);
        if (channel < ulong(N)) {
          ulong tile_index = channel * tiles_per_row + ulong(k0 / tile_values);
          WSPQRecord record = wspq_validate_record(
              blob, offsets, limits.x, limits.y, ulong(N) * tiles_per_row,
              tile_index, tile_values);
          if (!record.valid) {
            local_error = 1u;
          } else {
            record_begin[local_n] = record.begin;
            record_precision[local_n] = record.precision;
            record_corrections[local_n] = record.correction_count;
            record_base_bytes[local_n] = record.base_bytes;
          }
        }
      }
      validation_errors[tid] = local_error;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        *bad_record = 0u;
        *uses_p4 = 0u;
        *uses_p5 = 0u;
        for (uint i = 0u; i < 128u; ++i)
          *bad_record |= validation_errors[i];
        for (uint i = 0u; i < TN; ++i) {
          *uses_p4 |= uint(record_precision[i] == 4u);
          *uses_p5 |= uint(record_precision[i] == 5u);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (*bad_record != 0u) {
        wspq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
        return;
      }

      uint lut_error = 0u;
      for (uint i = tid; i < 256u; i += 128u) {
        if ((*uses_p4 != 0u && !isfinite(lut4[i])) ||
            (*uses_p5 != 0u && !isfinite(lut5[i])))
          lut_error = 1u;
      }
      validation_errors[tid] = lut_error;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        *bad_record = 0u;
        for (uint i = 0u; i < 128u; ++i)
          *bad_record |= validation_errors[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (*bad_record != 0u) {
        wspq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
        return;
      }
    }

    uint decoded_error = 0u;
    for (ulong i = ulong(tid); i < ulong(TK) * ulong(TN); i += 128ul) {
      uint local_k = uint(i / ulong(TN));
      uint local_n = uint(i % ulong(TN));
      ulong n = col_begin + ulong(local_n);
      uint kk = k0 + local_k;
      float value = 0.0f;
      if (n < ulong(N) && kk < K) {
        if (QBIT) {
          uint record_start = record_begin[local_n];
          uint precision = record_precision[local_n];
          uint correction_count = record_corrections[local_n];
          uint base_bytes = record_base_bytes[local_n];
          device const uchar *record = blob + record_start;
          uint within = kk % tile_values;
          if (precision == 16u) {
            value = wspq_bf16(record + 4u + within * 2u);
          } else {
            uint block_index = within / 256u;
            uint block_bytes = 8u + precision * 32u;
            device const uchar *block = record + 4u + block_index * block_bytes;
            uint code = wspq_code(block, within & 255u, precision);
            float mean = as_type<float>(wspq_u32(block));
            float sigma = as_type<float>(wspq_u32(block + 4u));
            float lut_value = precision == 4u ? lut4[code] : lut5[code];
            float scaled = sigma * lut_value;
            value = mean + scaled;

            device const uchar *corrections = record + 4u + base_bytes;
            uint lo = 0u, hi = correction_count;
            while (lo < hi) {
              uint mid = lo + (hi - lo) / 2u;
              uint position = wspq_position(corrections + mid * 4u);
              if (position < within)
                lo = mid + 1u;
              else
                hi = mid;
            }
            if (lo < correction_count) {
              device const uchar *entry = corrections + lo * 4u;
              if (wspq_position(entry) == within)
                value = wspq_replacement(entry);
            }
          }
          decoded_error |= uint(!isfinite(value));
        } else {
          // The pinned Linear weight is stored [N,K]; the panel is [K,N].
          value = dense_weight[n * ulong(K) + ulong(kk)];
        }
      }
      b_panel[i] = value;
    }

    if (QBIT) {
      validation_errors[tid] = decoded_error;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        *bad_record = 0u;
        for (uint i = 0u; i < 128u; ++i)
          *bad_record |= validation_errors[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (*bad_record != 0u) {
        wspq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
        return;
      }
    }

    for (ulong i = ulong(tid); i < ulong(TM) * ulong(TK); i += 128ul) {
      uint local_m = uint(i / ulong(TK));
      uint local_k = uint(i % ulong(TK));
      ulong m = row_begin + ulong(local_m);
      uint kk = k0 + local_k;
      a_panel[i] = m < ulong(M) && kk < K
          ? input[m * ulong(K) + ulong(kk)] : 0.0f;
    }

    // Publish both panels before any SIMD group consumes them. This barrier
    // also ensures every QBit decoder has completed its local B writes.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (uint kfrag = 0u; kfrag < TK; kfrag += 8u) {
      #pragma clang loop unroll(full)
      for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
        #pragma clang loop unroll(full)
        for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
          simdgroup_float8x8 a_fragment;
          simdgroup_float8x8 b_fragment;
          simdgroup_load(a_fragment,
              a_panel + (sg_row + rb * 8u) * TK + kfrag, TK);
          simdgroup_load(b_fragment,
              b_panel + kfrag * TN + sg_col + cb * 8u, TN);
          uint acc_index = rb * COL_BLOCKS + cb;
          simdgroup_multiply_accumulate(
              accum[acc_index], a_fragment, b_fragment, accum[acc_index]);
        }
      }
    }

    // All four SIMD groups finish reading the panel before its next overwrite.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (uint rb = 0u; rb < ROW_BLOCKS; ++rb) {
    #pragma clang loop unroll(full)
    for (uint cb = 0u; cb < COL_BLOCKS; ++cb) {
      uint acc_index = rb * COL_BLOCKS + cb;
      simdgroup_store(accum[acc_index],
          a_panel + (sg_row + rb * 8u) * TN + sg_col + cb * 8u, TN);
    }
  }

  // The A panel is dead after the final K read and has the same TM x 32
  // extent as this C tile. Store matrix results there, then add bias while
  // cooperatively copying only in-bounds outputs.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (ulong i = ulong(tid); i < ulong(TM) * ulong(TN); i += 128ul) {
    ulong m = row_begin + i / ulong(TN);
    ulong n = col_begin + i % ulong(TN);
    if (m < ulong(M) && n < ulong(N))
      output[m * ulong(N) + n] = a_panel[i] + bias[n];
  }
}

kernel void wan_shared_panel_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint4 &shape [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float a_panel[WTP_TM * WTP_TK];
  threadgroup float b_panel[WTP_TK * WTP_TN];
  threadgroup uint record_begin[WTP_TN];
  threadgroup uint record_precision[WTP_TN];
  threadgroup uint record_corrections[WTP_TN];
  threadgroup uint record_base_bytes[WTP_TN];
  threadgroup uint validation_errors[128];
  threadgroup uint incoming_failure;
  threadgroup uint bad_record;
  threadgroup uint uses_p4;
  threadgroup uint uses_p5;
  wsp_run<false>(input, weight, nullptr, nullptr, nullptr, nullptr, bias,
      output, nullptr, shape, uint4(0u), a_panel, b_panel, record_begin,
      record_precision, record_corrections, record_base_bytes,
      validation_errors, &incoming_failure, &bad_record, &uses_p4, &uses_p5,
      group, tid, lane, simdgroup, threads.x * threads.y);
}

kernel void wan_shared_panel_qbit_positional(
    device const float *input [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device const float *lut4 [[buffer(3)]],
    device const float *lut5 [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device float *output [[buffer(6)]],
    device atomic_uint *status [[buffer(7)]],
    constant uint4 &shape [[buffer(8)]],
    constant uint4 &limits [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint2 threads [[threads_per_threadgroup]]) {
  threadgroup float a_panel[WTP_TM * WTP_TK];
  threadgroup float b_panel[WTP_TK * WTP_TN];
  threadgroup uint record_begin[WTP_TN];
  threadgroup uint record_precision[WTP_TN];
  threadgroup uint record_corrections[WTP_TN];
  threadgroup uint record_base_bytes[WTP_TN];
  threadgroup uint validation_errors[128];
  threadgroup uint incoming_failure;
  threadgroup uint bad_record;
  threadgroup uint uses_p4;
  threadgroup uint uses_p5;
  wsp_run<true>(input, nullptr, blob, offsets, lut4, lut5, bias, output,
      status, shape, limits, a_panel, b_panel, record_begin,
      record_precision, record_corrections, record_base_bytes,
      validation_errors, &incoming_failure, &bad_record, &uses_p4, &uses_p5,
      group, tid, lane, simdgroup, threads.x * threads.y);
}
