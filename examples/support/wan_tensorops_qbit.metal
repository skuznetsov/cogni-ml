#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef WTP_TM
#define WTP_TM 32
#endif
#ifndef WTP_TN
#define WTP_TN 32
#endif
#ifndef WTP_TK
#define WTP_TK 32
#endif

#if WTP_TM < 1 || WTP_TN < 1 || WTP_TK < 1
#error "WTP tile dimensions must be positive"
#endif

// Runtime-compiled, opt-in QBit TensorOps experiment. The record parsing and
// arithmetic below follow qbit_positional_weights.metal's pos_u16/pos_u32,
// pos_bf16, pos_decode, and wan_positional_decode layout. They are inline here
// because the runtime compiler receives this shader as an independent source.
// This kernel admits only tile_values=1024, correction mode 0, and thus BF16
// replacement corrections. The caller must bind the pinned immutable t050
// positional arm; the shader checks its format, not its external provenance.
#pragma clang fp contract(off)

struct WTPQRecord {
  uint begin;
  uint precision;
  uint correction_count;
  uint base_bytes;
  bool valid;
};

static inline uint wtpq_u16(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8);
}

static inline uint wtpq_u32(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
         (uint(p[3]) << 24);
}

static inline float wtpq_bf16(device const uchar *p) {
  return as_type<float>(wtpq_u16(p) << 16);
}

static inline uint wtpq_position(device const uchar *entry) {
  // The selected 1024-value format stores correction positions as LE U16.
  return wtpq_u16(entry);
}

static inline float wtpq_replacement(device const uchar *entry) {
  // Mode zero stores a BF16 replacement after the U16 position.
  return wtpq_bf16(entry + 2);
}

static inline uint wtpq_code(device const uchar *block, uint within,
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

static inline WTPQRecord wtpq_validate_record(
    device const uchar *blob, device const uint *offsets, uint blob_bytes,
    uint offset_count, ulong tile_count, ulong tile_index, uint tile_values) {
  WTPQRecord result = {0u, 16u, 0u, 0u, false};
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
  uint correction_count = wtpq_u16(record + 2);
  // t050 uses mode zero only: BF16 replacement corrections, never F16 deltas.
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
      float mean = as_type<float>(wtpq_u32(block));
      float sigma = as_type<float>(wtpq_u32(block + 4));
      if (!isfinite(mean) || !isfinite(sigma) || sigma < 0.0f)
        return result;
    }
  }

  device const uchar *corrections = record + 4u + base_bytes;
  uint previous_position = 0u;
  for (uint i = 0u; i < correction_count; ++i) {
    device const uchar *entry = corrections + i * 4u;
    uint position = wtpq_position(entry);
    float replacement = wtpq_replacement(entry);
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

static inline void wtpq_poison_tile(device float *output,
                                    device atomic_uint *status, uint M,
                                    uint N, ulong row_begin, ulong col_begin,
                                    uint tid, bool count_error) {
  if (count_error && tid == 0u)
    atomic_fetch_add_explicit(status + 7u, 1u, memory_order_relaxed);

  constexpr ulong tile_area = ulong(WTP_TM) * ulong(WTP_TN);
  for (ulong i = ulong(tid); i < tile_area; i += 32ul) {
    ulong row = row_begin + i / ulong(WTP_TN);
    ulong column = col_begin + i % ulong(WTP_TN);
    if (row < ulong(M) && column < ulong(N))
      output[row * ulong(N) + column] = as_type<float>(0x7fc00000u);
  }
}

kernel void wan_tensorops_qbit_positional(
    device float *input [[buffer(0)]],
    device const uchar *blob [[buffer(1)]],
    device const uint *offsets [[buffer(2)]],
    device const float *lut_p4 [[buffer(3)]],
    device const float *lut_p5 [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device float *output [[buffer(6)]],
    device atomic_uint *status [[buffer(7)]],
    constant uint4 &shape [[buffer(8)]],
    constant uint4 &limits [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  // Caller contract: launch exactly one 32-thread simdgroup per threadgroup;
  // limits contain truthful capacities (bytes for blob, elements otherwise),
  // and input/bias/output/status are sized and alias-checked before dispatch.
  // status has at least eight atomic U32 words and is initialized once by the
  // caller; this kernel only reads [5:6] and increments sticky error word [7].
  constexpr int TM = WTP_TM;
  constexpr int TN = WTP_TN;
  constexpr int TK = WTP_TK;
  constexpr auto desc = matmul2d_descriptor(
      TM, TN, TK, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);

  const uint M = shape.x;
  const uint K = shape.y;
  const uint N = shape.z;
  const uint tile_values = shape.w;
  const uint blob_bytes = limits.x;
  const uint offset_count = limits.y;
  const uint lut4_count = limits.z;
  const uint lut5_count = limits.w;
  const ulong row_begin = ulong(group.y) * ulong(WTP_TM);
  const ulong col_begin = ulong(group.x) * ulong(WTP_TN);

  threadgroup uint record_begin[WTP_TN];
  threadgroup uint record_precision[WTP_TN];
  threadgroup uint record_corrections[WTP_TN];
  threadgroup uint record_base_bytes[WTP_TN];
  threadgroup uint validation_errors[32];
  threadgroup uint incoming_failure;
  threadgroup uint bad_record;
  threadgroup uint uses_p4;
  threadgroup uint uses_p5;

  if (tid == 0u) {
    uint failed_pack = atomic_load_explicit(status + 5u, memory_order_relaxed);
    uint failed_capacity = atomic_load_explicit(status + 6u, memory_order_relaxed);
    incoming_failure = uint(failed_pack != 0u || failed_capacity != 0u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (incoming_failure != 0u) {
    wtpq_poison_tile(output, status, M, N, row_begin, col_begin, tid, false);
    return;
  }

  bool shape_invalid = M == 0u || K == 0u || N == 0u ||
      M > 0x7fffffffu || K > 0x7fffffffu || N > 0x7fffffffu ||
      tile_values != 1024u || (lut4_count < 256u) || (lut5_count < 256u);
  if (tile_values == 0u || K % tile_values != 0u ||
      uint(WTP_TK) > tile_values || tile_values % uint(WTP_TK) != 0u ||
      K % uint(WTP_TK) != 0u)
    shape_invalid = true;
  ulong tiles_per_row = 0ul;
  ulong tile_count = 0ul;
  if (tile_values != 0u && K % tile_values == 0u) {
    tiles_per_row = ulong(K / tile_values);
    tile_count = ulong(N) * tiles_per_row;
  }
  if (tile_count >= 0xfffffffful ||
      ulong(offset_count) < tile_count + 1ul)
    shape_invalid = true;
  if (shape_invalid) {
    wtpq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
    return;
  }

  matmul2d<desc, execution_simdgroup> op;
  auto a = tensor(input, dextents<int, 2>{int(K), int(M)},
                  array<int, 2>{1, int(K)});
  auto dst = tensor(output, dextents<int, 2>{int(N), int(M)},
                    array<int, 2>{1, int(N)});
  auto b = op.get_right_input_cooperative_tensor<float, float, float>();
  auto ap = a.slice(0, int(row_begin));
  auto accum = op.get_destination_cooperative_tensor<decltype(ap), decltype(b), float>();

  #pragma clang loop unroll(full)
  for (uint i = 0u; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i))
      accum[i] = 0.0f;
  }

  for (uint k = 0u; k < K; k += uint(WTP_TK)) {
    if (k % tile_values == 0u) {
      uint local_error = 0u;
      for (uint local_n = tid; local_n < uint(WTP_TN); local_n += 32u) {
        record_begin[local_n] = 0u;
        record_precision[local_n] = 16u;
        record_corrections[local_n] = 0u;
        record_base_bytes[local_n] = 0u;
        ulong channel = col_begin + ulong(local_n);
        if (channel < ulong(N)) {
          ulong tile_index = channel * tiles_per_row + ulong(k / tile_values);
          WTPQRecord record = wtpq_validate_record(
              blob, offsets, blob_bytes, offset_count, tile_count,
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
        bad_record = 0u;
        uses_p4 = 0u;
        uses_p5 = 0u;
        for (uint i = 0u; i < 32u; ++i)
          bad_record |= validation_errors[i];
        for (uint i = 0u; i < uint(WTP_TN); ++i) {
          uses_p4 |= uint(record_precision[i] == 4u);
          uses_p5 |= uint(record_precision[i] == 5u);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (bad_record != 0u) {
        wtpq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
        return;
      }

      uint lut_error = 0u;
      for (uint i = tid; i < 256u; i += 32u) {
        if ((uses_p4 != 0u && !isfinite(lut_p4[i])) ||
            (uses_p5 != 0u && !isfinite(lut_p5[i])))
          lut_error = 1u;
      }
      validation_errors[tid] = lut_error;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        bad_record = 0u;
        for (uint i = 0u; i < 32u; ++i)
          bad_record |= validation_errors[i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (bad_record != 0u) {
        wtpq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
        return;
      }
    }

    uint decoded_value_error = 0u;
    #pragma clang loop unroll(full)
    for (uint i = 0u; i < b.get_capacity(); ++i) {
      if (b.is_valid_element(i)) {
        auto c = b.get_multidimensional_index(i);
        uint local_n = uint(c[0]);
        ulong n = col_begin + ulong(local_n);
        uint kk = k + uint(c[1]);
        float value = 0.0f;
        if (n < ulong(N) && kk < K) {
          uint record_start = record_begin[local_n];
          uint precision = record_precision[local_n];
          uint correction_count = record_corrections[local_n];
          uint base_bytes = record_base_bytes[local_n];
          device const uchar *record = blob + record_start;
          uint within = kk % tile_values;
          if (precision == 16u) {
            value = wtpq_bf16(record + 4u + within * 2u);
          } else {
            uint block_index = within / 256u;
            uint block_bytes = 8u + precision * 32u;
            device const uchar *block = record + 4u + block_index * block_bytes;
            uint code = wtpq_code(block, within & 255u, precision);
            float mean = as_type<float>(wtpq_u32(block));
            float sigma = as_type<float>(wtpq_u32(block + 4u));
            float lut_value = precision == 4u ? lut_p4[code] : lut_p5[code];
            float scaled = sigma * lut_value;
            value = mean + scaled;

            // Binary search matches the native decoder's ordered position
            // lookup. The admitted record mode always replaces with BF16.
            device const uchar *corrections = record + 4u + base_bytes;
            uint lo = 0u, hi = correction_count;
            while (lo < hi) {
              uint mid = lo + (hi - lo) / 2u;
              uint position = wtpq_position(corrections + mid * 4u);
              if (position < within)
                lo = mid + 1u;
              else
                hi = mid;
            }
            if (lo < correction_count) {
              device const uchar *entry = corrections + lo * 4u;
              if (wtpq_position(entry) == within)
                value = wtpq_replacement(entry);
            }
          }
          decoded_value_error |= uint(!isfinite(value));
        }
        b[i] = value;
      }
    }

    // Exactly one SIMD group owns this tile. A convergent vote preserves the
    // fail-closed decision without two threadgroup-memory barriers per K32
    // panel. Record metadata publication barriers above remain necessary.
    if (simd_any(decoded_value_error != 0u)) {
      wtpq_poison_tile(output, status, M, N, row_begin, col_begin, tid, true);
      return;
    }

    auto panel = a.slice(int(k), int(row_begin));
    op.run(panel, b, accum);
  }

  #pragma clang loop unroll(full)
  for (uint i = 0u; i < accum.get_capacity(); ++i) {
    if (accum.is_valid_element(i)) {
      auto c = accum.get_multidimensional_index(i);
      ulong n = col_begin + ulong(uint(c[0]));
      if (n < ulong(N))
        accum[i] = accum[i] + bias[uint(n)];
    }
  }
  auto out = dst.slice(int(col_begin), int(row_begin));
  accum.store(out);
}
