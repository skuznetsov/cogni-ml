#include <metal_stdlib>
using namespace metal;

constant uint HY_NOMAS_CACHE_QBIT_ROW_VALUES = 128u;
constant uint HY_NOMAS_CACHE_QBIT_PLANE_BYTES = 16u;
constant uint HY_NOMAS_CACHE_QBIT_F32_TIER = 32u;

// Counters are scoped to one reusable workspace slot. The plan clears this
// small record before each cache binding, then the row threads accumulate
// packed bytes, tier counts, and invalid-row count without a host roundtrip.
kernel void hy_nomas_cache_qbit_init(
    device atomic_uint *counters [[buffer(0)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) return;
  for (uint i = 0u; i < 6u; ++i) {
    atomic_store_explicit(&counters[i], 0u, memory_order_relaxed);
  }
}

inline uchar hy_nomas_cache_qbit_nearest_code(float value,
                                              device const float *codebook) {
  const float magnitude = fabs(value);
  uint low = 0u;
  uint high = 127u;
  while (low < high) {
    const uint mid = (low + high) >> 1u;
    const float boundary = (codebook[mid] + codebook[mid + 1u]) * 0.5f;
    if (magnitude <= boundary) {
      high = mid;
    } else {
      low = mid + 1u;
    }
  }
  const uint raw = value < 0.0f ? 255u - low : low;
  return uchar(raw);
}

inline uint hy_nomas_cache_qbit_plane_byte(uint within) {
  const uint bit_index = 127u - (within ^ 7u);
  return bit_index >> 3u;
}

inline uchar hy_nomas_cache_qbit_plane_mask(uint within) {
  const uint bit_index = 127u - (within ^ 7u);
  return uchar(1u << (bit_index & 7u));
}

kernel void hy_nomas_cache_qbit_pack(
    device const float *input [[buffer(0)]],
    device uint *headers [[buffer(1)]],
    device uchar *payload [[buffer(2)]],
    device atomic_uint *counters [[buffer(3)]],
    device const float *codebook [[buffer(4)]],
    constant uint &row_count [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= row_count) return;

  const uint input_base = row * HY_NOMAS_CACHE_QBIT_ROW_VALUES;
  bool finite_row = true;
  bool constant_bits = true;
  const uint first_bits = as_type<uint>(input[input_base]);
  const float first_value = input[input_base];
  float mean = 0.0f;
  for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
    const float value = input[input_base + i];
    if (!isfinite(value)) {
      finite_row = false;
    } else {
      // Scale before summing so a finite row cannot overflow merely because
      // its unscaled 128-value sum exceeds F32 range.
      mean += value * (1.0f / 128.0f);
    }
    if (as_type<uint>(value) != first_bits) constant_bits = false;
  }

  uint tier = HY_NOMAS_CACHE_QBIT_F32_TIER;
  float sigma = 0.0f;
  bool invalid_row = !finite_row;
  bool moments_valid = finite_row && isfinite(mean);
  if (moments_valid) {
    if (constant_bits) {
      // Keep exact source bits, including non-binary constants and -0.
      // Summing repeated F32 values can round the mean by one ulp.
      mean = first_value;
      sigma = 0.0f;
      tier = 4u;
    } else {
      float scale = 0.0f;
      for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
        const float delta = input[input_base + i] - mean;
        if (!isfinite(delta)) moments_valid = false;
        scale = max(scale, fabs(delta));
      }

      if (moments_valid && scale > 0.0f) {
        float square_sum = 0.0f;
        for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
          const float normalized_delta = (input[input_base + i] - mean) / scale;
          square_sum += normalized_delta * normalized_delta;
        }
        sigma = scale * sqrt(square_sum * (1.0f / 128.0f));
        moments_valid = isfinite(sigma) && sigma > 0.0f;
      } else {
        // Numerically equal but bitwise distinct values (for example +0/-0)
        // cannot use the constant representation without losing source bits.
        moments_valid = false;
      }
    }
  }

  if (moments_valid && sigma > 0.0f) {
    float max4 = 0.0f;
    float max5 = 0.0f;
    float max8 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum8 = 0.0f;
    bool normalized_finite = true;
    for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
      const float z = (input[input_base + i] - mean) / sigma;
      if (!isfinite(z)) {
        normalized_finite = false;
        break;
      }
      const uchar raw = hy_nomas_cache_qbit_nearest_code(z, codebook);
      const uint code = uint(raw);
      const float e4 = fabs(z - codebook[128u + code]);
      const float e5 = fabs(z - codebook[384u + code]);
      const float e8 = fabs(z - codebook[640u + code]);
      max4 = max(max4, e4);
      max5 = max(max5, e5);
      max8 = max(max8, e8);
      sum4 += e4 * e4;
      sum5 += e5 * e5;
      sum8 += e8 * e8;
    }

    if (normalized_finite) {
      const float rms4 = sqrt(sum4 * (1.0f / 128.0f));
      const float rms5 = sqrt(sum5 * (1.0f / 128.0f));
      const float rms8 = sqrt(sum8 * (1.0f / 128.0f));
      if (max4 <= 0.50f && rms4 <= 0.12f) {
        tier = 4u;
      } else if (max5 <= 0.25f && rms5 <= 0.06f) {
        tier = 5u;
      } else if (max8 <= 0.10f && rms8 <= 0.02f) {
        tier = 8u;
      }
    }
  }

  // A finite row with unrepresentable moments is an exact escape, not an
  // invalid source row. Only source NaN/Inf contributes to invalid_rows.
  if (!moments_valid) {
    mean = 0.0f;
    sigma = 0.0f;
    tier = HY_NOMAS_CACHE_QBIT_F32_TIER;
  }

  const uint payload_bytes = tier == 4u ? 64u :
                             tier == 5u ? 80u :
                             tier == 8u ? 128u : 512u;
  const uint payload_offset = atomic_fetch_add_explicit(
      &counters[0], payload_bytes, memory_order_relaxed);
  const uint header_base = row * 4u;
  headers[header_base] = as_type<uint>(mean);
  headers[header_base + 1u] = as_type<uint>(sigma);
  headers[header_base + 2u] = payload_offset;
  headers[header_base + 3u] = tier;

  if (tier == HY_NOMAS_CACHE_QBIT_F32_TIER) {
    for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
      const uint bits = as_type<uint>(input[input_base + i]);
      const uint byte_offset = payload_offset + i * 4u;
      payload[byte_offset] = uchar(bits & 0xffu);
      payload[byte_offset + 1u] = uchar((bits >> 8u) & 0xffu);
      payload[byte_offset + 2u] = uchar((bits >> 16u) & 0xffu);
      payload[byte_offset + 3u] = uchar((bits >> 24u) & 0xffu);
    }
    atomic_fetch_add_explicit(&counters[4], 1u, memory_order_relaxed);
  } else {
    for (uint i = 0u; i < payload_bytes; ++i) payload[payload_offset + i] = 0u;
    const uint plane_count = tier;
    for (uint within = 0u; within < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++within) {
      const float z = sigma == 0.0f ? 0.0f :
                      (input[input_base + within] - mean) / sigma;
      const uint raw = uint(hy_nomas_cache_qbit_nearest_code(z, codebook));
      const uint byte_index = hy_nomas_cache_qbit_plane_byte(within);
      const uchar bit_mask = hy_nomas_cache_qbit_plane_mask(within);
      for (uint plane = 0u; plane < plane_count; ++plane) {
        if ((raw & (1u << (7u - plane))) != 0u) {
          payload[payload_offset + plane * HY_NOMAS_CACHE_QBIT_PLANE_BYTES + byte_index] |= bit_mask;
        }
      }
    }
    if (tier == 4u) {
      atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
    } else if (tier == 5u) {
      atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
    } else {
      atomic_fetch_add_explicit(&counters[3], 1u, memory_order_relaxed);
    }
  }
  if (invalid_row) atomic_fetch_add_explicit(&counters[5], 1u, memory_order_relaxed);
}

kernel void hy_nomas_cache_qbit_mutate_control(
    device uint *headers [[buffer(0)]],
    device uchar *payload [[buffer(1)]],
    constant uint &header_row [[buffer(2)]],
    constant uint &bitplane_row [[buffer(3)]],
    constant float &header_delta [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0u) return;
  const uint header_base = header_row * 4u;
  const float mean = as_type<float>(headers[header_base]);
  headers[header_base] = as_type<uint>(mean + header_delta);

  const uint bitplane_base = bitplane_row * 4u;
  const uint tier = headers[bitplane_base + 3u];
  if (tier == 4u || tier == 5u || tier == 8u) {
    const uint payload_offset = headers[bitplane_base + 2u];
    // For the codec's 128-value transposed order, byte 15 bit 0 is the first
    // value's most significant plane bit.
    payload[payload_offset + 15u] ^= 1u;
  }
}

kernel void hy_nomas_cache_qbit_unpack(
    device float *output [[buffer(0)]],
    device const uint *headers [[buffer(1)]],
    device const uchar *payload [[buffer(2)]],
    device const float *codebook [[buffer(3)]],
    constant uint &row_count [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= row_count) return;
  const uint input_base = row * HY_NOMAS_CACHE_QBIT_ROW_VALUES;
  const uint header_base = row * 4u;
  const float mean = as_type<float>(headers[header_base]);
  const float sigma = as_type<float>(headers[header_base + 1u]);
  const uint payload_offset = headers[header_base + 2u];
  const uint tier = headers[header_base + 3u];

  if (tier == HY_NOMAS_CACHE_QBIT_F32_TIER) {
    for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
      const uint byte_offset = payload_offset + i * 4u;
      const uint bits = uint(payload[byte_offset]) |
                        (uint(payload[byte_offset + 1u]) << 8u) |
                        (uint(payload[byte_offset + 2u]) << 16u) |
                        (uint(payload[byte_offset + 3u]) << 24u);
      output[input_base + i] = as_type<float>(bits);
    }
    return;
  }

  const uint lut_offset = tier == 4u ? 128u : tier == 5u ? 384u : 640u;
  for (uint within = 0u; within < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++within) {
    if (sigma == 0.0f) {
      output[input_base + within] = mean;
      continue;
    }
    uint raw = 0u;
    const uint byte_index = hy_nomas_cache_qbit_plane_byte(within);
    const uchar bit_mask = hy_nomas_cache_qbit_plane_mask(within);
    for (uint plane = 0u; plane < tier; ++plane) {
      const uint byte_offset = payload_offset + plane * HY_NOMAS_CACHE_QBIT_PLANE_BYTES + byte_index;
      if ((payload[byte_offset] & bit_mask) != 0u) raw |= 1u << (7u - plane);
    }
    output[input_base + within] = mean + sigma * codebook[lut_offset + raw];
  }
}

// Compact-checkpoint writer. Each dispatch row owns one 512-byte F32 slot.
// The selected codes are fully materialized in private storage before any
// plane byte overwrites that row's source values.
kernel void hy_nomas_cache_qbit_pack_checkpoint(
    device float *input [[buffer(0)]],
    device uint *headers [[buffer(1)]],
    device atomic_uint *counters [[buffer(2)]],
    device const float *codebook [[buffer(3)]],
    constant uint &row_count [[buffer(4)]],
    constant uint &header_row_base [[buffer(5)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= row_count) return;

  const uint input_base = row * HY_NOMAS_CACHE_QBIT_ROW_VALUES;
  bool finite_row = true;
  bool constant_bits = true;
  const uint first_bits = as_type<uint>(input[input_base]);
  const float first_value = input[input_base];
  float mean = 0.0f;
  for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
    const float value = input[input_base + i];
    if (!isfinite(value)) {
      finite_row = false;
    } else {
      mean += value * (1.0f / 128.0f);
    }
    if (as_type<uint>(value) != first_bits) constant_bits = false;
  }

  uint tier = HY_NOMAS_CACHE_QBIT_F32_TIER;
  float sigma = 0.0f;
  bool invalid_row = !finite_row;
  bool moments_valid = finite_row && isfinite(mean);
  if (moments_valid) {
    if (constant_bits) {
      mean = first_value;
      sigma = 0.0f;
      tier = 4u;
    } else {
      float scale = 0.0f;
      for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
        const float delta = input[input_base + i] - mean;
        if (!isfinite(delta)) moments_valid = false;
        scale = max(scale, fabs(delta));
      }
      if (moments_valid && scale > 0.0f) {
        float square_sum = 0.0f;
        for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
          const float normalized_delta = (input[input_base + i] - mean) / scale;
          square_sum += normalized_delta * normalized_delta;
        }
        sigma = scale * sqrt(square_sum * (1.0f / 128.0f));
        moments_valid = isfinite(sigma) && sigma > 0.0f;
      } else {
        moments_valid = false;
      }
    }
  }

  if (moments_valid && sigma > 0.0f) {
    float max4 = 0.0f, max5 = 0.0f, max8 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum8 = 0.0f;
    bool normalized_finite = true;
    for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
      const float z = (input[input_base + i] - mean) / sigma;
      if (!isfinite(z)) { normalized_finite = false; break; }
      const uchar raw = hy_nomas_cache_qbit_nearest_code(z, codebook);
      const uint code = uint(raw);
      const float e4 = fabs(z - codebook[128u + code]);
      const float e5 = fabs(z - codebook[384u + code]);
      const float e8 = fabs(z - codebook[640u + code]);
      max4 = max(max4, e4); max5 = max(max5, e5); max8 = max(max8, e8);
      sum4 += e4 * e4; sum5 += e5 * e5; sum8 += e8 * e8;
    }
    if (normalized_finite) {
      const float rms4 = sqrt(sum4 * (1.0f / 128.0f));
      const float rms5 = sqrt(sum5 * (1.0f / 128.0f));
      const float rms8 = sqrt(sum8 * (1.0f / 128.0f));
      if (max4 <= 0.50f && rms4 <= 0.12f) tier = 4u;
      else if (max5 <= 0.25f && rms5 <= 0.06f) tier = 5u;
      else if (max8 <= 0.10f && rms8 <= 0.02f) tier = 8u;
    }
  }

  if (!moments_valid) {
    mean = 0.0f;
    sigma = 0.0f;
    tier = HY_NOMAS_CACHE_QBIT_F32_TIER;
  }

  const uint payload_bytes = tier == 4u ? 64u :
                             tier == 5u ? 80u :
                             tier == 8u ? 128u : 512u;
  const uint header_base = (header_row_base + row) * 4u;
  headers[header_base] = as_type<uint>(mean);
  headers[header_base + 1u] = as_type<uint>(sigma);
  headers[header_base + 2u] = input_base * sizeof(float);
  headers[header_base + 3u] = tier;

  if (tier != HY_NOMAS_CACHE_QBIT_F32_TIER) {
    uchar raw_codes[HY_NOMAS_CACHE_QBIT_ROW_VALUES];
    // This pass finishes all source reads before the row slot is overwritten.
    for (uint i = 0u; i < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++i) {
      const float z = sigma == 0.0f ? 0.0f : (input[input_base + i] - mean) / sigma;
      raw_codes[i] = hy_nomas_cache_qbit_nearest_code(z, codebook);
    }
    device uchar *payload = reinterpret_cast<device uchar *>(input);
    const uint payload_offset = input_base * sizeof(float);
    for (uint i = 0u; i < payload_bytes; ++i) payload[payload_offset + i] = 0u;
    for (uint within = 0u; within < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++within) {
      const uint raw = uint(raw_codes[within]);
      const uint byte_index = hy_nomas_cache_qbit_plane_byte(within);
      const uchar bit_mask = hy_nomas_cache_qbit_plane_mask(within);
      for (uint plane = 0u; plane < tier; ++plane) {
        if ((raw & (1u << (7u - plane))) != 0u) {
          payload[payload_offset + plane * HY_NOMAS_CACHE_QBIT_PLANE_BYTES + byte_index] |= bit_mask;
        }
      }
    }
  }

  atomic_fetch_add_explicit(&counters[0], payload_bytes, memory_order_relaxed);
  if (tier == 4u) atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
  else if (tier == 5u) atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
  else if (tier == 8u) atomic_fetch_add_explicit(&counters[3], 1u, memory_order_relaxed);
  else atomic_fetch_add_explicit(&counters[4], 1u, memory_order_relaxed);
  if (invalid_row) atomic_fetch_add_explicit(&counters[5], 1u, memory_order_relaxed);
}

// Fresh-process restore uses the same 512-byte row slots. Decode all raw code
// bytes to private storage before expanding the row over the packed prefix.
kernel void hy_nomas_cache_qbit_unpack_checkpoint(
    device float *output [[buffer(0)]],
    device const uint *headers [[buffer(1)]],
    device const float *codebook [[buffer(2)]],
    constant uint &row_count [[buffer(3)]],
    constant uint &header_row_base [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= row_count) return;
  const uint input_base = row * HY_NOMAS_CACHE_QBIT_ROW_VALUES;
  const uint header_base = (header_row_base + row) * 4u;
  const float mean = as_type<float>(headers[header_base]);
  const float sigma = as_type<float>(headers[header_base + 1u]);
  const uint payload_offset = headers[header_base + 2u];
  const uint tier = headers[header_base + 3u];
  if (tier == HY_NOMAS_CACHE_QBIT_F32_TIER) return;

  device uchar *payload = reinterpret_cast<device uchar *>(output);
  uchar raw_codes[HY_NOMAS_CACHE_QBIT_ROW_VALUES];
  for (uint within = 0u; within < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++within) {
    const uint byte_index = hy_nomas_cache_qbit_plane_byte(within);
    const uchar bit_mask = hy_nomas_cache_qbit_plane_mask(within);
    uint raw = 0u;
    for (uint plane = 0u; plane < tier; ++plane) {
      if ((payload[payload_offset + plane * HY_NOMAS_CACHE_QBIT_PLANE_BYTES + byte_index] & bit_mask) != 0u) {
        raw |= 1u << (7u - plane);
      }
    }
    raw_codes[within] = uchar(raw);
  }

  const uint lut_offset = tier == 4u ? 128u : tier == 5u ? 384u : 640u;
  for (uint within = 0u; within < HY_NOMAS_CACHE_QBIT_ROW_VALUES; ++within) {
    if (sigma == 0.0f) output[input_base + within] = mean;
    else output[input_base + within] = mean + sigma * codebook[lut_offset + uint(raw_codes[within])];
  }
}
