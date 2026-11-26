#include <metal_stdlib>
using namespace metal;

// Keep the diagnostic's ordinary F32 additions separate. The native lane
// uses an explicit fma at every logical reduction step; the exact lane below
// never passes through floating-point arithmetic until its final encoding.
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

constant uint DCAE_EXACT_LIMBS = 34u;
constant int DCAE_EXACT_BASE_EXP = -272;
constant ushort DCAE_CANONICAL_BF16_NAN = 0x7fc0u;
constant uint DCAE_CANONICAL_F32_NAN = 0x7fc00000u;

struct DCAEExactBins {
  ulong positive[DCAE_EXACT_LIMBS];
  ulong negative[DCAE_EXACT_LIMBS];
};

inline float dcae_bf16_to_f32(ushort bits) {
  return as_type<float>(uint(bits) << 16u);
}

inline bool dcae_bf16_finite(ushort bits) {
  return ((uint(bits) >> 7u) & 0xffu) != 0xffu;
}

// IEEE round-to-nearest, ties-to-even from F32 to BF16. Preserve infinities
// and make every NaN quiet even when its payload is confined to F32's low bits.
inline ushort dcae_f32_to_bf16_rne(float value) {
  const uint bits = as_type<uint>(value);
  const uint exponent = (bits >> 23u) & 0xffu;
  const uint fraction = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    if (fraction == 0u) return ushort(bits >> 16u);
    return ushort((bits >> 16u) | 0x0040u);
  }
  return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

inline void dcae_bins_zero(thread DCAEExactBins& bins) {
  #pragma clang loop unroll(disable)
  for (uint i = 0u; i < DCAE_EXACT_LIMBS; ++i) {
    bins.positive[i] = 0ul;
    bins.negative[i] = 0ul;
  }
}

// Add a nonnegative integer shifted by an exact number of base-2 bits. The
// caller's range keeps every insertion below 2^32; the documented 13,824-term
// Conv3D maximum keeps every unnormalized limb below 2^46.
inline void dcae_bins_add_shifted(thread ulong* limbs, ulong value,
                                  uint bit_shift) {
  const uint limb = bit_shift >> 4u;
  const uint offset = bit_shift & 15u;
  if (value == 0ul || limb >= DCAE_EXACT_LIMBS) return;
  limbs[limb] += value << offset;
}

// BF16 x BF16 = (integer mantissa product) * 2^exponent. For normals the
// integer mantissa has its hidden 128 bit; for subnormals it is the raw
// fraction. Using max(exp,1) for both operands gives the common exact form
// m * 2^(max(exp,1)-134), including subnormals. The product exponent is thus
// in [-266,240], so relative to base exponent -272 every insertion fits the
// 34-limb accumulator. Returns false for either nonfinite operand.
inline bool dcae_exact_add_product(thread DCAEExactBins& bins, ushort x,
                                   ushort w) {
  if (!dcae_bf16_finite(x) || !dcae_bf16_finite(w)) return false;

  const uint ex = (uint(x) >> 7u) & 0xffu;
  const uint ew = (uint(w) >> 7u) & 0xffu;
  const uint mx = (uint(x) & 0x7fu) | (ex == 0u ? 0u : 0x80u);
  const uint mw = (uint(w) & 0x7fu) | (ew == 0u ? 0u : 0x80u);
  const ulong product = ulong(mx) * ulong(mw);
  const int exponent = int(max(ex, 1u)) + int(max(ew, 1u)) - 268;
  const uint bit_shift = uint(exponent - DCAE_EXACT_BASE_EXP);
  const bool negative = (((uint(x) ^ uint(w)) & 0x8000u) != 0u);
  if (negative)
    dcae_bins_add_shifted(bins.negative, product, bit_shift);
  else
    dcae_bins_add_shifted(bins.positive, product, bit_shift);
  return true;
}

// Add one finite BF16 value exactly, for the post-convolution BF16 bias
// operation. Its exponent relative to -272 is max(exp,1)+138.
inline bool dcae_exact_add_bf16(thread DCAEExactBins& bins, ushort value) {
  if (!dcae_bf16_finite(value)) return false;
  const uint exponent = (uint(value) >> 7u) & 0xffu;
  const uint mantissa = (uint(value) & 0x7fu) |
      (exponent == 0u ? 0u : 0x80u);
  const uint bit_shift = max(exponent, 1u) + 138u;
  if ((value & 0x8000u) != 0u)
    dcae_bins_add_shifted(bins.negative, ulong(mantissa), bit_shift);
  else
    dcae_bins_add_shifted(bins.positive, ulong(mantissa), bit_shift);
  return true;
}

inline void dcae_bins_normalize(thread ulong* limbs) {
  #pragma clang loop unroll(disable)
  for (uint i = 0u; i + 1u < DCAE_EXACT_LIMBS; ++i) {
    const ulong carry = limbs[i] >> 16u;
    limbs[i] &= 0xfffful;
    limbs[i + 1u] += carry;
  }
}

inline int dcae_bins_compare(thread const ulong* a, thread const ulong* b) {
  #pragma clang loop unroll(disable)
  for (int i = int(DCAE_EXACT_LIMBS) - 1; i >= 0; --i) {
    if (a[i] > b[i]) return 1;
    if (a[i] < b[i]) return -1;
  }
  return 0;
}

inline void dcae_bins_subtract(thread const ulong* larger,
                               thread const ulong* smaller,
                               thread ulong* result) {
  ulong borrow = 0ul;
  #pragma clang loop unroll(disable)
  for (uint i = 0u; i < DCAE_EXACT_LIMBS; ++i) {
    const ulong a = larger[i];
    const ulong b = smaller[i] + borrow;
    if (a >= b) {
      result[i] = a - b;
      borrow = 0ul;
    } else {
      result[i] = 0x10000ul + a - b;
      borrow = 1ul;
    }
  }
}

// Resolve separate positive and negative product bins to one unsigned
// magnitude, including exact cancellation. Output limbs are normalized base
// 2^16 digits, and negative is zero for an exact zero result.
inline void dcae_bins_finish(thread DCAEExactBins& bins,
                             thread ulong* magnitude,
                             thread uint& negative) {
  dcae_bins_normalize(bins.positive);
  dcae_bins_normalize(bins.negative);
  const int comparison = dcae_bins_compare(bins.positive, bins.negative);
  if (comparison > 0) {
    dcae_bins_subtract(bins.positive, bins.negative, magnitude);
    negative = 0u;
  } else if (comparison < 0) {
    dcae_bins_subtract(bins.negative, bins.positive, magnitude);
    negative = 1u;
  } else {
    #pragma clang loop unroll(disable)
    for (uint i = 0u; i < DCAE_EXACT_LIMBS; ++i) magnitude[i] = 0ul;
    negative = 0u;
  }
}

inline uint dcae_big_bit(thread const ulong* magnitude, uint bit) {
  if ((bit >> 4u) >= DCAE_EXACT_LIMBS) return 0u;
  return uint((magnitude[bit >> 4u] >> (bit & 15u)) & 1ul);
}

inline bool dcae_big_any_below(thread const ulong* magnitude, uint limit) {
  #pragma clang loop unroll(disable)
  for (uint bit = 0u; bit < limit; ++bit)
    if (dcae_big_bit(magnitude, bit) != 0u) return true;
  return false;
}

inline int dcae_big_highest_bit(thread const ulong* magnitude) {
  #pragma clang loop unroll(disable)
  for (int limb = int(DCAE_EXACT_LIMBS) - 1; limb >= 0; --limb) {
    if (magnitude[limb] == 0ul) continue;
    #pragma clang loop unroll(disable)
    for (int bit = 15; bit >= 0; --bit)
      if (((magnitude[limb] >> uint(bit)) & 1ul) != 0ul)
        return limb * 16 + bit;
  }
  return -1;
}

inline ulong dcae_round_integer_rne(thread const ulong* magnitude,
                                    uint shift, uint precision) {
  ulong significand = 0ul;
  #pragma clang loop unroll(disable)
  for (uint bit = 0u; bit < precision; ++bit)
    significand |= ulong(dcae_big_bit(magnitude, shift + bit)) << bit;
  if (shift != 0u) {
    const bool guard = dcae_big_bit(magnitude, shift - 1u) != 0u;
    const bool sticky = dcae_big_any_below(magnitude, shift - 1u);
    if (guard && (sticky || (significand & 1ul) != 0ul)) ++significand;
  }
  return significand;
}

// Round the exact magnitude * 2^-272 directly to a binary interchange type.
// BF16 uses precision=8 and F32 uses precision=24. Normal and subnormal paths
// both implement guard/sticky/tie-even rounding; the direct BF16 path avoids
// double rounding through F32. Returns encoded bits in the low uint16/uint32.
inline ulong dcae_round_exact_binary(thread const ulong* magnitude,
                                    uint negative, uint precision,
                                    int min_exponent, int max_exponent,
                                    uint exponent_bits) {
  const int highest = dcae_big_highest_bit(magnitude);
  if (highest < 0) return 0ul;

  const uint fraction_bits = precision - 1u;
  const ulong sign = negative != 0u
      ? (1ul << (fraction_bits + exponent_bits)) : 0ul;
  const int unbiased = highest + DCAE_EXACT_BASE_EXP;
  if (unbiased < min_exponent) {
    const int quantum_exponent = min_exponent - int(fraction_bits);
    const uint shift = uint(quantum_exponent - DCAE_EXACT_BASE_EXP);
    const ulong fraction = dcae_round_integer_rne(magnitude, shift, precision);
    const ulong min_normal_significand = 1ul << fraction_bits;
    if (fraction >= min_normal_significand)
      return sign | min_normal_significand;  // rounded up to exponent field 1
    return sign | fraction;
  }

  const uint shift = uint(highest - int(fraction_bits));
  ulong significand = dcae_round_integer_rne(magnitude, shift, precision);
  int exponent = unbiased;
  if (significand == (1ul << precision)) {
    significand >>= 1u;
    ++exponent;
  }

  const ulong exponent_mask = (1ul << exponent_bits) - 1ul;
  if (exponent > max_exponent)
    return sign | (exponent_mask << fraction_bits);

  const int bias = int((1u << (exponent_bits - 1u)) - 1u);
  const ulong encoded_exponent = ulong(exponent + bias);
  const ulong fraction = significand - (1ul << fraction_bits);
  return sign | (encoded_exponent << fraction_bits) | fraction;
}

inline uint dcae_exact_f32_bits(thread const ulong* magnitude,
                                uint negative) {
  return uint(dcae_round_exact_binary(magnitude, negative, 24u,
      -126, 127, 8u));
}

inline ushort dcae_exact_bf16_bits(thread const ulong* magnitude,
                                  uint negative) {
  return ushort(dcae_round_exact_binary(magnitude, negative, 8u,
      -126, 127, 8u));
}

// Add two BF16 values with one exact BF16 rounding boundary. Inputs that are
// NaNs produce a canonical NaN; infinities follow IEEE addition, including
// opposite-signed infinity cancellation. Finite values use the same exact
// bins and direct rounding as the scalar reference kernel.
inline ushort dcae_bf16_add_rne(ushort a, ushort b) {
  const uint ea = (uint(a) >> 7u) & 0xffu;
  const uint eb = (uint(b) >> 7u) & 0xffu;
  const uint fa = uint(a) & 0x7fu;
  const uint fb = uint(b) & 0x7fu;
  const bool a_special = ea == 0xffu;
  const bool b_special = eb == 0xffu;
  if ((a_special && fa != 0u) || (b_special && fb != 0u))
    return DCAE_CANONICAL_BF16_NAN;
  if (a_special && b_special && ((a ^ b) & 0x8000u) != 0u)
    return DCAE_CANONICAL_BF16_NAN;
  if (a_special) return a;
  if (b_special) return b;

  if ((a & 0x7fffu) == 0u && (b & 0x7fffu) == 0u)
    return ((a & b & 0x8000u) != 0u) ? ushort(0x8000u) : ushort(0u);

  DCAEExactBins bins;
  dcae_bins_zero(bins);
  dcae_exact_add_bf16(bins, a);
  dcae_exact_add_bf16(bins, b);
  ulong magnitude[DCAE_EXACT_LIMBS];
  uint negative = 0u;
  dcae_bins_finish(bins, magnitude, negative);
  return dcae_exact_bf16_bits(magnitude, negative);
}

inline void dcae_store_nan(device float* raw_f32,
                           device ushort* pre_bf16,
                           device ushort* post_bf16, uint sample) {
  raw_f32[sample] = as_type<float>(DCAE_CANONICAL_F32_NAN);
  pre_bf16[sample] = DCAE_CANONICAL_BF16_NAN;
  post_bf16[sample] = DCAE_CANONICAL_BF16_NAN;
}

// Input is [1,C,T,H,W] BF16; cache is [C,H,W] BF16; weights are
// [C_out=C,C_in=C,3,3,3] in OI,KT,KH,KW order; bias is [C]. `indices` contains
// N linear NCDHW output offsets in [0,C*T*H*W), and every output buffer is
// indexed by sample ordinal (not by that full-output offset). At each output,
// kt/kh/kw address offsets -1..1 around the output coordinate. Spatial and
// right-temporal out-of-range inputs are zero; temporal offset -1 reads cache
// only when cachevalid is one. A padded term still consumes its logical
// reduction step and lane, preserving the requested grouping.
//
// params[0..8] = C,T,H,W,N,cachevalid,order,lanes,tree. Order 0 visits
// ic then kt/kh/kw; order 1 visits kt/kh/kw then ic. Lanes must be one of
// 1,2,4,8,16,32. Tree 0 sums lanes serially; tree 1 uses an adjacent balanced
// binary tree. Stage2's admitted geometry is [1,512,40,72,128], K=13,824;
// this sampled kernel is diagnostic and does not change model defaults.
kernel void dcae_reduce_native(
    device const ushort* x_bf16 [[buffer(0)]],
    device const ushort* cache_bf16 [[buffer(1)]],
    device const ushort* weight_bf16 [[buffer(2)]],
    device const uint* indices [[buffer(3)]],
    device float* raw_f32 [[buffer(4)]],
    device ushort* pre_bf16 [[buffer(5)]],
    device ushort* post_bf16 [[buffer(6)]],
    device const ushort* bias_bf16 [[buffer(7)]],
    constant uint* params [[buffer(8)]],
    uint sample [[thread_position_in_grid]]) {
  const uint C = params[0];
  const uint T = params[1];
  const uint H = params[2];
  const uint W = params[3];
  const uint N = params[4];
  const uint cachevalid = params[5];
  const uint order = params[6];
  const uint lanes = params[7];
  const uint tree = params[8];
  if (sample >= N) return;

  const bool valid_lanes = lanes == 1u || lanes == 2u || lanes == 4u ||
      lanes == 8u || lanes == 16u || lanes == 32u;
  if (C == 0u || T == 0u || H == 0u || W == 0u || cachevalid > 1u ||
      order > 1u || !valid_lanes || tree > 1u) {
    dcae_store_nan(raw_f32, pre_bf16, post_bf16, sample);
    return;
  }

  const ulong plane = ulong(H) * ulong(W);
  const ulong frame_extent = ulong(T) * plane;
  const ulong output_extent = ulong(C) * frame_extent;
  const uint output_linear = indices[sample];
  if (ulong(output_linear) >= output_extent) {
    dcae_store_nan(raw_f32, pre_bf16, post_bf16, sample);
    return;
  }

  const uint output_channel = uint(ulong(output_linear) / frame_extent);
  const ulong in_frame = ulong(output_linear) % frame_extent;
  const uint output_t = uint(in_frame / plane);
  const uint output_y = uint((in_frame % plane) / ulong(W));
  const uint output_x = uint(in_frame % ulong(W));
  const uint K = C * 27u;
  float lane_sum[32];
  #pragma clang loop unroll(disable)
  for (uint lane = 0u; lane < lanes; ++lane) lane_sum[lane] = 0.0f;

  #pragma clang loop unroll(disable)
  for (uint step = 0u; step < K; ++step) {
    uint ic;
    uint kt;
    uint kh;
    uint kw;
    if (order == 0u) {
      ic = step / 27u;
      const uint kernel_index = step % 27u;
      kt = kernel_index / 9u;
      kh = (kernel_index / 3u) % 3u;
      kw = kernel_index % 3u;
    } else {
      const uint kernel_index = step / C;
      ic = step % C;
      kt = kernel_index / 9u;
      kh = (kernel_index / 3u) % 3u;
      kw = kernel_index % 3u;
    }

    const int input_t = int(output_t) + int(kt) - 1;
    const int input_y = int(output_y) + int(kh) - 1;
    const int input_x = int(output_x) + int(kw) - 1;
    float input_value = 0.0f;
    if (input_y >= 0 && input_y < int(H) &&
        input_x >= 0 && input_x < int(W)) {
      if (input_t < 0) {
        if (cachevalid != 0u) {
          const ulong cache_index = ulong(ic) * plane +
              ulong(input_y) * ulong(W) + ulong(input_x);
          input_value = dcae_bf16_to_f32(cache_bf16[cache_index]);
        }
      } else if (input_t < int(T)) {
        const ulong input_index = ulong(ic) * frame_extent +
            ulong(input_t) * plane + ulong(input_y) * ulong(W) + ulong(input_x);
        input_value = dcae_bf16_to_f32(x_bf16[input_index]);
      }
    }

    const ulong weight_index = (((ulong(output_channel) * ulong(C) + ulong(ic)) *
        3ul + ulong(kt)) * 3ul + ulong(kh)) * 3ul + ulong(kw);
    const float weight_value = dcae_bf16_to_f32(weight_bf16[weight_index]);
    const uint lane = step % lanes;
    lane_sum[lane] = metal::fma(input_value, weight_value, lane_sum[lane]);
  }

  float reduction = 0.0f;
  if (tree == 0u) {
    #pragma clang loop unroll(disable)
    for (uint lane = 0u; lane < lanes; ++lane)
      reduction = reduction + lane_sum[lane];
  } else {
    #pragma clang loop unroll(disable)
    for (uint stride = 1u; stride < lanes; stride <<= 1u)
      #pragma clang loop unroll(disable)
      for (uint base = 0u; base < lanes; base += stride << 1u)
        lane_sum[base] = lane_sum[base] + lane_sum[base + stride];
    reduction = lane_sum[0];
  }

  const ushort pre = dcae_f32_to_bf16_rne(reduction);
  raw_f32[sample] = reduction;
  pre_bf16[sample] = pre;
  post_bf16[sample] = dcae_bf16_add_rne(pre, bias_bf16[output_channel]);
}

// Exact integer reference for the same sampled Conv3D mapping and buffers as
// dcae_reduce_native. It consumes params[0..5] only; order/lanes/tree cannot
// affect an exact sum. The per-sample K=13,824 reduction has positive and
// negative 34-limb radix-2^16 bins, avoiding cancellation loss. It is for at
// most 8,192 sampled outputs, never a performance path.
kernel void dcae_reduce_exact(
    device const ushort* x_bf16 [[buffer(0)]],
    device const ushort* cache_bf16 [[buffer(1)]],
    device const ushort* weight_bf16 [[buffer(2)]],
    device const uint* indices [[buffer(3)]],
    device float* raw_f32 [[buffer(4)]],
    device ushort* pre_bf16 [[buffer(5)]],
    device ushort* post_bf16 [[buffer(6)]],
    device const ushort* bias_bf16 [[buffer(7)]],
    constant uint* params [[buffer(8)]],
    uint sample [[thread_position_in_grid]]) {
  const uint C = params[0];
  const uint T = params[1];
  const uint H = params[2];
  const uint W = params[3];
  const uint N = params[4];
  const uint cachevalid = params[5];
  if (sample >= N) return;

  if (C == 0u || T == 0u || H == 0u || W == 0u || cachevalid > 1u) {
    dcae_store_nan(raw_f32, pre_bf16, post_bf16, sample);
    return;
  }
  const ulong plane = ulong(H) * ulong(W);
  const ulong frame_extent = ulong(T) * plane;
  const ulong output_extent = ulong(C) * frame_extent;
  const uint output_linear = indices[sample];
  if (ulong(output_linear) >= output_extent) {
    dcae_store_nan(raw_f32, pre_bf16, post_bf16, sample);
    return;
  }

  const uint output_channel = uint(ulong(output_linear) / frame_extent);
  const ulong in_frame = ulong(output_linear) % frame_extent;
  const uint output_t = uint(in_frame / plane);
  const uint output_y = uint((in_frame % plane) / ulong(W));
  const uint output_x = uint(in_frame % ulong(W));
  DCAEExactBins bins;
  dcae_bins_zero(bins);
  bool finite = dcae_bf16_finite(bias_bf16[output_channel]);

  // Exact ordering is immaterial, but this ic-major order mirrors native
  // order=0 and keeps source addressing easy to audit.
  #pragma clang loop unroll(disable)
  for (uint ic = 0u; ic < C; ++ic) {
    #pragma clang loop unroll(disable)
    for (uint kt = 0u; kt < 3u; ++kt) {
      #pragma clang loop unroll(disable)
      for (uint kh = 0u; kh < 3u; ++kh) {
        #pragma clang loop unroll(disable)
        for (uint kw = 0u; kw < 3u; ++kw) {
          const int input_t = int(output_t) + int(kt) - 1;
          const int input_y = int(output_y) + int(kh) - 1;
          const int input_x = int(output_x) + int(kw) - 1;
          ushort input_bits = 0u;
          if (input_y >= 0 && input_y < int(H) &&
              input_x >= 0 && input_x < int(W)) {
            if (input_t < 0) {
              if (cachevalid != 0u) {
                const ulong cache_index = ulong(ic) * plane +
                    ulong(input_y) * ulong(W) + ulong(input_x);
                input_bits = cache_bf16[cache_index];
              }
            } else if (input_t < int(T)) {
              const ulong input_index = ulong(ic) * frame_extent +
                  ulong(input_t) * plane + ulong(input_y) * ulong(W) +
                  ulong(input_x);
              input_bits = x_bf16[input_index];
            }
          }

          const ulong weight_index = (((ulong(output_channel) * ulong(C) +
              ulong(ic)) * 3ul + ulong(kt)) * 3ul + ulong(kh)) * 3ul +
              ulong(kw);
          const ushort weight_bits = weight_bf16[weight_index];
          finite = dcae_exact_add_product(bins, input_bits, weight_bits) && finite;
        }
      }
    }
  }

  if (!finite) {
    dcae_store_nan(raw_f32, pre_bf16, post_bf16, sample);
    return;
  }

  ulong magnitude[DCAE_EXACT_LIMBS];
  uint negative = 0u;
  dcae_bins_finish(bins, magnitude, negative);
  const uint raw_bits = dcae_exact_f32_bits(magnitude, negative);
  const ushort pre = dcae_exact_bf16_bits(magnitude, negative);
  raw_f32[sample] = as_type<float>(raw_bits);
  pre_bf16[sample] = pre;
  post_bf16[sample] = dcae_bf16_add_rne(pre, bias_bf16[output_channel]);
}

// Tiny exact dot-vector fixtures. params[0]=K terms per row and params[1]=row
// count; x and w are contiguous [count,K] BF16 arrays, with independent rows.
// Each row produces the exact pre-bias sum rounded directly to F32 and BF16.
// These vectors cover rounding/subnormal/overflow edge cases without touching
// the real-size Conv3D input or weight buffers.
kernel void dcae_reduce_exact_vectors(
    device const ushort* x_bf16 [[buffer(0)]],
    device const ushort* w_bf16 [[buffer(1)]],
    device float* raw_f32 [[buffer(2)]],
    device ushort* pre_bf16 [[buffer(3)]],
    constant uint* params [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
  const uint K = params[0];
  const uint count = params[1];
  if (row >= count) return;

  DCAEExactBins bins;
  dcae_bins_zero(bins);
  bool finite = true;
  const ulong row_base = ulong(row) * ulong(K);
  #pragma clang loop unroll(disable)
  for (uint k = 0u; k < K; ++k)
    finite = dcae_exact_add_product(bins, x_bf16[row_base + ulong(k)],
        w_bf16[row_base + ulong(k)]) && finite;

  if (!finite) {
    raw_f32[row] = as_type<float>(DCAE_CANONICAL_F32_NAN);
    pre_bf16[row] = DCAE_CANONICAL_BF16_NAN;
    return;
  }

  ulong magnitude[DCAE_EXACT_LIMBS];
  uint negative = 0u;
  dcae_bins_finish(bins, magnitude, negative);
  raw_f32[row] = as_type<float>(dcae_exact_f32_bits(magnitude, negative));
  pre_bf16[row] = dcae_exact_bf16_bits(magnitude, negative);
}
