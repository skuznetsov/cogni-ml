// Diagnostic only: canonical decoder, coefficient comparator and reduction controls.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void hy_q4_round_input(device const float *source [[buffer(0)]],
    device half *h16 [[buffer(1)]], device float *f32 [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid < count) { h16[gid] = half(source[gid]); f32[gid] = float(h16[gid]); }
}

kernel void hy_q4_canonical(device const uchar *packed [[buffer(0)]],
    device half *destination [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  const uint within = gid % HY_NOMAS_Q4_BLOCK_ELEMENTS;
  device const uchar *block = packed + (gid / HY_NOMAS_Q4_BLOCK_ELEMENTS) * HY_NOMAS_Q4_BLOCK_BYTES;
  const float d = hy_nomas_q4_load_f16_le(block);
  const float dmin = hy_nomas_q4_load_f16_le(block + 2u);
  const uint group = within / 64u;
  const uint rem = within % 64u;
  const uchar2 sm = hy_nomas_q4_scale_min(group * 2u + (rem >= 32u ? 1u : 0u), block + 4u);
  const uchar byte = block[16u + group * 32u + (rem & 31u)];
  const uint q = rem < 32u ? byte & 15u : byte >> 4u;
  const float ds = d * float(sm.x);
  const float dm = dmin * float(sm.y);
  destination[gid] = half(ds * float(q) - dm);
}

// Initialized on the host before encoding. All coefficient comparisons stay on GPU.
// Slots: mismatches, canonical/helper nonfinite, first mismatch, low/high nibble
// mismatches, two reserved words, compared blocks, completed. The same comparison
// executes real, identical-buffer and injected-single-bit controls.
kernel void hy_q4_compare(device const half *canonical [[buffer(0)]],
    device const half *helper [[buffer(1)]], device atomic_uint *stats [[buffer(2)]],
    constant uint &count [[buffer(3)]], constant uint &injected_index [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= count) return;
  ushort a = as_type<ushort>(canonical[gid]);
  if (gid == injected_index) a ^= 1u;
  const ushort b = as_type<ushort>(helper[gid]);
  if (a != b) {
    atomic_fetch_add_explicit(stats, 1u, memory_order_relaxed);
    atomic_fetch_min_explicit(stats + 3, gid, memory_order_relaxed);
    atomic_fetch_add_explicit(stats + ((gid % 64u) < 32u ? 4 : 5), 1u, memory_order_relaxed);
  }
  if (!isfinite(canonical[gid])) atomic_fetch_add_explicit(stats + 1, 1u, memory_order_relaxed);
  if (!isfinite(helper[gid])) atomic_fetch_add_explicit(stats + 2, 1u, memory_order_relaxed);
  if (gid % 256u == 0u) atomic_fetch_add_explicit(stats + 8, 1u, memory_order_relaxed);
  if (gid == count - 1u) atomic_store_explicit(stats + 9, 0x51443431u, memory_order_relaxed);
}

// Contiguous K prefixes for MPS; input/weight source strides remain full K.
kernel void hy_q4_pack_prefix(device const half *weights [[buffer(0)]],
    device const half *input [[buffer(1)]], device float *weight_stage [[buffer(2)]],
    device float *input_stage [[buffer(3)]], constant uint &full_k [[buffer(4)]],
    constant uint &k [[buffer(5)]], constant uint &n [[buffer(6)]],
    constant uint &m [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n * k) weight_stage[gid] = float(weights[(gid / k) * full_k + gid % k]);
  if (gid < m * k) input_stage[gid] = float(input[(gid / k) * full_k + gid % k]);
}

kernel void hy_q4_copy(device const uint *source [[buffer(0)]],
    device uint *destination [[buffer(1)]], constant uint &count [[buffer(2)]],
    constant uint &offset [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
  if (gid < count) destination[offset + gid] = source[gid];
}

// F16 products fit in F32; Neumaier compensation limits accumulation error.
// This diagnostic is not a proof of a correctly rounded result.
kernel void hy_q4_compensated(device const half *weights [[buffer(0)]],
    device const half *input [[buffer(1)]], device float *output [[buffer(2)]],
    constant uint &k [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &m [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= m * n) return;
  const uint row = gid / n, col = gid % n;
  float sum = 0.0f, correction = 0.0f;
  for (uint j = 0; j < k; ++j) {
    const float p = float(weights[col * k + j]) * float(input[row * k + j]);
    const float next = sum + p;
    correction += abs(sum) >= abs(p) ? (sum - next) + p : (p - next) + sum;
    sum = next;
  }
  output[gid] = sum + correction;
}
