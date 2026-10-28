#include <metal_stdlib>
using namespace metal;

inline float pgen_bf16(ushort bits) { return as_type<float>(uint(bits) << 16u); }
inline ushort pgen_round(float x) {
  uint bits = as_type<uint>(x);
  return ushort((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
kernel void pgen_probe_window(device ushort *window [[buffer(0)]],
  constant uint &which [[buffer(1)]], constant uint &count [[buffer(2)]],
  uint i [[thread_position_in_grid]]) {
  if (i < count) window[i] = pgen_round(float(int((i * 37u + which * 911u) % 8192u) - 4096) / 1024.0f);
}
kernel void pgen_probe_compare_bf16(device const ushort *actual [[buffer(0)]],
  device const ushort *expected [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
  constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  float a = pgen_bf16(actual[i]), b = pgen_bf16(expected[i]);
  bool finite = (as_type<uint>(a) & 0x7f800000u) != 0x7f800000u &&
                (as_type<uint>(b) & 0x7f800000u) != 0x7f800000u;
  float error = abs(a - b);
  if (!finite || error > 2e-5f + 2e-5f * abs(b)) atomic_fetch_add_explicit(status, 1u, memory_order_relaxed);
  if (finite) atomic_fetch_max_explicit(status + 1, as_type<uint>(error), memory_order_relaxed);
}
kernel void pgen_probe_compare_f32(device const float *actual [[buffer(0)]],
  device const float *expected [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
  constant uint &count [[buffer(3)]], device atomic_uint *frames [[buffer(4)]],
  uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  float a = actual[i], b = expected[i];
  bool finite = (as_type<uint>(a) & 0x7f800000u) != 0x7f800000u &&
                (as_type<uint>(b) & 0x7f800000u) != 0x7f800000u;
  float error = abs(a - b);
  if (!finite || error > 2e-5f + 2e-5f * abs(b)) {
    atomic_fetch_add_explicit(status, 1u, memory_order_relaxed);
    if (count > 1u) {
      uint frame = (i / (288u * 512u)) % 120u;
      atomic_fetch_add_explicit(frames + frame * 4u, 1u, memory_order_relaxed);
      atomic_fetch_min_explicit(frames + frame * 4u + 1u, i, memory_order_relaxed);
    }
  }
  if (finite) atomic_fetch_max_explicit(status + 1, as_type<uint>(error), memory_order_relaxed);
}
kernel void pgen_probe_sample_frames(device const float *actual [[buffer(0)]],
  device const float *expected [[buffer(1)]], device uint *frames [[buffer(2)]],
  uint frame [[thread_position_in_grid]]) {
  if (frame >= 120u || frames[frame * 4u] == 0u) return;
  uint i = frames[frame * 4u + 1u];
  frames[frame * 4u + 2u] = as_type<uint>(actual[i]);
  frames[frame * 4u + 3u] = as_type<uint>(expected[i]);
}
kernel void pgen_probe_nan(device float *actual [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (i == 0) actual[0] = as_type<float>(0x7fc00000u);
}
