// Adapted from PyTorch v2.8.0 sdpa_vector (Attention.metal), tag commit
// ba56102387ef21a3b04b357e5b183d48f0afefc7. Upstream file SHA-256:
// aaab42c45caaa222c62013af8dcac896ffa2abd889dbb9427c7648be6448c0cd.
// That implementation credits MLX scaled_dot_product_attention.metal.
// PyTorch BSD-3-Clause and Apple MLX MIT notices are retained under
// third_party/licenses and indexed by third_party/VIDEO_NOTICES.md.
// Changes: D=V=128, no mask/GQA, F32 BQHD bindings with explicit BF16
// input/output rounding, and a caller-selected compact query interval.
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

inline float hy_vector_bf16(float value) {
  uint bits = as_type<uint>(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u)
    return as_type<float>((bits & 0xffff0000u) | 0x00400000u);
  return as_type<float>((bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u);
}

kernel void hy_nomas_vector_attention_bf16(
    device const float *queries [[buffer(0)]],
    device const float *keys [[buffer(1)]],
    device const float *values [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint &input_query_rows [[buffer(4)]],
    constant uint &query_start [[buffer(5)]],
    constant uint &query_rows [[buffer(6)]],
    constant uint &N [[buffer(7)]],
    constant uint &heads [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr uint BN = 32;
  constexpr uint BD = 32;
  constexpr uint per_thread = 4;
  thread float q[per_thread];
  thread float k[per_thread];
  thread float o[per_thread];
  threadgroup float outputs[BN * BD];
  threadgroup float max_scores[BN];
  threadgroup float sum_exp_scores[BN];

  const uint batch = tid.x / heads;
  const uint head = tid.x % heads;
  const uint query = tid.y;
  queries += ((batch * input_query_rows + query_start + query) * heads + head) * 128
      + simd_lid * per_thread;
  keys += ((batch * N + simd_gid) * heads + head) * 128 + simd_lid * per_thread;
  values += ((batch * N + simd_gid) * heads + head) * 128 + simd_lid * per_thread;
  out += ((batch * query_rows + query) * heads + head) * 128 + simd_gid * per_thread;
  const uint inner_stride = BN * heads * 128;

  for (uint i = 0; i < per_thread; i++) q[i] = scale * hy_vector_bf16(queries[i]);
  for (uint i = 0; i < per_thread; i++) o[i] = 0;
  float max_score = -INFINITY;
  float sum_exp_score = 0;

  for (uint i = simd_gid; i < N; i += BN) {
    for (uint j = 0; j < per_thread; j++) k[j] = hy_vector_bf16(keys[j]);
    float score = 0;
    for (uint j = 0; j < per_thread; j++) score += q[j] * k[j];
    score = simd_sum(score);
    float new_max = max(max_score, score);
    float factor = metal::fast::exp(max_score - new_max);
    float exp_score = metal::fast::exp(score - new_max);
    max_score = new_max;
    sum_exp_score = sum_exp_score * factor + exp_score;
    for (uint j = 0; j < per_thread; j++)
      o[j] = o[j] * factor + exp_score * hy_vector_bf16(values[j]);
    keys += inner_stride;
    values += inner_stride;
  }

  if (simd_lid == 0) {
    max_scores[simd_gid] = max_score;
    sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = max_scores[simd_lid];
  float new_max = simd_max(max_score);
  float factor = metal::fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);
  for (uint i = 0; i < per_thread; i++) {
    outputs[simd_lid * BD + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float safe_sum = (sum_exp_score == 0 ? 1e-6f : sum_exp_score);
    o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor) / safe_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (simd_lid == 0)
    for (uint i = 0; i < per_thread; i++) out[i] = hy_vector_bf16(o[i]);
}
