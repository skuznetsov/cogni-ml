#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Probe-only F32 attention kernel. S SIMD groups process eight queries each
// for the same batch/head while sharing one 32x128 K tile and one V tile.
template <uint S>
inline void wan_bounded_shared_kv_attention_impl(
    device const float *q, device const float *k, device const float *v,
    device float *out, constant uint *p,
    threadgroup float *key_tile, threadgroup float *value_tile,
    uint local_tile, uint simdgroup, uint lane) {
  constexpr uint D = 128;
  constexpr uint TILE_KEYS = 32;
  constexpr uint TILE_ELEMENTS = TILE_KEYS * D;

  const uint nq = p[0], nk = p[1], heads = p[2], channels = heads * D;
  // Restore the caller's global tile after segmented dispatch.
  const uint tile = local_tile + p[4];
  const uint query_groups = (nq + 8u * S - 1u) / (8u * S);
  const uint head = tile % heads;
  const uint batch = tile / (heads * query_groups);
  const uint query_group_base = ((tile / heads) % query_groups) * (8u * S);
  const ulong qb = ulong(batch) * nq * channels;
  const ulong kb0 = ulong(batch) * nk * channels;

  const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
  const uint fn = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  const uint qr = query_group_base + simdgroup * 8u + fm;
  const uint tid = simdgroup * 32u + lane;
  const float scale = as_type<float>(p[3]);

  // Match the Q-fragment-hoist probe: capture each lane's two adjacent values
  // before the ascending key loop and reuse them for every 32-key tile.
  float2 qfragments[D / 8];
  for (uint dd = 0; dd < D / 8; ++dd) {
    const uint qc = dd * 8u + fn;
    const ulong qindex = qb + ulong(qr) * channels + head * D + qc;
    qfragments[dd] = float2(
        qr < nq ? q[qindex] : 0.f,
        qr < nq ? q[qindex + 1u] : 0.f);
  }

  float2 acco[D / 8];
  for (uint d = 0; d < D / 8; ++d) acco[d] = float2(0.f);
  float maximum = -MAXFLOAT, norm = 0.f;

  for (uint kb = 0; kb < nk; kb += TILE_KEYS) {
    // All S SIMD groups cooperatively stage the same head's K/V tile.
    // Invalid tail keys are zero-filled, but still participate in barriers.
    for (uint index = tid; index < TILE_ELEMENTS; index += S * 32u) {
      const uint key_in_tile = index / D;
      const uint dim = index % D;
      const uint key_row = kb + key_in_tile;
      const ulong source = kb0 + ulong(key_row) * channels + head * D + dim;
      const float kval = key_row < nk ? k[source] : 0.f;
      const float vval = key_row < nk ? v[source] : 0.f;
      key_tile[index] = kval;
      value_tile[index] = vval;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float2 prob[4];
    float new_max = maximum;
    for (uint sk = 0; sk < 4; ++sk) {
      simdgroup_float8x8 score(0.f);
      for (uint dd = 0; dd < D / 8; ++dd) {
        simdgroup_float8x8 aq, bk;
        for (uint j = 0; j < 2; ++j) {
          const uint key_in_tile = sk * 8u + fn + j;
          const uint key_dim = dd * 8u + fm;
          aq.thread_elements()[j] = qfragments[dd][j];
          bk.thread_elements()[j] = key_tile[key_in_tile * D + key_dim];
        }
        simdgroup_multiply_accumulate(score, aq, bk, score);
      }
      for (uint j = 0; j < 2; ++j) {
        const uint kr = kb + sk * 8u + fn + j;
        prob[sk][j] = qr < nq && kr < nk ? score.thread_elements()[j] * scale : -MAXFLOAT;
      }
      float local = max(prob[sk].x, prob[sk].y);
      local = max(local, simd_shuffle_xor(local, ushort(1)));
      local = max(local, simd_shuffle_xor(local, ushort(8)));
      new_max = max(new_max, local);
    }

    float sum = 0.f;
    for (uint sk = 0; sk < 4; ++sk) {
      prob[sk] = precise::exp(prob[sk] - new_max);
      float local = prob[sk].x + prob[sk].y;
      local = local + simd_shuffle_xor(local, ushort(1));
      local = local + simd_shuffle_xor(local, ushort(8));
      sum = sum + local;
    }
    const float factor = precise::exp(maximum - new_max);
    maximum = new_max;
    norm = norm * factor + sum;

    for (uint dd = 0; dd < D / 8; ++dd) {
      simdgroup_float8x8 output;
      output.thread_elements()[0] = acco[dd].x * factor;
      output.thread_elements()[1] = acco[dd].y * factor;
      for (uint sk = 0; sk < 4; ++sk) {
        simdgroup_float8x8 a, b;
        a.thread_elements()[0] = prob[sk].x;
        a.thread_elements()[1] = prob[sk].y;
        for (uint j = 0; j < 2; ++j) {
          const uint value_key_in_tile = sk * 8u + fm;
          const uint value_dim = dd * 8u + fn + j;
          b.thread_elements()[j] = value_tile[value_key_in_tile * D + value_dim];
        }
        simdgroup_multiply_accumulate(output, a, b, output);
      }
      acco[dd] = float2(output.thread_elements()[0], output.thread_elements()[1]);
    }

    // Every SIMD group reaches this before the shared arrays are overwritten,
    // including groups whose eight query rows are all outside nq.
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (qr < nq) {
    for (uint dd = 0; dd < D / 8; ++dd)
      for (uint j = 0; j < 2; ++j)
        out[qb + ulong(qr) * channels + head * D + dd * 8u + fn + j] = acco[dd][j] / norm;
  }
}

#define WAN_BOUNDED_SHARED_KV_ATTENTION(S, NAME) \
kernel void NAME(device const float *q [[buffer(0)]], \
    device const float *k [[buffer(1)]], device const float *v [[buffer(2)]], \
    device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], \
    uint local_tile [[threadgroup_position_in_grid]], \
    uint lane [[thread_index_in_simdgroup]], \
    uint simdgroup [[simdgroup_index_in_threadgroup]]) { \
  threadgroup float key_tile[32 * 128]; \
  threadgroup float value_tile[32 * 128]; \
  wan_bounded_shared_kv_attention_impl<S>(q, k, v, out, p, key_tile, value_tile, \
      local_tile, simdgroup, lane); \
}

WAN_BOUNDED_SHARED_KV_ATTENTION(2, wan_bounded_shared_kv_attention_128_sg2)
WAN_BOUNDED_SHARED_KV_ATTENTION(4, wan_bounded_shared_kv_attention_128_sg4)
