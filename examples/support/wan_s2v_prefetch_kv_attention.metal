#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Probe-only F32 D=128 attention. The next device fragment is requested before
// the current matrix multiply. The compiler decides the actual load schedule;
// source lookahead alone does not establish physical load/compute overlap.
// Keep one eight-query tile per SIMD group and the Q-cache arithmetic order.
kernel void wan_bounded_attention_prefetch_kv_128(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint local_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128;
  const uint nq = p[0], nk = p[1], heads = p[2], channels = heads * D;
  const uint tile = local_tile + p[4];
  const uint head = tile % heads;
  const uint query_tiles = (nq + 7) / 8;
  const uint batch = tile / (heads * query_tiles);
  const uint qbase = ((tile / heads) % query_tiles) * 8;
  const ulong qb = ulong(batch) * nq * channels;
  const ulong kb0 = ulong(batch) * nk * channels;
  const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
  const uint fn = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  const uint qr = qbase + fm;
  float2 qfragments[D / 8];
  for (uint dd = 0; dd < D / 8; ++dd) {
    const uint qc = dd * 8 + fn;
    qfragments[dd] = float2(
        qr < nq ? q[qb + ulong(qr) * channels + head * D + qc] : 0.f,
        qr < nq ? q[qb + ulong(qr) * channels + head * D + qc + 1] : 0.f);
  }
  const float scale = as_type<float>(p[3]);
  float2 acco[D / 8];
  for (uint dd = 0; dd < D / 8; ++dd) acco[dd] = float2(0.f);
  float maximum = -MAXFLOAT, norm = 0.f;
  for (uint kb = 0; kb < nk; kb += 32) {
    float2 prob[4];
    float new_max = maximum;
    for (uint sk = 0; sk < 4; ++sk) {
      simdgroup_float8x8 score(0.f);
      // Prime one fragment. Lookahead never crosses this sk's key rows.
      const uint kr0 = kb + sk * 8 + fn;
      const uint kr1 = kr0 + 1;
      float2 k_fragment(
          kr0 < nk ? k[kb0 + ulong(kr0) * channels + head * D + fm] : 0.f,
          kr1 < nk ? k[kb0 + ulong(kr1) * channels + head * D + fm] : 0.f);
      for (uint dd = 0; dd < D / 8; ++dd) {
        simdgroup_float8x8 aq, bk;
        aq.thread_elements()[0] = qfragments[dd].x;
        aq.thread_elements()[1] = qfragments[dd].y;
        bk.thread_elements()[0] = k_fragment.x;
        bk.thread_elements()[1] = k_fragment.y;
        float2 next_k(0.f);
        if (dd + 1 < D / 8) {
          const uint kd = (dd + 1) * 8 + fm;
          next_k = float2(
              kr0 < nk ? k[kb0 + ulong(kr0) * channels + head * D + kd] : 0.f,
              kr1 < nk ? k[kb0 + ulong(kr1) * channels + head * D + kd] : 0.f);
        }
        simdgroup_multiply_accumulate(score, aq, bk, score);
        k_fragment = next_k;
      }
      for (uint j = 0; j < 2; ++j) {
        const uint kr = kb + sk * 8 + fn + j;
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
    // Carry one V fragment through the original dd/sk traversal. The first
    // fragment and every lookahead stay inside the current 32-key tile.
    const uint vr0 = kb + fm;
    float2 v_fragment(
        vr0 < nk ? v[kb0 + ulong(vr0) * channels + head * D + fn] : 0.f,
        vr0 < nk ? v[kb0 + ulong(vr0) * channels + head * D + fn + 1] : 0.f);
    for (uint dd = 0; dd < D / 8; ++dd) {
      simdgroup_float8x8 output;
      output.thread_elements()[0] = acco[dd].x * factor;
      output.thread_elements()[1] = acco[dd].y * factor;
      for (uint sk = 0; sk < 4; ++sk) {
        simdgroup_float8x8 a, b;
        a.thread_elements()[0] = prob[sk].x;
        a.thread_elements()[1] = prob[sk].y;
        b.thread_elements()[0] = v_fragment.x;
        b.thread_elements()[1] = v_fragment.y;
        float2 next_v(0.f);
        if (sk + 1 < 4 || dd + 1 < D / 8) {
          const uint next_sk = sk + 1 < 4 ? sk + 1 : 0;
          const uint next_dd = sk + 1 < 4 ? dd : dd + 1;
          const uint vr = kb + next_sk * 8 + fm;
          const uint vd = next_dd * 8 + fn;
          next_v = float2(
              vr < nk ? v[kb0 + ulong(vr) * channels + head * D + vd] : 0.f,
              vr < nk ? v[kb0 + ulong(vr) * channels + head * D + vd + 1] : 0.f);
        }
        simdgroup_multiply_accumulate(output, a, b, output);
        v_fragment = next_v;
      }
      acco[dd] = float2(output.thread_elements()[0], output.thread_elements()[1]);
    }
  }
  if (qr < nq) {
    for (uint dd = 0; dd < D / 8; ++dd)
      for (uint j = 0; j < 2; ++j)
        out[qb + ulong(qr) * channels + head * D + dd * 8 + fn + j] = acco[dd][j] / norm;
  }
}
