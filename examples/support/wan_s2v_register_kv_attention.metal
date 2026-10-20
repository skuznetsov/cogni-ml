#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Probe-only F32 D=128 attention kernel. One SIMD group owns two independent
// eight-query tiles from the same head and reuses each K/V fragment between
// their otherwise separate attention accumulators.
kernel void wan_bounded_register_kv_attention_128_q2(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint local_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint D = 128;
  const uint nq = p[0], nk = p[1], heads = p[2], channels = heads * D;
  // Restore the caller's global tile after segmented dispatch.
  const uint tile = local_tile + p[4];
  const uint query_tiles = (nq + 15u) / 16u;
  const uint head = tile % heads;
  const uint batch = tile / (heads * query_tiles);
  const uint qbase = ((tile / heads) % query_tiles) * 16u;
  const ulong qb = ulong(batch) * nq * channels;
  const ulong kb0 = ulong(batch) * nk * channels;
  const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
  const uint fn = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  const uint qr0 = qbase + fm;
  const uint qr1 = qr0 + 8u;
  const float scale = as_type<float>(p[3]);

  // Cache both query tiles before streaming the ascending key blocks.
  float2 qfragments0[D / 8];
  float2 qfragments1[D / 8];
  for (uint dd = 0; dd < D / 8; ++dd) {
    const uint qc = dd * 8u + fn;
    qfragments0[dd] = float2(
        qr0 < nq ? q[qb + ulong(qr0) * channels + head * D + qc] : 0.f,
        qr0 < nq ? q[qb + ulong(qr0) * channels + head * D + qc + 1u] : 0.f);
    qfragments1[dd] = float2(
        qr1 < nq ? q[qb + ulong(qr1) * channels + head * D + qc] : 0.f,
        qr1 < nq ? q[qb + ulong(qr1) * channels + head * D + qc + 1u] : 0.f);
  }

  float2 acco0[D / 8];
  float2 acco1[D / 8];
  for (uint dd = 0; dd < D / 8; ++dd) {
    acco0[dd] = float2(0.f);
    acco1[dd] = float2(0.f);
  }
  float maximum0 = -MAXFLOAT, norm0 = 0.f;
  float maximum1 = -MAXFLOAT, norm1 = 0.f;

  for (uint kb = 0; kb < nk; kb += 32u) {
    float2 prob0[4];
    float2 prob1[4];
    float new_max0 = maximum0;
    float new_max1 = maximum1;
    for (uint sk = 0; sk < 4; ++sk) {
      simdgroup_float8x8 score0(0.f);
      simdgroup_float8x8 score1(0.f);
      for (uint dd = 0; dd < D / 8; ++dd) {
        simdgroup_float8x8 aq0, aq1, bk;
        for (uint j = 0; j < 2; ++j) {
          const uint kr = kb + sk * 8u + fn + j;
          const uint kd = dd * 8u + fm;
          aq0.thread_elements()[j] = qfragments0[dd][j];
          aq1.thread_elements()[j] = qfragments1[dd][j];
          bk.thread_elements()[j] = kr < nk
              ? k[kb0 + ulong(kr) * channels + head * D + kd]
              : 0.f;
        }
        simdgroup_multiply_accumulate(score0, aq0, bk, score0);
        simdgroup_multiply_accumulate(score1, aq1, bk, score1);
      }
      for (uint j = 0; j < 2; ++j) {
        const uint kr = kb + sk * 8u + fn + j;
        prob0[sk][j] = qr0 < nq && kr < nk
            ? score0.thread_elements()[j] * scale
            : -MAXFLOAT;
        prob1[sk][j] = qr1 < nq && kr < nk
            ? score1.thread_elements()[j] * scale
            : -MAXFLOAT;
      }
      float local0 = max(prob0[sk].x, prob0[sk].y);
      local0 = max(local0, simd_shuffle_xor(local0, ushort(1)));
      local0 = max(local0, simd_shuffle_xor(local0, ushort(8)));
      new_max0 = max(new_max0, local0);
      float local1 = max(prob1[sk].x, prob1[sk].y);
      local1 = max(local1, simd_shuffle_xor(local1, ushort(1)));
      local1 = max(local1, simd_shuffle_xor(local1, ushort(8)));
      new_max1 = max(new_max1, local1);
    }

    float sum0 = 0.f;
    float sum1 = 0.f;
    for (uint sk = 0; sk < 4; ++sk) {
      prob0[sk] = precise::exp(prob0[sk] - new_max0);
      float local0 = prob0[sk].x + prob0[sk].y;
      local0 = local0 + simd_shuffle_xor(local0, ushort(1));
      local0 = local0 + simd_shuffle_xor(local0, ushort(8));
      sum0 = sum0 + local0;

      prob1[sk] = precise::exp(prob1[sk] - new_max1);
      float local1 = prob1[sk].x + prob1[sk].y;
      local1 = local1 + simd_shuffle_xor(local1, ushort(1));
      local1 = local1 + simd_shuffle_xor(local1, ushort(8));
      sum1 = sum1 + local1;
    }
    const float factor0 = precise::exp(maximum0 - new_max0);
    const float factor1 = precise::exp(maximum1 - new_max1);
    maximum0 = new_max0;
    maximum1 = new_max1;
    norm0 = norm0 * factor0 + sum0;
    norm1 = norm1 * factor1 + sum1;

    for (uint dd = 0; dd < D / 8; ++dd) {
      simdgroup_float8x8 output0, output1;
      output0.thread_elements()[0] = acco0[dd].x * factor0;
      output0.thread_elements()[1] = acco0[dd].y * factor0;
      output1.thread_elements()[0] = acco1[dd].x * factor1;
      output1.thread_elements()[1] = acco1[dd].y * factor1;
      for (uint sk = 0; sk < 4; ++sk) {
        simdgroup_float8x8 a0, a1, b;
        a0.thread_elements()[0] = prob0[sk].x;
        a0.thread_elements()[1] = prob0[sk].y;
        a1.thread_elements()[0] = prob1[sk].x;
        a1.thread_elements()[1] = prob1[sk].y;
        for (uint j = 0; j < 2; ++j) {
          const uint vr = kb + sk * 8u + fm;
          const uint vd = dd * 8u + fn + j;
          b.thread_elements()[j] = vr < nk
              ? v[kb0 + ulong(vr) * channels + head * D + vd]
              : 0.f;
        }
        simdgroup_multiply_accumulate(output0, a0, b, output0);
        simdgroup_multiply_accumulate(output1, a1, b, output1);
      }
      acco0[dd] = float2(output0.thread_elements()[0], output0.thread_elements()[1]);
      acco1[dd] = float2(output1.thread_elements()[0], output1.thread_elements()[1]);
    }
  }

  if (qr0 < nq) {
    for (uint dd = 0; dd < D / 8; ++dd)
      for (uint j = 0; j < 2; ++j)
        out[qb + ulong(qr0) * channels + head * D + dd * 8u + fn + j] = acco0[dd][j] / norm0;
  }
  if (qr1 < nq) {
    for (uint dd = 0; dd < D / 8; ++dd)
      for (uint j = 0; j < 2; ++j)
        out[qb + ulong(qr1) * channels + head * D + dd * 8u + fn + j] = acco1[dd][j] / norm1;
  }
}
