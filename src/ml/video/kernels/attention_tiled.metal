#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Opt-in F32-only, unmasked, separate-query/key-length SDPA. One SIMD
// group owns eight query rows and streams ascending blocks of 32 key rows.
template <uint D>
inline void video_tiled_attention(device const float *q, device const float *k,
    device const float *v, device float *out, constant uint *p,
    uint tile, uint lane) {
  const uint nq = p[0], nk = p[1], heads = p[2], channels = heads * D;
  // threadgroup_position_in_grid is local to this dispatch. Restore the global
  // tile so segmented and full-range dispatches share identical indexing.
  tile += p[4];
  const uint head = tile % heads;
  const uint query_tiles = (nq + 7) / 8;
  const uint batch = tile / (heads * query_tiles);
  const uint qbase = ((tile / heads) % query_tiles) * 8;
  const ulong qb = ulong(batch) * nq * channels;
  const ulong kb0 = ulong(batch) * nk * channels;
  const uint fm = ((lane / 4) & 4) + ((lane / 2) % 4);
  const uint fn = (((lane / 4) & 2) * 2) + (lane % 2) * 2;
  const uint qr = qbase + fm;
  const float scale = as_type<float>(p[3]);
  float2 acco[D / 8];
  for (uint d = 0; d < D / 8; ++d) acco[d] = float2(0.f);
  float maximum = -MAXFLOAT, norm = 0.f;
  for (uint kb = 0; kb < nk; kb += 32) {
    float2 prob[4];
    float new_max = maximum;
    for (uint sk = 0; sk < 4; ++sk) {
      simdgroup_float8x8 score(0.f);
      for (uint dd = 0; dd < D / 8; ++dd) {
        simdgroup_float8x8 aq, bk;
        for (uint j = 0; j < 2; ++j) {
          const uint qc = dd * 8 + fn + j;
          const uint kr = kb + sk * 8 + fn + j;
          aq.thread_elements()[j] = qr < nq ? q[qb + ulong(qr) * channels + head * D + qc] : 0.f;
          bk.thread_elements()[j] = kr < nk ? k[kb0 + ulong(kr) * channels + head * D + dd * 8 + fm] : 0.f;
        }
        simdgroup_multiply_accumulate(score, aq, bk, score);
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
    for (uint dd = 0; dd < D / 8; ++dd) {
      simdgroup_float8x8 output;
      output.thread_elements()[0] = acco[dd].x * factor;
      output.thread_elements()[1] = acco[dd].y * factor;
      for (uint sk = 0; sk < 4; ++sk) {
        simdgroup_float8x8 a, b;
        a.thread_elements()[0] = prob[sk].x;
        a.thread_elements()[1] = prob[sk].y;
        for (uint j = 0; j < 2; ++j) {
          const uint vr = kb + sk * 8 + fm;
          b.thread_elements()[j] = vr < nk ? v[kb0 + ulong(vr) * channels + head * D + dd * 8 + fn + j] : 0.f;
        }
        simdgroup_multiply_accumulate(output, a, b, output);
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
#define VIDEO_TILED_ATTN(D) \
kernel void video_tiled_attention_##D(device const float *q [[buffer(0)]], \
    device const float *k [[buffer(1)]], device const float *v [[buffer(2)]], \
    device float *out [[buffer(3)]], constant uint *p [[buffer(4)]], \
    uint local_tile [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  video_tiled_attention<D>(q, k, v, out, p, local_tile, lane); \
}
VIDEO_TILED_ATTN(64)
VIDEO_TILED_ATTN(128)
