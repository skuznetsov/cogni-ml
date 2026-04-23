// Qwen 3.5 / 3.6 gated attention decode step (single query token).
//
// Formula (from qwen35_cpu.cr forward_full_attn_layer steps 8-9):
//   kv_h    = h / heads_per_group                     // GQA broadcast
//   scores[p] = (Q[h,:] · K[p, kv_h, :]) * scale      // for p in 0..pos
//   softmax(scores, 0..pos+1)
//   attn[h, d] = Σ_p  scores[p] * V[p, kv_h, d]
//   attn_gated[h, d] = attn[h, d] * sigmoid(gate[h, d])
//
// Layout (f32):
//   Q, gate, out   [n_head,    head_dim]
//   k_cache, v_cache  [max_seq, n_head_kv, head_dim]  (position-major)
//   cache_len = pos + 1 (# valid rows in the cache)
//
// Dispatch: (n_head, 1, 1) threadgroups × 32 threads each.
// Online (flash-style) softmax over tiles of 32 positions — each lane
// handles one position per tile for the score computation, then
// accumulates its slice of the output dim.
//
// Assumes head_dim <= 256 and head_dim % 4 == 0.

#include <metal_stdlib>
using namespace metal;

constant ushort QA_SG   =  32;    // threads per threadgroup = 1 simdgroup
constant uint   QA_HD   = 256;    // compile-time upper bound on head_dim

kernel void qwen35_attn_decode(
    device const float* Q        [[buffer(0)]],
    device const float* gate     [[buffer(1)]],
    device const float* k_cache  [[buffer(2)]],
    device const float* v_cache  [[buffer(3)]],
    device       float* out      [[buffer(4)]],
    constant     uint&  cache_len      [[buffer(5)]],
    constant     uint&  n_head         [[buffer(6)]],
    constant     uint&  n_head_kv      [[buffer(7)]],
    constant     uint&  head_dim       [[buffer(8)]],
    constant     uint&  heads_per_group[[buffer(9)]],
    constant     float& scale          [[buffer(10)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig;
    if (h >= n_head) return;

    const uint kv_h   = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint hd4    = head_dim / 4;
    const uint lane   = tiisg;

    // Q_h and gate[h,:] shared across lanes — keep in threadgroup memory.
    threadgroup float q_tg[QA_HD];
    threadgroup float gate_tg[QA_HD];
    threadgroup float tile_scores[QA_SG];

    for (uint d = lane; d < head_dim; d += QA_SG) {
        q_tg[d]    = Q[h * head_dim + d];
        gate_tg[d] = gate[h * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax state (per lane; simd_max / simd_sum reduce later).
    float m = -1e30f;
    float l = 0.0f;

    // Per-lane output-dim accumulators: 8 dims × 32 lanes = 256 slots.
    float o[QA_HD / QA_SG];
    for (uint i = 0; i < (QA_HD / QA_SG); ++i) o[i] = 0.0f;

    for (uint tile_start = 0; tile_start < cache_len; tile_start += QA_SG) {
        uint j = tile_start + lane;

        // 1) Score for this lane's position j (or -inf if past end).
        float score = -1e30f;
        if (j < cache_len) {
            device const float4* kj4 = (device const float4*)(
                k_cache + j * kv_dim + kv_h * head_dim);
            threadgroup const float4* qv4 = (threadgroup const float4*)q_tg;
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                float4 k = kj4[d];
                float4 q = qv4[d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            score = dot * scale;
        }

        // 2) Online softmax update across the simdgroup.
        float tile_max = simd_max(score);
        float m_new    = max(m, tile_max);
        float correction = exp(m - m_new);
        float p = (j < cache_len) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 3) Accumulate V for this tile. Each lane handles a strided
        //    subset of output dims (lane, lane+32, lane+64, ...).
        uint tile_len = min(tile_start + QA_SG, cache_len) - tile_start;
        for (uint dl = 0; dl < (QA_HD / QA_SG); dl++) {
            uint d = lane + dl * QA_SG;
            if (d >= head_dim) break;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; s++) {
                float vv = v_cache[(tile_start + s) * kv_dim + kv_h * head_dim + d];
                acc += tile_scores[s] * vv;
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    // 4) Normalize, gate, write out.
    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (uint dl = 0; dl < (QA_HD / QA_SG); dl++) {
        uint d = lane + dl * QA_SG;
        if (d >= head_dim) break;
        float g     = gate_tg[d];
        float sig_g = 1.0f / (1.0f + exp(-g));
        out[h * head_dim + d] = o[dl] * inv_l * sig_g;
    }
}
