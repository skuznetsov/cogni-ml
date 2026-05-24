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
constant uint   QA_GQA4_HD = 128; // specialized Qwen3.5/3.6 9B head_dim

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

// GQA-specialized decode attention for Qwen3.5/3.6 9B-style heads_per_group=4.
// One threadgroup owns one KV head and four query heads. The K tile is loaded
// once into threadgroup memory and reused by four simdgroups, preserving the
// same online-softmax math as qwen35_attn_decode while reducing redundant K
// cache traffic at long context.
kernel void qwen35_attn_decode_gqa4(
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
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    if (heads_per_group != 4 || head_dim > QA_GQA4_HD) return;

    const uint kv_h = tgpig;
    const uint local_h = sgitg;
    const uint h = kv_h * 4 + local_h;
    if (kv_h >= n_head_kv || h >= n_head) return;

    const uint kv_dim = n_head_kv * head_dim;
    const uint hd4 = head_dim / 4;
    const uint lane = tiisg;
    const uint linear = uint(sgitg) * QA_SG + lane;

    threadgroup float k_tile[QA_SG * QA_GQA4_HD];
    threadgroup float q_tg_all[4][QA_GQA4_HD];
    threadgroup float tile_scores_all[4][QA_SG];
    threadgroup float* q_tg = q_tg_all[local_h];
    threadgroup float* tile_scores = tile_scores_all[local_h];

    for (uint d = lane; d < head_dim; d += QA_SG) {
        q_tg[d] = Q[h * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1e30f;
    float l = 0.0f;

    float o[QA_GQA4_HD / QA_SG];
    for (uint i = 0; i < (QA_GQA4_HD / QA_SG); ++i) o[i] = 0.0f;

    for (uint tile_start = 0; tile_start < cache_len; tile_start += QA_SG) {
        const uint tile_len = min(tile_start + QA_SG, cache_len) - tile_start;

        for (uint idx = linear; idx < tile_len * head_dim; idx += 4 * QA_SG) {
            const uint s = idx / head_dim;
            const uint d = idx - s * head_dim;
            k_tile[s * head_dim + d] =
                k_cache[(tile_start + s) * kv_dim + kv_h * head_dim + d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint j = tile_start + lane;
        float score = -1e30f;
        if (j < cache_len) {
            threadgroup const float4* kj4 = (threadgroup const float4*)(
                k_tile + lane * head_dim);
            threadgroup const float4* qv4 = (threadgroup const float4*)q_tg;
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                float4 k = kj4[d];
                float4 q = qv4[d];
                dot += q.x * k.x + q.y * k.y + q.z * k.z + q.w * k.w;
            }
            score = dot * scale;
        }

        float tile_max = simd_max(score);
        float m_new = max(m, tile_max);
        float correction = exp(m - m_new);
        float p = (j < cache_len) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dl = 0; dl < (QA_GQA4_HD / QA_SG); dl++) {
            uint d = lane + dl * QA_SG;
            if (d >= head_dim) break;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; s++) {
                float vv = v_cache[(tile_start + s) * kv_dim + kv_h * head_dim + d];
                acc += tile_scores[s] * vv;
            }
            o[dl] = o[dl] * correction + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (uint dl = 0; dl < (QA_GQA4_HD / QA_SG); dl++) {
        uint d = lane + dl * QA_SG;
        if (d >= head_dim) break;
        float g = gate[h * head_dim + d];
        float sig_g = 1.0f / (1.0f + exp(-g));
        out[h * head_dim + d] = o[dl] * inv_l * sig_g;
    }
}

// Exact split-context decode attention. Stage 1 computes one online-softmax
// summary per (query head, context block):
//   partial_m = max score
//   partial_l = sum exp(score - partial_m)
//   partial_o = sum exp(score - partial_m) * V
// Stage 2 combines those summaries with the standard log-sum-exp correction.
// This exposes long-context decode parallelism that the single-simdgroup scan
// cannot use.
kernel void qwen35_attn_decode_splitk_stage1(
    device const float* Q          [[buffer(0)]],
    device const float* k_cache    [[buffer(1)]],
    device const float* v_cache    [[buffer(2)]],
    device       float* partial_o  [[buffer(3)]],
    device       float* partial_m  [[buffer(4)]],
    device       float* partial_l  [[buffer(5)]],
    constant     uint&  cache_len      [[buffer(6)]],
    constant     uint&  n_head         [[buffer(7)]],
    constant     uint&  n_head_kv      [[buffer(8)]],
    constant     uint&  head_dim       [[buffer(9)]],
    constant     uint&  heads_per_group[[buffer(10)]],
    constant     float& scale          [[buffer(11)]],
    constant     uint&  chunk_size     [[buffer(12)]],
    constant     uint&  n_blocks       [[buffer(13)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint block = tgpig.y;
    if (h >= n_head || block >= n_blocks || head_dim > QA_HD) return;

    const uint kv_h = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint hd4 = head_dim / 4;
    const uint lane = tiisg;
    const uint block_start = block * chunk_size;
    const uint block_end = min(block_start + chunk_size, cache_len);

    threadgroup float q_tg[QA_HD];
    threadgroup float tile_scores[QA_SG];

    for (uint d = lane; d < head_dim; d += QA_SG) {
        q_tg[d] = Q[h * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1e30f;
    float l = 0.0f;
    float o[QA_HD / QA_SG];
    for (uint i = 0; i < (QA_HD / QA_SG); ++i) o[i] = 0.0f;

    for (uint tile_start = block_start; tile_start < block_end; tile_start += QA_SG) {
        uint j = tile_start + lane;

        float score = -1e30f;
        if (j < block_end) {
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

        float tile_max = simd_max(score);
        float m_new = max(m, tile_max);
        float correction = exp(m - m_new);
        float p = (j < block_end) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        uint tile_len = min(tile_start + QA_SG, block_end) - tile_start;
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

    const uint mb = h * n_blocks + block;
    if (lane == 0) {
        partial_m[mb] = m;
        partial_l[mb] = l;
    }
    const uint out_base = mb * head_dim;
    for (uint dl = 0; dl < (QA_HD / QA_SG); dl++) {
        uint d = lane + dl * QA_SG;
        if (d >= head_dim) break;
        partial_o[out_base + d] = o[dl];
    }
}

kernel void qwen35_attn_decode_splitk_stage2(
    device const float* gate       [[buffer(0)]],
    device const float* partial_o  [[buffer(1)]],
    device const float* partial_m  [[buffer(2)]],
    device const float* partial_l  [[buffer(3)]],
    device       float* out        [[buffer(4)]],
    constant     uint&  n_head     [[buffer(5)]],
    constant     uint&  head_dim   [[buffer(6)]],
    constant     uint&  n_blocks   [[buffer(7)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig;
    if (h >= n_head || head_dim > QA_HD) return;

    const uint lane = tiisg;
    float m = -1e30f;
    for (uint b = 0; b < n_blocks; b++) {
        m = max(m, partial_m[h * n_blocks + b]);
    }

    float l_total = 0.0f;
    for (uint b = 0; b < n_blocks; b++) {
        const uint mb = h * n_blocks + b;
        l_total += partial_l[mb] * exp(partial_m[mb] - m);
    }
    const float inv_l = (l_total > 0.0f) ? 1.0f / l_total : 0.0f;

    for (uint dl = 0; dl < (QA_HD / QA_SG); dl++) {
        uint d = lane + dl * QA_SG;
        if (d >= head_dim) break;
        float acc = 0.0f;
        for (uint b = 0; b < n_blocks; b++) {
            const uint mb = h * n_blocks + b;
            acc += partial_o[mb * head_dim + d] * exp(partial_m[mb] - m);
        }
        float g = gate[h * head_dim + d];
        float sig_g = 1.0f / (1.0f + exp(-g));
        out[h * head_dim + d] = acc * inv_l * sig_g;
    }
}
