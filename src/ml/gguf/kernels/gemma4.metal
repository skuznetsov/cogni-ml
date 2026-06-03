#include <metal_stdlib>
using namespace metal;

kernel void gemma4_rmsnorm_heads_weighted(
    device       float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    constant     uint&  head_dim [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    device float* head = x + tgpig * head_dim;

    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        const float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv * weight[d];
    }
}

kernel void gemma4_rmsnorm_heads_plain(
    device       float* x        [[buffer(0)]],
    constant     uint&  head_dim [[buffer(1)]],
    constant     float& eps      [[buffer(2)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    device float* head = x + tgpig * head_dim;

    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        const float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv;
    }
}

kernel void gemma4_rope_neox(
    device       float* x            [[buffer(0)]],
    device const float* freq_factors [[buffer(1)]],
    constant     uint&  head_dim     [[buffer(2)]],
    constant     uint&  rope_dim     [[buffer(3)]],
    constant     uint&  pos          [[buffer(4)]],
    constant     float& freq_base    [[buffer(5)]],
    constant     uint&  use_factors  [[buffer(6)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint half_dim = rope_dim / 2;
    device float* head = x + tgpig * head_dim;

    for (uint i = tiisg; i < half_dim; i += 32) {
        const float factor = use_factors == 0 ? 1.0f : freq_factors[i];
        const float freq = pow(freq_base, -2.0f * float(i) / float(rope_dim)) / factor;
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i];
        const float x1 = head[i + half_dim];
        head[i] = x0 * c - x1 * s;
        head[i + half_dim] = x0 * s + x1 * c;
    }
}

constant uint GEMMA4_ATTN_SG = 32;
constant uint GEMMA4_ATTN_MAX_HD = 512;

kernel void gemma4_kv_write_one(
    device const float* k       [[buffer(0)]],
    device const float* v       [[buffer(1)]],
    device       float* k_cache [[buffer(2)]],
    device       float* v_cache [[buffer(3)]],
    constant     uint&  pos     [[buffer(4)]],
    constant     uint&  kv_dim  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= kv_dim) return;
    const uint off = pos * kv_dim + gid;
    k_cache[off] = k[gid];
    v_cache[off] = v[gid];
}

kernel void gemma4_attn_context_one(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  start_pos     [[buffer(4)]],
    constant     uint&  len           [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig;
    if (h >= n_head || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint kv_h = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q_tg[d] = q[h * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float o[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) {
        o[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < len; tile_start += GEMMA4_ATTN_SG) {
        const uint local_j = tile_start + lane;
        const uint pos = start_pos + local_j;

        float score = -INFINITY;
        if (local_j < len) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_tg[d] * k_cache[pos * kv_dim + kv_h * head_dim + d];
            }
            score = dot;
        }

        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float correction = exp(m - m_new);
        const float p = (local_j < len) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, len) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                acc += tile_scores[s] * v_cache[ppos * kv_dim + kv_h * head_dim + d];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        out[h * head_dim + d] = o[dl] * inv_l;
    }
}
