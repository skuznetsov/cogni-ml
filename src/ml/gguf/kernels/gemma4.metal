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

kernel void gemma4_rmsnorm_vec_weighted(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  count    [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float partial[256];

    float ss = 0.0f;
    for (uint i = tid; i < count; i += 256) {
        const float v = x[i];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(partial[0] / float(count) + eps);
    for (uint i = tid; i < count; i += 256) {
        out[i] = x[i] * inv * weight[i];
    }
}

kernel void gemma4_rmsnorm_rows_weighted(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    constant     uint&  row_dim  [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    uint   row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    device const float* src = x + row * row_dim;
    device       float* dst = out + row * row_dim;
    threadgroup float partial[256];

    float ss = 0.0f;
    for (uint i = tid; i < row_dim; i += 256) {
        const float v = src[i];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(partial[0] / float(row_dim) + eps);
    for (uint i = tid; i < row_dim; i += 256) {
        dst[i] = src[i] * inv * weight[i];
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

kernel void gemma4_rmsnorm_heads_weighted_rows(
    device       float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    constant     uint&  head_dim [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    constant     uint&  n_head   [[buffer(4)]],
    constant     uint&  n_tokens [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    device float* head = x + (t * n_head + h) * head_dim;

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

kernel void gemma4_rmsnorm_heads_plain_rows(
    device       float* x        [[buffer(0)]],
    constant     uint&  head_dim [[buffer(1)]],
    constant     float& eps      [[buffer(2)]],
    constant     uint&  n_head   [[buffer(3)]],
    constant     uint&  n_tokens [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    device float* head = x + (t * n_head + h) * head_dim;

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

kernel void gemma4_rope_neox_rows(
    device       float* x            [[buffer(0)]],
    device const float* freq_factors [[buffer(1)]],
    constant     uint&  head_dim     [[buffer(2)]],
    constant     uint&  rope_dim     [[buffer(3)]],
    constant     uint&  base_pos     [[buffer(4)]],
    constant     float& freq_base    [[buffer(5)]],
    constant     uint&  use_factors  [[buffer(6)]],
    constant     uint&  n_head       [[buffer(7)]],
    constant     uint&  n_tokens     [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    const uint half_dim = rope_dim / 2;
    const uint pos = base_pos + t;
    device float* head = x + (t * n_head + h) * head_dim;

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

kernel void gemma4_q_norm_rope_rows(
    device       float* q            [[buffer(0)]],
    device const float* weight       [[buffer(1)]],
    device const float* freq_factors [[buffer(2)]],
    constant     uint&  head_dim     [[buffer(3)]],
    constant     uint&  rope_dim     [[buffer(4)]],
    constant     uint&  base_pos     [[buffer(5)]],
    constant     float& eps          [[buffer(6)]],
    constant     float& freq_base    [[buffer(7)]],
    constant     uint&  use_factors  [[buffer(8)]],
    constant     uint&  n_head       [[buffer(9)]],
    constant     uint&  n_tokens     [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    device float* head = q + (t * n_head + h) * head_dim;
    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        const float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    const uint half_dim = rope_dim / 2;
    const uint pos = base_pos + t;
    for (uint i = tiisg; i < half_dim; i += 32) {
        const float factor = use_factors == 0 ? 1.0f : freq_factors[i];
        const float freq = pow(freq_base, -2.0f * float(i) / float(rope_dim)) / factor;
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i] * inv * weight[i];
        const float x1 = head[i + half_dim] * inv * weight[i + half_dim];
        head[i] = x0 * c - x1 * s;
        head[i + half_dim] = x0 * s + x1 * c;
    }
    for (uint d = rope_dim + tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv * weight[d];
    }
}

kernel void gemma4_k_norm_rope_write_rows(
    device       float* k            [[buffer(0)]],
    device const float* weight       [[buffer(1)]],
    device const float* freq_factors [[buffer(2)]],
    device       float* k_cache      [[buffer(3)]],
    constant     uint&  head_dim     [[buffer(4)]],
    constant     uint&  rope_dim     [[buffer(5)]],
    constant     uint&  base_pos     [[buffer(6)]],
    constant     float& eps          [[buffer(7)]],
    constant     float& freq_base    [[buffer(8)]],
    constant     uint&  use_factors  [[buffer(9)]],
    constant     uint&  n_head       [[buffer(10)]],
    constant     uint&  n_tokens     [[buffer(11)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    const uint kv_dim = n_head * head_dim;
    const uint cache_base = (base_pos + t) * kv_dim + h * head_dim;
    device float* head = k + (t * n_head + h) * head_dim;

    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        const float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    const uint half_dim = rope_dim / 2;
    const uint pos = base_pos + t;
    for (uint i = tiisg; i < half_dim; i += 32) {
        const float factor = use_factors == 0 ? 1.0f : freq_factors[i];
        const float freq = pow(freq_base, -2.0f * float(i) / float(rope_dim)) / factor;
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i] * inv * weight[i];
        const float x1 = head[i + half_dim] * inv * weight[i + half_dim];
        const float y0 = x0 * c - x1 * s;
        const float y1 = x0 * s + x1 * c;
        head[i] = y0;
        head[i + half_dim] = y1;
        k_cache[cache_base + i] = y0;
        k_cache[cache_base + i + half_dim] = y1;
    }
    for (uint d = rope_dim + tiisg; d < head_dim; d += 32) {
        const float y = head[d] * inv * weight[d];
        head[d] = y;
        k_cache[cache_base + d] = y;
    }
}

kernel void gemma4_v_norm_write_rows(
    device       float* v            [[buffer(0)]],
    device       float* v_cache      [[buffer(1)]],
    constant     uint&  head_dim     [[buffer(2)]],
    constant     uint&  base_pos     [[buffer(3)]],
    constant     float& eps          [[buffer(4)]],
    constant     uint&  n_head       [[buffer(5)]],
    constant     uint&  n_tokens     [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    const uint kv_dim = n_head * head_dim;
    const uint cache_base = (base_pos + t) * kv_dim + h * head_dim;
    device float* head = v + (t * n_head + h) * head_dim;

    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        const float x = head[d];
        ss += x * x;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        const float y = head[d] * inv;
        head[d] = y;
        v_cache[cache_base + d] = y;
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

kernel void gemma4_kv_write_rows(
    device const float* k        [[buffer(0)]],
    device const float* v        [[buffer(1)]],
    device       float* k_cache  [[buffer(2)]],
    device       float* v_cache  [[buffer(3)]],
    constant     uint&  base_pos [[buffer(4)]],
    constant     uint&  kv_dim   [[buffer(5)]],
    constant     uint&  n_tokens [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = n_tokens * kv_dim;
    if (gid >= total) return;

    const uint t = gid / kv_dim;
    const uint d = gid - t * kv_dim;
    const uint dst = (base_pos + t) * kv_dim + d;
    k_cache[dst] = k[gid];
    v_cache[dst] = v[gid];
}

kernel void gemma4_kv_write_rows_h16(
    device const float* k        [[buffer(0)]],
    device const float* v        [[buffer(1)]],
    device       half*  k_cache  [[buffer(2)]],
    device       half*  v_cache  [[buffer(3)]],
    constant     uint&  base_pos [[buffer(4)]],
    constant     uint&  kv_dim   [[buffer(5)]],
    constant     uint&  n_tokens [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = n_tokens * kv_dim;
    if (gid >= total) return;

    const uint t = gid / kv_dim;
    const uint d = gid - t * kv_dim;
    const uint dst = (base_pos + t) * kv_dim + d;
    k_cache[dst] = half(k[gid]);
    v_cache[dst] = half(v[gid]);
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

kernel void gemma4_attn_context_rows(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_h = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q_tg[d] = q[(t * n_head + h) * head_dim + d];
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
        out[(t * n_head + h) * head_dim + d] = o[dl] * inv_l;
    }
}


kernel void gemma4_attn_context_rows_kv_h16(
    device const float* q             [[buffer(0)]],
    device const half*  k_cache       [[buffer(1)]],
    device const half*  v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_h = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q_tg[d] = q[(t * n_head + h) * head_dim + d];
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
                dot += q_tg[d] * float(k_cache[pos * kv_dim + kv_h * head_dim + d]);
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
                acc += tile_scores[s] * float(v_cache[ppos * kv_dim + kv_h * head_dim + d]);
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
        out[(t * n_head + h) * head_dim + d] = o[dl] * inv_l;
    }
}

kernel void gemma4_attn_context_rows_gqa2_kv_h16(
    device const float* q             [[buffer(0)]],
    device const half*  k_cache       [[buffer(1)]],
    device const half*  v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pair_idx = tgpig.x;
    const uint t = tgpig.y;
    if (heads_per_group < 2 || (heads_per_group & 1) != 0 || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint h0 = pair_idx * 2;
    const uint h1 = h0 + 1;
    if (h1 >= n_head) return;
    const uint kv_h = h0 / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q0_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float q1_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores0[GEMMA4_ATTN_SG];
    threadgroup float tile_scores1[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q0_tg[d] = q[(t * n_head + h0) * head_dim + d];
        q1_tg[d] = q[(t * n_head + h1) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m0 = -INFINITY;
    float m1 = -INFINITY;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float o0[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    float o1[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < len; tile_start += GEMMA4_ATTN_SG) {
        const uint local_j = tile_start + lane;
        const uint pos = start_pos + local_j;

        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (local_j < len) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                const float kval = float(k_cache[pos * kv_dim + kv_h * head_dim + d]);
                dot0 += q0_tg[d] * kval;
                dot1 += q1_tg[d] * kval;
            }
            score0 = dot0;
            score1 = dot1;
        }

        const float tile_max0 = simd_max(score0);
        const float tile_max1 = simd_max(score1);
        const float m0_new = max(m0, tile_max0);
        const float m1_new = max(m1, tile_max1);
        const float correction0 = exp(m0 - m0_new);
        const float correction1 = exp(m1 - m1_new);
        const float p0 = (local_j < len) ? exp(score0 - m0_new) : 0.0f;
        const float p1 = (local_j < len) ? exp(score1 - m1_new) : 0.0f;
        l0 = l0 * correction0 + simd_sum(p0);
        l1 = l1 * correction1 + simd_sum(p1);

        tile_scores0[lane] = p0;
        tile_scores1[lane] = p1;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, len) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                const float vval = float(v_cache[ppos * kv_dim + kv_h * head_dim + d]);
                acc0 += tile_scores0[s] * vval;
                acc1 += tile_scores1[s] * vval;
            }
            o0[dl] = o0[dl] * correction0 + acc0;
            o1[dl] = o1[dl] * correction1 + acc1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m0 = m0_new;
        m1 = m1_new;
    }

    const float inv_l0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        out[(t * n_head + h0) * head_dim + d] = o0[dl] * inv_l0;
        out[(t * n_head + h1) * head_dim + d] = o1[dl] * inv_l1;
    }
}

kernel void gemma4_attn_context_rows_gqa2(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pair_idx = tgpig.x;
    const uint t = tgpig.y;
    if (heads_per_group < 2 || (heads_per_group & 1) != 0 || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint h0 = pair_idx * 2;
    const uint h1 = h0 + 1;
    if (h1 >= n_head) return;
    const uint kv_h = h0 / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q0_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float q1_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores0[GEMMA4_ATTN_SG];
    threadgroup float tile_scores1[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q0_tg[d] = q[(t * n_head + h0) * head_dim + d];
        q1_tg[d] = q[(t * n_head + h1) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m0 = -INFINITY;
    float m1 = -INFINITY;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float o0[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    float o1[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < len; tile_start += GEMMA4_ATTN_SG) {
        const uint local_j = tile_start + lane;
        const uint pos = start_pos + local_j;

        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (local_j < len) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                const float kval = k_cache[pos * kv_dim + kv_h * head_dim + d];
                dot0 += q0_tg[d] * kval;
                dot1 += q1_tg[d] * kval;
            }
            score0 = dot0;
            score1 = dot1;
        }

        const float tile_max0 = simd_max(score0);
        const float tile_max1 = simd_max(score1);
        const float m0_new = max(m0, tile_max0);
        const float m1_new = max(m1, tile_max1);
        const float correction0 = exp(m0 - m0_new);
        const float correction1 = exp(m1 - m1_new);
        const float p0 = (local_j < len) ? exp(score0 - m0_new) : 0.0f;
        const float p1 = (local_j < len) ? exp(score1 - m1_new) : 0.0f;
        l0 = l0 * correction0 + simd_sum(p0);
        l1 = l1 * correction1 + simd_sum(p1);

        tile_scores0[lane] = p0;
        tile_scores1[lane] = p1;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, len) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                const float vval = v_cache[ppos * kv_dim + kv_h * head_dim + d];
                acc0 += tile_scores0[s] * vval;
                acc1 += tile_scores1[s] * vval;
            }
            o0[dl] = o0[dl] * correction0 + acc0;
            o1[dl] = o1[dl] * correction1 + acc1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m0 = m0_new;
        m1 = m1_new;
    }

    const float inv_l0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        out[(t * n_head + h0) * head_dim + d] = o0[dl] * inv_l0;
        out[(t * n_head + h1) * head_dim + d] = o1[dl] * inv_l1;
    }
}


kernel void gemma4_attn_context_rows_swa256_vec(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens || head_dim != 256 || sliding_window == 0) return;
    if (sgitg >= 8) return;

    const uint kv_h = h / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (row_pos + 1 <= sliding_window) ? 0 : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * 256;

    threadgroup float q_tg[256];
    threadgroup float scores[8];
    threadgroup float probs[8];
    threadgroup float corr_tg;
    threadgroup float inv_l_tg;

    if (tid < 256) {
        q_tg[tid] = q[(t * n_head + h) * 256 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float o = 0.0f;

    for (uint tile_start = 0; tile_start < len; tile_start += 8) {
        const uint local_j = tile_start + uint(sgitg);
        const uint pos = start_pos + local_j;
        float dot = 0.0f;
        if (local_j < len) {
            for (uint d = uint(tiisg); d < 256; d += 32) {
                dot += q_tg[d] * k_cache[pos * kv_dim + kv_h * 256 + d];
            }
        } else {
            dot = -INFINITY;
        }
        dot = simd_sum(dot);
        if (tiisg == 0) {
            scores[sgitg] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float tile_max = -INFINITY;
            const uint tile_len = min(uint(8), len - tile_start);
            for (uint s = 0; s < tile_len; ++s) {
                tile_max = max(tile_max, scores[s]);
            }
            const float m_new = max(m, tile_max);
            const float correction = exp(m - m_new);
            float l_new = l * correction;
            for (uint s = 0; s < 8; ++s) {
                const float p = (s < tile_len) ? exp(scores[s] - m_new) : 0.0f;
                probs[s] = p;
                l_new += p;
            }
            corr_tg = correction;
            m = m_new;
            l = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            const uint tile_len = min(uint(8), len - tile_start);
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                acc += probs[s] * v_cache[ppos * kv_dim + kv_h * 256 + tid];
            }
            o = o * corr_tg + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        inv_l_tg = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256) {
        out[(t * n_head + h) * 256 + tid] = o * inv_l_tg;
    }
}

kernel void gemma4_attn_context_rows_swa256_vec_gqa2(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint pair_idx = tgpig.x;
    const uint t = tgpig.y;
    const uint h0 = pair_idx * 2;
    const uint h1 = h0 + 1;
    if (h1 >= n_head || t >= n_tokens || head_dim != 256 || sliding_window == 0) return;
    if (sgitg >= 8 || heads_per_group < 2 || (heads_per_group & 1) != 0) return;

    const uint kv_h = h0 / heads_per_group;
    if (kv_h >= n_head_kv || (h1 / heads_per_group) != kv_h) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (row_pos + 1 <= sliding_window) ? 0 : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * 256;

    threadgroup float q0_tg[256];
    threadgroup float q1_tg[256];
    threadgroup float scores0[8];
    threadgroup float scores1[8];
    threadgroup float probs0[8];
    threadgroup float probs1[8];
    threadgroup float corr0_tg;
    threadgroup float corr1_tg;
    threadgroup float inv_l0_tg;
    threadgroup float inv_l1_tg;

    if (tid < 256) {
        q0_tg[tid] = q[(t * n_head + h0) * 256 + tid];
        q1_tg[tid] = q[(t * n_head + h1) * 256 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m0 = -INFINITY;
    float l0 = 0.0f;
    float o0 = 0.0f;
    float m1 = -INFINITY;
    float l1 = 0.0f;
    float o1 = 0.0f;

    for (uint tile_start = 0; tile_start < len; tile_start += 8) {
        const uint local_j = tile_start + uint(sgitg);
        const uint pos = start_pos + local_j;
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        if (local_j < len) {
            for (uint d = uint(tiisg); d < 256; d += 32) {
                const float kv = k_cache[pos * kv_dim + kv_h * 256 + d];
                dot0 += q0_tg[d] * kv;
                dot1 += q1_tg[d] * kv;
            }
        } else {
            dot0 = -INFINITY;
            dot1 = -INFINITY;
        }
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        if (tiisg == 0) {
            scores0[sgitg] = dot0;
            scores1[sgitg] = dot1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float tile_max0 = -INFINITY;
            float tile_max1 = -INFINITY;
            const uint tile_len = min(uint(8), len - tile_start);
            for (uint s = 0; s < tile_len; ++s) {
                tile_max0 = max(tile_max0, scores0[s]);
                tile_max1 = max(tile_max1, scores1[s]);
            }
            const float m0_new = max(m0, tile_max0);
            const float m1_new = max(m1, tile_max1);
            const float correction0 = exp(m0 - m0_new);
            const float correction1 = exp(m1 - m1_new);
            float l0_new = l0 * correction0;
            float l1_new = l1 * correction1;
            for (uint s = 0; s < 8; ++s) {
                const float p0 = (s < tile_len) ? exp(scores0[s] - m0_new) : 0.0f;
                const float p1 = (s < tile_len) ? exp(scores1[s] - m1_new) : 0.0f;
                probs0[s] = p0;
                probs1[s] = p1;
                l0_new += p0;
                l1_new += p1;
            }
            corr0_tg = correction0;
            corr1_tg = correction1;
            m0 = m0_new;
            m1 = m1_new;
            l0 = l0_new;
            l1 = l1_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            const uint tile_len = min(uint(8), len - tile_start);
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                const float vv = v_cache[ppos * kv_dim + kv_h * 256 + tid];
                acc0 += probs0[s] * vv;
                acc1 += probs1[s] * vv;
            }
            o0 = o0 * corr0_tg + acc0;
            o1 = o1 * corr1_tg + acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        inv_l0_tg = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
        inv_l1_tg = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256) {
        out[(t * n_head + h0) * 256 + tid] = o0 * inv_l0_tg;
        out[(t * n_head + h1) * 256 + tid] = o1 * inv_l1_tg;
    }
}

kernel void gemma4_attn_context_rows_swa256_vec_tile16(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens || head_dim != 256 || sliding_window == 0) return;
    if (sgitg >= 8) return;

    const uint kv_h = h / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (row_pos + 1 <= sliding_window) ? 0 : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * 256;

    threadgroup float q_tg[256];
    threadgroup float scores[16];
    threadgroup float probs[16];
    threadgroup float corr_tg;
    threadgroup float inv_l_tg;

    if (tid < 256) {
        q_tg[tid] = q[(t * n_head + h) * 256 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -INFINITY;
    float l = 0.0f;
    float o = 0.0f;

    for (uint tile_start = 0; tile_start < len; tile_start += 16) {
        const uint local_j0 = tile_start + uint(sgitg);
        const uint local_j1 = local_j0 + 8;
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        if (local_j0 < len) {
            const uint pos0 = start_pos + local_j0;
            for (uint d = uint(tiisg); d < 256; d += 32) {
                dot0 += q_tg[d] * k_cache[pos0 * kv_dim + kv_h * 256 + d];
            }
        } else {
            dot0 = -INFINITY;
        }
        if (local_j1 < len) {
            const uint pos1 = start_pos + local_j1;
            for (uint d = uint(tiisg); d < 256; d += 32) {
                dot1 += q_tg[d] * k_cache[pos1 * kv_dim + kv_h * 256 + d];
            }
        } else {
            dot1 = -INFINITY;
        }
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        if (tiisg == 0) {
            scores[sgitg] = dot0;
            scores[8 + sgitg] = dot1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float tile_max = -INFINITY;
            const uint tile_len = min(uint(16), len - tile_start);
            for (uint s = 0; s < tile_len; ++s) {
                tile_max = max(tile_max, scores[s]);
            }
            const float m_new = max(m, tile_max);
            const float correction = exp(m - m_new);
            float l_new = l * correction;
            for (uint s = 0; s < 16; ++s) {
                const float p = (s < tile_len) ? exp(scores[s] - m_new) : 0.0f;
                probs[s] = p;
                l_new += p;
            }
            corr_tg = correction;
            m = m_new;
            l = l_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            const uint tile_len = min(uint(16), len - tile_start);
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                acc += probs[s] * v_cache[ppos * kv_dim + kv_h * 256 + tid];
            }
            o = o * corr_tg + acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        inv_l_tg = (l > 0.0f) ? 1.0f / l : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256) {
        out[(t * n_head + h) * 256 + tid] = o * inv_l_tg;
    }
}

kernel void gemma4_attn_context_rows_swa256_vec_gqa2_tile16(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint pair_idx = tgpig.x;
    const uint t = tgpig.y;
    const uint h0 = pair_idx * 2;
    const uint h1 = h0 + 1;
    if (h1 >= n_head || t >= n_tokens || head_dim != 256 || sliding_window == 0) return;
    if (sgitg >= 8 || heads_per_group < 2 || (heads_per_group & 1) != 0) return;

    const uint kv_h = h0 / heads_per_group;
    if (kv_h >= n_head_kv || (h1 / heads_per_group) != kv_h) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (row_pos + 1 <= sliding_window) ? 0 : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * 256;

    threadgroup float q0_tg[256];
    threadgroup float q1_tg[256];
    threadgroup float scores0[16];
    threadgroup float scores1[16];
    threadgroup float probs0[16];
    threadgroup float probs1[16];
    threadgroup float corr0_tg;
    threadgroup float corr1_tg;
    threadgroup float inv_l0_tg;
    threadgroup float inv_l1_tg;

    if (tid < 256) {
        q0_tg[tid] = q[(t * n_head + h0) * 256 + tid];
        q1_tg[tid] = q[(t * n_head + h1) * 256 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m0 = -INFINITY;
    float l0 = 0.0f;
    float o0 = 0.0f;
    float m1 = -INFINITY;
    float l1 = 0.0f;
    float o1 = 0.0f;

    for (uint tile_start = 0; tile_start < len; tile_start += 16) {
        const uint local_j0 = tile_start + uint(sgitg);
        const uint local_j1 = local_j0 + 8;
        float dot00 = 0.0f;
        float dot01 = 0.0f;
        float dot10 = 0.0f;
        float dot11 = 0.0f;
        if (local_j0 < len) {
            const uint pos0 = start_pos + local_j0;
            for (uint d = uint(tiisg); d < 256; d += 32) {
                const float kv = k_cache[pos0 * kv_dim + kv_h * 256 + d];
                dot00 += q0_tg[d] * kv;
                dot10 += q1_tg[d] * kv;
            }
        } else {
            dot00 = -INFINITY;
            dot10 = -INFINITY;
        }
        if (local_j1 < len) {
            const uint pos1 = start_pos + local_j1;
            for (uint d = uint(tiisg); d < 256; d += 32) {
                const float kv = k_cache[pos1 * kv_dim + kv_h * 256 + d];
                dot01 += q0_tg[d] * kv;
                dot11 += q1_tg[d] * kv;
            }
        } else {
            dot01 = -INFINITY;
            dot11 = -INFINITY;
        }
        dot00 = simd_sum(dot00);
        dot01 = simd_sum(dot01);
        dot10 = simd_sum(dot10);
        dot11 = simd_sum(dot11);
        if (tiisg == 0) {
            scores0[sgitg] = dot00;
            scores0[8 + sgitg] = dot01;
            scores1[sgitg] = dot10;
            scores1[8 + sgitg] = dot11;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            float tile_max0 = -INFINITY;
            float tile_max1 = -INFINITY;
            const uint tile_len = min(uint(16), len - tile_start);
            for (uint s = 0; s < tile_len; ++s) {
                tile_max0 = max(tile_max0, scores0[s]);
                tile_max1 = max(tile_max1, scores1[s]);
            }
            const float m0_new = max(m0, tile_max0);
            const float m1_new = max(m1, tile_max1);
            const float correction0 = exp(m0 - m0_new);
            const float correction1 = exp(m1 - m1_new);
            float l0_new = l0 * correction0;
            float l1_new = l1 * correction1;
            for (uint s = 0; s < 16; ++s) {
                const float p0 = (s < tile_len) ? exp(scores0[s] - m0_new) : 0.0f;
                const float p1 = (s < tile_len) ? exp(scores1[s] - m1_new) : 0.0f;
                probs0[s] = p0;
                probs1[s] = p1;
                l0_new += p0;
                l1_new += p1;
            }
            corr0_tg = correction0;
            corr1_tg = correction1;
            m0 = m0_new;
            m1 = m1_new;
            l0 = l0_new;
            l1 = l1_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 256) {
            const uint tile_len = min(uint(16), len - tile_start);
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                const float vv = v_cache[ppos * kv_dim + kv_h * 256 + tid];
                acc0 += probs0[s] * vv;
                acc1 += probs1[s] * vv;
            }
            o0 = o0 * corr0_tg + acc0;
            o1 = o1 * corr1_tg + acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        inv_l0_tg = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
        inv_l1_tg = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 256) {
        out[(t * n_head + h0) * 256 + tid] = o0 * inv_l0_tg;
        out[(t * n_head + h1) * 256 + tid] = o1 * inv_l1_tg;
    }
}

kernel void gemma4_attn_context_rows_gqa4(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint group_idx = tgpig.x;
    const uint t = tgpig.y;
    if (heads_per_group < 4 || (heads_per_group % 4) != 0 || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint h0 = group_idx * 4;
    const uint h1 = h0 + 1;
    const uint h2 = h0 + 2;
    const uint h3 = h0 + 3;
    if (h3 >= n_head) return;
    const uint kv_h = h0 / heads_per_group;
    if (kv_h >= n_head_kv || (h3 / heads_per_group) != kv_h) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q0_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float q1_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float q2_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float q3_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores0[GEMMA4_ATTN_SG];
    threadgroup float tile_scores1[GEMMA4_ATTN_SG];
    threadgroup float tile_scores2[GEMMA4_ATTN_SG];
    threadgroup float tile_scores3[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q0_tg[d] = q[(t * n_head + h0) * head_dim + d];
        q1_tg[d] = q[(t * n_head + h1) * head_dim + d];
        q2_tg[d] = q[(t * n_head + h2) * head_dim + d];
        q3_tg[d] = q[(t * n_head + h3) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m0 = -INFINITY;
    float m1 = -INFINITY;
    float m2 = -INFINITY;
    float m3 = -INFINITY;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float l2 = 0.0f;
    float l3 = 0.0f;
    float o0[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    float o1[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    float o2[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    float o3[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
        o2[i] = 0.0f;
        o3[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < len; tile_start += GEMMA4_ATTN_SG) {
        const uint local_j = tile_start + lane;
        const uint pos = start_pos + local_j;

        float score0 = -INFINITY;
        float score1 = -INFINITY;
        float score2 = -INFINITY;
        float score3 = -INFINITY;
        if (local_j < len) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            float dot3 = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                const float kval = k_cache[pos * kv_dim + kv_h * head_dim + d];
                dot0 += q0_tg[d] * kval;
                dot1 += q1_tg[d] * kval;
                dot2 += q2_tg[d] * kval;
                dot3 += q3_tg[d] * kval;
            }
            score0 = dot0;
            score1 = dot1;
            score2 = dot2;
            score3 = dot3;
        }

        const float tile_max0 = simd_max(score0);
        const float tile_max1 = simd_max(score1);
        const float tile_max2 = simd_max(score2);
        const float tile_max3 = simd_max(score3);
        const float m0_new = max(m0, tile_max0);
        const float m1_new = max(m1, tile_max1);
        const float m2_new = max(m2, tile_max2);
        const float m3_new = max(m3, tile_max3);
        const float correction0 = exp(m0 - m0_new);
        const float correction1 = exp(m1 - m1_new);
        const float correction2 = exp(m2 - m2_new);
        const float correction3 = exp(m3 - m3_new);
        const float p0 = (local_j < len) ? exp(score0 - m0_new) : 0.0f;
        const float p1 = (local_j < len) ? exp(score1 - m1_new) : 0.0f;
        const float p2 = (local_j < len) ? exp(score2 - m2_new) : 0.0f;
        const float p3 = (local_j < len) ? exp(score3 - m3_new) : 0.0f;
        l0 = l0 * correction0 + simd_sum(p0);
        l1 = l1 * correction1 + simd_sum(p1);
        l2 = l2 * correction2 + simd_sum(p2);
        l3 = l3 * correction3 + simd_sum(p3);

        tile_scores0[lane] = p0;
        tile_scores1[lane] = p1;
        tile_scores2[lane] = p2;
        tile_scores3[lane] = p3;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, len) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                const float vval = v_cache[ppos * kv_dim + kv_h * head_dim + d];
                acc0 += tile_scores0[s] * vval;
                acc1 += tile_scores1[s] * vval;
                acc2 += tile_scores2[s] * vval;
                acc3 += tile_scores3[s] * vval;
            }
            o0[dl] = o0[dl] * correction0 + acc0;
            o1[dl] = o1[dl] * correction1 + acc1;
            o2[dl] = o2[dl] * correction2 + acc2;
            o3[dl] = o3[dl] * correction3 + acc3;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m0 = m0_new;
        m1 = m1_new;
        m2 = m2_new;
        m3 = m3_new;
    }

    const float inv_l0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
    const float inv_l2 = (l2 > 0.0f) ? 1.0f / l2 : 0.0f;
    const float inv_l3 = (l3 > 0.0f) ? 1.0f / l3 : 0.0f;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        out[(t * n_head + h0) * head_dim + d] = o0[dl] * inv_l0;
        out[(t * n_head + h1) * head_dim + d] = o1[dl] * inv_l1;
        out[(t * n_head + h2) * head_dim + d] = o2[dl] * inv_l2;
        out[(t * n_head + h3) * head_dim + d] = o3[dl] * inv_l3;
    }
}



kernel void gemma4_attn_context_rows_splitk_stage1(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       float* partial_o     [[buffer(3)]],
    device       float* partial_m     [[buffer(4)]],
    device       float* partial_l     [[buffer(5)]],
    constant     uint&  base_pos      [[buffer(6)]],
    constant     uint&  query_start   [[buffer(7)]],
    constant     uint&  query_count   [[buffer(8)]],
    constant     uint&  n_head        [[buffer(9)]],
    constant     uint&  n_head_kv     [[buffer(10)]],
    constant     uint&  head_dim      [[buffer(11)]],
    constant     uint&  heads_per_group [[buffer(12)]],
    constant     uint&  chunk_size    [[buffer(13)]],
    constant     uint&  n_blocks      [[buffer(14)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint qt = tgpig.y;
    const uint block = tgpig.z;
    if (h >= n_head || qt >= query_count || block >= n_blocks || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint t = query_start + qt;
    const uint row_pos = base_pos + t;
    const uint block_start = block * chunk_size;
    const uint block_end = min(block_start + chunk_size, row_pos + 1);
    const uint kv_h = h / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q_tg[d] = q[(t * n_head + h) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1.0e30f;
    float l = 0.0f;
    float o[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) o[i] = 0.0f;

    for (uint tile_start = block_start; tile_start < block_end; tile_start += GEMMA4_ATTN_SG) {
        const uint j = tile_start + lane;

        float score = -1.0e30f;
        if (j < block_end) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_tg[d] * k_cache[j * kv_dim + kv_h * head_dim + d];
            }
            score = dot;
        }

        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float correction = exp(m - m_new);
        const float p = (j < block_end) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, block_end) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = tile_start + s;
                acc += tile_scores[s] * v_cache[ppos * kv_dim + kv_h * head_dim + d];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m = m_new;
    }

    const uint mb = ((qt * n_head + h) * n_blocks) + block;
    if (lane == 0) {
        partial_m[mb] = m;
        partial_l[mb] = l;
    }
    const uint out_base = mb * head_dim;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        partial_o[out_base + d] = o[dl];
    }
}

kernel void gemma4_attn_context_rows_splitk_kv_h16_stage1(
    device const float* q             [[buffer(0)]],
    device const half*  k_cache       [[buffer(1)]],
    device const half*  v_cache       [[buffer(2)]],
    device       float* partial_o     [[buffer(3)]],
    device       float* partial_m     [[buffer(4)]],
    device       float* partial_l     [[buffer(5)]],
    constant     uint&  base_pos      [[buffer(6)]],
    constant     uint&  query_start   [[buffer(7)]],
    constant     uint&  query_count   [[buffer(8)]],
    constant     uint&  n_head        [[buffer(9)]],
    constant     uint&  n_head_kv     [[buffer(10)]],
    constant     uint&  head_dim      [[buffer(11)]],
    constant     uint&  heads_per_group [[buffer(12)]],
    constant     uint&  chunk_size    [[buffer(13)]],
    constant     uint&  n_blocks      [[buffer(14)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint qt = tgpig.y;
    const uint block = tgpig.z;
    if (h >= n_head || qt >= query_count || block >= n_blocks || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint t = query_start + qt;
    const uint row_pos = base_pos + t;
    const uint block_start = block * chunk_size;
    const uint block_end = min(block_start + chunk_size, row_pos + 1);
    const uint kv_h = h / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q_tg[d] = q[(t * n_head + h) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1.0e30f;
    float l = 0.0f;
    float o[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) o[i] = 0.0f;

    for (uint tile_start = block_start; tile_start < block_end; tile_start += GEMMA4_ATTN_SG) {
        const uint j = tile_start + lane;

        float score = -1.0e30f;
        if (j < block_end) {
            float dot = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                dot += q_tg[d] * float(k_cache[j * kv_dim + kv_h * head_dim + d]);
            }
            score = dot;
        }

        const float tile_max = simd_max(score);
        const float m_new = max(m, tile_max);
        const float correction = exp(m - m_new);
        const float p = (j < block_end) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, block_end) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = tile_start + s;
                acc += tile_scores[s] * float(v_cache[ppos * kv_dim + kv_h * head_dim + d]);
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m = m_new;
    }

    const uint mb = ((qt * n_head + h) * n_blocks) + block;
    if (lane == 0) {
        partial_m[mb] = m;
        partial_l[mb] = l;
    }
    const uint out_base = mb * head_dim;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        partial_o[out_base + d] = o[dl];
    }
}


kernel void gemma4_attn_context_rows_splitk_stage2(
    device const float* partial_o     [[buffer(0)]],
    device const float* partial_m     [[buffer(1)]],
    device const float* partial_l     [[buffer(2)]],
    device       float* out           [[buffer(3)]],
    constant     uint&  query_start   [[buffer(4)]],
    constant     uint&  query_count   [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  head_dim      [[buffer(7)]],
    constant     uint&  n_blocks      [[buffer(8)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint qt = tgpig.y;
    if (h >= n_head || qt >= query_count || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint lane = tiisg;
    float m = -1.0e30f;
    for (uint b = 0; b < n_blocks; ++b) {
        const uint mb = ((qt * n_head + h) * n_blocks) + b;
        m = max(m, partial_m[mb]);
    }

    float l_total = 0.0f;
    for (uint b = 0; b < n_blocks; ++b) {
        const uint mb = ((qt * n_head + h) * n_blocks) + b;
        l_total += partial_l[mb] * exp(partial_m[mb] - m);
    }
    const float inv_l = (l_total > 0.0f) ? 1.0f / l_total : 0.0f;

    const uint t = query_start + qt;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        float acc = 0.0f;
        for (uint b = 0; b < n_blocks; ++b) {
            const uint mb = ((qt * n_head + h) * n_blocks) + b;
            acc += partial_o[mb * head_dim + d] * exp(partial_m[mb] - m);
        }
        out[(t * n_head + h) * head_dim + d] = acc * inv_l;
    }
}


kernel void gemma4_attn_context_rows_h16(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       half*  out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_h = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q_tg[d] = q[(t * n_head + h) * head_dim + d];
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
        out[(t * n_head + h) * head_dim + d] = half(o[dl] * inv_l);
    }
}


kernel void gemma4_attn_context_rows_gqa2_h16(
    device const float* q             [[buffer(0)]],
    device const float* k_cache       [[buffer(1)]],
    device const float* v_cache       [[buffer(2)]],
    device       half*  out           [[buffer(3)]],
    constant     uint&  base_pos      [[buffer(4)]],
    constant     uint&  n_tokens      [[buffer(5)]],
    constant     uint&  n_head        [[buffer(6)]],
    constant     uint&  n_head_kv     [[buffer(7)]],
    constant     uint&  head_dim      [[buffer(8)]],
    constant     uint&  heads_per_group [[buffer(9)]],
    constant     uint&  sliding_window [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pair_idx = tgpig.x;
    const uint t = tgpig.y;
    if (heads_per_group < 2 || (heads_per_group & 1) != 0 || t >= n_tokens || head_dim > GEMMA4_ATTN_MAX_HD) return;

    const uint h0 = pair_idx * 2;
    const uint h1 = h0 + 1;
    if (h1 >= n_head) return;
    const uint kv_h = h0 / heads_per_group;
    if (kv_h >= n_head_kv) return;

    const uint row_pos = base_pos + t;
    const uint start_pos = (sliding_window == 0 || row_pos + 1 <= sliding_window)
        ? 0
        : row_pos + 1 - sliding_window;
    const uint len = row_pos - start_pos + 1;
    const uint kv_dim = n_head_kv * head_dim;
    const uint lane = tiisg;

    threadgroup float q0_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float q1_tg[GEMMA4_ATTN_MAX_HD];
    threadgroup float tile_scores0[GEMMA4_ATTN_SG];
    threadgroup float tile_scores1[GEMMA4_ATTN_SG];

    for (uint d = lane; d < head_dim; d += GEMMA4_ATTN_SG) {
        q0_tg[d] = q[(t * n_head + h0) * head_dim + d];
        q1_tg[d] = q[(t * n_head + h1) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m0 = -INFINITY;
    float m1 = -INFINITY;
    float l0 = 0.0f;
    float l1 = 0.0f;
    float o0[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    float o1[GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG];
    for (uint i = 0; i < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++i) {
        o0[i] = 0.0f;
        o1[i] = 0.0f;
    }

    for (uint tile_start = 0; tile_start < len; tile_start += GEMMA4_ATTN_SG) {
        const uint local_j = tile_start + lane;
        const uint pos = start_pos + local_j;

        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (local_j < len) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                const float kval = k_cache[pos * kv_dim + kv_h * head_dim + d];
                dot0 += q0_tg[d] * kval;
                dot1 += q1_tg[d] * kval;
            }
            score0 = dot0;
            score1 = dot1;
        }

        const float tile_max0 = simd_max(score0);
        const float tile_max1 = simd_max(score1);
        const float m0_new = max(m0, tile_max0);
        const float m1_new = max(m1, tile_max1);
        const float correction0 = exp(m0 - m0_new);
        const float correction1 = exp(m1 - m1_new);
        const float p0 = (local_j < len) ? exp(score0 - m0_new) : 0.0f;
        const float p1 = (local_j < len) ? exp(score1 - m1_new) : 0.0f;
        l0 = l0 * correction0 + simd_sum(p0);
        l1 = l1 * correction1 + simd_sum(p1);

        tile_scores0[lane] = p0;
        tile_scores1[lane] = p1;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const uint tile_len = min(tile_start + GEMMA4_ATTN_SG, len) - tile_start;
        for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
            const uint d = lane + dl * GEMMA4_ATTN_SG;
            if (d >= head_dim) break;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < tile_len; ++s) {
                const uint ppos = start_pos + tile_start + s;
                const float vval = v_cache[ppos * kv_dim + kv_h * head_dim + d];
                acc0 += tile_scores0[s] * vval;
                acc1 += tile_scores1[s] * vval;
            }
            o0[dl] = o0[dl] * correction0 + acc0;
            o1[dl] = o1[dl] * correction1 + acc1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        m0 = m0_new;
        m1 = m1_new;
    }

    const float inv_l0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
    for (uint dl = 0; dl < (GEMMA4_ATTN_MAX_HD / GEMMA4_ATTN_SG); ++dl) {
        const uint d = lane + dl * GEMMA4_ATTN_SG;
        if (d >= head_dim) break;
        out[(t * n_head + h0) * head_dim + d] = half(o0[dl] * inv_l0);
        out[(t * n_head + h1) * head_dim + d] = half(o1[dl] * inv_l1);
    }
}

kernel void gemma4_add_vec(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a[gid] + b[gid];
}

kernel void gemma4_copy_f32(
    device const float* src    [[buffer(0)]],
    device       float* dst    [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dst[gid] = src[gid];
}

kernel void gemma4_add_scaled_vec(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    constant     float& scale  [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = (a[gid] + b[gid]) * scale;
}

kernel void gemma4_rmsnorm_add_rows(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device       float* out      [[buffer(3)]],
    constant     uint&  row_dim  [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    uint   row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    device const float* src = x + row * row_dim;
    device const float* res = residual + row * row_dim;
    device       float* dst = out + row * row_dim;
    threadgroup float partial[256];

    float ss = 0.0f;
    for (uint i = tid; i < row_dim; i += 256) {
        const float v = src[i];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(partial[0] / float(row_dim) + eps);
    for (uint i = tid; i < row_dim; i += 256) {
        dst[i] = res[i] + src[i] * inv * weight[i];
    }
}

kernel void gemma4_rmsnorm_add_scaled_rows(
    device const float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device       float* out      [[buffer(3)]],
    constant     uint&  row_dim  [[buffer(4)]],
    constant     float& eps      [[buffer(5)]],
    constant     float& scale    [[buffer(6)]],
    uint   row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    device const float* src = x + row * row_dim;
    device const float* res = residual + row * row_dim;
    device       float* dst = out + row * row_dim;
    threadgroup float partial[256];

    float ss = 0.0f;
    for (uint i = tid; i < row_dim; i += 256) {
        const float v = src[i];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(partial[0] / float(row_dim) + eps);
    for (uint i = tid; i < row_dim; i += 256) {
        dst[i] = (res[i] + src[i] * inv * weight[i]) * scale;
    }
}

kernel void gemma4_gelu_mul(
    device const float* gate  [[buffer(0)]],
    device const float* up    [[buffer(1)]],
    device       float* out   [[buffer(2)]],
    constant     uint&  count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const float x = gate[gid];
    const float arg = clamp(0.7978845608028654f * x * (1.0f + 0.044715f * x * x), -10.0f, 10.0f);
    const float gelu = 0.5f * x * (1.0f + tanh(arg));
    out[gid] = gelu * up[gid];
}

kernel void gemma4_logit_softcap(
    device       float* x     [[buffer(0)]],
    constant     uint&  count [[buffer(1)]],
    constant     float& cap   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count || cap <= 0.0f) return;
    x[gid] = tanh(x[gid] / cap) * cap;
}
