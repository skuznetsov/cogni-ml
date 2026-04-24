#include <metal_stdlib>
using namespace metal;

kernel void qwen35_split_qgate(
    device const float* q_full   [[buffer(0)]],
    device       float* q_out    [[buffer(1)]],
    device       float* gate_out [[buffer(2)]],
    constant     uint&  n_head   [[buffer(3)]],
    constant     uint&  head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint q_dim = n_head * head_dim;
    if (gid >= q_dim) return;

    const uint h = gid / head_dim;
    const uint d = gid % head_dim;
    const uint src_base = h * 2 * head_dim;
    q_out[gid] = q_full[src_base + d];
    gate_out[gid] = q_full[src_base + head_dim + d];
}

kernel void qwen35_split_qgate_rows(
    device const float* q_full   [[buffer(0)]],
    device       float* q_out    [[buffer(1)]],
    device       float* gate_out [[buffer(2)]],
    constant     uint&  n_head   [[buffer(3)]],
    constant     uint&  head_dim [[buffer(4)]],
    constant     uint&  n_tokens [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint q_dim = n_head * head_dim;
    const uint total = n_tokens * q_dim;
    if (gid >= total) return;

    const uint t = gid / q_dim;
    const uint local = gid - t * q_dim;
    const uint h = local / head_dim;
    const uint d = local % head_dim;
    const uint src_base = t * (2 * q_dim) + h * 2 * head_dim;
    q_out[gid] = q_full[src_base + d];
    gate_out[gid] = q_full[src_base + head_dim + d];
}

kernel void qwen35_rmsnorm_heads(
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
        float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv * weight[d];
    }
}

kernel void qwen35_rmsnorm_heads_rows(
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
        float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv * weight[d];
    }
}

kernel void qwen35_rope_partial(
    device       float* x         [[buffer(0)]],
    constant     uint&  head_dim  [[buffer(1)]],
    constant     uint&  rope_dim  [[buffer(2)]],
    constant     uint&  pos       [[buffer(3)]],
    constant     float& freq_base [[buffer(4)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint half_dim = rope_dim / 2;
    device float* head = x + tgpig * head_dim;

    for (uint i = tiisg; i < half_dim; i += 32) {
        const float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(rope_dim));
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i];
        const float x1 = head[i + half_dim];
        head[i] = x0 * c - x1 * s;
        head[i + half_dim] = x0 * s + x1 * c;
    }
}

kernel void qwen35_rope_partial_rows(
    device       float* x         [[buffer(0)]],
    constant     uint&  head_dim  [[buffer(1)]],
    constant     uint&  rope_dim  [[buffer(2)]],
    constant     uint&  base_pos  [[buffer(3)]],
    constant     float& freq_base [[buffer(4)]],
    constant     uint&  n_head    [[buffer(5)]],
    constant     uint&  n_tokens  [[buffer(6)]],
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
        const float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(rope_dim));
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i];
        const float x1 = head[i + half_dim];
        head[i] = x0 * c - x1 * s;
        head[i + half_dim] = x0 * s + x1 * c;
    }
}

kernel void qwen35_kv_write(
    device const float* k       [[buffer(0)]],
    device const float* v       [[buffer(1)]],
    device       float* k_cache [[buffer(2)]],
    device       float* v_cache [[buffer(3)]],
    constant     uint&  base    [[buffer(4)]],
    constant     uint&  kv_dim  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= kv_dim) return;
    k_cache[base + gid] = k[gid];
    v_cache[base + gid] = v[gid];
}

kernel void qwen35_kv_write_rows(
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

kernel void qwen35_attn_decode_rows(
    device const float* Q         [[buffer(0)]],
    device const float* gate      [[buffer(1)]],
    device const float* k_cache   [[buffer(2)]],
    device const float* v_cache   [[buffer(3)]],
    device       float* out       [[buffer(4)]],
    constant     uint&  base_pos       [[buffer(5)]],
    constant     uint&  n_tokens       [[buffer(6)]],
    constant     uint&  n_head         [[buffer(7)]],
    constant     uint&  n_head_kv      [[buffer(8)]],
    constant     uint&  head_dim       [[buffer(9)]],
    constant     uint&  heads_per_group[[buffer(10)]],
    constant     float& scale          [[buffer(11)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= n_head || t >= n_tokens) return;

    const uint kv_h = h / heads_per_group;
    const uint kv_dim = n_head_kv * head_dim;
    const uint hd4 = head_dim / 4;
    const uint lane = tiisg;
    const uint cache_len = base_pos + t + 1;

    threadgroup float q_tg[256];
    threadgroup float gate_tg[256];
    threadgroup float tile_scores[32];

    for (uint d = lane; d < head_dim; d += 32) {
        q_tg[d] = Q[(t * n_head + h) * head_dim + d];
        gate_tg[d] = gate[(t * n_head + h) * head_dim + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float m = -1e30f;
    float l = 0.0f;
    float o[8];
    for (uint i = 0; i < 8; ++i) o[i] = 0.0f;

    for (uint tile_start = 0; tile_start < cache_len; tile_start += 32) {
        uint j = tile_start + lane;

        float score = -1e30f;
        if (j < cache_len) {
            device const float4* kj4 = (device const float4*)(
                k_cache + j * kv_dim + kv_h * head_dim);
            threadgroup const float4* qv4 = (threadgroup const float4*)q_tg;
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                float4 k4 = kj4[d];
                float4 q4 = qv4[d];
                dot += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
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

        uint tile_len = min(tile_start + 32, cache_len) - tile_start;
        for (uint dl = 0; dl < 8; dl++) {
            uint d = lane + dl * 32;
            if (d >= head_dim) break;
            float acc = 0.0f;
            for (uint s = 0; s < tile_len; s++) {
                acc += tile_scores[s] *
                    v_cache[(tile_start + s) * kv_dim + kv_h * head_dim + d];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    const uint out_base = (t * n_head + h) * head_dim;
    for (uint dl = 0; dl < 8; dl++) {
        uint d = lane + dl * 32;
        if (d >= head_dim) break;
        const float g = gate_tg[d];
        const float sig_g = 1.0f / (1.0f + exp(-g));
        out[out_base + d] = o[dl] * inv_l * sig_g;
    }
}
