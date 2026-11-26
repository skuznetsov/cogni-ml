// Causal Flash Attention — supports both prefill (full sequence) and decode (single query)
// For prefill: processes full sequence with causal mask (j <= i)
// For decode: single query attends to full KV cache (incremental)
//
// Prefill dispatch: threadgroups = [n_heads, ceil(seq_len / N_CA_ROWS)]
//                   threads = [32, N_CA_ROWS]
// Decode dispatch:  threadgroups = [n_heads, 1]
//                   threads = [32, 1]

#include <metal_stdlib>
using namespace metal;

constant uint N_CA_ROWS = 4;  // simdgroups per threadgroup (prefill)

// ── Prefill: full sequence with causal mask ──
kernel void attention_causal_prefill(
    device const float* Q       [[buffer(0)]],   // [n_heads, seq_len, head_dim]
    device const float* K       [[buffer(1)]],   // [n_heads, seq_len, head_dim]
    device const float* V_t     [[buffer(2)]],   // [n_heads, head_dim, seq_len] TRANSPOSED
    device       float* output  [[buffer(3)]],   // [seq_len, n_heads * head_dim]
    constant     uint&  seq_len [[buffer(4)]],
    constant     uint&  n_heads [[buffer(5)]],
    constant     uint&  head_dim [[buffer(6)]],
    constant     float& scale   [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared_base [[threadgroup(0)]])
{
    const uint h = tgpig.x;
    const uint i = tgpig.y * N_CA_ROWS + sgitg;  // query position
    if (h >= n_heads || i >= seq_len) return;

    const uint h_off = h * seq_len * head_dim;
    const uint lane = tiisg;
    const uint hd4 = head_dim / 4;

    threadgroup float* tile_scores = shared_base + sgitg * 32;

    // Cache Q[i] in registers
    device const float4* qi4 = (device const float4*)(Q + h_off + i * head_dim);
    float4 qr[16];
    for (uint d = 0; d < hd4; d++) qr[d] = qi4[d];

    float m = -1e30f;
    float l = 0.0f;
    float o[2] = {0.0f, 0.0f};

    const uint vt_h_off = h * head_dim * seq_len;

    // Only attend to positions j <= i (causal mask)
    const uint max_j = i + 1;

    for (uint tile_start = 0; tile_start < max_j; tile_start += 32) {
        uint j = tile_start + lane;

        float score = -1e30f;
        if (j < max_j) {  // causal: j <= i
            device const float4* kj4 = (device const float4*)(K + h_off + j * head_dim);
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                dot += metal::dot(qr[d], kj4[d]);
            }
            score = dot * scale;
        }

        float tile_max = simd_max(score);
        float m_new = max(m, tile_max);
        float correction = exp(m - m_new);
        float p = (j < max_j) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dl = 0; dl < 2; dl++) {
            uint d = lane + dl * 32;
            if (d >= head_dim) continue;
            device const float* vt_row = V_t + vt_h_off + d * seq_len + tile_start;
            float acc = 0.0f;
            uint tile_end = min(tile_start + 32, max_j) - tile_start;
            for (uint s = 0; s < tile_end; s++) {
                acc += tile_scores[s] * vt_row[s];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    uint out_base = i * (n_heads * head_dim) + h * head_dim;
    for (uint dl = 0; dl < 2; dl++) {
        uint d = lane + dl * 32;
        if (d < head_dim) {
            output[out_base + d] = o[dl] * inv_l;
        }
    }
}

// ── Decode: single query token attends to full KV cache ──
// Q: [n_heads, 1, head_dim] — current token
// K: [n_heads, cache_len, head_dim] — all cached keys
// V_t: [n_heads, head_dim, cache_len] — all cached values (transposed)
// Dispatch: threadgroups = [n_heads, 1], threads = [32, 1]
kernel void attention_causal_decode(
    device const float* Q       [[buffer(0)]],   // [n_heads, head_dim]
    device const float* K       [[buffer(1)]],   // [n_heads, cache_len, head_dim]
    device const float* V_t     [[buffer(2)]],   // [n_heads, head_dim, cache_len]
    device       float* output  [[buffer(3)]],   // [n_heads * head_dim]
    constant     uint&  cache_len [[buffer(4)]],
    constant     uint&  n_heads  [[buffer(5)]],
    constant     uint&  head_dim [[buffer(6)]],
    constant     float& scale    [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    threadgroup float* tile_scores [[threadgroup(0)]])
{
    const uint h = tgpig.x;
    if (h >= n_heads) return;

    const uint lane = tiisg;
    const uint hd4 = head_dim / 4;
    const uint h_off = h * cache_len * head_dim;

    // Cache Q in registers
    device const float4* qi4 = (device const float4*)(Q + h * head_dim);
    float4 qr[16];
    for (uint d = 0; d < hd4; d++) qr[d] = qi4[d];

    float m = -1e30f;
    float l = 0.0f;
    float o[2] = {0.0f, 0.0f};

    const uint vt_h_off = h * head_dim * cache_len;

    for (uint tile_start = 0; tile_start < cache_len; tile_start += 32) {
        uint j = tile_start + lane;

        float score = -1e30f;
        if (j < cache_len) {
            device const float4* kj4 = (device const float4*)(K + h_off + j * head_dim);
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                dot += metal::dot(qr[d], kj4[d]);
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

        for (uint dl = 0; dl < 2; dl++) {
            uint d = lane + dl * 32;
            if (d >= head_dim) continue;
            device const float* vt_row = V_t + vt_h_off + d * cache_len + tile_start;
            float acc = 0.0f;
            uint tile_end = min(tile_start + 32, cache_len) - tile_start;
            for (uint s = 0; s < tile_end; s++) {
                acc += tile_scores[s] * vt_row[s];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    uint out_base = h * head_dim;
    for (uint dl = 0; dl < 2; dl++) {
        uint d = lane + dl * 32;
        if (d < head_dim) {
            output[out_base + d] = o[dl] * inv_l;
        }
    }
}

// ── Cross-attention: query from decoder, K/V from encoder/facts ──
// Same as decode but K/V come from external source (no causal mask needed)
// Dispatch: threadgroups = [n_heads, ceil(query_len / N_CA_ROWS)], threads = [32, N_CA_ROWS]
kernel void attention_cross(
    device const float* Q       [[buffer(0)]],   // [n_heads, query_len, head_dim]
    device const float* K       [[buffer(1)]],   // [n_heads, kv_len, head_dim]
    device const float* V_t     [[buffer(2)]],   // [n_heads, head_dim, kv_len]
    device       float* output  [[buffer(3)]],   // [query_len, n_heads * head_dim]
    constant     uint&  query_len [[buffer(4)]],
    constant     uint&  kv_len   [[buffer(5)]],
    constant     uint&  n_heads  [[buffer(6)]],
    constant     uint&  head_dim [[buffer(7)]],
    constant     float& scale    [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared_base [[threadgroup(0)]])
{
    const uint h = tgpig.x;
    const uint i = tgpig.y * N_CA_ROWS + sgitg;
    if (h >= n_heads || i >= query_len) return;

    const uint q_h_off = h * query_len * head_dim;
    const uint k_h_off = h * kv_len * head_dim;
    const uint lane = tiisg;
    const uint hd4 = head_dim / 4;

    threadgroup float* tile_scores = shared_base + sgitg * 32;

    device const float4* qi4 = (device const float4*)(Q + q_h_off + i * head_dim);
    float4 qr[16];
    for (uint d = 0; d < hd4; d++) qr[d] = qi4[d];

    float m = -1e30f;
    float l = 0.0f;
    float o[2] = {0.0f, 0.0f};

    const uint vt_h_off = h * head_dim * kv_len;

    // Full attention — no causal mask (facts are unordered)
    for (uint tile_start = 0; tile_start < kv_len; tile_start += 32) {
        uint j = tile_start + lane;

        float score = -1e30f;
        if (j < kv_len) {
            device const float4* kj4 = (device const float4*)(K + k_h_off + j * head_dim);
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                dot += metal::dot(qr[d], kj4[d]);
            }
            score = dot * scale;
        }

        float tile_max = simd_max(score);
        float m_new = max(m, tile_max);
        float correction = exp(m - m_new);
        float p = (j < kv_len) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        for (uint dl = 0; dl < 2; dl++) {
            uint d = lane + dl * 32;
            if (d >= head_dim) continue;
            device const float* vt_row = V_t + vt_h_off + d * kv_len + tile_start;
            float acc = 0.0f;
            uint tile_end = min(tile_start + 32, kv_len) - tile_start;
            for (uint s = 0; s < tile_end; s++) {
                acc += tile_scores[s] * vt_row[s];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    float inv_l = (l > 0.0f) ? 1.0f / l : 0.0f;
    uint out_base = i * (n_heads * head_dim) + h * head_dim;
    for (uint dl = 0; dl < 2; dl++) {
        uint d = lane + dl * 32;
        if (d < head_dim) {
            output[out_base + d] = o[dl] * inv_l;
        }
    }
}
