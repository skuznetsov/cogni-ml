// Flash Attention with online softmax + shared-memory tile scores
// Processes K/V in tiles of 32. Each tile's scores stored in small shared buffer.
// No O(n) shared memory — only O(32) per simdgroup.
//
// Dispatch: threadgroups = [n_heads, ceil(seq_len / N_FA_ROWS)]
//           threads_per_threadgroup = [32, N_FA_ROWS]
//           shared memory = N_FA_ROWS * 32 * sizeof(float)

#include <metal_stdlib>
using namespace metal;

constant uint N_FA_ROWS = 4;  // simdgroups per threadgroup

kernel void attention_flash(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V_t     [[buffer(2)]],  // [n_heads, head_dim, seq_len] TRANSPOSED
    device       float* output  [[buffer(3)]],
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
    const uint i = tgpig.y * N_FA_ROWS + sgitg;
    if (h >= n_heads || i >= seq_len) return;

    const uint h_off = h * seq_len * head_dim;
    const uint lane = tiisg;
    const uint hd4 = head_dim / 4;

    // Per-simdgroup shared tile scores (32 floats)
    threadgroup float* tile_scores = shared_base + sgitg * 32;

    // Cache Q in registers
    device const float4* qi4 = (device const float4*)(Q + h_off + i * head_dim);
    float4 qr[16];
    for (uint d = 0; d < hd4; d++) qr[d] = qi4[d];

    // Online softmax state
    float m = -1e30f;
    float l = 0.0f;
    float o[2] = {0.0f, 0.0f};  // 2 output dims per lane

    const uint vt_h_off = h * head_dim * seq_len;

    for (uint tile_start = 0; tile_start < seq_len; tile_start += 32) {
        uint j = tile_start + lane;

        // Q·K dot product
        float score = -1e30f;
        if (j < seq_len) {
            device const float4* kj4 = (device const float4*)(K + h_off + j * head_dim);
            float dot = 0.0f;
            for (uint d = 0; d < hd4; d++) {
                dot += metal::dot(qr[d], kj4[d]);
            }
            score = dot * scale;
        }

        // Online softmax
        float tile_max = simd_max(score);
        float m_new = max(m, tile_max);
        float correction = exp(m - m_new);
        float p = (j < seq_len) ? exp(score - m_new) : 0.0f;
        l = l * correction + simd_sum(p);

        // Store normalized p in shared tile (32 floats — tiny)
        tile_scores[lane] = p;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // V accumulation: lanes split over head_dim, read tile_scores for all 32 keys
        for (uint dl = 0; dl < 2; dl++) {
            uint d = lane + dl * 32;
            if (d >= head_dim) continue;
            device const float* vt_row = V_t + vt_h_off + d * seq_len + tile_start;
            float acc = 0.0f;
            uint tile_end = min(tile_start + 32, seq_len) - tile_start;
            for (uint s = 0; s < tile_end; s++) {
                acc += tile_scores[s] * vt_row[s];
            }
            o[dl] = o[dl] * correction + acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    // Finalize: output = o / l
    float inv_l = 1.0f / l;
    uint out_base = i * (n_heads * head_dim) + h * head_dim;
    for (uint dl = 0; dl < 2; dl++) {
        uint d = lane + dl * 32;
        if (d < head_dim) {
            output[out_base + d] = o[dl] * inv_l;
        }
    }
}
