// Tiled attention with half-precision K/V in shared memory
// Key insight: loading K/V tiles (32×64) as half into threadgroup memory
// eliminates strided device reads during Q·K and V accumulation.
//
// Dispatch: threadgroups = [n_heads, ceil(seq_len / NSG_A)]
//           threads_per_threadgroup = [32, NSG_A]

#include <metal_stdlib>
using namespace metal;

constant short NSG_A = 4;
constant short KV_TILE = 32;

kernel void attention_tiled_half(
    device const float* Q       [[buffer(0)]],
    device const float* K       [[buffer(1)]],
    device const float* V_t     [[buffer(2)]],  // [n_heads, head_dim, seq] TRANSPOSED
    device       float* output  [[buffer(3)]],
    constant     uint&  seq_len [[buffer(4)]],
    constant     uint&  n_heads [[buffer(5)]],
    constant     uint&  head_dim [[buffer(6)]],
    constant     float& scale   [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char* shmem_raw [[threadgroup(0)]])
{
    const uint h = tgpig.x;
    const uint i = tgpig.y * NSG_A + sgitg;  // query position
    if (h >= n_heads || i >= seq_len) return;

    const uint h_off = h * seq_len * head_dim;
    const uint lane = tiisg;
    const short DK = (short)head_dim;
    const short hd4 = DK / 4;

    // Shared per simdgroup:
    //   sk_h[KV_TILE][DK] as half = 32×64×2 = 4KB
    //   scores[KV_TILE] as float = 128B
    // Total = ~4.1KB per simdgroup. ×4 = 16.5KB < 32KB ✓
    const uint per_sg = KV_TILE * DK * 2 + KV_TILE * 4;  // bytes
    threadgroup half*  sk_h = (threadgroup half*)(shmem_raw + sgitg * per_sg);
    threadgroup float* tile_scores = (threadgroup float*)(sk_h + KV_TILE * DK);

    // Cache Q in registers as float4
    device const float4* qi4 = (device const float4*)(Q + h_off + i * DK);
    float4 qr[16];
    for (short d = 0; d < hd4; d++) qr[d] = qi4[d];

    // Online softmax accumulators
    float m = -1e30f, l = 0.0f;
    float o[2] = {0.0f, 0.0f};  // 2 output dims per lane

    for (uint kv_start = 0; kv_start < seq_len; kv_start += KV_TILE) {
        // === Cooperative K tile load: 32 lanes load 32×64 as half ===
        for (short c = 0; c < KV_TILE; c++) {
            uint ki = kv_start + c;
            device const float* kr = K + h_off + ki * DK;
            // Each lane loads 2 floats → 2 halfs (DK/32 = 2 iterations for DK=64)
            for (short d = lane; d < DK; d += 32) {
                sk_h[c * DK + d] = (ki < seq_len) ? half(kr[d]) : half(0);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // === Q·K^T from shared: each lane computes dot for its key ===
        for (short c = lane; c < KV_TILE; c += 32) {
            float dot = 0.0f;
            if (kv_start + c < seq_len) {
                threadgroup const half* kc = sk_h + c * DK;
                // float4 Q × half4 K — mixed precision dot
                for (short d = 0; d < DK; d += 4) {
                    float4 k4 = float4(kc[d], kc[d+1], kc[d+2], kc[d+3]);
                    dot += qr[d/4].x * k4.x + qr[d/4].y * k4.y + qr[d/4].z * k4.z + qr[d/4].w * k4.w;
                }
            }
            float s = dot * scale;
            tile_scores[c] = s;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // === Online softmax ===
        float tile_max = -1e30f;
        for (short c = lane; c < KV_TILE; c += 32) {
            if (kv_start + c < seq_len) tile_max = max(tile_max, tile_scores[c]);
        }
        tile_max = simd_max(tile_max);
        float m_new = max(m, tile_max);
        float correction = exp(m - m_new);

        float tile_sum = 0.0f;
        for (short c = lane; c < KV_TILE; c += 32) {
            float e = (kv_start + c < seq_len) ? exp(tile_scores[c] - m_new) : 0.0f;
            tile_scores[c] = e;
            tile_sum += e;
        }
        l = l * correction + simd_sum(tile_sum);

        // Correct previous output
        o[0] *= correction;
        o[1] *= correction;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // === V tile load as half from TRANSPOSED layout (reuse sk_h) ===
        // V_t layout: [n_heads, head_dim, seq_len]
        // We need tile_v[c][d] = V_t[h, d, kv_start+c]
        const uint vt_h_off = h * DK * seq_len;
        for (short d_blk = 0; d_blk < DK; d_blk += 32) {
            short d = d_blk + lane;
            if (d < DK) {
                device const float* vt_row = V_t + vt_h_off + d * seq_len + kv_start;
                for (short c = 0; c < KV_TILE; c++) {
                    sk_h[c * DK + d] = (kv_start + c < seq_len) ? half(vt_row[c]) : half(0);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // === V accumulation from shared ===
        uint tile_end = min(kv_start + KV_TILE, seq_len) - kv_start;
        for (short dl = 0; dl < 2; dl++) {
            uint d = lane + dl * 32;
            if (d >= DK) continue;
            float acc = 0.0f;
            for (short c = 0; c < (short)tile_end; c++) {
                acc += tile_scores[c] * float(sk_h[c * DK + d]);
            }
            o[dl] += acc;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        m = m_new;
    }

    // Finalize
    float inv_l = 1.0f / l;
    uint out_base = i * n_heads * DK + h * DK;
    for (short dl = 0; dl < 2; dl++) {
        uint d = lane + dl * 32;
        if (d < DK) output[out_base + d] = o[dl] * inv_l;
    }
}
