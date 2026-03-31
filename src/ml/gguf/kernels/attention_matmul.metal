// Flash attention with simdgroup_matrix — adapted from llama.cpp
// BERT encoder: no causal mask, no ALiBi, bidirectional.
// Q in shared, K/V read from device, scores in shared, O in shared.
//
// Parameters: Q_TILE=8 queries per SG, C_TILE=32 KV per tile, NSG=2 simdgroups
// DK=64 (head_dim), DV=64 (same for BERT)
//
// Shared memory layout per threadgroup:
//   sq[Q, DK] half       — query tile (Q = Q_TILE * NSG)
//   so[Q, PV] half       — output accumulator
//   ss[Q, 2*C] float     — scratch for scores + diagonal (via float2)
//   sk[8, DK] half       — K tile temp (per simdgroup, reused for V)
//
// Dispatch: threadgroups = [ceil(seq_len / Q_TOTAL), n_heads]
//           threads_per_threadgroup = [32, NSG]

#include <metal_stdlib>
using namespace metal;

constant short Q_PER_SG = 8;   // queries per simdgroup
constant short C_TILE = 32;    // KV positions per tile
constant short NSG_FA = 2;     // simdgroups per threadgroup
constant short NW = 32;        // simdgroup width

// head_dim fixed at 64 for nomic-bert
constant short DK = 64;
constant short DK8 = 8;        // DK/8
constant short DV = 64;
constant short DV4 = 16;       // DV/4
constant short DV8 = 8;        // DV/8
constant short PV = 64;        // padded DV (aligned to 64)
constant short PV4 = 16;
constant short PV8 = 8;

constant short Q_TOTAL = Q_PER_SG * NSG_FA;  // 16 queries per threadgroup
constant short NQ = Q_PER_SG / NSG_FA;       // 4 queries handled per SG

constant short SH = 2 * C_TILE;  // shared mem stride for scores (float2)

kernel void attention_matmul(
    device const half*  q_src   [[buffer(0)]],  // [n_heads, seq, DK]
    device const half*  k_src   [[buffer(1)]],  // [n_heads, seq, DK]
    device const half*  v_src   [[buffer(2)]],  // [n_heads, seq, DV]
    device       half*  output  [[buffer(3)]],  // [seq, n_heads * DV]
    constant     uint&  seq_len [[buffer(4)]],
    constant     uint&  n_heads [[buffer(5)]],
    constant     uint&  head_dim [[buffer(6)]],
    constant     float& scale   [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup half* shmem [[threadgroup(0)]])
{
    const uint iq1 = tgpig.x * Q_TOTAL;
    const uint h   = tgpig.y;
    if (h >= n_heads) return;

    const uint h_off_k = h * seq_len * DK;
    const uint h_off_v = h * seq_len * DV;

    // Stride for K/V in memory: head_dim (contiguous within each key vector)
    const uint NS_K = DK;  // stride between consecutive K rows
    const uint NS_V = DV;  // stride between consecutive V rows

    // Shared memory layout
    const short T = DK + 2 * PV;  // per query: DK(Q) + PV(O) + PV(O copy) ... simplified:
    // Actually simpler:
    //   sq[Q_TOTAL * DK] half  — all queries
    //   so[Q_TOTAL * PV] FLOAT — output accumulator (F32 for precision)
    //   ss[Q_TOTAL * SH] float — scores scratch
    threadgroup half4*   sq4 = (threadgroup half4*)shmem;
    threadgroup half*    sq  = (threadgroup half*)shmem;
    threadgroup float4*  so4 = (threadgroup float4*)(sq + Q_TOTAL * DK);
    threadgroup float*   so  = (threadgroup float*)(sq + Q_TOTAL * DK);
    threadgroup float*   ss  = (threadgroup float*)(so + Q_TOTAL * PV);
    threadgroup float2*  ss2 = (threadgroup float2*)ss;

    // Load Q into shared — contiguous per simdgroup (sg0: rows 0-7, sg1: rows 8-15)
    {
        const short q_off = sgitg * Q_PER_SG;
        for (short jj = 0; jj < Q_PER_SG; ++jj) {
            const short j = q_off + jj;
            device const half4* q4 = (device const half4*)(q_src + h_off_k + (iq1 + j) * DK);
            for (short i = tiisg; i < DK/4; i += NW) {
                sq4[j * DK/4 + i] = (iq1 + j < seq_len) ? q4[i] : half4(0);
            }
        }
    }

    // Zero output and scores — contiguous
    {
        const short q_off = sgitg * Q_PER_SG;
        for (short jj = 0; jj < Q_PER_SG; ++jj) {
            const short j = q_off + jj;
            for (short i = tiisg; i < DV4; i += NW) so4[j * PV4 + i] = float4(0);
            for (short i = tiisg; i < SH; i += NW)  ss[j * SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Running softmax state per query (Q_PER_SG=8 per simdgroup)
    float S[Q_PER_SG], M[Q_PER_SG];
    for (short j = 0; j < Q_PER_SG; j++) { S[j] = 0.0f; M[j] = -FLT_MAX/2; }

    // Loop over KV tiles
    for (uint ic = 0; ic < seq_len; ic += C_TILE) {
        // === Q·K^T via simdgroup_matrix ===
        // K is read directly from device memory
        device const half* pk = k_src + h_off_k + ic * DK;
        threadgroup const half* pq = sq + sgitg * Q_PER_SG * DK;  // Q rows for this SG

        // Each SG handles ALL C_TILE keys (not split across SGs) for its 8 queries
        threadgroup float* ps = ss + sgitg * Q_PER_SG * SH;  // score rows for this SG

        for (short cc = 0; cc < C_TILE/8; ++cc) {
            simdgroup_matrix<float, 8, 8> mqk = simdgroup_matrix<float, 8, 8>(0);

            for (short i = 0; i < DK8/2; ++i) {
                simdgroup_matrix<half, 8, 8> mq[2], mk[2];

                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_load(mq[0], pq + 16*i + 0, DK);
                simdgroup_load(mq[1], pq + 16*i + 8, DK);
                simdgroup_load(mk[0], pk + cc*8*NS_K + 16*i + 0, NS_K, 0, true);
                simdgroup_load(mk[1], pk + cc*8*NS_K + 16*i + 8, NS_K, 0, true);
                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
            }

            simdgroup_store(mqk, ps + cc*8, SH, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Online softmax ===
        for (short jj = 0; jj < Q_PER_SG; ++jj) {
            const short j = sgitg * Q_PER_SG + jj;
            const float m = M[jj];

            float2 s2 = ss2[j * SH/2 + tiisg] * scale;

            // Mask: beyond C_TILE (garbage from unused columns) and beyond seq_len
            uint k0 = ic + tiisg * 2;
            if (tiisg * 2     >= C_TILE || k0     >= seq_len) s2[0] = -FLT_MAX/2;
            if (tiisg * 2 + 1 >= C_TILE || k0 + 1 >= seq_len) s2[1] = -FLT_MAX/2;

            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

            const float  ms  = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);

            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);

            // Store softmax probabilities
            ss2[j * SH/2 + tiisg] = vs2;

            // Correct output accumulator: O *= exp(m_old - m_new)
            for (short i = tiisg; i < PV4; i += NW) {
                so4[j * PV4 + i] *= ms;  // float4 *= float
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === S · V via simdgroup_matrix: float += float × half ===
        // V original layout: [n_heads, seq, DV]
        {
            threadgroup float* my_so = so + sgitg * Q_PER_SG * PV;

            simdgroup_matrix<float, 8, 8> lo[DV8];
            for (short dk = 0; dk < DV8; ++dk)
                simdgroup_load(lo[dk], my_so + dk * 8, PV);

            const uint v_base = h * seq_len * DV;

            for (short cc = 0; cc < C_TILE/8; ++cc) {
                simdgroup_matrix<float, 8, 8> vs;
                simdgroup_load(vs, ss + sgitg * Q_PER_SG * SH + cc * 8, SH);

                for (short dk = 0; dk < DV8; ++dk) {
                    simdgroup_matrix<half, 8, 8> mv;
                    device const half* pv = v_src + v_base + (ic + cc*8) * DV + dk * 8;
                    simdgroup_load(mv, pv, DV);
                    simdgroup_multiply_accumulate(lo[dk], vs, mv, lo[dk]);
                }
            }

            for (short dk = 0; dk < DV8; ++dk)
                simdgroup_store(lo[dk], my_so + dk * 8, PV);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Finalize: O /= S ===
    for (short jj = 0; jj < Q_PER_SG; ++jj) {
        const short j = sgitg * Q_PER_SG + jj;
        if (iq1 + j >= seq_len) continue;

        const float inv_s = 1.0f / S[jj];

        for (short i = tiisg; i < DV4; i += NW) {
            float4 o = so4[j * PV4 + i] * inv_s;
            // Write to output: [seq, n_heads * DV]
            device half* out_row = output + (iq1 + j) * n_heads * DV + h * DV;
            device half4* out4 = (device half4*)out_row;
            out4[i] = half4(o);
        }
    }
}
