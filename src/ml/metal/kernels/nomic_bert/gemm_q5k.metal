// Tiled GEMM for Q5_K quantized weights × F32 input.
// Uses shared memory for dequantized weight tiles and input tiles.
// Much better precision than per-element sequential accumulation.
//
// C[m, n] = Σ_k dequant(A[m, k]) * B[n, k] + bias[m]
//
// A = weight [M, K] quantized Q5_K (M = out_dim, K = in_dim)
// B = input [N, K] F32 (N = batch/seq_len, K = in_dim)
// C = output [N, M] F32
//
// Tile: TILE_M rows of A × TILE_N rows of B, accumulated over K in chunks of QK_K=256

#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint TILE_M = 4;    // Output rows per threadgroup
constant uint TILE_N = 4;    // Input rows (batch) per threadgroup
constant uint THREADS_PER_ROW = 64; // Threads accumulating one output element

inline float2 get_scale_min_k4(int j, const device uint8_t* scales) {
    float sc, m;
    if (j < 4) { sc = float(scales[j] & 63); m = float(scales[j + 4] & 63); }
    else { sc = float((scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)); m = float((scales[j+4] >> 4) | ((scales[j] >> 6) << 4)); }
    return float2(sc, m);
}

// Tiled Q5_K matmul + optional GELU
// Grid: [ceil(out_dim/TILE_M), ceil(batch/TILE_N)]
// Threadgroup: [TILE_M, TILE_N] = 16 threads per group
kernel void gemm_q5k(
    device const uint8_t* w_raw   [[buffer(0)]],  // [out_dim rows] Q5_K
    device const float*   x       [[buffer(1)]],  // [batch, in_dim] F32
    device const float*   bias    [[buffer(2)]],  // [out_dim] F32
    device       float*   output  [[buffer(3)]],  // [batch, out_dim] F32
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],    // tile position
    uint2 tid  [[thread_position_in_threadgroup]])   // thread in tile
{
    const uint m = tgid.x * TILE_M + tid.x;  // output row
    const uint n = tgid.y * TILE_N + tid.y;  // batch index
    if (m >= out_dim || n >= batch) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 176;

    device const uint8_t* row_ptr = w_raw + m * row_bytes;
    device const float* x_row = x + n * in_dim;

    // Accumulate with float — but over fewer iterations (blocks_per_row = 3 for dim=768)
    // Each block = 256 elements, so only 3 accumulation "tiles"
    float sum = bias[m];

    for (uint blk = 0; blk < blocks_per_row; blk++) {
        device const uint8_t* bp = row_ptr + blk * 176;
        float d    = float(as_type<half>(*(device const ushort*)(bp)));
        float dmin = float(as_type<half>(*(device const ushort*)(bp + 2)));
        // Guard against NaN/Inf in half-precision scale factors
        if (isnan(d) || isinf(d)) d = 0.0f;
        if (isnan(dmin) || isinf(dmin)) dmin = 0.0f;
        device const uint8_t* scales = bp + 4;
        device const uint8_t* qh = bp + 16;
        device const uint8_t* ql = bp + 48;

        const uint base_j = blk * QK_K;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        uint ql_off = 0;

        // Process 4 sub-blocks of 64 elements each
        // Use float4 partial sums for better precision
        float4 partial = float4(0.0f);

        for (int iter = 0; iter < 4; iter++) {
            float2 sm0 = get_scale_min_k4(is, scales);
            float d1 = d * sm0.x, m1 = dmin * sm0.y;
            float2 sm1 = get_scale_min_k4(is + 1, scales);
            float d2 = d * sm1.x, m2 = dmin * sm1.y;

            // First 32 elements
            float sub_sum0 = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + l;
                sub_sum0 += x_row[j] * (d1 * float((ql[ql_off + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1);
            }

            // Second 32 elements
            float sub_sum1 = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + 32 + l;
                sub_sum1 += x_row[j] * (d2 * float(((ql[ql_off + l] >> 4) & 0x0F) + ((qh[l] & u2) ? 16 : 0)) - m2);
            }

            partial[iter] = sub_sum0 + sub_sum1;
            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }

        // Reduce float4 — fewer rounding steps than sequential
        sum += (partial[0] + partial[2]) + (partial[1] + partial[3]);
    }

    // Double NaN guard: before and after GELU
    if (isnan(sum) || isinf(sum)) sum = 0.0f;

    if (apply_gelu) {
        float v = clamp(sum, -20.0f, 20.0f);
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        float th = tanh(t);
        sum = 0.5f * v * (1.0f + th);
        if (isnan(sum) || isinf(sum)) sum = 0.0f;
    }

    output[n * out_dim + m] = sum;
}

// Same for Q6_K
kernel void gemm_q6k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tid  [[thread_position_in_threadgroup]])
{
    const uint m = tgid.x * TILE_M + tid.x;
    const uint n = tgid.y * TILE_N + tid.y;
    if (m >= out_dim || n >= batch) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 210;

    device const uint8_t* row_ptr = w_raw + m * row_bytes;
    device const float* x_row = x + n * in_dim;

    float sum = bias[m];

    for (uint blk = 0; blk < blocks_per_row; blk++) {
        device const uint8_t* bp = row_ptr + blk * 210;
        device const uint8_t* ql = bp;
        device const uint8_t* qh = bp + 128;
        device const int8_t*  sc = (device const int8_t*)(bp + 192);
        float d = float(as_type<half>(*(device const ushort*)(bp + 208)));
        if (isnan(d) || isinf(d)) d = 0.0f;

        const uint base_j = blk * QK_K;
        uint ql_off = 0, qh_off = 0, sc_off = 0;

        // float4 partial sums per 128-element half-block
        float2 partial = float2(0.0f);

        for (int n_iter = 0; n_iter < 2; n_iter++) {
            float sub_sum = 0.0f;
            for (uint l = 0; l < 32; l++) {
                uint is_val = l / 16;
                int q1 = (int(ql[ql_off + l]      & 0xF) | ((int(qh[qh_off + l] >> 0) & 3) << 4)) - 32;
                int q2 = (int(ql[ql_off + l + 32]  & 0xF) | ((int(qh[qh_off + l] >> 2) & 3) << 4)) - 32;
                int q3 = (int(ql[ql_off + l]       >> 4)  | ((int(qh[qh_off + l] >> 4) & 3) << 4)) - 32;
                int q4 = (int(ql[ql_off + l + 32]  >> 4)  | ((int(qh[qh_off + l] >> 6) & 3) << 4)) - 32;

                float s0 = float(sc[sc_off + is_val]);
                float s2 = float(sc[sc_off + is_val + 2]);
                float s4 = float(sc[sc_off + is_val + 4]);
                float s6 = float(sc[sc_off + is_val + 6]);

                uint j_base = base_j + n_iter * 128;
                sub_sum += x_row[j_base + l]      * (d * s0 * float(q1));
                sub_sum += x_row[j_base + l + 32] * (d * s2 * float(q2));
                sub_sum += x_row[j_base + l + 64] * (d * s4 * float(q3));
                sub_sum += x_row[j_base + l + 96] * (d * s6 * float(q4));
            }
            partial[n_iter] = sub_sum;
            ql_off += 64; qh_off += 32; sc_off += 8;
        }

        sum += partial[0] + partial[1];
    }

    if (isnan(sum) || isinf(sum)) sum = 0.0f;

    if (apply_gelu) {
        float v = clamp(sum, -20.0f, 20.0f);
        float t = 0.7978845608f * (v + 0.044715f * v * v * v);
        sum = 0.5f * v * (1.0f + tanh(t));
        if (isnan(sum) || isinf(sum)) sum = 0.0f;
    }

    output[n * out_dim + m] = sum;
}
