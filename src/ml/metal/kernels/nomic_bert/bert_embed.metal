// Fused dequantize-matmul + BERT ops for nomic-embed-text-v2-moe
// Supports Q5_K and Q6_K quantized weight matrices.
//
// Key kernel: matmul_dequant_q5k / q6k
//   Reads quantized weight bytes, dequantizes per-block, accumulates dot product.
//   Each thread computes one output element (one row of W dot input vector).

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Quantization constants and block structures
// ============================================================================

constant uint QK_K = 256;  // Super-block size

// Q5_K block: 176 bytes per 256 elements
// Layout: [d:f16][dmin:f16][scales:12B][qh:32B][qs:128B]
struct block_q5_K {
    half d;           // super-block scale
    half dmin;        // super-block min scale
    uint8_t scales[12]; // 6-bit scales/mins packed
    uint8_t qh[32];    // high bits
    uint8_t qs[128];   // low 4 bits
};

// Q6_K block: 210 bytes per 256 elements
// Layout: [ql:128B][qh:64B][scales:16B][d:f16]
struct block_q6_K {
    uint8_t ql[128];   // lower 4 bits
    uint8_t qh[64];    // upper 2 bits
    int8_t  scales[16]; // 8-bit scales
    half d;             // super-block scale
};

// Extract 6-bit scale and min from packed Q5_K scales
inline float2 get_scale_min_k4(int j, const device uint8_t* scales) {
    float sc, m;
    if (j < 4) {
        sc = float(scales[j] & 63);
        m  = float(scales[j + 4] & 63);
    } else {
        sc = float((scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4));
        m  = float((scales[j+4] >>  4) | ((scales[j]   >> 6) << 4));
    }
    return float2(sc, m);
}

// ============================================================================
// Fused dequant-matmul: Q5_K weights × F32 input → F32 output
// ============================================================================
//
// Each thread computes: output[row] = Σ dequant(W[row, j]) * x[j] + bias[row]
// W is stored as raw Q5_K blocks, row-major.
// Grid: [out_dim] threads, each handles one output row.

kernel void matmul_dequant_q5k(
    device const uint8_t* w_raw   [[buffer(0)]],  // Quantized weight [out_dim rows]
    device const float*   x       [[buffer(1)]],  // Input vector [in_dim]
    device const float*   bias    [[buffer(2)]],  // Bias [out_dim]
    device       float*   output  [[buffer(3)]],  // Output [out_dim]
    constant     uint&    in_dim  [[buffer(4)]],  // Input dimension
    constant     uint&    out_dim [[buffer(5)]],  // Output dimension
    uint tid [[thread_position_in_grid]])
{
    if (tid >= out_dim) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 176;  // Q5_K block = 176 bytes
    device const uint8_t* row_ptr = w_raw + tid * row_bytes;

    float sum = bias[tid];

    for (uint blk = 0; blk < blocks_per_row; blk++) {
        device const uint8_t* bp = row_ptr + blk * 176;
        const float d    = float(as_type<half>(*(device const ushort*)(bp)));
        const float dmin = float(as_type<half>(*(device const ushort*)(bp + 2)));
        device const uint8_t* scales = bp + 4;
        device const uint8_t* qh = bp + 16;
        device const uint8_t* ql = bp + 48;

        const uint base_j = blk * QK_K;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        uint ql_off = 0;

        for (int iter = 0; iter < 4; iter++) {
            float2 sm0 = get_scale_min_k4(is, scales);
            float d1 = d * sm0.x;
            float m1 = dmin * sm0.y;
            float2 sm1 = get_scale_min_k4(is + 1, scales);
            float d2 = d * sm1.x;
            float m2 = dmin * sm1.y;

            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + l;
                float val = d1 * float((ql[ql_off + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
                sum += x[j] * val;
            }
            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + 32 + l;
                float val = d2 * float(((ql[ql_off + l] >> 4) & 0x0F) + ((qh[l] & u2) ? 16 : 0)) - m2;
                sum += x[j] * val;
            }

            ql_off += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    output[tid] = sum;
}

// ============================================================================
// Fused dequant-matmul: Q6_K weights × F32 input → F32 output
// ============================================================================

kernel void matmul_dequant_q6k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= out_dim) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 210;
    device const uint8_t* row_ptr = w_raw + tid * row_bytes;

    float sum = bias[tid];

    for (uint blk = 0; blk < blocks_per_row; blk++) {
        device const uint8_t* bp = row_ptr + blk * 210;
        device const uint8_t* ql = bp;
        device const uint8_t* qh = bp + 128;
        device const int8_t*  sc = (device const int8_t*)(bp + 192);
        const float d = float(as_type<half>(*(device const ushort*)(bp + 208)));

        const uint base_j = blk * QK_K;
        uint ql_off = 0, qh_off = 0, sc_off = 0;

        for (int n_iter = 0; n_iter < 2; n_iter++) {
            for (uint l = 0; l < 32; l++) {
                uint is = l / 16;
                int q1 = (int(ql[ql_off + l]      & 0xF) | ((int(qh[qh_off + l] >> 0) & 3) << 4)) - 32;
                int q2 = (int(ql[ql_off + l + 32]  & 0xF) | ((int(qh[qh_off + l] >> 2) & 3) << 4)) - 32;
                int q3 = (int(ql[ql_off + l]       >> 4)  | ((int(qh[qh_off + l] >> 4) & 3) << 4)) - 32;
                int q4 = (int(ql[ql_off + l + 32]  >> 4)  | ((int(qh[qh_off + l] >> 6) & 3) << 4)) - 32;

                float s0 = float(sc[sc_off + is]);
                float s2 = float(sc[sc_off + is + 2]);
                float s4 = float(sc[sc_off + is + 4]);
                float s6 = float(sc[sc_off + is + 6]);

                uint j_base = base_j + n_iter * 128;
                sum += x[j_base + l]      * (d * s0 * float(q1));
                sum += x[j_base + l + 32] * (d * s2 * float(q2));
                sum += x[j_base + l + 64] * (d * s4 * float(q3));
                sum += x[j_base + l + 96] * (d * s6 * float(q4));
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }

    output[tid] = sum;
}

// ============================================================================
// Batched matmul for multiple input rows (e.g., seq_len > 1)
// Grid: [out_dim, batch] threads
// ============================================================================

kernel void matmul_dequant_q5k_batched(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],  // [batch, in_dim]
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],  // [batch, out_dim]
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]])  // x=out_dim, y=batch
{
    if (tid.x >= out_dim || tid.y >= batch) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 176;
    device const uint8_t* row_ptr = w_raw + tid.x * row_bytes;
    device const float* x_row = x + tid.y * in_dim;

    float sum = bias[tid.x];

    for (uint blk = 0; blk < blocks_per_row; blk++) {
        device const uint8_t* bp = row_ptr + blk * 176;
        const float d    = float(as_type<half>(*(device const ushort*)(bp)));
        const float dmin = float(as_type<half>(*(device const ushort*)(bp + 2)));
        device const uint8_t* scales = bp + 4;
        device const uint8_t* qh = bp + 16;
        device const uint8_t* ql = bp + 48;

        const uint base_j = blk * QK_K;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        uint ql_off = 0;

        for (int iter = 0; iter < 4; iter++) {
            float2 sm0 = get_scale_min_k4(is, scales);
            float d1 = d * sm0.x, m1 = dmin * sm0.y;
            float2 sm1 = get_scale_min_k4(is + 1, scales);
            float d2 = d * sm1.x, m2 = dmin * sm1.y;

            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + l;
                sum += x_row[j] * (d1 * float((ql[ql_off + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1);
            }
            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + 32 + l;
                sum += x_row[j] * (d2 * float(((ql[ql_off + l] >> 4) & 0x0F) + ((qh[l] & u2) ? 16 : 0)) - m2);
            }
            ql_off += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }

    output[tid.y * out_dim + tid.x] = sum;
}

// ============================================================================
// NeoX RoPE: rotate pairs (i, i + head_dim/2)
// Grid: [seq_len * n_heads] threads
// ============================================================================

kernel void rope_neox(
    device float* qk          [[buffer(0)]],  // [seq, dim] (Q or K, in-place)
    constant uint& seq_len    [[buffer(1)]],
    constant uint& n_heads    [[buffer(2)]],
    constant uint& head_dim   [[buffer(3)]],
    device const float* cos_cache [[buffer(4)]],  // [max_seq, head_dim/2]
    device const float* sin_cache [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = seq_len * n_heads;
    if (tid >= total) return;

    const uint pos = tid / n_heads;
    const uint h = tid % n_heads;
    const uint half_dim = head_dim / 2;
    const uint base = pos * (n_heads * head_dim) + h * head_dim;
    const uint rope_off = pos * half_dim;

    for (uint i = 0; i < half_dim; i++) {
        float c = cos_cache[rope_off + i];
        float s = sin_cache[rope_off + i];
        float v0 = qk[base + i];
        float v1 = qk[base + i + half_dim];
        qk[base + i]        = v0 * c - v1 * s;
        qk[base + i + half_dim] = v0 * s + v1 * c;
    }
}

// ============================================================================
// LayerNorm: in-place normalization
// Grid: [n_positions] threads
// ============================================================================

kernel void layernorm_bert(
    device float* x          [[buffer(0)]],  // [n_pos, dim], in-place
    device const float* w    [[buffer(1)]],  // [dim]
    device const float* b    [[buffer(2)]],  // [dim]
    constant uint& dim       [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    device float* row = x + tid * dim;

    // Mean
    float mean = 0.0f;
    for (uint j = 0; j < dim; j++) mean += row[j];
    mean /= float(dim);

    // Variance
    float var = 0.0f;
    for (uint j = 0; j < dim; j++) {
        float d = row[j] - mean;
        var += d * d;
    }
    var /= float(dim);

    float inv_std = rsqrt(var + eps);

    // Normalize + scale + shift
    for (uint j = 0; j < dim; j++) {
        row[j] = (row[j] - mean) * inv_std * w[j] + b[j];
    }
}

// ============================================================================
// GELU activation: in-place
// Grid: [n_elements] threads
// ============================================================================

kernel void gelu_inplace(
    device float* x [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    float v = x[tid];
    x[tid] = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
}

// ============================================================================
// Mean pooling + L2 normalize
// Grid: 1 thread (small operation, not worth parallelizing further)
// ============================================================================

kernel void mean_pool_normalize(
    device const float* hidden [[buffer(0)]],  // [seq_len, dim]
    device       float* output [[buffer(1)]],  // [dim]
    constant     uint&  seq_len [[buffer(2)]],
    constant     uint&  dim     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid > 0) return;

    float inv_seq = 1.0f / float(seq_len);

    // Mean pool
    for (uint j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (uint p = 0; p < seq_len; p++) {
            sum += hidden[p * dim + j];
        }
        output[j] = sum * inv_seq;
    }

    // L2 normalize
    float norm = 0.0f;
    for (uint j = 0; j < dim; j++) norm += output[j] * output[j];
    norm = rsqrt(norm + 1e-8f);
    for (uint j = 0; j < dim; j++) output[j] *= norm;
}
