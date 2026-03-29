// Fused BERT transformer layer — all ops in one kernel dispatch.
//
// For nomic-embed-text-v2-moe (seq≤512, dim=768, 12 heads, head_dim=64):
// Each threadgroup processes one output row of matmul.
// Multiple dispatches per layer but all in ONE command buffer.
//
// This file provides individual fused ops that the Crystal backend
// orchestrates into a single command buffer per forward pass.

#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;

// ============================================================================
// Q5_K block structure and helpers (same as bert_embed.metal)
// ============================================================================

inline float2 get_scale_min_k4(int j, const device uint8_t* scales) {
    float sc, m;
    if (j < 4) { sc = float(scales[j] & 63); m = float(scales[j + 4] & 63); }
    else { sc = float((scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)); m = float((scales[j+4] >> 4) | ((scales[j] >> 6) << 4)); }
    return float2(sc, m);
}

// ============================================================================
// Fused dequant-matmul + bias + optional GELU — single kernel
// Computes: output[row] = act(Σ dequant(W[row, j]) * x[j] + bias[row])
// act = GELU if apply_gelu=true, identity otherwise
//
// Grid: [out_dim × batch] threads
// ============================================================================

kernel void fused_q5k_matmul_gelu(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],  // [batch, in_dim]
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],  // [batch, out_dim]
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],  // 0 or 1
    uint2 gid [[thread_position_in_grid]])
{
    const uint o = gid.x;
    const uint b = gid.y;
    if (o >= out_dim || b >= batch) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 176;
    device const uint8_t* row_ptr = w_raw + o * row_bytes;
    device const float* x_row = x + b * in_dim;

    // Kahan compensated summation for near-double precision
    float sum = bias[o];
    float comp = 0.0f;  // compensation for lost low-order bits

    for (uint blk = 0; blk < blocks_per_row; blk++) {
        device const uint8_t* bp = row_ptr + blk * 176;
        const float d    = float(as_type<half>(*(device const ushort*)(bp)));
        const float dmin = float(as_type<half>(*(device const ushort*)(bp + 2)));
        device const uint8_t* scales = bp + 4;
        device const uint8_t* qh = bp + 16;
        device const uint8_t* ql = bp + 48;
        const uint base_j = blk * QK_K;
        uint8_t u1 = 1, u2 = 2;
        int is = 0; uint ql_off = 0;

        for (int iter = 0; iter < 4; iter++) {
            float2 sm0 = get_scale_min_k4(is, scales);
            float d1 = d * sm0.x, m1 = dmin * sm0.y;
            float2 sm1 = get_scale_min_k4(is + 1, scales);
            float d2 = d * sm1.x, m2 = dmin * sm1.y;

            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + l;
                float term = x_row[j] * (d1 * float((ql[ql_off + l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1);
                float y = term - comp; float t = sum + y; comp = (t - sum) - y; sum = t;
            }
            for (uint l = 0; l < 32; l++) {
                uint j = base_j + (is / 2) * 64 + 32 + l;
                float term = x_row[j] * (d2 * float(((ql[ql_off + l] >> 4) & 0x0F) + ((qh[l] & u2) ? 16 : 0)) - m2);
                float y = term - comp; float t = sum + y; comp = (t - sum) - y; sum = t;
            }
            ql_off += 32; is += 2; u1 <<= 2; u2 <<= 2;
        }
    }

    // Optional GELU
    // Guard: if accumulation produced NaN (rare GPU precision edge case), zero it
    if (isnan(sum)) sum = 0.0f;

    if (apply_gelu) {
        float v = clamp(sum, -20.0f, 20.0f);
        sum = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }

    output[b * out_dim + o] = sum;
}

// Same for Q6_K
kernel void fused_q6k_matmul_gelu(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint o = gid.x;
    const uint b = gid.y;
    if (o >= out_dim || b >= batch) return;

    const uint blocks_per_row = in_dim / QK_K;
    const uint row_bytes = blocks_per_row * 210;
    device const uint8_t* row_ptr = w_raw + o * row_bytes;
    device const float* x_row = x + b * in_dim;

    float sum = bias[o];
    float comp = 0.0f;

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
                { float term = x_row[j_base + l]      * (d * s0 * float(q1)); float y = term - comp; float t = sum + y; comp = (t - sum) - y; sum = t; }
                { float term = x_row[j_base + l + 32] * (d * s2 * float(q2)); float y = term - comp; float t = sum + y; comp = (t - sum) - y; sum = t; }
                { float term = x_row[j_base + l + 64] * (d * s4 * float(q3)); float y = term - comp; float t = sum + y; comp = (t - sum) - y; sum = t; }
                { float term = x_row[j_base + l + 96] * (d * s6 * float(q4)); float y = term - comp; float t = sum + y; comp = (t - sum) - y; sum = t; }
            }
            ql_off += 64; qh_off += 32; sc_off += 8;
        }
    }

    // Guard: if accumulation produced NaN (rare GPU precision edge case), zero it
    if (isnan(sum)) sum = 0.0f;

    if (apply_gelu) {
        float v = clamp(sum, -20.0f, 20.0f);
        sum = 0.5f * v * (1.0f + tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }

    output[b * out_dim + o] = sum;
}

// ============================================================================
// NeoX RoPE — in-place on Q and K buffers
// Grid: [seq_len * n_heads]
// ============================================================================

kernel void rope_neox_inplace(
    device float* qk          [[buffer(0)]],
    device const float* cos_t [[buffer(1)]],
    device const float* sin_t [[buffer(2)]],
    constant uint& seq_len    [[buffer(3)]],
    constant uint& n_heads    [[buffer(4)]],
    constant uint& head_dim   [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= seq_len * n_heads) return;
    const uint h = tid / seq_len;
    const uint pos = tid % seq_len;
    const uint hd2 = head_dim / 2;
    // Q/K layout: [n_heads, seq_len, head_dim]
    const uint base = h * (seq_len * head_dim) + pos * head_dim;
    const uint rope_off = pos * hd2;

    for (uint i = 0; i < hd2; i++) {
        float c = cos_t[rope_off + i];
        float s = sin_t[rope_off + i];
        float v0 = qk[base + i];
        float v1 = qk[base + i + hd2];
        qk[base + i]       = v0 * c - v1 * s;
        qk[base + i + hd2] = v0 * s + v1 * c;
    }
}

// ============================================================================
// Batched attention: scores = Q @ K^T * scale, softmax, output = scores @ V
// One thread per (head, query_pos) pair
// Grid: [n_heads * seq_len]
// ============================================================================

kernel void attention_forward(
    device const float* Q       [[buffer(0)]],  // [n_heads, seq_len, head_dim]
    device const float* K       [[buffer(1)]],  // [n_heads, seq_len, head_dim]
    device const float* V       [[buffer(2)]],  // [n_heads, seq_len, head_dim]
    device       float* output  [[buffer(3)]],  // [seq_len, n_heads * head_dim]
    constant     uint&  seq_len [[buffer(4)]],
    constant     uint&  n_heads [[buffer(5)]],
    constant     uint&  head_dim [[buffer(6)]],
    constant     float& scale   [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_heads * seq_len) return;

    const uint h = tid / seq_len;
    const uint i = tid % seq_len;  // query position
    const uint h_off = h * seq_len * head_dim;

    // Compute attention scores for this (head, query_pos)
    // scores[j] = Q[h,i] · K[h,j] * scale
    float max_score = -1e30f;

    // Use threadgroup memory would be better, but for short seqs this works
    // Stack allocation for scores (seq_len ≤ 512)
    float scores[512];

    for (uint j = 0; j < seq_len; j++) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d++) {
            dot += Q[h_off + i * head_dim + d] * K[h_off + j * head_dim + d];
        }
        scores[j] = dot * scale;
        max_score = max(max_score, scores[j]);
    }

    // Softmax
    float sum_exp = 0.0f;
    for (uint j = 0; j < seq_len; j++) {
        scores[j] = exp(scores[j] - max_score);
        sum_exp += scores[j];
    }
    float inv_sum = 1.0f / sum_exp;

    // Weighted sum of V → output[i, h*head_dim : (h+1)*head_dim]
    for (uint d = 0; d < head_dim; d++) {
        float val = 0.0f;
        for (uint j = 0; j < seq_len; j++) {
            val += scores[j] * inv_sum * V[h_off + j * head_dim + d];
        }
        output[i * (n_heads * head_dim) + h * head_dim + d] = val;
    }
}

// ============================================================================
// LayerNorm in-place: x = (x - mean) / sqrt(var + eps) * w + b
// Grid: [n_positions]
// ============================================================================

kernel void layernorm_inplace(
    device float* x          [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device const float* b    [[buffer(2)]],
    constant uint& dim       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    device float* row = x + tid * dim;
    float mean = 0.0f;
    for (uint j = 0; j < dim; j++) mean += row[j];
    mean /= float(dim);
    float var = 0.0f;
    for (uint j = 0; j < dim; j++) { float d = row[j] - mean; var += d * d; }
    var /= float(dim);
    float inv_std = rsqrt(var + 1e-5f);
    for (uint j = 0; j < dim; j++) {
        row[j] = (row[j] - mean) * inv_std * w[j] + b[j];
    }
}

// ============================================================================
// Residual add: x += y (in-place)
// Grid: [n_elements]
// ============================================================================

kernel void residual_add(
    device float* x       [[buffer(0)]],
    device const float* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    x[tid] += y[tid];
}

// ============================================================================
// QKV split from fused [seq, 3*dim] → separate Q, K, V in [n_heads, seq, head_dim]
// Grid: [seq_len * dim]
// ============================================================================

kernel void qkv_split(
    device const float* qkv   [[buffer(0)]],  // [seq, 3*dim]
    device       float* Q     [[buffer(1)]],   // [n_heads, seq, head_dim]
    device       float* K     [[buffer(2)]],
    device       float* V     [[buffer(3)]],
    constant     uint&  seq_len  [[buffer(4)]],
    constant     uint&  dim      [[buffer(5)]],
    constant     uint&  n_heads  [[buffer(6)]],
    constant     uint&  head_dim [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= seq_len * dim) return;
    const uint pos = tid / dim;
    const uint d = tid % dim;
    const uint h = d / head_dim;
    const uint hd = d % head_dim;

    const uint src_base = pos * 3 * dim;
    const uint dst = h * seq_len * head_dim + pos * head_dim + hd;

    Q[dst] = qkv[src_base + d];
    K[dst] = qkv[src_base + dim + d];
    V[dst] = qkv[src_base + 2 * dim + d];
}

// ============================================================================
// Fused MoE FFN: gate → softmax → top-2 → expert up+GELU → expert down → accumulate
// One thread per (position, output_dim) — each computes one output element.
// Grid: [dim, seq_len]
// ============================================================================

kernel void moe_ffn_fused(
    device const float*   hidden     [[buffer(0)]],   // [seq, dim] input
    device       float*   output     [[buffer(1)]],   // [seq, dim] output (additive: += result)
    device const float*   gate_w     [[buffer(2)]],   // [n_experts, dim] F32
    device const uint8_t* up_exps    [[buffer(3)]],   // All expert up weights (quantized)
    device const uint8_t* down_exps  [[buffer(4)]],   // All expert down weights (quantized)
    constant     uint&    dim        [[buffer(5)]],
    constant     uint&    ffn_dim    [[buffer(6)]],
    constant     uint&    n_experts  [[buffer(7)]],
    constant     uint&    n_used     [[buffer(8)]],   // top-k (=2)
    constant     uint&    up_expert_bytes  [[buffer(9)]],
    constant     uint&    down_expert_bytes [[buffer(10)]],
    constant     uint&    up_type    [[buffer(11)]],  // 0=Q5_K, 1=Q6_K
    constant     uint&    seq_len    [[buffer(12)]],
    uint2 gid [[thread_position_in_grid]])             // x=dim_idx, y=pos
{
    const uint d_out = gid.x;  // which output dimension
    const uint pos = gid.y;    // which token position
    if (d_out >= dim || pos >= seq_len) return;

    device const float* x = hidden + pos * dim;

    // Step 1: Gate logits (x @ gate_w^T) — computed per-thread for ALL experts
    float gate_logits[8];  // max 8 experts
    for (uint e = 0; e < n_experts; e++) {
        float dot = 0.0f;
        for (uint j = 0; j < dim; j++) {
            dot += x[j] * gate_w[e * dim + j];
        }
        gate_logits[e] = dot;
    }

    // Step 2: Softmax
    float max_g = gate_logits[0];
    for (uint e = 1; e < n_experts; e++) max_g = max(max_g, gate_logits[e]);
    float sum_exp = 0.0f;
    for (uint e = 0; e < n_experts; e++) {
        gate_logits[e] = exp(gate_logits[e] - max_g);
        sum_exp += gate_logits[e];
    }
    for (uint e = 0; e < n_experts; e++) gate_logits[e] /= sum_exp;

    // Step 3: Top-2 selection
    int top_idx[2] = {0, 1};
    float top_prob[2] = {gate_logits[0], gate_logits[1]};
    if (top_prob[1] > top_prob[0]) {
        int ti = top_idx[0]; top_idx[0] = top_idx[1]; top_idx[1] = ti;
        float tp = top_prob[0]; top_prob[0] = top_prob[1]; top_prob[1] = tp;
    }
    for (uint e = 2; e < n_experts; e++) {
        if (gate_logits[e] > top_prob[1]) {
            top_prob[1] = gate_logits[e];
            top_idx[1] = e;
            if (top_prob[1] > top_prob[0]) {
                int ti = top_idx[0]; top_idx[0] = top_idx[1]; top_idx[1] = ti;
                float tp = top_prob[0]; top_prob[0] = top_prob[1]; top_prob[1] = tp;
            }
        }
    }

    // Step 4: For each selected expert, compute up+GELU → down for output dim d_out
    float result = 0.0f;

    for (uint t = 0; t < n_used; t++) {
        int ei = top_idx[t];
        float w = top_prob[t];

        // Expert down: need full ffn_dim intermediate, but we compute one output dim
        // down[d_out] = Σ_f h[f] * down_w[d_out, f]
        // where h[f] = GELU(Σ_j x[j] * up_w[f, j])

        // This requires computing ALL ffn_dim values of h first (up+GELU),
        // then dot with down_w row d_out.
        // For ffn_dim=3072, this is expensive per thread but correct.

        // Expert up: compute h[f] = GELU(x @ up_w[ei][f, :]) for all f
        device const uint8_t* up_base = up_exps + ei * up_expert_bytes;
        device const uint8_t* dn_base = down_exps + ei * down_expert_bytes;

        // Down matmul: for output d_out, dot with h
        // down_w[ei] is [dim, ffn_dim] stored as [dim rows, ffn_dim cols] quantized
        // Row d_out covers ffn_dim input elements

        // Since computing ALL h[f] per thread is expensive (3072 dot products of 768),
        // we instead compute: out[d_out] = Σ_f GELU(up_dot_f) * down_w[d_out, f]
        // = fused up-GELU-down for one output element

        float acc = 0.0f;

        // For Q5_K: each block is 176 bytes, 256 elements
        // up_w[ei] is [ffn_dim, dim] = ffn_dim rows of dim elements
        // We need h[f] for all f, then dot with down row

        // This is O(ffn_dim * dim) per thread — too expensive.
        // With dim=768, ffn_dim=3072: 2.4M ops per thread per expert.
        // With 768 threads (one per d_out): total 1.8B ops per expert.
        // That's way too much redundant work.

        // Better approach: two-phase
        // Phase 1: compute h[f] = GELU(up[f] dot x) for all f (one thread per f)
        // Phase 2: compute out[d] = Σ_f h[f] * down[d,f] (one thread per d)
        // But this requires two dispatches and shared memory between them.

        // For a FUSED single-kernel approach: use threadgroup shared memory
        // Phase 1: all threads in threadgroup compute h[f] cooperatively
        // Phase 2: each thread computes its output dim from shared h

        // But this is complex. For now, let's do the two-phase approach
        // with separate dispatches per expert (still in same command buffer).

        // Actually, skip fused for now — just use separate up/down dispatches
        // with the weighted_add pattern. Same cmd, no sync.
        break;  // Can't do fully fused without shared memory
    }

    // Placeholder — this kernel is incomplete. Use dispatch-per-expert approach instead.
    // output[pos * dim + d_out] += result;
}

// ============================================================================
// Weighted add: dst[offset + i] += weight * src[i]
// Grid: [count]
// ============================================================================

kernel void weighted_add(
    device       float* dst    [[buffer(0)]],
    device const float* src    [[buffer(1)]],
    constant     uint&  offset [[buffer(2)]],
    constant     float& weight [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    dst[offset + tid] += weight * src[tid];
}

// ============================================================================
// Zero buffer region: dst[offset..offset+count] = 0
// Grid: [count]
// ============================================================================

kernel void zero_region(
    device float* dst      [[buffer(0)]],
    constant uint& offset  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[offset + tid] = 0.0f;
}

// ============================================================================
// Mean pool + L2 normalize: hidden[seq, dim] → output[dim]
// Single-thread kernel (fast enough for dim=768)
// ============================================================================

kernel void mean_pool_l2(
    device const float* hidden [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  seq_len [[buffer(2)]],
    constant     uint&  dim     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid > 0) return;
    float inv_seq = 1.0f / float(seq_len);
    for (uint j = 0; j < dim; j++) {
        float sum = 0.0f;
        for (uint p = 0; p < seq_len; p++) sum += hidden[p * dim + j];
        output[j] = sum * inv_seq;
    }
    float norm = 0.0f;
    for (uint j = 0; j < dim; j++) norm += output[j] * output[j];
    norm = rsqrt(norm + 1e-8f);
    for (uint j = 0; j < dim; j++) output[j] *= norm;
}
