// FP16 intermediate kernels for BERT transformer
// All intermediate buffers (hidden, QKV, attn, FFN) use half precision.
// Accumulation in F32, final conversion to half on output.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

// ============================================================================
// QKV split: [seq, 3*dim] half → Q, K half [n_heads, seq, head_dim]
//                                 V_t half [n_heads, head_dim, seq]
// ============================================================================
kernel void qkv_split(
    device const half*  qkv    [[buffer(0)]],
    device       half*  Q      [[buffer(1)]],
    device       half*  K      [[buffer(2)]],
    device       half*  V      [[buffer(3)]],  // original layout
    device       half*  V_t    [[buffer(4)]],  // transposed layout
    constant     uint&  seq_len  [[buffer(5)]],
    constant     uint&  dim      [[buffer(6)]],
    constant     uint&  n_heads  [[buffer(7)]],
    constant     uint&  head_dim [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= seq_len * dim) return;
    const uint pos = tid / dim;
    const uint d = tid % dim;
    const uint h = d / head_dim;
    const uint hd = d % head_dim;
    const uint src = pos * 3 * dim;
    const uint dst = h * seq_len * head_dim + pos * head_dim + hd;
    const uint vt = h * head_dim * seq_len + hd * seq_len + pos;
    half v_val = qkv[src + 2 * dim + d];
    Q[dst]  = qkv[src + d];
    K[dst]  = qkv[src + dim + d];
    V[dst]  = v_val;
    V_t[vt] = v_val;
}

// ============================================================================
// Fused QKV split + RoPE: [seq, 3*dim] → Q(rope'd), K(rope'd), V, V_t
// Eliminates 3 dispatches (split + 2×rope) + 2 barriers per layer
// Dispatch: dispatch_1d(seq_len * dim, 256)
// ============================================================================
kernel void qkv_split_rope(
    device const half*   qkv     [[buffer(0)]],
    device       half*   Q       [[buffer(1)]],
    device       half*   K       [[buffer(2)]],
    device       half*   V       [[buffer(3)]],
    device       half*   V_t     [[buffer(4)]],
    device const float*  cos_t   [[buffer(5)]],
    device const float*  sin_t   [[buffer(6)]],
    constant     uint&   seq_len  [[buffer(7)]],
    constant     uint&   dim      [[buffer(8)]],
    constant     uint&   n_heads  [[buffer(9)]],
    constant     uint&   head_dim [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= seq_len * dim) return;
    const uint pos = tid / dim;
    const uint d = tid % dim;
    const uint h = d / head_dim;
    const uint hd = d % head_dim;
    const uint hd2 = head_dim / 2;
    const uint src = pos * 3 * dim;
    const uint dst = h * seq_len * head_dim + pos * head_dim + hd;

    // V and V_t: straight copy (no RoPE)
    half v_val = qkv[src + 2 * dim + d];
    V[dst] = v_val;
    V_t[h * head_dim * seq_len + hd * seq_len + pos] = v_val;

    // Q and K: split + RoPE NeoX
    float q_raw = float(qkv[src + d]);
    float k_raw = float(qkv[src + dim + d]);

    if (hd < hd2) {
        // First half: v0 position
        float q_pair = float(qkv[src + h * head_dim + hd + hd2]);  // wrong: need full dim offset
        float k_pair = float(qkv[src + dim + h * head_dim + hd + hd2]);
        // Actually the pair element is at the same head, same pos, but hd+hd2
        // qkv layout: [pos * 3*dim + d] where d = h*head_dim + hd
        // pair: d_pair = h*head_dim + hd + hd2
        uint d_pair = h * head_dim + hd + hd2;
        float q1 = float(qkv[src + d_pair]);
        float k1 = float(qkv[src + dim + d_pair]);
        float c = cos_t[pos * hd2 + hd];
        float s = sin_t[pos * hd2 + hd];
        Q[dst] = half(q_raw * c - q1 * s);
        K[dst] = half(k_raw * c - k1 * s);
    } else {
        // Second half: v1 position
        uint hd_lo = hd - hd2;
        uint d_pair = h * head_dim + hd_lo;
        float q0 = float(qkv[src + d_pair]);
        float k0 = float(qkv[src + dim + d_pair]);
        float c = cos_t[pos * hd2 + hd_lo];
        float s = sin_t[pos * hd2 + hd_lo];
        Q[dst] = half(q0 * s + q_raw * c);
        K[dst] = half(k0 * s + k_raw * c);
    }
}

// ============================================================================
// RoPE NeoX in-place on half Q/K
// ============================================================================
kernel void rope_neox_inplace(
    device half*       qk      [[buffer(0)]],
    device const float* cos_t  [[buffer(1)]],
    device const float* sin_t  [[buffer(2)]],
    constant uint& seq_len     [[buffer(3)]],
    constant uint& n_heads     [[buffer(4)]],
    constant uint& head_dim    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= seq_len * n_heads) return;
    const uint h = tid / seq_len;
    const uint pos = tid % seq_len;
    const uint hd2 = head_dim / 2;
    const uint base = h * (seq_len * head_dim) + pos * head_dim;
    const uint rope_off = pos * hd2;
    for (uint i = 0; i < hd2; i++) {
        float c = cos_t[rope_off + i];
        float s = sin_t[rope_off + i];
        float v0 = float(qk[base + i]);
        float v1 = float(qk[base + i + hd2]);
        qk[base + i]       = half(v0 * c - v1 * s);
        qk[base + i + hd2] = half(v0 * s + v1 * c);
    }
}

// ============================================================================
// Attention: shared scores + V_t float4, FP16 I/O
// ============================================================================
constant uint N_QR = 8;

kernel void attention_forward(
    device const half*  Q       [[buffer(0)]],
    device const half*  K       [[buffer(1)]],
    device const half*  V_t     [[buffer(2)]],  // [n_heads, head_dim, seq] transposed
    device       half*  output  [[buffer(3)]],
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
    const uint i = tgpig.y * N_QR + sgitg;
    if (h >= n_heads || i >= seq_len) return;

    const uint h_off = h * seq_len * head_dim;
    const uint lane = tiisg;
    const uint hd4 = head_dim / 4;
    threadgroup float* shared = shared_base + sgitg * seq_len;

    // Cache Q in registers as float (promote from half)
    device const half* qi = Q + h_off + i * head_dim;
    float qr[64]; // max head_dim
    for (uint d = 0; d < head_dim; d++) qr[d] = float(qi[d]);

    // Q·K dot products (half4 K reads)
    float local_max = -1e30f;
    for (uint j = lane; j < seq_len; j += 32) {
        device const half4* kj4 = (device const half4*)(K + h_off + j * head_dim);
        float dot = 0.0f;
        for (uint d4 = 0; d4 < hd4; d4++) {
            half4 k4 = kj4[d4];
            dot += qr[d4*4]*float(k4.x) + qr[d4*4+1]*float(k4.y) + qr[d4*4+2]*float(k4.z) + qr[d4*4+3]*float(k4.w);
        }
        float s = dot * scale;
        shared[j] = s;
        local_max = max(local_max, s);
    }

    float global_max = simd_max(local_max);
    float local_sum = 0.0f;
    for (uint j = lane; j < seq_len; j += 32) {
        float e = exp(shared[j] - global_max);
        shared[j] = e;
        local_sum += e;
    }
    float inv_sum = 1.0f / simd_sum(local_sum);
    for (uint j = lane; j < seq_len; j += 32) shared[j] *= inv_sum;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // V accumulation from V_t (half4 reads for bandwidth, F32 accumulate)
    const uint vt_h_off = h * head_dim * seq_len;
    for (uint d = lane; d < head_dim; d += 32) {
        device const half* vt_row = V_t + vt_h_off + d * seq_len;
        float val = 0.0f;
        uint j = 0;
        for (; j + 3 < seq_len; j += 4) {
            half4 v4 = *(device const half4*)(vt_row + j);
            val += shared[j]*float(v4.x) + shared[j+1]*float(v4.y) + shared[j+2]*float(v4.z) + shared[j+3]*float(v4.w);
        }
        for (; j < seq_len; j++) val += shared[j] * float(vt_row[j]);
        output[i * (n_heads * head_dim) + h * head_dim + d] = half(val);
    }
}

// ============================================================================
// Fused residual + layernorm (half I/O, F32 compute)
// ============================================================================
kernel void residual_layernorm(
    device half*       x    [[buffer(0)]],
    device const half* y    [[buffer(1)]],
    device const float* w   [[buffer(2)]],
    device const float* b   [[buffer(3)]],
    constant uint& dim      [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pos = tgpig.x;
    const uint lane = tiisg;
    device half* row = x + pos * dim;
    device const half* y_row = y + pos * dim;

    float local_sum = 0.0f;
    for (uint j = lane; j < dim; j += 32) {
        float v = float(row[j]) + float(y_row[j]);
        row[j] = half(v);
        local_sum += v;
    }
    float mean = simd_sum(local_sum) / float(dim);

    float local_var = 0.0f;
    for (uint j = lane; j < dim; j += 32) { float d = float(row[j]) - mean; local_var += d * d; }
    float inv_std = rsqrt(simd_sum(local_var) / float(dim) + 1e-5f);

    for (uint j = lane; j < dim; j += 32) {
        row[j] = half((float(row[j]) - mean) * inv_std * w[j] + b[j]);
    }
}

// Variant: reads f32 residual (for post-atomic-scatter MoE norm2, skips f32→f16 dispatch)
kernel void residual_layernorm_f32(
    device half*        x    [[buffer(0)]],
    device const float* y_f32 [[buffer(1)]],
    device const float* w    [[buffer(2)]],
    device const float* b    [[buffer(3)]],
    constant uint& dim       [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pos = tgpig.x;
    const uint lane = tiisg;
    device half* row = x + pos * dim;

    float local_sum = 0.0f;
    for (uint j = lane; j < dim; j += 32) {
        float v = float(row[j]) + y_f32[pos * dim + j];
        row[j] = half(v);
        local_sum += v;
    }
    float mean = simd_sum(local_sum) / float(dim);

    float local_var = 0.0f;
    for (uint j = lane; j < dim; j += 32) { float d = float(row[j]) - mean; local_var += d * d; }
    float inv_std = rsqrt(simd_sum(local_var) / float(dim) + 1e-5f);

    for (uint j = lane; j < dim; j += 32) {
        row[j] = half((float(row[j]) - mean) * inv_std * w[j] + b[j]);
    }
}

// ============================================================================
// Residual + layernorm with COPY: out = layernorm(x + y) — different output buffer
// ============================================================================
kernel void residual_layernorm_copy(
    device const half* x    [[buffer(0)]],   // input 1 (read-only)
    device const half* y    [[buffer(1)]],   // input 2 (residual)
    device       half* out  [[buffer(2)]],   // output (different from x and y)
    device const float* w   [[buffer(3)]],
    device const float* b   [[buffer(4)]],
    constant uint& dim      [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pos = tgpig.x;
    const uint lane = tiisg;

    float local_sum = 0.0f;
    for (uint j = lane; j < dim; j += 32) {
        float v = float(x[pos * dim + j]) + float(y[pos * dim + j]);
        out[pos * dim + j] = half(v);
        local_sum += v;
    }
    float mean = simd_sum(local_sum) / float(dim);

    float local_var = 0.0f;
    for (uint j = lane; j < dim; j += 32) { float d = float(out[pos * dim + j]) - mean; local_var += d * d; }
    float inv_std = rsqrt(simd_sum(local_var) / float(dim) + 1e-5f);

    for (uint j = lane; j < dim; j += 32) {
        out[pos * dim + j] = half((float(out[pos * dim + j]) - mean) * inv_std * w[j] + b[j]);
    }
}

// ============================================================================
// GELU in-place (half)
// ============================================================================
kernel void gelu_inplace(
    device half* x      [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    float v = float(x[tid]);
    if (v > 10.0f) { x[tid] = half(v); return; }
    if (v < -10.0f) { x[tid] = half(0); return; }
    float t = 0.7978845608f * (v + 0.044715f * v * v * v);
    x[tid] = half(0.5f * v * (1.0f + tanh(t)));
}

// ============================================================================
// Fused SIMD gate + softmax + top-k + expert_count
// 1 simdgroup (32 threads) per token — parallel gate matmul via simd_sum.
// Dispatch: threadgroups = {seq_len, 1, 1}, threads = {32, 1, 1}
// ============================================================================
kernel void gate_softmax_topk_count(
    device const half*   hidden       [[buffer(0)]],
    device const float*  gate_w       [[buffer(1)]],
    device       int*    routing_ids  [[buffer(2)]],
    device       float*  routing_wts  [[buffer(3)]],
    device atomic_int*   expert_counts [[buffer(4)]],
    constant     uint&   dim          [[buffer(5)]],
    constant     uint&   n_experts    [[buffer(6)]],
    constant     uint&   k            [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pos = tgpig.x;
    const uint lane = tiisg;

    // Gate matmul: 32 threads cooperatively compute 8 dot products via simd_sum
    float partial[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    device const half* h_row = hidden + pos * dim;
    for (uint j = lane; j < dim; j += 32) {
        float hv = float(h_row[j]);
        for (uint e = 0; e < 8; e++) {
            partial[e] += hv * gate_w[e * dim + j];
        }
    }
    // Reduce across simdgroup → lane 0 gets final logits
    float logits[8];
    for (uint e = 0; e < 8; e++) {
        logits[e] = simd_sum(partial[e]);
    }

    // Softmax + top-k (lane 0 only, trivial for 8 elements)
    if (lane == 0) {
        float max_g = logits[0];
        for (uint e = 1; e < n_experts; e++) max_g = max(max_g, logits[e]);
        float sum_exp = 0.0f;
        for (uint e = 0; e < n_experts; e++) { logits[e] = exp(logits[e] - max_g); sum_exp += logits[e]; }
        float inv_sum = 1.0f / sum_exp;
        for (uint e = 0; e < n_experts; e++) logits[e] *= inv_sum;

        uint out_base = pos * k;
        for (uint i = 0; i < k; i++) {
            float best_p = -1.0f; int best_e = 0;
            for (uint e = 0; e < n_experts; e++) {
                if (logits[e] > best_p) { best_p = logits[e]; best_e = (int)e; }
            }
            routing_ids[out_base + i] = best_e;
            routing_wts[out_base + i] = best_p;
            logits[best_e] = -1.0f;
            atomic_fetch_add_explicit(&expert_counts[best_e], 1, memory_order_relaxed);
        }
    }
}

// ============================================================================
// Gate matmul: hidden(half) @ gate_w(F32) → logits(F32)
// ============================================================================
kernel void gate_matmul(
    device const half*  hidden   [[buffer(0)]],
    device const float* gate_w   [[buffer(1)]],
    device       float* output   [[buffer(2)]],
    constant     uint&  dim      [[buffer(3)]],
    constant     uint&  n_experts [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint e = gid.x;
    const uint pos = gid.y;
    float sum = 0.0f;
    for (uint j = 0; j < dim; j++) {
        sum += float(hidden[pos * dim + j]) * gate_w[e * dim + j];
    }
    output[pos * n_experts + e] = sum;
}

// ============================================================================
// Softmax + Top-K (same as before — operates on F32 logits)
// ============================================================================
kernel void softmax_topk(
    device const float* gate_logits  [[buffer(0)]],
    device       int*   routing_ids  [[buffer(1)]],
    device       float* routing_wts  [[buffer(2)]],
    constant     uint&  n_experts    [[buffer(3)]],
    constant     uint&  k            [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    device const float* row = gate_logits + tid * n_experts;
    float max_g = row[0];
    for (uint e = 1; e < n_experts; e++) max_g = max(max_g, row[e]);
    float sum_exp = 0.0f;
    float probs[8];
    for (uint e = 0; e < n_experts; e++) { probs[e] = exp(row[e] - max_g); sum_exp += probs[e]; }
    float inv_sum = 1.0f / sum_exp;
    for (uint e = 0; e < n_experts; e++) probs[e] *= inv_sum;
    uint out_base = tid * k;
    for (uint i = 0; i < k; i++) {
        float best_p = -1.0f; int best_e = 0;
        for (uint e = 0; e < n_experts; e++) {
            if (probs[e] > best_p) { best_p = probs[e]; best_e = (int)e; }
        }
        routing_ids[out_base + i] = best_e;
        routing_wts[out_base + i] = best_p;
        probs[best_e] = -1.0f;
    }
}

// ============================================================================
// Zero int32 buffer
// ============================================================================
kernel void zero_int(
    device int* x [[buffer(0)]],
    uint tid [[thread_position_in_grid]])
{
    x[tid] = 0;
}

// ============================================================================
// GPU MoE routing: build gather_map + scatter_wts from routing_ids/wts
// Uses atomic counters per expert to build contiguous expert groups.
// Grid: [seq_len]  (one thread per token position)
// ============================================================================

kernel void moe_build_routing(
    device const int*   routing_ids  [[buffer(0)]],  // [seq, k] expert indices
    device const float* routing_wts  [[buffer(1)]],  // [seq, k] weights
    device       int*   gather_map   [[buffer(2)]],  // [total_routing] output: pos indices
    device       float* scatter_wts  [[buffer(3)]],  // [total_routing] output: weights
    device       int*   expert_offsets [[buffer(4)]], // [n_experts+1] prefix sums (pre-computed)
    device atomic_int*  expert_counts [[buffer(5)]],  // [n_experts] atomic counters
    constant     uint&  k            [[buffer(6)]],  // n_experts_used (=2)
    constant     uint&  n_experts    [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    const uint pos = tid;
    for (uint ki = 0; ki < k; ki++) {
        int ei = routing_ids[pos * k + ki];
        float w = routing_wts[pos * k + ki];
        // Atomically get slot within this expert's group
        int slot = atomic_fetch_add_explicit(&expert_counts[ei], 1, memory_order_relaxed);
        int dest = expert_offsets[ei] + slot;
        gather_map[dest] = (int)pos;
        scatter_wts[dest] = w;
    }
}

// ============================================================================
// Count tokens per expert (for prefix sum)
// Grid: [seq_len]
// ============================================================================

kernel void moe_count_experts(
    device const int*   routing_ids  [[buffer(0)]],
    device atomic_int*  expert_counts [[buffer(1)]],
    constant     uint&  k            [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    for (uint ki = 0; ki < k; ki++) {
        int ei = routing_ids[tid * k + ki];
        atomic_fetch_add_explicit(&expert_counts[ei], 1, memory_order_relaxed);
    }
}

// ============================================================================
// Prefix sum for expert offsets (single-thread, tiny: 8 experts)
// Grid: [1]
// ============================================================================

kernel void moe_prefix_sum(
    device const int* expert_counts [[buffer(0)]],
    device       int* expert_offsets [[buffer(1)]],
    constant     uint& n_experts    [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    int sum = 0;
    for (uint e = 0; e < n_experts; e++) {
        expert_offsets[e] = sum;
        sum += expert_counts[e];
    }
    expert_offsets[n_experts] = sum;
}

// ============================================================================
// Fused: prefix_sum + zero_counts + build_routing + write_dispatch_args
// One dispatch replaces 4. Thread 0 computes prefix sum, then all build routing.
// Dispatch: {seq_len, 1, 1} threadgroups, {32, 1, 1} threads (or 1 TG if seq<32)
// ============================================================================
kernel void moe_route_and_dispatch(
    device const int*    routing_ids   [[buffer(0)]],
    device const float*  routing_wts   [[buffer(1)]],
    device       int*    gather_map    [[buffer(2)]],
    device       float*  scatter_wts   [[buffer(3)]],
    device       int*    expert_counts [[buffer(4)]],   // input: counts from gate kernel
    device       int*    expert_offsets [[buffer(5)]],   // output: prefix sums
    device       uint*   dispatch_args [[buffer(6)]],   // output: indirect dispatch args
    constant     uint&   k             [[buffer(7)]],
    constant     uint&   n_experts     [[buffer(8)]],
    constant     uint&   seq_len       [[buffer(9)]],
    constant     uint&   up_out_dim    [[buffer(10)]],
    constant     uint&   down_out_dim  [[buffer(11)]],
    constant     uint&   dim           [[buffer(12)]],
    uint tid  [[thread_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]])
{
    // Step 1: Thread 0 computes prefix sum + dispatch args + zeros counts
    if (tid == 0) {
        int sum = 0;
        for (uint e = 0; e < n_experts; e++) {
            int ec = atomic_load_explicit((device atomic_int*)&expert_counts[e], memory_order_relaxed);
            expert_offsets[e] = sum;
            atomic_store_explicit((device atomic_int*)&expert_counts[e], 0, memory_order_relaxed);
            sum += ec;

            uint eb_u = (uint)max(ec, 0);
            // UP dispatch args
            dispatch_args[e * 3 + 0] = (eb_u + 31) / 32;
            dispatch_args[e * 3 + 1] = (up_out_dim + 63) / 64;
            dispatch_args[e * 3 + 2] = 1;
            // DOWN dispatch args
            dispatch_args[(n_experts + e) * 3 + 0] = (eb_u + 31) / 32;
            dispatch_args[(n_experts + e) * 3 + 1] = (down_out_dim + 63) / 64;
            dispatch_args[(n_experts + e) * 3 + 2] = 1;
        }
        expert_offsets[n_experts] = sum;
    }

    // All threads wait for prefix sum + zeros to complete
    threadgroup_barrier(mem_flags::mem_device);

    // Step 2: All threads build routing (same as moe_build_routing)
    if (tid < seq_len) {
        for (uint ki = 0; ki < k; ki++) {
            int ei = routing_ids[tid * k + ki];
            float w = routing_wts[tid * k + ki];
            int slot = atomic_fetch_add_explicit(
                (device atomic_int*)&expert_counts[ei], 1, memory_order_relaxed);
            int dest = expert_offsets[ei] + slot;
            gather_map[dest] = (int)tid;
            scatter_wts[dest] = w;
        }
    }
}

// ============================================================================
// Write indirect dispatch args from expert_offsets (GPU-side)
// Grid: [n_experts]
// Layout: [8 UP args, 8 DOWN args, 8 scatter args] x 3 uint32 each
// ============================================================================
// Write indirect dispatch args for MoE mm kernels (simdgroup_matrix)
// Grid for mm: {ceil(eb/32), ceil(out_dim/64), 1}
// Grid for scatter: {ceil(eb*dim/256), 1, 1}
kernel void moe_write_dispatch_args(
    device const int*  expert_offsets [[buffer(0)]],
    device       uint* dispatch_args [[buffer(1)]],
    constant     uint& up_out_dim    [[buffer(2)]],   // ffn_dim (for UP matmul)
    constant     uint& down_out_dim  [[buffer(3)]],   // dim (for DOWN matmul)
    constant     uint& dim           [[buffer(4)]],   // dim (for scatter)
    constant     uint& n_experts     [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_experts) return;
    int eb = expert_offsets[tid + 1] - expert_offsets[tid];
    uint eb_u = (uint)max(eb, 0);

    // UP mm args: {ceil(eb/32), ceil(ffn_dim/64), 1}
    uint up_idx = tid * 3;
    dispatch_args[up_idx + 0] = (eb_u + 31) / 32;
    dispatch_args[up_idx + 1] = (up_out_dim + 63) / 64;
    dispatch_args[up_idx + 2] = 1;

    // DOWN mm args: {ceil(eb/32), ceil(dim/64), 1}
    uint down_idx = (n_experts + tid) * 3;
    dispatch_args[down_idx + 0] = (eb_u + 31) / 32;
    dispatch_args[down_idx + 1] = (down_out_dim + 63) / 64;
    dispatch_args[down_idx + 2] = 1;

    // Scatter args: {ceil(eb*dim/256), 1, 1}
    uint sc_idx = (2 * n_experts + tid) * 3;
    uint sc_threads = eb_u * dim;
    dispatch_args[sc_idx + 0] = (sc_threads + 255) / 256;
    dispatch_args[sc_idx + 1] = 1;
    dispatch_args[sc_idx + 2] = 1;
}

// ============================================================================
// Batched expert dispatch args: compute expert_tg_offsets (prefix sum of per-expert TG counts)
// + total grid size for batched_mm kernels. Single-thread kernel.
// Grid: [1]
// ============================================================================
kernel void moe_write_batched_args(
    device const int*  expert_offsets  [[buffer(0)]],  // [n_experts+1] token offsets
    device       int*  expert_tg_offs [[buffer(1)]],   // [n_experts+1] output: TG prefix sums
    device       uint* up_grid        [[buffer(2)]],   // [3] output: indirect dispatch args for batched UP
    device       uint* down_grid      [[buffer(3)]],   // [3] output: indirect dispatch args for batched DOWN
    constant     uint& n_experts      [[buffer(4)]],
    constant     uint& up_out_dim     [[buffer(5)]],   // ffn_dim
    constant     uint& down_out_dim   [[buffer(6)]],   // dim
    uint tid [[thread_position_in_grid]])
{
    int sum_tg = 0;
    for (uint e = 0; e < n_experts; e++) {
        expert_tg_offs[e] = sum_tg;
        int eb = expert_offsets[e + 1] - expert_offsets[e];
        sum_tg += (max(eb, 0) + 31) / 32;  // ceil(eb / MM_NR1=32)
    }
    expert_tg_offs[n_experts] = sum_tg;
    // UP indirect dispatch: {total_batch_tgs, ceil(ffn_dim/64), 1}
    up_grid[0] = (uint)sum_tg;
    up_grid[1] = (up_out_dim + 63) / 64;
    up_grid[2] = 1;
    // DOWN indirect dispatch: {total_batch_tgs, ceil(dim/64), 1}
    down_grid[0] = (uint)sum_tg;
    down_grid[1] = (down_out_dim + 63) / 64;
    down_grid[2] = 1;
}

// ============================================================================
// MoE gather: hidden(half) → moe_input(half)
// ============================================================================
kernel void moe_gather(
    device const half*  hidden     [[buffer(0)]],
    device       half*  moe_input  [[buffer(1)]],
    device const int*   gather_map [[buffer(2)]],
    constant     uint&  dim        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    const uint ri = tid / dim;
    const uint j  = tid % dim;
    const uint pos = (uint)gather_map[ri];
    moe_input[ri * dim + j] = hidden[pos * dim + j];
}

// ============================================================================
// Scatter weighted add (half)
// ============================================================================
kernel void scatter_weighted_add(
    device       half*  ffn_out     [[buffer(0)]],
    device const half*  expert_out  [[buffer(1)]],
    device const int*   scatter_map [[buffer(2)]],
    device const float* weights     [[buffer(3)]],
    constant     uint&  dim         [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    const uint ri = tid / dim;
    const uint j  = tid % dim;
    const uint pos = (uint)scatter_map[ri];
    ffn_out[pos * dim + j] = half(float(ffn_out[pos * dim + j]) + weights[ri] * float(expert_out[ri * dim + j]));
}

// ============================================================================
// Atomic MoE scatter — ALL routing slots in ONE dispatch, no sequential barriers
// Uses atomic_fetch_add on float buffer (Metal 3). After scatter, convert f32→f16.
// Dispatch: dispatch_1d(total_routing * dim, 256)
// ============================================================================
kernel void moe_scatter_atomic(
    device atomic_float* ffn_out_f32  [[buffer(0)]],  // float accumulator (zeroed before)
    device const half*   expert_out   [[buffer(1)]],  // packed expert output
    device const int*    gather_map   [[buffer(2)]],  // [total_routing] → token index
    device const float*  scatter_wts  [[buffer(3)]],  // [total_routing] weights
    constant     uint&   dim          [[buffer(4)]],
    constant     uint&   total_routing [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= total_routing * dim) return;
    const uint ri = tid / dim;
    const uint d  = tid % dim;
    const int pos = gather_map[ri];
    const float val = scatter_wts[ri] * float(expert_out[ri * dim + d]);
    atomic_fetch_add_explicit(&ffn_out_f32[pos * dim + d], val, memory_order_relaxed);
}

// Convert float buffer → half buffer (after atomic scatter)
// Dispatch: dispatch_1d(count, 256)
kernel void f32_to_f16(
    device const float* src [[buffer(0)]],
    device       half*  dst [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    dst[tid] = half(src[tid]);
}

// ============================================================================
// MoE scatter — GPU-side expert_offsets bounds, zero CPU sync
// Dispatch: dispatch_1d(seq_len * dim, 256) per expert (excess threads exit)
// ============================================================================
kernel void scatter_weighted_add_moe(
    device       half*  ffn_out       [[buffer(0)]],
    device const half*  expert_out    [[buffer(1)]],
    device const int*   scatter_map   [[buffer(2)]],
    device const float* weights       [[buffer(3)]],
    device const int*   expert_offs   [[buffer(4)]],
    constant     uint&  expert_id     [[buffer(5)]],
    constant     uint&  dim           [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const int base = expert_offs[expert_id];
    const int eb = expert_offs[expert_id + 1] - base;
    const int total = eb * (int)dim;
    if ((int)tid >= total) return;

    const uint ri = base + tid / dim;
    const uint j  = tid % dim;
    const uint pos = (uint)scatter_map[ri];
    ffn_out[pos * dim + j] = half(float(ffn_out[pos * dim + j]) + weights[ri] * float(expert_out[ri * dim + j]));
}

// ============================================================================
// MoE weighted scatter (half) — for sync-free path
// ============================================================================
kernel void moe_weighted_scatter(
    device       half*  ffn_out      [[buffer(0)]],
    device const half*  expert_out   [[buffer(1)]],
    device const int*   routing_ids  [[buffer(2)]],
    device const float* routing_wts  [[buffer(3)]],
    constant     uint&  dim          [[buffer(4)]],
    constant     uint&  seq_len      [[buffer(5)]],
    constant     uint&  k            [[buffer(6)]],
    constant     uint&  n_experts    [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    const uint pos = tid / dim;
    const uint j   = tid % dim;
    if (pos >= seq_len) return;
    float sum = 0.0f;
    for (uint ki = 0; ki < k; ki++) {
        const uint ei = (uint)routing_ids[pos * k + ki];
        sum += routing_wts[pos * k + ki] * float(expert_out[ei * seq_len * dim + pos * dim + j]);
    }
    ffn_out[pos * dim + j] = half(sum);
}

// ============================================================================
// Residual add (half)
// ============================================================================
kernel void residual_add(
    device half*       x [[buffer(0)]],
    device const half* y [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    x[tid] = half(float(x[tid]) + float(y[tid]));
}

// ============================================================================
// Zero region (half)
// ============================================================================
kernel void zero_region(
    device half* x     [[buffer(0)]],
    constant uint& off [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    x[off + tid] = half(0);
}

// ============================================================================
// Weighted add (half) — for sync-based MoE path
// ============================================================================
kernel void weighted_add(
    device half*       dst    [[buffer(0)]],
    device const half* src    [[buffer(1)]],
    constant uint& pos_offset [[buffer(2)]],
    constant float& weight   [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    dst[pos_offset + tid] = half(float(dst[pos_offset + tid]) + weight * float(src[tid]));
}

// ============================================================================
// Mean pool + L2 normalize (half input → F32 output)
// ============================================================================
kernel void mean_pool_l2(
    device const half*  hidden  [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  seq_len [[buffer(2)]],
    constant     uint&  dim     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    // Mean pool
    for (uint d = 0; d < dim; d++) {
        float sum = 0.0f;
        for (uint p = 0; p < seq_len; p++) sum += float(hidden[p * dim + d]);
        output[d] = sum / float(seq_len);
    }
    // L2 normalize
    float norm = 0.0f;
    for (uint d = 0; d < dim; d++) norm += output[d] * output[d];
    norm = rsqrt(norm + 1e-12f);
    for (uint d = 0; d < dim; d++) output[d] *= norm;
}

// ============================================================================
// LayerNorm in-place (half, for embedding norm before layers)
// ============================================================================
kernel void layernorm_inplace(
    device half*        x  [[buffer(0)]],
    device const float* w  [[buffer(1)]],
    device const float* b  [[buffer(2)]],
    constant uint& dim     [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint pos = tgpig.x;
    const uint lane = tiisg;
    device half* row = x + pos * dim;

    float local_sum = 0.0f;
    for (uint j = lane; j < dim; j += 32) local_sum += float(row[j]);
    float mean = simd_sum(local_sum) / float(dim);

    float local_var = 0.0f;
    for (uint j = lane; j < dim; j += 32) { float d = float(row[j]) - mean; local_var += d * d; }
    float inv_std = rsqrt(simd_sum(local_var) / float(dim) + 1e-5f);

    for (uint j = lane; j < dim; j += 32) {
        row[j] = half((float(row[j]) - mean) * inv_std * w[j] + b[j]);
    }
}
