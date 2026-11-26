// Neural Network compute kernels for GPU acceleration
// Linear, LayerNorm, fused operations

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants
// ============================================================================

constant uint TILE_SIZE = 16;
constant float GELU_COEFF = 0.7978845608f;  // sqrt(2/pi)
constant float GELU_COEFF2 = 0.044715f;

// ============================================================================
// Linear Layer Forward
// ============================================================================

// Linear forward (tiled): out = input @ weight^T + bias
// input: [batch, in_features]
// weight: [out_features, in_features] (stored as [out, in], will be transposed)
// bias: [out_features]
// output: [batch, out_features]
kernel void linear_forward_tiled(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    // out[b, o] = sum_i(input[b, i] * weight[o, i]) + bias[o]
    // This is: input[batch, in] @ weight^T[in, out] = output[batch, out]

    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load input tile: input[row, t*TILE_SIZE + tid.x]
        uint in_col = t * TILE_SIZE + tid.x;
        if (row < batch && in_col < in_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = input[row * in_features + in_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load weight tile (transposed): weight[col, t*TILE_SIZE + tid.y]
        uint w_row = t * TILE_SIZE + tid.y;
        if (col < out_features && w_row < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[col * in_features + w_row];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < out_features) {
        float result = sum;
        if (use_bias != 0) {
            result += bias[col];
        }
        output[row * out_features + col] = result;
    }
}

// Linear forward without tiling (for small matrices)
kernel void linear_forward_simple(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    if (row >= batch || col >= out_features) return;

    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        sum += input[row * in_features + i] * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    output[row * out_features + col] = sum;
}

// ============================================================================
// Fused Linear + GELU
// ============================================================================

// GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline float gelu(float x) {
    float x3 = x * x * x;
    float inner = GELU_COEFF * (x + GELU_COEFF2 * x3);
    return 0.5f * x * (1.0f + tanh(inner));
}

// Fused linear + GELU: out = GELU(input @ weight^T + bias)
kernel void linear_gelu_forward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint in_col = t * TILE_SIZE + tid.x;
        if (row < batch && in_col < in_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = input[row * in_features + in_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint w_row = t * TILE_SIZE + tid.y;
        if (col < out_features && w_row < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[col * in_features + w_row];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < out_features) {
        if (use_bias != 0) {
            sum += bias[col];
        }
        output[row * out_features + col] = gelu(sum);
    }
}

// ============================================================================
// Fused Linear + ReLU
// ============================================================================

kernel void linear_relu_forward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& use_bias [[buffer(7)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    uint numTiles = (in_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint in_col = t * TILE_SIZE + tid.x;
        if (row < batch && in_col < in_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = input[row * in_features + in_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint w_row = t * TILE_SIZE + tid.y;
        if (col < out_features && w_row < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[col * in_features + w_row];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < out_features) {
        if (use_bias != 0) {
            sum += bias[col];
        }
        output[row * out_features + col] = max(0.0f, sum);
    }
}

// ============================================================================
// LayerNorm Forward
// ============================================================================

// Layer normalization with mean/var computation per sample
// input: [batch, features]
// gamma, beta: [features]
// output: [batch, features]
kernel void layernorm_forward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    if (batch_idx >= batch) return;

    uint offset = batch_idx * features;

    // Compute mean
    float mean = 0.0f;
    for (uint f = 0; f < features; f++) {
        mean += input[offset + f];
    }
    mean /= float(features);

    // Compute variance
    float var = 0.0f;
    for (uint f = 0; f < features; f++) {
        float diff = input[offset + f] - mean;
        var += diff * diff;
    }
    var /= float(features);

    // Normalize and apply affine
    float inv_std = rsqrt(var + eps);
    for (uint f = 0; f < features; f++) {
        float normalized = (input[offset + f] - mean) * inv_std;
        output[offset + f] = normalized * gamma[f] + beta[f];
    }
}

// Parallel LayerNorm using threadgroup reduction for large feature dimensions
kernel void layernorm_parallel(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    threadgroup float* shared_sum [[threadgroup(0)]],
    threadgroup float* shared_sq_sum [[threadgroup(1)]],
    uint batch_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (batch_idx >= batch) return;

    uint offset = batch_idx * features;

    // Each thread computes partial sum over its assigned features
    float local_sum = 0.0f;
    for (uint f = tid; f < features; f += tg_size) {
        local_sum += input[offset + f];
    }
    shared_sum[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for mean
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(features);

    // Compute variance
    float local_sq_sum = 0.0f;
    for (uint f = tid; f < features; f += tg_size) {
        float diff = input[offset + f] - mean;
        local_sq_sum += diff * diff;
    }
    shared_sq_sum[tid] = local_sq_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction for variance
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float var = shared_sq_sum[0] / float(features);
    float inv_std = rsqrt(var + eps);

    // Normalize and apply affine (each thread handles multiple features)
    for (uint f = tid; f < features; f += tg_size) {
        float normalized = (input[offset + f] - mean) * inv_std;
        output[offset + f] = normalized * gamma[f] + beta[f];
    }
}

// ============================================================================
// RMSNorm Forward
// ============================================================================

// RMS normalization: out = x / sqrt(mean(x^2) + eps) * gamma
kernel void rmsnorm_forward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& features [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint batch_idx [[thread_position_in_grid]]
) {
    if (batch_idx >= batch) return;

    uint offset = batch_idx * features;

    // Compute mean of squares
    float ms = 0.0f;
    for (uint f = 0; f < features; f++) {
        float x = input[offset + f];
        ms += x * x;
    }
    ms /= float(features);

    // Normalize and scale
    float inv_rms = rsqrt(ms + eps);
    for (uint f = 0; f < features; f++) {
        output[offset + f] = input[offset + f] * inv_rms * gamma[f];
    }
}

// ============================================================================
// Fused LayerNorm + Linear
// ============================================================================

// Fused: LayerNorm(input) @ weight^T + bias
// Saves one global memory read/write cycle
kernel void layernorm_linear_forward(
    device const float* input [[buffer(0)]],
    device const float* ln_gamma [[buffer(1)]],
    device const float* ln_beta [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& use_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Simple version: each thread computes one output element
    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    if (row >= batch || col >= out_features) return;

    uint offset = row * in_features;

    // Compute LayerNorm statistics (each thread does this for its batch element)
    float mean = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        mean += input[offset + f];
    }
    mean /= float(in_features);

    float var = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        float diff = input[offset + f] - mean;
        var += diff * diff;
    }
    var /= float(in_features);
    float inv_std = rsqrt(var + eps);

    // Compute linear projection with on-the-fly normalization
    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float normalized_val = (input[offset + i] - mean) * inv_std * ln_gamma[i] + ln_beta[i];
        sum += normalized_val * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    output[row * out_features + col] = sum;
}

// ============================================================================
// Linear Backward
// ============================================================================

// Linear backward for input gradient: grad_input = grad_output @ weight
kernel void linear_backward_input(
    device const float* grad_output [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    // grad_input[b, i] = sum_o(grad_output[b, o] * weight[o, i])
    uint row = gid.y;  // batch index
    uint col = gid.x;  // in_features index

    float sum = 0.0f;
    uint numTiles = (out_features + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load grad_output tile
        uint out_col = t * TILE_SIZE + tid.x;
        if (row < batch && out_col < out_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = grad_output[row * out_features + out_col];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load weight tile (not transposed this time)
        uint w_row = t * TILE_SIZE + tid.y;
        if (w_row < out_features && col < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = weight[w_row * in_features + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < batch && col < in_features) {
        grad_input[row * in_features + col] = sum;
    }
}

// Linear backward for weight gradient: grad_weight = grad_output^T @ input
kernel void linear_backward_weight(
    device const float* grad_output [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* grad_weight [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]]
) {
    // grad_weight[o, i] = sum_b(grad_output[b, o] * input[b, i])
    uint row = gid.y;  // out_features index
    uint col = gid.x;  // in_features index

    float sum = 0.0f;
    uint numTiles = (batch + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        // Load grad_output^T tile: grad_output[t*TILE_SIZE + tid.y, row]
        uint b_idx = t * TILE_SIZE + tid.y;
        if (b_idx < batch && row < out_features) {
            tileA[tid.y * TILE_SIZE + tid.x] = grad_output[b_idx * out_features + row];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // Load input tile
        uint b_idx2 = t * TILE_SIZE + tid.y;
        if (b_idx2 < batch && col < in_features) {
            tileB[tid.y * TILE_SIZE + tid.x] = input[b_idx2 * in_features + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[k * TILE_SIZE + tid.y] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < out_features && col < in_features) {
        grad_weight[row * in_features + col] = sum;
    }
}

// Linear backward for bias gradient: grad_bias = sum(grad_output, dim=0)
kernel void linear_backward_bias(
    device const float* grad_output [[buffer(0)]],
    device float* grad_bias [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& out_features [[buffer(3)]],
    threadgroup float* shared [[threadgroup(0)]],
    uint out_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (out_idx >= out_features) return;

    // Each thread sums over part of the batch
    float local_sum = 0.0f;
    for (uint b = tid; b < batch; b += tg_size) {
        local_sum += grad_output[b * out_features + out_idx];
    }
    shared[tid] = local_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        grad_bias[out_idx] = shared[0];
    }
}

// ============================================================================
// Softmax (for attention)
// ============================================================================

// Online softmax per row with numerical stability
kernel void softmax_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) return;

    uint offset = row * cols;

    // Find max for numerical stability
    float max_val = input[offset];
    for (uint c = 1; c < cols; c++) {
        max_val = max(max_val, input[offset + c]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float e = exp(input[offset + c] - max_val);
        output[offset + c] = e;
        sum += e;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint c = 0; c < cols; c++) {
        output[offset + c] *= inv_sum;
    }
}

// ============================================================================
// Batched Matrix Multiply (for attention QK^T and attn @ V)
// ============================================================================

// Batched matmul: C[b] = A[b] @ B[b]
// A: [batch, M, K], B: [batch, K, N], C: [batch, M, N]
kernel void batched_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint b = gid.z;  // batch index
    uint row = gid.y;  // M index
    uint col = gid.x;  // N index

    if (b >= batch) return;

    uint A_offset = b * M * K;
    uint B_offset = b * K * N;
    uint C_offset = b * M * N;

    float sum = 0.0f;
    uint numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < numTiles; t++) {
        uint aCol = t * TILE_SIZE + tid.x;
        if (row < M && aCol < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[A_offset + row * K + aCol];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        uint bRow = t * TILE_SIZE + tid.y;
        if (bRow < K && col < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[B_offset + bRow * N + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[C_offset + row * N + col] = sum;
    }
}

// Batched matmul with transpose: C[b] = A[b] @ B[b]^T
kernel void batched_matmul_tn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // Simple implementation without tiling for clarity
    // A: [batch, M, K], B: [batch, N, K], C: [batch, M, N]
    // C = A @ B^T (each row of B is treated as a column)

    uint b = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (b >= batch || row >= M || col >= N) return;

    uint A_offset = b * M * K;
    uint B_offset = b * N * K;
    uint C_offset = b * M * N;

    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[A_offset + row * K + k] * B[B_offset + col * K + k];
    }

    C[C_offset + row * N + col] = sum * scale;
}

// ============================================================================
// Fused Scaled Dot-Product Attention
// ============================================================================

// Fused attention: output = softmax(Q @ K^T / sqrt(d)) @ V
// Q, K, V: [batch * num_heads, seq_len, head_dim]
// Output: [batch * num_heads, seq_len, head_dim]
// This kernel fuses: matmul -> scale -> softmax -> matmul
kernel void fused_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],  // batch * num_heads
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]  // (head_dim, batch_heads * seq_len)
) {
    uint bh_seq = gid.y;  // combined batch_heads and seq index
    uint d = gid.x;       // head_dim index

    if (d >= head_dim || bh_seq >= batch_heads * seq_len) return;

    uint bh = bh_seq / seq_len;  // batch_head index
    uint i = bh_seq % seq_len;   // query position

    uint Q_offset = bh * seq_len * head_dim;
    uint K_offset = bh * seq_len * head_dim;
    uint V_offset = bh * seq_len * head_dim;

    // Step 1: Compute attention scores for this query position
    // scores[j] = Q[i] @ K[j]^T * scale
    // Using online softmax to avoid storing all scores

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc = 0.0f;

    // First pass: find max (for numerical stability)
    for (uint j = 0; j < seq_len; j++) {
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;
        max_score = max(max_score, score);
    }

    // Second pass: compute softmax and weighted sum of V
    for (uint j = 0; j < seq_len; j++) {
        // Recompute score (trade compute for memory)
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;

        float attn_weight = exp(score - max_score);
        sum_exp += attn_weight;

        // Accumulate weighted V[j][d]
        output_acc += attn_weight * V[V_offset + j * head_dim + d];
    }

    // Normalize
    output[bh * seq_len * head_dim + i * head_dim + d] = output_acc / sum_exp;
}

// Optimized fused attention using tiled approach for larger sequences
// One threadgroup handles one query position across all d dimensions
kernel void fused_attention_tiled(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    threadgroup float* shared_scores [[threadgroup(0)]],  // [seq_len] for attention scores
    threadgroup float* shared_v [[threadgroup(1)]],       // [head_dim] for partial V accumulation
    uint bh_i [[threadgroup_position_in_grid]],  // batch_head * seq_len + query_i
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if (bh_i >= batch_heads * seq_len) return;

    uint bh = bh_i / seq_len;
    uint i = bh_i % seq_len;

    uint Q_offset = bh * seq_len * head_dim;
    uint K_offset = bh * seq_len * head_dim;
    uint V_offset = bh * seq_len * head_dim;

    // Step 1: Compute attention scores (parallelized over key positions)
    float local_max = -INFINITY;
    for (uint j = tid; j < seq_len; j += tg_size) {
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;
        shared_scores[j] = score;
        local_max = max(local_max, score);
    }

    // Reduce to find global max
    shared_v[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < tg_size) {
            shared_v[tid] = max(shared_v[tid], shared_v[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_score = shared_v[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: Compute softmax (exp and sum)
    float local_sum = 0.0f;
    for (uint j = tid; j < seq_len; j += tg_size) {
        float attn = exp(shared_scores[j] - max_score);
        shared_scores[j] = attn;  // Store attention weights
        local_sum += attn;
    }

    // Reduce sum
    shared_v[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < tg_size) {
            shared_v[tid] += shared_v[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared_v[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Normalize attention weights
    for (uint j = tid; j < seq_len; j += tg_size) {
        shared_scores[j] /= sum_exp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: Compute output = attention @ V (parallelized over head_dim)
    for (uint d = tid; d < head_dim; d += tg_size) {
        float out_val = 0.0f;
        for (uint j = 0; j < seq_len; j++) {
            out_val += shared_scores[j] * V[V_offset + j * head_dim + d];
        }
        output[bh * seq_len * head_dim + i * head_dim + d] = out_val;
    }
}

// Flash attention style - memory efficient, single pass with online softmax
kernel void flash_attention(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_heads [[buffer(4)]],
    constant uint& seq_len [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]  // (d, bh * seq_len)
) {
    uint bh_seq = gid.y;
    uint d = gid.x;

    if (d >= head_dim || bh_seq >= batch_heads * seq_len) return;

    uint bh = bh_seq / seq_len;
    uint i = bh_seq % seq_len;

    uint Q_offset = bh * seq_len * head_dim;
    uint K_offset = bh * seq_len * head_dim;
    uint V_offset = bh * seq_len * head_dim;

    // Online softmax with running statistics
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float output_acc = 0.0f;

    for (uint j = 0; j < seq_len; j++) {
        // Compute score
        float score = 0.0f;
        for (uint k = 0; k < head_dim; k++) {
            score += Q[Q_offset + i * head_dim + k] * K[K_offset + j * head_dim + k];
        }
        score *= scale;

        // Online softmax update
        float new_max = max(max_score, score);
        float correction = exp(max_score - new_max);

        // Update running sum and output
        sum_exp = sum_exp * correction + exp(score - new_max);
        output_acc = output_acc * correction + exp(score - new_max) * V[V_offset + j * head_dim + d];

        max_score = new_max;
    }

    output[bh * seq_len * head_dim + i * head_dim + d] = output_acc / sum_exp;
}

// ============================================================================
// Fused LayerNorm + Linear
// ============================================================================

// Fused: output = Linear(LayerNorm(input))
// Saves one global memory round-trip
kernel void fused_layernorm_linear(
    device const float* input [[buffer(0)]],
    device const float* ln_gamma [[buffer(1)]],
    device const float* ln_beta [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& use_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]  // (out_features, batch)
) {
    uint row = gid.y;  // batch index
    uint col = gid.x;  // output feature index

    if (row >= batch || col >= out_features) return;

    uint in_offset = row * in_features;

    // Step 1: Compute LayerNorm statistics for this row
    float mean = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        mean += input[in_offset + f];
    }
    mean /= float(in_features);

    float var = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        float diff = input[in_offset + f] - mean;
        var += diff * diff;
    }
    var /= float(in_features);
    float inv_std = rsqrt(var + eps);

    // Step 2: Compute Linear with on-the-fly normalized input
    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float normalized = (input[in_offset + i] - mean) * inv_std;
        float ln_out = normalized * ln_gamma[i] + ln_beta[i];
        sum += ln_out * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    output[row * out_features + col] = sum;
}

// Fused LayerNorm + Linear + GELU
kernel void fused_layernorm_linear_gelu(
    device const float* input [[buffer(0)]],
    device const float* ln_gamma [[buffer(1)]],
    device const float* ln_beta [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant uint& use_bias [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;

    if (row >= batch || col >= out_features) return;

    uint in_offset = row * in_features;

    // LayerNorm
    float mean = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        mean += input[in_offset + f];
    }
    mean /= float(in_features);

    float var = 0.0f;
    for (uint f = 0; f < in_features; f++) {
        float diff = input[in_offset + f] - mean;
        var += diff * diff;
    }
    var /= float(in_features);
    float inv_std = rsqrt(var + eps);

    // Linear
    float sum = 0.0f;
    for (uint i = 0; i < in_features; i++) {
        float normalized = (input[in_offset + i] - mean) * inv_std;
        float ln_out = normalized * ln_gamma[i] + ln_beta[i];
        sum += ln_out * weight[col * in_features + i];
    }

    if (use_bias != 0) {
        sum += bias[col];
    }

    // GELU
    float x = sum;
    float x3 = x * x * x;
    float inner = GELU_COEFF * (x + GELU_COEFF2 * x3);
    output[row * out_features + col] = 0.5f * x * (1.0f + tanh(inner));
}

// Elementwise add: out = a + b
kernel void add_tensors(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    out[id] = a[id] + b[id];
}

// ============================================================================
// Reshape utilities for attention
// ============================================================================

// Reshape [batch, seq, embed] -> [batch * heads, seq, head_dim]
kernel void reshape_for_heads(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]  // (head_dim, seq_len, batch * num_heads)
) {
    uint d = gid.x;
    uint s = gid.y;
    uint bh = gid.z;

    uint embed_dim = num_heads * head_dim;

    if (d >= head_dim || s >= seq_len || bh >= batch * num_heads) return;

    uint b = bh / num_heads;
    uint h = bh % num_heads;

    // Input index: [b, s, h * head_dim + d]
    uint in_idx = b * seq_len * embed_dim + s * embed_dim + h * head_dim + d;
    // Output index: [b * heads + h, s, d]
    uint out_idx = bh * seq_len * head_dim + s * head_dim + d;

    output[out_idx] = input[in_idx];
}

// Reshape [batch * heads, seq, head_dim] -> [batch, seq, embed]
kernel void reshape_from_heads(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]  // (embed_dim, seq_len, batch)
) {
    uint e = gid.x;  // embed_dim index
    uint s = gid.y;
    uint b = gid.z;

    uint embed_dim = num_heads * head_dim;

    if (e >= embed_dim || s >= seq_len || b >= batch) return;

    uint h = e / head_dim;
    uint d = e % head_dim;

    // Input index: [b * heads + h, s, d]
    uint in_idx = (b * num_heads + h) * seq_len * head_dim + s * head_dim + d;
    // Output index: [b, s, e]
    uint out_idx = b * seq_len * embed_dim + s * embed_dim + e;

    output[out_idx] = input[in_idx];
}

// Convert NCHW -> NHWC
kernel void nchw_to_nhwc(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& height [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]  // (width, height, batch * channels)
) {
    uint x = gid.x;
    uint y = gid.y;
    uint bc = gid.z;

    if (x >= width || y >= height || bc >= batch * channels) return;

    uint b = bc / channels;
    uint c = bc % channels;

    uint in_idx = b * channels * height * width + c * height * width + y * width + x;
    uint out_idx = b * height * width * channels + y * width * channels + x * channels + c;

    output[out_idx] = input[in_idx];
}

// Apply RoPE to [batch, seq, embed]
kernel void rope_apply(
    device const float* input [[buffer(0)]],
    device const float* freqs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& embed_dim [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& freqs_len [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]  // (embed_dim, seq_len, batch)
) {
    uint e = gid.x;
    uint s = gid.y;
    uint b = gid.z;

    if (e >= embed_dim || s >= seq_len || b >= batch) return;

    uint h = e / head_dim;
    uint d_in_head = e - h * head_dim;
    uint half_head = head_dim / 2;

    uint y = s / width;
    uint x = s - y * width;

    uint base = b * seq_len * embed_dim + s * embed_dim;
    float x_val = input[base + e];

    if (d_in_head < half_head) {
        uint freq_idx = d_in_head;
        if (freq_idx >= freqs_len) {
            output[base + e] = x_val;
            return;
        }
        float theta = float(y) * freqs[freq_idx];
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        uint pair_e = e + half_head;
        float x_pair = input[base + pair_e];
        output[base + e] = x_val * cos_t - x_pair * sin_t;
    } else {
        uint freq_idx = d_in_head - half_head;
        if (freq_idx >= freqs_len) {
            output[base + e] = x_val;
            return;
        }
        float theta = float(x) * freqs[freq_idx];
        float cos_t = cos(theta);
        float sin_t = sin(theta);
        uint pair_e = e - half_head;
        float x_pair = input[base + pair_e];
        output[base + e] = x_pair * sin_t + x_val * cos_t;
    }
}

// ============================================================================
// Conv2D Forward (NHWC format, optimized for small kernels)
// ============================================================================

// Conv2D forward: output = conv2d(input, weight) + bias
// input: [batch, H, W, in_channels] (NHWC format)
// weight: [out_channels, in_channels, kH, kW] (OIHW format)
// bias: [out_channels] or nullptr
// output: [batch, H_out, W_out, out_channels]
kernel void conv2d_forward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_h [[buffer(5)]],
    constant uint& in_w [[buffer(6)]],
    constant uint& in_channels [[buffer(7)]],
    constant uint& out_channels [[buffer(8)]],
    constant uint& kernel_h [[buffer(9)]],
    constant uint& kernel_w [[buffer(10)]],
    constant uint& stride [[buffer(11)]],
    constant uint& padding [[buffer(12)]],
    constant uint& use_bias [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]  // (out_w, out_h, batch * out_channels)
) {
    uint ox = gid.x;  // output x
    uint oy = gid.y;  // output y
    uint boc = gid.z; // batch * out_channels + out_channel

    uint out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    uint out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    if (ox >= out_w || oy >= out_h || boc >= batch * out_channels) return;

    uint b = boc / out_channels;
    uint oc = boc % out_channels;

    float sum = 0.0f;

    // Convolution kernel loop
    for (uint kh = 0; kh < kernel_h; kh++) {
        for (uint kw = 0; kw < kernel_w; kw++) {
            int ih = (int)(oy * stride + kh) - (int)padding;
            int iw = (int)(ox * stride + kw) - (int)padding;

            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                for (uint ic = 0; ic < in_channels; ic++) {
                    // Input: [batch, H, W, in_channels] - NHWC
                    uint in_idx = b * in_h * in_w * in_channels +
                                  ih * in_w * in_channels +
                                  iw * in_channels + ic;

                    // Weight: [out_channels, in_channels, kH, kW] - OIHW
                    uint w_idx = oc * in_channels * kernel_h * kernel_w +
                                 ic * kernel_h * kernel_w +
                                 kh * kernel_w + kw;

                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias
    if (use_bias != 0) {
        sum += bias[oc];
    }

    // Output: [batch, H_out, W_out, out_channels] - NHWC
    uint out_idx = b * out_h * out_w * out_channels +
                   oy * out_w * out_channels +
                   ox * out_channels + oc;

    output[out_idx] = sum;
}

// Conv2D forward with ReLU fused
kernel void conv2d_forward_relu(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& in_h [[buffer(5)]],
    constant uint& in_w [[buffer(6)]],
    constant uint& in_channels [[buffer(7)]],
    constant uint& out_channels [[buffer(8)]],
    constant uint& kernel_h [[buffer(9)]],
    constant uint& kernel_w [[buffer(10)]],
    constant uint& stride [[buffer(11)]],
    constant uint& padding [[buffer(12)]],
    constant uint& use_bias [[buffer(13)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint boc = gid.z;

    uint out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
    uint out_w = (in_w + 2 * padding - kernel_w) / stride + 1;

    if (ox >= out_w || oy >= out_h || boc >= batch * out_channels) return;

    uint b = boc / out_channels;
    uint oc = boc % out_channels;

    float sum = 0.0f;

    for (uint kh = 0; kh < kernel_h; kh++) {
        for (uint kw = 0; kw < kernel_w; kw++) {
            int ih = (int)(oy * stride + kh) - (int)padding;
            int iw = (int)(ox * stride + kw) - (int)padding;

            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                for (uint ic = 0; ic < in_channels; ic++) {
                    uint in_idx = b * in_h * in_w * in_channels +
                                  ih * in_w * in_channels +
                                  iw * in_channels + ic;
                    uint w_idx = oc * in_channels * kernel_h * kernel_w +
                                 ic * kernel_h * kernel_w +
                                 kh * kernel_w + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    if (use_bias != 0) {
        sum += bias[oc];
    }

    // ReLU fused
    sum = max(0.0f, sum);

    uint out_idx = b * out_h * out_w * out_channels +
                   oy * out_w * out_channels +
                   ox * out_channels + oc;

    output[out_idx] = sum;
}

// ============================================================================
// Upsampling Operations (for DPT)
// ============================================================================

// Nearest neighbor 2x upsampling
// input: [batch, in_h, in_w, channels] NHWC
// output: [batch, in_h*2, in_w*2, channels]
kernel void upsample_nearest_2x(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& in_h [[buffer(3)]],
    constant uint& in_w [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ox = gid.x;  // output x
    uint oy = gid.y;  // output y
    uint bc = gid.z;  // batch * channels combined

    uint out_h = in_h * 2;
    uint out_w = in_w * 2;

    if (ox >= out_w || oy >= out_h || bc >= batch * channels) return;

    uint b = bc / channels;
    uint c = bc % channels;

    // Map output to input (nearest neighbor)
    uint ix = ox / 2;
    uint iy = oy / 2;

    uint in_idx = b * in_h * in_w * channels + iy * in_w * channels + ix * channels + c;
    uint out_idx = b * out_h * out_w * channels + oy * out_w * channels + ox * channels + c;

    output[out_idx] = input[in_idx];
}

// Bilinear 2x upsampling (higher quality)
// input: [batch, in_h, in_w, channels] NHWC
// output: [batch, in_h*2, in_w*2, channels]
kernel void upsample_bilinear_2x(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& in_h [[buffer(3)]],
    constant uint& in_w [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint bc = gid.z;

    uint out_h = in_h * 2;
    uint out_w = in_w * 2;

    if (ox >= out_w || oy >= out_h || bc >= batch * channels) return;

    uint b = bc / channels;
    uint c = bc % channels;

    // Map output coordinate to input space
    float ix_f = (float(ox) + 0.5f) / 2.0f - 0.5f;
    float iy_f = (float(oy) + 0.5f) / 2.0f - 0.5f;

    int ix0 = max(0, int(floor(ix_f)));
    int iy0 = max(0, int(floor(iy_f)));
    int ix1 = min(int(in_w) - 1, ix0 + 1);
    int iy1 = min(int(in_h) - 1, iy0 + 1);

    float fx = ix_f - float(ix0);
    float fy = iy_f - float(iy0);

    // Sample 4 corners
    uint idx00 = b * in_h * in_w * channels + iy0 * in_w * channels + ix0 * channels + c;
    uint idx01 = b * in_h * in_w * channels + iy0 * in_w * channels + ix1 * channels + c;
    uint idx10 = b * in_h * in_w * channels + iy1 * in_w * channels + ix0 * channels + c;
    uint idx11 = b * in_h * in_w * channels + iy1 * in_w * channels + ix1 * channels + c;

    float v00 = input[idx00];
    float v01 = input[idx01];
    float v10 = input[idx10];
    float v11 = input[idx11];

    // Bilinear interpolation
    float val = (1 - fx) * (1 - fy) * v00 +
                fx * (1 - fy) * v01 +
                (1 - fx) * fy * v10 +
                fx * fy * v11;

    uint out_idx = b * out_h * out_w * channels + oy * out_w * channels + ox * channels + c;
    output[out_idx] = val;
}

// ============================================================================
// Image Preprocessing (fused resize + normalize)
// ============================================================================

// Bilinear resize + ImageNet normalization in one pass
// input: [in_h, in_w, 3] in [0, 1]
// output: [3, out_h, out_w] normalized to ImageNet mean/std, NCHW
kernel void resize_normalize_nchw(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& in_h [[buffer(2)]],
    constant uint& in_w [[buffer(3)]],
    constant uint& out_h [[buffer(4)]],
    constant uint& out_w [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint c = gid.z;

    if (ox >= out_w || oy >= out_h || c >= 3) return;

    // ImageNet normalization constants
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_inv[3] = {4.3668f, 4.4643f, 4.4444f};  // 1/0.229, 1/0.224, 1/0.225

    // Map output to input coordinates
    float scale_x = float(in_w) / float(out_w);
    float scale_y = float(in_h) / float(out_h);

    float ix_f = (float(ox) + 0.5f) * scale_x - 0.5f;
    float iy_f = (float(oy) + 0.5f) * scale_y - 0.5f;

    int ix0 = max(0, int(floor(ix_f)));
    int iy0 = max(0, int(floor(iy_f)));
    int ix1 = min(int(in_w) - 1, ix0 + 1);
    int iy1 = min(int(in_h) - 1, iy0 + 1);

    float fx = ix_f - float(ix0);
    float fy = iy_f - float(iy0);

    // Sample 4 corners (input is HWC)
    float v00 = input[iy0 * in_w * 3 + ix0 * 3 + c];
    float v01 = input[iy0 * in_w * 3 + ix1 * 3 + c];
    float v10 = input[iy1 * in_w * 3 + ix0 * 3 + c];
    float v11 = input[iy1 * in_w * 3 + ix1 * 3 + c];

    // Bilinear interpolation
    float val = (1 - fx) * (1 - fy) * v00 +
                fx * (1 - fy) * v01 +
                (1 - fx) * fy * v10 +
                fx * fy * v11;

    // Normalize
    val = (val - mean[c]) * std_inv[c];

    // Output is CHW format
    uint out_idx = c * out_h * out_w + oy * out_w + ox;
    output[out_idx] = val;
}

// ============================================================================
// ReLU (standalone for when not fused)
// ============================================================================

kernel void relu_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = max(0.0f, input[gid]);
}
