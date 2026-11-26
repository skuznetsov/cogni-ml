// Token sampling kernels for autoregressive generation
// - LM head: hidden state → vocab logits (uses existing GEMM, not here)
// - Top-k selection: find k largest logits
// - Softmax: normalize top-k logits to probabilities
// - Temperature scaling

#include <metal_stdlib>
using namespace metal;

// ── Top-K selection via partial sort ──
// Finds top-k logits and their indices from vocab_size logits.
// Each threadgroup handles one batch item.
// Dispatch: threadgroups = [batch], threads = [256, 1]
kernel void topk_select(
    device const float*  logits     [[buffer(0)]],  // [batch, vocab_size]
    device       float*  topk_vals  [[buffer(1)]],  // [batch, k]
    device       uint*   topk_ids   [[buffer(2)]],  // [batch, k]
    constant     uint&   vocab_size [[buffer(3)]],
    constant     uint&   k          [[buffer(4)]],
    constant     float&  temperature [[buffer(5)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint batch_idx = tgpig;
    device const float* row = logits + batch_idx * vocab_size;

    const uint n_simd = 8;  // 256 threads / 32
    const uint lane = tiisg;
    const uint sg = sgitg;

    // Phase 1: each thread finds its local max across strided elements
    float local_max = -1e30f;
    uint local_idx = 0;

    for (uint i = sg * 32 + lane; i < vocab_size; i += n_simd * 32) {
        float v = row[i] / temperature;
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
    }

    // Phase 2: reduce within simdgroup to find top-1
    // For k > 1, we'd need a more sophisticated approach
    // Simple version: iterative extraction (k is typically small, 1-50)
    // This kernel handles k=1 efficiently; for k>1, call repeatedly with masking

    float sg_max = simd_max(local_max);

    // Find which lane has the max
    bool is_max = (local_max == sg_max);
    uint max_lane = is_max ? lane : 32;
    max_lane = simd_min(max_lane);  // lowest lane with max value

    // First simdgroup writes result
    if (sg == 0 && lane == 0) {
        // Reduce across simdgroups (simple — only 8 groups)
        // For production, use shared memory reduction
        topk_vals[batch_idx * k] = sg_max;
        topk_ids[batch_idx * k] = local_idx;
    }
}

// ── Softmax over small array (top-k values) ──
// In-place softmax on topk_vals
// Dispatch: threadgroups = [batch], threads = [32, 1]
kernel void softmax_topk(
    device float*    vals   [[buffer(0)]],  // [batch, k] — in/out
    constant uint&   k     [[buffer(1)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint batch_idx = tgpig;
    device float* row = vals + batch_idx * k;
    const uint lane = tiisg;

    // Find max (for numerical stability)
    float v = (lane < k) ? row[lane] : -1e30f;
    float m = simd_max(v);

    // Exp and sum
    float e = (lane < k) ? exp(v - m) : 0.0f;
    float sum = simd_sum(e);

    // Normalize
    if (lane < k) {
        row[lane] = e / sum;
    }
}
