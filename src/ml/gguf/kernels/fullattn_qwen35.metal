#include <metal_stdlib>
using namespace metal;

kernel void qwen35_split_qgate(
    device const float* q_full   [[buffer(0)]],
    device       float* q_out    [[buffer(1)]],
    device       float* gate_out [[buffer(2)]],
    constant     uint&  n_head   [[buffer(3)]],
    constant     uint&  head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint q_dim = n_head * head_dim;
    if (gid >= q_dim) return;

    const uint h = gid / head_dim;
    const uint d = gid % head_dim;
    const uint src_base = h * 2 * head_dim;
    q_out[gid] = q_full[src_base + d];
    gate_out[gid] = q_full[src_base + head_dim + d];
}

kernel void qwen35_rmsnorm_heads(
    device       float* x        [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    constant     uint&  head_dim [[buffer(2)]],
    constant     float& eps      [[buffer(3)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    device float* head = x + tgpig * head_dim;

    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv * weight[d];
    }
}

kernel void qwen35_rope_partial(
    device       float* x         [[buffer(0)]],
    constant     uint&  head_dim  [[buffer(1)]],
    constant     uint&  rope_dim  [[buffer(2)]],
    constant     uint&  pos       [[buffer(3)]],
    constant     float& freq_base [[buffer(4)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint half_dim = rope_dim / 2;
    device float* head = x + tgpig * head_dim;

    for (uint i = tiisg; i < half_dim; i += 32) {
        const float freq = 1.0f / pow(freq_base, 2.0f * float(i) / float(rope_dim));
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i];
        const float x1 = head[i + half_dim];
        head[i] = x0 * c - x1 * s;
        head[i + half_dim] = x0 * s + x1 * c;
    }
}

kernel void qwen35_kv_write(
    device const float* k       [[buffer(0)]],
    device const float* v       [[buffer(1)]],
    device       float* k_cache [[buffer(2)]],
    device       float* v_cache [[buffer(3)]],
    constant     uint&  base    [[buffer(4)]],
    constant     uint&  kv_dim  [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= kv_dim) return;
    k_cache[base + gid] = k[gid];
    v_cache[base + gid] = v[gid];
}
