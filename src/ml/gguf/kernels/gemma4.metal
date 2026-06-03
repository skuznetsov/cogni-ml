#include <metal_stdlib>
using namespace metal;

kernel void gemma4_rmsnorm_heads_weighted(
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
        const float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv * weight[d];
    }
}

kernel void gemma4_rmsnorm_heads_plain(
    device       float* x        [[buffer(0)]],
    constant     uint&  head_dim [[buffer(1)]],
    constant     float& eps      [[buffer(2)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    device float* head = x + tgpig * head_dim;

    float ss = 0.0f;
    for (uint d = tiisg; d < head_dim; d += 32) {
        const float v = head[d];
        ss += v * v;
    }
    const float sum = simd_sum(ss);
    const float inv = rsqrt(sum / float(head_dim) + eps);

    for (uint d = tiisg; d < head_dim; d += 32) {
        head[d] = head[d] * inv;
    }
}

kernel void gemma4_rope_neox(
    device       float* x            [[buffer(0)]],
    device const float* freq_factors [[buffer(1)]],
    constant     uint&  head_dim     [[buffer(2)]],
    constant     uint&  rope_dim     [[buffer(3)]],
    constant     uint&  pos          [[buffer(4)]],
    constant     float& freq_base    [[buffer(5)]],
    constant     uint&  use_factors  [[buffer(6)]],
    uint   tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint half_dim = rope_dim / 2;
    device float* head = x + tgpig * head_dim;

    for (uint i = tiisg; i < half_dim; i += 32) {
        const float factor = use_factors == 0 ? 1.0f : freq_factors[i];
        const float freq = pow(freq_base, -2.0f * float(i) / float(rope_dim)) / factor;
        const float theta = float(pos) * freq;
        const float c = cos(theta);
        const float s = sin(theta);
        const float x0 = head[i];
        const float x1 = head[i + half_dim];
        head[i] = x0 * c - x1 * s;
        head[i + half_dim] = x0 * s + x1 * c;
    }
}
