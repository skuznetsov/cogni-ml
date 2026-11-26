// Test-only SiLU arithmetic candidates. PyTorch 2.8 MPS spells forward SiLU
// as input / (1 + exp(-input)); this file separates exp/divide choices without
// changing the production kernel or the frozen input operands.
#include <metal_stdlib>
using namespace metal;

kernel void hy_probe_silu_exp_div(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + exp(-x[id]);
        y[id] = x[id] / denominator;
    }
}

kernel void hy_probe_silu_precise_exp_div(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + precise::exp(-x[id]);
        y[id] = x[id] / denominator;
    }
}

kernel void hy_probe_silu_precise_exp_precise_div(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + precise::exp(-x[id]);
        y[id] = precise::divide(x[id], denominator);
    }
}

kernel void hy_probe_silu_exp_recip_mul(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + exp(-x[id]);
        y[id] = x[id] * (1.0f / denominator);
    }
}

kernel void hy_probe_silu_precise_exp_recip_mul(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + precise::exp(-x[id]);
        y[id] = x[id] * (1.0f / denominator);
    }
}

kernel void hy_probe_silu_exp2_div(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + exp2(-x[id] * 1.4426950408889634f);
        y[id] = x[id] / denominator;
    }
}

kernel void hy_probe_silu_precise_exp2_div(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) {
        const float denominator = 1.0f + precise::exp2(-x[id] * 1.4426950408889634f);
        y[id] = x[id] / denominator;
    }
}

inline float hy_probe_bf16_roundtrip_f32(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        const uint nan_bf16 = (bits >> 16) | 0x40u;
        return as_type<float>(nan_bf16 << 16);
    }
    const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(rounded & 0xffff0000u);
}

kernel void hy_probe_silu_roundtrip(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = hy_probe_bf16_roundtrip_f32(x[id]);
}
