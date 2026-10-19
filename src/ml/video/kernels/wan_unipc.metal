// Wan default Flow-UniPC, F32 arithmetic and caller-owned GPU history.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void wan_unipc_guidance(device const float *c [[buffer(0)]],
    device const float *u [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]], constant float &scale [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = u[i] + scale * (c[i] - u[i]);
}

kernel void wan_unipc_convert(device const float *x [[buffer(0)]],
    device const float *model [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &n [[buffer(3)]], constant float &sigma [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = x[i] - sigma * model[i];
}

kernel void wan_unipc_correct(device const float *last [[buffer(0)]],
    device const float *m0 [[buffer(1)]], device const float *m1 [[buffer(2)]],
    device const float *current [[buffer(3)]], device float *out [[buffer(4)]],
    constant uint &n [[buffer(5)]], constant float &ratio [[buffer(6)]],
    constant float &ab [[buffer(7)]], constant float &rk [[buffer(8)]],
    constant float &rho0 [[buffer(9)]], constant float &rho1 [[buffer(10)]],
    constant uint &order [[buffer(11)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float base = ratio * last[i] - ab * m0[i];
    const float residual = order == 2 ? rho0 * ((m1[i] - m0[i]) / rk) : 0.0f;
    out[i] = base - ab * (residual + rho1 * (current[i] - m0[i]));
}

kernel void wan_unipc_predict(device const float *x [[buffer(0)]],
    device const float *m0 [[buffer(1)]], device const float *m1 [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant float &ratio [[buffer(5)]], constant float &ab [[buffer(6)]],
    constant float &rk [[buffer(7)]], constant uint &order [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    const float base = ratio * x[i] - ab * m0[i];
    const float residual = order == 2 ? 0.5f * ((m1[i] - m0[i]) / rk) : 0.0f;
    out[i] = base - ab * residual;
}
