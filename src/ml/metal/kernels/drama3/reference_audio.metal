#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

kernel void dr3_reference_pcm16(
    device const float* pcm [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) output[i] = pcm[i] * (1.0f / 32768.0f);
}
