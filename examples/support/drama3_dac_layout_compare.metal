#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Exact diagnostic only: model arithmetic remains in the codec kernels.
kernel void dr3_dac_compare_stage(
    device const float* actual [[buffer(0)]],
    device const float* reference [[buffer(1)]],
    device atomic_uint* receipt [[buffer(2)]],
    constant ulong* params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (ulong(index) >= params[0]) return;

    uint base = uint(params[1]) * 2;
    float a = actual[index];
    float r = reference[index];
    if (as_type<uint>(a) != as_type<uint>(r)) {
        atomic_fetch_add_explicit(&receipt[base], 1u, memory_order_relaxed);
    }
    if (!isfinite(a) || !isfinite(r)) {
        atomic_fetch_add_explicit(&receipt[base + 1], 1u, memory_order_relaxed);
    }
}
