#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Diagnostic-only compact receipts. Production arithmetic stays in Resident::SOURCE.
kernel void dr3_dense_four_receipt_clear(
    device atomic_uint* receipt [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index < 2) atomic_store_explicit(&receipt[index], 0u, memory_order_relaxed);
}

kernel void dr3_dense_four_compare(
    device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
    device atomic_uint* receipt [[buffer(2)]], constant ulong* p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p[0]) return;
    float ref = reference[index], got = actual[index];
    if (as_type<uint>(ref) != as_type<uint>(got)) atomic_fetch_add_explicit(&receipt[0], 1u, memory_order_relaxed);
    if (!isfinite(ref) || !isfinite(got)) atomic_fetch_add_explicit(&receipt[1], 1u, memory_order_relaxed);
}
