#include <metal_stdlib>
using namespace metal;

// The shared helper pins IEEE round-to-nearest-ties-to-even and quiet-NaN
// behavior to the reference BF16 boundary used by preattention.
kernel void hy_nomas_reference_kv_cache_store_bf16(
    device const float *source [[buffer(0)]], device ushort *cache [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float rounded = hy_nomas_bf16_roundtrip_f32(source[id]);
    cache[id] = ushort(as_type<uint>(rounded) >> 16u);
}

kernel void hy_nomas_reference_kv_cache_load_f32(
    device const ushort *cache [[buffer(0)]], device float *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    destination[id] = as_type<float>(uint(cache[id]) << 16u);
}
