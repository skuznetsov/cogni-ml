// Adapted from Lightricks/LTX-2 BasicAVTransformerBlock RoPE handling at
// e98f6d927d83c92ef2be6aec76c8440b19306e39. Re-expressed for native Metal;
// covered by the LTX-2.x Community License (LICENSE-2_x).
#include <metal_stdlib>
using namespace metal;

// LTX split RoPE pairs the first and second halves of each head dimension.
// Input/output are row-major [tokens, heads, head_dim]; cos/sin are
// row-major [tokens, heads, head_dim / 2].
kernel void ltx2_split_rope_f32(
    device const float *x [[buffer(0)]], device const float *cos_freq [[buffer(1)]],
    device const float *sin_freq [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    const uint count = tokens * heads * head_dim;
    if (id >= count) return;

    const uint row_width = heads * head_dim;
    const uint token = id / row_width;
    const uint within_token = id % row_width;
    const uint head = within_token / head_dim;
    const uint component = within_token % head_dim;
    const uint half_dim = head_dim / 2;
    const uint pair = component < half_dim ? component : component - half_dim;
    const uint base = token * row_width + head * head_dim;
    const uint freq = (token * heads + head) * half_dim + pair;
    const float first = x[base + pair];
    const float second = x[base + pair + half_dim];
    const float c = cos_freq[freq];
    const float s = sin_freq[freq];
    y[id] = component < half_dim ? first * c - second * s : second * c + first * s;
}
