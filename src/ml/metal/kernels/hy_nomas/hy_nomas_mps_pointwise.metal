// Same-input arithmetic discriminators. PyTorch eager pointwise operators
// materialize each multiplication before the following addition.
#include <metal_stdlib>
using namespace metal;

kernel void hy_nomas_modulate_uncontracted_f32(
    device const float *x [[buffer(0)]], device const float *shift [[buffer(1)]],
    device const float *scale [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (id < count) {
        const float factor = 1.0f + scale[id];
        const float product = x[id] * factor;
        output[id] = product + shift[id];
    }
}

kernel void hy_nomas_residual_uncontracted_f32(
    device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
    device const float *gate [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &count [[buffer(4)]], uint id [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    if (id < count) {
        const float product = delta[id] * gate[id];
        output[id] = x[id] + product;
    }
}

// Pinned tuple-frequency RoPE with separately rounded products and sum.
// The flattened input is [B,T,H,Dh]; cosine and sine are [T,Dh] and broadcast
// over B and H. Keep this global entrypoint name unique in concatenated source.
kernel void hy_nomas_rope_uncontracted_f32(
    device const float *x [[buffer(0)]], device const float *cosine [[buffer(1)]],
    device const float *sine [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]], constant uint &batches [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
#pragma clang fp contract(off)
    const uint dim = heads * head_dim;
    const uint count = batches * tokens * dim;
    if (id < count) {
        const uint token = (id / dim) % tokens;
        const uint channel = id % head_dim;
        const uint pair = (channel & 1u) == 0u ? id + 1u : id - 1u;
        const float rotated = (channel & 1u) == 0u ? -x[pair] : x[pair];
        const uint table_index = token * head_dim + channel;
        const float product_x_cos = x[id] * cosine[table_index];
        const float product_rotated_sin = rotated * sine[table_index];
        output[id] = product_x_cos + product_rotated_sin;
    }
}
