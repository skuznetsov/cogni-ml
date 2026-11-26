// Adapted from Tencent Hy-Nomas at pinned revision
// 911cea6458a89c2663513ff7099561ec190dd491.
// Copyright (C) 2026 Tencent. Licensed under Apache-2.0.
// Modified for native Metal execution.
#include <metal_stdlib>
using namespace metal;

// IEEE round-to-nearest-ties-to-even from F32 to BF16 followed by expansion
// back to F32. NaNs retain sign and upper payload bits and are forced quiet;
// infinities pass through the normal exact conversion path.
inline float hy_nomas_bf16_roundtrip_f32(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        const uint nan_bf16 = (bits >> 16) | 0x40u;
        return as_type<float>(nan_bf16 << 16);
    }
    const uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return as_type<float>(rounded & 0xffff0000u);
}

kernel void hy_nomas_silu_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = x[id] / (1.0f + exp(-x[id]));
}

// Expand the upstream six-way modulation from either one row per batch or one
// row per token into token-major shift/scale/gate buffers.
kernel void hy_nomas_unpack_modulation_f32(
    device const float *modulation [[buffer(0)]],
    device float *attn_shift [[buffer(1)]], device float *attn_scale [[buffer(2)]],
    device float *attn_gate [[buffer(3)]], device float *ff_shift [[buffer(4)]],
    device float *ff_scale [[buffer(5)]], device float *ff_gate [[buffer(6)]],
    constant uint &rows [[buffer(7)]], constant uint &tokens [[buffer(8)]],
    constant uint &dim [[buffer(9)]], constant uint &mod_rows [[buffer(10)]],
    constant uint &batches [[buffer(11)]], uint id [[thread_position_in_grid]]) {
    if (id >= rows * dim) return;
    const uint row = id / dim;
    const uint batch = row / tokens;
    const uint source_row = mod_rows == batches ? batch : row;
    const uint column = id % dim;
    const uint base = source_row * (6 * dim) + column;
    attn_shift[id] = modulation[base];
    attn_scale[id] = modulation[base + dim];
    attn_gate[id] = modulation[base + 2 * dim];
    ff_shift[id] = modulation[base + 3 * dim];
    ff_scale[id] = modulation[base + 4 * dim];
    ff_gate[id] = modulation[base + 5 * dim];
}

// Adjacent-pair rotate_half: (r, i) -> (-i, r), followed by x*cos + rot*sin.
kernel void hy_nomas_rope_f32(
    device const float *x [[buffer(0)]], device const float *cosine [[buffer(1)]],
    device const float *sine [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &tokens [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]], constant uint &batches [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    const uint dim = heads * head_dim;
    const uint count = batches * tokens * dim;
    if (id >= count) return;
    const uint token = (id / dim) % tokens;
    const uint channel = id % head_dim;
    const uint pair = (channel & 1) == 0 ? id + 1 : id - 1;
    const float rotated = (channel & 1) == 0 ? -x[pair] : x[pair];
    const uint table_index = token * head_dim + channel;
    y[id] = x[id] * cosine[table_index] + rotated * sine[table_index];
}

kernel void hy_nomas_concat_f32(
    device const float *left [[buffer(0)]], device const float *right [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &batches [[buffer(3)]],
    constant uint &left_tokens [[buffer(4)]], constant uint &right_tokens [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &bf16_boundary [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    const uint tokens = left_tokens + right_tokens;
    const uint count = batches * tokens * dim;
    if (id >= count) return;
    const uint row = id / dim;
    const uint token = row % tokens;
    const uint batch = row / tokens;
    const uint column = id % dim;
    float value;
    if (token < left_tokens) {
        value = left[(batch * left_tokens + token) * dim + column];
    } else {
        const uint right_token = token - left_tokens;
        value = right[(batch * right_tokens + right_token) * dim + column];
    }
    output[id] = bf16_boundary != 0u ? hy_nomas_bf16_roundtrip_f32(value) : value;
}

kernel void hy_nomas_split_f32(
    device float *input [[buffer(0)]], device float *left [[buffer(1)]],
    device float *right [[buffer(2)]], constant uint &batches [[buffer(3)]],
    constant uint &left_tokens [[buffer(4)]], constant uint &right_tokens [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &bf16_boundary [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    const uint tokens = left_tokens + right_tokens;
    const uint count = batches * tokens * dim;
    if (id >= count) return;
    const uint row = id / dim;
    const uint token = row % tokens;
    const uint batch = row / tokens;
    const uint column = id % dim;
    float value = input[id];
    if (bf16_boundary != 0u) {
        value = hy_nomas_bf16_roundtrip_f32(value);
        // Each invocation owns one unique source element; this in-place write
        // preserves joint.attention as the post-boundary diagnostic checkpoint.
        input[id] = value;
    }
    if (token < left_tokens) {
        left[(batch * left_tokens + token) * dim + column] = value;
    } else {
        const uint right_token = token - left_tokens;
        right[(batch * right_tokens + right_token) * dim + column] = value;
    }
}
