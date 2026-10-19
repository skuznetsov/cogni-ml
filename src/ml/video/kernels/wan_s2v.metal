#include <metal_stdlib>
using namespace metal;

// Copyright 2024-2025 The Alibaba Wan Team Authors.
// Apache-2.0. Adapted for cogni-ml from Wan2.2 S2V at
// 1ea34ff48f87168174e12956e200b1d908b1c5ff.
// Wan2.2 S2V's input modulation has six fields and two video segments:
// [B,6,2,C]. The learned table is [6,C] and broadcasts over both axes.
kernel void wan_modulate_segments(
    device const float *x [[buffer(0)]], device const float *e [[buffer(1)]],
    device const float *base [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &seq [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &split [[buffer(7)]],
    constant uint &shift_field [[buffer(8)]], constant uint &scale_field [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= batches * seq * dim) return;
    const uint channel = id % dim;
    const uint token = (id / dim) % seq;
    const uint batch = id / (seq * dim);
    const uint segment = token < split ? 0 : 1;
    const uint shift_at = ((batch * 6 + shift_field) * 2 + segment) * dim + channel;
    const uint scale_at = ((batch * 6 + scale_field) * 2 + segment) * dim + channel;
    const float shift = e[shift_at] + base[shift_field * dim + channel];
    const float scale = e[scale_at] + base[scale_field * dim + channel];
    y[id] = x[id] * (1.0f + scale) + shift;
}

kernel void wan_segment_add(
    device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
    device const float *e [[buffer(2)]], device const float *base [[buffer(3)]],
    device float *y [[buffer(4)]], constant uint &batches [[buffer(5)]],
    constant uint &seq [[buffer(6)]], constant uint &dim [[buffer(7)]],
    constant uint &split [[buffer(8)]], constant uint &gate_field [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= batches * seq * dim) return;
    const uint channel = id % dim;
    const uint token = (id / dim) % seq;
    const uint batch = id / (seq * dim);
    const uint segment = token < split ? 0 : 1;
    const uint gate_at = ((batch * 6 + gate_field) * 2 + segment) * dim + channel;
    const float gate = e[gate_at] + base[gate_field * dim + channel];
    y[id] = x[id] + gate * delta[id];
}

// Direct-device three-axis RoPE. Positions are integer-valued F32 triples,
// including negative reference offsets. Axis dimensions are complex-pair
// counts exactly matching Wan's split of head_dim/2 into temporal/height/width.
kernel void wan_rope_f32(
    device const float *x [[buffer(0)]], device const float *positions [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &batches [[buffer(3)]],
    constant uint &seq [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]], constant float &theta [[buffer(7)]],
    constant uint &axis0 [[buffer(8)]], constant uint &axis1 [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    const uint total = batches * seq * heads * pairs;
    if (id >= total) return;
    const uint pair = id % pairs;
    const uint token_head = id / pairs;
    const uint row = token_head / heads;
    const uint token = row % seq;
    const uint batch = row / seq;
    uint axis, local_pair, axis_dim;
    if (pair < axis0) {
        axis = 0; local_pair = pair; axis_dim = axis0;
    } else if (pair < axis0 + axis1) {
        axis = 1; local_pair = pair - axis0; axis_dim = axis1;
    } else {
        axis = 2; local_pair = pair - axis0 - axis1; axis_dim = pairs - axis0 - axis1;
    }
    const float position = positions[(batch * seq + token) * 3 + axis];
    // `rope_params` takes the real component count 2 * axis_dim, so its
    // exponent `-2*k/real_dim` is `-k/axis_dim` in complex-pair coordinates.
    const float angle = position * pow(theta, -float(local_pair) / float(axis_dim));
    // The safe-mode GPU probe still found ~5e-5 sine error near 801 radians
    // with the ordinary intrinsics. Precise variants match the upstream FP64
    // RoPE table closely while preserving GPU-only runtime math.
    const float cosine = precise::cos(angle), sine = precise::sin(angle);
    const uint offset = token_head * head_dim + pair * 2;
    const float real = x[offset], imag = x[offset + 1];
    y[offset] = real * cosine - imag * sine;
    y[offset + 1] = real * sine + imag * cosine;
}

kernel void wan_audio_gather(
    device const float *x [[buffer(0)]], device float *gathered [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &seq [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &spatial [[buffer(5)]],
    constant uint &dim [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    const uint prefix = frames * spatial;
    const uint count = batches * prefix * dim;
    if (id >= count) return;
    const uint row = id / dim;
    const uint batch = row / prefix;
    const uint token = row % prefix;
    gathered[id] = x[(batch * seq + token) * dim + id % dim];
}

kernel void wan_audio_silu(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float value = x[id];
    y[id] = value / (1.0f + exp(-value));
}

// Diffusers AdaLayerNorm(chunk_dim=1): [shift, scale] after its linear layer.
kernel void wan_audio_adain(
    device const float *normalized [[buffer(0)]], device const float *projection [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &batches [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &spatial [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    const uint rows = batches * frames * spatial;
    const uint count = rows * dim;
    if (id >= count) return;
    const uint row = id / dim;
    const uint frame_row = row / spatial;
    const uint channel = id % dim;
    const float shift = projection[frame_row * (2 * dim) + channel];
    const float scale = projection[frame_row * (2 * dim) + dim + channel];
    y[id] = normalized[id] * (1.0f + scale) + shift;
}

// Prefix residual plus exact suffix copy. This preserves all reference/motion
// suffix values while adding audio attention only to the original video tokens.
kernel void wan_audio_scatter(
    device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &batches [[buffer(3)]],
    constant uint &seq [[buffer(4)]], constant uint &prefix [[buffer(5)]],
    constant uint &dim [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    const uint count = batches * seq * dim;
    if (id >= count) return;
    const uint token = (id / dim) % seq;
    const uint batch = id / (seq * dim);
    y[id] = token < prefix ? x[id] + delta[(batch * prefix + token) * dim + id % dim] : x[id];
}
