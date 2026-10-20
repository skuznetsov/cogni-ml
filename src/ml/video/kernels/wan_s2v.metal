#include <metal_stdlib>
using namespace metal;

// Copyright 2024-2025 The Alibaba Wan Team Authors.
// Apache-2.0. Adapted for cogni-ml from Wan2.2 S2V at
// 1ea34ff48f87168174e12956e200b1d908b1c5ff.
// Wan2.2 S2V's input modulation has six fields and two video segments:
// [B,6,2,C]. The learned table is [6,C] and broadcasts over both axes.
inline float wan_modulation_value(device const float *base_f32,
                                  device const ushort *base_bf16,
                                  uint index, uint is_bf16) {
    if (is_bf16 != 0) {
        return as_type<float>(uint(base_bf16[index]) << 16);
    }
    return base_f32[index];
}

kernel void wan_modulate_segments(
    device const float *x [[buffer(0)]], device const float *e [[buffer(1)]],
    device const float *base_f32 [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &seq [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &split [[buffer(7)]],
    constant uint &shift_field [[buffer(8)]], constant uint &scale_field [[buffer(9)]],
    device const ushort *base_bf16 [[buffer(10)]], constant uint &base_is_bf16 [[buffer(11)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= batches * seq * dim) return;
    const uint channel = id % dim;
    const uint token = (id / dim) % seq;
    const uint batch = id / (seq * dim);
    const uint segment = token < split ? 0 : 1;
    const uint shift_at = ((batch * 6 + shift_field) * 2 + segment) * dim + channel;
    const uint scale_at = ((batch * 6 + scale_field) * 2 + segment) * dim + channel;
    const float shift = e[shift_at] + wan_modulation_value(base_f32, base_bf16, shift_field * dim + channel, base_is_bf16);
    const float scale = e[scale_at] + wan_modulation_value(base_f32, base_bf16, scale_field * dim + channel, base_is_bf16);
    y[id] = x[id] * (1.0f + scale) + shift;
}

kernel void wan_segment_add(
    device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
    device const float *e [[buffer(2)]], device const float *base_f32 [[buffer(3)]],
    device float *y [[buffer(4)]], constant uint &batches [[buffer(5)]],
    constant uint &seq [[buffer(6)]], constant uint &dim [[buffer(7)]],
    constant uint &split [[buffer(8)]], constant uint &gate_field [[buffer(9)]],
    device const ushort *base_bf16 [[buffer(10)]], constant uint &base_is_bf16 [[buffer(11)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= batches * seq * dim) return;
    const uint channel = id % dim;
    const uint token = (id / dim) % seq;
    const uint batch = id / (seq * dim);
    const uint segment = token < split ? 0 : 1;
    const uint gate_at = ((batch * 6 + gate_field) * 2 + segment) * dim + channel;
    const float gate = e[gate_at] + wan_modulation_value(base_f32, base_bf16, gate_field * dim + channel, base_is_bf16);
    y[id] = x[id] + gate * delta[id];
}

// Direct-device three-axis RoPE. Positions are integer-valued F32 triples,
// including negative reference offsets. Axis dimensions are complex-pair
// counts exactly matching Wan's split of head_dim/2 into temporal/height/width.
kernel void wan_rope_f32(
    device const float *x [[buffer(0)]], device const float *positions [[buffer(1)]],
    device float *y [[buffer(2)]], device const float *frequency_hi [[buffer(3)]],
    device const float *frequency_lo [[buffer(4)]], constant uint &batches [[buffer(5)]],
    constant uint &seq [[buffer(6)]], constant uint &heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    uint id [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    const uint total = batches * seq * heads * pairs;
    if (id >= total) return;
    const uint pair = id % pairs;
    const uint token_head = id / pairs;
    const uint row = token_head / heads;
    const uint token = row % seq;
    const uint batch = row / seq;
    const uint axis0 = (pairs / 3) + (pairs % 3);
    const uint axis1 = pairs / 3;
    const uint axis = pair < axis0 ? 0 : (pair < axis0 + axis1 ? 1 : 2);
    const float position = positions[(batch * seq + token) * 3 + axis];
    // The upstream source creates its inverse frequencies and complex phase
    // table in FP64. Split the setup-only constants into hi/lo F32, then retain
    // the product residual with FMA. Two-component phase composition avoids
    // discarding that residual before the precise trig functions, including
    // for positions outside the small fixture coordinates.
    const float frequency = frequency_hi[pair];
    const float phase_hi = position * frequency;
    const float product_residual = fma(position, frequency, -phase_hi);
    const float phase_tail = fma(position, frequency_lo[pair], product_residual);
    const float phase_sum = phase_hi + phase_tail;
    const float phase_virtual = phase_sum - phase_hi;
    const float phase_residual = (phase_hi - (phase_sum - phase_virtual)) +
                                 (phase_tail - phase_virtual);
    const float cos_hi = precise::cos(phase_sum);
    const float sin_hi = precise::sin(phase_sum);
    const float cos_lo = precise::cos(phase_residual);
    const float sin_lo = precise::sin(phase_residual);
    const float cosine = cos_hi * cos_lo - sin_hi * sin_lo;
    const float sine = sin_hi * cos_lo + cos_hi * sin_lo;
    const uint offset = token_head * head_dim + pair * 2;
    const float real = x[offset], imag = x[offset + 1];
    y[offset] = real * cosine - imag * sine;
    y[offset + 1] = real * sine + imag * cosine;
}

// Optional final coefficient table for one Wan block encode. The table is
// indexed by [batch, token, complex-pair]; heads share the same three-axis
// position and therefore reuse these coefficients during both rotations.
kernel void wan_rope_table_build_f32(
    device const float *positions [[buffer(0)]],
    device float *cosine_table [[buffer(1)]], device float *sine_table [[buffer(2)]],
    device const float *frequency_hi [[buffer(3)]],
    device const float *frequency_lo [[buffer(4)]],
    constant uint &batches [[buffer(5)]], constant uint &seq [[buffer(6)]],
    constant uint &head_dim [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    const uint total = batches * seq * pairs;
    if (id >= total) return;
    const uint pair = id % pairs;
    const uint row = id / pairs;
    const uint token = row % seq;
    const uint batch = row / seq;
    const uint axis0 = (pairs / 3) + (pairs % 3);
    const uint axis1 = pairs / 3;
    const uint axis = pair < axis0 ? 0 : (pair < axis0 + axis1 ? 1 : 2);
    const float position = positions[(batch * seq + token) * 3 + axis];
    // Keep the exact two-F32 phase construction and precise trig expression
    // used by wan_rope_f32; only the coefficient lifetime is changed.
    const float frequency = frequency_hi[pair];
    const float phase_hi = position * frequency;
    const float product_residual = fma(position, frequency, -phase_hi);
    const float phase_tail = fma(position, frequency_lo[pair], product_residual);
    const float phase_sum = phase_hi + phase_tail;
    const float phase_virtual = phase_sum - phase_hi;
    const float phase_residual = (phase_hi - (phase_sum - phase_virtual)) +
                                 (phase_tail - phase_virtual);
    const float cos_hi = precise::cos(phase_sum);
    const float sin_hi = precise::sin(phase_sum);
    const float cos_lo = precise::cos(phase_residual);
    const float sin_lo = precise::sin(phase_residual);
    cosine_table[id] = cos_hi * cos_lo - sin_hi * sin_lo;
    sine_table[id] = sin_hi * cos_lo + cos_hi * sin_lo;
}

kernel void wan_rope_table_apply_f32(
    device const float *x [[buffer(0)]],
    device const float *cosine_table [[buffer(1)]],
    device const float *sine_table [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &seq [[buffer(5)]],
    constant uint &heads [[buffer(6)]], constant uint &head_dim [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    const uint pairs = head_dim / 2;
    const uint total = batches * seq * heads * pairs;
    if (id >= total) return;
    const uint pair = id % pairs;
    const uint token_head = id / pairs;
    const uint row = token_head / heads;
    const uint table_at = row * pairs + pair;
    const uint offset = token_head * head_dim + pair * 2;
    const float real = x[offset], imag = x[offset + 1];
    const float cosine = cosine_table[table_at];
    const float sine = sine_table[table_at];
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
