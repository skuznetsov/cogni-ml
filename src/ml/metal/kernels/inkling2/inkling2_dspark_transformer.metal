#include <metal_stdlib>
using namespace metal;

constant uint INKLING2_DSPARK_THREADS = 256;
constant uint INKLING2_DSPARK_SIMDGROUPS = 8;
constant uint INKLING2_DSPARK_HIDDEN = 2304;
constant uint INKLING2_DSPARK_HEAD_DIM = 128;
constant uint INKLING2_DSPARK_KV_HEADS = 8;
constant uint INKLING2_DSPARK_Q_HEADS = 24;
constant uint INKLING2_DSPARK_BLOCK = 8;
constant uint INKLING2_DSPARK_RING = 1024;
constant uint INKLING2_DSPARK_CONTEXT = 1023;
constant uint INKLING2_DSPARK_MAX_KEYS = INKLING2_DSPARK_CONTEXT + INKLING2_DSPARK_BLOCK;

inline float inkling2_dspark_bf16(device const ushort* values, uint index)
{
    return as_type<float>((uint)values[index] << 16);
}

inline float inkling2_dspark_round_bf16(float value)
{
    const uint bits = as_type<uint>(value);
    const uint rounding = 0x7fffU + ((bits >> 16) & 1U);
    return as_type<float>((bits + rounding) & 0xffff0000U);
}

inline float inkling2_dspark_rms_scale(
    float square_sum,
    ushort tiisg,
    ushort sgitg,
    threadgroup float* partials,
    uint width,
    float eps)
{
    const float simd_sum_squares = simd_sum(square_sum);
    if (tiisg == 0) partials[sgitg] = simd_sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint group = 0; group < INKLING2_DSPARK_SIMDGROUPS; ++group) {
        total += partials[group];
    }
    return 1.0f / sqrt(total / (float)width + eps);
}

kernel void inkling2_dspark_block_init(
    device const float* anchor      [[buffer(0)]],
    device const ushort* mask       [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint index = tgpig.x * INKLING2_DSPARK_THREADS + tid;
    const uint count = INKLING2_DSPARK_BLOCK * INKLING2_DSPARK_HIDDEN;
    if (index >= count) return;

    const uint row = index / INKLING2_DSPARK_HIDDEN;
    const uint column = index - row * INKLING2_DSPARK_HIDDEN;
    output[index] = row == 0 ? anchor[column] : inkling2_dspark_bf16(mask, column);
}

kernel void inkling2_dspark_rms_norm(
    device const float* input       [[buffer(0)]],
    device const ushort* norm       [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant     uint&  width       [[buffer(3)]],
    constant     uint&  rows        [[buffer(4)]],
    constant     float& eps         [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint row = tgpig.x;
    if (row >= rows) return;
    const size_t offset = (size_t)row * width;
    float square_sum = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_DSPARK_THREADS) {
        const float value = input[offset + column];
        square_sum += value * value;
    }
    threadgroup float partials[INKLING2_DSPARK_SIMDGROUPS];
    const float scale = inkling2_dspark_rms_scale(square_sum, tiisg, sgitg, partials, width, eps);
    for (uint column = tid; column < width; column += INKLING2_DSPARK_THREADS) {
        output[offset + column] = input[offset + column] * scale * inkling2_dspark_bf16(norm, column);
    }
}

// Mix the four committed target taps, apply fuse_weight[layer] softmax, then
// RMS-normalize with the layer's dflare_norm weight. Rows remain independent.
kernel void inkling2_dspark_mix_rms(
    device const float* tap0        [[buffer(0)]],
    device const float* tap1        [[buffer(1)]],
    device const float* tap2        [[buffer(2)]],
    device const float* tap3        [[buffer(3)]],
    device const ushort* fuse       [[buffer(4)]],
    device const ushort* norm       [[buffer(5)]],
    device       float* output      [[buffer(6)]],
    constant     uint&  width       [[buffer(7)]],
    constant     uint&  rows        [[buffer(8)]],
    constant     uint&  layer       [[buffer(9)]],
    constant     float& eps         [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint row = tgpig.x;
    if (row >= rows) return;

    float alpha[4];
    float alpha_max = -INFINITY;
    for (uint tap = 0; tap < 4; ++tap) {
        alpha[tap] = inkling2_dspark_bf16(fuse, layer * 4 + tap);
        alpha_max = max(alpha_max, alpha[tap]);
    }
    float alpha_sum = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
        alpha[tap] = exp(alpha[tap] - alpha_max);
        alpha_sum += alpha[tap];
    }
    for (uint tap = 0; tap < 4; ++tap) {
        alpha[tap] = inkling2_dspark_round_bf16(alpha[tap] / alpha_sum);
    }

    const size_t row_offset = (size_t)row * width;
    float square_sum = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_DSPARK_THREADS) {
        const float mixed = tap0[row_offset + column] * alpha[0] +
                            tap1[row_offset + column] * alpha[1] +
                            tap2[row_offset + column] * alpha[2] +
                            tap3[row_offset + column] * alpha[3];
        output[row_offset + column] = mixed;
        square_sum += mixed * mixed;
    }

    threadgroup float partials[INKLING2_DSPARK_SIMDGROUPS];
    const float scale = inkling2_dspark_rms_scale(square_sum, tiisg, sgitg, partials, width, eps);
    for (uint column = tid; column < width; column += INKLING2_DSPARK_THREADS) {
        output[row_offset + column] = output[row_offset + column] * scale *
                                      inkling2_dspark_bf16(norm, column);
    }
}

// Build context K/V cache entries. Keys receive per-head RMSNorm and
// non-traditional RoPE at their absolute target positions; values remain raw.
kernel void inkling2_dspark_context_commit(
    device const float* key_input   [[buffer(0)]],
    device const float* value_input [[buffer(1)]],
    device const ushort* key_norm  [[buffer(2)]],
    device       float* key_cache  [[buffer(3)]],
    device       float* value_cache [[buffer(4)]],
    constant     uint&  start       [[buffer(5)]],
    constant     uint&  rows        [[buffer(6)]],
    constant     float& eps         [[buffer(7)]],
    constant     float& rope_base   [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint head = tgpig.x;
    const uint row = tgpig.y;
    if (head >= INKLING2_DSPARK_KV_HEADS || row >= rows) return;

    const size_t vector_offset = ((size_t)row * INKLING2_DSPARK_KV_HEADS + head) * INKLING2_DSPARK_HEAD_DIM;
    float square_sum = 0.0f;
    for (uint dim = tid; dim < INKLING2_DSPARK_HEAD_DIM; dim += INKLING2_DSPARK_THREADS) {
        const float value = key_input[vector_offset + dim];
        square_sum += value * value;
    }
    threadgroup float partials[INKLING2_DSPARK_SIMDGROUPS];
    const float scale = inkling2_dspark_rms_scale(square_sum, tiisg, sgitg, partials,
        INKLING2_DSPARK_HEAD_DIM, eps);

    const uint absolute_position = start + row;
    const uint slot = absolute_position & (INKLING2_DSPARK_RING - 1);
    const size_t cache_offset = ((size_t)slot * INKLING2_DSPARK_KV_HEADS + head) * INKLING2_DSPARK_HEAD_DIM;
    const float angle_scale = (float)absolute_position;
    for (uint dim = tid; dim < INKLING2_DSPARK_HEAD_DIM; dim += INKLING2_DSPARK_THREADS) {
        const uint pair = dim < INKLING2_DSPARK_HEAD_DIM / 2 ? dim : dim - INKLING2_DSPARK_HEAD_DIM / 2;
        const size_t left_index = vector_offset + pair;
        const size_t right_index = vector_offset + pair + INKLING2_DSPARK_HEAD_DIM / 2;
        const float left = key_input[left_index] * scale * inkling2_dspark_bf16(key_norm, pair);
        const float right = key_input[right_index] * scale *
                            inkling2_dspark_bf16(key_norm, pair + INKLING2_DSPARK_HEAD_DIM / 2);
        const float angle = angle_scale * pow(rope_base, -((float)(2 * pair) / (float)INKLING2_DSPARK_HEAD_DIM));
        const float rotated = dim < INKLING2_DSPARK_HEAD_DIM / 2
            ? left * cos(angle) - right * sin(angle)
            : left * sin(angle) + right * cos(angle);
        key_cache[cache_offset + dim] = rotated;
        value_cache[cache_offset + dim] = value_input[vector_offset + dim];
    }
}

// Normalize and rotate draft Q or K vectors. `heads` is 24 for Q and 8 for K.
kernel void inkling2_dspark_block_qk_prepare(
    device const float* input       [[buffer(0)]],
    device const ushort* norm       [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant     uint&  heads       [[buffer(3)]],
    constant     uint&  rows        [[buffer(4)]],
    constant     uint&  position    [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    constant     float& rope_base   [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint head = tgpig.x;
    const uint row = tgpig.y;
    if (head >= heads || row >= rows) return;

    const size_t vector_offset = ((size_t)row * heads + head) * INKLING2_DSPARK_HEAD_DIM;
    float square_sum = 0.0f;
    for (uint dim = tid; dim < INKLING2_DSPARK_HEAD_DIM; dim += INKLING2_DSPARK_THREADS) {
        const float value = input[vector_offset + dim];
        square_sum += value * value;
    }
    threadgroup float partials[INKLING2_DSPARK_SIMDGROUPS];
    const float scale = inkling2_dspark_rms_scale(square_sum, tiisg, sgitg, partials,
        INKLING2_DSPARK_HEAD_DIM, eps);

    const uint absolute_position = position + row;
    const float angle_scale = (float)absolute_position;
    for (uint dim = tid; dim < INKLING2_DSPARK_HEAD_DIM; dim += INKLING2_DSPARK_THREADS) {
        const uint pair = dim < INKLING2_DSPARK_HEAD_DIM / 2 ? dim : dim - INKLING2_DSPARK_HEAD_DIM / 2;
        const size_t left_index = vector_offset + pair;
        const size_t right_index = vector_offset + pair + INKLING2_DSPARK_HEAD_DIM / 2;
        const float left = input[left_index] * scale * inkling2_dspark_bf16(norm, pair);
        const float right = input[right_index] * scale *
                            inkling2_dspark_bf16(norm, pair + INKLING2_DSPARK_HEAD_DIM / 2);
        const float angle = angle_scale * pow(rope_base, -((float)(2 * pair) / (float)INKLING2_DSPARK_HEAD_DIM));
        output[vector_offset + dim] = dim < INKLING2_DSPARK_HEAD_DIM / 2
            ? left * cos(angle) - right * sin(angle)
            : left * sin(angle) + right * cos(angle);
    }
}

// Full mutual attention over the eight-position draft block plus at most the
// newest 1023 committed context positions. Q/K are already RMS-normalized and
// rotated; context entries use absolute-position ring slots.
kernel void inkling2_dspark_block_attention(
    device const float* query        [[buffer(0)]],
    device const float* block_key    [[buffer(1)]],
    device const float* block_value  [[buffer(2)]],
    device const float* context_key  [[buffer(3)]],
    device const float* context_value [[buffer(4)]],
    device       float* output       [[buffer(5)]],
    constant     uint&  position     [[buffer(6)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint query_head = tgpig.x;
    const uint query_row = tgpig.y;
    if (query_head >= INKLING2_DSPARK_Q_HEADS || query_row >= INKLING2_DSPARK_BLOCK) return;

    const uint kv_head = query_head / (INKLING2_DSPARK_Q_HEADS / INKLING2_DSPARK_KV_HEADS);
    const size_t query_offset = ((size_t)query_row * INKLING2_DSPARK_Q_HEADS + query_head) * INKLING2_DSPARK_HEAD_DIM;
    const uint context_count = min(position, INKLING2_DSPARK_CONTEXT);
    const uint context_start = position - context_count;
    const uint key_count = context_count + INKLING2_DSPARK_BLOCK;
    constexpr float scale = 0.08838834764831845f;
    threadgroup float scores[INKLING2_DSPARK_MAX_KEYS];
    threadgroup float max_partials[INKLING2_DSPARK_SIMDGROUPS];
    threadgroup float sum_partials[INKLING2_DSPARK_SIMDGROUPS];

    for (uint key_index = tid; key_index < key_count; key_index += INKLING2_DSPARK_THREADS) {
        const device float* key_vector;
        if (key_index < context_count) {
            const uint absolute_key = context_start + key_index;
            const uint slot = absolute_key & (INKLING2_DSPARK_RING - 1);
            key_vector = context_key + ((size_t)slot * INKLING2_DSPARK_KV_HEADS + kv_head) * INKLING2_DSPARK_HEAD_DIM;
        } else {
            const uint block_row = key_index - context_count;
            key_vector = block_key + ((size_t)block_row * INKLING2_DSPARK_KV_HEADS + kv_head) * INKLING2_DSPARK_HEAD_DIM;
        }
        float dot = 0.0f;
        for (uint dim = 0; dim < INKLING2_DSPARK_HEAD_DIM; ++dim) {
            dot += query[query_offset + dim] * key_vector[dim];
        }
        scores[key_index] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_max = -INFINITY;
    for (uint key_index = tid; key_index < key_count; key_index += INKLING2_DSPARK_THREADS) {
        local_max = max(local_max, scores[key_index]);
    }
    const float simd_maximum = simd_max(local_max);
    if (tiisg == 0) max_partials[sgitg] = simd_maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float maximum = -INFINITY;
    for (uint group = 0; group < INKLING2_DSPARK_SIMDGROUPS; ++group) {
        maximum = max(maximum, max_partials[group]);
    }

    float local_sum = 0.0f;
    for (uint key_index = tid; key_index < key_count; key_index += INKLING2_DSPARK_THREADS) {
        const float probability = exp(scores[key_index] - maximum);
        scores[key_index] = probability;
        local_sum += probability;
    }
    const float simd_total = simd_sum(local_sum);
    if (tiisg == 0) sum_partials[sgitg] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint group = 0; group < INKLING2_DSPARK_SIMDGROUPS; ++group) {
        total += sum_partials[group];
    }
    for (uint key_index = tid; key_index < key_count; key_index += INKLING2_DSPARK_THREADS) {
        scores[key_index] /= total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t output_offset = ((size_t)query_row * INKLING2_DSPARK_Q_HEADS + query_head) * INKLING2_DSPARK_HEAD_DIM;
    for (uint dim = tid; dim < INKLING2_DSPARK_HEAD_DIM; dim += INKLING2_DSPARK_THREADS) {
        float value = 0.0f;
        for (uint key_index = 0; key_index < key_count; ++key_index) {
            const device float* value_vector;
            if (key_index < context_count) {
                const uint absolute_key = context_start + key_index;
                const uint slot = absolute_key & (INKLING2_DSPARK_RING - 1);
                value_vector = context_value + ((size_t)slot * INKLING2_DSPARK_KV_HEADS + kv_head) * INKLING2_DSPARK_HEAD_DIM;
            } else {
                const uint block_row = key_index - context_count;
                value_vector = block_value + ((size_t)block_row * INKLING2_DSPARK_KV_HEADS + kv_head) * INKLING2_DSPARK_HEAD_DIM;
            }
            value += scores[key_index] * value_vector[dim];
        }
        output[output_offset + dim] = value;
    }
}

kernel void inkling2_dspark_swiglu(
    device const float* gate         [[buffer(0)]],
    device const float* up           [[buffer(1)]],
    device       float* output       [[buffer(2)]],
    constant     uint&  count        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint index = tgpig.x * INKLING2_DSPARK_THREADS + tid;
    if (index >= count) return;
    const float value = gate[index];
    const float silu = value / (1.0f + exp(-value));
    output[index] = silu * up[index];
}
