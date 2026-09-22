#include <metal_stdlib>
using namespace metal;

constant uint QI21_MAX_SIMDGROUPS = 32;

inline float qi21_reduce_sum(float value,
                             threadgroup float* partials,
                             uint tid,
                             uint lane,
                             uint simdgroup,
                             uint simdgroups) {
    float sum = simd_sum(value);
    if (lane == 0) partials[simdgroup] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < simdgroups; ++i) total += partials[i];
        partials[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partials[0];
}

kernel void qi21_layernorm_modulate_gate(
    device const float* input [[buffer(0)]],
    device const float* modulation [[buffer(1)]],
    device float* normalized [[buffer(2)]],
    device float* gate [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    if (row >= tokens) return;
    threadgroup float partials[QI21_MAX_SIMDGROUPS];
    const uint simdgroups = (threads + 31) / 32;
    const uint base = row * dim;
    const uint mod_base = row * 4 * dim;

    float local_sum = 0.0f;
    for (uint d = tid; d < dim; d += threads) local_sum += input[base + d];
    const float mean = qi21_reduce_sum(local_sum, partials, tid, lane, simdgroup, simdgroups) / float(dim);

    float local_var = 0.0f;
    for (uint d = tid; d < dim; d += threads) {
        const float delta = input[base + d] - mean;
        local_var += delta * delta;
    }
    const float variance = qi21_reduce_sum(local_var, partials, tid, lane, simdgroup, simdgroups) / float(dim);
    const float inv_std = rsqrt(variance + eps);

    for (uint d = tid; d < dim; d += threads) {
        normalized[base + d] = (input[base + d] - mean) * inv_std *
                               (1.0f + modulation[mod_base + d]);
        gate[base + d] = tanh(modulation[mod_base + dim + d]);
    }
}

kernel void qi21_qk_rms_rope(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* q_weight [[buffer(2)]],
    device const float* k_weight [[buffer(3)]],
    device const int* positions [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& axis0 [[buffer(8)]],
    constant uint& axis1 [[buffer(9)]],
    constant uint& axis2 [[buffer(10)]],
    constant float& eps [[buffer(11)]],
    constant float& rope_theta [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    threadgroup float* scratch [[threadgroup(0)]]) {
    const uint token = group / heads;
    const uint head = group - token * heads;
    if (token >= tokens) return;
    threadgroup float partials_q[QI21_MAX_SIMDGROUPS];
    threadgroup float partials_k[QI21_MAX_SIMDGROUPS];
    threadgroup float inv_q;
    threadgroup float inv_k;
    const uint simdgroups = (threads + 31) / 32;
    const uint base = token * heads * head_dim + head * head_dim;

    const bool active = tid < head_dim;
    const float qv = active ? q[base + tid] : 0.0f;
    const float kv = active ? k[base + tid] : 0.0f;
    if (active) {
        scratch[tid] = qv;
        scratch[head_dim + tid] = kv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float qsum = simd_sum(qv * qv);
    float ksum = simd_sum(kv * kv);
    if (lane == 0) {
        partials_q[simdgroup] = qsum;
        partials_k[simdgroup] = ksum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float qtotal = 0.0f;
        float ktotal = 0.0f;
        for (uint i = 0; i < simdgroups; ++i) {
            qtotal += partials_q[i];
            ktotal += partials_k[i];
        }
        inv_q = rsqrt(qtotal / float(head_dim) + eps);
        inv_k = rsqrt(ktotal / float(head_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        scratch[tid] = qv * inv_q * q_weight[tid];
        scratch[head_dim + tid] = kv * inv_k * k_weight[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!active) return;

    uint axis_offset;
    uint axis_dim;
    uint axis;
    if (tid < axis0) {
        axis_offset = 0;
        axis_dim = axis0;
        axis = 0;
    } else if (tid < axis0 + axis1) {
        axis_offset = axis0;
        axis_dim = axis1;
        axis = 1;
    } else {
        axis_offset = axis0 + axis1;
        axis_dim = axis2;
        axis = 2;
    }
    const uint local_d = tid - axis_offset;
    const uint pair_base = axis_offset + (local_d & ~1u);
    const uint pair = (pair_base - axis_offset) / 2;
    const float frequency = pow(rope_theta, -float(pair * 2) / float(axis_dim));
    const float angle = float(positions[token * 3 + axis]) * frequency;
    const float c = cos(angle);
    const float s = sin(angle);

    const float qr = scratch[pair_base];
    const float qi = scratch[pair_base + 1];
    const float kr = scratch[head_dim + pair_base];
    const float ki = scratch[head_dim + pair_base + 1];
    if ((local_d & 1u) == 0) {
        q[base + tid] = qr * c - qi * s;
        k[base + tid] = kr * c - ki * s;
    } else {
        q[base + tid] = qr * s + qi * c;
        k[base + tid] = kr * s + ki * c;
    }
}

kernel void qi21_block_causal_attention(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* image_ids [[buffer(3)]],
    device const uchar* key_valid [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint query_token = group / heads;
    const uint head = group - query_token * heads;
    if (query_token >= tokens) return;
    threadgroup float partials[QI21_MAX_SIMDGROUPS];
    threadgroup float probability;
    threadgroup float correction;
    threadgroup float inverse_sum;
    const uint simdgroups = (threads + 31) / 32;
    const uint qbase = (query_token * heads + head) * head_dim;
    const float qv = tid < head_dim ? q[qbase + tid] : 0.0f;
    float accumulator = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint key_token = 0; key_token < tokens; ++key_token) {
        const bool same_image = image_ids[query_token] >= 0 &&
                                image_ids[query_token] == image_ids[key_token];
        const bool allowed = key_valid[key_token] != 0 &&
                             (query_token >= key_token || same_image);
        const uint kbase = (key_token * heads + head) * head_dim;
        const float product = allowed && tid < head_dim ? qv * k[kbase + tid] : 0.0f;
        const float subgroup_sum = simd_sum(product);
        if (lane == 0) partials[simdgroup] = subgroup_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float dot = 0.0f;
            for (uint i = 0; i < simdgroups; ++i) dot += partials[i];
            const float score = allowed ? dot * scale : -INFINITY;
            const float next_max = max(running_max, score);
            correction = isinf(running_max) ? 0.0f : exp(running_max - next_max);
            probability = allowed ? exp(score - next_max) : 0.0f;
            running_sum = running_sum * correction + probability;
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < head_dim) {
            accumulator = accumulator * correction + probability * v[kbase + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) inverse_sum = 1.0f / running_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < head_dim) output[qbase + tid] = accumulator * inverse_sum;
}

kernel void qi21_residual_layernorm_modulate_gate(
    device const float* hidden [[buffer(0)]],
    device const float* projected [[buffer(1)]],
    device const float* gate1 [[buffer(2)]],
    device const float* modulation [[buffer(3)]],
    device float* state [[buffer(4)]],
    device float* normalized [[buffer(5)]],
    device float* gate2 [[buffer(6)]],
    constant uint& tokens [[buffer(7)]],
    constant uint& dim [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    if (row >= tokens) return;
    threadgroup float partials[QI21_MAX_SIMDGROUPS];
    const uint simdgroups = (threads + 31) / 32;
    const uint base = row * dim;
    const uint mod_base = row * 4 * dim;

    for (uint d = tid; d < dim; d += threads) {
        state[base + d] = hidden[base + d] + gate1[base + d] * projected[base + d];
    }
    threadgroup_barrier(mem_flags::mem_device);

    float local_sum = 0.0f;
    for (uint d = tid; d < dim; d += threads) local_sum += state[base + d];
    const float mean = qi21_reduce_sum(local_sum, partials, tid, lane, simdgroup, simdgroups) / float(dim);

    float local_var = 0.0f;
    for (uint d = tid; d < dim; d += threads) {
        const float delta = state[base + d] - mean;
        local_var += delta * delta;
    }
    const float variance = qi21_reduce_sum(local_var, partials, tid, lane, simdgroup, simdgroups) / float(dim);
    const float inv_std = rsqrt(variance + eps);

    for (uint d = tid; d < dim; d += threads) {
        normalized[base + d] = (state[base + d] - mean) * inv_std *
                               (1.0f + modulation[mod_base + 2 * dim + d]);
        gate2[base + d] = tanh(modulation[mod_base + 3 * dim + d]);
    }
}

kernel void qi21_swiglu(
    device const float* fused [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& tokens [[buffer(2)]],
    constant uint& intermediate_dim [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint count = tokens * intermediate_dim;
    if (index >= count) return;
    const uint row = index / intermediate_dim;
    const uint column = index - row * intermediate_dim;
    const uint base = row * 2 * intermediate_dim;
    const float gate = fused[base + column];
    const float up = fused[base + intermediate_dim + column];
    output[index] = (gate / (1.0f + exp(-gate))) * up;
}

kernel void qi21_residual_gate_add(
    device const float* state [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* projected [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = state[index] + gate[index] * projected[index];
}
