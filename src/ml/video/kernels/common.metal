#include <metal_stdlib>
using namespace metal;

kernel void video_linear_f32(
    device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &din [[buffer(5)]],
    constant uint &dout [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id >= rows * dout) return;
    const uint row = id / dout, col = id % dout;
    float acc = 0.0f;
    for (uint i = 0; i < din; ++i) acc += x[row * din + i] * w[col * din + i];
    y[id] = acc + bias[col];
}

kernel void video_norm_f32(
    device const float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    constant uint &rms [[buffer(6)]], constant uint &affine [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partials[8];
    threadgroup float mean, inverse;
    float sum = 0.0f;
    if (!rms) for (uint d = tid; d < dim; d += threads) sum += x[row * dim + d];
    float reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        mean = total / float(dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint d = tid; d < dim; d += threads) {
        const float z = x[row * dim + d] - mean;
        sum += z * z;
    }
    reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        inverse = rsqrt(total / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < dim; d += threads) {
        float z = (x[row * dim + d] - mean) * inverse;
        y[row * dim + d] = affine ? z * weight[d] + bias[d] : z;
    }
}

kernel void video_modulate_f32(
    device const float *x [[buffer(0)]], device const float *shift [[buffer(1)]],
    device const float *scale [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &count [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant uint &mod_rows [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint m = mod_rows == 1 ? id % dim : id;
    y[id] = x[id] * (1.0f + scale[m]) + shift[m];
}

kernel void video_add_f32(
    device const float *x [[buffer(0)]], device const float *delta [[buffer(1)]],
    device const float *gate [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &count [[buffer(4)]], constant uint &gated [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &gate_rows [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float g = gated ? gate[gate_rows == 1 ? id % dim : id] : 1.0f;
    y[id] = x[id] + g * delta[id];
}

// Error-function approximation used only by explicitly requested exact-GELU
// probes. The admitted model blocks use PyTorch's tanh GELU variant.
inline float video_erf(float x) {
    const float a = abs(x), t = 1.0f / (1.0f + 0.3275911f * a);
    const float p = (((((1.061405429f * t - 1.453152027f) * t)
                      + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return copysign(1.0f - p * exp(-a * a), x);
}

kernel void video_gelu_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant uint &approximate [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float v = x[id];
    y[id] = approximate ? 0.5f * v * (1.0f + tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)))
                        : 0.5f * v * (1.0f + video_erf(v * 0.7071067811865475f));
}

kernel void video_copy_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = x[id];
}

kernel void video_attention_f32(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device const float *v [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &nq [[buffer(4)]], constant uint &nk [[buffer(5)]],
    constant uint &heads [[buffer(6)]], constant uint &dim [[buffer(7)]],
    constant float &scale [[buffer(8)]], uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    const uint batch = group / (nq * heads), qh = group % (nq * heads);
    const uint head = qh % heads, qb = (batch * nq * heads + qh) * dim;
    threadgroup float partials[8], probability, correction, inverse_sum;
    const float qv = tid < dim ? q[qb + tid] : 0.0f;
    float acc = 0.0f, running_max = -INFINITY, running_sum = 0.0f;
    for (uint key = 0; key < nk; ++key) {
        const uint kb = ((batch * nk + key) * heads + head) * dim;
        const float product = tid < dim ? qv * k[kb + tid] : 0.0f;
        const float sum = simd_sum(product);
        if (lane == 0) partials[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < threads / 32; ++i) total += partials[i];
            const float score = total * scale, next_max = max(running_max, score);
            correction = isinf(running_max) ? 0.0f : exp(running_max - next_max);
            probability = exp(score - next_max);
            running_sum = running_sum * correction + probability;
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < dim) acc = acc * correction + probability * v[kb + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) inverse_sum = 1.0f / running_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < dim) y[qb + tid] = acc * inverse_sum;
}
