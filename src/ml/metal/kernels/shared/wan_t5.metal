// Copyright 2024-2025 The Alibaba Wan Team Authors. Apache-2.0.
// F32 neural arithmetic; optional original BF16 resident weights.
#include <metal_stdlib>
using namespace metal;

inline float t5_bf16(ushort x) { return as_type<float>(uint(x) << 16); }

// T5LayerNorm uses torch.rsqrt(var + eps). PyTorch 2.8's MPS unary kernel
// implements rsqrt as 1 / precise::sqrt(x), rather than Metal's rsqrt
// intrinsic. Keep this reduction and rounding behavior local to T5.
kernel void wan_t5_rms_norm(device const float *x [[buffer(0)]],
    device const ushort *weight16 [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &rows [[buffer(3)]], constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]], constant uint &weight_bf16 [[buffer(6)]],
    device const float *weight32 [[buffer(7)]],
    constant uint &vector4 [[buffer(8)]],
    device float *mean_square_out [[buffer(9)]], device float *inverse_rms_out [[buffer(10)]],
    constant uint &capture_stats [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partials[32], mean_square, inverse;
    if (row >= rows) return;
    float sum = 0.0f;
    if (vector4 != 0) {
        device const float4 *row_x = reinterpret_cast<device const float4 *>(x + row * dim);
        float4 accum = 0.0f;
        for (uint i = tid; i < dim / 4; i += threads) {
            const float4 value = row_x[i];
            // PyTorch's pinned MPS oracle materializes pow(x, 2) before mean.
            // Match that elementwise square instead of allowing x*x contraction.
            accum += float4(metal::precise::pow(value.x, 2.0f),
                            metal::precise::pow(value.y, 2.0f),
                            metal::precise::pow(value.z, 2.0f),
                            metal::precise::pow(value.w, 2.0f));
        }
        sum = (accum.x + accum.y) + (accum.z + accum.w);
    } else {
        for (uint d = tid; d < dim; d += threads) {
            const float value = x[row * dim + d];
            sum += metal::precise::pow(value, 2.0f);
        }
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        mean_square = total / float(dim);
        inverse = 1.0f / metal::precise::sqrt(mean_square + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (capture_stats != 0 && tid == 0) {
        mean_square_out[row] = mean_square;
        inverse_rms_out[row] = inverse;
    }
    for (uint d = tid; d < dim; d += threads) {
        const float normalized = x[row * dim + d] * inverse;
        const float weight = weight_bf16 ? t5_bf16(weight16[d]) : weight32[d];
        y[row * dim + d] = weight * normalized;
    }
}

// Diagnostic-only counterpart: hold the scalar256 reduction topology fixed
// while forbidding multiply-add contraction in the sum-of-squares loop. This
// distinguishes product-rounding fusion from a different reduction tree.
kernel void wan_t5_rms_norm_contract_off(device const float *x [[buffer(0)]],
    device const ushort *weight16 [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &rows [[buffer(3)]], constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]], constant uint &weight_bf16 [[buffer(6)]],
    device const float *weight32 [[buffer(7)]],
    constant uint &vector4 [[buffer(8)]],
    device float *mean_square_out [[buffer(9)]], device float *inverse_rms_out [[buffer(10)]],
    constant uint &capture_stats [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    #pragma METAL fp contract(off)
    threadgroup float partials[32], mean_square, inverse;
    if (row >= rows) return;
    float sum = 0.0f;
    if (vector4 != 0) {
        device const float4 *row_x = reinterpret_cast<device const float4 *>(x + row * dim);
        float4 accum = 0.0f;
        for (uint i = tid; i < dim / 4; i += threads) {
            const float4 value = row_x[i];
            accum += value * value;
        }
        sum = (accum.x + accum.y) + (accum.z + accum.w);
    } else {
        for (uint d = tid; d < dim; d += threads) {
            const float value = x[row * dim + d];
            sum += value * value;
        }
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        mean_square = total / float(dim);
        inverse = 1.0f / metal::precise::sqrt(mean_square + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (capture_stats != 0 && tid == 0) {
        mean_square_out[row] = mean_square;
        inverse_rms_out[row] = inverse;
    }
    for (uint d = tid; d < dim; d += threads) {
        const float normalized = x[row * dim + d] * inverse;
        const float weight = weight_bf16 ? t5_bf16(weight16[d]) : weight32[d];
        y[row * dim + d] = weight * normalized;
    }
}

// PyTorch's MPS T5 path materializes `pow(x, 2)` before reducing the feature
// axis. Keep this as a separate dispatch so every F32 square is rounded to
// resident storage before the prepared MPSGraph mean consumes it.
kernel void wan_t5_rms_square(device const float *x [[buffer(0)]],
    device float *square [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) square[id] = metal::precise::pow(x[id], 2.0f);
}

// Complete RMS normalization from the source-shaped MPSGraph mean. MPS unary
// rsqrt uses 1 / precise::sqrt; preserve that form and the upstream
// (x * inverse_rms) * weight order.
kernel void wan_t5_rms_from_mean(device const float *x [[buffer(0)]],
    device const ushort *weight16 [[buffer(1)]], device float *y [[buffer(2)]],
    device const float *mean_square [[buffer(3)]],
    constant uint &rows [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]], constant uint &weight_bf16 [[buffer(7)]],
    device const float *weight32 [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float inverse;
    if (row >= rows) return;
    if (tid == 0) inverse = 1.0f / metal::precise::sqrt(mean_square[row] + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < dim; d += threads) {
        const float normalized = x[row * dim + d] * inverse;
        const float weight = weight_bf16 ? t5_bf16(weight16[d]) : weight32[d];
        y[row * dim + d] = normalized * weight;
    }
}

// Diagnostic/source-shaped counterpart to PyTorch's Wan T5 GELU expression.
// Keep each tensor-op boundary as a rounded F32 value. The generic shared
// GELU kernel intentionally remains the default until this path is qualified
// against the pinned MPS fixture.
kernel void wan_t5_gelu_source_f32(device const float *x [[buffer(0)]],
    device float *y [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    #pragma METAL fp contract(off)
    if (id >= count) return;
    const float value = x[id];
    volatile float cube = metal::precise::pow(value, 3.0f);
    volatile float cubic = 0.0447149984538555145263671875f * cube;
    volatile float inner = value + cubic;
    volatile float scaled = 0.79788458347320556640625f * inner;
    volatile float tangent = metal::precise::tanh(scaled);
    volatile float one_plus_tangent = 1.0f + tangent;
    volatile float half_value = 0.5f * value;
    volatile float result = half_value * one_plus_tangent;
    y[id] = result;
}

// Preserve low terms in long, cancellation-heavy T5 down projections. Both
// words and every product stay F32; no activation conversion or extra dispatch.
// MPSNDArray FC2 reads a persistent row-major [din,dout] F32 view backed by
// this transposed [dout,din] source matrix. BF16 conversion stays on GPU.
kernel void wan_t5_expand_fc2(device const float *w32 [[buffer(0)]],
    device const ushort *w16 [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &count [[buffer(3)]], constant uint &bf16 [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    out[id] = bf16 ? t5_bf16(w16[id]) : w32[id];
}

kernel void wan_t5_embed(device const float *ids [[buffer(0)]],
    device const ushort *w16 [[buffer(1)]], device float *y [[buffer(2)]],
    constant uint &count [[buffer(3)]], constant uint &dim [[buffer(4)]],
    constant uint &bf16 [[buffer(5)]], device const float *w32 [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint source = uint(ids[id / dim]) * dim + id % dim;
    y[id] = bf16 ? t5_bf16(w16[source]) : w32[source];
}

kernel void wan_t5_product(device float *x [[buffer(0)]], device const float *gate [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) x[id] *= gate[id];
}

kernel void wan_t5_mask_context(device const float *x [[buffer(0)]], device const float *mask [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &count [[buffer(3)]], constant uint &dim [[buffer(4)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = mask[id / dim] == 0.0f ? 0.0f : x[id];
}

// Materialize source-shaped [B,H,Q,K] bias for the prepared rank-three MPS
// products. Each batch has at least one visible key, checked during setup.
kernel void wan_t5_attention_bias(device const ushort *bias16 [[buffer(0)]],
    device const float *bias32 [[buffer(1)]], device const float *mask [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &seq [[buffer(4)]],
    constant uint &heads [[buffer(5)]], constant uint &buckets [[buffer(6)]],
    constant uint &max_distance [[buffer(7)]], constant uint &bf16 [[buffer(8)]],
    constant uint &count [[buffer(9)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint key = id % seq, query = (id / seq) % seq;
    const uint head = (id / (seq * seq)) % heads;
    const uint batch = id / (heads * seq * seq);
    const uint half_buckets = buckets / 2, exact = half_buckets / 2;
    const uint distance = uint(abs(int(key) - int(query)));
    const uint bucket = distance < exact ? distance : min(half_buckets - 1,
        exact + uint(log(float(distance) / float(exact)) /
        log(float(max_distance) / float(exact)) * float(half_buckets - exact)));
    const uint bi = ((key > query ? half_buckets : 0) + bucket) * heads + head;
    out[id] = mask[batch * seq + key] == 0.0f ? -FLT_MAX :
        (bf16 ? t5_bf16(bias16[bi]) : bias32[bi]);
}

// T5 has relative position bias and deliberately has no sqrt(head_dim) scale.
// Stream keys without a quadratic activation allocation. Every batch member
// must have at least one visible key, checked at input preparation.
kernel void wan_t5_attention(device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]], device const float *v [[buffer(2)]],
    device const ushort *bias16 [[buffer(3)]], device const float *mask [[buffer(4)]],
    device float *y [[buffer(5)]], constant uint &seq [[buffer(6)]],
    constant uint &heads [[buffer(7)]], constant uint &dim [[buffer(8)]],
    constant uint &buckets [[buffer(9)]], constant uint &max_distance [[buffer(10)]],
    constant uint &bf16 [[buffer(11)]], device const float *bias32 [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    const uint batch = group / (seq * heads), head = group % heads;
    const uint query = (group / heads) % seq;
    const uint offset = ((batch * seq + query) * heads + head) * dim;
    const float qv = tid < dim ? q[offset + tid] : 0.0f;
    threadgroup float partials[8], probability, correction, inverse;
    float acc = 0.0f, maximum = -INFINITY, sum = 0.0f;
    for (uint key = 0; key < seq; ++key) {
        if (mask[batch * seq + key] == 0.0f) continue;
        const uint kb = ((batch * seq + key) * heads + head) * dim;
        const float product = tid < dim ? qv * k[kb + tid] : 0.0f;
        const float total = simd_sum(product);
        if (lane == 0) partials[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float dot = 0.0f;
            for (uint i = 0; i < threads / 32; ++i) dot += partials[i];
            const uint half_buckets = buckets / 2, exact = half_buckets / 2;
            const uint distance = uint(abs(int(key) - int(query)));
            const uint bucket = distance < exact ? distance : min(half_buckets - 1,
                exact + uint(log(float(distance) / float(exact)) /
                log(float(max_distance) / float(exact)) * float(half_buckets - exact)));
            const uint bi = ((key > query ? half_buckets : 0) + bucket) * heads + head;
            const float score = dot + (bf16 ? t5_bf16(bias16[bi]) : bias32[bi]);
            const float next = max(maximum, score);
            correction = isinf(maximum) ? 0.0f : exp(maximum - next);
            probability = exp(score - next);
            sum = sum * correction + probability;
            maximum = next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < dim) acc = acc * correction + probability * v[kb + tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) inverse = 1.0f / sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < dim) y[offset + tid] = acc * inverse;
}
