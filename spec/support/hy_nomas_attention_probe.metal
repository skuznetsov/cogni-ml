// Test-only Hy-Nomas attention probes.
//
// The caller appends this source after MetalOps::SOURCE and
// HyNomasDualStreamBlock::SOURCE. It relies on their Metal standard-library
// imports and hy_nomas_bf16_roundtrip_f32 helper; it adds no production kernels.
//
// Attention buffers use the production layouts: Q/K/V and output are
// [B, tokens, heads, dim], while scores and probabilities are
// [B, query_tokens, heads, key_tokens]. The host dispatches one threadgroup per
// (batch, query, head) for qk/online probes, with
// ceil(dim / 32) * 32 threads per group. This matches the production attention
// geometry and its dim <= 256 limit. Softmax dispatches one thread per row;
// PV dispatches one thread per output element.

kernel void hy_probe_round(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = hy_nomas_bf16_roundtrip_f32(x[id]);
}

// Score each Q/K pair with the same F32 product, simd_sum, ordered simdgroup
// partial sum, and post-reduction scaling as video_attention_f32. Scores are
// stored as [B, query, head, key]. Host dispatch count is B * nq * heads.
kernel void hy_probe_qk(
    device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
    device float *scores [[buffer(2)]], constant uint &nq [[buffer(3)]],
    constant uint &nk [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant float &scale [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partials[8];
    const uint batch = group / (nq * heads);
    const uint qh = group % (nq * heads);
    const uint head = qh % heads;
    const uint qb = (batch * nq * heads + qh) * dim;
    const float qv = tid < dim ? q[qb + tid] : 0.0f;

    for (uint key = 0; key < nk; ++key) {
        const uint kb = ((batch * nk + key) * heads + head) * dim;
        const float product = tid < dim ? qv * k[kb + tid] : 0.0f;
        const float sum = simd_sum(product);
        if (lane == 0) partials[sg] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = 0.0f;
            for (uint i = 0; i < threads / 32; ++i) total += partials[i];
            scores[group * nk + key] = total * scale;
        }
        // All lanes must finish reading partials before the next key replaces
        // them. This is the score-only counterpart of the production barrier.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Explicit reference-style stable softmax. One invocation owns one complete
// score row and visits keys in increasing order for max, sum, then output.
// This serial reduction order is intentional and differs from any unspecified
// MPSGraph softmax tree. Host dispatch count is B * nq * heads rows.
kernel void hy_probe_softmax(
    device const float *scores [[buffer(0)]], device float *prob [[buffer(1)]],
    constant uint &nk [[buffer(2)]], uint row [[thread_position_in_grid]]) {
    float row_max = -INFINITY;
    for (uint key = 0; key < nk; ++key)
        row_max = max(row_max, scores[row * nk + key]);

    float row_sum = 0.0f;
    for (uint key = 0; key < nk; ++key)
        row_sum += exp(scores[row * nk + key] - row_max);

    const float inverse_sum = 1.0f / row_sum;
    for (uint key = 0; key < nk; ++key)
        prob[row * nk + key] = exp(scores[row * nk + key] - row_max) * inverse_sum;
}

// Frozen-probability P*V diagnostic: no softmax, increasing-key sequential
// F32 FMA reduction per output. `prob` must already be normalized and laid out
// [B, query, head, key]. Output layout is [B, query, head, dim]. Dispatch
// count must equal B * nq * heads * dim.
kernel void hy_probe_pv(
    device const float *prob [[buffer(0)]], device const float *v [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &nq [[buffer(3)]],
    constant uint &nk [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &count [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint group = id / dim;
    const uint channel = id % dim;
    const uint query_row = group / heads;
    const uint batch = query_row / nq;
    const uint head = group % heads;
    float acc = 0.0f;
    for (uint key = 0; key < nk; ++key) {
        const uint probability_index = group * nk + key;
        const uint value_index = ((batch * nk + key) * heads + head) * dim + channel;
        acc = fma(prob[probability_index], v[value_index], acc);
    }
    y[id] = acc;
}

// Apply the production online max/sum and V-accumulator recurrence to already
// scaled F32 scores. This bypasses QK while preserving the attention kernel's
// key order and recurrence. Host dispatch is one group per (batch, query, head)
// with ceil(dim / 32) * 32 threads per group.
kernel void hy_probe_online_scores(
    device const float *scores [[buffer(0)]], device const float *v [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &nq [[buffer(3)]],
    constant uint &nk [[buffer(4)]], constant uint &heads [[buffer(5)]],
    constant uint &dim [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float probability, correction, inverse_sum;
    const uint batch = group / (nq * heads);
    const uint qh = group % (nq * heads);
    const uint head = qh % heads;
    float acc = 0.0f;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (uint key = 0; key < nk; ++key) {
        if (tid == 0) {
            const float score = scores[group * nk + key];
            const float next_max = max(running_max, score);
            correction = isinf(running_max) ? 0.0f : exp(running_max - next_max);
            probability = exp(score - next_max);
            running_sum = running_sum * correction + probability;
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < dim) {
            const uint vb = ((batch * nk + key) * heads + head) * dim;
            acc = acc * correction + probability * v[vb + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) inverse_sum = 1.0f / running_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < dim) y[group * dim + tid] = acc * inverse_sum;
}
