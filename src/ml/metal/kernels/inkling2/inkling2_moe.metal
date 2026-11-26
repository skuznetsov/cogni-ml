// Single-token Inkling2 MoE kernels. The host appends this source to
// gemm_q56k.metal so the expert matvec can share its IQ4_NL block helper.

inline float inkling2_moe_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

// One 32-thread simdgroup computes sigmoid probabilities cooperatively, then
// lane 0 performs stable top-k selection. Ties keep the smaller expert ID.
// Threadgroup scratch is two adjacent arrays of n_experts floats.
kernel void inkling2_moe_route(
    device const float* logits       [[buffer(0)]],
    device const float* bias         [[buffer(1)]],
    device       uint*  selected_ids [[buffer(2)]],
    device       float* routing_wts  [[buffer(3)]],
    constant     uint&  n_experts    [[buffer(4)]],
    constant     uint&  k            [[buffer(5)]],
    constant     float& scale        [[buffer(6)]],
    constant     uint&  has_bias     [[buffer(7)]],
    threadgroup  float* scratch      [[threadgroup(0)]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint lane = (uint)tiisg;
    threadgroup float* probabilities = scratch;
    threadgroup float* selection_scores = scratch + n_experts;

    for (uint expert = lane; expert < n_experts; expert += 32) {
        const float logit = logits[expert];
        // NaN logits receive zero probability; signed infinities have their
        // limiting sigmoid values. This keeps output IDs inside the expert set.
        const float probability = isfinite(logit)
            ? inkling2_moe_sigmoid(logit)
            : (logit > 0.0f ? 1.0f : 0.0f);
        probabilities[expert] = probability;

        float score = probability;
        if (has_bias != 0) {
            score += bias[expert];
        }
        // A NaN correction cannot participate in an ordering; keep the route
        // deterministic by ranking that expert last.
        selection_scores[expert] = isnan(score) ? -INFINITY : score;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0) {
        const uint selected_count = min(k, n_experts);
        float selected_probability_sum = 0.0f;
        const uint output_count = k;

        for (uint rank = 0; rank < selected_count; ++rank) {
            bool found = false;
            float best_score = -INFINITY;
            uint best_expert = 0;
            for (uint expert = 0; expert < n_experts; ++expert) {
                const float score = selection_scores[expert];
                const bool consumed = as_type<uint>(score) == 0x7fc00000u;
                if (!consumed && (!found || score > best_score)) {
                    found = true;
                    best_score = score;
                    best_expert = expert;
                }
            }

            // selected_count <= n_experts, so a candidate always exists.
            // NaN marks a consumed score, including when every score is -Inf.
            selected_ids[rank] = best_expert;
            selected_probability_sum += probabilities[best_expert];
            selection_scores[best_expert] = as_type<float>(0x7fc00000u);
        }

        const float normalization = selected_probability_sum + 1.0e-20f;
        for (uint rank = 0; rank < selected_count; ++rank) {
            const uint expert = selected_ids[rank];
            routing_wts[rank] = probabilities[expert] / normalization * scale;
        }
        for (uint rank = selected_count; rank < output_count; ++rank) {
            // k may exceed n_experts only for malformed host input. Keep the
            // ABI output initialized and never emit an out-of-range ID.
            selected_ids[rank] = 0;
            routing_wts[rank] = 0.0f;
        }
    }
}

// One 64-thread group computes four rows for one selected expert. This keeps
// the established IQ4_NL lane partition and reduction arithmetic from GEMV.
kernel void inkling2_moe_matvec_iq4_nl(
    device const uint8_t* expert_weights [[buffer(0)]],
    device const float*   x              [[buffer(1)]],
    device const uint*    selected_ids   [[buffer(2)]],
    device       float*   output         [[buffer(3)]],
    constant     uint&    in_dim         [[buffer(4)]],
    constant     uint&    out_dim        [[buffer(5)]],
    constant     uint&    expert_bytes   [[buffer(6)]],
    constant     uint&    input_stride   [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tgpig.y;
    const uint first_row = (tgpig.x * 2 + (uint)sgitg) * 2;
    if (first_row >= out_dim) return;

    const uint expert_id = selected_ids[slot];
    const size_t expert_offset = (size_t)expert_id * (size_t)expert_bytes;
    device const uint8_t* w_raw = expert_weights + expert_offset;
    const uint nb = in_dim / IQ4_NL_QK;
    const uint row_bytes = nb * 18;
    device const float* y_base = x + (size_t)slot * (size_t)input_stride;
    float sumf[2] = {0.f, 0.f};

    const ushort ix = tiisg / 2;
    const ushort it = tiisg & 1;
    device const float* yb = y_base + ix * IQ4_NL_QK + it * 8;

    for (uint ib = ix; ib < nb; ib += 16) {
        device const float4* y4 = (device const float4*)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        for (short row = 0; row < 2; ++row) {
            const uint row_id = first_row + row;
            if (row_id >= out_dim) continue;
            device const block_iq4_nl_56* blk =
                (device const block_iq4_nl_56*)(w_raw + row_id * row_bytes) + ib;
            sumf[row] += (float)blk->d * iq4_nl_dot_half(blk, it, yl0, yl1, yl2, yl3);
        }

        yb += 16 * IQ4_NL_QK;
    }

    for (short row = 0; row < 2 && first_row + row < out_dim; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[(size_t)slot * (size_t)out_dim + first_row + row] = total;
        }
    }
}

// Q8_0 counterpart for checkpoint layers whose experts use Q8_0. Four
// simdgroups process four output rows per selected expert in each 128-thread
// group, matching simd_mv_q8_0_f32's row and lane geometry.
kernel void inkling2_moe_matvec_q8_0(
    device const uint8_t* expert_weights [[buffer(0)]],
    device const float*   x              [[buffer(1)]],
    device const uint*    selected_ids   [[buffer(2)]],
    device       float*   output         [[buffer(3)]],
    constant     uint&    in_dim         [[buffer(4)]],
    constant     uint&    out_dim        [[buffer(5)]],
    constant     uint&    expert_bytes   [[buffer(6)]],
    constant     uint&    input_stride   [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint slot = tgpig.y;
    const uint row_id = tgpig.x * MV8_NSG + (uint)sgitg;
    if (row_id >= out_dim) return;

    const uint expert_id = selected_ids[slot];
    const size_t expert_offset = (size_t)expert_id * (size_t)expert_bytes;
    device const uint8_t* w_raw = expert_weights + expert_offset;
    const uint nb = in_dim / Q8_0_QK;
    const uint row_bytes = nb * 34;
    device const float* y_base = x + (size_t)slot * (size_t)input_stride;
    float sum = 0.0f;

    for (uint ib = 0; ib < nb; ++ib) {
        const float y = y_base[ib * Q8_0_QK + tiisg];
        device const block_q8_0_56* blk =
            (device const block_q8_0_56*)(w_raw + row_id * row_bytes) + ib;
        sum += (float)blk->d * y * (float)blk->qs[tiisg];
    }

    const float total = simd_sum(sum);
    if (tiisg == 0) {
        output[(size_t)slot * (size_t)out_dim + row_id] = total;
    }
}

// Inkling2's bounded-gate, linear-gate and sigmoid activation (situ).
kernel void inkling2_moe_situ(
    device const float* gate_up    [[buffer(0)]],
    device       float* gated      [[buffer(1)]],
    constant     uint&  width      [[buffer(2)]],
    constant     uint&  k          [[buffer(3)]],
    constant     float& beta       [[buffer(4)]],
    constant     float& linear_beta [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = k * width;
    if (tid >= total) return;

    const uint slot = tid / width;
    const uint column = tid - slot * width;
    const size_t row_base = (size_t)slot * (size_t)(2 * width);
    const float gate = gate_up[row_base + column];
    const float up = gate_up[row_base + width + column];
    const float bounded_gate = beta * tanh(gate / beta);
    const float bounded_up = linear_beta * tanh(up / linear_beta);
    gated[tid] = bounded_gate * inkling2_moe_sigmoid(gate) * bounded_up;
}

// Ordered top-k accumulation, matching the CPU expert iteration order.
kernel void inkling2_moe_combine(
    device const float* values         [[buffer(0)]],
    device const float* mixing_weights [[buffer(1)]],
    device       float* result         [[buffer(2)]],
    constant     uint&  dim             [[buffer(3)]],
    constant     uint&  k               [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;

    float sum = 0.0f;
    for (uint slot = 0; slot < k; ++slot) {
        sum += mixing_weights[slot] * values[(size_t)slot * (size_t)dim + tid];
    }
    result[tid] = sum;
}
