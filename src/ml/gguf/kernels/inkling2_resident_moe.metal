// Batched Inkling2 projections and MoE kernels. This source is appended to
// gemm_q56k.metal so the IQ4_NL expert path shares its block decoder.

constant uint INKLING2_RESIDENT_TILE = 4;

// A simdgroup computes one output row and four adjacent input rows. Each
// thread loads one weight value and reuses it across the four token rows.
kernel void inkling2_resident_gemm_f32(
    device const float*   w         [[buffer(0)]],
    device const float*   x         [[buffer(1)]],
    device       float*   output    [[buffer(2)]],
    constant     uint&    in_dim    [[buffer(3)]],
    constant     uint&    out_dim   [[buffer(4)]],
    constant     uint&    batch     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgpig.x * INKLING2_RESIDENT_TILE + (uint)sgitg;
    const uint first_batch = tgpig.y * INKLING2_RESIDENT_TILE;
    if (row >= out_dim) return;

    const device float* w_row = w + (size_t)row * (size_t)in_dim;
    float4 sums = float4(0.0f);
    for (uint col = tiisg; col < in_dim; col += 32) {
        const float weight = w_row[col];
        float4 values = float4(0.0f);
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                values[b] = x[(size_t)batch_row * (size_t)in_dim + col];
            }
        }
        sums += values * weight;
    }

    const float4 totals = float4(simd_sum(sums.x), simd_sum(sums.y),
                                 simd_sum(sums.z), simd_sum(sums.w));
    if (tiisg == 0) {
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                output[(size_t)batch_row * (size_t)out_dim + row] = totals[b];
            }
        }
    }
}

kernel void inkling2_resident_gemm_q8_0(
    device const uint8_t* w_raw     [[buffer(0)]],
    device const float*   x         [[buffer(1)]],
    device       float*   output    [[buffer(2)]],
    constant     uint&    in_dim    [[buffer(3)]],
    constant     uint&    out_dim   [[buffer(4)]],
    constant     uint&    batch     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint row = tgpig.x * INKLING2_RESIDENT_TILE + (uint)sgitg;
    const uint first_batch = tgpig.y * INKLING2_RESIDENT_TILE;
    if (row >= out_dim) return;

    const uint row_bytes = nb * 34;
    device const block_q8_0_56* w_row =
        (device const block_q8_0_56*)(w_raw + (size_t)row * (size_t)row_bytes);
    float4 sums = float4(0.0f);

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q8_0_56* block = w_row + ib;
        const float scale = (float)block->d;
        const float quant = (float)block->qs[tiisg];
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                const float value = x[(size_t)batch_row * (size_t)in_dim +
                                      (size_t)ib * Q8_0_QK + tiisg];
                sums[b] += scale * value * quant;
            }
        }
    }

    const float4 totals = float4(simd_sum(sums.x), simd_sum(sums.y),
                                 simd_sum(sums.z), simd_sum(sums.w));
    if (tiisg == 0) {
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                output[(size_t)batch_row * (size_t)out_dim + row] = totals[b];
            }
        }
    }
}

inline float inkling2_resident_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

// One 32-thread group routes one token row. The host chunks the y dimension
// and supplies row_base so very long prefills do not rely on a huge grid.
kernel void inkling2_resident_moe_route(
    device const float* logits       [[buffer(0)]],
    device const float* bias         [[buffer(1)]],
    device       uint*  selected_ids [[buffer(2)]],
    device       float* routing_wts  [[buffer(3)]],
    constant     uint&  n_experts    [[buffer(4)]],
    constant     uint&  k            [[buffer(5)]],
    constant     float& scale        [[buffer(6)]],
    constant     uint&  has_bias     [[buffer(7)]],
    constant     uint&  rows         [[buffer(8)]],
    constant     uint&  row_base     [[buffer(9)]],
    threadgroup  float* scratch      [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint row = row_base + tgpig.y;
    if (row >= rows) return;

    const uint lane = (uint)tiisg;
    threadgroup float* probabilities = scratch;
    threadgroup float* selection_scores = scratch + n_experts;
    device const float* row_logits = logits + (size_t)row * n_experts;
    device uint* row_ids = selected_ids + (size_t)row * k;
    device float* row_weights = routing_wts + (size_t)row * k;

    for (uint expert = lane; expert < n_experts; expert += 32) {
        const float logit = row_logits[expert];
        const float probability = isfinite(logit)
            ? inkling2_resident_sigmoid(logit)
            : (logit > 0.0f ? 1.0f : 0.0f);
        probabilities[expert] = probability;

        float score = probability;
        if (has_bias != 0) score += bias[expert];
        selection_scores[expert] = isnan(score) ? -INFINITY : score;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float selected_probability_sum = 0.0f;
        for (uint rank = 0; rank < k; ++rank) {
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
            row_ids[rank] = best_expert;
            selected_probability_sum += probabilities[best_expert];
            selection_scores[best_expert] = as_type<float>(0x7fc00000u);
        }

        const float normalization = selected_probability_sum + 1.0e-20f;
        for (uint rank = 0; rank < k; ++rank) {
            const uint expert = row_ids[rank];
            row_weights[rank] = probabilities[expert] / normalization * scale;
        }
    }
}

// Each y group computes four output rows for one flattened (token, expert)
// route. routed_base keeps the grid bounded without changing row ordering.
kernel void inkling2_resident_moe_matvec_iq4_nl(
    device const uint8_t* expert_weights [[buffer(0)]],
    device const float*   x              [[buffer(1)]],
    device const uint*    selected_ids   [[buffer(2)]],
    device       float*   output         [[buffer(3)]],
    constant     uint&    in_dim         [[buffer(4)]],
    constant     uint&    out_dim        [[buffer(5)]],
    constant     uint&    expert_bytes   [[buffer(6)]],
    constant     uint&    input_stride   [[buffer(7)]],
    constant     uint&    top_k          [[buffer(8)]],
    constant     uint&    input_is_routed [[buffer(9)]],
    constant     uint&    routed_base    [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint route = routed_base + tgpig.y;
    const uint first_row = (tgpig.x * 2 + (uint)sgitg) * 2;
    if (first_row >= out_dim) return;

    const uint expert_id = selected_ids[route];
    const size_t expert_offset = (size_t)expert_id * (size_t)expert_bytes;
    device const uint8_t* w_raw = expert_weights + expert_offset;
    const uint nb = in_dim / IQ4_NL_QK;
    const uint row_bytes = nb * 18;
    const uint input_row = input_is_routed != 0 ? route : route / top_k;
    device const float* y_base = x + (size_t)input_row * input_stride;
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
            device const block_iq4_nl_56* block =
                (device const block_iq4_nl_56*)(w_raw + row_id * row_bytes) + ib;
            sumf[row] += (float)block->d * iq4_nl_dot_half(block, it, yl0, yl1, yl2, yl3);
        }
        yb += 16 * IQ4_NL_QK;
    }

    for (short row = 0; row < 2 && first_row + row < out_dim; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) output[(size_t)route * out_dim + first_row + row] = total;
    }
}

kernel void inkling2_resident_moe_matvec_q8_0(
    device const uint8_t* expert_weights [[buffer(0)]],
    device const float*   x              [[buffer(1)]],
    device const uint*    selected_ids   [[buffer(2)]],
    device       float*   output         [[buffer(3)]],
    constant     uint&    in_dim         [[buffer(4)]],
    constant     uint&    out_dim        [[buffer(5)]],
    constant     uint&    expert_bytes   [[buffer(6)]],
    constant     uint&    input_stride   [[buffer(7)]],
    constant     uint&    top_k          [[buffer(8)]],
    constant     uint&    input_is_routed [[buffer(9)]],
    constant     uint&    routed_base    [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint route = routed_base + tgpig.y;
    const uint row_id = tgpig.x * MV8_NSG + (uint)sgitg;
    if (row_id >= out_dim) return;

    const uint expert_id = selected_ids[route];
    const size_t expert_offset = (size_t)expert_id * (size_t)expert_bytes;
    device const uint8_t* w_raw = expert_weights + expert_offset;
    const uint nb = in_dim / Q8_0_QK;
    const uint row_bytes = nb * 34;
    const uint input_row = input_is_routed != 0 ? route : route / top_k;
    device const float* y_base = x + (size_t)input_row * input_stride;
    float sum = 0.0f;

    for (uint ib = 0; ib < nb; ++ib) {
        const float y = y_base[ib * Q8_0_QK + tiisg];
        device const block_q8_0_56* block =
            (device const block_q8_0_56*)(w_raw + row_id * row_bytes) + ib;
        sum += (float)block->d * y * (float)block->qs[tiisg];
    }

    const float total = simd_sum(sum);
    if (tiisg == 0) output[(size_t)route * out_dim + row_id] = total;
}

inline float inkling2_resident_situ(float gate, float up, float beta, float linear_beta) {
    const float bounded_gate = beta * tanh(gate / beta);
    const float bounded_up = linear_beta * tanh(up / linear_beta);
    return bounded_gate * inkling2_resident_sigmoid(gate) * bounded_up;
}

kernel void inkling2_resident_moe_situ(
    device const float* gate_up [[buffer(0)]],
    device       float* gated   [[buffer(1)]],
    constant     uint&  width   [[buffer(2)]],
    constant     uint&  total   [[buffer(3)]],
    constant     uint&  base    [[buffer(4)]],
    constant     float& beta    [[buffer(5)]],
    constant     float& linear_beta [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint element = base + tid;
    if (element >= total) return;

    const uint route = element / width;
    const uint column = element - route * width;
    const size_t row_base = (size_t)route * (size_t)(2 * width);
    const float gate = gate_up[row_base + column];
    const float up = gate_up[row_base + width + column];
    gated[element] = inkling2_resident_situ(gate, up, beta, linear_beta);
}

kernel void inkling2_resident_moe_combine(
    device const float* values         [[buffer(0)]],
    device const float* mixing_weights [[buffer(1)]],
    device       float* result         [[buffer(2)]],
    constant     uint&  dim            [[buffer(3)]],
    constant     uint&  k              [[buffer(4)]],
    constant     uint&  total          [[buffer(5)]],
    constant     uint&  base           [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint element = base + tid;
    if (element >= total) return;

    const uint row = element / dim;
    const uint column = element - row * dim;
    float sum = 0.0f;
    for (uint slot = 0; slot < k; ++slot) {
        const uint route = row * k + slot;
        sum += mixing_weights[route] * values[(size_t)route * dim + column];
    }
    result[element] = sum;
}
