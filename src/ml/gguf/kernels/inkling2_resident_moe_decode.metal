// Exact small-batch expert matvecs for Inkling 2 decode blocks. The route
// buckets and tile descriptors use the same three-uint layout as the
// resident grouped path, but each SIMD lane retains the one-row GEMV's
// accumulation order instead of using simdgroup matrix multiply-accumulate.

constant uint INKLING2_DECODE_ROUTE_CAPACITY = 4;

struct Inkling2DecodeExpertTile {
    uint expert;
    uint first_route;
    uint count;
};

struct Inkling2DecodeIq4NlHalf {
    float4 qf10;
    float4 qf11;
    float4 qf20;
    float4 qf21;
};

// Decode one lane-owned 16-value half once so grouped decode can share the
// weight work across all routes assigned to the expert tile.
inline Inkling2DecodeIq4NlHalf inkling2_decode_iq4_nl_half(
    device const block_iq4_nl_56* block,
    ushort it)
{
    device const uint16_t* q4 = (device const uint16_t*)(block->qs + 8 * it);
    uint32_t aux32[2];
    thread const uint8_t* q8 = (thread const uint8_t*)aux32;
    Inkling2DecodeIq4NlHalf decoded;

    aux32[0] = q4[0] | ((uint32_t)q4[1] << 16);
    aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
    aux32[0] &= 0x0f0f0f0f;
    decoded.qf10 = float4(kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]);
    decoded.qf11 = float4(kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]);

    aux32[0] = q4[2] | ((uint32_t)q4[3] << 16);
    aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
    aux32[0] &= 0x0f0f0f0f;
    decoded.qf20 = float4(kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]);
    decoded.qf21 = float4(kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]);
    return decoded;
}

inline float inkling2_decode_iq4_nl_dot(
    thread const Inkling2DecodeIq4NlHalf& weights,
    float4 yl0,
    float4 yl1,
    float4 yl2,
    float4 yl3)
{
    const float4 acc = yl0 * weights.qf10 + yl1 * weights.qf11 + yl2 * weights.qf20 + yl3 * weights.qf21;
    return acc[0] + acc[1] + acc[2] + acc[3];
}

// Each 64-thread group has two SIMD groups. Each SIMD group computes one
// column's gate/up pair, retaining the scalar IQ4_NL GEMV's lane/K order.
kernel void inkling2_resident_moe_matvec_situ_iq4_nl(
    device const uint8_t* weights      [[buffer(0)]],
    device const float*   x            [[buffer(1)]],
    device const uint*    selected_ids [[buffer(2)]],
    device       float*   gated        [[buffer(3)]],
    constant     uint&    in_dim       [[buffer(4)]],
    constant     uint&    expert_dim   [[buffer(5)]],
    constant     uint&    expert_bytes [[buffer(6)]],
    constant     uint&    top_k        [[buffer(7)]],
    constant     uint&    routed_base  [[buffer(8)]],
    constant     float&   beta         [[buffer(9)]],
    constant     float&   linear_beta  [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    const uint column = group.x * 2 + (uint)simd;
    if (column >= expert_dim) return;

    const uint route = routed_base + group.y;
    const uint expert_id = selected_ids[route];
    const device uint8_t* expert_raw = weights + (size_t)expert_id * expert_bytes;
    const uint nb = in_dim / IQ4_NL_QK;
    const uint row_bytes = nb * 18;
    const uint input_row = route / top_k;
    const device float* y_base = x + (size_t)input_row * in_dim;
    const ushort ix = lane / 2;
    const ushort it = lane & 1;
    float gate_sum = 0.f;
    float up_sum = 0.f;

    for (uint ib = ix; ib < nb; ib += 16) {
        const device float4* y4 = (device const float4*)(y_base + (size_t)ib * IQ4_NL_QK + it * 8);
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        const device block_iq4_nl_56* gate_block =
            (device const block_iq4_nl_56*)(expert_raw + (size_t)column * row_bytes) + ib;
        const device block_iq4_nl_56* up_block =
            (device const block_iq4_nl_56*)(expert_raw + (size_t)(column + expert_dim) * row_bytes) + ib;
        gate_sum += (float)gate_block->d * iq4_nl_dot_half(gate_block, it, yl0, yl1, yl2, yl3);
        up_sum += (float)up_block->d * iq4_nl_dot_half(up_block, it, yl0, yl1, yl2, yl3);
    }

    const float gate = simd_sum(gate_sum);
    const float up = simd_sum(up_sum);
    if (lane == 0) {
        gated[(size_t)route * expert_dim + column] =
            inkling2_resident_situ(gate, up, beta, linear_beta);
    }
}

kernel void inkling2_resident_moe_decode_grouped_q8_0(
    device const uint8_t* weights             [[buffer(0)]],
    device const float*   x                   [[buffer(1)]],
    device const uint*    routes              [[buffer(2)]],
    device const uint*    tile_map            [[buffer(3)]],
    device       float*   output              [[buffer(4)]],
    constant     uint&    in_dim              [[buffer(5)]],
    constant     uint&    out_dim             [[buffer(6)]],
    constant     uint&    expert_bytes        [[buffer(7)]],
    constant     uint&    input_route_divisor [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    if (group.y >= tile_map[0]) return;
    const device Inkling2DecodeExpertTile* descs =
        (device const Inkling2DecodeExpertTile*)(tile_map + 1);
    const Inkling2DecodeExpertTile tile = descs[group.y];
    routes += tile.first_route;

    const uint row = group.x * 4 + (uint)simd;
    if (row >= out_dim) return;

    const uint nb = in_dim / Q8_0_QK;
    const device block_q8_0_56* expert =
        (device const block_q8_0_56*)(weights + (size_t)tile.expert * expert_bytes);
    const device block_q8_0_56* weight_row = expert + (size_t)row * nb;
    float sums[INKLING2_DECODE_ROUTE_CAPACITY] = {0.f, 0.f, 0.f, 0.f};

    for (uint ib = 0; ib < nb; ++ib) {
        const device block_q8_0_56* block = weight_row + ib;
        const float scale = (float)block->d;
        const float quant = (float)block->qs[lane];
        for (uint slot = 0; slot < tile.count; ++slot) {
            const uint route = routes[slot];
            const uint input_row = route / input_route_divisor;
            const float value = x[(size_t)input_row * in_dim + (size_t)ib * Q8_0_QK + lane];
            sums[slot] += scale * value * quant;
        }
    }

    for (uint slot = 0; slot < tile.count; ++slot) {
        const float total = simd_sum(sums[slot]);
        if (lane == 0) {
            const uint route = routes[slot];
            output[(size_t)route * out_dim + row] = total;
        }
    }
}

kernel void inkling2_resident_moe_decode_grouped_iq4_nl(
    device const uint8_t* weights             [[buffer(0)]],
    device const float*   x                   [[buffer(1)]],
    device const uint*    routes              [[buffer(2)]],
    device const uint*    tile_map            [[buffer(3)]],
    device       float*   output              [[buffer(4)]],
    constant     uint&    in_dim              [[buffer(5)]],
    constant     uint&    out_dim             [[buffer(6)]],
    constant     uint&    expert_bytes        [[buffer(7)]],
    constant     uint&    input_route_divisor [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    if (group.y >= tile_map[0]) return;
    const device Inkling2DecodeExpertTile* descs =
        (device const Inkling2DecodeExpertTile*)(tile_map + 1);
    const Inkling2DecodeExpertTile tile = descs[group.y];
    routes += tile.first_route;

    const uint first_row = group.x * 4 + (uint)simd * 2;
    if (first_row >= out_dim) return;
    const bool has_second = first_row + 1 < out_dim;

    const uint nb = in_dim / IQ4_NL_QK;
    const device block_iq4_nl_56* expert =
        (device const block_iq4_nl_56*)(weights + (size_t)tile.expert * expert_bytes);
    const device block_iq4_nl_56* first_weights = expert + (size_t)first_row * nb;
    const device block_iq4_nl_56* second_weights = has_second ? first_weights + nb : first_weights;
    size_t input_offsets[INKLING2_DECODE_ROUTE_CAPACITY];
    float first_sums[INKLING2_DECODE_ROUTE_CAPACITY] = {0.f, 0.f, 0.f, 0.f};
    float second_sums[INKLING2_DECODE_ROUTE_CAPACITY] = {0.f, 0.f, 0.f, 0.f};

    for (uint slot = 0; slot < tile.count; ++slot) {
        input_offsets[slot] = (size_t)(routes[slot] / input_route_divisor) * in_dim;
    }

    const ushort ix = lane / 2;
    const ushort it = lane & 1;
    // Share the activation loads for a pair without changing either dot
    // product's block order or sharing its FP32 accumulator.
    for (uint ib = ix; ib < nb; ib += 16) {
        const device block_iq4_nl_56* first_block = first_weights + ib;
        const device block_iq4_nl_56* second_block = second_weights + ib;
        const Inkling2DecodeIq4NlHalf first_half = inkling2_decode_iq4_nl_half(first_block, it);
        const Inkling2DecodeIq4NlHalf second_half =
            has_second ? inkling2_decode_iq4_nl_half(second_block, it) : first_half;
        const float first_scale = (float)first_block->d;
        const float second_scale = has_second ? (float)second_block->d : first_scale;
        const size_t block_offset = (size_t)ib * IQ4_NL_QK + it * 8;
        for (uint slot = 0; slot < tile.count; ++slot) {
            const device float4* y4 = (const device float4*)(x + input_offsets[slot] + block_offset);
            const float4 yl0 = y4[0];
            const float4 yl1 = y4[4];
            const float4 yl2 = y4[1];
            const float4 yl3 = y4[5];
            first_sums[slot] += first_scale * inkling2_decode_iq4_nl_dot(first_half, yl0, yl1, yl2, yl3);
            if (has_second) {
                second_sums[slot] += second_scale * inkling2_decode_iq4_nl_dot(second_half, yl0, yl1, yl2, yl3);
            }
        }
    }

    for (uint slot = 0; slot < tile.count; ++slot) {
        const uint route = routes[slot];
        const size_t output_index = (size_t)route * out_dim + first_row;
        const float first = simd_sum(first_sums[slot]);
        if (lane == 0) {
            output[output_index] = first;
        }
        if (has_second) {
            const float second = simd_sum(second_sums[slot]);
            if (lane == 0) {
                output[output_index + 1] = second;
            }
        }
    }
}

// Grouped gate/up decode shares each expert weight unpack across its at most
// four routes, while each SIMD lane visits the same K blocks as scalar GEMV.
kernel void inkling2_resident_moe_decode_grouped_situ_iq4_nl(
    device const uint8_t* weights   [[buffer(0)]],
    device const float*   x         [[buffer(1)]],
    device const uint*    routes    [[buffer(2)]],
    device const uint*    tile_map  [[buffer(3)]],
    device       float*   gated     [[buffer(4)]],
    constant     uint&    in_dim    [[buffer(5)]],
    constant     uint&    expert_dim [[buffer(6)]],
    constant     uint&    expert_bytes [[buffer(7)]],
    constant     uint&    top_k     [[buffer(8)]],
    constant     float&   beta      [[buffer(9)]],
    constant     float&   linear_beta [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    if (group.y >= tile_map[0]) return;
    const device Inkling2DecodeExpertTile* descs =
        (device const Inkling2DecodeExpertTile*)(tile_map + 1);
    const Inkling2DecodeExpertTile tile = descs[group.y];
    routes += tile.first_route;

    const uint column = group.x * 2 + (uint)simd;
    if (column >= expert_dim) return;

    const uint nb = in_dim / IQ4_NL_QK;
    const device block_iq4_nl_56* expert =
        (device const block_iq4_nl_56*)(weights + (size_t)tile.expert * expert_bytes);
    const device block_iq4_nl_56* gate_row = expert + (size_t)column * nb;
    const device block_iq4_nl_56* up_row = expert + (size_t)(column + expert_dim) * nb;
    uint input_rows[INKLING2_DECODE_ROUTE_CAPACITY] = {0, 0, 0, 0};
    float gate_sums[INKLING2_DECODE_ROUTE_CAPACITY] = {0.f, 0.f, 0.f, 0.f};
    float up_sums[INKLING2_DECODE_ROUTE_CAPACITY] = {0.f, 0.f, 0.f, 0.f};

    // Hoist route division out of the K-block loop.
    for (uint slot = 0; slot < tile.count; ++slot) {
        input_rows[slot] = routes[slot] / top_k;
    }

    const ushort ix = lane / 2;
    const ushort it = lane & 1;
    for (uint ib = ix; ib < nb; ib += 16) {
        const device block_iq4_nl_56* gate_block = gate_row + ib;
        const device block_iq4_nl_56* up_block = up_row + ib;
        const Inkling2DecodeIq4NlHalf gate_weights = inkling2_decode_iq4_nl_half(gate_block, it);
        const Inkling2DecodeIq4NlHalf up_weights = inkling2_decode_iq4_nl_half(up_block, it);

        for (uint slot = 0; slot < tile.count; ++slot) {
            const device float* yb = x + (size_t)input_rows[slot] * in_dim +
                (size_t)ib * IQ4_NL_QK + it * 8;
            const device float4* y4 = (device const float4*)yb;
            const float4 yl0 = y4[0];
            const float4 yl1 = y4[4];
            const float4 yl2 = y4[1];
            const float4 yl3 = y4[5];
            gate_sums[slot] += (float)gate_block->d *
                inkling2_decode_iq4_nl_dot(gate_weights, yl0, yl1, yl2, yl3);
            up_sums[slot] += (float)up_block->d *
                inkling2_decode_iq4_nl_dot(up_weights, yl0, yl1, yl2, yl3);
        }
    }

    for (uint slot = 0; slot < tile.count; ++slot) {
        const float gate = simd_sum(gate_sums[slot]);
        const float up = simd_sum(up_sums[slot]);
        if (lane == 0) {
            const uint route = routes[slot];
            gated[(size_t)route * expert_dim + column] =
                inkling2_resident_situ(gate, up, beta, linear_beta);
        }
    }
}
