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

    const uint nb = in_dim / IQ4_NL_QK;
    const uint row_bytes = nb * 18;
    const device block_iq4_nl_56* expert =
        (device const block_iq4_nl_56*)(weights + (size_t)tile.expert * expert_bytes);
    float sums[INKLING2_DECODE_ROUTE_CAPACITY][2] = {
        {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f}, {0.f, 0.f},
    };

    const ushort ix = lane / 2;
    const ushort it = lane & 1;
    for (uint ib = ix; ib < nb; ib += 16) {
        for (short row = 0; row < 2; ++row) {
            const uint row_id = first_row + row;
            if (row_id >= out_dim) continue;

            const device block_iq4_nl_56* block =
                (device const block_iq4_nl_56*)(weights + (size_t)tile.expert * expert_bytes +
                    (size_t)row_id * row_bytes) + ib;
            const device uint16_t* q4 = (device const uint16_t*)(block->qs + 8 * it);
            uint32_t aux32[2];
            thread const uint8_t* q8 = (thread const uint8_t*)aux32;

            aux32[0] = q4[0] | ((uint32_t)q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            const float4 qf10 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            const float4 qf11 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};

            aux32[0] = q4[2] | ((uint32_t)q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            const float4 qf20 = {kvalues_iq4nl_f[q8[0]], kvalues_iq4nl_f[q8[1]], kvalues_iq4nl_f[q8[2]], kvalues_iq4nl_f[q8[3]]};
            const float4 qf21 = {kvalues_iq4nl_f[q8[4]], kvalues_iq4nl_f[q8[5]], kvalues_iq4nl_f[q8[6]], kvalues_iq4nl_f[q8[7]]};

            for (uint slot = 0; slot < tile.count; ++slot) {
                const uint route = routes[slot];
                const uint input_row = route / input_route_divisor;
                const device float* yb = x + (size_t)input_row * in_dim +
                    (size_t)ib * IQ4_NL_QK + it * 8;
                const device float4* y4 = (const device float4*)yb;
                const float4 yl0 = y4[0];
                const float4 yl1 = y4[4];
                const float4 yl2 = y4[1];
                const float4 yl3 = y4[5];
                const float4 acc = yl0 * qf10 + yl1 * qf11 + yl2 * qf20 + yl3 * qf21;
                const float dot = acc[0] + acc[1] + acc[2] + acc[3];
                sums[slot][row] += (float)block->d * dot;
            }
        }
    }

    for (uint slot = 0; slot < tile.count; ++slot) {
        const uint route = routes[slot];
        for (short row = 0; row < 2 && first_row + row < out_dim; ++row) {
            const float total = simd_sum(sums[slot][row]);
            if (lane == 0) {
                output[(size_t)route * out_dim + first_row + row] = total;
            }
        }
    }
}
