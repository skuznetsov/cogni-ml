#include <metal_simdgroup_matrix>

constant uint INKLING2_EXPERT_TILE_ROUTES = 16;

struct Inkling2ExpertTile {
    uint expert;
    uint first_route;
    uint count;
};

// Bucket each flattened (token, rank) route by expert. The router selects an
// expert at most once per token, so each bucket has room for all of its hits.
// Keep the original route index for the gather and scatter in both GEMMs.
kernel void inkling2_resident_moe_group_routes(
    device const uint* ids           [[buffer(0)]],
    device       uint* routes        [[buffer(1)]],
    device       uint* tile_map      [[buffer(2)]],
    constant     uint& rows          [[buffer(3)]],
    constant     uint& routed_rows   [[buffer(4)]],
    constant     uint& n_experts     [[buffer(5)]],
    threadgroup  atomic_uint* counts [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    threadgroup atomic_uint next_tile;
    if (tid == 0) atomic_store_explicit(&next_tile, 0, memory_order_relaxed);
    if (tid < n_experts) atomic_store_explicit(counts + tid, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint route = tid; route < routed_rows; route += threads) {
        const uint expert = ids[route];
        const uint slot = atomic_fetch_add_explicit(counts + expert, 1, memory_order_relaxed);
        routes[expert * rows + slot] = route;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < n_experts) {
        const uint count = atomic_load_explicit(counts + tid, memory_order_relaxed);
        if (count != 0) {
            const uint tiles = (count + INKLING2_EXPERT_TILE_ROUTES - 1) / INKLING2_EXPERT_TILE_ROUTES;
            const uint base = atomic_fetch_add_explicit(&next_tile, tiles, memory_order_relaxed);
            device Inkling2ExpertTile* descs = (device Inkling2ExpertTile*)(tile_map + 1);
            for (uint i = 0; i < tiles; ++i) {
                const uint start = i * INKLING2_EXPERT_TILE_ROUTES;
                descs[base + i] = {tid, tid * rows + start, min(INKLING2_EXPERT_TILE_ROUTES, count - start)};
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) tile_map[0] = atomic_load_explicit(&next_tile, memory_order_relaxed);
}

inline float inkling2_expert_quant(device const block_q8_0_56& block, uint column) {
    return (float)block.qs[column];
}

inline float inkling2_expert_quant(device const block_iq4_nl_56& block, uint column) {
    const uint packed = block.qs[column & 15];
    const uint code = (packed >> ((column / 16) * 4)) & 15;
    return kvalues_iq4nl_f[code];
}

// Four SIMD groups each own 16 adjacent output columns. Stage each block of
// quantized weights once per tile, including when there are more than eight
// routes. The second group of eight input rows is skipped for sparse tiles.
template <typename Block, bool FuseSiTU>
kernel void inkling2_resident_moe_gemm(
    device const uint8_t* weights             [[buffer(0)]],
    device const float*   x                   [[buffer(1)]],
    device const uint*    routes              [[buffer(2)]],
    device const uint*    tile_map            [[buffer(3)]],
    device       float*   output              [[buffer(4)]],
    constant     uint&    in_dim              [[buffer(5)]],
    constant     uint&    out_dim             [[buffer(6)]],
    constant     uint&    expert_bytes        [[buffer(7)]],
    constant     uint&    input_route_divisor [[buffer(8)]],
    constant     float&   beta                [[buffer(9)]],
    constant     float&   linear_beta         [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    if (group.y >= tile_map[0]) return;
    const device Inkling2ExpertTile* descs = (device const Inkling2ExpertTile*)(tile_map + 1);
    const Inkling2ExpertTile tile = descs[group.y];
    routes += tile.first_route;

    constexpr uint OUT = 64;
    constexpr uint K = 32;
    constexpr uint PAIRS = OUT / 2;
    const uint width = out_dim / 2;
    const uint first_out = group.x * (FuseSiTU ? PAIRS : OUT);
    const uint out_rows = min(out_dim - first_out, OUT);
    const uint k_blocks = in_dim / K;

    // Two threads unpack the lower and upper halves of each weight row.
    const uint weight_row = tid / 2;
    const uint weight_column = (tid & 1) * 16;
    device const Block* expert = (device const Block*)(weights + (size_t)tile.expert * expert_bytes);
    // A fused tile loads matching rows from the two halves of gate/up.
    // The host admits widths divisible by PAIRS via the quantized down input.
    const uint model_row = FuseSiTU
        ? first_out + (weight_row % PAIRS) + (weight_row / PAIRS) * width
        : first_out + min(weight_row, out_rows - 1);
    device const Block* row_weights = expert + (size_t)model_row * k_blocks;

    // Eight threads load the 32 activation values for one route.
    const uint route_row = tid / 8;
    const uint input_col = (tid % 8) * 4;
    const bool valid_route = route_row < tile.count;
    const uint input_row = valid_route ? routes[route_row] / input_route_divisor : 0;
    device const float* x_row = x + (size_t)input_row * in_dim + input_col;
    const bool has_upper = tile.count > 8;

    threadgroup float weight_tile[OUT * K];
    threadgroup float4 input_vectors[INKLING2_EXPERT_TILE_ROUTES * K / 4];
    const threadgroup float* input_tile = (const threadgroup float*)input_vectors;
    threadgroup float* weight_dest = weight_tile + weight_row * K + weight_column;

    simdgroup_float8x8 low0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 low1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 high0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 high1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const uint first_simd_out = simd * 16;
    for (uint block = 0; block < k_blocks; ++block) {
        const device Block& packed = row_weights[block];
        const float scale = (float)packed.d;
        #pragma unroll
        for (uint i = 0; i < 16; ++i) {
            weight_dest[i] = scale * inkling2_expert_quant(packed, weight_column + i);
        }
        input_vectors[tid] = valid_route
            ? float4(*(device const packed_float4*)(x_row + block * K)) : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < K; k += 8) {
            simdgroup_float8x8 a, b0, b1;
            simdgroup_load(a, input_tile + k, K);
            simdgroup_load(b0, weight_tile + first_simd_out * K + k, K, 0, true);
            simdgroup_load(b1, weight_tile + (first_simd_out + 8) * K + k, K, 0, true);
            simdgroup_multiply_accumulate(low0, a, b0, low0);
            simdgroup_multiply_accumulate(low1, a, b1, low1);
            if (has_upper) {
                simdgroup_load(a, input_tile + 8 * K + k, K);
                simdgroup_multiply_accumulate(high0, a, b0, high0);
                simdgroup_multiply_accumulate(high1, a, b1, high1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reuse the weight tile for the result, then scatter by original route.
    simdgroup_store(low0, weight_tile + first_simd_out, OUT);
    simdgroup_store(low1, weight_tile + first_simd_out + 8, OUT);
    if (has_upper) {
        simdgroup_store(high0, weight_tile + 8 * OUT + first_simd_out, OUT);
        simdgroup_store(high1, weight_tile + 8 * OUT + first_simd_out + 8, OUT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (FuseSiTU) {
        for (uint i = tid; i < tile.count * PAIRS; i += 128) {
            const uint row = i / PAIRS;
            const uint col = i % PAIRS;
            const uint offset = row * OUT + col;
            output[(size_t)routes[row] * width + first_out + col] =
                inkling2_resident_situ(weight_tile[offset], weight_tile[offset + PAIRS], beta, linear_beta);
        }
    } else {
        for (uint i = tid; i < tile.count * OUT; i += 128) {
            const uint col = i % OUT;
            if (col < out_rows) {
                const uint route = routes[i / OUT];
                output[(size_t)route * out_dim + first_out + col] = weight_tile[i];
            }
        }
    }
}

typedef decltype(inkling2_resident_moe_gemm<block_q8_0_56, false>) inkling2_resident_moe_gemm_t;
template [[host_name("inkling2_resident_moe_gemm_q8_0")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_q8_0_56, false>;
template [[host_name("inkling2_resident_moe_gemm_iq4_nl")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_iq4_nl_56, false>;
template [[host_name("inkling2_resident_moe_gemm_situ_q8_0")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_q8_0_56, true>;
template [[host_name("inkling2_resident_moe_gemm_situ_iq4_nl")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_iq4_nl_56, true>;
