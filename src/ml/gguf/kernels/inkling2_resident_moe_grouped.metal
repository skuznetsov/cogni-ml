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
inline uint inkling2_group_routes_m16(
    device const uint* ids,
    device uint* routes,
    device uint* tile_map,
    uint rows, uint routed_rows, uint n_experts,
    threadgroup atomic_uint* counts,
    threadgroup atomic_uint* next_tile,
    ushort tid, uint threads)
{
    if (tid == 0) atomic_store_explicit(next_tile, 0, memory_order_relaxed);
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
            const uint base = atomic_fetch_add_explicit(next_tile, tiles, memory_order_relaxed);
            device Inkling2ExpertTile* descs = (device Inkling2ExpertTile*)(tile_map + 1);
            for (uint i = 0; i < tiles; ++i) {
                const uint start = i * INKLING2_EXPERT_TILE_ROUTES;
                descs[base + i] = {tid, tid * rows + start, min(INKLING2_EXPERT_TILE_ROUTES, count - start)};
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint count = atomic_load_explicit(next_tile, memory_order_relaxed);
        tile_map[0] = count;
        return count;
    }
    return 0;
}

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
    inkling2_group_routes_m16(ids, routes, tile_map, rows, routed_rows, n_experts, counts, &next_tile, tid, threads);
}

inline void inkling2_write_expert_grids(
    device uint* dispatch_args,
    uint fused_up_x_groups, uint down_x_groups, uint count)
{
    const uint y = max(1u, count);
    dispatch_args[0] = fused_up_x_groups;
    dispatch_args[1] = y;
    dispatch_args[2] = 1;
    dispatch_args[3] = down_x_groups;
    dispatch_args[4] = y;
    dispatch_args[5] = 1;
}

kernel void inkling2_resident_moe_group_routes_indirect(
    device const uint* ids                [[buffer(0)]],
    device       uint* routes             [[buffer(1)]],
    device       uint* tile_map           [[buffer(2)]],
    constant     uint& rows               [[buffer(3)]],
    constant     uint& routed_rows        [[buffer(4)]],
    constant     uint& n_experts          [[buffer(5)]],
    device       uint* dispatch_args     [[buffer(7)]],
    constant     uint& fused_up_x_groups [[buffer(8)]],
    constant     uint& down_x_groups     [[buffer(9)]],
    threadgroup  atomic_uint* counts      [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    threadgroup atomic_uint next_tile;
    const uint count = inkling2_group_routes_m16(
        ids, routes, tile_map, rows, routed_rows, n_experts, counts, &next_tile, tid, threads);
    if (tid == 0) {
        inkling2_write_expert_grids(dispatch_args, fused_up_x_groups, down_x_groups, count);
    }
}

// Bucket routes once. M64 takes full tiles and tails wider than M32.
// The remaining routes use the M32 and M16 tail layout.
template <bool UseM64>
kernel void inkling2_resident_moe_group_routes_split(
    device const uint* ids                [[buffer(0)]],
    device       uint* routes             [[buffer(1)]],
    device       uint* tile_map           [[buffer(2)]],
    constant     uint& rows               [[buffer(3)]],
    constant     uint& routed_rows        [[buffer(4)]],
    constant     uint& n_experts          [[buffer(5)]],
    device       uint* wide_tile_map      [[buffer(6)]],
    device       uint* dispatch_args     [[buffer(7)]],
    constant     uint& fused_up_x_groups [[buffer(8)]],
    constant     uint& down_x_groups     [[buffer(9)]],
    device       uint* m64_tile_map       [[buffer(10)]],
    threadgroup  atomic_uint* counts      [[threadgroup(0)]],
    ushort tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]])
{
    constexpr uint WIDE_ROUTES = 32;
    constexpr uint NARROW_ROUTES = 16;
    constexpr uint M64_ROUTES = 64;

    threadgroup atomic_uint next_narrow_tile;
    threadgroup atomic_uint next_wide_tile;
    threadgroup atomic_uint next_m64_tile;
    if (tid == 0) {
        atomic_store_explicit(&next_narrow_tile, 0, memory_order_relaxed);
        atomic_store_explicit(&next_wide_tile, 0, memory_order_relaxed);
        if (UseM64) atomic_store_explicit(&next_m64_tile, 0, memory_order_relaxed);
    }
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
            // Prefer one M64 tile over two narrower tiles for a tail above 32.
            const uint m64_tiles = UseM64 ? (count + WIDE_ROUTES - 1) / M64_ROUTES : 0;
            const uint consumed = min(count, m64_tiles * M64_ROUTES);
            if (UseM64 && m64_tiles != 0) {
                const uint base = atomic_fetch_add_explicit(&next_m64_tile, m64_tiles, memory_order_relaxed);
                device Inkling2ExpertTile* descs = (device Inkling2ExpertTile*)(m64_tile_map + 1);
                for (uint i = 0; i < m64_tiles; ++i) {
                    const uint start = i * M64_ROUTES;
                    descs[base + i] = {tid, tid * rows + start, min(M64_ROUTES, count - start)};
                }
            }

            const uint remaining = count - consumed;
            const uint full_wide_tiles = remaining / WIDE_ROUTES;
            const uint tail = remaining & (WIDE_ROUTES - 1);
            const bool tail_is_wide = tail > NARROW_ROUTES;
            const bool has_narrow_tail = tail != 0 && !tail_is_wide;
            const uint expert_wide_tiles = full_wide_tiles + (tail_is_wide ? 1 : 0);

            uint narrow_base = 0;
            if (has_narrow_tail) {
                narrow_base = atomic_fetch_add_explicit(&next_narrow_tile, 1, memory_order_relaxed);
            }
            uint wide_base = 0;
            if (expert_wide_tiles != 0) {
                wide_base = atomic_fetch_add_explicit(&next_wide_tile, expert_wide_tiles, memory_order_relaxed);
            }

            device Inkling2ExpertTile* narrow_descs = (device Inkling2ExpertTile*)(tile_map + 1);
            device Inkling2ExpertTile* wide_descs = (device Inkling2ExpertTile*)(wide_tile_map + 1);
            for (uint i = 0; i < full_wide_tiles; ++i) {
                const uint start = consumed + i * WIDE_ROUTES;
                wide_descs[wide_base + i] = {tid, tid * rows + start, WIDE_ROUTES};
            }
            if (tail_is_wide) {
                const uint start = consumed + full_wide_tiles * WIDE_ROUTES;
                wide_descs[wide_base + full_wide_tiles] = {tid, tid * rows + start, tail};
            } else if (has_narrow_tail) {
                const uint start = consumed + full_wide_tiles * WIDE_ROUTES;
                narrow_descs[narrow_base] = {tid, tid * rows + start, tail};
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const uint narrow_count = atomic_load_explicit(&next_narrow_tile, memory_order_relaxed);
        const uint wide_count = atomic_load_explicit(&next_wide_tile, memory_order_relaxed);
        tile_map[0] = narrow_count;
        wide_tile_map[0] = wide_count;

        inkling2_write_expert_grids(dispatch_args, fused_up_x_groups, down_x_groups, narrow_count);
        inkling2_write_expert_grids(dispatch_args + 6, fused_up_x_groups, down_x_groups, wide_count);
        if (UseM64) {
            const uint m64_count = atomic_load_explicit(&next_m64_tile, memory_order_relaxed);
            m64_tile_map[0] = m64_count;
            inkling2_write_expert_grids(dispatch_args + 12, fused_up_x_groups, down_x_groups, m64_count);
        }
    }
}

typedef decltype(inkling2_resident_moe_group_routes_split<false>) inkling2_resident_moe_group_routes_split_t;
template [[host_name("inkling2_resident_moe_group_routes_split")]] kernel inkling2_resident_moe_group_routes_split_t inkling2_resident_moe_group_routes_split<false>;
template [[host_name("inkling2_resident_moe_group_routes_split_m64")]] kernel inkling2_resident_moe_group_routes_split_t inkling2_resident_moe_group_routes_split<true>;

inline float inkling2_expert_quant(device const block_q8_0_56& block, uint column) {
    return (float)block.qs[column];
}

inline float inkling2_expert_quant(device const block_iq4_nl_56& block, uint column) {
    const uint packed = block.qs[column & 15];
    const uint code = (packed >> ((column / 16) * 4)) & 15;
    return kvalues_iq4nl_f[code];
}

// Four SIMD groups each own 16 adjacent output columns. M64 adds a second
// set of four groups for routes 32–63, sharing the dequantized weights.
template <typename Block, bool FuseSiTU, bool BlockedWeights = false, uint RouteRows = 16>
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
    constexpr uint FRAGMENT = 8;
    constexpr uint FRAGMENT_SIZE = FRAGMENT * FRAGMENT;
    constexpr bool M64 = RouteRows == 64;
    constexpr uint THREADS = M64 ? 256 : 128;
    constexpr uint WEIGHT_LOADERS = THREADS / OUT;
    constexpr uint WEIGHT_COLUMNS = K / WEIGHT_LOADERS;
    constexpr uint FIRST_LOAD_ROWS = THREADS / 8;
    const uint width = out_dim / 2;
    const uint first_out = group.x * (FuseSiTU ? PAIRS : OUT);
    const uint out_rows = min(out_dim - first_out, OUT);
    const uint k_blocks = in_dim / K;

    // Each weight row is unpacked once, by two or four threads.
    const uint weight_row = tid / WEIGHT_LOADERS;
    const uint weight_column = (tid % WEIGHT_LOADERS) * WEIGHT_COLUMNS;
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
    const bool valid_route = M64 || route_row < tile.count;
    const uint input_row = valid_route ? routes[route_row] / input_route_divisor : 0;
    device const float* x_row = x + (size_t)input_row * in_dim + input_col;
    const uint second_route_row = route_row + FIRST_LOAD_ROWS;
    const bool valid_second_route = M64 || (RouteRows > FIRST_LOAD_ROWS && second_route_row < tile.count);
    // Pad partial M64 tiles with the last valid route. The scatter writes
    // only actual routes and every SIMD group keeps the same MMA loop.
    const uint second_input_row = valid_second_route
        ? routes[M64 ? min(second_route_row, tile.count - 1) : second_route_row] / input_route_divisor : 0;
    device const float* second_x_row = x + (size_t)second_input_row * in_dim + input_col;
    const bool has_upper = M64 || tile.count > 8;
    const bool has_third = M64 || (RouteRows > INKLING2_EXPERT_TILE_ROUTES && tile.count > 16);
    const bool has_fourth = M64 || (RouteRows > INKLING2_EXPERT_TILE_ROUTES && tile.count > 24);

    threadgroup float4 staging[(OUT + RouteRows) * K / 4];
    threadgroup float* weight_tile = (threadgroup float*)staging;
    threadgroup float4* input_vectors = staging + OUT * K / 4;
    const threadgroup float* input_tile = (const threadgroup float*)input_vectors;
    // IQ4_NL unpack writes the transposed 8x8 fragments directly. The MMA
    // loads each fragment at stride eight without gathering strided rows.
    const uint weight_fragment = (weight_column / FRAGMENT) * (OUT / FRAGMENT) + weight_row / FRAGMENT;
    threadgroup float* weight_dest = BlockedWeights
        ? weight_tile + weight_fragment * FRAGMENT_SIZE + weight_row % FRAGMENT
        : weight_tile + weight_row * K + weight_column;

    simdgroup_float8x8 low0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 low1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 high0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 high1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 third0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 third1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 fourth0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 fourth1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

    const uint first_simd_out = (M64 ? simd % 4 : simd) * 16;
    const uint first_simd_route = M64 ? (simd / 4) * 32 : 0;
    const threadgroup float* simd_inputs = input_tile + first_simd_route * K;
    for (uint block = 0; block < k_blocks; ++block) {
        const device Block& packed = row_weights[block];
        const float scale = (float)packed.d;
        #pragma unroll
        for (uint i = 0; i < WEIGHT_COLUMNS; ++i) {
            const uint index = BlockedWeights ? (i / FRAGMENT) * OUT * FRAGMENT + (i % FRAGMENT) * FRAGMENT : i;
            weight_dest[index] = scale * inkling2_expert_quant(packed, weight_column + i);
        }
        input_vectors[tid] = valid_route
            ? float4(*(device const packed_float4*)(x_row + block * K)) : float4(0.0f);
        if (has_third) {
            input_vectors[tid + FIRST_LOAD_ROWS * K / 4] = valid_second_route
                ? float4(*(device const packed_float4*)(second_x_row + block * K)) : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < K; k += FRAGMENT) {
            simdgroup_float8x8 a, b0, b1;
            simdgroup_load(a, simd_inputs + k, K);
            if (BlockedWeights) {
                const uint fragment_index = (k / FRAGMENT) * (OUT / FRAGMENT) + first_simd_out / FRAGMENT;
                const threadgroup float* weights_k = weight_tile + fragment_index * FRAGMENT_SIZE;
                simdgroup_load(b0, weights_k, FRAGMENT);
                simdgroup_load(b1, weights_k + FRAGMENT_SIZE, FRAGMENT);
            } else {
                simdgroup_load(b0, weight_tile + first_simd_out * K + k, K, 0, true);
                simdgroup_load(b1, weight_tile + (first_simd_out + FRAGMENT) * K + k, K, 0, true);
            }
            simdgroup_multiply_accumulate(low0, a, b0, low0);
            simdgroup_multiply_accumulate(low1, a, b1, low1);
            if (has_upper) {
                simdgroup_load(a, simd_inputs + 8 * K + k, K);
                simdgroup_multiply_accumulate(high0, a, b0, high0);
                simdgroup_multiply_accumulate(high1, a, b1, high1);
            }
            if (has_third) {
                simdgroup_load(a, simd_inputs + 16 * K + k, K);
                simdgroup_multiply_accumulate(third0, a, b0, third0);
                simdgroup_multiply_accumulate(third1, a, b1, third1);
                if (has_fourth) {
                    simdgroup_load(a, simd_inputs + 24 * K + k, K);
                    simdgroup_multiply_accumulate(fourth0, a, b0, fourth0);
                    simdgroup_multiply_accumulate(fourth1, a, b1, fourth1);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // After the final barrier, reuse staging for the epilogue and scatter.
    threadgroup float* results = (threadgroup float*)staging;
    threadgroup float* simd_result = results + first_simd_route * OUT + first_simd_out;
    simdgroup_store(low0, simd_result, OUT);
    simdgroup_store(low1, simd_result + 8, OUT);
    if (has_upper) {
        simdgroup_store(high0, simd_result + 8 * OUT, OUT);
        simdgroup_store(high1, simd_result + 8 * OUT + 8, OUT);
    }
    if (has_third) {
        simdgroup_store(third0, simd_result + 16 * OUT, OUT);
        simdgroup_store(third1, simd_result + 16 * OUT + 8, OUT);
    }
    if (has_fourth) {
        simdgroup_store(fourth0, simd_result + 24 * OUT, OUT);
        simdgroup_store(fourth1, simd_result + 24 * OUT + 8, OUT);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (FuseSiTU) {
        for (uint i = tid; i < tile.count * PAIRS; i += THREADS) {
            const uint row = i / PAIRS;
            const uint col = i % PAIRS;
            const uint offset = row * OUT + col;
            output[(size_t)routes[row] * width + first_out + col] =
                inkling2_resident_situ(results[offset], results[offset + PAIRS], beta, linear_beta);
        }
    } else {
        for (uint i = tid; i < tile.count * OUT; i += THREADS) {
            const uint col = i % OUT;
            if (col < out_rows) {
                const uint route = routes[i / OUT];
                output[(size_t)route * out_dim + first_out + col] = results[i];
            }
        }
    }
}

typedef decltype(inkling2_resident_moe_gemm<block_q8_0_56, false>) inkling2_resident_moe_gemm_t;
typedef decltype(inkling2_resident_moe_gemm<block_q8_0_56, false, false, 32>) inkling2_resident_moe_gemm_m32_t;
template [[host_name("inkling2_resident_moe_gemm_q8_0")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_q8_0_56, false>;
template [[host_name("inkling2_resident_moe_gemm_iq4_nl")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_iq4_nl_56, false, true>;
template [[host_name("inkling2_resident_moe_gemm_situ_q8_0")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_q8_0_56, true>;
template [[host_name("inkling2_resident_moe_gemm_situ_iq4_nl")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_iq4_nl_56, true, true>;
template [[host_name("inkling2_resident_moe_gemm_q8_0_m32")]] kernel inkling2_resident_moe_gemm_m32_t inkling2_resident_moe_gemm<block_q8_0_56, false, false, 32>;
template [[host_name("inkling2_resident_moe_gemm_iq4_nl_m32")]] kernel inkling2_resident_moe_gemm_m32_t inkling2_resident_moe_gemm<block_iq4_nl_56, false, true, 32>;
template [[host_name("inkling2_resident_moe_gemm_situ_q8_0_m32")]] kernel inkling2_resident_moe_gemm_m32_t inkling2_resident_moe_gemm<block_q8_0_56, true, false, 32>;
template [[host_name("inkling2_resident_moe_gemm_situ_iq4_nl_m32")]] kernel inkling2_resident_moe_gemm_m32_t inkling2_resident_moe_gemm<block_iq4_nl_56, true, true, 32>;
template [[host_name("inkling2_resident_moe_gemm_q8_0_m64")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_q8_0_56, false, false, 64>;
template [[host_name("inkling2_resident_moe_gemm_iq4_nl_m64")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_iq4_nl_56, false, true, 64>;
template [[host_name("inkling2_resident_moe_gemm_situ_q8_0_m64")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_q8_0_56, true, false, 64>;
template [[host_name("inkling2_resident_moe_gemm_situ_iq4_nl_m64")]] kernel inkling2_resident_moe_gemm_t inkling2_resident_moe_gemm<block_iq4_nl_56, true, true, 64>;
