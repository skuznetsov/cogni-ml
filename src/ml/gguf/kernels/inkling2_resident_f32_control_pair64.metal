#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Two F32 controller projections share a 64-row by 32-column rank panel.
// Each of eight SIMD groups computes one 32x16 output quadrant with eight
// FP32 MMA accumulators. Dispatch with 256 threads per threadgroup.
kernel void inkling2_resident_f32_control_pair64(
    device const uchar* model          [[buffer(0)]],
    device const float* rank           [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const float* message        [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant ulong* weight_offsets     [[buffer(5)]],
    constant uint&  in_dim             [[buffer(6)]],
    constant uint&  output_dim         [[buffer(7)]],
    constant uint&  rows               [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    constexpr uint Rank = 64;
    constexpr uint KTile = 32;
    constexpr uint RowTile = 64;
    constexpr uint ChannelTile = 32;
    constexpr uint Threads = 256;
    constexpr uint RankTileFloats = RowTile * KTile;
    constexpr uint WeightTileFloats = ChannelTile * KTile;
    constexpr uint StagingFloat4s = 1024;

    const uint first_channel = group.x * ChannelTile;
    const uint first_row = group.y * RowTile;
    if (in_dim != Rank || first_channel >= output_dim || first_row >= rows) {
        return;
    }

    const uint channel_count = min(output_dim - first_channel, ChannelTile);
    const uint row_count = min(rows - first_row, RowTile);
    const uint row_quadrant = (simd & 3) >> 1;
    const uint channel_quadrant = 16 * (simd & 1);
    const bool forget_projection = simd >= 4;

    device const float* gain_weights = (device const float*)(model + weight_offsets[0]);
    device const float* forget_weights = (device const float*)(model + weight_offsets[1]);

    // 8 KiB rank panel plus 4 KiB for each weight panel. The complete 16 KiB
    // staging area is overlaid with both 8 KiB result tiles after the last
    // MMA read has completed.
    threadgroup float4 staging[StagingFloat4s];
    threadgroup float* rank_tile = (threadgroup float*)staging;
    threadgroup float* gain_tile = rank_tile + RankTileFloats;
    threadgroup float* forget_tile = gain_tile + WeightTileFloats;

    simdgroup_float8x8 accum[8];
    #pragma unroll
    for (uint i = 0; i < 8; ++i) {
        accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_block = 0; k_block < Rank / KTile; ++k_block) {
        const uint k_base = k_block * KTile;

        // Scalar loads keep model offset requirements at ordinary F32
        // alignment, while adjacent lanes still fill contiguous tile values.
        for (uint i = tid; i < RankTileFloats; i += Threads) {
            const uint row = i / KTile;
            const uint col = i % KTile;
            rank_tile[i] = row < row_count
                ? rank[(size_t)(first_row + row) * Rank + k_base + col]
                : 0.0f;
        }
        for (uint i = tid; i < WeightTileFloats; i += Threads) {
            const uint channel = i / KTile;
            const uint col = i % KTile;
            const size_t weight_index =
                (size_t)(first_channel + channel) * Rank + k_base + col;
            gain_tile[i] = channel < channel_count ? gain_weights[weight_index] : 0.0f;
            forget_tile[i] = channel < channel_count ? forget_weights[weight_index] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* selected_weights = forget_projection ? forget_tile : gain_tile;
        const uint row_base = row_quadrant * 32;
        #pragma unroll
        for (uint k = 0; k < KTile; k += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1;
            simdgroup_load(a0, rank_tile + (row_base + 0) * KTile + k, KTile);
            simdgroup_load(a1, rank_tile + (row_base + 8) * KTile + k, KTile);
            simdgroup_load(a2, rank_tile + (row_base + 16) * KTile + k, KTile);
            simdgroup_load(a3, rank_tile + (row_base + 24) * KTile + k, KTile);
            simdgroup_load(b0, selected_weights + channel_quadrant * KTile + k,
                           KTile, 0, true);
            simdgroup_load(b1, selected_weights + (channel_quadrant + 8) * KTile + k,
                           KTile, 0, true);

            simdgroup_multiply_accumulate(accum[0], a0, b0, accum[0]);
            simdgroup_multiply_accumulate(accum[1], a0, b1, accum[1]);
            simdgroup_multiply_accumulate(accum[2], a1, b0, accum[2]);
            simdgroup_multiply_accumulate(accum[3], a1, b1, accum[3]);
            simdgroup_multiply_accumulate(accum[4], a2, b0, accum[4]);
            simdgroup_multiply_accumulate(accum[5], a2, b1, accum[5]);
            simdgroup_multiply_accumulate(accum[6], a3, b0, accum[6]);
            simdgroup_multiply_accumulate(accum[7], a3, b1, accum[7]);
        }

        // Protect the next K panel write, and on the last iteration the
        // staging overlay below, from every SIMD group's outstanding reads.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* result_storage = (threadgroup float*)staging;
    const uint result_projection_offset = forget_projection ? RankTileFloats : 0;
    threadgroup float* result_tile = result_storage + result_projection_offset;
    const uint row_base = row_quadrant * 32;
    #pragma unroll
    for (uint row_fragment = 0; row_fragment < 4; ++row_fragment) {
        const uint result_row = row_base + row_fragment * 8;
        threadgroup float* quadrant = result_tile + result_row * ChannelTile + channel_quadrant;
        simdgroup_store(accum[row_fragment * 2], quadrant, ChannelTile);
        simdgroup_store(accum[row_fragment * 2 + 1], quadrant + 8, ChannelTile);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < RankTileFloats; i += Threads) {
        const uint row = i / ChannelTile;
        const uint channel = i % ChannelTile;
        if (row < row_count && channel < channel_count) {
            const size_t output_index =
                (size_t)(first_row + row) * output_dim + first_channel + channel;
            const float gain_value = result_storage[(size_t)row * ChannelTile + channel];
            const float forget_value =
                result_storage[RankTileFloats + (size_t)row * ChannelTile + channel];
            const float source = input[output_index];
            const float forgotten = tanh(forget_value) * source;
            const float gained = tanh(gain_value) * message[output_index];
            output[output_index] = source - forgotten + gained;
        }
    }
}
