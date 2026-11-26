#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

constant constexpr uint INKLING2_CONTROLLER_DOWN_MMA_TILE = 32;
constant constexpr uint INKLING2_CONTROLLER_DOWN_MMA_VECTORS =
    INKLING2_CONTROLLER_DOWN_MMA_TILE * INKLING2_CONTROLLER_DOWN_MMA_TILE / 4;

// One 128-thread threadgroup computes a 32x32 output tile. Each SIMD group
// owns one 16x16 quadrant and uses four FP32 MMA accumulators for its 8x8
// fragments. Staging uses scalar loads so arbitrary input-width tails and
// four-byte-aligned resident views remain safe.
kernel void inkling2_resident_f32_controller_down_mma(
    device const float* weights [[buffer(0)]],
    device const float* x       [[buffer(1)]],
    device       float* output  [[buffer(2)]],
    constant     uint& in_dim   [[buffer(3)]],
    constant     uint& out_dim  [[buffer(4)]],
    constant     uint& batch    [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TILE = INKLING2_CONTROLLER_DOWN_MMA_TILE;
    const uint first_out = group.x * TILE;
    const uint first_in = group.y * TILE;
    const uint out_rows = min(out_dim - first_out, TILE);
    const uint in_rows = min(batch - first_in, TILE);
    const uint out_quadrant = 16 * (simd & 1);
    const uint in_quadrant = 16 * (simd >> 1);

    threadgroup float4 weight_vectors[INKLING2_CONTROLLER_DOWN_MMA_VECTORS];
    threadgroup float4 input_vectors[INKLING2_CONTROLLER_DOWN_MMA_VECTORS];
    threadgroup float* weight_tile = (threadgroup float*)weight_vectors;
    threadgroup float* input_tile = (threadgroup float*)input_vectors;

    // Four threads stage eight adjacent values from one matrix row.
    const uint load_row = tid / 4;
    const uint load_col = (tid % 4) * 8;
    const bool weight_valid = load_row < out_rows;
    const bool input_valid = load_row < in_rows;
    const uint safe_weight_row = first_out + min(load_row, out_rows - 1);
    const uint safe_input_row = first_in + min(load_row, in_rows - 1);
    const device float* weight_row = weights + (size_t)safe_weight_row * in_dim;
    const device float* input_row = x + (size_t)safe_input_row * in_dim;

    simdgroup_float8x8 accum[4];
    for (uint i = 0; i < 4; ++i) {
        accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_base = 0; k_base < in_dim; k_base += TILE) {
        const uint col = k_base + load_col;
        if ((in_dim & 3) == 0 && col + 7 < in_dim) {
            const device packed_float4* weight_values =
                (const device packed_float4*)(weight_row + col);
            const device packed_float4* input_values =
                (const device packed_float4*)(input_row + col);
            weight_vectors[2 * tid] = weight_valid ? float4(weight_values[0]) : float4(0.0f);
            weight_vectors[2 * tid + 1] = weight_valid ? float4(weight_values[1]) : float4(0.0f);
            input_vectors[2 * tid] = input_valid ? float4(input_values[0]) : float4(0.0f);
            input_vectors[2 * tid + 1] = input_valid ? float4(input_values[1]) : float4(0.0f);
        } else {
            #pragma unroll
            for (uint value = 0; value < 8; ++value) {
                const uint value_col = col + value;
                const uint staged = load_row * TILE + load_col + value;
                weight_tile[staged] = weight_valid && value_col < in_dim ? weight_row[value_col] : 0.0f;
                input_tile[staged] = input_valid && value_col < in_dim ? input_row[value_col] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint k = 0; k < TILE; k += 8) {
            simdgroup_float8x8 a0, a1, b0, b1;
            simdgroup_load(a0, input_tile + in_quadrant * TILE + k, TILE);
            simdgroup_load(a1, input_tile + (in_quadrant + 8) * TILE + k, TILE);
            simdgroup_load(b0, weight_tile + out_quadrant * TILE + k, TILE, 0, true);
            simdgroup_load(b1, weight_tile + (out_quadrant + 8) * TILE + k, TILE, 0, true);
            simdgroup_multiply_accumulate(accum[0], a0, b0, accum[0]);
            simdgroup_multiply_accumulate(accum[1], a0, b1, accum[1]);
            simdgroup_multiply_accumulate(accum[2], a1, b0, accum[2]);
            simdgroup_multiply_accumulate(accum[3], a1, b1, accum[3]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The last panel has been consumed; reuse the input staging area for the
    // result tile, then store only the in-bounds rows and output columns.
    threadgroup float* result_tile = (threadgroup float*)input_vectors;
    threadgroup float* quadrant = result_tile + in_quadrant * TILE + out_quadrant;
    simdgroup_store(accum[0], quadrant, TILE);
    simdgroup_store(accum[1], quadrant + 8, TILE);
    simdgroup_store(accum[2], quadrant + 8 * TILE, TILE);
    simdgroup_store(accum[3], quadrant + 8 * TILE + 8, TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < in_rows * out_rows; i += 128) {
        const uint row = i / out_rows;
        const uint col = i % out_rows;
        output[(size_t)(first_in + row) * out_dim + first_out + col] =
            result_tile[row * TILE + col];
    }
}
