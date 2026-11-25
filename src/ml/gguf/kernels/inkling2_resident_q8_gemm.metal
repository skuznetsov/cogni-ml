#include <metal_simdgroup_matrix>

// One SIMD group handles a 16-by-16 quadrant of the 32-by-32 output tile.
// Weights and inputs are staged as F32, with one Q8_0 block per weight row.
constant constexpr uint INKLING2_Q8_MATRIX_TILE = 32;
constant constexpr uint INKLING2_Q8_MATRIX_VECTORS = INKLING2_Q8_MATRIX_TILE * INKLING2_Q8_MATRIX_TILE / 4;

inline void inkling2_resident_q8_matrix_tile(
    device const block_q8_0_56* weights,
    device const float* x,
    device float* output,
    uint in_dim, uint out_dim, uint batch, uint3 group, ushort tid, ushort simd,
    threadgroup float4* weight_vectors, threadgroup float4* input_vectors,
    threadgroup float4* result_vectors, bool stage_input)
{
    constexpr uint TILE = INKLING2_Q8_MATRIX_TILE;
    const uint first_out = group.x * TILE;
    const uint first_in = group.y * TILE;
    const uint out_rows = min(out_dim - first_out, TILE);
    const uint in_rows = min(batch - first_in, TILE);
    const uint out_quadrant = 16 * (simd & 1);
    const uint in_quadrant = 16 * (simd >> 1);
    const uint blocks = in_dim / TILE;

    threadgroup float* weight_tile = (threadgroup float*)weight_vectors;
    threadgroup float* input_tile = (threadgroup float*)input_vectors;
    // Four threads stage eight adjacent values each from a weight and input row.
    const uint load_row = tid / 4;
    const uint load_col = (tid % 4) * 8;
    const bool weight_valid = load_row < out_rows;
    const bool input_valid = load_row < in_rows;
    const device block_q8_0_56* weight_row =
        weights + (size_t)(first_out + min(load_row, out_rows - 1)) * blocks;
    const device float* input_row =
        x + (size_t)(first_in + min(load_row, in_rows - 1)) * in_dim + load_col;
    simdgroup_float8x8 accum[4];
    for (uint i = 0; i < 4; ++i) {
        accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint block = 0; block < blocks; ++block) {
        if (weight_valid) {
            const device block_q8_0_56& packed = weight_row[block];
            const float scale = (float)packed.d;
            const device packed_char4* codes = (const device packed_char4*)(packed.qs + load_col);
            weight_vectors[2 * tid] = scale * float4((char4)codes[0]);
            weight_vectors[2 * tid + 1] = scale * float4((char4)codes[1]);
        } else {
            weight_vectors[2 * tid] = float4(0.0f);
            weight_vectors[2 * tid + 1] = float4(0.0f);
        }
        if (stage_input) {
            const device packed_float4* values = (const device packed_float4*)(input_row + block * TILE);
            input_vectors[2 * tid] = input_valid ? float4(values[0]) : float4(0.0f);
            input_vectors[2 * tid + 1] = input_valid ? float4(values[1]) : float4(0.0f);
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

    threadgroup float* result_tile = (threadgroup float*)result_vectors;
    threadgroup float* quadrant = result_tile + in_quadrant * TILE + out_quadrant;
    simdgroup_store(accum[0], quadrant, TILE);
    simdgroup_store(accum[1], quadrant + 8, TILE);
    simdgroup_store(accum[2], quadrant + 8 * TILE, TILE);
    simdgroup_store(accum[3], quadrant + 8 * TILE + 8, TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < in_rows * out_rows; i += 128) {
        const uint row = i / out_rows;
        const uint col = i % out_rows;
        output[(size_t)(first_in + row) * out_dim + first_out + col] = result_tile[row * TILE + col];
    }
}

kernel void inkling2_resident_gemm_q8_0_simd(
    device const block_q8_0_56* weights [[buffer(0)]],
    device const float* x              [[buffer(1)]],
    device       float* output         [[buffer(2)]],
    constant     uint&  in_dim         [[buffer(3)]],
    constant     uint&  out_dim        [[buffer(4)]],
    constant     uint&  batch          [[buffer(5)]],
    uint3  group [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort simd  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float4 weight_vectors[INKLING2_Q8_MATRIX_VECTORS];
    threadgroup float4 input_vectors[INKLING2_Q8_MATRIX_VECTORS];
    inkling2_resident_q8_matrix_tile(weights, x, output, in_dim, out_dim, batch, group, tid, simd,
                                    weight_vectors, input_vectors, input_vectors, true);
}

// Four SIMD groups per projection keep the original FP32 tile layout. The
// first four stage the shared activation panel for all eight groups.
kernel void inkling2_resident_gemm_q8_0_simd_pair(
    device const block_q8_0_56* first_weights  [[buffer(0)]],
    device const block_q8_0_56* second_weights [[buffer(1)]],
    device const float* x                     [[buffer(2)]],
    device       float* first_output          [[buffer(3)]],
    device       float* second_output         [[buffer(4)]],
    constant     uint&  in_dim                [[buffer(5)]],
    constant     uint&  out_dim               [[buffer(6)]],
    constant     uint&  batch                 [[buffer(7)]],
    uint3  group [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort simd  [[simdgroup_index_in_threadgroup]])
{
    threadgroup float4 weight_vectors[2 * INKLING2_Q8_MATRIX_VECTORS];
    threadgroup float4 input_vectors[INKLING2_Q8_MATRIX_VECTORS];
    const bool second = simd >= 4;
    threadgroup float4* tile = weight_vectors + (second ? INKLING2_Q8_MATRIX_VECTORS : 0);
    inkling2_resident_q8_matrix_tile(second ? second_weights : first_weights, x,
                                    second ? second_output : first_output, in_dim, out_dim, batch,
                                    group, tid % 128, simd % 4, tile, input_vectors, tile, !second);
}

// 64-row paired Q8_0 prefill candidate. Each SIMD group owns a 32x16 output
// region (eight 8x8 FP32 MMA accumulators). The two projections share the
// 64x32 activation panel while retaining distinct 32x32 weight panels.
kernel void inkling2_resident_gemm_q8_0_simd_pair64(
    device const block_q8_0_56* first_weights  [[buffer(0)]],
    device const block_q8_0_56* second_weights [[buffer(1)]],
    device const float* x                     [[buffer(2)]],
    device       float* first_output          [[buffer(3)]],
    device       float* second_output         [[buffer(4)]],
    constant     uint&  in_dim                [[buffer(5)]],
    constant     uint&  out_dim               [[buffer(6)]],
    constant     uint&  batch                 [[buffer(7)]],
    uint3  group [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort simd  [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TILE = 32;
    constexpr uint ROW_TILE = 64;
    constexpr uint STAGING_FLOAT4S = 1024;

    // 8 KiB input followed by two independent 4 KiB weight panels. After the
    // final MMA barrier, the same 16 KiB is reused for two 8 KiB output tiles.
    threadgroup float4 staging[STAGING_FLOAT4S];
    threadgroup float4* input_vectors = staging;
    threadgroup float4* first_weight_vectors = staging + 512;
    threadgroup float4* second_weight_vectors = staging + 768;

    const uint first_out = group.x * TILE;
    const uint first_in = group.y * ROW_TILE;
    const uint out_rows = min(out_dim - first_out, TILE);
    const uint in_rows = min(batch - first_in, ROW_TILE);
    const uint blocks = in_dim / TILE;
    const bool second = simd >= 4;
    const uint local_simd = simd & 3;
    const uint row_quadrant = local_simd >> 1;
    const uint out_quadrant = 16 * (local_simd & 1);
    threadgroup float4* selected_weights = second ? second_weight_vectors : first_weight_vectors;

    // Raw 0..255 tids stage 64x32 input values exactly once per threadgroup.
    const uint input_row = tid / 4;
    const uint input_col = (tid & 3) * 8;
    const bool input_valid = input_row < in_rows;
    const device float* input_source =
        x + (size_t)(first_in + min(input_row, in_rows - 1)) * in_dim + input_col;

    // Each projection owns four SIMD groups (128 threads), each loading one
    // distinct 4 KiB Q8-decoded weight tile with its projection-local tid.
    const uint weight_tid = tid & 127;
    const uint weight_row = weight_tid / 4;
    const uint weight_col = (weight_tid & 3) * 8;
    const bool weight_valid = weight_row < out_rows;
    const device block_q8_0_56* selected_source = second ? second_weights : first_weights;
    const device block_q8_0_56* row_source =
        selected_source + (size_t)(first_out + min(weight_row, out_rows - 1)) * blocks;

    simdgroup_float8x8 accum[8];
    for (uint i = 0; i < 8; ++i) {
        accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint block = 0; block < blocks; ++block) {
        if (weight_valid) {
            const device block_q8_0_56& packed = row_source[block];
            const float scale = (float)packed.d;
            const device packed_char4* codes = (const device packed_char4*)(packed.qs + weight_col);
            selected_weights[2 * weight_tid] = scale * float4((char4)codes[0]);
            selected_weights[2 * weight_tid + 1] = scale * float4((char4)codes[1]);
        } else {
            selected_weights[2 * weight_tid] = float4(0.0f);
            selected_weights[2 * weight_tid + 1] = float4(0.0f);
        }

        const device packed_float4* values =
            (const device packed_float4*)(input_source + block * TILE);
        input_vectors[2 * tid] = input_valid ? float4(values[0]) : float4(0.0f);
        input_vectors[2 * tid + 1] = input_valid ? float4(values[1]) : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float* input_tile = (threadgroup float*)input_vectors;
        threadgroup float* weight_tile = (threadgroup float*)selected_weights;
        const uint row_base = row_quadrant * 32;
        #pragma unroll
        for (uint k = 0; k < TILE; k += 8) {
            simdgroup_float8x8 a0, a1, a2, a3, b0, b1;
            simdgroup_load(a0, input_tile + (row_base + 0) * TILE + k, TILE);
            simdgroup_load(a1, input_tile + (row_base + 8) * TILE + k, TILE);
            simdgroup_load(a2, input_tile + (row_base + 16) * TILE + k, TILE);
            simdgroup_load(a3, input_tile + (row_base + 24) * TILE + k, TILE);
            simdgroup_load(b0, weight_tile + out_quadrant * TILE + k, TILE, 0, true);
            simdgroup_load(b1, weight_tile + (out_quadrant + 8) * TILE + k, TILE, 0, true);

            // Preserve each output's ascending K=0..31 FP32 MMA sequence.
            simdgroup_multiply_accumulate(accum[0], a0, b0, accum[0]);
            simdgroup_multiply_accumulate(accum[1], a0, b1, accum[1]);
            simdgroup_multiply_accumulate(accum[2], a1, b0, accum[2]);
            simdgroup_multiply_accumulate(accum[3], a1, b1, accum[3]);
            simdgroup_multiply_accumulate(accum[4], a2, b0, accum[4]);
            simdgroup_multiply_accumulate(accum[5], a2, b1, accum[5]);
            simdgroup_multiply_accumulate(accum[6], a3, b0, accum[6]);
            simdgroup_multiply_accumulate(accum[7], a3, b1, accum[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // All input and weight reads ended at the final loop barrier. Alias the
    // complete staging area for one 64x32 result tile per projection.
    threadgroup float* result_storage = (threadgroup float*)staging;
    threadgroup float* result_tile = result_storage + (second ? ROW_TILE * TILE : 0);
    const uint row_base = row_quadrant * 32;
    #pragma unroll
    for (uint row_fragment = 0; row_fragment < 4; ++row_fragment) {
        const uint result_row = row_base + row_fragment * 8;
        threadgroup float* quadrant = result_tile + result_row * TILE + out_quadrant;
        simdgroup_store(accum[row_fragment * 2], quadrant, TILE);
        simdgroup_store(accum[row_fragment * 2 + 1], quadrant + 8, TILE);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* output = second ? second_output : first_output;
    for (uint i = weight_tid; i < in_rows * out_rows; i += 128) {
        const uint row = i / out_rows;
        const uint col = i % out_rows;
        output[(size_t)(first_in + row) * out_dim + first_out + col] =
            result_tile[row * TILE + col];
    }
}
