#include <metal_simdgroup_matrix>

// One SIMD group handles a 16-by-16 quadrant of the 32-by-32 output tile.
// Weights and inputs are staged as F32, with one Q8_0 block per weight row.
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
    constexpr uint TILE = 32;
    const uint first_out = group.x * TILE;
    const uint first_in = group.y * TILE;
    const uint out_rows = min(out_dim - first_out, TILE);
    const uint in_rows = min(batch - first_in, TILE);
    const uint out_quadrant = 16 * (simd & 1);
    const uint in_quadrant = 16 * (simd >> 1);
    const uint blocks = in_dim / TILE;

    threadgroup float weight_tile[TILE * TILE];
    threadgroup float input_tile[TILE * TILE];
    simdgroup_float8x8 accum[4];
    for (uint i = 0; i < 4; ++i) {
        accum[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint block = 0; block < blocks; ++block) {
        for (uint i = tid; i < TILE * TILE; i += 128) {
            const uint row = i / TILE;
            const uint col = i % TILE;
            if (row < out_rows) {
                device const block_q8_0_56& packed = weights[(size_t)(first_out + row) * blocks + block];
                weight_tile[i] = (float)packed.d * (float)packed.qs[col];
            } else {
                weight_tile[i] = 0.0f;
            }
            input_tile[i] = row < in_rows ? x[(size_t)(first_in + row) * in_dim + block * TILE + col] : 0.0f;
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

    threadgroup float* quadrant = input_tile + in_quadrant * TILE + out_quadrant;
    simdgroup_store(accum[0], quadrant, TILE);
    simdgroup_store(accum[1], quadrant + 8, TILE);
    simdgroup_store(accum[2], quadrant + 8 * TILE, TILE);
    simdgroup_store(accum[3], quadrant + 8 * TILE + 8, TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < in_rows * out_rows; i += 128) {
        const uint row = i / out_rows;
        const uint col = i % out_rows;
        output[(size_t)(first_in + row) * out_dim + first_out + col] = input_tile[row * TILE + col];
    }
}
