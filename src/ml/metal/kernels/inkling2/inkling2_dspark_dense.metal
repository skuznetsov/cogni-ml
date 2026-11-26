#include <metal_stdlib>
using namespace metal;

static inline float inkling2_dspark_bf16_to_f32(ushort value) {
    return as_type<float>(((uint)value) << 16);
}

// Eight neighboring rows share each loaded BF16 weight. One simdgroup owns
// one output channel; its 32 lanes reduce F32 partial sums over the input
// width. The threadgroup computes an 8x8 output tile.
template <bool FullTile>
static inline void inkling2_dspark_dense_tile(
    device const ushort* weights,
    device const float* input,
    device float* output,
    uint in_dim,
    uint out_dim,
    uint rows,
    uint3 group,
    ushort lane,
    ushort simdgroup) {
    const uint output_column = group.x * 8 + uint(simdgroup);
    const uint first_row = FullTile ? 0 : group.y * 8;
    if (output_column >= out_dim || (!FullTile && first_row >= rows)) return;

    device const ushort* weight_row = weights + output_column * in_dim;
    const uint row_count = FullTile ? 8u : min(8u, rows - first_row);
    float sums[8] = {0.0f};
    for (uint column = uint(lane); column < in_dim; column += 32) {
        const float weight = inkling2_dspark_bf16_to_f32(weight_row[column]);
        #pragma unroll
        for (uint row = 0; row < 8; ++row) {
            if (row < row_count) {
                sums[row] += input[(first_row + row) * in_dim + column] * weight;
            }
        }
    }

    #pragma unroll
    for (uint row = 0; row < 8; ++row) {
        if (row < row_count) {
            const float total = simd_sum(sums[row]);
            if (lane == 0) {
                output[(first_row + row) * out_dim + output_column] = total;
            }
        }
    }
}

kernel void inkling2_dspark_dense(
    device const ushort* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    inkling2_dspark_dense_tile<false>(weights, input, output, in_dim, out_dim, rows, group, lane, simdgroup);
}

// The draft transformer always projects eight rows. Specialize that case
// so there are no row predicates in the accumulation loop.
kernel void inkling2_dspark_dense_rows8(
    device const ushort* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_dim [[buffer(3)]],
    constant uint& out_dim [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]) {
    inkling2_dspark_dense_tile<true>(weights, input, output, in_dim, out_dim, 8u, group, lane, simdgroup);
}
