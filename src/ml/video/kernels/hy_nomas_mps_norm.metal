// Adapted from PyTorch v2.8.0's non-affine F32
// layer_norm_single_row kernel.
// Upstream: aten/src/ATen/native/mps/kernels/LayerNorm.metal at v2.8.0.
// Copyright and BSD-3-Clause terms:
// third_party/licenses/pytorch-v2.8.0-BSD-3-Clause.txt.
//
// This compatibility path uses PyTorch's four-value per-thread accumulation,
// two-stage simd_sum reductions, population variance, negative-variance clamp,
// and precise reciprocal square root. Keep this kernel scoped to dim <= 4096.

kernel void hy_nomas_pytorch28_layer_norm_f32(
    device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
    constant uint &axis_size [[buffer(2)]], constant float &epsilon [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    constexpr uint SIMD_SIZE = 32;
    constexpr uint N_READS = 4;

    const uint row_offset = tg_id * axis_size;
    device const float *x = input + row_offset + tid * N_READS;
    device float *out = output + row_offset + tid * N_READS;
    const uint base_lane = tid * N_READS;

    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    if (base_lane + N_READS <= axis_size) {
        const float4 v4 = float4(x[0], x[1], x[2], x[3]);
        partial_sum = v4.x + v4.y + v4.z + v4.w;
        partial_sum_sq = dot(v4, v4);
    } else {
        const int remaining = int(axis_size - base_lane);
        if (remaining >= 3) {
            const float3 v3 = float3(x[0], x[1], x[2]);
            partial_sum = v3.x + v3.y + v3.z;
            partial_sum_sq = dot(v3, v3);
        } else if (remaining >= 2) {
            const float2 v2 = float2(x[0], x[1]);
            partial_sum = v2.x + v2.y;
            partial_sum_sq = dot(v2, v2);
        } else if (remaining >= 1) {
            const float v = x[0];
            partial_sum = v;
            partial_sum_sq = fma(v, v, partial_sum_sq);
        }
    }

    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_sums_sq[SIMD_SIZE];
    threadgroup float row_mean;
    threadgroup float row_inv_std;

    if (simdgroup_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
        local_sums_sq[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float group_partial_sum = simd_sum(partial_sum);
    const float group_partial_sum_sq = simd_sum(partial_sum_sq);
    if (simd_lane_id == 0) {
        local_sums[simdgroup_id] = group_partial_sum;
        local_sums_sq[simdgroup_id] = group_partial_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id == 0) {
        const float sum = simd_sum(local_sums[simd_lane_id]);
        const float sum_sq = simd_sum(local_sums_sq[simd_lane_id]);
        if (simd_lane_id == 0) {
            const float mean = sum / float(axis_size);
            float variance = sum_sq / float(axis_size) - mean * mean;
            variance = variance < 1e-6f ? 0.0f : variance;
            const float inv_std = metal::precise::rsqrt(variance + epsilon);
            row_mean = mean;
            row_inv_std = inv_std;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = row_mean;
    const float inv_std = row_inv_std;
    if (base_lane + N_READS <= axis_size) {
        #pragma unroll
        for (uint i = 0; i < N_READS; ++i) {
            const float value = float(x[i]);
            out[i] = (value - mean) * inv_std;
        }
    } else {
        #pragma unroll
        for (uint i = 0; i < N_READS; ++i) {
            const uint lane_idx = base_lane + i;
            if (lane_idx < axis_size) {
                const float value = float(x[i]);
                out[i] = (value - mean) * inv_std;
            }
        }
    }
}
