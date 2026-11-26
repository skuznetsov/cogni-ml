#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline uint drb_u32(device const uchar* b, ulong o) {
    return uint(b[o]) | uint(b[o + 1]) << 8 | uint(b[o + 2]) << 16 | uint(b[o + 3]) << 24;
}

inline float drb_float(device const uchar* b, ulong o, ulong i, ulong t) {
    if (t == 0) return as_type<float>(drb_u32(b, o + i * 4));
    o += i * 2;
    ushort v = ushort(b[o]) | ushort(b[o + 1]) << 8;
    return t == 1 ? as_type<float>(uint(v) << 16) : float(as_type<half>(v));
}

inline float drb_bf16_aligned(device const uchar* b, ulong o, ulong i) {
    ushort v = *(device const ushort*)(b + o + i * 2);
    return as_type<float>(uint(v) << 16);
}

kernel void dr3_batch_fixed_ids(device float* ids [[buffer(0)]],
                                constant ulong* p [[buffer(1)]],
                                uint i [[thread_position_in_grid]]) {
    if (i < p[0]) ids[i] = float(p[1]);
}

kernel void dr3_batch_repeat_rows(device const float* x [[buffer(0)]],
                                  device float* y [[buffer(1)]],
                                  constant ulong* p [[buffer(2)]],
                                  uint i [[thread_position_in_grid]]) {
    ulong rows = p[0], cols = p[1], top_k = p[2];
    if (i >= rows * cols * top_k) return;
    ulong token = (ulong(i) / cols) / top_k;
    ulong column = ulong(i) % cols;
    y[i] = x[token * cols + column];
}

kernel void dr3_dense_rows(device const uchar* w [[buffer(0)]],
                           device const float* x [[buffer(1)]],
                           device float* y [[buffer(2)]],
                           constant ulong* p [[buffer(3)]],
                           uint2 pos [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float parts[64];
    uint rows = uint(p[0]), cols = uint(p[1]);
    ulong row = p[4] * p[0] + pos.x;
    ulong input_base = ulong(pos.y) * cols;
    ulong output_index = ulong(pos.y) * rows + pos.x;
    float sum = 0;
    for (uint j = lane; j < cols; j += 64)
        sum += drb_float(w, p[2], row * cols + j, p[3]) * x[input_base + j];
    parts[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = 32; d > 0; d /= 2) {
        if (lane < d) parts[lane] += parts[lane + d];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) y[output_index] = parts[0];
}

kernel void dr3_dense_rows_simd(device const uchar* w [[buffer(0)]],
                                device const float* x [[buffer(1)]],
                                device float* y [[buffer(2)]],
                                constant ulong* p [[buffer(3)]],
                                uint2 pos [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint simd_lane [[thread_index_in_simdgroup]],
                                uint simdgroup [[simdgroup_index_in_threadgroup]],
                                uint simd_width [[threads_per_simdgroup]]) {
    threadgroup float parts[64];
    uint rows = uint(p[0]), cols = uint(p[1]);
    ulong row = p[4] * p[0] + pos.x;
    ulong input_base = ulong(pos.y) * cols;
    ulong output_index = ulong(pos.y) * rows + pos.x;
    float sum = 0;
    if (p[3] == 1 && (p[2] & 1) == 0) {
        for (uint j = lane; j < cols; j += 64)
            sum += drb_bf16_aligned(w, p[2], row * cols + j) * x[input_base + j];
    } else {
        for (uint j = lane; j < cols; j += 64)
            sum += drb_float(w, p[2], row * cols + j, p[3]) * x[input_base + j];
    }
    parts[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_width != 32) {
        for (uint d = 32; d > 0; d /= 2) {
            if (lane < d) parts[lane] += parts[lane + d];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0) y[output_index] = parts[0];
    } else if (simdgroup == 0) {
        float reduced = parts[simd_lane] + parts[simd_lane + 32];
        for (uint d = 16; d > 0; d /= 2) {
            float peer = simd_shuffle_down(reduced, ushort(d));
            if (simd_lane < d) reduced += peer;
        }
        if (simd_lane == 0) y[output_index] = reduced;
    }
}

kernel void dr3_dense_rows_simd32(device const uchar* w [[buffer(0)]],
                                  device const float* x [[buffer(1)]],
                                  device float* y [[buffer(2)]],
                                  constant ulong* p [[buffer(3)]],
                                  uint2 pos [[threadgroup_position_in_grid]],
                                  uint simd_lane [[thread_index_in_simdgroup]],
                                  uint simd_width [[threads_per_simdgroup]]) {
    if (simd_width != 32) return;
    uint rows = uint(p[0]), cols = uint(p[1]);
    ulong row = p[4] * p[0] + pos.x;
    ulong input_base = ulong(pos.y) * cols;
    ulong output_index = ulong(pos.y) * rows + pos.x;
    float lo = 0, hi = 0;
    if (p[3] == 1 && (p[2] & 1) == 0) {
        for (uint j = simd_lane; j < cols; j += 64)
            lo += drb_bf16_aligned(w, p[2], row * cols + j) * x[input_base + j];
        for (uint j = simd_lane + 32; j < cols; j += 64)
            hi += drb_bf16_aligned(w, p[2], row * cols + j) * x[input_base + j];
    } else {
        for (uint j = simd_lane; j < cols; j += 64)
            lo += drb_float(w, p[2], row * cols + j, p[3]) * x[input_base + j];
        for (uint j = simd_lane + 32; j < cols; j += 64)
            hi += drb_float(w, p[2], row * cols + j, p[3]) * x[input_base + j];
    }
    float reduced = lo + hi;
    for (uint d = 16; d > 0; d /= 2) {
        float peer = simd_shuffle_down(reduced, ushort(d));
        if (simd_lane < d) reduced += peer;
    }
    if (simd_lane == 0) y[output_index] = reduced;
}

inline float drb_sigmoid(float v) {
    return v >= 0 ? 1.0f / (1.0f + exp(-v)) : exp(v) / (1.0f + exp(v));
}

kernel void dr3_route_rows(device const float* x [[buffer(0)]],
                           device float* ids [[buffer(1)]],
                           device float* weights [[buffer(2)]],
                           constant ulong* p [[buffer(3)]],
                           uint row [[threadgroup_position_in_grid]]) {
    uint experts = uint(p[1]), top_k = uint(p[2]);
    ulong input_base = ulong(row) * experts;
    ulong output_base = ulong(row) * top_k;
    uint selected_ids[512];
    float scores[512];
    float maximum = -INFINITY, total = 0;
    for (uint e = 0; e < experts; e++) maximum = max(maximum, x[input_base + e]);
    for (uint e = 0; e < experts; e++) {
        scores[e] = exp(x[input_base + e] - maximum);
        total += scores[e];
    }
    for (uint e = 0; e < experts; e++) scores[e] /= total;

    float selected_total = 0;
    for (uint rank = 0; rank < top_k; rank++) {
        uint best = 0;
        float value = -INFINITY;
        for (uint e = 0; e < experts; e++) {
            bool used = false;
            for (uint previous = 0; previous < rank; previous++)
                used |= selected_ids[previous] == e;
            if (!used && scores[e] > value) {
                best = e;
                value = scores[e];
            }
        }
        selected_ids[rank] = best;
        ids[output_base + rank] = float(best);
        selected_total += scores[best];
    }
    for (uint rank = 0; rank < top_k; rank++)
        weights[output_base + rank] = scores[selected_ids[rank]] / (p[3] ? selected_total : 1.0f);
}

kernel void dr3_expert_sum_rows(device const float* down [[buffer(0)]],
                                device const float* weights [[buffer(1)]],
                                device float* out [[buffer(2)]],
                                constant ulong* p [[buffer(3)]],
                                uint i [[thread_position_in_grid]]) {
    ulong rows = p[0], hidden = p[1], top_k = p[2];
    if (i >= rows * hidden) return;
    ulong row = ulong(i) / hidden;
    ulong column = ulong(i) % hidden;
    float sum = 0;
    for (uint rank = 0; rank < top_k; rank++) {
        ulong flat_rank = row * top_k + rank;
        sum += down[flat_rank * hidden + column] * weights[flat_rank];
    }
    out[i] = sum;
}

kernel void dr3_weighted_add_rows(device float* out [[buffer(0)]],
                                  device const float* shared [[buffer(1)]],
                                  device const float* gate [[buffer(2)]],
                                  constant ulong* p [[buffer(3)]],
                                  uint i [[thread_position_in_grid]]) {
    ulong hidden = p[1];
    if (i >= p[0] * hidden) return;
    ulong row = ulong(i) / hidden;
    out[i] += shared[i] * drb_sigmoid(gate[row]);
}
