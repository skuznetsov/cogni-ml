#include <metal_stdlib>
using namespace metal;

constant ushort QWEN35_VEC_TG = 256;

// Elementwise SwiGLU combine:
//   out[i] = silu(gate[i]) * up[i]
kernel void qwen35_swiglu_mul(
    device const float* gate  [[buffer(0)]],
    device const float* up    [[buffer(1)]],
    device       float* out   [[buffer(2)]],
    constant     uint&  count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    const float g = gate[gid];
    const float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = (g * sig) * up[gid];
}

// Residual add + RMSNorm in one pass:
//   residual[i] = x[i] + y[i]
//   normed[i]   = residual[i] * rsqrt(mean(residual^2) + eps) * weight[i]
//
// One threadgroup handles one full vector.
kernel void qwen35_add_rmsnorm(
    device const float* x      [[buffer(0)]],
    device const float* y      [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device       float* residual[[buffer(3)]],
    device       float* normed [[buffer(4)]],
    constant     uint&  count  [[buffer(5)]],
    constant     float& eps    [[buffer(6)]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float partial[QWEN35_VEC_TG];

    float ss = 0.0f;
    for (uint i = tid; i < count; i += QWEN35_VEC_TG) {
        const float r = x[i] + y[i];
        ss += r * r;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = QWEN35_VEC_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / float(count) + eps);
    for (uint i = tid; i < count; i += QWEN35_VEC_TG) {
        const float r = x[i] + y[i];
        residual[i] = r;
        normed[i] = r * inv_rms * weight[i];
    }
}


// Elementwise vector add:
//   out[i] = a[i] + b[i]
kernel void qwen35_add_vec(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    out[gid] = a[gid] + b[gid];
}

// Plain RMSNorm on one vector:
//   out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
kernel void qwen35_rmsnorm_vec(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device       float* out    [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    constant     float& eps    [[buffer(4)]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float partial[QWEN35_VEC_TG];

    float ss = 0.0f;
    for (uint i = tid; i < count; i += QWEN35_VEC_TG) {
        const float v = x[i];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = QWEN35_VEC_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / float(count) + eps);
    for (uint i = tid; i < count; i += QWEN35_VEC_TG) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// Plain RMSNorm on token-major rows:
//   out[row, i] = x[row, i] * rsqrt(mean(row^2) + eps) * weight[i]
kernel void qwen35_rmsnorm_rows(
    device const float* x       [[buffer(0)]],
    device const float* weight  [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant     uint&  dim     [[buffer(3)]],
    constant     uint&  n_rows  [[buffer(4)]],
    constant     float& eps     [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    if (row >= n_rows) return;

    threadgroup float partial[QWEN35_VEC_TG];
    const uint base = row * dim;

    float ss = 0.0f;
    for (uint i = tid; i < dim; i += QWEN35_VEC_TG) {
        const float v = x[base + i];
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = QWEN35_VEC_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += QWEN35_VEC_TG) {
        out[base + i] = x[base + i] * inv_rms * weight[i];
    }
}

// Residual add + RMSNorm on token-major rows.
kernel void qwen35_add_rmsnorm_rows(
    device const float* x       [[buffer(0)]],
    device const float* y       [[buffer(1)]],
    device const float* weight  [[buffer(2)]],
    device       float* residual[[buffer(3)]],
    device       float* normed  [[buffer(4)]],
    constant     uint&  dim     [[buffer(5)]],
    constant     uint&  n_rows  [[buffer(6)]],
    constant     float& eps     [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    if (row >= n_rows) return;

    threadgroup float partial[QWEN35_VEC_TG];
    const uint base = row * dim;

    float ss = 0.0f;
    for (uint i = tid; i < dim; i += QWEN35_VEC_TG) {
        const float r = x[base + i] + y[base + i];
        ss += r * r;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = QWEN35_VEC_TG / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(partial[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += QWEN35_VEC_TG) {
        const float r = x[base + i] + y[base + i];
        residual[base + i] = r;
        normed[base + i] = r * inv_rms * weight[i];
    }
}
