// Diagnostic setup, sampling and full-output controls. All tensor arithmetic
// stays on the existing chain; only compact results are exported after drain.
#include <metal_stdlib>
using namespace metal;

inline float probe_bf16(ushort x) { return as_type<float>(uint(x) << 16); }
inline ushort probe_round(float x) {
    const uint b = as_type<uint>(x);
    if ((b & 0x7fffffffu) > 0x7f800000u) return ushort((b >> 16) | 0x40u);
    return ushort((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
}
inline uint probe_hash(uint v) {
    v ^= v >> 16; v *= 0x7feb352du;
    v ^= v >> 15; v *= 0x846ca68bu;
    return v ^ (v >> 16);
}
kernel void dcae_reduce_fill(device ushort *out [[buffer(0)]],
    constant uint *p [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    if (id >= p[0]) return;
    if (p[2] == 0u) {
        out[id] = ushort(((id + p[1]) % 2u ? 0x8000u : 0u) |
                         0x3f00u | ((id * 37u + p[1] * 13u) & 127u));
    } else {
        const uint h = probe_hash(id ^ p[1]);
        out[id] = ushort((h & 0x8000u) | ((124u + (h % 6u)) << 7) |
                         ((h >> 16) & 127u));
    }
}

// First 512 coordinates exercise channels and temporal/spatial edges. The
// rest are deterministic stratified coordinates independent of tensor values.
kernel void dcae_reduce_uniform(device uint *indices [[buffer(0)]],
    constant uint *p [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    const uint c = p[0], t = p[1], h = p[2], w = p[3], n = p[4];
    if (id >= n) return;
    if (id < min(512u, n)) {
        const uint ts[5] = {0u, 1u, t / 2u, t - 2u, t - 1u};
        const uint ys[3] = {0u, h / 2u, h - 1u};
        const uint xs[3] = {0u, w / 2u, w - 1u};
        indices[id] = (((id * 37u % c) * t + ts[id % 5u]) * h +
                       ys[(id / 5u) % 3u]) * w + xs[(id / 15u) % 3u];
    } else {
        const ulong total = ulong(c) * t * h * w;
        const uint bin = id - 512u, bins = n - 512u;
        const ulong start = total * bin / bins, end = total * (bin + 1u) / bins;
        indices[id] = uint(start + probe_hash(id) % uint(end - start));
    }
}

// One threadgroup per disjoint full-output bin. Choose the F32-lane value
// nearest a BF16 midpoint (distance in F32 ULPs), breaking ties by index.
// This is deliberate adversarial sampling, not an unbiased error estimate.
kernel void dcae_reduce_midpoints(device const float *raw [[buffer(0)]],
    device uint *indices [[buffer(1)]], constant uint *p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint scores[256], coords[256];
    const uint bins = p[1], offset = p[2];
    const uint start = uint(ulong(p[0]) * group / bins);
    const uint end = uint(ulong(p[0]) * (group + 1u) / bins);
    uint score = 0xffffffffu, coord = 0xffffffffu;
    for (uint j = start + lane; j < end; j += 256u) {
        const uint b = as_type<uint>(raw[j]);
        if ((b & 0x7f800000u) == 0x7f800000u) continue;
        const uint s = uint(abs(int(b & 0xffffu) - 0x8000));
        if (s < score || (s == score && j < coord)) { score = s; coord = j; }
    }
    scores[lane] = score; coords[lane] = coord;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 128u; step; step >>= 1u) {
        if (lane < step) {
            const uint s = scores[lane + step], j = coords[lane + step];
            if (s < scores[lane] || (s == scores[lane] && j < coords[lane])) {
                scores[lane] = s; coords[lane] = j;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) indices[offset + group] = coords[0];
}

kernel void dcae_reduce_compare_full(device const ushort *original [[buffer(0)]],
    device const ushort *pre [[buffer(1)]], device const ushort *post [[buffer(2)]],
    device const float *f32 [[buffer(3)]], device atomic_uint *stats [[buffer(4)]],
    constant uint &count [[buffer(5)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float a = probe_bf16(original[id]), b = probe_bf16(post[id]);
    const float c = probe_bf16(pre[id]), d = probe_bf16(probe_round(f32[id]));
    if (original[id] != post[id]) atomic_fetch_add_explicit(stats, 1u, memory_order_relaxed);
    if (!isfinite(a) || !isfinite(b) || abs(a - b) > 2e-5f + 2e-5f * abs(a))
        atomic_fetch_add_explicit(stats + 1, 1u, memory_order_relaxed);
    if (!isfinite(a) || !isfinite(b) || !isfinite(c))
        atomic_fetch_add_explicit(stats + 2, 1u, memory_order_relaxed);
    if (pre[id] != probe_round(f32[id])) atomic_fetch_add_explicit(stats + 3, 1u, memory_order_relaxed);
    if (!isfinite(c) || !isfinite(d) || abs(c - d) > 2e-5f + 2e-5f * abs(c))
        atomic_fetch_add_explicit(stats + 4, 1u, memory_order_relaxed);
    if (!isfinite(f32[id])) atomic_fetch_add_explicit(stats + 5, 1u, memory_order_relaxed);
}

kernel void dcae_reduce_gather(device const uint *indices [[buffer(0)]],
    device const ushort *original [[buffer(1)]], device const ushort *pre [[buffer(2)]],
    device const ushort *post [[buffer(3)]], device const float *raw [[buffer(4)]],
    device ushort *sample_original [[buffer(5)]], device ushort *sample_pre [[buffer(6)]],
    device ushort *sample_post [[buffer(7)]], device float *sample_raw [[buffer(8)]],
    constant uint *p [[buffer(9)]], uint id [[thread_position_in_grid]]) {
    if (id >= p[0]) return;
    const uint j = indices[id];
    if (j >= p[1]) {
        sample_original[id] = sample_pre[id] = sample_post[id] = 0x7fc0u;
        sample_raw[id] = as_type<float>(0x7fc00000u);
        return;
    }
    sample_original[id] = original[j]; sample_pre[id] = pre[j];
    sample_post[id] = post[j]; sample_raw[id] = raw[j];
}
