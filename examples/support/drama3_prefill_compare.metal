#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Diagnostic-only tiled reduction. No activation is copied to the host.
// [bit differences, tolerance failures, max abs, max ratio, nonfinite]
kernel void dr3_prefill_compare(device const float *actual [[buffer(0)]],
    device const float *reference [[buffer(1)]], device float *out [[buffer(2)]],
    constant ulong *p [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    ulong begin = ulong(gid) * 1024, end = min(begin + 1024, p[0]);
    float atol = as_type<float>(uint(p[1])), rtol = as_type<float>(uint(p[2]));
    float bits = 0, failures = 0, max_abs = 0, max_ratio = 0, nonfinite = 0;
    for (ulong i = begin; i < end; ++i) {
        float a = actual[i], b = reference[i];
        bits += as_type<uint>(a) != as_type<uint>(b);
        if (!isfinite(a) || !isfinite(b)) { nonfinite += 1; continue; }
        float error = abs(a - b), allowance = atol + rtol * abs(b);
        failures += error > allowance;
        max_abs = max(max_abs, error);
        max_ratio = max(max_ratio, allowance > 0 ? error / allowance : (error > 0 ? INFINITY : 0));
    }
    out[gid*5] = bits; out[gid*5+1] = failures; out[gid*5+2] = max_abs;
    out[gid*5+3] = max_ratio; out[gid*5+4] = nonfinite;
}

kernel void dr3_prefill_compare_reduce(device const float *parts [[buffer(0)]],
    device float *out [[buffer(1)]], constant ulong *p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid) return;
    float bits=0, failures=0, max_abs=0, max_ratio=0, nonfinite=0;
    for (ulong i=0; i<p[0]; ++i) {
        bits+=parts[i*5]; failures+=parts[i*5+1]; max_abs=max(max_abs,parts[i*5+2]);
        max_ratio=max(max_ratio,parts[i*5+3]); nonfinite+=parts[i*5+4];
    }
    out[0]=bits; out[1]=failures; out[2]=max_abs; out[3]=max_ratio; out[4]=nonfinite;
}

// Diagnostic canonical order only; production expert reduction is untouched.
// Each token's rank rows are sorted by expert ID, with rank as a stable tie.
kernel void dr3_prefill_canonical(device const float *ids [[buffer(0)]],
    device const float *values [[buffer(1)]], device float *out [[buffer(2)]],
    constant ulong *p [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    ulong rows=p[0], ranks=p[1], width=p[2];
    if (i>=rows*ranks*width) return;
    ulong row=ulong(i)/(ranks*width), source_rank=(ulong(i)/width)%ranks;
    ulong column=ulong(i)%width, target_rank=0;
    float id=ids[row*ranks+source_rank];
    for (ulong rank=0; rank<ranks; ++rank) {
        float other=ids[row*ranks+rank];
        target_rank += other<id || (other==id && rank<source_rank);
    }
    out[(row*ranks+target_rank)*width+column]=values[i];
}

kernel void dr3_prefill_i32_f32(device const int *input [[buffer(0)]],
    device float *out [[buffer(1)]], constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i<p[0]) out[i]=float(input[i]);
}
