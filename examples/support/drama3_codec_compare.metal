#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Diagnostic reductions only. Model activations and arithmetic are untouched.
constant uint DR3_CODEC_METRICS = 14;

inline void dr3_codec_kahan(float value, thread float& sum, thread float& correction) {
    float y = value - correction;
    float next = sum + y;
    correction = (next - sum) - y;
    sum = next;
}

kernel void dr3_codec_compare_parts(
    device const float* actual [[buffer(0)]],
    device const float* reference [[buffer(1)]],
    device float* parts [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint part [[thread_position_in_grid]]) {
    uint begin = part * 1024, end = min(begin + 1024, uint(p[0]));
    if (begin >= end) return;
    float sums[4] = {0, 0, 0, 0}, corrections[4] = {0, 0, 0, 0};
    float max_error = 0, max_reference = 0, max_actual = 0, max_pcm = 0;
    uint worst = begin, point_failures = 0, nonfinite = 0, pcm_exact = 0;
    for (uint i = begin; i < end; ++i) {
        float a = actual[i], r = reference[i];
        if (!isfinite(a) || !isfinite(r)) { ++nonfinite; continue; }
        float error = abs(a - r);
        if (error > max_error) { max_error = error; worst = i; }
        max_reference = max(max_reference, abs(r));
        max_actual = max(max_actual, abs(a));
        dr3_codec_kahan(error * error, sums[0], corrections[0]);
        dr3_codec_kahan(r * r, sums[1], corrections[1]);
        dr3_codec_kahan(a * a, sums[2], corrections[2]);
        if (p[1]) {
            if (error > max(5e-5f, abs(r) * 1e-3f)) ++point_failures;
            float pa = rint(clamp(a, -1.0f, 1.0f) * 32767.0f);
            float pr = rint(clamp(r, -1.0f, 1.0f) * 32767.0f);
            float delta = abs(pa - pr);
            max_pcm = max(max_pcm, delta);
            dr3_codec_kahan(delta * delta, sums[3], corrections[3]);
            if (delta == 0) ++pcm_exact;
        }
    }
    uint base = part * DR3_CODEC_METRICS;
    parts[base] = max_error;
    parts[base + 1] = sums[0];
    parts[base + 2] = sums[1];
    parts[base + 3] = sums[2];
    parts[base + 4] = max_reference;
    parts[base + 5] = max_actual;
    parts[base + 6] = float(point_failures);
    parts[base + 7] = float(nonfinite);
    parts[base + 8] = float(worst >> 16);
    parts[base + 9] = float(worst & 65535);
    parts[base + 10] = max_pcm;
    parts[base + 11] = sums[3];
    parts[base + 12] = float(pcm_exact);
    parts[base + 13] = 0;
}

kernel void dr3_codec_compare_reduce(
    device const float* parts [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong* p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid) return;
    float sums[4] = {0, 0, 0, 0}, corrections[4] = {0, 0, 0, 0};
    float max_error = 0, max_reference = 0, max_actual = 0, max_pcm = 0;
    uint worst = 0, point_failures = 0, nonfinite = 0, pcm_exact = 0;
    for (uint part = 0; part < p[1]; ++part) {
        uint base = part * DR3_CODEC_METRICS;
        float error = parts[base];
        if (error > max_error) {
            max_error = error;
            worst = (uint(parts[base + 8]) << 16) | uint(parts[base + 9]);
        }
        dr3_codec_kahan(parts[base + 1], sums[0], corrections[0]);
        dr3_codec_kahan(parts[base + 2], sums[1], corrections[1]);
        dr3_codec_kahan(parts[base + 3], sums[2], corrections[2]);
        dr3_codec_kahan(parts[base + 11], sums[3], corrections[3]);
        max_reference = max(max_reference, parts[base + 4]);
        max_actual = max(max_actual, parts[base + 5]);
        point_failures += uint(parts[base + 6]);
        nonfinite += uint(parts[base + 7]);
        max_pcm = max(max_pcm, parts[base + 10]);
        pcm_exact += uint(parts[base + 12]);
    }
    float rms = sqrt(sums[0] / float(p[0]));
    float reference_rms = sqrt(sums[1] / float(p[0]));
    output[0] = max_error;
    output[1] = rms;
    output[2] = reference_rms;
    output[3] = sqrt(sums[2] / float(p[0]));
    output[4] = max_reference;
    output[5] = max_actual;
    output[6] = float(point_failures);
    output[7] = float(nonfinite);
    output[8] = float(worst >> 16);
    output[9] = float(worst & 65535);
    output[10] = max_pcm;
    output[11] = sqrt(sums[3] / float(p[0]));
    output[12] = float(pcm_exact);
    // -1 denotes nonzero error against an exactly zero reference. Keep the
    // diagnostic finite while failing the normalized-RMS gate explicitly.
    output[13] = reference_rms > 0 ? rms / reference_rms : (rms == 0 ? 0 : -1);
}

kernel void dr3_codec_wave_windows(
    device const float* actual [[buffer(0)]],
    device const float* reference [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint window [[thread_position_in_grid]]) {
    uint begin = window * 22050, end = min(begin + 22050, uint(p[0]));
    if (begin >= end) return;
    float sum_a = 0, sum_r = 0, correction_a = 0, correction_r = 0;
    float peak_a = 0, peak_r = 0, pcm_a = 0, pcm_r = 0;
    for (uint i = begin; i < end; ++i) {
        float a = actual[i], r = reference[i];
        dr3_codec_kahan(a * a, sum_a, correction_a);
        dr3_codec_kahan(r * r, sum_r, correction_r);
        peak_a = max(peak_a, abs(a));
        peak_r = max(peak_r, abs(r));
        pcm_a = max(pcm_a, abs(rint(clamp(a, -1.0f, 1.0f) * 32767.0f)));
        pcm_r = max(pcm_r, abs(rint(clamp(r, -1.0f, 1.0f) * 32767.0f)));
    }
    output[window * 6] = sqrt(sum_a / float(end - begin));
    output[window * 6 + 1] = sqrt(sum_r / float(end - begin));
    output[window * 6 + 2] = peak_a;
    output[window * 6 + 3] = peak_r;
    output[window * 6 + 4] = pcm_a;
    output[window * 6 + 5] = pcm_r;
}
