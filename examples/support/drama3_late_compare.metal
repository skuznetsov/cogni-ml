#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Each lane owns one 1024-element tile. The integer records preserve raw bits
// (including NaNs and signed zero) until the one terminal compact-metrics read.
// Record layout: counts[3], max_abs_bits, max_ratio_bits, then three witnesses
// of [index_lo, index_hi, actual_bits, reference_bits, error_bits,
// allowance_bits, expert_id_bits]. UINT_MAX is the absent-witness sentinel.
kernel void dr3_late_compare_tile(device const float *actual [[buffer(0)]],
    device const float *reference [[buffer(1)]], device uint *parts [[buffer(2)]],
    device const int *route_ids [[buffer(3)]], constant ulong *p [[buffer(4)]],
    uint tile [[thread_position_in_grid]]) {
    ulong begin = ulong(tile) * 1024;
    ulong end = min(begin + 1024, p[0]);
    float atol = as_type<float>(uint(p[1]));
    float rtol = as_type<float>(uint(p[2]));
    ulong offset = p[3];
    bool has_routes = p[4] != 0;
    ulong hidden = max(p[5], 1ul);

    uint bits = 0, failures = 0, nonfinite = 0;
    float max_abs = 0.0f, max_ratio = 0.0f;
    uint first_bit = UINT_MAX, first_tolerance = UINT_MAX, first_nonfinite = UINT_MAX;
    uint bit_a = 0, bit_b = 0, bit_error = 0, bit_allowance = 0, bit_expert = UINT_MAX;
    uint tol_a = 0, tol_b = 0, tol_error = 0, tol_allowance = 0, tol_expert = UINT_MAX;
    uint nf_a = 0, nf_b = 0, nf_error = 0, nf_allowance = 0, nf_expert = UINT_MAX;
    uint bit_hi = UINT_MAX, tolerance_hi = UINT_MAX, nonfinite_hi = UINT_MAX;

    for (ulong i = begin; i < end; ++i) {
        float a = actual[i], b = reference[i];
        uint ab = as_type<uint>(a), bb = as_type<uint>(b);
        ulong global_index = offset + i;
        uint low = uint(global_index), high = uint(global_index >> 32);
        int expert = has_routes ? route_ids[i / hidden] : -1;
        uint expert_bits = as_type<uint>(expert);
        float error = abs(a - b);
        float allowance = atol + rtol * abs(b);

        if (ab != bb) {
            bits++;
            if (high < bit_hi || (high == bit_hi && low < first_bit)) {
                first_bit = low; bit_hi = high;
                bit_a = ab; bit_b = bb; bit_error = as_type<uint>(error);
                bit_allowance = as_type<uint>(allowance); bit_expert = expert_bits;
            }
        }
        if (!isfinite(a) || !isfinite(b)) {
            nonfinite++;
            if (high < nonfinite_hi || (high == nonfinite_hi && low < first_nonfinite)) {
                first_nonfinite = low; nonfinite_hi = high;
                nf_a = ab; nf_b = bb; nf_error = as_type<uint>(error);
                nf_allowance = as_type<uint>(allowance); nf_expert = expert_bits;
            }
            continue;
        }
        failures += error > allowance;
        max_abs = max(max_abs, error);
        float ratio = allowance > 0.0f ? error / allowance : (error > 0.0f ? INFINITY : 0.0f);
        max_ratio = max(max_ratio, ratio);
        if (error > allowance && (high < tolerance_hi || (high == tolerance_hi && low < first_tolerance))) {
            first_tolerance = low; tolerance_hi = high;
            tol_a = ab; tol_b = bb; tol_error = as_type<uint>(error);
            tol_allowance = as_type<uint>(allowance); tol_expert = expert_bits;
        }
    }

    uint base = tile * 26;
    parts[base + 0] = bits; parts[base + 1] = failures; parts[base + 2] = nonfinite;
    parts[base + 3] = as_type<uint>(max_abs); parts[base + 4] = as_type<uint>(max_ratio);
    parts[base + 5] = first_bit; parts[base + 6] = bit_hi; parts[base + 7] = bit_a;
    parts[base + 8] = bit_b; parts[base + 9] = bit_error; parts[base + 10] = bit_allowance;
    parts[base + 11] = bit_expert;
    parts[base + 12] = first_tolerance; parts[base + 13] = tolerance_hi; parts[base + 14] = tol_a;
    parts[base + 15] = tol_b; parts[base + 16] = tol_error; parts[base + 17] = tol_allowance;
    parts[base + 18] = tol_expert;
    parts[base + 19] = first_nonfinite; parts[base + 20] = nonfinite_hi; parts[base + 21] = nf_a;
    parts[base + 22] = nf_b; parts[base + 23] = nf_error; parts[base + 24] = nf_allowance;
    parts[base + 25] = nf_expert;
}

// One thread merges all tile receipts with integer min-index tie handling.
// Output is 52 finite floats: each uint32 receipt word is split into exact
// 16-bit numeric halves so Context.read never interprets witness NaNs as data.
kernel void dr3_late_compare_reduce(device const uint *parts [[buffer(0)]],
    device float *out [[buffer(1)]], constant ulong *p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    uint result[26];
    for (uint i = 0; i < 26; ++i) result[i] = 0;
    result[3] = as_type<uint>(0.0f); result[4] = as_type<uint>(0.0f);
    result[5] = result[12] = result[19] = UINT_MAX;
    result[6] = result[13] = result[20] = UINT_MAX;
    result[11] = result[18] = result[25] = UINT_MAX;

    for (ulong tile = 0; tile < p[0]; ++tile) {
        uint base = uint(tile) * 26;
        result[0] += parts[base + 0]; result[1] += parts[base + 1]; result[2] += parts[base + 2];
        float part_abs = as_type<float>(parts[base + 3]);
        float part_ratio = as_type<float>(parts[base + 4]);
        if (part_abs > as_type<float>(result[3])) result[3] = parts[base + 3];
        if (part_ratio > as_type<float>(result[4])) result[4] = parts[base + 4];
        for (uint witness = 0; witness < 3; ++witness) {
            uint dst = 5 + witness * 7;
            uint src = base + dst;
            uint candidate_lo = parts[src], candidate_hi = parts[src + 1];
            if (candidate_lo == UINT_MAX && candidate_hi == UINT_MAX) continue;
            uint old_lo = result[dst], old_hi = result[dst + 1];
            if (old_lo == UINT_MAX && old_hi == UINT_MAX || candidate_hi < old_hi ||
                (candidate_hi == old_hi && candidate_lo < old_lo)) {
                for (uint field = 0; field < 7; ++field) result[dst + field] = parts[src + field];
            }
        }
    }
    for (uint i = 0; i < 26; ++i) {
        out[i * 2] = float(result[i] & 0xffffu);
        out[i * 2 + 1] = float(result[i] >> 16);
    }
}

// Merge the compact per-chunk receipts for one logical case on device. The
// parameter buffer is [slot_count, metrics_slot_0, ...]; host code supplies
// only stable slot metadata and never chooses or inspects a witness.
kernel void dr3_late_compare_merge(device const float *receipts [[buffer(0)]],
    device float *out [[buffer(1)]], constant ulong *p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    uint result[26];
    for (uint i = 0; i < 26; ++i) result[i] = 0;
    result[3] = as_type<uint>(0.0f); result[4] = as_type<uint>(0.0f);
    result[5] = result[12] = result[19] = UINT_MAX;
    result[6] = result[13] = result[20] = UINT_MAX;
    result[11] = result[18] = result[25] = UINT_MAX;

    for (ulong item = 0; item < p[0]; ++item) {
        ulong base = p[item + 1] * 52;
        uint words[26];
        for (uint i = 0; i < 26; ++i)
            words[i] = uint(receipts[base + i * 2]) | (uint(receipts[base + i * 2 + 1]) << 16);
        result[0] += words[0]; result[1] += words[1]; result[2] += words[2];
        if (as_type<float>(words[3]) > as_type<float>(result[3])) result[3] = words[3];
        if (as_type<float>(words[4]) > as_type<float>(result[4])) result[4] = words[4];
        for (uint witness = 0; witness < 3; ++witness) {
            uint dst = 5 + witness * 7;
            uint candidate_lo = words[dst], candidate_hi = words[dst + 1];
            if (candidate_lo == UINT_MAX && candidate_hi == UINT_MAX) continue;
            uint old_lo = result[dst], old_hi = result[dst + 1];
            if ((old_lo == UINT_MAX && old_hi == UINT_MAX) || candidate_hi < old_hi ||
                (candidate_hi == old_hi && candidate_lo < old_lo)) {
                for (uint field = 0; field < 7; ++field) result[dst + field] = words[dst + field];
            }
        }
    }
    for (uint i = 0; i < 26; ++i) {
        out[i * 2] = float(result[i] & 0xffffu);
        out[i * 2 + 1] = float(result[i] >> 16);
    }
}

// Upstream route IDs are immutable int32 fixture bytes. Convert them on GPU
// for the existing affine experts_rows interface; comparator witnesses still
// use the original int32 buffer.
kernel void dr3_late_ids_i32_to_f32(device const int *input [[buffer(0)]],
    device float *output [[buffer(1)]], constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) output[i] = float(input[i]);
}

