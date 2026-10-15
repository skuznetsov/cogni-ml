// Fixed-logit sampler diagnostics only. Production sampling remains in
// src/ml/audio/drama3/kernels/resident.metal.
#include <metal_stdlib>
using namespace metal;

kernel void dr3_sampler_compare(
    device const float* x [[buffer(0)]],
    device const float* noise_uniforms [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    uint n = uint(p[0]);
    uint k = uint(p[1]);
    uint draw_count = uint(p[2]);
    float temperature = as_type<float>(uint(p[3]));
    float top_p = as_type<float>(uint(p[4]));
    uint ids[1024];
    float values[1024];
    float maximum = -INFINITY;

    for (uint j = 0; j < n; ++j) {
        if (!isfinite(x[j])) {
            y[0] = NAN;
            return;
        }
        float value = x[j] / temperature;
        maximum = max(maximum, value);
        uint pos = min(j, k);
        while (pos > 0 && (value > values[pos - 1] ||
               (value == values[pos - 1] && j < ids[pos - 1]))) {
            if (pos < k) {
                values[pos] = values[pos - 1];
                ids[pos] = ids[pos - 1];
            }
            --pos;
        }
        if (pos < k) {
            values[pos] = value;
            ids[pos] = j;
        }
    }

    // Keep the full-vocabulary denominator and top-p cumulative mass in the
    // same float32 accumulation order as production dr3_sample.
    float denominator = 0.0f;
    for (uint j = 0; j < n; ++j) denominator += exp(x[j] / temperature - maximum);

    uint offset_ids = 0;
    uint offset_logits = k;
    uint offset_kept = 2 * k;
    uint offset_full_probs = 3 * k;
    uint offset_full_cdf = 4 * k;
    uint offset_cond_probs = 5 * k;
    uint offset_cond_cdf = 6 * k;
    uint offset_kept_count = 7 * k;
    uint offset_denominator = offset_kept_count + 1;
    uint offset_retained_mass = offset_kept_count + 2;
    uint offset_gumbel_ids = offset_kept_count + 3;

    float full_cumulative = 0.0f;
    float retained_weight = 0.0f;
    uint kept_count = 0;
    for (uint rank = 0; rank < k; ++rank) {
        float weight = exp(values[rank] - maximum);
        float probability = weight / denominator;
        full_cumulative += probability;
        bool kept = rank == 0 || full_cumulative <= top_p;
        if (kept) {
            retained_weight += weight;
            ++kept_count;
        }
        y[offset_ids + rank] = float(ids[rank]);
        y[offset_logits + rank] = values[rank];
        y[offset_kept + rank] = kept ? 1.0f : 0.0f;
        y[offset_full_probs + rank] = probability;
        y[offset_full_cdf + rank] = full_cumulative;
    }

    float retained_mass = retained_weight / denominator;
    float conditional_cumulative = 0.0f;
    for (uint rank = 0; rank < k; ++rank) {
        if (y[offset_kept + rank] != 0.0f) {
            float weight = exp(values[rank] - maximum);
            float probability = weight / retained_weight;
            conditional_cumulative += probability;
            y[offset_cond_probs + rank] = probability;
            y[offset_cond_cdf + rank] = conditional_cumulative;
        } else {
            y[offset_cond_probs + rank] = 0.0f;
            y[offset_cond_cdf + rank] = 0.0f;
        }
    }
    y[offset_kept_count] = float(kept_count);
    y[offset_denominator] = denominator;
    y[offset_retained_mass] = retained_mass;

    // A shared-noise diagnostic, not a reproduction of MLX's seeded RNG.
    // Uniforms are indexed by stable native top-k rank and only retained
    // candidates participate; this is equivalent to masking dropped tokens
    // to -infinity before a categorical Gumbel-max.
    for (uint draw = 0; draw < draw_count; ++draw) {
        float best_score = -INFINITY;
        uint best_id = ids[0];
        for (uint rank = 0; rank < k; ++rank) {
            if (y[offset_kept + rank] == 0.0f) continue;
            float u = noise_uniforms[draw * k + rank];
            if (!(u > 0.0f && u < 1.0f) || !isfinite(u)) {
                y[0] = NAN;
                return;
            }
            float gumbel = -log(-log(u));
            float score = values[rank] + gumbel;
            if (score > best_score) {
                best_score = score;
                best_id = ids[rank];
            }
        }
        y[offset_gumbel_ids + draw] = float(best_id);
    }
}
