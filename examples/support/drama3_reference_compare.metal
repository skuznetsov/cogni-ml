#include <metal_stdlib>
using namespace metal;

// The existing backbone gate is pointwise: atol + rtol * abs(reference).
// Read only these reduced metrics after the ordinary generation boundaries.
kernel void dr3_reference_pointwise_gate(
    device const float *actual [[buffer(0)]],
    device const float *reference [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ulong &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    float failures = 0.0f;
    float nonfinite = 0.0f;
    float worst_ratio = 0.0f;
    for (ulong i = 0; i < count; ++i) {
        float a = actual[i];
        float r = reference[i];
        if (!isfinite(a) || !isfinite(r)) {
            failures += 1.0f;
            nonfinite += 1.0f;
        } else {
            float error = abs(a - r);
            float allowed = 3e-5f + 3e-4f * abs(r);
            failures += error > allowed ? 1.0f : 0.0f;
            worst_ratio = max(worst_ratio, error / allowed);
        }
    }
    output[0] = failures;
    output[1] = nonfinite;
    output[2] = worst_ratio;
}
