#include <metal_stdlib>
using namespace metal;

// Deliberately poison the previous display output before each complete replay.
// Matching the oracle then also checks that all timeline pixels were rewritten.
kernel void pixelumm_decoder_replay_poison(device float *out [[buffer(0)]],
                                           constant uint &count [[buffer(1)]],
                                           uint i [[thread_position_in_grid]]) {
    if (i < count) out[i] = as_type<float>(0x7fc00000u);
}
