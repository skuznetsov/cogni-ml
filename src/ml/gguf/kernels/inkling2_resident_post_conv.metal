// Static causal convolution for a four-row decode block.
// One lane owns a channel, including every read and write of its history.
// All output rows read the old prefix before the latest raw rows replace it.
inline void inkling2_static_convolution_rows(
    device const float* weights, device float* history, device float* output,
    float4 raw, uint channel, uint channels, uint taps, uint rows, uint start,
    uint history_slots)
{
    for (uint row = 0; row < rows; ++row) {
        const uint position = start + row;
        float static_sum = 0.0f;
        for (uint tap = 0; tap < taps; ++tap) {
            const int sample_position = (int)position - (int)(taps - 1 - tap);
            float sample = 0.0f;
            if (sample_position >= 0) {
                if ((uint)sample_position >= start) {
                    sample = raw[(uint)sample_position - start];
                } else {
                    sample = history[((uint)sample_position % history_slots) * channels + channel];
                }
            }
            static_sum += sample * weights[channel * taps + tap];
        }
        const float sum = raw[row] + static_sum + 0.0f;
        output[row * channels + channel] = sum / sqrt((float)taps + 1.0f);
    }
    const uint first = rows > history_slots ? rows - history_slots : 0;
    for (uint row = first; row < rows; ++row) {
        history[((start + row) % history_slots) * channels + channel] = raw[row];
    }
}

kernel void inkling2_resident_q8_post_conv_batch(
    device const uint8_t* weights         [[buffer(0)]],
    device const float*   x               [[buffer(1)]],
    device       float*   output          [[buffer(2)]],
    device       float*   history         [[buffer(3)]],
    device const float*   taps_weights    [[buffer(4)]],
    constant     uint&    in_dim          [[buffer(5)]],
    constant     uint&    channels        [[buffer(6)]],
    constant     uint&    rows            [[buffer(7)]],
    constant     uint&    start           [[buffer(8)]],
    constant     uint&    taps            [[buffer(9)]],
    constant     uint&    history_slots   [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort group [[simdgroup_index_in_threadgroup]])
{
    const uint channel = tgpig.x * MV8_NSG + group;
    if (channel >= channels) return;

    const uint nb = in_dim / Q8_0_QK;
    device const block_q8_0_56* w_row =
        (device const block_q8_0_56*)(weights + (size_t)channel * nb * 34);
    float4 sums = float4(0.0f);
    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q8_0_56* block = w_row + ib;
        const float scale = (float)block->d;
        const float quant = (float)block->qs[lane];
        for (uint row = 0; row < 4; ++row) {
            if (row < rows) {
                const float y = x[(size_t)row * in_dim + (size_t)ib * Q8_0_QK + lane];
                sums[row] += scale * y * quant;
            }
        }
    }
    const float4 totals = float4(simd_sum(sums.x), simd_sum(sums.y), simd_sum(sums.z), simd_sum(sums.w));
    if (lane == 0) {
        inkling2_static_convolution_rows(taps_weights, history, output, totals,
                                        channel, channels, taps, rows, start, history_slots);
    }
}
