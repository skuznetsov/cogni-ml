#include <metal_stdlib>
using namespace metal;

// Wav2Vec2's LayerNorm runs through PyTorch's MPS LayerNorm kernel. Match the
// PyTorch v2.8.0 LayerNorm.metal reduction contract (SHA-256
// 1a289179208970a07e5da56d72c8533adb2549cea49427541c0178ded525d597): four
// values per thread, SIMD partial sums, raw-moment variance with the 1e-6
// backend clamp, and precise rsqrt. Each row's statistics are complete before
// any output lane is written, so the XLSR path may normalize in place.
kernel void wan_audio_layer_norm_mps(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    constant uint &axis_size [[buffer(4)]],
    constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    constexpr uint SIMD_SIZE = 32;
    constexpr uint N_READS = 4;
    const uint row_offset = row * axis_size;
    const uint step = threads * N_READS;

    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    for (uint offset = 0; offset < axis_size; offset += step) {
        const uint base = offset + tid * N_READS;
        if (base + N_READS <= axis_size) {
            const float4 values = float4(input[row_offset + base], input[row_offset + base + 1],
                                         input[row_offset + base + 2], input[row_offset + base + 3]);
            partial_sum += values.x + values.y + values.z + values.w;
            partial_sum_sq += dot(values, values);
        } else if (base < axis_size) {
            const uint remaining = axis_size - base;
            if (remaining >= 3) {
                const float3 values = float3(input[row_offset + base], input[row_offset + base + 1],
                                             input[row_offset + base + 2]);
                partial_sum += values.x + values.y + values.z;
                partial_sum_sq += dot(values, values);
            } else if (remaining >= 2) {
                const float2 values = float2(input[row_offset + base], input[row_offset + base + 1]);
                partial_sum += values.x + values.y;
                partial_sum_sq += dot(values, values);
            } else {
                const float value = input[row_offset + base];
                partial_sum += value;
                partial_sum_sq = fma(value, value, partial_sum_sq);
            }
        }
    }

    threadgroup float local_sums[SIMD_SIZE];
    threadgroup float local_sums_sq[SIMD_SIZE];
    threadgroup float row_mean;
    threadgroup float row_inverse_std;
    if (simdgroup == 0) {
        local_sums[lane] = 0.0f;
        local_sums_sq[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float group_sum = simd_sum(partial_sum);
    const float group_sum_sq = simd_sum(partial_sum_sq);
    if (lane == 0) {
        local_sums[simdgroup] = group_sum;
        local_sums_sq[simdgroup] = group_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const float sum = simd_sum(local_sums[lane]);
        const float sum_sq = simd_sum(local_sums_sq[lane]);
        if (lane == 0) {
            const float mean = sum / float(axis_size);
            float variance = sum_sq / float(axis_size) - mean * mean;
            variance = variance < 1e-6f ? 0.0f : variance;
            row_mean = mean;
            row_inverse_std = metal::precise::rsqrt(variance + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float mean = row_mean;
    const float inverse_std = row_inverse_std;
    for (uint offset = 0; offset < axis_size; offset += step) {
        const uint base = offset + tid * N_READS;
        for (uint i = 0; i < N_READS && base + i < axis_size; ++i) {
            const uint index = row_offset + base + i;
            float normalized = (input[index] - mean) * inverse_std;
            normalized *= weight[base + i];
            normalized += bias[base + i];
            output[index] = normalized;
        }
    }
}

// MSL does not expose erf on the supported compiler. This Abramowitz-Stegun
// approximation has <2e-7 absolute error over the real line, preserving the
// exact (erf-form) GELU used by Wav2Vec2 within the strict parity gate.
inline float wan_audio_erf(float value) {
    const float sign = value < 0.0f ? -1.0f : 1.0f;
    const float x = abs(value);
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float polynomial = (((((1.061405429f * t - 1.453152027f) * t)
                              + 1.421413741f) * t - 0.284496736f)
                              * t + 0.254829592f) * t;
    return sign * (1.0f - polynomial * exp(-x * x));
}

// PyTorch 2.8.0 MPS `gelu_out_mps` builds exact GELU as separate graph ops:
// multiply by 1/sqrt(2), erf, add 1, multiply by 1/2, then multiply by x.
// Keep the same operation order for the XLSR feature extractor and avoid
// reassociating this into (x/2) * (1 + erf), which rounds differently.
inline float wan_audio_gelu_mps_order(float value) {
    volatile float erf_argument = value * 0.7071067811865475244f;
    volatile float erf_value = wan_audio_erf(erf_argument);
    volatile float cdf = erf_value + 1.0f;
    cdf = cdf * 0.5f;
    return cdf * value;
}

kernel void wan_audio_gelu_mps(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = wan_audio_gelu_mps_order(x[id]);
}

// Transformers Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm: population
// variance and epsilon 1e-7. One threadgroup computes scalar moments and then
// writes the normalized waveform; no host-side waveform intermediate exists.
kernel void wan_audio_normalize(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], constant float &eps [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    threadgroup float partials[8], mean, inverse;
    float sum = 0.0f;
    for (uint i = tid; i < count; i += threads) sum += x[i];
    float reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        mean = total / float(count);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = 0.0f;
    for (uint i = tid; i < count; i += threads) {
        const float centered = x[i] - mean;
        sum += centered * centered;
    }
    reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        inverse = rsqrt(total / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < count; i += threads) y[i] = (x[i] - mean) * inverse;
}

// Conv1d with PyTorch [out_channels, in_channels, kernel] weight layout and
// valid padding. The output is sequence-major [time, channel] for the native
// layer norm, attention and projection kernels.
kernel void wan_audio_conv1d(
    device const float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &input_len [[buffer(4)]], constant uint &output_len [[buffer(5)]],
    constant uint &input_channels [[buffer(6)]], constant uint &output_channels [[buffer(7)]],
    constant uint &kernel_size [[buffer(8)]], constant uint &stride [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= output_len * output_channels) return;
    const uint time = id / output_channels;
    const uint channel = id % output_channels;
    float value = 0.0f;
    const uint weight_base = channel * input_channels * kernel_size;
    for (uint in_channel = 0; in_channel < input_channels; ++in_channel) {
        const uint x_base = time * stride * input_channels + in_channel;
        const uint w_base = weight_base + in_channel * kernel_size;
        for (uint tap = 0; tap < kernel_size; ++tap) {
            value = fma(x[x_base + tap * input_channels], weight[w_base + tap], value);
        }
    }
    // PyTorch 2.8.0 MPS encodes convolution and broadcast bias as separate
    // MPSGraph operations; do not seed the convolution accumulator with bias.
    y[id] = value + bias[channel];
}

// Weight normalization is `dim=2` in the pinned Hugging Face implementation:
// one scale per kernel tap, normalized over output and grouped input channels.
kernel void wan_audio_position_scale(
    device const float *weight_v [[buffer(0)]], device const float *weight_g [[buffer(1)]],
    device float *scale [[buffer(2)]], constant uint &hidden [[buffer(3)]],
    constant uint &group_dim [[buffer(4)]], constant uint &kernel_size [[buffer(5)]],
    uint tap [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partials[8], norm;
    const uint count = hidden * group_dim;
    float sum = 0.0f;
    for (uint i = tid; i < count; i += threads) {
        const float v = weight_v[i * kernel_size + tap];
        sum += v * v;
    }
    const float reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        norm = rsqrt(total);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) scale[tap] = weight_g[tap] * norm;
}

// Grouped positional convolution from Wav2Vec2PositionalConvEmbedding. Even
// kernels use symmetric padding k//2 and Transformers SamePad removes the final
// extra output position; this loop computes only the retained T outputs.
kernel void wan_audio_position_conv(
    device const float *x [[buffer(0)]], device const float *weight_v [[buffer(1)]],
    device const float *scale [[buffer(2)]], device const float *bias [[buffer(3)]],
    device float *raw [[buffer(4)]], constant uint &frames [[buffer(5)]],
    constant uint &hidden [[buffer(6)]], constant uint &group_dim [[buffer(7)]],
    constant uint &kernel_size [[buffer(8)]], device float *y [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= frames * hidden) return;
    const uint time = id / hidden;
    const uint channel = id % hidden;
    const uint group_start = (channel / group_dim) * group_dim;
    const uint weight_base = channel * group_dim * kernel_size;
    const int padding = int(kernel_size / 2);
    float value = bias[channel];
    for (uint in_channel = 0; in_channel < group_dim; ++in_channel) {
        const uint in_index = group_start + in_channel;
        const uint w_base = weight_base + in_channel * kernel_size;
        for (uint tap = 0; tap < kernel_size; ++tap) {
            const int source_time = int(time) + int(tap) - padding;
            if (source_time >= 0 && source_time < int(frames)) {
                const float w = weight_v[w_base + tap] * scale[tap];
                value = fma(x[uint(source_time) * hidden + in_index], w, value);
            }
        }
    }
    raw[id] = value;
    // Wav2Vec2PositionalConvEmbedding applies configured GELU after SamePad.
    y[id] = wan_audio_gelu_mps_order(value);
}

inline uint wan_audio_round_sample(uint bucket, uint video_fps, uint target_fps) {
    const uint numerator = bucket * video_fps;
    uint quotient = numerator / target_fps;
    const uint remainder = numerator % target_fps;
    const uint twice = 2 * remainder;
    if (twice > target_fps || (twice == target_fps && (quotient & 1u) != 0u)) ++quotient;
    return quotient;
}

// Match Wan's align_corners=True interpolation to 30 fps followed by fixed
// 16-fps bucket selection. Padded source indices are written as zero, exactly
// as get_audio_embed_bucket_fps does when an index exceeds the real audio.
// Keep the operation order of PyTorch 2.8's MPS upsample_linear1d path:
// UpSample.mm computes one F32 scale, and UpSample.metal computes
// real_x = scale * output_x then x * v1 + (1 - x) * v0.
kernel void wan_audio_bucket(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &feature_frames [[buffer(2)]], constant uint &interpolation_frames [[buffer(3)]],
    constant uint &hidden [[buffer(4)]], constant uint &bucket_frames [[buffer(5)]],
    constant uint &video_fps [[buffer(6)]], constant uint &target_fps [[buffer(7)]],
    constant uint &layer [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    if (id >= hidden * bucket_frames) return;
    const uint channel = id / bucket_frames;
    const uint bucket = id % bucket_frames;
    const uint source_index = wan_audio_round_sample(bucket, video_fps, target_fps);
    float value = 0.0f;
    if (source_index < interpolation_frames) {
        const float scale = float(feature_frames - 1) / float(interpolation_frames - 1);
        const float position = scale * float(source_index);
        const uint lower = uint(floor(position));
        const uint upper = min(lower + 1, feature_frames - 1);
        const float fraction = fract(position);
        const float a = x[lower * hidden + channel];
        const float b = x[upper * hidden + channel];
        value = fraction * b + (1.0f - fraction) * a;
    }
    y[(layer * hidden + channel) * bucket_frames + bucket] = value;
}
