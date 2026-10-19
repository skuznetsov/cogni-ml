#include <metal_stdlib>
using namespace metal;

// MPS LayerNorm compatibility, scoped to this Wan CausalAudio path. Pinned
// to PyTorch v2.8.0 aten/src/ATen/native/mps/kernels/LayerNorm.metal
// (SHA-256 1a289179208970a07e5da56d72c8533adb2549cea49427541c0178ded525d597).
// Preserve N_READS=4 grouping, SIMD partial sums, raw-moment variance, the
// backend's var<1e-6 clamp, and precise rsqrt; centered-variance LayerNorm is
// observably different on near-constant rows. Input/output may alias because
// every row's statistics are complete before any output lane is written.
kernel void wan_s2v_audio_layer_norm_mps(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &axis_size [[buffer(2)]],
    constant float &epsilon [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    constexpr uint SIMD_SIZE = 32;
    constexpr uint N_READS = 4;
    uint row_offset = row * axis_size;
    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    uint step = threads * N_READS;

    for (uint offset = 0; offset < axis_size; offset += step) {
        uint base = offset + tid * N_READS;
        if (base + N_READS <= axis_size) {
            float4 values = float4(input[row_offset + base], input[row_offset + base + 1],
                                   input[row_offset + base + 2], input[row_offset + base + 3]);
            partial_sum += values.x + values.y + values.z + values.w;
            partial_sum_sq += dot(values, values);
        } else if (base < axis_size) {
            uint remaining = axis_size - base;
            if (remaining >= 3) {
                float3 values = float3(input[row_offset + base], input[row_offset + base + 1],
                                       input[row_offset + base + 2]);
                partial_sum += values.x + values.y + values.z;
                partial_sum_sq += dot(values, values);
            } else if (remaining >= 2) {
                float2 values = float2(input[row_offset + base], input[row_offset + base + 1]);
                partial_sum += values.x + values.y;
                partial_sum_sq += dot(values, values);
            } else {
                float value = input[row_offset + base];
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

    float group_sum = simd_sum(partial_sum);
    float group_sum_sq = simd_sum(partial_sum_sq);
    if (lane == 0) {
        local_sums[simdgroup] = group_sum;
        local_sums_sq[simdgroup] = group_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float sum = simd_sum(local_sums[lane]);
        float sum_sq = simd_sum(local_sums_sq[lane]);
        if (lane == 0) {
            float mean = sum / float(axis_size);
            float variance = sum_sq / float(axis_size) - mean * mean;
            variance = variance < 1e-6f ? 0.0f : variance;
            row_mean = mean;
            row_inverse_std = metal::precise::rsqrt(variance + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mean = row_mean;
    float inverse_std = row_inverse_std;
    for (uint offset = 0; offset < axis_size; offset += step) {
        uint base = offset + tid * N_READS;
        for (uint i = 0; i < N_READS && base + i < axis_size; ++i) {
            uint index = row_offset + base + i;
            float value = input[index];
            output[index] = (value - mean) * inverse_std;
        }
    }
}

// `features` is [layers, channels, bucket_frames]. Match upstream's
// SiLU(layer_weights), normalization by the layer-weight sum, and reduction
// across the layer axis. The first `prefix_frames` outputs repeat bucket 0.
kernel void wan_s2v_audio_weighted(
    device const float *features [[buffer(0)]],
    device const float *layer_weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &layers [[buffer(3)]],
    constant uint &channels [[buffer(4)]],
    constant uint &bucket_frames [[buffer(5)]],
    constant uint &prefix_frames [[buffer(6)]],
    constant uint &output_frames [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint count = output_frames * channels;
    if (gid >= count) return;
    uint frame = gid / channels;
    uint channel = gid - frame * channels;
    uint source_frame = frame < prefix_frames ? 0 : frame - prefix_frames;
    float weight_sum = 0.0f;
    for (uint layer = 0; layer < layers; ++layer) {
        float value = layer_weights[layer];
        weight_sum += value / (1.0f + exp(-value));
    }
    float value_sum = 0.0f;
    for (uint layer = 0; layer < layers; ++layer) {
        float value = layer_weights[layer];
        float weight = value / (1.0f + exp(-value));
        uint feature_index = (layer * channels + channel) * bucket_frames + source_frame;
        float weighted = features[feature_index] * weight;
        weighted = weighted / weight_sum;
        value_sum += weighted;
    }
    output[gid] = value_sum;
}

// Causal Conv1d with upstream replicate-left padding (kernel size 3). Input
// rows are [sequences, time, channels], output rows have the same sequence
// organization. `weight_groups > 1` is the initial local conv: each token
// head owns a distinct contiguous output-channel group while reading the same
// single weighted input sequence. Later convolutions share one weight set.
kernel void wan_s2v_audio_causal_conv1d(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &input_frames [[buffer(4)]],
    constant uint &output_frames [[buffer(5)]],
    constant uint &input_channels [[buffer(6)]],
    constant uint &output_channels [[buffer(7)]],
    constant uint &sequences [[buffer(8)]],
    constant uint &weight_groups [[buffer(9)]],
    constant uint &stride [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    uint count = sequences * output_frames * output_channels;
    if (gid >= count) return;
    uint output_channel = gid % output_channels;
    uint output_frame = (gid / output_channels) % output_frames;
    uint sequence = gid / (output_channels * output_frames);
    uint input_sequence = (weight_groups == sequences && sequences > 1) ? 0 : sequence;
    uint weight_group = weight_groups > 1 ? sequence : 0;
    float sum = bias[weight_group * output_channels + output_channel];
    for (uint input_channel = 0; input_channel < input_channels; ++input_channel) {
        for (uint kernel_index = 0; kernel_index < 3; ++kernel_index) {
            int source_frame = int(output_frame * stride + kernel_index) - 2;
            if (source_frame < 0) source_frame = 0;
            uint input_index = (input_sequence * input_frames + uint(source_frame)) * input_channels + input_channel;
            uint weight_index = ((weight_group * output_channels + output_channel) * input_channels + input_channel) * 3 + kernel_index;
            sum += input[input_index] * weight[weight_index];
        }
    }
    output[gid] = sum;
}

kernel void wan_s2v_audio_silu(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    float value = input[gid];
    output[gid] = value / (1.0f + exp(-value));
}

// Rearrange local `[tokens,time,channels]` into Wan `[time,tokens+pad,channels]`
// while removing the initial motion-latent frames. The learned pad token is
// broadcast without materializing a repeated tensor.
kernel void wan_s2v_audio_pack_local(
    device const float *motion [[buffer(0)]],
    device const float *pad [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &motion_frames [[buffer(3)]],
    constant uint &video_frames [[buffer(4)]],
    constant uint &drop_frames [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    constant uint &channels [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    uint slots = tokens + 1;
    uint count = video_frames * slots * channels;
    if (gid >= count) return;
    uint channel = gid % channels;
    uint slot = (gid / channels) % slots;
    uint frame = gid / (channels * slots);
    uint motion_frame = frame + drop_frames;
    if (slot < tokens) {
        uint source_index = (slot * motion_frames + motion_frame) * channels + channel;
        output[gid] = motion[source_index];
    } else {
        output[gid] = pad[channel];
    }
}

kernel void wan_s2v_audio_crop_global(
    device const float *motion [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &motion_frames [[buffer(2)]],
    constant uint &video_frames [[buffer(3)]],
    constant uint &drop_frames [[buffer(4)]],
    constant uint &channels [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    uint count = video_frames * channels;
    if (gid >= count) return;
    uint frame = gid / channels;
    uint channel = gid - frame * channels;
    output[gid] = motion[(frame + drop_frames) * channels + channel];
}
