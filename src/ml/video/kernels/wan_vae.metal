#include <metal_stdlib>
using namespace metal;

// Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
// Apache-2.0. Adapted from Wan2.1_VAE at Wan2.2 commit
// 1ea34ff48f87168174e12956e200b1d908b1c5ff.

inline float wan_vae_weight(device const float *weight32,
                            device const ushort *weight16,
                            uint index, uint is_bf16) {
    return is_bf16 ? as_type<float>(uint(weight16[index]) << 16) : weight32[index];
}

// NCDHW cross-correlation. `cache` contains 0..2 compact preceding frames;
// left padding is reduced by that history length for causal convolutions.
// kT=1 also implements the frame-independent Conv2d layers.
kernel void wan_vae_conv3d_f32(
    device const float *x [[buffer(0)]], device const float *cache [[buffer(1)]],
    device const float *weight32 [[buffer(2)]], device const ushort *weight16 [[buffer(3)]],
    device const float *bias32 [[buffer(4)]], device const ushort *bias16 [[buffer(5)]],
    device float *y [[buffer(6)]],
    constant uint &batches [[buffer(7)]], constant uint &in_channels [[buffer(8)]],
    constant uint &out_channels [[buffer(9)]], constant uint &input_t [[buffer(10)]],
    constant uint &input_h [[buffer(11)]], constant uint &input_w [[buffer(12)]],
    constant uint &cache_t [[buffer(13)]], constant uint &kernel_t [[buffer(14)]],
    constant uint &kernel_h [[buffer(15)]], constant uint &kernel_w [[buffer(16)]],
    constant uint &stride_t [[buffer(17)]], constant uint &stride_h [[buffer(18)]],
    constant uint &stride_w [[buffer(19)]], constant uint &pad_t [[buffer(20)]],
    constant uint &pad_h [[buffer(21)]], constant uint &pad_w [[buffer(22)]],
    constant uint &output_t [[buffer(23)]], constant uint &output_h [[buffer(24)]],
    constant uint &output_w [[buffer(25)]], constant uint &weight_bf16 [[buffer(26)]],
    constant uint &bias_bf16 [[buffer(27)]], constant uint &output_offset [[buffer(28)]],
    uint local_id [[thread_position_in_grid]]) {
    const uint id = output_offset + local_id;
    const uint count = batches * out_channels * output_t * output_h * output_w;
    if (id >= count) return;
    const uint ox = id % output_w;
    const uint oy = (id / output_w) % output_h;
    const uint ot = (id / (output_w * output_h)) % output_t;
    const uint oc = (id / (output_w * output_h * output_t)) % out_channels;
    const uint batch = id / (output_w * output_h * output_t * out_channels);
    float accumulators[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                             0.0f, 0.0f, 0.0f, 0.0f};
    uint ordinal = 0;
    for (uint ic = 0; ic < in_channels; ++ic) {
        for (uint kt = 0; kt < kernel_t; ++kt) {
            const int joined_t = int(ot * stride_t + kt) - int(pad_t);
            if (joined_t < 0 || joined_t >= int(cache_t + input_t)) continue;
            const bool from_cache = uint(joined_t) < cache_t;
            const uint it = from_cache ? uint(joined_t) : uint(joined_t) - cache_t;
            for (uint kh = 0; kh < kernel_h; ++kh) {
                const int iy = int(oy * stride_h + kh) - int(pad_h);
                if (iy < 0 || iy >= int(input_h)) continue;
                for (uint kw = 0; kw < kernel_w; ++kw) {
                    const int ix = int(ox * stride_w + kw) - int(pad_w);
                    if (ix < 0 || ix >= int(input_w)) continue;
                    const uint input_index = from_cache
                        ? ((((batch * in_channels + ic) * 2 + it) * input_h + uint(iy)) * input_w + uint(ix))
                        : ((((batch * in_channels + ic) * input_t + it) * input_h + uint(iy)) * input_w + uint(ix));
                    const uint weight_index = (((oc * in_channels + ic) * kernel_t + kt) * kernel_h + kh) * kernel_w + kw;
                    const float value = from_cache ? cache[input_index] : x[input_index];
                    const float weight = wan_vae_weight(weight32, weight16, weight_index, weight_bf16);
                    accumulators[ordinal & 7] = fma(value, weight, accumulators[ordinal & 7]);
                    ++ordinal;
                }
            }
        }
    }
    float sum = bias_bf16 ? wan_vae_weight(bias32, bias16, oc, 1) : bias32[oc];
    sum += ((accumulators[0] + accumulators[4]) + (accumulators[2] + accumulators[6])) +
           ((accumulators[1] + accumulators[5]) + (accumulators[3] + accumulators[7]));
    y[id] = sum;
}

kernel void wan_vae_rms_norm_f32(
    device const float *x [[buffer(0)]], device const float *gamma32 [[buffer(1)]],
    device const ushort *gamma16 [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &channels [[buffer(5)]],
    constant uint &frames [[buffer(6)]], constant uint &height [[buffer(7)]],
    constant uint &width [[buffer(8)]], constant uint &gamma_bf16 [[buffer(9)]],
    uint spatial_id [[thread_position_in_grid]]) {
    const uint spatial = frames * height * width;
    const uint locations = batches * spatial;
    if (spatial_id >= locations) return;
    const uint batch = spatial_id / spatial;
    const uint location = spatial_id % spatial;
    const uint base = batch * channels * spatial + location;
    float sum_sq = 0.0f;
    for (uint channel = 0; channel < channels; ++channel) {
        const float value = x[base + channel * spatial];
        sum_sq = fma(value, value, sum_sq);
    }
    const float inverse = rsqrt(max(sum_sq, 1.0e-24f));
    const float channel_scale = sqrt(float(channels));
    for (uint channel = 0; channel < channels; ++channel) {
        const uint index = base + channel * spatial;
        const float gamma = wan_vae_weight(gamma32, gamma16, channel, gamma_bf16);
        y[index] = ((x[index] * inverse) * channel_scale) * gamma;
    }
}

kernel void wan_vae_silu_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float value = x[id];
    y[id] = value / (1.0f + precise::exp(-value));
}

kernel void wan_vae_add_f32(
    device const float *x [[buffer(0)]], device const float *residual [[buffer(1)]],
    device float *y [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint id [[thread_position_in_grid]]) {
    if (id < count) y[id] = x[id] + residual[id];
}

// Cache buffers have fixed [B,C,2,H,W] storage and compact valid history at
// frame indices [0,history_len). A Rep call stores [zero,current] for the next
// temporal upsample, exactly as the upstream first-call branch does.
kernel void wan_vae_cache_last2_f32(
    device const float *x [[buffer(0)]], device const float *old_cache [[buffer(1)]],
    device float *new_cache [[buffer(2)]],
    constant uint &batches [[buffer(3)]], constant uint &channels [[buffer(4)]],
    constant uint &frames [[buffer(5)]], constant uint &height [[buffer(6)]],
    constant uint &width [[buffer(7)]], constant uint &old_count [[buffer(8)]],
    constant uint &replicate_zero [[buffer(9)]],
    constant uint &keep_last_only [[buffer(10)]], uint id [[thread_position_in_grid]]) {
    const uint frame_plane = batches * channels * height * width;
    const uint count = frame_plane * 2;
    if (id >= count) return;
    const uint pixels = height * width;
    const uint destination_frame = (id / pixels) % 2;
    const uint batch_channel = id / (2 * pixels);
    const uint pixel = id % pixels;
    uint new_count = min(2u, old_count + frames);
    uint source_frame;
    bool from_old;
    bool zero = false;
    if (keep_last_only != 0) {
        new_count = 1;
        source_frame = frames - 1;
        from_old = false;
    } else if (replicate_zero != 0) {
        new_count = 2;
        zero = destination_frame == 0;
        source_frame = frames - 1;
        from_old = false;
    } else {
        const uint start = old_count + frames - new_count;
        const uint combined_frame = start + destination_frame;
        from_old = combined_frame < old_count;
        source_frame = from_old ? combined_frame : combined_frame - old_count;
    }
    if (destination_frame >= new_count) {
        new_cache[id] = 0.0f;
    } else if (zero) {
        new_cache[id] = 0.0f;
    } else if (from_old) {
        new_cache[id] = old_cache[(batch_channel * 2 + source_frame) * pixels + pixel];
    } else {
        const uint source = (batch_channel * frames + source_frame) * height * width + pixel;
        new_cache[id] = x[source];
    }
}

kernel void wan_vae_nearest2x_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    const uint output_h = height * 2, output_w = width * 2;
    const uint count = batches * channels * frames * output_h * output_w;
    if (id >= count) return;
    const uint xcoord = id % output_w;
    const uint ycoord = (id / output_w) % output_h;
    const uint frame = (id / (output_w * output_h)) % frames;
    const uint channel = (id / (output_w * output_h * frames)) % channels;
    const uint batch = id / (output_w * output_h * frames * channels);
    const uint source = ((((batch * channels + channel) * frames + frame) * height + ycoord / 2) * width + xcoord / 2);
    y[id] = x[source];
}

kernel void wan_vae_temporal_shuffle_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]], uint id [[thread_position_in_grid]]) {
    const uint output_frames = frames * 2;
    const uint count = batches * channels * output_frames * height * width;
    if (id >= count) return;
    const uint pixel = id % (height * width);
    const uint ot = (id / (height * width)) % output_frames;
    const uint channel = (id / (height * width * output_frames)) % channels;
    const uint batch = id / (height * width * output_frames * channels);
    const uint part = ot % 2;
    const uint it = ot / 2;
    const uint input_channels = channels * 2;
    const uint source = ((((batch * input_channels + part * channels + channel) * frames + it) * height * width) + pixel);
    y[id] = x[source];
}

// Single-head, per-frame spatial SDPA. One threadgroup owns one query pixel;
// the channel reduction crosses simdgroups through the bounded partial array.
kernel void wan_vae_attention_f32(
    device const float *qkv [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &query_offset [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    const uint spatial = height * width;
    group += query_offset;
    if (group >= batches * frames * spatial) return;
    const uint query = group % spatial;
    const uint frame_batch = group / spatial;
    const uint frame = frame_batch % frames;
    const uint batch = frame_batch / frames;
    const uint plane = frames * spatial;
    const uint qbase = (batch * 3 * channels) * plane + frame * spatial + query;
    const float q = tid < channels ? qkv[qbase + tid * plane] : 0.0f;
    float accumulator = 0.0f;
    threadgroup float partials[16];
    threadgroup float correction;
    threadgroup float probability;
    threadgroup float inverse_sum;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (uint key = 0; key < spatial; ++key) {
        const uint kbase = (batch * 3 * channels + channels) * plane + frame * spatial + key;
        const float k = tid < channels ? qkv[kbase + tid * plane] : 0.0f;
        const float lane_dot = simd_sum(q * k);
        if (lane == 0) partials[sg] = lane_dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float dot = 0.0f;
            for (uint i = 0; i < threads / 32; ++i) dot += partials[i];
            const float score = dot * rsqrt(float(channels));
            const float next_max = max(running_max, score);
            correction = isinf(running_max) ? 0.0f : precise::exp(running_max - next_max);
            probability = precise::exp(score - next_max);
            running_sum = running_sum * correction + probability;
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < channels) {
            const uint vbase = (batch * 3 * channels + channels * 2) * plane + frame * spatial + key;
            accumulator = accumulator * correction + probability * qkv[vbase + tid * plane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) inverse_sum = 1.0f / running_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < channels) {
        const uint out = ((batch * channels + tid) * frames + frame) * spatial + query;
        y[out] = accumulator * inverse_sum;
    }
}

kernel void wan_vae_scale_f32(
    device const float *x [[buffer(0)]], device const float *mean [[buffer(1)]],
    device const float *factor [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &channels [[buffer(5)]],
    constant uint &frames [[buffer(6)]], constant uint &height [[buffer(7)]],
    constant uint &width [[buffer(8)]], constant uint &unscale [[buffer(9)]],
    uint id [[thread_position_in_grid]]) {
    const uint count = batches * channels * frames * height * width;
    if (id >= count) return;
    const uint channel = (id / (frames * height * width)) % channels;
    // Wan encode uses (x-mean)*inverse_std; decode uses z/inverse_std+mean.
    y[id] = unscale ? x[id] / factor[channel] + mean[channel]
                    : (x[id] - mean[channel]) * factor[channel];
}

kernel void wan_vae_split_scale_f32(
    device const float *x [[buffer(0)]], device const float *mean [[buffer(1)]],
    device const float *factor [[buffer(2)]], device float *y [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &channels [[buffer(5)]],
    constant uint &frames [[buffer(6)]], constant uint &height [[buffer(7)]],
    constant uint &width [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    const uint plane = frames * height * width;
    const uint count = batches * channels * plane;
    if (id >= count) return;
    const uint batch = id / (channels * plane);
    const uint channel = (id / plane) % channels;
    const uint spatial = id % plane;
    const uint source = (batch * channels * 2 + channel) * plane + spatial;
    y[id] = (x[source] - mean[channel]) * factor[channel];
}

kernel void wan_vae_copy_clamp_chunk_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &target_frames [[buffer(7)]],
    constant uint &frame_offset [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    const uint count = batches * channels * frames * height * width;
    if (id >= count || frame_offset + frames > target_frames) return;
    const uint pixel = id % (height * width);
    const uint frame = (id / (height * width)) % frames;
    const uint channel = (id / (height * width * frames)) % channels;
    const uint batch = id / (height * width * frames * channels);
    const uint destination = ((batch * channels + channel) * target_frames + frame_offset + frame) * height * width + pixel;
    y[destination] = clamp(x[id], -1.0f, 1.0f);
}

kernel void wan_vae_time_slice_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &frames [[buffer(4)]], constant uint &height [[buffer(5)]],
    constant uint &width [[buffer(6)]], constant uint &selected [[buffer(7)]],
    constant uint &slice_frames [[buffer(8)]],
    uint id [[thread_position_in_grid]]) {
    const uint count = batches * channels * slice_frames * height * width;
    if (id >= count) return;
    const uint frame = (id / (height * width)) % slice_frames;
    const uint batch = id / (channels * slice_frames * height * width);
    const uint channel = (id / (slice_frames * height * width)) % channels;
    const uint pixel = id % (height * width);
    const uint source = ((batch * channels + channel) * frames + selected + frame) * height * width + pixel;
    y[id] = x[source];
}

kernel void wan_vae_copy_frames_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &source_frames [[buffer(4)]], constant uint &source_offset [[buffer(5)]],
    constant uint &height [[buffer(6)]], constant uint &width [[buffer(7)]],
    constant uint &target_frames [[buffer(8)]], constant uint &target_offset [[buffer(9)]],
    constant uint &slice_frames [[buffer(10)]], uint id [[thread_position_in_grid]]) {
    const uint count = batches * channels * slice_frames * height * width;
    if (id >= count) return;
    const uint pixel = id % (height * width);
    const uint frame = (id / (height * width)) % slice_frames;
    const uint channel = (id / (height * width * slice_frames)) % channels;
    const uint batch = id / (height * width * slice_frames * channels);
    const uint source = ((batch * channels + channel) * source_frames + source_offset + frame) * height * width + pixel;
    const uint target = ((batch * channels + channel) * target_frames + target_offset + frame) * height * width + pixel;
    y[target] = x[source];
}

kernel void wan_vae_crop_frames_f32(
    device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &source_frames [[buffer(4)]], constant uint &source_offset [[buffer(5)]],
    constant uint &target_frames [[buffer(6)]], constant uint &height [[buffer(7)]],
    constant uint &width [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    const uint count = batches * channels * target_frames * height * width;
    if (id >= count) return;
    const uint pixel = id % (height * width);
    const uint frame = (id / (height * width)) % target_frames;
    const uint channel = (id / (height * width * target_frames)) % channels;
    const uint batch = id / (height * width * target_frames * channels);
    const uint source = ((batch * channels + channel) * source_frames + source_offset + frame) * height * width + pixel;
    y[id] = x[source];
}
