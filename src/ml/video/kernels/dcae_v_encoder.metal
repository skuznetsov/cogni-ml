#include <metal_stdlib>
using namespace metal;

inline float dcae_v_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort dcae_v_f32_to_bf16(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return ushort((bits >> 16) | 0x0040u);
    }
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float dcae_v_load(device const ushort *values, uint index) {
    return dcae_v_bf16_to_f32(values[index]);
}

inline float dcae_v_conv_value(
    device const ushort *x, device const ushort *cache,
    device const ushort *weight, uint batch,
    uint input_t, uint input_h, uint input_w,
    uint source_frames, uint input_frame_offset, uint cache_valid,
    uint output_t, uint output_y, uint output_x,
    uint kernel_t, uint kernel_h, uint kernel_w,
    uint out_channel, uint in_channels) {
    float accum = 0.0f;
    const uint plane = input_h * input_w;
    for (uint ic = 0; ic < in_channels; ++ic) {
        for (uint kt = 0; kt < kernel_t; ++kt) {
            const int joined_t = int(output_t) + int(kt) - 1;
            const bool from_cache = joined_t < 0;
            if (from_cache && cache_valid == 0) continue;
            if (!from_cache && joined_t >= int(input_t)) continue;
            const uint it = from_cache ? 0u : uint(joined_t);
            for (uint kh = 0; kh < kernel_h; ++kh) {
                const int iy = int(output_y) + int(kh) - 1;
                if (iy < 0 || iy >= int(input_h)) continue;
                for (uint kw = 0; kw < kernel_w; ++kw) {
                    const int ix = int(output_x) + int(kw) - 1;
                    if (ix < 0 || ix >= int(input_w)) continue;
                    const uint input_index = from_cache
                        ? ((batch * in_channels + ic) * plane + uint(iy) * input_w + uint(ix))
                        : (((batch * in_channels + ic) * source_frames +
                            input_frame_offset + it) * plane + uint(iy) * input_w + uint(ix));
                    const uint weight_index =
                        (((out_channel * in_channels + ic) * kernel_t + kt) * kernel_h + kh) * kernel_w + kw;
                    const float input_value = from_cache ? dcae_v_load(cache, input_index)
                                                         : dcae_v_load(x, input_index);
                    accum = fma(input_value, dcae_v_load(weight, weight_index), accum);
                }
            }
        }
    }
    return accum;
}

// Upstream converts pixels to BF16, divides in BF16, subtracts in BF16, then
// expands the still image over time. Keep those three operation boundaries.
kernel void dcae_v_normalize_repeat_bf16(
    device const uchar *rgb [[buffer(0)]], device ushort *clip [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &frames [[buffer(3)]],
    constant uint &height [[buffer(4)]], constant uint &width [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    const uint spatial = height * width;
    const uint count = batches * 3u * frames * spatial;
    if (id >= count) return;
    const uint channel = (id / (frames * spatial)) % 3u;
    const uint pixel = id % spatial;
    const float source = float(rgb[pixel * 3u + channel]);
    const float bf16_source = dcae_v_bf16_to_f32(dcae_v_f32_to_bf16(source));
    const float divided = dcae_v_bf16_to_f32(dcae_v_f32_to_bf16(bf16_source / 127.5f));
    const float normalized = dcae_v_bf16_to_f32(dcae_v_f32_to_bf16(divided - 1.0f));
    clip[id] = dcae_v_f32_to_bf16(normalized);
}

// Chunk-causal 3-D cross-correlation. Each invocation handles one 40-frame
// input tile (or the derived 10-frame latent-rate tile), reads one preceding
// BF16 cache frame when available, and writes BF16 after FP32 accumulation.
kernel void dcae_v_conv3d_bf16(
    device const ushort *x [[buffer(0)]], device const ushort *cache [[buffer(1)]],
    device const ushort *weight [[buffer(2)]], device const ushort *bias [[buffer(3)]],
    device ushort *y [[buffer(4)]],
    constant uint &batches [[buffer(5)]], constant uint &in_channels [[buffer(6)]],
    constant uint &out_channels [[buffer(7)]], constant uint &input_t [[buffer(8)]],
    constant uint &source_frames [[buffer(9)]], constant uint &input_h [[buffer(10)]],
    constant uint &input_w [[buffer(11)]], constant uint &input_frame_offset [[buffer(12)]],
    constant uint &cache_valid [[buffer(13)]], constant uint &use_silu [[buffer(14)]],
    constant uint &output_total_frames [[buffer(15)]], constant uint &output_frame_offset [[buffer(16)]],
    constant uint &has_bias [[buffer(17)]],
    uint id [[thread_position_in_grid]]) {
    const uint output_h = input_h;
    const uint output_w = input_w;
    const uint count = batches * out_channels * input_t * output_h * output_w;
    if (id >= count) return;
    const uint ox = id % output_w;
    const uint oy = (id / output_w) % output_h;
    const uint ot = (id / (output_w * output_h)) % input_t;
    const uint oc = (id / (output_w * output_h * input_t)) % out_channels;
    const uint batch = id / (output_w * output_h * input_t * out_channels);
    const float convolution = dcae_v_conv_value(x, cache, weight, batch,
        input_t, input_h, input_w, source_frames, input_frame_offset, cache_valid,
        ot, oy, ox, 3u, 3u, 3u, oc, in_channels);
    ushort rounded = dcae_v_f32_to_bf16(convolution);
    if (has_bias != 0) {
        const float biased = dcae_v_bf16_to_f32(rounded) + dcae_v_load(bias, oc);
        rounded = dcae_v_f32_to_bf16(biased);
    }
    if (use_silu != 0) {
        const float conv_result = dcae_v_bf16_to_f32(rounded);
        rounded = dcae_v_f32_to_bf16(conv_result / (1.0f + precise::exp(-conv_result)));
    }
    const uint destination = ((batch * out_channels + oc) * output_total_frames +
        output_frame_offset + ot) * output_h * output_w + oy * output_w + ox;
    y[destination] = rounded;
}

// ConvPixelUnshuffle, with the optional channel-averaging shortcut, is fused
// so the full-resolution intermediate never becomes a resident activation.
kernel void dcae_v_conv_unshuffle_bf16(
    device const ushort *x [[buffer(0)]], device const ushort *cache [[buffer(1)]],
    device const ushort *weight [[buffer(2)]], device const ushort *bias [[buffer(3)]],
    device ushort *y [[buffer(4)]],
    constant uint &batches [[buffer(5)]], constant uint &in_channels [[buffer(6)]],
    constant uint &out_channels [[buffer(7)]], constant uint &input_t [[buffer(8)]],
    constant uint &source_frames [[buffer(9)]], constant uint &input_h [[buffer(10)]],
    constant uint &input_w [[buffer(11)]], constant uint &input_frame_offset [[buffer(12)]],
    constant uint &cache_valid [[buffer(13)]], constant uint &spatial_factor [[buffer(14)]],
    constant uint &temporal_factor [[buffer(15)]], constant uint &output_t [[buffer(16)]],
    constant uint &output_total_frames [[buffer(17)]], constant uint &output_frame_offset [[buffer(18)]],
    constant uint &shortcut_enabled [[buffer(19)]], uint id [[thread_position_in_grid]]) {
    const uint output_h = input_h / spatial_factor;
    const uint output_w = input_w / spatial_factor;
    const uint count = batches * out_channels * output_t * output_h * output_w;
    if (id >= count) return;
    const uint ox = id % output_w;
    const uint oy = (id / output_w) % output_h;
    const uint ot = (id / (output_w * output_h)) % output_t;
    const uint oc = (id / (output_w * output_h * output_t)) % out_channels;
    const uint batch = id / (output_w * output_h * output_t * out_channels);
    const uint factor = spatial_factor * spatial_factor * temporal_factor;
    const uint unshuffle_index = oc % factor;
    const uint conv_channel = oc / factor;
    const uint conv_t = ot * temporal_factor + unshuffle_index / (spatial_factor * spatial_factor);
    const uint conv_y = oy * spatial_factor + (unshuffle_index / spatial_factor) % spatial_factor;
    const uint conv_x = ox * spatial_factor + unshuffle_index % spatial_factor;
    const float convolution = dcae_v_conv_value(x, cache, weight, batch,
        input_t, input_h, input_w, source_frames, input_frame_offset, cache_valid,
        conv_t, conv_y, conv_x, 3u, 3u, 3u, conv_channel, in_channels);
    ushort rounded = dcae_v_f32_to_bf16(convolution);
    const float biased = dcae_v_bf16_to_f32(rounded) + dcae_v_load(bias, conv_channel);
    rounded = dcae_v_f32_to_bf16(biased);
    if (shortcut_enabled != 0) {
        const uint group_size = in_channels * factor / out_channels;
        const uint input_plane = input_h * input_w;
        float shortcut_sum = 0.0f;
        for (uint group = 0; group < group_size; ++group) {
            const uint source_unshuffle = oc * group_size + group;
            const uint source_channel = source_unshuffle / factor;
            const uint source_factor = source_unshuffle % factor;
            const uint source_t = ot * temporal_factor + source_factor / (spatial_factor * spatial_factor);
            const uint source_y = oy * spatial_factor + (source_factor / spatial_factor) % spatial_factor;
            const uint source_x = ox * spatial_factor + source_factor % spatial_factor;
            const uint source_index = ((batch * in_channels + source_channel) * source_frames +
                input_frame_offset + source_t) * input_plane + source_y * input_w + source_x;
            shortcut_sum += dcae_v_load(x, source_index);
        }
        const float shortcut = dcae_v_bf16_to_f32(
            dcae_v_f32_to_bf16(shortcut_sum / float(group_size)));
        const float main_value = dcae_v_bf16_to_f32(rounded);
        rounded = dcae_v_f32_to_bf16(main_value + shortcut);
    }
    const uint destination = ((batch * out_channels + oc) * output_total_frames +
        output_frame_offset + ot) * output_h * output_w + oy * output_w + ox;
    y[destination] = rounded;
}

// The upstream RMSNorm3d is expressed as eager BF16 elementwise operations.
// Preserve each BF16 rounding boundary while accumulating the channel mean in F32.
kernel void dcae_v_rms_norm_bf16_inplace(
    device ushort *x [[buffer(0)]], device const ushort *weight [[buffer(1)]],
    device const ushort *bias [[buffer(2)]], constant uint &batches [[buffer(3)]],
    constant uint &channels [[buffer(4)]], constant uint &frames [[buffer(5)]],
    constant uint &height [[buffer(6)]], constant uint &width [[buffer(7)]],
    constant float &epsilon [[buffer(8)]], uint location [[thread_position_in_grid]]) {
    const uint spatial = frames * height * width;
    if (location >= batches * spatial) return;
    const uint base = (location / spatial) * channels * spatial + location % spatial;
    float square_sum = 0.0f;
    for (uint channel = 0; channel < channels; ++channel) {
        const uint index = base + channel * spatial;
        const float input_value = dcae_v_load(x, index);
        const float square = dcae_v_bf16_to_f32(dcae_v_f32_to_bf16(input_value * input_value));
        square_sum += square;
    }
    const float mean_square = dcae_v_bf16_to_f32(
        dcae_v_f32_to_bf16(square_sum / float(channels)));
    const float stabilized = dcae_v_bf16_to_f32(
        dcae_v_f32_to_bf16(mean_square + epsilon));
    const float denominator = dcae_v_bf16_to_f32(
        dcae_v_f32_to_bf16(sqrt(stabilized)));
    for (uint channel = 0; channel < channels; ++channel) {
        const uint index = base + channel * spatial;
        const float input_value = dcae_v_load(x, index);
        const float divided = dcae_v_bf16_to_f32(
            dcae_v_f32_to_bf16(input_value / denominator));
        const float scaled = dcae_v_bf16_to_f32(
            dcae_v_f32_to_bf16(divided * dcae_v_load(weight, channel)));
        x[index] = dcae_v_f32_to_bf16(scaled + dcae_v_load(bias, channel));
    }
}

// This is intentionally in-place on the main branch; each thread owns exactly
// one element and the residual buffer is disjoint.
kernel void dcae_v_residual_add_bf16_inplace(
    device ushort *main [[buffer(0)]], device const ushort *shortcut [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    main[id] = dcae_v_f32_to_bf16(dcae_v_load(main, id) + dcae_v_load(shortcut, id));
}

kernel void dcae_v_cache_last_frame_bf16(
    device const ushort *x [[buffer(0)]], device ushort *cache [[buffer(1)]],
    constant uint &channels [[buffer(2)]], constant uint &frames [[buffer(3)]],
    constant uint &source_frames [[buffer(4)]], constant uint &input_frame_offset [[buffer(5)]],
    constant uint &height [[buffer(6)]], constant uint &width [[buffer(7)]],
    constant uint &batches [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    const uint plane = height * width;
    const uint count = batches * channels * plane;
    if (id >= count) return;
    const uint batch_channel = id / plane;
    const uint spatial = id % plane;
    const uint source = (batch_channel * source_frames + input_frame_offset + frames - 1) * plane + spatial;
    cache[id] = x[source];
}

kernel void dcae_v_copy_bf16(
    device const ushort *source [[buffer(0)]], device ushort *destination [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    if (id < count) destination[id] = source[id];
}

kernel void dcae_v_scatter_latent_bf16(
    device const ushort *source [[buffer(0)]], device ushort *destination [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &channels [[buffer(3)]],
    constant uint &tile_frames [[buffer(4)]], constant uint &full_frames [[buffer(5)]],
    constant uint &height [[buffer(6)]], constant uint &width [[buffer(7)]],
    constant uint &frame_offset [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    const uint plane = height * width;
    const uint tile_values = batches * channels * tile_frames * plane;
    if (id >= tile_values) return;
    const uint t = (id / plane) % tile_frames;
    const uint channel = (id / (plane * tile_frames)) % channels;
    const uint batch = id / (plane * tile_frames * channels);
    const uint spatial = id % plane;
    const uint target = ((batch * channels + channel) * full_frames + frame_offset + t) * plane + spatial;
    destination[target] = source[id];
}
