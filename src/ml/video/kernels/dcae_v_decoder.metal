#include <metal_stdlib>
using namespace metal;

inline float dcae_v_decoder_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort dcae_v_decoder_f32_to_bf16(float value) {
    const uint bits = as_type<uint>(value);
    const uint magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) {
        return ushort((bits >> 16) | 0x0040u);
    }
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Gather a contiguous temporal tile from a BF16 BCTHW latent bank into a
// contiguous BF16 BCTHW tile. `count` is B*C*tile_frames*height*width and
// `frame_offset + tile_frames` must not exceed `bank_frames`.
kernel void dcae_v_decoder_gather_latent_bf16(
    device const ushort *bank [[buffer(0)]],
    device ushort *tile [[buffer(1)]],
    constant uint &bank_frames [[buffer(2)]],
    constant uint &tile_frames [[buffer(3)]],
    constant uint &height [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &frame_offset [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint spatial = height * width;
    const uint channel_batch = id / (tile_frames * spatial);
    const uint local_frame = (id / spatial) % tile_frames;
    const uint source = (channel_batch * bank_frames + frame_offset + local_frame) *
        spatial + id % spatial;
    tile[id] = bank[source];
}

// PixelShuffle3d and the decoder's optional ChannelDuplicatingPixelShuffle
// shortcut in one pass. `expanded` is the BF16 Conv3d result laid out as
// [B,out_channels*temporal_factor*spatial_factor^2,T,H,W]. `shortcut` is the
// original low-resolution BCTHW activation. When enabled, its channel is
// `expanded_channel / repeat_interleave`, matching repeat_interleave(dim=1)
// before PixelShuffle3d. The residual sum has one BF16 rounding boundary.
// `shortcut` must still be bound to a valid buffer when the flag is zero; it is
// then not read. `count` is B*out_channels*(T*tf)*(H*sf)*(W*sf).
kernel void dcae_v_decoder_pixel_shuffle_bf16(
    device const ushort *expanded [[buffer(0)]],
    device const ushort *shortcut [[buffer(1)]],
    device ushort *output [[buffer(2)]],
    constant uint &expanded_channels [[buffer(3)]],
    constant uint &shortcut_channels [[buffer(4)]],
    constant uint &out_channels [[buffer(5)]],
    constant uint &input_frames [[buffer(6)]],
    constant uint &input_height [[buffer(7)]],
    constant uint &input_width [[buffer(8)]],
    constant uint &spatial_factor [[buffer(9)]],
    constant uint &temporal_factor [[buffer(10)]],
    constant uint &repeat_interleave [[buffer(11)]],
    constant uint &shortcut_enabled [[buffer(12)]],
    constant uint &count [[buffer(13)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;

    const uint output_height = input_height * spatial_factor;
    const uint output_width = input_width * spatial_factor;
    const uint output_frames = input_frames * temporal_factor;
    const uint output_plane = output_height * output_width;
    const uint output_frame = (id / output_plane) % output_frames;
    const uint output_y = (id / output_width) % output_height;
    const uint output_x = id % output_width;
    const uint output_channel = (id / (output_plane * output_frames)) % out_channels;
    const uint batch = id / (output_plane * output_frames * out_channels);

    const uint factor = spatial_factor * spatial_factor * temporal_factor;
    const uint phase = (output_frame % temporal_factor) * spatial_factor * spatial_factor +
        (output_y % spatial_factor) * spatial_factor + output_x % spatial_factor;
    const uint expanded_channel = output_channel * factor + phase;
    const uint input_frame = output_frame / temporal_factor;
    const uint input_y = output_y / spatial_factor;
    const uint input_x = output_x / spatial_factor;
    const uint input_plane = input_height * input_width;
    const uint expanded_index = ((batch * expanded_channels + expanded_channel) *
        input_frames + input_frame) * input_plane + input_y * input_width + input_x;

    ushort result = expanded[expanded_index];
    if (shortcut_enabled != 0) {
        const uint shortcut_channel = expanded_channel / repeat_interleave;
        const uint shortcut_index = ((batch * shortcut_channels + shortcut_channel) *
            input_frames + input_frame) * input_plane + input_y * input_width + input_x;
        const float sum = dcae_v_decoder_bf16_to_f32(result) +
            dcae_v_decoder_bf16_to_f32(shortcut[shortcut_index]);
        result = dcae_v_decoder_f32_to_bf16(sum);
    }
    output[id] = result;
}

// Scatter one BF16 BCTHW decoded tile into its non-overlapping temporal range
// in the full BF16 BCTHW output bank. `count` is B*C*tile_frames*height*width.
kernel void dcae_v_decoder_scatter_output_bf16(
    device const ushort *tile [[buffer(0)]],
    device ushort *bank [[buffer(1)]],
    constant uint &total_frames [[buffer(2)]],
    constant uint &tile_frames [[buffer(3)]],
    constant uint &height [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &frame_offset [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint spatial = height * width;
    const uint channel_batch = id / (tile_frames * spatial);
    const uint local_frame = (id / spatial) % tile_frames;
    const uint destination = (channel_batch * total_frames + frame_offset + local_frame) *
        spatial + id % spatial;
    bank[destination] = tile[id];
}

// Convert raw decoder BF16 BCTHW [-1,1] pixels to channels-first F32 [0,1].
// This preserves no extra BF16 boundary: the display affine and clamp are F32.
kernel void dcae_v_decoder_display_bf16_f32(
    device const ushort *raw [[buffer(0)]],
    device float *display [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float value = dcae_v_decoder_bf16_to_f32(raw[id]);
    display[id] = clamp(value * 0.5f + 0.5f, 0.0f, 1.0f);
}

// Convert raw decoder BF16 BCTHW [-1,1] pixels directly to interleaved RGB8
// [B,T,H,W,3]. Match terminal export's multiply-then-clamp integer conversion
// by truncating the clamped display value times 255.
kernel void dcae_v_decoder_display_bf16_rgb8(
    device const ushort *raw [[buffer(0)]],
    device uchar *rgb8 [[buffer(1)]],
    constant uint &batches [[buffer(2)]],
    constant uint &frames [[buffer(3)]],
    constant uint &height [[buffer(4)]],
    constant uint &width [[buffer(5)]],
    constant uint &count [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint channel = id % 3u;
    const uint pixel = id / 3u;
    const uint spatial = height * width;
    const uint frame = (pixel / spatial) % frames;
    const uint batch = pixel / (spatial * frames);
    if (batch >= batches) return;
    const uint y = (pixel / width) % height;
    const uint x = pixel % width;
    const uint source = ((batch * 3u + channel) * frames + frame) * spatial + y * width + x;
    const float value = dcae_v_decoder_bf16_to_f32(raw[source]);
    const float display = clamp(value * 0.5f + 0.5f, 0.0f, 1.0f);
    rgb8[id] = uchar(display * 255.0f);
}
