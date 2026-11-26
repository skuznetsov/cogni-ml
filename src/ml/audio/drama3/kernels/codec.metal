#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// One lane owns one output element. Reductions intentionally follow the
// native codec's Float32 order; the bridge also compiles with safe math and
// fast math disabled.
kernel void drama3_codec_conv1d(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint input_time = p[0];
    const uint input_channels = p[1];
    const uint output_channels = p[2];
    const uint kernel_size = p[3];
    const uint stride = p[4];
    const uint groups = p[5];
    const uint dilation = p[6];
    const uint count = input_time * output_channels;
    if (gid >= count) return;

    const uint output_t = gid / output_channels;
    const uint output_channel = gid % output_channels;
    const uint input_per_group = input_channels / groups;
    const uint output_per_group = output_channels / groups;
    const uint group = output_channel / output_per_group;
    const int left_pad = int((kernel_size - 1) * dilation);
    float sum = bias[output_channel];

    for (uint in_local = 0; in_local < input_per_group; ++in_local) {
        const uint input_channel = group * input_per_group + in_local;
        for (uint tap = 0; tap < kernel_size; ++tap) {
            const int input_t = int(output_t * stride) - left_pad + int(tap * dilation);
            if (input_t < 0 || input_t >= int(input_time)) continue;
            const uint wi = (output_channel * input_per_group + in_local) * kernel_size + tap;
            const uint xi = uint(input_t) * input_channels + input_channel;
            sum += input[xi] * weight[wi];
        }
    }
    output[gid] = sum;
}

// The CPU implementation scatters in input-time, input-channel, tap, then
// output-channel order and adds bias after all products. Gathering here uses
// the same per-output contribution order while assigning each output once.
// Output length remains input_time * stride, so the usual right tail is cut.
kernel void drama3_codec_conv_transpose1d(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint *p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const uint input_time = p[0];
    const uint input_channels = p[1];
    const uint output_channels = p[2];
    const uint kernel_size = p[3];
    const uint stride = p[4];
    const uint output_time = input_time * stride;
    const uint count = output_time * output_channels;
    if (gid >= count) return;

    const uint output_t = gid / output_channels;
    const uint output_channel = gid % output_channels;
    float sum = 0.0f;

    const int lower = max(0, int(output_t) - int(kernel_size - 1));
    const uint first_input_t = (uint(lower) + stride - 1) / stride;
    const uint last_input_t = min(output_t / stride, input_time - 1);
    if (first_input_t <= last_input_t) {
        for (uint input_t = first_input_t; input_t <= last_input_t; ++input_t) {
            const uint tap = output_t - input_t * stride;
            for (uint input_channel = 0; input_channel < input_channels; ++input_channel) {
                const uint wi = (input_channel * output_channels + output_channel) * kernel_size + tap;
                sum += input[input_t * input_channels + input_channel] * weight[wi];
            }
        }
    }
    output[gid] = sum + bias[output_channel];
}
