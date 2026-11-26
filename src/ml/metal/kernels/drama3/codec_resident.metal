#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline float dr3c_float(constant ulong* p, uint i) {
    return as_type<float>(uint(p[i]));
}

// MSL does not expose erf on the supported compiler. This approximation has
// sub-2e-7 absolute error over the real line and keeps GELU device-resident.
inline float dr3c_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = abs(value);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t
                         - 0.284496736f) * t + 0.254829592f) * t;
    return sign * (1.0f - polynomial * exp(-x * x));
}

kernel void dr3c_wn_scales(
    device const float* weight [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= p[0]) return;
    float norm_sq = 0.0f;
    ulong base = ulong(channel) * p[1];
    for (uint i = 0; i < p[1]; ++i) {
        float value = weight[base + i];
        norm_sq += value * value;
    }
    scales[channel] = gain[channel] / (sqrt(norm_sq) + 1.0e-12f);
}

kernel void dr3c_wn_apply(
    device const float* weight [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* effective [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) effective[i] = weight[i] * scales[i / p[1]];
}

// Same per-element F32 multiply as dr3c_wn_apply, stored as [inner, out]
// for the ungrouped Conv1d kernel's output-channel lanes.
kernel void dr3c_wn_apply_coalesced(
    device const float* weight [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* effective [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p[0]) return;
    uint channels = uint(p[0] / p[1]);
    uint out_channel = i % channels;
    uint inner = i / channels;
    ulong source = ulong(out_channel) * p[1] + inner;
    effective[i] = weight[source] * scales[out_channel];
}

kernel void dr3c_conv1d(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), in_channels = uint(p[1]), out_channels = uint(p[2]);
    uint kernel_size = uint(p[3]), stride = uint(p[4]), groups = uint(p[5]), dilation = uint(p[6]);
    if (i >= rows * out_channels) return;
    uint t = i / out_channels, out_channel = i % out_channels;
    uint in_per_group = in_channels / groups, out_per_group = out_channels / groups;
    uint group = out_channel / out_per_group;
    int left_pad = int((kernel_size - 1) * dilation);
    float sum = bias[out_channel];
    for (uint in_local = 0; in_local < in_per_group; ++in_local) {
        uint in_channel = group * in_per_group + in_local;
        for (uint tap = 0; tap < kernel_size; ++tap) {
            int input_t = int(t * stride) - left_pad + int(tap * dilation);
            if (input_t < 0 || input_t >= int(rows)) continue;
            ulong wi = (ulong(out_channel) * in_per_group + in_local) * kernel_size + tap;
            sum += input[ulong(input_t) * in_channels + in_channel] * weight[wi];
        }
    }
    output[i] = sum;
}

// Ungrouped, output-channel-inner counterpart of dr3c_conv1d. The serial
// in_local -> tap accumulation and its F32 rounding order are unchanged.
kernel void dr3c_conv1d_coalesced(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), in_channels = uint(p[1]), out_channels = uint(p[2]);
    uint kernel_size = uint(p[3]), stride = uint(p[4]), dilation = uint(p[5]);
    if (i >= rows * out_channels) return;
    uint t = i / out_channels, out_channel = i % out_channels;
    int left_pad = int((kernel_size - 1) * dilation);
    float sum = bias[out_channel];
    for (uint in_local = 0; in_local < in_channels; ++in_local) {
        for (uint tap = 0; tap < kernel_size; ++tap) {
            int input_t = int(t * stride) - left_pad + int(tap * dilation);
            if (input_t < 0 || input_t >= int(rows)) continue;
            ulong wi = (ulong(in_local) * kernel_size + tap) * out_channels + out_channel;
            sum += input[ulong(input_t) * in_channels + in_local] * weight[wi];
        }
    }
    output[i] = sum;
}

kernel void dr3c_conv_transpose1d(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    uint input_rows = uint(p[0]), in_channels = uint(p[1]), out_channels = uint(p[2]);
    uint kernel_size = uint(p[3]), stride = uint(p[4]);
    uint output_rows = input_rows * stride;
    if (i >= output_rows * out_channels) return;
    uint output_t = i / out_channels, out_channel = i % out_channels;
    uint first_input_t = output_t >= kernel_size - 1 ? (output_t - kernel_size + 1 + stride - 1) / stride : 0;
    uint last_input_t = min(output_t / stride, input_rows - 1);
    float sum = 0.0f;
    if (first_input_t <= last_input_t) {
        for (uint t = first_input_t; t <= last_input_t; ++t) {
            uint tap = output_t - t * stride;
            for (uint in_channel = 0; in_channel < in_channels; ++in_channel) {
                ulong wi = (ulong(in_channel) * out_channels + out_channel) * kernel_size + tap;
                sum += input[ulong(t) * in_channels + in_channel] * weight[wi];
            }
        }
    }
    output[i] = sum + bias[out_channel];
}

kernel void dr3c_linear(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), in_channels = uint(p[1]), out_channels = uint(p[2]);
    if (i >= rows * out_channels) return;
    uint t = i / out_channels, out_channel = i % out_channels;
    float sum = 0.0f;
    ulong weight_base = ulong(out_channel) * in_channels;
    ulong input_base = ulong(t) * in_channels;
    for (uint c = 0; c < in_channels; ++c) sum += input[input_base + c] * weight[weight_base + c];
    output[i] = sum;
}

kernel void dr3c_linear_bias(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), in_channels = uint(p[1]), out_channels = uint(p[2]);
    if (i >= rows * out_channels) return;
    uint t = i / out_channels, out_channel = i % out_channels;
    float sum = 0.0f;
    ulong weight_base = ulong(out_channel) * in_channels;
    ulong input_base = ulong(t) * in_channels;
    for (uint c = 0; c < in_channels; ++c) sum += input[input_base + c] * weight[weight_base + c];
    output[i] = sum + bias[out_channel];
}

kernel void dr3c_embedding(
    device const float* codes [[buffer(0)]],
    device const float* table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), width = uint(p[1]), book = uint(p[2]);
    // Generation stores [main,c0,c1,...,c9]; Codec consumes c0 as the
    // semantic code and c1..c9 as residuals, leaving the main token unused.
    uint code_column = book + 1;
    uint vocab = uint(p[3]), codebook_dim = uint(p[4]);
    if (i >= rows * codebook_dim) return;
    uint t = i / codebook_dim, dim = i % codebook_dim;
    uint code_index = p[5] ? book * rows + t : t * width + code_column;
    int token = int(codes[code_index]);
    token = clamp(token, 0, int(vocab) - 1);
    output[i] = table[ulong(token) * codebook_dim + dim];
}

kernel void dr3c_qkv_split(
    device const float* qkv [[buffer(0)]],
    device float* query [[buffer(1)]],
    device float* key [[buffer(2)]],
    device float* value [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), channels = uint(p[1]);
    if (i >= rows * channels) return;
    uint t = i / channels, c = i % channels;
    ulong base = ulong(t) * channels * 3;
    query[i] = qkv[base + c];
    key[i] = qkv[base + channels + c];
    value[i] = qkv[base + channels * 2 + c];
}

kernel void dr3c_rope_all(
    device float* x [[buffer(0)]],
    constant ulong* p [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), heads = uint(p[1]), dim = uint(p[2]), pairs = dim / 2;
    if (id >= rows * heads * pairs) return;
    uint t = id / (heads * pairs), head = (id / pairs) % heads, pair = id % pairs;
    float base = dr3c_float(p, 3);
    float exponent = float(2 * pair) / float(dim);
    float frequency = 1.0f / pow(base, exponent);
    float angle = float(t) * frequency;
    float c = cos(angle), s = sin(angle);
    uint offset = t * heads * dim + head * dim + pair * 2;
    float left = x[offset], right = x[offset + 1];
    x[offset] = left * c - right * s;
    x[offset + 1] = right * c + left * s;
}

kernel void dr3c_attention_scores(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), heads = uint(p[1]), dim = uint(p[2]), window = uint(p[3]);
    if (i >= rows * heads * window) return;
    uint slot = i % window, head = (i / window) % heads, t = i / (window * heads);
    uint first_key = t >= window - 1 ? t - window + 1 : 0;
    uint valid = t - first_key + 1;
    if (slot >= valid) { scores[i] = 0.0f; return; }
    uint key_t = first_key + slot, channels = heads * dim;
    float score = 0.0f;
    ulong qbase = ulong(t) * channels + head * dim;
    ulong kbase = ulong(key_t) * channels + head * dim;
    for (uint d = 0; d < dim; ++d) score += query[qbase + d] * key[kbase + d];
    scores[i] = score * dr3c_float(p, 4);
}

kernel void dr3c_attention_probs(
    device const float* scores [[buffer(0)]],
    device float* probabilities [[buffer(1)]],
    constant ulong* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), heads = uint(p[1]), window = uint(p[2]);
    if (i >= rows * heads) return;
    uint t = i / heads, first_key = t >= window - 1 ? t - window + 1 : 0;
    uint valid = t - first_key + 1;
    ulong base = ulong(i) * window;
    float maximum = -INFINITY;
    for (uint s = 0; s < valid; ++s) maximum = max(maximum, scores[base + s]);
    float denom = 0.0f;
    for (uint s = 0; s < valid; ++s) {
        float value = exp(scores[base + s] - maximum);
        probabilities[base + s] = value;
        denom += value;
    }
    for (uint s = 0; s < valid; ++s) probabilities[base + s] /= denom;
    for (uint s = valid; s < window; ++s) probabilities[base + s] = 0.0f;
}

kernel void dr3c_attention_values(
    device const float* probabilities [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), heads = uint(p[1]), dim = uint(p[2]), window = uint(p[3]);
    if (i >= rows * heads * dim) return;
    uint d = i % dim, head = (i / dim) % heads, t = i / (dim * heads);
    uint first_key = t >= window - 1 ? t - window + 1 : 0;
    uint valid = t - first_key + 1, channels = heads * dim;
    float result = 0.0f;
    ulong pbase = (ulong(t) * heads + head) * window;
    for (uint s = 0; s < valid; ++s) {
        ulong vindex = ulong(first_key + s) * channels + head * dim + d;
        result += probabilities[pbase + s] * value[vindex];
    }
    output[i] = result;
}

kernel void dr3c_channel_scale(
    device const float* x [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint cols = uint(p[1]);
    if (i < p[0] * cols) y[i] = x[i] * scale[i % cols];
}

kernel void dr3c_silu_product(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) y[i] = (a[i] / (1.0f + exp(-a[i]))) * b[i];
}

kernel void dr3c_gelu(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant ulong* p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i < p[0]) y[i] = 0.5f * x[i] * (1.0f + dr3c_erf(x[i] * 0.7071067811865475244f));
}

kernel void dr3c_layer_norm(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    uint rows = uint(p[0]), cols = uint(p[1]);
    if (row >= rows) return;
    float mean = 0.0f;
    ulong base = ulong(row) * cols;
    for (uint c = 0; c < cols; ++c) mean += x[base + c];
    mean /= float(cols);
    float variance = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        float diff = x[base + c] - mean;
        variance += diff * diff;
    }
    variance /= float(cols);
    float inv = 1.0f / sqrt(variance + dr3c_float(p, 2));
    for (uint c = 0; c < cols; ++c) y[base + c] = (x[base + c] - mean) * inv * gamma[c] + beta[c];
}

kernel void dr3c_snake(
    device const float* x [[buffer(0)]],
    device const float* alpha [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant ulong* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    uint cols = uint(p[1]);
    if (i < p[0] * cols) {
        float value = x[i], a = alpha[i % cols], sine = sin(a * value);
        y[i] = value + sine * sine / (a + 1.0e-9f);
    }
}

kernel void dr3c_tanh_mono(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant ulong* p [[buffer(2)]],
    uint t [[thread_position_in_grid]]) {
    if (t < p[0]) y[t] = tanh(x[ulong(t) * p[1]]);
}
