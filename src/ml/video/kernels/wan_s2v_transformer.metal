// Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
// Apache-2.0. Adapted from Wan2.2 S2V at
// 1ea34ff48f87168174e12956e200b1d908b1c5ff.
#include <metal_stdlib>
using namespace metal;

inline float wan_s2v_bf16_to_float(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline float wan_s2v_typed_value(device const float *f32,
                                 device const ushort *bf16,
                                 uint index, uint is_bf16) {
    return is_bf16 != 0 ? wan_s2v_bf16_to_float(bf16[index]) : f32[index];
}

// First-request no-pose setup follows Wan's `COND[0] * 0` branch after VAE
// encoding. The VAE buffer is dead after this point, so the caller may zero it
// in place without allocating another full latent activation.
kernel void wan_s2v_zero_condition(device float *condition [[buffer(0)]],
                                   constant uint &count [[buffer(1)]],
                                   uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    condition[id] = 0.0f;
}

// Upstream sinusoidal_embedding_1d uses an FP64 outer product and emits all
// cosines followed by all sines. Setup supplies high/low F32 frequency parts;
// the compensated product retains the discarded frequency and product tails.
kernel void wan_s2v_sinusoidal(
    device const float *timestep [[buffer(0)]],
    device const float *frequency_hi [[buffer(1)]],
    device const float *frequency_lo [[buffer(2)]],
    device float *embedding [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &dim [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    const uint rows = batches + 1;
    if (id >= rows * dim) return;
    const uint row = id / dim, column = id % dim, half_dim = dim / 2;
    const float position = row < batches ? timestep[row] : 0.0f;
    const uint pair = column < half_dim ? column : column - half_dim;
    const float high = position * frequency_hi[pair];
    const float product_tail = fma(position, frequency_hi[pair], -high);
    const float tail = fma(position, frequency_lo[pair], product_tail);
    const float phase = high + tail;
    const float virtual_phase = phase - high;
    const float residual = (high - (phase - virtual_phase)) + (tail - virtual_phase);
    const float cosine = precise::cos(phase) * precise::cos(residual) -
                         precise::sin(phase) * precise::sin(residual);
    const float sine = precise::sin(phase) * precise::cos(residual) +
                       precise::cos(phase) * precise::sin(residual);
    embedding[id] = column < half_dim ? cosine : sine;
}

kernel void wan_s2v_silu(device const float *x [[buffer(0)]],
                         device float *y [[buffer(1)]],
                         constant uint &count [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const float value = x[id];
    y[id] = value / (1.0f + exp(-value));
}

// Convert [B+1,6,C] projection rows into [B,6,2,C]. Noisy video uses the
// sample timestep; reference and motion/framepack tokens use the appended
// zero-timestep row, matching the upstream S2V sequence split.
kernel void wan_s2v_time_expand(
    device const float *time_embedding_all [[buffer(0)]],
    device const float *projection_all [[buffer(1)]],
    device float *time_embedding [[buffer(2)]],
    device float *modulation [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &dim [[buffer(5)]],
    uint id [[thread_position_in_grid]]) {
    const uint time_count = batches * dim;
    const uint modulation_count = batches * 12 * dim;
    if (id < time_count) time_embedding[id] = time_embedding_all[id];
    if (id >= modulation_count) return;
    const uint channel = id % dim;
    const uint segment = (id / dim) % 2;
    const uint field = (id / (2 * dim)) % 6;
    const uint batch = id / (12 * dim);
    const uint source_batch = segment == 0 ? batch : batches;
    modulation[id] = projection_all[(source_batch * 6 + field) * dim + channel];
}

// Gather Conv3d patches into row-major (input-channel, t, h, w) vectors.
// `virtual_frames` right-aligns short FramePack inputs in the 19-frame window;
// out-of-range source frames are the zeros created by upstream FramePack.
kernel void wan_s2v_patchify(
    device const float *latent [[buffer(0)]], device float *tile [[buffer(1)]],
    constant uint &channels [[buffer(2)]], constant uint &source_frames [[buffer(3)]],
    constant uint &height [[buffer(4)]], constant uint &width [[buffer(5)]],
    constant uint &virtual_frames [[buffer(6)]], constant uint &patch_t [[buffer(7)]],
    constant uint &patch_h [[buffer(8)]], constant uint &patch_w [[buffer(9)]],
    constant uint &grid_h [[buffer(10)]], constant uint &grid_w [[buffer(11)]],
    constant uint &segment_frame_start [[buffer(12)]], constant uint &tile_start [[buffer(13)]],
    constant uint &tile_rows [[buffer(14)]], constant uint &batch [[buffer(15)]],
    uint id [[thread_position_in_grid]]) {
    const uint features = channels * patch_t * patch_h * patch_w;
    if (id >= tile_rows * features) return;
    const uint row = tile_start + id / features;
    const uint feature = id % features;
    const uint patch_x = row % grid_w, patch_y = (row / grid_w) % grid_h;
    const uint patch_frame = segment_frame_start + row / (grid_h * grid_w);
    const uint dx = feature % patch_w;
    const uint dy = (feature / patch_w) % patch_h;
    const uint dt = (feature / (patch_w * patch_h)) % patch_t;
    const uint channel = feature / (patch_w * patch_h * patch_t);
    const int source_frame = int(source_frames) - int(virtual_frames) +
                             int(patch_frame * patch_t + dt);
    if (source_frame < 0 || source_frame >= int(source_frames)) {
        tile[id] = 0.0f;
        return;
    }
    const uint y = patch_y * patch_h + dy, x = patch_x * patch_w + dx;
    const uint sample_values = channels * source_frames * height * width;
    const uint offset = batch * sample_values +
                        ((channel * source_frames + uint(source_frame)) * height + y) * width + x;
    tile[id] = latent[offset];
}

// Scatter one projected tile into the packed [B,S,C] sequence. `class_id=3`
// means add a projected condition patch without adding another class vector.
kernel void wan_s2v_scatter_tokens(
    device const float *projected [[buffer(0)]],
    device const float *class_f32 [[buffer(1)]],
    device const ushort *class_bf16 [[buffer(2)]],
    device float *sequence [[buffer(3)]],
    constant uint &batch [[buffer(4)]], constant uint &sequence_tokens [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &token_offset [[buffer(7)]],
    constant uint &tile_start [[buffer(8)]], constant uint &tile_rows [[buffer(9)]],
    constant uint &class_id [[buffer(10)]], constant uint &add [[buffer(11)]],
    constant uint &class_bf16_flag [[buffer(12)]], uint id [[thread_position_in_grid]]) {
    if (id >= tile_rows * dim) return;
    const uint row = tile_start + id / dim, channel = id % dim;
    const uint target = (batch * sequence_tokens + token_offset + row) * dim + channel;
    const float old = add != 0 ? sequence[target] : 0.0f;
    const float embedding = class_id < 3
        ? wan_s2v_typed_value(class_f32, class_bf16, class_id * dim + channel, class_bf16_flag)
        : 0.0f;
    sequence[target] = old + projected[id] + embedding;
}

// Non-affine LayerNorm over only the noisy prefix. Sequence rows are strided
// by the full sequence length because reference and motion suffixes are kept.
kernel void wan_s2v_head_norm(
    device float *x [[buffer(0)]], constant uint &batches [[buffer(1)]],
    constant uint &sequence_tokens [[buffer(2)]], constant uint &prefix_tokens [[buffer(3)]],
    constant uint &dim [[buffer(4)]], constant float &epsilon [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    if (row >= batches * prefix_tokens) return;
    const uint batch = row / prefix_tokens, token = row % prefix_tokens;
    const uint base = (batch * sequence_tokens + token) * dim;
    threadgroup float partials[8], mean, inverse;
    float sum = 0.0f;
    for (uint d = tid; d < dim; d += threads) sum += x[base + d];
    float reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        mean = total / float(dim);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum = 0.0f;
    for (uint d = tid; d < dim; d += threads) {
        const float centered = x[base + d] - mean;
        sum += centered * centered;
    }
    reduced = simd_sum(sum);
    if (lane == 0) partials[sg] = reduced;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < threads / 32; ++i) total += partials[i];
        inverse = rsqrt(total / float(dim) + epsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < dim; d += threads)
        x[base + d] = (x[base + d] - mean) * inverse;
}

kernel void wan_s2v_head_modulate(
    device float *x [[buffer(0)]], device const float *time_embedding [[buffer(1)]],
    device const float *modulation_f32 [[buffer(2)]],
    device const ushort *modulation_bf16 [[buffer(3)]],
    constant uint &sequence_tokens [[buffer(4)]], constant uint &prefix_tokens [[buffer(5)]],
    constant uint &dim [[buffer(6)]], constant uint &modulation_is_bf16 [[buffer(7)]],
    constant uint &count [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    if (id >= count) return;
    const uint channel = id % dim;
    const uint batch = id / (prefix_tokens * dim);
    const uint token = (id / dim) % prefix_tokens;
    const uint modulation_f32_offset = 0;
    const float shift = time_embedding[batch * dim + channel] +
        wan_s2v_typed_value(modulation_f32, modulation_bf16, modulation_f32_offset + channel,
                            modulation_is_bf16);
    const float scale = time_embedding[batch * dim + channel] +
        wan_s2v_typed_value(modulation_f32, modulation_bf16, dim + channel,
                            modulation_is_bf16);
    const uint at = (batch * sequence_tokens + token) * dim + channel;
    x[at] = x[at] * (1.0f + scale) + shift;
}

kernel void wan_s2v_head_linear(
    device const float *x [[buffer(0)]], device const ushort *weight_bf16 [[buffer(1)]],
    device const ushort *bias_bf16 [[buffer(2)]], device float *prediction [[buffer(3)]],
    constant uint &batches [[buffer(4)]], constant uint &sequence_tokens [[buffer(5)]],
    constant uint &prefix_tokens [[buffer(6)]], constant uint &dim [[buffer(7)]],
    constant uint &out_dim [[buffer(8)]], device const float *weight_f32 [[buffer(9)]],
    device const float *bias_f32 [[buffer(10)]], constant uint &weight_is_bf16 [[buffer(11)]],
    constant uint &bias_is_bf16 [[buffer(12)]], uint id [[thread_position_in_grid]]) {
    if (id >= batches * prefix_tokens * out_dim) return;
    const uint row = id / out_dim, column = id % out_dim;
    const uint batch = row / prefix_tokens, token = row % prefix_tokens;
    const uint x_offset = (batch * sequence_tokens + token) * dim;
    float sums[16] = {0.0f};
    uint i = 0;
    for (; i + 15 < dim; i += 16) {
        #pragma unroll
        for (uint j = 0; j < 16; ++j) {
            const uint wi = column * dim + i + j;
            const float weight = wan_s2v_typed_value(weight_f32, weight_bf16, wi, weight_is_bf16);
            sums[j] = fma(x[x_offset + i + j], weight, sums[j]);
        }
    }
    for (; i < dim; ++i) {
        const uint wi = column * dim + i;
        const float weight = wan_s2v_typed_value(weight_f32, weight_bf16, wi, weight_is_bf16);
        sums[0] = fma(x[x_offset + i], weight, sums[0]);
    }
    const float a = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    const float b = (sums[4] + sums[5]) + (sums[6] + sums[7]);
    const float c = (sums[8] + sums[9]) + (sums[10] + sums[11]);
    const float d = (sums[12] + sums[13]) + (sums[14] + sums[15]);
    const float bias = wan_s2v_typed_value(bias_f32, bias_bf16, column, bias_is_bf16);
    prediction[id] = ((a + b) + (c + d)) + bias;
}

// PyTorch Wan unpatchify implements einsum('fhwpqrc->cfphqwr').
kernel void wan_s2v_unpatchify(
    device const float *prediction [[buffer(0)]], device float *latent [[buffer(1)]],
    constant uint &batches [[buffer(2)]], constant uint &frames [[buffer(3)]],
    constant uint &grid_h [[buffer(4)]], constant uint &grid_w [[buffer(5)]],
    constant uint &channels [[buffer(6)]], constant uint &height [[buffer(7)]],
    constant uint &width [[buffer(8)]], uint id [[thread_position_in_grid]]) {
    const uint total = batches * channels * frames * height * width;
    if (id >= total) return;
    const uint x = id % width, y = (id / width) % height;
    const uint frame = (id / (width * height)) % frames;
    const uint channel = (id / (width * height * frames)) % channels;
    const uint batch = id / (width * height * frames * channels);
    const uint patch = (frame * grid_h + y / 2) * grid_w + x / 2;
    const uint feature = (((y % 2) * 2 + (x % 2)) * channels) + channel;
    latent[id] = prediction[(batch * frames * grid_h * grid_w + patch) * (channels * 4) + feature];
}
