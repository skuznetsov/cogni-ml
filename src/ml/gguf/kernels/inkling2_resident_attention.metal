// Resident Inkling 2 attention and causal convolution kernels. Inputs are
// row-major Float32 tiles; each query row attends only to its causal prefix.

#include <metal_stdlib>
using namespace metal;

kernel void inkling2_resident_convolution(
    device const float* input          [[buffer(0)]],
    device const float* weights        [[buffer(1)]],
    device const float* dynamic_taps   [[buffer(2)]],
    device const float* history        [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant     uint&  channels       [[buffer(5)]],
    constant     uint&  taps           [[buffer(6)]],
    constant     uint&  rows           [[buffer(7)]],
    constant     uint&  start          [[buffer(8)]],
    constant     uint&  dynamic_heads  [[buffer(9)]],
    constant     uint&  history_slots  [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    const uint count = rows * channels;
    if (tid >= count) return;

    const uint row = tid / channels;
    const uint channel = tid - row * channels;
    const uint position = start + row;
    float static_sum = 0.0f;

    for (uint tap = 0; tap < taps; ++tap) {
        const int sample_position = (int)position - (int)(taps - 1 - tap);
        float sample = 0.0f;
        if (sample_position >= 0) {
            if ((uint)sample_position >= start) {
                sample = input[((uint)sample_position - start) * channels + channel];
            } else if (history_slots > 0) {
                sample = history[((uint)sample_position % history_slots) * channels + channel];
            }
        }
        static_sum += sample * weights[channel * taps + tap];
    }

    float dynamic_sum = 0.0f;
    if (dynamic_heads > 0) {
        const uint channels_per_head = channels / dynamic_heads;
        const uint head = channel / channels_per_head;
        const size_t dynamic_base = (size_t)row * (size_t)dynamic_heads * (size_t)taps +
                                    (size_t)head * (size_t)taps;
        for (uint lag = 0; lag < taps; ++lag) {
            const int sample_position = (int)position - (int)lag;
            float sample = 0.0f;
            if (sample_position >= 0) {
                if ((uint)sample_position >= start) {
                    sample = input[((uint)sample_position - start) * channels + channel];
                } else if (history_slots > 0) {
                    sample = history[((uint)sample_position % history_slots) * channels + channel];
                }
            }
            dynamic_sum += sample * dynamic_taps[dynamic_base + lag];
        }
    }

    const float scale_terms = (dynamic_heads > 0 ? 2.0f : 1.0f) * (float)taps + 1.0f;
    const float sum = input[tid] + static_sum + dynamic_sum;
    output[tid] = sum / sqrt(scale_terms);
}

// Commit only the newest input for each history slot. This remains correct
// when a batch is larger than the history ring and avoids write races.
kernel void inkling2_resident_convolution_commit(
    device const float* input       [[buffer(0)]],
    device       float* history     [[buffer(1)]],
    constant     uint&  channels    [[buffer(2)]],
    constant     uint&  rows        [[buffer(3)]],
    constant     uint&  start       [[buffer(4)]],
    constant     uint&  history_slots [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    const uint count = channels * history_slots;
    if (tid >= count || history_slots == 0) return;

    const uint slot = tid / channels;
    const uint channel = tid - slot * channels;
    const uint end_position = start + rows - 1;
    const uint distance = (end_position % history_slots + history_slots - slot) % history_slots;
    if (distance <= end_position - start) {
        const uint candidate = end_position - distance;
        const uint row = candidate - start;
        history[tid] = input[row * channels + channel];
    }
}

// In-place per-head Q/relative clipping and full-attention tau scaling.
// The head dimension is bounded to 256 by the host, so each SIMD lane keeps
// at most eight values while the norm reduction is in flight.
kernel void inkling2_resident_clip_heads(
    device float* values            [[buffer(0)]],
    device const float* channel_scale [[buffer(1)]],
    constant uint& rows             [[buffer(2)]],
    constant uint& heads            [[buffer(3)]],
    constant uint& width            [[buffer(4)]],
    constant uint& start            [[buffer(5)]],
    constant uint& log_scale_floor  [[buffer(6)]],
    constant uint& sliding          [[buffer(7)]],
    constant float& threshold       [[buffer(8)]],
    constant float& log_scale_alpha [[buffer(9)]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint row = tgpig.y;
    const uint head = tgpig.z;
    if (row >= rows || head >= heads || width > 256) return;

    const size_t base = (size_t)row * (size_t)heads * (size_t)width + (size_t)head * (size_t)width;
    const uint chunks = (width + 31) / 32;
    float local_values[8];
    float norm_partial = 0.0f;
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint channel = (uint)lane + chunk * 32;
        float value = 0.0f;
        if (channel < width) {
            value = values[base + channel] * channel_scale[(size_t)head * width + channel];
            norm_partial += value * value;
        }
        local_values[chunk] = value;
    }

    const float norm = sqrt(simd_sum(norm_partial)) + 1.0e-6f;
    const float clip = min(1.0f, threshold / norm);
    float tau = 1.0f;
    if (sliding == 0) {
        const float ratio = ((float)(start + row) + 1.0f) / (float)log_scale_floor;
        tau = 1.0f + log_scale_alpha * log(max(1.0f, ratio));
    }

    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint channel = (uint)lane + chunk * 32;
        if (channel < width) {
            values[base + channel] = local_values[chunk] * clip * tau;
        }
    }
}

// In-place key clipping and per-head value RMS normalization.
kernel void inkling2_resident_prepare_kv(
    device float* keys       [[buffer(0)]],
    device float* values     [[buffer(1)]],
    constant uint& rows      [[buffer(2)]],
    constant uint& heads     [[buffer(3)]],
    constant uint& head_dim  [[buffer(4)]],
    constant float& qk_clip  [[buffer(5)]],
    constant float& rms_eps  [[buffer(6)]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint row = tgpig.y;
    const uint head = tgpig.z;
    if (row >= rows || head >= heads || head_dim > 256) return;

    const size_t base = (size_t)row * (size_t)heads * (size_t)head_dim + (size_t)head * (size_t)head_dim;
    const uint chunks = (head_dim + 31) / 32;
    float key_values[8];
    float value_values[8];
    float key_partial = 0.0f;
    float value_partial = 0.0f;
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint channel = (uint)lane + chunk * 32;
        float key = 0.0f;
        float value = 0.0f;
        if (channel < head_dim) {
            key = keys[base + channel];
            value = values[base + channel];
            key_partial += key * key;
            value_partial += value * value;
        }
        key_values[chunk] = key;
        value_values[chunk] = value;
    }

    const float key_norm = sqrt(simd_sum(key_partial)) + 1.0e-6f;
    const float key_clip = min(1.0f, qk_clip / key_norm);
    const float value_mean_square = simd_sum(value_partial) / (float)head_dim;
    const float value_scale = rsqrt(value_mean_square + rms_eps);
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint channel = (uint)lane + chunk * 32;
        if (channel < head_dim) {
            keys[base + channel] = key_values[chunk] * key_clip;
            values[base + channel] = value_values[chunk] * value_scale;
        }
    }
}

inline float inkling2_resident_score(
    device const float* query,
    device const float* key,
    device const float* query_relative,
    device const float* relative_bias,
    float sink,
    uint head_dim,
    uint relative_dim,
    uint distance,
    uint relative_extent,
    ushort lane)
{
    float qk_partial = 0.0f;
    for (uint dim = (uint)lane; dim < head_dim; dim += 32) {
        qk_partial += query[dim] * key[dim];
    }
    float score = simd_sum(qk_partial) * rsqrt((float)head_dim);
    if (distance < relative_extent) {
        float relative_partial = 0.0f;
        const size_t bias_base = (size_t)distance * (size_t)relative_dim;
        for (uint dim = (uint)lane; dim < relative_dim; dim += 32) {
            relative_partial += query_relative[dim] * relative_bias[bias_base + dim];
        }
        score += simd_sum(relative_partial);
    }
    return score - sink;
}

// One SIMD group handles one query row and one query head. Scores are streamed
// twice (max, then denominator/context), avoiding a context-length scratch tile.
kernel void inkling2_resident_attention(
    device const float* q               [[buffer(0)]],
    device const float* k_tile          [[buffer(1)]],
    device const float* v_tile          [[buffer(2)]],
    device const float* q_relative      [[buffer(3)]],
    device const float* gate            [[buffer(4)]],
    device const float* sink            [[buffer(5)]],
    device const float* relative_bias   [[buffer(6)]],
    device const float* k_cache         [[buffer(7)]],
    device const float* v_cache         [[buffer(8)]],
    device       float* output          [[buffer(9)]],
    constant uint& start                [[buffer(10)]],
    constant uint& rows                 [[buffer(11)]],
    constant uint& capacity             [[buffer(12)]],
    constant uint& kv_width             [[buffer(13)]],
    constant uint& n_head               [[buffer(14)]],
    constant uint& n_head_kv            [[buffer(15)]],
    constant uint& head_dim             [[buffer(16)]],
    constant uint& relative_dim         [[buffer(17)]],
    constant uint& relative_extent      [[buffer(18)]],
    constant uint& sliding_window       [[buffer(19)]],
    constant uint& sliding              [[buffer(20)]],
    constant uint& query_row_offset     [[buffer(21)]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint row = tgpig.y;
    const uint head = tgpig.z;
    if (row >= rows || head >= n_head || head_dim > 256) return;

    const uint position = start + query_row_offset + row;
    const uint kv_head = head / (n_head / n_head_kv);
    const size_t q_base = (size_t)row * (size_t)n_head * (size_t)head_dim + (size_t)head * (size_t)head_dim;
    const size_t relative_base = (size_t)row * (size_t)n_head * (size_t)relative_dim + (size_t)head * (size_t)relative_dim;
    const size_t gate_base = q_base;
    const size_t output_base = q_base;
    uint first_key = 0;
    if (sliding != 0 && position + 1 > sliding_window) {
        first_key = position + 1 - sliding_window;
    }

    float max_score = 0.0f; // Learned sink is a zero logit in the denominator.
    for (uint key_position = first_key; key_position <= position; ++key_position) {
        const bool in_tile = key_position >= start;
        const uint slot = sliding != 0 ? key_position % capacity : key_position;
        const size_t cache_base = (size_t)slot * (size_t)kv_width + (size_t)kv_head * (size_t)head_dim;
        device const float* key = k_cache + cache_base;
        if (in_tile) {
            const size_t tile_base = (size_t)(key_position - start) * (size_t)kv_width + (size_t)kv_head * (size_t)head_dim;
            key = k_tile + tile_base;
        }
        const uint distance = position - key_position;
        const float score = inkling2_resident_score(
            q + q_base, key, q_relative + relative_base, relative_bias,
            sink[(size_t)row * (size_t)n_head + head], head_dim, relative_dim,
            distance, relative_extent, lane);
        max_score = max(max_score, score);
    }

    float denominator = exp(-max_score);
    const uint chunks = (head_dim + 31) / 32;
    float context[8];
    for (uint chunk = 0; chunk < chunks; ++chunk) context[chunk] = 0.0f;
    for (uint key_position = first_key; key_position <= position; ++key_position) {
        const bool in_tile = key_position >= start;
        const uint slot = sliding != 0 ? key_position % capacity : key_position;
        const size_t cache_base = (size_t)slot * (size_t)kv_width + (size_t)kv_head * (size_t)head_dim;
        device const float* key = k_cache + cache_base;
        device const float* value = v_cache + cache_base;
        if (in_tile) {
            const size_t tile_base = (size_t)(key_position - start) * (size_t)kv_width + (size_t)kv_head * (size_t)head_dim;
            key = k_tile + tile_base;
            value = v_tile + tile_base;
        }
        const uint distance = position - key_position;
        const float score = inkling2_resident_score(
            q + q_base, key, q_relative + relative_base, relative_bias,
            sink[(size_t)row * (size_t)n_head + head], head_dim, relative_dim,
            distance, relative_extent, lane);
        const float weight = exp(score - max_score);
        denominator += weight;
        for (uint chunk = 0; chunk < chunks; ++chunk) {
            const uint channel = (uint)lane + chunk * 32;
            if (channel < head_dim) {
                context[chunk] += weight * value[channel];
            }
        }
    }

    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint channel = (uint)lane + chunk * 32;
        if (channel < head_dim) {
            const float gate_value = gate[gate_base + channel];
            const float sigmoid = 1.0f / (1.0f + exp(-gate_value));
            output[output_base + channel] = (context[chunk] / denominator) * sigmoid;
        }
    }
}

// Decode-only split-K attention. Each SIMD group owns a contiguous key range
// and writes one local softmax tuple (max, denominator, weighted V sum).
// A following kernel combines those tuples and contributes the learned sink
// exactly once as a zero logit.
kernel void inkling2_resident_attention_split(
    device const float* q               [[buffer(0)]],
    device const float* k_tile          [[buffer(1)]],
    device const float* v_tile          [[buffer(2)]],
    device const float* q_relative      [[buffer(3)]],
    device const float* gate            [[buffer(4)]],
    device const float* sink            [[buffer(5)]],
    device const float* relative_bias   [[buffer(6)]],
    device const float* k_cache         [[buffer(7)]],
    device const float* v_cache         [[buffer(8)]],
    device       float* partials        [[buffer(9)]],
    constant uint& start                [[buffer(10)]],
    constant uint& rows                 [[buffer(11)]],
    constant uint& capacity             [[buffer(12)]],
    constant uint& kv_width             [[buffer(13)]],
    constant uint& n_head               [[buffer(14)]],
    constant uint& n_head_kv            [[buffer(15)]],
    constant uint& head_dim             [[buffer(16)]],
    constant uint& relative_dim         [[buffer(17)]],
    constant uint& relative_extent      [[buffer(18)]],
    constant uint& sliding_window       [[buffer(19)]],
    constant uint& sliding              [[buffer(20)]],
    constant uint& partitions           [[buffer(21)]],
    constant uint& query_row_offset     [[buffer(22)]],
    constant uint& score_slots         [[buffer(23)]],
    threadgroup float* cached_scores   [[threadgroup(0)]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint partition = tgpig.x;
    const uint row = tgpig.y;
    const uint head = tgpig.z;
    if (partition >= partitions || row >= rows || head >= n_head || head_dim > 256) return;

    const uint position = start + query_row_offset + row;
    const uint kv_head = head / (n_head / n_head_kv);
    const size_t q_base = (size_t)row * (size_t)n_head * (size_t)head_dim + (size_t)head * (size_t)head_dim;
    const size_t relative_base = (size_t)row * (size_t)n_head * (size_t)relative_dim + (size_t)head * (size_t)relative_dim;
    uint first_key = 0;
    if (sliding != 0 && position + 1 > sliding_window) {
        first_key = position + 1 - sliding_window;
    }
    const uint key_count = position - first_key + 1;
    const uint partition_base = key_count / partitions;
    const uint partition_extra = key_count % partitions;
    const uint partition_begin = partition_base * partition + min(partition, partition_extra);
    const uint partition_end = partition_base * (partition + 1) + min(partition + 1, partition_extra);
    const size_t partial_base = (((size_t)row * (size_t)n_head + (size_t)head) * (size_t)partitions + (size_t)partition) * ((size_t)head_dim + 2);

    // One SIMD group owns the partition. Keep its scores for the second
    // pass; a larger partition recomputes them to bound scratch memory.
    const bool save_scores = partition_end - partition_begin <= score_slots;
    float max_score = -INFINITY;
    for (uint key_index = partition_begin; key_index < partition_end; ++key_index) {
        const uint key_position = first_key + key_index;
        const bool in_tile = key_position >= start;
        const uint slot = sliding != 0 ? key_position % capacity : key_position;
        const size_t cache_base = (size_t)slot * (size_t)kv_width + (size_t)kv_head * (size_t)head_dim;
        device const float* key = k_cache + cache_base;
        if (in_tile) {
            const size_t tile_base = (size_t)(key_position - start) * (size_t)kv_width + (size_t)kv_head * (size_t)head_dim;
            key = k_tile + tile_base;
        }
        const uint distance = position - key_position;
        const float score = inkling2_resident_score(
            q + q_base, key, q_relative + relative_base, relative_bias,
            sink[(size_t)row * (size_t)n_head + head], head_dim, relative_dim,
            distance, relative_extent, lane);
        max_score = max(max_score, score);
        if (save_scores && lane == 0) cached_scores[key_index - partition_begin] = score;
    }
    if (save_scores) simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunks = (head_dim + 31) / 32;
    float context[8];
    for (uint chunk = 0; chunk < chunks; ++chunk) context[chunk] = 0.0f;
    float denominator = 0.0f;
    for (uint key_index = partition_begin; key_index < partition_end; ++key_index) {
        const uint key_position = first_key + key_index;
        const bool in_tile = key_position >= start;
        const uint slot = sliding != 0 ? key_position % capacity : key_position;
        const uint kv_row = in_tile ? key_position - start : slot;
        const size_t kv_base = (size_t)kv_row * kv_width + (size_t)kv_head * head_dim;
        device const float* value = (in_tile ? v_tile : v_cache) + kv_base;
        float score;
        if (save_scores) {
            score = cached_scores[key_index - partition_begin];
        } else {
            device const float* key = (in_tile ? k_tile : k_cache) + kv_base;
            const uint distance = position - key_position;
            score = inkling2_resident_score(
                q + q_base, key, q_relative + relative_base, relative_bias,
                sink[(size_t)row * (size_t)n_head + head], head_dim, relative_dim,
                distance, relative_extent, lane);
        }
        const float weight = exp(score - max_score);
        denominator += weight;
        for (uint chunk = 0; chunk < chunks; ++chunk) {
            const uint channel = (uint)lane + chunk * 32;
            if (channel < head_dim) {
                context[chunk] += weight * value[channel];
            }
        }
    }

    if (lane == 0) {
        partials[partial_base] = max_score;
        partials[partial_base + 1] = denominator;
    }
    for (uint chunk = 0; chunk < chunks; ++chunk) {
        const uint channel = (uint)lane + chunk * 32;
        if (channel < head_dim) {
            partials[partial_base + 2 + channel] = context[chunk];
        }
    }
}

// Merge local softmax tuples. The sink contributes one zero logit globally,
// then the resulting context is gated exactly as in the regular kernel.
kernel void inkling2_resident_attention_combine(
    device const float* partials        [[buffer(0)]],
    device const float* gate            [[buffer(1)]],
    device       float* output          [[buffer(2)]],
    constant uint& rows                 [[buffer(3)]],
    constant uint& n_head               [[buffer(4)]],
    constant uint& head_dim             [[buffer(5)]],
    constant uint& partitions           [[buffer(6)]],
    ushort lane [[thread_index_in_simdgroup]],
    uint3 tgpig [[threadgroup_position_in_grid]])
{
    const uint row = tgpig.y;
    const uint head = tgpig.z;
    if (row >= rows || head >= n_head || head_dim > 256) return;

    const size_t head_base = (size_t)row * (size_t)n_head * (size_t)head_dim + (size_t)head * (size_t)head_dim;
    const size_t partial_head_base = ((size_t)row * (size_t)n_head + (size_t)head) * (size_t)partitions * ((size_t)head_dim + 2);
    float max_score = 0.0f; // The global zero-logit sink is the initial max.
    for (uint partition = 0; partition < partitions; ++partition) {
        const size_t base = partial_head_base + (size_t)partition * ((size_t)head_dim + 2);
        max_score = max(max_score, partials[base]);
    }

    float denominator = exp(-max_score);
    for (uint partition = 0; partition < partitions; ++partition) {
        const size_t base = partial_head_base + (size_t)partition * ((size_t)head_dim + 2);
        denominator += partials[base + 1] * exp(partials[base] - max_score);
    }

    for (uint dim = (uint)lane; dim < head_dim; dim += 32) {
        float context = 0.0f;
        for (uint partition = 0; partition < partitions; ++partition) {
            const size_t base = partial_head_base + (size_t)partition * ((size_t)head_dim + 2);
            context += partials[base + 2 + dim] * exp(partials[base] - max_score);
        }
        const float gate_value = gate[head_base + dim];
        const float sigmoid = 1.0f / (1.0f + exp(-gate_value));
        output[head_base + dim] = (context / denominator) * sigmoid;
    }
}

// Persist the post-convolution K/V tile after all queries have consumed its
// transient rows. For oversized sliding batches, only the last ring's rows are
// copied, so each destination slot has one writer.
kernel void inkling2_resident_cache_commit(
    device const float* k_tile       [[buffer(0)]],
    device const float* v_tile       [[buffer(1)]],
    device       float* k_cache      [[buffer(2)]],
    device       float* v_cache      [[buffer(3)]],
    constant uint& start             [[buffer(4)]],
    constant uint& rows              [[buffer(5)]],
    constant uint& capacity          [[buffer(6)]],
    constant uint& kv_width          [[buffer(7)]],
    constant uint& sliding           [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    const uint end_position = start + rows - 1;
    const uint write_rows = sliding != 0 ? min(rows, capacity) : rows;
    const uint write_start = sliding != 0 ? end_position + 1 - write_rows : start;
    const uint count = write_rows * kv_width;
    if (tid >= count) return;

    const uint row = tid / kv_width;
    const uint channel = tid - row * kv_width;
    const uint position = write_start + row;
    const uint slot = sliding != 0 ? position % capacity : position;
    const size_t source = (size_t)(position - start) * (size_t)kv_width + channel;
    const size_t destination = (size_t)slot * (size_t)kv_width + channel;
    k_cache[destination] = k_tile[source];
    v_cache[destination] = v_tile[source];
}

kernel void inkling2_resident_cache_copy(
    device const float* old_k      [[buffer(0)]],
    device const float* old_v      [[buffer(1)]],
    device       float* new_k      [[buffer(2)]],
    device       float* new_v      [[buffer(3)]],
    constant uint& first_element   [[buffer(4)]],
    constant uint& element_count   [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= element_count) return;
    const uint element = first_element + tid;
    new_k[element] = old_k[element];
    new_v[element] = old_v[element];
}
