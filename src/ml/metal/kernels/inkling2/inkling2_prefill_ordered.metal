#include <metal_stdlib>
using namespace metal;

constant uint ordered_query_rows = 4;
constant uint ordered_key_tile = 64;
constant uint ordered_qk_simdgroups = 4; // 128 threads per QK threadgroup.
constant uint ordered_channels = 128;
constant uint ordered_pv_stripe = 64;
constant uint gqa_heads = 3;
constant uint gqa_rows = 2;
constant uint gqa_key_tile = 32;
constant uint gqa_simdgroups = 2;

struct Inkling2PrefillOrderedParams {
    uint start;
    uint rows;
    uint query_heads;
    uint kv_heads;
    uint relative_extent;
    uint key_origin;
    uint packed_keys;
    uint score_stride;
    uint sliding_window;
    uint cache_capacity;
    uint source_start;
};

// Position and result are relative to the first packed key. Zero window
// denotes an unrestricted span, where every query starts at the same key.
static inline uint ordered_first_key(uint position, constant Inkling2PrefillOrderedParams& p) {
    if (p.sliding_window == 0) return 0;
    return position + 1 - min(p.sliding_window, p.key_origin + position + 1);
}

// Pack the union of visible keys in chronological order within each KV head.
// K groups each lane's QK operands and V groups adjacent output channels.
kernel void inkling2_prefill_ordered_pack(
    device const float* k                [[buffer(0)]],
    device const float4* v               [[buffer(1)]],
    device const float* k_cache          [[buffer(2)]],
    device const float4* v_cache         [[buffer(3)]],
    device       float4* packed_k        [[buffer(4)]],
    device       float4* packed_v        [[buffer(5)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]])
{
    const uint key = tid.y;
    const uint head = tid.z;
    if (key >= p.packed_keys) return;
    const uint position = p.key_origin + key;
    const bool in_tile = position >= p.start;
    uint source_row = in_tile ? position - p.start : position;
    if (!in_tile && position >= p.cache_capacity) source_row %= p.cache_capacity;
    const uint source = (source_row * p.kv_heads + head) * 32;
    const uint dst = (head * p.packed_keys + key) * 32 + tid.x;
    device const float* key_lane = (in_tile ? k : k_cache) + source * 4 + tid.x;
    packed_k[dst] = float4(key_lane[0], key_lane[32], key_lane[64], key_lane[96]);
    packed_v[dst] = (in_tile ? v : v_cache)[source + tid.x];
}

static inline float ordered_qk_dot(float4 q, float4 k) {
    float partial = 0.0f;
    partial += q.x * k.x;
    partial += q.y * k.y;
    partial += q.z * k.z;
    partial += q.w * k.w;
    return simd_sum(partial) * rsqrt((float)ordered_channels);
}

static inline float ordered_qk_relative(float q, device const float* bias, ushort lane) {
    float partial = 0.0f;
    if (lane < 16) partial += q * bias[lane];
    return simd_sum(partial);
}

kernel void inkling2_prefill_ordered_qk(
    device const float* q                [[buffer(0)]],
    device const float* q_rel            [[buffer(1)]],
    device const float* relative_bias    [[buffer(2)]],
    device const float* sink             [[buffer(3)]],
    device const float4* packed_k        [[buffer(4)]],
    device       float* scores           [[buffer(5)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    const uint row = group.y * ordered_query_rows;
    const uint head = group.z;
    const uint kv_head = head / (p.query_heads / p.kv_heads);
    const uint first_position = p.start + row - p.key_origin;
    const uint begin = ordered_first_key(first_position, p) + group.x * ordered_key_tile;
    const uint end = min(begin + ordered_key_tile, first_position + min(p.rows - row, ordered_query_rows));
    if (begin >= end) return;

    float4 query[ordered_query_rows];
    float relative[ordered_query_rows];
    float sinks[ordered_query_rows];
    uint first_keys[ordered_query_rows];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        const uint src = min(row + i, p.rows - 1) * p.query_heads + head;
        device const float* q_lane = q + src * ordered_channels + lane;
        query[i] = float4(q_lane[0], q_lane[32], q_lane[64], q_lane[96]);
        relative[i] = lane < 16 ? q_rel[src * 16 + lane] : 0.0f;
        sinks[i] = sink[src];
        first_keys[i] = ordered_first_key(first_position + i, p);
    }

    if (p.rows - row >= ordered_query_rows &&
        begin >= first_keys[ordered_query_rows - 1] &&
        begin + ordered_key_tile - 1 <= first_position &&
        first_position + ordered_query_rows - 1 - begin < p.relative_extent) {
        // The full key tile is visible and has relative bias for all four
        // queries. Overlap two key loads and keep each FP32 reduction independent.
        for (uint offset = 0; offset < ordered_key_tile; offset += 2 * ordered_qk_simdgroups) {
            const uint key0 = begin + offset + simd;
            const uint key1 = key0 + ordered_qk_simdgroups;
            device const float4* key_lane = packed_k + (kv_head * p.packed_keys + key0) * 32 + lane;
            const float4 k0 = key_lane[0];
            const float4 k1 = key_lane[ordered_qk_simdgroups * 32];
            const uint distance0 = first_position - key0;
            const uint distance1 = first_position - key1;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < ordered_query_rows; ++i) {
                float score0 = ordered_qk_dot(query[i], k0);
                float score1 = ordered_qk_dot(query[i], k1);
                score0 += ordered_qk_relative(relative[i], relative_bias + (distance0 + i) * 16, lane);
                score1 += ordered_qk_relative(relative[i], relative_bias + (distance1 + i) * 16, lane);
                if (lane == 0) {
                    device float* line = scores + (head * p.rows + row + i) * p.score_stride;
                    line[key0 - first_keys[i]] = score0 - sinks[i];
                    line[key1 - first_keys[i]] = score1 - sinks[i];
                }
            }
        }
        return;
    }

    // Each SIMD group scores a key for several adjacent queries with one K
    // load. Each dot uses the four lane-local additions and SIMD reduction
    // from inkling2_resident_score in exactly the same order.
    for (uint key = begin + simd; key < end; key += ordered_qk_simdgroups) {
        const float4 k = packed_k[(kv_head * p.packed_keys + key) * 32 + lane];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ordered_query_rows; ++i) {
            const uint position = first_position + i;
            if (row + i >= p.rows || key < first_keys[i] || key > position) continue;
            float score = ordered_qk_dot(query[i], k);
            const uint distance = position - key;
            if (distance < p.relative_extent) {
                score += ordered_qk_relative(relative[i], relative_bias + distance * 16, lane);
            }
            if (lane == 0) {
                scores[(head * p.rows + row + i) * p.score_stride + key - first_keys[i]] = score - sinks[i];
            }
        }
    }
}

static inline float ordered_qk_broadcast_relative(float q, float bias, ushort lane) {
    float partial = 0.0f;
    if (lane < 16) partial += q * bias;
    return simd_sum(partial);
}

// The global cache is row-major and never wraps. The active span stays
// visible through its K/V views until it is committed, including rows
// that preceded the current query panel.
template <bool DirectKV>
static inline float4 ordered_load_key(
    device const float* keys,
    device const float* key_cache,
    uint head,
    uint key,
    ushort lane,
    constant Inkling2PrefillOrderedParams& p)
{
    if (DirectKV) {
        const bool in_tile = key >= p.source_start;
        const uint row = in_tile ? key - p.source_start : key;
        device const float* k = (in_tile ? keys : key_cache) +
            (row * p.kv_heads + head) * ordered_channels + lane;
        return float4(k[0], k[32], k[64], k[96]);
    }
    return ((device const float4*)keys)[(head * p.packed_keys + key) * 32 + lane];
}

template <bool DirectKV>
static inline void ordered_qk_gqa3(
    device const float* q,
    device const float* q_rel,
    device const float* relative_bias,
    device const float* sink,
    device const float* keys,
    device const float* key_cache,
    device float* scores,
    constant Inkling2PrefillOrderedParams& p,
    uint3 group,
    ushort simd,
    ushort lane)
{
    const uint row = group.y * gqa_rows;
    const uint kv_head = group.z;
    const uint first_head = kv_head * gqa_heads;
    const uint first_position = p.start + row - (DirectKV ? 0 : p.key_origin);
    const uint begin = (DirectKV ? 0 : ordered_first_key(first_position, p)) + group.x * gqa_key_tile;
    const uint end = min(begin + gqa_key_tile, first_position + min(p.rows - row, gqa_rows));
    if (begin >= end) return;

    // A SIMD group loads K once and scores two rows for all three Q heads
    // in the KV group. Reuse the lane's relative-bias value across heads too.
    float4 query[gqa_rows][gqa_heads];
    float relative[gqa_rows][gqa_heads];
    float sinks[gqa_rows][gqa_heads];
    uint first_keys[gqa_rows];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < gqa_rows; ++i) {
        first_keys[i] = DirectKV ? 0 : ordered_first_key(first_position + i, p);
        #pragma clang loop unroll(full)
        for (uint h = 0; h < gqa_heads; ++h) {
            const uint src = min(row + i, p.rows - 1) * p.query_heads + first_head + h;
            device const float* q_lane = q + src * ordered_channels + lane;
            query[i][h] = float4(q_lane[0], q_lane[32], q_lane[64], q_lane[96]);
            relative[i][h] = lane < 16 ? q_rel[src * 16 + lane] : 0.0f;
            sinks[i][h] = sink[src];
        }
    }

    // Full causal tiles overlap two independent K loads. Tiles at the
    // relative-extent or causal boundary take the per-key checks below.
    const uint last_key = begin + gqa_key_tile - 1;
    const bool full_tile = p.rows - row >= gqa_rows &&
                           begin >= first_keys[gqa_rows - 1] && last_key <= first_position;
    if (full_tile && first_position + gqa_rows - 1 - begin < p.relative_extent) {
        for (uint offset = 0; offset < gqa_key_tile; offset += 2 * gqa_simdgroups) {
            const uint key0 = begin + offset + simd;
            const uint key1 = key0 + gqa_simdgroups;
            const float4 k0 = ordered_load_key<DirectKV>(keys, key_cache, kv_head, key0, lane, p);
            const float4 k1 = ordered_load_key<DirectKV>(keys, key_cache, kv_head, key1, lane, p);
            const uint distance0 = first_position - key0;
            const uint distance1 = first_position - key1;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < gqa_rows; ++i) {
                const float bias0 = lane < 16 ? relative_bias[(distance0 + i) * 16 + lane] : 0.0f;
                const float bias1 = lane < 16 ? relative_bias[(distance1 + i) * 16 + lane] : 0.0f;
                #pragma clang loop unroll(full)
                for (uint h = 0; h < gqa_heads; ++h) {
                    float score0 = ordered_qk_dot(query[i][h], k0);
                    float score1 = ordered_qk_dot(query[i][h], k1);
                    score0 += ordered_qk_broadcast_relative(relative[i][h], bias0, lane);
                    score1 += ordered_qk_broadcast_relative(relative[i][h], bias1, lane);
                    if (lane == 0) {
                        device float* line = scores + ((first_head + h) * p.rows + row + i) * p.score_stride;
                        line[key0 - first_keys[i]] = score0 - sinks[i][h];
                        line[key1 - first_keys[i]] = score1 - sinks[i][h];
                    }
                }
            }
        }
        return;
    }

    if (DirectKV && full_tile && first_position - last_key >= p.relative_extent) {
        // All queries are past relative extent for every key in this tile.
        for (uint offset = 0; offset < gqa_key_tile; offset += 2 * gqa_simdgroups) {
            const uint key0 = begin + offset + simd;
            const uint key1 = key0 + gqa_simdgroups;
            const float4 k0 = ordered_load_key<DirectKV>(keys, key_cache, kv_head, key0, lane, p);
            const float4 k1 = ordered_load_key<DirectKV>(keys, key_cache, kv_head, key1, lane, p);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < gqa_rows; ++i) {
                #pragma clang loop unroll(full)
                for (uint h = 0; h < gqa_heads; ++h) {
                    const float score0 = ordered_qk_dot(query[i][h], k0);
                    const float score1 = ordered_qk_dot(query[i][h], k1);
                    if (lane == 0) {
                        device float* line = scores + ((first_head + h) * p.rows + row + i) * p.score_stride;
                        line[key0] = score0 - sinks[i][h];
                        line[key1] = score1 - sinks[i][h];
                    }
                }
            }
        }
        return;
    }

    for (uint key = begin + simd; key < end; key += gqa_simdgroups) {
        const float4 k = ordered_load_key<DirectKV>(keys, key_cache, kv_head, key, lane, p);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < gqa_rows; ++i) {
            const uint position = first_position + i;
            if (row + i >= p.rows || key < first_keys[i] || key > position) continue;
            const uint distance = position - key;
            const bool has_bias = distance < p.relative_extent;
            const float bias = has_bias && lane < 16 ? relative_bias[distance * 16 + lane] : 0.0f;
            #pragma clang loop unroll(full)
            for (uint h = 0; h < gqa_heads; ++h) {
                float score = ordered_qk_dot(query[i][h], k);
                if (has_bias) score += ordered_qk_broadcast_relative(relative[i][h], bias, lane);
                if (lane == 0) {
                    scores[((first_head + h) * p.rows + row + i) * p.score_stride + key - first_keys[i]] = score - sinks[i][h];
                }
            }
        }
    }
}

kernel void inkling2_prefill_ordered_qk_gqa3(
    device const float* q                [[buffer(0)]],
    device const float* q_rel            [[buffer(1)]],
    device const float* relative_bias    [[buffer(2)]],
    device const float* sink             [[buffer(3)]],
    device const float4* packed_k        [[buffer(4)]],
    device       float* scores           [[buffer(5)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    ordered_qk_gqa3<false>(q, q_rel, relative_bias, sink, (device const float*)packed_k,
        nullptr, scores, p, group, simd, lane);
}

kernel void inkling2_prefill_global_qk_gqa3(
    device const float* q                [[buffer(0)]],
    device const float* q_rel            [[buffer(1)]],
    device const float* relative_bias    [[buffer(2)]],
    device const float* sink             [[buffer(3)]],
    device const float* keys             [[buffer(4)]],
    device       float* scores           [[buffer(5)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(6)]],
    device const float* key_cache        [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort simd [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]])
{
    ordered_qk_gqa3<true>(q, q_rel, relative_bias, sink, keys, key_cache, scores, p, group, simd, lane);
}

template <bool DirectKV>
static inline float4 ordered_load_value(
    device const float4* values,
    device const float4* value_cache,
    uint head,
    uint key,
    ushort lane,
    constant Inkling2PrefillOrderedParams& p)
{
    if (DirectKV) {
        const bool in_tile = key >= p.source_start;
        const uint row = in_tile ? key - p.source_start : key;
        return (in_tile ? values : value_cache)[(row * p.kv_heads + head) * 32 + lane];
    }
    return values[(head * p.packed_keys + key) * 32 + lane];
}

template <bool DirectKV>
static inline void ordered_softmax_pv(
    device const float4* values,
    device const float4* value_cache,
    device const float* scores,
    device const float4* gate,
    device float4* output,
    constant Inkling2PrefillOrderedParams& p,
    threadgroup float* cached_weights,
    uint3 group,
    ushort lane)
{
    const uint row = group.y * ordered_query_rows;
    const uint head = group.z;
    const uint kv_head = head / (p.query_heads / p.kv_heads);
    const uint common_end = p.start + row + 1 - (DirectKV ? 0 : p.key_origin);
    const uint active_rows = min(ordered_query_rows, p.rows - row);
    device const float* lines[ordered_query_rows];
    float maxima[ordered_query_rows];
    float denominator[ordered_query_rows];
    float4 context[ordered_query_rows];
    uint first_keys[ordered_query_rows];
    uint counts[ordered_query_rows];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        const uint active = min(i, active_rows - 1);
        lines[i] = scores + (head * p.rows + row + active) * p.score_stride;
        first_keys[i] = DirectKV ? 0 : ordered_first_key(common_end + active - 1, p);
        counts[i] = common_end + active - first_keys[i];
        maxima[i] = 0.0f;
        context[i] = float4(0.0f);
    }
    // The sink contributes a score of zero. Reduce only the maximum in
    // parallel and leave both FP32 sums in chronological order.
    const uint max_count = counts[ordered_query_rows - 1];
    for (uint key = lane; key < max_count; key += 32) {
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ordered_query_rows; ++i) {
            if (key < counts[i]) maxima[i] = max(maxima[i], lines[i][key]);
        }
    }
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        maxima[i] = simd_max(maxima[i]);
        denominator[i] = exp(-maxima[i]);
    }

    // Advance each query to the intersection in chronological order. The
    // intersection is empty if the logical window is narrower than the group.
    const uint common_begin = min(first_keys[ordered_query_rows - 1], common_end);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        for (uint key = first_keys[i]; key < common_begin; ++key) {
            const float weight = exp(lines[i][key - first_keys[i]] - maxima[i]);
            denominator[i] += weight;
            context[i] += weight * ordered_load_value<DirectKV>(values, value_cache, kv_head, key, lane, p);
        }
    }

    // One SIMD group stages a stripe of weights for all four queries and
    // reuses each V tuple in the intersection before overwriting the stripe.
    constexpr uint stripe = ordered_pv_stripe;
    for (uint block = common_begin; block < common_end; block += stripe) {
        const uint count = min(stripe, common_end - block);
        for (uint offset = lane; offset < count; offset += 32) {
            const uint key = block + offset;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < ordered_query_rows; ++i) {
                cached_weights[i * stripe + offset] = exp(lines[i][key - first_keys[i]] - maxima[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll_count(2)
        for (uint offset = 0; offset < count; ++offset) {
            const float4 value = ordered_load_value<DirectKV>(values, value_cache, kv_head, block + offset, lane, p);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < ordered_query_rows; ++i) {
                const float weight = cached_weights[i * stripe + offset];
                denominator[i] += weight;
                context[i] += weight * value;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Finish the extra causal keys for the later queries in the group.
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        if (i >= active_rows) continue;
        for (uint key = max(common_end, first_keys[i]); key < common_end + i; ++key) {
            const float weight = exp(lines[i][key - first_keys[i]] - maxima[i]);
            denominator[i] += weight;
            context[i] += weight * ordered_load_value<DirectKV>(values, value_cache, kv_head, key, lane, p);
        }
        const uint dst = ((row + i) * p.query_heads + head) * 32 + lane;
        const float4 sigmoid = 1.0f / (1.0f + exp(-gate[dst]));
        output[dst] = (context[i] / denominator[i]) * sigmoid;
    }
}

kernel void inkling2_prefill_ordered_softmax_pv(
    device const float4* packed_v        [[buffer(0)]],
    device const float*  scores           [[buffer(1)]],
    device const float4* gate            [[buffer(2)]],
    device       float4* output          [[buffer(3)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float cached_weights[ordered_query_rows * ordered_pv_stripe];
    ordered_softmax_pv<false>(packed_v, nullptr, scores, gate, output, p, cached_weights, group, lane);
}

kernel void inkling2_prefill_global_softmax_pv(
    device const float4* values          [[buffer(0)]],
    device const float* scores           [[buffer(1)]],
    device const float4* gate            [[buffer(2)]],
    device float4* output                [[buffer(3)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(4)]],
    device const float4* value_cache     [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]])
{
    threadgroup float cached_weights[ordered_query_rows * ordered_pv_stripe];
    ordered_softmax_pv<true>(values, value_cache, scores, gate, output, p, cached_weights, group, lane);
}
