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
};

// The selector bounds this path to the unwrapped prefix. Pack each SIMD lane's
// four QK operands together and leave the four adjacent PV output channels
// together. Both panels are chronological inside each KV head.
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
    const uint keys = p.start + p.rows;
    const uint key = tid.y;
    const uint head = tid.z;
    if (key >= keys) return;
    const bool in_tile = key >= p.start;
    const uint source = ((in_tile ? key - p.start : key) * p.kv_heads + head) * 32;
    const uint dst = (head * keys + key) * 32 + tid.x;
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
    const uint keys = p.start + p.rows;
    const uint row = group.y * ordered_query_rows;
    const uint head = group.z;
    const uint kv_head = head / (p.query_heads / p.kv_heads);
    const uint begin = group.x * ordered_key_tile;
    const uint end = min(begin + ordered_key_tile, p.start + min(p.rows, row + ordered_query_rows));
    if (begin >= end) return;

    float4 query[ordered_query_rows];
    float relative[ordered_query_rows];
    float sinks[ordered_query_rows];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        const uint src = min(row + i, p.rows - 1) * p.query_heads + head;
        device const float* q_lane = q + src * ordered_channels + lane;
        query[i] = float4(q_lane[0], q_lane[32], q_lane[64], q_lane[96]);
        relative[i] = lane < 16 ? q_rel[src * 16 + lane] : 0.0f;
        sinks[i] = sink[src];
    }

    const uint first_position = p.start + row;
    if (p.rows - row >= ordered_query_rows &&
        begin + ordered_key_tile - 1 <= first_position &&
        first_position + ordered_query_rows - 1 - begin < p.relative_extent) {
        // The full key tile is visible and has relative bias for all four
        // queries. Overlap two key loads and keep each FP32 reduction independent.
        for (uint offset = 0; offset < ordered_key_tile; offset += 2 * ordered_qk_simdgroups) {
            const uint key0 = begin + offset + simd;
            const uint key1 = key0 + ordered_qk_simdgroups;
            device const float4* key_lane = packed_k + (kv_head * keys + key0) * 32 + lane;
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
                    device float* line = scores + (head * p.rows + row + i) * keys;
                    line[key0] = score0 - sinks[i];
                    line[key1] = score1 - sinks[i];
                }
            }
        }
        return;
    }

    // Each SIMD group scores a key for several adjacent queries with one K
    // load. Each dot uses the four lane-local additions and SIMD reduction
    // from inkling2_resident_score in exactly the same order.
    for (uint key = begin + simd; key < end; key += ordered_qk_simdgroups) {
        const float4 k = packed_k[(kv_head * keys + key) * 32 + lane];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ordered_query_rows; ++i) {
            const uint position = p.start + row + i;
            if (row + i >= p.rows || key > position) continue;
            float score = ordered_qk_dot(query[i], k);
            const uint distance = position - key;
            if (distance < p.relative_extent) {
                score += ordered_qk_relative(relative[i], relative_bias + distance * 16, lane);
            }
            if (lane == 0) {
                scores[(head * p.rows + row + i) * keys + key] = score - sinks[i];
            }
        }
    }
}

static inline float ordered_qk_broadcast_relative(float q, float bias, ushort lane) {
    float partial = 0.0f;
    if (lane < 16) partial += q * bias;
    return simd_sum(partial);
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
    const uint keys = p.start + p.rows;
    const uint row = group.y * gqa_rows;
    const uint kv_head = group.z;
    const uint first_head = kv_head * gqa_heads;
    const uint begin = group.x * gqa_key_tile;
    const uint end = min(begin + gqa_key_tile, p.start + min(p.rows, row + gqa_rows));
    if (begin >= end) return;

    // A SIMD group loads K once and scores two rows for all three Q heads
    // in the KV group. Reuse the lane's relative-bias value across heads too.
    float4 query[gqa_rows][gqa_heads];
    float relative[gqa_rows][gqa_heads];
    float sinks[gqa_rows][gqa_heads];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < gqa_rows; ++i) {
        #pragma clang loop unroll(full)
        for (uint h = 0; h < gqa_heads; ++h) {
            const uint src = min(row + i, p.rows - 1) * p.query_heads + first_head + h;
            device const float* q_lane = q + src * ordered_channels + lane;
            query[i][h] = float4(q_lane[0], q_lane[32], q_lane[64], q_lane[96]);
            relative[i][h] = lane < 16 ? q_rel[src * 16 + lane] : 0.0f;
            sinks[i][h] = sink[src];
        }
    }

    const uint first_position = p.start + row;
    // Full causal tiles overlap two independent K loads. The edge path
    // below only writes visible scores and preserves the conditional bias.
    if (p.rows - row >= gqa_rows &&
        begin + gqa_key_tile - 1 <= first_position &&
        first_position + gqa_rows - 1 - begin < p.relative_extent) {
        for (uint offset = 0; offset < gqa_key_tile; offset += 2 * gqa_simdgroups) {
            const uint key0 = begin + offset + simd;
            device const float4* key_lane = packed_k + (kv_head * keys + key0) * 32 + lane;
            const float4 k0 = key_lane[0];
            const uint distance0 = first_position - key0;
            const uint key1 = key0 + gqa_simdgroups;
            const float4 k1 = key_lane[gqa_simdgroups * 32];
            const uint distance1 = distance0 - gqa_simdgroups;
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
                        device float* line = scores + ((first_head + h) * p.rows + row + i) * keys;
                        line[key0] = score0 - sinks[i][h];
                        line[key1] = score1 - sinks[i][h];
                    }
                }
            }
        }
        return;
    }

    for (uint key = begin + simd; key < end; key += gqa_simdgroups) {
        const float4 k = packed_k[(kv_head * keys + key) * 32 + lane];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < gqa_rows; ++i) {
            const uint position = first_position + i;
            if (row + i >= p.rows || key > position) continue;
            const uint distance = position - key;
            const bool has_bias = distance < p.relative_extent;
            const float bias = has_bias && lane < 16 ? relative_bias[distance * 16 + lane] : 0.0f;
            #pragma clang loop unroll(full)
            for (uint h = 0; h < gqa_heads; ++h) {
                float score = ordered_qk_dot(query[i][h], k);
                if (has_bias) score += ordered_qk_broadcast_relative(relative[i][h], bias, lane);
                if (lane == 0) {
                    scores[((first_head + h) * p.rows + row + i) * keys + key] = score - sinks[i][h];
                }
            }
        }
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
    const uint row = group.y * ordered_query_rows;
    const uint head = group.z;
    const uint kv_head = head / (p.query_heads / p.kv_heads);
    const uint keys = p.start + p.rows;
    const uint common = p.start + row + 1;
    const uint active_rows = min(ordered_query_rows, p.rows - row);
    device const float4* values = packed_v + kv_head * keys * 32 + lane;
    device const float* lines[ordered_query_rows];
    float maxima[ordered_query_rows];
    float denominator[ordered_query_rows];
    float4 context[ordered_query_rows];
    uint counts[ordered_query_rows];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        const uint active = min(i, active_rows - 1);
        lines[i] = scores + (head * p.rows + row + active) * keys;
        counts[i] = common + active;
        maxima[i] = 0.0f;
        context[i] = float4(0.0f);
    }
    // The sink contributes a score of zero. Reduce only the maximum in
    // parallel and leave both FP32 sums in chronological order.
    const uint max_count = common + active_rows - 1;
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

    // One SIMD group stages a fixed stripe of weights for all four queries
    // and reuses each V tuple before overwriting the stripe.
    constexpr uint stripe = ordered_pv_stripe;
    threadgroup float cached_weights[ordered_query_rows * stripe];
    for (uint block = 0; block < common; block += stripe) {
        const uint count = min(stripe, common - block);
        for (uint offset = lane; offset < count; offset += 32) {
            const uint key = block + offset;
            #pragma clang loop unroll(full)
            for (uint i = 0; i < ordered_query_rows; ++i) {
                cached_weights[i * stripe + offset] = exp(lines[i][key] - maxima[i]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll_count(2)
        for (uint offset = 0; offset < count; ++offset) {
            const float4 value = values[(block + offset) * 32];
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
        #pragma clang loop unroll(full)
        for (uint tail = 0; tail < i; ++tail) {
            const uint key = common + tail;
            const float weight = exp(lines[i][key] - maxima[i]);
            denominator[i] += weight;
            context[i] += weight * values[key * 32];
        }
        const uint dst = ((row + i) * p.query_heads + head) * 32 + lane;
        const float4 sigmoid = 1.0f / (1.0f + exp(-gate[dst]));
        output[dst] = (context[i] / denominator[i]) * sigmoid;
    }
}
