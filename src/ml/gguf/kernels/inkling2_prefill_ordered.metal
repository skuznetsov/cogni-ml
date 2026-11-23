#include <metal_stdlib>
using namespace metal;

constant uint ordered_query_rows = 4;
constant uint ordered_key_tile = 64;
constant uint ordered_qk_simdgroups = 4; // 128 threads per QK threadgroup.
constant uint ordered_channels = 128;

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

kernel void inkling2_prefill_ordered_weights(
    device       float* scores           [[buffer(0)]],
    device       float* sink_weights     [[buffer(1)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(2)]],
    threadgroup float* cached_scores     [[threadgroup(0)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    const uint row = group.y;
    const uint keys = p.start + p.rows;
    const uint count = p.start + row + 1;
    const uint line_index = group.z * p.rows + row;
    device float* line = scores + line_index * keys;
    for (uint key = tid; key < count; key += 128) cached_scores[key] = line[key];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float max_score = 0.0f;
        for (uint key = 0; key < count; ++key) max_score = max(max_score, cached_scores[key]);
        cached_scores[keys] = max_score;
        sink_weights[line_index] = exp(-max_score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float max_score = cached_scores[keys];
    for (uint key = tid; key < count; key += 128) line[key] = exp(cached_scores[key] - max_score);
}

kernel void inkling2_prefill_ordered_pv(
    device const float4* packed_v        [[buffer(0)]],
    device const float* weights          [[buffer(1)]],
    device const float* sink_weights     [[buffer(2)]],
    device const float4* gate            [[buffer(3)]],
    device       float4* output          [[buffer(4)]],
    constant Inkling2PrefillOrderedParams& p [[buffer(5)]],
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
    device const float* streams[ordered_query_rows];
    float4 context[ordered_query_rows];
    float denominator[ordered_query_rows];

    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        const uint line_index = head * p.rows + row + min(i, active_rows - 1);
        streams[i] = weights + line_index * keys;
        context[i] = float4(0.0f);
        denominator[i] = sink_weights[line_index];
    }

    // Reuse each V tuple for the independent query accumulators. Every
    // accumulator still visits keys one at a time in ascending order.
    for (uint key = 0; key < common; ++key) {
        const float4 value = values[key * 32];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < ordered_query_rows; ++i) {
            const float weight = streams[i][key];
            denominator[i] += weight;
            context[i] += weight * value;
        }
    }

    #pragma clang loop unroll(full)
    for (uint i = 0; i < ordered_query_rows; ++i) {
        if (i >= active_rows) continue;
        #pragma clang loop unroll(full)
        for (uint tail = 0; tail < i; ++tail) {
            const uint key = common + tail;
            const float weight = streams[i][key];
            denominator[i] += weight;
            context[i] += weight * values[key * 32];
        }
        const uint dst = ((row + i) * p.query_heads + head) * 32 + lane;
        const float4 sigmoid = 1.0f / (1.0f + exp(-gate[dst]));
        output[dst] = (context[i] / denominator[i]) * sigmoid;
    }
}
