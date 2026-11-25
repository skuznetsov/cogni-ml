#include <metal_stdlib>
using namespace metal;

constant uint inkling2_flash_bq = 32;
constant uint inkling2_flash_bk = 16;
constant uint inkling2_flash_threads = 128;
constant uint inkling2_flash_head_dim = 128;
constant uint inkling2_flash_relative_dim = 16;
constant uint inkling2_flash_ldk = inkling2_flash_bk + 8;
constant uint inkling2_flash_ldv = inkling2_flash_head_dim + 8;
constant float inkling2_flash_log2e = 1.4426950408889634f;

// Host ABI: eight uint32 fields in this exact order (32 bytes).
struct Inkling2FlashParams {
    uint start;
    uint rows;
    uint query_heads;
    uint kv_heads;
    uint relative_extent;
    uint sliding_window;
    uint cache_capacity;
    uint relative_stride;
};

// The streaming 8x16 online-softmax structure is independently implemented
// for this flat F32 ABI. Algorithmic reference: oMLX
// omlx/patches/inkling2/kernels.py::_FLASH_SOURCE (Apache-2.0).
static inline float inkling2_flash_relative_dot(
    device const float* query_relative,
    device const float* relative_bias,
    uint distance)
{
    float value = 0.0f;
    const size_t qbase = 0;
    const size_t bbase = (size_t)distance * inkling2_flash_relative_dim;
    #pragma clang loop unroll(full)
    for (uint dim = 0; dim < inkling2_flash_relative_dim; ++dim) {
        value += query_relative[qbase + dim] * relative_bias[bbase + dim];
    }
    return value;
}

template <bool FastMath>
static inline float inkling2_flash_exp2(float value)
{
    if (FastMath) return fast::exp2(value);
    return precise::exp2(value);
}

static inline float inkling2_flash_relative_value(
    device const float* q_rel,
    device const float* relative_bias,
    device const float* relative_table,
    uint row,
    uint head,
    uint distance,
    constant Inkling2FlashParams& p)
{
    if (distance >= p.relative_extent) return 0.0f;
    if (p.relative_stride != 0 && distance < p.relative_stride) {
        const size_t index = ((size_t)row * p.query_heads + head) * p.relative_stride + distance;
        return relative_table[index];
    }
    const size_t qbase = ((size_t)row * p.query_heads + head) * inkling2_flash_relative_dim;
    return inkling2_flash_relative_dot(q_rel + qbase, relative_bias, distance);
}

static inline float inkling2_flash_load_key(
    device const float* local_k,
    device const float* cache_k,
    uint key,
    uint kv_head,
    uint channel,
    constant Inkling2FlashParams& p)
{
    if (key >= p.start) {
        const size_t index = (((size_t)(key - p.start) * p.kv_heads + kv_head) * inkling2_flash_head_dim) + channel;
        return local_k[index];
    }
    if (p.cache_capacity == 0) return 0.0f;
    const uint cache_row = key % p.cache_capacity;
    const size_t index = (((size_t)cache_row * p.kv_heads + kv_head) * inkling2_flash_head_dim) + channel;
    return cache_k[index];
}

static inline float inkling2_flash_load_value(
    device const float* local_v,
    device const float* cache_v,
    uint key,
    uint kv_head,
    uint channel,
    constant Inkling2FlashParams& p)
{
    if (key >= p.start) {
        const size_t index = (((size_t)(key - p.start) * p.kv_heads + kv_head) * inkling2_flash_head_dim) + channel;
        return local_v[index];
    }
    if (p.cache_capacity == 0) return 0.0f;
    const uint cache_row = key % p.cache_capacity;
    const size_t index = (((size_t)cache_row * p.kv_heads + kv_head) * inkling2_flash_head_dim) + channel;
    return cache_v[index];
}

template <typename Operand, bool FastMath = false, bool VectorLoader = false>
static inline void inkling2_flash_core(
    device const float* q,
    device const float* local_k,
    device const float* local_v,
    device const float* q_rel,
    device const float* relative_bias,
    device const float* gate,
    device const float* sink,
    device const float* cache_k,
    device const float* cache_v,
    device float* output,
    device const float* relative_table,
    constant Inkling2FlashParams& p,
    uint3 group,
    uint tid,
    threadgroup Operand* key_stage,
    threadgroup Operand* value_stage)
{
    using OperandMatrix = simdgroup_matrix<Operand, 8, 8>;
    using AccumulatorMatrix = simdgroup_matrix<float, 8, 8>;

    const uint simd = tid / 32;
    const uint lane = tid % 32;
    const uint qid = lane / 4;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint q0 = group.x * inkling2_flash_bq;
    const uint head = group.y;
    const uint head_group = p.kv_heads == 0 ? 1u : max(1u, p.query_heads / p.kv_heads);
    const uint kv_head = head / head_group;
    const uint row = q0 + simd * 8u + fm;
    const bool row_ok = row < p.rows;
    const uint safe_row = p.rows == 0 ? 0u : min(row, p.rows - 1u);
    const uint position = p.start + safe_row;

    OperandMatrix query_matrix[inkling2_flash_head_dim / 8];
    AccumulatorMatrix output_matrix[inkling2_flash_head_dim / 8];
    #pragma clang loop unroll(full)
    for (uint db = 0; db < inkling2_flash_head_dim / 8; ++db) {
        const size_t qbase = ((size_t)safe_row * p.query_heads + head) * inkling2_flash_head_dim + db * 8u + fn;
        query_matrix[db].thread_elements()[0] = row_ok ? Operand(q[qbase]) : Operand(0.0f);
        query_matrix[db].thread_elements()[1] = row_ok ? Operand(q[qbase + 1u]) : Operand(0.0f);
        output_matrix[db] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint group_last_row = min(q0 + inkling2_flash_bq, p.rows);
    if (group_last_row == 0u) return;
    const uint last_position = p.start + group_last_row - 1u;
    uint first_key = 0u;
    if (p.sliding_window != 0u) {
        const uint first_position = p.start + q0;
        const uint visible = min(p.sliding_window, first_position + 1u);
        first_key = first_position + 1u - visible;
    }
    const uint key_begin = (first_key / inkling2_flash_bk) * inkling2_flash_bk;
    const float qscale = rsqrt((float)inkling2_flash_head_dim) * inkling2_flash_log2e;

    float maximum = row_ok ? sink[(size_t)safe_row * p.query_heads + head] * inkling2_flash_log2e : 0.0f;
    float denominator = 1.0f;

    for (uint key0 = key_begin; key0 <= last_position; key0 += inkling2_flash_bk) {
        if (VectorLoader) {
            // Four contiguous F32 channels share one source-row and base calculation
            // for both K and V. The typed vector loads require 16-byte-aligned
            // bindings and the native 128-channel row stride.
            for (uint item = tid; item < inkling2_flash_bk * (inkling2_flash_head_dim / 4u); item += inkling2_flash_threads) {
                const uint key_offset = item / (inkling2_flash_head_dim / 4u);
                const uint channel = (item % (inkling2_flash_head_dim / 4u)) * 4u;
                const uint key = key0 + key_offset;
                float4 kval = float4(0.0f);
                float4 vval = float4(0.0f);
                if (key < p.start + p.rows) {
                    if (key >= p.start) {
                        const uint source_row = key - p.start;
                        const size_t source_base = (((size_t)source_row * p.kv_heads + kv_head) * inkling2_flash_head_dim) + channel;
                        kval = *((device const float4*)(local_k + source_base));
                        vval = *((device const float4*)(local_v + source_base));
                    } else if (p.cache_capacity != 0u) {
                        const uint source_row = p.sliding_window == 0u
                            ? key : key % p.cache_capacity;
                        const size_t source_base = (((size_t)source_row * p.kv_heads + kv_head) * inkling2_flash_head_dim) + channel;
                        kval = *((device const float4*)(cache_k + source_base));
                        vval = *((device const float4*)(cache_v + source_base));
                    }
                }
                const size_t stage_k = (size_t)channel * inkling2_flash_ldk + key_offset;
                const size_t stage_v = (size_t)key_offset * inkling2_flash_ldv + channel;
                key_stage[stage_k] = Operand(kval.x);
                key_stage[stage_k + inkling2_flash_ldk] = Operand(kval.y);
                key_stage[stage_k + 2u * inkling2_flash_ldk] = Operand(kval.z);
                key_stage[stage_k + 3u * inkling2_flash_ldk] = Operand(kval.w);
                value_stage[stage_v] = Operand(vval.x);
                value_stage[stage_v + 1u] = Operand(vval.y);
                value_stage[stage_v + 2u] = Operand(vval.z);
                value_stage[stage_v + 3u] = Operand(vval.w);
            }
        } else {
            for (uint item = tid; item < inkling2_flash_bk * inkling2_flash_head_dim; item += inkling2_flash_threads) {
                const uint key_offset = item / inkling2_flash_head_dim;
                const uint channel = item % inkling2_flash_head_dim;
                const uint key = key0 + key_offset;
                const size_t stage_k = (size_t)channel * inkling2_flash_ldk + key_offset;
                const size_t stage_v = (size_t)key_offset * inkling2_flash_ldv + channel;
                const bool in_sequence = key < p.start + p.rows;
                const float kval = in_sequence
                    ? inkling2_flash_load_key(local_k, cache_k, key, kv_head, channel, p) : 0.0f;
                const float vval = in_sequence
                    ? inkling2_flash_load_value(local_v, cache_v, key, kv_head, channel, p) : 0.0f;
                key_stage[stage_k] = Operand(kval);
                value_stage[stage_v] = Operand(vval);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        AccumulatorMatrix score_matrix[inkling2_flash_bk / 8];
        #pragma clang loop unroll(full)
        for (uint kb = 0; kb < inkling2_flash_bk / 8; ++kb) {
            score_matrix[kb] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            #pragma clang loop unroll(full)
            for (uint db = 0; db < inkling2_flash_head_dim / 8; ++db) {
                OperandMatrix key_matrix;
                simdgroup_load(key_matrix, key_stage + db * 8u * inkling2_flash_ldk + kb * 8u, inkling2_flash_ldk);
                simdgroup_multiply_accumulate(score_matrix[kb], query_matrix[db], key_matrix, score_matrix[kb]);
            }
        }

        float row_max = -INFINITY;
        for (uint kb = 0; kb < inkling2_flash_bk / 8; ++kb) {
            #pragma clang loop unroll(full)
            for (uint element = 0; element < 2; ++element) {
                const uint key = key0 + kb * 8u + fn + element;
                const bool causal = key <= position;
                const uint distance = causal ? position - key : 0u;
                const bool visible = row_ok && causal &&
                    (p.sliding_window == 0u || distance < p.sliding_window);
                float score = -INFINITY;
                if (visible) {
                    score = score_matrix[kb].thread_elements()[element] * qscale;
                    const float relative = inkling2_flash_relative_value(
                        q_rel, relative_bias, relative_table, safe_row, head, distance, p);
                    score += relative * inkling2_flash_log2e;
                }
                score_matrix[kb].thread_elements()[element] = score;
                row_max = max(row_max, score);
            }
        }
        row_max = max(row_max, simd_shuffle_xor(row_max, 1));
        row_max = max(row_max, simd_shuffle_xor(row_max, 8));
        const float next_maximum = max(maximum, row_max);
        const float rescale = inkling2_flash_exp2<FastMath>(maximum - next_maximum);
        float tile_sum = 0.0f;
        OperandMatrix probability_matrix[inkling2_flash_bk / 8];
        for (uint kb = 0; kb < inkling2_flash_bk / 8; ++kb) {
            #pragma clang loop unroll(full)
            for (uint element = 0; element < 2; ++element) {
                const float probability = inkling2_flash_exp2<FastMath>(score_matrix[kb].thread_elements()[element] - next_maximum);
                probability_matrix[kb].thread_elements()[element] = Operand(probability);
                tile_sum += probability;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        denominator = denominator * rescale + tile_sum;
        maximum = next_maximum;

        #pragma clang loop unroll(full)
        for (uint db = 0; db < inkling2_flash_head_dim / 8; ++db) {
            output_matrix[db].thread_elements()[0] *= rescale;
            output_matrix[db].thread_elements()[1] *= rescale;
            #pragma clang loop unroll(full)
            for (uint kb = 0; kb < inkling2_flash_bk / 8; ++kb) {
                OperandMatrix value_matrix;
                simdgroup_load(value_matrix, value_stage + kb * 8u * inkling2_flash_ldv + db * 8u, inkling2_flash_ldv);
                simdgroup_multiply_accumulate(output_matrix[db], probability_matrix[kb], value_matrix, output_matrix[db]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_ok) {
        const size_t row_base = ((size_t)row * p.query_heads + head) * inkling2_flash_head_dim;
        #pragma clang loop unroll(full)
        for (uint db = 0; db < inkling2_flash_head_dim / 8; ++db) {
            const float context0 = output_matrix[db].thread_elements()[0] / denominator;
            const float context1 = output_matrix[db].thread_elements()[1] / denominator;
            const float gate0 = gate[row_base + db * 8u + fn];
            const float gate1 = gate[row_base + db * 8u + fn + 1u];
            output[row_base + db * 8u + fn] = context0 / (1.0f + precise::exp(-gate0));
            output[row_base + db * 8u + fn + 1u] = context1 / (1.0f + precise::exp(-gate1));
        }
    }
}

// Half-operand vector-load Flash prefill with inline relative bias.
kernel void inkling2_flash_f16_vec(
    device const float* q                  [[buffer(0)]],
    device const float* local_k            [[buffer(1)]],
    device const float* local_v            [[buffer(2)]],
    device const float* q_rel              [[buffer(3)]],
    device const float* relative_bias      [[buffer(4)]],
    device const float* gate                [[buffer(5)]],
    device const float* sink                [[buffer(6)]],
    device const float* cache_k             [[buffer(7)]],
    device const float* cache_v             [[buffer(8)]],
    device float* output                    [[buffer(9)]],
    constant Inkling2FlashParams& p         [[buffer(10)]],
    device const float* relative_table      [[buffer(11)]],
    uint3 group                             [[threadgroup_position_in_grid]],
    uint tid                                [[thread_index_in_threadgroup]])
{
    threadgroup half key_stage[inkling2_flash_head_dim * inkling2_flash_ldk];
    threadgroup half value_stage[inkling2_flash_bk * inkling2_flash_ldv];
    inkling2_flash_core<half, false, true>(q, local_k, local_v, q_rel, relative_bias, gate, sink,
        cache_k, cache_v, output, relative_table, p, group, tid, key_stage, value_stage);
}
