#include <metal_stdlib>
using namespace metal;

constant uint INKLING2_RESIDENT_THREADS = 256;
constant uint INKLING2_RESIDENT_SIMDGROUPS = 8;

inline float inkling2_resident_rms_scale(threadgroup const float* partials, uint width, float eps) {
    float total = 0.0f;
    for (uint group = 0; group < INKLING2_RESIDENT_SIMDGROUPS; ++group) {
        total += partials[group];
    }
    return 1.0f / sqrt(total / (float)width + eps);
}

inline void inkling2_resident_rms_row(
    device const float* x,
    device const float* weights,
    device       float* y,
    uint width,
    float eps,
    uint has_weights,
    ushort tiisg,
    ushort sgitg,
    ushort tid,
    threadgroup float* partials)
{
    float square_sum = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        const float value = x[column];
        square_sum += value * value;
    }

    const float simd_sum_squares = simd_sum(square_sum);
    if (tiisg == 0) partials[sgitg] = simd_sum_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = inkling2_resident_rms_scale(partials, width, eps);
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        const float normalized = x[column] * scale;
        y[column] = has_weights != 0 ? normalized * weights[column] : normalized;
    }
}

kernel void inkling2_resident_rms(
    device const float* input       [[buffer(0)]],
    device const float* weights     [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant     uint&  width       [[buffer(3)]],
    constant     uint&  rows        [[buffer(4)]],
    constant     float& eps         [[buffer(5)]],
    constant     uint&  has_weights [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint row = tgpig.x;
    if (row >= rows) return;

    const size_t offset = (size_t)row * width;
    threadgroup float partials[INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_rms_row(input + offset, weights, output + offset, width, eps, has_weights,
        tiisg, sgitg, tid, partials);
}

// Scalar Q, K, and V have independent weights and may read the same input.
// Give each row the same full threadgroup and reduction order as scalar RMS.
kernel void inkling2_resident_rms_triplet(
    device const float* x0          [[buffer(0)]],
    device const float* x1          [[buffer(1)]],
    device const float* x2          [[buffer(2)]],
    device const float* weight0     [[buffer(3)]],
    device const float* weight1     [[buffer(4)]],
    device const float* weight2     [[buffer(5)]],
    device       float* output      [[buffer(6)]],
    constant     uint&  width       [[buffer(7)]],
    constant     float& eps         [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint part = tgpig.x;
    device const float* input = part == 0 ? x0 : (part == 1 ? x1 : x2);
    device const float* weights = part == 0 ? weight0 : (part == 1 ? weight1 : weight2);
    threadgroup float partials[INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_rms_row(input, weights, output + (size_t)part * width, width, eps, 1,
        tiisg, sgitg, tid, partials);
}

inline void inkling2_resident_rms_concat_row(
    device const float* x_row,
    device const float* m_row,
    device       float* out_row,
    uint width,
    float eps,
    ushort tiisg,
    ushort sgitg,
    ushort tid,
    threadgroup float* partials)
{
    float x_squares = 0.0f;
    float m_squares = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        const float xv = x_row[column];
        const float mv = m_row[column];
        x_squares += xv * xv;
        m_squares += mv * mv;
    }

    const float x_group_sum = simd_sum(x_squares);
    const float m_group_sum = simd_sum(m_squares);
    if (tiisg == 0) {
        partials[sgitg] = x_group_sum;
        partials[INKLING2_RESIDENT_SIMDGROUPS + sgitg] = m_group_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float x_total = 0.0f;
    float m_total = 0.0f;
    for (uint group = 0; group < INKLING2_RESIDENT_SIMDGROUPS; ++group) {
        x_total += partials[group];
        m_total += partials[INKLING2_RESIDENT_SIMDGROUPS + group];
    }
    const float x_scale = 1.0f / sqrt(x_total / (float)width + eps);
    const float m_scale = 1.0f / sqrt(m_total / (float)width + eps);
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        out_row[column] = x_row[column] * x_scale;
        out_row[width + column] = m_row[column] * m_scale;
    }
}

kernel void inkling2_resident_rms_concat(
    device const float* x           [[buffer(0)]],
    device const float* message     [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant     uint&  width       [[buffer(3)]],
    constant     uint&  rows        [[buffer(4)]],
    constant     float& eps         [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint row = tgpig.x;
    if (row >= rows) return;

    device const float* x_row = x + (size_t)row * width;
    device const float* m_row = message + (size_t)row * width;
    device float* out_row = output + (size_t)row * (size_t)(2 * width);
    threadgroup float partials[2 * INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_rms_concat_row(x_row, m_row, out_row, width, eps,
        tiisg, sgitg, tid, partials);
}

// Three independent scalar RMS-concat rows share one dispatch. Each
// threadgroup retains the original 256-lane traversal and reduction order.
kernel void inkling2_resident_rms_concat_triplet(
    device const float* x0          [[buffer(0)]],
    device const float* x1          [[buffer(1)]],
    device const float* x2          [[buffer(2)]],
    device const float* messages    [[buffer(3)]],
    device       float* output      [[buffer(4)]],
    constant     uint&  width       [[buffer(5)]],
    constant     float& eps         [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint triplet = tgpig.x;
    if (triplet >= 3) return;

    device const float* x_row = x0;
    if (triplet == 1) x_row = x1;
    else if (triplet == 2) x_row = x2;
    device const float* m_row = messages + (size_t)triplet * (size_t)width;
    device float* out_row = output + (size_t)triplet * (size_t)2 * (size_t)width;
    threadgroup float partials[2 * INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_rms_concat_row(x_row, m_row, out_row, width, eps,
        tiisg, sgitg, tid, partials);
}

inline float inkling2_resident_depth_mix_dot(
    device const float* state, device const float* query, uint width, ushort tid)
{
    float dot = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        dot += state[column] * query[column];
    }
    return simd_sum(dot);
}

inline void inkling2_resident_depth_mix_weights(threadgroup float* scratch, uint key_count, uint width) {
    const uint scores_offset = key_count * INKLING2_RESIDENT_SIMDGROUPS;
    const float score_scale = 1.0f / sqrt((float)width);
    for (uint key = 0; key < key_count; ++key) {
        float score = 0.0f;
        for (uint group = 0; group < INKLING2_RESIDENT_SIMDGROUPS; ++group) {
            score += scratch[(size_t)key * INKLING2_RESIDENT_SIMDGROUPS + group];
        }
        scratch[scores_offset + key] = score * score_scale;
    }

    float max_score = scratch[scores_offset];
    for (uint key = 1; key < key_count; ++key) {
        max_score = max(max_score, scratch[scores_offset + key]);
    }

    float exp_sum = 0.0f;
    for (uint key = 0; key < key_count; ++key) {
        const float weight = exp(scratch[scores_offset + key] - max_score);
        scratch[scores_offset + key] = weight;
        exp_sum += weight;
    }
    for (uint key = 0; key < key_count; ++key) {
        scratch[scores_offset + key] /= exp_sum;
    }
}

kernel void inkling2_resident_depth_mix(
    device const float* bank        [[buffer(0)]],
    device const float* current     [[buffer(1)]],
    device const float* queries     [[buffer(2)]],
    device       float* output      [[buffer(3)]],
    constant     uint&  width       [[buffer(4)]],
    constant     uint&  rows        [[buffer(5)]],
    constant     uint&  bank_count  [[buffer(6)]],
    constant     uint&  query_count [[buffer(7)]],
    constant     uint&  has_current [[buffer(8)]],
    threadgroup  float* scratch     [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint row = tgpig.x;
    const uint query_id = tgpig.y;
    if (row >= rows || query_id >= query_count) return;

    const uint key_count = bank_count + has_current;
    device const float* query = queries + (size_t)query_id * width;
    const size_t row_offset = (size_t)row * width;

    // Store one partial dot per SIMD group and key. The first key_count slots
    // after the partials are reused for normalized attention weights.
    for (uint key = 0; key < key_count; ++key) {
        device const float* state = key < bank_count
            ? bank + (size_t)key * rows * width + row_offset
            : current + row_offset;
        const float group_dot = inkling2_resident_depth_mix_dot(state, query, width, tid);
        if (tiisg == 0) {
            scratch[(size_t)key * INKLING2_RESIDENT_SIMDGROUPS + sgitg] = group_dot;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint scores_offset = key_count * INKLING2_RESIDENT_SIMDGROUPS;
    if (tid == 0) inkling2_resident_depth_mix_weights(scratch, key_count, width);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float* out = output + ((size_t)query_id * rows + row) * width;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        float value = 0.0f;
        for (uint key = 0; key < key_count; ++key) {
            device const float* state = key < bank_count
                ? bank + (size_t)key * rows * width + row_offset
                : current + row_offset;
            value += scratch[scores_offset + key] * state[column];
        }
        out[column] = value;
    }
}

// Preserve lane ownership, key order and FP32 rounding between unit RMS,
// depth mix, and the weighted RMS or controller feature epilogue.
template <bool Concat>
inline void inkling2_resident_depth_mix_normalized_row(
    device const float* input,
    device       float* bank,
    device const float* query,
    device const float* weights_or_message,
    device       float* mixed,
    device       float* output,
    uint width,
    uint bank_count,
    float unit_eps,
    float output_eps,
    bool save_to_bank,
    threadgroup float* scratch,
    threadgroup float* rms_partials,
    ushort tiisg,
    ushort sgitg,
    ushort tid)
{
    const uint key_count = bank_count + 1;
    const uint scores_offset = key_count * INKLING2_RESIDENT_SIMDGROUPS;
    threadgroup float* row = scratch + scores_offset + key_count;

    float square_sum = 0.0f;
    float message_squares = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        const float value = input[column];
        square_sum += value * value;
        if (Concat) {
            const float message = weights_or_message[column];
            message_squares += message * message;
        }
    }
    if (Concat) {
        const float message_group_sum = simd_sum(message_squares);
        if (tiisg == 0) rms_partials[2 * INKLING2_RESIDENT_SIMDGROUPS + sgitg] = message_group_sum;
    }
    const float unit_squares = simd_sum(square_sum);
    if (tiisg == 0) rms_partials[sgitg] = unit_squares;
    for (uint key = 0; key < bank_count; ++key) {
        const float group_dot = inkling2_resident_depth_mix_dot(bank + (size_t)key * width, query, width, tid);
        if (tiisg == 0) scratch[key * INKLING2_RESIDENT_SIMDGROUPS + sgitg] = group_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float unit_scale = inkling2_resident_rms_scale(rms_partials, width, unit_eps);
    float dot = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        const float unit = input[column] * unit_scale;
        row[column] = unit;
        dot += unit * query[column];
    }
    const float current_dot = simd_sum(dot);
    if (tiisg == 0) scratch[bank_count * INKLING2_RESIDENT_SIMDGROUPS + sgitg] = current_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) inkling2_resident_depth_mix_weights(scratch, key_count, width);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    square_sum = 0.0f;
    for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
        const float unit = row[column];
        if (save_to_bank != 0) bank[(size_t)bank_count * width + column] = unit;

        float value = 0.0f;
        for (uint key = 0; key < bank_count; ++key) {
            value += scratch[scores_offset + key] * bank[(size_t)key * width + column];
        }
        value += scratch[scores_offset + bank_count] * unit;
        row[column] = value;
        if (Concat && mixed != nullptr) mixed[column] = value;
        square_sum += value * value;
    }
    const float output_squares = simd_sum(square_sum);
    if (tiisg == 0) rms_partials[INKLING2_RESIDENT_SIMDGROUPS + sgitg] = output_squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float output_scale = inkling2_resident_rms_scale(
        rms_partials + INKLING2_RESIDENT_SIMDGROUPS, width, output_eps);
    if (Concat) {
        const float message_scale = inkling2_resident_rms_scale(
            rms_partials + 2 * INKLING2_RESIDENT_SIMDGROUPS, width, output_eps);
        for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
            output[column] = row[column] * output_scale;
            output[width + column] = weights_or_message[column] * message_scale;
        }
    } else {
        for (uint column = tid; column < width; column += INKLING2_RESIDENT_THREADS) {
            const float normalized = row[column] * output_scale;
            output[column] = normalized * weights_or_message[column];
        }
    }
}

// One token and one depth query. Keep the two intermediate rows in
// threadgroup memory with the same lane ownership and FP32 reductions.
kernel void inkling2_resident_depth_mix_rms(
    device const float* input        [[buffer(0)]],
    device       float* bank         [[buffer(1)]],
    device const float* query        [[buffer(2)]],
    device const float* weights      [[buffer(3)]],
    device       float* output       [[buffer(4)]],
    constant     uint&  width        [[buffer(5)]],
    constant     uint&  bank_count   [[buffer(6)]],
    constant     float& unit_eps     [[buffer(7)]],
    constant     float& output_eps   [[buffer(8)]],
    constant     uint&  save_to_bank [[buffer(9)]],
    threadgroup  float* scratch      [[threadgroup(0)]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    threadgroup float rms_partials[2 * INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_depth_mix_normalized_row<false>(
        input, bank, query, weights, nullptr, output, width, bank_count, unit_eps, output_eps,
        save_to_bank != 0, scratch, rms_partials, tiisg, sgitg, tid);
}

kernel void inkling2_resident_depth_mix_rms_triplet(
    device const float* input        [[buffer(0)]],
    device       float* bank         [[buffer(1)]],
    device const float* queries      [[buffer(2)]],
    device const float* weight0      [[buffer(3)]],
    device const float* weight1      [[buffer(4)]],
    device const float* weight2      [[buffer(5)]],
    device       float* output       [[buffer(6)]],
    constant     uint&  width        [[buffer(7)]],
    constant     uint&  bank_count   [[buffer(8)]],
    constant     float& unit_eps     [[buffer(9)]],
    constant     float& output_eps   [[buffer(10)]],
    constant     uint&  save_to_bank [[buffer(11)]],
    threadgroup  float* scratch      [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint part = tgpig.x;
    const size_t offset = (size_t)part * width;
    device const float* weights = part == 0 ? weight0 : (part == 1 ? weight1 : weight2);
    threadgroup float rms_partials[2 * INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_depth_mix_normalized_row<false>(
        input, bank, queries + offset, weights, nullptr, output + offset, width, bank_count,
        unit_eps, output_eps, save_to_bank != 0 && part == 0, scratch, rms_partials, tiisg, sgitg, tid);
}

// Each feature pair uses an independent reduction group. Sliding attention
// shares the depth query, so only part zero publishes the raw mixture.
kernel void inkling2_resident_depth_mix_concat(
    device const float* input        [[buffer(0)]],
    device       float* bank         [[buffer(1)]],
    device const float* queries      [[buffer(2)]],
    device const float* messages     [[buffer(3)]],
    device       float* mixed        [[buffer(4)]],
    device       float* features     [[buffer(5)]],
    constant     uint&  width        [[buffer(6)]],
    constant     uint&  bank_count   [[buffer(7)]],
    constant     float& eps          [[buffer(8)]],
    constant     uint&  query_count  [[buffer(9)]],
    constant     uint&  save_to_bank [[buffer(10)]],
    threadgroup  float* scratch      [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    const uint part = tgpig.x;
    const size_t feature_offset = (size_t)part * width;
    const size_t query_offset = query_count == 1 ? 0 : feature_offset;
    device float* raw = query_count == 1 && part != 0 ? nullptr : mixed + query_offset;
    threadgroup float rms_partials[3 * INKLING2_RESIDENT_SIMDGROUPS];
    inkling2_resident_depth_mix_normalized_row<true>(
        input, bank, queries + query_offset, messages + feature_offset, raw,
        features + 2 * feature_offset, width, bank_count, eps, eps,
        save_to_bank != 0 && part == 0, scratch, rms_partials, tiisg, sgitg, tid);
}

kernel void inkling2_resident_add(
    device const float* x      [[buffer(0)]],
    device const float* y      [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant     uint&  count  [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) output[tid] = x[tid] + y[tid];
}

kernel void inkling2_resident_copy(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  count  [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) output[tid] = input[tid];
}

kernel void inkling2_resident_controls(
    device const float* input   [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device const float* gain    [[buffer(2)]],
    device const float* forget  [[buffer(3)]],
    device       float* output  [[buffer(4)]],
    constant     uint&  count   [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    const float source = input[tid];
    const float forgotten = tanh(forget[tid]) * source;
    const float gained = tanh(gain[tid]) * message[tid];
    output[tid] = source - forgotten + gained;
}

struct inkling2_resident_q8_0_block {
    half d;
    int8_t qs[32];
};

struct inkling2_resident_iq4_nl_block {
    half d;
    uint8_t qs[16];
};

constant float inkling2_resident_iq4_values[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
      1.0f,   13.0f,   25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f,
};

kernel void inkling2_resident_embed(
    device const uint8_t* weights   [[buffer(0)]],
    device const int32_t* token_ids [[buffer(1)]],
    device       float*   output    [[buffer(2)]],
    constant     uint&    width     [[buffer(3)]],
    constant     uint&    rows      [[buffer(4)]],
    constant     uint&    row_bytes [[buffer(5)]],
    constant     uint&    type      [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = width * rows;
    if (tid >= total) return;

    const uint row = tid / width;
    const uint column = tid - row * width;
    const int signed_token = token_ids[row];
    if (signed_token < 0) {
        output[tid] = 0.0f;
        return;
    }
    device const uint8_t* row_weights = weights + (size_t)(uint)signed_token * row_bytes;

    if (type == 0) {
        output[tid] = ((device const float*)row_weights)[column];
    } else if (type == 8) {
        const uint block = column / 32;
        const uint within = column - block * 32;
        device const inkling2_resident_q8_0_block* q8 =
            (device const inkling2_resident_q8_0_block*)(row_weights + (size_t)block * 34);
        output[tid] = (float)q8->d * (float)q8->qs[within];
    } else if (type == 20) {
        const uint block = column / 32;
        const uint within = column - block * 32;
        device const inkling2_resident_iq4_nl_block* iq4 =
            (device const inkling2_resident_iq4_nl_block*)(row_weights + (size_t)block * 18);
        const uint packed = iq4->qs[within & 15];
        const uint code = within < 16 ? (packed & 0x0f) : (packed >> 4);
        output[tid] = (float)iq4->d * inkling2_resident_iq4_values[code];
    }
}

inline float inkling2_logit_softcap(float value, float cap)
{
    return cap > 0.0f ? cap * tanh(value / cap) : value;
}

kernel void inkling2_resident_softcap(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    constant     uint&  vocab  [[buffer(2)]],
    constant     float& cap    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= vocab) return;
    output[tid] = inkling2_logit_softcap(input[tid], cap);
}

kernel void inkling2_resident_extrapolate(
    device const float* target    [[buffer(0)]],
    device const float* reference [[buffer(1)]],
    device       float* output    [[buffer(2)]],
    constant     uint&  count     [[buffer(3)]],
    constant     float& strength  [[buffer(4)]],
    constant     float& cap       [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    const float value = inkling2_logit_softcap(target[tid], cap);
    output[tid] = value + strength * (value - reference[tid]);
}

struct Inkling2GreedyCandidate {
    uint score_key;
    uint id;
};

constant uint INKLING2_NO_GREEDY_ID = 0xffffffffu;
constant uint INKLING2_GREEDY_TILE = 4 * INKLING2_RESIDENT_THREADS;

inline uint inkling2_greedy_score_key(float score)
{
    // Float SIMD comparisons can flush subnormals. Order finite IEEE bits
    // as unsigned integers, with both signed zeros treated as equal.
    const uint bits = as_type<uint>(score);
    const uint sign = 0x80000000u;
    if ((bits & ~sign) == 0) return sign;
    return bits & sign ? ~bits : bits ^ sign;
}

inline Inkling2GreedyCandidate inkling2_greedy_best(Inkling2GreedyCandidate a, Inkling2GreedyCandidate b)
{
    return b.score_key > a.score_key || (b.score_key == a.score_key && b.id < a.id) ? b : a;
}

inline Inkling2GreedyCandidate inkling2_greedy_simd_best(Inkling2GreedyCandidate best)
{
    const uint key = simd_max(best.score_key);
    const uint id = simd_min(best.score_key == key ? best.id : INKLING2_NO_GREEDY_ID);
    return {key, id};
}

inline Inkling2GreedyCandidate inkling2_greedy_group_best(
    Inkling2GreedyCandidate best, threadgroup Inkling2GreedyCandidate* partials,
    ushort lane, ushort group)
{
    best = inkling2_greedy_simd_best(best);
    if (lane == 0) partials[group] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0) {
        const Inkling2GreedyCandidate candidate =
            lane < INKLING2_RESIDENT_SIMDGROUPS ? partials[lane] : Inkling2GreedyCandidate{0u, INKLING2_NO_GREEDY_ID};
        best = inkling2_greedy_simd_best(candidate);
    }
    return best;
}

kernel void inkling2_resident_greedy_tiles(
    device const float*                   logits    [[buffer(0)]],
    device       Inkling2GreedyCandidate*  tiles     [[buffer(1)]],
    constant     uint&                    vocab     [[buffer(2)]],
    constant     uint&                    stride    [[buffer(3)]],
    constant     uint&                    count     [[buffer(4)]],
    constant     float&                   cap       [[buffer(5)]],
    uint2  tile  [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort group [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    threadgroup Inkling2GreedyCandidate partials[INKLING2_RESIDENT_SIMDGROUPS];
    device const float* row = logits + (size_t)tile.y * stride;
    const uint start = tile.x * INKLING2_GREEDY_TILE;
    const uint end = min(start + INKLING2_GREEDY_TILE, vocab);
    Inkling2GreedyCandidate best = {0u, INKLING2_NO_GREEDY_ID};
    for (uint id = start + tid; id < end; id += INKLING2_RESIDENT_THREADS) {
        const float score = inkling2_logit_softcap(row[id], cap);
        if (isfinite(score)) best = inkling2_greedy_best(best, {inkling2_greedy_score_key(score), id});
    }
    best = inkling2_greedy_group_best(best, partials, lane, group);
    if (tid == 0) tiles[(size_t)tile.y * count + tile.x] = best;
}

kernel void inkling2_resident_greedy_reduce(
    device const Inkling2GreedyCandidate* tiles  [[buffer(0)]],
    device       int*                    ids    [[buffer(1)]],
    constant     uint&                   count  [[buffer(2)]],
    uint   row   [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort group [[simdgroup_index_in_threadgroup]],
    ushort tid   [[thread_index_in_threadgroup]])
{
    threadgroup Inkling2GreedyCandidate partials[INKLING2_RESIDENT_SIMDGROUPS];
    device const Inkling2GreedyCandidate* row_tiles = tiles + (size_t)row * count;
    Inkling2GreedyCandidate best = {0u, INKLING2_NO_GREEDY_ID};
    for (uint i = tid; i < count; i += INKLING2_RESIDENT_THREADS) {
        best = inkling2_greedy_best(best, row_tiles[i]);
    }
    best = inkling2_greedy_group_best(best, partials, lane, group);
    if (tid == 0) ids[row] = best.id == INKLING2_NO_GREEDY_ID ? -1 : (int)best.id;
}
