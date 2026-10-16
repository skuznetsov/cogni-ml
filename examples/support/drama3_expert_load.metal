#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

inline uint dr3_expert_load_bytes(device const uchar* bytes, ulong offset) {
    return uint(bytes[offset]) | uint(bytes[offset + 1]) << 8 |
           uint(bytes[offset + 2]) << 16 | uint(bytes[offset + 3]) << 24;
}

// Match the existing byte-wise little-endian read for odd offsets. Metal
// buffer bases are aligned; the explicit offset check protects views whose
// packed tensor begins at a non-word boundary.
inline uint dr3_expert_load_word(device const uchar* bytes, ulong offset) {
    if ((offset & 3ul) == 0ul) {
        device const uint* words = reinterpret_cast<device const uint*>(bytes);
        return words[offset >> 2];
    }
    return dr3_expert_load_bytes(bytes, offset);
}

inline float dr3_expert_float(device const uchar* bytes, ulong offset, ulong index, ulong type) {
    if (type == 0) return as_type<float>(dr3_expert_load_bytes(bytes, offset + index * 4));
    offset += index * 2;
    ushort value = ushort(bytes[offset]) | ushort(bytes[offset + 1]) << 8;
    return type == 1 ? as_type<float>(uint(value) << 16) : float(as_type<half>(value));
}

// Receipts stay device-side until their pair's terminal drain. Slots are:
// bit mismatches, non-finite pairs, and aligned-word oracle mismatches.
kernel void dr3_expert_load_receipt_clear(
    device atomic_uint* receipt [[buffer(0)]], uint index [[thread_position_in_grid]]) {
    if (index < 3) atomic_store_explicit(&receipt[index], 0u, memory_order_relaxed);
}

kernel void dr3_expert_load_compare(
    device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
    device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= params[0]) return;
    float ref = reference[index];
    float got = actual[index];
    if (as_type<uint>(ref) != as_type<uint>(got)) {
        atomic_fetch_add_explicit(&receipt[0], 1u, memory_order_relaxed);
    }
    if (!isfinite(ref) || !isfinite(got)) {
        atomic_fetch_add_explicit(&receipt[1], 1u, memory_order_relaxed);
    }
}

kernel void dr3_expert_load_compare_rows(
    device const float* reference [[buffer(0)]], device const float* actual [[buffer(1)]],
    device atomic_uint* receipt [[buffer(2)]], constant ulong* params [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    ulong active_rows = params[0];
    ulong reference_rows = params[1];
    ulong row_count = active_rows * params[2];
    if (ulong(index) >= row_count) return;
    ulong rank = ulong(index) / active_rows;
    ulong row = ulong(index) % active_rows;
    float ref = reference[rank * reference_rows + row];
    float got = actual[index];
    if (as_type<uint>(ref) != as_type<uint>(got)) {
        atomic_fetch_add_explicit(&receipt[0], 1u, memory_order_relaxed);
    }
    if (!isfinite(ref) || !isfinite(got)) {
        atomic_fetch_add_explicit(&receipt[1], 1u, memory_order_relaxed);
    }
}

// Diagnostic copy of resident.metal::dr3_experts. The sole changed operation
// is loading each packed weight word; group partitioning, dequantization,
// multiply/add order, and lane-0 reduction order remain the same.
kernel void dr3_experts_word_load(
    device const uchar* weights [[buffer(0)]], device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]], device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]], device const float* ids [[buffer(5)]],
    constant ulong* params [[buffer(6)]], threadgroup float* parts [[threadgroup(0)]],
    uint2 position [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint row_output = position.x;
    uint rank = position.y;
    uint rows = uint(params[0]);
    uint cols = uint(params[1]);
    float selected = ids[rank];
    if (!isfinite(selected) || selected < 0 || selected >= float(params[9]) || floor(selected) != selected) {
        if (lane == 0) output[rank * rows + row_output] = NAN;
        return;
    }

    uint bits = uint(params[2]);
    uint group = uint(params[3]);
    uint expert_row = uint(selected) * rows + row_output;
    uint groups = cols / group;
    uint values_per_word = 32 / bits;
    uint mask = (1u << bits) - 1;
    uint input_offset = params[10] ? rank * cols : 0;

    for (uint group_index = lane; group_index < groups; group_index += 64) {
        uint tensor_index = expert_row * groups + group_index;
        float scale = dr3_expert_float(scales, params[5], tensor_index, params[7]);
        float bias = dr3_expert_float(biases, params[6], tensor_index, params[8]);
        float partial = 0;
        for (uint word_index = 0; word_index < group / values_per_word; word_index++) {
            uint column = group_index * group + word_index * values_per_word;
            uint packed = dr3_expert_load_word(
                weights, params[4] + (ulong(expert_row) * (cols / values_per_word) + column / values_per_word) * 4);
            for (uint value_index = 0; value_index < values_per_word; value_index++) {
                float weight = scale * float((packed >> (value_index * bits)) & mask) + bias;
                partial += weight * input[input_offset + column + value_index];
            }
        }
        parts[group_index] = partial;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        float sum = 0;
        for (uint group_index = 0; group_index < groups; group_index++) sum += parts[group_index];
        output[rank * rows + row_output] = sum;
    }
}

// Exact native arithmetic with several output rows packed into one 64-lane
// group. Supported `groups` divide 64; each lane computes one group's subtotal
// and the leader lane for each row sums those subtotals in ascending order.
// The caller passes physical checkpoint rows separately from active output
// rows so the odd-tail guard never changes expert storage addressing.
kernel void dr3_experts_packed_rows(
    device const uchar* weights [[buffer(0)]], device const uchar* scales [[buffer(1)]],
    device const uchar* biases [[buffer(2)]], device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]], device const float* ids [[buffer(5)]],
    constant ulong* params [[buffer(6)]], threadgroup float* parts [[threadgroup(0)]],
    uint2 position [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) {
    uint storage_rows = uint(params[0]);
    uint cols = uint(params[1]);
    uint bits = uint(params[2]);
    uint group = uint(params[3]);
    uint active_rows = uint(params[11]);
    uint groups = cols / group;
    uint rows_per_threadgroup = 64 / groups;
    uint row_slot = lane / groups;
    uint group_index = lane % groups;
    uint row = position.x * rows_per_threadgroup + row_slot;
    uint rank = position.y;
    float selected = ids[rank];
    bool valid_expert = isfinite(selected) && selected >= 0 && selected < float(params[9]) && floor(selected) == selected;
    uint values_per_word = 32 / bits;
    uint mask = (1u << bits) - 1;
    uint input_offset = params[10] ? rank * cols : 0;

    float partial = 0.0f;
    if (row < active_rows && valid_expert) {
        uint expert_row = uint(selected) * storage_rows + row;
        uint tensor_index = expert_row * groups + group_index;
        float scale = dr3_expert_float(scales, params[5], tensor_index, params[7]);
        float bias = dr3_expert_float(biases, params[6], tensor_index, params[8]);
        for (uint word_index = 0; word_index < group / values_per_word; word_index++) {
            uint column = group_index * group + word_index * values_per_word;
            uint packed = dr3_expert_load_bytes(
                weights, params[4] + (ulong(expert_row) * (cols / values_per_word) + column / values_per_word) * 4);
            for (uint value_index = 0; value_index < values_per_word; value_index++) {
                float weight = scale * float((packed >> (value_index * bits)) & mask) + bias;
                partial += weight * input[input_offset + column + value_index];
            }
        }
    }
    parts[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row < active_rows && group_index == 0) {
        if (!valid_expert) {
            output[rank * active_rows + row] = NAN;
        } else {
            float sum = 0.0f;
            uint row_base = row_slot * groups;
            for (uint group_slot = 0; group_slot < groups; group_slot++) {
                sum += parts[row_base + group_slot];
            }
            output[rank * active_rows + row] = sum;
        }
    }
}

// Exercised during an excluded warmup pair. Every offset is in-bounds; odd
// offsets prove the candidate takes the byte-assembly fallback.
kernel void dr3_expert_word_load_oracle(
    device const uchar* input [[buffer(0)]], device atomic_uint* receipt [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= 8) return;
    ulong offset = 0;
    switch (index) {
        case 0: offset = 0; break;
        case 1: offset = 1; break;
        case 2: offset = 3; break;
        case 3: offset = 4; break;
        case 4: offset = 7; break;
        case 5: offset = 8; break;
        case 6: offset = 11; break;
        default: offset = 12; break;
    }
    uint expected = dr3_expert_load_bytes(input, offset);
    uint actual = dr3_expert_load_word(input, offset);
    if (expected != actual) {
        atomic_fetch_add_explicit(&receipt[2], 1u, memory_order_relaxed);
    }
}
