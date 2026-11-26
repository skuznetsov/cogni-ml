// Batched Inkling2 projections and MoE kernels. This source is appended to
// gemm_q56k.metal so the IQ4_NL expert path shares its block decoder.

constant uint INKLING2_RESIDENT_TILE = 4;

static inline float inkling2_bf16_storage_widen(ushort value) {
    return as_type<float>((uint)value << 16);
}

kernel void inkling2_resident_simd_mv_bf16_f32(
    device const ushort* w       [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint first_row = (tgpig.x * 4 + sgitg);
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    device const float* y_base = x + n * in_dim;
    float sum = 0.0f;
    for (uint col = tiisg; col < in_dim; col += 32) {
        sum += inkling2_bf16_storage_widen(w[first_row * in_dim + col]) * y_base[col];
    }
    const float total = simd_sum(sum);
    if (tiisg == 0) output[n * out_dim + first_row] = total;
}

// Exact BF16 storage companion to the F32 controller triplet. It keeps the
// F32 multiply, lane traversal, accumulation and SIMD reduction unchanged.
kernel void inkling2_resident_bf16_projection_triplet(
    device const ushort* model          [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device       float* output         [[buffer(2)]],
    constant     ulong* weight_offsets [[buffer(3)]],
    constant     uint&  in_dim         [[buffer(4)]],
    constant     uint&  out_dim        [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    const uint row = group.x * 4 + simd;
    if (row >= out_dim) return;
    device const ushort* weights = model + weight_offsets[group.y];
    device const float* x = input + (size_t)group.y * in_dim;
    float sum = 0.0f;
    for (uint col = lane; col < in_dim; col += 32) {
        sum += inkling2_bf16_storage_widen(weights[row * in_dim + col]) * x[col];
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[(size_t)group.y * out_dim + row] = total;
}

// Independent projections occupy the y dimension. Keep the lane's column
// traversal and SIMD reduction identical to simd_mv_f32_f32.
kernel void inkling2_resident_f32_projection_triplet(
    device const uchar* model          [[buffer(0)]],
    device const float* input          [[buffer(1)]],
    device       float* output         [[buffer(2)]],
    constant     ulong* weight_offsets [[buffer(3)]],
    constant     uint&  in_dim         [[buffer(4)]],
    constant     uint&  out_dim        [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    const uint row = group.x * 4 + simd;
    if (row >= out_dim) return;
    device const float* weights = (device const float*)(model + weight_offsets[group.y]);
    device const float* x = input + (size_t)group.y * in_dim;
    float sum = 0.0f;
    for (uint col = lane; col < in_dim; col += 32) {
        sum += weights[row * in_dim + col] * x[col];
    }
    const float total = simd_sum(sum);
    if (lane == 0) output[(size_t)group.y * out_dim + row] = total;
}

// Fuse two scalar F32 projections and their controller epilogue. Each lane
// visits the same columns, in the same order, as simd_mv_f32_f32; the shared
// rank value feeds both independent accumulators before their usual reductions.
kernel void inkling2_resident_f32_control_pair(
    device const uchar* model          [[buffer(0)]],
    device const float* rank           [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const float* message        [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant ulong* weight_offsets    [[buffer(5)]],
    constant uint&  in_dim             [[buffer(6)]],
    constant uint&  out_dim            [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    const uint row = group.x * 4 + (uint)simd;
    if (row >= out_dim) return;

    device const float* gain_weights = (device const float*)(model + weight_offsets[0]);
    device const float* forget_weights = (device const float*)(model + weight_offsets[1]);
    const size_t weight_base = (size_t)row * in_dim;
    float gain_sum = 0.0f;
    float forget_sum = 0.0f;
    for (uint col = (uint)lane; col < in_dim; col += 32) {
        const float rank_value = rank[col];
        gain_sum += gain_weights[weight_base + col] * rank_value;
        forget_sum += forget_weights[weight_base + col] * rank_value;
    }

    const float gain_value = simd_sum(gain_sum);
    const float forget_value = simd_sum(forget_sum);
    if (lane == 0) {
        const float source = input[row];
        const float forgotten = tanh(forget_value) * source;
        const float gained = tanh(gain_value) * message[row];
        output[row] = source - forgotten + gained;
    }
}

// Scalar decode storage-only companion; all arithmetic remains F32.
kernel void inkling2_resident_bf16_control_pair(
    device const ushort* model          [[buffer(0)]],
    device const float* rank           [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const float* message        [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant ulong* weight_offsets    [[buffer(5)]],
    constant uint&  in_dim             [[buffer(6)]],
    constant uint&  out_dim            [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    const uint row = group.x * 4 + (uint)simd;
    if (row >= out_dim) return;

    device const ushort* gain_weights = model + weight_offsets[0];
    device const ushort* forget_weights = model + weight_offsets[1];
    const size_t weight_base = (size_t)row * in_dim;
    float gain_sum = 0.0f;
    float forget_sum = 0.0f;
    for (uint col = (uint)lane; col < in_dim; col += 32) {
        const float rank_value = rank[col];
        gain_sum += inkling2_bf16_storage_widen(gain_weights[weight_base + col]) * rank_value;
        forget_sum += inkling2_bf16_storage_widen(forget_weights[weight_base + col]) * rank_value;
    }

    const float gain_value = simd_sum(gain_sum);
    const float forget_value = simd_sum(forget_sum);
    if (lane == 0) {
        const float source = input[row];
        const float forgotten = tanh(forget_value) * source;
        const float gained = tanh(gain_value) * message[row];
        output[row] = source - forgotten + gained;
    }
}

// Batched controller projections use the same four-row accumulator and SIMD
// reduction order as inkling2_resident_gemm_f32 while sharing each rank read
// between the gain and forget matrices.
kernel void inkling2_resident_f32_control_pair_batch(
    device const uchar* model          [[buffer(0)]],
    device const float* rank           [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const float* message        [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant ulong* weight_offsets    [[buffer(5)]],
    constant uint&  in_dim             [[buffer(6)]],
    constant uint&  out_dim            [[buffer(7)]],
    constant uint&  rows               [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    const uint output_row = group.x * 4 + (uint)simd;
    if (output_row >= out_dim) return;

    const uint first_batch = group.y * 4;
    device const float* gain_weights = (device const float*)(model + weight_offsets[0]);
    device const float* forget_weights = (device const float*)(model + weight_offsets[1]);
    const size_t weight_base = (size_t)output_row * in_dim;
    float4 gain_sums = float4(0.0f);
    float4 forget_sums = float4(0.0f);
    for (uint col = (uint)lane; col < in_dim; col += 32) {
        float4 rank_values = float4(0.0f);
        for (uint batch_row = 0; batch_row < 4; ++batch_row) {
            const uint row = first_batch + batch_row;
            if (row < rows) {
                rank_values[batch_row] = rank[(size_t)row * in_dim + col];
            }
        }
        gain_sums += rank_values * gain_weights[weight_base + col];
        forget_sums += rank_values * forget_weights[weight_base + col];
    }

    const float4 gain_values = float4(simd_sum(gain_sums.x), simd_sum(gain_sums.y),
                                     simd_sum(gain_sums.z), simd_sum(gain_sums.w));
    const float4 forget_values = float4(simd_sum(forget_sums.x), simd_sum(forget_sums.y),
                                        simd_sum(forget_sums.z), simd_sum(forget_sums.w));
    if (lane == 0) {
        for (uint batch_row = 0; batch_row < 4; ++batch_row) {
            const uint row = first_batch + batch_row;
            if (row < rows) {
                const size_t output_index = (size_t)row * out_dim + output_row;
                const float source = input[output_index];
                const float forgotten = tanh(forget_values[batch_row]) * source;
                const float gained = tanh(gain_values[batch_row]) * message[output_index];
                output[output_index] = source - forgotten + gained;
            }
        }
    }
}

static inline float4 inkling2_control_pair_sum(float4 sums) {
    return float4(simd_sum(sums.x), simd_sum(sums.y), simd_sum(sums.z), simd_sum(sums.w));
}

// Use the original float4 accumulators and SIMD reduction for every row.
// The four SIMD groups share the rank tile. They leave reduced values in
// threadgroup memory so the epilogue can use contiguous output channels.
kernel void inkling2_resident_f32_control_pair_coalesced8(
    device const uchar* model          [[buffer(0)]],
    device const float* rank           [[buffer(1)]],
    device const float* input          [[buffer(2)]],
    device const float* message        [[buffer(3)]],
    device       float* output         [[buffer(4)]],
    constant ulong* weight_offsets    [[buffer(5)]],
    constant uint&  in_dim             [[buffer(6)]],
    constant uint&  out_dim            [[buffer(7)]],
    constant uint&  rows               [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    constexpr uint Rank = 64;
    constexpr uint BatchRows = 8;
    constexpr uint Channels = 32;
    constexpr uint ChannelsPerSimd = Channels / 4;
    constexpr uint RowVectors = BatchRows / 4;
    if (in_dim != Rank) return;
    const uint first_batch = group.y * BatchRows;
    const uint batch_count = min(rows - first_batch, BatchRows);
    const uint first_channel = group.x * Channels;
    const uint channel_count = min(out_dim - first_channel, Channels);
    device const float* gain_weights = (device const float*)(model + weight_offsets[0]);
    device const float* forget_weights = (device const float*)(model + weight_offsets[1]);
    device const float* rank_rows = rank + (size_t)first_batch * Rank;
    threadgroup float rank_tile[BatchRows * Rank];
    threadgroup float gain_tile[BatchRows * Channels];
    threadgroup float forget_tile[BatchRows * Channels];

    for (uint i = tid; i < BatchRows * Rank; i += 128) {
        rank_tile[i] = i < batch_count * Rank ? rank_rows[i] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 rank_values[2][RowVectors];
    for (uint step = 0; step < 2; ++step) {
        const uint col = (uint)lane + step * 32;
        for (uint b = 0; b < BatchRows; ++b) {
            const uint offset = b * Rank + col;
            rank_values[step][b / 4][b % 4] = rank_tile[offset];
        }
    }

    for (uint i = 0; i < ChannelsPerSimd; ++i) {
        const uint channel = (uint)simd * ChannelsPerSimd + i;
        if (channel >= channel_count) break;
        const uint output_row = first_channel + channel;
        const size_t weight_base = (size_t)output_row * Rank;
        float4 gain_sums[RowVectors];
        float4 forget_sums[RowVectors];
        for (uint b = 0; b < RowVectors; ++b) {
            gain_sums[b] = float4(0.0f);
            forget_sums[b] = float4(0.0f);
        }
        for (uint step = 0; step < 2; ++step) {
            const size_t offset = weight_base + (uint)lane + step * 32;
            const float gain_weight = gain_weights[offset];
            const float forget_weight = forget_weights[offset];
            for (uint b = 0; b < RowVectors; ++b) {
                gain_sums[b] += rank_values[step][b] * gain_weight;
                forget_sums[b] += rank_values[step][b] * forget_weight;
            }
        }
        for (uint b = 0; b < RowVectors; ++b) {
            const float4 gain_values = inkling2_control_pair_sum(gain_sums[b]);
            const float4 forget_values = inkling2_control_pair_sum(forget_sums[b]);
            if (lane == 0) {
                for (uint j = 0; j < 4; ++j) {
                    const uint batch_row = b * 4 + j;
                    if (batch_row >= batch_count) break;
                    const uint offset = batch_row * Channels + channel;
                    gain_tile[offset] = gain_values[j];
                    forget_tile[offset] = forget_values[j];
                }
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < batch_count * Channels; i += 128) {
        const uint channel = i % Channels;
        if (channel < channel_count) {
            const size_t index = (size_t)(first_batch + i / Channels) * out_dim + first_channel + channel;
            const float source = input[index];
            const float forgotten = tanh(forget_tile[i]) * source;
            const float gained = tanh(gain_tile[i]) * message[index];
            output[index] = source - forgotten + gained;
        }
    }
}

// A simdgroup computes one output row and four adjacent input rows. Each
// thread loads one weight value and reuses it across the four token rows.
kernel void inkling2_resident_gemm_f32(
    device const float*   w         [[buffer(0)]],
    device const float*   x         [[buffer(1)]],
    device       float*   output    [[buffer(2)]],
    constant     uint&    in_dim    [[buffer(3)]],
    constant     uint&    out_dim   [[buffer(4)]],
    constant     uint&    batch     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgpig.x * INKLING2_RESIDENT_TILE + (uint)sgitg;
    const uint first_batch = tgpig.y * INKLING2_RESIDENT_TILE;
    if (row >= out_dim) return;

    const device float* w_row = w + (size_t)row * (size_t)in_dim;
    float4 sums = float4(0.0f);
    for (uint col = tiisg; col < in_dim; col += 32) {
        const float weight = w_row[col];
        float4 values = float4(0.0f);
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                values[b] = x[(size_t)batch_row * (size_t)in_dim + col];
            }
        }
        sums += values * weight;
    }

    const float4 totals = float4(simd_sum(sums.x), simd_sum(sums.y),
                                 simd_sum(sums.z), simd_sum(sums.w));
    if (tiisg == 0) {
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                output[(size_t)batch_row * (size_t)out_dim + row] = totals[b];
            }
        }
    }
}

// Batched storage-only companion to the resident F32 projection. The
// accumulator shape, input traversal and reduction order are identical.
kernel void inkling2_resident_gemm_bf16(
    device const ushort* w         [[buffer(0)]],
    device const float*   x         [[buffer(1)]],
    device       float*   output    [[buffer(2)]],
    constant     uint&    in_dim    [[buffer(3)]],
    constant     uint&    out_dim   [[buffer(4)]],
    constant     uint&    batch     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint row = tgpig.x * INKLING2_RESIDENT_TILE + (uint)sgitg;
    const uint first_batch = tgpig.y * INKLING2_RESIDENT_TILE;
    if (row >= out_dim) return;

    const device ushort* w_row = w + (size_t)row * (size_t)in_dim;
    float4 sums = float4(0.0f);
    for (uint col = tiisg; col < in_dim; col += 32) {
        const float weight = inkling2_bf16_storage_widen(w_row[col]);
        float4 values = float4(0.0f);
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                values[b] = x[(size_t)batch_row * (size_t)in_dim + col];
            }
        }
        sums += values * weight;
    }

    const float4 totals = float4(simd_sum(sums.x), simd_sum(sums.y),
                                 simd_sum(sums.z), simd_sum(sums.w));
    if (tiisg == 0) {
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                output[(size_t)batch_row * (size_t)out_dim + row] = totals[b];
            }
        }
    }
}

kernel void inkling2_resident_gemm_q8_0(
    device const uint8_t* w_raw     [[buffer(0)]],
    device const float*   x         [[buffer(1)]],
    device       float*   output    [[buffer(2)]],
    constant     uint&    in_dim    [[buffer(3)]],
    constant     uint&    out_dim   [[buffer(4)]],
    constant     uint&    batch     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint row = tgpig.x * INKLING2_RESIDENT_TILE + (uint)sgitg;
    const uint first_batch = tgpig.y * INKLING2_RESIDENT_TILE;
    if (row >= out_dim) return;

    const uint row_bytes = nb * 34;
    device const block_q8_0_56* w_row =
        (device const block_q8_0_56*)(w_raw + (size_t)row * (size_t)row_bytes);
    float4 sums = float4(0.0f);

    for (uint ib = 0; ib < nb; ++ib) {
        device const block_q8_0_56* block = w_row + ib;
        const float scale = (float)block->d;
        const float quant = (float)block->qs[tiisg];
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                const float value = x[(size_t)batch_row * (size_t)in_dim +
                                      (size_t)ib * Q8_0_QK + tiisg];
                sums[b] += scale * value * quant;
            }
        }
    }

    const float4 totals = float4(simd_sum(sums.x), simd_sum(sums.y),
                                 simd_sum(sums.z), simd_sum(sums.w));
    if (tiisg == 0) {
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            const uint batch_row = first_batch + b;
            if (batch_row < batch) {
                output[(size_t)batch_row * (size_t)out_dim + row] = totals[b];
            }
        }
    }
}

struct inkling2_q8_projection {
    ulong weight_offset;
    ulong input_offset;
    ulong output_offset;
    ulong end_group;
};

inline inkling2_q8_projection inkling2_q8_group_descriptor(
    constant inkling2_q8_projection* projections, uint group, thread uint& first_group)
{
    first_group = 0;
    uint projection = 0;
    while (group >= projections[projection].end_group) {
        first_group = uint(projections[projection].end_group);
        ++projection;
    }
    return projections[projection];
}

// Each threadgroup owns four output channels of one independent projection.
kernel void inkling2_resident_q8_projection_group(
    device const uchar* model                          [[buffer(0)]],
    device const float* input                          [[buffer(1)]],
    device       float* output                         [[buffer(2)]],
    constant inkling2_q8_projection* projections       [[buffer(3)]],
    constant uint& in_dim                              [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    uint first_group;
    const inkling2_q8_projection descriptor = inkling2_q8_group_descriptor(projections, group.x, first_group);
    const uint row = (group.x - first_group) * INKLING2_RESIDENT_TILE + simd;
    const uint nb = in_dim / Q8_0_QK;
    device const block_q8_0_56* weights = (device const block_q8_0_56*)(model + descriptor.weight_offset)
        + (size_t)row * nb;
    device const float* x = input + descriptor.input_offset;
    device float* y = output + descriptor.output_offset;
    float sum = 0.0f;
    for (uint ib = 0; ib < nb; ++ib) {
        const float value = x[ib * Q8_0_QK + lane];
        sum += (float)weights[ib].d * value * (float)weights[ib].qs[lane];
    }
    const float total = simd_sum(sum);
    if (lane == 0) y[row] = total;
}

// Two simdgroups each compute a pair of output channels for up to four tokens.
// The pair shares activation loads and retains a separate reduction per channel.
kernel void inkling2_resident_q8_projection_group_batch(
    device const uchar* model                          [[buffer(0)]],
    device const float* input                          [[buffer(1)]],
    device       float* output                         [[buffer(2)]],
    constant inkling2_q8_projection* projections       [[buffer(3)]],
    constant uint& in_dim                              [[buffer(4)]],
    constant uint& rows                                [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]])
{
    uint first_group;
    const inkling2_q8_projection descriptor = inkling2_q8_group_descriptor(projections, group.x, first_group);
    const uint out_dim = (uint(descriptor.end_group) - first_group) * INKLING2_RESIDENT_TILE;
    const uint row = (group.x - first_group) * INKLING2_RESIDENT_TILE + 2 * (uint)simd;
    const uint nb = in_dim / Q8_0_QK;
    device const block_q8_0_56* weights0 = (device const block_q8_0_56*)(model + descriptor.weight_offset)
        + (size_t)row * nb;
    device const block_q8_0_56* weights1 = weights0 + nb;
    device const float* x = input + descriptor.input_offset;
    device float* y = output + descriptor.output_offset;
    float4 sums0 = float4(0.0f);
    float4 sums1 = float4(0.0f);
    for (uint ib = 0; ib < nb; ++ib) {
        const float scale0 = (float)weights0[ib].d;
        const float scale1 = (float)weights1[ib].d;
        const float quant0 = (float)weights0[ib].qs[lane];
        const float quant1 = (float)weights1[ib].qs[lane];
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            if (b < rows) {
                const float value = x[(size_t)b * in_dim + ib * Q8_0_QK + lane];
                sums0[b] += scale0 * value * quant0;
                sums1[b] += scale1 * value * quant1;
            }
        }
    }
    const float4 totals0 = float4(simd_sum(sums0.x), simd_sum(sums0.y),
                                  simd_sum(sums0.z), simd_sum(sums0.w));
    const float4 totals1 = float4(simd_sum(sums1.x), simd_sum(sums1.y),
                                  simd_sum(sums1.z), simd_sum(sums1.w));
    if (lane == 0) {
        for (uint b = 0; b < INKLING2_RESIDENT_TILE; ++b) {
            if (b < rows) {
                const size_t index = (size_t)b * out_dim + row;
                y[index] = totals0[b];
                y[index + 1] = totals1[b];
            }
        }
    }
}

inline float inkling2_resident_sigmoid(float value) {
    return 1.0f / (1.0f + exp(-value));
}

// One 32-thread group routes one token row. The host chunks the y dimension
// and supplies row_base so very long prefills do not rely on a huge grid.
kernel void inkling2_resident_moe_route(
    device const float* logits       [[buffer(0)]],
    device const float* bias         [[buffer(1)]],
    device       uint*  selected_ids [[buffer(2)]],
    device       float* routing_wts  [[buffer(3)]],
    constant     uint&  n_experts    [[buffer(4)]],
    constant     uint&  k            [[buffer(5)]],
    constant     float& scale        [[buffer(6)]],
    constant     uint&  has_bias     [[buffer(7)]],
    constant     uint&  rows         [[buffer(8)]],
    constant     uint&  row_base     [[buffer(9)]],
    threadgroup  float* scratch      [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint row = row_base + tgpig.y;
    if (row >= rows) return;

    const uint lane = (uint)tiisg;
    threadgroup float* probabilities = scratch;
    threadgroup float* selection_scores = scratch + n_experts;
    device const float* row_logits = logits + (size_t)row * n_experts;
    device uint* row_ids = selected_ids + (size_t)row * k;
    device float* row_weights = routing_wts + (size_t)row * k;

    for (uint expert = lane; expert < n_experts; expert += 32) {
        const float logit = row_logits[expert];
        const float probability = isfinite(logit)
            ? inkling2_resident_sigmoid(logit)
            : (logit > 0.0f ? 1.0f : 0.0f);
        probabilities[expert] = probability;

        float score = probability;
        if (has_bias != 0) score += bias[expert];
        selection_scores[expert] = isnan(score) ? -INFINITY : score;
    }

    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint no_expert = 0xffffffffu;
    float selected_probability_sum = 0.0f;
    for (uint rank = 0; rank < k; ++rank) {
        float local_score = -INFINITY;
        uint local_expert = no_expert;
        for (uint expert = lane; expert < n_experts; expert += 32) {
            const float score = selection_scores[expert];
            const bool consumed = as_type<uint>(score) == 0x7fc00000u;
            if (!consumed && (local_expert == no_expert || score > local_score)) {
                local_score = score;
                local_expert = expert;
            }
        }

        const float best_score = simd_max(local_score);
        const uint best_expert = simd_min(local_score == best_score ? local_expert : no_expert);

        // Each lane reads and consumes only its own strided scores.
        if ((best_expert & 31u) == lane) {
            selection_scores[best_expert] = as_type<float>(0x7fc00000u);
        }
        if (lane == 0) {
            row_ids[rank] = best_expert;
            selected_probability_sum += probabilities[best_expert];
        }
    }

    // Keep accumulation and normalization in the original rank order.
    if (lane == 0) {
        const float normalization = selected_probability_sum + 1.0e-20f;
        for (uint rank = 0; rank < k; ++rank) {
            const uint expert = row_ids[rank];
            row_weights[rank] = probabilities[expert] / normalization * scale;
        }
    }
}

// Each y group computes four output rows for one flattened (token, expert)
// route. routed_base keeps the grid bounded without changing row ordering.
kernel void inkling2_resident_moe_matvec_iq4_nl(
    device const uint8_t* expert_weights [[buffer(0)]],
    device const float*   x              [[buffer(1)]],
    device const uint*    selected_ids   [[buffer(2)]],
    device       float*   output         [[buffer(3)]],
    constant     uint&    in_dim         [[buffer(4)]],
    constant     uint&    out_dim        [[buffer(5)]],
    constant     uint&    expert_bytes   [[buffer(6)]],
    constant     uint&    input_stride   [[buffer(7)]],
    constant     uint&    top_k          [[buffer(8)]],
    constant     uint&    input_is_routed [[buffer(9)]],
    constant     uint&    routed_base    [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint route = routed_base + tgpig.y;
    const uint first_row = (tgpig.x * 2 + (uint)sgitg) * 2;
    if (first_row >= out_dim) return;

    const uint expert_id = selected_ids[route];
    const size_t expert_offset = (size_t)expert_id * (size_t)expert_bytes;
    device const uint8_t* w_raw = expert_weights + expert_offset;
    const uint nb = in_dim / IQ4_NL_QK;
    const uint row_bytes = nb * 18;
    const uint input_row = input_is_routed != 0 ? route : route / top_k;
    device const float* y_base = x + (size_t)input_row * input_stride;
    float sumf[2] = {0.f, 0.f};

    const ushort ix = tiisg / 2;
    const ushort it = tiisg & 1;
    device const float* yb = y_base + ix * IQ4_NL_QK + it * 8;

    for (uint ib = ix; ib < nb; ib += 16) {
        device const float4* y4 = (device const float4*)yb;
        const float4 yl0 = y4[0];
        const float4 yl1 = y4[4];
        const float4 yl2 = y4[1];
        const float4 yl3 = y4[5];

        for (short row = 0; row < 2; ++row) {
            const uint row_id = first_row + row;
            if (row_id >= out_dim) continue;
            device const block_iq4_nl_56* block =
                (device const block_iq4_nl_56*)(w_raw + row_id * row_bytes) + ib;
            sumf[row] += (float)block->d * iq4_nl_dot_half(block, it, yl0, yl1, yl2, yl3);
        }
        yb += 16 * IQ4_NL_QK;
    }

    for (short row = 0; row < 2 && first_row + row < out_dim; ++row) {
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) output[(size_t)route * out_dim + first_row + row] = total;
    }
}

kernel void inkling2_resident_moe_matvec_q8_0(
    device const uint8_t* expert_weights [[buffer(0)]],
    device const float*   x              [[buffer(1)]],
    device const uint*    selected_ids   [[buffer(2)]],
    device       float*   output         [[buffer(3)]],
    constant     uint&    in_dim         [[buffer(4)]],
    constant     uint&    out_dim        [[buffer(5)]],
    constant     uint&    expert_bytes   [[buffer(6)]],
    constant     uint&    input_stride   [[buffer(7)]],
    constant     uint&    top_k          [[buffer(8)]],
    constant     uint&    input_is_routed [[buffer(9)]],
    constant     uint&    routed_base    [[buffer(10)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint route = routed_base + tgpig.y;
    const uint row_id = tgpig.x * MV8_NSG + (uint)sgitg;
    if (row_id >= out_dim) return;

    const uint expert_id = selected_ids[route];
    const size_t expert_offset = (size_t)expert_id * (size_t)expert_bytes;
    device const uint8_t* w_raw = expert_weights + expert_offset;
    const uint nb = in_dim / Q8_0_QK;
    const uint row_bytes = nb * 34;
    const uint input_row = input_is_routed != 0 ? route : route / top_k;
    device const float* y_base = x + (size_t)input_row * input_stride;
    float sum = 0.0f;

    for (uint ib = 0; ib < nb; ++ib) {
        const float y = y_base[ib * Q8_0_QK + tiisg];
        device const block_q8_0_56* block =
            (device const block_q8_0_56*)(w_raw + row_id * row_bytes) + ib;
        sum += (float)block->d * y * (float)block->qs[tiisg];
    }

    const float total = simd_sum(sum);
    if (tiisg == 0) output[(size_t)route * out_dim + row_id] = total;
}

inline float inkling2_resident_situ(float gate, float up, float beta, float linear_beta) {
    const float bounded_gate = beta * tanh(gate / beta);
    const float bounded_up = linear_beta * tanh(up / linear_beta);
    return bounded_gate * inkling2_resident_sigmoid(gate) * bounded_up;
}

kernel void inkling2_resident_moe_situ(
    device const float* gate_up [[buffer(0)]],
    device       float* gated   [[buffer(1)]],
    constant     uint&  width   [[buffer(2)]],
    constant     uint&  total   [[buffer(3)]],
    constant     uint&  base    [[buffer(4)]],
    constant     float& beta    [[buffer(5)]],
    constant     float& linear_beta [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint element = base + tid;
    if (element >= total) return;

    const uint route = element / width;
    const uint column = element - route * width;
    const size_t row_base = (size_t)route * (size_t)(2 * width);
    const float gate = gate_up[row_base + column];
    const float up = gate_up[row_base + width + column];
    gated[element] = inkling2_resident_situ(gate, up, beta, linear_beta);
}

kernel void inkling2_resident_moe_combine(
    device const float* values         [[buffer(0)]],
    device const float* mixing_weights [[buffer(1)]],
    device       float* result         [[buffer(2)]],
    constant     uint&  dim            [[buffer(3)]],
    constant     uint&  k              [[buffer(4)]],
    constant     uint&  total          [[buffer(5)]],
    constant     uint&  base           [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    const uint element = base + tid;
    if (element >= total) return;

    const uint row = element / dim;
    const uint column = element - row * dim;
    float sum = 0.0f;
    for (uint slot = 0; slot < k; ++slot) {
        const uint route = row * k + slot;
        sum += mixing_weights[route] * values[(size_t)route * dim + column];
    }
    result[element] = sum;
}
