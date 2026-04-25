// Q5_K and Q6_K Metal matmul kernels — F32 input, F32 output, no bias.
// Direct ports of llama.cpp's kernel_mul_mv_q{5,6}_K_f32_impl.
//
// Phase 2.5: GEMV only (batch=1) — sufficient for single-token decode
// and for lm_head (Q6_K 4096→248320 per token). Prefill currently uses
// CPU for Q5_K/Q6_K weights. GEMM ports are a Phase 4 optimization.
//
// Q5_K block: [d:2B][dmin:2B][scales:12B][qh:32B][qs:128B] = 176 bytes.
// Q6_K block: [ql:128B][qh:64B][scales:16B int8][d:2B]      = 210 bytes.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

constant uint Q56K_QK_K = 256;
constant uint Q8_0_QK = 32;

// ============================================================================
// Q8_0
// ============================================================================
struct block_q8_0_56 {
    half   d;
    int8_t qs[32];
};

constant short MV8_NSG = 4;
constant short MV8_NR0 = 1;

kernel void simd_mv_q8_0_f32(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint first_row = (tgpig.x * MV8_NSG + sgitg) * MV8_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 34;
    device const float * y_base = x + n * in_dim;
    float sumf[MV8_NR0] = {0.f};

    for (uint ib = 0; ib < nb; ++ib) {
        const float y = y_base[ib * Q8_0_QK + tiisg];
        for (short row = 0; row < MV8_NR0; ++row) {
            const uint row_id = first_row + row;
            if (row_id >= out_dim) continue;
            device const block_q8_0_56 * blk =
                (device const block_q8_0_56 *)(w_raw + row_id * row_bytes) + ib;
            sumf[row] += (float)blk->d * y * (float)blk->qs[tiisg];
        }
    }

    for (short row = 0; row < MV8_NR0 && first_row + row < out_dim; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[n * out_dim + first_row + row] = tot;
        }
    }
}

kernel void simd_mv_q8_0_dual_f32(
    device const uint8_t* gate_w_raw [[buffer(0)]],
    device const uint8_t* up_w_raw   [[buffer(1)]],
    device const float*   x          [[buffer(2)]],
    device       float*   gate_out   [[buffer(3)]],
    device       float*   up_out     [[buffer(4)]],
    constant     uint&    in_dim     [[buffer(5)]],
    constant     uint&    out_dim    [[buffer(6)]],
    constant     uint&    batch      [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint first_row = (tgpig.x * MV8_NSG + sgitg) * MV8_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 34;
    device const float * y_base = x + n * in_dim;
    float gate_sum[MV8_NR0] = {0.f};
    float up_sum[MV8_NR0] = {0.f};

    for (uint ib = 0; ib < nb; ++ib) {
        const float y = y_base[ib * Q8_0_QK + tiisg];
        for (short row = 0; row < MV8_NR0; ++row) {
            const uint row_id = first_row + row;
            if (row_id >= out_dim) continue;
            device const block_q8_0_56 * gate_blk =
                (device const block_q8_0_56 *)(gate_w_raw + row_id * row_bytes) + ib;
            device const block_q8_0_56 * up_blk =
                (device const block_q8_0_56 *)(up_w_raw + row_id * row_bytes) + ib;
            gate_sum[row] += (float)gate_blk->d * y * (float)gate_blk->qs[tiisg];
            up_sum[row] += (float)up_blk->d * y * (float)up_blk->qs[tiisg];
        }
    }

    for (short row = 0; row < MV8_NR0 && first_row + row < out_dim; ++row) {
        const float gate_tot = simd_sum(gate_sum[row]);
        const float up_tot = simd_sum(up_sum[row]);
        if (tiisg == 0) {
            const uint out_idx = n * out_dim + first_row + row;
            gate_out[out_idx] = gate_tot;
            up_out[out_idx] = up_tot;
        }
    }
}

constant short MV8_TOP_ROWS_PER_TG = 12;

kernel void simd_mv_q8_0_top1_tiles_f32(
    device const uint8_t* w_raw       [[buffer(0)]],
    device const float*   x           [[buffer(1)]],
    device       float*   tile_values [[buffer(2)]],
    device       uint*    tile_ids    [[buffer(3)]],
    constant     uint&    in_dim      [[buffer(4)]],
    constant     uint&    out_dim     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint nb = in_dim / Q8_0_QK;
    const uint row_bytes = nb * 34;

    threadgroup float sg_values[MV8_NSG];
    threadgroup uint  sg_ids[MV8_NSG];

    float best = -INFINITY;
    uint best_id = 0;

    for (short tile_row = 0; tile_row < (MV8_TOP_ROWS_PER_TG + MV8_NSG - 1) / MV8_NSG; ++tile_row) {
        const uint row_id = tgpig.x * MV8_TOP_ROWS_PER_TG + tile_row * MV8_NSG + sgitg;
        if (row_id >= out_dim) continue;

        float sumf = 0.0f;
        for (uint ib = 0; ib < nb; ++ib) {
            device const block_q8_0_56 * blk =
                (device const block_q8_0_56 *)(w_raw + row_id * row_bytes) + ib;
            sumf += (float)blk->d * x[ib * Q8_0_QK + tiisg] * (float)blk->qs[tiisg];
        }

        const float total = simd_sum(sumf);
        if (tiisg == 0 && (total > best || (total == best && row_id < best_id))) {
            best = total;
            best_id = row_id;
        }
    }

    if (tiisg == 0) {
        sg_values[sgitg] = best;
        sg_ids[sgitg] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float group_best = sg_values[0];
        uint group_best_id = sg_ids[0];
        for (uint i = 1; i < MV8_NSG; ++i) {
            const float v = sg_values[i];
            const uint id = sg_ids[i];
            if (v > group_best || (v == group_best && id < group_best_id)) {
                group_best = v;
                group_best_id = id;
            }
        }
        tile_values[tgpig.x] = group_best;
        tile_ids[tgpig.x] = group_best_id;
    }
}

// ============================================================================
// Q5_K
// ============================================================================
struct block_q5_K_56 {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

// Match llama.cpp ggml-metal-impl.h: N_SG_Q5_K=2, N_R0_Q5_K=1.
constant short MV5_NSG = 2;
constant short MV5_NR0 = 1;

// GEMV for Q5_K, direct port of llama.cpp kernel_mul_mv_q5_K_f32_impl.
kernel void simd_mv_q5k_f32(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2 * iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    const uint nb = in_dim / Q56K_QK_K;
    const uint first_row = (tgpig.x * MV5_NSG + sgitg) * MV5_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 176;

    float sumf[MV5_NR0] = {0.f};
    float yl[16], yh[16];
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y_base = x + n * in_dim;
    device const float * y1 = y_base + ix * Q56K_QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l + 0] = y1[l +  0]; sumy[0] += yl[l + 0];
            yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
            yh[l + 0] = y2[l +  0]; sumy[2] += yh[l + 0];
            yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
        }

        for (short row = 0; row < MV5_NR0; ++row) {
            device const block_q5_K_56 * blk =
                (device const block_q5_K_56 *)(w_raw + (first_row + row) * row_bytes) + i;

            device const uint8_t  * q1 = blk->qs + q_offset;
            device const uint8_t  * q2 = q1 + 64;
            device const uint8_t  * qh = blk->qh + l0;
            device const half     * dh = &blk->d;
            device const uint16_t * a  = (device const uint16_t *)blk->scales + iq;

            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL for (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l + 0] * (q1[l] & 0x0F);
                acc1[1] += yl[l + 8] * (q1[l] & 0xF0);
                acc1[2] += yh[l + 0] * (q2[l] & 0x0F);
                acc1[3] += yh[l + 8] * (q2[l] & 0xF0);
                acc2[0] += h & hm1 ? yl[l + 0] : 0.f;
                acc2[1] += h & hm2 ? yl[l + 8] : 0.f;
                acc2[2] += h & hm3 ? yh[l + 0] : 0.f;
                acc2[3] += h & hm4 ? yh[l + 8] : 0.f;
            }

            sumf[row] += dh[0] * (sc8[0] * (acc1[0]        + 16.f * acc2[0]) +
                                  sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                                  sc8[4] * (acc1[2]        + 16.f * acc2[2]) +
                                  sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y1 += 4 * Q56K_QK_K;
    }

    for (short row = 0; row < MV5_NR0 && first_row + row < out_dim; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[n * out_dim + first_row + row] = tot;
        }
    }
}

// ============================================================================
// Q6_K
// ============================================================================
struct block_q6_K_56 {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half    d;
};

// Q6_K uses one row per simdgroup. On Apple M2 Max this reduces register
// pressure enough to slightly beat the extra input-load amortization from NR0=2.
constant short MV6_NSG = 2;
constant short MV6_NR0 = 1;

// GEMV for Q6_K. Direct port of llama.cpp kernel_mul_mv_q6_K_f32_impl.
kernel void simd_mv_q6k_f32(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const uint nb = in_dim / Q56K_QK_K;
    const uint first_row = (tgpig.x * MV6_NSG + sgitg) * MV6_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 210;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    float sumf[MV6_NR0] = {0.f};
    float yl[16];

    device const float * y_base = x + n * in_dim;

    for (uint i = ix; i < nb; i += 2) {
        device const float * y = y_base + i * Q56K_QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < MV6_NR0; ++row) {
            device const block_q6_K_56 * blk =
                (device const block_q6_K_56 *)(w_raw + (first_row + row) * row_bytes) + i;

            device const uint8_t * q1 = blk->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = blk->qh + q_offset_h;
            device const int8_t  * sc = blk->scales + is;
            device const half    * dh = &blk->d;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                  sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < MV6_NR0 && first_row + row < out_dim; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[n * out_dim + first_row + row] = tot;
        }
    }
}

// Same arithmetic as simd_mv_q6k_f32, but folds the final residual add into
// the output write. Used by the decode FFN-down path to remove one add kernel.
kernel void simd_mv_q6k_f32_add(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    device const float*   residual[[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const uint nb = in_dim / Q56K_QK_K;
    const uint first_row = (tgpig.x * MV6_NSG + sgitg) * MV6_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 210;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    float sumf[MV6_NR0] = {0.f};
    float yl[16];

    device const float * y_base = x + n * in_dim;

    for (uint i = ix; i < nb; i += 2) {
        device const float * y = y_base + i * Q56K_QK_K + y_offset;
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < MV6_NR0; ++row) {
            device const block_q6_K_56 * blk =
                (device const block_q6_K_56 *)(w_raw + (first_row + row) * row_bytes) + i;

            device const uint8_t * q1 = blk->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = blk->qh + q_offset_h;
            device const int8_t  * sc = blk->scales + is;
            device const half    * dh = &blk->d;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                                  sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < MV6_NR0 && first_row + row < out_dim; ++row) {
        float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const uint out_idx = n * out_dim + first_row + row;
            output[out_idx] = residual[out_idx] + tot;
        }
    }
}

// Q6_K lm-head greedy top1. This computes the same per-row dot products as
// `simd_mv_q6k_f32`, but each threadgroup emits only the best row in a small
// tile instead of materializing the full vocab logits vector. It is intended
// only for greedy decode experiments where full logits are not needed.
constant short MV6_TOP_ROWS_PER_TG = 12;

kernel void simd_mv_q6k_top1_tiles_f32(
    device const uint8_t* w_raw       [[buffer(0)]],
    device const float*   x           [[buffer(1)]],
    device       float*   tile_values [[buffer(2)]],
    device       uint*    tile_ids    [[buffer(3)]],
    constant     uint&    in_dim      [[buffer(4)]],
    constant     uint&    out_dim     [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const uint nb = in_dim / Q56K_QK_K;
    const uint row_bytes = nb * 210;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    threadgroup float sg_values[MV6_NSG];
    threadgroup uint  sg_ids[MV6_NSG];

    float best = -INFINITY;
    uint best_id = 0;

    for (short tile_row = 0; tile_row < MV6_TOP_ROWS_PER_TG / MV6_NSG; ++tile_row) {
        const uint row_id = tgpig.x * MV6_TOP_ROWS_PER_TG + tile_row * MV6_NSG + sgitg;
        if (row_id >= out_dim) continue;

        float sumf = 0.0f;
        float yl[16];

        for (uint i = ix; i < nb; i += 2) {
            device const float * y = x + i * Q56K_QK_K + y_offset;
            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = y[l +  0];
                yl[4*l + 1] = y[l + 32];
                yl[4*l + 2] = y[l + 64];
                yl[4*l + 3] = y[l + 96];
            }

            device const block_q6_K_56 * blk =
                (device const block_q6_K_56 *)(w_raw + row_id * row_bytes) + i;

            device const uint8_t * q1 = blk->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = blk->qh + q_offset_h;
            device const int8_t  * sc = blk->scales + is;
            device const half    * dh = &blk->d;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                             sums[2] * sc[4] + sums[3] * sc[6]);
        }

        const float total = simd_sum(sumf);
        if (tiisg == 0 && (total > best || (total == best && row_id < best_id))) {
            best = total;
            best_id = row_id;
        }
    }

    if (tiisg == 0) {
        sg_values[sgitg] = best;
        sg_ids[sgitg] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float group_best = sg_values[0];
        uint group_best_id = sg_ids[0];
        for (uint i = 1; i < MV6_NSG; ++i) {
            const float v = sg_values[i];
            const uint id = sg_ids[i];
            if (v > group_best || (v == group_best && id < group_best_id)) {
                group_best = v;
                group_best_id = id;
            }
        }
        tile_values[tgpig.x] = group_best;
        tile_ids[tgpig.x] = group_best_id;
    }
}

kernel void simd_mv_q6k_top1_tiles_batch_f32(
    device const uint8_t* w_raw       [[buffer(0)]],
    device const float*   x           [[buffer(1)]],
    device       float*   tile_values [[buffer(2)]],
    device       uint*    tile_ids    [[buffer(3)]],
    constant     uint&    in_dim      [[buffer(4)]],
    constant     uint&    out_dim     [[buffer(5)]],
    constant     uint&    tile_count  [[buffer(6)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const uint nb = in_dim / Q56K_QK_K;
    const uint row_bytes = nb * 210;
    const uint batch_id = tgpig.y;
    device const float* x_row = x + batch_id * in_dim;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    threadgroup float sg_values[MV6_NSG];
    threadgroup uint  sg_ids[MV6_NSG];

    float best = -INFINITY;
    uint best_id = 0;

    for (short tile_row = 0; tile_row < MV6_TOP_ROWS_PER_TG / MV6_NSG; ++tile_row) {
        const uint row_id = tgpig.x * MV6_TOP_ROWS_PER_TG + tile_row * MV6_NSG + sgitg;
        if (row_id >= out_dim) continue;

        float sumf = 0.0f;
        float yl[16];

        for (uint i = ix; i < nb; i += 2) {
            device const float * y = x_row + i * Q56K_QK_K + y_offset;
            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = y[l +  0];
                yl[4*l + 1] = y[l + 32];
                yl[4*l + 2] = y[l + 64];
                yl[4*l + 3] = y[l + 96];
            }

            device const block_q6_K_56 * blk =
                (device const block_q6_K_56 *)(w_raw + row_id * row_bytes) + i;

            device const uint8_t * q1 = blk->ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = blk->qh + q_offset_h;
            device const int8_t  * sc = blk->scales + is;
            device const half    * dh = &blk->d;

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] +
                             sums[2] * sc[4] + sums[3] * sc[6]);
        }

        const float total = simd_sum(sumf);
        if (tiisg == 0 && (total > best || (total == best && row_id < best_id))) {
            best = total;
            best_id = row_id;
        }
    }

    if (tiisg == 0) {
        sg_values[sgitg] = best;
        sg_ids[sgitg] = best_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        float group_best = sg_values[0];
        uint group_best_id = sg_ids[0];
        for (uint i = 1; i < MV6_NSG; ++i) {
            const float v = sg_values[i];
            const uint id = sg_ids[i];
            if (v > group_best || (v == group_best && id < group_best_id)) {
                group_best = v;
                group_best_id = id;
            }
        }
        const uint out_idx = batch_id * tile_count + tgpig.x;
        tile_values[out_idx] = group_best;
        tile_ids[out_idx] = group_best_id;
    }
}

kernel void qwen35_top1_reduce_tiles(
    device const float* tile_values [[buffer(0)]],
    device const uint*  tile_ids    [[buffer(1)]],
    device       uint*  top_id      [[buffer(2)]],
    device       float* top_value   [[buffer(3)]],
    constant     uint&  tile_count  [[buffer(4)]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float local_values[256];
    threadgroup uint  local_ids[256];

    float best = -INFINITY;
    uint best_id = 0;
    for (uint i = tid; i < tile_count; i += 256) {
        const float v = tile_values[i];
        const uint id = tile_ids[i];
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }

    local_values[tid] = best;
    local_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float group_best = local_values[0];
        uint group_best_id = local_ids[0];
        for (uint i = 1; i < 256; ++i) {
            const float v = local_values[i];
            const uint id = local_ids[i];
            if (v > group_best || (v == group_best && id < group_best_id)) {
                group_best = v;
                group_best_id = id;
            }
        }
        top_id[0] = group_best_id;
        top_value[0] = group_best;
    }
}

kernel void qwen35_top1_reduce_tiles_batch(
    device const float* tile_values [[buffer(0)]],
    device const uint*  tile_ids    [[buffer(1)]],
    device       uint*  top_id      [[buffer(2)]],
    device       float* top_value   [[buffer(3)]],
    constant     uint&  tile_count  [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float local_values[256];
    threadgroup uint  local_ids[256];

    float best = -INFINITY;
    uint best_id = 0;
    const uint base = row * tile_count;
    for (uint i = tid; i < tile_count; i += 256) {
        const float v = tile_values[base + i];
        const uint id = tile_ids[base + i];
        if (v > best || (v == best && id < best_id)) {
            best = v;
            best_id = id;
        }
    }

    local_values[tid] = best;
    local_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float group_best = local_values[0];
        uint group_best_id = local_ids[0];
        for (uint i = 1; i < 256; ++i) {
            const float v = local_values[i];
            const uint id = local_ids[i];
            if (v > group_best || (v == group_best && id < group_best_id)) {
                group_best = v;
                group_best_id = id;
            }
        }
        top_id[row] = group_best_id;
        top_value[row] = group_best;
    }
}

kernel void qwen35_top1_reduce_f16_rows(
    device const half* logits    [[buffer(0)]],
    device       uint* top_id    [[buffer(1)]],
    device       float* top_value [[buffer(2)]],
    constant     uint& out_dim   [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float local_values[256];
    threadgroup uint  local_ids[256];

    float best = -INFINITY;
    uint best_id = 0;
    const uint base = row * out_dim;
    for (uint i = tid; i < out_dim; i += 256) {
        const float v = float(logits[base + i]);
        if (v > best || (v == best && i < best_id)) {
            best = v;
            best_id = i;
        }
    }

    local_values[tid] = best;
    local_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float group_best = local_values[0];
        uint group_best_id = local_ids[0];
        for (uint i = 1; i < 256; ++i) {
            const float v = local_values[i];
            const uint id = local_ids[i];
            if (v > group_best || (v == group_best && id < group_best_id)) {
                group_best = v;
                group_best_id = id;
            }
        }
        top_id[row] = group_best_id;
        top_value[row] = group_best;
    }
}

static inline void qwen35_update_top2(float v,
                                      uint id,
                                      thread float &best,
                                      thread uint &best_id,
                                      thread float &second,
                                      thread uint &second_id)
{
    if (v > best || (v == best && id < best_id)) {
        second = best;
        second_id = best_id;
        best = v;
        best_id = id;
    } else if (id != best_id && (v > second || (v == second && id < second_id))) {
        second = v;
        second_id = id;
    }
}

kernel void qwen35_top2_reduce_f16_rows(
    device const half* logits        [[buffer(0)]],
    device       uint* top_id        [[buffer(1)]],
    device       float* top_value    [[buffer(2)]],
    device       uint* second_id     [[buffer(3)]],
    device       float* second_value [[buffer(4)]],
    constant     uint& out_dim       [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]])
{
    threadgroup float local_best_values[256];
    threadgroup uint  local_best_ids[256];
    threadgroup float local_second_values[256];
    threadgroup uint  local_second_ids[256];

    float best = -INFINITY;
    uint best_id = 0;
    float second = -INFINITY;
    uint second_id_local = 0;
    const uint base = row * out_dim;
    for (uint i = tid; i < out_dim; i += 256) {
        qwen35_update_top2(float(logits[base + i]), i, best, best_id, second, second_id_local);
    }

    local_best_values[tid] = best;
    local_best_ids[tid] = best_id;
    local_second_values[tid] = second;
    local_second_ids[tid] = second_id_local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float group_best = -INFINITY;
        uint group_best_id = 0;
        float group_second = -INFINITY;
        uint group_second_id = 0;
        for (uint i = 0; i < 256; ++i) {
            qwen35_update_top2(local_best_values[i], local_best_ids[i],
                               group_best, group_best_id, group_second, group_second_id);
            qwen35_update_top2(local_second_values[i], local_second_ids[i],
                               group_best, group_best_id, group_second, group_second_id);
        }
        top_id[row] = group_best_id;
        top_value[row] = group_best;
        second_id[row] = group_second_id;
        second_value[row] = group_second;
    }
}
