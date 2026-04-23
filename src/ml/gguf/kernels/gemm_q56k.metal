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

// Match Q4_K kernel dispatch: NSG=2 simdgroups, NR0=2 rows/simdgroup.
constant short MV5_NSG = 2;
constant short MV5_NR0 = 2;

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

// Q6_K uses NSG=2, NR0=2 (symmetric with Q5_K).
constant short MV6_NSG = 2;
constant short MV6_NR0 = 2;

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
