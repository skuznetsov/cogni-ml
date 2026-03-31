// SIMD-group GEMM for Q5_K / Q6_K — matches llama.cpp's accumulation pattern.
// Each output element computed by 32 threads (1 SIMD group).
// Integer quant values accumulated separately from scale multiplication
// to avoid large intermediate products that lose F32 precision.
//
// Standard: threadgroups = [ceil(out_dim/(N_ROWS*NR0)), batch, 1]
// MoE:      threadgroups = [ceil(out_dim/(N_ROWS*NR0)), seq_len, 1]
//           (kernel reads expert_offsets to compute actual batch + offset)

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

constant uint QK_K = 256;
constant uint N_ROWS = 2;   // simdgroups per threadgroup
constant uint NR0 = 2;      // output rows per simdgroup (amortize input reads)  // output rows per threadgroup (2 simdgroups)

// ============================================================================
// Q5_K SIMD matmul — llama.cpp style accumulation
// ============================================================================
kernel void simd_gemm_q5k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const half*    x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       half*    output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup processes NR0 consecutive output rows, sharing input reads
    const uint first_row = (tgpig.x * N_ROWS + sgitg) * NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint nb = in_dim / QK_K;
    const uint row_bytes = nb * 176;

    device const half* y1 = x + n * in_dim;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;

    const short l0       = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uint8_t hm1 = 1u << (2*iq);
    const uint8_t hm2 = hm1 << 1;
    const uint8_t hm3 = hm1 << 4;
    const uint8_t hm4 = hm2 << 4;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    float sumf[NR0] = {0.f};  // accumulator per row
    device const half* yp = y1 + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        // Load y data ONCE, reuse for all NR0 rows
        device const half* y2 = yp + 128;
        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL for (short l = 0; l < 8; ++l) {
            yl[l+0] = yp[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = yp[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        // Process NR0 rows with shared y data
        for (short row = 0; row < (short)NR0; ++row) {
            device const uint8_t* bp = w_raw + (first_row + row) * row_bytes + i * 176;
            device const uint8_t* q1 = bp + 48 + q_offset;
            device const uint8_t* qh = bp + 16 + l0;
            device const half*    dh = (device const half*)bp;
            device const uint16_t* a = (device const uint16_t*)(bp + 4) + iq;
            device const uint8_t* q2 = q1 + 64;

            uint16_t sc16[4];
            sc16[0] = a[0] & kmask1;
            sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

            float4 acc1 = {0.f}, acc2 = {0.f};
            FOR_UNROLL for (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * float(q1[l] & 0x0F);
                acc1[1] += yl[l+8] * float(q1[l] & 0xF0);
                acc1[2] += yh[l+0] * float(q2[l] & 0x0F);
                acc1[3] += yh[l+8] * float(q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }

            sumf[row] +=
                dh[0] * (  sc8[0] * (acc1[0]       + 16.f*acc2[0])
                          + sc8[1] * (acc1[1]/16.f  + 16.f*acc2[1])
                          + sc8[4] * (acc1[2]       + 16.f*acc2[2])
                          + sc8[5] * (acc1[3]/16.f  + 16.f*acc2[3]) )
              - dh[1] * (  sumy[0]*sc8[2] + sumy[1]*sc8[3]
                         + sumy[2]*sc8[6] + sumy[3]*sc8[7] );
        }

        yp += 4 * QK_K;
    }

    // SIMD reduction + output for each row
    for (short row = 0; row < (short)NR0 && first_row + row < out_dim; ++row) {
        float sum = simd_sum(sumf[row]) + bias[first_row + row];
        if (apply_gelu) {
            if (sum > 10.0f) { }
            else if (sum < -10.0f) { sum = 0.0f; }
            else {
                float t = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
                sum = 0.5f * sum * (1.0f + tanh(t));
            }
        }
        if (tiisg == 0) {
            output[n * out_dim + first_row + row] = half(sum);
        }
    }
}

// ============================================================================
// Q6_K SIMD matmul — exact llama.cpp pattern
// Block layout: [ql:128B][qh:64B][sc:16B][d:2B] = 210 bytes
// ============================================================================
kernel void simd_gemm_q6k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const half*    x       [[buffer(1)]],  // FP16 input
    device const float*   bias    [[buffer(2)]],  // F32 bias
    device       half*    output  [[buffer(3)]],  // FP16 output
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup processes NR0 consecutive output rows, sharing input reads
    const uint first_row = (tgpig.x * N_ROWS + sgitg) * NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint nb = in_dim / QK_K;
    const uint row_bytes = nb * 210;

    device const half* yy = x + n * in_dim;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    // Thread decomposition (matches llama.cpp)
    const short tid = tiisg / 2;    // 0..15
    const short ix  = tiisg % 2;    // 0..1 stride
    const short ip  = tid / 8;      // 0..1 which 128-elem half
    const short il  = tid % 8;      // 0..7
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    float sumf[NR0] = {0.f};  // accumulator per row
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        // Load y data ONCE, reuse for all NR0 rows
        device const half* y = yy + i * QK_K + y_offset;

        FOR_UNROLL for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        // Process NR0 rows with shared y data
        for (short row = 0; row < (short)NR0; ++row) {
            device const uint8_t* bp = w_raw + (first_row + row) * row_bytes + i * 210;
            device const uint8_t* q1 = bp + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = bp + 128 + q_offset_h;
            device const int8_t*  sc = (device const int8_t*)(bp + 192) + is;
            device const half*    dh = (device const half*)(bp + 208);

            float4 sums = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * float((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * float((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * float((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * float((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < (short)NR0 && first_row + row < out_dim; ++row) {
        float sum = simd_sum(sumf[row]) + bias[first_row + row];
        if (apply_gelu) {
            if (sum > 10.0f) { }
            else if (sum < -10.0f) { sum = 0.0f; }
            else {
                float t = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
                sum = 0.5f * sum * (1.0f + tanh(t));
            }
        }
        if (tiisg == 0) {
            output[n * out_dim + first_row + row] = half(sum);
        }
    }
}

// ============================================================================
// MoE variants — GPU-side expert_offsets, zero CPU-GPU sync
// Input/output are packed contiguously: [expert0_tokens..., expert1_tokens..., ...]
// Kernel reads expert_offsets to compute base offset + batch bounds
// Dispatch: threadgroups = [ceil(out_dim/(N_ROWS*NR0)), seq_len, 1]
// ============================================================================
kernel void simd_gemm_q5k_moe(
    device const uint8_t* w_raw        [[buffer(0)]],   // weights for THIS expert
    device const half*    x_packed     [[buffer(1)]],   // full packed input
    device const float*   bias         [[buffer(2)]],
    device       half*    out_packed   [[buffer(3)]],   // full packed output
    device const int*     expert_offs  [[buffer(4)]],   // [n_experts+1] offsets
    constant     uint&    expert_id    [[buffer(5)]],
    constant     uint&    in_dim       [[buffer(6)]],
    constant     uint&    out_dim      [[buffer(7)]],
    constant     uint&    apply_gelu   [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint first_row = (tgpig.x * N_ROWS + sgitg) * NR0;
    const uint n_local = tgpig.y;
    const uint base = expert_offs[expert_id];
    const uint eb = expert_offs[expert_id + 1] - base;
    if (first_row >= out_dim || n_local >= eb) return;

    const uint n = base + n_local;
    const uint nb = in_dim / QK_K;
    const uint row_bytes = nb * 176;
    device const half* y1 = x_packed + n * in_dim;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;
    const short l0       = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;
    const uint8_t hm1 = 1u << (2*iq), hm2 = hm1 << 1, hm3 = hm1 << 4, hm4 = hm2 << 4;
    constexpr uint16_t kmask1 = 0x3f3f, kmask2 = 0x0f0f, kmask3 = 0xc0c0;

    float sumf[NR0] = {0.f};
    device const half* yp = y1 + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        device const half* y2 = yp + 128;
        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        FOR_UNROLL for (short l = 0; l < 8; ++l) {
            yl[l+0] = yp[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = yp[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }
        for (short row = 0; row < (short)NR0; ++row) {
            device const uint8_t* bp = w_raw + (first_row + row) * row_bytes + i * 176;
            device const uint8_t* q1 = bp + 48 + q_offset;
            device const uint8_t* qh = bp + 16 + l0;
            device const half*    dh = (device const half*)bp;
            device const uint16_t* a = (device const uint16_t*)(bp + 4) + iq;
            device const uint8_t* q2 = q1 + 64;
            uint16_t sc16[4];
            sc16[0] = a[0] & kmask1; sc16[1] = a[2] & kmask1;
            sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
            sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
            thread const uint8_t* sc8 = (thread const uint8_t*)sc16;
            float4 acc1 = {0.f}, acc2 = {0.f};
            FOR_UNROLL for (short l = 0; l < 8; ++l) {
                uint8_t h = qh[l];
                acc1[0] += yl[l+0] * float(q1[l] & 0x0F);
                acc1[1] += yl[l+8] * float(q1[l] & 0xF0);
                acc1[2] += yh[l+0] * float(q2[l] & 0x0F);
                acc1[3] += yh[l+8] * float(q2[l] & 0xF0);
                acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
                acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
                acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
                acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
            }
            sumf[row] +=
                dh[0] * (  sc8[0] * (acc1[0]       + 16.f*acc2[0])
                          + sc8[1] * (acc1[1]/16.f  + 16.f*acc2[1])
                          + sc8[4] * (acc1[2]       + 16.f*acc2[2])
                          + sc8[5] * (acc1[3]/16.f  + 16.f*acc2[3]) )
              - dh[1] * (  sumy[0]*sc8[2] + sumy[1]*sc8[3]
                         + sumy[2]*sc8[6] + sumy[3]*sc8[7] );
        }
        yp += 4 * QK_K;
    }

    for (short row = 0; row < (short)NR0 && first_row + row < out_dim; ++row) {
        float sum = simd_sum(sumf[row]) + bias[first_row + row];
        if (apply_gelu) {
            if (sum > 10.0f) { }
            else if (sum < -10.0f) { sum = 0.0f; }
            else {
                float t = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
                sum = 0.5f * sum * (1.0f + tanh(t));
            }
        }
        if (tiisg == 0) {
            out_packed[n * out_dim + first_row + row] = half(sum);
        }
    }
}

kernel void simd_gemm_q6k_moe(
    device const uint8_t* w_raw        [[buffer(0)]],
    device const half*    x_packed     [[buffer(1)]],
    device const float*   bias         [[buffer(2)]],
    device       half*    out_packed   [[buffer(3)]],
    device const int*     expert_offs  [[buffer(4)]],
    constant     uint&    expert_id    [[buffer(5)]],
    constant     uint&    in_dim       [[buffer(6)]],
    constant     uint&    out_dim      [[buffer(7)]],
    constant     uint&    apply_gelu   [[buffer(8)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint first_row = (tgpig.x * N_ROWS + sgitg) * NR0;
    const uint n_local = tgpig.y;
    const uint base = expert_offs[expert_id];
    const uint eb = expert_offs[expert_id + 1] - base;
    if (first_row >= out_dim || n_local >= eb) return;

    const uint n = base + n_local;
    const uint nb = in_dim / QK_K;
    const uint row_bytes = nb * 210;
    device const half* yy = x_packed + n * in_dim;

    constexpr uint8_t kmask1 = 0x03, kmask2 = 0x0C, kmask3 = 0x30, kmask4 = 0xC0;
    const short tid = tiisg / 2, ix = tiisg % 2;
    const short ip = tid / 8, il = tid % 8;
    const short l0 = 4 * il, is = 8 * ip + l0 / 16;
    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    float sumf[NR0] = {0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const half* y = yy + i * QK_K + y_offset;
        FOR_UNROLL for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0]; yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64]; yl[4*l + 3] = y[l + 96];
        }
        for (short row = 0; row < (short)NR0; ++row) {
            device const uint8_t* bp = w_raw + (first_row + row) * row_bytes + i * 210;
            device const uint8_t* q1 = bp + q_offset_l;
            device const uint8_t* q2 = q1 + 32;
            device const uint8_t* qh = bp + 128 + q_offset_h;
            device const int8_t*  sc = (device const int8_t*)(bp + 192) + is;
            device const half*    dh = (device const half*)(bp + 208);
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * float((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * float((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * float((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * float((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
        }
    }

    for (short row = 0; row < (short)NR0 && first_row + row < out_dim; ++row) {
        float sum = simd_sum(sumf[row]) + bias[first_row + row];
        if (apply_gelu) {
            if (sum > 10.0f) { }
            else if (sum < -10.0f) { sum = 0.0f; }
            else {
                float t = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
                sum = 0.5f * sum * (1.0f + tanh(t));
            }
        }
        if (tiisg == 0) {
            out_packed[n * out_dim + first_row + row] = half(sum);
        }
    }
}
