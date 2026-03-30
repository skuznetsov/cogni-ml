// SIMD-group GEMM for Q5_K / Q6_K — matches llama.cpp's accumulation pattern.
// Each output element computed by 32 threads (1 SIMD group).
// Integer quant values accumulated separately from scale multiplication
// to avoid large intermediate products that lose F32 precision.
//
// Dispatch: threadgroups = [ceil(out_dim/N_ROWS), batch, 1]
//           threads_per_threadgroup = [32, N_ROWS, 1]

#include <metal_stdlib>
using namespace metal;

constant uint QK_K = 256;
constant uint N_ROWS = 2;  // output rows per threadgroup (2 simdgroups)

// ============================================================================
// Q5_K SIMD matmul — llama.cpp style accumulation
// ============================================================================
kernel void simd_gemm_q5k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint m = tgpig.x * N_ROWS + sgitg;  // output row
    const uint n = tgpig.y;                     // batch index
    if (m >= out_dim || n >= batch) return;

    const uint nb = in_dim / QK_K;  // blocks per row
    const uint row_bytes = nb * 176;

    device const uint8_t* row_ptr = w_raw + m * row_bytes;
    device const float* y1 = x + n * in_dim;

    // Thread decomposition within 32-lane SIMD group (matches llama.cpp)
    const short tid = tiisg / 4;    // 0..7
    const short ix  = tiisg % 4;    // 0..3 — stride over blocks
    const short iq  = tid / 4;      // 0..1 — which 128-elem half
    const short ir  = tid % 4;      // 0..3 — which 8-elem sub-group

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

    float sumf = 0.0f;
    device const float * yp = y1 + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        device const uint8_t* bp = row_ptr + i * 176;
        device const uint8_t* q1 = bp + 48 + q_offset;   // qs + q_offset
        device const uint8_t* qh = bp + 16 + l0;         // qh + l0
        device const half*    dh = (device const half*)bp;
        device const uint16_t* a = (device const uint16_t*)(bp + 4) + iq;

        device const uint8_t* q2 = q1 + 64;
        device const float* y2 = yp + 128;

        // Load 16 src1 floats and accumulate per-quadrant sums
        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = yp[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = yp[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        // Unpack 6-bit scales
        uint16_t sc16[4];
        sc16[0] = a[0] & kmask1;
        sc16[1] = a[2] & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);
        thread const uint8_t* sc8 = (thread const uint8_t*)sc16;

        // Separate integer accumulation (no scale multiplication per element)
        float4 acc1 = {0.f};  // low 4-bit products
        float4 acc2 = {0.f};  // high bit (5th bit) contributions

        for (short l = 0; l < 8; ++l) {
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

        // Scale multiplication only ONCE per block (not per element)
        sumf +=
            dh[0] * (  sc8[0] * (acc1[0]       + 16.f*acc2[0])
                      + sc8[1] * (acc1[1]/16.f  + 16.f*acc2[1])
                      + sc8[4] * (acc1[2]       + 16.f*acc2[2])
                      + sc8[5] * (acc1[3]/16.f  + 16.f*acc2[3]) )
          - dh[1] * (  sumy[0]*sc8[2] + sumy[1]*sc8[3]
                     + sumy[2]*sc8[6] + sumy[3]*sc8[7] );

        yp += 4 * QK_K;
    }

    // SIMD reduction
    float sum = simd_sum(sumf) + bias[m];

    if (apply_gelu) {
        // Safe GELU: short-circuit for |v|>10 to avoid Metal tanh(large) → NaN
        if (sum > 10.0f) { /* GELU ≈ identity */ }
        else if (sum < -10.0f) { sum = 0.0f; }
        else {
            float t = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
            sum = 0.5f * sum * (1.0f + tanh(t));
        }
    }

    if (tiisg == 0) {
        output[n * out_dim + m] = sum;
    }
}

// ============================================================================
// Q6_K SIMD matmul — exact llama.cpp pattern
// Block layout: [ql:128B][qh:64B][sc:16B][d:2B] = 210 bytes
// ============================================================================
kernel void simd_gemm_q6k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       float*   output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const uint m = tgpig.x * N_ROWS + sgitg;
    const uint n = tgpig.y;
    if (m >= out_dim || n >= batch) return;

    const uint nb = in_dim / QK_K;
    const uint row_bytes = nb * 210;

    device const uint8_t* row_base = w_raw + m * row_bytes;
    device const float* yy = x + n * in_dim;

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

    float sumf = 0.0f;
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const uint8_t* bp = row_base + i * 210;
        device const uint8_t* q1 = bp + q_offset_l;           // ql
        device const uint8_t* q2 = q1 + 32;
        device const uint8_t* qh = bp + 128 + q_offset_h;     // qh
        device const int8_t*  sc = (device const int8_t*)(bp + 192) + is;
        device const half*    dh = (device const half*)(bp + 208);

        device const float* y = yy + i * QK_K + y_offset;

        // Load 4 elements from each of 4 sub-rows (stride 32)
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        float4 sums = {0.f, 0.f, 0.f, 0.f};

        for (short l = 0; l < 4; ++l) {
            sums[0] += yl[4*l + 0] * float((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
            sums[1] += yl[4*l + 1] * float((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
            sums[2] += yl[4*l + 2] * float((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
            sums[3] += yl[4*l + 3] * float((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
        }

        sumf += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);
    }

    float sum = simd_sum(sumf) + bias[m];

    if (apply_gelu) {
        if (sum > 10.0f) { /* GELU ≈ identity */ }
        else if (sum < -10.0f) { sum = 0.0f; }
        else {
            float t = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
            sum = 0.5f * sum * (1.0f + tanh(t));
        }
    }

    if (tiisg == 0) {
        output[n * out_dim + m] = sum;
    }
}
