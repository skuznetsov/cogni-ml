// Q4_K Metal matmul kernels — F32 input, F32 output.
// Used by Qwen 3.5/3.6 (and any arch with Q4_K weights).
//
// Two variants:
//   simd_mv_q4k_f32  — GEMV for decode (n_tokens=1). Ported from llama.cpp
//                      kernel_mul_mv_q4_K_f32_impl. Packed int accumulation:
//                      integer quant values accumulated separately from scales
//                      to preserve F32 precision with many terms.
//   simd_mm_q4k_f32  — GEMM for prefill (n_tokens>1). Adapted from our
//                      simd_mm_q5k with Q4_K dequant swap and F32 output.
//                      Keeps double-buffered shmem + cooperative write.
//
// Q4_K block layout: [d:2B][dmin:2B][scales:12B][qs:128B] = 144 bytes.
// 256 elements per block (QK_K). 16 sub-blocks of 16 elements each.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

constant uint QK_K = 256;

// ============================================================================
// Q4_K block structure and dequant helper (for GEMM path)
// ============================================================================
struct block_q4_K {
    half    d;
    half    dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

static inline uchar2 get_scale_min_k4_scalar(uint j, device const uchar * q) {
    if (j < 4) {
        return uchar2{uchar(q[j] & 63), uchar(q[j + 4] & 63)};
    }
    return uchar2{
        uchar((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4)),
        uchar((q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    };
}

// Extract 6-bit (scale, min) pair for sub-block index j, k from packed scales.
// Matches llama.cpp get_scale_min_k4_just2.
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

kernel void embed_q4k_f32_from_token_id(
    device const uint8_t* w_raw       [[buffer(0)]],
    device const uint*    token_ids   [[buffer(1)]],
    device       float*   output      [[buffer(2)]],
    constant     uint&    hidden_dim  [[buffer(3)]],
    constant     uint&    vocab_size  [[buffer(4)]],
    constant     uint&    token_index [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= hidden_dim) return;

    const uint token_id = token_ids[token_index];
    if (token_id >= vocab_size) {
        output[tid] = 0.0f;
        return;
    }

    const uint nb = hidden_dim / QK_K;
    const uint row_bytes = nb * 144;
    const uint block_id = tid / QK_K;
    const uint within = tid - block_id * QK_K;
    const uint group = within / 64;
    const uint rem = within - group * 64;
    const uint lane = rem & 31;
    const uint scale_idx = group * 2 + (rem >= 32 ? 1 : 0);

    device const block_q4_K * row = (device const block_q4_K *)(w_raw + token_id * row_bytes);
    device const block_q4_K * blk = row + block_id;
    const uchar2 sc_min = get_scale_min_k4_scalar(scale_idx, blk->scales);
    const uchar q = blk->qs[group * 32 + lane];
    const float qv = rem < 32 ? float(q & 0x0F) : float(q >> 4);
    output[tid] = float(blk->d) * float(sc_min.x) * qv - float(blk->dmin) * float(sc_min.y);
}

// Dequantize 16 elements of sub-block `il` (0..15) into a 4x4 half register.
// Matches llama.cpp dequantize_q4_K.
void dequantize_q4_K_fn(device const block_q4_K *xb, short il, thread half4x4 & reg) {
    device const uint8_t * q = xb->qs;

    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? (float)xb->d : (float)xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// ============================================================================
// simd_mv_q4k_f32 — GEMV for decode (n_tokens==1 or small batch).
// Direct port of llama.cpp kernel_mul_mv_q4_K_f32_impl, simplified for our
// flat memory layout: weight [out_dim, in_dim] row-major Q4_K,
//                    input  [batch, in_dim] row-major F32,
//                    output [batch, out_dim] row-major F32.
//
// Dispatch:
//   threadgroups   = [ceil(out_dim / (NSG*NR0)), batch, 1]
//   threads/tg     = [32 * NSG, 1, 1]          (NSG simdgroups of 32 threads)
// ============================================================================
constant short MV_NSG = 2;   // simdgroups per threadgroup
constant short MV_NR0 = 2;   // output rows per simdgroup (amortize y loads)

kernel void simd_mv_q4k_f32(
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

    const short ix = tiisg / 8;   // 0..3  — block stride within simdgroup
    const short it = tiisg % 8;   // 0..7
    const short iq = it / 4;      // 0 or 1
    const short ir = it % 4;      // 0..3

    const uint nb = in_dim / QK_K;

    const uint first_row = (tgpig.x * MV_NSG + sgitg) * MV_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 144;  // Q4_K block = 144 B

    // Input pointer for this batch row, offset to this thread's y-window.
    device const float * y_base = x + n * in_dim;
    device const float * y4 = y_base + ix * QK_K + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[MV_NR0] = {0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        // Load y block (32 floats total for this thread's lane).
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // For each of MV_NR0 rows this simdgroup handles:
        for (short row = 0; row < MV_NR0; ++row) {
            device const block_q4_K * blk =
                (device const block_q4_K *)(w_raw + (first_row + row) * row_bytes) + ib;

            device const uint16_t * sc = (device const uint16_t *)blk->scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)blk->qs + 16 * iq + 4 * ir;
            device const half     * dh = &blk->d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            // Shift compensations: 0x0F00 = 16 << 8 (value × 256 pre-shift,
            // so multiply by 1/256 to recover real nibble value).
            // 0xF0 = 16 × value, so multiply by 1/16 to recover.
            sumf[row] += dh[0] * ((acc1[0] + (1.f/256.f) * acc1[1]) * sc8[0] +
                                  (acc1[2] + (1.f/256.f) * acc1[3]) * sc8[1] * (1.f/16.f) +
                                  (acc2[0] + (1.f/256.f) * acc2[1]) * sc8[4] +
                                  (acc2[2] + (1.f/256.f) * acc2[3]) * sc8[5] * (1.f/16.f)) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    // Reduce across 32 threads of this simdgroup, write one output per row.
    for (short row = 0; row < MV_NR0 && first_row + row < out_dim; ++row) {
        float sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[n * out_dim + first_row + row] = sum;
        }
    }
}

// Same arithmetic as simd_mv_q4k_f32, but folds the final residual add into
// the output write. Used by the decode FFN-down path to remove one add kernel.
kernel void simd_mv_q4k_f32_add(
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
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const uint nb = in_dim / QK_K;

    const uint first_row = (tgpig.x * MV_NSG + sgitg) * MV_NR0;
    const uint n = tgpig.y;
    if (first_row >= out_dim || n >= batch) return;

    const uint row_bytes = nb * 144;

    device const float * y_base = x + n * in_dim;
    device const float * y4 = y_base + ix * QK_K + 64 * iq + 8 * ir;

    float yl[16];
    float yh[16];
    float sumf[MV_NR0] = {0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < MV_NR0; ++row) {
            device const block_q4_K * blk =
                (device const block_q4_K *)(w_raw + (first_row + row) * row_bytes) + ib;

            device const uint16_t * sc = (device const uint16_t *)blk->scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)blk->qs + 16 * iq + 4 * ir;
            device const half     * dh = &blk->d;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const uint16_t * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + (1.f/256.f) * acc1[1]) * sc8[0] +
                                  (acc1[2] + (1.f/256.f) * acc1[3]) * sc8[1] * (1.f/16.f) +
                                  (acc2[0] + (1.f/256.f) * acc2[1]) * sc8[4] +
                                  (acc2[2] + (1.f/256.f) * acc2[3]) * sc8[5] * (1.f/16.f)) -
                         dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                  sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < MV_NR0 && first_row + row < out_dim; ++row) {
        float sum = simd_sum(sumf[row]);
        if (tiisg == 0) {
            const uint out_idx = n * out_dim + first_row + row;
            output[out_idx] = residual[out_idx] + sum;
        }
    }
}

// ============================================================================
// simd_mm_q4k_f32 — GEMM for prefill (batch>1).
// Adapted from our simd_mm_q5k. F32 input + F32 output, no bias, no GELU.
// Double-buffered tile load (1 barrier per iter instead of 2).
// Cooperative output write (128 threads vs 32).
//
// Output tile: 64 rows × 32 batch elements.
// Dispatch:
//   threadgroups   = [ceil(batch/32), ceil(out_dim/64), 1]
//   threads/tg     = [128, 1, 1]     (4 simdgroups)
//   threadgroup_memory = 16384 B     (double buffer, 6144 B per tile,
//                                     reused for output scratch)
// ============================================================================
constant int MM_NR0 = 64;
constant int MM_NR1 = 32;
constant int MM_NK  = 32;
constant int MM_NL0 = 2;     // NK/16
constant int MM_NL1 = 4;     // NK/8
constant int MM_NL  = 16;    // QK_NL — sub-blocks per super-block (256/16)

constant int MM_SA_SIZE   = 4096;                     // per-tile weight shmem (fp16)
constant int MM_SB_SIZE   = 2048;                     // per-tile input shmem  (fp16)
constant int MM_TILE_SIZE = MM_SA_SIZE + MM_SB_SIZE;  // 6144 B per tile

kernel void simd_mm_q4k_f32(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Double-buffered shmem: tile 0 and tile 1 alternate
    threadgroup half * sa_buf[2] = {
        (threadgroup half *)(shmem),
        (threadgroup half *)(shmem + MM_TILE_SIZE)
    };
    threadgroup half * sb_buf[2] = {
        (threadgroup half *)(shmem + MM_SA_SIZE),
        (threadgroup half *)(shmem + MM_TILE_SIZE + MM_SA_SIZE)
    };

    const int r0 = tgpig.y * MM_NR0;  // first output row in this tile
    const int r1 = tgpig.x * MM_NR1;  // first batch row in this tile

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, (int)batch   - r1);

    const short lr0 = min((short)(tiitg / MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / MM_NL1), (short)(nr1 - 1));

    const short il0 = tiitg % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 144;
    device const block_q4_K * xw =
        (device const block_q4_K *)(w_raw + (r0 + lr0) * row_bytes) + il0 / MM_NL;

    const short iy = 8 * (tiitg % MM_NL1);
    device const float * y = x + (r1 + lr1) * in_dim + iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Preload first tile into buffer 0.
    {
        threadgroup half * sa = sa_buf[0];
        threadgroup half * sb = sb_buf[0];

        half4x4 temp_a;
        dequantize_q4_K_fn(xw, il, temp_a);
        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/MM_NL0)/8;
            const short lx = (tiitg/MM_NL0)%8;
            const short ly = i%8;
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
        }
        {
            // Convert 8 F32 inputs to half on load into shmem.
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            threadgroup half * dst = sb + 64*(4*sx + sy) + 8*ly;
            FOR_UNROLL for (short i = 0; i < 8; i++) {
                dst[i] = (half)y[i];
            }
        }
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
        y += MM_NK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        short cur = iter % 2;
        short nxt = 1 - cur;
        threadgroup half * sa = sa_buf[cur];
        threadgroup half * sb = sb_buf[cur];

        // Compute from current buffer
        threadgroup const half * lsma = sa + 4*64*(sgitg % 2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg / 2);
        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            lsma += 8*64; lsmb += 4*64;
        }

        // Load next tile into alternate buffer (overlap with compute)
        if (iter + 1 < n_iter) {
            threadgroup half * sa_n = sa_buf[nxt];
            threadgroup half * sb_n = sb_buf[nxt];
            half4x4 temp_a;
            dequantize_q4_K_fn(xw, il, temp_a);
            FOR_UNROLL for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/MM_NL0)/8;
                const short lx = (tiitg/MM_NL0)%8;
                const short ly = i%8;
                *(sa_n + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
            }
            {
                const short sx = (tiitg % MM_NL1);
                const short sy = (tiitg/MM_NL1)/8;
                const short ly = (tiitg/MM_NL1)%8;
                threadgroup half * dst = sb_n + 64*(4*sx + sy) + 8*ly;
                FOR_UNROLL for (short i = 0; i < 8; i++) {
                    dst[i] = (half)y[i];
                }
            }
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
            y += MM_NK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Most Qwen prefill shapes are exact 64x32 tiles. For those, write the
    // simdgroup accumulators directly to device memory and avoid the extra
    // shmem staging barrier. Keep the cooperative path for edge tiles.
    if (nr0 == MM_NR0 && nr1 == MM_NR1) {
        device float * C = output + (r0 + 32*(sgitg & 1)) + (r1 + 16*(sgitg >> 1))*out_dim;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
        }
        return;
    }

    // Stage edge-tile accumulators to shmem, then cooperative F32 output write.
    threadgroup float * temp = (threadgroup float *)shmem;
    {
        threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ALL 128 threads cooperatively write output.
    const int total_out = nr0 * nr1;
    for (int idx = (int)tiitg; idx < total_out; idx += 128) {
        const int i = idx % nr0;
        const int j = idx / nr0;
        output[(r1 + j) * out_dim + r0 + i] = temp[j * MM_NR0 + i];
    }
}

// ============================================================================
// simd_mm_q4k_h16 — same GEMM as simd_mm_q4k_f32, but with pre-converted
// F16 input. This is exact for the current F32 kernel because it also rounds
// F32 inputs to half before simdgroup MMA, but does so per output tile.
// ============================================================================
kernel void simd_mm_q4k_h16(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const half*    x       [[buffer(1)]],
    device       float*   output  [[buffer(2)]],
    constant     uint&    in_dim  [[buffer(3)]],
    constant     uint&    out_dim [[buffer(4)]],
    constant     uint&    batch   [[buffer(5)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Double-buffered shmem: tile 0 and tile 1 alternate
    threadgroup half * sa_buf[2] = {
        (threadgroup half *)(shmem),
        (threadgroup half *)(shmem + MM_TILE_SIZE)
    };
    threadgroup half * sb_buf[2] = {
        (threadgroup half *)(shmem + MM_SA_SIZE),
        (threadgroup half *)(shmem + MM_TILE_SIZE + MM_SA_SIZE)
    };

    const int r0 = tgpig.y * MM_NR0;  // first output row in this tile
    const int r1 = tgpig.x * MM_NR1;  // first batch row in this tile

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, (int)batch   - r1);

    const short lr0 = min((short)(tiitg / MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg / MM_NL1), (short)(nr1 - 1));

    const short il0 = tiitg % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 144;
    device const block_q4_K * xw =
        (device const block_q4_K *)(w_raw + (r0 + lr0) * row_bytes) + il0 / MM_NL;

    const short iy = 8 * (tiitg % MM_NL1);
    device const half * y = x + (r1 + lr1) * in_dim + iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Preload first tile into buffer 0.
    {
        threadgroup half * sa = sa_buf[0];
        threadgroup half * sb = sb_buf[0];

        half4x4 temp_a;
        dequantize_q4_K_fn(xw, il, temp_a);
        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/MM_NL0)/8;
            const short lx = (tiitg/MM_NL0)%8;
            const short ly = i%8;
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
        }
        {
            // Input is pre-converted to F16 once per matmul.
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            threadgroup half * dst = sb + 64*(4*sx + sy) + 8*ly;
            *(threadgroup half2x4 *)dst = *(device const half2x4 *)y;
        }
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
        y += MM_NK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        short cur = iter % 2;
        short nxt = 1 - cur;
        threadgroup half * sa = sa_buf[cur];
        threadgroup half * sb = sb_buf[cur];

        // Compute from current buffer
        threadgroup const half * lsma = sa + 4*64*(sgitg % 2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg / 2);
        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            lsma += 8*64; lsmb += 4*64;
        }

        // Load next tile into alternate buffer (overlap with compute)
        if (iter + 1 < n_iter) {
            threadgroup half * sa_n = sa_buf[nxt];
            threadgroup half * sb_n = sb_buf[nxt];
            half4x4 temp_a;
            dequantize_q4_K_fn(xw, il, temp_a);
            FOR_UNROLL for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/MM_NL0)/8;
                const short lx = (tiitg/MM_NL0)%8;
                const short ly = i%8;
                *(sa_n + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
            }
            {
                const short sx = (tiitg % MM_NL1);
                const short sy = (tiitg/MM_NL1)/8;
                const short ly = (tiitg/MM_NL1)%8;
                threadgroup half * dst = sb_n + 64*(4*sx + sy) + 8*ly;
                *(threadgroup half2x4 *)dst = *(device const half2x4 *)y;
            }
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
            y += MM_NK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Most Qwen prefill shapes are exact 64x32 tiles. For those, write the
    // simdgroup accumulators directly to device memory and avoid the extra
    // shmem staging barrier. Keep the cooperative path for edge tiles.
    if (nr0 == MM_NR0 && nr1 == MM_NR1) {
        device float * C = output + (r0 + 32*(sgitg & 1)) + (r1 + 16*(sgitg >> 1))*out_dim;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
        }
        return;
    }

    // Stage edge-tile accumulators to shmem, then cooperative F32 output write.
    threadgroup float * temp = (threadgroup float *)shmem;
    {
        threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ALL 128 threads cooperatively write output.
    const int total_out = nr0 * nr1;
    for (int idx = (int)tiitg; idx < total_out; idx += 128) {
        const int i = idx % nr0;
        const int j = idx / nr0;
        output[(r1 + j) * out_dim + r0 + i] = temp[j * MM_NR0 + i];
    }
}
