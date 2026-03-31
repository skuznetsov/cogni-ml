// Matrix-matrix GEMM for Q5_K / Q6_K using simdgroup_matrix_multiply_accumulate
// Adapted from llama.cpp's kernel_mul_mm — hardware-accelerated 8×8 tiles
//
// Each threadgroup computes a 64×32 output tile (64 output rows × 32 batch elements)
// using 4 simdgroups (128 threads), dequantizing weights to FP16 shared memory.
//
// For batch > ~8, this is ~5× faster than the scalar SIMD-group GEMM.
//
// Dispatch: threadgroups = [ceil(batch/32), ceil(out_dim/64), 1]
//           threads_per_threadgroup = [128, 1, 1]
//           threadgroup_memory = 8192 bytes

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

constant uint QK_K = 256;

// ============================================================================
// Q5_K dequantization: 16 elements from one sub-block → 4×4 half register
// Block layout: [d:2B][dmin:2B][scales:12B][qh:32B][qs:128B] = 176 bytes
// il = 0..15 selects which 16-element chunk of the 256-element super-block
// ============================================================================
struct block_q5_K {
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                           uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

void dequantize_q5_K_fn(device const block_q5_K *xb, short il, thread half4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;

    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, xb->scales);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask  = il<2 ? 0x0F : 0xF0;
    const float qh_val = il<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
    }
}

// ============================================================================
// Q6_K dequantization: 16 elements from one sub-block → 4×4 half register
// Block layout: [ql:128B][qh:64B][scales:16B][d:2B] = 210 bytes
// ============================================================================
struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    half d;
};

void dequantize_q6_K_fn(device const block_q6_K *xb, short il, thread half4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml  = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 *  ((half)(q & 0xFF))       - ml;
        reg[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
}

// ============================================================================
// Shared matmul core: dequant to shared mem → simdgroup_matrix_multiply_accumulate
// ============================================================================
constant int MM_NR0 = 64;   // output rows per threadgroup
constant int MM_NR1 = 32;   // batch elements per threadgroup
constant int MM_NK  = 32;   // K elements per iteration
constant int MM_NL0 = 2;    // NK/16 — threads sharing weight dequantization
constant int MM_NL1 = 4;    // NK/8  — threads sharing input loading
constant int MM_NL  = 16;   // QK_NL — sub-blocks per super-block (256/16)
// Double-buffered shared memory: sa[0]/sb[0] and sa[1]/sb[1]
// Eliminates 1 of 2 threadgroup_barriers per K-iteration (50% fewer inner barriers)
constant int MM_SA_SIZE = 4096;  // bytes per weight tile in shmem
constant int MM_SB_SIZE = 2048;  // bytes per input tile in shmem
constant int MM_TILE_SIZE = MM_SA_SIZE + MM_SB_SIZE;  // 6144 bytes per tile

// Q5_K matrix-matrix multiply with simdgroup_matrix
kernel void simd_mm_q5k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const half*    x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       half*    output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Double-buffered shared memory: tile 0 and tile 1 alternate
    threadgroup half * sa_buf[2] = {
        (threadgroup half *)(shmem),
        (threadgroup half *)(shmem + MM_TILE_SIZE)
    };
    threadgroup half * sb_buf[2] = {
        (threadgroup half *)(shmem + MM_SA_SIZE),
        (threadgroup half *)(shmem + MM_TILE_SIZE + MM_SA_SIZE)
    };

    const int r0 = tgpig.y * MM_NR0;
    const int r1 = tgpig.x * MM_NR1;

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, (int)batch - r1);

    const short lr0 = min((short)(tiitg/MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg/MM_NL1), (short)(nr1 - 1));

    const short il0 = tiitg % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 176;
    device const block_q5_K * xw = (device const block_q5_K *)(w_raw + (r0 + lr0) * row_bytes) + il0 / MM_NL;

    const short iy = 8 * (tiitg % MM_NL1);
    device const half * y = x + (r1 + lr1) * in_dim + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Load first tile into buffer 0
    {
        threadgroup half * sa = sa_buf[0];
        threadgroup half * sb = sb_buf[0];
        half4x4 temp_a;
        dequantize_q5_K_fn(xw, il, temp_a);
        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/MM_NL0)/8;
            const short lx = (tiitg/MM_NL0)%8;
            const short ly = i%8;
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
        }
        {
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = *(device const half2x4 *)y;
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

        // Load next tile into alternate buffer (overlap with compute above on next iteration)
        if (iter + 1 < n_iter) {
            threadgroup half * sa_n = sa_buf[nxt];
            threadgroup half * sb_n = sb_buf[nxt];
            half4x4 temp_a;
            dequantize_q5_K_fn(xw, il, temp_a);
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
                *(threadgroup half2x4 *)(sb_n + 64*(4*sx + sy) + 8*ly) = *(device const half2x4 *)y;
            }
            il = (il + 2 < MM_NL) ? il + 2 : il % 2;
            xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
            y += MM_NK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // ONE barrier (was TWO)
    }

    // Write output: ALL 128 threads participate (4× faster than sgitg==0 only)
    // Store accumulators to shared, barrier, then cooperative bias+GELU+write
    threadgroup float * temp = (threadgroup float *)shmem;
    {
        threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ALL 128 threads cooperatively write output with bias + optional GELU
    // nr0/nr1 already computed above
    // Total elements: nr0 * nr1 ≤ 64*32 = 2048. Each of 128 threads handles ~16 elements.
    const int total_out = nr0 * nr1;
    for (int idx = (int)tiitg; idx < total_out; idx += 128) {
        const int i = idx % nr0;  // output row within tile
        const int j = idx / nr0;  // batch element within tile
        float val = temp[j * MM_NR0 + i] + bias[r0 + i];
        if (apply_gelu) {
            if (val > 10.0f) { }
            else if (val < -10.0f) { val = 0.0f; }
            else {
                float t = 0.7978845608f * (val + 0.044715f * val * val * val);
                val = 0.5f * val * (1.0f + tanh(t));
            }
        }
        output[(r1 + j) * out_dim + r0 + i] = half(val);
    }

}

// Q6_K matrix-matrix multiply with simdgroup_matrix
kernel void simd_mm_q6k(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const half*    x       [[buffer(1)]],
    device const float*   bias    [[buffer(2)]],
    device       half*    output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    constant     uint&    apply_gelu [[buffer(7)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * MM_NR0;
    const int r1 = tgpig.x * MM_NR1;

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, (int)batch - r1);

    const short lr0 = min((short)(tiitg/MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg/MM_NL1), (short)(nr1 - 1));

    const short il0 = tiitg % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 210;
    const short offset1 = il0 / MM_NL;

    device const block_q6_K * xw = (device const block_q6_K *)(w_raw + (r0 + lr0) * row_bytes) + offset1;

    const short iy = 8 * (tiitg % MM_NL1);
    device const half * y = x + (r1 + lr1) * in_dim + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    for (uint loop_k = 0; loop_k < in_dim; loop_k += MM_NK) {
        half4x4 temp_a;
        dequantize_q6_K_fn(xw, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/MM_NL0)/8;
            const short lx = (tiitg/MM_NL0)%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
        }

        {
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            const short ib = 4*sx + sy;
            *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = *(device const half2x4 *)y;
        }

        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
        y += MM_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg % 2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg / 2);

        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 4; i++) {
                simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 2; i++) {
                simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // Write output with bias + optional GELU
    threadgroup float * temp = (threadgroup float *)shmem;
    threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ALL 128 threads cooperatively write output with bias + GELU
    {
        const int total_out = nr0 * nr1;
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            float val = temp[j * MM_NR0 + i] + bias[r0 + i];
            if (apply_gelu) {
                if (val > 10.0f) { }
                else if (val < -10.0f) { val = 0.0f; }
                else {
                    float t = 0.7978845608f * (val + 0.044715f * val * val * val);
                    val = 0.5f * val * (1.0f + tanh(t));
                }
            }
            output[(r1 + j) * out_dim + r0 + i] = half(val);
        }
    }
}

// ============================================================================
// MoE variants of matrix-matrix GEMM — GPU-side expert_offsets, zero CPU sync
// Same simdgroup_matrix approach, but input/output are packed by expert.
// Dispatch: indirect with grid = [ceil(eb/32), ceil(out_dim/64), 1]
// ============================================================================

kernel void simd_mm_q5k_moe(
    device const uint8_t* w_raw        [[buffer(0)]],  // weights for THIS expert
    device const half*    x_packed     [[buffer(1)]],  // full packed input
    device const float*   bias         [[buffer(2)]],
    device       half*    out_packed   [[buffer(3)]],  // full packed output
    device const int*     expert_offs  [[buffer(4)]],
    constant     uint&    expert_id    [[buffer(5)]],
    constant     uint&    in_dim       [[buffer(6)]],
    constant     uint&    out_dim      [[buffer(7)]],
    constant     uint&    apply_gelu   [[buffer(8)]],
    threadgroup  char*    shmem        [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int base = expert_offs[expert_id];
    const int eb = expert_offs[expert_id + 1] - base;
    if (eb <= 0) return;

    const int r0 = tgpig.y * MM_NR0;  // output row tile
    const int r1 = tgpig.x * MM_NR1;  // batch tile within this expert

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, eb - r1);
    if (nr0 <= 0 || nr1 <= 0) return;

    const short lr0 = min((short)(tiitg/MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg/MM_NL1), (short)(nr1 - 1));

    const short il0 = tiitg % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 176;
    device const block_q5_K * xw = (device const block_q5_K *)(w_raw + (r0 + lr0) * row_bytes) + il0/MM_NL;

    const short iy = 8 * (tiitg % MM_NL1);
    device const half * y = x_packed + (base + r1 + lr1) * in_dim + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint loop_k = 0; loop_k < in_dim; loop_k += MM_NK) {
        half4x4 temp_a;
        dequantize_q5_K_fn(xw, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/MM_NL0)/8;
            const short lx = (tiitg/MM_NL0)%8;
            const short ly = i%8;
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
        }
        {
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = *(device const half2x4 *)y;
        }
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
        y += MM_NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    }

    threadgroup float * temp = (threadgroup float *)shmem;
    threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int total_out = nr0 * nr1;
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            float val = temp[j * MM_NR0 + i] + bias[r0 + i];
            if (apply_gelu) {
                if (val > 10.0f) {} else if (val < -10.0f) { val = 0.0f; }
                else { float t = 0.7978845608f * (val + 0.044715f * val * val * val); val = 0.5f * val * (1.0f + tanh(t)); }
            }
            out_packed[(base + r1 + j) * out_dim + r0 + i] = half(val);
        }
    }
}

kernel void simd_mm_q6k_moe(
    device const uint8_t* w_raw        [[buffer(0)]],
    device const half*    x_packed     [[buffer(1)]],
    device const float*   bias         [[buffer(2)]],
    device       half*    out_packed   [[buffer(3)]],
    device const int*     expert_offs  [[buffer(4)]],
    constant     uint&    expert_id    [[buffer(5)]],
    constant     uint&    in_dim       [[buffer(6)]],
    constant     uint&    out_dim      [[buffer(7)]],
    constant     uint&    apply_gelu   [[buffer(8)]],
    threadgroup  char*    shmem        [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int base = expert_offs[expert_id];
    const int eb = expert_offs[expert_id + 1] - base;
    if (eb <= 0) return;

    const int r0 = tgpig.y * MM_NR0;
    const int r1 = tgpig.x * MM_NR1;

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, eb - r1);
    if (nr0 <= 0 || nr1 <= 0) return;

    const short lr0 = min((short)(tiitg/MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg/MM_NL1), (short)(nr1 - 1));

    const short il0 = tiitg % MM_NL0;
    short il = il0;

    const uint row_bytes = (in_dim / QK_K) * 210;
    device const block_q6_K * xw = (device const block_q6_K *)(w_raw + (r0 + lr0) * row_bytes) + il0/MM_NL;

    const short iy = 8 * (tiitg % MM_NL1);
    device const half * y = x_packed + (base + r1 + lr1) * in_dim + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint loop_k = 0; loop_k < in_dim; loop_k += MM_NK) {
        half4x4 temp_a;
        dequantize_q6_K_fn(xw, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = (tiitg/MM_NL0)/8;
            const short lx = (tiitg/MM_NL0)%8;
            const short ly = i%8;
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
        }
        {
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = *(device const half2x4 *)y;
        }
        il = (il + 2 < MM_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw;
        y += MM_NK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    }

    threadgroup float * temp = (threadgroup float *)shmem;
    threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
    for (short i = 0; i < 8; i++) simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const int total_out = nr0 * nr1;
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            float val = temp[j * MM_NR0 + i] + bias[r0 + i];
            if (apply_gelu) {
                if (val > 10.0f) {} else if (val < -10.0f) { val = 0.0f; }
                else { float t = 0.7978845608f * (val + 0.044715f * val * val * val); val = 0.5f * val * (1.0f + tanh(t)); }
            }
            out_packed[(base + r1 + j) * out_dim + r0 + i] = half(val);
        }
    }
}

// ============================================================================
// Batched expert GEMM — ALL experts in ONE dispatch (LTP Diamond surgery)
// tgpig.x = flattened batch tiles across ALL experts
// tgpig.y = output row tile
// Binary search expert_tg_offsets to find expert_id from tgpig.x
// Grid: {sum_of_all_expert_batch_tgs, ceil(out_dim/64), 1}
// ============================================================================

inline int find_expert(device const int* tg_offsets, int flat_x, int n) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (tg_offsets[mid] <= flat_x) lo = mid; else hi = mid - 1;
    }
    return lo;
}

#define BATCHED_MM_BODY(BLOCK_T, BLOCK_BYTES, DEQUANT_FN) \
    threadgroup half * sa = (threadgroup half *)(shmem); \
    threadgroup half * sb = (threadgroup half *)(shmem + 4096); \
    const int eid = find_expert(expert_tg_offs, (int)tgpig.x, (int)n_experts); \
    const int local_x = (int)tgpig.x - expert_tg_offs[eid]; \
    const int base = expert_offs[eid]; \
    const int eb = expert_offs[eid + 1] - base; \
    if (eb <= 0) return; \
    const int r0 = tgpig.y * MM_NR0; \
    const int r1 = local_x * MM_NR1; \
    const short nr0 = min(MM_NR0, (int)out_dim - r0); \
    const short nr1 = min(MM_NR1, eb - r1); \
    if (nr0 <= 0 || nr1 <= 0) return; \
    const short lr0 = min((short)(tiitg/MM_NL0), (short)(nr0 - 1)); \
    const short lr1 = min((short)(tiitg/MM_NL1), (short)(nr1 - 1)); \
    const short il0 = tiitg % MM_NL0; \
    short il = il0; \
    const uint row_bytes = (in_dim / QK_K) * BLOCK_BYTES; \
    device const BLOCK_T * xw = (device const BLOCK_T *)(all_weights + eid * weight_stride + (r0 + lr0) * row_bytes) + il0/MM_NL; \
    const short iy = 8 * (tiitg % MM_NL1); \
    device const half * y = x_packed + (base + r1 + lr1) * in_dim + iy; \
    simdgroup_half8x8 ma[4]; simdgroup_half8x8 mb[2]; simdgroup_float8x8 mc[8]; \
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f); \
    for (uint loop_k = 0; loop_k < in_dim; loop_k += MM_NK) { \
        half4x4 temp_a; DEQUANT_FN(xw, il, temp_a); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        FOR_UNROLL for (short i = 0; i < 16; i++) { \
            const short sx = 2*il0 + i/8, sy = (tiitg/MM_NL0)/8, lx = (tiitg/MM_NL0)%8, ly = i%8; \
            *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4]; } \
        { const short sx = tiitg%MM_NL1, sy = (tiitg/MM_NL1)/8, ly = (tiitg/MM_NL1)%8; \
          *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = *(device const half2x4 *)y; } \
        il = (il + 2 < MM_NL) ? il + 2 : il % 2; \
        xw = (il < 2) ? xw + (2 + MM_NL - 1)/MM_NL : xw; y += MM_NK; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        threadgroup const half * lsma = sa + 4*64*(sgitg%2), * lsmb = sb + 2*64*(sgitg/2); \
        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) { \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false); \
            simdgroup_barrier(mem_flags::mem_none); \
            FOR_UNROLL for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]); \
            lsma += 8*64; lsmb += 4*64; } \
    } \
    threadgroup float * temp = (threadgroup float *)shmem; \
    { threadgroup float * sg_out = temp + 32*(sgitg&1) + 16*(sgitg>>1)*MM_NR0; \
      for (short i = 0; i < 8; i++) simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false); } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    { const int total_out = nr0 * nr1; \
      for (int idx = (int)tiitg; idx < total_out; idx += 128) { \
          const int i = idx%nr0, j = idx/nr0; \
          float val = temp[j*MM_NR0 + i] + bias[r0 + i]; \
          if (apply_gelu) { if (val > 10.0f) {} else if (val < -10.0f) { val = 0.0f; } \
            else { float t = 0.7978845608f*(val + 0.044715f*val*val*val); val = 0.5f*val*(1.0f + tanh(t)); } } \
          out_packed[(base + r1 + j)*out_dim + r0 + i] = half(val); } }

kernel void batched_mm_q5k(
    device const uint8_t* all_weights    [[buffer(0)]],
    device const half*    x_packed       [[buffer(1)]],
    device const float*   bias           [[buffer(2)]],
    device       half*    out_packed     [[buffer(3)]],
    device const int*     expert_offs    [[buffer(4)]],
    device const int*     expert_tg_offs [[buffer(5)]],
    constant     uint&    in_dim         [[buffer(6)]],
    constant     uint&    out_dim        [[buffer(7)]],
    constant     uint&    apply_gelu     [[buffer(8)]],
    constant     uint&    n_experts      [[buffer(9)]],
    constant     uint&    weight_stride  [[buffer(10)]],
    threadgroup  char*    shmem          [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{ BATCHED_MM_BODY(block_q5_K, 176, dequantize_q5_K_fn) }

kernel void batched_mm_q6k(
    device const uint8_t* all_weights    [[buffer(0)]],
    device const half*    x_packed       [[buffer(1)]],
    device const float*   bias           [[buffer(2)]],
    device       half*    out_packed     [[buffer(3)]],
    device const int*     expert_offs    [[buffer(4)]],
    device const int*     expert_tg_offs [[buffer(5)]],
    constant     uint&    in_dim         [[buffer(6)]],
    constant     uint&    out_dim        [[buffer(7)]],
    constant     uint&    apply_gelu     [[buffer(8)]],
    constant     uint&    n_experts      [[buffer(9)]],
    constant     uint&    weight_stride  [[buffer(10)]],
    threadgroup  char*    shmem          [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{ BATCHED_MM_BODY(block_q6_K, 210, dequantize_q6_K_fn) }
