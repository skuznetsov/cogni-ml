// FP16×FP16 Matrix-Matrix GEMM using simdgroup_matrix_multiply_accumulate
// Pre-dequantized weights — no Q5K/Q6K dequant in the hot loop.
//
// Each threadgroup computes a 64×32 output tile.
// Weights are FP16 in row-major layout: w[out_dim, in_dim]
// Input is FP16: x[batch, in_dim]
// Output is FP16: output[batch, out_dim] (with F32 bias + optional GELU)
//
// Dispatch: threadgroups = [ceil(batch/32), ceil(out_dim/64), 1]
//           threads_per_threadgroup = [128, 1, 1]
//           threadgroup_memory = 8192 bytes

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define FOR_UNROLL _Pragma("clang loop unroll(full)")

constant int MM_NR0 = 64;   // output rows per threadgroup
constant int MM_NR1 = 32;   // batch elements per threadgroup
constant int MM_NK  = 32;   // K elements per iteration

kernel void simd_mm_f16(
    device const half*   w       [[buffer(0)]],  // pre-dequantized FP16 [out_dim, in_dim]
    device const half*   x       [[buffer(1)]],  // FP16 input [batch, in_dim]
    device const float*  bias    [[buffer(2)]],  // F32 bias [out_dim]
    device       half*   output  [[buffer(3)]],  // FP16 output [batch, out_dim]
    constant     uint&   in_dim  [[buffer(4)]],
    constant     uint&   out_dim [[buffer(5)]],
    constant     uint&   batch   [[buffer(6)]],
    constant     uint&   apply_gelu [[buffer(7)]],
    threadgroup  char*   shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // Shared memory: sa for weights [NR0, NK], sb for input [NR1, NK]
    // Layout: 64 elements per row (8-aligned for simdgroup_load)
    threadgroup half * sa = (threadgroup half *)(shmem);          // 64*64 = 4096 half = 8192 bytes? No.
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);   // 4096 bytes for B

    const int r0 = tgpig.y * MM_NR0;  // first output row
    const int r1 = tgpig.x * MM_NR1;  // first batch element

    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, (int)batch - r1);

    // Thread decomposition for loading
    // 128 threads load NR0×NK = 64×32 = 2048 weight elements (16 per thread)
    // And NR1×NK = 32×32 = 1024 input elements (8 per thread)
    const short lr0 = min((short)(tiitg / 2), (short)(nr0 - 1));   // weight row (0..63)
    const short lr1 = min((short)(tiitg / 4), (short)(nr1 - 1));   // input row (0..31)

    // Weight pointer for this thread's row
    device const half * w_row = w + (r0 + lr0) * in_dim;
    // Input pointer for this thread's row
    device const half * x_row = x + (r1 + lr1) * in_dim;

    // Simdgroup matrices
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    // Main K-loop
    for (uint k = 0; k < in_dim; k += MM_NK) {
        // Load weight tile [NR0, NK] into sa
        // 128 threads, each loads 16 elements (2 rows × 8 elements)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const short row_in_tile = tiitg / 2;     // 0..63
            const short col_start = (tiitg % 2) * 16; // 0 or 16
            if (row_in_tile < nr0) {
                device const half * src = w + (r0 + row_in_tile) * in_dim + k + col_start;
                threadgroup half * dst = sa + 64 * (row_in_tile / 8) + 8 * (row_in_tile % 8);
                // Store in simdgroup-friendly layout: 64 * (row/8) + 8 * (row%8) + col_block*64
                // Actually, use the same layout as llama.cpp for simdgroup_load compatibility
                for (short j = 0; j < 16; j++) {
                    short sx = col_start/8 + j/8;  // 0..3 (K block of 8)
                    short sy = row_in_tile / 8;     // 0..7 (row block of 8)
                    short lx = row_in_tile % 8;     // 0..7 (row within block)
                    short ly = j % 8;               // 0..7 (col within K-block of 8)
                    short ib = 8 * sx + sy;
                    *(sa + 64*ib + 8*ly + lx) = (k + col_start + j < in_dim) ? src[j] : half(0);
                }
            }
        }

        // Load input tile [NR1, NK] into sb
        {
            const short row_in_tile = tiitg / 4;     // 0..31
            const short col_start = (tiitg % 4) * 8;  // 0, 8, 16, 24
            if (row_in_tile < nr1) {
                device const half * src = x + (r1 + row_in_tile) * in_dim + k + col_start;
                short sx = col_start / 8;       // 0..3
                short sy = row_in_tile / 8;     // 0..3
                short ly = row_in_tile % 8;     // 0..7
                short ib = 4*sx + sy;
                *(threadgroup half2x4 *)(sb + 64*ib + 8*ly) = (k + col_start < in_dim) ?
                    *(device const half2x4 *)src : half2x4(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: 4 simdgroups, each handles a 32×16 quadrant of the 64×32 tile
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

    // Store results to shared memory, apply bias + GELU, write FP16 output
    threadgroup float * temp = (threadgroup float *)shmem;
    threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
    for (short i = 0; i < 8; i++) {
        simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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

// MoE variant — reads expert_offsets from GPU for base pointer computation
kernel void simd_mm_f16_moe(
    device const half*   w           [[buffer(0)]],  // weights for THIS expert
    device const half*   x_packed    [[buffer(1)]],  // full packed input
    device const float*  bias        [[buffer(2)]],
    device       half*   out_packed  [[buffer(3)]],  // full packed output
    device const int*    expert_offs [[buffer(4)]],
    constant     uint&   expert_id   [[buffer(5)]],
    constant     uint&   in_dim      [[buffer(6)]],
    constant     uint&   out_dim     [[buffer(7)]],
    constant     uint&   apply_gelu  [[buffer(8)]],
    threadgroup  char*   shmem       [[threadgroup(0)]],
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

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    for (uint k = 0; k < in_dim; k += MM_NK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load weight tile
        {
            const short row_in_tile = tiitg / 2;
            const short col_start = (tiitg % 2) * 16;
            if (row_in_tile < nr0) {
                device const half * src = w + (r0 + row_in_tile) * in_dim + k + col_start;
                for (short j = 0; j < 16; j++) {
                    short sx = col_start/8 + j/8;
                    short sy = row_in_tile / 8;
                    short lx = row_in_tile % 8;
                    short ly = j % 8;
                    *(sa + 64*(8*sx + sy) + 8*ly + lx) = (k + col_start + j < in_dim) ? src[j] : half(0);
                }
            }
        }

        // Load input tile (from packed buffer at expert base offset)
        {
            const short row_in_tile = tiitg / 4;
            const short col_start = (tiitg % 4) * 8;
            if (row_in_tile < nr1) {
                device const half * src = x_packed + (base + r1 + row_in_tile) * in_dim + k + col_start;
                short sx = col_start / 8;
                short sy = row_in_tile / 8;
                short ly = row_in_tile % 8;
                *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = (k + col_start < in_dim) ?
                    *(device const half2x4 *)src : half2x4(0);
            }
        }

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
