# CUDA Q4_K repacked-layout microbench.
#
# This probe keeps model quality exact at the quantized-weight level, but changes
# storage layout offline: per-weight 4-bit nibbles are expanded to one byte and
# packed scale/min metadata is pre-expanded to f32. Runtime GEMV then avoids GGUF
# scale bit extraction and nibble unpacking. It is intentionally a memory/speed
# trade-off probe, not a production route yet.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/gguf/dequant"
require "../src/ml/cuda/driver"

DEFAULT_MODEL  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TENSOR = "blk.0.ffn_up.weight"
Q4K_PTX = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q4K_SCALE_REGS_PTX = Q4K_PTX
  .gsub("q4_k_gemv_warp4_f32", "q4_k_gemv_scale_regs_warp4_f32")
  .gsub(<<-PTX, <<-PTX)
    add.s64 %rd9, %rd7, 4;      // scales base
    add.s64 %rd10, %rd7, 16;    // qs base

    mov.u32 %r13, 0;            // group 0..3
PTX
    add.s64 %rd9, %rd7, 4;      // scales base
    add.s64 %rd10, %rd7, 16;    // qs base
    ld.global.u32 %r70, [%rd9]; // scale bytes 0..3
    add.s64 %rd68, %rd9, 4;
    ld.global.u32 %r71, [%rd68]; // scale bytes 4..7
    add.s64 %rd69, %rd9, 8;
    ld.global.u32 %r72, [%rd69]; // scale bytes 8..11

    mov.u32 %r13, 0;            // group 0..3
PTX
  .gsub(<<-PTX, <<-PTX)
WARP_SCALE1_HIGH:
    add.u32 %r15, %r14, 4;
    cvt.u64.u32 %rd11, %r15;
    add.s64 %rd12, %rd9, %rd11;
    ld.global.u8 %r16, [%rd12];
    and.b32 %r20, %r16, 15;
    add.u32 %r17, %r14, -4;
    cvt.u64.u32 %rd13, %r17;
    add.s64 %rd14, %rd9, %rd13;
    ld.global.u8 %r18, [%rd14];
    shr.u32 %r19, %r18, 6;
    shl.b32 %r19, %r19, 4;
    or.b32 %r20, %r20, %r19;
    shr.u32 %r21, %r16, 4;
    cvt.u64.u32 %rd15, %r14;
    add.s64 %rd16, %rd9, %rd15;
    ld.global.u8 %r22, [%rd16];
    shr.u32 %r23, %r22, 6;
    shl.b32 %r23, %r23, 4;
    or.b32 %r21, %r21, %r23;
    bra WARP_SCALE1_DONE;

WARP_SCALE1_LOW:
    cvt.u64.u32 %rd17, %r14;
    add.s64 %rd18, %rd9, %rd17;
    ld.global.u8 %r20, [%rd18];
    and.b32 %r20, %r20, 63;
    add.u32 %r24, %r14, 4;
    cvt.u64.u32 %rd19, %r24;
    add.s64 %rd20, %rd9, %rd19;
    ld.global.u8 %r21, [%rd20];
    and.b32 %r21, %r21, 63;
PTX
WARP_SCALE1_HIGH:
    add.u32 %r17, %r14, -4;
    shl.b32 %r18, %r17, 3;
    shr.u32 %r16, %r72, %r18;
    and.b32 %r16, %r16, 255;
    and.b32 %r20, %r16, 15;
    shr.u32 %r19, %r70, %r18;
    and.b32 %r19, %r19, 255;
    shr.u32 %r19, %r19, 6;
    shl.b32 %r19, %r19, 4;
    or.b32 %r20, %r20, %r19;
    shr.u32 %r21, %r16, 4;
    and.b32 %r21, %r21, 15;
    shr.u32 %r22, %r71, %r18;
    and.b32 %r22, %r22, 255;
    shr.u32 %r23, %r22, 6;
    shl.b32 %r23, %r23, 4;
    or.b32 %r21, %r21, %r23;
    bra WARP_SCALE1_DONE;

WARP_SCALE1_LOW:
    shl.b32 %r18, %r14, 3;
    shr.u32 %r20, %r70, %r18;
    and.b32 %r20, %r20, 63;
    shr.u32 %r21, %r71, %r18;
    and.b32 %r21, %r21, 63;
PTX
  .gsub(<<-PTX, <<-PTX)
WARP_SCALE2_HIGH:
    add.u32 %r26, %r25, 4;
    cvt.u64.u32 %rd21, %r26;
    add.s64 %rd22, %rd9, %rd21;
    ld.global.u8 %r27, [%rd22];
    and.b32 %r30, %r27, 15;
    add.u32 %r28, %r25, -4;
    cvt.u64.u32 %rd23, %r28;
    add.s64 %rd24, %rd9, %rd23;
    ld.global.u8 %r29, [%rd24];
    shr.u32 %r31, %r29, 6;
    shl.b32 %r31, %r31, 4;
    or.b32 %r30, %r30, %r31;
    shr.u32 %r32, %r27, 4;
    cvt.u64.u32 %rd25, %r25;
    add.s64 %rd26, %rd9, %rd25;
    ld.global.u8 %r33, [%rd26];
    shr.u32 %r34, %r33, 6;
    shl.b32 %r34, %r34, 4;
    or.b32 %r31, %r32, %r34;
    bra WARP_SCALE2_DONE;

WARP_SCALE2_LOW:
    cvt.u64.u32 %rd27, %r25;
    add.s64 %rd28, %rd9, %rd27;
    ld.global.u8 %r30, [%rd28];
    and.b32 %r30, %r30, 63;
    add.u32 %r35, %r25, 4;
    cvt.u64.u32 %rd29, %r35;
    add.s64 %rd30, %rd9, %rd29;
    ld.global.u8 %r31, [%rd30];
    and.b32 %r31, %r31, 63;
PTX
WARP_SCALE2_HIGH:
    add.u32 %r28, %r25, -4;
    shl.b32 %r29, %r28, 3;
    shr.u32 %r27, %r72, %r29;
    and.b32 %r27, %r27, 255;
    and.b32 %r30, %r27, 15;
    shr.u32 %r31, %r70, %r29;
    and.b32 %r31, %r31, 255;
    shr.u32 %r31, %r31, 6;
    shl.b32 %r31, %r31, 4;
    or.b32 %r30, %r30, %r31;
    shr.u32 %r32, %r27, 4;
    and.b32 %r32, %r32, 15;
    shr.u32 %r33, %r71, %r29;
    and.b32 %r33, %r33, 255;
    shr.u32 %r34, %r33, 6;
    shl.b32 %r34, %r34, 4;
    or.b32 %r31, %r32, %r34;
    bra WARP_SCALE2_DONE;

WARP_SCALE2_LOW:
    shl.b32 %r29, %r25, 3;
    shr.u32 %r30, %r70, %r29;
    and.b32 %r30, %r30, 63;
    shr.u32 %r31, %r71, %r29;
    and.b32 %r31, %r31, 63;
PTX

REPACK_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry q4_k_repacked_gemv_warp4_f32(
    .param .u64 qvals,
    .param .u64 scales,
    .param .u64 mins,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [qvals];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [mins];
    ld.param.u64 %rd4, [x];
    ld.param.u64 %rd5, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;       // lane
    shr.u32 %r6, %r3, 5;        // warp id inside CTA
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;      // row
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra DONE;

    shr.u32 %r9, %r1, 8;        // blocks_per_row
    mov.f32 %f1, 0f00000000;    // acc
    mov.u32 %r10, 0;            // block index

BLOCK_LOOP:
    setp.ge.u32 %p2, %r10, %r9;
    @%p2 bra REDUCE;

    mad.lo.u32 %r11, %r8, %r9, %r10; // row_block
    shl.b32 %r12, %r11, 3;           // metadata base = row_block * 8
    shl.b32 %r13, %r11, 8;           // q base = row_block * 256
    shl.b32 %r14, %r10, 8;           // x block base
    mov.u32 %r15, 0;                 // subblock 0..7

SUB_LOOP:
    setp.ge.u32 %p3, %r15, 8;
    @%p3 bra NEXT_BLOCK;

    add.u32 %r16, %r12, %r15;
    mul.wide.u32 %rd6, %r16, 4;
    add.s64 %rd7, %rd2, %rd6;
    add.s64 %rd8, %rd3, %rd6;
    ld.global.f32 %f2, [%rd7];       // scale
    ld.global.f32 %f3, [%rd8];       // min

    shl.b32 %r17, %r15, 5;
    add.u32 %r18, %r17, %r5;         // offset inside 256 block
    add.u32 %r19, %r13, %r18;
    cvt.u64.u32 %rd9, %r19;
    add.s64 %rd10, %rd1, %rd9;
    ld.global.u8 %r20, [%rd10];
    cvt.rn.f32.u32 %f4, %r20;
    mul.rn.f32 %f5, %f2, %f4;
    sub.rn.f32 %f5, %f5, %f3;

    add.u32 %r21, %r14, %r18;
    mul.wide.u32 %rd11, %r21, 4;
    add.s64 %rd12, %rd4, %rd11;
    ld.global.f32 %f6, [%rd12];
    fma.rn.f32 %f1, %f6, %f5, %f1;

    add.u32 %r15, %r15, 1;
    bra SUB_LOOP;

NEXT_BLOCK:
    add.u32 %r10, %r10, 1;
    bra BLOCK_LOOP;

REDUCE:
    mov.u32 %r22, 0xffffffff;
    mov.b32 %r23, %f1;
    shfl.sync.down.b32 %r24, %r23, 16, 31, %r22;
    mov.b32 %f7, %r24;
    add.rn.f32 %f1, %f1, %f7;
    mov.b32 %r23, %f1;
    shfl.sync.down.b32 %r24, %r23, 8, 31, %r22;
    mov.b32 %f7, %r24;
    add.rn.f32 %f1, %f1, %f7;
    mov.b32 %r23, %f1;
    shfl.sync.down.b32 %r24, %r23, 4, 31, %r22;
    mov.b32 %f7, %r24;
    add.rn.f32 %f1, %f1, %f7;
    mov.b32 %r23, %f1;
    shfl.sync.down.b32 %r24, %r23, 2, 31, %r22;
    mov.b32 %f7, %r24;
    add.rn.f32 %f1, %f1, %f7;
    mov.b32 %r23, %f1;
    shfl.sync.down.b32 %r24, %r23, 1, 31, %r22;
    mov.b32 %f7, %r24;
    add.rn.f32 %f1, %f1, %f7;

    setp.ne.u32 %p4, %r5, 0;
    @%p4 bra DONE;
    mul.wide.u32 %rd13, %r8, 4;
    add.s64 %rd14, %rd5, %rd13;
    st.global.f32 [%rd14], %f1;

DONE:
    ret;
}
PTX

F32_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry f32_gemv_warp4_f32(
    .param .u64 w,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd1, [w];
    ld.param.u64 %rd2, [x];
    ld.param.u64 %rd3, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;
    shr.u32 %r6, %r3, 5;
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra F32_DONE;

    mov.f32 %f1, 0f00000000;
    mov.u32 %r9, %r5;
F32_LOOP:
    setp.ge.u32 %p2, %r9, %r1;
    @%p2 bra F32_REDUCE;
    mad.lo.u32 %r10, %r8, %r1, %r9;
    mul.wide.u32 %rd4, %r10, 4;
    mul.wide.u32 %rd5, %r9, 4;
    add.s64 %rd6, %rd1, %rd4;
    add.s64 %rd7, %rd2, %rd5;
    ld.global.f32 %f2, [%rd6];
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32 %f1, %f2, %f3, %f1;
    add.u32 %r9, %r9, 32;
    bra F32_LOOP;

F32_REDUCE:
    mov.u32 %r11, 0xffffffff;
    mov.b32 %r12, %f1;
    shfl.sync.down.b32 %r13, %r12, 16, 31, %r11;
    mov.b32 %f4, %r13;
    add.rn.f32 %f1, %f1, %f4;
    mov.b32 %r12, %f1;
    shfl.sync.down.b32 %r13, %r12, 8, 31, %r11;
    mov.b32 %f4, %r13;
    add.rn.f32 %f1, %f1, %f4;
    mov.b32 %r12, %f1;
    shfl.sync.down.b32 %r13, %r12, 4, 31, %r11;
    mov.b32 %f4, %r13;
    add.rn.f32 %f1, %f1, %f4;
    mov.b32 %r12, %f1;
    shfl.sync.down.b32 %r13, %r12, 2, 31, %r11;
    mov.b32 %f4, %r13;
    add.rn.f32 %f1, %f1, %f4;
    mov.b32 %r12, %f1;
    shfl.sync.down.b32 %r13, %r12, 1, 31, %r11;
    mov.b32 %f4, %r13;
    add.rn.f32 %f1, %f1, %f4;

    setp.ne.u32 %p3, %r5, 0;
    @%p3 bra F32_DONE;
    mul.wide.u32 %rd8, %r8, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f1;

F32_DONE:
    ret;
}
PTX

META_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry q4_k_meta_gemv_warp4_f32(
    .param .u64 w_raw,
    .param .u64 scales,
    .param .u64 mins,
    .param .u64 x,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<6>;
    .reg .b32 %r<72>;
    .reg .b64 %rd<64>;
    .reg .f32 %f<20>;

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [mins];
    ld.param.u64 %rd4, [x];
    ld.param.u64 %rd5, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;       // lane
    shr.u32 %r6, %r3, 5;        // warp id inside CTA
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;      // row
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra META_DONE;

    shr.u32 %r9, %r1, 8;        // blocks_per_row
    mul.lo.u32 %r10, %r9, 144;  // raw row bytes
    mul.wide.u32 %rd6, %r8, %r10;
    add.s64 %rd7, %rd1, %rd6;   // raw row base

    mov.f32 %f1, 0f00000000;    // acc
    mov.u32 %r11, 0;            // block index

META_BLOCK_LOOP:
    setp.ge.u32 %p2, %r11, %r9;
    @%p2 bra META_REDUCE;

    mul.lo.u32 %r12, %r11, 144;
    cvt.u64.u32 %rd8, %r12;
    add.s64 %rd9, %rd7, %rd8;   // block base
    add.s64 %rd10, %rd9, 16;    // qs base
    mad.lo.u32 %r13, %r8, %r9, %r11; // row_block
    shl.b32 %r14, %r13, 3;      // metadata base
    shl.b32 %r15, %r11, 8;      // x block base
    mov.u32 %r16, 0;            // group 0..3

META_GROUP_LOOP:
    setp.ge.u32 %p3, %r16, 4;
    @%p3 bra META_NEXT_BLOCK;

    shl.b32 %r17, %r16, 1;      // subblock index 0,2,4,6
    add.u32 %r18, %r14, %r17;
    mul.wide.u32 %rd11, %r18, 4;
    add.s64 %rd12, %rd2, %rd11;
    add.s64 %rd13, %rd3, %rd11;
    ld.global.f32 %f2, [%rd12]; // sc low
    ld.global.f32 %f3, [%rd13]; // min low

    add.u32 %r19, %r18, 1;
    mul.wide.u32 %rd14, %r19, 4;
    add.s64 %rd15, %rd2, %rd14;
    add.s64 %rd16, %rd3, %rd14;
    ld.global.f32 %f4, [%rd15]; // sc high
    ld.global.f32 %f5, [%rd16]; // min high

    shl.b32 %r20, %r16, 5;
    add.u32 %r21, %r20, %r5;
    cvt.u64.u32 %rd17, %r21;
    add.s64 %rd18, %rd10, %rd17;
    ld.global.u8 %r22, [%rd18];

    and.b32 %r23, %r22, 15;
    cvt.rn.f32.u32 %f6, %r23;
    mul.rn.f32 %f7, %f2, %f6;
    sub.rn.f32 %f7, %f7, %f3;

    shr.u32 %r24, %r22, 4;
    cvt.rn.f32.u32 %f8, %r24;
    mul.rn.f32 %f9, %f4, %f8;
    sub.rn.f32 %f9, %f9, %f5;

    shl.b32 %r25, %r16, 6;
    add.u32 %r26, %r15, %r25;
    add.u32 %r27, %r26, %r5;
    mul.wide.u32 %rd19, %r27, 4;
    add.s64 %rd20, %rd4, %rd19;
    ld.global.f32 %f10, [%rd20];
    fma.rn.f32 %f1, %f10, %f7, %f1;

    add.u32 %r28, %r27, 32;
    mul.wide.u32 %rd21, %r28, 4;
    add.s64 %rd22, %rd4, %rd21;
    ld.global.f32 %f11, [%rd22];
    fma.rn.f32 %f1, %f11, %f9, %f1;

    add.u32 %r16, %r16, 1;
    bra META_GROUP_LOOP;

META_NEXT_BLOCK:
    add.u32 %r11, %r11, 1;
    bra META_BLOCK_LOOP;

META_REDUCE:
    mov.u32 %r40, 0xffffffff;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 16, 31, %r40;
    mov.b32 %f12, %r42;
    add.rn.f32 %f1, %f1, %f12;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 8, 31, %r40;
    mov.b32 %f12, %r42;
    add.rn.f32 %f1, %f1, %f12;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 4, 31, %r40;
    mov.b32 %f12, %r42;
    add.rn.f32 %f1, %f1, %f12;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 2, 31, %r40;
    mov.b32 %f12, %r42;
    add.rn.f32 %f1, %f1, %f12;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 1, 31, %r40;
    mov.b32 %f12, %r42;
    add.rn.f32 %f1, %f1, %f12;

    setp.ne.u32 %p4, %r5, 0;
    @%p4 bra META_DONE;
    mul.wide.u32 %rd23, %r8, 4;
    add.s64 %rd24, %rd5, %rd23;
    st.global.f32 [%rd24], %f1;

META_DONE:
    ret;
}
PTX

record RepackedQ4, qvals : Bytes, scales : Array(Float32), mins : Array(Float32)
record PackedQ8Input, packs : Array(UInt32), scales : Array(Float32)

Q8_DP4A_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry q4_k_q8_dp4a_gemv_warp4_f32(
    .param .u64 w_raw,
    .param .u64 scales,
    .param .u64 mins,
    .param .u64 q8_packs,
    .param .u64 q8_scales,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<96>;
    .reg .b64 %rd<72>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [scales];
    ld.param.u64 %rd3, [mins];
    ld.param.u64 %rd4, [q8_packs];
    ld.param.u64 %rd5, [q8_scales];
    ld.param.u64 %rd6, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;       // lane
    shr.u32 %r6, %r3, 5;        // warp id inside CTA
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;      // row
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra Q8_DONE;

    shr.u32 %r9, %r1, 8;        // blocks_per_row
    mul.lo.u32 %r10, %r9, 144;  // raw row bytes
    mul.wide.u32 %rd7, %r8, %r10;
    add.s64 %rd8, %rd1, %rd7;   // row base

    mov.f32 %f1, 0f00000000;    // acc
    mov.u32 %r11, 0;            // q4 block index

Q8_BLOCK_LOOP:
    setp.ge.u32 %p2, %r11, %r9;
    @%p2 bra Q8_REDUCE;

    mul.lo.u32 %r12, %r11, 144;
    cvt.u64.u32 %rd9, %r12;
    add.s64 %rd10, %rd8, %rd9;  // q4 block base
    add.s64 %rd11, %rd10, 16;   // qs base

    // Each warp lane processes two 4-element packs per 256-value Q4_K block.
    mov.u32 %r13, 0;

Q8_PACK_ITER:
    setp.ge.u32 %p3, %r13, 2;
    @%p3 bra Q8_NEXT_BLOCK;

    mad.lo.u32 %r14, %r13, 32, %r5; // pack index 0..63
    shr.u32 %r15, %r14, 3;          // subblock 0..7, each 32 values
    and.b32 %r16, %r14, 7;          // 4-value pack inside subblock
    shl.b32 %r17, %r16, 2;          // byte offset inside 32-byte subblock
    shr.u32 %r18, %r15, 1;          // q4 byte group 0..3
    shl.b32 %r19, %r18, 5;
    add.u32 %r20, %r19, %r17;
    cvt.u64.u32 %rd12, %r20;
    add.s64 %rd13, %rd11, %rd12;
    ld.global.u32 %r21, [%rd13];    // four packed q4 bytes

    and.b32 %r22, %r15, 1;
    setp.ne.u32 %p4, %r22, 0;
    @%p4 bra Q8_HIGH_NIBBLE;
Q8_LOW_NIBBLE:
    and.b32 %r23, %r21, 252645135;  // 0x0f0f0f0f
    bra Q8_NIBBLE_DONE;
Q8_HIGH_NIBBLE:
    shr.u32 %r24, %r21, 4;
    and.b32 %r23, %r24, 252645135;  // 0x0f0f0f0f

Q8_NIBBLE_DONE:
    shl.b32 %r25, %r11, 6;
    add.u32 %r26, %r25, %r14;
    mul.wide.u32 %rd14, %r26, 4;
    add.s64 %rd15, %rd4, %rd14;
    ld.global.u32 %r27, [%rd15];    // four signed q8 bytes

    mov.u32 %r28, 0;
    dp4a.s32.s32 %r28, %r23, %r27, %r28;
    mov.u32 %r29, 0;
    mov.u32 %r30, 16843009;         // 0x01010101
    dp4a.s32.s32 %r29, %r30, %r27, %r29;

    mad.lo.u32 %r31, %r8, %r9, %r11;
    shl.b32 %r32, %r31, 3;
    add.u32 %r33, %r32, %r15;
    mul.wide.u32 %rd16, %r33, 4;
    add.s64 %rd17, %rd2, %rd16;
    add.s64 %rd18, %rd3, %rd16;
    ld.global.f32 %f2, [%rd17];     // Q4 d*scale
    ld.global.f32 %f3, [%rd18];     // Q4 dmin*min

    shl.b32 %r34, %r11, 3;
    add.u32 %r35, %r34, %r15;
    mul.wide.u32 %rd19, %r35, 4;
    add.s64 %rd20, %rd5, %rd19;
    ld.global.f32 %f4, [%rd20];     // Q8 activation scale

    cvt.rn.f32.s32 %f5, %r28;
    cvt.rn.f32.s32 %f6, %r29;
    mul.rn.f32 %f7, %f2, %f5;
    neg.f32 %f8, %f6;
    fma.rn.f32 %f7, %f3, %f8, %f7;
    fma.rn.f32 %f1, %f4, %f7, %f1;

    add.u32 %r13, %r13, 1;
    bra Q8_PACK_ITER;

Q8_NEXT_BLOCK:
    add.u32 %r11, %r11, 1;
    bra Q8_BLOCK_LOOP;

Q8_REDUCE:
    mov.u32 %r40, 0xffffffff;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 16, 31, %r40;
    mov.b32 %f9, %r42;
    add.rn.f32 %f1, %f1, %f9;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 8, 31, %r40;
    mov.b32 %f9, %r42;
    add.rn.f32 %f1, %f1, %f9;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 4, 31, %r40;
    mov.b32 %f9, %r42;
    add.rn.f32 %f1, %f1, %f9;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 2, 31, %r40;
    mov.b32 %f9, %r42;
    add.rn.f32 %f1, %f1, %f9;
    mov.b32 %r41, %f1;
    shfl.sync.down.b32 %r42, %r41, 1, 31, %r40;
    mov.b32 %f9, %r42;
    add.rn.f32 %f1, %f1, %f9;

    setp.ne.u32 %p5, %r5, 0;
    @%p5 bra Q8_DONE;
    mul.wide.u32 %rd21, %r8, 4;
    add.s64 %rd22, %rd6, %rd21;
    st.global.f32 [%rd22], %f1;

Q8_DONE:
    ret;
}
PTX

Q8_RAW_DP4A_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry q4_k_raw_q8_dp4a_gemv_warp4_f32(
    .param .u64 w_raw,
    .param .u64 q8_packs,
    .param .u64 q8_scales,
    .param .u64 out,
    .param .u32 in_dim,
    .param .u32 out_dim
)
{
    .reg .pred %p<10>;
    .reg .b16 %h<3>;
    .reg .b32 %r<112>;
    .reg .b64 %rd<80>;
    .reg .f32 %f<20>;

    ld.param.u64 %rd1, [w_raw];
    ld.param.u64 %rd2, [q8_packs];
    ld.param.u64 %rd3, [q8_scales];
    ld.param.u64 %rd4, [out];
    ld.param.u32 %r1, [in_dim];
    ld.param.u32 %r2, [out_dim];

    mov.u32 %r3, %tid.x;
    mov.u32 %r4, %ctaid.x;
    and.b32 %r5, %r3, 31;       // lane
    shr.u32 %r6, %r3, 5;        // warp id inside CTA
    shl.b32 %r7, %r4, 2;
    add.u32 %r8, %r7, %r6;      // row
    setp.ge.u32 %p1, %r8, %r2;
    @%p1 bra RAW_Q8_DONE;

    shr.u32 %r9, %r1, 8;        // blocks_per_row
    mul.lo.u32 %r10, %r9, 144;  // raw row bytes
    mul.wide.u32 %rd7, %r8, %r10;
    add.s64 %rd8, %rd1, %rd7;   // row base

    mov.f32 %f1, 0f00000000;    // acc
    mov.u32 %r11, 0;            // q4 block index

RAW_Q8_BLOCK_LOOP:
    setp.ge.u32 %p2, %r11, %r9;
    @%p2 bra RAW_Q8_REDUCE;

    mul.lo.u32 %r12, %r11, 144;
    cvt.u64.u32 %rd9, %r12;
    add.s64 %rd10, %rd8, %rd9;  // q4 block base
    ld.global.u16 %h1, [%rd10];
    add.s64 %rd20, %rd10, 2;
    ld.global.u16 %h2, [%rd20];
    cvt.f32.f16 %f2, %h1;       // d
    cvt.f32.f16 %f3, %h2;       // dmin
    add.s64 %rd21, %rd10, 4;    // scales base
    add.s64 %rd11, %rd10, 16;   // qs base

    // Each warp lane processes two 4-element packs per 256-value Q4_K block.
    mov.u32 %r13, 0;

RAW_Q8_PACK_ITER:
    setp.ge.u32 %p3, %r13, 2;
    @%p3 bra RAW_Q8_NEXT_BLOCK;

    mad.lo.u32 %r14, %r13, 32, %r5; // pack index 0..63
    shr.u32 %r15, %r14, 3;          // subblock 0..7
    and.b32 %r16, %r14, 7;          // 4-value pack inside subblock
    shl.b32 %r17, %r16, 2;          // byte offset inside 32-byte subblock
    shr.u32 %r18, %r15, 1;          // q4 byte group 0..3
    shl.b32 %r19, %r18, 5;
    add.u32 %r20, %r19, %r17;
    cvt.u64.u32 %rd12, %r20;
    add.s64 %rd13, %rd11, %rd12;
    ld.global.u32 %r21, [%rd13];    // four packed q4 bytes

    and.b32 %r22, %r15, 1;
    setp.ne.u32 %p4, %r22, 0;
    @%p4 bra RAW_Q8_HIGH_NIBBLE;
RAW_Q8_LOW_NIBBLE:
    and.b32 %r23, %r21, 252645135;  // 0x0f0f0f0f
    bra RAW_Q8_NIBBLE_DONE;
RAW_Q8_HIGH_NIBBLE:
    shr.u32 %r24, %r21, 4;
    and.b32 %r23, %r24, 252645135;  // 0x0f0f0f0f

RAW_Q8_NIBBLE_DONE:
    shl.b32 %r25, %r11, 6;
    add.u32 %r26, %r25, %r14;
    mul.wide.u32 %rd14, %r26, 4;
    add.s64 %rd15, %rd2, %rd14;
    ld.global.u32 %r27, [%rd15];    // four signed q8 bytes

    mov.u32 %r28, 0;
    dp4a.s32.s32 %r28, %r23, %r27, %r28;
    mov.u32 %r29, 0;
    mov.u32 %r30, 16843009;         // 0x01010101
    dp4a.s32.s32 %r29, %r30, %r27, %r29;

    setp.lt.u32 %p5, %r15, 4;
    @%p5 bra RAW_Q8_SCALE_LOW;

RAW_Q8_SCALE_HIGH:
    add.u32 %r40, %r15, 4;
    cvt.u64.u32 %rd30, %r40;
    add.s64 %rd31, %rd21, %rd30;
    ld.global.u8 %r41, [%rd31];     // packed high scale/min
    and.b32 %r42, %r41, 15;

    add.u32 %r43, %r15, -4;
    cvt.u64.u32 %rd32, %r43;
    add.s64 %rd33, %rd21, %rd32;
    ld.global.u8 %r44, [%rd33];
    shr.u32 %r45, %r44, 6;
    shl.b32 %r45, %r45, 4;
    or.b32 %r50, %r42, %r45;        // sc

    shr.u32 %r46, %r41, 4;
    cvt.u64.u32 %rd34, %r15;
    add.s64 %rd35, %rd21, %rd34;
    ld.global.u8 %r47, [%rd35];
    shr.u32 %r48, %r47, 6;
    shl.b32 %r48, %r48, 4;
    or.b32 %r51, %r46, %r48;        // min
    bra RAW_Q8_SCALE_DONE;

RAW_Q8_SCALE_LOW:
    cvt.u64.u32 %rd36, %r15;
    add.s64 %rd37, %rd21, %rd36;
    ld.global.u8 %r50, [%rd37];
    and.b32 %r50, %r50, 63;         // sc
    add.u32 %r52, %r15, 4;
    cvt.u64.u32 %rd38, %r52;
    add.s64 %rd39, %rd21, %rd38;
    ld.global.u8 %r51, [%rd39];
    and.b32 %r51, %r51, 63;         // min

RAW_Q8_SCALE_DONE:
    shl.b32 %r34, %r11, 3;
    add.u32 %r35, %r34, %r15;
    mul.wide.u32 %rd19, %r35, 4;
    add.s64 %rd22, %rd3, %rd19;
    ld.global.f32 %f4, [%rd22];     // Q8 activation scale

    cvt.rn.f32.s32 %f5, %r28;
    cvt.rn.f32.s32 %f6, %r29;
    cvt.rn.f32.u32 %f7, %r50;
    cvt.rn.f32.u32 %f8, %r51;
    mul.rn.f32 %f9, %f2, %f7;
    mul.rn.f32 %f10, %f3, %f8;
    mul.rn.f32 %f11, %f9, %f5;
    neg.f32 %f12, %f6;
    fma.rn.f32 %f11, %f10, %f12, %f11;
    fma.rn.f32 %f1, %f4, %f11, %f1;

    add.u32 %r13, %r13, 1;
    bra RAW_Q8_PACK_ITER;

RAW_Q8_NEXT_BLOCK:
    add.u32 %r11, %r11, 1;
    bra RAW_Q8_BLOCK_LOOP;

RAW_Q8_REDUCE:
    mov.u32 %r60, 0xffffffff;
    mov.b32 %r61, %f1;
    shfl.sync.down.b32 %r62, %r61, 16, 31, %r60;
    mov.b32 %f13, %r62;
    add.rn.f32 %f1, %f1, %f13;
    mov.b32 %r61, %f1;
    shfl.sync.down.b32 %r62, %r61, 8, 31, %r60;
    mov.b32 %f13, %r62;
    add.rn.f32 %f1, %f1, %f13;
    mov.b32 %r61, %f1;
    shfl.sync.down.b32 %r62, %r61, 4, 31, %r60;
    mov.b32 %f13, %r62;
    add.rn.f32 %f1, %f1, %f13;
    mov.b32 %r61, %f1;
    shfl.sync.down.b32 %r62, %r61, 2, 31, %r60;
    mov.b32 %f13, %r62;
    add.rn.f32 %f1, %f1, %f13;
    mov.b32 %r61, %f1;
    shfl.sync.down.b32 %r62, %r61, 1, 31, %r60;
    mov.b32 %f13, %r62;
    add.rn.f32 %f1, %f1, %f13;

    setp.ne.u32 %p6, %r5, 0;
    @%p6 bra RAW_Q8_DONE;
    mul.wide.u32 %rd23, %r8, 4;
    add.s64 %rd24, %rd4, %rd23;
    st.global.f32 [%rd24], %f1;

RAW_Q8_DONE:
    ret;
}
PTX

Q8_QUANT_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry quantize_q8_1_f32(
    .param .u64 x,
    .param .u64 q8_packs,
    .param .u64 q8_scales,
    .param .u32 in_dim
)
{
    .reg .pred %p<8>;
    .reg .b32 %r<72>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<16>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [q8_packs];
    ld.param.u64 %rd3, [q8_scales];
    ld.param.u32 %r1, [in_dim];

    mov.u32 %r2, %tid.x;       // lane 0..31
    mov.u32 %r3, %ctaid.x;     // 32-value subblock
    shl.b32 %r4, %r3, 5;
    add.u32 %r5, %r4, %r2;     // input index
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra Q8Q_DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.s64 %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];
    abs.f32 %f2, %f1;

    mov.u32 %r6, 0xffffffff;
    mov.b32 %r7, %f2;
    shfl.sync.down.b32 %r8, %r7, 16, 31, %r6;
    mov.b32 %f3, %r8;
    max.f32 %f2, %f2, %f3;
    mov.b32 %r7, %f2;
    shfl.sync.down.b32 %r8, %r7, 8, 31, %r6;
    mov.b32 %f3, %r8;
    max.f32 %f2, %f2, %f3;
    mov.b32 %r7, %f2;
    shfl.sync.down.b32 %r8, %r7, 4, 31, %r6;
    mov.b32 %f3, %r8;
    max.f32 %f2, %f2, %f3;
    mov.b32 %r7, %f2;
    shfl.sync.down.b32 %r8, %r7, 2, 31, %r6;
    mov.b32 %f3, %r8;
    max.f32 %f2, %f2, %f3;
    mov.b32 %r7, %f2;
    shfl.sync.down.b32 %r8, %r7, 1, 31, %r6;
    mov.b32 %f3, %r8;
    max.f32 %f2, %f2, %f3;

    mov.b32 %r7, %f2;
    shfl.sync.idx.b32 %r8, %r7, 0, 31, %r6;
    mov.b32 %f4, %r8;          // amax
    setp.gt.f32 %p2, %f4, 0f00000000;
    @%p2 bra Q8Q_SCALE_NONZERO;
    mov.f32 %f5, 0f3f800000;   // scale = 1.0
    bra Q8Q_SCALE_DONE;
Q8Q_SCALE_NONZERO:
    div.rn.f32 %f5, %f4, 0f42fe0000; // /127.0
Q8Q_SCALE_DONE:
    div.rn.f32 %f6, %f1, %f5;
    cvt.rni.s32.f32 %r9, %f6;
    max.s32 %r9, %r9, -127;
    min.s32 %r9, %r9, 127;

    and.b32 %r10, %r2, 3;
    mov.u32 %r11, %r9;
    and.b32 %r11, %r11, 255;

    add.u32 %r12, %r2, 1;
    shfl.sync.idx.b32 %r13, %r9, %r12, 31, %r6;
    and.b32 %r13, %r13, 255;
    shl.b32 %r13, %r13, 8;
    or.b32 %r11, %r11, %r13;

    add.u32 %r14, %r2, 2;
    shfl.sync.idx.b32 %r15, %r9, %r14, 31, %r6;
    and.b32 %r15, %r15, 255;
    shl.b32 %r15, %r15, 16;
    or.b32 %r11, %r11, %r15;

    add.u32 %r16, %r2, 3;
    shfl.sync.idx.b32 %r17, %r9, %r16, 31, %r6;
    and.b32 %r17, %r17, 255;
    shl.b32 %r17, %r17, 24;
    or.b32 %r11, %r11, %r17;

    setp.ne.u32 %p3, %r10, 0;
    @%p3 bra Q8Q_MAYBE_SCALE;

    shr.u32 %r18, %r2, 2;      // pack 0..7
    shl.b32 %r19, %r3, 3;
    add.u32 %r20, %r19, %r18;
    mul.wide.u32 %rd6, %r20, 4;
    add.s64 %rd7, %rd2, %rd6;
    st.global.u32 [%rd7], %r11;

Q8Q_MAYBE_SCALE:
    setp.ne.u32 %p4, %r2, 0;
    @%p4 bra Q8Q_DONE;
    mul.wide.u32 %rd8, %r3, 4;
    add.s64 %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f5;

Q8Q_DONE:
    ret;
}
PTX

def bytesize_f32(elements : Int32) : LibC::SizeT
  (elements * sizeof(Float32)).to_u64
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise ArgumentError.new("size mismatch") unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |v, i|
    d = (v - b[i]).abs
    max = d if d > max
  end
  max
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  dot = 0.0_f64
  na = 0.0_f64
  nb = 0.0_f64
  a.each_with_index do |v, i|
    av = v.to_f64
    bv = b[i].to_f64
    dot += av * bv
    na += av * av
    nb += bv * bv
  end
  dot / Math.sqrt(na * nb)
end

def max_abs_diff_f32(a : Array(Float32), b : Array(Float32)) : Float32
  max_abs_diff(a, b)
end

def repack_q4_k(raw : Bytes, in_dim : Int32, out_dim : Int32) : RepackedQ4
  blocks = in_dim // 256
  qvals = Bytes.new(out_dim * blocks * 256)
  scales = Array(Float32).new(out_dim * blocks * 8, 0.0_f32)
  mins = Array(Float32).new(out_dim * blocks * 8, 0.0_f32)

  out_dim.times do |row|
    row_base = row * blocks * 144
    blocks.times do |b|
      off = row_base + b * 144
      d = ML::GGUF::Dequant.fp16_to_f32(raw[off, 2])
      dmin = ML::GGUF::Dequant.fp16_to_f32(raw[off + 2, 2])
      scales_ptr = raw.to_unsafe + off + 4
      qs_ptr = raw.to_unsafe + off + 16
      rb = row * blocks + b

      8.times do |sub|
        sc, m = ML::GGUF::Dequant.get_scale_min_k4(sub, scales_ptr)
        scales[rb * 8 + sub] = d * sc
        mins[rb * 8 + sub] = dmin * m
      end

      4.times do |group|
        32.times do |lane|
          q = qs_ptr[group * 32 + lane]
          qvals[rb * 256 + group * 64 + lane] = (q & 0x0F).to_u8
          qvals[rb * 256 + group * 64 + 32 + lane] = (q.to_u32 >> 4).to_u8
        end
      end
    end
  end

  RepackedQ4.new(qvals, scales, mins)
end

def quantize_q8_1_input(x : Array(Float32), in_dim : Int32) : PackedQ8Input
  raise ArgumentError.new("Q8_1 input requires in_dim multiple of 32") unless in_dim % 32 == 0

  subblocks = in_dim // 32
  packs = Array(UInt32).new(subblocks * 8, 0_u32)
  scales = Array(Float32).new(subblocks, 0.0_f32)

  subblocks.times do |sub|
    base = sub * 32
    amax = 0.0_f32
    32.times do |i|
      v = x[base + i].abs
      amax = v if v > amax
    end
    scale = amax > 0.0_f32 ? (amax / 127.0_f32) : 1.0_f32
    scales[sub] = scale

    8.times do |pack|
      word = 0_u32
      4.times do |j|
        q = (x[base + pack * 4 + j] / scale).round.to_i
        q = -127 if q < -127
        q = 127 if q > 127
        word |= ((q & 0xff).to_u32 << (j * 8))
      end
      packs[sub * 8 + pack] = word
    end
  end

  PackedQ8Input.new(packs, scales)
end

def run_kernel(fn : ML::CUDA::KernelFunction, grid : UInt32, block : UInt32, params : Void**, reps : Int32, warmup : Int32) : Float64
  warmup.times do
    ML::CUDA.launch!(fn, grid, 1_u32, 1_u32, block, 1_u32, 1_u32, params, "warmup")
  end
  ML::CUDA.synchronize!("cuCtxSynchronize(warmup)") if warmup > 0

  t0 = Time.instant
  reps.times do
    ML::CUDA.launch!(fn, grid, 1_u32, 1_u32, block, 1_u32, 1_u32, params, "timed")
  end
  ML::CUDA.synchronize!("cuCtxSynchronize(timed)")
  (Time.instant - t0).total_milliseconds / reps
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
tensor_name = DEFAULT_TENSOR
seed = 23_u64
reps = 20
warmup = 3

OptionParser.parse do |p|
  p.banner = "Usage: cuda_q4k_repack_probe [--model PATH] [--tensor NAME] [--seed N] [--reps N] [--warmup N]"
  p.on("--model PATH", "Q4_K GGUF model path") { |v| model = v }
  p.on("--tensor NAME", "Q4_K tensor name") { |v| tensor_name = v }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed kernel launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

gguf = ML::GGUF::GGUFFile.new(model)
info = gguf.tensor(tensor_name) || raise "missing tensor #{tensor_name.inspect}"
raise "expected Q4_K tensor, got #{info.type.name}" unless info.type.q4_k?
raise "expected matrix tensor, got dims=#{info.dims}" unless info.dims.size >= 2

in_dim = info.dims[0].to_i32
out_dim = info.dims[1].to_i32
raise "Q4_K GEMV requires in_dim multiple of 256, got #{in_dim}" unless in_dim % 256 == 0

w_raw = gguf.read_tensor_raw(info)
rng = Random.new(seed)
x = Array(Float32).new(in_dim) { rng.rand(-1.0_f32..1.0_f32) }
zero_bias = Array(Float32).new(out_dim, 0.0_f32)

cpu_t0 = Time.instant
cpu = ML::GGUF::QuantMatmul.matmul_add(x, 1, in_dim, w_raw, ML::GGUF::TensorType::Q4_K, out_dim, zero_bias)
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

repack_t0 = Time.instant
repacked = repack_q4_k(w_raw, in_dim, out_dim)
repack_ms = (Time.instant - repack_t0).total_milliseconds

q8_t0 = Time.instant
q8_input = quantize_q8_1_input(x, in_dim)
q8_pack_ms = (Time.instant - q8_t0).total_milliseconds

f32_t0 = Time.instant
f32_weights = ML::GGUF::Dequant.dequantize_q4_k(w_raw, in_dim * out_dim)
f32_repack_ms = (Time.instant - f32_t0).total_milliseconds

ctx = ML::CUDA::Context.create
modules = [] of ML::CUDA::CUDAModule
buffers = [] of ML::CUDA::DeviceBuffer
begin
  raw_mod = ML::CUDA::CUDAModule.load(Q4K_PTX, "q4_raw")
  scale_regs_mod = ML::CUDA::CUDAModule.load(Q4K_SCALE_REGS_PTX, "q4_scale_regs")
  meta_mod = ML::CUDA::CUDAModule.load(META_PTX, "q4_meta")
  repack_mod = ML::CUDA::CUDAModule.load(REPACK_PTX, "q4_repack")
  f32_mod = ML::CUDA::CUDAModule.load(F32_PTX, "q4_f32")
  q8_mod = ML::CUDA::CUDAModule.load(Q8_DP4A_PTX, "q4_q8_dp4a")
  q8_raw_mod = ML::CUDA::CUDAModule.load(Q8_RAW_DP4A_PTX, "q4_raw_q8_dp4a")
  q8_quant_mod = ML::CUDA::CUDAModule.load(Q8_QUANT_PTX, "q8_quant")
  modules.concat([raw_mod, scale_regs_mod, meta_mod, repack_mod, f32_mod, q8_mod, q8_raw_mod, q8_quant_mod])
  raw_fn = raw_mod.function("q4_k_gemv_warp4_f32")
  scale_regs_fn = scale_regs_mod.function("q4_k_gemv_scale_regs_warp4_f32")
  meta_fn = meta_mod.function("q4_k_meta_gemv_warp4_f32")
  repack_fn = repack_mod.function("q4_k_repacked_gemv_warp4_f32")
  f32_fn = f32_mod.function("f32_gemv_warp4_f32")
  q8_fn = q8_mod.function("q4_k_q8_dp4a_gemv_warp4_f32")
  q8_raw_fn = q8_raw_mod.function("q4_k_raw_q8_dp4a_gemv_warp4_f32")
  q8_quant_fn = q8_quant_mod.function("quantize_q8_1_f32")

  d_raw = ML::CUDA::DeviceBuffer.new(w_raw.size.to_u64)
  d_qvals = ML::CUDA::DeviceBuffer.new(repacked.qvals.size.to_u64)
  d_scales = ML::CUDA::DeviceBuffer.new(bytesize_f32(repacked.scales.size))
  d_mins = ML::CUDA::DeviceBuffer.new(bytesize_f32(repacked.mins.size))
  d_q8_packs = ML::CUDA::DeviceBuffer.new(q8_input.packs.size.to_u64 * 4_u64)
  d_q8_scales = ML::CUDA::DeviceBuffer.new(bytesize_f32(q8_input.scales.size))
  d_q8_gpu_packs = ML::CUDA::DeviceBuffer.new(q8_input.packs.size.to_u64 * 4_u64)
  d_q8_gpu_scales = ML::CUDA::DeviceBuffer.new(bytesize_f32(q8_input.scales.size))
  d_f32 = ML::CUDA::DeviceBuffer.new(bytesize_f32(f32_weights.size))
  d_x = ML::CUDA::DeviceBuffer.new(bytesize_f32(in_dim))
  d_raw_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_scale_regs_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_meta_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_repack_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_f32_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_q8_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_q8_raw_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_q8_raw_gpu_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_q8_gpu_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  buffers.concat([d_raw, d_qvals, d_scales, d_mins, d_q8_packs, d_q8_scales, d_q8_gpu_packs, d_q8_gpu_scales, d_f32, d_x, d_raw_out, d_scale_regs_out, d_meta_out, d_repack_out, d_f32_out, d_q8_out, d_q8_raw_out, d_q8_raw_gpu_out, d_q8_gpu_out])

  ML::CUDA.copy_htod!(d_raw.ptr, w_raw.to_unsafe.as(Void*), w_raw.size.to_u64, "raw")
  ML::CUDA.copy_htod!(d_qvals.ptr, repacked.qvals.to_unsafe.as(Void*), repacked.qvals.size.to_u64, "qvals")
  ML::CUDA.copy_htod!(d_scales.ptr, repacked.scales.to_unsafe.as(Void*), bytesize_f32(repacked.scales.size), "scales")
  ML::CUDA.copy_htod!(d_mins.ptr, repacked.mins.to_unsafe.as(Void*), bytesize_f32(repacked.mins.size), "mins")
  ML::CUDA.copy_htod!(d_q8_packs.ptr, q8_input.packs.to_unsafe.as(Void*), q8_input.packs.size.to_u64 * 4_u64, "q8_packs")
  ML::CUDA.copy_htod!(d_q8_scales.ptr, q8_input.scales.to_unsafe.as(Void*), bytesize_f32(q8_input.scales.size), "q8_scales")
  ML::CUDA.copy_htod!(d_f32.ptr, f32_weights.to_unsafe.as(Void*), bytesize_f32(f32_weights.size), "f32_weights")
  ML::CUDA.copy_htod!(d_x.ptr, x.to_unsafe.as(Void*), bytesize_f32(in_dim), "x")

  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  grid = ((out_dim + 3) // 4).to_u32
  block = 128_u32

  raw_params = Pointer(Void*).malloc(5)
  raw_w = d_raw.ptr
  raw_x = d_x.ptr
  raw_out = d_raw_out.ptr
  raw_params[0] = pointerof(raw_w).as(Void*)
  raw_params[1] = pointerof(raw_x).as(Void*)
  raw_params[2] = pointerof(raw_out).as(Void*)
  raw_params[3] = pointerof(in_dim_u32).as(Void*)
  raw_params[4] = pointerof(out_dim_u32).as(Void*)

  scale_regs_params = Pointer(Void*).malloc(5)
  scale_regs_w = d_raw.ptr
  scale_regs_x = d_x.ptr
  scale_regs_out = d_scale_regs_out.ptr
  scale_regs_params[0] = pointerof(scale_regs_w).as(Void*)
  scale_regs_params[1] = pointerof(scale_regs_x).as(Void*)
  scale_regs_params[2] = pointerof(scale_regs_out).as(Void*)
  scale_regs_params[3] = pointerof(in_dim_u32).as(Void*)
  scale_regs_params[4] = pointerof(out_dim_u32).as(Void*)

  meta_params = Pointer(Void*).malloc(7)
  meta_w = d_raw.ptr
  meta_scales = d_scales.ptr
  meta_mins = d_mins.ptr
  meta_x = d_x.ptr
  meta_out = d_meta_out.ptr
  meta_params[0] = pointerof(meta_w).as(Void*)
  meta_params[1] = pointerof(meta_scales).as(Void*)
  meta_params[2] = pointerof(meta_mins).as(Void*)
  meta_params[3] = pointerof(meta_x).as(Void*)
  meta_params[4] = pointerof(meta_out).as(Void*)
  meta_params[5] = pointerof(in_dim_u32).as(Void*)
  meta_params[6] = pointerof(out_dim_u32).as(Void*)

  repack_params = Pointer(Void*).malloc(7)
  qvals_ptr = d_qvals.ptr
  scales_ptr = d_scales.ptr
  mins_ptr = d_mins.ptr
  repack_x = d_x.ptr
  repack_out = d_repack_out.ptr
  repack_params[0] = pointerof(qvals_ptr).as(Void*)
  repack_params[1] = pointerof(scales_ptr).as(Void*)
  repack_params[2] = pointerof(mins_ptr).as(Void*)
  repack_params[3] = pointerof(repack_x).as(Void*)
  repack_params[4] = pointerof(repack_out).as(Void*)
  repack_params[5] = pointerof(in_dim_u32).as(Void*)
  repack_params[6] = pointerof(out_dim_u32).as(Void*)

  f32_params = Pointer(Void*).malloc(5)
  f32_w = d_f32.ptr
  f32_x = d_x.ptr
  f32_out = d_f32_out.ptr
  f32_params[0] = pointerof(f32_w).as(Void*)
  f32_params[1] = pointerof(f32_x).as(Void*)
  f32_params[2] = pointerof(f32_out).as(Void*)
  f32_params[3] = pointerof(in_dim_u32).as(Void*)
  f32_params[4] = pointerof(out_dim_u32).as(Void*)

  q8_params = Pointer(Void*).malloc(8)
  q8_w = d_raw.ptr
  q8_scales_ptr = d_scales.ptr
  q8_mins_ptr = d_mins.ptr
  q8_packs_ptr = d_q8_packs.ptr
  q8_input_scales_ptr = d_q8_scales.ptr
  q8_out = d_q8_out.ptr
  q8_params[0] = pointerof(q8_w).as(Void*)
  q8_params[1] = pointerof(q8_scales_ptr).as(Void*)
  q8_params[2] = pointerof(q8_mins_ptr).as(Void*)
  q8_params[3] = pointerof(q8_packs_ptr).as(Void*)
  q8_params[4] = pointerof(q8_input_scales_ptr).as(Void*)
  q8_params[5] = pointerof(q8_out).as(Void*)
  q8_params[6] = pointerof(in_dim_u32).as(Void*)
  q8_params[7] = pointerof(out_dim_u32).as(Void*)

  q8_raw_params = Pointer(Void*).malloc(6)
  q8_raw_w = d_raw.ptr
  q8_raw_packs_ptr = d_q8_packs.ptr
  q8_raw_input_scales_ptr = d_q8_scales.ptr
  q8_raw_out = d_q8_raw_out.ptr
  q8_raw_params[0] = pointerof(q8_raw_w).as(Void*)
  q8_raw_params[1] = pointerof(q8_raw_packs_ptr).as(Void*)
  q8_raw_params[2] = pointerof(q8_raw_input_scales_ptr).as(Void*)
  q8_raw_params[3] = pointerof(q8_raw_out).as(Void*)
  q8_raw_params[4] = pointerof(in_dim_u32).as(Void*)
  q8_raw_params[5] = pointerof(out_dim_u32).as(Void*)

  q8_quant_params = Pointer(Void*).malloc(4)
  q8_quant_x = d_x.ptr
  q8_quant_packs = d_q8_gpu_packs.ptr
  q8_quant_scales = d_q8_gpu_scales.ptr
  q8_quant_params[0] = pointerof(q8_quant_x).as(Void*)
  q8_quant_params[1] = pointerof(q8_quant_packs).as(Void*)
  q8_quant_params[2] = pointerof(q8_quant_scales).as(Void*)
  q8_quant_params[3] = pointerof(in_dim_u32).as(Void*)

  q8_gpu_params = Pointer(Void*).malloc(8)
  q8_gpu_w = d_raw.ptr
  q8_gpu_scales_ptr = d_scales.ptr
  q8_gpu_mins_ptr = d_mins.ptr
  q8_gpu_packs_ptr = d_q8_gpu_packs.ptr
  q8_gpu_input_scales_ptr = d_q8_gpu_scales.ptr
  q8_gpu_out_ptr = d_q8_gpu_out.ptr
  q8_gpu_params[0] = pointerof(q8_gpu_w).as(Void*)
  q8_gpu_params[1] = pointerof(q8_gpu_scales_ptr).as(Void*)
  q8_gpu_params[2] = pointerof(q8_gpu_mins_ptr).as(Void*)
  q8_gpu_params[3] = pointerof(q8_gpu_packs_ptr).as(Void*)
  q8_gpu_params[4] = pointerof(q8_gpu_input_scales_ptr).as(Void*)
  q8_gpu_params[5] = pointerof(q8_gpu_out_ptr).as(Void*)
  q8_gpu_params[6] = pointerof(in_dim_u32).as(Void*)
  q8_gpu_params[7] = pointerof(out_dim_u32).as(Void*)

  q8_raw_gpu_params = Pointer(Void*).malloc(6)
  q8_raw_gpu_w = d_raw.ptr
  q8_raw_gpu_packs_ptr = d_q8_gpu_packs.ptr
  q8_raw_gpu_input_scales_ptr = d_q8_gpu_scales.ptr
  q8_raw_gpu_out_ptr = d_q8_raw_gpu_out.ptr
  q8_raw_gpu_params[0] = pointerof(q8_raw_gpu_w).as(Void*)
  q8_raw_gpu_params[1] = pointerof(q8_raw_gpu_packs_ptr).as(Void*)
  q8_raw_gpu_params[2] = pointerof(q8_raw_gpu_input_scales_ptr).as(Void*)
  q8_raw_gpu_params[3] = pointerof(q8_raw_gpu_out_ptr).as(Void*)
  q8_raw_gpu_params[4] = pointerof(in_dim_u32).as(Void*)
  q8_raw_gpu_params[5] = pointerof(out_dim_u32).as(Void*)

  raw_ms = run_kernel(raw_fn, grid, block, raw_params, reps, warmup)
  scale_regs_ms = run_kernel(scale_regs_fn, grid, block, scale_regs_params, reps, warmup)
  meta_ms_gpu = run_kernel(meta_fn, grid, block, meta_params, reps, warmup)
  repack_ms_gpu = run_kernel(repack_fn, grid, block, repack_params, reps, warmup)
  f32_ms_gpu = run_kernel(f32_fn, grid, block, f32_params, reps, warmup)
  q8_ms_gpu = run_kernel(q8_fn, grid, block, q8_params, reps, warmup)
  q8_raw_ms_gpu = run_kernel(q8_raw_fn, grid, block, q8_raw_params, reps, warmup)
  q8_quant_grid = (in_dim // 32).to_u32
  q8_quant_ms_gpu = run_kernel(q8_quant_fn, q8_quant_grid, 32_u32, q8_quant_params, reps, warmup)
  ML::CUDA.launch!(q8_quant_fn, q8_quant_grid, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, q8_quant_params, "q8_quant_for_output")
  ML::CUDA.synchronize!("cuCtxSynchronize(q8_quant_for_output)")
  q8_gpu_ms = run_kernel(q8_fn, grid, block, q8_gpu_params, reps, warmup)
  q8_raw_gpu_ms = run_kernel(q8_raw_fn, grid, block, q8_raw_gpu_params, reps, warmup)

  raw_gpu = Array(Float32).new(out_dim, 0.0_f32)
  scale_regs_gpu = Array(Float32).new(out_dim, 0.0_f32)
  meta_gpu = Array(Float32).new(out_dim, 0.0_f32)
  repack_gpu = Array(Float32).new(out_dim, 0.0_f32)
  f32_gpu = Array(Float32).new(out_dim, 0.0_f32)
  q8_gpu = Array(Float32).new(out_dim, 0.0_f32)
  q8_raw_gpu = Array(Float32).new(out_dim, 0.0_f32)
  q8_gpu_quant = Array(Float32).new(out_dim, 0.0_f32)
  q8_raw_gpu_quant = Array(Float32).new(out_dim, 0.0_f32)
  q8_gpu_packs_host = Array(UInt32).new(q8_input.packs.size, 0_u32)
  q8_gpu_scales_host = Array(Float32).new(q8_input.scales.size, 0.0_f32)
  ML::CUDA.copy_dtoh!(raw_gpu.to_unsafe.as(Void*), d_raw_out.ptr, bytesize_f32(out_dim), "raw_out")
  ML::CUDA.copy_dtoh!(scale_regs_gpu.to_unsafe.as(Void*), d_scale_regs_out.ptr, bytesize_f32(out_dim), "scale_regs_out")
  ML::CUDA.copy_dtoh!(meta_gpu.to_unsafe.as(Void*), d_meta_out.ptr, bytesize_f32(out_dim), "meta_out")
  ML::CUDA.copy_dtoh!(repack_gpu.to_unsafe.as(Void*), d_repack_out.ptr, bytesize_f32(out_dim), "repack_out")
  ML::CUDA.copy_dtoh!(f32_gpu.to_unsafe.as(Void*), d_f32_out.ptr, bytesize_f32(out_dim), "f32_out")
  ML::CUDA.copy_dtoh!(q8_gpu.to_unsafe.as(Void*), d_q8_out.ptr, bytesize_f32(out_dim), "q8_out")
  ML::CUDA.copy_dtoh!(q8_raw_gpu.to_unsafe.as(Void*), d_q8_raw_out.ptr, bytesize_f32(out_dim), "q8_raw_out")
  ML::CUDA.copy_dtoh!(q8_gpu_quant.to_unsafe.as(Void*), d_q8_gpu_out.ptr, bytesize_f32(out_dim), "q8_gpu_quant_out")
  ML::CUDA.copy_dtoh!(q8_raw_gpu_quant.to_unsafe.as(Void*), d_q8_raw_gpu_out.ptr, bytesize_f32(out_dim), "q8_raw_gpu_quant_out")
  ML::CUDA.copy_dtoh!(q8_gpu_packs_host.to_unsafe.as(Void*), d_q8_gpu_packs.ptr, q8_input.packs.size.to_u64 * 4_u64, "q8_gpu_packs")
  ML::CUDA.copy_dtoh!(q8_gpu_scales_host.to_unsafe.as(Void*), d_q8_gpu_scales.ptr, bytesize_f32(q8_input.scales.size), "q8_gpu_scales")

  raw_max = max_abs_diff(raw_gpu, cpu)
  raw_cos = cosine(raw_gpu, cpu)
  scale_regs_max = max_abs_diff(scale_regs_gpu, cpu)
  scale_regs_cos = cosine(scale_regs_gpu, cpu)
  meta_max = max_abs_diff(meta_gpu, cpu)
  meta_cos = cosine(meta_gpu, cpu)
  repack_max = max_abs_diff(repack_gpu, cpu)
  repack_cos = cosine(repack_gpu, cpu)
  f32_max = max_abs_diff(f32_gpu, cpu)
  f32_cos = cosine(f32_gpu, cpu)
  q8_max = max_abs_diff(q8_gpu, cpu)
  q8_cos = cosine(q8_gpu, cpu)
  q8_raw_max = max_abs_diff(q8_raw_gpu, cpu)
  q8_raw_cos = cosine(q8_raw_gpu, cpu)
  q8_gpu_quant_max = max_abs_diff(q8_gpu_quant, cpu)
  q8_gpu_quant_cos = cosine(q8_gpu_quant, cpu)
  q8_raw_gpu_quant_max = max_abs_diff(q8_raw_gpu_quant, cpu)
  q8_raw_gpu_quant_cos = cosine(q8_raw_gpu_quant, cpu)
  q8_pack_mismatches = 0
  first_q8_pack_mismatch = ""
  q8_input.packs.each_with_index do |v, i|
    next if v == q8_gpu_packs_host[i]
    q8_pack_mismatches += 1
    first_q8_pack_mismatch = "#{i}:host=0x#{v.to_s(16)} gpu=0x#{q8_gpu_packs_host[i].to_s(16)}" if first_q8_pack_mismatch.empty?
  end
  q8_scale_max = max_abs_diff_f32(q8_input.scales, q8_gpu_scales_host)

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "tensor=#{tensor_name}"
  puts "shape=#{in_dim}x#{out_dim}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "raw_bytes=#{w_raw.size}"
  puts "meta_bytes=#{w_raw.size + repacked.scales.size * 4 + repacked.mins.size * 4}"
  puts "meta_ratio=#{((w_raw.size + repacked.scales.size * 4 + repacked.mins.size * 4).to_f64 / w_raw.size).round(3)}"
  puts "repacked_bytes=#{repacked.qvals.size + repacked.scales.size * 4 + repacked.mins.size * 4}"
  puts "repack_ratio=#{((repacked.qvals.size + repacked.scales.size * 4 + repacked.mins.size * 4).to_f64 / w_raw.size).round(3)}"
  puts "f32_bytes=#{f32_weights.size * 4}"
  puts "f32_ratio=#{((f32_weights.size * 4).to_f64 / w_raw.size).round(3)}"
  puts "q8_input_bytes=#{q8_input.packs.size * 4 + q8_input.scales.size * 4}"
  puts "host_repack_ms=#{repack_ms.round(3)}"
  puts "host_q8_pack_ms=#{q8_pack_ms.round(3)}"
  puts "host_f32_dequant_ms=#{f32_repack_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "raw_cuda_ms=#{raw_ms.round(4)}"
  puts "scale_regs_cuda_ms=#{scale_regs_ms.round(4)}"
  puts "meta_cuda_ms=#{meta_ms_gpu.round(4)}"
  puts "repacked_cuda_ms=#{repack_ms_gpu.round(4)}"
  puts "f32_cuda_ms=#{f32_ms_gpu.round(4)}"
  puts "q8_dp4a_cuda_ms=#{q8_ms_gpu.round(4)}"
  puts "q8_raw_dp4a_cuda_ms=#{q8_raw_ms_gpu.round(4)}"
  puts "q8_quant_cuda_ms=#{q8_quant_ms_gpu.round(4)}"
  puts "q8_dp4a_gpu_quant_cuda_ms=#{q8_gpu_ms.round(4)}"
  puts "q8_raw_dp4a_gpu_quant_cuda_ms=#{q8_raw_gpu_ms.round(4)}"
  puts "scale_regs_speedup=#{(raw_ms / scale_regs_ms).round(4)}"
  puts "meta_speedup=#{(raw_ms / meta_ms_gpu).round(4)}"
  puts "repacked_speedup=#{(raw_ms / repack_ms_gpu).round(4)}"
  puts "f32_speedup=#{(raw_ms / f32_ms_gpu).round(4)}"
  puts "q8_dp4a_speedup=#{(raw_ms / q8_ms_gpu).round(4)}"
  puts "q8_raw_dp4a_speedup=#{(raw_ms / q8_raw_ms_gpu).round(4)}"
  puts "q8_dp4a_oneuse_with_quant_speedup=#{(raw_ms / (q8_quant_ms_gpu + q8_gpu_ms)).round(4)}"
  puts "q8_dp4a_reuse2_with_quant_speedup=#{((2.0_f64 * raw_ms) / (q8_quant_ms_gpu + 2.0_f64 * q8_gpu_ms)).round(4)}"
  puts "q8_raw_dp4a_oneuse_with_quant_speedup=#{(raw_ms / (q8_quant_ms_gpu + q8_raw_gpu_ms)).round(4)}"
  puts "q8_raw_dp4a_reuse2_with_quant_speedup=#{((2.0_f64 * raw_ms) / (q8_quant_ms_gpu + 2.0_f64 * q8_raw_gpu_ms)).round(4)}"
  puts "raw_cos=#{raw_cos.round(8)}"
  puts "raw_max_diff=#{raw_max}"
  puts "scale_regs_cos=#{scale_regs_cos.round(8)}"
  puts "scale_regs_max_diff=#{scale_regs_max}"
  puts "meta_cos=#{meta_cos.round(8)}"
  puts "meta_max_diff=#{meta_max}"
  puts "repacked_cos=#{repack_cos.round(8)}"
  puts "repacked_max_diff=#{repack_max}"
  puts "f32_cos=#{f32_cos.round(8)}"
  puts "f32_max_diff=#{f32_max}"
  puts "q8_dp4a_cos=#{q8_cos.round(8)}"
  puts "q8_dp4a_max_diff=#{q8_max}"
  puts "q8_raw_dp4a_cos=#{q8_raw_cos.round(8)}"
  puts "q8_raw_dp4a_max_diff=#{q8_raw_max}"
  puts "q8_dp4a_gpu_quant_cos=#{q8_gpu_quant_cos.round(8)}"
  puts "q8_dp4a_gpu_quant_max_diff=#{q8_gpu_quant_max}"
  puts "q8_raw_dp4a_gpu_quant_cos=#{q8_raw_gpu_quant_cos.round(8)}"
  puts "q8_raw_dp4a_gpu_quant_max_diff=#{q8_raw_gpu_quant_max}"
  puts "q8_quant_pack_mismatches=#{q8_pack_mismatches}"
  puts "q8_quant_first_pack_mismatch=#{first_q8_pack_mismatch}"
  puts "q8_quant_scale_max_diff=#{q8_scale_max}"
  puts "q8_dp4a_note=approximate_input_quantization_upper_bound"
  puts "ok=#{raw_cos >= 0.99999 && raw_max <= 1.0e-3_f32 && scale_regs_cos >= 0.99999 && scale_regs_max <= 1.0e-3_f32 && meta_cos >= 0.99999 && meta_max <= 1.0e-3_f32 && repack_cos >= 0.99999 && repack_max <= 1.0e-3_f32 && f32_cos >= 0.99999 && f32_max <= 1.0e-3_f32}"
ensure
  buffers.each(&.close)
  modules.each(&.close)
  ctx.close
  gguf.close
end
