#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
GEMM_SOURCE   = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/gemm_mm.metal") }}
GEMV_SOURCE   = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/gemm_q56k.metal") }}
DELTA_SOURCE  = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/delta_net.metal") }}
FFN_SOURCE    = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/ffn_qwen35.metal") }}
REC_SOURCE    = {{ read_file("#{__DIR__}/../src/ml/gguf/kernels/recurrent_qwen35.metal") }}
MM_NR0        =    64
MM_NR1        =    32
MM_TG         =   128
MM_SHMEM      = 12288
MV_Q8_NSG     =     4
MV_Q8_NR0     =     1

record Q8GemvLayout, nsg : Int32, nr0 : Int32

record Op, name : String, qw : ML::GGUF::QuantWeight
record BenchOp, name : String, qw : ML::GGUF::QuantWeight, w_buf : ML::MetalBuffer, out_buf : ML::MetalBuffer
record Sample, wall_ms : Float64, gpu_ms : Float64
record Stats, p10 : Float64, p50 : Float64, p90 : Float64, avg : Float64

private def percentile(sorted : Array(Float64), pct : Float64) : Float64
  raise "empty sample" if sorted.empty?
  idx = ((sorted.size - 1) * pct).round.to_i.clamp(0, sorted.size - 1)
  sorted[idx]
end

private def stats(values : Array(Float64)) : Stats
  sorted = values.sort
  Stats.new(
    percentile(sorted, 0.10),
    percentile(sorted, 0.50),
    percentile(sorted, 0.90),
    values.sum / values.size
  )
end

private def fmt(v : Float64) : String
  "%.4f" % v
end

private def rec_proj_tilefast_source : String
  original = <<-METAL
    const uint row_bytes = (in_dim / 32) * 34;
    const int mixed_row = r0 + lr0;
    device const uint8_t *base_raw;
    int local_row;
    if (mixed_row < gate_start) {
        base_raw = qkv_w_raw;
        local_row = mixed_row;
    } else if (mixed_row < alpha_start) {
        base_raw = gate_w_raw;
        local_row = mixed_row - gate_start;
    } else if (mixed_row < beta_start) {
        base_raw = alpha_w_raw;
        local_row = mixed_row - alpha_start;
    } else {
        base_raw = beta_w_raw;
        local_row = mixed_row - beta_start;
    }
  METAL
  replacement = <<-METAL
    const uint row_bytes = (in_dim / 32) * 34;
    const int mixed_row = r0 + lr0;
    device const uint8_t *base_raw;
    int local_row;
    if (r0 + MM_NR0 <= gate_start) {
        base_raw = qkv_w_raw;
        local_row = mixed_row;
    } else if (r0 >= gate_start && r0 + MM_NR0 <= alpha_start) {
        base_raw = gate_w_raw;
        local_row = mixed_row - gate_start;
    } else if (mixed_row < gate_start) {
        base_raw = qkv_w_raw;
        local_row = mixed_row;
    } else if (mixed_row < alpha_start) {
        base_raw = gate_w_raw;
        local_row = mixed_row - gate_start;
    } else if (mixed_row < beta_start) {
        base_raw = alpha_w_raw;
        local_row = mixed_row - alpha_start;
    } else {
        base_raw = beta_w_raw;
        local_row = mixed_row - beta_start;
    }
  METAL
  source = GEMM_SOURCE.sub(
    "kernel void simd_mm_q8_0_f32in_f32out_rec_proj_mixed(",
    "kernel void simd_mm_q8_0_f32in_f32out_rec_proj_mixed_tilefast("
  ).sub(original, replacement)
  raise "failed to build rec_proj tilefast source" if source == GEMM_SOURCE || !source.includes?("rec_proj_mixed_tilefast")
  source
end

private def rec_proj_mixed_singlebuf_source : String
  source = GEMM_SOURCE.sub(
    "kernel void simd_mm_q8_0_f32in_f32out_rec_proj_mixed(",
    "kernel void simd_mm_q8_0_f32in_f32out_rec_proj_mixed_single("
  )
  start = source.index("kernel void simd_mm_q8_0_f32in_f32out_rec_proj_mixed_single(").not_nil!
  finish = source.index("\nkernel void simd_mm_q8_0_f32in_f32out_single(", start).not_nil!
  kernel = source[start...finish]

  single_decl = <<-METAL
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + MM_SA_SIZE);
  METAL
  decl_start = kernel.index("    threadgroup half * sa_buf[2]").not_nil!
  decl_finish = kernel.index("\n\n    const int gate_start", decl_start).not_nil!
  kernel = kernel[0...decl_start] + single_decl + kernel[decl_finish..]

  body_start = kernel.index("    {\n        threadgroup half * sa = sa_buf[0];").not_nil!
  body_finish = kernel.index("\n\n    if (r1 + MM_NR1", body_start).not_nil!
  single_body = <<-METAL
    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        half4x4 temp_a;
        dequantize_q8_0_fn(xw, il, temp_a);

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
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = (half2x4)(*(device const float2x4 *)y);
        }

        il = (il + 2 < MM8_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + 1 : xw;
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
  METAL
  kernel = kernel[0...body_start] + single_body + kernel[body_finish..]
  source = source[0...start] + kernel + source[finish..]
  raise "failed to build rec_proj mixed singlebuf source" unless source.includes?("rec_proj_mixed_single") && !kernel.includes?("sa_buf")
  source
end

private def no_sg_barrier_source : String
  source = GEMM_SOURCE.gsub("simdgroup_barrier(mem_flags::mem_none);", "")
  raise "failed to remove simdgroup barriers from Q8 source" if source == GEMM_SOURCE || source.includes?("simdgroup_barrier(mem_flags::mem_none);")
  source
end

private def q8_gemv_layout_source(layout : Q8GemvLayout) : String
  raise "nsg must be positive" unless layout.nsg > 0
  raise "nr0 must be positive" unless layout.nr0 > 0
  source = GEMV_SOURCE
    .gsub("constant short MV8_NSG = 4;", "constant short MV8_NSG = #{layout.nsg};")
    .gsub("constant short MV8_NR0 = 1;", "constant short MV8_NR0 = #{layout.nr0};")
  raise "failed to build Q8 GEMV layout source" if source == GEMV_SOURCE && (layout.nsg != MV_Q8_NSG || layout.nr0 != MV_Q8_NR0)
  source
end

private def two_output_tilefast_source : String
  original = <<-METAL
    const uint row_bytes = (in_dim / 32) * 34;
    const int mixed_row = r0 + lr0;
    device const uint8_t *base_raw = mixed_row < (int)qkv_dim ? qkv_w_raw : gate_w_raw;
    const int local_row = mixed_row < (int)qkv_dim ? mixed_row : mixed_row - (int)qkv_dim;
  METAL
  replacement = <<-METAL
    const uint row_bytes = (in_dim / 32) * 34;
    const int mixed_row = r0 + lr0;
    device const uint8_t *base_raw;
    int local_row;
    if (r0 + MM_NR0 <= (int)qkv_dim) {
        base_raw = qkv_w_raw;
        local_row = mixed_row;
    } else if (r0 >= (int)qkv_dim) {
        base_raw = gate_w_raw;
        local_row = mixed_row - (int)qkv_dim;
    } else if (mixed_row < (int)qkv_dim) {
        base_raw = qkv_w_raw;
        local_row = mixed_row;
    } else {
        base_raw = gate_w_raw;
        local_row = mixed_row - (int)qkv_dim;
    }
  METAL
  source = GEMM_SOURCE.sub(
    "kernel void simd_mm_q8_0_f32in_f32out_qkv_gate_mixed(",
    "kernel void simd_mm_q8_0_f32in_f32out_qkv_gate_mixed_tilefast("
  ).sub(original, replacement)
  raise "failed to build two-output tilefast source" if source == GEMM_SOURCE || !source.includes?("qkv_gate_mixed_tilefast")
  source
end

private def post_oproj_fused_source : String
  GEMM_SOURCE + <<-METAL

kernel void qwen35_dn_post_norm_gate_chunk_out(
    device const float* y        [[buffer(0)]],
    device const float* z        [[buffer(1)]],
    device const float* ssm_norm [[buffer(2)]],
    device       float* out      [[buffer(3)]],
    constant     uint&  h_v      [[buffer(4)]],
    constant     uint&  s        [[buffer(5)]],
    constant     float& eps      [[buffer(6)]],
    constant     uint&  n_tokens [[buffer(7)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= h_v || t >= n_tokens) return;

    device const float* Y = y + (t * h_v + h) * s;
    device const float* Z = z + (t * h_v + h) * s;
    device float* O = out + (t * h_v + h) * s;

    float ss = 0.0f;
    for (uint d = tiisg; d < s; d += 32) {
        const float v = Y[d];
        ss += v * v;
    }
    const float inv = rsqrt(simd_sum(ss) / float(s) + eps);

    for (uint d = tiisg; d < s; d += 32) {
        const float zv = Z[d];
        const float sig = 1.0f / (1.0f + exp(-zv));
        O[d] = Y[d] * inv * ssm_norm[d] * (zv * sig);
    }
}

kernel void qwen35_dn_post_inv_rms_rows(
    device const float* y        [[buffer(0)]],
    device       float* inv_rms  [[buffer(1)]],
    constant     uint&  h_v      [[buffer(2)]],
    constant     uint&  s        [[buffer(3)]],
    constant     float& eps      [[buffer(4)]],
    constant     uint&  n_tokens [[buffer(5)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]])
{
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    if (h >= h_v || t >= n_tokens) return;

    device const float* Y = y + (t * h_v + h) * s;
    float ss = 0.0f;
    for (uint d = tiisg; d < s; d += 32) {
        const float v = Y[d];
        ss += v * v;
    }
    inv_rms[t * h_v + h] = rsqrt(simd_sum(ss) / float(s) + eps);
}

kernel void simd_mm_q8_0_f32in_f32out_dn_post_oproj(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   y_raw   [[buffer(1)]],
    device const float*   z       [[buffer(2)]],
    device const float*   norm    [[buffer(3)]],
    device const float*   inv_rms [[buffer(4)]],
    device       float*   output  [[buffer(5)]],
    constant     uint&    in_dim  [[buffer(6)]],
    constant     uint&    out_dim [[buffer(7)]],
    constant     uint&    batch   [[buffer(8)]],
    constant     uint&    s       [[buffer(9)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
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

    const uint row_bytes = (in_dim / 32) * 34;
    device const block_q8_0 * xw = (device const block_q8_0 *)(w_raw + (r0 + lr0) * row_bytes);
    const short iy = 8 * (tiitg % MM_NL1);
    const uint token = r1 + lr1;
    const uint base_col = iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    {
        threadgroup half * sa = sa_buf[0];
        threadgroup half * sb = sb_buf[0];
        half4x4 temp_a;
        dequantize_q8_0_fn(xw, il, temp_a);
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
            half vals[8];
            FOR_UNROLL for (short k = 0; k < 8; ++k) {
                const uint col = base_col + k;
                const uint h = col / s;
                const uint d = col - h * s;
                const uint idx = token * in_dim + col;
                const float zv = z[idx];
                const float sig = 1.0f / (1.0f + exp(-zv));
                vals[k] = half(y_raw[idx] * inv_rms[token * (in_dim / s) + h] * norm[d] * (zv * sig));
            }
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = half2x4{
                half4(vals[0], vals[1], vals[2], vals[3]),
                half4(vals[4], vals[5], vals[6], vals[7])
            };
        }
        il = (il + 2 < MM8_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + 1 : xw;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    uint y_col_base = base_col + MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        const short cur = iter % 2;
        const short nxt = 1 - cur;
        threadgroup half * sa = sa_buf[cur];
        threadgroup half * sb = sb_buf[cur];
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
        if (iter + 1 < n_iter) {
            threadgroup half * sa_n = sa_buf[nxt];
            threadgroup half * sb_n = sb_buf[nxt];
            half4x4 temp_a;
            dequantize_q8_0_fn(xw, il, temp_a);
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
                half vals[8];
                FOR_UNROLL for (short k = 0; k < 8; ++k) {
                    const uint col = y_col_base + k;
                    const uint h = col / s;
                    const uint d = col - h * s;
                    const uint idx = token * in_dim + col;
                    const float zv = z[idx];
                    const float sig = 1.0f / (1.0f + exp(-zv));
                    vals[k] = half(y_raw[idx] * inv_rms[token * (in_dim / s) + h] * norm[d] * (zv * sig));
                }
                *(threadgroup half2x4 *)(sb_n + 64*(4*sx + sy) + 8*ly) = half2x4{
                    half4(vals[0], vals[1], vals[2], vals[3]),
                    half4(vals[4], vals[5], vals[6], vals[7])
                };
            }
            il = (il + 2 < MM8_NL) ? il + 2 : il % 2;
            xw = (il < 2) ? xw + 1 : xw;
            y_col_base += MM_NK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (r0 + MM_NR0 <= (int)out_dim && r1 + MM_NR1 <= (int)batch) {
        device float * C = output + r0 + 32*(sgitg & 1) + (r1 + 16*(sgitg >> 1)) * out_dim;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
        }
    } else {
        threadgroup float * temp = (threadgroup float *)shmem;
        {
            threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
            for (short i = 0; i < 8; i++) {
                simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int total_out = nr0 * nr1;
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            output[(r1 + j) * out_dim + r0 + i] = temp[j * MM_NR0 + i];
        }
    }
}
METAL
end

private def rec_conv_token_parallel_source : String
  REC_SOURCE + <<-METAL

kernel void qwen35_recurrent_conv_shift_chunk_token_parallel(
    device const float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    device const float* conv1d     [[buffer(2)]],
    device       float* q_out      [[buffer(3)]],
    device       float* k_out      [[buffer(4)]],
    device       float* v_out      [[buffer(5)]],
    constant     uint&  h_k        [[buffer(6)]],
    constant     uint&  h_v        [[buffer(7)]],
    constant     uint&  s          [[buffer(8)]],
    constant     uint&  conv_k     [[buffer(9)]],
    constant     uint&  n_tokens   [[buffer(10)]],
    uint gid [[thread_position_in_grid]])
{
    const uint qkv_dim = 2 * h_k * s + h_v * s;
    const uint total = n_tokens * qkv_dim;
    if (gid >= total) return;

    const uint tok = gid / qkv_dim;
    const uint d = gid - tok * qkv_dim;
    const uint w_base = d * conv_k;
    const int history = int(conv_k) - 1;

    float acc = 0.0f;
    for (uint kt = 0; kt < conv_k; ++kt) {
        const int src_pos = int(tok) + int(kt) - history;
        float x;
        if (src_pos < 0) {
            x = conv_state[(src_pos + history) * int(qkv_dim) + int(d)];
        } else {
            x = qkv_mixed[uint(src_pos) * qkv_dim + d];
        }
        acc += x * conv1d[w_base + kt];
    }

    const float sig = 1.0f / (1.0f + exp(-acc));
    const float val = acc * sig;

    const uint q_dim = h_k * s;
    const uint k_dim = h_k * s;
    if (d < q_dim) {
        q_out[tok * q_dim + d] = val;
    } else if (d < q_dim + k_dim) {
        k_out[tok * k_dim + d - q_dim] = val;
    } else {
        v_out[tok * h_v * s + d - q_dim - k_dim] = val;
    }
}

kernel void qwen35_recurrent_conv_shift_chunk_token_parallel_state(
    device       float* conv_state [[buffer(0)]],
    device const float* qkv_mixed  [[buffer(1)]],
    constant     uint&  qkv_dim    [[buffer(2)]],
    constant     uint&  conv_k     [[buffer(3)]],
    constant     uint&  n_tokens   [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint state_rows = conv_k - 1;
    const uint total = state_rows * qkv_dim;
    if (gid >= total) return;

    const uint row = gid / qkv_dim;
    const uint d = gid - row * qkv_dim;
    const int src_pos = int(n_tokens) + int(row) - int(state_rows);
    if (src_pos < 0) {
        conv_state[row * qkv_dim + d] = conv_state[uint(src_pos + int(state_rows)) * qkv_dim + d];
    } else {
        conv_state[row * qkv_dim + d] = qkv_mixed[uint(src_pos) * qkv_dim + d];
    }
}
METAL
end

private def ffn_down_fused_input_source : String
  GEMM_SOURCE + <<-METAL

kernel void simd_mm_q8_0_f32in_f32out_ffn_down_swiglu(
    device const uint8_t* w_raw   [[buffer(0)]],
    device const float*   gate    [[buffer(1)]],
    device const float*   up      [[buffer(2)]],
    device       float*   output  [[buffer(3)]],
    constant     uint&    in_dim  [[buffer(4)]],
    constant     uint&    out_dim [[buffer(5)]],
    constant     uint&    batch   [[buffer(6)]],
    threadgroup  char*    shmem   [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
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

    const uint row_bytes = (in_dim / 32) * 34;
    device const block_q8_0 * xw = (device const block_q8_0 *)(w_raw + (r0 + lr0) * row_bytes);
    const short iy = 8 * (tiitg % MM_NL1);
    const uint token = r1 + lr1;
    uint col_base = iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

    {
        threadgroup half * sa = sa_buf[0];
        threadgroup half * sb = sb_buf[0];
        half4x4 temp_a;
        dequantize_q8_0_fn(xw, il, temp_a);
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
            half vals[8];
            FOR_UNROLL for (short k = 0; k < 8; ++k) {
                const uint idx = token * in_dim + col_base + k;
                const float g = gate[idx];
                const float sig = 1.0f / (1.0f + exp(-g));
                vals[k] = half((g * sig) * up[idx]);
            }
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = half2x4{
                half4(vals[0], vals[1], vals[2], vals[3]),
                half4(vals[4], vals[5], vals[6], vals[7])
            };
        }
        il = (il + 2 < MM8_NL) ? il + 2 : il % 2;
        xw = (il < 2) ? xw + 1 : xw;
        col_base += MM_NK;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        const short cur = iter % 2;
        const short nxt = 1 - cur;
        threadgroup half * sa = sa_buf[cur];
        threadgroup half * sb = sb_buf[cur];
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
        if (iter + 1 < n_iter) {
            threadgroup half * sa_n = sa_buf[nxt];
            threadgroup half * sb_n = sb_buf[nxt];
            half4x4 temp_a;
            dequantize_q8_0_fn(xw, il, temp_a);
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
                half vals[8];
                FOR_UNROLL for (short k = 0; k < 8; ++k) {
                    const uint idx = token * in_dim + col_base + k;
                    const float g = gate[idx];
                    const float sig = 1.0f / (1.0f + exp(-g));
                    vals[k] = half((g * sig) * up[idx]);
                }
                *(threadgroup half2x4 *)(sb_n + 64*(4*sx + sy) + 8*ly) = half2x4{
                    half4(vals[0], vals[1], vals[2], vals[3]),
                    half4(vals[4], vals[5], vals[6], vals[7])
                };
            }
            il = (il + 2 < MM8_NL) ? il + 2 : il % 2;
            xw = (il < 2) ? xw + 1 : xw;
            col_base += MM_NK;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (r0 + MM_NR0 <= (int)out_dim && r1 + MM_NR1 <= (int)batch) {
        device float * C = output + r0 + 32*(sgitg & 1) + (r1 + 16*(sgitg >> 1)) * out_dim;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
        }
    } else {
        threadgroup float * temp = (threadgroup float *)shmem;
        {
            threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
            for (short i = 0; i < 8; i++) {
                simdgroup_store(mc[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const int total_out = nr0 * nr1;
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            output[(r1 + j) * out_dim + r0 + i] = temp[j * MM_NR0 + i];
        }
    }
}
METAL
end

private def ffn_upgate_pair_rows_source : String
  GEMM_SOURCE + <<-METAL

kernel void simd_mm_q8_0_f32in_f32out_ffn_upgate_pair_rows(
    device const uint8_t* gate_w_raw [[buffer(0)]],
    device const uint8_t* up_w_raw   [[buffer(1)]],
    device const float*   x          [[buffer(2)]],
    device       float*   gate_out   [[buffer(3)]],
    device       float*   up_out     [[buffer(4)]],
    constant     uint&    in_dim     [[buffer(5)]],
    constant     uint&    out_dim    [[buffer(6)]],
    constant     uint&    batch      [[buffer(7)]],
    threadgroup  char*    shmem      [[threadgroup(0)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + MM_SA_SIZE);

    const int r0 = tgpig.y * MM_NR0;
    const int r1 = tgpig.x * MM_NR1;
    const short nr0 = min(MM_NR0, (int)out_dim - r0);
    const short nr1 = min(MM_NR1, (int)batch - r1);
    const short lr0 = min((short)(tiitg/MM_NL0), (short)(nr0 - 1));
    const short lr1 = min((short)(tiitg/MM_NL1), (short)(nr1 - 1));
    const short il0 = tiitg % MM_NL0;
    short il_gate = il0;
    short il_up = il0;

    const uint row_bytes = (in_dim / 32) * 34;
    device const block_q8_0 * gate_w = (device const block_q8_0 *)(gate_w_raw + (r0 + lr0) * row_bytes);
    device const block_q8_0 * up_w = (device const block_q8_0 *)(up_w_raw + (r0 + lr0) * row_bytes);
    const short iy = 8 * (tiitg % MM_NL1);
    device const float * y = x + (r1 + lr1) * in_dim + iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];
    for (short i = 0; i < 8; i++) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    }

    const uint n_iter = (in_dim + MM_NK - 1) / MM_NK;
    for (uint iter = 0; iter < n_iter; iter++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            half4x4 temp_a;
            dequantize_q8_0_fn(gate_w, il_gate, temp_a);
            FOR_UNROLL for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/MM_NL0)/8;
                const short lx = (tiitg/MM_NL0)%8;
                const short ly = i%8;
                *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }
        {
            const short sx = (tiitg % MM_NL1);
            const short sy = (tiitg/MM_NL1)/8;
            const short ly = (tiitg/MM_NL1)%8;
            *(threadgroup half2x4 *)(sb + 64*(4*sx + sy) + 8*ly) = (half2x4)(*(device const float2x4 *)y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_gate = sa + 4*64*(sgitg % 2);
        threadgroup const half * lsmb_gate = sb + 2*64*(sgitg / 2);
        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma_gate + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb_gate + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc_gate[i], mb[i/4], ma[i%4], mc_gate[i]);
            lsma_gate += 8*64; lsmb_gate += 4*64;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            half4x4 temp_a;
            dequantize_q8_0_fn(up_w, il_up, temp_a);
            FOR_UNROLL for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/MM_NL0)/8;
                const short lx = (tiitg/MM_NL0)%8;
                const short ly = i%8;
                *(sa + 64*(8*sx + sy) + 8*ly + lx) = temp_a[i/4][i%4];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_up = sa + 4*64*(sgitg % 2);
        threadgroup const half * lsmb_up = sb + 2*64*(sgitg / 2);
        FOR_UNROLL for (short ik = 0; ik < MM_NK/8; ik++) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma_up + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb_up + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL for (short i = 0; i < 8; i++) simdgroup_multiply_accumulate(mc_up[i], mb[i/4], ma[i%4], mc_up[i]);
            lsma_up += 8*64; lsmb_up += 4*64;
        }

        il_gate = (il_gate + 2 < MM8_NL) ? il_gate + 2 : il_gate % 2;
        gate_w = (il_gate < 2) ? gate_w + 1 : gate_w;
        il_up = (il_up + 2 < MM8_NL) ? il_up + 2 : il_up % 2;
        up_w = (il_up < 2) ? up_w + 1 : up_w;
        y += MM_NK;
    }

    if (r0 + MM_NR0 <= (int)out_dim && r1 + MM_NR1 <= (int)batch) {
        device float * G = gate_out + r0 + 32*(sgitg & 1) + (r1 + 16*(sgitg >> 1)) * out_dim;
        device float * U = up_out + r0 + 32*(sgitg & 1) + (r1 + 16*(sgitg >> 1)) * out_dim;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc_gate[i], G + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
            simdgroup_store(mc_up[i], U + 8*(i%4) + 8*out_dim*(i/4), out_dim, 0, false);
        }
    } else {
        threadgroup float * temp = (threadgroup float *)shmem;
        {
            threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
            for (short i = 0; i < 8; i++) {
                simdgroup_store(mc_gate[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int total_out = nr0 * nr1;
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            gate_out[(r1 + j) * out_dim + r0 + i] = temp[j * MM_NR0 + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup float * sg_out = temp + 32*(sgitg & 1) + 16*(sgitg >> 1)*MM_NR0;
            for (short i = 0; i < 8; i++) {
                simdgroup_store(mc_up[i], sg_out + 8*(i%4) + 8*MM_NR0*(i/4), MM_NR0, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = (int)tiitg; idx < total_out; idx += 128) {
            const int i = idx % nr0;
            const int j = idx / nr0;
            up_out[(r1 + j) * out_dim + r0 + i] = temp[j * MM_NR0 + i];
        }
    }
}
METAL
end

private def f32_to_f16_bits(f : Float32) : UInt16
  bits = f.unsafe_as(UInt32)
  sign = (bits >> 16) & 0x8000_u32
  exp = ((bits >> 23) & 0xff).to_i32 - 127 + 15
  mant = (bits >> 13) & 0x03ff_u32
  if exp <= 0
    sign.to_u16
  elsif exp >= 31
    (sign | 0x7c00_u32).to_u16
  else
    (sign | (exp.to_u32 << 10) | mant).to_u16
  end
end

private def metal_buffer_from_u16(data : Array(UInt16)) : ML::MetalBuffer
  buf = ML::MetalBuffer.new(data.size.to_i64 * sizeof(UInt16))
  buf.write_bytes(data.to_unsafe.as(Pointer(UInt8)), data.size * sizeof(UInt16))
  buf
end

private def first_recurrent_layer(weights : ML::GGUF::Qwen35Weights) : ML::GGUF::Qwen35RecurrentWeights
  weights.layers.each do |layer|
    case layer
    when ML::GGUF::Qwen35RecurrentWeights
      return layer
    else
    end
  end
  raise "no recurrent layer found"
end

private def corridor_ops(name : String, rec : ML::GGUF::Qwen35RecurrentWeights) : Array(Op)
  case name
  when "rec_proj"
    [
      Op.new("rec.proj.qkv", rec.attn_qkv_qw),
      Op.new("rec.proj.gate", rec.attn_gate_qw),
      Op.new("rec.proj.alpha", rec.ssm_alpha_qw),
      Op.new("rec.proj.beta", rec.ssm_beta_qw),
    ]
  when "ffn_upgate"
    [
      Op.new("rec.ffn.gate", rec.ffn_gate_qw),
      Op.new("rec.ffn.up", rec.ffn_up_qw),
    ]
  else
    raise "unknown corridor #{name.inspect}; expected rec_proj, ffn_upgate, or all"
  end
end

private def upload_weights(qw : ML::GGUF::QuantWeight) : ML::MetalBuffer
  buf = ML::MetalBuffer.new(qw.raw.size.to_i64)
  buf.write_bytes(qw.raw.to_unsafe, qw.raw.size)
  buf
end

private def build_bench_ops(ops : Array(Op), batch : Int32) : Array(BenchOp)
  ops.map do |op|
    qw = op.qw
    raise "#{op.name}: expected Q8_0, got #{qw.type.name}" unless qw.type.q8_0?
    BenchOp.new(
      op.name,
      qw,
      upload_weights(qw),
      ML::MetalBuffer.new((batch * qw.out_dim).to_i64 * sizeof(Float32))
    )
  end
end

private def encode_q8_gemm(enc : ML::Metal::ComputeEncoder,
                           pipe : ML::Metal::ComputePipeline,
                           op : BenchOp,
                           x_buf : ML::MetalBuffer,
                           batch : Int32) : Nil
  qw = op.qw
  enc.set_pipeline(pipe)
  enc.set_buffer(op.w_buf, 0)
  enc.set_buffer(x_buf, 1)
  enc.set_buffer(op.out_buf, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(qw.in_dim.to_u32, 3)
  enc.set_value(qw.out_dim.to_u32, 4)
  enc.set_value(batch.to_u32, 5)
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (qw.out_dim + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def encode_q8_gemm_h16_input(enc : ML::Metal::ComputeEncoder,
                                     pipe : ML::Metal::ComputePipeline,
                                     op : BenchOp,
                                     x16_buf : ML::MetalBuffer,
                                     batch : Int32) : Nil
  qw = op.qw
  enc.set_pipeline(pipe)
  enc.set_buffer(op.w_buf, 0)
  enc.set_buffer(x16_buf, 1)
  enc.set_buffer(op.out_buf, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(qw.in_dim.to_u32, 3)
  enc.set_value(qw.out_dim.to_u32, 4)
  enc.set_value(batch.to_u32, 5)
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (qw.out_dim + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def run_corridor(ops : Array(BenchOp),
                         x_buf : ML::MetalBuffer,
                         batch : Int32,
                         pipe : ML::Metal::ComputePipeline) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    ops.each do |op|
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_q8_gemm(enc, pipe, op, x_buf, batch)
      enc.end_encoding
    end
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_corridor_h16_input(ops : Array(BenchOp),
                                   x16_buf : ML::MetalBuffer,
                                   batch : Int32,
                                   pipe : ML::Metal::ComputePipeline) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    ops.each do |op|
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_q8_gemm_h16_input(enc, pipe, op, x16_buf, batch)
      enc.end_encoding
    end
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def encode_dn_post_out(enc : ML::Metal::ComputeEncoder,
                               pipe : ML::Metal::ComputePipeline,
                               y_buf : ML::MetalBuffer,
                               z_buf : ML::MetalBuffer,
                               norm_buf : ML::MetalBuffer,
                               out_buf : ML::MetalBuffer,
                               h_v : Int32,
                               s : Int32,
                               batch : Int32,
                               eps : Float32) : Nil
  enc.set_pipeline(pipe)
  enc.set_buffer(y_buf, 0)
  enc.set_buffer(z_buf, 1)
  enc.set_buffer(norm_buf, 2)
  enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_value(h_v.to_u32, 4)
  enc.set_value(s.to_u32, 5)
  enc.set_value(eps, 6)
  enc.set_value(batch.to_u32, 7)
  enc.dispatch_threadgroups({h_v, batch, 1}, {32, 1, 1})
end

private def encode_dn_post_inv_rms(enc : ML::Metal::ComputeEncoder,
                                   pipe : ML::Metal::ComputePipeline,
                                   y_buf : ML::MetalBuffer,
                                   inv_buf : ML::MetalBuffer,
                                   h_v : Int32,
                                   s : Int32,
                                   batch : Int32,
                                   eps : Float32) : Nil
  enc.set_pipeline(pipe)
  enc.set_buffer(y_buf, 0)
  enc.set_buffer(inv_buf, 1, ML::Metal::BufferAccess::Write)
  enc.set_value(h_v.to_u32, 2)
  enc.set_value(s.to_u32, 3)
  enc.set_value(eps, 4)
  enc.set_value(batch.to_u32, 5)
  enc.dispatch_threadgroups({h_v, batch, 1}, {32, 1, 1})
end

private def encode_q8_post_oproj_fused(enc : ML::Metal::ComputeEncoder,
                                       pipe : ML::Metal::ComputePipeline,
                                       op : BenchOp,
                                       y_buf : ML::MetalBuffer,
                                       z_buf : ML::MetalBuffer,
                                       norm_buf : ML::MetalBuffer,
                                       inv_buf : ML::MetalBuffer,
                                       out_buf : ML::MetalBuffer,
                                       h_v : Int32,
                                       s : Int32,
                                       batch : Int32) : Nil
  qw = op.qw
  raise "post o_proj in_dim mismatch" unless qw.in_dim == h_v * s
  enc.set_pipeline(pipe)
  enc.set_buffer(op.w_buf, 0)
  enc.set_buffer(y_buf, 1)
  enc.set_buffer(z_buf, 2)
  enc.set_buffer(norm_buf, 3)
  enc.set_buffer(inv_buf, 4)
  enc.set_buffer(out_buf, 5, ML::Metal::BufferAccess::Write)
  enc.set_value(qw.in_dim.to_u32, 6)
  enc.set_value(qw.out_dim.to_u32, 7)
  enc.set_value(batch.to_u32, 8)
  enc.set_value(s.to_u32, 9)
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (qw.out_dim + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def encode_swiglu(enc : ML::Metal::ComputeEncoder,
                          pipe : ML::Metal::ComputePipeline,
                          gate_buf : ML::MetalBuffer,
                          up_buf : ML::MetalBuffer,
                          out_buf : ML::MetalBuffer,
                          count : Int32) : Nil
  enc.set_pipeline(pipe)
  enc.set_buffer(gate_buf, 0)
  enc.set_buffer(up_buf, 1)
  enc.set_buffer(out_buf, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(count.to_u32, 3)
  enc.dispatch_1d(count, 256)
end

private def encode_q8_ffn_down_swiglu(enc : ML::Metal::ComputeEncoder,
                                      pipe : ML::Metal::ComputePipeline,
                                      op : BenchOp,
                                      gate_buf : ML::MetalBuffer,
                                      up_buf : ML::MetalBuffer,
                                      out_buf : ML::MetalBuffer,
                                      batch : Int32) : Nil
  qw = op.qw
  enc.set_pipeline(pipe)
  enc.set_buffer(op.w_buf, 0)
  enc.set_buffer(gate_buf, 1)
  enc.set_buffer(up_buf, 2)
  enc.set_buffer(out_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_value(qw.in_dim.to_u32, 4)
  enc.set_value(qw.out_dim.to_u32, 5)
  enc.set_value(batch.to_u32, 6)
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (qw.out_dim + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def encode_q8_ffn_upgate_pair_rows(enc : ML::Metal::ComputeEncoder,
                                           pipe : ML::Metal::ComputePipeline,
                                           gate : BenchOp,
                                           up : BenchOp,
                                           x_buf : ML::MetalBuffer,
                                           batch : Int32) : Nil
  gate_qw = gate.qw
  up_qw = up.qw
  raise "gate/up in_dim mismatch" unless gate_qw.in_dim == up_qw.in_dim
  raise "gate/up out_dim mismatch" unless gate_qw.out_dim == up_qw.out_dim
  enc.set_pipeline(pipe)
  enc.set_buffer(gate.w_buf, 0)
  enc.set_buffer(up.w_buf, 1)
  enc.set_buffer(x_buf, 2)
  enc.set_buffer(gate.out_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_buffer(up.out_buf, 4, ML::Metal::BufferAccess::Write)
  enc.set_value(gate_qw.in_dim.to_u32, 5)
  enc.set_value(gate_qw.out_dim.to_u32, 6)
  enc.set_value(batch.to_u32, 7)
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (gate_qw.out_dim + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def run_ffn_upgate_pair_rows(pair : Array(BenchOp),
                                     x_buf : ML::MetalBuffer,
                                     batch : Int32,
                                     pipe : ML::Metal::ComputePipeline) : Sample
  raise "expected gate/up pair" unless pair.size == 2
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_ffn_upgate_pair_rows(enc, pipe, pair[0], pair[1], x_buf, batch)
    enc.end_encoding
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_post_oproj_default(op : BenchOp,
                                   y_buf : ML::MetalBuffer,
                                   z_buf : ML::MetalBuffer,
                                   norm_buf : ML::MetalBuffer,
                                   post_buf : ML::MetalBuffer,
                                   batch : Int32,
                                   h_v : Int32,
                                   s : Int32,
                                   eps : Float32,
                                   post_pipe : ML::Metal::ComputePipeline,
                                   q8_pipe : ML::Metal::ComputePipeline) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    post_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_dn_post_out(post_enc, post_pipe, y_buf, z_buf, norm_buf, post_buf, h_v, s, batch, eps)
    post_enc.end_encoding

    gemm_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_gemm(gemm_enc, q8_pipe, op, post_buf, batch)
    gemm_enc.end_encoding

    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_post_oproj_fused(op : BenchOp,
                                 y_buf : ML::MetalBuffer,
                                 z_buf : ML::MetalBuffer,
                                 norm_buf : ML::MetalBuffer,
                                 inv_buf : ML::MetalBuffer,
                                 fused_out_buf : ML::MetalBuffer,
                                 batch : Int32,
                                 h_v : Int32,
                                 s : Int32,
                                 eps : Float32,
                                 inv_pipe : ML::Metal::ComputePipeline,
                                 fused_pipe : ML::Metal::ComputePipeline) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    inv_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_dn_post_inv_rms(inv_enc, inv_pipe, y_buf, inv_buf, h_v, s, batch, eps)
    inv_enc.end_encoding

    gemm_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_post_oproj_fused(gemm_enc, fused_pipe, op, y_buf, z_buf, norm_buf, inv_buf, fused_out_buf, h_v, s, batch)
    gemm_enc.end_encoding

    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_ffn_down_default(upgate_ops : Array(BenchOp),
                                 down_op : BenchOp,
                                 x_buf : ML::MetalBuffer,
                                 act_buf : ML::MetalBuffer,
                                 batch : Int32,
                                 upgate_pipe : ML::Metal::ComputePipeline,
                                 swiglu_pipe : ML::Metal::ComputePipeline,
                                 q8_pipe : ML::Metal::ComputePipeline) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    upgate_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_qkv_gate_mixed(upgate_enc, upgate_pipe, upgate_ops[0], upgate_ops[1], x_buf, batch)
    upgate_enc.end_encoding

    swiglu_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_swiglu(swiglu_enc, swiglu_pipe, upgate_ops[0].out_buf, upgate_ops[1].out_buf, act_buf, batch * upgate_ops[0].qw.out_dim)
    swiglu_enc.end_encoding

    down_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_gemm(down_enc, q8_pipe, down_op, act_buf, batch)
    down_enc.end_encoding

    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_ffn_down_fused(upgate_ops : Array(BenchOp),
                               down_op : BenchOp,
                               x_buf : ML::MetalBuffer,
                               fused_out_buf : ML::MetalBuffer,
                               batch : Int32,
                               upgate_pipe : ML::Metal::ComputePipeline,
                               fused_down_pipe : ML::Metal::ComputePipeline) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    upgate_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_qkv_gate_mixed(upgate_enc, upgate_pipe, upgate_ops[0], upgate_ops[1], x_buf, batch)
    upgate_enc.end_encoding

    down_enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_ffn_down_swiglu(down_enc, fused_down_pipe, down_op, upgate_ops[0].out_buf, upgate_ops[1].out_buf, fused_out_buf, batch)
    down_enc.end_encoding

    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def encode_q8_dual_gemv(enc : ML::Metal::ComputeEncoder,
                                pipe : ML::Metal::ComputePipeline,
                                alpha : BenchOp,
                                beta : BenchOp,
                                x_buf : ML::MetalBuffer,
                                batch : Int32) : Nil
  aqw = alpha.qw
  bqw = beta.qw
  raise "alpha/beta in_dim mismatch" unless aqw.in_dim == bqw.in_dim
  raise "alpha/beta out_dim mismatch" unless aqw.out_dim == bqw.out_dim
  enc.set_pipeline(pipe)
  enc.set_buffer(alpha.w_buf, 0)
  enc.set_buffer(beta.w_buf, 1)
  enc.set_buffer(x_buf, 2)
  enc.set_buffer(alpha.out_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_buffer(beta.out_buf, 4, ML::Metal::BufferAccess::Write)
  enc.set_value(aqw.in_dim.to_u32, 5)
  enc.set_value(aqw.out_dim.to_u32, 6)
  enc.set_value(batch.to_u32, 7)
  rows_per_tg = MV_Q8_NSG * MV_Q8_NR0
  grid = {
    (aqw.out_dim + rows_per_tg - 1) // rows_per_tg,
    batch,
    1,
  }
  enc.dispatch_threadgroups(grid, {MV_Q8_NSG * 32, 1, 1})
end

private def run_alpha_beta_dual(ops : Array(BenchOp),
                                x_buf : ML::MetalBuffer,
                                batch : Int32,
                                pipe : ML::Metal::ComputePipeline) : Sample
  raise "expected alpha/beta pair" unless ops.size == 2
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_dual_gemv(enc, pipe, ops[0], ops[1], x_buf, batch)
    enc.end_encoding
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def encode_q8_gemv_layout(enc : ML::Metal::ComputeEncoder,
                                  pipe : ML::Metal::ComputePipeline,
                                  op : BenchOp,
                                  x_buf : ML::MetalBuffer,
                                  batch : Int32,
                                  layout : Q8GemvLayout) : Nil
  qw = op.qw
  enc.set_pipeline(pipe)
  enc.set_buffer(op.w_buf, 0)
  enc.set_buffer(x_buf, 1)
  enc.set_buffer(op.out_buf, 2, ML::Metal::BufferAccess::Write)
  enc.set_value(qw.in_dim.to_u32, 3)
  enc.set_value(qw.out_dim.to_u32, 4)
  enc.set_value(batch.to_u32, 5)
  rows_per_tg = layout.nsg * layout.nr0
  grid = {
    (qw.out_dim + rows_per_tg - 1) // rows_per_tg,
    batch,
    1,
  }
  enc.dispatch_threadgroups(grid, {layout.nsg * 32, 1, 1})
end

private def run_q8_gemv_layout(ops : Array(BenchOp),
                               x_buf : ML::MetalBuffer,
                               batch : Int32,
                               pipe : ML::Metal::ComputePipeline,
                               layout : Q8GemvLayout) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    ops.each do |op|
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_q8_gemv_layout(enc, pipe, op, x_buf, batch, layout)
      enc.end_encoding
    end
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def encode_q8_qkv_gate_mixed(enc : ML::Metal::ComputeEncoder,
                                     pipe : ML::Metal::ComputePipeline,
                                     qkv : BenchOp,
                                     gate : BenchOp,
                                     x_buf : ML::MetalBuffer,
                                     batch : Int32) : Nil
  qkv_qw = qkv.qw
  gate_qw = gate.qw
  raise "qkv/gate in_dim mismatch" unless qkv_qw.in_dim == gate_qw.in_dim
  enc.set_pipeline(pipe)
  enc.set_buffer(qkv.w_buf, 0)
  enc.set_buffer(gate.w_buf, 1)
  enc.set_buffer(x_buf, 2)
  enc.set_buffer(qkv.out_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_buffer(gate.out_buf, 4, ML::Metal::BufferAccess::Write)
  enc.set_value(qkv_qw.in_dim.to_u32, 5)
  enc.set_value(qkv_qw.out_dim.to_u32, 6)
  enc.set_value(gate_qw.out_dim.to_u32, 7)
  enc.set_value(batch.to_u32, 8)
  total_out = qkv_qw.out_dim + gate_qw.out_dim
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (total_out + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def run_qkv_gate_mixed(ops : Array(BenchOp),
                               x_buf : ML::MetalBuffer,
                               batch : Int32,
                               pipe : ML::Metal::ComputePipeline) : Sample
  raise "expected qkv/gate pair" unless ops.size == 2
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_qkv_gate_mixed(enc, pipe, ops[0], ops[1], x_buf, batch)
    enc.end_encoding
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def encode_q8_rec_proj_mixed(enc : ML::Metal::ComputeEncoder,
                                     pipe : ML::Metal::ComputePipeline,
                                     qkv : BenchOp,
                                     gate : BenchOp,
                                     alpha : BenchOp,
                                     beta : BenchOp,
                                     x_buf : ML::MetalBuffer,
                                     batch : Int32) : Nil
  qkv_qw = qkv.qw
  gate_qw = gate.qw
  alpha_qw = alpha.qw
  beta_qw = beta.qw
  raise "rec-proj in_dim mismatch" unless qkv_qw.in_dim == gate_qw.in_dim && qkv_qw.in_dim == alpha_qw.in_dim && qkv_qw.in_dim == beta_qw.in_dim
  enc.set_pipeline(pipe)
  enc.set_buffer(qkv.w_buf, 0)
  enc.set_buffer(gate.w_buf, 1)
  enc.set_buffer(alpha.w_buf, 2)
  enc.set_buffer(beta.w_buf, 3)
  enc.set_buffer(x_buf, 4)
  enc.set_buffer(qkv.out_buf, 5, ML::Metal::BufferAccess::Write)
  enc.set_buffer(gate.out_buf, 6, ML::Metal::BufferAccess::Write)
  enc.set_buffer(alpha.out_buf, 7, ML::Metal::BufferAccess::Write)
  enc.set_buffer(beta.out_buf, 8, ML::Metal::BufferAccess::Write)
  enc.set_value(qkv_qw.in_dim.to_u32, 9)
  enc.set_value(qkv_qw.out_dim.to_u32, 10)
  enc.set_value(gate_qw.out_dim.to_u32, 11)
  enc.set_value(alpha_qw.out_dim.to_u32, 12)
  enc.set_value(beta_qw.out_dim.to_u32, 13)
  enc.set_value(batch.to_u32, 14)
  total_out = qkv_qw.out_dim + gate_qw.out_dim + alpha_qw.out_dim + beta_qw.out_dim
  grid = {
    (batch + MM_NR1 - 1) // MM_NR1,
    (total_out + MM_NR0 - 1) // MM_NR0,
    1,
  }
  enc.set_threadgroup_memory(MM_SHMEM, 0)
  enc.dispatch_threadgroups(grid, {MM_TG, 1, 1})
end

private def run_rec_proj_mixed(ops : Array(BenchOp),
                               x_buf : ML::MetalBuffer,
                               batch : Int32,
                               pipe : ML::Metal::ComputePipeline) : Sample
  raise "expected full rec-proj quad" unless ops.size == 4
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_q8_rec_proj_mixed(enc, pipe, ops[0], ops[1], ops[2], ops[3], x_buf, batch)
    enc.end_encoding
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def read_outputs(ops : Array(BenchOp), batch : Int32) : Array(Array(Float32))
  ops.map { |op| op.out_buf.read(batch * op.qw.out_dim) }
end

private def max_delta(default_outputs : Array(Array(Float32)), other_outputs : Array(Array(Float32))) : Float64
  max = 0.0_f64
  default_outputs.each_with_index do |default_out, op_i|
    other = other_outputs[op_i]
    default_out.each_with_index do |v, i|
      d = (v - other[i]).to_f64.abs
      max = d if d > max
    end
  end
  max
end

private def checksum(outputs : Array(Array(Float32))) : Float64
  sum = 0.0_f64
  outputs.each do |result|
    sum += result[0].to_f64
    sum += result[result.size // 2].to_f64
    sum += result[-1].to_f64
  end
  sum
end

private def run_h16_input_probe(label : String,
                                bench_ops : Array(BenchOp),
                                x_buf : ML::MetalBuffer,
                                x16_buf : ML::MetalBuffer,
                                batch : Int32,
                                f32_pipe : ML::Metal::ComputePipeline,
                                h16_pipe : ML::Metal::ComputePipeline,
                                warmup : Int32,
                                runs : Int32,
                                indent : String = "  ") : Nil
  run_corridor(bench_ops, x_buf, batch, f32_pipe)
  f32_outputs = read_outputs(bench_ops, batch)
  run_corridor_h16_input(bench_ops, x16_buf, batch, h16_pipe)
  h16_outputs = read_outputs(bench_ops, batch)
  delta = max_delta(f32_outputs, h16_outputs)
  f32_checksum = checksum(f32_outputs)
  h16_checksum = checksum(h16_outputs)

  f32_wall = [] of Float64
  h16_wall = [] of Float64
  f32_gpu = [] of Float64
  h16_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      f = run_corridor(bench_ops, x_buf, batch, f32_pipe)
      h = run_corridor_h16_input(bench_ops, x16_buf, batch, h16_pipe)
    else
      h = run_corridor_h16_input(bench_ops, x16_buf, batch, h16_pipe)
      f = run_corridor(bench_ops, x_buf, batch, f32_pipe)
    end

    if i >= warmup
      f32_wall << f.wall_ms
      f32_gpu << f.gpu_ms
      h16_wall << h.wall_ms
      h16_gpu << h.gpu_ms
    end
  end

  fw = stats(f32_wall)
  hw = stats(h16_wall)
  fg = stats(f32_gpu)
  hg = stats(h16_gpu)
  puts "#{indent}bench=#{label} ops=#{bench_ops.size}"
  puts "#{indent}  f32_input_wall_ms p10=#{fmt(fw.p10)} p50=#{fmt(fw.p50)} p90=#{fmt(fw.p90)} avg=#{fmt(fw.avg)}"
  puts "#{indent}  h16_input_wall_ms p10=#{fmt(hw.p10)} p50=#{fmt(hw.p50)} p90=#{fmt(hw.p90)} avg=#{fmt(hw.avg)} speedup=#{fmt(fw.p50 / hw.p50)}x"
  puts "#{indent}  f32_input_gpu_ms  p10=#{fmt(fg.p10)} p50=#{fmt(fg.p50)} p90=#{fmt(fg.p90)} avg=#{fmt(fg.avg)}"
  puts "#{indent}  h16_input_gpu_ms  p10=#{fmt(hg.p10)} p50=#{fmt(hg.p50)} p90=#{fmt(hg.p90)} avg=#{fmt(hg.avg)} speedup=#{fmt(fg.p50 / hg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_f32=#{fmt(f32_checksum)} checksum_h16=#{fmt(h16_checksum)}"
end

private def run_paired(label : String,
                       bench_ops : Array(BenchOp),
                       x_buf : ML::MetalBuffer,
                       batch : Int32,
                       default_pipe : ML::Metal::ComputePipeline,
                       single_pipe : ML::Metal::ComputePipeline,
                       warmup : Int32,
                       runs : Int32,
                       indent : String = "  ") : Nil
  run_corridor(bench_ops, x_buf, batch, default_pipe)
  default_outputs = read_outputs(bench_ops, batch)
  run_corridor(bench_ops, x_buf, batch, single_pipe)
  single_outputs = read_outputs(bench_ops, batch)
  delta = max_delta(default_outputs, single_outputs)
  default_checksum = checksum(default_outputs)
  single_checksum = checksum(single_outputs)

  default_wall = [] of Float64
  single_wall = [] of Float64
  default_gpu = [] of Float64
  single_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      d = run_corridor(bench_ops, x_buf, batch, default_pipe)
      s = run_corridor(bench_ops, x_buf, batch, single_pipe)
    else
      s = run_corridor(bench_ops, x_buf, batch, single_pipe)
      d = run_corridor(bench_ops, x_buf, batch, default_pipe)
    end

    if i >= warmup
      default_wall << d.wall_ms
      default_gpu << d.gpu_ms
      single_wall << s.wall_ms
      single_gpu << s.gpu_ms
    end
  end

  dw = stats(default_wall)
  sw = stats(single_wall)
  dg = stats(default_gpu)
  sg = stats(single_gpu)
  puts "#{indent}#{label}"
  puts "#{indent}  default_wall_ms p10=#{fmt(dw.p10)} p50=#{fmt(dw.p50)} p90=#{fmt(dw.p90)} avg=#{fmt(dw.avg)}"
  puts "#{indent}  single_wall_ms  p10=#{fmt(sw.p10)} p50=#{fmt(sw.p50)} p90=#{fmt(sw.p90)} avg=#{fmt(sw.avg)} speedup=#{fmt(dw.p50 / sw.p50)}x"
  puts "#{indent}  default_gpu_ms  p10=#{fmt(dg.p10)} p50=#{fmt(dg.p50)} p90=#{fmt(dg.p90)} avg=#{fmt(dg.avg)}"
  puts "#{indent}  single_gpu_ms   p10=#{fmt(sg.p10)} p50=#{fmt(sg.p50)} p90=#{fmt(sg.p90)} avg=#{fmt(sg.avg)} speedup=#{fmt(dg.p50 / sg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_default=#{fmt(default_checksum)} checksum_single=#{fmt(single_checksum)}"
end

private def run_alpha_beta_dual_probe(bench_ops : Array(BenchOp),
                                      x_buf : ML::MetalBuffer,
                                      batch : Int32,
                                      default_pipe : ML::Metal::ComputePipeline,
                                      dual_pipe : ML::Metal::ComputePipeline,
                                      warmup : Int32,
                                      runs : Int32,
                                      indent : String = "  ") : Nil
  alpha_beta = bench_ops.select { |op| op.name == "rec.proj.alpha" || op.name == "rec.proj.beta" }
  raise "rec_proj alpha/beta pair not found" unless alpha_beta.size == 2

  run_corridor(alpha_beta, x_buf, batch, default_pipe)
  separate_outputs = read_outputs(alpha_beta, batch)
  run_alpha_beta_dual(alpha_beta, x_buf, batch, dual_pipe)
  dual_outputs = read_outputs(alpha_beta, batch)
  delta = max_delta(separate_outputs, dual_outputs)
  separate_checksum = checksum(separate_outputs)
  dual_checksum = checksum(dual_outputs)

  separate_wall = [] of Float64
  dual_wall = [] of Float64
  separate_gpu = [] of Float64
  dual_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      s = run_corridor(alpha_beta, x_buf, batch, default_pipe)
      d = run_alpha_beta_dual(alpha_beta, x_buf, batch, dual_pipe)
    else
      d = run_alpha_beta_dual(alpha_beta, x_buf, batch, dual_pipe)
      s = run_corridor(alpha_beta, x_buf, batch, default_pipe)
    end

    if i >= warmup
      separate_wall << s.wall_ms
      separate_gpu << s.gpu_ms
      dual_wall << d.wall_ms
      dual_gpu << d.gpu_ms
    end
  end

  sw = stats(separate_wall)
  dw = stats(dual_wall)
  sg = stats(separate_gpu)
  dg = stats(dual_gpu)
  puts "#{indent}bench=alpha_beta_dual_gemv ops=2"
  puts "#{indent}  separate_gemm_wall_ms p10=#{fmt(sw.p10)} p50=#{fmt(sw.p50)} p90=#{fmt(sw.p90)} avg=#{fmt(sw.avg)}"
  puts "#{indent}  dual_gemv_wall_ms     p10=#{fmt(dw.p10)} p50=#{fmt(dw.p50)} p90=#{fmt(dw.p90)} avg=#{fmt(dw.avg)} speedup=#{fmt(sw.p50 / dw.p50)}x"
  puts "#{indent}  separate_gemm_gpu_ms  p10=#{fmt(sg.p10)} p50=#{fmt(sg.p50)} p90=#{fmt(sg.p90)} avg=#{fmt(sg.avg)}"
  puts "#{indent}  dual_gemv_gpu_ms      p10=#{fmt(dg.p10)} p50=#{fmt(dg.p50)} p90=#{fmt(dg.p90)} avg=#{fmt(dg.avg)} speedup=#{fmt(sg.p50 / dg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_separate=#{fmt(separate_checksum)} checksum_dual=#{fmt(dual_checksum)}"
end

private def run_q8_gemv_layout_sweep_probe(bench_ops : Array(BenchOp),
                                           x_buf : ML::MetalBuffer,
                                           batch : Int32,
                                           base_pipe : ML::Metal::ComputePipeline,
                                           variant_pipes : Array({Q8GemvLayout, ML::Metal::ComputePipeline}),
                                           warmup : Int32,
                                           runs : Int32,
                                           indent : String = "  ") : Nil
  base_layout = Q8GemvLayout.new(MV_Q8_NSG, MV_Q8_NR0)
  run_q8_gemv_layout(bench_ops, x_buf, batch, base_pipe, base_layout)
  base_outputs = read_outputs(bench_ops, batch)
  base_checksum = checksum(base_outputs)

  puts "#{indent}bench=q8_gemv_layout_sweep ops=#{bench_ops.size} base_nsg=#{base_layout.nsg} base_nr0=#{base_layout.nr0}"

  variant_pipes.each do |layout, pipe|
    run_q8_gemv_layout(bench_ops, x_buf, batch, pipe, layout)
    variant_outputs = read_outputs(bench_ops, batch)
    delta = max_delta(base_outputs, variant_outputs)
    variant_checksum = checksum(variant_outputs)

    base_wall = [] of Float64
    variant_wall = [] of Float64
    base_gpu = [] of Float64
    variant_gpu = [] of Float64

    (warmup + runs).times do |i|
      if i.even?
        b = run_q8_gemv_layout(bench_ops, x_buf, batch, base_pipe, base_layout)
        v = run_q8_gemv_layout(bench_ops, x_buf, batch, pipe, layout)
      else
        v = run_q8_gemv_layout(bench_ops, x_buf, batch, pipe, layout)
        b = run_q8_gemv_layout(bench_ops, x_buf, batch, base_pipe, base_layout)
      end

      if i >= warmup
        base_wall << b.wall_ms
        base_gpu << b.gpu_ms
        variant_wall << v.wall_ms
        variant_gpu << v.gpu_ms
      end
    end

    bw = stats(base_wall)
    vw = stats(variant_wall)
    bg = stats(base_gpu)
    vg = stats(variant_gpu)
    puts "#{indent}  layout nsg=#{layout.nsg} nr0=#{layout.nr0}"
    puts "#{indent}    base_wall_ms    p10=#{fmt(bw.p10)} p50=#{fmt(bw.p50)} p90=#{fmt(bw.p90)} avg=#{fmt(bw.avg)}"
    puts "#{indent}    variant_wall_ms p10=#{fmt(vw.p10)} p50=#{fmt(vw.p50)} p90=#{fmt(vw.p90)} avg=#{fmt(vw.avg)} speedup=#{fmt(bw.p50 / vw.p50)}x"
    puts "#{indent}    base_gpu_ms     p10=#{fmt(bg.p10)} p50=#{fmt(bg.p50)} p90=#{fmt(bg.p90)} avg=#{fmt(bg.avg)}"
    puts "#{indent}    variant_gpu_ms  p10=#{fmt(vg.p10)} p50=#{fmt(vg.p50)} p90=#{fmt(vg.p90)} avg=#{fmt(vg.avg)} speedup=#{fmt(bg.p50 / vg.p50)}x"
    puts "#{indent}    max_delta=#{fmt(delta)} checksum_base=#{fmt(base_checksum)} checksum_variant=#{fmt(variant_checksum)}"
  end
end

private def run_two_output_mixed_probe(label : String,
                                       pair : Array(BenchOp),
                                       x_buf : ML::MetalBuffer,
                                       batch : Int32,
                                       default_pipe : ML::Metal::ComputePipeline,
                                       mixed_pipe : ML::Metal::ComputePipeline,
                                       warmup : Int32,
                                       runs : Int32,
                                       indent : String = "  ") : Nil
  raise "#{label}: expected two operations" unless pair.size == 2

  run_corridor(pair, x_buf, batch, default_pipe)
  separate_outputs = read_outputs(pair, batch)
  run_qkv_gate_mixed(pair, x_buf, batch, mixed_pipe)
  mixed_outputs = read_outputs(pair, batch)
  delta = max_delta(separate_outputs, mixed_outputs)
  separate_checksum = checksum(separate_outputs)
  mixed_checksum = checksum(mixed_outputs)

  separate_wall = [] of Float64
  mixed_wall = [] of Float64
  separate_gpu = [] of Float64
  mixed_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      s = run_corridor(pair, x_buf, batch, default_pipe)
      m = run_qkv_gate_mixed(pair, x_buf, batch, mixed_pipe)
    else
      m = run_qkv_gate_mixed(pair, x_buf, batch, mixed_pipe)
      s = run_corridor(pair, x_buf, batch, default_pipe)
    end

    if i >= warmup
      separate_wall << s.wall_ms
      separate_gpu << s.gpu_ms
      mixed_wall << m.wall_ms
      mixed_gpu << m.gpu_ms
    end
  end

  sw = stats(separate_wall)
  mw = stats(mixed_wall)
  sg = stats(separate_gpu)
  mg = stats(mixed_gpu)
  puts "#{indent}bench=#{label} ops=2"
  puts "#{indent}  separate_gemm_wall_ms p10=#{fmt(sw.p10)} p50=#{fmt(sw.p50)} p90=#{fmt(sw.p90)} avg=#{fmt(sw.avg)}"
  puts "#{indent}  mixed_wall_ms        p10=#{fmt(mw.p10)} p50=#{fmt(mw.p50)} p90=#{fmt(mw.p90)} avg=#{fmt(mw.avg)} speedup=#{fmt(sw.p50 / mw.p50)}x"
  puts "#{indent}  separate_gemm_gpu_ms  p10=#{fmt(sg.p10)} p50=#{fmt(sg.p50)} p90=#{fmt(sg.p90)} avg=#{fmt(sg.avg)}"
  puts "#{indent}  mixed_gpu_ms         p10=#{fmt(mg.p10)} p50=#{fmt(mg.p50)} p90=#{fmt(mg.p90)} avg=#{fmt(mg.avg)} speedup=#{fmt(sg.p50 / mg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_separate=#{fmt(separate_checksum)} checksum_mixed=#{fmt(mixed_checksum)}"
end

private def run_qkv_gate_mixed_probe(bench_ops : Array(BenchOp),
                                     x_buf : ML::MetalBuffer,
                                     batch : Int32,
                                     default_pipe : ML::Metal::ComputePipeline,
                                     mixed_pipe : ML::Metal::ComputePipeline,
                                     warmup : Int32,
                                     runs : Int32,
                                     indent : String = "  ") : Nil
  qkv_gate = bench_ops.select { |op| op.name == "rec.proj.qkv" || op.name == "rec.proj.gate" }
  raise "rec_proj qkv/gate pair not found" unless qkv_gate.size == 2
  run_two_output_mixed_probe("qkv_gate_mixed_dispatch", qkv_gate, x_buf, batch, default_pipe, mixed_pipe, warmup, runs, indent)
end

private def run_ffn_upgate_mixed_probe(bench_ops : Array(BenchOp),
                                       x_buf : ML::MetalBuffer,
                                       batch : Int32,
                                       default_pipe : ML::Metal::ComputePipeline,
                                       mixed_pipe : ML::Metal::ComputePipeline,
                                       warmup : Int32,
                                       runs : Int32,
                                       indent : String = "  ") : Nil
  upgate = bench_ops.select { |op| op.name == "rec.ffn.gate" || op.name == "rec.ffn.up" }
  raise "ffn gate/up pair not found" unless upgate.size == 2
  run_two_output_mixed_probe("ffn_upgate_mixed_dispatch", upgate, x_buf, batch, default_pipe, mixed_pipe, warmup, runs, indent)
end

private def run_rec_proj_mixed_probe(bench_ops : Array(BenchOp),
                                     x_buf : ML::MetalBuffer,
                                     batch : Int32,
                                     default_pipe : ML::Metal::ComputePipeline,
                                     mixed_pipe : ML::Metal::ComputePipeline,
                                     warmup : Int32,
                                     runs : Int32,
                                     indent : String = "  ") : Nil
  rec_ops = bench_ops.select { |op| op.name.starts_with?("rec.proj.") }
  raise "full rec_proj quad not found" unless rec_ops.size == 4

  run_corridor(rec_ops, x_buf, batch, default_pipe)
  separate_outputs = read_outputs(rec_ops, batch)
  run_rec_proj_mixed(rec_ops, x_buf, batch, mixed_pipe)
  mixed_outputs = read_outputs(rec_ops, batch)
  delta = max_delta(separate_outputs, mixed_outputs)
  separate_checksum = checksum(separate_outputs)
  mixed_checksum = checksum(mixed_outputs)

  separate_wall = [] of Float64
  mixed_wall = [] of Float64
  separate_gpu = [] of Float64
  mixed_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      s = run_corridor(rec_ops, x_buf, batch, default_pipe)
      m = run_rec_proj_mixed(rec_ops, x_buf, batch, mixed_pipe)
    else
      m = run_rec_proj_mixed(rec_ops, x_buf, batch, mixed_pipe)
      s = run_corridor(rec_ops, x_buf, batch, default_pipe)
    end

    if i >= warmup
      separate_wall << s.wall_ms
      separate_gpu << s.gpu_ms
      mixed_wall << m.wall_ms
      mixed_gpu << m.gpu_ms
    end
  end

  sw = stats(separate_wall)
  mw = stats(mixed_wall)
  sg = stats(separate_gpu)
  mg = stats(mixed_gpu)
  puts "#{indent}bench=rec_proj_mixed_dispatch ops=4"
  puts "#{indent}  separate_gemm_wall_ms p10=#{fmt(sw.p10)} p50=#{fmt(sw.p50)} p90=#{fmt(sw.p90)} avg=#{fmt(sw.avg)}"
  puts "#{indent}  mixed_wall_ms        p10=#{fmt(mw.p10)} p50=#{fmt(mw.p50)} p90=#{fmt(mw.p90)} avg=#{fmt(mw.avg)} speedup=#{fmt(sw.p50 / mw.p50)}x"
  puts "#{indent}  separate_gemm_gpu_ms  p10=#{fmt(sg.p10)} p50=#{fmt(sg.p50)} p90=#{fmt(sg.p90)} avg=#{fmt(sg.avg)}"
  puts "#{indent}  mixed_gpu_ms         p10=#{fmt(mg.p10)} p50=#{fmt(mg.p50)} p90=#{fmt(mg.p90)} avg=#{fmt(mg.avg)} speedup=#{fmt(sg.p50 / mg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_separate=#{fmt(separate_checksum)} checksum_mixed=#{fmt(mixed_checksum)}"
end

private def run_rec_proj_tilefast_probe(bench_ops : Array(BenchOp),
                                        x_buf : ML::MetalBuffer,
                                        batch : Int32,
                                        mixed_pipe : ML::Metal::ComputePipeline,
                                        tilefast_pipe : ML::Metal::ComputePipeline,
                                        warmup : Int32,
                                        runs : Int32,
                                        indent : String = "  ") : Nil
  rec_ops = bench_ops.select { |op| op.name.starts_with?("rec.proj.") }
  raise "full rec_proj quad not found" unless rec_ops.size == 4

  run_rec_proj_mixed(rec_ops, x_buf, batch, mixed_pipe)
  mixed_outputs = read_outputs(rec_ops, batch)
  run_rec_proj_mixed(rec_ops, x_buf, batch, tilefast_pipe)
  tilefast_outputs = read_outputs(rec_ops, batch)
  delta = max_delta(mixed_outputs, tilefast_outputs)
  mixed_checksum = checksum(mixed_outputs)
  tilefast_checksum = checksum(tilefast_outputs)

  mixed_wall = [] of Float64
  tilefast_wall = [] of Float64
  mixed_gpu = [] of Float64
  tilefast_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      m = run_rec_proj_mixed(rec_ops, x_buf, batch, mixed_pipe)
      t = run_rec_proj_mixed(rec_ops, x_buf, batch, tilefast_pipe)
    else
      t = run_rec_proj_mixed(rec_ops, x_buf, batch, tilefast_pipe)
      m = run_rec_proj_mixed(rec_ops, x_buf, batch, mixed_pipe)
    end

    if i >= warmup
      mixed_wall << m.wall_ms
      mixed_gpu << m.gpu_ms
      tilefast_wall << t.wall_ms
      tilefast_gpu << t.gpu_ms
    end
  end

  mw = stats(mixed_wall)
  tw = stats(tilefast_wall)
  mg = stats(mixed_gpu)
  tg = stats(tilefast_gpu)
  puts "#{indent}bench=rec_proj_tilefast_mixed ops=4"
  puts "#{indent}  mixed_wall_ms    p10=#{fmt(mw.p10)} p50=#{fmt(mw.p50)} p90=#{fmt(mw.p90)} avg=#{fmt(mw.avg)}"
  puts "#{indent}  tilefast_wall_ms p10=#{fmt(tw.p10)} p50=#{fmt(tw.p50)} p90=#{fmt(tw.p90)} avg=#{fmt(tw.avg)} speedup=#{fmt(mw.p50 / tw.p50)}x"
  puts "#{indent}  mixed_gpu_ms     p10=#{fmt(mg.p10)} p50=#{fmt(mg.p50)} p90=#{fmt(mg.p90)} avg=#{fmt(mg.avg)}"
  puts "#{indent}  tilefast_gpu_ms  p10=#{fmt(tg.p10)} p50=#{fmt(tg.p50)} p90=#{fmt(tg.p90)} avg=#{fmt(tg.avg)} speedup=#{fmt(mg.p50 / tg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_mixed=#{fmt(mixed_checksum)} checksum_tilefast=#{fmt(tilefast_checksum)}"
end

private def run_rec_proj_mixed_variant_probe(label : String,
                                             variant_label : String,
                                             bench_ops : Array(BenchOp),
                                             x_buf : ML::MetalBuffer,
                                             batch : Int32,
                                             base_pipe : ML::Metal::ComputePipeline,
                                             variant_pipe : ML::Metal::ComputePipeline,
                                             warmup : Int32,
                                             runs : Int32,
                                             indent : String = "  ") : Nil
  rec_ops = bench_ops.select { |op| op.name.starts_with?("rec.proj.") }
  raise "full rec_proj quad not found" unless rec_ops.size == 4

  run_rec_proj_mixed(rec_ops, x_buf, batch, base_pipe)
  base_outputs = read_outputs(rec_ops, batch)
  run_rec_proj_mixed(rec_ops, x_buf, batch, variant_pipe)
  variant_outputs = read_outputs(rec_ops, batch)
  delta = max_delta(base_outputs, variant_outputs)
  base_checksum = checksum(base_outputs)
  variant_checksum = checksum(variant_outputs)

  base_wall = [] of Float64
  variant_wall = [] of Float64
  base_gpu = [] of Float64
  variant_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      b = run_rec_proj_mixed(rec_ops, x_buf, batch, base_pipe)
      v = run_rec_proj_mixed(rec_ops, x_buf, batch, variant_pipe)
    else
      v = run_rec_proj_mixed(rec_ops, x_buf, batch, variant_pipe)
      b = run_rec_proj_mixed(rec_ops, x_buf, batch, base_pipe)
    end

    if i >= warmup
      base_wall << b.wall_ms
      base_gpu << b.gpu_ms
      variant_wall << v.wall_ms
      variant_gpu << v.gpu_ms
    end
  end

  bw = stats(base_wall)
  vw = stats(variant_wall)
  bg = stats(base_gpu)
  vg = stats(variant_gpu)
  puts "#{indent}bench=#{label} ops=4"
  puts "#{indent}  base_wall_ms       p10=#{fmt(bw.p10)} p50=#{fmt(bw.p50)} p90=#{fmt(bw.p90)} avg=#{fmt(bw.avg)}"
  puts "#{indent}  #{variant_label}_wall_ms p10=#{fmt(vw.p10)} p50=#{fmt(vw.p50)} p90=#{fmt(vw.p90)} avg=#{fmt(vw.avg)} speedup=#{fmt(bw.p50 / vw.p50)}x"
  puts "#{indent}  base_gpu_ms        p10=#{fmt(bg.p10)} p50=#{fmt(bg.p50)} p90=#{fmt(bg.p90)} avg=#{fmt(bg.avg)}"
  puts "#{indent}  #{variant_label}_gpu_ms  p10=#{fmt(vg.p10)} p50=#{fmt(vg.p50)} p90=#{fmt(vg.p90)} avg=#{fmt(vg.avg)} speedup=#{fmt(bg.p50 / vg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_base=#{fmt(base_checksum)} checksum_#{variant_label}=#{fmt(variant_checksum)}"
end

private def run_two_output_mixed_variant_probe(label : String,
                                               variant_label : String,
                                               pair : Array(BenchOp),
                                               x_buf : ML::MetalBuffer,
                                               batch : Int32,
                                               base_pipe : ML::Metal::ComputePipeline,
                                               variant_pipe : ML::Metal::ComputePipeline,
                                               warmup : Int32,
                                               runs : Int32,
                                               indent : String = "  ") : Nil
  raise "#{label}: expected two operations" unless pair.size == 2

  run_qkv_gate_mixed(pair, x_buf, batch, base_pipe)
  base_outputs = read_outputs(pair, batch)
  run_qkv_gate_mixed(pair, x_buf, batch, variant_pipe)
  variant_outputs = read_outputs(pair, batch)
  delta = max_delta(base_outputs, variant_outputs)
  base_checksum = checksum(base_outputs)
  variant_checksum = checksum(variant_outputs)

  base_wall = [] of Float64
  variant_wall = [] of Float64
  base_gpu = [] of Float64
  variant_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      b = run_qkv_gate_mixed(pair, x_buf, batch, base_pipe)
      v = run_qkv_gate_mixed(pair, x_buf, batch, variant_pipe)
    else
      v = run_qkv_gate_mixed(pair, x_buf, batch, variant_pipe)
      b = run_qkv_gate_mixed(pair, x_buf, batch, base_pipe)
    end

    if i >= warmup
      base_wall << b.wall_ms
      base_gpu << b.gpu_ms
      variant_wall << v.wall_ms
      variant_gpu << v.gpu_ms
    end
  end

  bw = stats(base_wall)
  vw = stats(variant_wall)
  bg = stats(base_gpu)
  vg = stats(variant_gpu)
  puts "#{indent}bench=#{label} ops=2"
  puts "#{indent}  base_wall_ms       p10=#{fmt(bw.p10)} p50=#{fmt(bw.p50)} p90=#{fmt(bw.p90)} avg=#{fmt(bw.avg)}"
  puts "#{indent}  #{variant_label}_wall_ms p10=#{fmt(vw.p10)} p50=#{fmt(vw.p50)} p90=#{fmt(vw.p90)} avg=#{fmt(vw.avg)} speedup=#{fmt(bw.p50 / vw.p50)}x"
  puts "#{indent}  base_gpu_ms        p10=#{fmt(bg.p10)} p50=#{fmt(bg.p50)} p90=#{fmt(bg.p90)} avg=#{fmt(bg.avg)}"
  puts "#{indent}  #{variant_label}_gpu_ms  p10=#{fmt(vg.p10)} p50=#{fmt(vg.p50)} p90=#{fmt(vg.p90)} avg=#{fmt(vg.avg)} speedup=#{fmt(bg.p50 / vg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_base=#{fmt(base_checksum)} checksum_#{variant_label}=#{fmt(variant_checksum)}"
end

private def run_qkv_gate_tilefast_probe(bench_ops : Array(BenchOp),
                                        x_buf : ML::MetalBuffer,
                                        batch : Int32,
                                        base_pipe : ML::Metal::ComputePipeline,
                                        tilefast_pipe : ML::Metal::ComputePipeline,
                                        warmup : Int32,
                                        runs : Int32,
                                        indent : String = "  ") : Nil
  qkv_gate = bench_ops.select { |op| op.name == "rec.proj.qkv" || op.name == "rec.proj.gate" }
  raise "rec_proj qkv/gate pair not found" unless qkv_gate.size == 2
  run_two_output_mixed_variant_probe("qkv_gate_mixed_tilefast", "tilefast", qkv_gate, x_buf, batch,
    base_pipe, tilefast_pipe, warmup, runs, indent)
end

private def run_ffn_upgate_tilefast_probe(bench_ops : Array(BenchOp),
                                          x_buf : ML::MetalBuffer,
                                          batch : Int32,
                                          base_pipe : ML::Metal::ComputePipeline,
                                          tilefast_pipe : ML::Metal::ComputePipeline,
                                          warmup : Int32,
                                          runs : Int32,
                                          indent : String = "  ") : Nil
  upgate = bench_ops.select { |op| op.name == "rec.ffn.gate" || op.name == "rec.ffn.up" }
  raise "ffn gate/up pair not found" unless upgate.size == 2
  run_two_output_mixed_variant_probe("ffn_upgate_mixed_tilefast", "tilefast", upgate, x_buf, batch,
    base_pipe, tilefast_pipe, warmup, runs, indent)
end

private def run_ffn_upgate_pair_rows_probe(bench_ops : Array(BenchOp),
                                           x_buf : ML::MetalBuffer,
                                           batch : Int32,
                                           base_pipe : ML::Metal::ComputePipeline,
                                           pair_rows_pipe : ML::Metal::ComputePipeline,
                                           warmup : Int32,
                                           runs : Int32,
                                           indent : String = "  ") : Nil
  upgate = bench_ops.select { |op| op.name == "rec.ffn.gate" || op.name == "rec.ffn.up" }
  raise "ffn gate/up pair not found" unless upgate.size == 2

  run_qkv_gate_mixed(upgate, x_buf, batch, base_pipe)
  base_outputs = read_outputs(upgate, batch)
  run_ffn_upgate_pair_rows(upgate, x_buf, batch, pair_rows_pipe)
  pair_outputs = read_outputs(upgate, batch)
  delta = max_delta(base_outputs, pair_outputs)
  base_checksum = checksum(base_outputs)
  pair_checksum = checksum(pair_outputs)

  base_wall = [] of Float64
  pair_wall = [] of Float64
  base_gpu = [] of Float64
  pair_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      b = run_qkv_gate_mixed(upgate, x_buf, batch, base_pipe)
      p = run_ffn_upgate_pair_rows(upgate, x_buf, batch, pair_rows_pipe)
    else
      p = run_ffn_upgate_pair_rows(upgate, x_buf, batch, pair_rows_pipe)
      b = run_qkv_gate_mixed(upgate, x_buf, batch, base_pipe)
    end

    if i >= warmup
      base_wall << b.wall_ms
      base_gpu << b.gpu_ms
      pair_wall << p.wall_ms
      pair_gpu << p.gpu_ms
    end
  end

  bw = stats(base_wall)
  pw = stats(pair_wall)
  bg = stats(base_gpu)
  pg = stats(pair_gpu)
  puts "#{indent}bench=ffn_upgate_pair_rows ops=2"
  puts "#{indent}  mixed_wall_ms     p10=#{fmt(bw.p10)} p50=#{fmt(bw.p50)} p90=#{fmt(bw.p90)} avg=#{fmt(bw.avg)}"
  puts "#{indent}  pair_rows_wall_ms p10=#{fmt(pw.p10)} p50=#{fmt(pw.p50)} p90=#{fmt(pw.p90)} avg=#{fmt(pw.avg)} speedup=#{fmt(bw.p50 / pw.p50)}x"
  puts "#{indent}  mixed_gpu_ms      p10=#{fmt(bg.p10)} p50=#{fmt(bg.p50)} p90=#{fmt(bg.p90)} avg=#{fmt(bg.avg)}"
  puts "#{indent}  pair_rows_gpu_ms  p10=#{fmt(pg.p10)} p50=#{fmt(pg.p50)} p90=#{fmt(pg.p90)} avg=#{fmt(pg.avg)} speedup=#{fmt(bg.p50 / pg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_mixed=#{fmt(base_checksum)} checksum_pair_rows=#{fmt(pair_checksum)}"
end

private def run_post_oproj_fused_probe(rec : ML::GGUF::Qwen35RecurrentWeights,
                                       batch : Int32,
                                       post_pipe : ML::Metal::ComputePipeline,
                                       q8_pipe : ML::Metal::ComputePipeline,
                                       inv_pipe : ML::Metal::ComputePipeline,
                                       fused_pipe : ML::Metal::ComputePipeline,
                                       warmup : Int32,
                                       runs : Int32,
                                       indent : String = "  ") : Nil
  qw = rec.ssm_out_qw
  raise "post o_proj expected Q8_0, got #{qw.type.name}" unless qw.type.q8_0?
  s = 128
  raise "unexpected post o_proj in_dim #{qw.in_dim}" unless qw.in_dim % s == 0
  h_v = qw.in_dim // s
  eps = 1.0e-6_f32
  rng = Random.new(20260618)

  y = Array(Float32).new(batch * qw.in_dim) { rng.rand(-1.0_f32..1.0_f32) }
  z = Array(Float32).new(batch * qw.in_dim) { rng.rand(-2.0_f32..2.0_f32) }
  norm = Array(Float32).new(s) { rng.rand(0.5_f32..1.5_f32) }
  y_buf = ML::MetalBuffer.from_array(y)
  z_buf = ML::MetalBuffer.from_array(z)
  norm_buf = ML::MetalBuffer.from_array(norm)
  post_buf = ML::MetalBuffer.new((batch * qw.in_dim).to_i64 * sizeof(Float32))
  inv_buf = ML::MetalBuffer.new((batch * h_v).to_i64 * sizeof(Float32))
  fused_out_buf = ML::MetalBuffer.new((batch * qw.out_dim).to_i64 * sizeof(Float32))
  op = BenchOp.new("rec.o_proj", qw, upload_weights(qw), ML::MetalBuffer.new((batch * qw.out_dim).to_i64 * sizeof(Float32)))

  run_post_oproj_default(op, y_buf, z_buf, norm_buf, post_buf, batch, h_v, s, eps, post_pipe, q8_pipe)
  default_outputs = [op.out_buf.read(batch * qw.out_dim)]
  run_post_oproj_fused(op, y_buf, z_buf, norm_buf, inv_buf, fused_out_buf, batch, h_v, s, eps, inv_pipe, fused_pipe)
  fused_outputs = [fused_out_buf.read(batch * qw.out_dim)]
  delta = max_delta(default_outputs, fused_outputs)
  default_checksum = checksum(default_outputs)
  fused_checksum = checksum(fused_outputs)

  default_wall = [] of Float64
  fused_wall = [] of Float64
  default_gpu = [] of Float64
  fused_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      d = run_post_oproj_default(op, y_buf, z_buf, norm_buf, post_buf, batch, h_v, s, eps, post_pipe, q8_pipe)
      f = run_post_oproj_fused(op, y_buf, z_buf, norm_buf, inv_buf, fused_out_buf, batch, h_v, s, eps, inv_pipe, fused_pipe)
    else
      f = run_post_oproj_fused(op, y_buf, z_buf, norm_buf, inv_buf, fused_out_buf, batch, h_v, s, eps, inv_pipe, fused_pipe)
      d = run_post_oproj_default(op, y_buf, z_buf, norm_buf, post_buf, batch, h_v, s, eps, post_pipe, q8_pipe)
    end

    if i >= warmup
      default_wall << d.wall_ms
      default_gpu << d.gpu_ms
      fused_wall << f.wall_ms
      fused_gpu << f.gpu_ms
    end
  end

  dw = stats(default_wall)
  fw = stats(fused_wall)
  dg = stats(default_gpu)
  fg = stats(fused_gpu)
  puts
  puts "corridor=post_oproj"
  puts "#{indent}op=rec.o_proj shape=#{qw.in_dim}x#{qw.out_dim} type=#{qw.type.name} raw_mib=#{fmt(qw.raw.size.to_f64 / 1024.0 / 1024.0)} h_v=#{h_v} s=#{s}"
  puts "#{indent}bench=post_oproj_fused_input"
  puts "#{indent}  default_wall_ms p10=#{fmt(dw.p10)} p50=#{fmt(dw.p50)} p90=#{fmt(dw.p90)} avg=#{fmt(dw.avg)}"
  puts "#{indent}  fused_wall_ms   p10=#{fmt(fw.p10)} p50=#{fmt(fw.p50)} p90=#{fmt(fw.p90)} avg=#{fmt(fw.avg)} speedup=#{fmt(dw.p50 / fw.p50)}x"
  puts "#{indent}  default_gpu_ms  p10=#{fmt(dg.p10)} p50=#{fmt(dg.p50)} p90=#{fmt(dg.p90)} avg=#{fmt(dg.avg)}"
  puts "#{indent}  fused_gpu_ms    p10=#{fmt(fg.p10)} p50=#{fmt(fg.p50)} p90=#{fmt(fg.p90)} avg=#{fmt(fg.avg)} speedup=#{fmt(dg.p50 / fg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_default=#{fmt(default_checksum)} checksum_fused=#{fmt(fused_checksum)}"
end

private def run_ffn_down_fused_probe(rec : ML::GGUF::Qwen35RecurrentWeights,
                                     batch : Int32,
                                     upgate_pipe : ML::Metal::ComputePipeline,
                                     swiglu_pipe : ML::Metal::ComputePipeline,
                                     q8_pipe : ML::Metal::ComputePipeline,
                                     fused_down_pipe : ML::Metal::ComputePipeline,
                                     warmup : Int32,
                                     runs : Int32,
                                     indent : String = "  ") : Nil
  gate = BenchOp.new("rec.ffn.gate", rec.ffn_gate_qw, upload_weights(rec.ffn_gate_qw), ML::MetalBuffer.new((batch * rec.ffn_gate_qw.out_dim).to_i64 * sizeof(Float32)))
  up = BenchOp.new("rec.ffn.up", rec.ffn_up_qw, upload_weights(rec.ffn_up_qw), ML::MetalBuffer.new((batch * rec.ffn_up_qw.out_dim).to_i64 * sizeof(Float32)))
  down = BenchOp.new("rec.ffn.down", rec.ffn_down_qw, upload_weights(rec.ffn_down_qw), ML::MetalBuffer.new((batch * rec.ffn_down_qw.out_dim).to_i64 * sizeof(Float32)))
  raise "expected Q8 FFN weights" unless gate.qw.type.q8_0? && up.qw.type.q8_0? && down.qw.type.q8_0?
  raise "gate/up/down shape mismatch" unless gate.qw.out_dim == up.qw.out_dim && down.qw.in_dim == gate.qw.out_dim

  rng = Random.new(20260619)
  x = Array(Float32).new(batch * gate.qw.in_dim) { rng.rand(-1.0_f32..1.0_f32) }
  x_buf = ML::MetalBuffer.from_array(x)
  act_buf = ML::MetalBuffer.new((batch * gate.qw.out_dim).to_i64 * sizeof(Float32))
  fused_out_buf = ML::MetalBuffer.new((batch * down.qw.out_dim).to_i64 * sizeof(Float32))
  upgate_ops = [gate, up]

  run_ffn_down_default(upgate_ops, down, x_buf, act_buf, batch, upgate_pipe, swiglu_pipe, q8_pipe)
  default_outputs = [down.out_buf.read(batch * down.qw.out_dim)]
  run_ffn_down_fused(upgate_ops, down, x_buf, fused_out_buf, batch, upgate_pipe, fused_down_pipe)
  fused_outputs = [fused_out_buf.read(batch * down.qw.out_dim)]
  delta = max_delta(default_outputs, fused_outputs)
  default_checksum = checksum(default_outputs)
  fused_checksum = checksum(fused_outputs)

  default_wall = [] of Float64
  fused_wall = [] of Float64
  default_gpu = [] of Float64
  fused_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      d = run_ffn_down_default(upgate_ops, down, x_buf, act_buf, batch, upgate_pipe, swiglu_pipe, q8_pipe)
      f = run_ffn_down_fused(upgate_ops, down, x_buf, fused_out_buf, batch, upgate_pipe, fused_down_pipe)
    else
      f = run_ffn_down_fused(upgate_ops, down, x_buf, fused_out_buf, batch, upgate_pipe, fused_down_pipe)
      d = run_ffn_down_default(upgate_ops, down, x_buf, act_buf, batch, upgate_pipe, swiglu_pipe, q8_pipe)
    end

    if i >= warmup
      default_wall << d.wall_ms
      default_gpu << d.gpu_ms
      fused_wall << f.wall_ms
      fused_gpu << f.gpu_ms
    end
  end

  dw = stats(default_wall)
  fw = stats(fused_wall)
  dg = stats(default_gpu)
  fg = stats(fused_gpu)
  puts
  puts "corridor=ffn_down"
  puts "#{indent}op=rec.ffn gate/up/down shapes=#{gate.qw.in_dim}x#{gate.qw.out_dim}+#{up.qw.out_dim}->#{down.qw.out_dim} type=#{gate.qw.type.name}"
  puts "#{indent}bench=ffn_down_fused_input"
  puts "#{indent}  default_wall_ms p10=#{fmt(dw.p10)} p50=#{fmt(dw.p50)} p90=#{fmt(dw.p90)} avg=#{fmt(dw.avg)}"
  puts "#{indent}  fused_wall_ms   p10=#{fmt(fw.p10)} p50=#{fmt(fw.p50)} p90=#{fmt(fw.p90)} avg=#{fmt(fw.avg)} speedup=#{fmt(dw.p50 / fw.p50)}x"
  puts "#{indent}  default_gpu_ms  p10=#{fmt(dg.p10)} p50=#{fmt(dg.p50)} p90=#{fmt(dg.p90)} avg=#{fmt(dg.avg)}"
  puts "#{indent}  fused_gpu_ms    p10=#{fmt(fg.p10)} p50=#{fmt(fg.p50)} p90=#{fmt(fg.p90)} avg=#{fmt(fg.avg)} speedup=#{fmt(dg.p50 / fg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)} checksum_default=#{fmt(default_checksum)} checksum_fused=#{fmt(fused_checksum)}"
end

private def encode_rec_conv_shift(enc : ML::Metal::ComputeEncoder,
                                  pipe : ML::Metal::ComputePipeline,
                                  conv_state_buf : ML::MetalBuffer,
                                  qkv_buf : ML::MetalBuffer,
                                  conv_w_buf : ML::MetalBuffer,
                                  q_buf : ML::MetalBuffer,
                                  k_buf : ML::MetalBuffer,
                                  v_buf : ML::MetalBuffer,
                                  h_k : Int32,
                                  h_v : Int32,
                                  s : Int32,
                                  conv_k : Int32,
                                  batch : Int32) : Nil
  qkv_dim = 2 * h_k * s + h_v * s
  enc.set_pipeline(pipe)
  enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
  enc.set_buffer(qkv_buf, 1)
  enc.set_buffer(conv_w_buf, 2)
  enc.set_buffer(q_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_buffer(k_buf, 4, ML::Metal::BufferAccess::Write)
  enc.set_buffer(v_buf, 5, ML::Metal::BufferAccess::Write)
  enc.set_value(h_k.to_u32, 6)
  enc.set_value(h_v.to_u32, 7)
  enc.set_value(s.to_u32, 8)
  enc.set_value(conv_k.to_u32, 9)
  enc.set_value(batch.to_u32, 10)
  enc.dispatch_1d(qkv_dim, 256)
end

private def encode_l2_heads_chunk(enc : ML::Metal::ComputeEncoder,
                                  pipe : ML::Metal::ComputePipeline,
                                  x_buf : ML::MetalBuffer,
                                  n_heads : Int32,
                                  s : Int32,
                                  batch : Int32,
                                  eps : Float32) : Nil
  enc.set_pipeline(pipe)
  enc.set_buffer(x_buf, 0, ML::Metal::BufferAccess::ReadWrite)
  enc.set_value(n_heads.to_u32, 1)
  enc.set_value(s.to_u32, 2)
  enc.set_value(eps, 3)
  enc.dispatch_threadgroups({n_heads, batch, 1}, {32, 1, 1})
end

private def encode_recurrent_ab_chunk(enc : ML::Metal::ComputeEncoder,
                                      pipe : ML::Metal::ComputePipeline,
                                      alpha_buf : ML::MetalBuffer,
                                      beta_buf : ML::MetalBuffer,
                                      dt_bias_buf : ML::MetalBuffer,
                                      ssm_a_buf : ML::MetalBuffer,
                                      g_buf : ML::MetalBuffer,
                                      h_v : Int32,
                                      batch : Int32) : Nil
  enc.set_pipeline(pipe)
  enc.set_buffer(alpha_buf, 0)
  enc.set_buffer(beta_buf, 1, ML::Metal::BufferAccess::ReadWrite)
  enc.set_buffer(dt_bias_buf, 2)
  enc.set_buffer(ssm_a_buf, 3)
  enc.set_buffer(g_buf, 4, ML::Metal::BufferAccess::Write)
  enc.set_value(h_v.to_u32, 5)
  enc.set_value(batch.to_u32, 6)
  enc.dispatch_1d(batch * h_v, 64)
end

private def encode_rec_conv_shift_token_parallel(enc : ML::Metal::ComputeEncoder,
                                                 conv_pipe : ML::Metal::ComputePipeline,
                                                 state_pipe : ML::Metal::ComputePipeline,
                                                 conv_state_buf : ML::MetalBuffer,
                                                 qkv_buf : ML::MetalBuffer,
                                                 conv_w_buf : ML::MetalBuffer,
                                                 q_buf : ML::MetalBuffer,
                                                 k_buf : ML::MetalBuffer,
                                                 v_buf : ML::MetalBuffer,
                                                 h_k : Int32,
                                                 h_v : Int32,
                                                 s : Int32,
                                                 conv_k : Int32,
                                                 batch : Int32) : Nil
  qkv_dim = 2 * h_k * s + h_v * s
  enc.set_pipeline(conv_pipe)
  enc.set_buffer(conv_state_buf, 0)
  enc.set_buffer(qkv_buf, 1)
  enc.set_buffer(conv_w_buf, 2)
  enc.set_buffer(q_buf, 3, ML::Metal::BufferAccess::Write)
  enc.set_buffer(k_buf, 4, ML::Metal::BufferAccess::Write)
  enc.set_buffer(v_buf, 5, ML::Metal::BufferAccess::Write)
  enc.set_value(h_k.to_u32, 6)
  enc.set_value(h_v.to_u32, 7)
  enc.set_value(s.to_u32, 8)
  enc.set_value(conv_k.to_u32, 9)
  enc.set_value(batch.to_u32, 10)
  enc.dispatch_1d(batch * qkv_dim, 128)
  enc.set_pipeline(state_pipe)
  enc.set_buffer(conv_state_buf, 0, ML::Metal::BufferAccess::ReadWrite)
  enc.set_buffer(qkv_buf, 1)
  enc.set_value(qkv_dim.to_u32, 2)
  enc.set_value(conv_k.to_u32, 3)
  enc.set_value(batch.to_u32, 4)
  enc.dispatch_1d((conv_k - 1) * qkv_dim, 128)
end

private def run_rec_conv_token_parallel_sample(conv_pipe : ML::Metal::ComputePipeline,
                                               state_pipe : ML::Metal::ComputePipeline,
                                               conv_state_buf : ML::MetalBuffer,
                                               qkv_buf : ML::MetalBuffer,
                                               conv_w_buf : ML::MetalBuffer,
                                               q_buf : ML::MetalBuffer,
                                               k_buf : ML::MetalBuffer,
                                               v_buf : ML::MetalBuffer,
                                               h_k : Int32,
                                               h_v : Int32,
                                               s : Int32,
                                               conv_k : Int32,
                                               batch : Int32) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    enc = ML::Metal::ComputeEncoder.new(cmd)
    encode_rec_conv_shift_token_parallel(enc, conv_pipe, state_pipe,
      conv_state_buf, qkv_buf, conv_w_buf, q_buf, k_buf, v_buf,
      h_k, h_v, s, conv_k, batch)
    enc.end_encoding
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_rec_prep_sample(kind : String,
                                conv_pipe : ML::Metal::ComputePipeline,
                                l2_pipe : ML::Metal::ComputePipeline,
                                ab_pipe : ML::Metal::ComputePipeline,
                                conv_state_buf : ML::MetalBuffer,
                                qkv_buf : ML::MetalBuffer,
                                conv_w_buf : ML::MetalBuffer,
                                q_buf : ML::MetalBuffer,
                                k_buf : ML::MetalBuffer,
                                v_buf : ML::MetalBuffer,
                                alpha_buf : ML::MetalBuffer,
                                beta_buf : ML::MetalBuffer,
                                dt_bias_buf : ML::MetalBuffer,
                                ssm_a_buf : ML::MetalBuffer,
                                g_buf : ML::MetalBuffer,
                                h_k : Int32,
                                h_v : Int32,
                                s : Int32,
                                conv_k : Int32,
                                batch : Int32,
                                eps : Float32) : Sample
  gpu_ms = 0.0
  elapsed = Time.measure do
    cmd = ML::Metal::CommandBuffer.new
    case kind
    when "conv"
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_rec_conv_shift(enc, conv_pipe, conv_state_buf, qkv_buf, conv_w_buf, q_buf, k_buf, v_buf, h_k, h_v, s, conv_k, batch)
      enc.end_encoding
    when "qnorm"
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_l2_heads_chunk(enc, l2_pipe, q_buf, h_k, s, batch, eps)
      enc.end_encoding
    when "knorm"
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_l2_heads_chunk(enc, l2_pipe, k_buf, h_k, s, batch, eps)
      enc.end_encoding
    when "ab"
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_recurrent_ab_chunk(enc, ab_pipe, alpha_buf, beta_buf, dt_bias_buf, ssm_a_buf, g_buf, h_v, batch)
      enc.end_encoding
    when "all"
      enc = ML::Metal::ComputeEncoder.new(cmd)
      encode_rec_conv_shift(enc, conv_pipe, conv_state_buf, qkv_buf, conv_w_buf, q_buf, k_buf, v_buf, h_k, h_v, s, conv_k, batch)
      enc.end_encoding
      qenc = ML::Metal::ComputeEncoder.new(cmd)
      encode_l2_heads_chunk(qenc, l2_pipe, q_buf, h_k, s, batch, eps)
      qenc.end_encoding
      kenc = ML::Metal::ComputeEncoder.new(cmd)
      encode_l2_heads_chunk(kenc, l2_pipe, k_buf, h_k, s, batch, eps)
      kenc.end_encoding
      abenc = ML::Metal::ComputeEncoder.new(cmd)
      encode_recurrent_ab_chunk(abenc, ab_pipe, alpha_buf, beta_buf, dt_bias_buf, ssm_a_buf, g_buf, h_v, batch)
      abenc.end_encoding
    else
      raise "unknown rec prep kind #{kind.inspect}"
    end
    cmd.commit
    gpu_ms = cmd.wait_gpu_elapsed_ms
  end
  Sample.new(elapsed.total_milliseconds, gpu_ms)
end

private def run_rec_prep_split_probe(weights : ML::GGUF::Qwen35Weights,
                                     rec : ML::GGUF::Qwen35RecurrentWeights,
                                     batch : Int32,
                                     conv_pipe : ML::Metal::ComputePipeline,
                                     l2_pipe : ML::Metal::ComputePipeline,
                                     ab_pipe : ML::Metal::ComputePipeline,
                                     warmup : Int32,
                                     runs : Int32,
                                     indent : String = "  ") : Nil
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  conv_k = hp.ssm_conv_kernel
  eps = hp.rms_eps
  qkv_dim = 2 * h_k * s + h_v * s
  raise "qkv shape mismatch #{rec.attn_qkv_qw.out_dim} != #{qkv_dim}" unless rec.attn_qkv_qw.out_dim == qkv_dim

  rng = Random.new(20260620)
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim) { rng.rand(-0.25_f32..0.25_f32) }
  qkv = Array(Float32).new(batch * qkv_dim) { rng.rand(-1.0_f32..1.0_f32) }
  alpha = Array(Float32).new(batch * h_v) { rng.rand(-2.0_f32..2.0_f32) }
  beta = Array(Float32).new(batch * h_v) { rng.rand(-2.0_f32..2.0_f32) }
  conv_state_buf = ML::MetalBuffer.from_array(conv_state)
  qkv_buf = ML::MetalBuffer.from_array(qkv)
  conv_w_buf = ML::MetalBuffer.from_array(rec.ssm_conv1d)
  q_buf = ML::MetalBuffer.new((batch * h_k * s).to_i64 * sizeof(Float32))
  k_buf = ML::MetalBuffer.new((batch * h_k * s).to_i64 * sizeof(Float32))
  v_buf = ML::MetalBuffer.new((batch * h_v * s).to_i64 * sizeof(Float32))
  alpha_buf = ML::MetalBuffer.from_array(alpha)
  beta_buf = ML::MetalBuffer.from_array(beta)
  dt_bias_buf = ML::MetalBuffer.from_array(rec.ssm_dt_bias)
  ssm_a_buf = ML::MetalBuffer.from_array(rec.ssm_a)
  g_buf = ML::MetalBuffer.new((batch * h_v).to_i64 * sizeof(Float32))

  kinds = ["conv", "qnorm", "knorm", "ab", "all"]
  samples = Hash(String, {Array(Float64), Array(Float64)}).new
  kinds.each { |kind| samples[kind] = {[] of Float64, [] of Float64} }

  (warmup + runs).times do |i|
    kinds.each do |kind|
      sample = run_rec_prep_sample(kind, conv_pipe, l2_pipe, ab_pipe,
        conv_state_buf, qkv_buf, conv_w_buf, q_buf, k_buf, v_buf,
        alpha_buf, beta_buf, dt_bias_buf, ssm_a_buf, g_buf,
        h_k, h_v, s, conv_k, batch, eps)
      if i >= warmup
        samples[kind][0] << sample.wall_ms
        samples[kind][1] << sample.gpu_ms
      end
    end
  end

  puts
  puts "corridor=rec_prep"
  puts "#{indent}shape h_k=#{h_k} h_v=#{h_v} s=#{s} conv_k=#{conv_k} qkv_dim=#{qkv_dim} batch=#{batch}"
  kinds.each do |kind|
    wall = stats(samples[kind][0])
    gpu = stats(samples[kind][1])
    puts "#{indent}bench=rec_prep.#{kind}"
    puts "#{indent}  wall_ms p10=#{fmt(wall.p10)} p50=#{fmt(wall.p50)} p90=#{fmt(wall.p90)} avg=#{fmt(wall.avg)}"
    puts "#{indent}  gpu_ms  p10=#{fmt(gpu.p10)} p50=#{fmt(gpu.p50)} p90=#{fmt(gpu.p90)} avg=#{fmt(gpu.avg)}"
  end
  q_checksum = q_buf.read(batch * h_k * s)[0].to_f64 + k_buf.read(batch * h_k * s)[0].to_f64 + g_buf.read(batch * h_v)[0].to_f64
  puts "#{indent}  checksum=#{fmt(q_checksum)}"
end

private def run_rec_conv_token_parallel_probe(weights : ML::GGUF::Qwen35Weights,
                                              rec : ML::GGUF::Qwen35RecurrentWeights,
                                              batch : Int32,
                                              default_pipe : ML::Metal::ComputePipeline,
                                              token_pipe : ML::Metal::ComputePipeline,
                                              token_state_pipe : ML::Metal::ComputePipeline,
                                              warmup : Int32,
                                              runs : Int32,
                                              indent : String = "  ") : Nil
  hp = weights.hparams
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  conv_k = hp.ssm_conv_kernel
  qkv_dim = 2 * h_k * s + h_v * s
  raise "qkv shape mismatch #{rec.attn_qkv_qw.out_dim} != #{qkv_dim}" unless rec.attn_qkv_qw.out_dim == qkv_dim

  rng = Random.new(20260621)
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim) { rng.rand(-0.25_f32..0.25_f32) }
  qkv = Array(Float32).new(batch * qkv_dim) { rng.rand(-1.0_f32..1.0_f32) }
  qkv_buf = ML::MetalBuffer.from_array(qkv)
  conv_w_buf = ML::MetalBuffer.from_array(rec.ssm_conv1d)

  default_state_buf = ML::MetalBuffer.from_array(conv_state)
  variant_state_buf = ML::MetalBuffer.from_array(conv_state)
  default_q = ML::MetalBuffer.new((batch * h_k * s).to_i64 * sizeof(Float32))
  default_k = ML::MetalBuffer.new((batch * h_k * s).to_i64 * sizeof(Float32))
  default_v = ML::MetalBuffer.new((batch * h_v * s).to_i64 * sizeof(Float32))
  variant_q = ML::MetalBuffer.new((batch * h_k * s).to_i64 * sizeof(Float32))
  variant_k = ML::MetalBuffer.new((batch * h_k * s).to_i64 * sizeof(Float32))
  variant_v = ML::MetalBuffer.new((batch * h_v * s).to_i64 * sizeof(Float32))

  run_rec_prep_sample("conv", default_pipe, default_pipe, default_pipe,
    default_state_buf, qkv_buf, conv_w_buf, default_q, default_k, default_v,
    default_q, default_k, default_q, default_k, default_v,
    h_k, h_v, s, conv_k, batch, 0.0_f32)
  run_rec_conv_token_parallel_sample(token_pipe, token_state_pipe,
    variant_state_buf, qkv_buf, conv_w_buf, variant_q, variant_k, variant_v,
    h_k, h_v, s, conv_k, batch)

  delta = max_delta(
    [
      default_q.read(batch * h_k * s),
      default_k.read(batch * h_k * s),
      default_v.read(batch * h_v * s),
      default_state_buf.read((conv_k - 1) * qkv_dim),
    ],
    [
      variant_q.read(batch * h_k * s),
      variant_k.read(batch * h_k * s),
      variant_v.read(batch * h_v * s),
      variant_state_buf.read((conv_k - 1) * qkv_dim),
    ]
  )

  default_wall = [] of Float64
  variant_wall = [] of Float64
  default_gpu = [] of Float64
  variant_gpu = [] of Float64

  (warmup + runs).times do |i|
    if i.even?
      d = run_rec_prep_sample("conv", default_pipe, default_pipe, default_pipe,
        default_state_buf, qkv_buf, conv_w_buf, default_q, default_k, default_v,
        default_q, default_k, default_q, default_k, default_v,
        h_k, h_v, s, conv_k, batch, 0.0_f32)
      v = run_rec_conv_token_parallel_sample(token_pipe, token_state_pipe,
        variant_state_buf, qkv_buf, conv_w_buf, variant_q, variant_k, variant_v,
        h_k, h_v, s, conv_k, batch)
    else
      v = run_rec_conv_token_parallel_sample(token_pipe, token_state_pipe,
        variant_state_buf, qkv_buf, conv_w_buf, variant_q, variant_k, variant_v,
        h_k, h_v, s, conv_k, batch)
      d = run_rec_prep_sample("conv", default_pipe, default_pipe, default_pipe,
        default_state_buf, qkv_buf, conv_w_buf, default_q, default_k, default_v,
        default_q, default_k, default_q, default_k, default_v,
        h_k, h_v, s, conv_k, batch, 0.0_f32)
    end

    if i >= warmup
      default_wall << d.wall_ms
      default_gpu << d.gpu_ms
      variant_wall << v.wall_ms
      variant_gpu << v.gpu_ms
    end
  end

  dw = stats(default_wall)
  vw = stats(variant_wall)
  dg = stats(default_gpu)
  vg = stats(variant_gpu)
  puts
  puts "corridor=rec_conv"
  puts "#{indent}shape h_k=#{h_k} h_v=#{h_v} s=#{s} conv_k=#{conv_k} qkv_dim=#{qkv_dim} batch=#{batch}"
  puts "#{indent}bench=rec_conv_token_parallel"
  puts "#{indent}  default_wall_ms p10=#{fmt(dw.p10)} p50=#{fmt(dw.p50)} p90=#{fmt(dw.p90)} avg=#{fmt(dw.avg)}"
  puts "#{indent}  token_wall_ms   p10=#{fmt(vw.p10)} p50=#{fmt(vw.p50)} p90=#{fmt(vw.p90)} avg=#{fmt(vw.avg)} speedup=#{fmt(dw.p50 / vw.p50)}x"
  puts "#{indent}  default_gpu_ms  p10=#{fmt(dg.p10)} p50=#{fmt(dg.p50)} p90=#{fmt(dg.p90)} avg=#{fmt(dg.avg)}"
  puts "#{indent}  token_gpu_ms    p10=#{fmt(vg.p10)} p50=#{fmt(vg.p50)} p90=#{fmt(vg.p90)} avg=#{fmt(vg.avg)} speedup=#{fmt(dg.p50 / vg.p50)}x"
  puts "#{indent}  max_delta=#{fmt(delta)}"
end

model = ENV["QWEN35_DRAFT"]? || DEFAULT_MODEL
batch = 256
runs = 9
warmup = 2
corridor = "all"
per_op = false
alpha_beta_dual = false
qkv_gate_mixed = false
rec_proj_mixed = false
ffn_upgate_mixed = false
h16_input = false
rec_proj_tilefast = false
rec_proj_mixed_singlebuf = false
post_oproj_fused = false
ffn_down_fused = false
no_sg_barriers = false
two_output_tilefast = false
rec_prep_split = false
ffn_upgate_pair_rows = false
q8_gemv_layout_sweep = false
rec_conv_token_parallel = false

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_q8_hotshape_micro [--model PATH] [--batch N] [--runs N] [--warmup N] [--corridor rec_proj|ffn_upgate|all] [--per-op] [--alpha-beta-dual] [--qkv-gate-mixed] [--rec-proj-mixed] [--ffn-upgate-mixed] [--h16-input] [--rec-proj-tilefast] [--rec-proj-mixed-singlebuf] [--post-oproj-fused] [--ffn-down-fused] [--no-sg-barriers] [--two-output-tilefast] [--rec-prep-split] [--ffn-upgate-pair-rows] [--q8-gemv-layout-sweep] [--rec-conv-token-parallel]"
  p.on("--model=PATH", "Qwen3.5 0.8B Q8_0 GGUF path") { |v| model = v }
  p.on("--batch=N", "Prompt batch rows (default: 256)") { |v| batch = v.to_i }
  p.on("--runs=N", "Timed paired runs (default: 9)") { |v| runs = v.to_i }
  p.on("--warmup=N", "Warmup paired runs (default: 2)") { |v| warmup = v.to_i }
  p.on("--corridor=NAME", "rec_proj, ffn_upgate, or all") { |v| corridor = v }
  p.on("--per-op", "Also time each operation in the selected corridor") { per_op = true }
  p.on("--alpha-beta-dual", "Compare rec_proj alpha/beta separate Q8 GEMMs against existing dual Q8 GEMV") { alpha_beta_dual = true }
  p.on("--qkv-gate-mixed", "Compare rec_proj qkv/gate separate Q8 GEMMs against one mixed-dispatch Q8 GEMM") { qkv_gate_mixed = true }
  p.on("--rec-proj-mixed", "Compare rec_proj qkv/gate/alpha/beta separate Q8 GEMMs against one mixed-dispatch Q8 GEMM") { rec_proj_mixed = true }
  p.on("--ffn-upgate-mixed", "Compare FFN gate/up separate Q8 GEMMs against one mixed-dispatch Q8 GEMM") { ffn_upgate_mixed = true }
  p.on("--h16-input", "Compare Q8 F32-input GEMM body against H16-input GEMM body on the same corridor") { h16_input = true }
  p.on("--rec-proj-tilefast", "Compare rec_proj mixed kernel against a micro-only tile-level weight-base specialization") { rec_proj_tilefast = true }
  p.on("--rec-proj-mixed-singlebuf", "Compare rec_proj mixed double-buffer kernel against a micro-only single-buffer variant") { rec_proj_mixed_singlebuf = true }
  p.on("--post-oproj-fused", "Compare dn_post+Q8 o_proj against a micro-only fused-input Q8 o_proj diamond") { post_oproj_fused = true }
  p.on("--ffn-down-fused", "Compare FFN upgate+SwiGLU+down against a micro-only fused-input Q8 FFN-down diamond") { ffn_down_fused = true }
  p.on("--no-sg-barriers", "Compare current Q8 GEMM/mixed bodies against a micro-only source variant without mem_none simdgroup barriers") { no_sg_barriers = true }
  p.on("--two-output-tilefast", "Compare two-output Q8 mixed kernel against a micro-only tile-boundary weight-base specialization") { two_output_tilefast = true }
  p.on("--rec-prep-split", "Time recurrent prep subphases: conv, qnorm, knorm, alpha/beta, and combined") { rec_prep_split = true }
  p.on("--ffn-upgate-pair-rows", "Compare FFN gate/up mixed output tiling against a micro-only paired-row tile kernel") { ffn_upgate_pair_rows = true }
  p.on("--q8-gemv-layout-sweep", "Compare batch=1 Q8 GEMV compile-time layouts against the current 4x1 layout") { q8_gemv_layout_sweep = true }
  p.on("--rec-conv-token-parallel", "Compare recurrent conv channel-serial scan against a micro-only token-parallel conv scan") { rec_conv_token_parallel = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "batch must be positive" unless batch > 0
raise "runs must be positive" unless runs > 0
raise "warmup must be non-negative" unless warmup >= 0

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
rec = first_recurrent_layer(weights)
hidden_dim = rec.attn_qkv_qw.in_dim
rng = Random.new(20260614)
x = Array(Float32).new(batch * hidden_dim) { rng.rand(-1.0_f32..1.0_f32) }
x_buf = ML::MetalBuffer.from_array(x)
x16_buf = h16_input ? metal_buffer_from_u16(x.map { |v| f32_to_f16_bits(v) }) : nil
default_pipe = ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out", GEMM_SOURCE)
h16_pipe = h16_input ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32out", GEMM_SOURCE) : nil
single_pipe = ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_single", GEMM_SOURCE)
dual_pipe = alpha_beta_dual ? ML::Metal::ComputePipeline.new("simd_mv_q8_0_dual_f32", GEMV_SOURCE) : nil
mixed_pipe = (qkv_gate_mixed || ffn_upgate_mixed) ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", GEMM_SOURCE) : nil
rec_proj_mixed_pipe = rec_proj_mixed ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed", GEMM_SOURCE) : nil
rec_proj_tilefast_pipe = rec_proj_tilefast ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed_tilefast", rec_proj_tilefast_source) : nil
rec_proj_mixed_single_pipe = rec_proj_mixed_singlebuf ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed_single", rec_proj_mixed_singlebuf_source) : nil
post_oproj_source = post_oproj_fused ? post_oproj_fused_source : nil
post_oproj_post_pipe = post_oproj_fused ? ML::Metal::ComputePipeline.new("qwen35_dn_post_norm_gate_chunk_out", post_oproj_source.not_nil!) : nil
post_oproj_inv_pipe = post_oproj_fused ? ML::Metal::ComputePipeline.new("qwen35_dn_post_inv_rms_rows", post_oproj_source.not_nil!) : nil
post_oproj_fused_pipe = post_oproj_fused ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_dn_post_oproj", post_oproj_source.not_nil!) : nil
ffn_down_source = ffn_down_fused ? ffn_down_fused_input_source : nil
ffn_swiglu_pipe = ffn_down_fused ? ML::Metal::ComputePipeline.new("qwen35_swiglu_mul", FFN_SOURCE) : nil
ffn_down_fused_pipe = ffn_down_fused ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_ffn_down_swiglu", ffn_down_source.not_nil!) : nil
no_sg_source = no_sg_barriers ? no_sg_barrier_source : nil
no_sg_default_pipe = no_sg_barriers ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out", no_sg_source.not_nil!) : nil
no_sg_mixed_pipe = no_sg_barriers ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", no_sg_source.not_nil!) : nil
no_sg_rec_proj_mixed_pipe = no_sg_barriers ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed", no_sg_source.not_nil!) : nil
two_output_tilefast_pipe = two_output_tilefast ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed_tilefast", two_output_tilefast_source) : nil
rec_conv_pipe = rec_prep_split ? ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk", REC_SOURCE) : nil
rec_l2_pipe = rec_prep_split ? ML::Metal::ComputePipeline.new("qwen35_l2_heads_chunk", REC_SOURCE) : nil
rec_ab_pipe = rec_prep_split ? ML::Metal::ComputePipeline.new("qwen35_recurrent_ab_chunk", REC_SOURCE) : nil
rec_conv_token_source = rec_conv_token_parallel ? rec_conv_token_parallel_source : nil
rec_conv_token_default_pipe = rec_conv_token_parallel ? ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk", REC_SOURCE) : nil
rec_conv_token_pipe = rec_conv_token_parallel ? ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk_token_parallel", rec_conv_token_source.not_nil!) : nil
rec_conv_token_state_pipe = rec_conv_token_parallel ? ML::Metal::ComputePipeline.new("qwen35_recurrent_conv_shift_chunk_token_parallel_state", rec_conv_token_source.not_nil!) : nil
ffn_upgate_pair_rows_pipe = ffn_upgate_pair_rows ? ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_ffn_upgate_pair_rows", ffn_upgate_pair_rows_source) : nil
q8_gemv_base_pipe = q8_gemv_layout_sweep ? ML::Metal::ComputePipeline.new("simd_mv_q8_0_f32", GEMV_SOURCE) : nil
q8_gemv_layouts = [
  Q8GemvLayout.new(2, 1),
  Q8GemvLayout.new(4, 2),
  Q8GemvLayout.new(8, 1),
  Q8GemvLayout.new(8, 2),
]
q8_gemv_layout_pipes = q8_gemv_layout_sweep ? q8_gemv_layouts.map do |layout|
  {
    layout,
    ML::Metal::ComputePipeline.new("simd_mv_q8_0_f32", q8_gemv_layout_source(layout)),
  }
end : [] of {Q8GemvLayout, ML::Metal::ComputePipeline}
corridors = corridor == "all" ? ["rec_proj", "ffn_upgate"] : [corridor]

puts "Qwen35 Q8 hot-shape microprobe"
puts "model=#{model}"
puts "batch=#{batch} runs=#{runs} warmup=#{warmup} per_op=#{per_op} alpha_beta_dual=#{alpha_beta_dual} qkv_gate_mixed=#{qkv_gate_mixed} rec_proj_mixed=#{rec_proj_mixed} ffn_upgate_mixed=#{ffn_upgate_mixed} h16_input=#{h16_input} rec_proj_tilefast=#{rec_proj_tilefast} rec_proj_mixed_singlebuf=#{rec_proj_mixed_singlebuf} post_oproj_fused=#{post_oproj_fused} ffn_down_fused=#{ffn_down_fused} no_sg_barriers=#{no_sg_barriers} two_output_tilefast=#{two_output_tilefast} rec_prep_split=#{rec_prep_split} ffn_upgate_pair_rows=#{ffn_upgate_pair_rows} q8_gemv_layout_sweep=#{q8_gemv_layout_sweep} rec_conv_token_parallel=#{rec_conv_token_parallel}"
puts "boundary=direct Metal F32-input Q8 GEMM kernels; one command buffer per corridor; readback only for delta"
puts "kernels=simd_mm_q8_0_f32in_f32out vs simd_mm_q8_0_f32in_f32out_single"
puts "dual_kernel=simd_mv_q8_0_dual_f32" if alpha_beta_dual
puts "mixed_kernel=simd_mm_q8_0_f32in_f32out_qkv_gate_mixed" if qkv_gate_mixed
puts "rec_proj_mixed_kernel=simd_mm_q8_0_f32in_f32out_rec_proj_mixed" if rec_proj_mixed
puts "rec_proj_tilefast_kernel=simd_mm_q8_0_f32in_f32out_rec_proj_mixed_tilefast" if rec_proj_tilefast
puts "rec_proj_mixed_singlebuf_kernel=simd_mm_q8_0_f32in_f32out_rec_proj_mixed_single" if rec_proj_mixed_singlebuf
puts "post_oproj_fused_kernel=simd_mm_q8_0_f32in_f32out_dn_post_oproj" if post_oproj_fused
puts "ffn_down_fused_kernel=simd_mm_q8_0_f32in_f32out_ffn_down_swiglu" if ffn_down_fused
puts "no_sg_barriers_source=GEMM_SOURCE without simdgroup_barrier(mem_none)" if no_sg_barriers
puts "two_output_tilefast_kernel=simd_mm_q8_0_f32in_f32out_qkv_gate_mixed_tilefast" if two_output_tilefast
puts "rec_prep_kernels=qwen35_recurrent_conv_shift_chunk,qwen35_l2_heads_chunk,qwen35_recurrent_ab_chunk" if rec_prep_split
puts "rec_conv_token_parallel_kernels=qwen35_recurrent_conv_shift_chunk_token_parallel,+state" if rec_conv_token_parallel
puts "ffn_upgate_pair_rows_kernel=simd_mm_q8_0_f32in_f32out_ffn_upgate_pair_rows" if ffn_upgate_pair_rows
puts "q8_gemv_layouts=#{q8_gemv_layouts.map { |layout| "#{layout.nsg}x#{layout.nr0}" }.join(",")}" if q8_gemv_layout_sweep

corridors.each do |corridor_name|
  bench_ops = build_bench_ops(corridor_ops(corridor_name, rec), batch)
  puts
  puts "corridor=#{corridor_name}"
  bench_ops.each do |op|
    qw = op.qw
    puts "  op=#{op.name} shape=#{qw.in_dim}x#{qw.out_dim} type=#{qw.type.name} raw_mib=#{fmt(qw.raw.size.to_f64 / 1024.0 / 1024.0)}"
  end

  run_paired("bench=corridor ops=#{bench_ops.size}", bench_ops, x_buf, batch, default_pipe, single_pipe, warmup, runs)

  if no_sg_barriers
    run_paired("bench=no_sg_barriers_separate ops=#{bench_ops.size}", bench_ops, x_buf, batch, default_pipe, no_sg_default_pipe.not_nil!, warmup, runs)
  end

  if per_op
    bench_ops.each do |op|
      qw = op.qw
      run_paired("bench=op:#{op.name} shape=#{qw.in_dim}x#{qw.out_dim}", [op], x_buf, batch, default_pipe, single_pipe, warmup, runs)
      if h16_input
        run_h16_input_probe("h16_input_op:#{op.name} shape=#{qw.in_dim}x#{qw.out_dim}", [op], x_buf, x16_buf.not_nil!, batch, default_pipe, h16_pipe.not_nil!, warmup, runs)
      end
    end
  end

  if h16_input
    run_h16_input_probe("h16_input_corridor:#{corridor_name}", bench_ops, x_buf, x16_buf.not_nil!, batch, default_pipe, h16_pipe.not_nil!, warmup, runs)
  end

  if alpha_beta_dual && corridor_name == "rec_proj"
    run_alpha_beta_dual_probe(bench_ops, x_buf, batch, default_pipe, dual_pipe.not_nil!, warmup, runs)
  end

  if qkv_gate_mixed && corridor_name == "rec_proj"
    run_qkv_gate_mixed_probe(bench_ops, x_buf, batch, default_pipe, mixed_pipe.not_nil!, warmup, runs)
  end

  if two_output_tilefast && corridor_name == "rec_proj"
    mixed = mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", GEMM_SOURCE)
    run_qkv_gate_tilefast_probe(bench_ops, x_buf, batch, mixed, two_output_tilefast_pipe.not_nil!, warmup, runs)
  end

  if rec_proj_mixed && corridor_name == "rec_proj"
    run_rec_proj_mixed_probe(bench_ops, x_buf, batch, default_pipe, rec_proj_mixed_pipe.not_nil!, warmup, runs)
  end

  if rec_proj_tilefast && corridor_name == "rec_proj"
    mixed = rec_proj_mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed", GEMM_SOURCE)
    run_rec_proj_tilefast_probe(bench_ops, x_buf, batch, mixed, rec_proj_tilefast_pipe.not_nil!, warmup, runs)
  end

  if rec_proj_mixed_singlebuf && corridor_name == "rec_proj"
    mixed = rec_proj_mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed", GEMM_SOURCE)
    run_rec_proj_mixed_variant_probe("rec_proj_mixed_singlebuf", "singlebuf", bench_ops, x_buf, batch,
      mixed, rec_proj_mixed_single_pipe.not_nil!, warmup, runs)
  end

  if no_sg_barriers && corridor_name == "rec_proj"
    mixed = rec_proj_mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_rec_proj_mixed", GEMM_SOURCE)
    run_rec_proj_mixed_variant_probe("rec_proj_mixed_no_sg_barriers", "no_sg", bench_ops, x_buf, batch,
      mixed, no_sg_rec_proj_mixed_pipe.not_nil!, warmup, runs)
  end

  if ffn_upgate_mixed && corridor_name == "ffn_upgate"
    run_ffn_upgate_mixed_probe(bench_ops, x_buf, batch, default_pipe, mixed_pipe.not_nil!, warmup, runs)
  end

  if two_output_tilefast && corridor_name == "ffn_upgate"
    mixed = mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", GEMM_SOURCE)
    run_ffn_upgate_tilefast_probe(bench_ops, x_buf, batch, mixed, two_output_tilefast_pipe.not_nil!, warmup, runs)
  end

  if no_sg_barriers && corridor_name == "ffn_upgate"
    mixed = mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", GEMM_SOURCE)
    run_two_output_mixed_variant_probe("ffn_upgate_mixed_no_sg_barriers", "no_sg", bench_ops, x_buf, batch,
      mixed, no_sg_mixed_pipe.not_nil!, warmup, runs)
  end

  if ffn_upgate_pair_rows && corridor_name == "ffn_upgate"
    mixed = mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", GEMM_SOURCE)
    run_ffn_upgate_pair_rows_probe(bench_ops, x_buf, batch, mixed, ffn_upgate_pair_rows_pipe.not_nil!, warmup, runs)
  end

  if q8_gemv_layout_sweep
    run_q8_gemv_layout_sweep_probe(bench_ops, x_buf, batch, q8_gemv_base_pipe.not_nil!,
      q8_gemv_layout_pipes, warmup, runs)
  end
end

if post_oproj_fused
  run_post_oproj_fused_probe(rec, batch, post_oproj_post_pipe.not_nil!, default_pipe,
    post_oproj_inv_pipe.not_nil!, post_oproj_fused_pipe.not_nil!, warmup, runs)
end

if ffn_down_fused
  upgate = mixed_pipe || ML::Metal::ComputePipeline.new("simd_mm_q8_0_f32in_f32out_qkv_gate_mixed", GEMM_SOURCE)
  run_ffn_down_fused_probe(rec, batch, upgate, ffn_swiglu_pipe.not_nil!, default_pipe,
    ffn_down_fused_pipe.not_nil!, warmup, runs)
end

if rec_prep_split
  run_rec_prep_split_probe(weights, rec, batch, rec_conv_pipe.not_nil!, rec_l2_pipe.not_nil!, rec_ab_pipe.not_nil!, warmup, runs)
end

if rec_conv_token_parallel
  run_rec_conv_token_parallel_probe(weights, rec, batch, rec_conv_token_default_pipe.not_nil!,
    rec_conv_token_pipe.not_nil!, rec_conv_token_state_pipe.not_nil!, warmup, runs)
end
