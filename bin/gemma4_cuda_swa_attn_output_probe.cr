# CogniGemma CUDA SWA attention-output composition smoke.
#
# Composes the certified corridor:
# input RMSNorm -> Q/K/V projection -> per-head norms -> RoPE -> KV write ->
# parallel SWA context -> attn_output projection. It stops before residual add,
# post-attention norm, FFN, and layer-output scale.

require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/cuda/driver"
require "../src/ml/cuda/qwen_full_attn_projection_runner"
require "../src/ml/cuda/gemma4_swa_context_ptx"

NORM_PTX = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
Q4K_PTX  = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q6K_PTX  = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}
ROPE_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry rope_neox_apply_batched_probe(
    .param .u64 x,
    .param .u64 cos_t,
    .param .u64 sin_t,
    .param .u32 head_dim,
    .param .u32 n_rot
)
{
    .reg .pred %p<4>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<9>;

    ld.param.u64 %rd1, [x];
    ld.param.u64 %rd2, [cos_t];
    ld.param.u64 %rd3, [sin_t];
    ld.param.u32 %r1, [head_dim];
    ld.param.u32 %r2, [n_rot];

    mov.u32 %r3, %ctaid.x;       // row
    mov.u32 %r4, %tid.x;         // i
    shr.u32 %r5, %r2, 1;         // half
    setp.ge.u32 %p1, %r4, %r5;
    @%p1 bra DONE;

    mul.lo.u32 %r6, %r3, %r1;
    add.u32 %r7, %r6, %r4;       // a
    add.u32 %r8, %r7, %r5;       // b

    mul.wide.u32 %rd4, %r7, 4;
    add.s64 %rd5, %rd1, %rd4;
    mul.wide.u32 %rd6, %r8, 4;
    add.s64 %rd7, %rd1, %rd6;
    mul.wide.u32 %rd8, %r4, 4;
    add.s64 %rd9, %rd2, %rd8;
    add.s64 %rd10, %rd3, %rd8;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd7];
    ld.global.f32 %f3, [%rd9];
    ld.global.f32 %f4, [%rd10];

    mul.rn.f32 %f5, %f1, %f3;
    neg.f32 %f7, %f4;
    fma.rn.f32 %f5, %f2, %f7, %f5;
    mul.rn.f32 %f6, %f1, %f4;
    fma.rn.f32 %f6, %f2, %f3, %f6;
    st.global.f32 [%rd5], %f5;
    st.global.f32 [%rd7], %f6;

DONE:
    ret;
}
PTX

DEFAULT_MODEL = ENV["GEMMA4_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

def bytesize_f32(elements : Int32) : LibC::SizeT
  (elements * sizeof(Float32)).to_u64
end

def max_abs_diff(a : Array(Float32), b : Array(Float32)) : Float32
  raise ArgumentError.new("size mismatch") unless a.size == b.size
  max = 0.0_f32
  a.each_with_index do |v, i|
    diff = (v - b[i]).abs
    max = diff if diff > max
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

def rope_tables(pos : Int32, n_rot : Int32, freq_base : Float32) : {Array(Float32), Array(Float32)}
  half = n_rot // 2
  cos = Array(Float32).new(half, 0.0_f32)
  sin = Array(Float32).new(half, 0.0_f32)
  half.times do |i|
    i0 = 2 * i
    theta = pos.to_f32 * (freq_base ** (-i0.to_f32 / n_rot.to_f32))
    cos[i] = Math.cos(theta).to_f32
    sin[i] = Math.sin(theta).to_f32
  end
  {cos, sin}
end

def launch_norm(fn : ML::CUDA::KernelFunction,
                x_ptr : ML::CUDA::DevicePtr,
                norm_ptr : ML::CUDA::DevicePtr,
                out_ptr : ML::CUDA::DevicePtr,
                rows : Int32,
                dim : Int32,
                eps : Float32,
                label : String) : Nil
  d_x = x_ptr
  d_norm = norm_ptr
  d_out = out_ptr
  dim_u32 = dim.to_u32
  eps_f32 = eps
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_x).as(Void*)
  params[1] = pointerof(d_norm).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(dim_u32).as(Void*)
  params[4] = pointerof(eps_f32).as(Void*)
  ML::CUDA.launch!(fn, rows.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, label)
end

def launch_rope(fn : ML::CUDA::KernelFunction,
                x_ptr : ML::CUDA::DevicePtr,
                cos_ptr : ML::CUDA::DevicePtr,
                sin_ptr : ML::CUDA::DevicePtr,
                rows : Int32,
                head_dim : Int32,
                n_rot : Int32,
                label : String) : Nil
  d_x = x_ptr
  d_cos = cos_ptr
  d_sin = sin_ptr
  head_dim_u32 = head_dim.to_u32
  n_rot_u32 = n_rot.to_u32
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_x).as(Void*)
  params[1] = pointerof(d_cos).as(Void*)
  params[2] = pointerof(d_sin).as(Void*)
  params[3] = pointerof(head_dim_u32).as(Void*)
  params[4] = pointerof(n_rot_u32).as(Void*)
  block = {n_rot // 2, 256}.min.to_u32
  ML::CUDA.launch!(fn, rows.to_u32, 1_u32, 1_u32, block, 1_u32, 1_u32, params, label)
end

def launch_gemv(fn : ML::CUDA::KernelFunction,
                d_w : ML::CUDA::DevicePtr,
                d_x : ML::CUDA::DevicePtr,
                d_out : ML::CUDA::DevicePtr,
                in_dim : Int32,
                out_dim : Int32,
                label : String) : Nil
  in_dim_u32 = in_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  params = Pointer(Void*).malloc(5)
  params[0] = pointerof(d_w).as(Void*)
  params[1] = pointerof(d_x).as(Void*)
  params[2] = pointerof(d_out).as(Void*)
  params[3] = pointerof(in_dim_u32).as(Void*)
  params[4] = pointerof(out_dim_u32).as(Void*)
  grid = ((out_dim + 3) // 4).to_u32
  ML::CUDA.launch!(fn, grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, params, label)
end

model = DEFAULT_MODEL
layer = 0
tokens = 4
base_pos = 0
splitk_chunk = 1024
seed = 23_u64
reps = 10
warmup = 2

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_swa_attn_output_probe [--model PATH] [--layer N] [--tokens N] [--base-pos N] [--splitk-chunk N] [--reps N] [--warmup N] [--seed N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layer N", "Gemma4 SWA layer index with explicit V") { |v| layer = v.to_i }
  p.on("--tokens N", "Synthetic token span length") { |v| tokens = v.to_i }
  p.on("--base-pos N", "Absolute start position for the synthetic span") { |v| base_pos = v.to_i }
  p.on("--splitk-chunk N", "Parallel context chunk; use 1024 for exact one-window route") { |v| splitk_chunk = v.to_i }
  p.on("--reps N", "Timed launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokens must be positive" unless tokens > 0
raise "base-pos must be non-negative" unless base_pos >= 0
raise "splitk-chunk must be positive" unless splitk_chunk > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
hp = weights.hparams
raise "layer #{layer} is not SWA" unless hp.sliding_window?(layer)
lw = weights.layers[layer]
v_qw = lw.attn_v_qw || raise "SWA layer #{layer} unexpectedly lacks explicit V"
head_dim = hp.head_dim_for_layer(layer)
n_head = hp.n_head
n_head_kv = hp.n_head_kv(layer)
heads_per_group = n_head // n_head_kv
hidden = hp.n_embd
q_dim = n_head * head_dim
kv_dim = n_head_kv * head_dim
max_seq = base_pos + tokens
active_window_len = Math.min(max_seq, hp.sliding_window)
splitk_chunks = (active_window_len + splitk_chunk - 1) // splitk_chunk

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }
initial_k_cache = Array(Float32).new(max_seq * kv_dim) { rng.rand(-0.25_f32..0.25_f32) }
initial_v_cache = Array(Float32).new(max_seq * kv_dim) { rng.rand(-0.25_f32..0.25_f32) }

state = ML::GGUF::Gemma4CPU::State.new(hp, max_seq)
state.layers[layer].k_cache = initial_k_cache.dup
state.layers[layer].v_cache = initial_v_cache.dup
cpu = Array(Float32).new(tokens * hidden, 0.0_f32)
cpu_t0 = Time.instant
tokens.times do |tok|
  abs_pos = base_pos + tok
  row = xs[tok * hidden, hidden]
  out = ML::GGUF::Gemma4CPU.attention_projected_output(weights, layer, row, abs_pos, state)
  hidden.times { |i| cpu[tok * hidden + i] = out[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

proj_weights = ML::CUDA::QwenFullAttnProjectionRunner::Weights.new(
  hidden, q_dim, kv_dim, kv_dim,
  lw.attn_q_qw.raw, lw.attn_k_qw.raw, v_qw.raw, v_qw.type)

ctx = nil.as(ML::CUDA::Context?)
proj_runner = nil.as(ML::CUDA::QwenFullAttnProjectionRunner?)
norm_mod = nil.as(ML::CUDA::CUDAModule?)
rope_mod = nil.as(ML::CUDA::CUDAModule?)
attn_mod = nil.as(ML::CUDA::CUDAModule?)
kv_mod = nil.as(ML::CUDA::CUDAModule?)
q4_mod = nil.as(ML::CUDA::CUDAModule?)
q6_mod = nil.as(ML::CUDA::CUDAModule?)
buffers = [] of ML::CUDA::DeviceBuffer

begin
  ctx = ML::CUDA::Context.create
  proj_runner = ML::CUDA::QwenFullAttnProjectionRunner.from_weights_with_input_norm(proj_weights, tokens, xs, lw.attn_norm, hp.rms_eps)
  norm_mod = ML::CUDA::CUDAModule.load(NORM_PTX, "gemma4_swa_attn_out_norm")
  rope_mod = ML::CUDA::CUDAModule.load(ROPE_PTX, "gemma4_swa_attn_out_rope")
  attn_mod = ML::CUDA::CUDAModule.load(ML::CUDA::Gemma4SWAContextPTX::ATTN_PTX, "gemma4_swa_attn_out_context")
  kv_mod = ML::CUDA::CUDAModule.load(ML::CUDA::Gemma4SWAContextPTX::KV_WRITE_PTX, "gemma4_swa_attn_out_kv")
  q4_mod = ML::CUDA::CUDAModule.load(Q4K_PTX, "gemma4_swa_attn_out_q4")
  q6_mod = ML::CUDA::CUDAModule.load(Q6K_PTX, "gemma4_swa_attn_out_q6")

  norm_fn = norm_mod.function("rmsnorm_vec_parallel_batched_probe")
  rope_fn = rope_mod.function("rope_neox_apply_batched_probe")
  kv_write_fn = kv_mod.function("gemma4_kv_cache_write_probe")
  splitk_part_fn = attn_mod.function("gemma4_swa_ungated_attn_splitk_part_probe")
  splitk_reduce_fn = attn_mod.function("gemma4_swa_ungated_attn_splitk_reduce_probe")
  q4_fn = q4_mod.function("q4_k_gemv_warp4_f32")
  q6_fn = q6_mod.function("q6_k_gemv_warp4_f32")
  out_fn = lw.attn_output_qw.type.q4_k? ? q4_fn : q6_fn

  q_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_q_norm.size)); buffers << q_norm_buf
  k_norm_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(lw.attn_k_norm.size)); buffers << k_norm_buf
  ones_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(head_dim)); buffers << ones_buf
  q_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * q_dim)); buffers << q_buf
  k_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * kv_dim)); buffers << k_buf
  v_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * kv_dim)); buffers << v_buf
  cos_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(hp.rope_dim_for_layer(layer) // 2)); buffers << cos_buf
  sin_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(hp.rope_dim_for_layer(layer) // 2)); buffers << sin_buf
  k_cache_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(max_seq * kv_dim)); buffers << k_cache_buf
  v_cache_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(max_seq * kv_dim)); buffers << v_cache_buf
  scores_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * n_head * max_seq)); buffers << scores_buf
  ctx_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * q_dim)); buffers << ctx_buf
  splitk_meta_count = tokens * n_head * splitk_chunks
  splitk_o_count = splitk_meta_count * head_dim
  splitk_m_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(splitk_meta_count)); buffers << splitk_m_buf
  splitk_l_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(splitk_meta_count)); buffers << splitk_l_buf
  splitk_o_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(splitk_o_count)); buffers << splitk_o_buf
  start_pos_buf = ML::CUDA::DeviceBuffer.new(sizeof(UInt32).to_u64); buffers << start_pos_buf
  out_w_buf = ML::CUDA::DeviceBuffer.new(lw.attn_output_qw.raw.size.to_u64); buffers << out_w_buf
  out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * hidden)); buffers << out_buf

  ones = Array(Float32).new(head_dim, 1.0_f32)
  start_pos_value = base_pos.to_u32
  ML::CUDA.copy_htod!(q_norm_buf.ptr, lw.attn_q_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_q_norm.size), "q_norm")
  ML::CUDA.copy_htod!(k_norm_buf.ptr, lw.attn_k_norm.to_unsafe.as(Void*), bytesize_f32(lw.attn_k_norm.size), "k_norm")
  ML::CUDA.copy_htod!(ones_buf.ptr, ones.to_unsafe.as(Void*), bytesize_f32(ones.size), "plain_v_norm")
  ML::CUDA.copy_htod!(k_cache_buf.ptr, initial_k_cache.to_unsafe.as(Void*), bytesize_f32(initial_k_cache.size), "k_cache_init")
  ML::CUDA.copy_htod!(v_cache_buf.ptr, initial_v_cache.to_unsafe.as(Void*), bytesize_f32(initial_v_cache.size), "v_cache_init")
  ML::CUDA.copy_htod!(start_pos_buf.ptr, pointerof(start_pos_value).as(Void*), sizeof(UInt32).to_u64, "start_pos")
  ML::CUDA.copy_htod!(out_w_buf.ptr, lw.attn_output_qw.raw.to_unsafe.as(Void*), lw.attn_output_qw.raw.size.to_u64, "attn_output_w")
  proj_runner.upload_weights
  proj_runner.reset_sequence

  kv_dim_u32 = kv_dim.to_u32
  start_u32 = base_pos.to_u32
  d_k = k_buf.ptr
  d_v = v_buf.ptr
  d_kc = k_cache_buf.ptr
  d_vc = v_cache_buf.ptr
  kv_params = Pointer(Void*).malloc(6)
  kv_params[0] = pointerof(d_k).as(Void*)
  kv_params[1] = pointerof(d_v).as(Void*)
  kv_params[2] = pointerof(d_kc).as(Void*)
  kv_params[3] = pointerof(d_vc).as(Void*)
  kv_params[4] = pointerof(kv_dim_u32).as(Void*)
  kv_params[5] = pointerof(start_u32).as(Void*)

  d_q = q_buf.ptr
  d_scores = scores_buf.ptr
  d_ctx = ctx_buf.ptr
  d_start_pos = start_pos_buf.ptr
  d_splitk_m = splitk_m_buf.ptr
  d_splitk_l = splitk_l_buf.ptr
  d_splitk_o = splitk_o_buf.ptr
  n_head_u32 = n_head.to_u32
  n_head_kv_u32 = n_head_kv.to_u32
  head_dim_u32 = head_dim.to_u32
  hpg_u32 = heads_per_group.to_u32
  max_seq_u32 = max_seq.to_u32
  window_size_u32 = hp.sliding_window.to_u32
  scale = 1.0_f32
  splitk_chunk_u32 = splitk_chunk.to_u32
  splitk_chunks_u32 = splitk_chunks.to_u32
  part_params = Pointer(Void*).malloc(17)
  part_params[0] = pointerof(d_q).as(Void*)
  part_params[1] = pointerof(d_kc).as(Void*)
  part_params[2] = pointerof(d_vc).as(Void*)
  part_params[3] = pointerof(d_scores).as(Void*)
  part_params[4] = pointerof(d_splitk_m).as(Void*)
  part_params[5] = pointerof(d_splitk_l).as(Void*)
  part_params[6] = pointerof(d_splitk_o).as(Void*)
  part_params[7] = pointerof(n_head_u32).as(Void*)
  part_params[8] = pointerof(n_head_kv_u32).as(Void*)
  part_params[9] = pointerof(head_dim_u32).as(Void*)
  part_params[10] = pointerof(hpg_u32).as(Void*)
  part_params[11] = pointerof(d_start_pos).as(Void*)
  part_params[12] = pointerof(max_seq_u32).as(Void*)
  part_params[13] = pointerof(window_size_u32).as(Void*)
  part_params[14] = pointerof(scale).as(Void*)
  part_params[15] = pointerof(splitk_chunk_u32).as(Void*)
  part_params[16] = pointerof(splitk_chunks_u32).as(Void*)

  reduce_params = Pointer(Void*).malloc(9)
  reduce_params[0] = pointerof(d_splitk_m).as(Void*)
  reduce_params[1] = pointerof(d_splitk_l).as(Void*)
  reduce_params[2] = pointerof(d_splitk_o).as(Void*)
  reduce_params[3] = pointerof(d_ctx).as(Void*)
  reduce_params[4] = pointerof(n_head_u32).as(Void*)
  reduce_params[5] = pointerof(head_dim_u32).as(Void*)
  reduce_params[6] = pointerof(d_start_pos).as(Void*)
  reduce_params[7] = pointerof(max_seq_u32).as(Void*)
  reduce_params[8] = pointerof(splitk_chunks_u32).as(Void*)

  run_once = -> {
    proj_runner.not_nil!.run_sequence
    launch_norm(norm_fn, proj_runner.not_nil!.q_device_ptr, q_norm_buf.ptr, q_buf.ptr, tokens * n_head, head_dim, hp.rms_eps, "q_head_norm")
    launch_norm(norm_fn, proj_runner.not_nil!.k_device_ptr, k_norm_buf.ptr, k_buf.ptr, tokens * n_head_kv, head_dim, hp.rms_eps, "k_head_norm")
    launch_norm(norm_fn, proj_runner.not_nil!.v_device_ptr, ones_buf.ptr, v_buf.ptr, tokens * n_head_kv, head_dim, hp.rms_eps, "v_plain_norm")

    tokens.times do |tok|
      abs_pos = base_pos + tok
      cos, sin = rope_tables(abs_pos, hp.rope_dim_for_layer(layer), hp.rope_freq_base_for_layer(layer))
      ML::CUDA.copy_htod!(cos_buf.ptr, cos.to_unsafe.as(Void*), bytesize_f32(cos.size), "rope_cos")
      ML::CUDA.copy_htod!(sin_buf.ptr, sin.to_unsafe.as(Void*), bytesize_f32(sin.size), "rope_sin")
      launch_rope(rope_fn, q_buf.ptr + bytesize_f32(tok * q_dim), cos_buf.ptr, sin_buf.ptr, n_head, head_dim, hp.rope_dim_for_layer(layer), "q_rope")
      launch_rope(rope_fn, k_buf.ptr + bytesize_f32(tok * kv_dim), cos_buf.ptr, sin_buf.ptr, n_head_kv, head_dim, hp.rope_dim_for_layer(layer), "k_rope")
    end

    ML::CUDA.launch!(kv_write_fn, tokens.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, kv_params, "kv_write")
    ML::CUDA.launch!(splitk_part_fn, tokens.to_u32, n_head.to_u32, splitk_chunks_u32, 256_u32, 1_u32, 1_u32, part_params, "swa_attn_part")
    ML::CUDA.launch!(splitk_reduce_fn, tokens.to_u32, n_head.to_u32, 1_u32, 256_u32, 1_u32, 1_u32, reduce_params, "swa_attn_reduce")
    tokens.times do |tok|
      launch_gemv(out_fn, out_w_buf.ptr, ctx_buf.ptr + bytesize_f32(tok * q_dim), out_buf.ptr + bytesize_f32(tok * hidden), q_dim, hidden, "attn_output")
    end
  }

  warmup.times { run_once.call }
  ML::CUDA.synchronize!("warmup")
  t0 = Time.instant
  reps.times { run_once.call }
  ML::CUDA.synchronize!("timed")
  cuda_ms = (Time.instant - t0).total_milliseconds / reps

  gpu = Array(Float32).new(tokens * hidden, 0.0_f32)
  ML::CUDA.copy_dtoh!(gpu.to_unsafe.as(Void*), out_buf.ptr, bytesize_f32(gpu.size), "attn_projected")
  cos_v = cosine(gpu, cpu)
  diff = max_abs_diff(gpu, cpu)
  ok = cos_v >= 0.99999 && diff <= 1.0e-3_f32

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "tokens=#{tokens}"
  puts "base_pos=#{base_pos}"
  puts "sliding_window=#{hp.sliding_window}"
  puts "active_window_len=#{active_window_len}"
  puts "splitk_chunk=#{splitk_chunk}"
  puts "splitk_chunks=#{splitk_chunks}"
  puts "hidden=#{hidden}"
  puts "q_dim=#{q_dim}"
  puts "kv_dim=#{kv_dim}"
  puts "cuda_ms=#{cuda_ms.round(4)}"
  puts "cuda_ms_per_token=#{(cuda_ms / tokens).round(4)}"
  puts "cpu_ms=#{cpu_ms.round(4)}"
  puts "cos=#{cos_v.round(8)}"
  puts "max_diff=#{diff}"
  puts "ok=#{ok}"
  exit(ok ? 0 : 1)
ensure
  buffers.each(&.close)
  q6_mod.try(&.close)
  q4_mod.try(&.close)
  kv_mod.try(&.close)
  attn_mod.try(&.close)
  rope_mod.try(&.close)
  norm_mod.try(&.close)
  proj_runner.try(&.close)
  ctx.try(&.close)
end
