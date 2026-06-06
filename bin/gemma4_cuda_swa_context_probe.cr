# CogniGemma CUDA SWA KV-cache + attention-context smoke.
#
# Uses already-normalized/RoPE'd synthetic Q/K/V rows. A tiny CUDA copy kernel
# appends K/V rows into resident caches, then a Gemma-specific ungated serial
# CUDA GQA attention kernel reads the cache. This stops before attn_output
# projection.

require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/cuda/driver"

FULL_ATTN_PTX               = {{ read_file("src/ml/cuda/kernels/fullattn_post_probe.ptx") }}
GATED_ATTENTION_START       = "// Correctness-first gated GQA attention decode over a resident KV cache."
GATED_ATTENTION_END         = "// Split-K long-context GQA attention decode, stage 1."
GATED_ATTENTION_START_INDEX = FULL_ATTN_PTX.index(GATED_ATTENTION_START) || raise "gated attention kernel start not found"
GATED_ATTENTION_END_INDEX   = FULL_ATTN_PTX.index(GATED_ATTENTION_END) || raise "gated attention kernel end not found"
GATED_ATTENTION             = FULL_ATTN_PTX[GATED_ATTENTION_START_INDEX, GATED_ATTENTION_END_INDEX - GATED_ATTENTION_START_INDEX]
UNGATED_ATTENTION           = GATED_ATTENTION
  .gsub("gated GQA attention decode", "ungated GQA attention decode")
  .gsub("full_attn_decode_cache_probe", "gemma4_ungated_attn_decode_cache_probe")
  .gsub("    .param .u64 gate_in,\n", "")
  .gsub("    ld.param.u64 %rd2, [gate_in];\n", "")
  .gsub("    mul.lo.u32 %r16, %r15, %r3; // q/gate/out base", "    mul.lo.u32 %r16, %r15, %r3; // q/out base")
  .gsub(%(A_GATE_WRITE:
    mul.rn.f32 %f17, %f14, %f13;
    add.u32 %r33, %r16, %r27;
    mul.wide.u32 %rd19, %r33, 4;
    add.s64 %rd20, %rd2, %rd19;
    ld.global.f32 %f18, [%rd20]; // gate
    neg.f32 %f19, %f18;
    mov.f32 %f20, 0f3FB8AA3B;   // log2(e)
    mul.rn.f32 %f21, %f19, %f20;
    ex2.approx.ftz.f32 %f22, %f21;
    mov.f32 %f23, 0f3F800000;   // 1.0
    add.rn.f32 %f24, %f23, %f22;
    rcp.approx.ftz.f32 %f25, %f24;
    mul.rn.f32 %f26, %f17, %f25;
    add.s64 %rd21, %rd6, %rd19;
    st.global.f32 [%rd21], %f26;), %(A_GATE_WRITE:
    mul.rn.f32 %f17, %f14, %f13;
    add.u32 %r33, %r16, %r27;
    mul.wide.u32 %rd19, %r33, 4;
    add.s64 %rd21, %rd6, %rd19;
    st.global.f32 [%rd21], %f17;))
raise "ungated attention rewrite failed" unless UNGATED_ATTENTION.includes?("gemma4_ungated_attn_decode_cache_probe") && !UNGATED_ATTENTION.includes?("gate_in")

ATTN_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

#{UNGATED_ATTENTION}
PTX
KV_WRITE_PTX = <<-PTX
.version 8.0
.target sm_80
.address_size 64

.visible .entry gemma4_kv_cache_write_probe(
    .param .u64 k_in,
    .param .u64 v_in,
    .param .u64 k_cache,
    .param .u64 v_cache,
    .param .u32 kv_dim,
    .param .u32 start_pos
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<18>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd1, [k_in];
    ld.param.u64 %rd2, [v_in];
    ld.param.u64 %rd3, [k_cache];
    ld.param.u64 %rd4, [v_cache];
    ld.param.u32 %r1, [kv_dim];
    ld.param.u32 %r2, [start_pos];

    mov.u32 %r3, %ctaid.x;      // token row
    mov.u32 %r4, %tid.x;
    mov.u32 %r5, %ntid.x;
    mul.lo.u32 %r6, %r3, %r1;   // input base
    add.u32 %r7, %r2, %r3;      // absolute position
    mul.lo.u32 %r8, %r7, %r1;   // cache base
    mov.u32 %r9, %r4;

LOOP:
    setp.ge.u32 %p1, %r9, %r1;
    @%p1 bra DONE;
    add.u32 %r10, %r6, %r9;
    add.u32 %r11, %r8, %r9;
    mul.wide.u32 %rd5, %r10, 4;
    mul.wide.u32 %rd6, %r11, 4;
    add.s64 %rd7, %rd1, %rd5;
    add.s64 %rd8, %rd2, %rd5;
    add.s64 %rd9, %rd3, %rd6;
    add.s64 %rd10, %rd4, %rd6;
    ld.global.f32 %f1, [%rd7];
    st.global.f32 [%rd9], %f1;
    ld.global.f32 %f1, [%rd8];
    st.global.f32 [%rd10], %f1;
    add.u32 %r9, %r9, %r5;
    bra LOOP;

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

model = DEFAULT_MODEL
layer = 0
tokens = 4
seed = 23_u64
reps = 20
warmup = 3

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_swa_context_probe [--model PATH] [--layer N] [--tokens N] [--seed N] [--reps N] [--warmup N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layer N", "Gemma4 SWA layer index") { |v| layer = v.to_i }
  p.on("--tokens N", "Short context length; must fit inside SWA window") { |v| tokens = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokens must be positive" unless tokens > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
hp = weights.hparams
raise "layer #{layer} is not SWA" unless hp.sliding_window?(layer)
raise "tokens exceed SWA window" if tokens > hp.sliding_window
head_dim = hp.head_dim_for_layer(layer)
n_head = hp.n_head
n_head_kv = hp.n_head_kv(layer)
heads_per_group = n_head // n_head_kv
q_dim = n_head * head_dim
kv_dim = n_head_kv * head_dim
max_seq = tokens

rng = Random.new(seed)
q = Array(Float32).new(tokens * q_dim) { rng.rand(-0.25_f32..0.25_f32) }
k = Array(Float32).new(tokens * kv_dim) { rng.rand(-0.25_f32..0.25_f32) }
v = Array(Float32).new(tokens * kv_dim) { rng.rand(-0.25_f32..0.25_f32) }

state = ML::GGUF::Gemma4CPU::LayerState.new
cpu = Array(Float32).new(tokens * q_dim, 0.0_f32)
cpu_t0 = Time.instant
tokens.times do |tok|
  proj = ML::GGUF::Gemma4CPU::AttentionProjection.new(
    q[tok * q_dim, q_dim],
    k[tok * kv_dim, kv_dim],
    v[tok * kv_dim, kv_dim],
    false)
  out = ML::GGUF::Gemma4CPU.attention_context_from_projection!(proj, hp, layer, tok, state, max_seq)
  q_dim.times { |i| cpu[tok * q_dim + i] = out[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

ctx = nil.as(ML::CUDA::Context?)
attn_mod = nil.as(ML::CUDA::CUDAModule?)
kv_mod = nil.as(ML::CUDA::CUDAModule?)
buffers = [] of ML::CUDA::DeviceBuffer

begin
  ctx = ML::CUDA::Context.create
  attn_mod = ML::CUDA::CUDAModule.load(ATTN_PTX, "gemma4_context_attn")
  kv_mod = ML::CUDA::CUDAModule.load(KV_WRITE_PTX, "gemma4_context_kv")
  attn_fn = attn_mod.function("gemma4_ungated_attn_decode_cache_probe")
  kv_write_fn = kv_mod.function("gemma4_kv_cache_write_probe")

  q_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(q.size)); buffers << q_buf
  k_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(k.size)); buffers << k_buf
  v_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(v.size)); buffers << v_buf
  k_cache_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(max_seq * kv_dim)); buffers << k_cache_buf
  v_cache_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(max_seq * kv_dim)); buffers << v_cache_buf
  scores_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * n_head * max_seq)); buffers << scores_buf
  out_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(tokens * q_dim)); buffers << out_buf
  start_pos_buf = ML::CUDA::DeviceBuffer.new(sizeof(UInt32).to_u64); buffers << start_pos_buf
  zero_start = 0_u32

  ML::CUDA.copy_htod!(q_buf.ptr, q.to_unsafe.as(Void*), bytesize_f32(q.size), "q")
  ML::CUDA.copy_htod!(k_buf.ptr, k.to_unsafe.as(Void*), bytesize_f32(k.size), "k")
  ML::CUDA.copy_htod!(v_buf.ptr, v.to_unsafe.as(Void*), bytesize_f32(v.size), "v")
  ML::CUDA.copy_htod!(start_pos_buf.ptr, pointerof(zero_start).as(Void*), sizeof(UInt32).to_u64, "start_pos")

  kv_dim_u32 = kv_dim.to_u32
  start_u32 = 0_u32
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
  d_out = out_buf.ptr
  d_start_pos = start_pos_buf.ptr
  n_head_u32 = n_head.to_u32
  n_head_kv_u32 = n_head_kv.to_u32
  head_dim_u32 = head_dim.to_u32
  hpg_u32 = heads_per_group.to_u32
  max_seq_u32 = max_seq.to_u32
  scale = 1.0_f32
  attn_params = Pointer(Void*).malloc(12)
  attn_params[0] = pointerof(d_q).as(Void*)
  attn_params[1] = pointerof(d_kc).as(Void*)
  attn_params[2] = pointerof(d_vc).as(Void*)
  attn_params[3] = pointerof(d_scores).as(Void*)
  attn_params[4] = pointerof(d_out).as(Void*)
  attn_params[5] = pointerof(n_head_u32).as(Void*)
  attn_params[6] = pointerof(n_head_kv_u32).as(Void*)
  attn_params[7] = pointerof(head_dim_u32).as(Void*)
  attn_params[8] = pointerof(hpg_u32).as(Void*)
  attn_params[9] = pointerof(d_start_pos).as(Void*)
  attn_params[10] = pointerof(max_seq_u32).as(Void*)
  attn_params[11] = pointerof(scale).as(Void*)

  run_once = -> {
    ML::CUDA.launch!(kv_write_fn, tokens.to_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, kv_params, "kv_write")
    ML::CUDA.launch!(attn_fn, tokens.to_u32, n_head.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, attn_params, "gemma4_ungated_attn_context")
  }

  warmup.times { run_once.call }
  ML::CUDA.synchronize!("warmup")
  t0 = Time.instant
  reps.times { run_once.call }
  ML::CUDA.synchronize!("timed")
  cuda_ms = (Time.instant - t0).total_milliseconds / reps

  gpu = Array(Float32).new(tokens * q_dim, 0.0_f32)
  ML::CUDA.copy_dtoh!(gpu.to_unsafe.as(Void*), out_buf.ptr, bytesize_f32(gpu.size), "attn_out")

  cos = cosine(gpu, cpu)
  diff = max_abs_diff(gpu, cpu)
  ok = cos >= 0.99999 && diff <= 5.0e-4_f32
  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "tokens=#{tokens}"
  puts "head_dim=#{head_dim}"
  puts "n_head=#{n_head}"
  puts "n_head_kv=#{n_head_kv}"
  puts "heads_per_group=#{heads_per_group}"
  puts "cuda_ms=#{cuda_ms.round(4)}"
  puts "cuda_ms_per_token=#{(cuda_ms / tokens).round(4)}"
  puts "cpu_ms=#{cpu_ms.round(4)}"
  puts "cos=#{cos.round(8)}"
  puts "max_diff=#{diff}"
  puts "ok=#{ok}"
  exit(ok ? 0 : 1)
ensure
  buffers.each(&.close)
  kv_mod.try(&.close)
  attn_mod.try(&.close)
  ctx.try(&.close)
end
