# CogniGemma CUDA RoPE primitive smoke.
#
# Applies NeoX RoPE with precomputed cos/sin tables to Gemma4 SWA and
# full-attention Q/K rows. V is intentionally excluded: Gemma4 does not RoPE V.

require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/cuda/driver"

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

    ld.global.f32 %f1, [%rd5];   // x0
    ld.global.f32 %f2, [%rd7];   // x1
    ld.global.f32 %f3, [%rd9];   // cos
    ld.global.f32 %f4, [%rd10];  // sin

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

def rope_tables(pos : Int32, n_rot : Int32, freq_base : Float32, factors : Array(Float32)?) : {Array(Float32), Array(Float32)}
  half = n_rot // 2
  cos = Array(Float32).new(half, 0.0_f32)
  sin = Array(Float32).new(half, 0.0_f32)
  half.times do |i|
    i0 = 2 * i
    factor = factors ? factors.not_nil![i] : 1.0_f32
    theta = pos.to_f32 * (freq_base ** (-i0.to_f32 / n_rot.to_f32)) / factor
    cos[i] = Math.cos(theta).to_f32
    sin[i] = Math.sin(theta).to_f32
  end
  {cos, sin}
end

def cpu_apply_rope_rows!(x : Array(Float32), rows : Int32, head_dim : Int32,
                         n_rot : Int32, pos : Int32, base : Float32,
                         factors : Array(Float32)?) : Nil
  rows.times do |row|
    ML::GGUF::Gemma4CPU.rope_neox_slice!(x, row * head_dim, n_rot, head_dim, pos, base, factors)
  end
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

def run_case(name : String,
             fn : ML::CUDA::KernelFunction,
             rng : Random,
             rows : Int32,
             head_dim : Int32,
             n_rot : Int32,
             pos : Int32,
             base : Float32,
             factors : Array(Float32)?,
             reps : Int32,
             warmup : Int32) : Bool
  x_cpu = Array(Float32).new(rows * head_dim) { rng.rand(-1.0_f32..1.0_f32) }
  x_gpu = x_cpu.dup
  cpu_apply_rope_rows!(x_cpu, rows, head_dim, n_rot, pos, base, factors)
  cos, sin = rope_tables(pos, n_rot, base, factors)

  x_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(x_gpu.size))
  cos_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(cos.size))
  sin_buf = ML::CUDA::DeviceBuffer.new(bytesize_f32(sin.size))
  begin
    ML::CUDA.copy_htod!(x_buf.ptr, x_gpu.to_unsafe.as(Void*), bytesize_f32(x_gpu.size), "#{name} x")
    ML::CUDA.copy_htod!(cos_buf.ptr, cos.to_unsafe.as(Void*), bytesize_f32(cos.size), "#{name} cos")
    ML::CUDA.copy_htod!(sin_buf.ptr, sin.to_unsafe.as(Void*), bytesize_f32(sin.size), "#{name} sin")

    warmup.times { launch_rope(fn, x_buf.ptr, cos_buf.ptr, sin_buf.ptr, rows, head_dim, n_rot, "#{name} warmup") }
    ML::CUDA.synchronize!("warmup #{name}") if warmup > 0

    # Reset after warmup because RoPE is in-place.
    x_gpu = Array(Float32).new(rows * head_dim) { rng.rand(-1.0_f32..1.0_f32) }
    x_cpu = x_gpu.dup
    cpu_apply_rope_rows!(x_cpu, rows, head_dim, n_rot, pos, base, factors)
    ML::CUDA.copy_htod!(x_buf.ptr, x_gpu.to_unsafe.as(Void*), bytesize_f32(x_gpu.size), "#{name} reset")

    t0 = Time.instant
    reps.times do
      # Reset each rep to keep the operation identical and avoid repeated RoPE.
      ML::CUDA.copy_htod!(x_buf.ptr, x_gpu.to_unsafe.as(Void*), bytesize_f32(x_gpu.size), "#{name} timed_reset")
      launch_rope(fn, x_buf.ptr, cos_buf.ptr, sin_buf.ptr, rows, head_dim, n_rot, name)
    end
    ML::CUDA.synchronize!("timed #{name}")
    cuda_ms = (Time.instant - t0).total_milliseconds / reps

    gpu_out = Array(Float32).new(rows * head_dim, 0.0_f32)
    ML::CUDA.copy_dtoh!(gpu_out.to_unsafe.as(Void*), x_buf.ptr, bytesize_f32(gpu_out.size), "#{name} out")
    cos_v = cosine(gpu_out, x_cpu)
    diff = max_abs_diff(gpu_out, x_cpu)
    ok = cos_v >= 0.999999 && diff <= 2.0e-5_f32
    puts "case=#{name}"
    puts "rows=#{rows}"
    puts "head_dim=#{head_dim}"
    puts "n_rot=#{n_rot}"
    puts "pos=#{pos}"
    puts "base=#{base}"
    puts "factor_table=#{!factors.nil?}"
    puts "cuda_ms=#{cuda_ms.round(4)}"
    puts "cuda_ms_per_row=#{(cuda_ms / rows).round(6)}"
    puts "cos=#{cos_v.round(8)}"
    puts "max_diff=#{diff}"
    puts "ok=#{ok}"
    puts
    ok
  ensure
    sin_buf.close
    cos_buf.close
    x_buf.close
  end
end

model = DEFAULT_MODEL
seed = 23_u64
tokens = 4
pos = 7
reps = 20
warmup = 3

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_rope_probe [--model PATH] [--tokens N] [--pos N] [--reps N] [--warmup N] [--seed N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--tokens N", "Token rows to multiply by head counts") { |v| tokens = v.to_i }
  p.on("--pos N", "Position used for RoPE") { |v| pos = v.to_i }
  p.on("--reps N", "Timed launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokens must be positive" unless tokens > 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
hp = weights.hparams
ctx = nil.as(ML::CUDA::Context?)
mod = nil.as(ML::CUDA::CUDAModule?)

begin
  ctx = ML::CUDA::Context.create
  mod = ML::CUDA::CUDAModule.load(ROPE_PTX, "gemma4_rope")
  fn = mod.function("rope_neox_apply_batched_probe")
  rng = Random.new(seed)

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts

  ok_all = true
  ok_all &&= run_case("swa_q_rope", fn, rng, tokens * hp.n_head, hp.head_dim_swa,
    hp.rope_dim_count_swa, pos, hp.rope_freq_base_swa, nil, reps, warmup)
  ok_all &&= run_case("swa_k_rope", fn, rng, tokens * hp.n_head_kv(0), hp.head_dim_swa,
    hp.rope_dim_count_swa, pos, hp.rope_freq_base_swa, nil, reps, warmup)
  ok_all &&= run_case("full_q_rope", fn, rng, tokens * hp.n_head, hp.head_dim,
    hp.rope_dim_count, pos, hp.rope_freq_base, weights.rope_freqs, reps, warmup)
  ok_all &&= run_case("full_k_rope", fn, rng, tokens * hp.n_head_kv(5), hp.head_dim,
    hp.rope_dim_count, pos, hp.rope_freq_base, weights.rope_freqs, reps, warmup)
  puts "summary_ok=#{ok_all}"
  exit(ok_all ? 0 : 1)
ensure
  mod.try(&.close)
  ctx.try(&.close)
end
