require "option_parser"
require "../src/ml/cuda/driver"

PCA_UPDOWN_PTX = {{ read_file("src/ml/cuda/kernels/pca_updown_probe.ptx") }}

hidden = 4096
rank = 32
warmup = 5
reps = 50
seed = 1234_u64

OptionParser.parse do |p|
  p.banner = "Usage: cuda_pca_updown_probe [--hidden N] [--rank N] [--warmup N] [--reps N] [--seed N]"
  p.on("--hidden N", "Hidden dimension, default 4096") { |v| hidden = v.to_i }
  p.on("--rank N", "PCA-updown rank <= 64, default 32") { |v| rank = v.to_i }
  p.on("--warmup N", "Warmup iterations, default 5") { |v| warmup = v.to_i }
  p.on("--reps N", "Timed repetitions, default 50") { |v| reps = v.to_i }
  p.on("--seed N", "Deterministic data seed") { |v| seed = v.to_u64 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "hidden must be positive" unless hidden > 0
raise "rank must be in 1..64" unless rank > 0 && rank <= 64
raise "warmup must be non-negative" unless warmup >= 0
raise "reps must be positive" unless reps > 0

private def bytesize_f32(count : Int32) : LibC::SizeT
  (count.to_i64 * sizeof(Float32)).to_u64
end

private def fill_random(count : Int32, rng : Random, scale : Float32) : Array(Float32)
  Array(Float32).new(count) { ((rng.rand - 0.5) * 2.0 * scale).to_f32 }
end

private def cpu_updown(x : Array(Float32),
                       x_mean : Array(Float32),
                       c_mean : Array(Float32),
                       coeff_w : Array(Float32),
                       down : Array(Float32),
                       hidden : Int32,
                       rank : Int32) : Array(Float32)
  coeffs = Array(Float32).new(rank, 0.0_f32)
  rank.times do |j|
    acc = c_mean[j]
    base = j * hidden
    hidden.times do |d|
      acc += (x[d] - x_mean[d]) * coeff_w[base + d]
    end
    coeffs[j] = acc
  end

  out = Array(Float32).new(hidden, 0.0_f32)
  hidden.times do |d|
    acc = 0.0_f32
    rank.times do |j|
      acc += coeffs[j] * down[j * hidden + d]
    end
    out[d] = acc
  end
  out
end

rng = Random.new(seed)
x = fill_random(hidden, rng, 0.5_f32)
x_mean = fill_random(hidden, rng, 0.1_f32)
c_mean = fill_random(rank, rng, 0.1_f32)
coeff_w = fill_random(rank * hidden, rng, 0.002_f32)
down = fill_random(rank * hidden, rng, 0.02_f32)
expected = cpu_updown(x, x_mean, c_mean, coeff_w, down, hidden, rank)
actual = Array(Float32).new(hidden, 0.0_f32)

ctx = nil.as(ML::CUDA::Context?)
mod = nil.as(ML::CUDA::CUDAModule?)
buffers = [] of ML::CUDA::DeviceBuffer

begin
  ctx = ML::CUDA::Context.create
  mod = ML::CUDA::CUDAModule.load(PCA_UPDOWN_PTX, "pca_updown")
  fn = mod.not_nil!.function("ffn_pca_updown_fused_probe")

  sizes = [
    bytesize_f32(hidden),
    bytesize_f32(hidden),
    bytesize_f32(rank),
    bytesize_f32(rank * hidden),
    bytesize_f32(rank * hidden),
    bytesize_f32(hidden),
  ]
  ptrs = sizes.map do |size|
    buf = ML::CUDA::DeviceBuffer.new(size)
    buffers << buf
    buf.ptr
  end
  d_x, d_x_mean, d_c_mean, d_coeff_w, d_down, d_out = ptrs

  ML::CUDA.copy_htod!(d_x, x.to_unsafe.as(Void*), bytesize_f32(hidden), "x")
  ML::CUDA.copy_htod!(d_x_mean, x_mean.to_unsafe.as(Void*), bytesize_f32(hidden), "x_mean")
  ML::CUDA.copy_htod!(d_c_mean, c_mean.to_unsafe.as(Void*), bytesize_f32(rank), "c_mean")
  ML::CUDA.copy_htod!(d_coeff_w, coeff_w.to_unsafe.as(Void*), bytesize_f32(rank * hidden), "coeff_w")
  ML::CUDA.copy_htod!(d_down, down.to_unsafe.as(Void*), bytesize_f32(rank * hidden), "down")

  param_keepalive = [] of Void*
  box_ptr = ->(value : ML::CUDA::DevicePtr) {
    ptr = Pointer(ML::CUDA::DevicePtr).malloc(1)
    ptr.value = value
    param_keepalive << ptr.as(Void*)
    ptr
  }
  box_u32 = ->(value : UInt32) {
    ptr = Pointer(UInt32).malloc(1)
    ptr.value = value
    param_keepalive << ptr.as(Void*)
    ptr
  }

  params = Pointer(Void*).malloc(8)
  params[0] = box_ptr.call(d_x).as(Void*)
  params[1] = box_ptr.call(d_x_mean).as(Void*)
  params[2] = box_ptr.call(d_c_mean).as(Void*)
  params[3] = box_ptr.call(d_coeff_w).as(Void*)
  params[4] = box_ptr.call(d_down).as(Void*)
  params[5] = box_ptr.call(d_out).as(Void*)
  params[6] = box_u32.call(hidden.to_u32).as(Void*)
  params[7] = box_u32.call(rank.to_u32).as(Void*)

  launch = -> {
    ML::CUDA.launch!(fn, 1_u32, 1_u32, 1_u32, 256_u32, 1_u32, 1_u32, params, "pca updown")
  }

  warmup.times { launch.call }
  ML::CUDA.synchronize!("cuCtxSynchronize(pca updown warmup)")

  t0 = Time.instant
  reps.times { launch.call }
  ML::CUDA.synchronize!("cuCtxSynchronize(pca updown reps)")
  elapsed_ms = (Time.instant - t0).total_milliseconds

  ML::CUDA.copy_dtoh!(actual.to_unsafe.as(Void*), d_out, bytesize_f32(hidden), "out")

  max_abs = 0.0_f32
  sum_sq = 0.0_f64
  hidden.times do |i|
    delta = (actual[i] - expected[i]).abs
    max_abs = delta if delta > max_abs
    sum_sq += delta.to_f64 * delta.to_f64
  end
  rmse = Math.sqrt(sum_sq / hidden)

  puts "cuda_device=#{ctx.not_nil!.device_name}"
  puts "hidden=#{hidden}"
  puts "rank=#{rank}"
  puts "warmup=#{warmup}"
  puts "reps=#{reps}"
  puts "cuda_ms_total=#{elapsed_ms.round(6)}"
  puts "cuda_ms_per_call=#{(elapsed_ms / reps).round(6)}"
  puts "max_abs=#{max_abs}"
  puts "rmse=#{rmse}"
  puts "ok=#{max_abs < 1.0e-4_f32}"
ensure
  buffers.each(&.close)
  mod.try(&.close)
  ctx.try(&.close)
end
