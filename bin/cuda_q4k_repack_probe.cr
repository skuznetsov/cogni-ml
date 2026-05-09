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

record RepackedQ4, qvals : Bytes, scales : Array(Float32), mins : Array(Float32)

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

f32_t0 = Time.instant
f32_weights = ML::GGUF::Dequant.dequantize_q4_k(w_raw, in_dim * out_dim)
f32_repack_ms = (Time.instant - f32_t0).total_milliseconds

ctx = ML::CUDA::Context.create
modules = [] of ML::CUDA::CUDAModule
buffers = [] of ML::CUDA::DeviceBuffer
begin
  raw_mod = ML::CUDA::CUDAModule.load(Q4K_PTX, "q4_raw")
  repack_mod = ML::CUDA::CUDAModule.load(REPACK_PTX, "q4_repack")
  f32_mod = ML::CUDA::CUDAModule.load(F32_PTX, "q4_f32")
  modules.concat([raw_mod, repack_mod, f32_mod])
  raw_fn = raw_mod.function("q4_k_gemv_warp4_f32")
  repack_fn = repack_mod.function("q4_k_repacked_gemv_warp4_f32")
  f32_fn = f32_mod.function("f32_gemv_warp4_f32")

  d_raw = ML::CUDA::DeviceBuffer.new(w_raw.size.to_u64)
  d_qvals = ML::CUDA::DeviceBuffer.new(repacked.qvals.size.to_u64)
  d_scales = ML::CUDA::DeviceBuffer.new(bytesize_f32(repacked.scales.size))
  d_mins = ML::CUDA::DeviceBuffer.new(bytesize_f32(repacked.mins.size))
  d_f32 = ML::CUDA::DeviceBuffer.new(bytesize_f32(f32_weights.size))
  d_x = ML::CUDA::DeviceBuffer.new(bytesize_f32(in_dim))
  d_raw_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_repack_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  d_f32_out = ML::CUDA::DeviceBuffer.new(bytesize_f32(out_dim))
  buffers.concat([d_raw, d_qvals, d_scales, d_mins, d_f32, d_x, d_raw_out, d_repack_out, d_f32_out])

  ML::CUDA.copy_htod!(d_raw.ptr, w_raw.to_unsafe.as(Void*), w_raw.size.to_u64, "raw")
  ML::CUDA.copy_htod!(d_qvals.ptr, repacked.qvals.to_unsafe.as(Void*), repacked.qvals.size.to_u64, "qvals")
  ML::CUDA.copy_htod!(d_scales.ptr, repacked.scales.to_unsafe.as(Void*), bytesize_f32(repacked.scales.size), "scales")
  ML::CUDA.copy_htod!(d_mins.ptr, repacked.mins.to_unsafe.as(Void*), bytesize_f32(repacked.mins.size), "mins")
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

  raw_ms = run_kernel(raw_fn, grid, block, raw_params, reps, warmup)
  repack_ms_gpu = run_kernel(repack_fn, grid, block, repack_params, reps, warmup)
  f32_ms_gpu = run_kernel(f32_fn, grid, block, f32_params, reps, warmup)

  raw_gpu = Array(Float32).new(out_dim, 0.0_f32)
  repack_gpu = Array(Float32).new(out_dim, 0.0_f32)
  f32_gpu = Array(Float32).new(out_dim, 0.0_f32)
  ML::CUDA.copy_dtoh!(raw_gpu.to_unsafe.as(Void*), d_raw_out.ptr, bytesize_f32(out_dim), "raw_out")
  ML::CUDA.copy_dtoh!(repack_gpu.to_unsafe.as(Void*), d_repack_out.ptr, bytesize_f32(out_dim), "repack_out")
  ML::CUDA.copy_dtoh!(f32_gpu.to_unsafe.as(Void*), d_f32_out.ptr, bytesize_f32(out_dim), "f32_out")

  raw_max = max_abs_diff(raw_gpu, cpu)
  raw_cos = cosine(raw_gpu, cpu)
  repack_max = max_abs_diff(repack_gpu, cpu)
  repack_cos = cosine(repack_gpu, cpu)
  f32_max = max_abs_diff(f32_gpu, cpu)
  f32_cos = cosine(f32_gpu, cpu)

  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "tensor=#{tensor_name}"
  puts "shape=#{in_dim}x#{out_dim}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "raw_bytes=#{w_raw.size}"
  puts "repacked_bytes=#{repacked.qvals.size + repacked.scales.size * 4 + repacked.mins.size * 4}"
  puts "repack_ratio=#{((repacked.qvals.size + repacked.scales.size * 4 + repacked.mins.size * 4).to_f64 / w_raw.size).round(3)}"
  puts "f32_bytes=#{f32_weights.size * 4}"
  puts "f32_ratio=#{((f32_weights.size * 4).to_f64 / w_raw.size).round(3)}"
  puts "host_repack_ms=#{repack_ms.round(3)}"
  puts "host_f32_dequant_ms=#{f32_repack_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "raw_cuda_ms=#{raw_ms.round(4)}"
  puts "repacked_cuda_ms=#{repack_ms_gpu.round(4)}"
  puts "f32_cuda_ms=#{f32_ms_gpu.round(4)}"
  puts "repacked_speedup=#{(raw_ms / repack_ms_gpu).round(4)}"
  puts "f32_speedup=#{(raw_ms / f32_ms_gpu).round(4)}"
  puts "raw_cos=#{raw_cos.round(8)}"
  puts "raw_max_diff=#{raw_max}"
  puts "repacked_cos=#{repack_cos.round(8)}"
  puts "repacked_max_diff=#{repack_max}"
  puts "f32_cos=#{f32_cos.round(8)}"
  puts "f32_max_diff=#{f32_max}"
  puts "ok=#{raw_cos >= 0.99999 && raw_max <= 1.0e-3_f32 && repack_cos >= 0.99999 && repack_max <= 1.0e-3_f32 && f32_cos >= 0.99999 && f32_max <= 1.0e-3_f32}"
ensure
  buffers.each(&.close)
  modules.each(&.close)
  ctx.close
  gguf.close
end
