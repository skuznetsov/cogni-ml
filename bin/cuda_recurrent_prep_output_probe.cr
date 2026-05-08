# CUDA recurrent prep + DeltaNet output slice probe for Qwen GGUF weights.
#
# Runs the real recurrent input projection bundle through CUDA recurrent conv
# prep, alpha/beta transforms, DeltaNet state update, post RMSNorm/SiLU
# gating, and the real Q4_K ssm_out projection. This is the first one-token recurrent-attention slice facade; residuals and
# FFN are still outside the probe.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/gguf/qwen35_cpu"

@[Link(ldflags: "-lcuda")]
lib LibCUDARecPrepOut
  alias CUdevice = Int32
  alias CUcontext = Void*
  alias CUmodule = Void*
  alias CUfunction = Void*
  alias CUdeviceptr = UInt64

  fun cuInit(flags : UInt32) : Int32
  fun cuDeviceGet(device : CUdevice*, ordinal : Int32) : Int32
  fun cuDeviceGetName(name : UInt8*, len : Int32, dev : CUdevice) : Int32
  fun cuDeviceComputeCapability(major : Int32*, minor : Int32*, dev : CUdevice) : Int32
  fun cuCtxCreate_v2(ctx : CUcontext*, flags : UInt32, dev : CUdevice) : Int32
  fun cuCtxDestroy_v2(ctx : CUcontext) : Int32
  fun cuModuleLoadData(mod : CUmodule*, image : Void*) : Int32
  fun cuModuleUnload(mod : CUmodule) : Int32
  fun cuModuleGetFunction(fn : CUfunction*, mod : CUmodule, name : UInt8*) : Int32
  fun cuMemAlloc_v2(dptr : CUdeviceptr*, bytesize : LibC::SizeT) : Int32
  fun cuMemFree_v2(dptr : CUdeviceptr) : Int32
  fun cuMemcpyHtoD_v2(dst : CUdeviceptr, src : Void*, bytesize : LibC::SizeT) : Int32
  fun cuMemcpyDtoH_v2(dst : Void*, src : CUdeviceptr, bytesize : LibC::SizeT) : Int32
  fun cuLaunchKernel(fn : CUfunction, grid_x : UInt32, grid_y : UInt32, grid_z : UInt32,
                     block_x : UInt32, block_y : UInt32, block_z : UInt32,
                     shared_mem_bytes : UInt32, stream : Void*,
                     kernel_params : Void**, extra : Void**) : Int32
  fun cuCtxSynchronize : Int32
end

DN_PTX        = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
Q4K_PTX       = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q5K_PTX       = {{ read_file("src/ml/cuda/kernels/q5k_gemv_probe.ptx") }}
DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def cuda!(code : Int32, what : String) : Nil
  raise "#{what} failed with CUDA error #{code}" unless code == 0
end

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

def report_pair(name : String, gpu : Array(Float32), cpu : Array(Float32), lines : Array(String), max_allowed : Float32) : Bool
  cos = cosine(gpu, cpu)
  max_diff = max_abs_diff(gpu, cpu)
  ok = cos >= 0.99999 && max_diff <= max_allowed
  lines << "#{name}_cos=#{cos.round(8)}"
  lines << "#{name}_max_diff=#{max_diff}"
  lines << "#{name}_ok=#{ok}"
  ok
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layer = 0
seed = 31_u64
reps = 1
warmup = 0

OptionParser.parse do |p|
  p.banner = "Usage: cuda_recurrent_prep_output_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Recurrent layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed recurrent-prep output launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0

h_k = 16
h_v = 32
s = 128
conv_k = 4
q_dim = h_k * s
v_dim = h_v * s
qkv_dim = 2 * q_dim + v_dim
inner_dim = v_dim
scale = (1.0 / Math.sqrt(s.to_f64)).to_f32
eps = 1.0e-6_f32

gguf = ML::GGUF::GGUFFile.new(model)
prefix = "blk.#{layer}"
qkv_info = gguf.tensor("#{prefix}.attn_qkv.weight") || raise "missing #{prefix}.attn_qkv.weight"
gate_info = gguf.tensor("#{prefix}.attn_gate.weight") || raise "missing #{prefix}.attn_gate.weight"
alpha_info = gguf.tensor("#{prefix}.ssm_alpha.weight") || raise "missing #{prefix}.ssm_alpha.weight"
beta_info = gguf.tensor("#{prefix}.ssm_beta.weight") || raise "missing #{prefix}.ssm_beta.weight"
out_info = gguf.tensor("#{prefix}.ssm_out.weight") || raise "missing #{prefix}.ssm_out.weight"
conv_info = gguf.tensor("#{prefix}.ssm_conv1d.weight") || raise "missing #{prefix}.ssm_conv1d.weight"
dt_info = gguf.tensor("#{prefix}.ssm_dt.bias") || raise "missing #{prefix}.ssm_dt.bias"
a_info = gguf.tensor("#{prefix}.ssm_a") || raise "missing #{prefix}.ssm_a"
norm_info = gguf.tensor("#{prefix}.ssm_norm.weight") || raise "missing #{prefix}.ssm_norm.weight"
raise "expected Q5_K attn_qkv" unless qkv_info.type.q5_k?
raise "expected Q4_K gate/alpha/beta" unless gate_info.type.q4_k? && alpha_info.type.q4_k? && beta_info.type.q4_k?
raise "expected Q4_K ssm_out" unless out_info.type.q4_k?
hidden = qkv_info.dims[0].to_i32
raise "attn_qkv shape mismatch" unless qkv_info.dims[1].to_i32 == qkv_dim
raise "attn_gate shape mismatch" unless gate_info.dims[0].to_i32 == hidden && gate_info.dims[1].to_i32 == inner_dim
raise "ssm_alpha/beta shape mismatch" unless alpha_info.dims[0].to_i32 == hidden && alpha_info.dims[1].to_i32 == h_v &&
                                      beta_info.dims[0].to_i32 == hidden && beta_info.dims[1].to_i32 == h_v
raise "ssm_out input mismatch" unless out_info.dims[0].to_i32 == inner_dim
out_dim = out_info.dims[1].to_i32
qkv_raw = gguf.read_tensor_raw(qkv_info)
gate_raw = gguf.read_tensor_raw(gate_info)
alpha_raw = gguf.read_tensor_raw(alpha_info)
beta_raw_w = gguf.read_tensor_raw(beta_info)
out_raw = gguf.read_tensor_raw(out_info)
conv1d = gguf.read_tensor_f32(conv_info)
dt_bias = gguf.read_tensor_f32(dt_info)
ssm_a = gguf.read_tensor_f32(a_info)
ssm_norm = gguf.read_tensor_f32(norm_info)
raise "conv1d size mismatch" unless conv1d.size == qkv_dim * conv_k
raise "dt/ssm_a size mismatch" unless dt_bias.size == h_v && ssm_a.size == h_v
raise "ssm_norm size mismatch" unless ssm_norm.size == s

rng = Random.new(seed)
x = Array(Float32).new(hidden) { ((rng.next_float - 0.5) * 0.2).to_f32 }
conv_state_init = Array(Float32).new((conv_k - 1) * qkv_dim) { ((rng.next_float - 0.5) * 0.05).to_f32 }
ssm_state_init = Array(Float32).new(h_v * s * s) { ((rng.next_float - 0.5) * 0.05).to_f32 }
qkv_mixed = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, qkv_raw, ML::GGUF::TensorType::Q5_K, qkv_dim, Array(Float32).new(qkv_dim, 0.0_f32))
z = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, gate_raw, ML::GGUF::TensorType::Q4_K, inner_dim, Array(Float32).new(inner_dim, 0.0_f32))
alpha = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, alpha_raw, ML::GGUF::TensorType::Q4_K, h_v, Array(Float32).new(h_v, 0.0_f32))
beta_raw = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, beta_raw_w, ML::GGUF::TensorType::Q4_K, h_v, Array(Float32).new(h_v, 0.0_f32))

cpu_t0 = Time.instant
conv_state_cpu = conv_state_init.dup
conv_out = Array(Float32).new(qkv_dim) do |ch|
  acc = 0.0_f32
  w_base = ch * conv_k
  (conv_k - 1).times { |t| acc += conv_state_cpu[t * qkv_dim + ch] * conv1d[w_base + t] }
  acc += qkv_mixed[ch] * conv1d[w_base + conv_k - 1]
  sig = 1.0_f32 / (1.0_f32 + Math.exp(-acc).to_f32)
  acc * sig
end
(conv_k - 2).times do |t|
  src = (t + 1) * qkv_dim
  dst = t * qkv_dim
  qkv_dim.times { |ch| conv_state_cpu[dst + ch] = conv_state_cpu[src + ch] }
end
last = (conv_k - 2) * qkv_dim
qkv_dim.times { |ch| conv_state_cpu[last + ch] = qkv_mixed[ch] }
q_cpu = conv_out[0, q_dim]
k_cpu = conv_out[q_dim, q_dim]
v_cpu = conv_out[2 * q_dim, v_dim]
h_k.times do |h|
  ML::GGUF::Qwen35CPU.l2_norm_slice!(q_cpu, h * s, s, eps)
  ML::GGUF::Qwen35CPU.l2_norm_slice!(k_cpu, h * s, s, eps)
end
g_cpu = Array(Float32).new(h_v, 0.0_f32)
b_cpu = Array(Float32).new(h_v, 0.0_f32)
h_v.times do |h|
  b_cpu[h] = 1.0_f32 / (1.0_f32 + Math.exp(-beta_raw[h]).to_f32)
  xi = alpha[h] + dt_bias[h]
  sp = xi > 20.0_f32 ? xi : Math.log(1.0_f32 + Math.exp(xi).to_f32).to_f32
  g_cpu[h] = Math.exp((sp * ssm_a[h]).to_f64).to_f32
end
ssm_state_cpu = ssm_state_init.dup
y_cpu = Array(Float32).new(inner_dim, 0.0_f32)
ML::GGUF::Qwen35CPU.delta_net_step!(ssm_state_cpu, q_cpu, k_cpu, v_cpu, g_cpu, b_cpu, y_cpu, h_k, h_v, s, scale)
h_v.times do |h|
  base = h * s
  sumsq = 0.0_f32
  s.times { |d| yv = y_cpu[base + d]; sumsq += yv * yv }
  inv_rms = 1.0_f32 / Math.sqrt(sumsq / s + eps).to_f32
  s.times do |d|
    idx = base + d
    zv = z[idx]
    sig = 1.0_f32 / (1.0_f32 + Math.exp(-zv).to_f32)
    y_cpu[idx] = y_cpu[idx] * inv_rms * ssm_norm[d] * (zv * sig)
  end
end
proj_cpu = ML::GGUF::QuantMatmul.matmul_add(y_cpu, 1, inner_dim, out_raw, ML::GGUF::TensorType::Q4_K, out_dim, Array(Float32).new(out_dim, 0.0_f32))
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda! LibCUDARecPrepOut.cuInit(0_u32), "cuInit"
dev = uninitialized LibCUDARecPrepOut::CUdevice
cuda! LibCUDARecPrepOut.cuDeviceGet(pointerof(dev), 0), "cuDeviceGet"
name_buf = Bytes.new(256)
cuda! LibCUDARecPrepOut.cuDeviceGetName(name_buf.to_unsafe, name_buf.size, dev), "cuDeviceGetName"
device_name = String.new(name_buf.to_unsafe).strip
cc_major = uninitialized Int32
cc_minor = uninitialized Int32
cuda! LibCUDARecPrepOut.cuDeviceComputeCapability(pointerof(cc_major), pointerof(cc_minor), dev), "cuDeviceComputeCapability"
ctx = Pointer(Void).null
cuda! LibCUDARecPrepOut.cuCtxCreate_v2(pointerof(ctx), 0_u32, dev), "cuCtxCreate"

dn_mod = Pointer(Void).null
q4_mod = Pointer(Void).null
q5_mod = Pointer(Void).null
conv_fn = Pointer(Void).null
norm_fn = Pointer(Void).null
ab_fn = Pointer(Void).null
dn_fn = Pointer(Void).null
post_fn = Pointer(Void).null
q4_fn = Pointer(Void).null
q5_fn = Pointer(Void).null
ptrs = [] of UInt64

begin
  cuda! LibCUDARecPrepOut.cuModuleLoadData(pointerof(dn_mod), DN_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(delta)"
  cuda! LibCUDARecPrepOut.cuModuleLoadData(pointerof(q4_mod), Q4K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q4)"
  cuda! LibCUDARecPrepOut.cuModuleLoadData(pointerof(q5_mod), Q5K_PTX.to_unsafe.as(Void*)), "cuModuleLoadData(q5)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(conv_fn), dn_mod, "recurrent_conv1d_silu_step_probe"), "cuModuleGetFunction(conv)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(norm_fn), dn_mod, "l2_norm_128_probe"), "cuModuleGetFunction(norm)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(ab_fn), dn_mod, "alpha_beta_transform_probe"), "cuModuleGetFunction(alpha_beta)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(dn_fn), dn_mod, "deltanet_step_128_probe"), "cuModuleGetFunction(delta)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(post_fn), dn_mod, "deltanet_post_norm_gate_128_probe"), "cuModuleGetFunction(post)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(q4_fn), q4_mod, "q4_k_gemv_warp4_f32"), "cuModuleGetFunction(q4)"
  cuda! LibCUDARecPrepOut.cuModuleGetFunction(pointerof(q5_fn), q5_mod, "q5_k_gemv_warp4_f32"), "cuModuleGetFunction(q5)"

  sizes = [bytesize_f32(hidden), qkv_raw.size.to_u64, gate_raw.size.to_u64, alpha_raw.size.to_u64, beta_raw_w.size.to_u64,
           bytesize_f32(conv_state_init.size), bytesize_f32(ssm_state_init.size), bytesize_f32(qkv_mixed.size), bytesize_f32(conv1d.size),
           bytesize_f32(qkv_dim), bytesize_f32(alpha.size), bytesize_f32(beta_raw.size), bytesize_f32(dt_bias.size), bytesize_f32(ssm_a.size),
           bytesize_f32(g_cpu.size), bytesize_f32(b_cpu.size), bytesize_f32(z.size), bytesize_f32(ssm_norm.size), out_raw.size.to_u64,
           bytesize_f32(out_dim)]
  sizes.each_with_index do |size_bytes, i|
    pdev = 0_u64
    cuda! LibCUDARecPrepOut.cuMemAlloc_v2(pointerof(pdev), size_bytes), "cuMemAlloc(#{i})"
    ptrs << pdev
  end
  d_x, d_qkv_w, d_gate_w, d_alpha_w, d_beta_w, d_conv_state, d_ssm_state, d_qkv, d_conv_w, d_conv_out, d_alpha, d_beta_raw, d_dt, d_a, d_g, d_b, d_z, d_norm, d_out_w, d_proj = ptrs

  copy_inputs = -> {
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_x, x.to_unsafe.as(Void*), bytesize_f32(hidden)), "cuMemcpyHtoD(x)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_qkv_w, qkv_raw.to_unsafe.as(Void*), qkv_raw.size.to_u64), "cuMemcpyHtoD(qkv_w)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_gate_w, gate_raw.to_unsafe.as(Void*), gate_raw.size.to_u64), "cuMemcpyHtoD(gate_w)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_alpha_w, alpha_raw.to_unsafe.as(Void*), alpha_raw.size.to_u64), "cuMemcpyHtoD(alpha_w)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_beta_w, beta_raw_w.to_unsafe.as(Void*), beta_raw_w.size.to_u64), "cuMemcpyHtoD(beta_w)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_conv_state, conv_state_init.to_unsafe.as(Void*), bytesize_f32(conv_state_init.size)), "cuMemcpyHtoD(conv_state)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_ssm_state, ssm_state_init.to_unsafe.as(Void*), bytesize_f32(ssm_state_init.size)), "cuMemcpyHtoD(ssm_state)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_conv_w, conv1d.to_unsafe.as(Void*), bytesize_f32(conv1d.size)), "cuMemcpyHtoD(conv_w)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_dt, dt_bias.to_unsafe.as(Void*), bytesize_f32(dt_bias.size)), "cuMemcpyHtoD(dt)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_a, ssm_a.to_unsafe.as(Void*), bytesize_f32(ssm_a.size)), "cuMemcpyHtoD(a)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_norm, ssm_norm.to_unsafe.as(Void*), bytesize_f32(ssm_norm.size)), "cuMemcpyHtoD(norm)"
    cuda! LibCUDARecPrepOut.cuMemcpyHtoD_v2(d_out_w, out_raw.to_unsafe.as(Void*), out_raw.size.to_u64), "cuMemcpyHtoD(out_w)"
  }
  copy_inputs.call

  qkv_dim_u32 = qkv_dim.to_u32
  h_k_u32 = h_k.to_u32
  h_v_u32 = h_v.to_u32
  hidden_u32 = hidden.to_u32
  inner_u32 = inner_dim.to_u32
  out_dim_u32 = out_dim.to_u32
  qkv_dim_u32_for_proj = qkv_dim.to_u32
  q4_grid = ((out_dim + 3) // 4).to_u32
  qkv_grid = ((qkv_dim + 3) // 4).to_u32
  inner_grid = ((inner_dim + 3) // 4).to_u32
  h_v_grid = ((h_v + 3) // 4).to_u32
  conv_grid = ((qkv_dim + 127) // 128).to_u32
  d_q = d_conv_out
  d_k = d_conv_out + bytesize_f32(q_dim)
  d_v = d_conv_out + bytesize_f32(2 * q_dim)

  qkv_proj_params = Pointer(Void*).malloc(5)
  qkv_proj_params[0] = pointerof(d_qkv_w).as(Void*)
  qkv_proj_params[1] = pointerof(d_x).as(Void*)
  qkv_proj_params[2] = pointerof(d_qkv).as(Void*)
  qkv_proj_params[3] = pointerof(hidden_u32).as(Void*)
  qkv_proj_params[4] = pointerof(qkv_dim_u32_for_proj).as(Void*)

  gate_proj_params = Pointer(Void*).malloc(5)
  gate_proj_params[0] = pointerof(d_gate_w).as(Void*)
  gate_proj_params[1] = pointerof(d_x).as(Void*)
  gate_proj_params[2] = pointerof(d_z).as(Void*)
  gate_proj_params[3] = pointerof(hidden_u32).as(Void*)
  gate_proj_params[4] = pointerof(inner_u32).as(Void*)

  alpha_proj_params = Pointer(Void*).malloc(5)
  alpha_proj_params[0] = pointerof(d_alpha_w).as(Void*)
  alpha_proj_params[1] = pointerof(d_x).as(Void*)
  alpha_proj_params[2] = pointerof(d_alpha).as(Void*)
  alpha_proj_params[3] = pointerof(hidden_u32).as(Void*)
  alpha_proj_params[4] = pointerof(h_v_u32).as(Void*)

  beta_proj_params = Pointer(Void*).malloc(5)
  beta_proj_params[0] = pointerof(d_beta_w).as(Void*)
  beta_proj_params[1] = pointerof(d_x).as(Void*)
  beta_proj_params[2] = pointerof(d_beta_raw).as(Void*)
  beta_proj_params[3] = pointerof(hidden_u32).as(Void*)
  beta_proj_params[4] = pointerof(h_v_u32).as(Void*)

  conv_params = Pointer(Void*).malloc(5)
  conv_params[0] = pointerof(d_conv_state).as(Void*)
  conv_params[1] = pointerof(d_qkv).as(Void*)
  conv_params[2] = pointerof(d_conv_w).as(Void*)
  conv_params[3] = pointerof(d_conv_out).as(Void*)
  conv_params[4] = pointerof(qkv_dim_u32).as(Void*)

  q_norm_params = Pointer(Void*).malloc(3)
  q_norm_params[0] = pointerof(d_q).as(Void*)
  q_norm_params[1] = pointerof(h_k_u32).as(Void*)
  q_norm_params[2] = pointerof(eps).as(Void*)
  k_norm_params = Pointer(Void*).malloc(3)
  k_norm_params[0] = pointerof(d_k).as(Void*)
  k_norm_params[1] = pointerof(h_k_u32).as(Void*)
  k_norm_params[2] = pointerof(eps).as(Void*)

  ab_params = Pointer(Void*).malloc(7)
  ab_params[0] = pointerof(d_alpha).as(Void*)
  ab_params[1] = pointerof(d_beta_raw).as(Void*)
  ab_params[2] = pointerof(d_dt).as(Void*)
  ab_params[3] = pointerof(d_a).as(Void*)
  ab_params[4] = pointerof(d_g).as(Void*)
  ab_params[5] = pointerof(d_b).as(Void*)
  ab_params[6] = pointerof(h_v_u32).as(Void*)

  dn_params = Pointer(Void*).malloc(10)
  dn_params[0] = pointerof(d_ssm_state).as(Void*)
  dn_params[1] = pointerof(d_q).as(Void*)
  dn_params[2] = pointerof(d_k).as(Void*)
  dn_params[3] = pointerof(d_v).as(Void*)
  dn_params[4] = pointerof(d_g).as(Void*)
  dn_params[5] = pointerof(d_b).as(Void*)
  dn_params[6] = pointerof(d_v).as(Void*)
  dn_params[7] = pointerof(h_k_u32).as(Void*)
  dn_params[8] = pointerof(h_v_u32).as(Void*)
  dn_params[9] = pointerof(scale).as(Void*)

  post_params = Pointer(Void*).malloc(5)
  post_params[0] = pointerof(d_v).as(Void*)
  post_params[1] = pointerof(d_z).as(Void*)
  post_params[2] = pointerof(d_norm).as(Void*)
  post_params[3] = pointerof(h_v_u32).as(Void*)
  post_params[4] = pointerof(eps).as(Void*)

  q4_params = Pointer(Void*).malloc(5)
  q4_params[0] = pointerof(d_out_w).as(Void*)
  q4_params[1] = pointerof(d_v).as(Void*)
  q4_params[2] = pointerof(d_proj).as(Void*)
  q4_params[3] = pointerof(inner_u32).as(Void*)
  q4_params[4] = pointerof(out_dim_u32).as(Void*)

  run_bundle = -> {
    cuda! LibCUDARecPrepOut.cuLaunchKernel(q5_fn, qkv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, qkv_proj_params, Pointer(Void*).null), "qkv proj"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(q4_fn, inner_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, gate_proj_params, Pointer(Void*).null), "gate proj"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, alpha_proj_params, Pointer(Void*).null), "alpha proj"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(q4_fn, h_v_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, beta_proj_params, Pointer(Void*).null), "beta proj"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(conv_fn, conv_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, conv_params, Pointer(Void*).null), "conv prep"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(norm_fn, h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, q_norm_params, Pointer(Void*).null), "q norm"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(norm_fn, h_k.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, k_norm_params, Pointer(Void*).null), "k norm"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(ab_fn, 1_u32, 1_u32, 1_u32, 32_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, ab_params, Pointer(Void*).null), "alpha beta"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(dn_fn, h_v.to_u32, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, dn_params, Pointer(Void*).null), "delta step"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(post_fn, h_v.to_u32, 1_u32, 1_u32, 1_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, post_params, Pointer(Void*).null), "post gate"
    cuda! LibCUDARecPrepOut.cuLaunchKernel(q4_fn, q4_grid, 1_u32, 1_u32, 128_u32, 1_u32, 1_u32, 0_u32, Pointer(Void).null, q4_params, Pointer(Void*).null), "ssm_out"
  }

  warmup.times { run_bundle.call }
  cuda! LibCUDARecPrepOut.cuCtxSynchronize, "cuCtxSynchronize(warmup)" if warmup > 0
  copy_inputs.call
  gpu_t0 = Time.instant
  reps.times { run_bundle.call }
  cuda! LibCUDARecPrepOut.cuCtxSynchronize, "cuCtxSynchronize"
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / reps

  copy_inputs.call
  run_bundle.call
  cuda! LibCUDARecPrepOut.cuCtxSynchronize, "cuCtxSynchronize(correctness)"

  conv_state_gpu = Array(Float32).new(conv_state_init.size, 0.0_f32)
  ssm_state_gpu = Array(Float32).new(ssm_state_init.size, 0.0_f32)
  y_gpu = Array(Float32).new(inner_dim, 0.0_f32)
  proj_gpu = Array(Float32).new(out_dim, 0.0_f32)
  cuda! LibCUDARecPrepOut.cuMemcpyDtoH_v2(conv_state_gpu.to_unsafe.as(Void*), d_conv_state, bytesize_f32(conv_state_gpu.size)), "cuMemcpyDtoH(conv_state)"
  cuda! LibCUDARecPrepOut.cuMemcpyDtoH_v2(ssm_state_gpu.to_unsafe.as(Void*), d_ssm_state, bytesize_f32(ssm_state_gpu.size)), "cuMemcpyDtoH(ssm_state)"
  cuda! LibCUDARecPrepOut.cuMemcpyDtoH_v2(y_gpu.to_unsafe.as(Void*), d_v, bytesize_f32(y_gpu.size)), "cuMemcpyDtoH(y)"
  cuda! LibCUDARecPrepOut.cuMemcpyDtoH_v2(proj_gpu.to_unsafe.as(Void*), d_proj, bytesize_f32(proj_gpu.size)), "cuMemcpyDtoH(proj)"

  lines = [] of String
  ok = true
  ok &&= report_pair("conv_state", conv_state_gpu, conv_state_cpu, lines, 1.0e-5_f32)
  ok &&= report_pair("ssm_state", ssm_state_gpu, ssm_state_cpu, lines, 5.0e-4_f32)
  ok &&= report_pair("post_y", y_gpu, y_cpu, lines, 5.0e-3_f32)
  ok &&= report_pair("proj", proj_gpu, proj_cpu, lines, 5.0e-3_f32)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "qkv_dim=#{qkv_dim}"
  puts "inner_dim=#{inner_dim}"
  puts "out_dim=#{out_dim}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  ptrs.each { |ptr| LibCUDARecPrepOut.cuMemFree_v2(ptr) unless ptr == 0_u64 }
  LibCUDARecPrepOut.cuModuleUnload(dn_mod) unless dn_mod.null?
  LibCUDARecPrepOut.cuModuleUnload(q4_mod) unless q4_mod.null?
  LibCUDARecPrepOut.cuModuleUnload(q5_mod) unless q5_mod.null?
  LibCUDARecPrepOut.cuCtxDestroy_v2(ctx) unless ctx.null?
  gguf.close
end
