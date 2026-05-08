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
require "../src/ml/cuda/qwen_recurrent_layer_runner"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

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
tokens = 1

OptionParser.parse do |p|
  p.banner = "Usage: cuda_recurrent_prep_output_probe [--model PATH] [--layer N] [--seed N] [--reps N] [--warmup N] [--tokens N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Recurrent layer index") { |v| layer = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--reps N", "Timed recurrent-prep output launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--tokens N", "GPU-resident sequence length for recurrent state progression") { |v| tokens = v.to_i }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
raise "tokens must be positive" unless tokens > 0

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
attn_norm_info = gguf.tensor("#{prefix}.attn_norm.weight") || raise "missing #{prefix}.attn_norm.weight"
qkv_info = gguf.tensor("#{prefix}.attn_qkv.weight") || raise "missing #{prefix}.attn_qkv.weight"
gate_info = gguf.tensor("#{prefix}.attn_gate.weight") || raise "missing #{prefix}.attn_gate.weight"
alpha_info = gguf.tensor("#{prefix}.ssm_alpha.weight") || raise "missing #{prefix}.ssm_alpha.weight"
beta_info = gguf.tensor("#{prefix}.ssm_beta.weight") || raise "missing #{prefix}.ssm_beta.weight"
out_info = gguf.tensor("#{prefix}.ssm_out.weight") || raise "missing #{prefix}.ssm_out.weight"
post_norm_info = gguf.tensor("#{prefix}.post_attention_norm.weight") || raise "missing #{prefix}.post_attention_norm.weight"
ffn_gate_info = gguf.tensor("#{prefix}.ffn_gate.weight") || raise "missing #{prefix}.ffn_gate.weight"
ffn_up_info = gguf.tensor("#{prefix}.ffn_up.weight") || raise "missing #{prefix}.ffn_up.weight"
ffn_down_info = gguf.tensor("#{prefix}.ffn_down.weight") || raise "missing #{prefix}.ffn_down.weight"
conv_info = gguf.tensor("#{prefix}.ssm_conv1d.weight") || raise "missing #{prefix}.ssm_conv1d.weight"
dt_info = gguf.tensor("#{prefix}.ssm_dt.bias") || raise "missing #{prefix}.ssm_dt.bias"
a_info = gguf.tensor("#{prefix}.ssm_a") || raise "missing #{prefix}.ssm_a"
norm_info = gguf.tensor("#{prefix}.ssm_norm.weight") || raise "missing #{prefix}.ssm_norm.weight"
raise "expected Q5_K attn_qkv" unless qkv_info.type.q5_k?
raise "expected Q4_K gate/alpha/beta" unless gate_info.type.q4_k? && alpha_info.type.q4_k? && beta_info.type.q4_k?
raise "expected Q4_K ssm_out" unless out_info.type.q4_k?
raise "expected Q4_K ffn gate/up and Q6_K ffn down" unless ffn_gate_info.type.q4_k? && ffn_up_info.type.q4_k? && ffn_down_info.type.q6_k?
hidden = qkv_info.dims[0].to_i32
ffn_dim = ffn_gate_info.dims[1].to_i32
raise "attn_qkv shape mismatch" unless qkv_info.dims[1].to_i32 == qkv_dim
raise "attn_gate shape mismatch" unless gate_info.dims[0].to_i32 == hidden && gate_info.dims[1].to_i32 == inner_dim
raise "ssm_alpha/beta shape mismatch" unless alpha_info.dims[0].to_i32 == hidden && alpha_info.dims[1].to_i32 == h_v &&
                                      beta_info.dims[0].to_i32 == hidden && beta_info.dims[1].to_i32 == h_v
raise "ssm_out input mismatch" unless out_info.dims[0].to_i32 == inner_dim
raise "ffn shape mismatch" unless ffn_gate_info.dims[0].to_i32 == hidden && ffn_up_info.dims[0].to_i32 == hidden &&
                                ffn_up_info.dims[1].to_i32 == ffn_dim && ffn_down_info.dims[0].to_i32 == ffn_dim &&
                                ffn_down_info.dims[1].to_i32 == hidden
out_dim = out_info.dims[1].to_i32
attn_norm = gguf.read_tensor_f32(attn_norm_info)
qkv_raw = gguf.read_tensor_raw(qkv_info)
gate_raw = gguf.read_tensor_raw(gate_info)
alpha_raw = gguf.read_tensor_raw(alpha_info)
beta_raw_w = gguf.read_tensor_raw(beta_info)
out_raw = gguf.read_tensor_raw(out_info)
post_norm = gguf.read_tensor_f32(post_norm_info)
ffn_gate_raw = gguf.read_tensor_raw(ffn_gate_info)
ffn_up_raw = gguf.read_tensor_raw(ffn_up_info)
ffn_down_raw = gguf.read_tensor_raw(ffn_down_info)
conv1d = gguf.read_tensor_f32(conv_info)
dt_bias = gguf.read_tensor_f32(dt_info)
ssm_a = gguf.read_tensor_f32(a_info)
ssm_norm = gguf.read_tensor_f32(norm_info)
raise "conv1d size mismatch" unless conv1d.size == qkv_dim * conv_k
raise "dt/ssm_a size mismatch" unless dt_bias.size == h_v && ssm_a.size == h_v
raise "ssm_norm size mismatch" unless ssm_norm.size == s
raise "norm size mismatch" unless attn_norm.size == hidden && post_norm.size == hidden

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { ((rng.next_float - 0.5) * 0.2).to_f32 }
conv_state_init = Array(Float32).new((conv_k - 1) * qkv_dim) { ((rng.next_float - 0.5) * 0.05).to_f32 }
ssm_state_init = Array(Float32).new(h_v * s * s) { ((rng.next_float - 0.5) * 0.05).to_f32 }

cpu_t0 = Time.instant
conv_state_cpu = conv_state_init.dup
ssm_state_cpu = ssm_state_init.dup
attn_out_cpu = Array(Float32).new(hidden, 0.0_f32)
final_cpu_all = Array(Float32).new(tokens * hidden, 0.0_f32)

tokens.times do |tok|
  x_offset = tok * hidden
  x = xs[x_offset, hidden]
  cur = ML::GGUF::Qwen35CPU.rms_norm(x, attn_norm, eps)
  qkv_mixed = ML::GGUF::QuantMatmul.matmul_add(cur, 1, hidden, qkv_raw, ML::GGUF::TensorType::Q5_K, qkv_dim, Array(Float32).new(qkv_dim, 0.0_f32))
  z = ML::GGUF::QuantMatmul.matmul_add(cur, 1, hidden, gate_raw, ML::GGUF::TensorType::Q4_K, inner_dim, Array(Float32).new(inner_dim, 0.0_f32))
  alpha = ML::GGUF::QuantMatmul.matmul_add(cur, 1, hidden, alpha_raw, ML::GGUF::TensorType::Q4_K, h_v, Array(Float32).new(h_v, 0.0_f32))
  beta_raw = ML::GGUF::QuantMatmul.matmul_add(cur, 1, hidden, beta_raw_w, ML::GGUF::TensorType::Q4_K, h_v, Array(Float32).new(h_v, 0.0_f32))
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
  attn_out_cpu = ML::GGUF::QuantMatmul.matmul_add(y_cpu, 1, inner_dim, out_raw, ML::GGUF::TensorType::Q4_K, out_dim, Array(Float32).new(out_dim, 0.0_f32))
  residual_cpu = Array(Float32).new(hidden) { |i| x[i] + attn_out_cpu[i] }
  cur2 = ML::GGUF::Qwen35CPU.rms_norm(residual_cpu, post_norm, eps)
  ffn_gate_cpu = ML::GGUF::QuantMatmul.matmul_add(cur2, 1, hidden, ffn_gate_raw, ML::GGUF::TensorType::Q4_K, ffn_dim, Array(Float32).new(ffn_dim, 0.0_f32))
  ffn_up_cpu = ML::GGUF::QuantMatmul.matmul_add(cur2, 1, hidden, ffn_up_raw, ML::GGUF::TensorType::Q4_K, ffn_dim, Array(Float32).new(ffn_dim, 0.0_f32))
  ffn_comb_cpu = Array(Float32).new(ffn_dim) do |i|
    gv = ffn_gate_cpu[i]
    (gv / (1.0_f32 + Math.exp(-gv).to_f32)) * ffn_up_cpu[i]
  end
  ffn_out_cpu = ML::GGUF::QuantMatmul.matmul_add(ffn_comb_cpu, 1, ffn_dim, ffn_down_raw, ML::GGUF::TensorType::Q6_K, hidden, Array(Float32).new(hidden, 0.0_f32))
  hidden.times { |i| final_cpu_all[x_offset + i] = residual_cpu[i] + ffn_out_cpu[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds / tokens

cuda_ctx = nil.as(ML::CUDA::Context?)
runner = nil.as(ML::CUDA::QwenRecurrentLayerRunner?)

begin
  cuda_ctx = ML::CUDA::Context.create
  device_name = cuda_ctx.device_name
  cc_major = cuda_ctx.compute_capability_major
  cc_minor = cuda_ctx.compute_capability_minor

  runner = ML::CUDA::QwenRecurrentLayerRunner.new(
    tokens, hidden, ffn_dim, qkv_dim, q_dim, inner_dim, h_k, h_v, scale, eps,
    xs, conv_state_init, ssm_state_init,
    attn_norm, qkv_raw, gate_raw, alpha_raw, beta_raw_w, out_raw,
    post_norm, ffn_gate_raw, ffn_up_raw, ffn_down_raw,
    conv1d, dt_bias, ssm_a, ssm_norm
  )

  upload_t0 = Time.instant
  runner.upload_weights
  ML::CUDA.synchronize!("cuCtxSynchronize(upload_weights)")
  weight_upload_ms = (Time.instant - upload_t0).total_milliseconds
  runner.reset_sequence

  warmup.times { runner.run_sequence }
  ML::CUDA.synchronize!("cuCtxSynchronize(warmup)") if warmup > 0
  runner.reset_sequence

  gpu_t0 = Time.instant
  timed_steps = runner.run_repeated(reps)
  ML::CUDA.synchronize!("cuCtxSynchronize")
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds / timed_steps

  runner.reset_sequence
  runner.run_sequence
  ML::CUDA.synchronize!("cuCtxSynchronize(correctness)")
  runner.read_outputs

  lines = [] of String
  ok = true
  ok &&= report_pair("conv_state", runner.conv_state_gpu, conv_state_cpu, lines, 1.0e-5_f32)
  ok &&= report_pair("ssm_state", runner.ssm_state_gpu, ssm_state_cpu, lines, 5.0e-4_f32)
  ok &&= report_pair("attn_out", runner.attn_out_gpu, attn_out_cpu, lines, 5.0e-3_f32)
  ok &&= report_pair("final_all", runner.final_gpu_all, final_cpu_all, lines, 5.0e-3_f32)

  puts "device=#{device_name}"
  puts "compute_capability=#{cc_major}.#{cc_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "hidden=#{hidden}"
  puts "ffn_dim=#{ffn_dim}"
  puts "qkv_dim=#{qkv_dim}"
  puts "inner_dim=#{inner_dim}"
  puts "out_dim=#{out_dim}"
  puts "tokens=#{tokens}"
  puts "reps=#{reps}"
  puts "warmup=#{warmup}"
  puts "timed_steps=#{timed_steps}"
  puts "weight_upload_ms=#{weight_upload_ms.round(3)}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cuda_ms_per_token=#{gpu_ms.round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cpu_ms_per_token=#{cpu_ms.round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  runner.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
