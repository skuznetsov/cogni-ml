# CUDA full-attention post-projection boundary probe.
#
# Runs Q/K/V projection, then CUDA split+RMSNorm+RoPE+KV-cache write, and
# compares Q, gate, K, K-cache, and V-cache against the CPU reference.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_meta"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/cuda/qwen_full_attn_projection_runner"
require "../src/ml/cuda/qwen_full_attn_kv_runner"

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

def rope_tables(tokens : Int32, start_pos : Int32, rope_dim : Int32, freq_base : Float32) : {Array(Float32), Array(Float32)}
  half = rope_dim // 2
  cos_table = Array(Float32).new(tokens * half, 0.0_f32)
  sin_table = Array(Float32).new(tokens * half, 0.0_f32)
  tokens.times do |tok|
    pos = start_pos + tok
    half.times do |i|
      freq = 1.0_f32 / (freq_base ** (2.0_f32 * i / rope_dim))
      theta = pos.to_f32 * freq
      cos_table[tok * half + i] = Math.cos(theta).to_f32
      sin_table[tok * half + i] = Math.sin(theta).to_f32
    end
  end
  {cos_table, sin_table}
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layer = 3
seed = 23_u64
tokens = 4
start_pos = 0
max_seq = 16

OptionParser.parse do |p|
  p.banner = "Usage: cuda_full_attn_kv_probe [--model PATH] [--layer N] [--tokens N] [--start-pos N] [--max-seq N] [--seed N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layer N", "Full-attention layer index") { |v| layer = v.to_i }
  p.on("--tokens N", "Sequence length") { |v| tokens = v.to_i }
  p.on("--start-pos N", "KV cache start position") { |v| start_pos = v.to_i }
  p.on("--max-seq N", "KV cache capacity") { |v| max_seq = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layer must be non-negative" unless layer >= 0
raise "tokens must be positive" unless tokens > 0
raise "start-pos must be non-negative" unless start_pos >= 0
raise "max-seq must cover start-pos + tokens" unless max_seq >= start_pos + tokens

gguf = ML::GGUF::GGUFFile.new(model)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
raise "layer #{layer} is not full-attention" unless hparams.full_attention?(layer)

proj_weights = ML::CUDA::QwenFullAttnProjectionRunner::Weights.load(gguf, layer)
kv_weights = ML::CUDA::QwenFullAttnKVRunner::Weights.load(gguf, layer)
hidden = proj_weights.hidden
n_head = hparams.n_head
n_head_kv = hparams.n_head_kv
head_dim = hparams.head_dim
rope_dim = hparams.rope_dim_count
q_dim = n_head * head_dim
kv_dim = n_head_kv * head_dim
raise "projection q dimension mismatch" unless proj_weights.q_dim == 2 * q_dim
raise "projection k/v dimension mismatch" unless proj_weights.k_dim == kv_dim && proj_weights.v_dim == kv_dim

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }
cos_table, sin_table = rope_tables(tokens, start_pos, rope_dim, hparams.rope_freq_base)

q_cpu_all = Array(Float32).new(tokens * q_dim, 0.0_f32)
gate_cpu_all = Array(Float32).new(tokens * q_dim, 0.0_f32)
k_cpu_all = Array(Float32).new(tokens * kv_dim, 0.0_f32)
attn_cpu_all = Array(Float32).new(tokens * q_dim, 0.0_f32)
k_cache_cpu = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
v_cache_cpu = Array(Float32).new(max_seq * kv_dim, 0.0_f32)
heads_per_group = n_head // n_head_kv
scale = (1.0_f64 / Math.sqrt(head_dim.to_f64)).to_f32

tokens.times do |tok|
  x = xs[tok * hidden, hidden]
  q_full = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, proj_weights.q_raw, ML::GGUF::TensorType::Q4_K, proj_weights.q_dim, Array(Float32).new(proj_weights.q_dim, 0.0_f32))
  k = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, proj_weights.k_raw, ML::GGUF::TensorType::Q4_K, kv_dim, Array(Float32).new(kv_dim, 0.0_f32))
  v = ML::GGUF::QuantMatmul.matmul_add(x, 1, hidden, proj_weights.v_raw, proj_weights.v_type, kv_dim, Array(Float32).new(kv_dim, 0.0_f32))

  q = Array(Float32).new(q_dim, 0.0_f32)
  gate = Array(Float32).new(q_dim, 0.0_f32)
  n_head.times do |h|
    src_base = h * 2 * head_dim
    dst_base = h * head_dim
    head_dim.times do |d|
      q[dst_base + d] = q_full[src_base + d]
      gate[dst_base + d] = q_full[src_base + head_dim + d]
    end
  end
  n_head.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(q, h * head_dim, head_dim, kv_weights.q_norm, hparams.rms_eps) }
  n_head_kv.times { |h| ML::GGUF::Qwen35CPU.rms_norm_slice!(k, h * head_dim, head_dim, kv_weights.k_norm, hparams.rms_eps) }
  n_head.times { |h| ML::GGUF::Qwen35CPU.rope_partial!(q, h * head_dim, rope_dim, head_dim, start_pos + tok, hparams.rope_freq_base) }
  n_head_kv.times { |h| ML::GGUF::Qwen35CPU.rope_partial!(k, h * head_dim, rope_dim, head_dim, start_pos + tok, hparams.rope_freq_base) }

  q_dim.times do |i|
    q_cpu_all[tok * q_dim + i] = q[i]
    gate_cpu_all[tok * q_dim + i] = gate[i]
  end
  kv_dim.times do |i|
    k_cpu_all[tok * kv_dim + i] = k[i]
    cache_idx = (start_pos + tok) * kv_dim + i
    k_cache_cpu[cache_idx] = k[i]
    v_cache_cpu[cache_idx] = v[i]
  end

  cache_len = start_pos + tok + 1
  scores = Array(Float32).new(cache_len, 0.0_f32)
  n_head.times do |h|
    kv_h = h // heads_per_group
    q_off = h * head_dim
    cache_len.times do |p|
      k_off = p * kv_dim + kv_h * head_dim
      dot = 0.0_f32
      head_dim.times { |d| dot += q[q_off + d] * k_cache_cpu[k_off + d] }
      scores[p] = dot * scale
    end
    ML::GGUF::Qwen35CPU.softmax_slice!(scores, 0, cache_len)
    out_off = tok * q_dim + h * head_dim
    head_dim.times do |d|
      acc = 0.0_f32
      cache_len.times do |p|
        v_off = p * kv_dim + kv_h * head_dim
        acc += scores[p] * v_cache_cpu[v_off + d]
      end
      gate_v = gate[h * head_dim + d]
      attn_cpu_all[out_off + d] = acc * (1.0_f32 / (1.0_f32 + Math.exp(-gate_v).to_f32))
    end
  end
end

cuda_ctx = nil.as(ML::CUDA::Context?)
proj = nil.as(ML::CUDA::QwenFullAttnProjectionRunner?)
kv = nil.as(ML::CUDA::QwenFullAttnKVRunner?)

begin
  cuda_ctx = ML::CUDA::Context.create
  proj = ML::CUDA::QwenFullAttnProjectionRunner.from_weights(proj_weights, tokens, xs)
  proj.upload_weights
  proj.reset_sequence
  proj.run_sequence
  ML::CUDA.synchronize!("cuCtxSynchronize(projection)")

  kv = ML::CUDA::QwenFullAttnKVRunner.new(tokens, max_seq, start_pos, n_head, n_head_kv, head_dim, rope_dim,
    hparams.rms_eps, proj.q_device_ptr, proj.k_device_ptr, proj.v_device_ptr, kv_weights, cos_table, sin_table)
  kv.upload_constants
  kv.run_sequence
  ML::CUDA.synchronize!("cuCtxSynchronize(kv)")
  kv.read_outputs

  lines = [] of String
  ok = true
  ok &&= report_pair("q", kv.q_gpu_all, q_cpu_all, lines, 2.0e-4_f32)
  ok &&= report_pair("gate", kv.gate_gpu_all, gate_cpu_all, lines, 1.0e-3_f32)
  ok &&= report_pair("k", kv.k_gpu_all, k_cpu_all, lines, 2.0e-4_f32)
  ok &&= report_pair("attn", kv.attn_gpu_all, attn_cpu_all, lines, 2.0e-3_f32)
  ok &&= report_pair("k_cache", kv.k_cache_gpu, k_cache_cpu, lines, 2.0e-4_f32)
  ok &&= report_pair("v_cache", kv.v_cache_gpu, v_cache_cpu, lines, 1.0e-3_f32)

  puts "device=#{cuda_ctx.device_name}"
  puts "compute_capability=#{cuda_ctx.compute_capability_major}.#{cuda_ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layer=#{layer}"
  puts "tokens=#{tokens}"
  puts "start_pos=#{start_pos}"
  puts "max_seq=#{max_seq}"
  puts "hidden=#{hidden}"
  puts "n_head=#{n_head}"
  puts "n_head_kv=#{n_head_kv}"
  puts "head_dim=#{head_dim}"
  puts "rope_dim=#{rope_dim}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  kv.try(&.close)
  proj.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
