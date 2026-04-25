#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"
DEFAULT_PROMPT    = "The quick brown fox jumps over the lazy dog. Describe this scene in detail, then explain how weather, geometry, and memory interact in a compact machine learning runtime. " \
                    "Use precise technical language and include several short code-like phrases so the token stream is varied."

private def softplus(x : Float32) : Float32
  x > 20.0_f32 ? x : Math.log(1.0_f32 + Math.exp(x)).to_f32
end

private def silu!(x : Array(Float32)) : Nil
  x.size.times do |i|
    v = x[i]
    x[i] = v / (1.0_f32 + Math.exp(-v).to_f32)
  end
end

private def l2_norm_slice!(x : Array(Float32), offset : Int32, len : Int32, eps : Float32) : Nil
  ss = 0.0_f64
  len.times { |i| ss += x[offset + i].to_f64 * x[offset + i].to_f64 }
  inv = (1.0 / Math.sqrt(ss + eps.to_f64)).to_f32
  len.times { |i| x[offset + i] *= inv }
end

private def recurrent_k_vectors_for_prompt(weights : ML::GGUF::Qwen35Weights,
                                           token_ids : Array(Int32),
                                           layer_index : Int32) : Array(Array(Array(Float64)))
  hp = weights.hparams
  target_layer = weights.layers[layer_index].as?(ML::GGUF::Qwen35RecurrentWeights) ||
                 raise "layer #{layer_index} is not recurrent"
  h_k = hp.ssm_group_count
  h_v = hp.ssm_time_step_rank
  s = hp.ssm_state_size
  qkv_dim = 2 * h_k * s + h_v * s
  conv_k = hp.ssm_conv_kernel
  conv_state = Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: token_ids.size + 2)
  per_head = Array.new(h_k) { [] of Array(Float64) }

  token_ids.each_with_index do |token_id, pos|
    x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, token_id)

    layer_index.times do |il|
      case layer = weights.layers[il]
      in ML::GGUF::Qwen35FullAttnWeights
        x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      in ML::GGUF::Qwen35RecurrentWeights
        x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos.to_i32, layer, state.layers[il], hp, state.max_seq)
      end
    end

    cur = ML::GGUF::Qwen35CPU.rms_norm(x, target_layer.attn_norm, hp.rms_eps)
    qkv_mixed = ML::GGUF::Qwen35CPU.qmatvec_nobias(target_layer.attn_qkv_qw, cur)

    conv_out = Array(Float32).new(qkv_dim) do |ch|
      acc = 0.0_f32
      w_base = ch * conv_k
      (conv_k - 1).times do |t|
        acc += conv_state[t * qkv_dim + ch] * target_layer.ssm_conv1d[w_base + t]
      end
      acc + qkv_mixed[ch] * target_layer.ssm_conv1d[w_base + (conv_k - 1)]
    end

    (conv_k - 2).times do |t|
      src = (t + 1) * qkv_dim
      dst = t * qkv_dim
      qkv_dim.times { |ch| conv_state[dst + ch] = conv_state[src + ch] }
    end
    last = (conv_k - 2) * qkv_dim
    qkv_dim.times { |ch| conv_state[last + ch] = qkv_mixed[ch] }

    silu!(conv_out)
    k_offset = h_k * s
    h_k.times { |h| l2_norm_slice!(conv_out, k_offset + h * s, s, hp.rms_eps) }

    h_k.times do |h|
      off = k_offset + h * s
      per_head[h] << Array.new(s) { |d| conv_out[off + d].to_f64 }
    end
  end

  per_head
end

private def dot(a : Array(Float64), b : Array(Float64)) : Float64
  acc = 0.0
  a.size.times { |i| acc += a[i] * b[i] }
  acc
end

private def residual_norm(v : Array(Float64), basis : Array(Array(Float64)), rank : Int32) : Float64
  residual = v.dup
  limit = Math.min(rank, basis.size)
  limit.times do |i|
    b = basis[i]
    coeff = dot(residual, b)
    residual.size.times { |d| residual[d] -= coeff * b[d] }
  end
  Math.sqrt(dot(residual, residual))
end

private def greedy_basis(vectors : Array(Array(Float64)), max_rank : Int32, eps : Float64 = 1.0e-6) : Array(Array(Float64))
  basis = [] of Array(Float64)
  vectors.each do |v|
    break if basis.size >= max_rank
    residual = v.dup
    basis.each do |b|
      coeff = dot(residual, b)
      residual.size.times { |d| residual[d] -= coeff * b[d] }
    end
    norm = Math.sqrt(dot(residual, residual))
    next if norm <= eps
    basis << residual.map { |x| x / norm }
  end
  basis
end

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = DEFAULT_PROMPT
tokens_limit = 96
calib_tokens = 32
layer_index = 0
ranks = [8, 16, 32, 64, 96, 128]
thresholds = [0.05, 0.10, 0.20, 0.35, 0.50]

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_fixed_basis_probe [--model PATH] [--tokenizer PATH] [--prompt TEXT] [--tokens N] [--calib-tokens N] [--layer N] [--ranks LIST]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--tokenizer=PATH", "llama-tokenize path") { |v| tokenizer_bin = v }
  p.on("--prompt=TEXT", "Prompt text") { |v| prompt = v }
  p.on("--tokens=N", "Max prompt tokens to use") { |v| tokens_limit = v.to_i }
  p.on("--calib-tokens=N", "Tokens used to build the fixed basis") { |v| calib_tokens = v.to_i }
  p.on("--layer=N", "Recurrent layer index to probe (default: 0)") { |v| layer_index = v.to_i }
  p.on("--ranks=LIST", "Comma-separated ranks") { |v| ranks = v.split(',').map(&.to_i) }
  p.on("--thresholds=LIST", "Comma-separated residual thresholds for pass-rate reporting") { |v| thresholds = v.split(',').map(&.to_f) }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokenizer not found: #{tokenizer_bin}" unless File.exists?(tokenizer_bin)
raise "tokens must be positive" unless tokens_limit > 0
raise "calib-tokens must be positive" unless calib_tokens > 0
raise "ranks must not be empty" if ranks.empty?

gguf = ML::GGUF::GGUFFile.new(model)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model, tokenizer_bin)
token_ids = tok.encode(prompt, add_bos_override: false)
while token_ids.size < tokens_limit
  token_ids.concat(tok.encode(prompt, add_bos_override: false))
end
token_ids = token_ids[0, tokens_limit]

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
per_head = recurrent_k_vectors_for_prompt(weights, token_ids, layer_index)
max_rank = ranks.max
calib_count = Math.min(calib_tokens, token_ids.size - 1)
raise "need at least one held-out token" unless calib_count > 0 && calib_count < token_ids.size

puts "Qwen35 DeltaNet fixed-basis K residual probe"
puts "model=#{File.basename(model)}"
puts "layer=#{layer_index} token_vectors=#{token_ids.size} calib_tokens=#{calib_count} heldout_tokens=#{token_ids.size - calib_count}"
puts "heads=#{per_head.size} state_size=#{per_head[0][0].size} ranks=#{ranks.join(',')}"
puts "basis=per-head greedy modified Gram-Schmidt over first calib_tokens; reports held-out L2 residual for normalized K vectors"
puts "thresholds=#{thresholds.map { |t| t.round(4) }.join(',')}"

ranks.each do |rank|
  all_residuals = [] of Float64
  per_head.each do |vectors|
    basis = greedy_basis(vectors[0, calib_count], max_rank)
    vectors[calib_count, vectors.size - calib_count].each do |v|
      all_residuals << residual_norm(v, basis, rank)
    end
  end

  sorted = all_residuals.sort
  mean = all_residuals.sum / all_residuals.size
  p50 = sorted[sorted.size // 2]
  p90 = sorted[(sorted.size * 90 // 100).clamp(0, sorted.size - 1)]
  p99 = sorted[(sorted.size * 99 // 100).clamp(0, sorted.size - 1)]
  max = sorted[-1]
  pass = thresholds.map do |threshold|
    passed = all_residuals.count { |r| r <= threshold }
    "#{threshold.round(4)}:#{(100.0 * passed / all_residuals.size).round(2)}%"
  end
  puts "rank=#{rank} mean_residual=#{mean.round(6)} p50=#{p50.round(6)} p90=#{p90.round(6)} p99=#{p99.round(6)} max=#{max.round(6)} pass_rates=#{pass.join(',')}"
end
