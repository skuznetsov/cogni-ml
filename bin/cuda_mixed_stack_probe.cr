# CUDA mixed recurrent/full-attention stack scaffold probe for Qwen GGUF weights.
#
# This is a correctness scaffold, not an end-to-end decoder: it composes
# recurrent-layer and full-attention-layer CUDA runners in model layer order
# with device-resident hidden handoff between layers.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/quant_matmul"
require "../src/ml/cuda/qwen_recurrent_layer_runner"
require "../src/ml/cuda/qwen_full_attn_layer_runner"
require "../src/ml/cuda/qwen_output_head_runner"
require "../src/ml/cuda/qwen_mixed_stack_runner"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

def parse_layers(value : String) : Array(Int32)
  value.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
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
layers = [0, 1, 2, 3, 4]
seed = 41_u64
tokens = 2
start_pos = 0
max_seq = 16
warmup = 0
read_logits = false
profile_phases = false

OptionParser.parse do |p|
  p.banner = "Usage: cuda_mixed_stack_probe [--model PATH] [--layers LIST] [--tokens N] [--start-pos N] [--max-seq N] [--seed N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layers LIST", "Comma-separated layer ids in model order") { |v| layers = parse_layers(v) }
  p.on("--tokens N", "Sequence length for state progression") { |v| tokens = v.to_i }
  p.on("--start-pos N", "Starting decode position for full-attention KV cache") { |v| start_pos = v.to_i }
  p.on("--max-seq N", "KV cache capacity") { |v| max_seq = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--warmup N", "Untimed warmup stack runs") { |v| warmup = v.to_i }
  p.on("--read-logits", "Read full logits back for attribution; default reads resident CUDA top1 only") { read_logits = true }
  p.on("--profile-phases", "Synchronize after each runner and print attribution timings; slower than default") { profile_phases = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layers must not be empty" if layers.empty?
raise "layers must be non-negative" unless layers.all? { |layer| layer >= 0 }
raise "layers must be strictly increasing for this probe" unless layers.each_cons(2).all? { |pair| pair[0] < pair[1] }
raise "tokens must be positive" unless tokens > 0
raise "start-pos must be non-negative" unless start_pos >= 0
raise "max-seq must cover start-pos + tokens" unless max_seq >= start_pos + tokens
raise "warmup must be non-negative" unless warmup >= 0

eps = 1.0e-6_f32
gguf = ML::GGUF::GGUFFile.new(model)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
layers.each { |layer| raise "layer #{layer} out of range" unless layer < hparams.n_layer }
hidden = hparams.n_embd

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { ((rng.next_float - 0.5) * 0.2).to_f32 }

recurrent_weights = {} of Int32 => ML::CUDA::QwenRecurrentLayerRunner::Weights
full_weights = {} of Int32 => ML::CUDA::QwenFullAttnLayerRunner::Weights
head_weights = ML::CUDA::QwenOutputHeadRunner::Weights.load(gguf)
conv_state_inits = {} of Int32 => Array(Float32)
ssm_state_inits = {} of Int32 => Array(Float32)

layers.each do |layer|
  if hparams.full_attention?(layer)
    full_weights[layer] = ML::CUDA::QwenFullAttnLayerRunner::Weights.load(gguf, layer)
  else
    weights = ML::CUDA::QwenRecurrentLayerRunner::Weights.load(gguf, layer, eps)
    recurrent_weights[layer] = weights
    conv_state_inits[layer] = Array(Float32).new((weights.conv_k - 1) * weights.qkv_dim) { ((rng.next_float - 0.5) * 0.05).to_f32 }
    ssm_state_inits[layer] = Array(Float32).new(weights.h_v * weights.s * weights.s) { ((rng.next_float - 0.5) * 0.05).to_f32 }
  end
end

cpu_weights = ML::GGUF::Qwen35Weights.new(gguf, hparams)
cpu_states = Array(ML::GGUF::Qwen35CPU::LayerState).new(hparams.n_layer) { ML::GGUF::Qwen35CPU::LayerState.new }
recurrent_weights.each_key do |layer|
  cpu_states[layer].conv_state = conv_state_inits[layer].dup
  cpu_states[layer].ssm_state = ssm_state_inits[layer].dup
end

cpu_t0 = Time.instant
cpu_current = xs.dup
layers.each do |layer|
  lw = cpu_weights.layers[layer]
  out = Array(Float32).new(tokens * hidden, 0.0_f32)
  tokens.times do |tok|
    row = cpu_current[tok * hidden, hidden]
    y = case lw
        in ML::GGUF::Qwen35FullAttnWeights
          ML::GGUF::Qwen35CPU.forward_full_attn_layer(row, start_pos + tok, lw, cpu_states[layer], hparams, max_seq)
        in ML::GGUF::Qwen35RecurrentWeights
          ML::GGUF::Qwen35CPU.forward_recurrent_layer(row, 0, lw, cpu_states[layer], hparams, max_seq)
        end
    hidden.times { |i| out[tok * hidden + i] = y[i] }
  end
  cpu_current = out
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds
cpu_logits_all = read_logits ? Array(Float32).new(tokens * head_weights.vocab, 0.0_f32) : [] of Float32
cpu_top1_ids = Array(Int32).new(tokens)
tokens.times do |tok|
  row = cpu_current[tok * hidden, hidden]
  normed = ML::GGUF::Qwen35CPU.rms_norm(row, head_weights.norm, hparams.rms_eps)
  logits = ML::GGUF::QuantMatmul.matmul_add(normed, 1, head_weights.hidden,
    head_weights.output_raw, head_weights.output_type, head_weights.vocab,
    Array(Float32).new(head_weights.vocab, 0.0_f32))
  best_id = 0
  best = logits[0]
  head_weights.vocab.times do |i|
    cpu_logits_all[tok * head_weights.vocab + i] = logits[i] if read_logits
    if logits[i] > best
      best = logits[i]
      best_id = i
    end
  end
  cpu_top1_ids << best_id
end

cuda_ctx = nil.as(ML::CUDA::Context?)
runners = [] of ML::CUDA::QwenMixedStackRunner::LayerRunner
head = nil.as(ML::CUDA::QwenOutputHeadRunner?)
stack = nil.as(ML::CUDA::QwenMixedStackRunner?)
final_gpu_all = Array(Float32).new(tokens * hidden, 0.0_f32)

begin
  cuda_ctx = ML::CUDA::Context.create
  cos_table, sin_table = rope_tables(tokens, start_pos, hparams.rope_dim_count, hparams.rope_freq_base)

  layers.each_with_index do |layer, idx|
    layer_input = idx == 0 ? xs : Array(Float32).new(tokens * hidden, 0.0_f32)
    if hparams.full_attention?(layer)
      runners << ML::CUDA::QwenFullAttnLayerRunner.from_weights(full_weights[layer], tokens, max_seq, start_pos,
        hparams.n_head, hparams.n_head_kv, hparams.head_dim, hparams.rope_dim_count, hparams.rms_eps,
        layer_input, cos_table, sin_table)
    else
      runners << ML::CUDA::QwenRecurrentLayerRunner.from_weights(recurrent_weights[layer], tokens, layer_input,
        conv_state_inits[layer], ssm_state_inits[layer])
    end
  end
  head = ML::CUDA::QwenOutputHeadRunner.from_weights(head_weights, tokens,
    Array(Float32).new(tokens * hidden, 0.0_f32), hparams.rms_eps, read_logits: read_logits)
  output_head = head.not_nil!
  stack = ML::CUDA::QwenMixedStackRunner.new(layers, runners, output_head, tokens, hidden, xs)
  mixed_stack = stack.not_nil!

  weight_upload_ms = mixed_stack.upload_weights(profile: profile_phases)

  warmup.times { mixed_stack.run_sequence(profile_phases: false) }

  gpu_ms = mixed_stack.run_sequence(profile_phases: profile_phases)
  final_gpu_all = mixed_stack.final_gpu_all

  lines = [] of String
  ok = report_pair("final_all", final_gpu_all, cpu_current, lines, 1.0e-2_f32)
  gpu_top1_ids = output_head.top1_ids
  top1_ok = gpu_top1_ids == cpu_top1_ids
  if read_logits
    logits_ok = report_pair("logits", output_head.logits_gpu_all, cpu_logits_all, lines, 5.0e-3_f32)
    ok = ok && logits_ok
  else
    lines << "logits_readback=false"
  end
  lines << "top1_gpu=#{gpu_top1_ids.join(",")}"
  lines << "top1_cpu=#{cpu_top1_ids.join(",")}"
  lines << "top1_values_gpu=#{output_head.top1_values_gpu.map { |v| v.round(6) }.join(",")}"
  lines << "top1_ok=#{top1_ok}"
  ok = ok && top1_ok
  runners.each_with_index do |runner, idx|
    layer = layers[idx]
    case runner
    in ML::CUDA::QwenRecurrentLayerRunner
      conv_ok = report_pair("layer#{layer}_conv_state", runner.conv_state_gpu, cpu_states[layer].conv_state.not_nil!, lines, 2.0e-5_f32)
      ssm_ok = report_pair("layer#{layer}_ssm_state", runner.ssm_state_gpu, cpu_states[layer].ssm_state.not_nil!, lines, 1.0e-3_f32)
      ok = ok && conv_ok && ssm_ok
    in ML::CUDA::QwenFullAttnLayerRunner
      kv = runner.kv
      k_cpu = cpu_states[layer].k_cache || Array(Float32).new(max_seq * hparams.n_head_kv * hparams.head_dim, 0.0_f32)
      v_cpu = cpu_states[layer].v_cache || Array(Float32).new(max_seq * hparams.n_head_kv * hparams.head_dim, 0.0_f32)
      k_ok = report_pair("layer#{layer}_k_cache", kv.k_cache_gpu, k_cpu, lines, 2.0e-4_f32)
      v_ok = report_pair("layer#{layer}_v_cache", kv.v_cache_gpu, v_cpu, lines, 1.0e-3_f32)
      ok = ok && k_ok && v_ok
    end
  end

  puts "device=#{cuda_ctx.device_name}"
  puts "compute_capability=#{cuda_ctx.compute_capability_major}.#{cuda_ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layers=#{layers.join(",")}"
  puts "tokens=#{tokens}"
  puts "start_pos=#{start_pos}"
  puts "max_seq=#{max_seq}"
  puts "warmup=#{warmup}"
  puts "read_logits=#{read_logits}"
  puts "profile_phases=#{profile_phases}"
  puts "hidden=#{hidden}"
  puts "vocab=#{head_weights.vocab}"
  puts "weight_upload_ms=#{weight_upload_ms.round(3)}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cuda_ms_per_token=#{(gpu_ms / tokens).round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cpu_ms_per_token=#{(cpu_ms / tokens).round(3)}"
  mixed_stack.phase_lines.each { |line| puts line }
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  stack.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
