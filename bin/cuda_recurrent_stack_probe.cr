# CUDA recurrent multi-layer scaffold probe for Qwen GGUF weights.
#
# This is a correctness scaffold, not an end-to-end decoder: it chains
# recurrent-layer CUDA runners and can compare device-resident layer handoff
# against the older host-handoff debug mode.

require "option_parser"
require "../src/ml/gguf/reader"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/cuda/qwen_recurrent_stack_runner"

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

model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
layers = [0, 2, 4]
seed = 31_u64
tokens = 2
warmup = 0
host_handoff = false

OptionParser.parse do |p|
  p.banner = "Usage: cuda_recurrent_stack_probe [--model PATH] [--layers LIST] [--tokens N] [--seed N] [--warmup N]"
  p.on("--model PATH", "Qwen Q4_K_M GGUF model path") { |v| model = v }
  p.on("--layers LIST", "Comma-separated recurrent layer ids") { |v| layers = parse_layers(v) }
  p.on("--tokens N", "Sequence length for recurrent state progression") { |v| tokens = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("--warmup N", "Untimed warmup stack runs") { |v| warmup = v.to_i }
  p.on("--host-handoff", "Debug mode: copy each layer output through host before the next layer") { host_handoff = true }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "layers must not be empty" if layers.empty?
raise "layer ids must be non-negative" unless layers.all? { |layer| layer >= 0 }
raise "tokens must be positive" unless tokens > 0
raise "warmup must be non-negative" unless warmup >= 0

eps = 1.0e-6_f32
gguf = ML::GGUF::GGUFFile.new(model)
hparams = ML::GGUF::Qwen35Hparams.new(gguf)
layers.each do |layer|
  raise "layer #{layer} out of range" unless layer < hparams.n_layer
  raise "layer #{layer} is not recurrent" unless hparams.recurrent?(layer)
end

cuda_weights = layers.map { |layer| ML::CUDA::QwenRecurrentLayerRunner::Weights.load(gguf, layer, eps) }
hidden = cuda_weights.first.hidden
cuda_weights.each { |weight| raise "mixed hidden sizes are unsupported" unless weight.hidden == hidden }

rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { ((rng.next_float - 0.5) * 0.2).to_f32 }
conv_state_inits = cuda_weights.map do |weight|
  Array(Float32).new((weight.conv_k - 1) * weight.qkv_dim) { ((rng.next_float - 0.5) * 0.05).to_f32 }
end
ssm_state_inits = cuda_weights.map do |weight|
  Array(Float32).new(weight.h_v * weight.s * weight.s) { ((rng.next_float - 0.5) * 0.05).to_f32 }
end

cpu_weights = ML::GGUF::Qwen35Weights.new(gguf, hparams)
cpu_states = Array(ML::GGUF::Qwen35CPU::LayerState).new(layers.size) do |idx|
  state = ML::GGUF::Qwen35CPU::LayerState.new
  state.conv_state = conv_state_inits[idx].dup
  state.ssm_state = ssm_state_inits[idx].dup
  state
end

cpu_t0 = Time.instant
cpu_current = xs.dup
layers.each_with_index do |layer, idx|
  layer_weights = cpu_weights.layers[layer]
  recurrent_weights = layer_weights.as?(ML::GGUF::Qwen35RecurrentWeights) || raise "layer #{layer} is not recurrent weights"
  out = Array(Float32).new(tokens * hidden, 0.0_f32)
  tokens.times do |tok|
    row = cpu_current[tok * hidden, hidden]
    y = ML::GGUF::Qwen35CPU.forward_recurrent_layer(row, 0, recurrent_weights, cpu_states[idx], hparams, tokens)
    hidden.times { |i| out[tok * hidden + i] = y[i] }
  end
  cpu_current = out
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

cuda_ctx = nil.as(ML::CUDA::Context?)
stack = nil.as(ML::CUDA::QwenRecurrentStackRunner?)

begin
  cuda_ctx = ML::CUDA::Context.create
  stack = ML::CUDA::QwenRecurrentStackRunner.new(layers, cuda_weights, tokens, xs, conv_state_inits, ssm_state_inits, host_handoff)

  upload_t0 = Time.instant
  stack.upload_weights
  ML::CUDA.synchronize!("cuCtxSynchronize(stack upload)")
  weight_upload_ms = (Time.instant - upload_t0).total_milliseconds

  warmup.times { stack.run_sequence }

  gpu_t0 = Time.instant
  stack.run_sequence
  gpu_ms = (Time.instant - gpu_t0).total_milliseconds

  lines = [] of String
  ok = report_pair("final_all", stack.final_gpu_all, cpu_current, lines, 5.0e-3_f32)
  stack.runners.each_with_index do |runner, idx|
    conv_ok = report_pair("layer#{layers[idx]}_conv_state", runner.conv_state_gpu, cpu_states[idx].conv_state.not_nil!, lines, 2.0e-5_f32)
    ssm_ok = report_pair("layer#{layers[idx]}_ssm_state", runner.ssm_state_gpu, cpu_states[idx].ssm_state.not_nil!, lines, 5.0e-4_f32)
    ok = ok && conv_ok && ssm_ok
  end

  puts "device=#{cuda_ctx.device_name}"
  puts "compute_capability=#{cuda_ctx.compute_capability_major}.#{cuda_ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layers=#{layers.join(",")}"
  puts "tokens=#{tokens}"
  puts "warmup=#{warmup}"
  puts "handoff=#{host_handoff ? "host" : "device"}"
  puts "hidden=#{hidden}"
  puts "weight_upload_ms=#{weight_upload_ms.round(3)}"
  puts "cuda_ms=#{gpu_ms.round(3)}"
  puts "cuda_ms_per_token=#{(gpu_ms / tokens).round(3)}"
  puts "cpu_ms=#{cpu_ms.round(3)}"
  puts "cpu_ms_per_token=#{(cpu_ms / tokens).round(3)}"
  lines.each { |line| puts line }
  puts "ok=#{ok}"
ensure
  stack.try(&.close)
  cuda_ctx.try(&.close)
  gguf.close
end
