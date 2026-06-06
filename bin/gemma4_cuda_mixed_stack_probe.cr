# CogniGemma CUDA mixed-layer stack smoke.
#
# Chains probe-grade Gemma4 CUDA layer runners with device-resident hidden
# handoff, then compares the selected-layer prefix against Gemma4CPU.

require "option_parser"
require "../src/ml/gguf/gemma4_cpu"
require "../src/ml/cuda/driver"
require "../src/ml/cuda/gemma4_layer_runner"

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
layers_arg = "0,1,2,3,4,5"
tokens = 2
base_pos = 0
seed = 23_u64
reps = 3
warmup = 1

OptionParser.parse do |p|
  p.banner = "Usage: gemma4_cuda_mixed_stack_probe [--model PATH] [--layers LIST] [--tokens N] [--base-pos N] [--reps N] [--warmup N] [--seed N]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--layers LIST", "Comma-separated layer ids to compose") { |v| layers_arg = v }
  p.on("--tokens N", "Synthetic token span length") { |v| tokens = v.to_i }
  p.on("--base-pos N", "Absolute start position for the synthetic span") { |v| base_pos = v.to_i }
  p.on("--reps N", "Timed launches") { |v| reps = v.to_i }
  p.on("--warmup N", "Untimed warmup launches") { |v| warmup = v.to_i }
  p.on("--seed N", "Random seed") { |v| seed = v.to_u64 }
  p.on("-h", "--help", "Show help") { puts p; exit 0 }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "tokens must be positive" unless tokens > 0
raise "base-pos must be non-negative" unless base_pos >= 0
raise "reps must be positive" unless reps > 0
raise "warmup must be non-negative" unless warmup >= 0
layers = layers_arg.split(',').map(&.strip).reject(&.empty?).map(&.to_i)
raise "layers must be non-empty" if layers.empty?

weights = ML::GGUF::Gemma4Weights.from_gguf(model)
hp = weights.hparams
hidden = hp.n_embd
max_seq = base_pos + tokens
rng = Random.new(seed)
xs = Array(Float32).new(tokens * hidden) { rng.rand(-1.0_f32..1.0_f32) }

state = ML::GGUF::Gemma4CPU::State.new(hp, max_seq)
cpu = Array(Float32).new(tokens * hidden, 0.0_f32)
cpu_t0 = Time.instant
tokens.times do |tok|
  abs_pos = base_pos + tok
  row = xs[tok * hidden, hidden]
  layers.each do |layer|
    row = ML::GGUF::Gemma4CPU.forward_layer(weights, layer, row, abs_pos, state)
  end
  hidden.times { |i| cpu[tok * hidden + i] = row[i] }
end
cpu_ms = (Time.instant - cpu_t0).total_milliseconds

ctx = nil.as(ML::CUDA::Context?)
runners = [] of ML::CUDA::Gemma4LayerRunner
begin
  ctx = ML::CUDA::Context.create
  dummy = Array(Float32).new(tokens * hidden, 0.0_f32)
  layers.each_with_index do |layer, idx|
    input = idx == 0 ? xs : dummy
    runner = ML::CUDA::Gemma4LayerRunner.new(weights, layer, tokens, max_seq, base_pos, input)
    runners << runner
  end

  runners.each(&.upload_weights)

  run_once = -> {
    previous = 0_u64
    runners.each_with_index do |runner, idx|
      if idx == 0
        runner.reset_sequence
      else
        runner.use_device_sequence_input(previous)
        runner.reset_sequence
      end
      runner.run_sequence
      previous = runner.output_device_ptr
    end
  }

  warmup.times { run_once.call }
  ML::CUDA.synchronize!("warmup")
  t0 = Time.instant
  reps.times { run_once.call }
  ML::CUDA.synchronize!("timed")
  cuda_ms = (Time.instant - t0).total_milliseconds / reps

  last = runners.last
  last.read_outputs
  gpu = last.final_gpu_all
  cos_v = cosine(gpu, cpu)
  diff = max_abs_diff(gpu, cpu)
  ok = cos_v >= 0.99999 && diff <= 5.0e-4_f32

  full_count = layers.count { |l| hp.full_attention?(l) }
  swa_count = layers.size - full_count
  puts "device=#{ctx.device_name}"
  puts "compute_capability=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  puts "model=#{model}"
  puts "layers=#{layers.join(',')}"
  puts "layer_count=#{layers.size}"
  puts "swa_count=#{swa_count}"
  puts "full_attention_count=#{full_count}"
  puts "tokens=#{tokens}"
  puts "base_pos=#{base_pos}"
  puts "hidden=#{hidden}"
  puts "cuda_ms=#{cuda_ms.round(4)}"
  puts "cuda_ms_per_token=#{(cuda_ms / tokens).round(4)}"
  puts "cpu_ms=#{cpu_ms.round(4)}"
  puts "cos=#{cos_v.round(8)}"
  puts "max_diff=#{diff}"
  puts "ok=#{ok}"
  exit(ok ? 0 : 1)
ensure
  runners.reverse_each(&.close)
  ctx.try(&.close)
end
