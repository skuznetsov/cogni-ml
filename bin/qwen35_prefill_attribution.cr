require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

model = MODEL_PATH
prompt_len = 64
warmup = 1
reps = 3
compare_env = nil.as(String?)
compare_off = "1"

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_prefill_attribution [--model PATH] [--prompt N] [--warmup N] [--reps N] [--compare-env NAME]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--prompt=N", "Prompt tokens for prefill attribution (default: 64)") { |v| prompt_len = v.to_i }
  p.on("--warmup=N", "Warmup runs before profiling (default: 1)") { |v| warmup = v.to_i }
  p.on("--reps=N", "Measured repetitions for wall timing (default: 3)") { |v| reps = v.to_i }
  p.on("--compare-env=NAME", "Also run A/B with NAME unset vs NAME=1") { |v| compare_env = v }
  p.on("--compare-off=VALUE", "Off value for --compare-env (default: 1)") { |v| compare_off = v }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt must be positive" unless prompt_len > 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "--reps must be positive" unless reps > 0

def prompt_tokens(n : Int32) : Array(Int32)
  Array(Int32).new(n) { |i| ((i * 7 + 11) % 1000).to_i32 }
end

def run_prefill_once(w : ML::GGUF::Qwen35Weights,
                     prompt : Array(Int32),
                     profile : Bool) : Float64
  state = ML::GGUF::Qwen35CPU::State.new(w.hparams, max_seq: prompt.size + 4)
  ML::GGUF::Qwen35Metal::Profile.reset if profile
  ML::GGUF::Qwen35Metal::Profile.enable! if profile
  t0 = Time.instant
  ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, state)
  wall_ms = (Time.instant - t0).total_milliseconds
  ML::GGUF::Qwen35Metal::Profile.disable! if profile
  wall_ms
end

def percentile(xs : Array(Float64), pct : Int32) : Float64
  sorted = xs.sort
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

def mean(xs : Array(Float64)) : Float64
  xs.sum / xs.size
end

def set_env(name : String, value : String?) : Nil
  if value
    ENV[name] = value
  else
    ENV.delete(name)
  end
end

def measure_wall(w, prompt, warmup : Int32, reps : Int32) : Array(Float64)
  warmup.times { run_prefill_once(w, prompt, profile: false) }
  Array(Float64).new(reps) { run_prefill_once(w, prompt, profile: false) }
end

w = ML::GGUF::Qwen35Weights.from_gguf(model)
prompt = prompt_tokens(prompt_len)

puts "Qwen35 prefill attribution"
puts "model=#{model}"
puts "prompt=#{prompt_len} warmup=#{warmup} reps=#{reps}"

warmup.times { run_prefill_once(w, prompt, profile: false) }
profile_ms = run_prefill_once(w, prompt, profile: true)
puts
print ML::GGUF::Qwen35Metal::Profile.report_io
printf "  profiled wall: %.2f ms  %.2f tok/s\n", profile_ms, prompt_len * 1000.0 / profile_ms

times = Array(Float64).new(reps) { run_prefill_once(w, prompt, profile: false) }
printf "  wall reps: avg=%.2f ms p50=%.2f ms p90=%.2f ms p50=%.2f tok/s\n",
  mean(times), percentile(times, 50), percentile(times, 90),
  prompt_len * 1000.0 / percentile(times, 50)

if env = compare_env
  old = ENV[env]?
  begin
    set_env(env, nil)
    on = measure_wall(w, prompt, warmup, reps)
    set_env(env, compare_off)
    off = measure_wall(w, prompt, warmup, reps)
    puts
    puts "A/B #{env}: default vs #{compare_off.inspect}"
    printf "  default: avg=%.2f ms p50=%.2f ms %.2f tok/s\n",
      mean(on), percentile(on, 50), prompt_len * 1000.0 / percentile(on, 50)
    printf "  off:     avg=%.2f ms p50=%.2f ms %.2f tok/s\n",
      mean(off), percentile(off, 50), prompt_len * 1000.0 / percentile(off, 50)
    printf "  delta default-off: %.2f ms\n", mean(on) - mean(off)
  ensure
    set_env(env, old)
  end
end
