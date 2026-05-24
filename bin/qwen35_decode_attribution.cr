require "option_parser"
require "../src/ml/bench_load_guard"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_metal"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

model = MODEL_PATH
prompt_len = 0
gen_len = 64
warmup = 1
reps = 3
compare_env = nil.as(String?)
compare_off = "1"
compare_off_set = false
load_warning_threshold = 50.0
load_total_warning_threshold = 100.0
wait_quiet_ms = 0
quiet_poll_ms = 1000
require_quiet = false
greedy_chain = false
gpu_token_chain = false
decode_mode = "top1"

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_decode_attribution [--model PATH] [--prompt N] [--gen N] [--warmup N] [--reps N] [--compare-env NAME] [--body-only] [--gpu-token-chain]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--prompt=N", "Prompt tokens to prefill before timed decode; 0 matches benchmark synthetic decode state (default: 0)") { |v| prompt_len = v.to_i }
  p.on("--gen=N", "Decode tokens for attribution (default: 64)") { |v| gen_len = v.to_i }
  p.on("--warmup=N", "Warmup runs before profiling (default: 1)") { |v| warmup = v.to_i }
  p.on("--reps=N", "Measured repetitions for wall timing (default: 3)") { |v| reps = v.to_i }
  p.on("--compare-env=NAME", "Also run A/B with NAME unset vs NAME=VALUE; NAME=VALUE is accepted") { |v| compare_env = v }
  p.on("--compare-off=VALUE", "Off value for --compare-env (default: 1)") { |v| compare_off = v; compare_off_set = true }
  p.on("--body-only", "Measure decoder body/state update only, matching llama-bench tg logits=nullptr") { decode_mode = "body" }
  p.on("--top1", "Measure product-shaped greedy top1 decode (default)") { decode_mode = "top1" }
  p.on("--greedy-chain", "Feed each generated top1 token into the next step instead of benchmark synthetic input tokens") { greedy_chain = true }
  p.on("--gpu-token-chain", "Use GPU-resident exact greedy token handoff for the timed decode suffix") { gpu_token_chain = true }
  p.on("--load-warning-threshold=PCT", "Warn if another process uses at least PCT CPU before benchmarking (default: 50, 0 disables)") { |v| load_warning_threshold = v.to_f }
  p.on("--load-total-warning-threshold=PCT", "Warn if total observed process CPU exceeds PCT before benchmarking (default: 100, 0 disables)") { |v| load_total_warning_threshold = v.to_f }
  p.on("--wait-quiet-ms=N", "Wait up to N ms for host load to fall below benchmark thresholds before measuring") { |v| wait_quiet_ms = v.to_i }
  p.on("--quiet-poll-ms=N", "Polling interval for --wait-quiet-ms (default: 1000)") { |v| quiet_poll_ms = v.to_i }
  p.on("--require-quiet", "Abort instead of warning when host CPU load exceeds process or total thresholds") { require_quiet = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt must be non-negative" unless prompt_len >= 0
raise "--gen must be positive" unless gen_len > 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "--reps must be positive" unless reps > 0
raise "--wait-quiet-ms must be non-negative" unless wait_quiet_ms >= 0
raise "--quiet-poll-ms must be positive" unless quiet_poll_ms > 0
raise "--gpu-token-chain requires top1 mode" if gpu_token_chain && decode_mode != "top1"
if env = compare_env
  if env.includes?("=")
    name, value = env.split("=", 2)
    compare_env = name
    compare_off = value unless compare_off_set
  end
  raise "--compare-env name must not be empty" if compare_env.try(&.empty?)
  raise "--compare-env name is invalid: #{compare_env}" unless compare_env.not_nil!.matches?(/\A[A-Za-z_][A-Za-z0-9_]*\z/)
end

ML::BenchLoadGuard.wait_until_quiet!(load_warning_threshold, load_total_warning_threshold, wait_quiet_ms, quiet_poll_ms)
if require_quiet
  ML::BenchLoadGuard.require_quiet!(load_warning_threshold, load_total_warning_threshold)
else
  ML::BenchLoadGuard.warn_if_busy(load_warning_threshold, load_total_warning_threshold)
end

def prompt_tokens(n : Int32) : Array(Int32)
  Array(Int32).new(n) { |i| ((i * 7 + 11) % 1000).to_i32 }
end

def synthetic_decode_tokens(n : Int32) : Array(Int32)
  Array(Int32).new(n) { |i| ((i * 13 + 11751) % 32000).to_i32 }
end

def prepare_state(w : ML::GGUF::Qwen35Weights, prompt : Array(Int32), gen_len : Int32) : ML::GGUF::Qwen35CPU::State
  state = ML::GGUF::Qwen35CPU::State.new(w.hparams, max_seq: prompt.size + gen_len + 4)
  ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, state) unless prompt.empty?
  state
end

def run_decode_once(w : ML::GGUF::Qwen35Weights,
                    prompt : Array(Int32),
                    gen_len : Int32,
                    profile : Bool,
                    greedy_chain : Bool,
                    gpu_token_chain : Bool,
                    decode_mode : String) : Float64
  state = prepare_state(w, prompt, gen_len)
  token = prompt.empty? ? 11751_i32 : prompt[-1]
  synthetic = synthetic_decode_tokens(gen_len)
  start_pos = prompt.size

  ML::GGUF::Qwen35Metal::Profile.reset if profile
  ML::GGUF::Qwen35Metal::Profile.enable! if profile
  t0 = Time.instant
  if gpu_token_chain
    tokens = ML::GGUF::Qwen35CPU.forward_top1_chain_gpu(w, token, start_pos.to_i32, state, gen_len)
    raise "GPU token chain route unavailable" if tokens.nil?
  else
    gen_len.times do |i|
      input = if greedy_chain || !prompt.empty?
                token
              else
                synthetic[i]
              end
      if decode_mode == "body"
        ML::GGUF::Qwen35CPU.prefill_token(w, input, (start_pos + i).to_i32, state)
      else
        top1, _logit = ML::GGUF::Qwen35CPU.forward_top1(w, input, (start_pos + i).to_i32, state)
        token = top1
      end
    end
  end
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

def measure_paired_env(w,
                       prompt,
                       gen_len : Int32,
                       env : String,
                       alternate_value : String,
                       warmup : Int32,
                       reps : Int32,
                       greedy_chain : Bool,
                       gpu_token_chain : Bool,
                       decode_mode : String) : {Array(Float64), Array(Float64)}
  default = [] of Float64
  alternate = [] of Float64

  warmup.times do
    set_env(env, nil)
    run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
    set_env(env, alternate_value)
    run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
  end

  reps.times do |i|
    if i.even?
      set_env(env, nil)
      a = run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
      set_env(env, alternate_value)
      b = run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
    else
      set_env(env, alternate_value)
      b = run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
      set_env(env, nil)
      a = run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
    end
    default << a
    alternate << b
  end

  {default, alternate}
ensure
  set_env(env, nil)
end

w = ML::GGUF::Qwen35Weights.from_gguf(model)
prompt = prompt_tokens(prompt_len)

puts "Qwen35 decode attribution"
puts "model=#{model}"
mode = if gpu_token_chain
         "gpu_token_chain"
       elsif decode_mode == "body"
         "body_only"
       elsif greedy_chain || prompt_len > 0
         "greedy_chain"
       else
         "top1_synthetic_inputs"
       end
puts "prompt=#{prompt_len} gen=#{gen_len} warmup=#{warmup} reps=#{reps} mode=#{mode}"

warmup.times { run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode) }
profile_ms = run_decode_once(w, prompt, gen_len, profile: true, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode)
puts
print ML::GGUF::Qwen35Metal::Profile.report_io
printf "  profiled wall: %.2f ms  %.2f tok/s\n", profile_ms, gen_len * 1000.0 / profile_ms

times = Array(Float64).new(reps) { run_decode_once(w, prompt, gen_len, profile: false, greedy_chain: greedy_chain, gpu_token_chain: gpu_token_chain, decode_mode: decode_mode) }
printf "  wall reps: avg=%.2f ms p50=%.2f ms p90=%.2f ms p50=%.2f tok/s\n",
  mean(times), percentile(times, 50), percentile(times, 90),
  gen_len * 1000.0 / percentile(times, 50)

if env = compare_env
  old = ENV[env]?
  begin
    on, off = measure_paired_env(w, prompt, gen_len, env, compare_off, warmup, reps, greedy_chain, gpu_token_chain, decode_mode)
    wins = on.zip(off).count { |a, b| a < b }
    puts
    puts "A/B #{env}: default vs #{compare_off.inspect} (paired interleaved)"
    printf "  default: avg=%.2f ms p50=%.2f ms %.2f tok/s\n",
      mean(on), percentile(on, 50), gen_len * 1000.0 / percentile(on, 50)
    printf "  other:   avg=%.2f ms p50=%.2f ms %.2f tok/s\n",
      mean(off), percentile(off, 50), gen_len * 1000.0 / percentile(off, 50)
    printf "  default-other: %.2f ms  wins=%d/%d\n", mean(on) - mean(off), wins, reps
  ensure
    set_env(env, old)
  end
end
