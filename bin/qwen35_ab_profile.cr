require "option_parser"
require "../src/ml/bench_load_guard"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"

model = MODEL_PATH
env_name = "QWEN35_WAVE_CHUNK_LAYERS"
a_value = "2"
b_value = "4"
prompt = 64
gen = 32
trials = 8
warmup = 1
top1 = true
load_warning_threshold = 50.0
load_total_warning_threshold = 100.0
wait_quiet_ms = 0
quiet_poll_ms = 1000
require_quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen35_ab_profile [--env NAME] --a VALUE --b VALUE [--prompt N] [--gen N] [--trials N]"
  p.on("--model=PATH", "GGUF model path") { |v| model = v }
  p.on("--env=NAME", "Environment variable to switch (default: QWEN35_WAVE_CHUNK_LAYERS)") { |v| env_name = v }
  p.on("--a=VALUE", "A value for env var") { |v| a_value = v }
  p.on("--b=VALUE", "B value for env var") { |v| b_value = v }
  p.on("--prompt=N", "Prefill tokens before measured decode (default: 64)") { |v| prompt = v.to_i }
  p.on("--gen=N", "Measured decode tokens per trial/config (default: 32)") { |v| gen = v.to_i }
  p.on("--trials=N", "Paired A/B trials (default: 8)") { |v| trials = v.to_i }
  p.on("--warmup=N", "Warmup runs per config before trials (default: 1)") { |v| warmup = v.to_i }
  p.on("--full-logits", "Measure full logits instead of fused top1") { top1 = false }
  p.on("--load-warning-threshold=PCT", "Warn if another process uses at least PCT CPU before benchmarking (default: 50, 0 disables)") { |v| load_warning_threshold = v.to_f }
  p.on("--load-total-warning-threshold=PCT", "Warn if total observed process CPU exceeds PCT before benchmarking (default: 100, 0 disables)") { |v| load_total_warning_threshold = v.to_f }
  p.on("--wait-quiet-ms=N", "Wait up to N ms for host load to fall below benchmark thresholds before measuring") { |v| wait_quiet_ms = v.to_i }
  p.on("--quiet-poll-ms=N", "Polling interval for --wait-quiet-ms (default: 1000)") { |v| quiet_poll_ms = v.to_i }
  p.on("--require-quiet", "Abort instead of warning when host CPU load exceeds process or total thresholds") { require_quiet = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "--prompt must be positive" unless prompt > 0
raise "--gen must be positive" unless gen > 0
raise "--trials must be positive" unless trials > 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "--wait-quiet-ms must be non-negative" unless wait_quiet_ms >= 0
raise "--quiet-poll-ms must be positive" unless quiet_poll_ms > 0
raise "model not found: #{model}" unless File.exists?(model)

ML::BenchLoadGuard.wait_until_quiet!(load_warning_threshold, load_total_warning_threshold, wait_quiet_ms, quiet_poll_ms)
if require_quiet
  ML::BenchLoadGuard.require_quiet!(load_warning_threshold, load_total_warning_threshold)
else
  ML::BenchLoadGuard.warn_if_busy(load_warning_threshold, load_total_warning_threshold)
end

def set_env(name : String, value : String) : Nil
  if value == "<unset>"
    ENV.delete(name)
  else
    ENV[name] = value
  end
end

def token_at(i : Int32) : Int32
  ((i * 7 + 11) % 1000).to_i32
end

def forward_for_ab(w, tok : Int32, pos : Int32, state, top1 : Bool)
  if top1
    ML::GGUF::Qwen35CPU.forward_top1(w, tok, pos, state)
  else
    ML::GGUF::Qwen35CPU.forward(w, tok, pos, state)
  end
end

def run_once(w, base_state, env_name : String, value : String, prompt : Int32, gen : Int32, top1 : Bool) : Float64
  set_env(env_name, value)
  state = base_state.fork
  t0 = Time.instant
  gen.times do |r|
    forward_for_ab(w, 11751_i32, prompt + r, state, top1)
  end
  (Time.instant - t0).total_milliseconds / gen
end

def mean(xs : Array(Float64)) : Float64
  xs.sum / xs.size
end

def percentile(xs : Array(Float64), pct : Int32) : Float64
  sorted = xs.sort
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

old_env = ENV[env_name]?
old_top1 = ENV["QWEN35_HEAD_TOP1_FUSED"]?

begin
  ENV["QWEN35_HEAD_TOP1_FUSED"] = "1" if top1

  w = ML::GGUF::Qwen35Weights.from_gguf(model)
  hp = w.hparams
  base = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: prompt + gen + 2)

  puts "Qwen35 paired A/B profile"
  puts "model=#{model}"
  puts "env=#{env_name} A=#{a_value.inspect} B=#{b_value.inspect}"
  puts "prompt=#{prompt} gen=#{gen} trials=#{trials} warmup=#{warmup} mode=#{top1 ? "top1" : "full_logits"}"
  puts "Building base prefill state..."
  prompt.times { |i| forward_for_ab(w, token_at(i), i, base, top1) }

  warmup.times do
    run_once(w, base, env_name, a_value, prompt, [gen, 4].min, top1)
    run_once(w, base, env_name, b_value, prompt, [gen, 4].min, top1)
  end

  a_ms = [] of Float64
  b_ms = [] of Float64
  a_wins = 0
  b_wins = 0

  trials.times do |i|
    if i.even?
      a = run_once(w, base, env_name, a_value, prompt, gen, top1)
      b = run_once(w, base, env_name, b_value, prompt, gen, top1)
    else
      b = run_once(w, base, env_name, b_value, prompt, gen, top1)
      a = run_once(w, base, env_name, a_value, prompt, gen, top1)
    end
    a_ms << a
    b_ms << b
    if a < b
      a_wins += 1
    elsif b < a
      b_wins += 1
    end
    printf "trial=%02d A=%.3f ms/tok B=%.3f ms/tok delta(A-B)=%.3f winner=%s\n",
      i + 1, a, b, a - b, a < b ? "A" : (b < a ? "B" : "tie")
  end

  puts
  printf "A mean=%.3f p50=%.3f p90=%.3f wins=%d/%d\n",
    mean(a_ms), percentile(a_ms, 50), percentile(a_ms, 90), a_wins, trials
  printf "B mean=%.3f p50=%.3f p90=%.3f wins=%d/%d\n",
    mean(b_ms), percentile(b_ms, 50), percentile(b_ms, 90), b_wins, trials
  printf "delta_mean(A-B)=%.3f ms/tok\n", mean(a_ms) - mean(b_ms)
ensure
  if old_env
    ENV[env_name] = old_env
  else
    ENV.delete(env_name)
  end

  if old_top1
    ENV["QWEN35_HEAD_TOP1_FUSED"] = old_top1
  else
    ENV.delete("QWEN35_HEAD_TOP1_FUSED")
  end
end
