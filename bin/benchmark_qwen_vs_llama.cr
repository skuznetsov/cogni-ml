require "json"
require "option_parser"
require "../src/ml/bench_load_guard"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_weights"

MODEL_PATH  = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_BENCH = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-bench"

record NativeStats,
  avg_ms : Float64,
  p50_ms : Float64,
  p95_ms : Float64,
  tok_s_avg : Float64,
  tok_s_p50 : Float64

record LlamaStats,
  avg_ts : Float64,
  stddev_ts : Float64,
  avg_ns : Int64

def percentile(sorted : Array(Float64), pct : Int32) : Float64
  idx = (sorted.size * pct // 100).clamp(0, sorted.size - 1)
  sorted[idx]
end

def measure_native_prefill(w : ML::GGUF::Qwen35Weights, n_prompt : Int32, reps : Int32, warmup : Int32) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }

  warmup.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    run_native_prefill(w, prompt, state)
  end

  times = Array(Float64).new(reps)
  reps.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    t0 = Time.instant
    run_native_prefill(w, prompt, state)
    times << (Time.instant - t0).total_milliseconds
  end

  sorted = times.sort
  avg_ms = times.sum / times.size
  p50_ms = percentile(sorted, 50)
  p95_ms = percentile(sorted, 95)
  NativeStats.new(
    avg_ms: avg_ms,
    p50_ms: p50_ms,
    p95_ms: p95_ms,
    tok_s_avg: (n_prompt * 1000.0) / avg_ms,
    tok_s_p50: (n_prompt * 1000.0) / p50_ms,
  )
end

def measure_native_prefill_prepared_state(w : ML::GGUF::Qwen35Weights, n_prompt : Int32, reps : Int32, warmup : Int32) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }

  warmup.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    run_native_prefill(w, prompt, state)
  end

  times = Array(Float64).new(reps)
  reps.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    t0 = Time.instant
    run_native_prefill(w, prompt, state)
    times << (Time.instant - t0).total_milliseconds
  end

  sorted = times.sort
  avg_ms = times.sum / times.size
  p50_ms = percentile(sorted, 50)
  p95_ms = percentile(sorted, 95)
  NativeStats.new(
    avg_ms: avg_ms,
    p50_ms: p50_ms,
    p95_ms: p95_ms,
    tok_s_avg: (n_prompt * 1000.0) / avg_ms,
    tok_s_p50: (n_prompt * 1000.0) / p50_ms,
  )
end

def clear_float_buffer(buf : ML::MetalBuffer?) : Nil
  return unless b = buf

  b.contents.as(Pointer(UInt8)).clear(b.size)
end

def clear_float_array(values : Array(Float32)?) : Nil
  return unless xs = values

  xs.fill(0.0_f32)
end

def reset_prefill_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  state.layers.each do |layer|
    layer.position = 0
    # At start_pos=0 full-attention K/V rows used by the prompt are overwritten
    # before attention reads them. DeltaNet conv/SSM state is true recurrence
    # state and must be reset for exact repeated timing.
    clear_float_array(layer.conv_state)
    clear_float_array(layer.ssm_state)
    clear_float_buffer(layer.conv_state_buf)
    clear_float_buffer(layer.ssm_state_buf)
  end
end

def measure_native_prefill_preallocated(w : ML::GGUF::Qwen35Weights, n_prompt : Int32, reps : Int32, warmup : Int32) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)

  # Allocate the backing GPU state buffers once, outside the timed section.
  run_native_prefill(w, prompt, state)
  reset_prefill_state!(state)

  warmup.times do
    run_native_prefill(w, prompt, state)
    reset_prefill_state!(state)
  end

  times = Array(Float64).new(reps)
  reps.times do
    t0 = Time.instant
    run_native_prefill(w, prompt, state)
    times << (Time.instant - t0).total_milliseconds
    reset_prefill_state!(state)
  end

  sorted = times.sort
  avg_ms = times.sum / times.size
  p50_ms = percentile(sorted, 50)
  p95_ms = percentile(sorted, 95)
  NativeStats.new(
    avg_ms: avg_ms,
    p50_ms: p50_ms,
    p95_ms: p95_ms,
    tok_s_avg: (n_prompt * 1000.0) / avg_ms,
    tok_s_p50: (n_prompt * 1000.0) / p50_ms,
  )
end

def measure_native_prefill_cached(w : ML::GGUF::Qwen35Weights,
                                  model : String,
                                  n_prompt : Int32,
                                  reps : Int32,
                                  warmup : Int32) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }
  root = File.tempname("qwen35-bench-prompt-cache")
  Dir.mkdir_p(root)

  begin
    store = ML::GGUF::Qwen35PromptCache::Store.new(root)
    seeded = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    run_native_prefill(w, prompt, seeded)
    model_id = ML::GGUF::Qwen35PromptCache.short_hash("bench-model\0#{model}")
    tokenizer_id = "synthetic-token-ids-v1"
    entry = store.save(
      session_id: "benchmark",
      model_id: model_id,
      tokenizer_id: tokenizer_id,
      prompt_text: "",
      token_ids: prompt,
      state: seeded,
    )

    warmup.times do
      restored = store.restore(entry, hp)
      raise "prompt-cache restore layer mismatch" unless restored.layers.size == hp.n_layer
    end

    times = Array(Float64).new(reps)
    reps.times do
      t0 = Time.instant
      restored = store.restore(entry, hp)
      raise "prompt-cache restore layer mismatch" unless restored.layers.size == hp.n_layer
      times << (Time.instant - t0).total_milliseconds
    end

    sorted = times.sort
    avg_ms = times.sum / times.size
    p50_ms = percentile(sorted, 50)
    p95_ms = percentile(sorted, 95)
    NativeStats.new(
      avg_ms: avg_ms,
      p50_ms: p50_ms,
      p95_ms: p95_ms,
      tok_s_avg: (n_prompt * 1000.0) / avg_ms,
      tok_s_p50: (n_prompt * 1000.0) / p50_ms,
    )
  ensure
    FileUtils.rm_rf(root) if Dir.exists?(root)
  end
end

def run_native_prefill(w : ML::GGUF::Qwen35Weights,
                       prompt : Array(Int32),
                       state : ML::GGUF::Qwen35CPU::State) : Nil
  return if prompt.empty?

  ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, state)
end

def forward_decode_token(w : ML::GGUF::Qwen35Weights, tok : Int32, pos : Int32,
                         state : ML::GGUF::Qwen35CPU::State, top1 : Bool) : Nil
  if top1
    ML::GGUF::Qwen35CPU.forward_top1(w, tok, pos, state)
  else
    ML::GGUF::Qwen35CPU.forward(w, tok, pos, state)
  end
end

def measure_native_decode(w : ML::GGUF::Qwen35Weights, n_gen : Int32, reps : Int32, warmup : Int32, top1 : Bool) : NativeStats
  hp = w.hparams
  decode_tokens = Array(Int32).new(n_gen) { |i| ((i * 13 + 11751) % 32000).to_i32 }

  warmup.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_gen + 4)
    decode_tokens.each_with_index { |tok, pos| forward_decode_token(w, tok, pos.to_i32, state, top1) }
  end

  times = Array(Float64).new(reps)
  reps.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_gen + 4)
    t0 = Time.instant
    decode_tokens.each_with_index { |tok, pos| forward_decode_token(w, tok, pos.to_i32, state, top1) }
    times << (Time.instant - t0).total_milliseconds
  end

  sorted = times.sort
  avg_ms = times.sum / times.size
  p50_ms = percentile(sorted, 50)
  p95_ms = percentile(sorted, 95)
  NativeStats.new(
    avg_ms: avg_ms,
    p50_ms: p50_ms,
    p95_ms: p95_ms,
    tok_s_avg: (n_gen * 1000.0) / avg_ms,
    tok_s_p50: (n_gen * 1000.0) / p50_ms,
  )
end

def run_llama_bench(llama_bench : String, model : String, n_prompt : Int32, n_gen : Int32, reps : Int32, n_gpu_layers : Int32, threads : Int32, flash_attn : Bool) : LlamaStats
  output = IO::Memory.new
  error = IO::Memory.new
  args = [
    "-m", model,
    "-p", n_prompt.to_s,
    "-n", n_gen.to_s,
    "-ngl", n_gpu_layers.to_s,
    "-t", threads.to_s,
    "-fa", flash_attn ? "1" : "0",
    "-r", reps.to_s,
    "-o", "json",
  ]
  status = Process.run(llama_bench, args: args, output: output, error: error)
  unless status.success?
    raise "llama-bench failed: #{error.to_s}"
  end

  parsed = JSON.parse(output.to_s)
  row = parsed.as_a.first
  LlamaStats.new(
    avg_ts: row["avg_ts"].as_f,
    stddev_ts: row["stddev_ts"].as_f,
    avg_ns: row["avg_ns"].as_i64,
  )
end

def pct_gap(native : Float64, llama : Float64) : Float64
  ((native / llama) - 1.0) * 100.0
end

model = MODEL_PATH
llama_bench = LLAMA_BENCH
n_prompt = 64
n_gen = 64
reps = 5
warmup = 2
n_gpu_layers = 99
threads = 8
flash_attn = false
native_decode_top1 = true
native_prefill_cache = false
native_prefill_prealloc = false
native_prefill_prepare_state = false
load_warning_threshold = 50.0
load_total_warning_threshold = 100.0
wait_quiet_ms = 0
quiet_poll_ms = 1000
require_quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: benchmark_qwen_vs_llama [options]"
  p.on("--model=PATH", "Path to Qwen GGUF") { |v| model = v }
  p.on("--llama-bench=PATH", "Path to llama-bench binary") { |v| llama_bench = v }
  p.on("--prompt=N", "Prompt tokens for prefill benchmark (default: 64)") { |v| n_prompt = v.to_i }
  p.on("--gen=N", "Generated tokens for decode benchmark (default: 64)") { |v| n_gen = v.to_i }
  p.on("--reps=N", "Repetitions (default: 5)") { |v| reps = v.to_i }
  p.on("--warmup=N", "Warmup repetitions for native path (default: 2)") { |v| warmup = v.to_i }
  p.on("--ngl=N", "llama.cpp GPU layers (default: 99)") { |v| n_gpu_layers = v.to_i }
  p.on("--threads=N", "llama.cpp CPU threads (default: 8)") { |v| threads = v.to_i }
  p.on("--flash-attn", "Enable flash attention in llama.cpp") { flash_attn = true }
  p.on("--native-full-logits", "Measure native decode with full lm-head logits instead of greedy top1") { native_decode_top1 = false }
  p.on("--native-prefill-cache", "Measure native prefill as exact prompt-cache restore after one seeded run") { native_prefill_cache = true }
  p.on("--native-prefill-prealloc", "Measure native prefill with state buffers allocated outside the timed loop") { native_prefill_prealloc = true }
  p.on("--native-prefill-prepare-state", "Prepare a fresh state's Metal buffers before each timed native prefill") { native_prefill_prepare_state = true }
  p.on("--load-warning-threshold=PCT", "Warn if another process uses at least PCT CPU before benchmarking (default: 50, 0 disables)") { |v| load_warning_threshold = v.to_f }
  p.on("--load-total-warning-threshold=PCT", "Warn if total observed process CPU exceeds PCT before benchmarking (default: 100, 0 disables)") { |v| load_total_warning_threshold = v.to_f }
  p.on("--wait-quiet-ms=N", "Wait up to N ms for host load to fall below benchmark thresholds before measuring") { |v| wait_quiet_ms = v.to_i }
  p.on("--quiet-poll-ms=N", "Polling interval for --wait-quiet-ms (default: 1000)") { |v| quiet_poll_ms = v.to_i }
  p.on("--require-quiet", "Abort instead of warning when host CPU load exceeds process or total thresholds") { require_quiet = true }
end

raise "Model not found: #{model}" unless File.exists?(model)
raise "llama-bench not found: #{llama_bench}" unless File.exists?(llama_bench)
raise "--wait-quiet-ms must be non-negative" unless wait_quiet_ms >= 0
raise "--quiet-poll-ms must be positive" unless quiet_poll_ms > 0

ML::BenchLoadGuard.wait_until_quiet!(load_warning_threshold, load_total_warning_threshold, wait_quiet_ms, quiet_poll_ms)
if require_quiet
  ML::BenchLoadGuard.require_quiet!(load_warning_threshold, load_total_warning_threshold)
else
  ML::BenchLoadGuard.warn_if_busy(load_warning_threshold, load_total_warning_threshold)
end

w = ML::GGUF::Qwen35Weights.from_gguf(model)

native_prefill = if native_prefill_cache
                   measure_native_prefill_cached(w, model, n_prompt, reps, warmup)
                 elsif native_prefill_prealloc
                   measure_native_prefill_preallocated(w, n_prompt, reps, warmup)
                 elsif native_prefill_prepare_state
                   measure_native_prefill_prepared_state(w, n_prompt, reps, warmup)
                 else
                   measure_native_prefill(w, n_prompt, reps, warmup)
                 end
native_decode = measure_native_decode(w, n_gen, reps, warmup, native_decode_top1)

llama_prefill = run_llama_bench(llama_bench, model, n_prompt, 0, reps, n_gpu_layers, threads, flash_attn)
llama_decode = run_llama_bench(llama_bench, model, 0, n_gen, reps, n_gpu_layers, threads, flash_attn)

puts "Qwen 3.5 9B benchmark vs llama.cpp"
puts "model: #{model}"
puts "llama-bench: #{llama_bench}"
native_prefill_mode = if native_prefill_cache
                        "prompt_cache_restore_after_seed"
                      elsif native_prefill_prealloc
                        "preallocated_state_chunked_prompt_plus_final_top1"
                      elsif native_prefill_prepare_state
                        "prepared_state_chunked_prompt_plus_final_top1"
                      else
                        "chunked_prompt_plus_final_top1"
                      end
puts "settings: prompt=#{n_prompt} gen=#{n_gen} reps=#{reps} warmup=#{warmup} ngl=#{n_gpu_layers} threads=#{threads} flash_attn=#{flash_attn} native_prefill=#{native_prefill_mode} native_decode=#{native_decode_top1 ? "top1" : "full_logits"}"
puts
puts "Prefill"
puts "  cogni-ml:  avg=#{native_prefill.avg_ms.round(2)} ms  p50=#{native_prefill.p50_ms.round(2)} ms  p95=#{native_prefill.p95_ms.round(2)} ms  avg=#{native_prefill.tok_s_avg.round(2)} tok/s  p50=#{native_prefill.tok_s_p50.round(2)} tok/s"
puts "  llama.cpp: avg=#{(llama_prefill.avg_ns / 1_000_000.0).round(2)} ms  avg=#{llama_prefill.avg_ts.round(2)} tok/s  stddev=#{llama_prefill.stddev_ts.round(2)} tok/s"
puts "  gap vs llama.cpp: #{pct_gap(native_prefill.tok_s_p50, llama_prefill.avg_ts).round(2)}%"
puts
puts "Decode"
puts "  cogni-ml:  avg=#{native_decode.avg_ms.round(2)} ms  p50=#{native_decode.p50_ms.round(2)} ms  p95=#{native_decode.p95_ms.round(2)} ms  avg=#{native_decode.tok_s_avg.round(2)} tok/s  p50=#{native_decode.tok_s_p50.round(2)} tok/s"
puts "  llama.cpp: avg=#{(llama_decode.avg_ns / 1_000_000.0).round(2)} ms  avg=#{llama_decode.avg_ts.round(2)} tok/s  stddev=#{llama_decode.stddev_ts.round(2)} tok/s"
puts "  gap vs llama.cpp: #{pct_gap(native_decode.tok_s_p50, llama_decode.avg_ts).round(2)}%"
