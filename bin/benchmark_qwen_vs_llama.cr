require "json"
require "option_parser"
require "../src/ml/bench_load_guard"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/qwen_vs_llama_benchmark_contract"

MODEL_PATH      = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
LLAMA_BENCH     = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-bench"
BENCHMARK_MODES = {"both", "native", "llama"}

alias BenchmarkContract = ML::QwenVsLlamaBenchmarkContract
alias BenchmarkHeadMode = ML::QwenVsLlamaBenchmarkContract::HeadMode

record NativeStats,
  avg_ms : Float64,
  p50_ms : Float64,
  p95_ms : Float64,
  tok_s_p50 : Float64,
  mean_ts : Float64

record LlamaStats,
  avg_ts : Float64,
  stddev_ts : Float64,
  avg_ns : Int64,
  n_depth : Int32

def percentile(sorted : Array(Float64), pct : Int32) : Float64
  idx = (sorted.size * pct // 100).clamp(0, sorted.size - 1)
  sorted[idx]
end

def native_stats(times : Array(Float64), n_tokens : Int32) : NativeStats
  sorted = times.sort
  avg_ms = times.sum / times.size
  p50_ms = percentile(sorted, 50)
  NativeStats.new(
    avg_ms: avg_ms,
    p50_ms: p50_ms,
    p95_ms: percentile(sorted, 95),
    tok_s_p50: (n_tokens * 1000.0) / p50_ms,
    mean_ts: BenchmarkContract.mean_throughput(times, n_tokens),
  )
end

def measure_native_prefill(w : ML::GGUF::Qwen35Weights, n_prompt : Int32, reps : Int32, warmup : Int32,
                           head_mode : BenchmarkHeadMode = BenchmarkContract::DEFAULT_PREFILL_HEAD) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }

  warmup.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    run_native_prefill(w, prompt, state, head_mode)
  end

  times = Array(Float64).new(reps)
  reps.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    t0 = Time.instant
    run_native_prefill(w, prompt, state, head_mode)
    times << (Time.instant - t0).total_milliseconds
  end

  native_stats(times, n_prompt)
end

def measure_native_prefill_prepared_state(w : ML::GGUF::Qwen35Weights, n_prompt : Int32, reps : Int32, warmup : Int32,
                                          head_mode : BenchmarkHeadMode = BenchmarkContract::DEFAULT_PREFILL_HEAD) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }

  warmup.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    run_native_prefill(w, prompt, state, head_mode)
  end

  times = Array(Float64).new(reps)
  reps.times do
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    t0 = Time.instant
    run_native_prefill(w, prompt, state, head_mode)
    times << (Time.instant - t0).total_milliseconds
  end

  native_stats(times, n_prompt)
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

def measure_native_prefill_preallocated(w : ML::GGUF::Qwen35Weights, n_prompt : Int32, reps : Int32, warmup : Int32,
                                        head_mode : BenchmarkHeadMode = BenchmarkContract::DEFAULT_PREFILL_HEAD) : NativeStats
  hp = w.hparams
  prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)

  # Allocate the backing GPU state buffers once, outside the timed section.
  run_native_prefill(w, prompt, state, head_mode)
  reset_prefill_state!(state)

  warmup.times do
    run_native_prefill(w, prompt, state, head_mode)
    reset_prefill_state!(state)
  end

  times = Array(Float64).new(reps)
  reps.times do
    t0 = Time.instant
    run_native_prefill(w, prompt, state, head_mode)
    times << (Time.instant - t0).total_milliseconds
    reset_prefill_state!(state)
  end

  native_stats(times, n_prompt)
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
    run_native_prefill(w, prompt, seeded, BenchmarkHeadMode::DecoderBodyLowerBound)
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

    native_stats(times, n_prompt)
  ensure
    FileUtils.rm_rf(root) if Dir.exists?(root)
  end
end

def measure_native_prefill_cached_prefix(w : ML::GGUF::Qwen35Weights,
                                         model : String,
                                         n_prompt : Int32,
                                         suffix_tokens : Int32,
                                         reps : Int32,
                                         warmup : Int32) : NativeStats
  raise "--native-prefill-cache-prefix-suffix must be positive" unless suffix_tokens > 0
  raise "--native-prefill-cache-prefix-suffix must be smaller than --prompt" unless suffix_tokens < n_prompt

  hp = w.hparams
  full_prompt = Array(Int32).new(n_prompt) { |i| ((i * 7 + 11) % 1000).to_i32 }
  prefix_len = n_prompt - suffix_tokens
  prefix_prompt = full_prompt[0, prefix_len]
  root = File.tempname("qwen35-bench-prompt-cache-prefix")
  Dir.mkdir_p(root)

  begin
    store = ML::GGUF::Qwen35PromptCache::Store.new(root)
    seeded = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_prompt + 4)
    run_native_prefill(w, prefix_prompt, seeded, BenchmarkHeadMode::DecoderBodyLowerBound)
    model_id = ML::GGUF::Qwen35PromptCache.short_hash("bench-model\0#{model}")
    tokenizer_id = "synthetic-token-ids-v1"
    entry = store.save(
      session_id: "benchmark-prefix",
      model_id: model_id,
      tokenizer_id: tokenizer_id,
      prompt_text: "",
      token_ids: prefix_prompt,
      state: seeded,
    )

    warmup.times do
      replay = store.restore_and_replay_suffix(entry, w, full_prompt)
      raise "prompt-cache prefix replay layer mismatch" unless replay.state.layers.size == hp.n_layer
      raise "prompt-cache prefix replay suffix mismatch" unless replay.replayed_tokens == suffix_tokens
      raise "prompt-cache prefix replay missing next token" unless replay.next_token_id
    end

    times = Array(Float64).new(reps)
    reps.times do
      t0 = Time.instant
      replay = store.restore_and_replay_suffix(entry, w, full_prompt)
      raise "prompt-cache prefix replay layer mismatch" unless replay.state.layers.size == hp.n_layer
      raise "prompt-cache prefix replay suffix mismatch" unless replay.replayed_tokens == suffix_tokens
      raise "prompt-cache prefix replay missing next token" unless replay.next_token_id
      times << (Time.instant - t0).total_milliseconds
    end

    native_stats(times, n_prompt)
  ensure
    FileUtils.rm_rf(root) if Dir.exists?(root)
  end
end

def run_native_prefill(w : ML::GGUF::Qwen35Weights,
                       prompt : Array(Int32),
                       state : ML::GGUF::Qwen35CPU::State,
                       head_mode : BenchmarkHeadMode) : Nil
  return if prompt.empty?

  case head_mode
  in .decoder_body_lower_bound?
    ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, state)
  in .fused_top1?
    ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, state)
  in .full_logits?
    if prompt.size > 1
      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt[0...-1], 0, state)
    end
    ML::GGUF::Qwen35CPU.forward(w, prompt[-1], prompt.size.to_i32 - 1, state)
  end
end

def forward_decode_token(w : ML::GGUF::Qwen35Weights, tok : Int32, pos : Int32,
                         state : ML::GGUF::Qwen35CPU::State, mode : BenchmarkHeadMode) : Nil
  case mode
  in .decoder_body_lower_bound?
    ML::GGUF::Qwen35CPU.prefill_token(w, tok, pos, state)
  in .fused_top1?
    ML::GGUF::Qwen35CPU.forward_top1(w, tok, pos, state)
  in .full_logits?
    ML::GGUF::Qwen35CPU.forward(w, tok, pos, state)
  end
end

def run_native_decode_iteration(w : ML::GGUF::Qwen35Weights,
                                n_gen : Int32,
                                n_depth : Int32,
                                mode : BenchmarkHeadMode) : Float64
  hp = w.hparams
  decode_tokens = Array(Int32).new(n_gen) { |i| ((i * 13 + 11751) % 32000).to_i32 }
  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: n_depth + n_gen + 4)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
  next_token = decode_tokens[0]

  if n_depth > 0
    prompt = Array(Int32).new(n_depth) { |i| ((i * 7 + 11) % 1000).to_i32 }
    if mode.fused_top1?
      next_token, _ = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, prompt, 0, state)
    else
      ML::GGUF::Qwen35CPU.prefill_tokens(w, prompt, 0, state)
    end
  end

  t0 = Time.instant
  if mode.fused_top1?
    n_gen.times do |i|
      next_token, _ = ML::GGUF::Qwen35CPU.forward_top1(w, next_token, n_depth + i, state)
    end
  else
    decode_tokens.each_with_index do |tok, i|
      forward_decode_token(w, tok, n_depth + i, state, mode)
    end
  end
  (Time.instant - t0).total_milliseconds
end

def measure_native_decode(w : ML::GGUF::Qwen35Weights,
                          n_gen : Int32,
                          n_depth : Int32,
                          reps : Int32,
                          warmup : Int32,
                          mode : BenchmarkHeadMode) : NativeStats
  warmup.times { run_native_decode_iteration(w, n_gen, n_depth, mode) }

  times = Array(Float64).new(reps)
  reps.times do
    times << run_native_decode_iteration(w, n_gen, n_depth, mode)
  end

  native_stats(times, n_gen)
end

def run_llama_bench(llama_bench : String,
                    model : String,
                    n_prompt : Int32,
                    n_gen : Int32,
                    n_depth : Int32,
                    reps : Int32,
                    n_gpu_layers : Int32,
                    threads : Int32,
                    flash_attn : Bool,
                    cache_type_k : String?,
                    cache_type_v : String?,
                    extra_args : Array(String)) : LlamaStats
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
  args.concat(BenchmarkContract.llama_depth_args(n_depth))
  if ctk = cache_type_k
    args << "-ctk" << ctk
  end
  if ctv = cache_type_v
    args << "-ctv" << ctv
  end
  args.concat(extra_args)
  status = Process.run(llama_bench, args: args, output: output, error: error)
  unless status.success?
    raise "llama-bench failed: #{error.to_s}\nargs=#{args.join(" ")}"
  end

  parsed = JSON.parse(output.to_s)
  rows = parsed.as_a
  row = rows.first
  actual_prompt = row["n_prompt"].as_i.to_i32
  actual_gen = row["n_gen"].as_i.to_i32
  actual_depth = row["n_depth"].as_i.to_i32
  BenchmarkContract.validate_llama_result_shape!(
    rows.size.to_i32,
    actual_prompt,
    actual_gen,
    actual_depth,
    n_prompt,
    n_gen,
    n_depth,
  )
  LlamaStats.new(
    avg_ts: row["avg_ts"].as_f,
    stddev_ts: row["stddev_ts"].as_f,
    avg_ns: row["avg_ns"].as_i64,
    n_depth: actual_depth,
  )
end

def pct_gap(native : Float64, llama : Float64) : Float64
  ((native / llama) - 1.0) * 100.0
end

model = MODEL_PATH
llama_bench = LLAMA_BENCH
n_prompt = 64
n_gen = 64
decode_depth = 0
reps = 5
warmup = 1
n_gpu_layers = 99
threads = 8
flash_attn = false
llama_cache_type_k = nil.as(String?)
llama_cache_type_v = nil.as(String?)
llama_extra_args = [] of String
native_decode_mode = BenchmarkContract::DEFAULT_DECODE_HEAD
native_prefill_cache = false
native_prefill_cache_prefix_suffix = 0
native_prefill_prealloc = false
native_prefill_prepare_state = true
native_prefill_head_mode = BenchmarkContract::DEFAULT_PREFILL_HEAD
load_warning_threshold = 50.0
load_total_warning_threshold = 100.0
wait_quiet_ms = 0
quiet_poll_ms = 1000
require_quiet = false
benchmark_mode = "both"

OptionParser.parse do |p|
  p.banner = "Usage: benchmark_qwen_vs_llama [options]"
  p.on("--model=PATH", "Path to Qwen GGUF") { |v| model = v }
  p.on("--llama-bench=PATH", "Path to llama-bench binary") { |v| llama_bench = v }
  p.on("--prompt=N", "Prompt tokens for prefill benchmark (default: 64)") { |v| n_prompt = v.to_i }
  p.on("--gen=N", "Generated tokens for decode benchmark (default: 64)") { |v| n_gen = v.to_i }
  p.on("--decode-depth=N", "Seed N KV/recurrent positions before timed decode; passed to llama-bench -d (default: 0)") { |v| decode_depth = v.to_i }
  p.on("--reps=N", "Repetitions (default: 5)") { |v| reps = v.to_i }
  p.on("--warmup=N", "Warmup repetitions for native path (default: 1; llama-bench has one built-in warmup)") { |v| warmup = v.to_i }
  p.on("--ngl=N", "llama.cpp GPU layers (default: 99)") { |v| n_gpu_layers = v.to_i }
  p.on("--threads=N", "llama.cpp CPU threads (default: 8)") { |v| threads = v.to_i }
  p.on("--flash-attn", "Enable flash attention in llama.cpp") { flash_attn = true }
  p.on("--llama-cache-k=TYPE", "llama.cpp KV cache K type for llama-bench, for example q8_0") { |v| llama_cache_type_k = v }
  p.on("--llama-cache-v=TYPE", "llama.cpp KV cache V type for llama-bench, for example q4_0") { |v| llama_cache_type_v = v }
  p.on("--llama-extra-arg=ARG", "Append one raw argument to llama-bench; repeat for flag/value pairs") { |v| llama_extra_args << v }
  p.on("--native-decode-body-only", "Measure a native decoder-body lower bound; not comparable to unmodified llama-bench") { native_decode_mode = BenchmarkHeadMode::DecoderBodyLowerBound }
  p.on("--native-decode-top1", "Measure product greedy decode with fused top1; reported separately from llama-bench") { native_decode_mode = BenchmarkHeadMode::FusedTop1 }
  p.on("--native-decode-full-logits", "Measure native decode with full lm-head logits (default)") { native_decode_mode = BenchmarkHeadMode::FullLogits }
  p.on("--native-full-logits", "Alias for --native-decode-full-logits") { native_decode_mode = BenchmarkHeadMode::FullLogits }
  p.on("--native-prefill-body-only", "Measure a native prompt-body lower bound; not comparable to unmodified llama-bench") { native_prefill_head_mode = BenchmarkHeadMode::DecoderBodyLowerBound }
  p.on("--native-prefill-final-top1", "Measure product prompt processing plus final output-head top1") { native_prefill_head_mode = BenchmarkHeadMode::FusedTop1 }
  p.on("--native-prefill-full-logits", "Measure prompt processing plus final full logits (default)") { native_prefill_head_mode = BenchmarkHeadMode::FullLogits }
  p.on("--native-prefill-cache", "Measure native prefill as exact prompt-cache restore after one seeded run") { native_prefill_cache = true }
  p.on("--native-prefill-cache-prefix-suffix=N", "Measure prompt-cache prefix restore plus exact replay of the last N prompt tokens") { |v| native_prefill_cache = true; native_prefill_cache_prefix_suffix = v.to_i }
  p.on("--native-prefill-prealloc", "Measure native prefill with state buffers allocated outside the timed loop") { native_prefill_prealloc = true }
  p.on("--native-prefill-prepare-state", "Prepare a fresh state's Metal buffers before each timed native prefill") { native_prefill_prepare_state = true }
  p.on("--native-prefill-first-touch", "Include fresh Metal state allocation in native prefill timing; diagnostic only") { native_prefill_prepare_state = false }
  p.on("--load-warning-threshold=PCT", "Warn if another process uses at least PCT CPU before benchmarking (default: 50, 0 disables)") { |v| load_warning_threshold = v.to_f }
  p.on("--load-total-warning-threshold=PCT", "Warn if total observed process CPU exceeds PCT before benchmarking (default: 100, 0 disables)") { |v| load_total_warning_threshold = v.to_f }
  p.on("--wait-quiet-ms=N", "Wait up to N ms for host load to fall below benchmark thresholds before measuring") { |v| wait_quiet_ms = v.to_i }
  p.on("--quiet-poll-ms=N", "Polling interval for --wait-quiet-ms (default: 1000)") { |v| quiet_poll_ms = v.to_i }
  p.on("--require-quiet", "Abort instead of warning when host CPU load exceeds process or total thresholds") { require_quiet = true }
  p.on("--mode=MODE", "Benchmark mode: both, native, or llama (default: both)") { |v| benchmark_mode = v }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--mode must be one of #{BENCHMARK_MODES.join(", ")}" unless BENCHMARK_MODES.includes?(benchmark_mode)
raise "--prompt must be positive" unless n_prompt > 0
raise "--gen must be positive" unless n_gen > 0
raise "--decode-depth must be non-negative" unless decode_depth >= 0
raise "--reps must be positive" unless reps > 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "Model not found: #{model}" unless File.exists?(model)
raise "llama-bench not found: #{llama_bench}" if benchmark_mode != "native" && !File.exists?(llama_bench)
raise "--wait-quiet-ms must be non-negative" unless wait_quiet_ms >= 0
raise "--quiet-poll-ms must be positive" unless quiet_poll_ms > 0

ML::BenchLoadGuard.wait_until_quiet!(load_warning_threshold, load_total_warning_threshold, wait_quiet_ms, quiet_poll_ms)
if require_quiet
  ML::BenchLoadGuard.require_quiet!(load_warning_threshold, load_total_warning_threshold)
else
  ML::BenchLoadGuard.warn_if_busy(load_warning_threshold, load_total_warning_threshold)
end

native_prefill = nil.as(NativeStats?)
native_decode = nil.as(NativeStats?)
llama_prefill = nil.as(LlamaStats?)
llama_decode = nil.as(LlamaStats?)

if benchmark_mode != "llama"
  w = ML::GGUF::Qwen35Weights.from_gguf(model)

  native_prefill = if native_prefill_cache && native_prefill_cache_prefix_suffix > 0
                     measure_native_prefill_cached_prefix(w, model, n_prompt, native_prefill_cache_prefix_suffix, reps, warmup)
                   elsif native_prefill_cache
                     measure_native_prefill_cached(w, model, n_prompt, reps, warmup)
                   elsif native_prefill_prealloc
                     measure_native_prefill_preallocated(w, n_prompt, reps, warmup, native_prefill_head_mode)
                   elsif native_prefill_prepare_state
                     measure_native_prefill_prepared_state(w, n_prompt, reps, warmup, native_prefill_head_mode)
                   else
                     measure_native_prefill(w, n_prompt, reps, warmup, native_prefill_head_mode)
                   end
  native_decode = measure_native_decode(w, n_gen, decode_depth, reps, warmup, native_decode_mode)
end

if benchmark_mode != "native"
  llama_prefill = run_llama_bench(llama_bench, model, n_prompt, 0, 0, reps, n_gpu_layers, threads, flash_attn, llama_cache_type_k, llama_cache_type_v, llama_extra_args)
  llama_decode = run_llama_bench(llama_bench, model, 0, n_gen, decode_depth, reps, n_gpu_layers, threads, flash_attn, llama_cache_type_k, llama_cache_type_v, llama_extra_args)
end

puts "Qwen benchmark vs llama.cpp"
puts "model: #{model}"
puts "llama-bench: #{llama_bench}"
native_prefill_mode = if native_prefill_cache && native_prefill_cache_prefix_suffix > 0
                        "prompt_cache_prefix_restore_suffix#{native_prefill_cache_prefix_suffix}"
                      elsif native_prefill_cache
                        "prompt_cache_restore_after_seed"
                      elsif native_prefill_prealloc
                        "preallocated_state_#{BenchmarkContract.prefill_label(native_prefill_head_mode)}"
                      elsif native_prefill_prepare_state
                        "prepared_state_#{BenchmarkContract.prefill_label(native_prefill_head_mode)}"
                      else
                        "first_touch_#{BenchmarkContract.prefill_label(native_prefill_head_mode)}"
                      end
native_decode_label = BenchmarkContract.decode_label(native_decode_mode)
prefill_comparison = BenchmarkContract.prefill_comparison(native_prefill_head_mode, cached: native_prefill_cache)
decode_comparison = BenchmarkContract.decode_comparison(native_decode_mode)
puts "settings: mode=#{benchmark_mode} prompt=#{n_prompt} gen=#{n_gen} decode_depth=#{decode_depth} reps=#{reps} warmup=#{warmup} ngl=#{n_gpu_layers} threads=#{threads} flash_attn=#{flash_attn} llama_cache_k=#{llama_cache_type_k || "default"} llama_cache_v=#{llama_cache_type_v || "default"} llama_extra_args=#{llama_extra_args.inspect} native_prefill=#{native_prefill_mode} native_decode=#{native_decode_label}"
puts
puts "Prefill"
if np = native_prefill
  puts "  cogni-ml:  avg=#{np.avg_ms.round(2)} ms  p50=#{np.p50_ms.round(2)} ms  p95=#{np.p95_ms.round(2)} ms  mean=#{np.mean_ts.round(2)} tok/s  p50=#{np.tok_s_p50.round(2)} tok/s"
else
  puts "  cogni-ml:  skipped"
end
if lp = llama_prefill
  puts "  llama.cpp: avg=#{(lp.avg_ns / 1_000_000.0).round(2)} ms  avg=#{lp.avg_ts.round(2)} tok/s  stddev=#{lp.stddev_ts.round(2)} tok/s"
else
  puts "  llama.cpp: skipped"
end
if np = native_prefill
  if lp = llama_prefill
    case prefill_comparison.level
    in .strict?
      puts "  comparison: #{prefill_comparison.scope} (#{prefill_comparison.reason})"
      puts "  mean gap vs llama.cpp avg_ts: #{pct_gap(np.mean_ts, lp.avg_ts).round(2)}%"
    in .diagnostic?
      puts "  comparison: diagnostic only — #{prefill_comparison.scope} (#{prefill_comparison.reason})"
      puts "  diagnostic mean delta vs llama.cpp avg_ts: #{pct_gap(np.mean_ts, lp.avg_ts).round(2)}%"
    in .incomparable?
      puts "  comparison: not comparable (#{prefill_comparison.reason})"
    end
  end
end
puts
puts "Decode"
if nd = native_decode
  puts "  cogni-ml:  avg=#{nd.avg_ms.round(2)} ms  p50=#{nd.p50_ms.round(2)} ms  p95=#{nd.p95_ms.round(2)} ms  mean=#{nd.mean_ts.round(2)} tok/s  p50=#{nd.tok_s_p50.round(2)} tok/s"
else
  puts "  cogni-ml:  skipped"
end
if ld = llama_decode
  puts "  llama.cpp: avg=#{(ld.avg_ns / 1_000_000.0).round(2)} ms  avg=#{ld.avg_ts.round(2)} tok/s  stddev=#{ld.stddev_ts.round(2)} tok/s"
else
  puts "  llama.cpp: skipped"
end
if nd = native_decode
  if ld = llama_decode
    case decode_comparison.level
    in .strict?
      puts "  comparison: #{decode_comparison.scope} depth=#{ld.n_depth} (#{decode_comparison.reason})"
      puts "  mean gap vs llama.cpp avg_ts: #{pct_gap(nd.mean_ts, ld.avg_ts).round(2)}%"
    in .diagnostic?
      puts "  comparison: diagnostic only — #{decode_comparison.scope} depth=#{ld.n_depth} (#{decode_comparison.reason})"
      puts "  diagnostic mean delta vs llama.cpp avg_ts: #{pct_gap(nd.mean_ts, ld.avg_ts).round(2)}%"
    in .incomparable?
      puts "  comparison: not comparable (#{decode_comparison.reason})"
    end
  end
end
