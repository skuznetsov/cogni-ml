require "json"
require "option_parser"
require "../src/ml/bench_load_guard"

MODEL_PATH = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"
LLAMA_BENCH = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-bench"
NATIVE_PROFILE_BIN = "/tmp/gemma4_metal_decode_profile_bench"

record NativeStats,
  prefill_ms : Float64,
  prefill_tok_s : Float64,
  decode_ms_per_tok : Float64,
  decode_tok_s : Float64

record LlamaStats,
  avg_ts : Float64,
  stddev_ts : Float64,
  avg_ns : Int64

def run_checked(cmd : String, args : Array(String), env : Hash(String, String)? = nil) : String
  output = IO::Memory.new
  error = IO::Memory.new
  status = Process.run(cmd, args: args, env: env, output: output, error: error)
  unless status.success?
    raise "#{cmd} failed: #{error}\nargs=#{args.join(" ")}"
  end
  output.to_s
end

def build_native_profile!(bin : String) : Nil
  args = [
    "build",
    "bin/gemma4_metal_decode_profile.cr",
    "-o", bin,
    "--error-trace",
    "--link-flags=#{Dir.current}/build/bridge.o -framework Metal -framework Foundation -framework MetalPerformanceShaders -lc++",
  ]
  run_checked("/opt/homebrew/bin/crystal", args, {"CRYSTAL_CACHE_DIR" => "/tmp/cogni_ml_gemma4_bench_profile_build"})
end

def synthetic_tokens(n : Int32) : String
  Array(String).new(n) { |i| (42 + (i % 4096)).to_s }.join(",")
end

def parse_native(output : String, prompt_tokens : Int32) : NativeStats
  prefill_ms = nil.as(Float64?)
  prefill_tok_s = nil.as(Float64?)
  decode_ms_per_tok = nil.as(Float64?)

  output.each_line do |line|
    if m = line.match(/prefill_p50_ms=([0-9.]+).*prefill_p50_tok_s=([0-9.]+)/)
      prefill_ms = m[1].to_f
      prefill_tok_s = m[2].to_f
    elsif m = line.match(/^decode_ms_per_token_p50=([0-9.]+)/)
      decode_ms_per_tok = m[1].to_f
    end
  end

  raise "native profile output missing prefill metrics" unless prefill_ms && prefill_tok_s
  raise "native profile output missing decode metrics" unless decode_ms_per_tok
  NativeStats.new(
    prefill_ms: prefill_ms.not_nil!,
    prefill_tok_s: prefill_tok_s.not_nil!,
    decode_ms_per_tok: decode_ms_per_tok.not_nil!,
    decode_tok_s: 1000.0 / decode_ms_per_tok.not_nil!,
  )
end

def run_native(profile_bin : String,
               model : String,
               prompt_tokens : Int32,
               gen_tokens : Int32,
               reps : Int32,
               warmups : Int32,
               mode : String,
               prefill_chunk : Int32,
               decode_wave : Bool,
               top1_resident : Bool,
               top1_chain : Int32,
               decode_only_seed : Int32? = nil) : NativeStats
  args = [
    "--model", model,
    "--tokens", synthetic_tokens(prompt_tokens),
    "--generate", gen_tokens.to_s,
    "--max-seq", Math.max((decode_only_seed ? gen_tokens : prompt_tokens + gen_tokens) + 8, prompt_tokens * 2).to_s,
    "--runs", reps.to_s,
    "--warmups", warmups.to_s,
    "--prefill-mode", "rows",
    "--prefill-chunk", prefill_chunk.to_s,
  ]
  if mode == "body"
    args << "--body-only"
    args << "--prefill-no-head"
  end
  args << "--decode-layerwise" unless decode_wave
  args << "--top1-wave-resident" if top1_resident
  if mode == "top1" && top1_chain > 1
    args << "--top1-chain"
    args << top1_chain.to_s
  end
  if seed = decode_only_seed
    args << "--decode-only-seed"
    args << seed.to_s
  end
  env = ENV.to_h
  env["GEMMA4_ROW_PREFILL_ALLOW_GEMM"] = "1"
  output = run_checked(profile_bin, args, env)
  parse_native(output, prompt_tokens)
end

def run_llama_bench(llama_bench : String,
                    model : String,
                    n_prompt : Int32,
                    n_gen : Int32,
                    reps : Int32,
                    n_gpu_layers : Int32,
                    threads : Int32,
                    flash_attn : Bool,
                    extra_args : Array(String)) : LlamaStats
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
  args.concat(extra_args)
  parsed = JSON.parse(run_checked(llama_bench, args))
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
native_bin = NATIVE_PROFILE_BIN
prompt_tokens = 256
gen_tokens = 64
reps = 3
warmups = 1
n_gpu_layers = 99
threads = 8
flash_attn = true
build_native = false
mode = "body"
prefill_chunk = 0
decode_wave = true
top1_resident = true
top1_chain = 1
decode_only_seed = 11751
llama_extra_args = [] of String
load_warning_threshold = 50.0
load_total_warning_threshold = 100.0
wait_quiet_ms = 0
quiet_poll_ms = 1000
require_quiet = false

OptionParser.parse do |p|
  p.banner = "Usage: benchmark_gemma4_vs_llama [options]"
  p.on("--model=PATH", "Path to Gemma4 GGUF") { |v| model = v }
  p.on("--llama-bench=PATH", "Path to llama-bench") { |v| llama_bench = v }
  p.on("--native-bin=PATH", "Path to built gemma4_metal_decode_profile binary") { |v| native_bin = v }
  p.on("--build-native", "Build the native profile binary before running") { build_native = true }
  p.on("--prompt=N", "Prompt tokens for prefill benchmark") { |v| prompt_tokens = v.to_i }
  p.on("--gen=N", "Generated tokens for decode benchmark") { |v| gen_tokens = v.to_i }
  p.on("--reps=N", "Repetitions") { |v| reps = v.to_i }
  p.on("--warmups=N", "Native warmup repetitions") { |v| warmups = v.to_i }
  p.on("--ngl=N", "llama.cpp GPU layers") { |v| n_gpu_layers = v.to_i }
  p.on("--threads=N", "llama.cpp CPU threads") { |v| threads = v.to_i }
  p.on("--flash-attn=BOOL", "llama.cpp flash attention true/false") { |v| flash_attn = v.downcase.in?({"1", "true", "yes", "on"}) }
  p.on("--llama-extra-arg=ARG", "Append raw llama-bench argument; repeat for flag/value") { |v| llama_extra_args << v }
  p.on("--native-mode=MODE", "Native decode mode: body or top1") { |v| mode = v }
  p.on("--native-decode-only-seed=N", "Native tg seed token for llama-bench-compatible empty-KV decode") { |v| decode_only_seed = v.to_i }
  p.on("--prefill-chunk=N", "Native row prefill chunk size, default prompt length") { |v| prefill_chunk = v.to_i }
  p.on("--decode-layerwise", "Disable native decode wave") { decode_wave = false }
  p.on("--top1-wave-resident", "Use native resident top1 wave in top1 mode (default)") { top1_resident = true }
  p.on("--no-top1-wave-resident", "Use legacy hidden-readback + separate top1 head path") { top1_resident = false }
  p.on("--native-top1-chain=N", "Use exact GPU-resident native top1 chain chunks in native top1 mode") { |v| top1_chain = v.to_i }
  p.on("--load-warning-threshold=PCT", "Warn if a process exceeds PCT CPU") { |v| load_warning_threshold = v.to_f }
  p.on("--load-total-warning-threshold=PCT", "Warn if total observed CPU exceeds PCT") { |v| load_total_warning_threshold = v.to_f }
  p.on("--wait-quiet-ms=N", "Wait up to N ms for quiet host") { |v| wait_quiet_ms = v.to_i }
  p.on("--quiet-poll-ms=N", "Quiet polling interval") { |v| quiet_poll_ms = v.to_i }
  p.on("--require-quiet", "Abort if host is still busy") { require_quiet = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "llama-bench not found: #{llama_bench}" unless File.exists?(llama_bench)
raise "prompt/gen/reps/warmups must be positive" unless prompt_tokens > 0 && gen_tokens > 0 && reps > 0 && warmups >= 0
raise "--native-mode must be body or top1" unless {"body", "top1"}.includes?(mode)
raise "--native-top1-chain must be positive" unless top1_chain > 0
prefill_chunk = prompt_tokens if prefill_chunk <= 0

build_native_profile!(native_bin) if build_native
raise "native profile binary not found: #{native_bin}" unless File.exists?(native_bin)

ML::BenchLoadGuard.wait_until_quiet!(load_warning_threshold, load_total_warning_threshold, wait_quiet_ms, quiet_poll_ms)
if require_quiet
  ML::BenchLoadGuard.require_quiet!(load_warning_threshold, load_total_warning_threshold)
else
  ML::BenchLoadGuard.warn_if_busy(load_warning_threshold, load_total_warning_threshold)
end

native_prefill = run_native(native_bin, model, prompt_tokens, 1, reps, warmups, "body", prefill_chunk, decode_wave, top1_resident, 1)
native_decode = run_native(native_bin, model, 1, gen_tokens, reps, warmups, mode, prefill_chunk, decode_wave, top1_resident, top1_chain, decode_only_seed)
llama_prefill = run_llama_bench(llama_bench, model, prompt_tokens, 0, reps, n_gpu_layers, threads, flash_attn, llama_extra_args)
llama_decode = run_llama_bench(llama_bench, model, 0, gen_tokens, reps, n_gpu_layers, threads, flash_attn, llama_extra_args)

puts "Gemma4 benchmark vs llama.cpp"
puts "model: #{model}"
puts "settings: prompt=#{prompt_tokens} gen=#{gen_tokens} reps=#{reps} warmups=#{warmups} native_mode=#{mode} prefill_chunk=#{prefill_chunk} decode_wave=#{decode_wave} top1_resident=#{top1_resident} native_top1_chain=#{top1_chain} native_decode_only_seed=#{decode_only_seed} ngl=#{n_gpu_layers} threads=#{threads} flash_attn=#{flash_attn} llama_extra_args=#{llama_extra_args.inspect}"
puts "note: native pp is measured body-only; native tg is measured decode-only with empty KV to match llama-bench tg semantics."
puts
puts "Prefill"
puts "  cogni-ml:  p50=#{native_prefill.prefill_ms.round(2)} ms  p50=#{native_prefill.prefill_tok_s.round(2)} tok/s"
puts "  llama.cpp: avg=#{(llama_prefill.avg_ns / 1_000_000.0).round(2)} ms  avg=#{llama_prefill.avg_ts.round(2)} tok/s  stddev=#{llama_prefill.stddev_ts.round(2)} tok/s"
puts "  gap vs llama.cpp: #{pct_gap(native_prefill.prefill_tok_s, llama_prefill.avg_ts).round(2)}%"
puts
puts "Decode"
puts "  cogni-ml:  p50=#{native_decode.decode_ms_per_tok.round(2)} ms/tok  p50=#{native_decode.decode_tok_s.round(2)} tok/s"
puts "  llama.cpp: avg=#{(llama_decode.avg_ns / 1_000_000.0).round(2)} ms  avg=#{llama_decode.avg_ts.round(2)} tok/s  stddev=#{llama_decode.stddev_ts.round(2)} tok/s"
puts "  gap vs llama.cpp: #{pct_gap(native_decode.decode_tok_s, llama_decode.avg_ts).round(2)}%"
