require "json"
require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/llm/llama"
require "../src/ml/qwen_vs_llama_benchmark_contract"

DEFAULT_MODEL       = (Path.home / ".cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf").to_s
WORKER_SCHEMA       = "cogni-ml/qwen-decode-split-worker-v1"
SPLIT_PROCESS_ORDER = ["native", "llama", "llama", "native"]

alias BenchmarkContract = ML::QwenVsLlamaBenchmarkContract

record TimedRows, milliseconds : Float64, logits : Array(Array(Float32))

record Summary,
  avg_ms : Float64,
  p50_ms : Float64,
  p95_ms : Float64,
  mean_ts : Float64

record WorkerMeasurement,
  samples_ms : Array(Float64),
  step_top1 : Array(Int32),
  step_top2 : Array(Int32),
  final_logits : Array(Float32)

class WorkerResult
  include JSON::Serializable

  getter schema : String
  getter engine : String
  getter seed_count : Int32
  getter forced_count : Int32
  getter seed_sha256 : String
  getter forced_sha256 : String
  getter samples_ms : Array(Float64)
  getter output_width : Int32
  getter step_top1 : Array(Int32)
  getter step_top2 : Array(Int32)
  getter final_logits : Array(Float32)
  getter effective_n_batch : Int32?
  getter effective_n_ubatch : Int32?

  def initialize(@engine : String,
                 @seed_count : Int32,
                 @forced_count : Int32,
                 @seed_sha256 : String,
                 @forced_sha256 : String,
                 @samples_ms : Array(Float64),
                 @output_width : Int32,
                 @step_top1 : Array(Int32),
                 @step_top2 : Array(Int32),
                 @final_logits : Array(Float32),
                 @effective_n_batch : Int32?,
                 @effective_n_ubatch : Int32?,
                 @schema : String = WORKER_SCHEMA)
  end
end

def clear_float_buffer(buffer : ML::MetalBuffer?) : Nil
  buffer.try { |value| value.contents.as(Pointer(UInt8)).clear(value.size) }
end

def reset_native_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  raise "same-token decode state unexpectedly owns adaptive KV" if state.adaptive_kv?

  state.layers.each do |layer|
    layer.position = 0
    layer.conv_state.try(&.fill(0.0_f32))
    layer.ssm_state.try(&.fill(0.0_f32))
    clear_float_buffer(layer.conv_state_buf)
    clear_float_buffer(layer.ssm_state_buf)
  end
end

class NativeDecodeRunner
  getter output_width : Int32

  def initialize(@weights : ML::GGUF::Qwen35Weights,
                 @seed_tokens : Array(Int32),
                 @forced_tokens : Array(Int32),
                 kv_cache_f16 : Bool)
    @output_width = @weights.output.out_dim
    @state = ML::GGUF::Qwen35CPU::State.new(
      @weights.hparams,
      max_seq: (@seed_tokens.size + @forced_tokens.size + 4).to_i32,
      kv_cache_f16: kv_cache_f16,
    )
    @closed = false
    begin
      ML::GGUF::Qwen35CPU.prepare_state_metal!(
        @state,
        @weights.hparams,
        clear: true,
        admit_adaptive_resident_kv: false,
      )
    rescue ex
      close
      raise ex
    end
  end

  def reset_and_seed! : Nil
    reset_native_state!(@state)
    ML::GGUF::Qwen35CPU.prefill_tokens(@weights, @seed_tokens, 0, @state)
  end

  def run : Array(Array(Float32))
    rows = Array(Array(Float32)).new(@forced_tokens.size)
    @forced_tokens.each_with_index do |token, index|
      logits = ML::GGUF::Qwen35CPU.forward(
        @weights,
        token,
        (@seed_tokens.size + index).to_i32,
        @state,
      )
      raise "native full-logit width mismatch" unless logits.size == @output_width
      rows << logits
    end
    rows
  end

  def close : Nil
    return if @closed

    ML::GGUF::Qwen35CPU.release_state_metal!(@state)
    @closed = true
  end
end

class LlamaDecodeRunner
  getter output_width : Int32

  def effective_n_batch : Int32
    @context.n_batch.to_i32
  end

  def effective_n_ubatch : Int32
    @context.n_ubatch.to_i32
  end

  def initialize(@model : ML::LLM::Model,
                 @seed_tokens : Array(Int32),
                 @forced_tokens : Array(Int32),
                 n_batch : Int32,
                 n_ubatch : Int32,
                 n_threads : Int32,
                 flash_attn : Bool,
                 cache_type : ML::LLM::LlamaFFI::GgmlType)
    raise "same-token decode requires one llama logical seed batch" if @seed_tokens.size > n_batch

    @output_width = @model.vocab_size
    @context = @model.create_context(
      n_ctx: (@seed_tokens.size + @forced_tokens.size + 4).to_i32,
      n_batch: n_batch,
      n_ubatch: n_ubatch,
      n_threads: n_threads,
      flash_attn: flash_attn,
      cache_type_k: cache_type,
      cache_type_v: cache_type,
    )
    begin
      raise "same-token decode requires one effective llama logical seed batch" if @seed_tokens.size > @context.n_batch
    rescue ex
      @context.free
      raise ex
    end
  end

  def reset_and_seed! : Nil
    @context.kv_clear(data: false)
    raise "llama.cpp seed prefill failed" unless @context.eval(@seed_tokens)
    # llama_decode is asynchronous. Fence the seeded history before the timed
    # forced-token loop, matching llama-bench's restored-state timing boundary.
    @context.get_logits
    raise "llama.cpp seed depth mismatch" unless @context.position == @seed_tokens.size
  end

  def run : Array(Array(Float32))
    rows = Array(Array(Float32)).new(@forced_tokens.size)
    @forced_tokens.each do |token|
      raise "llama.cpp forced decode failed" unless @context.eval([token])
      # llama_get_logits_ith performs the synchronization. Copy one complete
      # output row inside the timed boundary, matching the native Array result.
      logits = @context.get_logits.to_a
      raise "llama.cpp full-logit width mismatch" unless logits.size == @output_width
      rows << logits
    end
    expected = @seed_tokens.size + @forced_tokens.size
    raise "llama.cpp final decode depth mismatch" unless @context.position == expected
    rows
  end

  def close : Nil
    @context.free
  end
end

def top2(logits : Array(Float32)) : {Int32, Int32}
  best_id = -1
  second_id = -1
  best = -Float32::INFINITY
  second = -Float32::INFINITY
  logits.each_with_index do |value, index|
    raise "non-finite logit at #{index}" unless value.finite?
    if value > best
      second = best
      second_id = best_id
      best = value
      best_id = index
    elsif value > second
      second = value
      second_id = index
    end
  end
  {best_id.to_i32, second_id.to_i32}
end

def cosine(a : Array(Float32), b : Array(Float32)) : Float64
  raise "logit width mismatch" unless a.size == b.size

  dot = 0.0_f64
  aa = 0.0_f64
  bb = 0.0_f64
  a.each_with_index do |av, index|
    bv = b[index]
    raise "non-finite logit at #{index}" unless av.finite? && bv.finite?
    af = av.to_f64
    bf = bv.to_f64
    dot += af * bf
    aa += af * af
    bb += bf * bf
  end
  raise "zero-norm logit vector" unless aa > 0.0 && bb > 0.0
  dot / Math.sqrt(aa * bb)
end

def timed(&block : -> Array(Array(Float32))) : TimedRows
  started = Time.instant
  rows = yield
  TimedRows.new((Time.instant - started).total_milliseconds, rows)
end

def percentile(sorted : Array(Float64), pct : Int32) : Float64
  sorted[(sorted.size * pct // 100).clamp(0, sorted.size - 1)]
end

def summarize(samples : Array(Float64), token_count : Int32) : Summary
  sorted = samples.sort
  Summary.new(
    avg_ms: samples.sum / samples.size,
    p50_ms: percentile(sorted, 50),
    p95_ms: percentile(sorted, 95),
    mean_ts: BenchmarkContract.mean_throughput(samples, token_count),
  )
end

def measure(runner : NativeDecodeRunner | LlamaDecodeRunner,
            warmup : Int32,
            reps : Int32) : WorkerMeasurement
  warmup.times do
    runner.reset_and_seed!
    runner.run
  end

  samples = Array(Float64).new(reps)
  expected_top1 = nil.as(Array(Int32)?)
  expected_top2 = nil.as(Array(Int32)?)
  final_logits = nil.as(Array(Float32)?)
  reps.times do
    runner.reset_and_seed!
    result = timed { runner.run }
    raise "decode row count mismatch" unless result.logits.size > 0
    pairs = result.logits.map { |row| top2(row) }
    top1_ids = pairs.map(&.[0])
    top2_ids = pairs.map(&.[1])
    if prior = expected_top1
      raise "decode top-1 changed between repetitions" unless prior == top1_ids
      raise "decode top-2 changed between repetitions" unless expected_top2 == top2_ids
    else
      expected_top1 = top1_ids
      expected_top2 = top2_ids
    end
    samples << result.milliseconds
    final_logits = result.logits.last
  end

  WorkerMeasurement.new(samples, expected_top1.not_nil!, expected_top2.not_nil!, final_logits.not_nil!)
end

def declaration(seed_tokens : Array(Int32),
                forced_tokens : Array(Int32),
                output_width : Int32,
                warmup : Int32) : BenchmarkContract::DecodeWorkloadDeclaration
  BenchmarkContract::DecodeWorkloadDeclaration.new(
    seed_tokens: seed_tokens,
    forced_tokens: forced_tokens,
    initial_depth: seed_tokens.size.to_i32,
    final_depth: (seed_tokens.size + forced_tokens.size).to_i32,
    logical_sequences: 1,
    output_rows_per_token: 1,
    output_width: output_width,
    full_logits: true,
    synchronized_per_token: true,
    state_seeded_outside_timing: true,
    host_copy_per_token_inside_timing: true,
    warmup_runs: warmup,
  )
end

def write_result(path : String, result : WorkerResult) : Nil
  temporary = "#{path}.tmp.#{Process.pid}"
  begin
    File.open(temporary, "w") { |file| result.to_json(file) }
    File.rename(temporary, path)
  ensure
    File.delete(temporary) if File.exists?(temporary)
  end
end

def run_worker!(engine : String,
                result_path : String,
                model_path : String,
                seed_count : Int32,
                forced_count : Int32,
                reps : Int32,
                warmup : Int32,
                n_gpu_layers : Int32,
                n_batch : Int32,
                n_ubatch : Int32,
                n_threads : Int32,
                flash_attn : Bool,
                native_cache_f16 : Bool,
                llama_cache_type : ML::LLM::LlamaFFI::GgmlType) : Nil
  model : ML::LLM::Model? = nil
  weights : ML::GGUF::Qwen35Weights? = nil
  runner : NativeDecodeRunner | LlamaDecodeRunner | Nil = nil
  backend_initialized = false
  begin
    output_width = case engine
                   when "native"
                     weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
                     weights.not_nil!.output.out_dim
                   when "llama"
                     ML::LLM.init
                     backend_initialized = true
                     model = ML::LLM::Model.new(model_path, n_gpu_layers: n_gpu_layers)
                     model.not_nil!.vocab_size
                   else
                     raise "unsupported worker engine: #{engine}"
                   end
    seed_tokens = BenchmarkContract.synthetic_prefill_tokens(seed_count, output_width)
    forced_tokens = BenchmarkContract.synthetic_decode_tokens(forced_count, output_width)

    runner = if engine == "native"
               NativeDecodeRunner.new(weights.not_nil!, seed_tokens, forced_tokens, native_cache_f16)
             else
               LlamaDecodeRunner.new(
                 model.not_nil!, seed_tokens, forced_tokens,
                 n_batch, n_ubatch, n_threads, flash_attn, llama_cache_type,
               )
             end
    measured = measure(runner.not_nil!, warmup, reps)
    effective_n_batch = runner.is_a?(LlamaDecodeRunner) ? runner.effective_n_batch : nil
    effective_n_ubatch = runner.is_a?(LlamaDecodeRunner) ? runner.effective_n_ubatch : nil
    write_result(result_path, WorkerResult.new(
      engine,
      seed_count,
      forced_count,
      BenchmarkContract.token_stream_sha256(seed_tokens),
      BenchmarkContract.token_stream_sha256(forced_tokens),
      measured.samples_ms,
      output_width,
      measured.step_top1,
      measured.step_top2,
      measured.final_logits,
      effective_n_batch,
      effective_n_ubatch,
    ))
  ensure
    runner.try(&.close)
    weights.try(&.close)
    model.try(&.free)
    ML::LLM.cleanup if backend_initialized
  end
end

def worker_args(engine : String,
                result_path : String,
                model_path : String,
                seed_count : Int32,
                forced_count : Int32,
                reps : Int32,
                warmup : Int32,
                n_gpu_layers : Int32,
                n_batch : Int32,
                n_ubatch : Int32,
                n_threads : Int32,
                flash_attn : Bool,
                native_cache_f16 : Bool,
                llama_cache_type : String) : Array(String)
  args = [
    "--worker=#{engine}", "--worker-result=#{result_path}",
    "--model=#{model_path}", "--depth=#{seed_count}", "--gen=#{forced_count}",
    "--reps=#{reps}", "--warmup=#{warmup}", "--ngl=#{n_gpu_layers}",
    "--n-batch=#{n_batch}", "--n-ubatch=#{n_ubatch}", "--threads=#{n_threads}",
    "--native-kv=#{native_cache_f16 ? "f16" : "f32"}", "--llama-kv=#{llama_cache_type}",
  ]
  args << "--flash-attn" if flash_attn
  args
end

def validate_result!(result : WorkerResult,
                     engine : String,
                     seed_count : Int32,
                     forced_count : Int32,
                     reps : Int32,
                     expected_n_batch : Int32,
                     expected_n_ubatch : Int32) : Nil
  raise "worker schema mismatch" unless result.schema == WORKER_SCHEMA
  raise "worker engine mismatch" unless result.engine == engine
  raise "worker shape mismatch" unless result.seed_count == seed_count && result.forced_count == forced_count
  raise "worker sample count mismatch" unless result.samples_ms.size == reps
  raise "worker timing invalid" unless result.samples_ms.all? { |sample| sample > 0.0 && sample.finite? }
  raise "worker top-1 count mismatch" unless result.step_top1.size == forced_count
  raise "worker top-2 count mismatch" unless result.step_top2.size == forced_count
  raise "worker full-logit width mismatch" unless result.output_width > 0 && result.final_logits.size == result.output_width
  seed_tokens = BenchmarkContract.synthetic_prefill_tokens(seed_count, result.output_width)
  forced_tokens = BenchmarkContract.synthetic_decode_tokens(forced_count, result.output_width)
  raise "worker seed stream hash mismatch" unless result.seed_sha256 == BenchmarkContract.token_stream_sha256(seed_tokens)
  raise "worker forced stream hash mismatch" unless result.forced_sha256 == BenchmarkContract.token_stream_sha256(forced_tokens)
  top2(result.final_logits)
  if engine == "llama"
    raise "llama batch geometry missing" unless result.effective_n_batch && result.effective_n_ubatch
    effective_batch = result.effective_n_batch.not_nil!
    effective_ubatch = result.effective_n_ubatch.not_nil!
    unless effective_batch >= seed_count && effective_batch <= expected_n_batch
      raise "llama effective logical batch does not cover the seeded history"
    end
    unless effective_ubatch > 0 && effective_ubatch <= effective_batch && effective_ubatch <= expected_n_ubatch
      raise "llama effective microbatch exceeds its declared bounds"
    end
  elsif result.effective_n_batch || result.effective_n_ubatch
    raise "native batch geometry unexpectedly present"
  end
end

model_path = DEFAULT_MODEL
seed_count = 256
forced_count = 16
reps = 4
warmup = 1
n_gpu_layers = 99
n_batch = 2048
n_ubatch = 512
n_threads = 8
flash_attn = false
native_cache_f16 = true
llama_cache_type = ML::LLM::LlamaFFI::GgmlType::F16
worker_engine : String? = nil
worker_result_path : String? = nil
llama_first = false

OptionParser.parse do |parser|
  parser.banner = "Usage: benchmark_qwen_decode_vs_llama_same_token [options]"
  parser.on("--model=PATH", "Path to a Qwen GGUF") { |value| model_path = value }
  parser.on("--depth=N", "Seeded context tokens outside timing (default: 256)") { |value| seed_count = value.to_i }
  parser.on("--gen=N", "Forced decode tokens inside timing (default: 16)") { |value| forced_count = value.to_i }
  parser.on("--reps=N", "Measured repetitions; divisible by four (default: 4)") { |value| reps = value.to_i }
  parser.on("--warmup=N", "Warmups per worker (default: 1)") { |value| warmup = value.to_i }
  parser.on("--ngl=N", "llama.cpp GPU layers (default: 99)") { |value| n_gpu_layers = value.to_i }
  parser.on("--n-batch=N", "llama.cpp logical batch size (default: 2048)") { |value| n_batch = value.to_i }
  parser.on("--n-ubatch=N", "llama.cpp physical microbatch size (default: 512)") { |value| n_ubatch = value.to_i }
  parser.on("--threads=N", "llama.cpp CPU threads (default: 8)") { |value| n_threads = value.to_i }
  parser.on("--flash-attn", "Enable llama.cpp flash attention") { flash_attn = true }
  parser.on("--llama-first", "Reverse the balanced worker order to llama-native-native-llama") { llama_first = true }
  parser.on("--native-kv=TYPE", "Native K/V type: f16 or f32 (default: f16)") do |value|
    native_cache_f16 = case value
                       when "f16" then true
                       when "f32" then false
                       else            raise "unsupported --native-kv type: #{value}"
                       end
  end
  parser.on("--llama-kv=TYPE", "llama.cpp K/V type: f16 or f32 (default: f16)") do |value|
    llama_cache_type = case value
                       when "f16" then ML::LLM::LlamaFFI::GgmlType::F16
                       when "f32" then ML::LLM::LlamaFFI::GgmlType::F32
                       else            raise "unsupported --llama-kv type: #{value}"
                       end
  end
  parser.on("--worker=ENGINE", "Internal worker: native or llama") { |value| worker_engine = value }
  parser.on("--worker-result=PATH", "Internal worker result path") { |value| worker_result_path = value }
  parser.on("-h", "--help", "Show help") do
    puts parser
    exit
  end
end

raise "model not found: #{model_path}" unless File.exists?(model_path)
model_path = File.realpath(model_path)
raise "--depth must be positive" unless seed_count > 0
raise "--gen must be positive" unless forced_count > 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "--n-batch must cover the seed" unless n_batch >= seed_count
raise "--n-ubatch must be positive and no larger than --n-batch" unless n_ubatch > 0 && n_ubatch <= n_batch
raise "--threads must be positive" unless n_threads > 0

if worker = worker_engine
  raise "--worker must be native or llama" unless {"native", "llama"}.includes?(worker)
  raise "--worker-result is required" unless worker_result_path
  raise "--worker reps must be positive" unless reps > 0
  run_worker!(
    worker, worker_result_path.not_nil!, model_path, seed_count, forced_count,
    reps, warmup, n_gpu_layers, n_batch, n_ubatch, n_threads, flash_attn,
    native_cache_f16, llama_cache_type,
  )
  exit
end

raise "--worker-result requires --worker" if worker_result_path
raise "--reps must be positive and divisible by four" unless reps > 0 && reps % 4 == 0

worker_executable = Process.executable_path || raise "cannot resolve worker executable"
model_identity = File.info(model_path)
worker_reps = reps // 2
result_paths = [] of String
results = [] of WorkerResult
llama_cache_name = llama_cache_type == ML::LLM::LlamaFFI::GgmlType::F16 ? "f16" : "f32"
process_order = llama_first ? ["llama", "native", "native", "llama"] : SPLIT_PROCESS_ORDER

begin
  process_order.each_with_index do |engine, index|
    current_identity = File.info(model_path)
    unless model_identity.same_file?(current_identity) &&
           model_identity.size == current_identity.size &&
           model_identity.modification_time == current_identity.modification_time
      raise "model identity changed between workers"
    end

    result_path = File.tempname("qwen-decode-#{engine}-#{index}", ".json")
    result_paths << result_path
    args = worker_args(
      engine, result_path, model_path, seed_count, forced_count, worker_reps,
      warmup, n_gpu_layers, n_batch, n_ubatch, n_threads, flash_attn,
      native_cache_f16, llama_cache_name,
    )
    status = Process.run(worker_executable, args, output: STDOUT, error: STDERR)
    raise "worker #{engine}[#{index}] failed: #{status.exit_reason}" unless status.success?
    current_identity = File.info(model_path)
    unless model_identity.same_file?(current_identity) &&
           model_identity.size == current_identity.size &&
           model_identity.modification_time == current_identity.modification_time
      raise "model identity changed while worker was running"
    end
    raise "worker #{engine}[#{index}] produced no result" unless File.file?(result_path)
    result = WorkerResult.from_json(File.read(result_path))
    validate_result!(result, engine, seed_count, forced_count, worker_reps, n_batch, n_ubatch)
    results << result
  end

  native_results = results.select { |result| result.engine == "native" }
  llama_results = results.select { |result| result.engine == "llama" }
  raise "split order did not yield two results per engine" unless native_results.size == 2 && llama_results.size == 2
  widths = results.map(&.output_width).uniq
  seed_hashes = results.map(&.seed_sha256).uniq
  forced_hashes = results.map(&.forced_sha256).uniq
  raise "worker output widths differ" unless widths.size == 1
  raise "worker seed streams differ" unless seed_hashes.size == 1
  raise "worker forced streams differ" unless forced_hashes.size == 1
  raise "native top-1 changed across processes" unless native_results.map(&.step_top1).uniq.size == 1
  raise "native top-2 changed across processes" unless native_results.map(&.step_top2).uniq.size == 1
  raise "llama top-1 changed across processes" unless llama_results.map(&.step_top1).uniq.size == 1
  raise "llama top-2 changed across processes" unless llama_results.map(&.step_top2).uniq.size == 1
  raise "llama effective logical batch changed across processes" unless llama_results.map(&.effective_n_batch).uniq.size == 1
  raise "llama effective microbatch changed across processes" unless llama_results.map(&.effective_n_ubatch).uniq.size == 1

  seed_tokens = BenchmarkContract.synthetic_prefill_tokens(seed_count, widths[0])
  forced_tokens = BenchmarkContract.synthetic_decode_tokens(forced_count, widths[0])
  comparison = BenchmarkContract.same_token_decode_comparison(
    declaration(seed_tokens, forced_tokens, widths[0], warmup),
    declaration(seed_tokens.dup, forced_tokens.dup, widths[0], warmup),
  )
  raise "same-token decode declaration rejected: #{comparison.reason}" unless comparison.level.same_token?

  native_stats = summarize(native_results.flat_map(&.samples_ms), forced_count)
  llama_stats = summarize(llama_results.flat_map(&.samples_ms), forced_count)
  gap = ((native_stats.mean_ts / llama_stats.mean_ts) - 1.0) * 100.0
  native_top1 = native_results[0].step_top1
  native_top2 = native_results[0].step_top2
  llama_top1 = llama_results[0].step_top1
  llama_top2 = llama_results[0].step_top2
  top1_matches = native_top1.zip(llama_top1).count { |pair| pair[0] == pair[1] }
  top2_matches = native_top1.zip(native_top2, llama_top1, llama_top2).count do |row|
    row[0] == row[2] && row[1] == row[3]
  end
  min_final_cosine = native_results.flat_map do |native|
    llama_results.map { |llama| cosine(native.final_logits, llama.final_logits) }
  end.min

  native_cache_name = native_cache_f16 ? "f16" : "f32"
  llama_batch = llama_results[0].effective_n_batch.not_nil!
  llama_ubatch = llama_results[0].effective_n_ubatch.not_nil!
  puts "Qwen same-token forced decode external workload vs llama.cpp"
  puts "model: #{model_path}"
  puts "settings: depth=#{seed_count} gen=#{forced_count} reps=#{reps} warmup=#{warmup} order=process-#{process_order.join('-')} output=one_full_logits_row_with_host_copy_per_token state=seeded_outside_timing native_kv=#{native_cache_name} llama_kv=#{llama_cache_name} llama_batch=#{llama_batch}/#{llama_ubatch} flash_attn=#{flash_attn} threads=#{n_threads} ngl=#{n_gpu_layers}"
  results.each_with_index do |result, index|
    puts "worker_samples_ms[#{index}]: engine=#{result.engine} values=#{result.samples_ms.to_json}"
  end
  puts "seed_sha256: #{seed_hashes[0]}"
  puts "forced_sha256: #{forced_hashes[0]}"
  puts "native: mean=#{native_stats.mean_ts.round(2)} tok/s p50=#{(forced_count * 1000.0 / native_stats.p50_ms).round(2)} tok/s avg=#{native_stats.avg_ms.round(2)} ms"
  puts "llama:  mean=#{llama_stats.mean_ts.round(2)} tok/s p50=#{(forced_count * 1000.0 / llama_stats.p50_ms).round(2)} tok/s avg=#{llama_stats.avg_ms.round(2)} ms"
  puts "gap: #{gap.round(2)}%"
  puts "quality: top1=#{top1_matches}/#{forced_count} top2_pair=#{top2_matches}/#{forced_count} final_logits_cosine=#{min_final_cosine.round(8)}"
  puts "contract: #{comparison.scope}"
ensure
  result_paths.each { |path| File.delete(path) if File.exists?(path) }
end
