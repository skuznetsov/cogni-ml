require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/llm/llama"
require "../src/ml/qwen_vs_llama_benchmark_contract"

DEFAULT_MODEL  = (Path.home / ".cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf").to_s
FLASH_D256_ENV = "QWEN35_PREFILL_ATTN_FLASH_D256"

alias BenchmarkContract = ML::QwenVsLlamaBenchmarkContract

record TimedLogits, milliseconds : Float64, logits : Array(Float32)

record Summary,
  avg_ms : Float64,
  p50_ms : Float64,
  p95_ms : Float64,
  mean_ts : Float64

record Quality,
  cosine : Float64,
  native_top1 : Int32,
  native_top2 : Int32,
  llama_top1 : Int32,
  llama_top2 : Int32

def clear_float_buffer(buffer : ML::MetalBuffer?) : Nil
  return unless value = buffer

  value.contents.as(Pointer(UInt8)).clear(value.size)
end

def clear_float_array(values : Array(Float32)?) : Nil
  values.try(&.fill(0.0_f32))
end

def reset_native_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  raise "same-token prefill state unexpectedly owns adaptive KV" if state.adaptive_kv?

  state.layers.each do |layer|
    layer.position = 0
    # Prompt rows overwrite typed K/V before reading them. DeltaNet recurrence
    # is history-bearing and must be cleared before every timed repetition.
    clear_float_array(layer.conv_state)
    clear_float_array(layer.ssm_state)
    clear_float_buffer(layer.conv_state_buf)
    clear_float_buffer(layer.ssm_state_buf)
  end
end

class NativePrefillRunner
  getter output_width : Int32
  getter terminal_last_used : Bool

  def initialize(@weights : ML::GGUF::Qwen35Weights,
                 @tokens : Array(Int32),
                 kv_cache_f16 : Bool,
                 @flash_d256 : Bool = false)
    hp = @weights.hparams
    @output_width = @weights.output.out_dim
    @state = ML::GGUF::Qwen35CPU::State.new(
      hp,
      max_seq: @tokens.size.to_i32 + 4,
      kv_cache_f16: kv_cache_f16,
    )
    with_flash_d256 do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(
        @state,
        hp,
        clear: true,
        admit_adaptive_resident_kv: false,
      )
    end
    @terminal_last_used = false
  end

  def reset! : Nil
    reset_native_state!(@state)
  end

  def run : Array(Float32)
    route_used = [false]
    logits = with_flash_d256 do
      ML::GGUF::Qwen35CPU.prefill_tokens_logits(@weights, @tokens, 0, @state, route_used)
    end
    @terminal_last_used = route_used[0]
    raise "native full-logit width mismatch" unless logits.size == @output_width
    logits
  end

  private def with_flash_d256(&)
    old = ENV[FLASH_D256_ENV]?
    ENV[FLASH_D256_ENV] = @flash_d256 ? "1" : "0"
    yield
  ensure
    if old
      ENV[FLASH_D256_ENV] = old
    else
      ENV.delete(FLASH_D256_ENV)
    end
  end
end

class LlamaPrefillRunner
  getter output_width : Int32

  def initialize(@model : ML::LLM::Model,
                 @tokens : Array(Int32),
                 n_batch : Int32,
                 n_ubatch : Int32,
                 n_threads : Int32,
                 flash_attn : Bool,
                 cache_type : ML::LLM::LlamaFFI::GgmlType)
    raise "same-token prefill requires one llama logical batch" if @tokens.size > n_batch

    @output_width = @model.vocab_size
    @context = @model.create_context(
      n_ctx: @tokens.size.to_i32 + 4,
      n_batch: n_batch,
      n_ubatch: n_ubatch,
      n_threads: n_threads,
      flash_attn: flash_attn,
      cache_type_k: cache_type,
      cache_type_v: cache_type,
    )
    raise "same-token prefill requires one effective llama logical batch" if @tokens.size > @context.n_batch
  end

  def reset! : Nil
    @context.kv_clear
  end

  def run : Array(Float32)
    raise "llama.cpp prefill failed" unless @context.eval(@tokens)
    ML::LLM::LlamaFFI.llama_synchronize(@context.handle)
    raise "llama.cpp logical depth mismatch" unless @context.position == @tokens.size

    # Native full logits are materialized as a host Array inside its timed call.
    # Copy llama's final full-logit row inside the same timing boundary.
    logits = @context.get_logits.to_a
    raise "llama.cpp full-logit width mismatch" unless logits.size == @output_width
    logits
  end

  def close : Nil
    @context.free
  end
end

def timed(&block : -> Array(Float32)) : TimedLogits
  started = Time.instant
  logits = yield
  TimedLogits.new((Time.instant - started).total_milliseconds, logits)
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

def quality(native : Array(Float32), llama : Array(Float32)) : Quality
  native_top1, native_top2 = top2(native)
  llama_top1, llama_top2 = top2(llama)
  Quality.new(
    cosine(native, llama),
    native_top1,
    native_top2,
    llama_top1,
    llama_top2,
  )
end

def declaration(tokens : Array(Int32), output_width : Int32, warmup : Int32) : BenchmarkContract::PrefillWorkloadDeclaration
  BenchmarkContract::PrefillWorkloadDeclaration.new(
    tokens: tokens,
    initial_depth: 0,
    final_depth: tokens.size.to_i32,
    logical_prompts: 1,
    output_rows: 1,
    output_width: output_width,
    full_logits: true,
    synchronized: true,
    state_reused: true,
    setup_outside_timing: true,
    host_copy_inside_timing: true,
    warmup_runs: warmup,
  )
end

def measure_abba(native : NativePrefillRunner,
                 llama : LlamaPrefillRunner,
                 token_count : Int32,
                 reps : Int32) : {Summary, Summary, Quality}
  native_times = Array(Float64).new(reps)
  llama_times = Array(Float64).new(reps)
  last_quality = nil.as(Quality?)

  reps.times do |index|
    native.reset!
    llama.reset!
    native_result = nil.as(TimedLogits?)
    llama_result = nil.as(TimedLogits?)

    native_first = {true, false, false, true}[index % 4]
    if native_first
      native_result = timed { native.run }
      llama_result = timed { llama.run }
    else
      llama_result = timed { llama.run }
      native_result = timed { native.run }
    end

    native_value = native_result.not_nil!
    llama_value = llama_result.not_nil!
    native_times << native_value.milliseconds
    llama_times << llama_value.milliseconds
    measured_quality = quality(native_value.logits, llama_value.logits)
    if previous = last_quality
      unless previous.native_top1 == measured_quality.native_top1 &&
             previous.native_top2 == measured_quality.native_top2 &&
             previous.llama_top1 == measured_quality.llama_top1 &&
             previous.llama_top2 == measured_quality.llama_top2
        raise "top-2 output changed between measured repetitions"
      end
      last_quality = previous.copy_with(cosine: Math.min(previous.cosine, measured_quality.cosine))
    else
      last_quality = measured_quality
    end
  end

  {summarize(native_times, token_count), summarize(llama_times, token_count), last_quality.not_nil!}
end

def measure_native_route_abba(left : NativePrefillRunner,
                              right : NativePrefillRunner,
                              token_count : Int32,
                              reps : Int32) : {Summary, Summary, Quality}
  left_times = Array(Float64).new(reps)
  right_times = Array(Float64).new(reps)
  last_quality = nil.as(Quality?)

  reps.times do |index|
    left.reset!
    right.reset!
    left_result = nil.as(TimedLogits?)
    right_result = nil.as(TimedLogits?)

    left_first = {true, false, false, true}[index % 4]
    if left_first
      left_result = timed { left.run }
      right_result = timed { right.run }
    else
      right_result = timed { right.run }
      left_result = timed { left.run }
    end

    left_value = left_result.not_nil!
    right_value = right_result.not_nil!
    left_times << left_value.milliseconds
    right_times << right_value.milliseconds
    measured_quality = quality(left_value.logits, right_value.logits)
    if previous = last_quality
      unless previous.native_top1 == measured_quality.native_top1 &&
             previous.native_top2 == measured_quality.native_top2 &&
             previous.llama_top1 == measured_quality.llama_top1 &&
             previous.llama_top2 == measured_quality.llama_top2
        raise "F16/F32 top-2 output changed between measured repetitions"
      end
      last_quality = previous.copy_with(cosine: Math.min(previous.cosine, measured_quality.cosine))
    else
      last_quality = measured_quality
    end
  end

  {summarize(left_times, token_count), summarize(right_times, token_count), last_quality.not_nil!}
end

model_path = DEFAULT_MODEL
prompt_sizes = [256, 512, 1024, 2048]
reps = 4
warmup = 1
n_gpu_layers = 99
n_batch = 2048
n_ubatch = 512
n_threads = 8
flash_attn = false
llama_cache_type = ML::LLM::LlamaFFI::GgmlType::F16
native_cache_f16 = false
native_cache_ab = false
native_flash = false
native_flash_ab = false

OptionParser.parse do |parser|
  parser.banner = "Usage: benchmark_qwen_prefill_vs_llama_same_token [options]"
  parser.on("--model=PATH", "Path to a Qwen3.5 GGUF") { |value| model_path = value }
  parser.on("--prompts=LIST", "Comma-separated prompt sizes (default: 256,512,1024,2048)") do |value|
    prompt_sizes = value.split(',').map(&.to_i)
  end
  parser.on("--reps=N", "Measured ABBA repetitions; must be divisible by four (default: 4)") { |value| reps = value.to_i }
  parser.on("--warmup=N", "Warmups per engine and prompt size (default: 1)") { |value| warmup = value.to_i }
  parser.on("--ngl=N", "llama.cpp GPU layers (default: 99)") { |value| n_gpu_layers = value.to_i }
  parser.on("--n-batch=N", "llama.cpp logical batch size (default: 2048)") { |value| n_batch = value.to_i }
  parser.on("--n-ubatch=N", "llama.cpp physical microbatch size (default: 512)") { |value| n_ubatch = value.to_i }
  parser.on("--threads=N", "llama.cpp CPU threads (default: 8)") { |value| n_threads = value.to_i }
  parser.on("--flash-attn", "Enable llama.cpp flash attention") { flash_attn = true }
  parser.on("--native-kv=TYPE", "Native K/V cache type: f16 or f32 (default: f32)") do |value|
    native_cache_f16 = case value
                       when "f16" then true
                       when "f32" then false
                       else            raise "unsupported --native-kv type: #{value}"
                       end
  end
  parser.on("--native-kv-ab", "Also measure native F16 versus F32 KV in-process") { native_cache_ab = true }
  parser.on("--native-flash", "Use the admitted d256 Flash-MMA route for the native-vs-llama comparison") { native_flash = true }
  parser.on("--native-flash-ab", "Also measure opt-in d256 Flash-MMA versus the native baseline in-process") { native_flash_ab = true }
  parser.on("--llama-kv=TYPE", "llama.cpp K/V cache type: f16 or f32 (default: f16)") do |value|
    llama_cache_type = case value
                       when "f16" then ML::LLM::LlamaFFI::GgmlType::F16
                       when "f32" then ML::LLM::LlamaFFI::GgmlType::F32
                       else            raise "unsupported --llama-kv type: #{value}"
                       end
  end
  parser.on("-h", "--help", "Show help") do
    puts parser
    exit
  end
end

raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "prompt sizes must be positive" if prompt_sizes.empty? || prompt_sizes.any? { |size| size <= 0 }
raise "--reps must be positive and divisible by four" unless reps > 0 && reps % 4 == 0
raise "--warmup must be non-negative" unless warmup >= 0
raise "--n-batch must cover the largest prompt so llama emits one output row" unless n_batch >= prompt_sizes.max
raise "--n-ubatch must be positive and no larger than --n-batch" unless n_ubatch > 0 && n_ubatch <= n_batch
raise "--threads must be positive" unless n_threads > 0
raise "--native-kv-ab requires --native-kv=f16" if native_cache_ab && !native_cache_f16
raise "--native-flash requires --native-kv=f16" if native_flash && !native_cache_f16
raise "--native-flash-ab requires --native-kv=f16" if native_flash_ab && !native_cache_f16

native_weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
ML::LLM.init
llama_model = ML::LLM::Model.new(model_path, n_gpu_layers: n_gpu_layers)

begin
  native_vocab = native_weights.output.out_dim
  llama_vocab = llama_model.vocab_size
  raise "model vocabulary mismatch: native=#{native_vocab} llama=#{llama_vocab}" unless native_vocab == llama_vocab

  puts "Qwen same-token prefill external workload vs llama.cpp"
  puts "model: #{model_path}"
  native_cache_name = native_cache_f16 ? "f16" : "f32"
  puts "settings: prompts=#{prompt_sizes.join(',')} reps=#{reps} warmup=#{warmup} order=ABBA ngl=#{n_gpu_layers} n_batch=#{n_batch} n_ubatch=#{n_ubatch} threads=#{n_threads} flash_attn=#{flash_attn} output=one_terminal_full_logits_with_host_copy state=reused_cleared native_kv=#{native_cache_name} native_flash=#{native_flash} llama_kv=#{llama_cache_type.to_s.downcase}"
  puts
  puts "# pp  token_sha256  native_tok/s  llama_tok/s  gap  min_logits_cosine  native_top2  llama_top2  terminal_last  contract"
  puts "# native-kv-ab: pp f16_tok/s f32_tok/s f16_gain min_logits_cosine f16_top2 f32_top2" if native_cache_ab
  puts "# native-flash-ab: pp flash_tok/s baseline_tok/s flash_gain min_logits_cosine flash_top2 baseline_top2" if native_flash_ab

  prompt_sizes.each do |prompt_size|
    canonical = BenchmarkContract.synthetic_prefill_tokens(prompt_size.to_i32, native_vocab)
    native_tokens = canonical.dup
    llama_tokens = canonical.dup
    native_runner = NativePrefillRunner.new(native_weights, native_tokens, native_cache_f16, native_flash)
    native_f32_runner = NativePrefillRunner.new(native_weights, native_tokens, false) if native_cache_ab
    native_flash_runner = NativePrefillRunner.new(native_weights, native_tokens, true, true) if native_flash_ab
    native_flash_baseline_runner = NativePrefillRunner.new(native_weights, native_tokens, true, false) if native_flash_ab
    llama_runner = LlamaPrefillRunner.new(
      llama_model,
      llama_tokens,
      n_batch,
      n_ubatch,
      n_threads,
      flash_attn,
      llama_cache_type,
    )

    begin
      warmup.times do |index|
        if index.even?
          native_runner.reset!
          native_runner.run
          llama_runner.reset!
          llama_runner.run
        else
          llama_runner.reset!
          llama_runner.run
          native_runner.reset!
          native_runner.run
        end
      end

      native_declaration = declaration(native_tokens, native_runner.output_width, warmup)
      llama_declaration = declaration(llama_tokens, llama_runner.output_width, warmup)
      comparison = BenchmarkContract.same_token_prefill_comparison(native_declaration, llama_declaration)
      raise "same-token workload declaration rejected: #{comparison.reason}" unless comparison.level.same_token?

      native_stats, llama_stats, result_quality = measure_abba(
        native_runner,
        llama_runner,
        prompt_size.to_i32,
        reps,
      )
      gap = ((native_stats.mean_ts / llama_stats.mean_ts) - 1.0) * 100.0
      hash = BenchmarkContract.token_stream_sha256(native_tokens)
      puts "#{prompt_size.to_s.rjust(4)}  #{hash[0, 16]}  #{native_stats.mean_ts.round(2).to_s.rjust(12)}  #{llama_stats.mean_ts.round(2).to_s.rjust(11)}  #{gap.round(2).to_s.rjust(6)}%  #{result_quality.cosine.round(8)}  #{result_quality.native_top1}/#{result_quality.native_top2}  #{result_quality.llama_top1}/#{result_quality.llama_top2}  #{native_runner.terminal_last_used}  #{comparison.scope}"

      if f32_runner = native_f32_runner
        f32_runner.reset!
        f32_runner.run
        f16_stats, f32_stats, cache_quality = measure_native_route_abba(
          native_runner,
          f32_runner,
          prompt_size.to_i32,
          reps,
        )
        gain = ((f16_stats.mean_ts / f32_stats.mean_ts) - 1.0) * 100.0
        puts "native-kv-ab: #{prompt_size} #{f16_stats.mean_ts.round(2)} #{f32_stats.mean_ts.round(2)} #{gain.round(2)}% #{cache_quality.cosine.round(8)} #{cache_quality.native_top1}/#{cache_quality.native_top2} #{cache_quality.llama_top1}/#{cache_quality.llama_top2}"
      end

      if flash_runner = native_flash_runner
        flash_baseline = native_flash_baseline_runner.not_nil!
        warmup.times do
          flash_runner.reset!
          flash_runner.run
          flash_baseline.reset!
          flash_baseline.run
        end
        flash_stats, baseline_stats, flash_quality = measure_native_route_abba(
          flash_runner,
          flash_baseline,
          prompt_size.to_i32,
          reps,
        )
        gain = ((flash_stats.mean_ts / baseline_stats.mean_ts) - 1.0) * 100.0
        puts "native-flash-ab: #{prompt_size} #{flash_stats.mean_ts.round(2)} #{baseline_stats.mean_ts.round(2)} #{gain.round(2)}% #{flash_quality.cosine.round(8)} #{flash_quality.native_top1}/#{flash_quality.native_top2} #{flash_quality.llama_top1}/#{flash_quality.llama_top2}"
      end
    ensure
      llama_runner.close
    end
  end
ensure
  llama_model.free
  ML::LLM.cleanup
end
