#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/qwen_vs_llama_benchmark_contract"

FLASH_ENV     = "QWEN35_PREFILL_ATTN_FLASH_D256"
DEFAULT_MODEL = (Path.home / ".cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf").to_s

record Sample, milliseconds : Float64, logits : Array(Float32)

private def with_flash(enabled : Bool, &)
  old = ENV[FLASH_ENV]?
  ENV[FLASH_ENV] = enabled ? "1" : "0"
  yield
ensure
  if old
    ENV[FLASH_ENV] = old
  else
    ENV.delete(FLASH_ENV)
  end
end

private def reset_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  state.layers.each do |layer|
    layer.position = 0
    layer.conv_state.try(&.fill(0.0_f32))
    layer.ssm_state.try(&.fill(0.0_f32))
    layer.conv_state_buf.try { |buffer| buffer.contents.as(Pointer(UInt8)).clear(buffer.size) }
    layer.ssm_state_buf.try { |buffer| buffer.contents.as(Pointer(UInt8)).clear(buffer.size) }
  end
end

private def run(weights : ML::GGUF::Qwen35Weights,
                tokens : Array(Int32),
                state : ML::GGUF::Qwen35CPU::State,
                flash : Bool) : Sample
  reset_state!(state)
  started = Time.instant
  logits = with_flash(flash) do
    ML::GGUF::Qwen35CPU.prefill_tokens_logits(weights, tokens, 0, state)
  end
  Sample.new((Time.instant - started).total_milliseconds, logits)
end

private def top2(logits : Array(Float32)) : {Int32, Int32}
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

private def cosine(a : Array(Float32), b : Array(Float32)) : Float64
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
  dot / Math.sqrt(aa * bb)
end

private def max_abs_error(a : Array(Float32), b : Array(Float32)) : Float64
  raise "logit width mismatch" unless a.size == b.size
  a.each_with_index.max_of { |value, index| (value.to_f64 - b[index].to_f64).abs }
end

private def assert_flash_route!(weights : ML::GGUF::Qwen35Weights,
                                tokens : Array(Int32),
                                state : ML::GGUF::Qwen35CPU::State) : Nil
  profile = ML::GGUF::Qwen35Metal::Profile
  profile.reset
  profile.enable!
  run(weights, tokens, state, true)
  flash_hits = profile.route_count("prefill_attn_flash_d256")
  raise "Flash-MMA route was not selected" unless flash_hits > 0

  profile.reset
  run(weights, tokens, state, false)
  baseline_hits = profile.route_count("prefill_attn_flash_d256")
  raise "baseline unexpectedly selected Flash-MMA" unless baseline_hits == 0
ensure
  ML::GGUF::Qwen35Metal::Profile.disable!
  ML::GGUF::Qwen35Metal::Profile.reset
end

private def summarize(milliseconds : Array(Float64), tokens : Int32)
  sorted = milliseconds.sort
  avg = milliseconds.sum / milliseconds.size
  middle = sorted.size // 2
  p50 = sorted.size.odd? ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0
  throughput = tokens.to_f64 * 1000.0 / avg
  {avg: avg, p50: p50, throughput: throughput}
end

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  middle = sorted.size // 2
  sorted.size.odd? ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2.0
end

model_path = DEFAULT_MODEL
prompt_sizes = [1024, 2048]
warmup = 1
reps = 4

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_prefill_flash_d256_abba [options]"
  parser.on("--model=PATH", "Qwen3.5/Qwen3.8 GGUF") { |value| model_path = value }
  parser.on("--prompts=LIST", "Comma-separated prompt sizes") { |value| prompt_sizes = value.split(',').map(&.to_i) }
  parser.on("--warmup=N", "Warmup pairs") { |value| warmup = value.to_i }
  parser.on("--reps=N", "Measured repetitions, divisible by four") { |value| reps = value.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "prompt sizes must be positive" if prompt_sizes.empty? || prompt_sizes.any? { |size| size <= 0 }
raise "warmup must be positive so lazy pipeline compilation stays outside measurements" unless warmup > 0
raise "reps must be positive and divisible by four" unless reps > 0 && reps % 4 == 0

weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
begin
  hp = weights.hparams
  puts "Qwen d256 Flash-MMA native ABBA"
  puts "model=#{model_path} device=#{ML::Metal::Device.instance.name.inspect} heads=#{hp.n_head}/#{hp.n_head_kv} head_dim=#{hp.head_dim} prompts=#{prompt_sizes.join(',')} warmup=#{warmup} reps=#{reps}"
  puts "# pp flash_tok/s baseline_tok/s mean_gain paired_median_gain flash_p50_ms baseline_p50_ms min_cosine max_abs flash_top2 baseline_top2"

  prompt_sizes.each do |prompt_size|
    tokens = ML::QwenVsLlamaBenchmarkContract.synthetic_prefill_tokens(
      prompt_size.to_i32, weights.output.out_dim,
    )
    state = ML::GGUF::Qwen35CPU::State.new(
      hp, max_seq: prompt_size.to_i32 + 4, kv_cache_f16: true)
    with_flash(true) do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(
        state, hp, clear: true, admit_adaptive_resident_kv: false)
    end
    assert_flash_route!(weights, tokens, state)

    warmup.times do |index|
      if index.even?
        run(weights, tokens, state, true)
        run(weights, tokens, state, false)
      else
        run(weights, tokens, state, false)
        run(weights, tokens, state, true)
      end
    end

    flash_ms = Array(Float64).new(reps)
    baseline_ms = Array(Float64).new(reps)
    min_cosine = 1.0_f64
    max_abs = 0.0_f64
    flash_top2 = {-1_i32, -1_i32}
    baseline_top2 = {-1_i32, -1_i32}
    reps.times do |index|
      flash = nil.as(Sample?)
      baseline = nil.as(Sample?)
      if {true, false, false, true}[index % 4]
        flash = run(weights, tokens, state, true)
        baseline = run(weights, tokens, state, false)
      else
        baseline = run(weights, tokens, state, false)
        flash = run(weights, tokens, state, true)
      end
      f = flash.not_nil!
      b = baseline.not_nil!
      flash_ms << f.milliseconds
      baseline_ms << b.milliseconds
      min_cosine = Math.min(min_cosine, cosine(f.logits, b.logits))
      max_abs = Math.max(max_abs, max_abs_error(f.logits, b.logits))
      current_flash_top2 = top2(f.logits)
      current_baseline_top2 = top2(b.logits)
      raise "Flash-MMA changed top-2" unless current_flash_top2 == current_baseline_top2
      if flash_top2[0] >= 0
        raise "top-2 changed across repetitions" unless flash_top2 == current_flash_top2
      end
      flash_top2 = current_flash_top2
      baseline_top2 = current_baseline_top2
    end

    flash_stats = summarize(flash_ms, prompt_size.to_i32)
    baseline_stats = summarize(baseline_ms, prompt_size.to_i32)
    gain = (flash_stats[:throughput] / baseline_stats[:throughput] - 1.0) * 100.0
    paired_gains = flash_ms.zip(baseline_ms).map do |flash_value, baseline_value|
      (baseline_value / flash_value - 1.0) * 100.0
    end
    paired_median_gain = median(paired_gains)
    raise "Flash-MMA logit cosine below 0.9999" unless min_cosine >= 0.9999
    puts "#{prompt_size} #{flash_stats[:throughput].round(2)} #{baseline_stats[:throughput].round(2)} #{gain.round(2)}% #{paired_median_gain.round(2)}% #{flash_stats[:p50].round(3)} #{baseline_stats[:p50].round(3)} #{min_cosine.round(8)} #{max_abs.round(8)} #{flash_top2[0]}/#{flash_top2[1]} #{baseline_top2[0]}/#{baseline_top2[1]}"
  end
ensure
  weights.close
end
