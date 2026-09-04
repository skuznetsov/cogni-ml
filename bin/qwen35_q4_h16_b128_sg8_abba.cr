#!/usr/bin/env crystal

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/qwen_vs_llama_benchmark_contract"

ENV_KEY = "QWEN35_Q4K_H16_B128_SG8"

record Sample, milliseconds : Float64, logits : Array(Float32)

private def with_mode(value : String?, &)
  old = ENV[ENV_KEY]?
  if value
    ENV[ENV_KEY] = value
  else
    ENV.delete(ENV_KEY)
  end
  yield
ensure
  if old
    ENV[ENV_KEY] = old
  else
    ENV.delete(ENV_KEY)
  end
end

private def reset_state!(state : ML::GGUF::Qwen35CPU::State) : Nil
  state.layers.each do |layer|
    layer.position = 0
    layer.conv_state.try(&.fill(0.0_f32))
    layer.ssm_state.try(&.fill(0.0_f32))
    layer.conv_state_buf.try { |buf| buf.contents.as(Pointer(UInt8)).clear(buf.size) }
    layer.ssm_state_buf.try { |buf| buf.contents.as(Pointer(UInt8)).clear(buf.size) }
  end
end

private def run(weights, tokens, state, mode : String?) : Sample
  reset_state!(state)
  started = Time.instant
  logits = with_mode(mode) do
    ML::GGUF::Qwen35CPU.prefill_tokens_logits(weights, tokens, 0, state)
  end
  Sample.new((Time.instant - started).total_milliseconds, logits)
end

private def top2(logits : Array(Float32)) : {Int32, Int32}
  best_id = -1_i32
  second_id = -1_i32
  best = -Float32::INFINITY
  second = -Float32::INFINITY
  logits.each_with_index do |value, index|
    if value > best
      second = best
      second_id = best_id
      best = value
      best_id = index.to_i32
    elsif value > second
      second = value
      second_id = index.to_i32
    end
  end
  {best_id, second_id}
end

private def compare(a : Array(Float32), b : Array(Float32)) : {Float64, Float64}
  raise "logit width mismatch" unless a.size == b.size
  dot = aa = bb = max_abs = 0.0_f64
  a.each_with_index do |av, i|
    bv = b[i]
    raise "non-finite logit" unless av.finite? && bv.finite?
    af, bf = av.to_f64, bv.to_f64
    dot += af * bf
    aa += af * af
    bb += bf * bf
    max_abs = Math.max(max_abs, (af - bf).abs)
  end
  {dot / Math.sqrt(aa * bb), max_abs}
end

private def median(values : Array(Float64)) : Float64
  sorted = values.sort
  mid = sorted.size // 2
  sorted.size.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0
end

model = ARGV[0]? || raise "usage: qwen35_q4_b128_sg8_abba MODEL"
candidate_raw = ENV["QWEN35_B128_ABBA_CANDIDATE"]?
baseline_raw = ENV["QWEN35_B128_ABBA_BASELINE"]? || "0"
candidate_mode = candidate_raw == "auto" ? nil : candidate_raw
baseline_mode = baseline_raw == "auto" ? nil : baseline_raw
prompts = (ENV["QWEN35_B128_ABBA_PROMPTS"]? || "256,512,1024,2048").split(',').map(&.to_i)
reps = (ENV["QWEN35_B128_ABBA_REPS"]? || "4").to_i
raise "QWEN35_B128_ABBA_REPS must be positive" unless reps > 0
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
begin
  hp = weights.hparams
  puts "Q4 H16 B128 SG8 same-process ABBA device=#{ML::Metal::Device.instance.name.inspect} reps=#{reps} candidate=#{candidate_mode.inspect} baseline=#{baseline_mode.inspect}"
  puts "# pp candidate_tok/s baseline_tok/s mean_gain paired_median_gain min_cosine max_abs top2 candidate_regular candidate_fused baseline_regular baseline_fused"
  prompts.each do |pp|
    tokens = ML::QwenVsLlamaBenchmarkContract.synthetic_prefill_tokens(pp.to_i32, weights.output.out_dim)
    state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: pp.to_i32 + 4, kv_cache_f16: true)
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp, clear: true, admit_adaptive_resident_kv: false)

    profile = ML::GGUF::Qwen35Metal::Profile
    profile.reset
    profile.enable!
    run(weights, tokens, state, candidate_mode)
    regular_hits = profile.route_count("q4_h16_b128_sg8")
    fused_hits = profile.route_count("q4_h16_b128_sg8_swiglu_h16")
    raise "candidate route absent at pp#{pp}" unless regular_hits > 0 && fused_hits > 0
    profile.reset
    run(weights, tokens, state, baseline_mode)
    baseline_regular_hits = profile.route_count("q4_h16_b128_sg8")
    baseline_fused_hits = profile.route_count("q4_h16_b128_sg8_swiglu_h16")
    if baseline_mode == "0"
      raise "rollback entered candidate route" unless baseline_regular_hits == 0 && baseline_fused_hits == 0
    else
      raise "candidate did not widen the measured route" unless regular_hits > baseline_regular_hits || fused_hits > baseline_fused_hits
    end
    profile.disable!
    profile.reset

    run(weights, tokens, state, baseline_mode)
    run(weights, tokens, state, candidate_mode)
    candidate_ms = [] of Float64
    baseline_ms = [] of Float64
    min_cosine = 1.0_f64
    max_abs = 0.0_f64
    selected_top2 = {-1_i32, -1_i32}
    reps.times do |i|
      candidate = nil.as(Sample?)
      baseline = nil.as(Sample?)
      if {true, false, false, true}[i % 4]
        candidate = run(weights, tokens, state, candidate_mode)
        baseline = run(weights, tokens, state, baseline_mode)
      else
        baseline = run(weights, tokens, state, baseline_mode)
        candidate = run(weights, tokens, state, candidate_mode)
      end
      c, b = candidate.not_nil!, baseline.not_nil!
      cosine, error = compare(c.logits, b.logits)
      min_cosine = Math.min(min_cosine, cosine)
      max_abs = Math.max(max_abs, error)
      raise "top-2 changed" unless top2(c.logits) == top2(b.logits)
      selected_top2 = top2(c.logits)
      candidate_ms << c.milliseconds
      baseline_ms << b.milliseconds
    end
    candidate_tps = pp * 1000.0 / (candidate_ms.sum / reps)
    baseline_tps = pp * 1000.0 / (baseline_ms.sum / reps)
    gain = (candidate_tps / baseline_tps - 1.0) * 100.0
    paired = candidate_ms.zip(baseline_ms).map { |c, b| (b / c - 1.0) * 100.0 }
    raise "candidate logit mismatch" unless min_cosine >= 0.999999 && max_abs <= 1.0e-6
    puts "#{pp} #{candidate_tps.round(2)} #{baseline_tps.round(2)} #{gain.round(2)}% #{median(paired).round(2)}% #{min_cosine.round(10)} #{max_abs.round(9)} #{selected_top2[0]}/#{selected_top2[1]} #{regular_hits} #{fused_hits} #{baseline_regular_hits} #{baseline_fused_hits}"
  end
ensure
  ML::GGUF::Qwen35Metal::Profile.disable!
  ML::GGUF::Qwen35Metal::Profile.reset
  weights.close
end
