#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen_qbit_quality_metrics"
require "../src/ml/qwen_vs_llama_benchmark_contract"

FLASH_ENV     = "QWEN35_PREFILL_ATTN_FLASH_D256"
DEFAULT_MODEL = (Path.home / ".cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf").to_s

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

private def top2(logits : Array(Float32)) : ML::GGUF::QwenQBitQualityMetrics::Top2
  ML::GGUF::QwenQBitQualityMetrics.top2(logits)
end

private def token_ecs(weights : ML::GGUF::Qwen35Weights,
                      exact : Int32,
                      candidate : Int32) : Float64
  exact_embedding = ML::GGUF::Qwen35CPU.embedding_lookup(weights.output, exact)
  candidate_embedding = ML::GGUF::Qwen35CPU.embedding_lookup(weights.output, candidate)
  ML::GGUF::QwenQBitQualityMetrics.embedding_cosine(exact_embedding, candidate_embedding)
end

model_path = DEFAULT_MODEL
prompt_tokens = 2048
steps = 8

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_prefill_flash_d256_quality_probe [options]"
  parser.on("--model=PATH", "Qwen3.5/Qwen3.8 GGUF") { |value| model_path = value }
  parser.on("--prompt-tokens=N", "Synthetic prefill length (default: 2048)") { |value| prompt_tokens = value.to_i }
  parser.on("--steps=N", "Teacher-forced continuation steps (default: 8)") { |value| steps = value.to_i }
  parser.on("-h", "--help", "Show help") { puts parser; exit }
end

raise "model not found: #{model_path}" unless File.file?(model_path)
raise "prompt tokens must be positive" unless prompt_tokens > 0
raise "steps must be positive" unless steps > 0
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

metadata = ML::GGUF::GGUFFile.new(model_path, mmap_tensors: false)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(metadata, model_path)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
baseline_state = nil.as(ML::GGUF::Qwen35CPU::State?)
flash_state = nil.as(ML::GGUF::Qwen35CPU::State?)
begin
  hp = weights.hparams
  tokens = ML::QwenVsLlamaBenchmarkContract.synthetic_prefill_tokens(
    prompt_tokens.to_i32, weights.output.out_dim,
  )
  baseline_state = ML::GGUF::Qwen35CPU::State.new(
    hp, max_seq: (prompt_tokens + steps + 4).to_i32, kv_cache_f16: true)
  flash_state = ML::GGUF::Qwen35CPU::State.new(
    hp, max_seq: (prompt_tokens + steps + 4).to_i32, kv_cache_f16: true)
  with_flash(false) do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(baseline_state.not_nil!, hp,
      clear: true, admit_adaptive_resident_kv: false)
  end
  with_flash(true) do
    ML::GGUF::Qwen35CPU.prepare_state_metal!(flash_state.not_nil!, hp,
      clear: true, admit_adaptive_resident_kv: false)
  end

  profile = ML::GGUF::Qwen35Metal::Profile
  profile.reset
  profile.enable!
  baseline_logits = with_flash(false) do
    ML::GGUF::Qwen35CPU.prefill_tokens_logits(weights, tokens, 0, baseline_state.not_nil!)
  end
  raise "baseline unexpectedly selected Flash-MMA" unless profile.route_count("prefill_attn_flash_d256") == 0
  profile.reset
  flash_logits = with_flash(true) do
    ML::GGUF::Qwen35CPU.prefill_tokens_logits(weights, tokens, 0, flash_state.not_nil!)
  end
  raise "Flash-MMA route was not selected" unless profile.route_count("prefill_attn_flash_d256") > 0
  profile.disable!
  profile.reset

  exact = top2(baseline_logits)
  candidate = top2(flash_logits)
  top1_matches = 0
  ranked_top2_matches = 0
  top2_overlap = 0
  max_first_logit_delta = 0.0_f32
  min_ecs = 1.0_f64
  exact_ids = [] of Int32
  candidate_ids = [] of Int32

  steps.times do |index|
    comparison = ML::GGUF::QwenQBitQualityMetrics.compare_top2(exact, candidate)
    top1_matches += 1 if exact.first_id == candidate.first_id
    ranked_top2_matches += comparison.ranked_matches
    top2_overlap += comparison.set_overlap
    max_first_logit_delta = Math.max(max_first_logit_delta, comparison.first_logit_delta)
    min_ecs = Math.min(min_ecs, token_ecs(weights, exact.first_id, candidate.first_id))
    exact_ids << exact.first_id
    candidate_ids << candidate.first_id
    break if index + 1 == steps

    teacher = exact.first_id
    pos = prompt_tokens + index
    first, first_logit, second, second_logit = with_flash(false) do
      ML::GGUF::Qwen35CPU.forward_top2(weights, teacher, pos, baseline_state.not_nil!)
    end
    exact = ML::GGUF::QwenQBitQualityMetrics::Top2.new(first, first_logit, second, second_logit)
    first, first_logit, second, second_logit = with_flash(true) do
      ML::GGUF::Qwen35CPU.forward_top2(weights, teacher, pos, flash_state.not_nil!)
    end
    candidate = ML::GGUF::QwenQBitQualityMetrics::Top2.new(first, first_logit, second, second_logit)
  end

  ranked_count = steps * 2
  exact_text = tokenizer.decode(exact_ids)
  candidate_text = tokenizer.decode(candidate_ids)
  quality_pass = top1_matches == steps && ranked_top2_matches == ranked_count &&
                 top2_overlap == ranked_count && min_ecs >= 0.999999
  puts "qwen35_prefill_flash_d256_quality_probe"
  puts "  model=#{model_path} heads=#{hp.n_head}/#{hp.n_head_kv} prompt_tokens=#{prompt_tokens} steps=#{steps}"
  puts "  top1=#{top1_matches}/#{steps} ranked_top2=#{ranked_top2_matches}/#{ranked_count} top2_overlap=#{top2_overlap}/#{ranked_count} min_ecs=#{min_ecs.round(8)} max_first_logit_delta=#{max_first_logit_delta.round(8)}"
  puts "  exact_ids=#{exact_ids.join(',')}"
  puts "  candidate_ids=#{candidate_ids.join(',')}"
  puts "  exact_text=#{exact_text.inspect}"
  puts "  candidate_text=#{candidate_text.inspect}"
  puts "  quality_pass=#{quality_pass}"
  raise "Flash-MMA continuation quality gate failed" unless quality_pass
ensure
  ML::GGUF::Qwen35Metal::Profile.disable!
  ML::GGUF::Qwen35Metal::Profile.reset
  ML::GGUF::Qwen35CPU.release_state_metal!(flash_state.not_nil!) if flash_state
  ML::GGUF::Qwen35CPU.release_state_metal!(baseline_state.not_nil!) if baseline_state
  weights.close
  metadata.close
end
