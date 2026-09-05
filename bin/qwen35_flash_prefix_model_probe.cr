# Bounded full-width F16 prefix-append probe, not the terminal-row product API.
require "json"
require "option_parser"
require "digest/sha256"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_quality_metrics"

alias CPU = ML::GGUF::Qwen35CPU
alias GPU = ML::GGUF::Qwen35Metal
alias QM = ML::GGUF::QwenQBitQualityMetrics

# Fixed before model runs. These are bounded diagnostic tolerances, not a
# general semantic-quality guarantee. Every component must pass separately.
STATE_ATOL    =   0.02_f64
STATE_RTOL    =   0.01_f64
LOGIT_ATOL    =    0.1_f64
LOGIT_COS     = 0.9999_f64
TOKEN_ECS_MIN =   0.99_f64

class Delta
  getter count = 0_i64
  getter nonfinite = 0_i64
  getter outside = 0_i64
  getter max_abs = 0.0_f64
  @square = 0.0_f64

  def add(a : Float64, b : Float64, atol : Float64, rtol : Float64 = 0.0)
    @count += 1
    unless a.finite? && b.finite?
      @nonfinite += 1
      return
    end
    d = (a - b).abs
    @max_abs = Math.max(@max_abs, d)
    @square += d * d
    @outside += 1 if d > atol + rtol * a.abs
  end

  def passed? : Bool
    @count > 0 && @nonfinite == 0 && @outside == 0
  end

  def summary
    {count: @count, nonfinite: @nonfinite, outside: @outside,
     max_abs: @max_abs, rmse: Math.sqrt(@square / Math.max(@count, 1)), passed: passed?}
  end
end

private def half_value(bits : UInt16) : Float64
  sign = (bits & 0x8000) == 0 ? 1.0 : -1.0
  exponent = ((bits >> 10) & 31).to_i
  mantissa = (bits & 1023).to_i
  return mantissa == 0 ? sign * Float64::INFINITY : Float64::NAN if exponent == 31
  return sign * mantissa * (2.0 ** -24) if exponent == 0
  sign * (1.0 + mantissa / 1024.0) * (2.0 ** (exponent - 15))
end

private def release_state(state : CPU::State)
  ML::Metal::Device.synchronize
  state.layers.each do |layer|
    layer.k_cache_buf.try(&.release)
    layer.v_cache_buf.try(&.release)
    layer.conv_state_buf.try(&.release)
    layer.ssm_state_buf.try(&.release)
    layer.k_cache_buf = layer.v_cache_buf = nil
    layer.conv_state_buf = layer.ssm_state_buf = nil
  end
end

private def verify_owners(state : CPU::State, hp : ML::GGUF::Qwen35Hparams)
  raise "F16 nonadaptive owners required" unless state.kv_cache_f16? && !state.adaptive_kv?
  state.layers.each_with_index do |layer, i|
    if hp.full_attention?(i)
      raise "missing F16 KV owner at #{i}" unless layer.k_cache_buf && layer.v_cache_buf && !layer.k_cache && !layer.v_cache
      expected = state.max_seq.to_i64 * hp.n_head_kv * hp.head_dim * 2
      raise "wrong F16 KV capacity at #{i}" unless layer.k_cache_buf.not_nil!.size == expected && layer.v_cache_buf.not_nil!.size == expected
    else
      raise "missing recurrent owner at #{i}" unless layer.conv_state_buf && layer.ssm_state_buf && !layer.conv_state && !layer.ssm_state
    end
  end
end

private def compare_state(a : CPU::State, b : CPU::State, hp : ML::GGUF::Qwen35Hparams, live : Int32, label : String, exact : Bool = false) : Bool
  ML::Metal::Device.synchronize
  verify_owners(a, hp)
  verify_owners(b, hp)
  raise "bad live span" unless live > 0 && live <= a.max_seq && a.max_seq == b.max_seq
  groups = {"k" => Delta.new, "v" => Delta.new, "conv" => Delta.new, "ssm" => Delta.new}
  a.layers.each_with_index do |left, i|
    right = b.layers[i]
    # Position fields are caller-maintained, not a live-length certificate.
    raise "position-field mismatch" unless left.position == right.position
    pairs = hp.full_attention?(i) ? [{"k", left.k_cache_buf, right.k_cache_buf}, {"v", left.v_cache_buf, right.v_cache_buf}] : [{"conv", left.conv_state_buf, right.conv_state_buf}, {"ssm", left.ssm_state_buf, right.ssm_state_buf}]
    pairs.each do |name, x_optional, y_optional|
      x, y = x_optional.not_nil!, y_optional.not_nil!
      raise "state buffer shape mismatch" unless x.size == y.size
      half = name == "k" || name == "v"
      n = half ? live * hp.n_head_kv * hp.head_dim : (x.size // 4).to_i32
      local = Delta.new
      worst_index = 0
      worst_a = worst_b = 0.0
      n.times do |j|
        av = half ? half_value(x.contents.as(UInt16*)[j]) : x.contents.as(Float32*)[j].to_f64
        bv = half ? half_value(y.contents.as(UInt16*)[j]) : y.contents.as(Float32*)[j].to_f64
        if (av - bv).abs > local.max_abs
          worst_index, worst_a, worst_b = j, av, bv
        end
        local.add(av, bv, exact ? 0.0 : STATE_ATOL, exact ? 0.0 : STATE_RTOL)
        groups[name].add(av, bv, exact ? 0.0 : STATE_ATOL, exact ? 0.0 : STATE_RTOL)
      end
      unless local.passed?
        puts({event: "state_outlier", label: label, layer: i, component: name,
              worst_index: worst_index, worst_token_row: half ? worst_index // (hp.n_head_kv * hp.head_dim) : nil,
              baseline_value: worst_a, candidate_value: worst_b, metrics: local.summary}.to_json)
      end
    end
  end
  puts({event: "state", label: label, live_tokens_checked: live,
        position_fields_are_live_lengths: false, components: groups.transform_values(&.summary)}.to_json)
  groups.values.all?(&.passed?)
end

private def route_ok?(count : Int64, flash : Bool, expected : Int32) : Bool
  count == (flash ? expected : 0)
end

private def prefill_prefix(weights : ML::GGUF::Qwen35Weights, prefix : Array(Int32), capacity : Int32)
  ENV["QWEN35_PREFILL_ATTN_FLASH_D256"] = "0"
  state = CPU::State.new(weights.hparams, capacity, kv_cache_f16: true)
  CPU.prepare_state_metal!(state, weights.hparams)
  ML::Metal::Device.synchronize
  started = Time.instant
  CPU.prefill_tokens(weights, prefix, 0, state)
  ML::Metal::Device.synchronize
  {state, (Time.instant - started).total_milliseconds}
rescue ex
  release_state(state) if state
  raise ex
end

private def append_logits(weights : ML::GGUF::Qwen35Weights, state : CPU::State, ids : Array(Int32), prefix : Int32, flash : Bool)
  ENV["QWEN35_PREFILL_ATTN_FLASH_D256"] = flash ? "1" : "0"
  GPU::Profile.reset
  GPU::Profile.enable!
  ML::Metal::Device.synchronize
  started = Time.instant
  hidden = CPU.prefill_tokens_last_hidden(weights, ids, prefix, state)
  logits = GPU.rmsnorm_project(hidden, weights.output_norm, weights.output, weights.hparams.rms_eps)
  raise "full-logit GPU head unavailable" unless logits
  ML::Metal::Device.synchronize
  elapsed = (Time.instant - started).total_milliseconds
  count = GPU::Profile.route_count("prefill_attn_flash_d256")
  puts({event: "append", flash: flash, prefix_tokens: prefix, actual_prefill_rows: ids.size,
        flash_dispatches: count, wall_ms: elapsed}.to_json)
  raise "wrong executed Flash route count" unless route_ok?(count, flash, weights.hparams.full_attention_layers.size)
  {logits, elapsed}
ensure
  GPU::Profile.disable!
end

private def self_test
  same = Delta.new
  same.add(1.0, 1.0, STATE_ATOL, STATE_RTOL)
  bad = Delta.new
  bad.add(1.0, 2.0, STATE_ATOL, STATE_RTOL)
  nan = Delta.new
  nan.add(1.0, Float64::NAN, STATE_ATOL, STATE_RTOL)
  raise "comparator self-test failed" unless same.passed? && !bad.passed? && !nan.passed?
  raise "route self-test failed" unless route_ok?(16_i64, true, 16) && !route_ok?(0_i64, true, 16)
  raise "half decoder self-test failed" unless half_value(0x3c00_u16) == 1.0 && half_value(0xbc00_u16) == -1.0 && half_value(0x0001_u16) == 2.0 ** -24 && half_value(0x7e00_u16).nan?
  raise "top2 self-test failed" unless QM.top2([1.0_f32, 2.0_f32, 0.0_f32]).first_id == 1
  puts "self_test=PASS (equal, perturbed, nonfinite, route, half decoder, top2)"
end

model = ENV["QWEN35_MODEL"]? || "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
prefix_count = 256
append_count = 65
generation = 8
warmup = false
reverse = false
test_only = false
control = false
OptionParser.parse do |p|
  p.on("--model PATH", "GGUF path") { |v| model = v }
  p.on("--prefix N", "Common prefix token count") { |v| prefix_count = v.to_i }
  p.on("--append N", "Actual full-width appended rows") { |v| append_count = v.to_i }
  p.on("--gen N", "Teacher/free continuation positions") { |v| generation = v.to_i }
  p.on("--warmup", "Warm both full append shapes before measurements") { warmup = true }
  p.on("--reverse", "Measure candidate before baseline") { reverse = true }
  p.on("--self-test", "No-model comparator qualification") { test_only = true }
  p.on("--control", "A/A control: keep Flash off in candidate too") { control = true }
  p.on("--help", "Show help") { puts p; exit }
end
self_test
exit if test_only
raise "bounded shape required" unless (2..2048).includes?(prefix_count) && (2..2048).includes?(append_count) && (2..32).includes?(generation) && prefix_count + append_count + generation <= 8192

# Pin the relevant route controls, without disabling memory/watchdog guards.
ENV.keys.select { |k| k.starts_with?("QWEN35_") }.each { |k| ENV.delete(k) }
ENV["QWEN35_PREFILL_CHUNK_SIZE"] = "2048"
ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = "1"
ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = "50"
partial = prefix_count % 4 != 0 || append_count % 4 != 0
ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_OFF"] = partial ? "1" : "0"
ENV["QWEN35_PREFILL_ATTN_FLASH_D256"] = "1"
gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
begin
  tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model)
ensure
  gguf.close
end
raise "Flash F16 pipelines unavailable" unless GPU.kv_cache_f16_pipelines_supported?
weights = ML::GGUF::Qwen35Weights.from_gguf(model)
hp = weights.hparams
raise "model/device not admitted" unless GPU.prefill_attn_flash_d256_policy?(ML::Metal::Device.instance.name, prefix_count, append_count, hp.n_head, hp.n_head_kv, hp.head_dim, true, false, "1")
capacity = prefix_count + append_count + generation
filler = tokenizer.encode("# Keep insertion order and remove repeated integers.\nvalues = [3, 1, 3, 2, 1]\n" * 300)
suffix = tokenizer.encode("\n# Return unique integers in their original order.\ndef stable_unique(values):\n    ")
raise "fixture suffix exceeds append" unless suffix.size < append_count
tokens = filler.first(prefix_count + append_count - suffix.size) + suffix
raise "fixture token count mismatch" unless tokens.size == prefix_count + append_count
prefix, appended = tokens.first(prefix_count), tokens[prefix_count, append_count]
puts({event: "config", model: model, device: ML::Metal::Device.instance.name, layers: hp.n_layer,
      prefix: prefix_count, append: append_count, generation: generation, capacity: capacity,
      fixture: "raw_code_completion_fixed_token_count", tokens_sha256: Digest::SHA256.hexdigest(tokens.join(",")),
      comparator: partial ? "rows_partial_group_guard" : "default_sg4", ecs_basis: "token_embd.weight",
      timing_contract: "full_width_last_hidden_plus_full_gpu_logits_fenced", warmup: warmup, reverse: reverse, control: control,
      state_atol: STATE_ATOL, state_rtol: STATE_RTOL, logit_atol: LOGIT_ATOL, logit_cosine_min: LOGIT_COS,
      teacher_ecs_min: TOKEN_ECS_MIN, state_gate: "conservative_value_equivalence_not_semantic_quality",
      ranked_top2_is_diagnostic: true}.to_json)

states = [] of CPU::State
passed = true
begin
  if warmup
    [false, true].each do |flash|
      state, _ = prefill_prefix(weights, prefix, capacity)
      begin
        append_logits(weights, state, appended, prefix_count, flash && !control)
      ensure
        release_state(state)
      end
    end
  end
  baseline, base_prefix_ms = prefill_prefix(weights, prefix, capacity)
  states << baseline
  candidate, cand_prefix_ms = prefill_prefix(weights, prefix, capacity)
  states << candidate
  passed &= compare_state(baseline, candidate, hp, prefix_count, "common_prefix", exact: true)
  raise "common prefix is not exact" unless passed
  base_logits = [] of Float32
  cand_logits = [] of Float32
  base_ms = cand_ms = 0.0
  (reverse ? [true, false] : [false, true]).each do |flash|
    logits, ms = append_logits(weights, flash ? candidate : baseline, appended, prefix_count, flash && !control)
    if flash
      cand_logits, cand_ms = logits, ms
    else
      base_logits, base_ms = logits, ms
    end
  end
  state_passed = compare_state(baseline, candidate, hp, tokens.size, "after_append")
  teacher_passed = true
  base_ids, teacher_ids = [] of Int32, [] of Int32
  ranked = covered = 0
  min_ecs = min_cos = 1.0
  max_logit_delta = 0.0
  generation.times do |step|
    a, b = QM.top2(base_logits), QM.top2(cand_logits)
    delta = Delta.new
    raise "logit width mismatch" unless base_logits.size == cand_logits.size
    base_logits.each_with_index { |v, i| delta.add(v.to_f64, cand_logits[i].to_f64, LOGIT_ATOL) }
    cosine = QM.embedding_cosine(base_logits, cand_logits)
    ecs = a.first_id == b.first_id ? 1.0 : QM.embedding_cosine(CPU.embedding_lookup(weights.token_embd, a.first_id), CPU.embedding_lookup(weights.token_embd, b.first_id))
    cmp = QM.compare_top2(a, b)
    ranked += cmp.ranked_matches
    covered += 1 if cmp.exact_top1_covered
    min_ecs, min_cos = Math.min(min_ecs, ecs), Math.min(min_cos, cosine)
    max_logit_delta = Math.max(max_logit_delta, delta.max_abs)
    teacher_passed &= delta.passed? && cosine >= LOGIT_COS && cmp.exact_top1_covered && ecs >= TOKEN_ECS_MIN
    base_ids << a.first_id
    teacher_ids << b.first_id
    puts({event: "teacher", step: step, baseline_top2: {a.first_id, a.second_id}, candidate_top2: {b.first_id, b.second_id},
          token_ecs: ecs, logit_cosine: cosine, delta: delta.summary}.to_json)
    if step + 1 < generation
      pos = tokens.size + step
      base_logits = CPU.forward(weights, a.first_id, pos, baseline)
      cand_logits = CPU.forward(weights, a.first_id, pos, candidate)
      ML::Metal::Device.synchronize
    end
  end
  state_passed &= compare_state(baseline, candidate, hp, tokens.size + generation - 1, "after_teacher")
  states.each { |s| release_state(s) }
  states.clear
  free, _ = prefill_prefix(weights, prefix, capacity)
  states << free
  free_logits, _ = append_logits(weights, free, appended, prefix_count, !control)
  free_ids = [] of Int32
  generation.times do |step|
    id = QM.top2(free_logits).first_id
    free_ids << id
    if step + 1 < generation
      free_logits = CPU.forward(weights, id, tokens.size + step, free)
    end
  end
  free_match = free_ids == base_ids
  passed = state_passed && teacher_passed && free_match
  puts({event: "summary", passed: passed, state_passed: state_passed, teacher_passed: teacher_passed,
        baseline_prefix_ms: base_prefix_ms, candidate_prefix_ms: cand_prefix_ms,
        baseline_append_ms: base_ms, candidate_append_ms: cand_ms, append_ratio: base_ms / cand_ms,
        timing_is_diagnostic: true, top1_matches: base_ids.zip(teacher_ids).count { |a, b| a == b },
        top2_ranked_matches: ranked, top2_ranked_count: 2 * generation, exact_top1_covered: covered,
        token_ecs_min: min_ecs, logit_cosine_min: min_cos, logit_max_abs: max_logit_delta,
        free_match: free_match, baseline_ids: base_ids, candidate_free_ids: free_ids,
        baseline_text: tokenizer.decode(base_ids), candidate_text: tokenizer.decode(free_ids),
        eos_stopping: false, semantic_task_scored: false}.to_json)
  raise "full-model prefix gate FAILED; retain explicit-only admission" unless passed
ensure
  states.each { |s| release_state(s) }
  weights.close
end
