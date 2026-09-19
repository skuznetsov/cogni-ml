# Bounded real-model F32 append parity: ordinary rows versus direct SG4.
# Optional bounded warm timing is append+head+fence, not a kernel benchmark.
require "json"
require "digest/sha256"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen_qbit_quality_metrics"
require "../src/ml/metal/process_lease"

alias CPU = ML::GGUF::Qwen35CPU
alias GPU = ML::GGUF::Qwen35Metal
alias QM = ML::GGUF::QwenQBitQualityMetrics

STATE_ATOL    =   0.02_f64
STATE_RTOL    =   0.01_f64
LOGIT_ATOL    =    0.1_f64
LOGIT_COS     = 0.9999_f64
TOKEN_ECS_MIN =   0.99_f64
GENERATION    =          4
TIMING_WARMUP = ["baseline", "candidate", "candidate", "baseline"]
TIMING_ORDER  = ["baseline", "candidate", "candidate", "baseline",
                 "candidate", "baseline", "baseline", "candidate"] * 2

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

private def parse_shape(args : Array(String)) : {Int32, Int32, Bool, Bool, Bool, Bool}
  timing = args.includes?("--timing")
  prefix_profile = args.includes?("--prefix-profile")
  split_submit = args.includes?("--split-submit")
  raise ArgumentError.new("split submit requires prefix profile") if split_submit && !prefix_profile
  prefix_trace = args.includes?("--prefix-trace") || prefix_profile
  raise ArgumentError.new("timing and prefix trace are exclusive") if timing && prefix_trace
  dry = args.last? == "--dry-run"
  mode = timing ? ["--timing"] : (prefix_profile ? ["--prefix-profile"] : (prefix_trace ? ["--prefix-trace"] : [] of String))
  expected_tail = mode + (split_submit ? ["--split-submit"] : [] of String) + (dry ? ["--dry-run"] : [] of String)
  raise ArgumentError.new("expected shape [--timing|--prefix-trace|--prefix-profile [--split-submit]] [--dry-run]") unless args.size == 1 + expected_tail.size && args.skip(1) == expected_tail
  raise ArgumentError.new("prefix trace admits only 7839:193") if prefix_trace && args.first != "--shape=7839:193"
  case args.first
  when "--shape=256:195"  then {256, 195, dry, timing, prefix_trace, prefix_profile}
  when "--shape=7839:193" then {7839, 193, dry, timing, prefix_trace, prefix_profile}
  else                         raise ArgumentError.new("only 256:195 and 7839:193 are admitted")
  end
end

private def self_test
  same, bad, nan, empty = Delta.new, Delta.new, Delta.new, Delta.new
  same.add(1.0, 1.0, STATE_ATOL, STATE_RTOL)
  bad.add(1.0, 2.0, STATE_ATOL, STATE_RTOL)
  nan.add(1.0, Float64::NAN, STATE_ATOL, STATE_RTOL)
  raise "comparator self-test failed" unless same.passed? && !bad.passed? && !nan.passed? && !empty.passed?
  raise "top2 self-test failed" unless QM.top2([1.0_f32, 2.0_f32, 0.0_f32]).first_id == 1
  raise "shape parse failed" unless parse_shape(["--shape=256:195"]) == {256, 195, false, false, false, false} && parse_shape(["--shape=7839:193", "--dry-run"]) == {7839, 193, true, false, false, false}
  raise "timing parse failed" unless parse_shape(["--shape=256:195", "--timing", "--dry-run"]) == {256, 195, true, true, false, false}
  raise "prefix trace parse failed" unless parse_shape(["--shape=7839:193", "--prefix-trace", "--dry-run"]) == {7839, 193, true, false, true, false}
  raise "prefix profile parse failed" unless parse_shape(["--shape=7839:193", "--prefix-profile", "--dry-run"]) == {7839, 193, true, false, true, true}
  raise "split submit parse failed" unless parse_shape(["--shape=7839:193", "--prefix-profile", "--split-submit", "--dry-run"]) == {7839, 193, true, false, true, true}
  raise "unbalanced timing" unless TIMING_ORDER.size == 16 && TIMING_ORDER.count("baseline") == 8 && TIMING_WARMUP.count("baseline") == 2 && TIMING_WARMUP.count("candidate") == 2
  timing_quality!([1.0_f32, 2.0_f32], [1.0_f32, 2.0_f32], "same", "same")
  failures = 0
  [{[1.0_f32, 3.0_f32], "same"}, {[Float32::NAN, 2.0_f32], "same"}, {[1.0_f32, 2.0_f32], "changed"}].each do |values, hash|
    begin
      timing_quality!(values, [1.0_f32, 2.0_f32], hash, "same")
    rescue
      failures += 1
    end
  end
  raise "timing quality checker accepted defect" unless failures == 3
  rejected = 0
  [[] of String, ["--shape=256:194"], ["--shape=07839:193"], ["--shape=7839:195"],
   ["--shape=256:195", "--warmup"], ["--self-test", "--shape=256:195"],
   ["--shape=256:195", "--dry-run", "--dry-run"], ["--dry-run"],
   ["--shape=256:195", "--prefix-trace"], ["--shape=7839:193", "--prefix-trace", "--timing"],
   ["--shape=7839:193", "--prefix-trace", "--prefix-trace"], ["--shape=7839:193", "--dry-run", "--prefix-trace"],
   ["--shape=256:195", "--prefix-profile"], ["--shape=7839:193", "--prefix-profile", "--timing"],
   ["--shape=7839:193", "--prefix-profile", "--prefix-trace"], ["--shape=7839:193", "--prefix-profile", "--prefix-profile"],
   ["--shape=7839:193", "--dry-run", "--prefix-profile"],
   ["--shape=7839:193", "--split-submit"], ["--shape=7839:193", "--prefix-trace", "--split-submit"],
   ["--shape=7839:193", "--prefix-profile", "--split-submit", "--split-submit"],
   ["--shape=7839:193", "--split-submit", "--prefix-profile"],
   ["--shape=7839:193", "--prefix-profile", "--dry-run", "--split-submit"]].each do |args|
    begin
      parse_shape(args)
    rescue ArgumentError
      rejected += 1
    end
  end
  raise "invalid CLI accepted" unless rejected == 22
  puts({event: "self_test", passed: true, rejected_cli: rejected, gpu: false}.to_json)
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
  raise "F32 nonadaptive owners required" if state.kv_cache_f16? || state.adaptive_kv?
  raise "wrong layer count" unless state.layers.size == hp.n_layer
  state.layers.each_with_index do |layer, i|
    if hp.full_attention?(i)
      raise "missing F32 KV owner at #{i}" unless layer.k_cache_buf && layer.v_cache_buf && !layer.k_cache && !layer.v_cache
      expected = state.max_seq.to_i64 * hp.n_head_kv * hp.head_dim * 4
      raise "wrong KV capacity at #{i}" unless layer.k_cache_buf.not_nil!.size == expected && layer.v_cache_buf.not_nil!.size == expected
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
  owners = Set(UInt64).new
  a.layers.each_with_index do |left, i|
    right = b.layers[i]
    raise "position-field mismatch" unless left.position == right.position
    pairs = hp.full_attention?(i) ? [{"k", left.k_cache_buf, right.k_cache_buf}, {"v", left.v_cache_buf, right.v_cache_buf}] : [{"conv", left.conv_state_buf, right.conv_state_buf}, {"ssm", left.ssm_state_buf, right.ssm_state_buf}]
    pairs.each do |name, x_optional, y_optional|
      x, y = x_optional.not_nil!, y_optional.not_nil!
      raise "state buffer shape mismatch" unless x.size == y.size
      [x, y].each do |buffer|
        address = buffer.contents.address
        raise "aliased state owner" if owners.includes?(address)
        owners.add(address)
      end
      kv = name == "k" || name == "v"
      n = kv ? live * hp.n_head_kv * hp.head_dim : (x.size // 4).to_i32
      local = Delta.new
      n.times do |j|
        av, bv = x.contents.as(Float32*)[j].to_f64, y.contents.as(Float32*)[j].to_f64
        local.add(av, bv, exact ? 0.0 : STATE_ATOL, exact ? 0.0 : STATE_RTOL)
        groups[name].add(av, bv, exact ? 0.0 : STATE_ATOL, exact ? 0.0 : STATE_RTOL)
      end
      unless local.passed?
        puts({event: "state_outlier", label: label, layer: i, component: name, metrics: local.summary}.to_json)
      end
    end
  end
  puts({event: "state", label: label, live_tokens_checked: live, distinct_buffers: owners.size,
        position_fields_are_live_lengths: false, components: groups.transform_values(&.summary)}.to_json)
  STDOUT.flush
  groups.values.all?(&.passed?)
end

private def append_logits(weights : ML::GGUF::Qwen35Weights, state : CPU::State, ids : Array(Int32), prefix : Int32, arm : String)
  ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_OFF"] = arm == "baseline" ? "1" : "0"
  ML::Metal::Device.synchronize
  STDERR.puts("sg4_model_append_begin arm=#{arm}")
  STDERR.flush
  started = Time.instant
  hidden = CPU.prefill_tokens_last_hidden(weights, ids, prefix, state)
  logits = GPU.rmsnorm_project(hidden, weights.output_norm, weights.output, weights.hparams.rms_eps)
  raise "full GPU logit head unavailable" unless logits
  ML::Metal::Device.synchronize
  elapsed = (Time.instant - started).total_milliseconds
  STDERR.puts("sg4_model_append_end arm=#{arm}")
  STDERR.flush
  puts({event: "append", arm: arm, prefix_tokens: prefix, append_tokens: ids.size, wall_ms: elapsed}.to_json)
  STDOUT.flush
  {logits, elapsed}
end

private def state_fingerprint(state : CPU::State, hp : ML::GGUF::Qwen35Hparams, live : Int32, validate_finite : Bool = false) : String
  ML::Metal::Device.synchronize
  verify_owners(state, hp)
  raise "invalid fingerprint span" unless 0 < live <= state.max_seq
  digest = Digest::SHA256.new
  state.layers.each_with_index do |layer, i|
    buffers = hp.full_attention?(i) ? [layer.k_cache_buf.not_nil!, layer.v_cache_buf.not_nil!] : [layer.conv_state_buf.not_nil!, layer.ssm_state_buf.not_nil!]
    buffers.each_with_index do |buffer, component|
      bytes = hp.full_attention?(i) ? live.to_i64 * hp.n_head_kv * hp.head_dim * 4 : buffer.size
      raise "fingerprint span overflow" unless 0 < bytes <= buffer.size && bytes <= Int32::MAX
      if validate_finite
        raise "nonfinite reference state" unless Slice.new(buffer.contents.as(Float32*), (bytes // 4).to_i32).all?(&.finite?)
      end
      digest.update("#{i}:#{component}:#{bytes}:")
      digest.update(Slice.new(buffer.contents.as(UInt8*), bytes.to_i32))
    end
  end
  digest.final.hexstring
end

private def timing_quality!(logits : Array(Float32), reference : Array(Float32), state_hash : String, reference_hash : String)
  raise "timing sample changed state" unless state_hash == reference_hash
  raise "timing sample changed logits" unless !logits.empty? && logits == reference && logits.all?(&.finite?)
end

private def timing_run(weights : ML::GGUF::Qwen35Weights, prefix_state : CPU::State, work : CPU::State, ids : Array(Int32), prefix : Int32)
  # Tracing is captured at process startup. Refuse instrumented timing rather
  # than trying to disable its output after pipelines have initialized.
  raise "pipeline tracing must be disabled for timing" if ML::Metal::PipelineSelectionTrace::PREFIX.try { |p| !p.empty? }
  hp = weights.hparams
  prefix_hash = state_fingerprint(prefix_state, hp, prefix)
  reference = nil.as(Array(Float32)?)
  reference_hash = nil.as(String?)
  (TIMING_WARMUP + TIMING_ORDER).each_with_index do |arm, index|
    warmup = index < TIMING_WARMUP.size
    ML::Metal::Device.synchronize
    reset_started = Time.instant
    work.copy_from!(prefix_state)
    ML::Metal::Device.synchronize
    reset_ms = (Time.instant - reset_started).total_milliseconds
    pipelines_before = ML::Metal::PipelineCache.entry_count
    logits, elapsed = append_logits(weights, work, ids, prefix, arm)
    pipelines_after = ML::Metal::PipelineCache.entry_count
    raise "measured sample populated pipeline cache" if !warmup && pipelines_before != pipelines_after
    hash = state_fingerprint(work, hp, prefix + ids.size, validate_finite: reference.nil?)
    if reference.nil?
      raise "nonfinite warmup logits" unless logits.all?(&.finite?)
      reference, reference_hash = logits.dup, hash
    end
    timing_quality!(logits, reference.not_nil!, hash, reference_hash.not_nil!)
    top = QM.top2(logits)
    puts({event: "timing_sample", phase: warmup ? "warmup" : "measured", index: warmup ? index : index - TIMING_WARMUP.size,
          arm: arm, append_head_fence_ms: elapsed, reset_excluded_ms: reset_ms, state_sha256: hash,
          pipeline_entries_before: pipelines_before, pipeline_entries_after: pipelines_after,
          exact_logits: true, top2: {top.first_id, top.second_id}, passed: true}.to_json)
    STDOUT.flush
  end
  raise "immutable prefix changed" unless state_fingerprint(prefix_state, hp, prefix) == prefix_hash
  puts({event: "summary", passed: true, mode: "timing", measured_per_arm: 8, warmup_per_arm: 2,
        state_and_logits_exact: true, scope: "warm_append_plus_head_and_fence_not_kernel_or_pp_tg", timing_order: TIMING_ORDER}.to_json)
end

if ARGV == ["--self-test"]
  self_test
  exit
end
prefix_count, append_count, dry, timing, prefix_trace, prefix_profile = parse_shape(ARGV)
split_submit = ARGV.includes?("--split-submit")
model = ENV["QWEN35_MODEL"]? || "/Users/sergey/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
ENV.keys.select { |k| k.starts_with?("QWEN35_") }.each { |k| ENV.delete(k) }
ENV.delete("COGNI_METAL_SUBMIT_PROFILE")
ENV["COGNI_METAL_SUBMIT_PROFILE"] = "1" if split_submit
ENV["QWEN35_PREFILL_CHUNK_SIZE"] = "2048"
ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = "1"
ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = "50"
ENV["QWEN35_PREFILL_ATTN_FLASH_D256"] = "0"
ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_OFF"] = "0"
ENV["QWEN35_PREFILL_ATTN_ROWS_SG4_DIRECT_GATE_MIN"] = "1"
ENV["QWEN35_PREFILL_COMMAND_TRACE"] = "1" if prefix_trace
ENV["QWEN35_PREFILL_BOUNDARY_PROFILE"] = "1" if prefix_profile
ENV["COGNI_METAL_LEASE_WAIT_MS"] = "0"
gguf = ML::GGUF::GGUFFile.new(model, mmap_tensors: false)
begin
  hp = ML::GGUF::Qwen35Hparams.new(gguf)
  raise "model geometry not admitted" unless hp.n_layer == 64 && hp.n_head == 24 && hp.n_head_kv == 4 && hp.head_dim == 256 && hp.full_attention_layers.size == 16
  tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model)
ensure
  gguf.close
end
filler = tokenizer.encode("# Keep insertion order and remove repeated integers.\nvalues = [3, 1, 3, 2, 1]\n" * 300)
suffix = tokenizer.encode("\n# Return unique integers in their original order.\ndef stable_unique(values):\n    ")
raise "fixture suffix exceeds append" unless suffix.size < append_count
tokens = filler.first(prefix_count + append_count - suffix.size) + suffix
raise "fixture token count mismatch" unless tokens.size == prefix_count + append_count
capacity = tokens.size + GENERATION
puts({event: "config", model: model, prefix_tokens: prefix_count, append_tokens: append_count,
      generation: GENERATION, capacity: capacity, token_sha256: Digest::SHA256.hexdigest(tokens.join(",")),
      fixture: "public_raw_code_completion", layers: hp.n_layer, heads: hp.n_head, kv_heads: hp.n_head_kv, head_dim: hp.head_dim,
      baseline: "ordinary_rows", candidate: "direct_sg4", prefix: "shared_direct_prefill_deep_copy",
      state_atol: STATE_ATOL, state_rtol: STATE_RTOL, logit_atol: LOGIT_ATOL, logit_cosine_min: LOGIT_COS,
      token_ecs_min: TOKEN_ECS_MIN, dry_run: dry, timing: timing, timing_order: timing ? TIMING_ORDER : nil,
      warmup_order: timing ? TIMING_WARMUP : nil, prefix_trace: prefix_trace, prefix_profile: prefix_profile, split_submit: split_submit,
      prefix_token_sha256: Digest::SHA256.hexdigest(tokens.first(prefix_count).join(",")),
      controls: ENV.select { |k, _| k.starts_with?("QWEN35_") || k == "COGNI_METAL_SUBMIT_PROFILE" }.to_h, semantic_task_scored: false}.to_json)
STDOUT.flush
exit if dry

lease = ML::Metal::ProcessLease.acquire
states = [] of CPU::State
weights = nil.as(ML::GGUF::Qwen35Weights?)
begin
  device = ML::Metal::Device.instance.name
  raise "only Apple M2 Max admitted" unless device == "Apple M2 Max"
  loaded = ML::GGUF::Qwen35Weights.from_gguf(model)
  weights = loaded
  baseline = CPU::State.new(hp, capacity, kv_cache_f16: false)
  states << baseline
  CPU.prepare_state_metal!(baseline, hp, admit_adaptive_resident_kv: false)
  if prefix_trace
    puts({event: "prefix", phase: "begin", tokens: prefix_count, capacity: capacity}.to_json)
    STDOUT.flush
  end
  CPU.prefill_tokens(loaded, tokens.first(prefix_count), 0, baseline)
  ML::Metal::Device.synchronize
  if prefix_trace
    puts({event: "prefix", phase: "end", tokens: prefix_count, capacity: capacity}.to_json)
    puts({event: "summary", mode: prefix_profile ? "prefix_profile" : "prefix_trace", passed: true,
          scope: "prefix_completion_only_not_parity_speed_or_stability"}.to_json)
  else
    candidate = CPU::State.new(hp, capacity, kv_cache_f16: false)
    states << candidate
    candidate.copy_from!(baseline)
    raise "copied prefix is not exact" unless compare_state(baseline, candidate, hp, prefix_count, "common_prefix", exact: true)
    appended = tokens[prefix_count, append_count]
    if timing
      timing_run(loaded, baseline, candidate, appended, prefix_count)
    else
      base_logits, base_ms = append_logits(loaded, baseline, appended, prefix_count, "baseline")
      cand_logits, cand_ms = append_logits(loaded, candidate, appended, prefix_count, "candidate")
      raise "append state parity failed" unless compare_state(baseline, candidate, hp, tokens.size, "after_append")
      base_ids, cand_ids = [] of Int32, [] of Int32
      ranked = covered = 0
      min_ecs = min_cos = 1.0
      max_delta = 0.0
      GENERATION.times do |step|
        a, b = QM.top2(base_logits), QM.top2(cand_logits)
        delta = Delta.new
        raise "logit width mismatch" unless base_logits.size == cand_logits.size
        base_logits.each_with_index { |v, i| delta.add(v.to_f64, cand_logits[i].to_f64, LOGIT_ATOL) }
        cosine = QM.embedding_cosine(base_logits, cand_logits)
        ecs = a.first_id == b.first_id ? 1.0 : QM.embedding_cosine(CPU.embedding_lookup(loaded.token_embd, a.first_id), CPU.embedding_lookup(loaded.token_embd, b.first_id))
        cmp = QM.compare_top2(a, b)
        ranked += cmp.ranked_matches
        covered += 1 if cmp.exact_top1_covered
        min_ecs, min_cos = Math.min(min_ecs, ecs), Math.min(min_cos, cosine)
        max_delta = Math.max(max_delta, delta.max_abs)
        base_ids << a.first_id
        cand_ids << b.first_id
        passed = delta.passed? && cosine >= LOGIT_COS && cmp.exact_top1_covered && ecs >= TOKEN_ECS_MIN && a.first_id == b.first_id
        puts({event: "greedy", step: step, passed: passed, baseline_top2: {a.first_id, a.second_id}, candidate_top2: {b.first_id, b.second_id},
              token_ecs: ecs, logit_cosine: cosine, delta: delta.summary}.to_json)
        STDOUT.flush
        raise "greedy/logit gate failed; histories must not diverge" unless passed
        if step + 1 < GENERATION
          # Independent greedy consumers; equal histories are checked above.
          base_logits = CPU.forward(loaded, a.first_id, tokens.size + step, baseline)
          cand_logits = CPU.forward(loaded, b.first_id, tokens.size + step, candidate)
          ML::Metal::Device.synchronize
        end
      end
      raise "continuation state parity failed" unless compare_state(baseline, candidate, hp, tokens.size + GENERATION - 1, "after_greedy")
      puts({event: "summary", passed: true, device: device, top1_matches: GENERATION,
            top1_count: GENERATION, top2_ranked_matches: ranked, top2_ranked_count: 2 * GENERATION,
            exact_top1_covered: covered, token_ecs_min: min_ecs, logit_cosine_min: min_cos, logit_max_abs: max_delta,
            baseline_ids: base_ids, candidate_ids: cand_ids, baseline_text: tokenizer.decode(base_ids), candidate_text: tokenizer.decode(cand_ids),
            baseline_append_ms: base_ms, candidate_append_ms: cand_ms, timing_is_diagnostic: true,
            eos_stopping: false, semantic_task_scored: false}.to_json)
    end
  end
rescue ex
  puts({event: "summary", passed: false, error: ex.message}.to_json)
  raise ex
ensure
  states.each { |state| release_state(state) }
  weights.try(&.close)
  lease.close
end
