# Same-process product decode A/B for adaptive T8 and fused-stage2 routes.
#
# The normal timing mode forces both states along the same greedy trajectory.
# `--quality-top2` instead self-feeds two independent real-prefix trajectories.
# Execution order is alternated per position so model loading and order drift
# cannot be mistaken for a kernel speedup.
#
# `--synthetic-prefix` restores a canonical zero-valued adaptive cache into the
# real model state. It isolates long-context decode shape and scheduling without
# the quadratic prefill cost; it is performance/correctness evidence, not a
# semantic-quality substitute for a real prompt.

require "json"
require "option_parser"
require "digest/sha256"

require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen_qbit_quality_metrics"

alias QM = ML::GGUF::QwenQBitQualityMetrics

DEFAULT_QWEN38_MODEL       = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_RESIDENT_MAP       = "p4;27=bf16,43=bf16,47=bf16,51=bf16"
PRODUCT_PREFILL_CHUNK_SIZE = ENV.fetch("QWEN35_ADAPTIVE_PROBE_PREFILL_CHUNK_SIZE", "512").to_i32
PRODUCT_APPEND_MAX_GROUPS  =   1
PRODUCT_APPEND_COOLDOWN_MS = 100
T8_ENV_KEYS                = [
  "QWEN35_ADAPTIVE_P4_SPLITK_T8",
  "QWEN35_ADAPTIVE_BF16_SPLITK_T8",
]
DIRECT_QK_ENV_KEY    = "QWEN35_ADAPTIVE_P4_SPLITK_DIRECT_QK"
V_CONTIGUOUS_ENV_KEY = "QWEN35_ADAPTIVE_P4_SPLITK_V_CONTIGUOUS"
ADAPTIVE_ENV_KEYS    = [
  "QWEN35_ADAPTIVE_RESIDENT_KV_LAYER",
  "QWEN35_ADAPTIVE_RESIDENT_KV_TIER",
  "QWEN35_ADAPTIVE_RESIDENT_KV_MAP",
  "QWEN35_PREFILL_APPEND_CMD_OFF",
  "QWEN35_PREFILL_RESIDENT_BOUNDARY_OFF",
  "QWEN35_PREFILL_FUSE_FULL_REC_OFF",
  "QWEN35_FULL_PREFILL_CHUNK_OFF",
  "QWEN35_PREFILL_REC_RUN_OFF",
  "QWEN35_PREFILL_CHUNK_OFF",
  "QWEN35_PREFILL_CHUNK_SIZE",
  "QWEN35_PREFILL_APPEND_MAX_GROUPS",
  "QWEN35_PREFILL_APPEND_COOLDOWN_MS",
  "QWEN35_PREFILL_GC_GUARD_OFF",
  "QWEN35_SCRATCH_OFF",
  "QWEN35_ADAPTIVE_SPLITK",
  "QWEN35_ADAPTIVE_SPLITK_MIN_CTX",
  "QWEN35_ADAPTIVE_SPLITK_CHUNK",
  "QWEN35_ADAPTIVE_GQA6_TILE",
  "QWEN35_ADAPTIVE_DEQUANT_T4",
  "QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF",
  "QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED",
  "QWEN35_ADAPTIVE_PACK_PREFIX_QUANT",
] + T8_ENV_KEYS + [DIRECT_QK_ENV_KEY, V_CONTIGUOUS_ENV_KEY]

RELEASE_BUILD           = {% if flag?(:release) %} true {% else %} false {% end %}
QUALITY_LOGIT_TOLERANCE = 1e-4_f32

raise "QWEN35_ADAPTIVE_PROBE_PREFILL_CHUNK_SIZE must be positive" unless PRODUCT_PREFILL_CHUNK_SIZE > 0

record DecodeSample,
  index : Int32,
  order : String,
  baseline_input_id : Int32,
  candidate_input_id : Int32,
  baseline_output_id : Int32,
  candidate_output_id : Int32,
  baseline_ms : Float64,
  candidate_ms : Float64,
  logit_delta : Float32,
  quality : DecodeQualitySample?

record DecodeQualitySample,
  phase : String,
  index : Int32,
  baseline : QM::Top2,
  candidate : QM::Top2,
  comparison : QM::Top2Comparison,
  token_ecs : Float64,
  paired_logits_valid : Bool

record RouteCertificate,
  p4_t8_owners : Int32,
  bf16_t8_owners : Int32,
  p4_direct_qk_owners : Int32,
  p4_v_contiguous_owners : Int32,
  packed_len : Int32

private def with_adaptive_probe_env(resident_map : String,
                                    candidate : Bool,
                                    compare_stage2 : Bool,
                                    compare_splitk_chunk : Bool,
                                    compare_direct_qk : Bool,
                                    compare_v_contiguous : Bool,
                                    compare_v_contiguous_auto : Bool,
                                    compare_p4_stage1 : Bool,
                                    &)
  old = ADAPTIVE_ENV_KEYS.to_h { |key| {key, ENV[key]?} }
  ADAPTIVE_ENV_KEYS.each { |key| ENV.delete(key) }
  ENV["QWEN35_ADAPTIVE_RESIDENT_KV_MAP"] = resident_map
  ENV["QWEN35_PREFILL_CHUNK_SIZE"] = PRODUCT_PREFILL_CHUNK_SIZE.to_s
  ENV["QWEN35_PREFILL_APPEND_MAX_GROUPS"] = PRODUCT_APPEND_MAX_GROUPS.to_s
  ENV["QWEN35_PREFILL_APPEND_COOLDOWN_MS"] = PRODUCT_APPEND_COOLDOWN_MS.to_s
  ENV["QWEN35_PREFILL_GC_GUARD_OFF"] = "0"
  ENV["QWEN35_SCRATCH_OFF"] = "0"
  ENV["QWEN35_ADAPTIVE_SPLITK"] = "1"
  ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"] = "256"
  ENV["QWEN35_ADAPTIVE_UNIFORM_PREFILL_OFF"] = "0"
  if compare_p4_stage1
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
    ENV["QWEN35_ADAPTIVE_GQA6_TILE"] = "15"
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    ENV[DIRECT_QK_ENV_KEY] = candidate ? "1" : "0"
    ENV[V_CONTIGUOUS_ENV_KEY] = candidate ? "1" : "0"
  elsif compare_v_contiguous || compare_v_contiguous_auto
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
    ENV["QWEN35_ADAPTIVE_GQA6_TILE"] = "15"
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    ENV[DIRECT_QK_ENV_KEY] = "0"
    unless candidate && compare_v_contiguous_auto
      ENV[V_CONTIGUOUS_ENV_KEY] = candidate ? "1" : "0"
    end
  elsif compare_direct_qk
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
    ENV["QWEN35_ADAPTIVE_GQA6_TILE"] = "15"
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    ENV[DIRECT_QK_ENV_KEY] = candidate ? "1" : "0"
    ENV[V_CONTIGUOUS_ENV_KEY] = "0"
  elsif compare_splitk_chunk
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "1"
    ENV["QWEN35_ADAPTIVE_GQA6_TILE"] = "15"
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = candidate ? "60" : "64"
  elsif compare_stage2
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    T8_ENV_KEYS.each { |key| ENV[key] = "1" }
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "0" unless candidate
  else
    ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"] = "64"
    ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"] = "0"
    value = candidate ? "1" : "0"
    T8_ENV_KEYS.each { |key| ENV[key] = value }
  end
  yield
ensure
  old.try &.each do |key, value|
    if value
      ENV[key] = value
    else
      ENV.delete(key)
    end
  end
end

private def verify_finite_top1!(label : String,
                                token_id : Int32,
                                logit : Float32,
                                vocab_size : Int32) : Nil
  unless token_id >= 0 && token_id < vocab_size
    raise "#{label} top-1 token #{token_id} is outside vocabulary size #{vocab_size}"
  end
  raise "#{label} top-1 logit is not finite" unless logit.finite?
end

private def verify_finite_top2!(label : String,
                                top2 : QM::Top2,
                                vocab_size : Int32) : Nil
  verify_finite_top1!(label, top2.first_id, top2.first_logit, vocab_size)
  unless top2.second_id >= 0 && top2.second_id < vocab_size
    raise "#{label} top-2 token #{top2.second_id} is outside vocabulary size #{vocab_size}"
  end
  raise "#{label} top-2 logit is not finite" unless top2.second_logit.finite?
  raise "#{label} top-2 token duplicates top-1" if top2.second_id == top2.first_id
  if top2.second_logit > top2.first_logit
    raise "#{label} top-2 logit exceeds top-1"
  end
end

private def token_embedding_ecs(weights : ML::GGUF::Qwen35Weights,
                                baseline_id : Int32,
                                candidate_id : Int32,
                                cache : Hash(Int32, Array(Float32))) : Float64
  return 1.0 if baseline_id == candidate_id

  baseline = cache.fetch(baseline_id) do
    embedding = ML::GGUF::Qwen35CPU.embedding_lookup(weights.output, baseline_id)
    cache[baseline_id] = embedding
    embedding
  end
  candidate = cache.fetch(candidate_id) do
    embedding = ML::GGUF::Qwen35CPU.embedding_lookup(weights.output, candidate_id)
    cache[candidate_id] = embedding
    embedding
  end
  QM.embedding_cosine(baseline, candidate)
end

private def compare_decode_quality(phase : String,
                                   index : Int32,
                                   baseline : QM::Top2,
                                   candidate : QM::Top2,
                                   weights : ML::GGUF::Qwen35Weights,
                                   embedding_cache : Hash(Int32, Array(Float32)),
                                   paired_logits_valid : Bool = true) : DecodeQualitySample
  comparison = QM.compare_top2(baseline, candidate)
  DecodeQualitySample.new(
    phase,
    index,
    baseline,
    candidate,
    comparison,
    token_embedding_ecs(weights, baseline.first_id, candidate.first_id, embedding_cache),
    paired_logits_valid,
  )
end

private def emit_decode_quality(sample : DecodeQualitySample) : Nil
  payload = JSON.build do |json|
    json.object do
      json.field "event", "decode_quality"
      json.field "phase", sample.phase
      json.field "index", sample.index
      json.field "baseline_top1_id", sample.baseline.first_id
      json.field "baseline_top1_logit", sample.baseline.first_logit
      json.field "baseline_top2_id", sample.baseline.second_id
      json.field "baseline_top2_logit", sample.baseline.second_logit
      json.field "baseline_margin", sample.baseline.margin
      json.field "candidate_top1_id", sample.candidate.first_id
      json.field "candidate_top1_logit", sample.candidate.first_logit
      json.field "candidate_top2_id", sample.candidate.second_id
      json.field "candidate_top2_logit", sample.candidate.second_logit
      json.field "candidate_margin", sample.candidate.margin
      json.field "ranked_top2_matches", sample.comparison.ranked_matches
      json.field "top2_set_overlap", sample.comparison.set_overlap
      json.field "exact_top1_covered", sample.comparison.exact_top1_covered
      json.field "exact_top2_covered", sample.comparison.exact_top2_covered
      json.field "first_logit_delta", sample.comparison.first_logit_delta
      json.field "second_logit_delta", sample.comparison.second_logit_delta
      json.field "margin_delta", sample.comparison.margin_delta
      json.field "token_ecs", sample.token_ecs
      json.field "paired_logits_valid", sample.paired_logits_valid
    end
  end
  puts "QBIT_T8_DECODE_QUALITY_JSON=#{payload}"
  STDOUT.flush
end

private def append_quality_violations!(violations : Array(String),
                                       label : String,
                                       sample : DecodeQualitySample) : Nil
  return unless sample.paired_logits_valid

  comparison = sample.comparison
  violations << "#{label}_top1_mismatch" unless sample.baseline.first_id == sample.candidate.first_id
  violations << "#{label}_ranked_top2_mismatch" unless comparison.ranked_matches == 2
  if comparison.first_logit_delta > QUALITY_LOGIT_TOLERANCE
    violations << "#{label}_first_logit_delta=#{comparison.first_logit_delta}"
  end
  if comparison.second_logit_delta > QUALITY_LOGIT_TOLERANCE
    violations << "#{label}_second_logit_delta=#{comparison.second_logit_delta}"
  end
  if comparison.margin_delta > QUALITY_LOGIT_TOLERANCE
    violations << "#{label}_margin_delta=#{comparison.margin_delta}"
  end
end

private def verify_candidate_t8_route!(state : ML::GGUF::Qwen35CPU::State,
                                       hp : ML::GGUF::Qwen35Hparams,
                                       resident_map : String,
                                       compare_stage2 : Bool,
                                       compare_splitk_chunk : Bool,
                                       compare_direct_qk : Bool,
                                       compare_v_contiguous : Bool,
                                       compare_v_contiguous_auto : Bool,
                                       compare_p4_stage1 : Bool) : RouteCertificate
  device_name = ML::Metal::Device.instance.name
  p4_owners = 0_i32
  bf16_owners = 0_i32
  p4_direct_qk_owners = 0_i32
  p4_v_contiguous_owners = 0_i32
  packed_len = -1_i32

  with_adaptive_probe_env(resident_map, true, compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1) do
    hp.full_attention_layers.each do |layer_index|
      cache = state.layers[layer_index].adaptive_kv
      raise "adaptive KV is missing layer #{layer_index}" unless cache
      cache.with_snapshot_buffers do |_k_base, k_sidecar, _v_base, v_sidecar, k_plan, v_plan, cache_len|
        packed_len = cache_len if packed_len < 0
        raise "adaptive T8 route owners have different prefix lengths" unless packed_len == cache_len
        tier = k_plan.uniform_tier
        unless tier && v_plan.uniform_tier == tier
          raise "adaptive T8 route requires a uniform K/V tier at layer #{layer_index}"
        end
        unless ML::GGUF::QwenQBitAdaptiveMetalPolicy.decode_splitk?(
                 cache_len, 1, true,
                 ENV["QWEN35_ADAPTIVE_SPLITK"]?,
                 ENV["QWEN35_ADAPTIVE_SPLITK_MIN_CTX"]?,
               )
          raise "adaptive split-K route is inactive at layer #{layer_index}"
        end

        case tier
        when ML::GGUF::QwenQBitAdaptiveKV::Tier::P4
          if compare_splitk_chunk && ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"]? != "60"
            raise "candidate split-K chunk 60 route is inactive at layer #{layer_index}"
          end
          p4_t8 = ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_t8?(
            device_name, cache_len, ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
          )
          unless p4_t8
            raise "P4 T8 route is inactive at layer #{layer_index}"
          end
          if (compare_stage2 || compare_splitk_chunk || compare_direct_qk || compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) &&
             !ML::GGUF::QwenQBitAdaptiveMetalPolicy.splitk_stage2_fused?(
               device_name, true, false, cache_len,
               ENV["QWEN35_ADAPTIVE_P4_SPLITK_T8"]?,
               ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"]?,
             )
            raise "automatic P4 fused stage2 route is inactive at layer #{layer_index}"
          end
          direct_qk = ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_direct_qk?(
            true,
            true,
            ENV[DIRECT_QK_ENV_KEY]?,
          )
          if (compare_direct_qk || compare_p4_stage1) && !direct_qk
            raise "P4 direct-QK route is inactive at layer #{layer_index}"
          end
          v_contiguous = ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_v_contiguous?(
            device_name,
            cache_len,
            true,
            p4_t8,
            ENV[V_CONTIGUOUS_ENV_KEY]?,
          )
          if (compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) && !v_contiguous
            raise "P4 contiguous-V route is inactive at layer #{layer_index}"
          end
          p4_direct_qk_owners += 1 if direct_qk
          p4_v_contiguous_owners += 1 if v_contiguous
          p4_owners += 1
        when ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16
          if compare_splitk_chunk && ENV["QWEN35_ADAPTIVE_SPLITK_CHUNK"]? != "60"
            raise "candidate split-K chunk 60 route is inactive at layer #{layer_index}"
          end
          unless !k_sidecar.contents.null? && k_sidecar.contents.address % 16_u64 == 0_u64 &&
                 !v_sidecar.contents.null? && v_sidecar.contents.address % 16_u64 == 0_u64
            raise "BF16 T8 sidecar alignment is invalid at layer #{layer_index}"
          end
          unless ML::GGUF::QwenQBitAdaptiveMetalPolicy.bf16_splitk_t8?(
                   device_name, cache_len, ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"]?,
                 )
            raise "BF16 T8 route is inactive at layer #{layer_index}"
          end
          if (compare_splitk_chunk || compare_direct_qk || compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) &&
             !ML::GGUF::QwenQBitAdaptiveMetalPolicy.splitk_stage2_fused?(
               device_name, false, true, cache_len,
               ENV["QWEN35_ADAPTIVE_BF16_SPLITK_T8"]?,
               ENV["QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED"]?,
             )
            raise "forced BF16 fused stage2 route is inactive at layer #{layer_index}"
          end
          if ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_direct_qk?(
               false,
               false,
               ENV[DIRECT_QK_ENV_KEY]?,
             )
            raise "P4 direct-QK route leaked into BF16 layer #{layer_index}"
          end
          if ML::GGUF::QwenQBitAdaptiveMetalPolicy.p4_splitk_v_contiguous?(
               device_name,
               cache_len,
               false,
               false,
               ENV[V_CONTIGUOUS_ENV_KEY]?,
             )
            raise "P4 contiguous-V route leaked into BF16 layer #{layer_index}"
          end
          bf16_owners += 1
        else
          raise "adaptive T8 probe found unsupported tier #{tier} at layer #{layer_index}"
        end
      end
    end
  end

  unless p4_owners > 0 && p4_owners + bf16_owners == hp.full_attention_layers.size
    raise "adaptive T8 route coverage is incomplete: P4=#{p4_owners}, BF16=#{bf16_owners}"
  end
  if compare_stage2
    raise "stage2 comparison requires every adaptive owner to use P4" unless bf16_owners == 0
  else
    raise "T8 comparison requires both P4 and BF16 owners" unless bf16_owners > 0
  end
  if (compare_direct_qk || compare_p4_stage1) && p4_direct_qk_owners != p4_owners
    raise "P4 direct-QK coverage is incomplete: #{p4_direct_qk_owners}/#{p4_owners}"
  end
  if (compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) && p4_v_contiguous_owners != p4_owners
    raise "P4 contiguous-V coverage is incomplete: #{p4_v_contiguous_owners}/#{p4_owners}"
  end
  RouteCertificate.new(p4_owners, bf16_owners, p4_direct_qk_owners, p4_v_contiguous_owners, packed_len)
end

private def release_state!(state : ML::GGUF::Qwen35CPU::State?) : Nil
  return unless state
  ML::GGUF::Qwen35CPU.release_state_metal!(state) if ML::Metal::Device.available?
end

private def verify_resident_state!(state : ML::GGUF::Qwen35CPU::State,
                                   hp : ML::GGUF::Qwen35Hparams,
                                   expected_tokens : Int32) : Nil
  unless state.adaptive_kv_layer_indices == hp.full_attention_layers
    raise "adaptive KV does not own every full-attention layer"
  end

  f32_owners = hp.full_attention_layers.select do |layer_index|
    layer = state.layers[layer_index]
    !!(layer.k_cache || layer.v_cache || layer.k_cache_buf || layer.v_cache_buf)
  end
  raise "adaptive KV has Float32 owners at layers #{f32_owners}" unless f32_owners.empty?

  hp.full_attention_layers.each do |layer_index|
    cache = state.layers[layer_index].adaptive_kv
    raise "adaptive KV is missing layer #{layer_index}" unless cache
    cache.with_snapshot_buffers do |_k_base, _k_sidecar, _v_base, _v_sidecar, _k_plan, _v_plan, cache_len|
      unless cache_len == expected_tokens
        raise "adaptive KV layer #{layer_index} has #{cache_len} tokens, expected #{expected_tokens}"
      end
    end
  end

  state.layers.each_with_index do |layer, layer_index|
    unless layer.position == expected_tokens
      raise "layer #{layer_index} position is #{layer.position}, expected #{expected_tokens}"
    end
  end
end

private def verify_independent_states!(baseline : ML::GGUF::Qwen35CPU::State,
                                       candidate : ML::GGUF::Qwen35CPU::State,
                                       hp : ML::GGUF::Qwen35Hparams) : Nil
  raise "baseline and candidate state objects alias" if baseline.same?(candidate)
  baseline.layers.each_with_index do |baseline_layer, layer_index|
    candidate_layer = candidate.layers[layer_index]
    raise "baseline and candidate layer #{layer_index} alias" if baseline_layer.same?(candidate_layer)
    if hp.full_attention?(layer_index)
      baseline_cache = baseline_layer.adaptive_kv
      candidate_cache = candidate_layer.adaptive_kv
      unless baseline_cache && candidate_cache
        raise "adaptive KV is missing layer #{layer_index}"
      end
      if baseline_cache.same?(candidate_cache)
        raise "baseline and candidate adaptive KV layer #{layer_index} alias"
      end
    else
      if (baseline_conv = baseline_layer.conv_state_buf) && baseline_conv.same?(candidate_layer.conv_state_buf)
        raise "baseline and candidate recurrent conv layer #{layer_index} alias"
      end
      if (baseline_ssm = baseline_layer.ssm_state_buf) && baseline_ssm.same?(candidate_layer.ssm_state_buf)
        raise "baseline and candidate recurrent SSM layer #{layer_index} alias"
      end
    end
  end
end

private def seed_synthetic_adaptive_prefix!(state : ML::GGUF::Qwen35CPU::State,
                                            hp : ML::GGUF::Qwen35Hparams,
                                            prefix_tokens : Int32) : Nil
  raise "synthetic prefix must be positive" unless prefix_tokens > 0
  prefix_rows64 = prefix_tokens.to_i64 * hp.n_head_kv
  raise "synthetic prefix row count exceeds Int32" if prefix_rows64 > Int32::MAX
  prefix_rows = prefix_rows64.to_i32

  hp.full_attention_layers.each do |layer_index|
    cache = state.layers[layer_index].adaptive_kv
    raise "adaptive KV is missing layer #{layer_index}" unless cache

    k_plan = nil.as(ML::GGUF::QwenQBitAdaptiveKV::Plan?)
    v_plan = nil.as(ML::GGUF::QwenQBitAdaptiveKV::Plan?)
    cache.with_snapshot_buffers do |_k_base, _k_sidecar, _v_base, _v_sidecar, current_k_plan, current_v_plan, cache_len|
      raise "synthetic prefix target is not empty at layer #{layer_index}" unless cache_len == 0
      k_plan = current_k_plan
      v_plan = current_v_plan
    end

    k = ML::GGUF::QwenQBitAdaptiveKV.empty_encoded(k_plan.not_nil!, prefix_rows)
    v = ML::GGUF::QwenQBitAdaptiveKV.empty_encoded(v_plan.not_nil!, prefix_rows)
    ML::GGUF::QwenQBitAdaptiveResidentKV.restore_snapshot!(cache, k, v, prefix_tokens)
  end

  state.layers.each { |layer| layer.position = prefix_tokens }
  verify_resident_state!(state, hp, prefix_tokens)
end

private def prepare_state(weights : ML::GGUF::Qwen35Weights,
                          tokens : Array(Int32),
                          max_seq : Int32,
                          resident_map : String,
                          candidate : Bool,
                          compare_stage2 : Bool,
                          compare_splitk_chunk : Bool,
                          compare_direct_qk : Bool,
                          compare_v_contiguous : Bool,
                          compare_v_contiguous_auto : Bool,
                          compare_p4_stage1 : Bool,
                          synthetic_prefix : Int32,
                          quality_top2 : Bool,
                          quality_top2_production_prefill : Bool)
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  begin
    first = QM::Top2.new(-1_i32, Float32::NAN, -1_i32, Float32::NAN)
    with_adaptive_probe_env(resident_map, candidate, compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1) do
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams)
      if synthetic_prefix > 0
        seed_synthetic_adaptive_prefix!(state, weights.hparams, synthetic_prefix)
        first = QM::Top2.new(tokens.last, 0.0_f32, -1_i32, -Float32::INFINITY)
      elsif quality_top2 && !quality_top2_production_prefill
        first = QM.top2(ML::GGUF::Qwen35CPU.prefill_tokens_logits(
          weights, tokens, 0, state,
        ))
      else
        first_id, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(
          weights, tokens, 0, state,
        )
        first = QM::Top2.new(first_id, first_logit, -1_i32, -Float32::INFINITY)
      end
    end
    expected_tokens = synthetic_prefix > 0 ? synthetic_prefix : tokens.size.to_i32
    state.layers.each { |layer| layer.position = expected_tokens }
    verify_resident_state!(state, weights.hparams, expected_tokens)
    {state, first}
  rescue ex
    release_state!(state)
    raise ex
  end
end

private def decode_step(weights : ML::GGUF::Qwen35Weights,
                        state : ML::GGUF::Qwen35CPU::State,
                        token_id : Int32,
                        position : Int32,
                        resident_map : String,
                        candidate : Bool,
                        compare_stage2 : Bool,
                        compare_splitk_chunk : Bool,
                        compare_direct_qk : Bool,
                        compare_v_contiguous : Bool,
                        compare_v_contiguous_auto : Bool,
                        compare_p4_stage1 : Bool,
                        quality_top2 : Bool)
  output = QM::Top2.new(-1_i32, Float32::NAN, -1_i32, Float32::NAN)
  elapsed_ms = 0.0_f64
  with_adaptive_probe_env(resident_map, candidate, compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1) do
    started = Time.instant
    if quality_top2
      first_id, first_logit, second_id, second_logit = ML::GGUF::Qwen35CPU.forward_top2(
        weights, token_id, position, state,
      )
      output = QM::Top2.new(first_id, first_logit, second_id, second_logit)
    else
      first_id, first_logit = ML::GGUF::Qwen35CPU.forward_top1(
        weights, token_id, position, state,
      )
      output = QM::Top2.new(first_id, first_logit, -1_i32, -Float32::INFINITY)
    end
    state.layers.each { |layer| layer.position = position + 1 }
    elapsed_ms = (Time.instant - started).total_milliseconds
  end
  {output, elapsed_ms}
end

private def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  midpoint = sorted.size // 2
  sorted.size.odd? ? sorted[midpoint] : (sorted[midpoint - 1] + sorted[midpoint]) / 2.0
end

model_path = DEFAULT_QWEN38_MODEL
resident_map = DEFAULT_RESIDENT_MAP
prompt = "Explain one safe optimization for a compressed GPU KV cache."
prompt_file = nil.as(String?)
sample_count = 10
requested_max_seq = 0
prompt_repeats = 1
chat_mode = true
candidate_first = false
compare_stage2 = false
compare_splitk_chunk = false
compare_direct_qk = false
compare_v_contiguous = false
compare_v_contiguous_auto = false
compare_p4_stage1 = false
quality_top2 = false
quality_top2_production_prefill = false
semantic_coding_quality = false
synthetic_prefix = 0

OptionParser.parse do |parser|
  parser.banner = "Usage: qwen35_adaptive_t8_decode_probe [options] [prompt]"
  parser.on("--model PATH", "Qwen GGUF path") { |value| model_path = value }
  parser.on("--prompt-file PATH", "Read the complete prompt from PATH") { |value| prompt_file = value }
  parser.on("--samples N", "Matched measured decode positions (default: 10)") { |value| sample_count = value.to_i }
  parser.on("--max-seq N", "Cache capacity; 0 selects the minimum") { |value| requested_max_seq = value.to_i }
  parser.on("--repeat-prompt N", "Repeat prompt text before rendering (default: 1)") { |value| prompt_repeats = value.to_i }
  parser.on("--resident-map MAP", "Adaptive resident tier map") { |value| resident_map = value }
  parser.on("--compare-stage2", "Hold T8 on and compare legacy stage2 with automatic policy") { compare_stage2 = true }
  parser.on("--compare-splitk-chunk", "Compare explicit split-K chunks 64 and 60") { compare_splitk_chunk = true }
  parser.on("--compare-direct-qk", "Hold P4/BF16 T8 and fused stage2 on; toggle P4 direct-QK") { compare_direct_qk = true }
  parser.on("--compare-v-contiguous", "Hold legacy shared-K on; toggle contiguous shared-V accumulation") { compare_v_contiguous = true }
  parser.on("--compare-v-contiguous-auto", "Compare explicit contiguous-V off with automatic admission") { compare_v_contiguous_auto = true }
  parser.on("--compare-p4-stage1", "Compare legacy P4 T8 stage1 with forced direct-QK plus contiguous-V") { compare_p4_stage1 = true }
  parser.on("--quality-top2", "Run real-prefix free trajectories with top-2, margin, and output-weight ECS diagnostics; disables timing admission") { quality_top2 = true }
  parser.on("--quality-top2-production-prefill", "Use the production top-1 prefill boundary, then compare top-2 free trajectories; disables timing admission") { quality_top2_production_prefill = true }
  parser.on("--semantic-coding-quality", "Run production-prefill top-2 trajectories until aligned EOS; numerical deltas stay diagnostic and external task scoring remains required") do
    semantic_coding_quality = true
    quality_top2_production_prefill = true
  end
  parser.on("--synthetic-prefix N", "Restore a zero-valued adaptive prefix for decode-only timing") { |value| synthetic_prefix = value.to_i }
  parser.on("--raw", "Do not render the Qwen chat template") { chat_mode = false }
  parser.on("--candidate-first", "Run candidate first during warmup and the first measured pair") { candidate_first = true }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

raise "--prompt-file cannot be combined with a positional prompt" if prompt_file && !ARGV.empty?
if path = prompt_file
  raise "prompt file does not exist: #{path}" unless File.file?(path)
  prompt = File.read(path)
elsif !ARGV.empty?
  prompt = ARGV.join(" ")
end

raise "model does not exist: #{model_path}" unless File.file?(model_path)
raise "--samples must be at least 10" unless sample_count >= 10
if semantic_coding_quality && sample_count < 256
  raise "--semantic-coding-quality requires --samples at least 256 so the pinned coding fixtures can reach EOS"
end
raise "--repeat-prompt must be positive" unless prompt_repeats > 0
raise "--max-seq cannot be negative" if requested_max_seq < 0
raise "--synthetic-prefix cannot be negative" if synthetic_prefix < 0
raise "resident map cannot be empty" if resident_map.strip.empty?
raise "select only one top-2 quality prefill mode" if quality_top2 && quality_top2_production_prefill
quality_top2 ||= quality_top2_production_prefill
raise "--semantic-coding-quality requires chat rendering" if semantic_coding_quality && !chat_mode
comparison_count = {compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1}.count(true)
raise "select only one comparison" if comparison_count > 1
if synthetic_prefix > 0 && !(compare_splitk_chunk || compare_direct_qk || compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1)
  raise "--synthetic-prefix requires a split-K stage1 comparison"
end
raise "--quality-top2 requires a real prompt prefix" if quality_top2 && synthetic_prefix > 0
raise "Metal is unavailable" unless ML::GGUF::Qwen35Metal.available?

tokenizer_gguf = ML::GGUF::GGUFFile.new(model_path, mmap_tensors: false)
weights = nil.as(ML::GGUF::Qwen35Weights?)
baseline_state = nil.as(ML::GGUF::Qwen35CPU::State?)
candidate_state = nil.as(ML::GGUF::Qwen35CPU::State?)

begin
  tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(tokenizer_gguf, model_path)
  vocab_size = tokenizer.vocab.size.to_i32
  weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
  hp = weights.hparams
  raise "probe requires Qwen3.8 head dimension 256" unless hp.head_dim == 256
  device_name = ML::Metal::Device.instance.name
  raise "probe requires exact Apple M2 Max" unless device_name == "Apple M2 Max"

  repeated_prompt = Array.new(prompt_repeats, prompt).join("\n\n")
  model_prompt = chat_mode ? ML::GGUF::Qwen35Chat.render_user_prompt(repeated_prompt, enable_thinking: false) : repeated_prompt
  tokens = tokenizer.encode(model_prompt)
  raise "prompt encoded to zero tokens" if tokens.empty?
  if compare_p4_stage1 && tokens.size < ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT
    raise "--compare-p4-stage1 requires at least #{ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT} prompt tokens; got #{tokens.size}"
  end
  prefix_tokens = synthetic_prefix > 0 ? synthetic_prefix : tokens.size.to_i32
  if compare_v_contiguous_auto &&
     prefix_tokens.to_i64 + 1_i64 < ML::GGUF::QwenQBitAdaptiveMetalPolicy::P4_SPLITK_V_CONTIGUOUS_MIN_VISIBLE_CONTEXT
    raise "--compare-v-contiguous-auto requires at least #{ML::GGUF::QwenQBitAdaptiveMetalPolicy::P4_SPLITK_V_CONTIGUOUS_MIN_VISIBLE_CONTEXT} visible tokens; got #{prefix_tokens.to_i64 + 1_i64}"
  end
  minimum_max_seq = prefix_tokens + sample_count + 2
  max_seq = requested_max_seq == 0 ? minimum_max_seq : requested_max_seq
  raise "prefix plus warmup and samples exceeds --max-seq" if max_seq < minimum_max_seq

  baseline_state, baseline_boundary = prepare_state(
    weights, tokens, max_seq, resident_map, false, compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, synthetic_prefix.to_i32, quality_top2, quality_top2_production_prefill,
  )
  candidate_state, candidate_boundary = prepare_state(
    weights, tokens, max_seq, resident_map, true, compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, synthetic_prefix.to_i32, quality_top2, quality_top2_production_prefill,
  )
  verify_independent_states!(baseline_state, candidate_state, hp)
  semantic_quality_valid = synthetic_prefix == 0
  if semantic_quality_valid
    if quality_top2 && !quality_top2_production_prefill
      verify_finite_top2!("baseline prefill", baseline_boundary, vocab_size)
      verify_finite_top2!("candidate prefill", candidate_boundary, vocab_size)
    else
      verify_finite_top1!("baseline prefill", baseline_boundary.first_id, baseline_boundary.first_logit, vocab_size)
      verify_finite_top1!("candidate prefill", candidate_boundary.first_id, candidate_boundary.first_logit, vocab_size)
    end
  elsif baseline_boundary.first_id < 0 || baseline_boundary.first_id >= vocab_size
    raise "synthetic seed token is outside the vocabulary: #{baseline_boundary.first_id}"
  end
  quality_violations = [] of String
  prefill_logit_delta = (baseline_boundary.first_logit - candidate_boundary.first_logit).abs
  if quality_top2_production_prefill && baseline_boundary.first_id != candidate_boundary.first_id
    quality_violations << "prefill_top1_id_mismatch"
  elsif !quality_top2 && baseline_boundary.first_id != candidate_boundary.first_id
    raise "prefill top-1 mismatch: #{baseline_boundary.first_id} != #{candidate_boundary.first_id}"
  end
  if quality_top2_production_prefill && prefill_logit_delta > QUALITY_LOGIT_TOLERANCE
    quality_violations << "prefill_top1_logit_delta=#{prefill_logit_delta}"
  elsif !quality_top2 && semantic_quality_valid && prefill_logit_delta > QUALITY_LOGIT_TOLERANCE
    raise "prefill logit mismatch: #{prefill_logit_delta}"
  end
  route_certificate = verify_candidate_t8_route!(
    candidate_state, hp, resident_map, compare_stage2, compare_splitk_chunk, compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1,
  )

  embedding_cache = {} of Int32 => Array(Float32)
  quality_steps = [] of DecodeQualitySample
  if quality_top2 && !quality_top2_production_prefill
    prefill_quality = compare_decode_quality(
      "prefill_boundary", -2_i32, baseline_boundary, candidate_boundary, weights, embedding_cache,
    )
    quality_steps << prefill_quality
    emit_decode_quality(prefill_quality)
    append_quality_violations!(quality_violations, "prefill", prefill_quality)
  end

  baseline_input_id = baseline_boundary.first_id
  candidate_input_id = candidate_boundary.first_id
  if quality_top2 && (baseline_input_id == tokenizer.eos_id || candidate_input_id == tokenizer.eos_id)
    raise "prefill boundary reached EOS before the decode comparison"
  end
  position = prefix_tokens.to_i32
  baseline_warm = nil.as(QM::Top2?)
  candidate_warm = nil.as(QM::Top2?)
  trajectories_aligned = baseline_input_id == candidate_input_id
  if candidate_first
    candidate_warm, _ = decode_step(
      weights, candidate_state, candidate_input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
      compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
    )
    baseline_warm, _ = decode_step(
      weights, baseline_state, baseline_input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
      compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
    )
  else
    baseline_warm, _ = decode_step(
      weights, baseline_state, baseline_input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
      compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
    )
    candidate_warm, _ = decode_step(
      weights, candidate_state, candidate_input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
      compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
    )
  end
  baseline_warm = baseline_warm.not_nil!
  candidate_warm = candidate_warm.not_nil!
  warm_quality = nil.as(DecodeQualitySample?)
  if quality_top2
    verify_finite_top2!("baseline warmup", baseline_warm, vocab_size)
    verify_finite_top2!("candidate warmup", candidate_warm, vocab_size)
    warm_quality = compare_decode_quality(
      "warmup", -1_i32, baseline_warm, candidate_warm, weights, embedding_cache, trajectories_aligned,
    )
    quality_steps << warm_quality.not_nil!
    emit_decode_quality(warm_quality)
  end
  unless quality_top2 || baseline_warm.first_id == candidate_warm.first_id
    raise "warmup top-1 mismatch: #{baseline_warm.first_id} != #{candidate_warm.first_id}"
  end
  verify_finite_top1!("baseline warmup", baseline_warm.first_id, baseline_warm.first_logit, vocab_size)
  verify_finite_top1!("candidate warmup", candidate_warm.first_id, candidate_warm.first_logit, vocab_size)
  warm_logit_delta = (baseline_warm.first_logit - candidate_warm.first_logit).abs
  if quality_top2
    append_quality_violations!(quality_violations, "warmup", warm_quality.not_nil!) if trajectories_aligned
  elsif warm_logit_delta > QUALITY_LOGIT_TOLERANCE
    raise "warmup logit mismatch: #{warm_logit_delta}"
  end
  verify_resident_state!(baseline_state, hp, (prefix_tokens + 1).to_i32)
  verify_resident_state!(candidate_state, hp, (prefix_tokens + 1).to_i32)
  trajectories_aligned &&= baseline_warm.first_id == candidate_warm.first_id
  baseline_input_id = baseline_warm.first_id
  candidate_input_id = quality_top2 ? candidate_warm.first_id : baseline_warm.first_id
  position += 1

  samples = [] of DecodeSample
  baseline_output_ids = [baseline_boundary.first_id, baseline_warm.first_id]
  candidate_output_ids = [candidate_boundary.first_id, candidate_warm.first_id]
  termination_reason = "requested_samples_completed"
  sample_count.times do |index|
    if baseline_input_id == tokenizer.eos_id || candidate_input_id == tokenizer.eos_id
      termination_reason = "eos_before_sample_#{index}"
      break
    end

    baseline_top2 = nil.as(QM::Top2?)
    baseline_ms = 0.0_f64
    candidate_top2 = nil.as(QM::Top2?)
    candidate_ms = 0.0_f64
    run_candidate_first = candidate_first == index.even?
    order = run_candidate_first ? "BA" : "AB"

    if run_candidate_first
      candidate_top2, candidate_ms = decode_step(
        weights, candidate_state, candidate_input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
        compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
      )
      baseline_top2, baseline_ms = decode_step(
        weights, baseline_state, baseline_input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
        compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
      )
    else
      baseline_top2, baseline_ms = decode_step(
        weights, baseline_state, baseline_input_id, position, resident_map, false, compare_stage2, compare_splitk_chunk,
        compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
      )
      candidate_top2, candidate_ms = decode_step(
        weights, candidate_state, candidate_input_id, position, resident_map, true, compare_stage2, compare_splitk_chunk,
        compare_direct_qk, compare_v_contiguous, compare_v_contiguous_auto, compare_p4_stage1, quality_top2,
      )
    end

    baseline_top2 = baseline_top2.not_nil!
    candidate_top2 = candidate_top2.not_nil!
    quality = nil.as(DecodeQualitySample?)
    if quality_top2
      verify_finite_top2!("baseline sample #{index}", baseline_top2, vocab_size)
      verify_finite_top2!("candidate sample #{index}", candidate_top2, vocab_size)
      quality = compare_decode_quality(
        "sample", index.to_i32, baseline_top2, candidate_top2, weights, embedding_cache, trajectories_aligned,
      )
      quality_steps << quality.not_nil!
      emit_decode_quality(quality)
    end
    unless quality_top2 || baseline_top2.first_id == candidate_top2.first_id
      raise "measured top-1 mismatch at sample #{index}: #{baseline_top2.first_id} != #{candidate_top2.first_id}"
    end
    verify_finite_top1!("baseline sample #{index}", baseline_top2.first_id, baseline_top2.first_logit, vocab_size)
    verify_finite_top1!("candidate sample #{index}", candidate_top2.first_id, candidate_top2.first_logit, vocab_size)
    logit_delta = (baseline_top2.first_logit - candidate_top2.first_logit).abs
    if quality_top2
      append_quality_violations!(quality_violations, "sample_#{index}", quality.not_nil!) if trajectories_aligned
    elsif logit_delta > QUALITY_LOGIT_TOLERANCE
      raise "measured logit mismatch at sample #{index}: #{logit_delta}"
    end
    expected_tokens = (prefix_tokens + index + 2).to_i32
    verify_resident_state!(baseline_state, hp, expected_tokens)
    verify_resident_state!(candidate_state, hp, expected_tokens)

    samples << DecodeSample.new(
      index.to_i32,
      order,
      baseline_input_id,
      candidate_input_id,
      baseline_top2.first_id,
      candidate_top2.first_id,
      baseline_ms,
      candidate_ms,
      logit_delta,
      quality,
    )
    baseline_output_ids << baseline_top2.first_id
    candidate_output_ids << candidate_top2.first_id
    trajectories_aligned &&= baseline_top2.first_id == candidate_top2.first_id
    baseline_input_id = baseline_top2.first_id
    candidate_input_id = quality_top2 ? candidate_top2.first_id : baseline_top2.first_id
    position += 1
  end

  raise "decode probe produced no measured samples" if samples.empty?
  baseline_values = samples.map(&.baseline_ms)
  candidate_values = samples.map(&.candidate_ms)
  baseline_mean = baseline_values.sum / baseline_values.size
  candidate_mean = candidate_values.sum / candidate_values.size
  improvement_pct = (baseline_mean - candidate_mean) / baseline_mean * 100.0
  wins = samples.count { |sample| sample.candidate_ms < sample.baseline_ms }
  observed_samples = samples.size
  requested_samples_completed = observed_samples == sample_count
  baseline_eos = baseline_input_id == tokenizer.eos_id
  candidate_eos = candidate_input_id == tokenizer.eos_id
  aligned_eos = baseline_eos && candidate_eos
  baseline_eos_step = baseline_output_ids.index(tokenizer.eos_id)
  candidate_eos_step = candidate_output_ids.index(tokenizer.eos_id)
  coding_completion = if aligned_eos && baseline_eos_step == candidate_eos_step
                        "aligned_eos"
                      elsif baseline_eos || candidate_eos
                        "unaligned_eos"
                      else
                        "sample_limit"
                      end
  minimum_wins = (observed_samples * 4 + 4) // 5
  timing_gate_valid = !quality_top2
  gate_passed = timing_gate_valid && requested_samples_completed && RELEASE_BUILD &&
                improvement_pct >= 3.0 && wins >= minimum_wins
  quality_ranked_matches = quality_steps.sum(&.comparison.ranked_matches)
  quality_ranked_count = quality_steps.size * 2
  quality_min_set_overlap = quality_steps.min_of?(&.comparison.set_overlap)
  quality_exact_top1_covered = quality_steps.count(&.comparison.exact_top1_covered)
  quality_exact_top2_covered = quality_steps.count(&.comparison.exact_top2_covered)
  quality_min_token_ecs = quality_steps.min_of?(&.token_ecs)
  aligned_quality_steps = quality_steps.select(&.paired_logits_valid)
  quality_max_second_logit_delta = aligned_quality_steps.max_of?(&.comparison.second_logit_delta)
  quality_max_margin_delta = aligned_quality_steps.max_of?(&.comparison.margin_delta)
  quality_min_baseline_margin = quality_steps.min_of?(&.baseline.margin)
  common_prefix = baseline_output_ids.each_with_index.take_while do |id, index|
    candidate_output_ids[index]? == id
  end.size
  first_divergence_step = common_prefix < baseline_output_ids.size ? common_prefix : nil
  quality_violations << termination_reason unless requested_samples_completed || (semantic_coding_quality && aligned_eos)
  if quality_top2 && first_divergence_step
    quality_violations << "free_run_divergence_step=#{first_divergence_step}"
  end
  quality_gate_passed = quality_top2 && semantic_quality_valid && requested_samples_completed &&
                        quality_violations.empty? &&
                        common_prefix == baseline_output_ids.size
  semantic_trajectory_gate_passed = semantic_coding_quality && semantic_quality_valid && aligned_eos &&
                                    coding_completion == "aligned_eos" &&
                                    common_prefix == baseline_output_ids.size && first_divergence_step.nil? &&
                                    quality_steps.all? do |quality|
                                      quality.paired_logits_valid &&
                                        quality.baseline.first_id == quality.candidate.first_id &&
                                        quality.baseline.second_id == quality.candidate.second_id &&
                                        quality.comparison.ranked_matches == 2 &&
                                        quality.comparison.set_overlap == 2 &&
                                        quality.comparison.exact_top1_covered &&
                                        quality.comparison.exact_top2_covered &&
                                        quality.token_ecs == 1.0
                                    end
  strict_numeric_gate_passed = semantic_trajectory_gate_passed && quality_violations.empty?
  prefill_boundary_mode = if synthetic_prefix > 0
                            "synthetic_seed"
                          elsif quality_top2_production_prefill
                            "production_top1"
                          elsif quality_top2
                            "full_logits_top2"
                          else
                            "production_top1"
                          end
  prompt_sha256 = Digest::SHA256.hexdigest(repeated_prompt.to_slice)
  comparison = if compare_p4_stage1
                 "p4_stage1_bundle"
               elsif compare_v_contiguous_auto
                 "p4_v_contiguous_auto"
               elsif compare_v_contiguous
                 "p4_v_contiguous"
               elsif compare_direct_qk
                 "p4_direct_qk"
               elsif compare_splitk_chunk
                 "splitk_chunk"
               elsif compare_stage2
                 "p4_stage2"
               else
                 "t8_loaders"
               end
  state_source = synthetic_prefix > 0 ? "synthetic_zero_prefix" : "real_prompt_prefill"
  baseline_stage2 = (compare_splitk_chunk || compare_direct_qk || compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) ? "forced_on" : "forced_off"
  candidate_stage2 = if compare_splitk_chunk || compare_direct_qk || compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1
                       "forced_on"
                     elsif compare_stage2
                       "automatic"
                     else
                       "forced_off"
                     end

  puts "qwen35_adaptive_t8_decode_probe"
  puts "  release_build=#{RELEASE_BUILD} device=#{device_name.inspect} comparison=#{comparison} state_source=#{state_source} prompt_sha256=#{prompt_sha256}"
  puts "  semantic_quality_valid=#{semantic_quality_valid}"
  quality_scope = if quality_top2_production_prefill
                    "real_prefix_production_top1_then_free_run_top2"
                  elsif quality_top2
                    "real_prefix_free_run"
                  else
                    "disabled"
                  end
  quality_gate_kind = if quality_top2_production_prefill
                        "production_top1_boundary_plus_aligned_top2_numeric_plus_free_prefix"
                      elsif quality_top2
                        "aligned_top2_numeric_plus_free_prefix"
                      end
  puts "  quality_top2=#{quality_top2} quality_scope=#{quality_scope} prefill_boundary_mode=#{prefill_boundary_mode} timing_gate_valid=#{timing_gate_valid}"
  puts "  semantic_coding_quality=#{semantic_coding_quality} coding_completion=#{coding_completion} baseline_eos=#{baseline_eos} candidate_eos=#{candidate_eos} aligned_eos=#{aligned_eos} semantic_trajectory_gate=#{semantic_trajectory_gate_passed ? "PASS" : "FAIL"} strict_numeric_gate=#{strict_numeric_gate_passed ? "PASS" : "FAIL"}"
  puts "  baseline_stage2=#{baseline_stage2} candidate_stage2=#{candidate_stage2}"
  puts "  p4_stage1_admission=#{compare_p4_stage1 ? "forced_off_vs_forced_on" : "not_compared"}"
  baseline_direct_qk = (compare_direct_qk || compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) ? false : nil
  candidate_direct_qk = if compare_direct_qk || compare_p4_stage1
                          true
                        elsif compare_v_contiguous || compare_v_contiguous_auto
                          false
                        end
  baseline_v_contiguous = (compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) ? false : nil
  candidate_v_contiguous = (compare_v_contiguous || compare_v_contiguous_auto || compare_p4_stage1) ? true : nil
  baseline_v_contiguous_override = baseline_v_contiguous.nil? ? nil : "0"
  candidate_v_contiguous_override = if compare_v_contiguous_auto
                                      nil
                                    elsif candidate_v_contiguous
                                      "1"
                                    end
  baseline_v_contiguous_override_display = baseline_v_contiguous_override || "unset"
  candidate_v_contiguous_override_display = candidate_v_contiguous_override || "unset"
  puts "  baseline_direct_qk=#{baseline_direct_qk} candidate_direct_qk=#{candidate_direct_qk}"
  puts "  baseline_v_contiguous=#{baseline_v_contiguous} candidate_v_contiguous=#{candidate_v_contiguous}"
  puts "  baseline_v_contiguous_override=#{baseline_v_contiguous_override_display} candidate_v_contiguous_override=#{candidate_v_contiguous_override_display}"
  puts "  baseline_splitk_chunk=64 candidate_splitk_chunk=#{compare_splitk_chunk ? 60 : 64}"
  puts "  prompt_tokens=#{tokens.size} seeded_prefix_tokens=#{prefix_tokens} prompt_repeats=#{prompt_repeats} requested_samples=#{sample_count} observed_samples=#{observed_samples} termination_reason=#{termination_reason} max_seq=#{max_seq} full_attention_layers=#{hp.full_attention_layers.size}"
  puts "  prefill_chunk_size=#{PRODUCT_PREFILL_CHUNK_SIZE} append_max_groups=#{PRODUCT_APPEND_MAX_GROUPS} append_cooldown_ms=#{PRODUCT_APPEND_COOLDOWN_MS} pooled_scratch=true gc_guard=true"
  puts "  automatic_t8_min_context=#{ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT}"
  puts "  route_certificate_kind=policy_eligibility"
  puts "  candidate_t8_route_owners=p4:#{route_certificate.p4_t8_owners},bf16:#{route_certificate.bf16_t8_owners},p4_direct_qk:#{route_certificate.p4_direct_qk_owners},p4_v_contiguous:#{route_certificate.p4_v_contiguous_owners} packed_len=#{route_certificate.packed_len}"
  puts "  baseline_mean_ms=#{baseline_mean.round(3)} candidate_mean_ms=#{candidate_mean.round(3)} improvement_pct=#{improvement_pct.round(3)} wins=#{wins}/#{observed_samples} minimum_wins=#{minimum_wins} gate=#{gate_passed ? "PASS" : "FAIL"}"
  puts "  baseline_median_ms=#{median(baseline_values).round(3)} candidate_median_ms=#{median(candidate_values).round(3)} ratio=#{(baseline_mean / candidate_mean).round(5)}x"
  if quality_top2
    puts "  quality_ranked_top2_matches=#{quality_ranked_matches}/#{quality_ranked_count} min_set_overlap=#{quality_min_set_overlap} exact_top1_covered=#{quality_exact_top1_covered}/#{quality_steps.size} exact_top2_covered=#{quality_exact_top2_covered}/#{quality_steps.size}"
    puts "  quality_min_token_ecs=#{quality_min_token_ecs} max_second_logit_delta=#{quality_max_second_logit_delta} max_margin_delta=#{quality_max_margin_delta} min_baseline_margin=#{quality_min_baseline_margin}"
    puts "  quality_gate=#{quality_gate_passed ? "PASS" : "FAIL"} kind=#{quality_gate_kind} tolerance=#{QUALITY_LOGIT_TOLERANCE} aligned_steps=#{aligned_quality_steps.size}/#{quality_steps.size} free_common_prefix=#{common_prefix}/#{baseline_output_ids.size} first_divergence_step=#{first_divergence_step} violations=#{quality_violations.join(';')}"
  end
  baseline_text = tokenizer.decode(baseline_output_ids)
  candidate_text = tokenizer.decode(candidate_output_ids)
  puts "  baseline_output_ids=#{baseline_output_ids.join(',')} baseline_text=#{baseline_text.inspect}"
  puts "  candidate_output_ids=#{candidate_output_ids.join(',')} candidate_text=#{candidate_text.inspect}"

  payload = JSON.build do |json|
    json.object do
      json.field "schema", "qwen-adaptive-t8-decode-ab-v12"
      json.field "comparison", comparison
      json.field "state_source", state_source
      json.field "semantic_quality_valid", semantic_quality_valid
      json.field "quality_top2", quality_top2
      json.field "quality_top2_production_prefill", quality_top2_production_prefill
      json.field "semantic_coding_quality", semantic_coding_quality
      json.field "quality_scope", quality_scope
      json.field "quality_measurement_valid", quality_top2 && semantic_quality_valid
      json.field "quality_gate_kind", quality_gate_kind
      json.field "quality_logit_tolerance", quality_top2 ? QUALITY_LOGIT_TOLERANCE : nil
      json.field "ecs_basis", quality_top2 ? "output.weight" : nil
      json.field "ecs_interpretation", quality_top2 ? "static_output_row_cosine_token_proxy" : nil
      json.field "ecs_equal_ids_short_circuit_to_one", quality_top2
      json.field "semantic_task_scored", false
      json.field "semantic_trajectory_gate_passed", semantic_trajectory_gate_passed
      json.field "strict_numeric_gate_passed", strict_numeric_gate_passed
      json.field "timing_gate_valid", timing_gate_valid
      json.field "prefill_boundary_mode", prefill_boundary_mode
      json.field "prefill_boundary_top2_available", quality_top2 && !quality_top2_production_prefill
      json.field "baseline_prefill_top1_id", baseline_boundary.first_id
      json.field "baseline_prefill_top1_logit", baseline_boundary.first_logit
      json.field "candidate_prefill_top1_id", candidate_boundary.first_id
      json.field "candidate_prefill_top1_logit", candidate_boundary.first_logit
      json.field "baseline_stage2", baseline_stage2
      json.field "candidate_stage2", candidate_stage2
      json.field "p4_stage1_admission", compare_p4_stage1 ? "forced_off_vs_forced_on" : "not_compared"
      json.field "baseline_direct_qk", baseline_direct_qk
      json.field "candidate_direct_qk", candidate_direct_qk
      json.field "baseline_v_contiguous", baseline_v_contiguous
      json.field "candidate_v_contiguous", candidate_v_contiguous
      json.field "baseline_v_contiguous_override", baseline_v_contiguous_override
      json.field "candidate_v_contiguous_override", candidate_v_contiguous_override
      json.field "release_build", RELEASE_BUILD
      json.field "device", device_name
      json.field "model", File.basename(model_path)
      json.field "prompt_sha256", prompt_sha256
      json.field "resident_map", resident_map
      json.field "effective_splitk", true
      json.field "effective_splitk_min_context", 256
      json.field "baseline_splitk_chunk", 64
      json.field "candidate_splitk_chunk", compare_splitk_chunk ? 60 : 64
      json.field "automatic_t8_min_context", ML::GGUF::QwenQBitAdaptiveMetalPolicy::SPLITK_T8_MIN_CONTEXT
      json.field "route_certificate_kind", "policy_eligibility"
      json.field "candidate_p4_t8_owners", route_certificate.p4_t8_owners
      json.field "candidate_bf16_t8_owners", route_certificate.bf16_t8_owners
      json.field "candidate_p4_direct_qk_owners", route_certificate.p4_direct_qk_owners
      json.field "candidate_p4_v_contiguous_owners", route_certificate.p4_v_contiguous_owners
      json.field "prompt_tokens", tokens.size
      json.field "seeded_prefix_tokens", prefix_tokens
      json.field "prompt_repeats", prompt_repeats
      json.field "candidate_first", candidate_first
      json.field "samples", sample_count
      json.field "observed_samples", observed_samples
      json.field "requested_samples_completed", requested_samples_completed
      json.field "termination_reason", termination_reason
      json.field "baseline_eos", baseline_eos
      json.field "candidate_eos", candidate_eos
      json.field "aligned_eos", aligned_eos
      json.field "baseline_eos_step", baseline_eos_step
      json.field "candidate_eos_step", candidate_eos_step
      json.field "coding_completion", coding_completion
      json.field "max_seq", max_seq
      json.field "prefill_chunk_size", PRODUCT_PREFILL_CHUNK_SIZE
      json.field "prefill_append_max_groups", PRODUCT_APPEND_MAX_GROUPS
      json.field "prefill_append_cooldown_ms", PRODUCT_APPEND_COOLDOWN_MS
      json.field "pooled_scratch", true
      json.field "prefill_gc_guard", true
      json.field "baseline_mean_ms", baseline_mean
      json.field "candidate_mean_ms", candidate_mean
      json.field "baseline_median_ms", median(baseline_values)
      json.field "candidate_median_ms", median(candidate_values)
      json.field "improvement_pct", improvement_pct
      json.field "ratio", baseline_mean / candidate_mean
      json.field "candidate_wins", wins
      json.field "minimum_candidate_wins", minimum_wins
      json.field "gate_passed", gate_passed
      json.field "quality_ranked_top2_matches", quality_ranked_matches if quality_top2
      json.field "quality_ranked_top2_count", quality_ranked_count if quality_top2
      json.field "quality_min_set_overlap", quality_min_set_overlap if quality_top2
      json.field "quality_exact_top1_covered", quality_exact_top1_covered if quality_top2
      json.field "quality_exact_top2_covered", quality_exact_top2_covered if quality_top2
      json.field "quality_step_count", quality_steps.size if quality_top2
      json.field "quality_min_token_ecs", quality_min_token_ecs if quality_top2
      json.field "quality_max_second_logit_delta", quality_max_second_logit_delta if quality_top2
      json.field "quality_max_margin_delta", quality_max_margin_delta if quality_top2
      json.field "quality_min_baseline_margin", quality_min_baseline_margin if quality_top2
      json.field "quality_aligned_step_count", aligned_quality_steps.size if quality_top2
      json.field "quality_gate_passed", quality_gate_passed if quality_top2
      json.field "quality_violations", quality_violations if quality_top2
      json.field "free_common_prefix", common_prefix if quality_top2
      json.field "first_divergence_step", first_divergence_step if quality_top2
      json.field "prefill_logit_delta", prefill_logit_delta
      json.field "warm_logit_delta", warm_logit_delta
      json.field "baseline_output_ids", baseline_output_ids
      json.field "candidate_output_ids", candidate_output_ids
      json.field "baseline_text", baseline_text
      json.field "candidate_text", candidate_text
      if quality_top2
        json.field "quality_steps" do
          json.array do
            quality_steps.each do |quality|
              json.object do
                json.field "phase", quality.phase
                json.field "index", quality.index
                json.field "baseline_top1_id", quality.baseline.first_id
                json.field "baseline_top1_logit", quality.baseline.first_logit
                json.field "baseline_top2_id", quality.baseline.second_id
                json.field "baseline_top2_logit", quality.baseline.second_logit
                json.field "baseline_margin", quality.baseline.margin
                json.field "candidate_top1_id", quality.candidate.first_id
                json.field "candidate_top1_logit", quality.candidate.first_logit
                json.field "candidate_top2_id", quality.candidate.second_id
                json.field "candidate_top2_logit", quality.candidate.second_logit
                json.field "candidate_margin", quality.candidate.margin
                json.field "ranked_top2_matches", quality.comparison.ranked_matches
                json.field "top2_set_overlap", quality.comparison.set_overlap
                json.field "exact_top1_covered", quality.comparison.exact_top1_covered
                json.field "exact_top2_covered", quality.comparison.exact_top2_covered
                json.field "first_logit_delta", quality.comparison.first_logit_delta
                json.field "second_logit_delta", quality.comparison.second_logit_delta
                json.field "margin_delta", quality.comparison.margin_delta
                json.field "token_ecs", quality.token_ecs
                json.field "paired_logits_valid", quality.paired_logits_valid
              end
            end
          end
        end
      end
      json.field "pairs" do
        json.array do
          samples.each do |sample|
            json.object do
              json.field "index", sample.index
              json.field "order", sample.order
              json.field "baseline_input_id", sample.baseline_input_id
              json.field "candidate_input_id", sample.candidate_input_id
              json.field "baseline_output_id", sample.baseline_output_id
              json.field "candidate_output_id", sample.candidate_output_id
              json.field "baseline_ms", sample.baseline_ms
              json.field "candidate_ms", sample.candidate_ms
              json.field "logit_delta", sample.logit_delta
              if quality = sample.quality
                json.field "baseline_top2_id", quality.baseline.second_id
                json.field "baseline_top2_logit", quality.baseline.second_logit
                json.field "baseline_margin", quality.baseline.margin
                json.field "candidate_top2_id", quality.candidate.second_id
                json.field "candidate_top2_logit", quality.candidate.second_logit
                json.field "candidate_margin", quality.candidate.margin
                json.field "ranked_top2_matches", quality.comparison.ranked_matches
                json.field "top2_set_overlap", quality.comparison.set_overlap
                json.field "exact_top1_covered", quality.comparison.exact_top1_covered
                json.field "exact_top2_covered", quality.comparison.exact_top2_covered
                json.field "second_logit_delta", quality.comparison.second_logit_delta
                json.field "margin_delta", quality.comparison.margin_delta
                json.field "token_ecs", quality.token_ecs
                json.field "paired_logits_valid", quality.paired_logits_valid
              end
            end
          end
        end
      end
    end
  end
  puts "QBIT_T8_DECODE_JSON=#{payload}"
  STDOUT.flush
  if semantic_coding_quality && !semantic_trajectory_gate_passed
    raise "semantic coding trajectory gate failed: #{quality_violations.join("; ")}"
  elsif quality_top2 && !quality_gate_passed
    raise "quality gate failed: #{quality_violations.join("; ")}"
  end
ensure
  release_state!(candidate_state)
  release_state!(baseline_state)
  weights.try(&.close)
  tokenizer_gguf.close
end
