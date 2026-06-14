require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL       = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_BASE_ENV    = "DIFFUSION_GEMMA_C8_RESIDENT_DECODE_POLICY=1 DIFFUSION_GEMMA_PROMPT_CACHE_POLICY=1"
DEFAULT_VARIANT_ENV = "#{DEFAULT_BASE_ENV} DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_GRAPH=1 DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_BATCH_GRAPH_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM_MAX_CANVAS=16 DIFFUSION_GEMMA_FFN_RESIDENT_SCRATCH=1 DIFFUSION_GEMMA_FFN_RESIDENT_GRAPH_CACHE=1"

struct ArmEnv
  getter raw : String
  getter pairs : Array(Tuple(String, String))

  def initialize(@raw)
    @pairs = parse_pairs(raw)
  end

  private def parse_pairs(raw : String) : Array(Tuple(String, String))
    pairs = [] of Tuple(String, String)
    raw.split(/\s+/).reject(&.empty?).each do |token|
      key, value = token.split("=", 2)
      raise "env token must be KEY=VALUE, got #{token.inspect}" unless value
      raise "env key must not be empty" if key.empty?
      pairs << {key, value}
    end
    pairs
  end
end

struct ArmResult
  getter cache : ML::GGUF::DiffusionGemmaCPU::PromptLayerCache
  getter timing : ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePredictionTiming
  getter cache_ms : Float64
  getter predict_ms : Float64

  def initialize(@cache, @timing, @cache_ms, @predict_ms)
  end
end

struct FullVocabTop1ArmResult
  getter cache : ML::GGUF::DiffusionGemmaCPU::PromptLayerCache
  getter decode : ML::GGUF::DiffusionGemmaCPU::DecodeCanvasRowsTiming
  getter top1s : Array(ML::GGUF::DiffusionGemmaCPU::OutputTop1)
  getter cache_ms : Float64
  getter predict_ms : Float64
  getter output_head_ms : Float64

  def initialize(@cache, @decode, @top1s, @cache_ms, @predict_ms, @output_head_ms)
  end
end

struct FullVocabTop2ArmResult
  getter cache : ML::GGUF::DiffusionGemmaCPU::PromptLayerCache
  getter decode : ML::GGUF::DiffusionGemmaCPU::DecodeCanvasRowsTiming
  getter top2s : Array(ML::GGUF::DiffusionGemmaCPU::OutputTop2)
  getter cache_ms : Float64
  getter predict_ms : Float64
  getter output_head_ms : Float64

  def initialize(@cache, @decode, @top2s, @cache_ms, @predict_ms, @output_head_ms)
  end
end

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 16
canvas_len = 8
prompt_token = 1
canvas_token = 0
token_windows_arg = nil.as(String?)
certificate_mode = "bounded"
candidate_count = 64
candidate_offsets_arg = "0"
candidate_stride = 1
max_candidate_row_size = 8192
max_layers = 30
seed = 7
temp_inv = 1.0_f32
base_env = ArmEnv.new(DEFAULT_BASE_ENV)
variant_env = ArmEnv.new(DEFAULT_VARIANT_ENV)
require_argmax_match = false
require_sampled_match = false
min_base_logit_margin = nil.as(Float64?)
min_variant_logit_margin = nil.as(Float64?)
max_logit_delta = nil.as(Float64?)

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_prompt_output_cert_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 16)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Synthetic canvas length (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--canvas-token ID", "Synthetic canvas start token id (default: 0)") { |v| canvas_token = v.to_i }
  p.on("--token-windows LIST", "Comma/space separated prompt:canvas token starts, e.g. 1:0,17:100") { |v| token_windows_arg = v }
  p.on("--certificate-mode MODE", "bounded, full-vocab-top1-metal/cpu, or full-vocab-top2-metal/cpu (default: bounded)") { |v| certificate_mode = v }
  p.on("--full-vocab-top1-metal", "Use Metal full-vocab top1 argmax certificate instead of bounded candidates") { certificate_mode = "full-vocab-top1-metal" }
  p.on("--full-vocab-top1-cpu", "Use CPU full-vocab top1 argmax certificate instead of bounded candidates") { certificate_mode = "full-vocab-top1-cpu" }
  p.on("--full-vocab-top2-metal", "Use Metal full-vocab top2 argmax+margin certificate instead of bounded candidates") { certificate_mode = "full-vocab-top2-metal" }
  p.on("--full-vocab-top2-cpu", "Use CPU full-vocab top2 argmax+margin certificate instead of bounded candidates") { certificate_mode = "full-vocab-top2-cpu" }
  p.on("--candidate-count N", "Candidate ids per candidate span, generated from the row token plus each offset (default: 64)") { |v| candidate_count = v.to_i }
  p.on("--candidate-offsets LIST", "Comma/space separated candidate span offsets from each canvas token (default: 0)") { |v| candidate_offsets_arg = v }
  p.on("--candidate-stride N", "Token stride inside each candidate span (default: 1)") { |v| candidate_stride = v.to_i }
  p.on("--max-candidate-row-size N", "Fail when merged per-row candidate count exceeds N; 0 disables (default: 8192)") { |v| max_candidate_row_size = v.to_i }
  p.on("--max-layers N", "Prompt-cache/decode layer count (default: 30)") { |v| max_layers = v.to_i }
  p.on("--seed N", "Deterministic sample-u seed for sampled-token comparison (default: 7)") { |v| seed = v.to_i }
  p.on("--temp-inv F", "Inverse sampling temperature for bounded logits (default: 1.0)") { |v| temp_inv = v.to_f32 }
  p.on("--base-env ENV", "Whitespace-separated KEY=VALUE env for base arm") { |v| base_env = ArmEnv.new(v) }
  p.on("--variant-env ENV", "Whitespace-separated KEY=VALUE env for variant arm") { |v| variant_env = ArmEnv.new(v) }
  p.on("--require-argmax-match", "Exit 4 when any canvas row changes bounded argmax") { require_argmax_match = true }
  p.on("--require-sampled-match", "Exit 4 when any canvas row changes deterministic sampled token") { require_sampled_match = true }
  p.on("--min-base-logit-margin F", "Exit 4 when the minimum base top1/top2 logit margin is below F") { |v| min_base_logit_margin = v.to_f64 }
  p.on("--min-variant-logit-margin F", "Exit 4 when the minimum variant top1/top2 logit margin is below F") { |v| min_variant_logit_margin = v.to_f64 }
  p.on("--max-logit-delta F", "Exit 4 when any candidate logit absolute delta exceeds F") { |v| max_logit_delta = v.to_f64 }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

def apply_env(arm_env : ArmEnv) : Hash(String, String?)
  old = Hash(String, String?).new
  arm_env.pairs.each do |key, value|
    old[key] = ENV[key]?
    ENV[key] = value
  end
  old
end

def restore_env(old : Hash(String, String?)) : Nil
  old.each do |key, value|
    if value
      ENV[key] = value.not_nil!
    else
      ENV.delete(key)
    end
  end
end

def generated_token_sequence(default_token : Int32, count : Int32, vocab_size : Int32, label : String) : Array(Int32)
  raise "#{label} must be positive" unless count > 0
  raise "#{label} exceeds vocab size" if count > vocab_size
  Array(Int32).new(count) { |i| (default_token + i) % vocab_size }
end

def parse_token_windows(raw : String) : Array(Tuple(Int32, Int32))
  windows = [] of Tuple(Int32, Int32)
  raw.split(/[,\s]+/).reject(&.empty?).each do |entry|
    prompt_raw, canvas_raw = entry.split(":", 2)
    raise "--token-windows entries must be prompt:canvas, got #{entry.inspect}" unless canvas_raw
    windows << {prompt_raw.to_i, canvas_raw.to_i}
  end
  raise "--token-windows must contain at least one entry" if windows.empty?
  windows
end

def parse_int_list(raw : String, label : String) : Array(Int32)
  values = raw.split(/[,\s]+/).reject(&.empty?).map(&.to_i)
  raise "#{label} must contain at least one integer" if values.empty?
  values
end

def wrap_vocab_id(value : Int64, vocab_size : Int32) : Int32
  wrapped = value % vocab_size
  wrapped += vocab_size if wrapped < 0
  wrapped.to_i32
end

def generated_candidate_rows(canvas_tokens : Array(Int32),
                             count : Int32,
                             vocab_size : Int32,
                             offsets : Array(Int32),
                             stride : Int32,
                             max_row_size : Int32) : Array(Array(Int32))
  raise "candidate count must be positive" unless count > 0
  raise "candidate count exceeds vocab size" if count > vocab_size
  raise "candidate stride must be positive" unless stride > 0
  raise "max candidate row size must be non-negative" if max_row_size < 0
  raise "candidate offsets must not be empty" if offsets.empty?

  canvas_tokens.map do |token_id|
    candidates = [] of Int32
    offsets.each do |offset|
      count.times do |i|
        candidates << wrap_vocab_id(token_id.to_i64 + offset.to_i64 + i.to_i64 * stride.to_i64, vocab_size)
      end
    end
    candidates.uniq!
    candidates.sort!
    if max_row_size > 0 && candidates.size > max_row_size
      raise "candidate row size #{candidates.size} exceeds --max-candidate-row-size #{max_row_size}"
    end
    candidates
  end
end

def prompt_rows_from_tokens(weights : ML::GGUF::DiffusionGemmaWeights, tokens : Array(Int32)) : Array(Float32)
  rows = [] of Float32
  tokens.each do |token_id|
    rows.concat(ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(weights, token_id))
  end
  rows
end

def checksum_rows(rows : Array(Float32)) : Float64
  stride = Math.max(1, rows.size // 257)
  sum = 0.0
  index = 0
  while index < rows.size
    sum += rows[index].to_f64
    index += stride
  end
  sum
end

def diff_stats(a : Array(Float32), b : Array(Float32)) : NamedTuple(max_abs: Float64, mean_abs: Float64, checksum_a: Float64, checksum_b: Float64, checksum_delta: Float64)
  raise "row size mismatch #{a.size} != #{b.size}" unless a.size == b.size
  max_abs = 0.0
  sum_abs = 0.0
  checksum_a = 0.0
  checksum_b = 0.0
  a.size.times do |i|
    av = a[i].to_f64
    bv = b[i].to_f64
    abs = (av - bv).abs
    max_abs = abs if abs > max_abs
    sum_abs += abs
    checksum_a += av
    checksum_b += bv
  end
  {
    max_abs:        max_abs,
    mean_abs:       a.empty? ? 0.0 : sum_abs / a.size,
    checksum_a:     checksum_a,
    checksum_b:     checksum_b,
    checksum_delta: (checksum_a - checksum_b).abs,
  }
end

def rank_order(prediction : ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction) : Array(Int32)
  order = (0...prediction.candidate_token_ids.size).to_a
  order.sort! do |a, b|
    cmp = prediction.logits[b] <=> prediction.logits[a]
    cmp == 0 ? prediction.candidate_token_ids[a] <=> prediction.candidate_token_ids[b] : cmp
  end
  order
end

def margin(values : Array(Float32), order : Array(Int32)) : Float64
  return 0.0 if order.size < 2
  values[order[0]].to_f64 - values[order[1]].to_f64
end

def prediction_delta(a : ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction,
                     b : ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction) : NamedTuple(max_logit_abs: Float64, max_prob_abs: Float64)
  raise "candidate row size mismatch" unless a.candidate_token_ids == b.candidate_token_ids
  max_logit = 0.0
  max_prob = 0.0
  a.candidate_token_ids.size.times do |i|
    logit_abs = (a.logits[i].to_f64 - b.logits[i].to_f64).abs
    prob_abs = (a.probabilities[i].to_f64 - b.probabilities[i].to_f64).abs
    max_logit = logit_abs if logit_abs > max_logit
    max_prob = prob_abs if prob_abs > max_prob
  end
  {max_logit_abs: max_logit, max_prob_abs: max_prob}
end

def run_arm(weights : ML::GGUF::DiffusionGemmaWeights,
            prompt_rows : Array(Float32),
            canvas_rows : Array(Float32),
            mask : ML::GGUF::DiffusionGemmaAttentionMask,
            candidate_rows : Array(Array(Int32)),
            sample_us : Array(Float32),
            max_layers : Int32,
            temp_inv : Float32,
            env : ArmEnv) : ArmResult
  old = apply_env(env)
  begin
    cache_t0 = Time.instant
    cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      materialize_final_rows: true,
    )
    cache_ms = (Time.instant - cache_t0).total_milliseconds

    predict_t0 = Time.instant
    timing = ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_predictions_timed(
      weights: weights,
      canvas_rows: canvas_rows,
      mask: mask,
      prompt_cache: cache,
      candidate_token_ids_by_canvas_row: candidate_rows,
      max_layers: max_layers,
      temp_inv: temp_inv,
      sample_us: sample_us,
    )
    predict_ms = (Time.instant - predict_t0).total_milliseconds
    ArmResult.new(cache, timing, cache_ms, predict_ms)
  ensure
    restore_env(old)
  end
end

def run_full_vocab_top1_arm(weights : ML::GGUF::DiffusionGemmaWeights,
                            prompt_rows : Array(Float32),
                            canvas_rows : Array(Float32),
                            mask : ML::GGUF::DiffusionGemmaAttentionMask,
                            max_layers : Int32,
                            env : ArmEnv,
                            use_metal : Bool) : FullVocabTop1ArmResult
  old = apply_env(env)
  begin
    cache_t0 = Time.instant
    cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      materialize_final_rows: true,
    )
    cache_ms = (Time.instant - cache_t0).total_milliseconds

    predict_t0 = Time.instant
    decode = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache_timed(
      weights: weights,
      canvas_rows: canvas_rows,
      mask: mask,
      prompt_cache: cache,
      max_layers: max_layers,
    )

    hp = weights.hparams
    output_t0 = Time.instant
    top1s = Array(ML::GGUF::DiffusionGemmaCPU::OutputTop1).new(mask.canvas_len) do |canvas_pos|
      hidden = decode.rows[canvas_pos * hp.n_embd, hp.n_embd]
      if use_metal
        top1 = ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_metal(weights, hidden)
        raise "Metal full-vocab top1 unavailable" unless top1
        top1.not_nil!
      else
        ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_cpu(weights, hidden)
      end
    end
    output_head_ms = (Time.instant - output_t0).total_milliseconds
    predict_ms = (Time.instant - predict_t0).total_milliseconds
    FullVocabTop1ArmResult.new(cache, decode, top1s, cache_ms, predict_ms, output_head_ms)
  ensure
    restore_env(old)
  end
end

def run_full_vocab_top2_arm(weights : ML::GGUF::DiffusionGemmaWeights,
                            prompt_rows : Array(Float32),
                            canvas_rows : Array(Float32),
                            mask : ML::GGUF::DiffusionGemmaAttentionMask,
                            max_layers : Int32,
                            env : ArmEnv,
                            use_metal : Bool) : FullVocabTop2ArmResult
  old = apply_env(env)
  begin
    cache_t0 = Time.instant
    cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      materialize_final_rows: true,
    )
    cache_ms = (Time.instant - cache_t0).total_milliseconds

    predict_t0 = Time.instant
    decode = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache_timed(
      weights: weights,
      canvas_rows: canvas_rows,
      mask: mask,
      prompt_cache: cache,
      max_layers: max_layers,
    )

    hp = weights.hparams
    output_t0 = Time.instant
    top2s = Array(ML::GGUF::DiffusionGemmaCPU::OutputTop2).new(mask.canvas_len) do |canvas_pos|
      hidden = decode.rows[canvas_pos * hp.n_embd, hp.n_embd]
      if use_metal
        top2 = ML::GGUF::DiffusionGemmaCPU.output_top2_full_vocab_metal(weights, hidden)
        raise "Metal full-vocab top2 unavailable" unless top2
        top2.not_nil!
      else
        ML::GGUF::DiffusionGemmaCPU.output_top2_full_vocab_cpu(weights, hidden)
      end
    end
    output_head_ms = (Time.instant - output_t0).total_milliseconds
    predict_ms = (Time.instant - predict_t0).total_milliseconds
    FullVocabTop2ArmResult.new(cache, decode, top2s, cache_ms, predict_ms, output_head_ms)
  ensure
    restore_env(old)
  end
end

def format_f64(value : Float64) : String
  "%.9f" % value
end

def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  mid = sorted.size // 2
  sorted.size.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt-len must be positive" unless prompt_len > 0
raise "--canvas-len must be positive" unless canvas_len > 0
valid_certificate_modes = ["bounded", "full-vocab-top1-metal", "full-vocab-top1-cpu", "full-vocab-top2-metal", "full-vocab-top2-cpu"]
raise "--certificate-mode must be one of #{valid_certificate_modes.join(", ")}" unless valid_certificate_modes.includes?(certificate_mode)
raise "--candidate-count must be positive" unless candidate_count > 0
raise "--candidate-stride must be positive" unless candidate_stride > 0
raise "--max-candidate-row-size must be non-negative" if max_candidate_row_size < 0
raise "--max-layers must be positive" unless max_layers > 0
raise "--temp-inv must be finite and positive" unless temp_inv.finite? && temp_inv > 0.0_f32
if certificate_mode != "bounded"
  raise "--require-sampled-match is incompatible with #{certificate_mode}" if require_sampled_match
  unless certificate_mode.starts_with?("full-vocab-top2-")
    raise "--min-base-logit-margin is incompatible with #{certificate_mode}" if min_base_logit_margin
    raise "--min-variant-logit-margin is incompatible with #{certificate_mode}" if min_variant_logit_margin
  end
end
if threshold = min_base_logit_margin
  raise "--min-base-logit-margin must be finite" unless threshold.finite?
end
if threshold = min_variant_logit_margin
  raise "--min-variant-logit-margin must be finite" unless threshold.finite?
end
if threshold = max_logit_delta
  raise "--max-logit-delta must be finite and non-negative" unless threshold.finite? && threshold >= 0.0
end

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length

windows = token_windows_arg ? parse_token_windows(token_windows_arg.not_nil!) : [{prompt_token, canvas_token}]
candidate_offsets = parse_int_list(candidate_offsets_arg, "--candidate-offsets")
windows.each do |window|
  p_start = window[0]
  c_start = window[1]
  raise "prompt token start out of range" if p_start < 0 || p_start >= hp.vocab_size
  raise "canvas token start out of range" if c_start < 0 || c_start >= hp.vocab_size
end

puts "# load_ms=#{format_f64(load_ms)}"
puts "# base_env=#{base_env.raw.empty? ? "<empty>" : base_env.raw}"
puts "# variant_env=#{variant_env.raw.empty? ? "<empty>" : variant_env.raw}"
puts "# certificate_mode=#{certificate_mode}"
puts [
  "kind",
  "window",
  "prompt_token",
  "canvas_token",
  "row",
  "candidate_count",
  "base_argmax",
  "variant_argmax",
  "argmax_match",
  "base_sampled",
  "variant_sampled",
  "sampled_match",
  "base_logit_margin",
  "variant_logit_margin",
  "base_prob_margin",
  "variant_prob_margin",
  "max_logit_abs_delta",
  "max_prob_abs_delta",
].join('\t')

if certificate_mode != "bounded"
  use_metal = certificate_mode.ends_with?("-metal")
  use_top2 = certificate_mode.starts_with?("full-vocab-top2-")
  aggregate_rows = 0
  aggregate_argmax_matches = 0
  aggregate_max_logit_abs_delta = 0.0
  aggregate_min_base_logit_margin = Float64::INFINITY
  aggregate_min_variant_logit_margin = Float64::INFINITY
  cache_speedups = [] of Float64
  predict_speedups = [] of Float64

  windows.each_with_index do |window, window_index|
    prompt_start = window[0]
    canvas_start = window[1]
    prompt_tokens = generated_token_sequence(prompt_start, prompt_len, hp.vocab_size, "--prompt-len")
    canvas_tokens = generated_token_sequence(canvas_start, canvas_len, hp.vocab_size, "--canvas-len")
    prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
    canvas_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(weights, canvas_tokens)
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)

    if use_top2
      base_top2_result = run_full_vocab_top2_arm(weights, prompt_rows, canvas_rows, mask, max_layers, base_env, use_metal)
      variant_top2_result = run_full_vocab_top2_arm(weights, prompt_rows, canvas_rows, mask, max_layers, variant_env, use_metal)
      base_top1s = base_top2_result.top2s.map(&.top1)
      variant_top1s = variant_top2_result.top2s.map(&.top1)
      base = base_top2_result
      variant = variant_top2_result
    else
      base_top1_result = run_full_vocab_top1_arm(weights, prompt_rows, canvas_rows, mask, max_layers, base_env, use_metal)
      variant_top1_result = run_full_vocab_top1_arm(weights, prompt_rows, canvas_rows, mask, max_layers, variant_env, use_metal)
      base_top1s = base_top1_result.top1s
      variant_top1s = variant_top1_result.top1s
      base = base_top1_result
      variant = variant_top1_result
    end
    hidden = diff_stats(base.cache.final_rows, variant.cache.final_rows)
    cache_speedup = base.cache_ms / variant.cache_ms
    predict_speedup = base.predict_ms / variant.predict_ms
    cache_speedups << cache_speedup
    predict_speedups << predict_speedup

    puts [
      "timing_summary",
      "window=#{window_index}",
      "prompt_token=#{prompt_start}",
      "canvas_token=#{canvas_start}",
      "prompt_len=#{prompt_len}",
      "canvas_len=#{canvas_len}",
      "certificate_mode=#{certificate_mode}",
      "candidate_count=#{hp.vocab_size}",
      "candidate_offsets=full_vocab",
      "candidate_stride=1",
      "min_candidate_row_size=#{hp.vocab_size}",
      "max_candidate_row_size=#{hp.vocab_size}",
      "max_layers=#{max_layers}",
      "base_cache_ms=#{format_f64(base.cache_ms)}",
      "variant_cache_ms=#{format_f64(variant.cache_ms)}",
      "cache_speedup=#{format_f64(cache_speedup)}",
      "base_predict_ms=#{format_f64(base.predict_ms)}",
      "variant_predict_ms=#{format_f64(variant.predict_ms)}",
      "predict_speedup=#{format_f64(predict_speedup)}",
      "base_decode_stack_ms=#{format_f64(base.predict_ms - base.output_head_ms)}",
      "variant_decode_stack_ms=#{format_f64(variant.predict_ms - variant.output_head_ms)}",
      "base_output_head_ms=#{format_f64(base.output_head_ms)}",
      "variant_output_head_ms=#{format_f64(variant.output_head_ms)}",
    ].join('\t')
    puts [
      "hidden_summary",
      "window=#{window_index}",
      "prompt_token=#{prompt_start}",
      "canvas_token=#{canvas_start}",
      "max_abs=#{format_f64(hidden[:max_abs])}",
      "mean_abs=#{format_f64(hidden[:mean_abs])}",
      "checksum_base=#{format_f64(hidden[:checksum_a])}",
      "checksum_variant=#{format_f64(hidden[:checksum_b])}",
      "checksum_delta=#{format_f64(hidden[:checksum_delta])}",
      "sampled_checksum_base=#{format_f64(checksum_rows(base.cache.final_rows))}",
      "sampled_checksum_variant=#{format_f64(checksum_rows(variant.cache.final_rows))}",
    ].join('\t')

    argmax_matches = 0
    max_logit_abs_delta = 0.0
    min_base_margin = Float64::INFINITY
    min_variant_margin = Float64::INFINITY
    base_top1s.each_with_index do |base_top1, row|
      variant_top1 = variant_top1s[row]
      argmax_match = base_top1.token_id == variant_top1.token_id
      argmax_matches += 1 if argmax_match
      aggregate_argmax_matches += 1 if argmax_match
      aggregate_rows += 1
      logit_delta = (base_top1.logit.to_f64 - variant_top1.logit.to_f64).abs
      if use_top2
        base_top2 = base.as(FullVocabTop2ArmResult).top2s[row]
        variant_top2 = variant.as(FullVocabTop2ArmResult).top2s[row]
        base_margin = base_top2.margin.to_f64
        variant_margin = variant_top2.margin.to_f64
        min_base_margin = base_margin if base_margin < min_base_margin
        min_variant_margin = variant_margin if variant_margin < min_variant_margin
        aggregate_min_base_logit_margin = base_margin if base_margin < aggregate_min_base_logit_margin
        aggregate_min_variant_logit_margin = variant_margin if variant_margin < aggregate_min_variant_logit_margin
        second_delta = (base_top2.second_logit.to_f64 - variant_top2.second_logit.to_f64).abs
        logit_delta = second_delta if second_delta > logit_delta
      end
      max_logit_abs_delta = logit_delta if logit_delta > max_logit_abs_delta
      aggregate_max_logit_abs_delta = logit_delta if logit_delta > aggregate_max_logit_abs_delta

      puts [
        "row",
        window_index.to_s,
        prompt_start.to_s,
        canvas_start.to_s,
        row.to_s,
        hp.vocab_size.to_s,
        base_top1.token_id.to_s,
        variant_top1.token_id.to_s,
        argmax_match.to_s,
        "n/a",
        "n/a",
        "n/a",
        use_top2 ? format_f64(base.as(FullVocabTop2ArmResult).top2s[row].margin.to_f64) : "n/a",
        use_top2 ? format_f64(variant.as(FullVocabTop2ArmResult).top2s[row].margin.to_f64) : "n/a",
        "n/a",
        "n/a",
        format_f64(logit_delta),
        "n/a",
      ].join('\t')
    end

    puts [
      "cert_summary",
      "window=#{window_index}",
      "prompt_token=#{prompt_start}",
      "canvas_token=#{canvas_start}",
      "certificate_mode=#{certificate_mode}",
      "argmax_matches=#{argmax_matches}/#{canvas_len}",
      "sampled_matches=n/a",
      "all_argmax_match=#{argmax_matches == canvas_len}",
      "all_sampled_match=n/a",
      "min_base_logit_margin=#{use_top2 ? format_f64(min_base_margin) : "n/a"}",
      "min_variant_logit_margin=#{use_top2 ? format_f64(min_variant_margin) : "n/a"}",
      "min_base_prob_margin=n/a",
      "min_variant_prob_margin=n/a",
      "max_logit_abs_delta=#{format_f64(max_logit_abs_delta)}",
      "max_prob_abs_delta=n/a",
    ].join('\t')
  end

  all_argmax_match = aggregate_argmax_matches == aggregate_rows
  puts [
    "aggregate_summary",
    "windows=#{windows.size}",
    "rows=#{aggregate_rows}",
    "argmax_matches=#{aggregate_argmax_matches}/#{aggregate_rows}",
    "sampled_matches=n/a",
    "candidate_offsets=full_vocab",
    "candidate_stride=1",
    "min_candidate_row_size=#{hp.vocab_size}",
    "max_candidate_row_size=#{hp.vocab_size}",
    "certificate_mode=#{certificate_mode}",
    "all_argmax_match=#{all_argmax_match}",
    "all_sampled_match=n/a",
    "min_base_logit_margin=#{use_top2 ? format_f64(aggregate_min_base_logit_margin) : "n/a"}",
    "min_variant_logit_margin=#{use_top2 ? format_f64(aggregate_min_variant_logit_margin) : "n/a"}",
    "min_base_prob_margin=n/a",
    "min_variant_prob_margin=n/a",
    "max_logit_abs_delta=#{format_f64(aggregate_max_logit_abs_delta)}",
    "max_prob_abs_delta=n/a",
    "median_cache_speedup=#{format_f64(median(cache_speedups))}",
    "median_predict_speedup=#{format_f64(median(predict_speedups))}",
  ].join('\t')

  failures = [] of String
  failures << "argmax" if require_argmax_match && !all_argmax_match
  if threshold = min_base_logit_margin
    failures << "base_margin" if aggregate_min_base_logit_margin < threshold
  end
  if threshold = min_variant_logit_margin
    failures << "variant_margin" if aggregate_min_variant_logit_margin < threshold
  end
  if threshold = max_logit_delta
    failures << "logit_delta" if aggregate_max_logit_abs_delta > threshold
  end

  unless failures.empty?
    STDERR.puts "output_cert status=fail certificate_mode=#{certificate_mode} failures=#{failures.join(",")} argmax_matches=#{aggregate_argmax_matches}/#{aggregate_rows} sampled_matches=n/a"
    exit 4
  end

  STDERR.puts "output_cert status=#{all_argmax_match ? "argmax_match" : "argmax_mismatch"} certificate_mode=#{certificate_mode} argmax_matches=#{aggregate_argmax_matches}/#{aggregate_rows} sampled_matches=n/a"
  exit
end

aggregate_rows = 0
aggregate_argmax_matches = 0
aggregate_sampled_matches = 0
aggregate_max_logit_abs_delta = 0.0
aggregate_max_prob_abs_delta = 0.0
aggregate_min_base_logit_margin = Float64::INFINITY
aggregate_min_variant_logit_margin = Float64::INFINITY
aggregate_min_base_prob_margin = Float64::INFINITY
aggregate_min_variant_prob_margin = Float64::INFINITY
cache_speedups = [] of Float64
predict_speedups = [] of Float64
candidate_row_sizes = [] of Int32

windows.each_with_index do |window, window_index|
  prompt_start = window[0]
  canvas_start = window[1]
  prompt_tokens = generated_token_sequence(prompt_start, prompt_len, hp.vocab_size, "--prompt-len")
  canvas_tokens = generated_token_sequence(canvas_start, canvas_len, hp.vocab_size, "--canvas-len")
  prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
  canvas_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(weights, canvas_tokens)
  candidate_rows = generated_candidate_rows(canvas_tokens, candidate_count, hp.vocab_size, candidate_offsets, candidate_stride, max_candidate_row_size)
  candidate_row_sizes.concat(candidate_rows.map(&.size))
  sample_us = ML::GGUF::DiffusionGemmaCPU.sample_u_rows(seed, canvas_len)
  mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)

  base = run_arm(weights, prompt_rows, canvas_rows, mask, candidate_rows, sample_us, max_layers, temp_inv, base_env)
  variant = run_arm(weights, prompt_rows, canvas_rows, mask, candidate_rows, sample_us, max_layers, temp_inv, variant_env)
  hidden = diff_stats(base.cache.final_rows, variant.cache.final_rows)
  cache_speedup = base.cache_ms / variant.cache_ms
  predict_speedup = base.predict_ms / variant.predict_ms
  cache_speedups << cache_speedup
  predict_speedups << predict_speedup

  puts [
    "timing_summary",
    "window=#{window_index}",
    "prompt_token=#{prompt_start}",
    "canvas_token=#{canvas_start}",
    "prompt_len=#{prompt_len}",
    "canvas_len=#{canvas_len}",
    "candidate_count=#{candidate_count}",
    "candidate_offsets=#{candidate_offsets.join(",")}",
    "candidate_stride=#{candidate_stride}",
    "min_candidate_row_size=#{candidate_rows.map(&.size).min}",
    "max_candidate_row_size=#{candidate_rows.map(&.size).max}",
    "max_layers=#{max_layers}",
    "base_cache_ms=#{format_f64(base.cache_ms)}",
    "variant_cache_ms=#{format_f64(variant.cache_ms)}",
    "cache_speedup=#{format_f64(cache_speedup)}",
    "base_predict_ms=#{format_f64(base.predict_ms)}",
    "variant_predict_ms=#{format_f64(variant.predict_ms)}",
    "predict_speedup=#{format_f64(predict_speedup)}",
    "base_decode_stack_ms=#{format_f64(base.timing.decode_stack_ms)}",
    "variant_decode_stack_ms=#{format_f64(variant.timing.decode_stack_ms)}",
    "base_output_head_ms=#{format_f64(base.timing.output_head_ms)}",
    "variant_output_head_ms=#{format_f64(variant.timing.output_head_ms)}",
  ].join('\t')
  puts [
    "hidden_summary",
    "window=#{window_index}",
    "prompt_token=#{prompt_start}",
    "canvas_token=#{canvas_start}",
    "max_abs=#{format_f64(hidden[:max_abs])}",
    "mean_abs=#{format_f64(hidden[:mean_abs])}",
    "checksum_base=#{format_f64(hidden[:checksum_a])}",
    "checksum_variant=#{format_f64(hidden[:checksum_b])}",
    "checksum_delta=#{format_f64(hidden[:checksum_delta])}",
    "sampled_checksum_base=#{format_f64(checksum_rows(base.cache.final_rows))}",
    "sampled_checksum_variant=#{format_f64(checksum_rows(variant.cache.final_rows))}",
  ].join('\t')

  argmax_matches = 0
  sampled_matches = 0
  max_logit_abs_delta = 0.0
  max_prob_abs_delta = 0.0
  min_base_margin = Float64::INFINITY
  min_variant_margin = Float64::INFINITY
  min_base_prob_margin = Float64::INFINITY
  min_variant_prob_margin = Float64::INFINITY

  base.timing.predictions.each_with_index do |base_prediction, row|
    variant_prediction = variant.timing.predictions[row]
    base_order = rank_order(base_prediction)
    variant_order = rank_order(variant_prediction)
    deltas = prediction_delta(base_prediction, variant_prediction)
    max_logit_abs_delta = deltas[:max_logit_abs] if deltas[:max_logit_abs] > max_logit_abs_delta
    max_prob_abs_delta = deltas[:max_prob_abs] if deltas[:max_prob_abs] > max_prob_abs_delta
    aggregate_max_logit_abs_delta = deltas[:max_logit_abs] if deltas[:max_logit_abs] > aggregate_max_logit_abs_delta
    aggregate_max_prob_abs_delta = deltas[:max_prob_abs] if deltas[:max_prob_abs] > aggregate_max_prob_abs_delta

    base_logit_margin = margin(base_prediction.logits, base_order)
    variant_logit_margin = margin(variant_prediction.logits, variant_order)
    base_prob_margin = margin(base_prediction.probabilities, base_order)
    variant_prob_margin = margin(variant_prediction.probabilities, variant_order)
    min_base_margin = base_logit_margin if base_logit_margin < min_base_margin
    min_variant_margin = variant_logit_margin if variant_logit_margin < min_variant_margin
    min_base_prob_margin = base_prob_margin if base_prob_margin < min_base_prob_margin
    min_variant_prob_margin = variant_prob_margin if variant_prob_margin < min_variant_prob_margin
    aggregate_min_base_logit_margin = base_logit_margin if base_logit_margin < aggregate_min_base_logit_margin
    aggregate_min_variant_logit_margin = variant_logit_margin if variant_logit_margin < aggregate_min_variant_logit_margin
    aggregate_min_base_prob_margin = base_prob_margin if base_prob_margin < aggregate_min_base_prob_margin
    aggregate_min_variant_prob_margin = variant_prob_margin if variant_prob_margin < aggregate_min_variant_prob_margin

    argmax_match = base_prediction.argmax_token_id == variant_prediction.argmax_token_id
    sampled_match = base_prediction.sampled_token_id == variant_prediction.sampled_token_id
    argmax_matches += 1 if argmax_match
    sampled_matches += 1 if sampled_match
    aggregate_argmax_matches += 1 if argmax_match
    aggregate_sampled_matches += 1 if sampled_match
    aggregate_rows += 1

    puts [
      "row",
      window_index.to_s,
      prompt_start.to_s,
      canvas_start.to_s,
      row.to_s,
      base_prediction.candidate_token_ids.size.to_s,
      base_prediction.argmax_token_id.to_s,
      variant_prediction.argmax_token_id.to_s,
      argmax_match.to_s,
      base_prediction.sampled_token_id.to_s,
      variant_prediction.sampled_token_id.to_s,
      sampled_match.to_s,
      format_f64(base_logit_margin),
      format_f64(variant_logit_margin),
      format_f64(base_prob_margin),
      format_f64(variant_prob_margin),
      format_f64(deltas[:max_logit_abs]),
      format_f64(deltas[:max_prob_abs]),
    ].join('\t')
  end

  puts [
    "cert_summary",
    "window=#{window_index}",
    "prompt_token=#{prompt_start}",
    "canvas_token=#{canvas_start}",
    "argmax_matches=#{argmax_matches}/#{canvas_len}",
    "sampled_matches=#{sampled_matches}/#{canvas_len}",
    "all_argmax_match=#{argmax_matches == canvas_len}",
    "all_sampled_match=#{sampled_matches == canvas_len}",
    "min_base_logit_margin=#{format_f64(min_base_margin)}",
    "min_variant_logit_margin=#{format_f64(min_variant_margin)}",
    "min_base_prob_margin=#{format_f64(min_base_prob_margin)}",
    "min_variant_prob_margin=#{format_f64(min_variant_prob_margin)}",
    "max_logit_abs_delta=#{format_f64(max_logit_abs_delta)}",
    "max_prob_abs_delta=#{format_f64(max_prob_abs_delta)}",
  ].join('\t')
end

all_argmax_match = aggregate_argmax_matches == aggregate_rows
all_sampled_match = aggregate_sampled_matches == aggregate_rows
puts [
  "aggregate_summary",
  "windows=#{windows.size}",
  "rows=#{aggregate_rows}",
  "argmax_matches=#{aggregate_argmax_matches}/#{aggregate_rows}",
  "sampled_matches=#{aggregate_sampled_matches}/#{aggregate_rows}",
  "candidate_offsets=#{candidate_offsets.join(",")}",
  "candidate_stride=#{candidate_stride}",
  "min_candidate_row_size=#{candidate_row_sizes.min}",
  "max_candidate_row_size=#{candidate_row_sizes.max}",
  "all_argmax_match=#{all_argmax_match}",
  "all_sampled_match=#{all_sampled_match}",
  "min_base_logit_margin=#{format_f64(aggregate_min_base_logit_margin)}",
  "min_variant_logit_margin=#{format_f64(aggregate_min_variant_logit_margin)}",
  "min_base_prob_margin=#{format_f64(aggregate_min_base_prob_margin)}",
  "min_variant_prob_margin=#{format_f64(aggregate_min_variant_prob_margin)}",
  "max_logit_abs_delta=#{format_f64(aggregate_max_logit_abs_delta)}",
  "max_prob_abs_delta=#{format_f64(aggregate_max_prob_abs_delta)}",
  "median_cache_speedup=#{format_f64(median(cache_speedups))}",
  "median_predict_speedup=#{format_f64(median(predict_speedups))}",
].join('\t')

failures = [] of String
failures << "argmax" if require_argmax_match && !all_argmax_match
failures << "sampled" if require_sampled_match && !all_sampled_match
if threshold = min_base_logit_margin
  failures << "base_margin" if aggregate_min_base_logit_margin < threshold
end
if threshold = min_variant_logit_margin
  failures << "variant_margin" if aggregate_min_variant_logit_margin < threshold
end
if threshold = max_logit_delta
  failures << "logit_delta" if aggregate_max_logit_abs_delta > threshold
end

unless failures.empty?
  STDERR.puts "output_cert status=fail failures=#{failures.join(",")} argmax_matches=#{aggregate_argmax_matches}/#{aggregate_rows} sampled_matches=#{aggregate_sampled_matches}/#{aggregate_rows}"
  exit 4
end

STDERR.puts "output_cert status=#{all_argmax_match ? "argmax_match" : "argmax_mismatch"} argmax_matches=#{aggregate_argmax_matches}/#{aggregate_rows} sampled_matches=#{aggregate_sampled_matches}/#{aggregate_rows}"
