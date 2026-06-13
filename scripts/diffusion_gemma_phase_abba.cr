require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

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

struct PhaseSample
  getter arm : String
  getter cycle : Int32
  getter sequence_index : Int32
  getter measured : Bool
  getter total_ms : Float64
  getter qkv_ms : Float64
  getter context_ms : Float64
  getter context_score_ms : Float64
  getter context_softmax_ms : Float64
  getter context_value_ms : Float64
  getter attention_out_ms : Float64
  getter shared_ffn_ms : Float64
  getter moe_ffn_ms : Float64
  getter moe_grouped_prep_ms : Float64
  getter moe_grouped_gate_up_ms : Float64
  getter moe_grouped_activation_ms : Float64
  getter moe_grouped_down_ms : Float64
  getter moe_grouped_scatter_combine_norm_ms : Float64
  getter ffn_resident_ms : Float64
  getter combine_scale_ms : Float64
  getter checksum : Float64
  getter shared_rows : Bool
  getter shared_resident : Bool
  getter ffn_resident : Bool
  getter moe_rows : Bool
  getter grouped_moe : Bool
  getter moe_router_batch : Bool
  getter moe_gpu_gather : Bool
  getter moe_gpu_prenorm : Bool
  getter moe_gpu_reduce : Bool
  getter attention_out_rows : Bool
  getter attention_residual_metal_rows : Bool
  getter attention_residual_context_buffer : Bool

  def initialize(@arm, @cycle, @sequence_index, @measured, @total_ms, @qkv_ms,
                 @context_ms, @context_score_ms, @context_softmax_ms,
                 @context_value_ms, @attention_out_ms, @shared_ffn_ms,
                 @moe_ffn_ms, @moe_grouped_prep_ms,
                 @moe_grouped_gate_up_ms, @moe_grouped_activation_ms,
                 @moe_grouped_down_ms, @moe_grouped_scatter_combine_norm_ms,
                 @ffn_resident_ms, @combine_scale_ms, @checksum, @shared_rows,
                 @shared_resident, @ffn_resident, @moe_rows, @grouped_moe, @moe_router_batch,
                 @moe_gpu_gather, @moe_gpu_prenorm, @moe_gpu_reduce,
                 @attention_out_rows, @attention_residual_metal_rows,
                 @attention_residual_context_buffer)
  end

  def value(metric : String) : Float64
    case metric
    when "total_ms"                            then total_ms
    when "qkv_ms"                              then qkv_ms
    when "context_ms"                          then context_ms
    when "context_score_ms"                    then context_score_ms
    when "context_softmax_ms"                  then context_softmax_ms
    when "context_value_ms"                    then context_value_ms
    when "attention_out_ms"                    then attention_out_ms
    when "shared_ffn_ms"                       then shared_ffn_ms
    when "moe_ffn_ms"                          then moe_ffn_ms
    when "moe_grouped_prep_ms"                 then moe_grouped_prep_ms
    when "moe_grouped_gate_up_ms"              then moe_grouped_gate_up_ms
    when "moe_grouped_activation_ms"           then moe_grouped_activation_ms
    when "moe_grouped_down_ms"                 then moe_grouped_down_ms
    when "moe_grouped_scatter_combine_norm_ms" then moe_grouped_scatter_combine_norm_ms
    when "ffn_resident_ms"                     then ffn_resident_ms
    when "combine_scale_ms"                    then combine_scale_ms
    else
      raise "unknown metric #{metric}"
    end
  end
end

TSV_HEADER = [
  "kind",
  "arm",
  "cycle",
  "sequence_index",
  "measured",
  "prompt_len",
  "canvas_len",
  "max_layers",
  "single_route",
  "shared_rows",
  "shared_resident",
  "ffn_resident",
  "moe_rows",
  "grouped_moe",
  "moe_router_batch",
  "moe_gpu_gather",
  "moe_gpu_prenorm",
  "moe_gpu_reduce",
  "attention_out_rows",
  "attention_residual_metal_rows",
  "attention_residual_context_buffer",
  "total_ms",
  "qkv_ms",
  "context_ms",
  "context_score_ms",
  "context_softmax_ms",
  "context_value_ms",
  "attention_out_ms",
  "shared_ffn_ms",
  "moe_ffn_ms",
  "moe_grouped_prep_ms",
  "moe_grouped_gate_up_ms",
  "moe_grouped_activation_ms",
  "moe_grouped_down_ms",
  "moe_grouped_scatter_combine_norm_ms",
  "ffn_resident_ms",
  "combine_scale_ms",
  "checksum",
]

METRICS = [
  "total_ms",
  "qkv_ms",
  "context_ms",
  "context_score_ms",
  "context_softmax_ms",
  "context_value_ms",
  "attention_out_ms",
  "shared_ffn_ms",
  "moe_ffn_ms",
  "moe_grouped_prep_ms",
  "moe_grouped_gate_up_ms",
  "moe_grouped_activation_ms",
  "moe_grouped_down_ms",
  "moe_grouped_scatter_combine_norm_ms",
  "ffn_resident_ms",
  "combine_scale_ms",
]

def generated_token_sequence(default_token : Int32, count : Int32, vocab_size : Int32, label : String) : Array(Int32)
  raise "#{label} must be positive" unless count > 0
  raise "#{label} exceeds vocab size" if count > vocab_size
  Array(Int32).new(count) { |i| (default_token + i) % vocab_size }
end

def prompt_rows_from_tokens(weights : ML::GGUF::DiffusionGemmaWeights, tokens : Array(Int32)) : Array(Float32)
  rows = [] of Float32
  tokens.each do |token_id|
    rows.concat(ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(weights, token_id))
  end
  rows
end

def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  mid = sorted.size // 2
  sorted.size.odd? ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0
end

def format_f64(value : Float64) : String
  "%.6f" % value
end

def format_sample_values(values : Array(Float64)) : String
  values.map { |v| "%.3f" % v }.join(",")
end

def trimmed_values(values : Array(Float64), trim_per_side : Int32) : Array(Float64)
  raise "trim_per_side must be non-negative" if trim_per_side < 0
  return values if trim_per_side == 0

  sorted = values.sort
  return sorted if sorted.size <= trim_per_side * 2 + 1

  sorted[trim_per_side, sorted.size - trim_per_side * 2]
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

def run_arm(weights : ML::GGUF::DiffusionGemmaWeights,
            canvas_rows : Array(Float32),
            mask : ML::GGUF::DiffusionGemmaAttentionMask,
            prompt_cache : ML::GGUF::DiffusionGemmaCPU::PromptLayerCache,
            max_layers : Int32,
            routes_by_layer_by_canvas_row : Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?,
            arm : String,
            cycle : Int32,
            sequence_index : Int32,
            measured : Bool) : PhaseSample
  t0 = Time.instant
  timed = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache_timed(
    weights: weights,
    canvas_rows: canvas_rows,
    mask: mask,
    prompt_cache: prompt_cache,
    max_layers: max_layers,
    routes_by_layer_by_canvas_row: routes_by_layer_by_canvas_row,
  )
  elapsed_ms = (Time.instant - t0).total_milliseconds

  PhaseSample.new(
    arm: arm,
    cycle: cycle,
    sequence_index: sequence_index,
    measured: measured,
    total_ms: elapsed_ms,
    qkv_ms: timed.qkv_ms,
    context_ms: timed.context_ms,
    context_score_ms: timed.context_score_ms,
    context_softmax_ms: timed.context_softmax_ms,
    context_value_ms: timed.context_value_ms,
    attention_out_ms: timed.attention_out_ms,
    shared_ffn_ms: timed.shared_ffn_ms,
    moe_ffn_ms: timed.moe_ffn_ms,
    moe_grouped_prep_ms: timed.moe_grouped_prep_ms,
    moe_grouped_gate_up_ms: timed.moe_grouped_gate_up_ms,
    moe_grouped_activation_ms: timed.moe_grouped_activation_ms,
    moe_grouped_down_ms: timed.moe_grouped_down_ms,
    moe_grouped_scatter_combine_norm_ms: timed.moe_grouped_scatter_combine_norm_ms,
    ffn_resident_ms: timed.ffn_resident_ms,
    combine_scale_ms: timed.combine_scale_ms,
    checksum: checksum_rows(timed.rows),
    shared_rows: ML::GGUF::DiffusionGemmaCPU.shared_ffn_batch_rows_enabled?(mask.canvas_len),
    shared_resident: ML::GGUF::DiffusionGemmaCPU.shared_ffn_resident_graph_enabled?(mask.canvas_len),
    ffn_resident: ML::GGUF::DiffusionGemmaCPU.ffn_residual_resident_graph_enabled?(mask.canvas_len),
    moe_rows: ML::GGUF::DiffusionGemmaCPU.moe_ffn_batch_rows_enabled?(mask.canvas_len),
    grouped_moe: ML::GGUF::DiffusionGemmaCPU.moe_ffn_grouped_expert_rows_enabled?(mask.canvas_len),
    moe_router_batch: ML::GGUF::DiffusionGemmaCPU.moe_grouped_router_batch_rows_enabled?(mask.canvas_len),
    moe_gpu_gather: ML::GGUF::DiffusionGemmaCPU.moe_grouped_gpu_gather_enabled?(mask.canvas_len) &&
                    ML::GGUF::DiffusionGemmaCPU.moe_grouped_resident_batch_graph_enabled?(mask.canvas_len),
    moe_gpu_prenorm: ML::GGUF::DiffusionGemmaCPU.moe_grouped_gpu_prenorm_enabled?(mask.canvas_len),
    moe_gpu_reduce: ML::GGUF::DiffusionGemmaCPU.moe_grouped_gpu_reduce_enabled?(mask.canvas_len),
    attention_out_rows: ML::GGUF::DiffusionGemmaCPU.attention_out_batch_rows_enabled?(mask.canvas_len),
    attention_residual_metal_rows: ML::GGUF::DiffusionGemmaCPU.attention_residual_metal_rows_enabled?(mask.canvas_len),
    attention_residual_context_buffer: timed.attention_residual_context_buffer,
  )
end

def print_sample(sample : PhaseSample, prompt_len : Int32, canvas_len : Int32, max_layers : Int32, single_route : Bool) : Nil
  puts [
    "sample",
    sample.arm,
    sample.cycle.to_s,
    sample.sequence_index.to_s,
    sample.measured.to_s,
    prompt_len.to_s,
    canvas_len.to_s,
    max_layers.to_s,
    single_route.to_s,
    sample.shared_rows.to_s,
    sample.shared_resident.to_s,
    sample.ffn_resident.to_s,
    sample.moe_rows.to_s,
    sample.grouped_moe.to_s,
    sample.moe_router_batch.to_s,
    sample.moe_gpu_gather.to_s,
    sample.moe_gpu_prenorm.to_s,
    sample.moe_gpu_reduce.to_s,
    sample.attention_out_rows.to_s,
    sample.attention_residual_metal_rows.to_s,
    sample.attention_residual_context_buffer.to_s,
    format_f64(sample.total_ms),
    format_f64(sample.qkv_ms),
    format_f64(sample.context_ms),
    format_f64(sample.context_score_ms),
    format_f64(sample.context_softmax_ms),
    format_f64(sample.context_value_ms),
    format_f64(sample.attention_out_ms),
    format_f64(sample.shared_ffn_ms),
    format_f64(sample.moe_ffn_ms),
    format_f64(sample.moe_grouped_prep_ms),
    format_f64(sample.moe_grouped_gate_up_ms),
    format_f64(sample.moe_grouped_activation_ms),
    format_f64(sample.moe_grouped_down_ms),
    format_f64(sample.moe_grouped_scatter_combine_norm_ms),
    format_f64(sample.ffn_resident_ms),
    format_f64(sample.combine_scale_ms),
    format_f64(sample.checksum),
  ].join('\t')
end

def print_metric_summary(kind : String,
                         metric : String,
                         base_values : Array(Float64),
                         variant_values : Array(Float64)) : Nil
  base_median = median(base_values)
  variant_median = median(variant_values)
  delta = base_median - variant_median
  combined_range = (base_values.max - base_values.min) + (variant_values.max - variant_values.min)
  range_over_delta = delta.abs > 0.0 ? combined_range / delta.abs : Float64::INFINITY
  speedup = variant_median > 0.0 ? base_median / variant_median : 0.0
  puts [
    kind,
    metric,
    format_f64(base_median),
    format_f64(variant_median),
    format_f64(speedup),
    format_f64(delta),
    format_f64(combined_range),
    format_f64(range_over_delta),
    format_sample_values(base_values),
    format_sample_values(variant_values),
  ].join('\t')
end

def print_summary(samples : Array(PhaseSample), trim_per_arm : Int32) : Nil
  measured = samples.select(&.measured)
  by_arm = measured.group_by(&.arm)
  return unless by_arm["base"]? && by_arm["variant"]?

  METRICS.each do |metric|
    base_values = by_arm["base"].map { |sample| sample.value(metric) }
    variant_values = by_arm["variant"].map { |sample| sample.value(metric) }
    print_metric_summary("summary", metric, base_values, variant_values)
    if trim_per_arm > 0
      print_metric_summary(
        "trimmed_summary",
        metric,
        trimmed_values(base_values, trim_per_arm),
        trimmed_values(variant_values, trim_per_arm),
      )
    end
  end
end

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 128
canvas_len = 8
prompt_token = 1
canvas_token = 0
max_layers = 1
warmups = 1
repeats = 4
trim_per_arm = 0
sequence = "base variant variant base"
base_env = ArmEnv.new("")
variant_env = ArmEnv.new("")
single_route = true

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_phase_abba [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 128)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Synthetic canvas length (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--canvas-token ID", "Synthetic canvas start token id (default: 0)") { |v| canvas_token = v.to_i }
  p.on("--max-layers N", "Decode layers (default: 1)") { |v| max_layers = v.to_i }
  p.on("--warmups N", "Unmeasured ABBA cycles before samples (default: 1)") { |v| warmups = v.to_i }
  p.on("--repeats N", "Measured ABBA cycles (default: 4)") { |v| repeats = v.to_i }
  p.on("--trim-per-arm N", "Emit trimmed summaries after dropping N low/high values per arm and metric (default: 0)") { |v| trim_per_arm = v.to_i }
  p.on("--sequence LIST", "Arm sequence per cycle (default: base variant variant base)") { |v| sequence = v }
  p.on("--base-env ENV", "Whitespace-separated KEY=VALUE env for base arm") { |v| base_env = ArmEnv.new(v) }
  p.on("--variant-env ENV", "Whitespace-separated KEY=VALUE env for variant arm") { |v| variant_env = ArmEnv.new(v) }
  p.on("--full-routes", "Use model-computed full MoE routes instead of the top-1 smoke shortcut") { single_route = false }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt-len must be positive" unless prompt_len > 0
raise "--canvas-len must be positive" unless canvas_len > 0
raise "--max-layers must be positive" unless max_layers > 0
raise "--warmups must be non-negative" unless warmups >= 0
raise "--repeats must be positive" unless repeats > 0
raise "--trim-per-arm must be non-negative" unless trim_per_arm >= 0
arms = sequence.split(/\s+/).reject(&.empty?)
raise "--sequence must contain at least one arm" if arms.empty?
arms.each { |arm| raise "--sequence arms must be base or variant, got #{arm.inspect}" unless {"base", "variant"}.includes?(arm) }
raise "single-route smoke currently supports --max-layers 1; pass --full-routes for deeper runs" if single_route && max_layers != 1

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length

prompt_tokens = generated_token_sequence(prompt_token, prompt_len, hp.vocab_size, "--prompt-len")
canvas_tokens = generated_token_sequence(canvas_token, canvas_len, hp.vocab_size, "--canvas-len")
prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
canvas_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(weights, canvas_tokens)
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)
routes_by_layer_by_canvas_row = nil.as(Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?)
if single_route
  route_rows = canvas_tokens.map_with_index do |_, i|
    row = canvas_rows[i * hp.n_embd, hp.n_embd]
    ML::GGUF::DiffusionGemmaCPU.route_experts(weights, 0, row)[0, 1]
  end
  routes_by_layer_by_canvas_row = [route_rows]
end

cache_t0 = Time.instant
cache_old_env = apply_env(base_env)
prompt_cache = begin
  ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
    weights,
    prompt_rows,
    mask,
    max_layers: max_layers,
    materialize_final_rows: false,
  )
ensure
  restore_env(cache_old_env)
end
cache_ms = (Time.instant - cache_t0).total_milliseconds

puts "# load_ms=#{format_f64(load_ms)}"
puts "# prompt_cache_ms=#{format_f64(cache_ms)}"
puts "# base_env=#{base_env.raw.empty? ? "<empty>" : base_env.raw}"
puts "# variant_env=#{variant_env.raw.empty? ? "<empty>" : variant_env.raw}"
puts TSV_HEADER.join('\t')

samples = [] of PhaseSample
total_cycles = warmups + repeats
total_cycles.times do |cycle|
  measured = cycle >= warmups
  arms.each_with_index do |arm, sequence_index|
    env = arm == "base" ? base_env : variant_env
    old = apply_env(env)
    begin
      sample = run_arm(
        weights,
        canvas_rows,
        mask,
        prompt_cache,
        max_layers,
        routes_by_layer_by_canvas_row,
        arm,
        cycle,
        sequence_index,
        measured,
      )
      samples << sample
      print_sample(sample, prompt_len, canvas_len, max_layers, single_route)
    ensure
      restore_env(old)
    end
  end
end
print_summary(samples, trim_per_arm)
