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

struct PromptCacheSample
  getter arm : String
  getter cycle : Int32
  getter sequence_index : Int32
  getter measured : Bool
  getter total_ms : Float64
  getter projection_ms : Float64
  getter projection_norm_ms : Float64
  getter projection_matmul_ms : Float64
  getter projection_assemble_ms : Float64
  getter projection_copy_ms : Float64
  getter projection_head_norm_ms : Float64
  getter projection_q_norm_ms : Float64
  getter projection_k_norm_ms : Float64
  getter projection_v_norm_ms : Float64
  getter projection_rope_ms : Float64
  getter projection_rope_table_ms : Float64
  getter projection_rope_apply_ms : Float64
  getter projection_rope_q_apply_ms : Float64
  getter projection_rope_k_apply_ms : Float64
  getter materialize_ms : Float64
  getter materialize_context_ms : Float64
  getter materialize_attention_out_ms : Float64
  getter materialize_shared_ffn_ms : Float64
  getter materialize_moe_ffn_ms : Float64
  getter materialize_combine_scale_ms : Float64
  getter materialize_moe_grouped_prep_ms : Float64
  getter materialize_moe_grouped_gate_up_ms : Float64
  getter materialize_moe_grouped_activation_ms : Float64
  getter materialize_moe_grouped_down_ms : Float64
  getter materialize_moe_grouped_scatter_combine_norm_ms : Float64
  getter materialize_moe_grouped_active_experts : Int32
  getter materialize_moe_grouped_route_slots : Int32
  getter materialize_moe_grouped_max_expert_batch : Int32
  getter materialize_moe_grouped_over_threshold_experts : Int32
  getter checksum : Float64
  getter prompt_cache_policy : Bool
  getter materialize_batch_rows : Bool
  getter materialize_grouped_moe : Bool
  getter projection_backend : String
  getter fused_norm_rope : Bool

  def initialize(@arm, @cycle, @sequence_index, @measured, @total_ms,
                 @projection_ms, @projection_norm_ms, @projection_matmul_ms,
                 @projection_assemble_ms, @projection_copy_ms,
                 @projection_head_norm_ms, @projection_q_norm_ms,
                 @projection_k_norm_ms, @projection_v_norm_ms,
                 @projection_rope_ms, @projection_rope_table_ms,
                 @projection_rope_apply_ms, @projection_rope_q_apply_ms,
                 @projection_rope_k_apply_ms, @materialize_ms,
                 @materialize_context_ms, @materialize_attention_out_ms,
                 @materialize_shared_ffn_ms, @materialize_moe_ffn_ms,
                 @materialize_combine_scale_ms, @materialize_moe_grouped_prep_ms,
                 @materialize_moe_grouped_gate_up_ms,
                 @materialize_moe_grouped_activation_ms,
                 @materialize_moe_grouped_down_ms,
                 @materialize_moe_grouped_scatter_combine_norm_ms,
                 @materialize_moe_grouped_active_experts,
                 @materialize_moe_grouped_route_slots,
                 @materialize_moe_grouped_max_expert_batch,
                 @materialize_moe_grouped_over_threshold_experts,
                 @checksum,
                 @prompt_cache_policy, @materialize_batch_rows,
                 @materialize_grouped_moe, @projection_backend,
                 @fused_norm_rope)
  end

  def value(metric : String) : Float64
    case metric
    when "total_ms"                                        then total_ms
    when "projection_ms"                                   then projection_ms
    when "projection_norm_ms"                              then projection_norm_ms
    when "projection_matmul_ms"                            then projection_matmul_ms
    when "projection_assemble_ms"                          then projection_assemble_ms
    when "projection_copy_ms"                              then projection_copy_ms
    when "projection_head_norm_ms"                         then projection_head_norm_ms
    when "projection_q_norm_ms"                            then projection_q_norm_ms
    when "projection_k_norm_ms"                            then projection_k_norm_ms
    when "projection_v_norm_ms"                            then projection_v_norm_ms
    when "projection_rope_ms"                              then projection_rope_ms
    when "projection_rope_table_ms"                        then projection_rope_table_ms
    when "projection_rope_apply_ms"                        then projection_rope_apply_ms
    when "projection_rope_q_apply_ms"                      then projection_rope_q_apply_ms
    when "projection_rope_k_apply_ms"                      then projection_rope_k_apply_ms
    when "materialize_ms"                                  then materialize_ms
    when "materialize_context_ms"                          then materialize_context_ms
    when "materialize_attention_out_ms"                    then materialize_attention_out_ms
    when "materialize_shared_ffn_ms"                       then materialize_shared_ffn_ms
    when "materialize_moe_ffn_ms"                          then materialize_moe_ffn_ms
    when "materialize_combine_scale_ms"                    then materialize_combine_scale_ms
    when "materialize_moe_grouped_prep_ms"                 then materialize_moe_grouped_prep_ms
    when "materialize_moe_grouped_gate_up_ms"              then materialize_moe_grouped_gate_up_ms
    when "materialize_moe_grouped_activation_ms"           then materialize_moe_grouped_activation_ms
    when "materialize_moe_grouped_down_ms"                 then materialize_moe_grouped_down_ms
    when "materialize_moe_grouped_scatter_combine_norm_ms" then materialize_moe_grouped_scatter_combine_norm_ms
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
  "materialize_final_rows",
  "prompt_cache_policy",
  "materialize_batch_rows",
  "materialize_grouped_moe",
  "projection_backend",
  "fused_norm_rope",
  "total_ms",
  "projection_ms",
  "projection_norm_ms",
  "projection_matmul_ms",
  "projection_assemble_ms",
  "projection_copy_ms",
  "projection_head_norm_ms",
  "projection_q_norm_ms",
  "projection_k_norm_ms",
  "projection_v_norm_ms",
  "projection_rope_ms",
  "projection_rope_table_ms",
  "projection_rope_apply_ms",
  "projection_rope_q_apply_ms",
  "projection_rope_k_apply_ms",
  "materialize_ms",
  "materialize_context_ms",
  "materialize_attention_out_ms",
  "materialize_shared_ffn_ms",
  "materialize_moe_ffn_ms",
  "materialize_combine_scale_ms",
  "materialize_moe_grouped_prep_ms",
  "materialize_moe_grouped_gate_up_ms",
  "materialize_moe_grouped_activation_ms",
  "materialize_moe_grouped_down_ms",
  "materialize_moe_grouped_scatter_combine_norm_ms",
  "materialize_moe_grouped_active_experts",
  "materialize_moe_grouped_route_slots",
  "materialize_moe_grouped_max_expert_batch",
  "materialize_moe_grouped_over_threshold_experts",
  "checksum",
]

METRICS = [
  "total_ms",
  "projection_ms",
  "projection_norm_ms",
  "projection_matmul_ms",
  "projection_assemble_ms",
  "projection_copy_ms",
  "projection_head_norm_ms",
  "projection_q_norm_ms",
  "projection_k_norm_ms",
  "projection_v_norm_ms",
  "projection_rope_ms",
  "projection_rope_table_ms",
  "projection_rope_apply_ms",
  "projection_rope_q_apply_ms",
  "projection_rope_k_apply_ms",
  "materialize_ms",
  "materialize_context_ms",
  "materialize_attention_out_ms",
  "materialize_shared_ffn_ms",
  "materialize_moe_ffn_ms",
  "materialize_combine_scale_ms",
  "materialize_moe_grouped_prep_ms",
  "materialize_moe_grouped_gate_up_ms",
  "materialize_moe_grouped_activation_ms",
  "materialize_moe_grouped_down_ms",
  "materialize_moe_grouped_scatter_combine_norm_ms",
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

def projection_backend(prompt_len : Int32) : String
  if ML::GGUF::DiffusionGemmaCPU.prompt_projection_metal_enabled? &&
     prompt_len >= ML::GGUF::DiffusionGemmaCPU.prompt_projection_metal_min_batch
    "metal"
  else
    "cpu"
  end
end

def top1_prompt_routes(weights : ML::GGUF::DiffusionGemmaWeights,
                       prompt_rows : Array(Float32),
                       prompt_len : Int32) : Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))
  hp = weights.hparams
  route_rows = prompt_len.times.map do |i|
    row = prompt_rows[i * hp.n_embd, hp.n_embd]
    ML::GGUF::DiffusionGemmaCPU.route_experts(weights, 0, row)[0, 1]
  end.to_a
  [route_rows]
end

def run_arm(weights : ML::GGUF::DiffusionGemmaWeights,
            prompt_rows : Array(Float32),
            mask : ML::GGUF::DiffusionGemmaAttentionMask,
            max_layers : Int32,
            materialize_final_rows : Bool,
            routes_by_layer_by_prompt_row : Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?,
            arm : String,
            cycle : Int32,
            sequence_index : Int32,
            measured : Bool) : PromptCacheSample
  prompt_len = mask.prompt_len
  t0 = Time.instant
  cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
    weights,
    prompt_rows,
    mask,
    max_layers: max_layers,
    routes_by_layer_by_prompt_row: routes_by_layer_by_prompt_row,
    materialize_final_rows: materialize_final_rows,
  )
  elapsed_ms = (Time.instant - t0).total_milliseconds

  PromptCacheSample.new(
    arm: arm,
    cycle: cycle,
    sequence_index: sequence_index,
    measured: measured,
    total_ms: elapsed_ms,
    projection_ms: cache.projection_ms_by_layer.sum,
    projection_norm_ms: cache.projection_norm_ms_by_layer.sum,
    projection_matmul_ms: cache.projection_matmul_ms_by_layer.sum,
    projection_assemble_ms: cache.projection_assemble_ms_by_layer.sum,
    projection_copy_ms: cache.projection_copy_ms_by_layer.sum,
    projection_head_norm_ms: cache.projection_head_norm_ms_by_layer.sum,
    projection_q_norm_ms: cache.projection_q_norm_ms_by_layer.sum,
    projection_k_norm_ms: cache.projection_k_norm_ms_by_layer.sum,
    projection_v_norm_ms: cache.projection_v_norm_ms_by_layer.sum,
    projection_rope_ms: cache.projection_rope_ms_by_layer.sum,
    projection_rope_table_ms: cache.projection_rope_table_ms_by_layer.sum,
    projection_rope_apply_ms: cache.projection_rope_apply_ms_by_layer.sum,
    projection_rope_q_apply_ms: cache.projection_rope_q_apply_ms_by_layer.sum,
    projection_rope_k_apply_ms: cache.projection_rope_k_apply_ms_by_layer.sum,
    materialize_ms: cache.materialize_ms_by_layer.sum,
    materialize_context_ms: cache.materialize_context_ms_by_layer.sum,
    materialize_attention_out_ms: cache.materialize_attention_out_ms_by_layer.sum,
    materialize_shared_ffn_ms: cache.materialize_shared_ffn_ms_by_layer.sum,
    materialize_moe_ffn_ms: cache.materialize_moe_ffn_ms_by_layer.sum,
    materialize_combine_scale_ms: cache.materialize_combine_scale_ms_by_layer.sum,
    materialize_moe_grouped_prep_ms: cache.materialize_moe_grouped_prep_ms_by_layer.sum,
    materialize_moe_grouped_gate_up_ms: cache.materialize_moe_grouped_gate_up_ms_by_layer.sum,
    materialize_moe_grouped_activation_ms: cache.materialize_moe_grouped_activation_ms_by_layer.sum,
    materialize_moe_grouped_down_ms: cache.materialize_moe_grouped_down_ms_by_layer.sum,
    materialize_moe_grouped_scatter_combine_norm_ms: cache.materialize_moe_grouped_scatter_combine_norm_ms_by_layer.sum,
    materialize_moe_grouped_active_experts: cache.materialize_moe_grouped_active_experts_by_layer.sum,
    materialize_moe_grouped_route_slots: cache.materialize_moe_grouped_route_slots_by_layer.sum,
    materialize_moe_grouped_max_expert_batch: cache.materialize_moe_grouped_max_expert_batch_by_layer.max? || 0,
    materialize_moe_grouped_over_threshold_experts: cache.materialize_moe_grouped_over_threshold_experts_by_layer.sum,
    checksum: checksum_rows(cache.final_rows),
    prompt_cache_policy: ML::GGUF::DiffusionGemmaCPU.prompt_cache_policy_requested?,
    materialize_batch_rows: ML::GGUF::DiffusionGemmaCPU.prompt_materialize_batch_rows_enabled?(prompt_len),
    materialize_grouped_moe: ML::GGUF::DiffusionGemmaCPU.prompt_materialize_grouped_moe_enabled?,
    projection_backend: projection_backend(prompt_len),
    fused_norm_rope: ML::GGUF::DiffusionGemmaCPU.prompt_projection_fused_norm_rope_enabled?,
  )
end

def capture_routes_for_arm(weights : ML::GGUF::DiffusionGemmaWeights,
                           prompt_rows : Array(Float32),
                           mask : ML::GGUF::DiffusionGemmaAttentionMask,
                           max_layers : Int32,
                           env : ArmEnv,
                           arm : String) : Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))
  old = apply_env(env)
  begin
    ML::GGUF::DiffusionGemmaCPU.clear_ffn_resident_graph_cache
    t0 = Time.instant
    cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      materialize_final_rows: true,
    )
    elapsed_ms = (Time.instant - t0).total_milliseconds
    routes = cache.routes_by_layer_by_prompt_row
    raise "route capture for #{arm} produced #{routes.size}/#{max_layers} layers" unless routes.size == max_layers
    route_rows = routes.sum(&.size)
    route_slots = routes.sum { |layer| layer.sum(&.size) }
    puts [
      "# route_capture",
      arm,
      "layers=#{routes.size}",
      "rows=#{route_rows}",
      "slots=#{route_slots}",
      "elapsed_ms=#{format_f64(elapsed_ms)}",
      "checksum=#{format_f64(checksum_rows(cache.final_rows))}",
    ].join('\t')
    routes
  ensure
    restore_env(old)
    ML::GGUF::DiffusionGemmaCPU.clear_ffn_resident_graph_cache
  end
end

def print_sample(sample : PromptCacheSample,
                 prompt_len : Int32,
                 canvas_len : Int32,
                 max_layers : Int32,
                 single_route : Bool,
                 materialize_final_rows : Bool) : Nil
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
    materialize_final_rows.to_s,
    sample.prompt_cache_policy.to_s,
    sample.materialize_batch_rows.to_s,
    sample.materialize_grouped_moe.to_s,
    sample.projection_backend,
    sample.fused_norm_rope.to_s,
    format_f64(sample.total_ms),
    format_f64(sample.projection_ms),
    format_f64(sample.projection_norm_ms),
    format_f64(sample.projection_matmul_ms),
    format_f64(sample.projection_assemble_ms),
    format_f64(sample.projection_copy_ms),
    format_f64(sample.projection_head_norm_ms),
    format_f64(sample.projection_q_norm_ms),
    format_f64(sample.projection_k_norm_ms),
    format_f64(sample.projection_v_norm_ms),
    format_f64(sample.projection_rope_ms),
    format_f64(sample.projection_rope_table_ms),
    format_f64(sample.projection_rope_apply_ms),
    format_f64(sample.projection_rope_q_apply_ms),
    format_f64(sample.projection_rope_k_apply_ms),
    format_f64(sample.materialize_ms),
    format_f64(sample.materialize_context_ms),
    format_f64(sample.materialize_attention_out_ms),
    format_f64(sample.materialize_shared_ffn_ms),
    format_f64(sample.materialize_moe_ffn_ms),
    format_f64(sample.materialize_combine_scale_ms),
    format_f64(sample.materialize_moe_grouped_prep_ms),
    format_f64(sample.materialize_moe_grouped_gate_up_ms),
    format_f64(sample.materialize_moe_grouped_activation_ms),
    format_f64(sample.materialize_moe_grouped_down_ms),
    format_f64(sample.materialize_moe_grouped_scatter_combine_norm_ms),
    sample.materialize_moe_grouped_active_experts.to_s,
    sample.materialize_moe_grouped_route_slots.to_s,
    sample.materialize_moe_grouped_max_expert_batch.to_s,
    sample.materialize_moe_grouped_over_threshold_experts.to_s,
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

def print_summary(samples : Array(PromptCacheSample), trim_per_arm : Int32) : Nil
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

def print_checksum_summary(samples : Array(PromptCacheSample)) : Float64?
  measured = samples.select(&.measured)
  by_arm = measured.group_by(&.arm)
  return nil unless by_arm["base"]? && by_arm["variant"]?

  base_values = by_arm["base"].map(&.checksum)
  variant_values = by_arm["variant"].map(&.checksum)
  base_median = median(base_values)
  variant_median = median(variant_values)
  delta = (base_median - variant_median).abs
  puts [
    "checksum_summary",
    format_f64(base_median),
    format_f64(variant_median),
    format_f64(delta),
    format_sample_values(base_values),
    format_sample_values(variant_values),
  ].join('\t')
  delta
end

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 16
canvas_len = 8
prompt_token = 1
max_layers = 1
warmups = 1
repeats = 4
trim_per_arm = 0
sequence = "base variant variant base"
mirror_sequence = nil.as(String?)
base_env = ArmEnv.new("")
variant_env = ArmEnv.new("")
single_route = true
materialize_final_rows = false
checksum_tolerance = nil.as(Float64?)
route_replay_base = false
route_replay_variant = false

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_prompt_cache_abba [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 16)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Synthetic canvas length used for mask/cache shape (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--max-layers N", "Prompt-cache layers (default: 1)") { |v| max_layers = v.to_i }
  p.on("--warmups N", "Unmeasured ABBA cycles before samples (default: 1)") { |v| warmups = v.to_i }
  p.on("--repeats N", "Measured ABBA cycles (default: 4)") { |v| repeats = v.to_i }
  p.on("--trim-per-arm N", "Emit trimmed summaries after dropping N low/high values per arm and metric (default: 0)") { |v| trim_per_arm = v.to_i }
  p.on("--sequence LIST", "Arm sequence per cycle (default: base variant variant base)") { |v| sequence = v }
  p.on("--mirror-sequence LIST", "Alternate this arm sequence on odd cycles, preserving sequence_index positions") { |v| mirror_sequence = v }
  p.on("--base-env ENV", "Whitespace-separated KEY=VALUE env for base arm") { |v| base_env = ArmEnv.new(v) }
  p.on("--variant-env ENV", "Whitespace-separated KEY=VALUE env for variant arm") { |v| variant_env = ArmEnv.new(v) }
  p.on("--full-routes", "Use model-computed full MoE routes instead of the top-1 smoke shortcut") { single_route = false }
  p.on("--materialize-final-rows", "Materialize the final prompt-cache layer rows") { materialize_final_rows = true }
  p.on("--checksum-tolerance F", "Exit 4 when base/variant checksum median delta exceeds F") { |v| checksum_tolerance = v.to_f64 }
  p.on("--base-replay-routes", "Pre-capture and replay base prompt MoE routes for base arm") { route_replay_base = true }
  p.on("--variant-replay-routes", "Pre-capture and replay variant prompt MoE routes for variant arm") { route_replay_variant = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

route_replay_requested = route_replay_base || route_replay_variant
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
mirror_arms = mirror_sequence.try { |raw| raw.split(/\s+/).reject(&.empty?) }
if alt_arms = mirror_arms
  raise "--mirror-sequence must contain at least one arm" if alt_arms.empty?
  alt_arms.each { |arm| raise "--mirror-sequence arms must be base or variant, got #{arm.inspect}" unless {"base", "variant"}.includes?(arm) }
  raise "--mirror-sequence must have the same length as --sequence" unless alt_arms.size == arms.size
end
raise "single-route smoke currently supports --max-layers 1; pass --full-routes for deeper runs" if single_route && max_layers != 1
if route_replay_requested
  raise "route replay requires --full-routes" if single_route
  raise "route replay requires --materialize-final-rows" unless materialize_final_rows
end

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length

prompt_tokens = generated_token_sequence(prompt_token, prompt_len, hp.vocab_size, "--prompt-len")
prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)
routes_by_layer_by_prompt_row = if single_route && materialize_final_rows
                                  top1_prompt_routes(weights, prompt_rows, prompt_len)
                                else
                                  nil
                                end
base_replay_routes = route_replay_base ? capture_routes_for_arm(weights, prompt_rows, mask, max_layers, base_env, "base") : nil
variant_replay_routes = route_replay_variant ? capture_routes_for_arm(weights, prompt_rows, mask, max_layers, variant_env, "variant") : nil

puts "# load_ms=#{format_f64(load_ms)}"
puts "# base_env=#{base_env.raw.empty? ? "<empty>" : base_env.raw}"
puts "# variant_env=#{variant_env.raw.empty? ? "<empty>" : variant_env.raw}"
puts "# sequence=#{sequence}"
puts "# mirror_sequence=#{mirror_sequence || "<none>"}"
puts "# route_replay_base=#{route_replay_base}"
puts "# route_replay_variant=#{route_replay_variant}"
puts TSV_HEADER.join('\t')

samples = [] of PromptCacheSample
total_cycles = warmups + repeats
total_cycles.times do |cycle|
  measured = cycle >= warmups
  cycle_arms = if (alt_arms = mirror_arms) && cycle.odd?
                 alt_arms
               else
                 arms
               end
  cycle_arms.each_with_index do |arm, sequence_index|
    env = arm == "base" ? base_env : variant_env
    arm_routes = case arm
                 when "base"
                   base_replay_routes || routes_by_layer_by_prompt_row
                 when "variant"
                   variant_replay_routes || routes_by_layer_by_prompt_row
                 else
                   routes_by_layer_by_prompt_row
                 end
    old = apply_env(env)
    begin
      sample = run_arm(
        weights,
        prompt_rows,
        mask,
        max_layers,
        materialize_final_rows,
        arm_routes,
        arm,
        cycle,
        sequence_index,
        measured,
      )
      samples << sample
      print_sample(sample, prompt_len, canvas_len, max_layers, single_route, materialize_final_rows)
    ensure
      restore_env(old)
    end
  end
end
print_summary(samples, trim_per_arm)
checksum_delta = print_checksum_summary(samples)
if tolerance = checksum_tolerance
  if delta = checksum_delta
    STDOUT.flush
    if delta > tolerance
      STDERR.puts "checksum_verdict status=fail delta=#{format_f64(delta)} tolerance=#{format_f64(tolerance)}"
      exit 4
    end
    STDERR.puts "checksum_verdict status=ok delta=#{format_f64(delta)} tolerance=#{format_f64(tolerance)}"
  end
end
