require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

alias ExpertRoute = ML::GGUF::DiffusionGemmaCPU::ExpertRoute

struct RouteOverlapStats
  getter layer : Int32
  getter prompt_len : Int32
  getter canvas_len : Int32
  getter route_width : Int32
  getter total_slots : Int32
  getter unique_experts : Int32
  getter duplicate_slots : Int32
  getter grouped_experts : Int32
  getter groupable_slots : Int32
  getter groupable_fraction : Float64
  getter max_multiplicity : Int32
  getter mean_multiplicity : Float64
  getter pair_count : Int32
  getter pair_overlap_mean : Float64
  getter pair_overlap_max : Int32
  getter pair_weight_overlap_mean : Float64

  def initialize(@layer, @prompt_len, @canvas_len, @route_width, @total_slots,
                 @unique_experts, @duplicate_slots, @grouped_experts,
                 @groupable_slots, @groupable_fraction, @max_multiplicity,
                 @mean_multiplicity, @pair_count, @pair_overlap_mean,
                 @pair_overlap_max, @pair_weight_overlap_mean)
  end

  def values : Array(String)
    [
      layer.to_s,
      prompt_len.to_s,
      canvas_len.to_s,
      route_width.to_s,
      total_slots.to_s,
      unique_experts.to_s,
      duplicate_slots.to_s,
      grouped_experts.to_s,
      groupable_slots.to_s,
      format_float(groupable_fraction),
      max_multiplicity.to_s,
      format_float(mean_multiplicity),
      pair_count.to_s,
      format_float(pair_overlap_mean),
      pair_overlap_max.to_s,
      format_float(pair_weight_overlap_mean),
    ]
  end
end

def plan_values(plan : ML::GGUF::DiffusionGemmaCPU::CogniGraphPlan) : Array(String)
  [
    plan.n_ops.to_s,
    plan.n_waves.to_s,
    plan.n_barriers.to_s,
    plan.max_wave_width.to_s,
    plan.active_experts.to_s,
    plan.route_slots.to_s,
    plan.wave_widths.join(","),
    plan.phi,
  ]
end

def parse_positive_counts(raw : String, label : String) : Array(Int32)
  counts = raw.split(/[,\s]+/).map(&.strip).reject(&.empty?).map(&.to_i)
  raise "#{label} must contain at least one count" if counts.empty?
  counts.each do |count|
    raise "#{label} entries must be positive" unless count > 0
  end
  counts
end

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

def route_rows(weights : ML::GGUF::DiffusionGemmaWeights,
               layer : Int32,
               rows : Array(Float32),
               row_count : Int32) : Array(Array(ExpertRoute))
  hp = weights.hparams
  routes = [] of Array(ExpertRoute)
  row_count.times do |row|
    routes << ML::GGUF::DiffusionGemmaCPU.route_experts(
      weights,
      layer,
      rows[row * hp.n_embd, hp.n_embd],
    )
  end
  routes
end

def attention_residual_rows(weights : ML::GGUF::DiffusionGemmaWeights,
                            layer : Int32,
                            prompt_projections : Array(ML::GGUF::DiffusionGemmaCPU::AttentionProjection),
                            canvas_rows : Array(Float32),
                            mask : ML::GGUF::DiffusionGemmaAttentionMask) : Array(Float32)
  hp = weights.hparams
  q_context_dim = hp.n_head * hp.head_dim_for_layer(layer)
  canvas_projections = [] of ML::GGUF::DiffusionGemmaCPU::AttentionProjection
  mask.canvas_len.times do |canvas_pos|
    row = canvas_rows[canvas_pos * hp.n_embd, hp.n_embd]
    canvas_projections << ML::GGUF::DiffusionGemmaCPU.attention_project_normed(
      weights,
      layer,
      row,
      mask.prompt_len + canvas_pos,
    )
  end

  context_rows = Array(Float32).new(mask.canvas_len * q_context_dim, 0.0_f32)
  mask.canvas_len.times do |canvas_pos|
    context = ML::GGUF::DiffusionGemmaCPU.attention_context_decode_timed(
      prompt_projections,
      canvas_projections,
      hp,
      layer,
      canvas_query_index: canvas_pos,
      mask: mask,
    ).context
    context.size.times do |i|
      context_rows[canvas_pos * q_context_dim + i] = context[i]
    end
  end

  ML::GGUF::DiffusionGemmaCPU.attention_residual_from_context_rows(
    weights,
    layer,
    canvas_rows,
    context_rows,
    mask.canvas_len,
  )
end

def weighted_overlap(a : Array(ExpertRoute), b : Array(ExpertRoute)) : Float64
  by_expert = Hash(Int32, Float32).new
  a.each { |route| by_expert[route.expert] = route.weight }
  overlap = 0.0
  b.each do |route|
    if weight = by_expert[route.expert]?
      overlap += Math.min(weight, route.weight).to_f64
    end
  end
  overlap
end

def overlap_stats(routes : Array(Array(ExpertRoute)), layer : Int32, prompt_len : Int32, canvas_len : Int32) : RouteOverlapStats
  route_width = routes.empty? ? 0 : routes[0].size
  total_slots = routes.sum(&.size)
  counts = Hash(Int32, Int32).new(0)
  routes.each do |row_routes|
    row_routes.each do |route|
      counts[route.expert] += 1
    end
  end

  unique_experts = counts.size
  duplicate_slots = total_slots - unique_experts
  grouped_experts = counts.count { |_, multiplicity| multiplicity >= 2 }
  groupable_slots = counts.sum { |_, multiplicity| multiplicity >= 2 ? multiplicity : 0 }
  groupable_fraction = total_slots > 0 ? groupable_slots.to_f64 / total_slots : 0.0
  max_multiplicity = counts.values.max? || 0
  mean_multiplicity = unique_experts > 0 ? total_slots.to_f64 / unique_experts : 0.0

  pair_count = 0
  pair_overlap_sum = 0
  pair_overlap_max = 0
  pair_weight_overlap_sum = 0.0
  routes.each_with_index do |left, i|
    ((i + 1)...routes.size).each do |j|
      right = routes[j]
      overlap = left.count { |route| right.any? { |other| other.expert == route.expert } }
      pair_count += 1
      pair_overlap_sum += overlap
      pair_overlap_max = overlap if overlap > pair_overlap_max
      pair_weight_overlap_sum += weighted_overlap(left, right)
    end
  end
  pair_overlap_mean = pair_count > 0 ? pair_overlap_sum.to_f64 / pair_count : 0.0
  pair_weight_overlap_mean = pair_count > 0 ? pair_weight_overlap_sum / pair_count : 0.0

  RouteOverlapStats.new(
    layer: layer,
    prompt_len: prompt_len,
    canvas_len: canvas_len,
    route_width: route_width,
    total_slots: total_slots,
    unique_experts: unique_experts,
    duplicate_slots: duplicate_slots,
    grouped_experts: grouped_experts,
    groupable_slots: groupable_slots,
    groupable_fraction: groupable_fraction,
    max_multiplicity: max_multiplicity,
    mean_multiplicity: mean_multiplicity,
    pair_count: pair_count,
    pair_overlap_mean: pair_overlap_mean,
    pair_overlap_max: pair_overlap_max,
    pair_weight_overlap_mean: pair_weight_overlap_mean,
  )
end

def format_float(value : Float64) : String
  "%.6f" % value
end

def print_keyvalue(stats : RouteOverlapStats, plan : ML::GGUF::DiffusionGemmaCPU::CogniGraphPlan? = nil) : Nil
  names = TSV_HEADER
  values = stats.values
  names.each_with_index do |name, i|
    puts "#{name}=#{values[i]}"
  end
  if plan
    PLAN_TSV_HEADER.each_with_index do |name, i|
      puts "#{name}=#{plan_values(plan)[i]}"
    end
  end
end

TSV_HEADER = [
  "layer",
  "prompt_len",
  "canvas_len",
  "route_width",
  "total_slots",
  "unique_experts",
  "duplicate_slots",
  "grouped_experts",
  "groupable_slots",
  "groupable_fraction",
  "max_multiplicity",
  "mean_multiplicity",
  "pair_count",
  "pair_overlap_mean",
  "pair_overlap_max",
  "pair_weight_overlap_mean",
]

PLAN_TSV_HEADER = [
  "plan_ops",
  "plan_waves",
  "plan_barriers",
  "plan_max_wave_width",
  "plan_active_experts",
  "plan_route_slots",
  "plan_wave_widths",
  "plan_phi",
]

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_token = 1
canvas_token = 0
prompt_lengths_arg = "128"
canvas_lengths_arg = "2,4,8"
max_layers = 1
format = "tsv"
with_cognigraph_plan = false

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_moe_route_overlap_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--canvas-token ID", "Synthetic canvas start token id (default: 0)") { |v| canvas_token = v.to_i }
  p.on("--prompt-lengths LIST", "Synthetic prompt lengths, comma or space separated (default: 128)") { |v| prompt_lengths_arg = v }
  p.on("--canvas-lengths LIST", "Synthetic canvas lengths, comma or space separated (default: 2,4,8)") { |v| canvas_lengths_arg = v }
  p.on("--max-layers N", "Collect route overlap before each of N decode layers (default: 1)") { |v| max_layers = v.to_i }
  p.on("--format FORMAT", "Output format: tsv or keyvalue (default: tsv)") { |v| format = v.downcase }
  p.on("--with-cognigraph-plan", "Append CogniGraph dry-plan ops/waves/barriers columns") { with_cognigraph_plan = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--format must be tsv or keyvalue" unless {"tsv", "keyvalue"}.includes?(format)
raise "--max-layers must be positive" unless max_layers > 0

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer

prompt_lengths = parse_positive_counts(prompt_lengths_arg, "--prompt-lengths")
canvas_lengths = parse_positive_counts(canvas_lengths_arg, "--canvas-lengths")

puts "load_ms=#{format_float(load_ms)}" if format == "keyvalue"
puts((with_cognigraph_plan ? TSV_HEADER + PLAN_TSV_HEADER : TSV_HEADER).join('\t')) if format == "tsv"

prompt_lengths.each do |prompt_len|
  prompt_tokens = generated_token_sequence(prompt_token, prompt_len, hp.vocab_size, "--prompt-lengths entry")
  prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)

  canvas_lengths.each do |canvas_len|
    raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length

    canvas_tokens = generated_token_sequence(canvas_token, canvas_len, hp.vocab_size, "--canvas-lengths entry")
    canvas_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(weights, canvas_tokens)
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(
      prompt_len: prompt_len,
      canvas_len: canvas_len,
      sliding_window: hp.sliding_window,
    )
    prompt_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      materialize_final_rows: false,
    )

    rows = canvas_rows
    max_layers.times do |layer|
      attn_out_rows = attention_residual_rows(
        weights,
        layer,
        prompt_cache.projections_by_layer[layer],
        rows,
        mask,
      )
      routes = route_rows(weights, layer, attn_out_rows, canvas_len)
      stats = overlap_stats(routes, layer, prompt_len, canvas_len)
      plan = with_cognigraph_plan ? ML::GGUF::DiffusionGemmaCPU.grouped_moe_cognigraph_plan(routes, hp.n_embd, hp.expert_ff, hp.expert_count) : nil
      if format == "tsv"
        puts((plan ? stats.values + plan_values(plan) : stats.values).join('\t'))
      else
        print_keyvalue(stats, plan)
      end

      timed = ML::GGUF::DiffusionGemmaCPU.layer_forward_decode_canvas_rows_with_prompt_projections_timed(
        weights: weights,
        il: layer,
        prompt_projections: prompt_cache.projections_by_layer[layer],
        canvas_rows: rows,
        mask: mask,
        prompt_metal_cache: prompt_cache.metal_cache_by_layer[layer]?,
      )
      rows = timed.rows
    end
  end
end
