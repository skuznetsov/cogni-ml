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

def print_keyvalue(stats : RouteOverlapStats) : Nil
  names = TSV_HEADER
  values = stats.values
  names.each_with_index do |name, i|
    puts "#{name}=#{values[i]}"
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

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_token = 1
canvas_token = 0
prompt_lengths_arg = "128"
canvas_lengths_arg = "2,4,8"
max_layers = 1
format = "tsv"

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_moe_route_overlap_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--canvas-token ID", "Synthetic canvas start token id (default: 0)") { |v| canvas_token = v.to_i }
  p.on("--prompt-lengths LIST", "Synthetic prompt lengths, comma or space separated (default: 128)") { |v| prompt_lengths_arg = v }
  p.on("--canvas-lengths LIST", "Synthetic canvas lengths, comma or space separated (default: 2,4,8)") { |v| canvas_lengths_arg = v }
  p.on("--max-layers N", "Collect route overlap before each of N decode layers (default: 1)") { |v| max_layers = v.to_i }
  p.on("--format FORMAT", "Output format: tsv or keyvalue (default: tsv)") { |v| format = v.downcase }
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
puts TSV_HEADER.join('\t') if format == "tsv"

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
      stats = overlap_stats(route_rows(weights, layer, rows, canvas_len), layer, prompt_len, canvas_len)
      if format == "tsv"
        puts stats.values.join('\t')
      else
        print_keyvalue(stats)
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
