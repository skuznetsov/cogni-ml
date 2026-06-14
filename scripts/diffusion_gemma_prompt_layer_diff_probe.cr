require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 16
canvas_len = 8
prompt_token = 1
layer = 0

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_prompt_layer_diff_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 16)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Canvas length for mask/cache shape (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--layer N", "Layer to compare (default: 0)") { |v| layer = v.to_i }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
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

def append_row!(rows : Array(Float32), row : Array(Float32)) : Nil
  rows.concat(row)
end

def row_slice(rows : Array(Float32), row : Int32, width : Int32) : Array(Float32)
  rows[row * width, width]
end

def copy_row!(dst : Array(Float32), row : Int32, width : Int32, src : Array(Float32)) : Nil
  (dst.to_unsafe + row * width).copy_from(src.to_unsafe, width)
end

def diff_stats(a : Array(Float32), b : Array(Float32)) : NamedTuple(max_abs: Float64, mean_abs: Float64, checksum_delta: Float64)
  raise "diff size mismatch #{a.size} != #{b.size}" unless a.size == b.size
  max_abs = 0.0
  sum_abs = 0.0
  checksum_delta = 0.0
  a.size.times do |i|
    delta = (a[i] - b[i]).to_f64
    abs = delta.abs
    max_abs = abs if abs > max_abs
    sum_abs += abs
    checksum_delta += delta
  end
  {
    max_abs:        max_abs,
    mean_abs:       a.empty? ? 0.0 : sum_abs / a.size,
    checksum_delta: checksum_delta,
  }
end

def print_diff(name : String, a : Array(Float32), b : Array(Float32)) : Nil
  stats = diff_stats(a, b)
  puts "component=#{name} max_abs=#{"%.9f" % stats[:max_abs]} mean_abs=#{"%.9f" % stats[:mean_abs]} checksum_delta=#{"%.9f" % stats[:checksum_delta]}"
end

def route_diff_stats(a : Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)),
                     b : Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute))) : NamedTuple(rows: Int32, expert_mismatch: Int32, weight_max_abs: Float64, weight_mean_abs: Float64)
  raise "route row count mismatch" unless a.size == b.size
  expert_mismatch = 0
  weight_max_abs = 0.0
  weight_sum_abs = 0.0
  weight_count = 0
  a.each_with_index do |routes, row|
    other = b[row]
    raise "route size mismatch at row #{row}" unless routes.size == other.size
    routes.each_with_index do |route, i|
      other_route = other[i]
      expert_mismatch += 1 if route.expert != other_route.expert
      diff = (route.weight - other_route.weight).abs.to_f64
      weight_max_abs = diff if diff > weight_max_abs
      weight_sum_abs += diff
      weight_count += 1
    end
  end
  {
    rows:            a.size,
    expert_mismatch: expert_mismatch,
    weight_max_abs:  weight_max_abs,
    weight_mean_abs: weight_count == 0 ? 0.0 : weight_sum_abs / weight_count,
  }
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt-len must be positive" unless prompt_len > 0
raise "--canvas-len must be positive" unless canvas_len > 0
raise "--layer must be non-negative" unless layer >= 0

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--layer exceeds model layer count" if layer >= hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length

prompt_tokens = generated_token_sequence(prompt_token, prompt_len, hp.vocab_size, "--prompt-len")
prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)

projection_t0 = Time.instant
projections = ML::GGUF::DiffusionGemmaCPU.prompt_attention_projections(weights, layer, prompt_rows, mask)
projection_ms = (Time.instant - projection_t0).total_milliseconds

q_context_dim = hp.n_head * hp.head_dim_for_layer(layer)
context_rows = Array(Float32).new(prompt_len * q_context_dim, 0.0_f32)
scalar_attn_rows = Array(Float32).new(prompt_len * hp.n_embd, 0.0_f32)
prompt_len.times do |pos|
  x = row_slice(prompt_rows, pos, hp.n_embd)
  context = ML::GGUF::DiffusionGemmaCPU.attention_context_prompt(
    projections,
    hp,
    layer,
    query_pos: pos,
    sliding_window: mask.sliding_window,
  )
  copy_row!(context_rows, pos, q_context_dim, context)
  attn = ML::GGUF::DiffusionGemmaCPU.attention_residual_from_context(weights, layer, x, context)
  copy_row!(scalar_attn_rows, pos, hp.n_embd, attn)
end

batched_attn_rows = ML::GGUF::DiffusionGemmaCPU.attention_residual_from_context_rows(
  weights,
  layer,
  prompt_rows,
  context_rows,
  prompt_len,
)

scalar_routes = Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)).new(prompt_len) do |row|
  ML::GGUF::DiffusionGemmaCPU.route_experts(weights, layer, row_slice(scalar_attn_rows, row, hp.n_embd))
end
batched_routes = ML::GGUF::DiffusionGemmaCPU.route_experts_rows(weights, layer, batched_attn_rows, prompt_len)

scalar_shared_rows = [] of Float32
prompt_len.times do |row|
  append_row!(scalar_shared_rows, ML::GGUF::DiffusionGemmaCPU.shared_dense_ffn(weights, layer, row_slice(scalar_attn_rows, row, hp.n_embd)))
end
batched_shared_from_scalar_attn = ML::GGUF::DiffusionGemmaCPU.shared_dense_ffn_rows(weights, layer, scalar_attn_rows, prompt_len)
batched_shared_rows = ML::GGUF::DiffusionGemmaCPU.shared_dense_ffn_rows(weights, layer, batched_attn_rows, prompt_len)

scalar_moe_rows = [] of Float32
prompt_len.times do |row|
  append_row!(scalar_moe_rows, ML::GGUF::DiffusionGemmaCPU.moe_ffn(weights, layer, row_slice(scalar_attn_rows, row, hp.n_embd), scalar_routes[row]))
end
batched_moe_from_scalar_attn_routes = ML::GGUF::DiffusionGemmaCPU.moe_ffn_rows(weights, layer, scalar_attn_rows, prompt_len, scalar_routes)
batched_moe_rows = ML::GGUF::DiffusionGemmaCPU.moe_ffn_rows(weights, layer, batched_attn_rows, prompt_len, batched_routes)

scalar_final_rows = [] of Float32
prompt_len.times do |row|
  ffn = ML::GGUF::DiffusionGemmaCPU.ffn_residual_from_parts(
    weights,
    layer,
    row_slice(scalar_attn_rows, row, hp.n_embd),
    row_slice(scalar_shared_rows, row, hp.n_embd),
    row_slice(scalar_moe_rows, row, hp.n_embd),
  )
  append_row!(scalar_final_rows, ML::GGUF::DiffusionGemmaCPU.scale_layer_output(weights, layer, ffn, canvas: false))
end
batched_final_rows = ML::GGUF::DiffusionGemmaCPU.ffn_residual_from_parts_rows(
  weights,
  layer,
  batched_attn_rows,
  batched_shared_rows,
  batched_moe_rows,
  prompt_len,
  canvas: false,
)

route_stats = route_diff_stats(scalar_routes, batched_routes)
puts "load_ms=#{"%.3f" % load_ms}"
puts "projection_ms=#{"%.3f" % projection_ms}"
puts "prompt_len=#{prompt_len}"
puts "canvas_len=#{canvas_len}"
puts "layer=#{layer}"
puts "projection_backend=#{ML::GGUF::DiffusionGemmaCPU.prompt_projection_metal_enabled? && prompt_len >= ML::GGUF::DiffusionGemmaCPU.prompt_projection_metal_min_batch ? "metal" : "cpu"}"
puts "fused_norm_rope=#{ML::GGUF::DiffusionGemmaCPU.prompt_projection_fused_norm_rope_enabled?}"
puts "attention_residual_metal_rows=#{ML::GGUF::DiffusionGemmaCPU.attention_residual_metal_rows_enabled?(prompt_len)}"
print_diff("attn_out.scalar_vs_batched", scalar_attn_rows, batched_attn_rows)
puts "component=routes rows=#{route_stats[:rows]} expert_mismatch=#{route_stats[:expert_mismatch]} weight_max_abs=#{"%.9f" % route_stats[:weight_max_abs]} weight_mean_abs=#{"%.9f" % route_stats[:weight_mean_abs]}"
print_diff("shared.same_scalar_attn.scalar_vs_batched", scalar_shared_rows, batched_shared_from_scalar_attn)
print_diff("shared.full.scalar_vs_batched", scalar_shared_rows, batched_shared_rows)
print_diff("moe.same_scalar_attn_routes.scalar_vs_batched", scalar_moe_rows, batched_moe_from_scalar_attn_routes)
print_diff("moe.full.scalar_vs_batched", scalar_moe_rows, batched_moe_rows)
print_diff("final.scalar_vs_batched", scalar_final_rows, batched_final_rows)
