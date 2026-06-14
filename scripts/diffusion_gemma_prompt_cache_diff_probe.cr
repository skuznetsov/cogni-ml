require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 16
canvas_len = 8
max_layers = 30
base_env = "DIFFUSION_GEMMA_C8_RESIDENT_DECODE_POLICY=1 DIFFUSION_GEMMA_PROMPT_CACHE_POLICY=1"
variant_env = "#{base_env} DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_DOWN_COMMAND=1"

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_prompt_cache_diff_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 16)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Canvas length for mask/cache shape (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--max-layers N", "Layer count to materialize (default: 30)") { |v| max_layers = v.to_i }
  p.on("--base-env TEXT", "Whitespace-separated KEY=VALUE env for base run") { |v| base_env = v }
  p.on("--variant-env TEXT", "Whitespace-separated KEY=VALUE env for variant run") { |v| variant_env = v }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

def parse_env_assignments(text : String) : Hash(String, String)
  env = {} of String => String
  text.split.each do |token|
    key, value = token.split("=", 2)
    raise "invalid env assignment: #{token}" unless key && value
    env[key] = value
  end
  env
end

def with_env(assignments : Hash(String, String), &block)
  old_env = assignments.keys.to_h { |key| {key, ENV[key]?} }
  begin
    assignments.each { |key, value| ENV[key] = value }
    yield
  ensure
    old_env.each do |key, value|
      if value
        ENV[key] = value
      else
        ENV.delete(key)
      end
    end
  end
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

def build_cache(weights : ML::GGUF::DiffusionGemmaWeights,
                prompt_rows : Array(Float32),
                mask : ML::GGUF::DiffusionGemmaAttentionMask,
                max_layers : Int32,
                env_text : String) : ML::GGUF::DiffusionGemmaCPU::PromptLayerCache
  with_env(parse_env_assignments(env_text)) do
    return ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
    )
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt-len must be positive" unless prompt_len > 0
raise "--canvas-len must be positive" unless canvas_len > 0
raise "--max-layers must be positive" unless max_layers > 0

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length

prompt_rows = [] of Float32
prompt_len.times do |i|
  prompt_rows.concat(ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(weights, i + 1))
end
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(
  prompt_len: prompt_len,
  canvas_len: canvas_len,
  sliding_window: hp.sliding_window,
)

base_t0 = Time.instant
base = build_cache(weights, prompt_rows, mask, max_layers, base_env)
base_ms = (Time.instant - base_t0).total_milliseconds
variant_t0 = Time.instant
variant = build_cache(weights, prompt_rows, mask, max_layers, variant_env)
variant_ms = (Time.instant - variant_t0).total_milliseconds
stats = diff_stats(base.final_rows, variant.final_rows)

puts "load_ms=#{"%.3f" % load_ms}"
puts "prompt_len=#{prompt_len}"
puts "canvas_len=#{canvas_len}"
puts "max_layers=#{max_layers}"
puts "base_ms=#{"%.3f" % base_ms}"
puts "variant_ms=#{"%.3f" % variant_ms}"
puts "base_env=#{base_env}"
puts "variant_env=#{variant_env}"
puts "base_materialize_ms=#{"%.3f" % base.materialize_ms_by_layer.sum}"
puts "variant_materialize_ms=#{"%.3f" % variant.materialize_ms_by_layer.sum}"
puts "base_moe_ms=#{"%.3f" % base.materialize_moe_ffn_ms_by_layer.sum}"
puts "variant_moe_ms=#{"%.3f" % variant.materialize_moe_ffn_ms_by_layer.sum}"
puts "max_abs=#{"%.9f" % stats[:max_abs]}"
puts "mean_abs=#{"%.9f" % stats[:mean_abs]}"
puts "checksum_base=#{"%.9f" % stats[:checksum_a]}"
puts "checksum_variant=#{"%.9f" % stats[:checksum_b]}"
puts "checksum_delta=#{"%.9f" % stats[:checksum_delta]}"
