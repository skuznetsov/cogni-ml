require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_ENV   = "DIFFUSION_GEMMA_C8_RESIDENT_DECODE_POLICY=1 DIFFUSION_GEMMA_PROMPT_CACHE_POLICY=1 DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_GRAPH=1 DIFFUSION_GEMMA_MOE_GROUPED_RESIDENT_BATCH_GRAPH_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_GATHER_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_REDUCE_MAX_CANVAS=16 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM=1 DIFFUSION_GEMMA_MOE_GROUPED_GPU_PRENORM_MAX_CANVAS=16 DIFFUSION_GEMMA_FFN_RESIDENT_SCRATCH=1 DIFFUSION_GEMMA_FFN_RESIDENT_GRAPH_CACHE=1"

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

def max_abs_delta(a : Array(Float32), b : Array(Float32)) : Float64
  raise "row size mismatch" unless a.size == b.size
  max = 0.0
  a.each_with_index do |value, i|
    delta = (value.to_f64 - b[i].to_f64).abs
    max = delta if delta > max
  end
  max
end

def format_f64(value : Float64) : String
  "%.6f" % value
end

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 16
canvas_len = 8
prompt_token = 1
max_layers = 30
env = ArmEnv.new(ENV["ROUTE_REPLAY_ENV"]? || DEFAULT_ENV)
clear_graph_cache_between = false
max_abs_tolerance = 1e-5
warmups = 1

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_prompt_route_replay_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 16)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Synthetic canvas length for mask shape (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--max-layers N", "Prompt-cache layer count (default: 30)") { |v| max_layers = v.to_i }
  p.on("--env ENV", "Whitespace-separated KEY=VALUE env for both source and replay") { |v| env = ArmEnv.new(v) }
  p.on("--clear-graph-cache-between", "Clear resident graph caches between source and route replay builds") { clear_graph_cache_between = true }
  p.on("--max-abs-tolerance F", "Fail when replay final rows differ above F (default: 1e-5)") { |v| max_abs_tolerance = v.to_f64 }
  p.on("--warmups N", "Unmeasured source builds before timed source/replay (default: 1)") { |v| warmups = v.to_i }
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
raise "--max-abs-tolerance must be finite and non-negative" unless max_abs_tolerance.finite? && max_abs_tolerance >= 0.0

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length
raise "prompt token start out of range" if prompt_token < 0 || prompt_token >= hp.vocab_size

prompt_tokens = generated_token_sequence(prompt_token, prompt_len, hp.vocab_size, "--prompt-len")
prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)

old = apply_env(env)
begin
  ML::GGUF::DiffusionGemmaCPU.clear_ffn_resident_graph_cache
  warmups.times do
    ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      materialize_final_rows: true,
    )
    ML::GGUF::DiffusionGemmaCPU.clear_ffn_resident_graph_cache
  end

  source_t0 = Time.instant
  source = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
    weights,
    prompt_rows,
    mask,
    max_layers: max_layers,
    materialize_final_rows: true,
  )
  source_ms = (Time.instant - source_t0).total_milliseconds
  routes = source.routes_by_layer_by_prompt_row
  raise "source did not capture full-depth routes: #{routes.size}/#{max_layers}" unless routes.size == max_layers

  ML::GGUF::DiffusionGemmaCPU.clear_ffn_resident_graph_cache if clear_graph_cache_between
  replay_t0 = Time.instant
  replay = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
    weights,
    prompt_rows,
    mask,
    max_layers: max_layers,
    routes_by_layer_by_prompt_row: routes,
    materialize_final_rows: true,
  )
  replay_ms = (Time.instant - replay_t0).total_milliseconds

  max_abs = max_abs_delta(source.final_rows, replay.final_rows)
  checksum_source = checksum_rows(source.final_rows)
  checksum_replay = checksum_rows(replay.final_rows)
  checksum_delta = (checksum_source - checksum_replay).abs
  route_slots = routes.sum { |layer| layer.sum(&.size) }
  active_rows = routes.sum(&.size)
  source_prep_ms = source.materialize_moe_grouped_prep_ms_by_layer.sum
  replay_prep_ms = replay.materialize_moe_grouped_prep_ms_by_layer.sum
  source_moe_ms = source.materialize_moe_ffn_ms_by_layer.sum
  replay_moe_ms = replay.materialize_moe_ffn_ms_by_layer.sum
  total_speedup = replay_ms > 0.0 ? source_ms / replay_ms : 0.0
  prep_speedup = replay_prep_ms > 0.0 ? source_prep_ms / replay_prep_ms : 0.0

  puts "# load_ms=#{format_f64(load_ms)}"
  puts "# env=#{env.raw.empty? ? "<empty>" : env.raw}"
  puts [
    "summary",
    "prompt_len=#{prompt_len}",
    "canvas_len=#{canvas_len}",
    "max_layers=#{max_layers}",
    "warmups=#{warmups}",
    "route_layers=#{routes.size}",
    "route_rows=#{active_rows}",
    "route_slots=#{route_slots}",
    "source_ms=#{format_f64(source_ms)}",
    "replay_ms=#{format_f64(replay_ms)}",
    "total_speedup=#{format_f64(total_speedup)}",
    "source_prep_ms=#{format_f64(source_prep_ms)}",
    "replay_prep_ms=#{format_f64(replay_prep_ms)}",
    "prep_speedup=#{format_f64(prep_speedup)}",
    "source_moe_ms=#{format_f64(source_moe_ms)}",
    "replay_moe_ms=#{format_f64(replay_moe_ms)}",
    "max_abs=#{format_f64(max_abs)}",
    "checksum_source=#{format_f64(checksum_source)}",
    "checksum_replay=#{format_f64(checksum_replay)}",
    "checksum_delta=#{format_f64(checksum_delta)}",
    "clear_graph_cache_between=#{clear_graph_cache_between}",
  ].join('\t')

  if max_abs > max_abs_tolerance
    STDERR.puts "route_replay status=fail max_abs=#{format_f64(max_abs)} tolerance=#{format_f64(max_abs_tolerance)}"
    exit 4
  end
  STDERR.puts "route_replay status=ok max_abs=#{format_f64(max_abs)} route_layers=#{routes.size}"
ensure
  restore_env(old)
end
