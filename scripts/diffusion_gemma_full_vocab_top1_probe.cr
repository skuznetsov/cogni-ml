require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/reader"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_ENV   = "DIFFUSION_GEMMA_C8_RESIDENT_DECODE_POLICY=1 DIFFUSION_GEMMA_PROMPT_CACHE_POLICY=1"

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

def format_f64(value : Float64) : String
  "%.6f" % value
end

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_len = 16
canvas_len = 8
prompt_token = 17
canvas_token = 100
max_layers = 30
rows_arg = "0"
env = ArmEnv.new(ENV["TOP1_ENV"]? || DEFAULT_ENV)
metal_warmups = 1
check_top2 = false
batched_top1_metal = false

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_full_vocab_top1_probe [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-len N", "Synthetic prompt length (default: 16)") { |v| prompt_len = v.to_i }
  p.on("--canvas-len N", "Synthetic canvas length (default: 8)") { |v| canvas_len = v.to_i }
  p.on("--prompt-token ID", "Synthetic prompt start token id (default: 17)") { |v| prompt_token = v.to_i }
  p.on("--canvas-token ID", "Synthetic canvas start token id (default: 100)") { |v| canvas_token = v.to_i }
  p.on("--max-layers N", "Prompt-cache/decode layer count (default: 30)") { |v| max_layers = v.to_i }
  p.on("--rows LIST", "Comma/space separated canvas row indexes to check (default: 0)") { |v| rows_arg = v }
  p.on("--env ENV", "Whitespace-separated KEY=VALUE env for prompt/decode route") { |v| env = ArmEnv.new(v) }
  p.on("--metal-warmups N", "Unmeasured Metal top1 calls on the first checked row (default: 1)") { |v| metal_warmups = v.to_i }
  p.on("--top2", "Compare CPU and Metal full-vocab top2 ids/logits instead of top1") { check_top2 = true }
  p.on("--batched-top1-metal", "Use the row-batched Metal top1 helper for top1 checks") { batched_top1_metal = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--prompt-len must be positive" unless prompt_len > 0
raise "--canvas-len must be positive" unless canvas_len > 0
raise "--max-layers must be positive" unless max_layers > 0
raise "--metal-warmups must be non-negative" if metal_warmups < 0
raise "--batched-top1-metal is incompatible with --top2" if batched_top1_metal && check_top2
rows = rows_arg.split(/[,\s]+/).reject(&.empty?).map(&.to_i)
raise "--rows must contain at least one row" if rows.empty?

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
raise "--max-layers exceeds model layer count" if max_layers > hp.n_layer
raise "prompt+canvas exceeds context_length" if prompt_len + canvas_len > hp.context_length
raise "prompt token start out of range" if prompt_token < 0 || prompt_token >= hp.vocab_size
raise "canvas token start out of range" if canvas_token < 0 || canvas_token >= hp.vocab_size
rows.each do |row|
  raise "row index #{row} out of range 0...#{canvas_len}" if row < 0 || row >= canvas_len
end

prompt_tokens = generated_token_sequence(prompt_token, prompt_len, hp.vocab_size, "--prompt-len")
canvas_tokens = generated_token_sequence(canvas_token, canvas_len, hp.vocab_size, "--canvas-len")
prompt_rows = prompt_rows_from_tokens(weights, prompt_tokens)
canvas_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(weights, canvas_tokens)
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: prompt_len, canvas_len: canvas_len, sliding_window: hp.sliding_window)

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

  decode_t0 = Time.instant
  decode = ML::GGUF::DiffusionGemmaCPU.decode_canvas_rows_with_prompt_cache_timed(
    weights: weights,
    canvas_rows: canvas_rows,
    mask: mask,
    prompt_cache: cache,
    max_layers: max_layers,
  )
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  puts "# load_ms=#{format_f64(load_ms)}"
  puts "# env=#{env.raw.empty? ? "<empty>" : env.raw}"
  puts [
    "kind",
    "row",
    "cpu_token",
    "metal_token",
    "match",
    "cpu_logit",
    "metal_logit",
    "logit_abs_delta",
    "cpu_ms",
    "metal_ms",
  ].join('\t')

  checked = 0
  matches = 0
  total_cpu_ms = 0.0
  total_metal_ms = 0.0
  max_logit_delta = 0.0
  first_hidden = decode.rows[rows[0] * hp.n_embd, hp.n_embd]
  metal_warmups.times do
    if check_top2
      ML::GGUF::DiffusionGemmaCPU.output_top2_full_vocab_metal(weights, first_hidden)
    elsif batched_top1_metal
      ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_rows_metal(weights, decode.rows, mask.canvas_len)
    else
      ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_metal(weights, first_hidden)
    end
  end

  batched_top1s = nil.as(Array(ML::GGUF::DiffusionGemmaCPU::OutputTop1)?)
  batched_top1_ms = 0.0
  if batched_top1_metal
    batched_t0 = Time.instant
    batched_top1s = ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_rows_metal(weights, decode.rows, mask.canvas_len)
    batched_top1_ms = (Time.instant - batched_t0).total_milliseconds
    raise "Metal full-vocab batched top1 unavailable" unless batched_top1s
  end

  rows.each do |row|
    hidden = decode.rows[row * hp.n_embd, hp.n_embd]

    cpu_t0 = Time.instant
    cpu = check_top2 ? ML::GGUF::DiffusionGemmaCPU.output_top2_full_vocab_cpu(weights, hidden) : ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_cpu(weights, hidden)
    cpu_ms = (Time.instant - cpu_t0).total_milliseconds

    metal_t0 = Time.instant
    metal = if batched_top1s
              batched_top1s.not_nil![row]
            else
              check_top2 ? ML::GGUF::DiffusionGemmaCPU.output_top2_full_vocab_metal(weights, hidden) : ML::GGUF::DiffusionGemmaCPU.output_top1_full_vocab_metal(weights, hidden)
            end
    metal_ms = batched_top1s && row != rows[0] ? 0.0 : (batched_top1s ? batched_top1_ms : (Time.instant - metal_t0).total_milliseconds)
    raise "Metal full-vocab #{check_top2 ? "top2" : "top1"} unavailable" unless metal

    metal_result = metal.not_nil!
    match = cpu.token_id == metal_result.token_id
    delta = (cpu.logit.to_f64 - metal_result.logit.to_f64).abs
    if check_top2
      cpu_top2 = cpu.as(ML::GGUF::DiffusionGemmaCPU::OutputTop2)
      metal_top2 = metal_result.as(ML::GGUF::DiffusionGemmaCPU::OutputTop2)
      second_delta = (cpu_top2.second_logit.to_f64 - metal_top2.second_logit.to_f64).abs
      delta = second_delta if second_delta > delta
      match &&= cpu_top2.second_token_id == metal_top2.second_token_id
    end
    max_logit_delta = delta if delta > max_logit_delta
    checked += 1
    matches += 1 if match
    total_cpu_ms += cpu_ms
    total_metal_ms += metal_ms

    puts [
      check_top2 ? "top2" : "top1",
      row.to_s,
      cpu.token_id.to_s,
      metal_result.token_id.to_s,
      match.to_s,
      format_f64(cpu.logit.to_f64),
      format_f64(metal_result.logit.to_f64),
      format_f64(delta),
      format_f64(cpu_ms),
      format_f64(metal_ms),
    ].join('\t')
  end

  speedup = total_metal_ms > 0.0 ? total_cpu_ms / total_metal_ms : 0.0
  puts [
    "summary",
    "rows=#{checked}",
    "matches=#{matches}/#{checked}",
    "all_match=#{matches == checked}",
    "load_ms=#{format_f64(load_ms)}",
    "cache_ms=#{format_f64(cache_ms)}",
    "decode_ms=#{format_f64(decode_ms)}",
    "cpu_#{check_top2 ? "top2" : "top1"}_ms=#{format_f64(total_cpu_ms)}",
    "metal_#{check_top2 ? "top2" : "top1"}_ms=#{format_f64(total_metal_ms)}",
    "metal_mode=#{batched_top1_metal ? "batched_rows" : "per_row"}",
    "#{check_top2 ? "top2" : "top1"}_speedup=#{format_f64(speedup)}",
    "max_logit_abs_delta=#{format_f64(max_logit_delta)}",
  ].join('\t')

  unless matches == checked
    STDERR.puts "full_vocab_#{check_top2 ? "top2" : "top1"} status=fail matches=#{matches}/#{checked}"
    exit 4
  end

  STDERR.puts "full_vocab_#{check_top2 ? "top2" : "top1"} status=ok matches=#{matches}/#{checked}"
ensure
  restore_env(old)
end
