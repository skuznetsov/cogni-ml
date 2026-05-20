# Resident-process request timing probe for Qwen 3.5/3.6 Metal inference.
#
# This is intentionally a benchmark probe, not a product server. It loads the
# model/tokenizer once, optionally performs explicit warmup requests, then times
# fresh request states inside the same process so one-shot CLI costs are not
# confused with steady-state request latency.

require "option_parser"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL_PATH
tokenizer_bin = ENV["QWEN35_TOKENIZER_BIN"]? || DEFAULT_TOKENIZER_BIN
prompt = "The capital of France is"
n_gen = 16
requests = 5
warmups = 1
max_seq = 1024
prepare_state = ENV["QWEN35_PREPARE_STATE_OFF"]? != "1"
quiet = false

parser = OptionParser.new do |p|
  p.banner = "Usage: qwen35_warm_request_probe [options] [prompt]"
  p.on("--model PATH", "Target GGUF model path (default: QWEN35_MODEL or local 9B)") { |v| model_path = v }
  p.on("--tokenizer-bin PATH", "External llama-tokenize fallback path") { |v| tokenizer_bin = v }
  p.on("--gen N", "Generated token count per request (default: 16)") { |v| n_gen = v.to_i }
  p.on("--requests N", "Measured request count (default: 5)") { |v| requests = v.to_i }
  p.on("--warmups N", "Explicit unmeasured warmup request count (default: 1)") { |v| warmups = v.to_i }
  p.on("--max-seq N", "State max sequence length (default: 1024)") { |v| max_seq = v.to_i }
  p.on("--no-prepare-state", "Do not eagerly prepare Metal state buffers before prefill") { prepare_state = false }
  p.on("--quiet", "Suppress generated token id rows") { quiet = true }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end
parser.parse(ARGV)
prompt = ARGV.join(" ") unless ARGV.empty?

raise "--gen must be positive" unless n_gen > 0
raise "--requests must be positive" unless requests > 0
raise "--warmups must be non-negative" unless warmups >= 0
raise "--max-seq must be positive" unless max_seq > 0

struct RequestTiming
  getter total_ms : Float64
  getter tokenize_ms : Float64
  getter state_prepare_ms : Float64
  getter prefill_ms : Float64
  getter decode_ms : Float64
  getter prompt_tokens : Int32
  getter output_tokens : Int32
  getter first_token : Int32

  def initialize(@total_ms, @tokenize_ms, @state_prepare_ms, @prefill_ms, @decode_ms,
                 @prompt_tokens, @output_tokens, @first_token)
  end

  def ms_per_output : Float64
    output_tokens > 0 ? total_ms / output_tokens : 0.0
  end
end

def run_request(weights : ML::GGUF::Qwen35Weights,
                tokenizer : ML::GGUF::Qwen35Tokenizer,
                prompt : String,
                n_gen : Int32,
                max_seq : Int32,
                prepare_state : Bool) : {RequestTiming, Array(Int32)}
  hp = weights.hparams
  request_t0 = Time.instant

  tokenize_t0 = Time.instant
  ids = tokenizer.encode(prompt)
  tokenize_ms = (Time.instant - tokenize_t0).total_milliseconds
  raise "prompt encoded to zero tokens" if ids.empty?
  raise "request exceeds max_seq: prompt=#{ids.size} gen=#{n_gen} max_seq=#{max_seq}" if ids.size + n_gen >= max_seq

  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)

  state_prepare_ms = 0.0
  if prepare_state
    prepare_t0 = Time.instant
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
    state_prepare_ms = (Time.instant - prepare_t0).total_milliseconds
  end

  prefill_t0 = Time.instant
  top, _top_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, ids, 0, state)
  prefill_ms = (Time.instant - prefill_t0).total_milliseconds

  output_ids = [top]
  pos = ids.size
  decode_t0 = Time.instant
  while output_ids.size < n_gen
    next_id, _next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, output_ids[-1], pos, state)
    output_ids << next_id
    pos += 1
    break if next_id == tokenizer.eos_id
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  total_ms = (Time.instant - request_t0).total_milliseconds
  timing = RequestTiming.new(
    total_ms,
    tokenize_ms,
    state_prepare_ms,
    prefill_ms,
    decode_ms,
    ids.size,
    output_ids.size,
    output_ids[0],
  )
  {timing, output_ids}
end

startup_t0 = Time.instant
g = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(g, model_path, tokenizer_bin)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
startup_ms = (Time.instant - startup_t0).total_milliseconds

puts "qwen35_warm_request_probe"
puts "  model=#{model_path}"
puts "  prompt=#{prompt.inspect} gen=#{n_gen} requests=#{requests} warmups=#{warmups} max_seq=#{max_seq} prepare_state=#{prepare_state}"
puts "  startup_ms=#{startup_ms.round(1)}"

warmup_ms = 0.0
warmups.times do
  warm_t0 = Time.instant
  run_request(weights, tokenizer, prompt, n_gen, max_seq, prepare_state)
  warmup_ms += (Time.instant - warm_t0).total_milliseconds
end
puts "  explicit_warmup_ms=#{warmup_ms.round(1)}" if warmups > 0

timings = [] of RequestTiming
requests.times do |i|
  timing, output_ids = run_request(weights, tokenizer, prompt, n_gen, max_seq, prepare_state)
  timings << timing
  unless quiet
    puts "  request #{i + 1}: ids=#{output_ids.inspect}"
  end
  puts "  request #{i + 1} summary: total_ms=#{timing.total_ms.round(1)} ms_per_tok=#{timing.ms_per_output.round(2)} tokenize_ms=#{timing.tokenize_ms.round(1)} state_prepare_ms=#{timing.state_prepare_ms.round(1)} prefill_ms=#{timing.prefill_ms.round(1)} decode_ms=#{timing.decode_ms.round(1)} prompt_tokens=#{timing.prompt_tokens} output_tokens=#{timing.output_tokens} first_token=#{timing.first_token}"
end

totals = timings.map(&.total_ms).sort
decode_totals = timings.map(&.decode_ms).sort
prefills = timings.map(&.prefill_ms).sort
mid = totals.size // 2
avg_total = timings.sum(&.total_ms) / timings.size
avg_tok = timings.sum(&.output_tokens).to_f
avg_ms_per_tok = timings.sum(&.total_ms) / avg_tok

puts "  aggregate: avg_total_ms=#{avg_total.round(1)} p50_total_ms=#{totals[mid].round(1)} avg_ms_per_tok=#{avg_ms_per_tok.round(2)} p50_prefill_ms=#{prefills[mid].round(1)} p50_decode_ms=#{decode_totals[mid].round(1)}"
