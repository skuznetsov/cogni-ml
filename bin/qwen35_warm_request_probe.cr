# Resident-process request timing probe for Qwen 3.5/3.6 Metal inference.
#
# This is intentionally a benchmark probe, not a product server. It loads the
# model/tokenizer once, optionally performs explicit warmup requests, then times
# fresh request states inside the same process so one-shot CLI costs are not
# confused with steady-state request latency.

require "option_parser"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_prompt_cache"
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
source_replay = false
prompt_cache_replay = false
resident_states = (ENV["QWEN35_PROMPT_CACHE_RESIDENT_STATES"]? || "0").to_i
metal_profile = ENV["QWEN35_METAL_PROFILE"]? == "1"
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
  p.on("--source-replay", "Measure resident exact source-history replay instead of plain greedy decode") { source_replay = true }
  p.on("--prompt-cache-replay", "Measure real Store restore + exact source replay in a resident process") { prompt_cache_replay = true }
  p.on("--resident-states N", "Resident Store state-cache entries for --prompt-cache-replay") { |v| resident_states = v.to_i }
  p.on("--metal-profile", "Print Qwen35Metal profile report for measured requests") { metal_profile = true }
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
raise "--resident-states must be non-negative" unless resident_states >= 0
raise "--source-replay and --prompt-cache-replay are mutually exclusive" if source_replay && prompt_cache_replay

struct RequestTiming
  getter total_ms : Float64
  getter tokenize_ms : Float64
  getter state_prepare_ms : Float64
  getter restore_ms : Float64
  getter prefill_ms : Float64
  getter decode_ms : Float64
  getter prompt_tokens : Int32
  getter output_tokens : Int32
  getter first_token : Int32

  def initialize(@total_ms, @tokenize_ms, @state_prepare_ms, @restore_ms, @prefill_ms, @decode_ms,
                 @prompt_tokens, @output_tokens, @first_token)
  end

  def ms_per_output : Float64
    output_tokens > 0 ? total_ms / output_tokens : 0.0
  end
end

struct SourceReplayTemplate
  getter prompt_tokens : Int32
  getter state : ML::GGUF::Qwen35CPU::State
  getter output_ids : Array(Int32)

  def initialize(@prompt_tokens, @state, @output_ids)
  end
end

struct PromptCacheReplayTemplate
  getter store : ML::GGUF::Qwen35PromptCache::Store
  getter entry : ML::GGUF::Qwen35PromptCache::Entry
  getter prompt_tokens : Array(Int32)
  getter output_ids : Array(Int32)

  def initialize(@store, @entry, @prompt_tokens, @output_ids)
  end
end

def copy_prompt_state!(dst : ML::GGUF::Qwen35CPU::State,
                       src : ML::GGUF::Qwen35CPU::State,
                       hp : ML::GGUF::Qwen35Hparams,
                       used_tokens : Int32,
                       prepare_state : Bool) : Nil
  {% if flag?(:cpu_only) %}
    dst.copy_from!(src)
  {% else %}
    if prepare_state && ML::GGUF::Qwen35Metal.available?
      ML::GGUF::Qwen35CPU.copy_state_metal_used!(dst, src, hp, used_tokens: used_tokens)
    else
      dst.copy_from!(src)
    end
  {% end %}
end

def build_source_replay_template(weights : ML::GGUF::Qwen35Weights,
                                 tokenizer : ML::GGUF::Qwen35Tokenizer,
                                 prompt : String,
                                 n_gen : Int32,
                                 max_seq : Int32,
                                 prepare_state : Bool) : SourceReplayTemplate
  hp = weights.hparams
  ids = tokenizer.encode(prompt)
  raise "prompt encoded to zero tokens" if ids.empty?
  raise "request exceeds max_seq: prompt=#{ids.size} gen=#{n_gen} max_seq=#{max_seq}" if ids.size + n_gen >= max_seq

  prompt_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(prompt_state, hp) if prepare_state
  first_token, _first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, ids, 0, prompt_state)

  gen_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(gen_state, hp, clear: false) if prepare_state
  copy_prompt_state!(gen_state, prompt_state, hp, ids.size, prepare_state)

  output_ids = [first_token]
  pos = ids.size
  while output_ids.size < n_gen
    next_id, _next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, output_ids[-1], pos, gen_state)
    output_ids << next_id
    pos += 1
    break if next_id == tokenizer.eos_id
  end

  SourceReplayTemplate.new(ids.size, prompt_state, output_ids)
end

def build_prompt_cache_replay_template(weights : ML::GGUF::Qwen35Weights,
                                       tokenizer : ML::GGUF::Qwen35Tokenizer,
                                       prompt : String,
                                       n_gen : Int32,
                                       max_seq : Int32,
                                       prepare_state : Bool,
                                       resident_states : Int32) : PromptCacheReplayTemplate
  hp = weights.hparams
  ids = tokenizer.encode(prompt)
  raise "prompt encoded to zero tokens" if ids.empty?
  raise "request exceeds max_seq: prompt=#{ids.size} gen=#{n_gen} max_seq=#{max_seq}" if ids.size + n_gen >= max_seq

  root = File.tempname("qwen35-warm-prompt-cache")
  Dir.mkdir_p(root)
  store = ML::GGUF::Qwen35PromptCache::Store.new(root, resident_state_cache_entries: resident_states)

  prompt_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(prompt_state, hp) if prepare_state
  first_token, first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, ids, 0, prompt_state)

  entry = store.save(
    session_id: "warm-request-probe",
    model_id: "warm-model",
    tokenizer_id: "warm-tokenizer",
    prompt_text: "",
    token_ids: ids,
    state: prompt_state,
    next_token_id: first_token,
    next_token_logit: first_logit,
  )

  gen_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(gen_state, hp, clear: false) if prepare_state
  copy_prompt_state!(gen_state, prompt_state, hp, ids.size, prepare_state)

  output_ids = [first_token]
  pos = ids.size
  while output_ids.size < n_gen
    next_id, _next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, output_ids[-1], pos, gen_state)
    output_ids << next_id
    pos += 1
    break if next_id == tokenizer.eos_id
  end

  PromptCacheReplayTemplate.new(store, entry, ids, output_ids)
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
    0.0,
    prefill_ms,
    decode_ms,
    ids.size,
    output_ids.size,
    output_ids[0],
  )
  {timing, output_ids}
end

def run_source_replay_request(weights : ML::GGUF::Qwen35Weights,
                              replay : SourceReplayTemplate,
                              max_seq : Int32,
                              prepare_state : Bool) : {RequestTiming, Array(Int32)}
  hp = weights.hparams
  request_t0 = Time.instant

  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  state_prepare_ms = 0.0
  if prepare_state
    prepare_t0 = Time.instant
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp, clear: false)
    state_prepare_ms = (Time.instant - prepare_t0).total_milliseconds
  end

  restore_t0 = Time.instant
  copy_prompt_state!(state, replay.state, hp, replay.prompt_tokens, prepare_state)
  restore_ms = (Time.instant - restore_t0).total_milliseconds

  source = replay.output_ids
  raise "source replay has no generated ids" if source.empty?

  decode_t0 = Time.instant
  if source.size > 1
    # The final emitted token does not need a next-token prediction for this
    # request, matching qwen35_generate's source-history tail-skip contract.
    verify_ids = source[0, source.size - 1]
    target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verify_ids, replay.prompt_tokens, state)
    expected = source[0]
    source.each_with_index do |candidate, i|
      raise "source replay mismatch at token #{i}: candidate=#{candidate} expected=#{expected}" unless candidate == expected
      expected = target_nexts[i][0] if i < target_nexts.size
    end
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  total_ms = (Time.instant - request_t0).total_milliseconds
  timing = RequestTiming.new(
    total_ms,
    0.0,
    state_prepare_ms,
    restore_ms,
    0.0,
    decode_ms,
    replay.prompt_tokens,
    source.size,
    source[0],
  )
  {timing, source}
end

def run_prompt_cache_replay_request(weights : ML::GGUF::Qwen35Weights,
                                    replay : PromptCacheReplayTemplate,
                                    max_seq : Int32,
                                    prepare_state : Bool) : {RequestTiming, Array(Int32)}
  hp = weights.hparams
  request_t0 = Time.instant

  state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  state_prepare_ms = 0.0
  if prepare_state
    prepare_t0 = Time.instant
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp, clear: false)
    state_prepare_ms = (Time.instant - prepare_t0).total_milliseconds
  end

  restore_t0 = Time.instant
  restored = replay.store.restore_and_replay_suffix(replay.entry, weights, replay.prompt_tokens, reuse_state: state)
  restore_ms = (Time.instant - restore_t0).total_milliseconds
  state = restored.state

  source = replay.output_ids
  raise "prompt-cache replay has no generated ids" if source.empty?
  raise "prompt-cache replay first token mismatch" unless restored.next_token_id == source[0]

  decode_t0 = Time.instant
  if source.size > 1
    verify_ids = source[0, source.size - 1]
    target_nexts = ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, verify_ids, replay.prompt_tokens.size, state)
    expected = source[0]
    source.each_with_index do |candidate, i|
      raise "prompt-cache replay mismatch at token #{i}: candidate=#{candidate} expected=#{expected}" unless candidate == expected
      expected = target_nexts[i][0] if i < target_nexts.size
    end
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  total_ms = (Time.instant - request_t0).total_milliseconds
  timing = RequestTiming.new(
    total_ms,
    0.0,
    state_prepare_ms,
    restore_ms,
    0.0,
    decode_ms,
    replay.prompt_tokens.size,
    source.size,
    source[0],
  )
  {timing, source}
end

startup_t0 = Time.instant
g = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(g, model_path, tokenizer_bin)
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
startup_ms = (Time.instant - startup_t0).total_milliseconds

puts "qwen35_warm_request_probe"
puts "  model=#{model_path}"
mode = prompt_cache_replay ? "prompt_cache_replay" : (source_replay ? "source_replay" : "greedy")
puts "  prompt=#{prompt.inspect} gen=#{n_gen} requests=#{requests} warmups=#{warmups} max_seq=#{max_seq} prepare_state=#{prepare_state} mode=#{mode} resident_states=#{resident_states}"
puts "  startup_ms=#{startup_ms.round(1)}"

source_template = source_replay ? build_source_replay_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state) : nil
prompt_cache_template = prompt_cache_replay ? build_prompt_cache_replay_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state, resident_states) : nil

warmup_ms = 0.0
warmups.times do
  warm_t0 = Time.instant
  if replay = prompt_cache_template
    run_prompt_cache_replay_request(weights, replay, max_seq, prepare_state)
  elsif replay = source_template
    run_source_replay_request(weights, replay, max_seq, prepare_state)
  else
    run_request(weights, tokenizer, prompt, n_gen, max_seq, prepare_state)
  end
  warmup_ms += (Time.instant - warm_t0).total_milliseconds
end
puts "  explicit_warmup_ms=#{warmup_ms.round(1)}" if warmups > 0

timings = [] of RequestTiming
{% unless flag?(:cpu_only) %}
  if metal_profile
    ML::GGUF::Qwen35Metal::Profile.reset
    ML::GGUF::Qwen35Metal::Profile.enable!
  end
{% end %}

requests.times do |i|
  timing, output_ids = if replay = prompt_cache_template
                         run_prompt_cache_replay_request(weights, replay, max_seq, prepare_state)
                       elsif replay = source_template
                         run_source_replay_request(weights, replay, max_seq, prepare_state)
                       else
                         run_request(weights, tokenizer, prompt, n_gen, max_seq, prepare_state)
                       end
  timings << timing
  unless quiet
    puts "  request #{i + 1}: ids=#{output_ids.inspect}"
  end
  puts "  request #{i + 1} summary: total_ms=#{timing.total_ms.round(1)} ms_per_tok=#{timing.ms_per_output.round(2)} tokenize_ms=#{timing.tokenize_ms.round(1)} state_prepare_ms=#{timing.state_prepare_ms.round(1)} restore_ms=#{timing.restore_ms.round(1)} prefill_ms=#{timing.prefill_ms.round(1)} decode_ms=#{timing.decode_ms.round(1)} prompt_tokens=#{timing.prompt_tokens} output_tokens=#{timing.output_tokens} first_token=#{timing.first_token}"
end

{% unless flag?(:cpu_only) %}
  if metal_profile
    ML::GGUF::Qwen35Metal::Profile.disable!
    STDOUT << ML::GGUF::Qwen35Metal::Profile.report_io
  end
{% end %}

totals = timings.map(&.total_ms).sort
decode_totals = timings.map(&.decode_ms).sort
prefills = timings.map(&.prefill_ms).sort
restores = timings.map(&.restore_ms).sort
mid = totals.size // 2
avg_total = timings.sum(&.total_ms) / timings.size
avg_tok = timings.sum(&.output_tokens).to_f
avg_ms_per_tok = timings.sum(&.total_ms) / avg_tok

puts "  aggregate: avg_total_ms=#{avg_total.round(1)} p50_total_ms=#{totals[mid].round(1)} avg_ms_per_tok=#{avg_ms_per_tok.round(2)} p50_restore_ms=#{restores[mid].round(1)} p50_prefill_ms=#{prefills[mid].round(1)} p50_decode_ms=#{decode_totals[mid].round(1)}"
