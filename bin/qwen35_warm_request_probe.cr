# Resident-process request timing probe for Qwen 3.5/3.6 Metal inference.
#
# This is intentionally a benchmark probe, not a product server. It loads the
# model/tokenizer once, optionally performs explicit warmup requests, then times
# fresh request states inside the same process so one-shot CLI costs are not
# confused with steady-state request latency.

require "option_parser"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_serving_route"
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
prompt_cache_fast_forward = false
prompt_cache_direct_output = false
prompt_cache_serving_route = false
serving_route_continuation = ENV["QWEN35_SERVING_ROUTE_CONTINUATION"]? == "1"
serving_route_direct_miss = ENV["QWEN35_SERVING_ROUTE_DIRECT_MISS"]? == "1"
serving_route_active_cursor = ENV["QWEN35_SERVING_ROUTE_ACTIVE_CURSOR"]? == "1"
resident_states = (ENV["QWEN35_PROMPT_CACHE_RESIDENT_STATES"]? || "0").to_i
metal_profile = ENV["QWEN35_METAL_PROFILE"]? == "1"
artifact_codec = ENV["QWEN35_PROMPT_CACHE_ARTIFACT_CODEC"]?
artifact_codec_block = (ENV["QWEN35_PROMPT_CACHE_ARTIFACT_CODEC_BLOCK"]? || "8").to_i
artifact_live_kv = ENV["QWEN35_PROMPT_CACHE_LIVE_KV_ARTIFACTS"]? == "1"
reuse_request_state = ENV["QWEN35_WARM_PROBE_REUSE_REQUEST_STATE"]? == "1"
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
  p.on("--prompt-cache-fast-forward", "Measure trusted Store state restore + cached output emission with no verifier body") { prompt_cache_fast_forward = true }
  p.on("--prompt-cache-direct-output", "Measure resident direct output-certificate lookup + validation with no state restore") { prompt_cache_direct_output = true }
  p.on("--prompt-cache-serving-route", "Measure resident serving route: direct terminal hit, or state fast-forward when continuation state is required") { prompt_cache_serving_route = true }
  p.on("--serving-route-continuation", "With --prompt-cache-serving-route, require continuation state and bypass terminal direct-output emission") { serving_route_continuation = true }
  p.on("--serving-route-direct-miss", "With --prompt-cache-serving-route, omit the direct output certificate so terminal requests exercise exact source-span fallback") { serving_route_direct_miss = true }
  p.on("--serving-route-active-cursor", "With --prompt-cache-serving-route --serving-route-continuation, prewarm continuation state once and measure active-session cursor handoff") { serving_route_active_cursor = true }
  p.on("--resident-states N", "Resident Store state-cache entries for --prompt-cache-replay") { |v| resident_states = v.to_i }
  p.on("--artifact-codec CODEC", "Prompt-cache artifact codec for cache modes (raw, recurrent-bf16, recurrent-int8)") { |v| artifact_codec = v == "raw" ? nil : v }
  p.on("--artifact-codec-block N", "Prompt-cache recurrent-int8 artifact block size (default: 8)") { |v| artifact_codec_block = v.to_i }
  p.on("--live-kv-artifacts", "Write prompt-cache artifacts with only live KV rows") { artifact_live_kv = true }
  p.on("--reuse-request-state", "Reuse one prepared destination state across warm measured requests") { reuse_request_state = true }
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
raise "--artifact-codec-block must be positive" unless artifact_codec_block > 0
mode_count = (source_replay ? 1 : 0) + (prompt_cache_replay ? 1 : 0) + (prompt_cache_fast_forward ? 1 : 0) + (prompt_cache_direct_output ? 1 : 0) + (prompt_cache_serving_route ? 1 : 0)
raise "--source-replay, --prompt-cache-replay, --prompt-cache-fast-forward, --prompt-cache-direct-output, and --prompt-cache-serving-route are mutually exclusive" if mode_count > 1
raise "--serving-route-continuation requires --prompt-cache-serving-route" if serving_route_continuation && !prompt_cache_serving_route
raise "--serving-route-direct-miss requires --prompt-cache-serving-route" if serving_route_direct_miss && !prompt_cache_serving_route
raise "--serving-route-active-cursor requires --prompt-cache-serving-route --serving-route-continuation" if serving_route_active_cursor && !(prompt_cache_serving_route && serving_route_continuation)

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
  getter route : String

  def initialize(@total_ms, @tokenize_ms, @state_prepare_ms, @restore_ms, @prefill_ms, @decode_ms,
                 @prompt_tokens, @output_tokens, @first_token, @route = "")
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

struct PromptCacheFastForwardTemplate
  getter store : ML::GGUF::Qwen35PromptCache::Store
  getter entry : ML::GGUF::Qwen35PromptCache::Entry
  getter cached_prefix_tokens : Array(Int32)
  getter full_history_tokens : Array(Int32)
  getter output_ids : Array(Int32)

  def initialize(@store, @entry, @cached_prefix_tokens, @full_history_tokens, @output_ids)
  end
end

struct PromptCacheDirectOutputTemplate
  getter store : ML::GGUF::Qwen35PromptCache::Store
  getter prompt : String
  getter output_ids : Array(Int32)

  def initialize(@store, @prompt, @output_ids)
  end
end

struct PromptCacheServingRouteTemplate
  getter fast_forward : PromptCacheFastForwardTemplate
  getter prompt : String

  def initialize(@fast_forward, @prompt)
  end
end

struct PromptCacheActiveCursorTemplate
  getter state : ML::GGUF::Qwen35CPU::State
  getter prompt_token_count : Int32
  getter output_ids : Array(Int32)

  def initialize(@state, @prompt_token_count, @output_ids)
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
                                       resident_states : Int32,
                                       artifact_codec : String?,
                                       artifact_codec_block : Int32,
                                       artifact_live_kv : Bool) : PromptCacheReplayTemplate
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
    artifact_codec: artifact_codec,
    artifact_codec_block: artifact_codec ? artifact_codec_block : nil,
    artifact_live_kv_tokens: artifact_live_kv ? ids.size : nil,
    artifact_validation_kind: artifact_codec ? "warm-request-prefix" : nil,
    artifact_validation_steps: artifact_codec ? ids.size : nil,
    artifact_validation_hash: artifact_codec ? ML::GGUF::Qwen35PromptCache.token_hash(ids) : nil,
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

def build_prompt_cache_fast_forward_template(weights : ML::GGUF::Qwen35Weights,
                                             tokenizer : ML::GGUF::Qwen35Tokenizer,
                                             prompt : String,
                                             n_gen : Int32,
                                             max_seq : Int32,
                                             prepare_state : Bool,
                                             resident_states : Int32,
                                             artifact_codec : String?,
                                             artifact_codec_block : Int32,
                                             artifact_live_kv : Bool) : PromptCacheFastForwardTemplate
  hp = weights.hparams
  ids = tokenizer.encode(prompt)
  raise "prompt encoded to zero tokens" if ids.empty?
  raise "request exceeds max_seq: prompt=#{ids.size} gen=#{n_gen} max_seq=#{max_seq}" if ids.size + n_gen >= max_seq

  root = File.tempname("qwen35-warm-fast-forward-cache")
  Dir.mkdir_p(root)
  store = ML::GGUF::Qwen35PromptCache::Store.new(root, resident_state_cache_entries: resident_states)

  prompt_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(prompt_state, hp) if prepare_state
  first_token, _first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, ids, 0, prompt_state)

  span_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(span_state, hp, clear: false) if prepare_state
  copy_prompt_state!(span_state, prompt_state, hp, ids.size, prepare_state)

  output_ids = [first_token]
  pos = ids.size
  while output_ids.size < n_gen
    next_id, _next_logit = ML::GGUF::Qwen35CPU.forward_top1(weights, output_ids[-1], pos, span_state)
    output_ids << next_id
    pos += 1
    break if next_id == tokenizer.eos_id
  end

  # Decode state after emitting N tokens has processed the first N-1 generated
  # tokens; the final emitted token remains as `next_token_id`, matching the
  # normal greedy tail-skip contract and allowing exact continuation later.
  processed_output = output_ids.size > 1 ? output_ids[0, output_ids.size - 1] : [] of Int32
  cached_prefix_tokens = ids + processed_output
  full_history_tokens = ids + output_ids
  entry = store.save(
    session_id: "warm-request-probe",
    model_id: "warm-model",
    tokenizer_id: "warm-tokenizer",
    prompt_text: "",
    token_ids: cached_prefix_tokens,
    state: span_state,
    artifact_codec: artifact_codec,
    artifact_codec_block: artifact_codec ? artifact_codec_block : nil,
    artifact_live_kv_tokens: artifact_live_kv ? cached_prefix_tokens.size : nil,
    artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
    artifact_validation_steps: output_ids.size,
    artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history_tokens),
    next_token_id: output_ids[-1],
  )

  PromptCacheFastForwardTemplate.new(store, entry, cached_prefix_tokens, full_history_tokens, output_ids)
end

def build_prompt_cache_direct_output_template(weights : ML::GGUF::Qwen35Weights,
                                              tokenizer : ML::GGUF::Qwen35Tokenizer,
                                              prompt : String,
                                              n_gen : Int32,
                                              max_seq : Int32,
                                              prepare_state : Bool,
                                              resident_states : Int32,
                                              artifact_codec : String?,
                                              artifact_codec_block : Int32,
                                              artifact_live_kv : Bool) : PromptCacheDirectOutputTemplate
  fast_forward = build_prompt_cache_fast_forward_template(
    weights,
    tokenizer,
    prompt,
    n_gen,
    max_seq,
    prepare_state,
    resident_states,
    artifact_codec,
    artifact_codec_block,
    artifact_live_kv,
  )
  prompt_tokens_len = fast_forward.full_history_tokens.size - fast_forward.output_ids.size
  prompt_token_ids = fast_forward.full_history_tokens[0, prompt_tokens_len]
  fast_forward.store.save_output_fast_forward(
    session_id: "warm-request-probe",
    model_id: "warm-model",
    tokenizer_id: "warm-tokenizer",
    prompt_text: prompt,
    prompt_token_ids: prompt_token_ids,
    output_token_ids: fast_forward.output_ids,
    generated_text: tokenizer.decode(fast_forward.output_ids),
    exact_entry: fast_forward.entry,
  )
  PromptCacheDirectOutputTemplate.new(fast_forward.store, prompt, fast_forward.output_ids)
end

def build_prompt_cache_serving_route_template(weights : ML::GGUF::Qwen35Weights,
                                              tokenizer : ML::GGUF::Qwen35Tokenizer,
                                              prompt : String,
                                              n_gen : Int32,
                                              max_seq : Int32,
                                              prepare_state : Bool,
                                              resident_states : Int32,
                                              artifact_codec : String?,
                                              artifact_codec_block : Int32,
                                              artifact_live_kv : Bool,
                                              direct_miss : Bool = false) : PromptCacheServingRouteTemplate
  fast_forward = build_prompt_cache_fast_forward_template(
    weights,
    tokenizer,
    prompt,
    n_gen,
    max_seq,
    prepare_state,
    resident_states,
    artifact_codec,
    artifact_codec_block,
    artifact_live_kv,
  )
  unless direct_miss
    prompt_tokens_len = fast_forward.full_history_tokens.size - fast_forward.output_ids.size
    prompt_token_ids = fast_forward.full_history_tokens[0, prompt_tokens_len]
    fast_forward.store.save_output_fast_forward(
      session_id: "warm-request-probe",
      model_id: "warm-model",
      tokenizer_id: "warm-tokenizer",
      prompt_text: prompt,
      prompt_token_ids: prompt_token_ids,
      output_token_ids: fast_forward.output_ids,
      generated_text: tokenizer.decode(fast_forward.output_ids),
      exact_entry: fast_forward.entry,
    )
  end
  PromptCacheServingRouteTemplate.new(fast_forward, prompt)
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
    "greedy",
  )
  {timing, output_ids}
end

def run_prompt_cache_serving_route_request(weights : ML::GGUF::Qwen35Weights,
                                           replay : PromptCacheServingRouteTemplate,
                                           max_seq : Int32,
                                           prepare_state : Bool,
                                           continuation_required : Bool,
                                           reuse_state : ML::GGUF::Qwen35CPU::State? = nil) : {RequestTiming, Array(Int32)}
  fast_forward = replay.fast_forward
  request_t0 = Time.instant

  state_prepare_ms = 0.0
  route_reuse_state = nil.as(ML::GGUF::Qwen35CPU::State?)
  unless continuation_required
    route_reuse_state = reuse_state
  else
    state = reuse_state || ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
    raise "reused request state max_seq mismatch" if state.max_seq != max_seq
    if prepare_state && reuse_state.nil?
      prepare_t0 = Time.instant
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams, clear: false)
      state_prepare_ms = (Time.instant - prepare_t0).total_milliseconds
    end
    route_reuse_state = state
  end

  result = ML::GGUF::Qwen35ServingRoute.serve_exact_cached_span(
    fast_forward.store,
    weights,
    "warm-model",
    "warm-request-probe",
    replay.prompt,
    fast_forward.output_ids,
    fast_forward.entry,
    fast_forward.full_history_tokens,
    continuation_required: continuation_required,
    reuse_state: route_reuse_state,
  )
  total_ms = (Time.instant - request_t0).total_milliseconds
  restore_ms = result.replay ? total_ms - state_prepare_ms : 0.0
  output_ids = result.output_token_ids
  timing = RequestTiming.new(
    total_ms,
    0.0,
    state_prepare_ms,
    restore_ms,
    0.0,
    0.0,
    result.prompt_token_count,
    output_ids.size,
    output_ids[0],
    result.route,
  )
  {timing, output_ids}
end

def build_prompt_cache_active_cursor_template(weights : ML::GGUF::Qwen35Weights,
                                              replay : PromptCacheServingRouteTemplate,
                                              max_seq : Int32,
                                              prepare_state : Bool,
                                              reuse_state : ML::GGUF::Qwen35CPU::State? = nil) : PromptCacheActiveCursorTemplate
  fast_forward = replay.fast_forward
  state = reuse_state || ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  raise "active cursor state max_seq mismatch" if state.max_seq != max_seq
  if prepare_state && reuse_state.nil?
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, weights.hparams, clear: false)
  end
  result = ML::GGUF::Qwen35ServingRoute.serve_exact_cached_span(
    fast_forward.store,
    weights,
    "warm-model",
    "warm-request-probe",
    replay.prompt,
    fast_forward.output_ids,
    fast_forward.entry,
    fast_forward.full_history_tokens,
    continuation_required: true,
    reuse_state: state,
  )
  restored = result.replay.try(&.state)
  raise "active cursor prewarm did not restore continuation state" unless restored
  PromptCacheActiveCursorTemplate.new(restored, result.prompt_token_count, result.output_token_ids)
end

def run_prompt_cache_active_cursor_request(cursor : PromptCacheActiveCursorTemplate) : {RequestTiming, Array(Int32)}
  request_t0 = Time.instant
  # The active cursor owns an already-restored continuation state. This probe
  # measures the server-session handoff floor, not a reusable cache restore.
  output_ids = cursor.output_ids.dup
  total_ms = (Time.instant - request_t0).total_milliseconds
  timing = RequestTiming.new(
    total_ms,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    cursor.prompt_token_count,
    output_ids.size,
    output_ids[0],
    "state_fast_forward_active_cursor",
  )
  {timing, output_ids}
end

def run_prompt_cache_direct_output_request(replay : PromptCacheDirectOutputTemplate) : {RequestTiming, Array(Int32)}
  request_t0 = Time.instant
  hit = replay.store.lookup_output_fast_forward(
    "warm-model",
    "warm-request-probe",
    replay.prompt,
    replay.output_ids.size,
  )
  raise "direct output fast-forward miss" unless hit
  raise "direct output fast-forward token mismatch" unless hit.output_token_ids == replay.output_ids

  total_ms = (Time.instant - request_t0).total_milliseconds
  output_ids = hit.output_token_ids
  timing = RequestTiming.new(
    total_ms,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    hit.prompt_token_count,
    hit.output_token_count,
    output_ids[0],
    "direct_output",
  )
  {timing, output_ids}
end

def run_source_replay_request(weights : ML::GGUF::Qwen35Weights,
                              replay : SourceReplayTemplate,
                              max_seq : Int32,
                              prepare_state : Bool,
                              reuse_state : ML::GGUF::Qwen35CPU::State? = nil) : {RequestTiming, Array(Int32)}
  hp = weights.hparams
  request_t0 = Time.instant

  state = reuse_state || ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  raise "reused request state max_seq mismatch" if state.max_seq != max_seq
  state_prepare_ms = 0.0
  if prepare_state && reuse_state.nil?
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
    "source_replay",
  )
  {timing, source}
end

def run_prompt_cache_replay_request(weights : ML::GGUF::Qwen35Weights,
                                    replay : PromptCacheReplayTemplate,
                                    max_seq : Int32,
                                    prepare_state : Bool,
                                    reuse_state : ML::GGUF::Qwen35CPU::State? = nil) : {RequestTiming, Array(Int32)}
  hp = weights.hparams
  request_t0 = Time.instant

  state = reuse_state || ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  raise "reused request state max_seq mismatch" if state.max_seq != max_seq
  state_prepare_ms = 0.0
  if prepare_state && reuse_state.nil?
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
    "prompt_cache_replay",
  )
  {timing, source}
end

def run_prompt_cache_fast_forward_request(weights : ML::GGUF::Qwen35Weights,
                                          replay : PromptCacheFastForwardTemplate,
                                          max_seq : Int32,
                                          prepare_state : Bool,
                                          reuse_state : ML::GGUF::Qwen35CPU::State? = nil) : {RequestTiming, Array(Int32)}
  hp = weights.hparams
  request_t0 = Time.instant

  state = reuse_state || ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  raise "reused request state max_seq mismatch" if state.max_seq != max_seq
  state_prepare_ms = 0.0
  if prepare_state && reuse_state.nil?
    prepare_t0 = Time.instant
    ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp, clear: false)
    state_prepare_ms = (Time.instant - prepare_t0).total_milliseconds
  end

  unless ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(replay.entry, replay.full_history_tokens, replay.output_ids.size)
    raise "fast-forward artifact validation mismatch"
  end

  restore_t0 = Time.instant
  restored = replay.store.restore_and_replay_suffix(replay.entry, weights, replay.cached_prefix_tokens, reuse_state: state)
  restore_ms = (Time.instant - restore_t0).total_milliseconds
  raise "fast-forward restored prefix mismatch" unless restored.reused_prefix_len == replay.cached_prefix_tokens.size
  raise "fast-forward restored replayed unexpected suffix" unless restored.replayed_tokens == 0
  raise "fast-forward restored next token mismatch" unless restored.next_token_id == replay.output_ids[-1]

  output_ids = replay.output_ids
  total_ms = (Time.instant - request_t0).total_milliseconds
  timing = RequestTiming.new(
    total_ms,
    0.0,
    state_prepare_ms,
    restore_ms,
    0.0,
    0.0,
    replay.full_history_tokens.size - output_ids.size,
    output_ids.size,
    output_ids[0],
    "state_fast_forward",
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
mode = if prompt_cache_serving_route
         "prompt_cache_serving_route"
       elsif prompt_cache_direct_output
         "prompt_cache_direct_output"
       elsif prompt_cache_fast_forward
         "prompt_cache_fast_forward"
       elsif prompt_cache_replay
         "prompt_cache_replay"
       elsif source_replay
         "source_replay"
       else
         "greedy"
       end
puts "  prompt=#{prompt.inspect} gen=#{n_gen} requests=#{requests} warmups=#{warmups} max_seq=#{max_seq} prepare_state=#{prepare_state} mode=#{mode} serving_route_continuation=#{serving_route_continuation} serving_route_direct_miss=#{serving_route_direct_miss} serving_route_active_cursor=#{serving_route_active_cursor} resident_states=#{resident_states} artifact_codec=#{artifact_codec || "raw"} artifact_codec_block=#{artifact_codec_block} artifact_live_kv=#{artifact_live_kv} reuse_request_state=#{reuse_request_state}"
puts "  startup_ms=#{startup_ms.round(1)}"

source_template = source_replay ? build_source_replay_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state) : nil
prompt_cache_template = prompt_cache_replay ? build_prompt_cache_replay_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state, resident_states, artifact_codec, artifact_codec_block, artifact_live_kv) : nil
prompt_cache_fast_forward_template = prompt_cache_fast_forward ? build_prompt_cache_fast_forward_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state, resident_states, artifact_codec, artifact_codec_block, artifact_live_kv) : nil
prompt_cache_direct_output_template = prompt_cache_direct_output ? build_prompt_cache_direct_output_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state, resident_states, artifact_codec, artifact_codec_block, artifact_live_kv) : nil
prompt_cache_serving_route_template = prompt_cache_serving_route ? build_prompt_cache_serving_route_template(weights, tokenizer, prompt, n_gen, max_seq, prepare_state, resident_states, artifact_codec, artifact_codec_block, artifact_live_kv, serving_route_direct_miss) : nil
reusable_request_state = nil.as(ML::GGUF::Qwen35CPU::State?)
if reuse_request_state
  reusable_request_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(reusable_request_state.not_nil!, weights.hparams, clear: false) if prepare_state
end
prompt_cache_active_cursor_template = if serving_route_active_cursor && (replay = prompt_cache_serving_route_template)
                                        build_prompt_cache_active_cursor_template(weights, replay, max_seq, prepare_state, reusable_request_state)
                                      else
                                        nil
                                      end

warmup_ms = 0.0
warmups.times do
  warm_t0 = Time.instant
  if cursor = prompt_cache_active_cursor_template
    run_prompt_cache_active_cursor_request(cursor)
  elsif replay = prompt_cache_serving_route_template
    run_prompt_cache_serving_route_request(weights, replay, max_seq, prepare_state, serving_route_continuation, reusable_request_state)
  elsif replay = prompt_cache_direct_output_template
    run_prompt_cache_direct_output_request(replay)
  elsif replay = prompt_cache_fast_forward_template
    run_prompt_cache_fast_forward_request(weights, replay, max_seq, prepare_state, reusable_request_state)
  elsif replay = prompt_cache_template
    run_prompt_cache_replay_request(weights, replay, max_seq, prepare_state, reusable_request_state)
  elsif replay = source_template
    run_source_replay_request(weights, replay, max_seq, prepare_state, reusable_request_state)
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
  timing, output_ids = if cursor = prompt_cache_active_cursor_template
                         run_prompt_cache_active_cursor_request(cursor)
                       elsif replay = prompt_cache_serving_route_template
                         run_prompt_cache_serving_route_request(weights, replay, max_seq, prepare_state, serving_route_continuation, reusable_request_state)
                       elsif replay = prompt_cache_direct_output_template
                         run_prompt_cache_direct_output_request(replay)
                       elsif replay = prompt_cache_fast_forward_template
                         run_prompt_cache_fast_forward_request(weights, replay, max_seq, prepare_state, reusable_request_state)
                       elsif replay = prompt_cache_template
                         run_prompt_cache_replay_request(weights, replay, max_seq, prepare_state, reusable_request_state)
                       elsif replay = source_template
                         run_source_replay_request(weights, replay, max_seq, prepare_state, reusable_request_state)
                       else
                         run_request(weights, tokenizer, prompt, n_gen, max_seq, prepare_state)
                       end
  timings << timing
  unless quiet
    puts "  request #{i + 1}: ids=#{output_ids.inspect}"
  end
  puts "  request #{i + 1} summary: total_ms=#{timing.total_ms.round(3)} ms_per_tok=#{timing.ms_per_output.round(4)} tokenize_ms=#{timing.tokenize_ms.round(3)} state_prepare_ms=#{timing.state_prepare_ms.round(3)} restore_ms=#{timing.restore_ms.round(3)} prefill_ms=#{timing.prefill_ms.round(3)} decode_ms=#{timing.decode_ms.round(3)} prompt_tokens=#{timing.prompt_tokens} output_tokens=#{timing.output_tokens} first_token=#{timing.first_token} route=#{timing.route}"
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
route_counts = Hash(String, Int32).new(0)
timings.each { |timing| route_counts[timing.route] += 1 }
route_summary = route_counts.keys.sort.map { |route| "#{route}=#{route_counts[route]}" }.join(",")

puts "  aggregate: avg_total_ms=#{avg_total.round(3)} p50_total_ms=#{totals[mid].round(3)} avg_ms_per_tok=#{avg_ms_per_tok.round(4)} p50_restore_ms=#{restores[mid].round(3)} p50_prefill_ms=#{prefills[mid].round(3)} p50_decode_ms=#{decode_totals[mid].round(3)} routes=#{route_summary}"
