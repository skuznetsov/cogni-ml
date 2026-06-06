require "option_parser"
require "json"
require "../src/ml/gguf/gemma4_metal"
require "../src/ml/gguf/gemma4_prompt_cache"
require "../src/ml/gguf/gemma4_tokenizer"
require "../src/ml/gguf/qwen35_constraints"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/gemma-4-12B-it-GGUF/gemma-4-12B-it-Q4_K_M.gguf"

model = ENV["GEMMA4_MODEL"]? || DEFAULT_MODEL
prompt = [42, 43, 44, 45, 46, 47, 48, 49]
prompt_text = nil.as(String?)
prompt_file = nil.as(String?)
chat_user = nil.as(String?)
generate = 8
max_seq = 1024
warmups = 1
runs = 3
profile = false
profile_decode_only = false
print_generated_ids = false
print_generated_text = false
llama_tokenize_bin = ENV["LLAMA_TOKENIZE_BIN"]? || "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"
decode_mode = "top1"
decode_wave = ENV["GEMMA4_DECODE_WAVE_OFF"]? != "1"
top1_wave_resident = ENV["GEMMA4_TOP1_WAVE_RESIDENT_OFF"]? != "1"
top1_chain_explicit = !ENV["GEMMA4_TOP1_CHAIN"]?.nil?
body_chain_explicit = !ENV["GEMMA4_BODY_CHAIN"]?.nil?
top1_chain = (ENV["GEMMA4_TOP1_CHAIN"]? || "8").to_i
body_chain = (ENV["GEMMA4_BODY_CHAIN"]? || "8").to_i
prefill_mode = "serial"
prefill_chunk = 8
prefill_head = true
stop_layer = nil.as(Int32?)
decode_stop_layer_after_step = nil.as(Int32?)
decode_stop_layer_after_layer = nil.as(Int32?)
prompt_cache_root = nil.as(String?)
prompt_cache_session = "profile"
prompt_cache_snapshot_mib = 0
prompt_cache_snapshot_min_free_mib = (ENV["GEMMA4_PROMPT_CACHE_SNAPSHOT_MIN_FREE_MIB"]? || "4096").to_i
prompt_cache_snapshot_entries = 1
decode_only_seed = nil.as(Int32?)
allowed_ids = [] of Int32
constrained_literal_prefix = nil.as(String?)
constrained_tool_call_prefix = false
constrained_tool_required_param_prefix = false
tools_json = ENV["GEMMA4_TOOLS_JSON"]?
literal_force_single = true
literal_force_span = true

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_metal_decode_profile [--tokens 42,43] [--generate 8] [--max-seq 1024] [--runs 3]"
  p.on("--model PATH", "Gemma4 GGUF path") { |v| model = v }
  p.on("--tokens IDS", "Comma-separated prompt token ids") { |v| prompt = v.split(',').reject(&.empty?).map(&.to_i) }
  p.on("--prompt TEXT", "Prompt text; tokenized through llama-tokenize oracle") { |v| prompt_text = v }
  p.on("--prompt-file PATH", "Read prompt text from a file and tokenize through llama-tokenize oracle") { |v| prompt_file = v }
  p.on("--chat-user TEXT", "Format a single Gemma4 user turn and tokenize through llama-tokenize oracle") { |v| chat_user = v }
  p.on("--llama-tokenize-bin PATH", "llama.cpp llama-tokenize binary for Gemma4 text prompts") { |v| llama_tokenize_bin = v }
  p.on("--generate N", "Measured generated tokens per run") { |v| generate = v.to_i }
  p.on("--max-seq N", "KV cache sequence capacity") { |v| max_seq = v.to_i }
  p.on("--warmups N", "Warmup runs") { |v| warmups = v.to_i }
  p.on("--runs N", "Measured runs") { |v| runs = v.to_i }
  p.on("--profile", "Print shared Metal matmul/profile attribution for the final measured run") { profile = true }
  p.on("--profile-decode-only", "Reset profile counters after prefill and report only generated-token work") { profile_decode_only = true }
  p.on("--print-generated-ids", "Print seed + generated token ids for parity diagnostics") { print_generated_ids = true }
  p.on("--print-generated-text", "Print conservative detokenized generated text; requires text tokenizer metadata") { print_generated_text = true }
  p.on("--decode-wave", "Use one command buffer per decode token instead of one wait per layer (default)") { decode_wave = true }
  p.on("--decode-layerwise", "Use legacy one-wait-per-layer decode path") { decode_wave = false }
  p.on("--top1-wave-resident", "Fuse decode wave, output RMSNorm, and top1 head into one resident command buffer (default)") { top1_wave_resident = true }
  p.on("--no-top1-wave-resident", "Use legacy hidden-readback + separate top1 head path") { top1_wave_resident = false }
  p.on("--top1-chain N", "Generate exact greedy top1 in GPU-resident chunks of N; requires --decode-wave and --top1-wave-resident") { |v| top1_chain = v.to_i; top1_chain_explicit = true }
  p.on("--body-chain N", "Run body-only decode in GPU-resident chunks of N known synthetic tokens; use 1 for llama-bench tg parity") { |v| body_chain = v.to_i; body_chain_explicit = true }
  p.on("--prefill-mode MODE", "Prompt prefill mode: serial or rows (default: serial)") { |v| prefill_mode = v }
  p.on("--prefill-chunk N", "Row prefill chunk size; exact path clamps above 8; GEMMA4_ROW_PREFILL_ALLOW_GEMM=1 defaults cap to 512") { |v| prefill_chunk = v.to_i }
  p.on("--prefill-no-head", "Measure pure prompt body only; requires --body-only because no next-token seed is computed") { prefill_head = false }
  p.on("--stop-layer N", "Run only the first N layers for attribution (default: all)") { |v| stop_layer = v.to_i }
  p.on("--decode-stop-layer-after START:LAYER", "Decode full-depth until loop step START, then use first LAYER layers") do |v|
    parts = v.split(':', limit: 2)
    raise "--decode-stop-layer-after expects START:LAYER" unless parts.size == 2
    decode_stop_layer_after_step = parts[0].to_i
    decode_stop_layer_after_layer = parts[1].to_i
  end
  p.on("--prompt-cache-root PATH", "Enable exact Gemma prompt-cache save/restore at this root") { |v| prompt_cache_root = v }
  p.on("--prompt-cache-session ID", "Prompt-cache session id") { |v| prompt_cache_session = v }
  p.on("--prompt-cache-snapshot-mib N", "Enable Store resident snapshot cache with this MiB budget") { |v| prompt_cache_snapshot_mib = v.to_i }
  p.on("--prompt-cache-snapshot-min-free-mib N", "Clamp snapshot cache to leave at least this much available memory; default env GEMMA4_PROMPT_CACHE_SNAPSHOT_MIN_FREE_MIB or 4096") { |v| prompt_cache_snapshot_min_free_mib = v.to_i }
  p.on("--prompt-cache-snapshot-entries N", "Store resident snapshot cache entry limit") { |v| prompt_cache_snapshot_entries = v.to_i }
  p.on("--decode-only-seed N", "Skip prompt prefill and start decode from this token id with empty KV state") { |v| decode_only_seed = v.to_i }
  p.on("--body-only", "During measured generation, update resident state from fixed synthetic tokens without final logits/top1") { decode_mode = "body" }
  p.on("--top1", "During measured generation, run exact greedy top1 chain (default)") { decode_mode = "top1" }
  p.on("--top2", "Diagnostic: compute resident top2 each token and continue with top1") { decode_mode = "top2" }
  p.on("--allowed-ids IDS", "Diagnostic: constrain measured decode head to these comma-separated token ids") do |v|
    allowed_ids = v.split(',').reject(&.empty?).map(&.to_i)
    decode_mode = "allowed"
  end
  p.on("--constrained-literal-prefix TEXT", "Diagnostic: tokenizer-derived literal frontier after the initial seed token") do |v|
    constrained_literal_prefix = v
    decode_mode = "literal"
  end
  p.on("--tools-json JSON", "OpenAI/Qwen-style tool schema JSON for constrained tool-prefix diagnostics") { |v| tools_json = v }
  p.on("--constrained-tool-call-prefix", "Diagnostic: constrain XML tool-call prefix over function names from --tools-json/GEMMA4_TOOLS_JSON") do
    constrained_tool_call_prefix = true
    decode_mode = "literal"
  end
  p.on("--constrained-tool-required-param-prefix", "Diagnostic: constrain tool/function plus required parameter-open prefixes from --tools-json/GEMMA4_TOOLS_JSON") do
    constrained_tool_call_prefix = true
    constrained_tool_required_param_prefix = true
    decode_mode = "literal"
  end
  p.on("--literal-force-single-off", "Disable forced single-token literal frontier elimination") { literal_force_single = false }
  p.on("--literal-force-span-off", "Disable batched forced literal spans") { literal_force_span = false }
  p.on("-h", "--help", "Show help") { puts p; exit }
end

raise "model not found: #{model}" unless File.exists?(model)
text_modes = [prompt_text, prompt_file, chat_user].count { |v| !v.nil? }
raise "--prompt, --prompt-file, and --chat-user are mutually exclusive" if text_modes > 1
if constrained_tool_call_prefix && constrained_literal_prefix
  raise "--constrained-tool-call-prefix and --constrained-literal-prefix are mutually exclusive"
end
tools = [] of JSON::Any
tool_names = [] of String
tool_required_params = {} of String => Array(String)
if constrained_tool_call_prefix
  raw_tools = tools_json
  raise "--constrained-tool-call-prefix requires --tools-json or GEMMA4_TOOLS_JSON" if raw_tools.nil? || raw_tools.empty?
  parsed_tools = JSON.parse(raw_tools)
  tools = parsed_tools.as_a? || raise "--tools-json must be a JSON array"
  tool_names = ML::GGUF::Qwen35Constraints.tool_function_names(tools)
  tool_required_params = ML::GGUF::Qwen35Constraints.tool_required_parameters(tools)
  raise "--constrained-tool-call-prefix found no function names" if tool_names.empty?
end
if file = prompt_file
  prompt_text = File.read(file)
end
if user = chat_user
  prompt_text = "<|turn>user\n#{user}<turn|>\n<|turn>model\n"
end
raise "generate must be positive" unless generate > 0
raise "runs must be positive" unless runs > 0
raise "prompt-cache snapshot MiB must be non-negative" unless prompt_cache_snapshot_mib >= 0
raise "prompt-cache snapshot min-free MiB must be non-negative" unless prompt_cache_snapshot_min_free_mib >= 0
raise "prompt-cache snapshot entries must be non-negative" unless prompt_cache_snapshot_entries >= 0

def percentile(sorted : Array(Float64), p : Float64) : Float64
  return 0.0 if sorted.empty?
  idx = ((sorted.size - 1).to_f64 * p).round.to_i
  sorted[idx]
end

def summarize(label : String, samples : Array(Float64), tokens : Int32) : Nil
  sorted = samples.sort
  mean = samples.sum / samples.size
  p50 = percentile(sorted, 0.50)
  p90 = percentile(sorted, 0.90)
  tok_s = p50 > 0.0 ? tokens.to_f64 / (p50 / 1000.0) : 0.0
  puts "#{label}_runs=#{samples.map { |v| v.round(3) }.join(',')}"
  puts "#{label}_mean_ms=#{mean.round(3)} #{label}_p50_ms=#{p50.round(3)} #{label}_p90_ms=#{p90.round(3)} #{label}_p50_tok_s=#{tok_s.round(3)}"
end

class Gemma4LiteralTokenIndex
  @texts : Array(String)
  @by_first : Hash(Char, Array({Int32, String}))

  def initialize(tokenizer : ML::GGUF::Gemma4Tokenizer)
    @texts = Array(String).new(tokenizer.vocab.size, "")
    @by_first = Hash(Char, Array({Int32, String})).new { |h, k| h[k] = [] of {Int32, String} }
    tokenizer.vocab.each_index do |id|
      text = begin
        tokenizer.decode_single(id.to_i32)
      rescue
        ""
      end
      @texts[id] = text
      next if text.empty?

      @by_first[text[0]] << {id.to_i32, text}
    end
  end

  def text_for_id(id : Int32) : String
    return "" if id < 0 || id >= @texts.size

    @texts[id]
  end

  def literal_frontier_ids(remaining_literals : Array(String)) : Array(Int32)
    return [] of Int32 if remaining_literals.empty?

    allowed = [] of Int32
    seen = Set(Int32).new
    remaining_literals.each do |literal|
      next if literal.empty?

      bucket = @by_first[literal[0]]?
      next unless bucket

      bucket.each do |id, decoded|
        next if seen.includes?(id)
        next unless literal.starts_with?(decoded)

        allowed << id
        seen << id
      end
    end
    allowed
  end
end

def gemma4_advance_literal_options(remaining_literals : Array(String), emitted : String) : Array(String)
  return remaining_literals if emitted.empty?

  next_literals = [] of String
  remaining_literals.each do |literal|
    next unless literal.starts_with?(emitted)

    suffix = literal[emitted.size..]
    next_literals << suffix if suffix && !suffix.empty?
  end
  next_literals
end

def top1_id(logits : Array(Float32)) : Int32
  best_id = 0
  best = logits[0]
  logits.each_with_index do |v, i|
    if v > best
      best = v
      best_id = i
    end
  end
  best_id.to_i32
end

def forward_top1(weights : ML::GGUF::Gemma4Weights, token_id : Int32, pos : Int32,
                 state : ML::GGUF::Gemma4Metal::ResidentState,
                 decode_wave : Bool = false,
                 top1_wave_resident : Bool = false,
                 stop_layer : Int32? = nil) : Int32
  layer_count = stop_layer || weights.hparams.n_layer
  if decode_wave && top1_wave_resident
    if id = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_wave(weights, token_id, pos, state, layer_count)
      return id
    end
  end

  hidden = if decode_wave
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(weights, token_id, pos, state, layer_count).not_nil!
           else
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos, state, layer_count).not_nil!
           end
  logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
  top1_id(logits)
end

def forward_top2_top1(weights : ML::GGUF::Gemma4Weights, token_id : Int32, pos : Int32,
                      state : ML::GGUF::Gemma4Metal::ResidentState,
                      decode_wave : Bool = false,
                      top1_wave_resident : Bool = false,
                      stop_layer : Int32? = nil) : Int32
  layer_count = stop_layer || weights.hparams.n_layer
  if decode_wave && top1_wave_resident
    if top2 = ML::GGUF::Gemma4Metal.forward_top2_resident_cache_wave(weights, token_id, pos, state, layer_count)
      return top2[0].to_i32
    end
  end

  hidden = if decode_wave
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(weights, token_id, pos, state, layer_count).not_nil!
           else
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos, state, layer_count).not_nil!
           end
  normed = ML::GGUF::Gemma4CPU.rms_norm(hidden, weights.output_norm, weights.hparams.rms_eps)
  top2 = ML::GGUF::Qwen35Metal.project_top2_no_norm(weights.token_embd, normed).not_nil!
  top2[0].to_i32
end

def forward_allowed_top1(weights : ML::GGUF::Gemma4Weights, token_id : Int32, pos : Int32,
                         state : ML::GGUF::Gemma4Metal::ResidentState,
                         allowed_ids : Array(Int32),
                         decode_wave : Bool = false,
                         top1_wave_resident : Bool = false,
                         stop_layer : Int32? = nil) : Int32
  layer_count = stop_layer || weights.hparams.n_layer
  if decode_wave && top1_wave_resident
    if top1 = ML::GGUF::Gemma4Metal.forward_top1_allowed_resident_cache_wave(weights, token_id, pos, allowed_ids, state, layer_count)
      return top1[0].to_i32
    end
  end

  hidden = if decode_wave
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache_wave(weights, token_id, pos, state, layer_count).not_nil!
           else
             ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos, state, layer_count).not_nil!
           end
  normed = ML::GGUF::Gemma4CPU.rms_norm(hidden, weights.output_norm, weights.hparams.rms_eps)
  logits = ML::GGUF::Qwen35Metal.matmul(weights.token_embd, normed, 1).not_nil!
  best_id = allowed_ids[0]
  best = logits[best_id]
  allowed_ids.each do |id|
    value = logits[id]
    if value > best
      best = value
      best_id = id
    end
  end
  best_id.to_i32
end

def top1_allowed_from_hidden(weights : ML::GGUF::Gemma4Weights,
                             hidden : Array(Float32),
                             allowed_ids : Array(Int32),
                             resident_allowed_head : Bool = false) : Int32
  raise "top1_allowed_from_hidden requires at least one allowed id" if allowed_ids.empty?
  if resident_allowed_head
    if top1 = ML::GGUF::Qwen35Metal.rmsnorm_project_top1_allowed_ids(
         hidden, weights.output_norm, weights.token_embd, weights.hparams.rms_eps, allowed_ids)
      return top1[0].to_i32
    end
  end

  logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
  best_id = allowed_ids[0]
  best = logits[best_id]
  allowed_ids.each do |id|
    value = logits[id]
    if value > best
      best = value
      best_id = id
    end
  end
  best_id.to_i32
end

def synthetic_decode_token(i : Int32) : Int32
  ((i * 13 + 11751) % 32000).to_i32
end

def prefill_prompt_hidden(weights : ML::GGUF::Gemma4Weights,
                          prompt : Array(Int32),
                          state : ML::GGUF::Gemma4Metal::ResidentState,
                          prefill_mode : String,
                          prefill_chunk : Int32,
                          stop_layer : Int32?,
                          read_last_hidden : Bool = true) : Array(Float32)
  case prefill_mode
  when "serial"
    last = [] of Float32
    prompt.each_with_index do |token_id, pos|
      last = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, token_id, pos.to_i32, state, stop_layer || weights.hparams.n_layer).not_nil!
    end
    last
  when "rows"
    ML::GGUF::Gemma4Metal.prefill_tokens_last_hidden_resident_rows(weights, prompt, 0, state, chunk_size: prefill_chunk, stop_layer: stop_layer || weights.hparams.n_layer, read_last_hidden: read_last_hidden).not_nil!
  else
    raise "prefill mode must be serial or rows"
  end
end

def run_once(weights : ML::GGUF::Gemma4Weights, prompt : Array(Int32), generate : Int32,
             max_seq : Int32, decode_mode : String, prefill_mode : String, prefill_chunk : Int32,
             prefill_head : Bool = true,
             stop_layer : Int32? = nil,
             profile : Bool = false,
             profile_decode_only : Bool = false,
             decode_wave : Bool = false,
             top1_wave_resident : Bool = false,
             top1_chain : Int32 = 1,
             body_chain : Int32 = 1,
             decode_stop_layer_after_step : Int32? = nil,
             decode_stop_layer_after_layer : Int32? = nil,
             prompt_cache_store : ML::GGUF::Gemma4PromptCache::Store? = nil,
             prompt_cache_model_id : String = "",
             prompt_cache_tokenizer_id : String = "synthetic-token-ids",
             prompt_cache_session_id : String = "profile",
             decode_only_seed : Int32? = nil,
             allowed_ids : Array(Int32) = [] of Int32,
             literal_index : Gemma4LiteralTokenIndex? = nil,
             literal_remaining_start : Array(String) = [] of String,
             literal_force_single : Bool = true,
             literal_force_span : Bool = true) : NamedTuple(prefill_ms: Float64, decode_ms: Float64, first_id: Int32, last_id: Int32, token_trace: Array(Int32), cache_route: String, cache_restore_ms: Float64, literal_forced_single: Int32, literal_allowed_head: Int32, literal_forced_span_batches: Int32, literal_forced_span_tokens: Int32)
  state = ML::GGUF::Gemma4Metal::ResidentState.new(weights.hparams, max_seq)

  ML::GGUF::Qwen35Metal::Profile.reset if profile
  ML::GGUF::Qwen35Metal::Profile.enable! if profile

  prefill_t0 = Time.instant
  hidden = [] of Float32
  next_id = -1_i32
  cache_route = prompt_cache_store ? "miss" : "none"
  cache_restore_ms = 0.0
  save_cache_after_head = false
  literal_remaining = literal_remaining_start.dup
  literal_forced_single = 0
  literal_allowed_head = 0
  literal_forced_span_batches = 0
  literal_forced_span_tokens = 0
  if seed = decode_only_seed
    next_id = seed.to_i32
    cache_route = "decode_only"
  elsif store = prompt_cache_store
    if hit = store.lookup_prompt(prompt_cache_model_id, prompt_cache_tokenizer_id, "", prompt)
      restore_t0 = Time.instant
      store.restore(hit, reuse_state: state)
      cache_restore_ms = (Time.instant - restore_t0).total_milliseconds
      if prefill_head && (cached_next = hit.next_token_id)
        next_id = cached_next
        cache_route = "hit_next_id"
      elsif prefill_head
        # Older cache entries store authoritative K/V rows, not final prompt
        # logits. Replaying the last prompt token recovers exact logits while
        # rewriting the same K/V row.
        hidden = ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(
          weights,
          prompt.last,
          (prompt.size - 1).to_i32,
          state,
          stop_layer || weights.hparams.n_layer
        ).not_nil!
        cache_route = "hit_replay_last"
      else
        cache_route = "hit_restore_only"
      end
    end
  end

  if decode_only_seed
    # Decode-only mode mirrors llama-bench tg: empty KV state, no prompt ingest.
  elsif cache_route == "miss"
    hidden = prefill_prompt_hidden(weights, prompt, state, prefill_mode, prefill_chunk, stop_layer, prefill_head)
    save_cache_after_head = true
    cache_route = "miss_save"
  elsif cache_route == "none"
    hidden = prefill_prompt_hidden(weights, prompt, state, prefill_mode, prefill_chunk, stop_layer, prefill_head)
  end
  if prefill_head && next_id < 0
    if decode_mode == "literal" && !literal_remaining.empty?
      index = literal_index.not_nil!
      dynamic_allowed = index.literal_frontier_ids(literal_remaining)
      raise "literal first-token frontier is empty for #{literal_remaining.inspect}" if dynamic_allowed.empty?
      next_id = if literal_force_single && dynamic_allowed.size == 1
                  literal_forced_single += 1
                  dynamic_allowed[0]
                else
                  literal_allowed_head += 1
                  top1_allowed_from_hidden(weights, hidden, dynamic_allowed, top1_wave_resident)
                end
      emitted = index.text_for_id(next_id)
      literal_remaining = gemma4_advance_literal_options(literal_remaining, emitted)
    else
      logits = ML::GGUF::Gemma4Metal.forward_logits_from_hidden(weights, hidden).not_nil!
      next_id = top1_id(logits)
    end
  end
  if save_cache_after_head
    prompt_cache_store.not_nil!.save_resident_state(
      state,
      prompt,
      model_id: prompt_cache_model_id,
      tokenizer_id: prompt_cache_tokenizer_id,
      prompt_text: "",
      session_id: prompt_cache_session_id,
      next_token_id: prefill_head ? next_id : nil,
    )
  end
  prefill_ms = (Time.instant - prefill_t0).total_milliseconds

  if profile && profile_decode_only
    ML::GGUF::Qwen35Metal::Profile.reset
    ML::GGUF::Qwen35Metal::Profile.enable!
  end

  decode_t0 = Time.instant
  cur = prefill_head ? next_id : synthetic_decode_token(0)
  token_trace = [cur]
  i = 0
  while i < generate
    pos = decode_only_seed ? i.to_i32 : (prompt.size + i).to_i32
    decode_layer_count = stop_layer || weights.hparams.n_layer
    if start_step = decode_stop_layer_after_step
      if scheduled_layer = decode_stop_layer_after_layer
        decode_layer_count = scheduled_layer if i >= start_step
      end
    end
    if decode_mode == "body"
      if body_chain > 1 && decode_wave
        chunk = Math.min(body_chain, generate - i)
        chunk_tokens = Array(Int32).new(chunk) { |j| synthetic_decode_token(i + j) }
        unless ML::GGUF::Gemma4Metal.forward_resident_cache_body_chain(weights, chunk_tokens, pos, state, decode_layer_count)
          raise "Gemma4 body decode chain failed"
        end
        token_trace.concat(chunk_tokens)
        cur = chunk_tokens.last
        i += chunk
        next
      else
        cur = synthetic_decode_token(i)
        if decode_wave
          unless ML::GGUF::Gemma4Metal.forward_resident_cache_wave_no_read(weights, cur, pos, state, decode_layer_count)
            raise "Gemma4 body decode wave failed"
          end
        else
          ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, cur, pos, state, decode_layer_count).not_nil!
        end
        i += 1
      end
    else
      if decode_mode == "literal" && literal_force_span && decode_wave && !literal_remaining.empty?
        index = literal_index.not_nil!
        probe_remaining = literal_remaining
        emitted_ids = [] of Int32
        max_span = generate - i
        while emitted_ids.size < max_span && !probe_remaining.empty?
          dynamic_allowed = index.literal_frontier_ids(probe_remaining)
          break unless dynamic_allowed.size == 1

          id = dynamic_allowed[0]
          emitted_ids << id
          emitted = index.text_for_id(id)
          probe_remaining = gemma4_advance_literal_options(probe_remaining, emitted)
        end

        if emitted_ids.size >= 2
          consumed_tokens = [cur]
          consumed_tokens.concat(emitted_ids[0, emitted_ids.size - 1])
          unless ML::GGUF::Gemma4Metal.forward_resident_cache_body_chain(weights, consumed_tokens, pos, state, decode_layer_count)
            raise "Gemma4 literal forced span body chain failed"
          end
          token_trace.concat(emitted_ids)
          cur = emitted_ids.last
          literal_remaining = probe_remaining
          literal_forced_span_batches += 1
          literal_forced_span_tokens += emitted_ids.size
          i += emitted_ids.size
          next
        end
      end

      if top1_chain > 1 && decode_wave && top1_wave_resident
        chunk = Math.min(top1_chain, generate - i)
        ids = ML::GGUF::Gemma4Metal.forward_top1_resident_cache_chain(weights, cur, pos, state, chunk, decode_layer_count)
        raise "Gemma4 top1 resident chain failed" unless ids
        token_trace.concat(ids.not_nil!)
        cur = ids.not_nil!.last
        i += ids.not_nil!.size
        next
      else
        cur = if decode_mode == "top2"
                forward_top2_top1(weights, cur, pos, state, decode_wave, top1_wave_resident, decode_layer_count)
              elsif decode_mode == "allowed"
                forward_allowed_top1(weights, cur, pos, state, allowed_ids, decode_wave, top1_wave_resident, decode_layer_count)
              elsif decode_mode == "literal" && !literal_remaining.empty?
                index = literal_index.not_nil!
                dynamic_allowed = index.literal_frontier_ids(literal_remaining)
                raise "literal frontier is empty for #{literal_remaining.inspect}" if dynamic_allowed.empty?
                id = if literal_force_single && dynamic_allowed.size == 1
                       if decode_wave
                         unless ML::GGUF::Gemma4Metal.forward_resident_cache_wave_no_read(weights, cur, pos, state, decode_layer_count)
                           raise "Gemma4 literal forced body wave failed"
                         end
                       else
                         ML::GGUF::Gemma4Metal.forward_hidden_resident_cache(weights, cur, pos, state, decode_layer_count).not_nil!
                       end
                       literal_forced_single += 1
                       dynamic_allowed[0]
                     else
                       literal_allowed_head += 1
                       forward_allowed_top1(weights, cur, pos, state, dynamic_allowed, decode_wave, top1_wave_resident, decode_layer_count)
                     end
                emitted = index.text_for_id(id)
                literal_remaining = gemma4_advance_literal_options(literal_remaining, emitted)
                id
              else
                forward_top1(weights, cur, pos, state, decode_wave, top1_wave_resident, decode_layer_count)
              end
        i += 1
      end
    end
    token_trace << cur
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds

  if profile
    ML::GGUF::Qwen35Metal::Profile.disable!
    puts ML::GGUF::Qwen35Metal::Profile.report_io
  end

  {prefill_ms: prefill_ms, decode_ms: decode_ms, first_id: next_id, last_id: cur, token_trace: token_trace, cache_route: cache_route, cache_restore_ms: cache_restore_ms, literal_forced_single: literal_forced_single, literal_allowed_head: literal_allowed_head, literal_forced_span_batches: literal_forced_span_batches, literal_forced_span_tokens: literal_forced_span_tokens}
end

started = Time.instant
weights = ML::GGUF::Gemma4Weights.from_gguf(model)
load_ms = (Time.instant - started).total_milliseconds
raise "Metal not available" unless ML::GGUF::Gemma4Metal.available?

tokenizer = nil.as(ML::GGUF::Gemma4Tokenizer?)
structured_literal_enabled = !constrained_literal_prefix.nil? || constrained_tool_call_prefix
if prompt_text || print_generated_text || structured_literal_enabled
  g = ML::GGUF::GGUFFile.new(model)
  tokenizer = ML::GGUF::Gemma4Tokenizer.from_gguf(g, model, llama_tokenize_bin)
  g.close
end
if text = prompt_text
  prompt = tokenizer.not_nil!.encode(text)
end
raise "prompt tokens must not be empty" if prompt.empty?
raise "max-seq too small" if max_seq < prompt.size + generate

raise "decode mode must be top1, top2, allowed, literal, or body" unless {"top1", "top2", "allowed", "literal", "body"}.includes?(decode_mode)
raise "--allowed-ids must not be empty" if decode_mode == "allowed" && allowed_ids.empty?
if decode_mode == "literal"
  prefix = constrained_literal_prefix
  raise "--constrained-literal-prefix must not be empty" if !constrained_tool_call_prefix && (prefix.nil? || prefix.empty?)
end
allowed_ids.each do |id|
  raise "--allowed-ids token #{id} out of range" if id < 0 || id >= weights.token_embd.out_dim
end
raise "prefill mode must be serial or rows" unless {"serial", "rows"}.includes?(prefill_mode)
raise "prefill chunk must be positive" unless prefill_chunk > 0
raise "--top1-chain must be positive" unless top1_chain > 0
raise "--body-chain must be positive" unless body_chain > 0
top1_chain_supported = decode_mode == "top1" && decode_wave && top1_wave_resident
body_chain_supported = decode_mode == "body" && decode_wave
if top1_chain > 1 && !top1_chain_supported
  raise "--top1-chain > 1 requires --top1 with --decode-wave and --top1-wave-resident" if top1_chain_explicit
  top1_chain = 1
end
if body_chain > 1 && !body_chain_supported
  raise "--body-chain > 1 requires --body-only with --decode-wave" if body_chain_explicit
  body_chain = 1
end
raise "--prefill-no-head requires --body-only" if !prefill_head && decode_mode != "body"
if sl = stop_layer
  raise "--stop-layer must be non-negative" if sl < 0
  raise "--stop-layer exceeds model layer count" if sl > weights.hparams.n_layer
end
if start_step = decode_stop_layer_after_step
  raise "--decode-stop-layer-after START must be non-negative" if start_step < 0
  layer = decode_stop_layer_after_layer.not_nil!
  raise "--decode-stop-layer-after LAYER must be positive" if layer <= 0
  raise "--decode-stop-layer-after LAYER exceeds model layer count" if layer > weights.hparams.n_layer
  if top1_chain > 1
    raise "--top1-chain > 1 cannot be combined with --decode-stop-layer-after yet" if top1_chain_explicit
    top1_chain = 1
  end
end
if prompt_cache_root && stop_layer && stop_layer != weights.hparams.n_layer
  raise "--prompt-cache-root currently requires full-layer prefill/decode"
end
raise "--decode-only-seed cannot be combined with --prompt-cache-root" if decode_only_seed && prompt_cache_root

cache_store = nil.as(ML::GGUF::Gemma4PromptCache::Store?)
cache_model_id = File.basename(model)
cache_tokenizer_id = prompt_text ? "gemma4-llama-tokenize-oracle" : "synthetic-token-ids"
literal_index = structured_literal_enabled ? Gemma4LiteralTokenIndex.new(tokenizer.not_nil!) : nil
literal_remaining_start = if constrained_tool_call_prefix
                            if constrained_tool_required_param_prefix
                              options = [] of String
                              tool_names.each do |name|
                                required = tool_required_params[name]? || [] of String
                                if required.empty?
                                  options << "<tool_call>\n<function=#{name}>\n"
                                else
                                  required.each do |parameter_name|
                                    options << "<tool_call>\n<function=#{name}>\n<parameter=#{parameter_name}>\n"
                                  end
                                end
                              end
                              options
                            else
                              ML::GGUF::Qwen35Constraints.qwen_tool_call_prefix_options(tool_names)
                            end
                          elsif constrained_literal_prefix
                            [constrained_literal_prefix.not_nil!]
                          else
                            [] of String
                          end
if root = prompt_cache_root
  cache_store = ML::GGUF::Gemma4PromptCache::Store.new(
    root,
    snapshot_cache_byte_limit: prompt_cache_snapshot_mib.to_i64 * 1024_i64 * 1024_i64,
    snapshot_cache_min_free_bytes: prompt_cache_snapshot_min_free_mib.to_i64 * 1024_i64 * 1024_i64,
    snapshot_cache_entry_limit: prompt_cache_snapshot_entries,
  )
end

decode_schedule = decode_stop_layer_after_step ? "#{decode_stop_layer_after_step}:#{decode_stop_layer_after_layer}" : ""
puts "model=#{File.basename(model)} prompt_tokens=#{prompt.join(',')} prompt_len=#{prompt.size} prompt_text_mode=#{chat_user ? "chat_user" : (prompt_text ? "raw_text" : "token_ids")} generate=#{generate} max_seq=#{max_seq} warmups=#{warmups} runs=#{runs} mode=#{decode_mode} decode_wave=#{decode_wave} top1_wave_resident=#{top1_wave_resident} top1_chain=#{top1_chain} body_chain=#{body_chain} body_chain_note=#{body_chain == 1 ? "llama_bench_parity" : "graph_chunk_known_tokens"} allowed_ids=#{allowed_ids.join(',')} constrained_literal_prefix=#{constrained_literal_prefix || ""} constrained_tool_call_prefix=#{constrained_tool_call_prefix} constrained_tool_required_param_prefix=#{constrained_tool_required_param_prefix} tool_names=#{tool_names.join(',')} literal_options=#{literal_remaining_start.size} literal_force_single=#{literal_force_single} literal_force_span=#{literal_force_span} prefill_mode=#{prefill_mode} prefill_chunk=#{prefill_chunk} prefill_head=#{prefill_head} stop_layer=#{stop_layer || weights.hparams.n_layer} decode_stop_layer_after=#{decode_schedule} profile=#{profile} profile_decode_only=#{profile_decode_only} decode_only_seed=#{decode_only_seed || ""} prompt_cache_enabled=#{!cache_store.nil?} prompt_cache_root=#{prompt_cache_root || ""} prompt_cache_snapshot_mib=#{prompt_cache_snapshot_mib} prompt_cache_snapshot_min_free_mib=#{prompt_cache_snapshot_min_free_mib} prompt_cache_snapshot_entries=#{prompt_cache_snapshot_entries} load_ms=#{load_ms.round(3)}"

warmups.times do
  run_once(
    weights, prompt, generate, max_seq, decode_mode, prefill_mode, prefill_chunk, prefill_head, stop_layer,
    profile_decode_only: false,
    decode_wave: decode_wave,
    top1_wave_resident: top1_wave_resident,
    top1_chain: top1_chain,
    body_chain: body_chain,
    decode_stop_layer_after_step: decode_stop_layer_after_step,
    decode_stop_layer_after_layer: decode_stop_layer_after_layer,
    prompt_cache_store: cache_store,
    prompt_cache_model_id: cache_model_id,
    prompt_cache_tokenizer_id: cache_tokenizer_id,
    prompt_cache_session_id: prompt_cache_session,
    decode_only_seed: decode_only_seed,
    allowed_ids: allowed_ids,
    literal_index: literal_index,
    literal_remaining_start: literal_remaining_start,
    literal_force_single: literal_force_single,
    literal_force_span: literal_force_span,
  )
end

prefill_samples = [] of Float64
decode_samples = [] of Float64
cache_restore_samples = [] of Float64
cache_route_counts = Hash(String, Int32).new(0)
literal_forced_single_total = 0
literal_allowed_head_total = 0
literal_forced_span_batches_total = 0
literal_forced_span_tokens_total = 0
first_id = 0_i32
last_id = 0_i32
last_token_trace = [] of Int32
runs.times do
  result = run_once(
    weights, prompt, generate, max_seq, decode_mode, prefill_mode, prefill_chunk, prefill_head, stop_layer, profile && runs == 1,
    profile_decode_only: profile_decode_only,
    decode_wave: decode_wave,
    top1_wave_resident: top1_wave_resident,
    top1_chain: top1_chain,
    body_chain: body_chain,
    decode_stop_layer_after_step: decode_stop_layer_after_step,
    decode_stop_layer_after_layer: decode_stop_layer_after_layer,
    prompt_cache_store: cache_store,
    prompt_cache_model_id: cache_model_id,
    prompt_cache_tokenizer_id: cache_tokenizer_id,
    prompt_cache_session_id: prompt_cache_session,
    decode_only_seed: decode_only_seed,
    allowed_ids: allowed_ids,
    literal_index: literal_index,
    literal_remaining_start: literal_remaining_start,
    literal_force_single: literal_force_single,
    literal_force_span: literal_force_span,
  )
  prefill_samples << result[:prefill_ms]
  decode_samples << result[:decode_ms]
  cache_route_counts[result[:cache_route]] += 1
  literal_forced_single_total += result[:literal_forced_single]
  literal_allowed_head_total += result[:literal_allowed_head]
  literal_forced_span_batches_total += result[:literal_forced_span_batches]
  literal_forced_span_tokens_total += result[:literal_forced_span_tokens]
  cache_restore_samples << result[:cache_restore_ms] if result[:cache_restore_ms] > 0.0
  first_id = result[:first_id]
  last_id = result[:last_id]
  last_token_trace = result[:token_trace]
end

summarize("prefill", prefill_samples, prompt.size)
summarize("decode", decode_samples, generate)
summarize("cache_restore", cache_restore_samples, prompt.size) unless cache_restore_samples.empty?
decode_p50 = percentile(decode_samples.sort, 0.50)
puts "decode_ms_per_token_p50=#{(decode_p50 / generate).round(3)} first_id=#{first_id} last_id=#{last_id}"
puts "literal_summary=forced_single:#{literal_forced_single_total},allowed_head:#{literal_allowed_head_total},span_batches:#{literal_forced_span_batches_total},span_tokens:#{literal_forced_span_tokens_total}" if decode_mode == "literal"
puts "token_trace=#{last_token_trace.join(',')}" if print_generated_ids
if print_generated_text
  puts "generated_text=#{tokenizer.not_nil!.decode(last_token_trace).inspect}"
end
unless cache_route_counts.empty?
  route_summary = cache_route_counts.map { |route, count| "#{route}:#{count}" }.join(",")
  puts "prompt_cache_routes=#{route_summary}"
end
if store = cache_store
  puts "prompt_cache_snapshot_requested_bytes=#{store.snapshot_cache_requested_byte_limit} prompt_cache_snapshot_effective_byte_limit=#{store.snapshot_cache_byte_limit} prompt_cache_snapshot_min_free_bytes=#{store.snapshot_cache_min_free_bytes} prompt_cache_snapshot_bytes=#{store.snapshot_cache_bytes} prompt_cache_snapshot_hits=#{store.snapshot_cache_hits} prompt_cache_snapshot_misses=#{store.snapshot_cache_misses}"
end
