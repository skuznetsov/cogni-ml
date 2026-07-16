require "json"

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_proposal_route"
require "../src/ml/gguf/qwen35_request_state_pool"
require "../src/ml/gguf/qwen35_resident_session"
require "../src/ml/gguf/qwen35_serving_route"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"

DEFAULT_MODEL_PATH    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

model_path = ENV["QWEN35_MODEL"]? || ENV["QWEN35_MODEL_PATH"]? || DEFAULT_MODEL_PATH
tokenizer_bin = ENV["QWEN35_TOKENIZER_BIN"]? || DEFAULT_TOKENIZER_BIN
cache_root = ENV["QWEN35_PROMPT_CACHE_ROOT"]? || ML::GGUF::Qwen35PromptCache.default_root
max_seq = (ENV["QWEN35_RESIDENT_MAX_SEQ"]? || "1024").to_i
pool_capacity = (ENV["QWEN35_RESIDENT_STATE_POOL"]? || "1").to_i
prepare_metal = ENV["QWEN35_PREPARE_STATE_OFF"]? != "1"

raise "model not found: #{model_path}" unless File.exists?(model_path)
raise "QWEN35_RESIDENT_MAX_SEQ must be positive" unless max_seq > 0
raise "QWEN35_RESIDENT_STATE_POOL must be positive" unless pool_capacity > 0

def json_s(obj : JSON::Any, key : String, default : String? = nil) : String?
  return default unless value = obj.as_h[key]?
  value.as_s
end

def json_i(obj : JSON::Any, key : String, default : Int32) : Int32
  return default unless value = obj.as_h[key]?
  value.as_i.to_i32
end

def json_bool(obj : JSON::Any, key : String, default : Bool = false) : Bool
  return default unless value = obj.as_h[key]?
  value.as_bool
end

def session_key(session_id : String, turn_id : String?) : String
  "#{session_id}\0#{turn_id}"
end

startup_t0 = Time.instant
gguf = ML::GGUF::GGUFFile.new(model_path)
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model_path, tokenizer_bin)
gguf.close
weights = ML::GGUF::Qwen35Weights.from_gguf(model_path)
store = ML::GGUF::Qwen35PromptCache::Store.new(cache_root)
model_id = ML::GGUF::Qwen35ProposalRoute.model_id(model_path)
tokenizer_id = ML::GGUF::Qwen35ProposalRoute.tokenizer_id(model_id, tokenizer)
state_pool = ML::GGUF::Qwen35RequestStatePool.new(weights.hparams, max_seq: max_seq, capacity: pool_capacity, prepare_metal: prepare_metal)
resident_sessions = {} of String => ML::GGUF::Qwen35ResidentSession
cursor_states = {} of String => ML::GGUF::Qwen35CPU::State
startup_ms = (Time.instant - startup_t0).total_milliseconds

STDERR.puts "qwen35_resident_jsonl model=#{model_path} cache_root=#{cache_root} max_seq=#{max_seq} pool=#{pool_capacity} startup_ms=#{startup_ms.round(1)}"

STDIN.each_line do |line|
  line = line.strip
  next if line.empty?

  request_t0 = Time.instant
  state = nil.as(ML::GGUF::Qwen35CPU::State?)
  route = "error"
  token_cache_hit = false
  tokenize_ms = 0.0
  prefill_ms = 0.0
  decode_ms = 0.0
  restore_ms = 0.0
  cache_save_ms = 0.0
  prewarm_ms = 0.0
  active_cursor_ready = false
  prompt_ids = [] of Int32
  output_ids = [] of Int32
  generated_text = ""
  error = nil.as(String?)

  begin
    req = JSON.parse(line)
    prompt = json_s(req, "prompt") || raise "request requires string prompt"
    n_gen = json_i(req, "gen", 16)
    session_id = json_s(req, "session_id", "default").not_nil!
    turn_id = json_s(req, "turn_id")
    continuation_required = json_bool(req, "continuation_required")
    active_cursor = json_bool(req, "active_cursor")
    prewarm_active_cursor = json_bool(req, "prewarm_active_cursor")
    cached_gen = json_i(req, "cached_gen", n_gen)
    raise "gen must be positive" unless n_gen > 0
    raise "cached_gen must be positive" unless cached_gen > 0
    raise "cached_gen cannot exceed gen" if cached_gen > n_gen
    raise "active_cursor requires continuation_required" if active_cursor && !continuation_required

    direct_t0 = Time.instant
    hit = if continuation_required
            store.lookup_output_fast_forward(model_id, session_id, prompt, cached_gen, tokenizer_id: tokenizer_id, turn_id: turn_id)
          else
            store.lookup_output_fast_forward_at_most(model_id, session_id, prompt, n_gen, tokenizer_id: tokenizer_id, turn_id: turn_id)
          end
    if hit
      token_cache_hit = true
      prompt_ids = hit.prompt_token_ids
      output_ids = hit.output_token_ids
      tokenize_ms = (Time.instant - direct_t0).total_milliseconds
      if continuation_required
        exact_entry = store.lookup_exact_known_span_for_output_fast_forward(hit) || raise "output fast-forward exact artifact missing"
        full_history = hit.prompt_token_ids + hit.output_token_ids
        key = session_key(session_id, turn_id)
        session = resident_sessions[key]? || begin
          created = ML::GGUF::Qwen35ResidentSession.new(store, weights, model_id, session_id, turn_id)
          resident_sessions[key] = created
          created
        end
        state = state_pool.checkout unless active_cursor && session.active_cursor?
        restore_t0 = Time.instant
        result = session.serve_exact_cached_span(
          prompt,
          hit.output_token_ids,
          exact_entry,
          full_history,
          continuation_required: true,
          reuse_state: state,
        )
        restore_ms = (Time.instant - restore_t0).total_milliseconds
        route = result.route
        output_ids = result.output_token_ids
        if result.route == ML::GGUF::Qwen35ResidentSession::ACTIVE_CURSOR
          state = cursor_states.delete(key)
        end

        replay = result.replay
        if output_ids.size < n_gen
          raise "continuation route missing restored state" unless replay
          next_id = replay.next_token_id
          raise "continuation route missing next token" unless next_id

          decode_t0 = Time.instant
          pos = result.prompt_token_count + output_ids.size - 1
          while output_ids.size < n_gen
            top, _ = ML::GGUF::Qwen35CPU.forward_top1(weights, next_id, pos, replay.state)
            next_id = top.to_i32
            output_ids << next_id
            pos += 1
            break if next_id == tokenizer.eos_id
          end
          decode_ms = (Time.instant - decode_t0).total_milliseconds
        end
        generated_text = tokenizer.decode(output_ids)
      else
        route = hit.output_token_ids.size == n_gen ? ML::GGUF::Qwen35ServingRoute::DIRECT_OUTPUT : "direct_output_short"
        generated_text = hit.generated_text || tokenizer.decode(output_ids)
      end
    else
      tokenize_t0 = Time.instant
      if tokenized = store.lookup_tokenized_prompt(model_id, tokenizer_id, prompt)
        token_cache_hit = true
        prompt_ids = tokenized.token_ids
      else
        prompt_ids = tokenizer.encode(prompt)
        store.save_tokenized_prompt(model_id, tokenizer_id, prompt, prompt_ids)
      end
      tokenize_ms = (Time.instant - tokenize_t0).total_milliseconds
      raise "prompt encoded to zero tokens" if prompt_ids.empty?
      raise "request exceeds max_seq: prompt=#{prompt_ids.size} gen=#{n_gen} max_seq=#{max_seq}" if prompt_ids.size + n_gen >= max_seq

      state = state_pool.checkout
      route = "greedy"
      prefill_t0 = Time.instant
      first_token, _first_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, state.not_nil!)
      prefill_ms = (Time.instant - prefill_t0).total_milliseconds
      output_ids << first_token

      decode_t0 = Time.instant
      pos = prompt_ids.size
      while output_ids.size < n_gen
        top, _ = ML::GGUF::Qwen35CPU.forward_top1(weights, output_ids[-1], pos, state.not_nil!)
        next_id = top.to_i32
        output_ids << next_id
        pos += 1
        break if next_id == tokenizer.eos_id
      end
      decode_ms = (Time.instant - decode_t0).total_milliseconds
      generated_text = tokenizer.decode(output_ids)

      save_t0 = Time.instant
      full_history = prompt_ids + output_ids
      cached_prefix = full_history[0, full_history.size - 1]
      exact_entry = store.save(
        session_id: session_id,
        turn_id: turn_id,
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        prompt_text: "",
        token_ids: cached_prefix,
        state: state.not_nil!,
        artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: output_ids.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history),
        next_token_id: output_ids[-1],
      )
      store.save_source_history(
        session_id: session_id,
        turn_id: turn_id,
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        token_ids: full_history,
        generated_token_count: output_ids.size,
        generated_text: generated_text,
      )
      store.save_output_fast_forward(
        session_id: session_id,
        turn_id: turn_id,
        model_id: model_id,
        tokenizer_id: tokenizer_id,
        prompt_text: prompt,
        prompt_token_ids: prompt_ids,
        output_token_ids: output_ids,
        generated_text: generated_text,
        exact_entry: exact_entry,
        terminal_token_id: output_ids.last? == tokenizer.eos_id ? tokenizer.eos_id : nil,
      )
      cache_save_ms = (Time.instant - save_t0).total_milliseconds

      if prewarm_active_cursor
        key = session_key(session_id, turn_id)
        session = resident_sessions[key]? || begin
          created = ML::GGUF::Qwen35ResidentSession.new(store, weights, model_id, session_id, turn_id)
          resident_sessions[key] = created
          created
        end
        if old_state = cursor_states.delete(key)
          session.clear_active_cursor
          state_pool.release(old_state)
        end
        prewarm_t0 = Time.instant
        session.prewarm_continuation_cursor(
          prompt,
          output_ids,
          exact_entry,
          full_history,
          reuse_state: state.not_nil!,
        )
        prewarm_ms = (Time.instant - prewarm_t0).total_milliseconds
        cursor_states[key] = state.not_nil!
        state = nil
        active_cursor_ready = true
      end
    end
    if turn_id
      key = session_key(session_id, turn_id)
      active_cursor_ready ||= resident_sessions[key]?.try(&.active_cursor?) || false
    else
      key = session_key(session_id, nil)
      active_cursor_ready ||= resident_sessions[key]?.try(&.active_cursor?) || false
    end
  rescue ex
    error = ex.message || ex.class.name
  ensure
    state_pool.release(state.not_nil!) if state
  end

  total_ms = (Time.instant - request_t0).total_milliseconds
  JSON.build(STDOUT) do |json|
    json.object do
      json.field "ok", error.nil?
      json.field "route", route
      json.field "error", error if error
      json.field "total_ms", total_ms.round(3)
      json.field "tokenize_ms", tokenize_ms.round(3)
      json.field "restore_ms", restore_ms.round(3)
      json.field "prefill_ms", prefill_ms.round(3)
      json.field "decode_ms", decode_ms.round(3)
      json.field "cache_save_ms", cache_save_ms.round(3)
      json.field "prewarm_ms", prewarm_ms.round(3)
      json.field "token_cache_hit", token_cache_hit
      json.field "active_cursor_ready", active_cursor_ready
      json.field "prompt_tokens", prompt_ids.size
      json.field "output_tokens", output_ids.size
      json.field "output_ids", output_ids
      json.field "generated_text", generated_text
    end
  end
  STDOUT << '\n'
  STDOUT.flush
end
