# Greedy generation demo for Qwen 3.5 9B.
#
# Usage:
#   crystal run bin/qwen35_generate.cr -- "Your prompt here" [n_tokens]
#
# Uses the native Metal wave path when available; set
# `QWEN35_DECODE_WAVE_OFF=1` for the slow CPU reference path.

require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_chat"
require "../src/ml/gguf/qwen35_constraints"
require "../src/ml/gguf/ngram_draft"
require "../src/ml/gguf/qwen35_prompt_cache"
require "../src/ml/gguf/qwen35_serving_route"
require "../src/ml/gguf/qwen35_weights"
require "../src/ml/gguf/qwen35_tokenizer"

MODEL_PATH         = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DRAFT_MODEL_PATH   = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
LLAMA_TOKENIZE_BIN = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

request_t0 = Time.instant
model_load_ms = 0.0
draft_load_ms = 0.0
tokenize_ms = 0.0
state_prepare_ms = 0.0
source_history_lookup_ms = 0.0
cache_restore_ms = 0.0
prefill_ms = 0.0
decode_ms = 0.0
source_history_save_ms = 0.0
token_cache_hit = false
cache_route = "none"

prompt = ARGV[0]? || "The capital of France is"
n_gen = (ARGV[1]? || "8").to_i
chat_tools = if tools_json = ENV["QWEN35_TOOLS_JSON"]?
               ML::GGUF::Qwen35Chat.parse_tools_json(tools_json)
             else
               [] of JSON::Any
             end
chat_messages = ENV["QWEN35_MESSAGES_JSON"]?.try { |json| ML::GGUF::Qwen35Chat.messages_from_openai_json(json) }
chat_mode = ENV["QWEN35_CHAT"]? == "1" || !chat_tools.empty? || !chat_messages.nil?
tool_response_json_format = (ENV["QWEN35_TOOL_RESPONSE_JSON"]? || "").downcase
unless tool_response_json_format.empty? || tool_response_json_format == "simple" || tool_response_json_format == "openai"
  raise "QWEN35_TOOL_RESPONSE_JSON must be simple or openai"
end
model_prompt = if messages = chat_messages
                 ML::GGUF::Qwen35Chat.render(messages, tools: chat_tools)
               elsif chat_mode
                 ML::GGUF::Qwen35Chat.render_user_prompt(
                   prompt,
                   system: ENV["QWEN35_CHAT_SYSTEM"]?,
                   tools: chat_tools,
                 )
               else
                 prompt
               end
prompt_cache_enabled = ENV["QWEN35_PROMPT_CACHE"]? == "1"
prompt_cache_source_history_enabled = ENV["QWEN35_PROMPT_CACHE_SOURCE_HISTORY"]? == "1"
prompt_cache_fast_forward_enabled = prompt_cache_enabled && prompt_cache_source_history_enabled && ENV["QWEN35_PROMPT_CACHE_FAST_FORWARD"]? == "1"
prompt_cache_preweight_fast_forward_enabled = prompt_cache_fast_forward_enabled && ENV["QWEN35_PROMPT_CACHE_PREWEIGHT_FAST_FORWARD_OFF"]? != "1"
prompt_token_cache_enabled = prompt_cache_enabled && ENV["QWEN35_PROMPT_TOKEN_CACHE_OFF"]? != "1"
prompt_cache_full_hit_min_gen = (ENV["QWEN35_PROMPT_CACHE_FULL_HIT_MIN_GEN"]? || "64").to_i
prompt_cache_artifact_codec = ENV["QWEN35_PROMPT_CACHE_ARTIFACT_CODEC"]?.try(&.downcase)
prompt_cache_artifact_codec = nil if prompt_cache_artifact_codec == "raw" || prompt_cache_artifact_codec == ""
prompt_cache_artifact_codec_block = (ENV["QWEN35_PROMPT_CACHE_ARTIFACT_CODEC_BLOCK"]? || "8").to_i
prompt_cache_live_kv_artifacts = ENV["QWEN35_PROMPT_CACHE_LIVE_KV_ARTIFACTS"]? == "1"
trace_steps = ENV["QWEN35_TRACE_STEPS_OFF"]? != "1" && ENV["QWEN35_QUIET"]? != "1"
decode_policy = (ENV["QWEN35_DECODE_POLICY"]? || "").downcase
unless decode_policy.empty? || decode_policy == "greedy" || decode_policy == "ngram" || decode_policy == "speculative" || decode_policy == "auto"
  raise "QWEN35_DECODE_POLICY must be greedy, ngram, speculative, or auto"
end
legacy_speculative_decode_enabled = ENV["QWEN35_SPECULATIVE_DECODE"]? == "1" || ENV.has_key?("QWEN35_DRAFT_MODEL")
legacy_ngram_decode_enabled = ENV["QWEN35_NGRAM_DECODE"]? == "1"
if decode_policy.empty? && legacy_speculative_decode_enabled && legacy_ngram_decode_enabled
  raise "QWEN35_SPECULATIVE_DECODE and QWEN35_NGRAM_DECODE are mutually exclusive; set QWEN35_DECODE_POLICY=ngram or speculative"
end
speculative_decode_enabled = false
ngram_decode_enabled = false
case decode_policy
when "greedy"
  # Explicit policy overrides legacy env toggles.
when "ngram", "auto"
  ngram_decode_enabled = true
when "speculative"
  speculative_decode_enabled = true
else
  speculative_decode_enabled = legacy_speculative_decode_enabled
  ngram_decode_enabled = legacy_ngram_decode_enabled
end
draft_model_path = ENV["QWEN35_DRAFT_MODEL"]? || DRAFT_MODEL_PATH
spec_gamma = (ENV["QWEN35_SPEC_GAMMA"]? || "4").to_i
spec_max_gamma = (ENV["QWEN35_SPEC_MAX_GAMMA"]? || "32").to_i
spec_plain_fallback_enabled = ENV["QWEN35_SPEC_PLAIN_FALLBACK_OFF"]? != "1"
spec_plain_fallback_gamma = (ENV["QWEN35_SPEC_PLAIN_FALLBACK_GAMMA"]? || "2").to_i
spec_full_accept_streak = (ENV["QWEN35_SPEC_FULL_ACCEPT_STREAK"]? || "2").to_i
spec_fast_regrow_min_gamma = (ENV["QWEN35_SPEC_FAST_REGROW_MIN_GAMMA"]? || "8").to_i
spec_bootstrap_gamma = (ENV["QWEN35_SPEC_BOOTSTRAP_GAMMA"]? || "0").to_i
spec_bootstrap_streak = (ENV["QWEN35_SPEC_BOOTSTRAP_STREAK"]? || "1").to_i
spec_single_fast_enabled = ENV["QWEN35_SPEC_SINGLE_FAST_OFF"]? != "1"
spec_verify_mode = (ENV["QWEN35_SPEC_VERIFY"]? || "chunk-inplace").downcase
spec_skip_draft_before_fallback = ENV["QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF"]? != "1"
spec_skip_draft_backup_before_fallback = ENV["QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF"]? != "1"
ngram_gamma = (ENV["QWEN35_NGRAM_GAMMA"]? || "32").to_i
ngram_min = (ENV["QWEN35_NGRAM_MIN"]? || "6").to_i
ngram_max = (ENV["QWEN35_NGRAM_MAX"]? || "8").to_i
ngram_stage_min = (ENV["QWEN35_NGRAM_STAGE_MIN"]? || (decode_policy == "auto" ? (ngram_gamma + 1).to_s : "0")).to_i
ngram_stage_gate = (ENV["QWEN35_NGRAM_STAGE_GATE"]? || "4").to_i
ngram_risk_min_size = (ENV["QWEN35_NGRAM_RISK_MIN_SIZE"]? || "16").to_i
ngram_min_candidates = (ENV["QWEN35_NGRAM_MIN_CANDIDATES"]? || (decode_policy == "auto" ? "8" : "0")).to_i
ngram_risk_gate = if value = ENV["QWEN35_NGRAM_RISK_GATE"]?
                    value == "1"
                  else
                    decode_policy == "auto"
                  end
ngram_corridor_gate = if value = ENV["QWEN35_NGRAM_CORRIDOR_GATE"]?
                        value == "1"
                      else
                        decode_policy == "auto"
                      end
ngram_corridor_min_size = (ENV["QWEN35_NGRAM_CORRIDOR_MIN_SIZE"]? || "4").to_i
ngram_corridor_match_len_min = (ENV["QWEN35_NGRAM_CORRIDOR_MATCH_LEN_MIN"]? || (decode_policy == "auto" ? "8" : "0")).to_i
ngram_corridor_lag4_min = (ENV["QWEN35_NGRAM_CORRIDOR_LAG4_MIN"]? || "0.25").to_f
ngram_corridor_lag8_min = (ENV["QWEN35_NGRAM_CORRIDOR_LAG8_MIN"]? || (decode_policy == "auto" ? "2.0" : "0.5")).to_f
ngram_corridor_entropy_max = (ENV["QWEN35_NGRAM_CORRIDOR_ENTROPY_MAX"]? || "0.6").to_f
ngram_recursive = ENV["QWEN35_NGRAM_RECURSIVE_OFF"]? != "1"
ngram_disable_after_reject = ENV["QWEN35_NGRAM_DISABLE_AFTER_REJECT_OFF"]? != "1"
ngram_replay_on_reject = ENV["QWEN35_NGRAM_REPLAY_ON_REJECT"]? == "1"
ngram_index_enabled = ENV["QWEN35_NGRAM_INDEX_OFF"]? != "1"
ngram_cache_min_remaining = (ENV["QWEN35_NGRAM_CACHE_MIN_REMAINING"]? || (decode_policy == "auto" ? "64" : "0")).to_i
prepare_state_metal = ENV["QWEN35_PREPARE_STATE_OFF"]? != "1"
metal_profile_enabled = ENV["QWEN35_METAL_PROFILE"]? == "1"
constrained_literal_prefix = ENV["QWEN35_CONSTRAINED_LITERAL_PREFIX"]?
constrained_tool_call_prefix_enabled = ENV["QWEN35_CONSTRAINED_TOOL_CALL_PREFIX"]? == "1"
structured_constraint_enabled = (constrained_literal_prefix && !constrained_literal_prefix.not_nil!.empty?) || constrained_tool_call_prefix_enabled

raise "QWEN35_PROMPT_CACHE_FULL_HIT_MIN_GEN must be non-negative" unless prompt_cache_full_hit_min_gen >= 0
raise "QWEN35_PROMPT_CACHE_ARTIFACT_CODEC_BLOCK must be positive" unless prompt_cache_artifact_codec_block > 0
unless prompt_cache_artifact_codec.nil? || prompt_cache_artifact_codec == "recurrent-bf16" || prompt_cache_artifact_codec == "recurrent-int8"
  raise "QWEN35_PROMPT_CACHE_ARTIFACT_CODEC must be raw, recurrent-bf16, or recurrent-int8"
end
if prompt_cache_artifact_codec == "recurrent-int8" && ENV["QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE"]? != "1"
  raise "QWEN35_PROMPT_CACHE_ARTIFACT_CODEC=recurrent-int8 requires QWEN35_PROMPT_CACHE_METAL_INT8_RESTORE=1"
end
raise "QWEN35_NGRAM_GAMMA must be positive" unless ngram_gamma > 0
raise "QWEN35_NGRAM_MIN must be positive" unless ngram_min > 0
raise "QWEN35_NGRAM_MAX must be >= QWEN35_NGRAM_MIN" unless ngram_max >= ngram_min
raise "QWEN35_NGRAM_STAGE_MIN must be non-negative" unless ngram_stage_min >= 0
raise "QWEN35_NGRAM_STAGE_GATE must be positive" unless ngram_stage_gate > 0
raise "QWEN35_NGRAM_RISK_MIN_SIZE must be positive" unless ngram_risk_min_size > 0
raise "QWEN35_NGRAM_MIN_CANDIDATES must be non-negative" unless ngram_min_candidates >= 0
raise "QWEN35_NGRAM_CACHE_MIN_REMAINING must be non-negative" unless ngram_cache_min_remaining >= 0
raise "QWEN35_NGRAM_CORRIDOR_MIN_SIZE must be positive" unless ngram_corridor_min_size > 0
raise "QWEN35_NGRAM_CORRIDOR_MATCH_LEN_MIN must be non-negative" unless ngram_corridor_match_len_min >= 0
raise "QWEN35_NGRAM_CORRIDOR_LAG4_MIN must be non-negative" unless ngram_corridor_lag4_min >= 0.0
raise "QWEN35_NGRAM_CORRIDOR_LAG8_MIN must be non-negative" unless ngram_corridor_lag8_min >= 0.0
raise "QWEN35_NGRAM_CORRIDOR_ENTROPY_MAX must be non-negative" unless ngram_corridor_entropy_max >= 0.0
raise "QWEN35_SPEC_GAMMA must be positive" unless spec_gamma > 0
raise "QWEN35_SPEC_MAX_GAMMA must be positive" unless spec_max_gamma > 0
raise "QWEN35_SPEC_PLAIN_FALLBACK_GAMMA must be positive" unless spec_plain_fallback_gamma > 0
raise "QWEN35_SPEC_FULL_ACCEPT_STREAK must be positive" unless spec_full_accept_streak > 0
raise "QWEN35_SPEC_FAST_REGROW_MIN_GAMMA must be non-negative" unless spec_fast_regrow_min_gamma >= 0
raise "QWEN35_SPEC_BOOTSTRAP_GAMMA must be non-negative" unless spec_bootstrap_gamma >= 0
raise "QWEN35_SPEC_BOOTSTRAP_STREAK must be positive" unless spec_bootstrap_streak > 0
unless spec_verify_mode == "chunk-inplace" || spec_verify_mode == "hybrid" || spec_verify_mode == "serial"
  raise "QWEN35_SPEC_VERIFY must be chunk-inplace, hybrid, or serial"
end
if constrained_literal_prefix && !constrained_literal_prefix.not_nil!.empty? && constrained_tool_call_prefix_enabled
  raise "QWEN35_CONSTRAINED_LITERAL_PREFIX and QWEN35_CONSTRAINED_TOOL_CALL_PREFIX are mutually exclusive"
end
if structured_constraint_enabled
  raise "constrained structured decoding is currently supported only with greedy decode" if speculative_decode_enabled || ngram_decode_enabled
  raise "constrained structured decoding is currently incompatible with prompt cache fast paths" if prompt_cache_enabled
end
if constrained_tool_call_prefix_enabled && chat_tools.empty?
  raise "QWEN35_CONSTRAINED_TOOL_CALL_PREFIX requires QWEN35_TOOLS_JSON with at least one function tool"
end
spec_max_gamma = Math.max(spec_max_gamma, spec_gamma)

def cache_model_id(path : String) : String
  info = File.info(path)
  ML::GGUF::Qwen35PromptCache.short_hash("model\0#{path}\0#{info.size}\0#{info.modification_time.to_unix}")
end

def cache_tokenizer_id(model_id : String, tok : ML::GGUF::Qwen35Tokenizer) : String
  ML::GGUF::Qwen35PromptCache.short_hash("tokenizer\0#{model_id}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}")
end

def prefill_next(weights : ML::GGUF::Qwen35Weights,
                 token_ids : Array(Int32),
                 state : ML::GGUF::Qwen35CPU::State) : Int32
  top, _logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, token_ids, 0, state)
  top.to_i32
end

def advance_next(weights : ML::GGUF::Qwen35Weights,
                 token_id : Int32,
                 pos : Int32,
                 state : ML::GGUF::Qwen35CPU::State) : Int32
  top, _logit = ML::GGUF::Qwen35CPU.forward_top1(weights, token_id, pos, state)
  top.to_i32
end

def literal_constraint_complete?(remaining : Array(String)) : Bool
  remaining.empty? || remaining.any?(&.empty?)
end

def advance_next_maybe_literal_constrained(weights : ML::GGUF::Qwen35Weights,
                                           tokenizer : ML::GGUF::Qwen35Tokenizer,
                                           token_index : ML::GGUF::Qwen35Constraints::TokenTextIndex,
                                           token_id : Int32,
                                           pos : Int32,
                                           state : ML::GGUF::Qwen35CPU::State,
                                           remaining : Array(String)) : {Int32, Float32, Array(String), String, Bool}
  if literal_constraint_complete?(remaining)
    top, logit = ML::GGUF::Qwen35CPU.forward_top1(weights, token_id, pos, state)
    return {top.to_i32, logit, [] of String, "", false}
  end

  allowed = ML::GGUF::Qwen35Constraints.literal_frontier_ids(token_index, remaining)
  if allowed.empty?
    top, logit = ML::GGUF::Qwen35CPU.forward_top1(weights, token_id, pos, state)
    return {top.to_i32, logit, [] of String, "", false}
  end

  top, logit = ML::GGUF::Qwen35CPU.forward_top1_allowed(weights, token_id, pos, state, allowed)
  piece = token_index.text_for_id(top)
  next_remaining = ML::GGUF::Qwen35Constraints.advance_literal_options(remaining, piece)
  next_remaining = [] of String if literal_constraint_complete?(next_remaining)
  {top.to_i32, logit, next_remaining, piece, true}
end

def advance_tool_literal_stage(stage : String,
                               emitted : String,
                               required_by_function : Hash(String, Array(String)),
                               required_parameters : Array(String),
                               parameter_index : Int32,
                               value_options_by_function : Hash(String, Hash(String, Array(String))),
                               value_options_by_parameter : Hash(String, Array(String))) : {String, Array(String), Array(String), Int32, Hash(String, Array(String)), Bool}
  if stage == "parameter_open"
    next_stage, next_literals = next_tool_value_corridor(required_parameters, parameter_index, value_options_by_parameter)
    return {next_stage, next_literals, required_parameters, parameter_index, value_options_by_parameter, true}
  end

  if stage == "parameter_separator"
    next_index = parameter_index + 1
    next_stage, next_literals = next_tool_value_corridor(required_parameters, next_index, value_options_by_parameter)
    return {next_stage, next_literals, required_parameters, next_index, value_options_by_parameter, true}
  end

  if stage == "value_literal"
    next_stage, next_literals = next_tool_value_close_corridor(required_parameters, parameter_index)
    return {next_stage, next_literals, required_parameters, parameter_index, value_options_by_parameter, false}
  end

  if stage == "closing_parameter"
    return {"done", [] of String, required_parameters, parameter_index, value_options_by_parameter, false}
  end

  return {stage, [] of String, required_parameters, parameter_index, value_options_by_parameter, false} unless stage == "function_prefix"

  required_by_function.each do |function_name, required|
    next unless emitted.ends_with?("<function=#{function_name}>\n")
    next if required.empty?

    options = ML::GGUF::Qwen35Constraints.qwen_parameter_open_options([required[0]])
    selected_value_options = value_options_by_function[function_name]? || {} of String => Array(String)
    return {"parameter_open", options, required, 0, selected_value_options, false}
  end

  {"done", [] of String, required_parameters, parameter_index, value_options_by_parameter, false}
end

def next_tool_value_corridor(required_parameters : Array(String),
                             parameter_index : Int32,
                             value_options_by_parameter : Hash(String, Array(String))) : {String, Array(String)}
  parameter_name = required_parameters[parameter_index]?
  return {"value", [] of String} unless parameter_name

  value_options = value_options_by_parameter[parameter_name]?
  return {"value", [] of String} unless value_options && !value_options.empty?

  {"value_literal", value_options}
end

def next_tool_value_close_corridor(required_parameters : Array(String),
                                   parameter_index : Int32) : {String, Array(String)}
  next_parameter_index = parameter_index + 1
  if next_parameter_index < required_parameters.size
    return {"parameter_separator", ML::GGUF::Qwen35Constraints.qwen_parameter_continue_options([required_parameters[next_parameter_index]])}
  end

  {"closing_parameter", ML::GGUF::Qwen35Constraints.qwen_single_parameter_close_options}
end

def maybe_start_tool_value_close(stage : String,
                                 value_text : String,
                                 required_parameters : Array(String),
                                 parameter_index : Int32) : {String, Array(String), Int32, String}
  return {stage, [] of String, parameter_index, value_text} unless stage == "value"
  newline_index = value_text.index('\n')
  return {stage, [] of String, parameter_index, value_text} unless newline_index
  return {stage, [] of String, parameter_index, value_text} if value_text[0...newline_index].strip.empty?

  next_parameter_index = parameter_index + 1
  has_next_parameter = next_parameter_index < required_parameters.size
  close_stage, close_options = next_tool_value_close_corridor(required_parameters, parameter_index)
  emitted_after_newline = value_text[(newline_index + 1)..]
  return {close_stage, close_options, parameter_index, value_text} if emitted_after_newline.empty?

  remaining = ML::GGUF::Qwen35Constraints.advance_literal_options(close_options, emitted_after_newline)
  if literal_constraint_complete?(remaining)
    if has_next_parameter
      return {"value", [] of String, next_parameter_index, ""}
    else
      return {"done", [] of String, parameter_index, value_text}
    end
  end
  return {stage, [] of String, parameter_index, value_text} if remaining.empty?

  {close_stage, remaining, parameter_index, value_text}
end

def resync_draft!(weights : ML::GGUF::Qwen35Weights,
                  state : ML::GGUF::Qwen35CPU::State,
                  base : ML::GGUF::Qwen35CPU::State,
                  accepted_or_corrected : Array(Int32),
                  start_pos : Int32) : Int32
  state.copy_from!(base)
  next_id = -1
  accepted_or_corrected.each_with_index do |tok, i|
    next_id = advance_next(weights, tok, start_pos + i, state)
  end
  next_id
end

def with_guarded_full_rows_disabled(&)
  old_guard = ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"]?
  ENV.delete("QWEN35_HEAD_FULL_ROWS_GUARDED")
  yield
ensure
  if old_guard
    ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"] = old_guard
  else
    ENV.delete("QWEN35_HEAD_FULL_ROWS_GUARDED")
  end
end

def print_qwen_tool_calls_if_any(text : String,
                                 chat_mode : Bool,
                                 tool_response_json_format : String,
                                 tools : Array(JSON::Any) = [] of JSON::Any) : Nil
  return unless chat_mode || !tool_response_json_format.empty?

  calls = ML::GGUF::Qwen35Chat.parse_tool_calls(text)
  content = ML::GGUF::Qwen35Chat.content_without_tool_calls(text)

  unless calls.empty?
    puts "\n=== Parsed tool calls ==="
    puts ML::GGUF::Qwen35Chat.tool_calls_to_json(calls, tools)
  end

  unless tool_response_json_format.empty?
    puts "\n=== Tool response JSON ==="
    if tool_response_json_format == "openai"
      puts ML::GGUF::Qwen35Chat.tool_response_to_openai_json(calls, content, tools)
    else
      puts ML::GGUF::Qwen35Chat.tool_response_to_json(calls, content, tools)
    end
  end
end

def replay_target_state(weights : ML::GGUF::Qwen35Weights,
                        prompt_ids : Array(Int32),
                        generated_ids : Array(Int32),
                        max_seq : Int32,
                        prepare_state_metal : Bool) : {ML::GGUF::Qwen35CPU::State, Int32}
  replay_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(replay_state, weights.hparams) if prepare_state_metal
  replay_ids = prompt_ids.dup
  replay_ids.concat(generated_ids)
  {replay_state, prefill_next(weights, replay_ids, replay_state)}
end

session_id = ENV["QWEN35_SESSION_ID"]? || "default"
turn_id = ENV["QWEN35_TURN_ID"]?
cache_store = nil.as(ML::GGUF::Qwen35PromptCache::Store?)
cache_model = ""
cache_tokenizer = ""
cache_root = ""
source_history_hit = nil.as(ML::GGUF::Qwen35PromptCache::SourceHistoryEntry?)
output_ids = [] of Int32
output_text = nil.as(String?)
cached_prompt_ids = nil.as(Array(Int32)?)

if prompt_cache_preweight_fast_forward_enabled && prompt_token_cache_enabled
  preflight_t0 = Time.instant
  cache_root = ENV["QWEN35_PROMPT_CACHE_ROOT"]? || ML::GGUF::Qwen35PromptCache.default_root
  cache_store = ML::GGUF::Qwen35PromptCache::Store.new(cache_root)
  cache_model = cache_model_id(MODEL_PATH)
  if output_hit = cache_store.not_nil!.lookup_output_fast_forward(cache_model, session_id, model_prompt, n_gen, turn_id: turn_id)
    token_cache_hit = true
    cached_prompt_ids = output_hit.prompt_token_ids
    cache_tokenizer = output_hit.tokenizer_id
    output_ids = output_hit.output_token_ids
    output_text = output_hit.generated_text
    cache_route = ML::GGUF::Qwen35ServingRoute::DIRECT_OUTPUT
    source_history_lookup_ms = (Time.instant - preflight_t0).total_milliseconds
    tokenize_ms = source_history_lookup_ms
    STDOUT << "\nPrompt cache direct output fast-forward hit before tokenizer/weight load: emitted #{output_ids.size} cached tokens\n"
    total_ms = (Time.instant - request_t0).total_milliseconds
    STDOUT << "  request summary: total_ms=#{total_ms.round(1)} model_load_ms=#{model_load_ms.round(1)} draft_load_ms=#{draft_load_ms.round(1)} tokenize_ms=#{tokenize_ms.round(1)} token_cache_hit=#{token_cache_hit} cache_route=#{cache_route} state_prepare_ms=#{state_prepare_ms.round(1)} source_history_lookup_ms=#{source_history_lookup_ms.round(1)} cache_restore_ms=#{cache_restore_ms.round(1)} prefill_ms=#{prefill_ms.round(1)} decode_ms=#{decode_ms.round(1)} source_history_save_ms=#{source_history_save_ms.round(1)} prompt_tokens=#{cached_prompt_ids.not_nil!.size} output_tokens=#{output_ids.size}\n"

    puts "\n=== Generated token ids ==="
    puts output_ids.inspect
    puts "\n=== Generated text ==="
    puts output_text
    puts "\n=== Full output ==="
    puts model_prompt + output_text.not_nil!
    print_qwen_tool_calls_if_any(output_text.not_nil!, chat_mode, tool_response_json_format, chat_tools)
    exit
  elsif tokenized_hit = cache_store.not_nil!.lookup_tokenized_prompt_for_model(cache_model, model_prompt)
    token_cache_hit = true
    cached_prompt_ids = tokenized_hit.token_ids
    cache_tokenizer = tokenized_hit.tokenizer_id
    source_history_hit = cache_store.not_nil!.lookup_source_history(session_id, cache_model, cache_tokenizer, turn_id: turn_id)
    source_history_lookup_ms = (Time.instant - preflight_t0).total_milliseconds
    if source = source_history_hit
      ids = cached_prompt_ids.not_nil!
      replay_start = ids.size
      source_remaining = source.token_ids.size - replay_start
      if source.token_ids.size > replay_start &&
         source_remaining >= n_gen &&
         ML::GGUF::Qwen35PromptCache.generated_text_metadata_valid?(source, n_gen) &&
         (cached_text = source.generated_text) &&
        ML::GGUF::Qwen35PromptCache.source_history_prefix_match?(source.token_ids, ids, replay_start)
        full_history_len = ids.size + n_gen
        cached_prefix_len = full_history_len - 1
        if fast_hit = cache_store.not_nil!.lookup_token_prefix(
             cache_model,
             cache_tokenizer,
             source.token_ids,
             cached_prefix_len)
          if ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(fast_hit, source.token_ids, n_gen, full_history_len)
            output_ids = source.token_ids[ids.size, n_gen]
            output_text = cached_text
            cache_route = "source_history_direct_output"
            tokenize_ms = source_history_lookup_ms
            STDOUT << "\nPrompt cache output fast-forward hit before tokenizer/weight load: emitted #{output_ids.size} cached tokens\n"
            total_ms = (Time.instant - request_t0).total_milliseconds
            STDOUT << "  request summary: total_ms=#{total_ms.round(1)} model_load_ms=#{model_load_ms.round(1)} draft_load_ms=#{draft_load_ms.round(1)} tokenize_ms=#{tokenize_ms.round(1)} token_cache_hit=#{token_cache_hit} cache_route=#{cache_route} state_prepare_ms=#{state_prepare_ms.round(1)} source_history_lookup_ms=#{source_history_lookup_ms.round(1)} cache_restore_ms=#{cache_restore_ms.round(1)} prefill_ms=#{prefill_ms.round(1)} decode_ms=#{decode_ms.round(1)} source_history_save_ms=#{source_history_save_ms.round(1)} prompt_tokens=#{ids.size} output_tokens=#{output_ids.size}\n"

            puts "\n=== Generated token ids ==="
            puts output_ids.inspect
            puts "\n=== Generated text ==="
            puts output_text
            puts "\n=== Full output ==="
            puts model_prompt + output_text.not_nil!
            print_qwen_tool_calls_if_any(output_text.not_nil!, chat_mode, tool_response_json_format, chat_tools)
            exit
          end
        end
      end
    end
  end
end

puts "Loading tokenizer metadata..."
t0 = Time.instant
g = ML::GGUF::GGUFFile.new(MODEL_PATH)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(g, MODEL_PATH, LLAMA_TOKENIZE_BIN)
g.close
model_load_ms = (Time.instant - t0).total_milliseconds
puts "Loaded tokenizer metadata in #{(model_load_ms / 1000.0).round(1)}s. vocab=#{tok.vocab.size}"
puts "Qwen chat mode enabled: tools=#{chat_tools.size} rendered_prompt_chars=#{model_prompt.size}" if chat_mode

if prompt_cache_enabled && cache_store.nil?
  cache_root = ENV["QWEN35_PROMPT_CACHE_ROOT"]? || ML::GGUF::Qwen35PromptCache.default_root
  cache_store = ML::GGUF::Qwen35PromptCache::Store.new(cache_root)
  cache_model = cache_model_id(MODEL_PATH)
  cache_tokenizer = cache_tokenizer_id(cache_model, tok)
elsif prompt_cache_enabled && cache_tokenizer.empty?
  cache_tokenizer = cache_tokenizer_id(cache_model, tok)
end

# Encode prompt
tokenize_t0 = Time.instant
ids = if cached = cached_prompt_ids
        cached
      elsif prompt_token_cache_enabled && (tokenized_hit = cache_store.not_nil!.lookup_tokenized_prompt(cache_model, cache_tokenizer, model_prompt))
        token_cache_hit = true
        tokenized_hit.token_ids
      else
        encoded = tok.encode(model_prompt)
        cache_store.not_nil!.save_tokenized_prompt(cache_model, cache_tokenizer, model_prompt, encoded) if prompt_token_cache_enabled
        encoded
      end
tokenize_ms = (Time.instant - tokenize_t0).total_milliseconds
puts "Prompt tokens (#{ids.size}): #{ids.inspect}"
puts "Prompt decoded: #{tok.decode(ids).inspect}"
constraint_token_index = structured_constraint_enabled ? ML::GGUF::Qwen35Constraints::TokenTextIndex.new(tok) : nil

if prompt_cache_enabled && prompt_cache_source_history_enabled && source_history_hit.nil?
  source_lookup_t0 = Time.instant
  source_history_hit = cache_store.not_nil!.lookup_source_history(session_id, cache_model, cache_tokenizer, turn_id: turn_id)
  source_history_lookup_ms = (Time.instant - source_lookup_t0).total_milliseconds
end

if prompt_cache_preweight_fast_forward_enabled && (source = source_history_hit)
  replay_start = ids.size
  source_remaining = source.token_ids.size - replay_start
  if source.token_ids.size > replay_start &&
     source_remaining >= n_gen &&
     ML::GGUF::Qwen35PromptCache.source_history_prefix_match?(source.token_ids, ids, replay_start)
    full_history_len = ids.size + n_gen
    cached_prefix_len = full_history_len - 1
    if fast_hit = cache_store.not_nil!.lookup_token_prefix(
         cache_model,
         cache_tokenizer,
         source.token_ids,
         cached_prefix_len)
      if ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(fast_hit, source.token_ids, n_gen, full_history_len)
        output_ids = source.token_ids[ids.size, n_gen]
        cache_route = "source_history_direct_output"
        STDOUT << "\nPrompt cache output fast-forward hit before weight load: emitted #{output_ids.size} cached tokens\n"
        total_ms = (Time.instant - request_t0).total_milliseconds
        STDOUT << "  request summary: total_ms=#{total_ms.round(1)} model_load_ms=#{model_load_ms.round(1)} draft_load_ms=#{draft_load_ms.round(1)} tokenize_ms=#{tokenize_ms.round(1)} token_cache_hit=#{token_cache_hit} cache_route=#{cache_route} state_prepare_ms=#{state_prepare_ms.round(1)} source_history_lookup_ms=#{source_history_lookup_ms.round(1)} cache_restore_ms=#{cache_restore_ms.round(1)} prefill_ms=#{prefill_ms.round(1)} decode_ms=#{decode_ms.round(1)} source_history_save_ms=#{source_history_save_ms.round(1)} prompt_tokens=#{ids.size} output_tokens=#{output_ids.size}\n"

        puts "\n=== Generated token ids ==="
        puts output_ids.inspect
        generated_text = tok.decode(output_ids)
        puts "\n=== Generated text ==="
        puts generated_text
        puts "\n=== Full output ==="
        puts model_prompt + generated_text
        print_qwen_tool_calls_if_any(generated_text, chat_mode, tool_response_json_format, chat_tools)
        exit
      end
    end
  end
end

puts "Loading weights..."
t0 = Time.instant
w = ML::GGUF::Qwen35Weights.from_gguf(MODEL_PATH)
hp = w.hparams
model_load_ms += (Time.instant - t0).total_milliseconds
puts "Loaded weights in #{(model_load_ms / 1000.0).round(1)}s total. n_layer=#{hp.n_layer} n_embd=#{hp.n_embd} n_ff=#{hp.n_ff} vocab=#{w.output.out_dim}"

draft = nil.as(ML::GGUF::Qwen35Weights?)
if speculative_decode_enabled
  raise "draft model not found: #{draft_model_path}" unless File.exists?(draft_model_path)
  tstart = Time.instant
  draft = ML::GGUF::Qwen35Weights.from_gguf(draft_model_path)
  raise "target/draft vocab mismatch: #{w.output.out_dim} != #{draft.not_nil!.output.out_dim}" unless w.output.out_dim == draft.not_nil!.output.out_dim
  draft_load_ms = (Time.instant - tstart).total_milliseconds
  puts "Loaded draft in #{(draft_load_ms / 1000.0).round(1)}s. n_layer=#{draft.not_nil!.hparams.n_layer} n_embd=#{draft.not_nil!.hparams.n_embd}"
end

max_seq = ids.size + n_gen + 8
state_prepare_t0 = Time.instant
state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp) if prepare_state_metal
state_prepare_ms += (Time.instant - state_prepare_t0).total_milliseconds
ngram_source_history = [] of Int32
ngram_replay_cursor = nil.as(Int32?)
prompt_cache_reused = false
prompt_cache_fast_forward_used = false
tool_required_parameters = constrained_tool_call_prefix_enabled ? ML::GGUF::Qwen35Constraints.tool_required_parameters(chat_tools) : {} of String => Array(String)
tool_finite_value_options = constrained_tool_call_prefix_enabled ? ML::GGUF::Qwen35Constraints.tool_finite_parameter_value_options(chat_tools) : {} of String => Hash(String, Array(String))
literal_remaining = if prefix = constrained_literal_prefix
                      prefix.empty? ? [] of String : [prefix]
                    elsif constrained_tool_call_prefix_enabled
                      names = ML::GGUF::Qwen35Constraints.tool_function_names(chat_tools)
                      raise "QWEN35_CONSTRAINED_TOOL_CALL_PREFIX found no function names in QWEN35_TOOLS_JSON" if names.empty?
                      ML::GGUF::Qwen35Constraints.qwen_tool_call_prefix_options(names)
                    else
                      [] of String
                    end
literal_constrained_steps = 0
literal_emitted = ""
tool_value_text = ""
tool_literal_stage = constrained_tool_call_prefix_enabled ? "function_prefix" : "none"
tool_required_sequence = [] of String
tool_value_options_by_parameter = {} of String => Array(String)
tool_parameter_index = 0
tool_literal_stage_counts = Hash(String, Int32).new(0)
tool_freeform_value_steps = 0
tool_value_boundary_hits = 0
unless literal_remaining.empty?
  label = constrained_tool_call_prefix_enabled ? "tool-call prefix options=#{literal_remaining.size}" : constrained_literal_prefix.not_nil!.inspect
  STDOUT << "Constrained literal prefix enabled: #{label}\n"
end

pos = 0

if prompt_cache_enabled
  use_full_prompt_hit = n_gen >= prompt_cache_full_hit_min_gen
  max_prefix_len = if ids.empty?
                     0
                   elsif use_full_prompt_hit
                     ids.size
                   else
                     ids.size - 1
                   end

  if prompt_cache_source_history_enabled
    if source = source_history_hit
      replay_start = ids.size
      if source.token_ids.size > replay_start &&
         ML::GGUF::Qwen35PromptCache.source_history_prefix_match?(source.token_ids, ids, replay_start)
        source_history_hit = source
        source_remaining = source.token_ids.size - replay_start
        if source_remaining >= ngram_cache_min_remaining
          ngram_source_history = source.token_ids
          ngram_replay_cursor = replay_start
          STDOUT << "\nPrompt source-history hit: tokens=#{source.token_count} replay_start=#{replay_start} remaining=#{source_remaining}\n"
        else
          STDOUT << "\nPrompt source-history hit below cache n-gram minimum: remaining=#{source_remaining} min=#{ngram_cache_min_remaining}; exact fallback remains active\n"
        end
      else
        STDOUT << "\nPrompt source-history found but prefix did not validate; exact fallback remains active\n"
      end
    else
      STDOUT << "\nPrompt source-history miss (root=#{cache_root})\n"
    end
  end

  if prompt_cache_fast_forward_enabled && output_ids.empty? && (source = source_history_hit)
      source_remaining = source.token_ids.size - ids.size
    if source_remaining >= n_gen
      full_history_len = ids.size + n_gen
      cached_prefix_len = full_history_len - 1
      if fast_hit = cache_store.not_nil!.lookup_token_prefix(
           cache_model,
           cache_tokenizer,
           source.token_ids,
           cached_prefix_len)
        if ML::GGUF::Qwen35PromptCache.exact_known_span_entry_valid?(fast_hit, source.token_ids, n_gen, full_history_len)
          tstart = Time.instant
          reuse_state = fast_hit.max_seq == state.max_seq ? state : nil
          cached_output_ids = source.token_ids[ids.size, n_gen]
          route = ML::GGUF::Qwen35ServingRoute.serve_exact_cached_span(
            cache_store.not_nil!,
            w,
            cache_model,
            session_id,
            model_prompt,
            cached_output_ids,
            fast_hit,
            source.token_ids,
            full_history_len: full_history_len,
            continuation_required: false,
            turn_id: turn_id,
            reuse_state: reuse_state,
          )
          cache_restore_ms = (Time.instant - tstart).total_milliseconds
          if replay = route.replay
            state = replay.state
            pos = cached_prefix_len
          end
          output_ids = route.output_token_ids
          prompt_cache_reused = true
          prompt_cache_fast_forward_used = true
          cache_route = route.route
          STDOUT << "\nPrompt cache fast-forward hit: emitted #{output_ids.size} cached tokens, route=#{route.route}, reused_state_prefix=#{cached_prefix_len}, route took #{(cache_restore_ms / 1000.0).round(3)}s\n"
        else
          STDOUT << "\nPrompt cache fast-forward artifact failed validation; exact fallback remains active\n"
        end
      else
        STDOUT << "\nPrompt cache fast-forward state miss; exact fallback remains active\n"
      end
    else
      STDOUT << "\nPrompt cache fast-forward source span too short: remaining=#{source_remaining} requested=#{n_gen}; exact fallback remains active\n"
    end
  end

  if max_prefix_len > 0 && (hit = cache_store.not_nil!.lookup_longest_prefix(cache_model, cache_tokenizer, ids, max_prefix_len: max_prefix_len))
    unless prompt_cache_fast_forward_used
      tstart = Time.instant
      reuse_state = hit.max_seq == state.max_seq ? state : nil
      replay = cache_store.not_nil!.restore_and_replay_suffix(hit, w, ids, reuse_state: reuse_state)
      dt = (Time.instant - tstart).total_seconds
      cache_restore_ms += dt * 1000.0
      state = replay.state
      pos = ids.size
      if top = replay.next_token_id
        output_ids << top
        prompt_cache_reused = true
        cache_route = "prompt_state_restore"
        STDOUT << "\nPrompt cache hit: reused #{replay.reused_prefix_len}/#{ids.size} prompt tokens, replayed #{replay.replayed_tokens}, restore+replay took #{dt.round(3)}s\n"
      else
        STDOUT << "\nPrompt cache hit had no suffix logits; falling back to normal prefill\n"
        pos = 0
        state_prepare_t0 = Time.instant
        state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
        ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp) if prepare_state_metal
        state_prepare_ms += (Time.instant - state_prepare_t0).total_milliseconds
      end
    end
  else
    STDOUT << "\nPrompt cache miss (root=#{cache_root})\n" unless prompt_cache_fast_forward_used
  end
end

# Prefill known non-final prompt tokens through the shared helper, then run the
# final token for next-token logits. The recurrent chunk path is default-on;
# set `QWEN35_PREFILL_CHUNK_OFF=1` to force the older whole-token prefill loop.
if output_ids.empty?
  puts "\nPrefilling #{ids.size} tokens..."
  if !prompt_cache_enabled && ids.size > 1 && literal_remaining.empty?
    tstart = Time.instant
    top, top_logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(w, ids, pos, state)
    output_ids << top.to_i32
    dt = (Time.instant - tstart).total_seconds
    prefill_ms += dt * 1000.0
    STDOUT << "  chunked #{ids.size}/#{ids.size} tokens with final top1 took #{dt.round(2)}s\n"
    STDOUT.flush
    pos += ids.size
  elsif ids.size > 1
    prefix_ids = ids[0...-1]
    tstart = Time.instant
    ML::GGUF::Qwen35CPU.prefill_tokens(w, prefix_ids, pos, state)
    dt = (Time.instant - tstart).total_seconds
    prefill_ms += dt * 1000.0
    pos += prefix_ids.size
    STDOUT << "  prefix #{prefix_ids.size}/#{ids.size} tokens took #{dt.round(2)}s\n"
    STDOUT.flush

    if prompt_cache_enabled
      preview = ENV["QWEN35_PROMPT_CACHE_PREVIEW"]? == "1" ? tok.decode(prefix_ids) : nil
      saved = cache_store.not_nil!.save(
        session_id: ENV["QWEN35_SESSION_ID"]? || "default",
        turn_id: turn_id,
        model_id: cache_model,
        tokenizer_id: cache_tokenizer,
        prompt_text: "",
        token_ids: prefix_ids,
        state: state,
        prompt_preview: preview,
      )
      STDOUT << "  saved prompt-cache prefix #{prefix_ids.size} tokens sha=#{saved.artifact_sha256[0, 12]}\n"
    end
  end

  if output_ids.empty? && (final_id = ids.last?)
    tstart = Time.instant
    constrained_generated = false
    if literal_remaining.empty?
      top, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, final_id, pos, state)
    else
      constrained_stage = tool_literal_stage
      top, top_logit, literal_remaining, emitted_piece, constrained = advance_next_maybe_literal_constrained(
        w, tok, constraint_token_index.not_nil!, final_id, pos, state, literal_remaining)
      constrained_generated = constrained
      if constrained
        literal_constrained_steps += 1
        tool_literal_stage_counts[constrained_stage] += 1 if constrained_tool_call_prefix_enabled
        literal_emitted += emitted_piece
        if literal_remaining.empty? && constrained_tool_call_prefix_enabled
          tool_literal_stage, literal_remaining, tool_required_sequence, tool_parameter_index, tool_value_options_by_parameter, reset_tool_value = advance_tool_literal_stage(
            tool_literal_stage, literal_emitted, tool_required_parameters, tool_required_sequence, tool_parameter_index, tool_finite_value_options, tool_value_options_by_parameter)
          tool_value_text = "" if reset_tool_value
        end
      end
    end
    output_ids << top.to_i32
    generated_piece = tok.decode_single(top)
    if !constrained_generated && constrained_tool_call_prefix_enabled && literal_remaining.empty? && tool_literal_stage == "value"
      tool_freeform_value_steps += 1
      tool_value_text += generated_piece
      next_stage, next_literals, next_parameter_index, next_value_text = maybe_start_tool_value_close(
        tool_literal_stage, tool_value_text, tool_required_sequence, tool_parameter_index)
      if next_stage != tool_literal_stage || !next_literals.empty?
        tool_value_boundary_hits += 1
        tool_literal_stage = next_stage
        literal_remaining = next_literals
        tool_parameter_index = next_parameter_index
        tool_value_text = next_value_text
      end
    end
    dt = (Time.instant - tstart).total_seconds
    prefill_ms += dt * 1000.0
    STDOUT << "  final token #{ids.size}/#{ids.size} id=#{final_id} took #{dt.round(2)}s\n"
    STDOUT.flush
    pos += 1
    if prompt_cache_enabled
      preview = ENV["QWEN35_PROMPT_CACHE_PREVIEW"]? == "1" ? tok.decode(ids) : nil
      saved = cache_store.not_nil!.save(
        session_id: ENV["QWEN35_SESSION_ID"]? || "default",
        turn_id: turn_id,
        model_id: cache_model,
        tokenizer_id: cache_tokenizer,
        prompt_text: "",
        token_ids: ids,
        state: state,
        prompt_preview: preview,
        next_token_id: top.to_i32,
        next_token_logit: top_logit,
      )
      STDOUT << "  saved prompt-cache full #{ids.size} tokens sha=#{saved.artifact_sha256[0, 12]}\n"
    end
  end
end

# Decode loop
{% unless flag?(:cpu_only) %}
  if metal_profile_enabled
    ML::GGUF::Qwen35Metal::Profile.reset
    ML::GGUF::Qwen35Metal::Profile.enable!
  end
{% end %}

if output_ids.size >= n_gen
  STDOUT << "\nGeneration satisfied from validated cache; no decode loop needed.\n"
  decode_ms = 0.0
elsif speculative_decode_enabled && !output_ids.empty?
  puts "\nGenerating #{n_gen} tokens with exact neural speculative decode..."
  puts "  draft=#{draft_model_path}"
  puts "  gamma=#{spec_gamma} max_gamma=#{spec_max_gamma} bootstrap_gamma=#{spec_bootstrap_gamma} bootstrap_streak=#{spec_bootstrap_streak} fallback=#{spec_plain_fallback_enabled} fallback_gamma=#{spec_plain_fallback_gamma} full_accept_streak=#{spec_full_accept_streak} fast_regrow_min_gamma=#{spec_fast_regrow_min_gamma} single_fast=#{spec_single_fast_enabled} verify=#{spec_verify_mode}"

  decode_t0 = Time.instant
  target_next = output_ids.pop
  draft_weights = draft.not_nil!
  draft_state = ML::GGUF::Qwen35CPU::State.new(draft_weights.hparams, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(draft_state, draft_weights.hparams) if prepare_state_metal
  draft_next = prefill_next(draft_weights, ids, draft_state)
  target_backup_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
  draft_cycle_base = ML::GGUF::Qwen35CPU::State.new(draft_weights.hparams, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(target_backup_state, hp) if prepare_state_metal
  ML::GGUF::Qwen35CPU.prepare_state_metal!(draft_cycle_base, draft_weights.hparams) if prepare_state_metal

  current_gamma = spec_gamma
  full_accept_streak = 0
  adaptive_growth_allowed = true
  accepted = 0
  proposed = 0
  cycles = 0
  plain_fallback_steps = 0
  early_rejects = 0
  single_fast = 0
  target_verify_ms = 0.0
  draft_ms = 0.0
  draft_resync_ms = 0.0
  draft_backup_skips = 0
  draft_resync_skips = 0

  while output_ids.size < n_gen
    if spec_plain_fallback_enabled && !adaptive_growth_allowed && current_gamma <= spec_plain_fallback_gamma
      emitted = target_next
      tstart = Time.instant
      target_next = advance_next(w, emitted, pos, state)
      target_verify_ms += (Time.instant - tstart).total_milliseconds
      output_ids << emitted
      piece = tok.decode_single(emitted)
      if trace_steps
        STDOUT << "  gen #{output_ids.size}/#{n_gen} pos=#{pos} id=#{emitted} piece=#{piece.inspect} mode=target-fallback\n"
        STDOUT.flush
      end
      pos += 1
      plain_fallback_steps += 1
      break if emitted == tok.eos_id
      next
    end

    cycles += 1
    cycle_start_pos = pos
    cycle_gamma = Math.min(current_gamma, n_gen - output_ids.size)
    correction_or_accepted = [] of Int32
    candidates = [] of Int32
    rejected = false

    if draft_next != target_next
      will_plain_fallback_after_reject = spec_plain_fallback_enabled &&
                                         spec_skip_draft_before_fallback &&
                                         Math.max(1, current_gamma // 2) <= spec_plain_fallback_gamma
      emitted = target_next
      tstart = Time.instant
      target_next = advance_next(w, emitted, pos, state)
      target_verify_ms += (Time.instant - tstart).total_milliseconds
      output_ids << emitted
      correction_or_accepted << emitted
      proposed += 1
      pos += 1
      rejected = true
      early_rejects += 1
      draft_resync_skips += 1 if will_plain_fallback_after_reject
      if trace_steps
        STDOUT << "  spec cycle=#{cycles} early_reject emitted=1 gamma=#{current_gamma}\n"
        STDOUT.flush
      end
    elsif spec_single_fast_enabled && cycle_gamma == 1 && draft_next == target_next
      emitted = draft_next
      candidates << emitted
      output_ids << emitted
      correction_or_accepted << emitted
      accepted += 1
      proposed += 1

      tstart = Time.instant
      draft_next = advance_next(draft_weights, emitted, pos, draft_state)
      draft_ms += (Time.instant - tstart).total_milliseconds

      tstart = Time.instant
      target_next = advance_next(w, emitted, pos, state)
      target_verify_ms += (Time.instant - tstart).total_milliseconds

      pos += 1
      single_fast += 1
      if trace_steps
        STDOUT << "  spec cycle=#{cycles} single_fast emitted=1 gamma=#{current_gamma}\n"
        STDOUT.flush
      end
    else
      tstart = Time.instant
      skip_draft_backup_for_fallback = spec_plain_fallback_enabled &&
                                       spec_skip_draft_before_fallback &&
                                       spec_skip_draft_backup_before_fallback &&
                                       Math.max(1, current_gamma // 2) <= spec_plain_fallback_gamma
      unless skip_draft_backup_for_fallback
        draft_cycle_base.copy_from!(draft_state)
      else
        draft_backup_skips += 1
      end
      cycle_gamma.times do |i|
        candidates << draft_next
        draft_next = advance_next(draft_weights, draft_next, pos + i, draft_state)
      end
      draft_ms += (Time.instant - tstart).total_milliseconds
      proposed += candidates.size

      serial_verify = spec_verify_mode == "serial" || (spec_verify_mode == "hybrid" && cycles == 1)
      if serial_verify
        candidates.each do |cand|
          if cand == target_next
            output_ids << cand
            correction_or_accepted << cand
            accepted += 1
            tstart = Time.instant
            target_next = advance_next(w, cand, pos, state)
            target_verify_ms += (Time.instant - tstart).total_milliseconds
            pos += 1
            break if cand == tok.eos_id
          else
            expected = target_next
            output_ids << expected
            correction_or_accepted << expected
            tstart = Time.instant
            target_next = advance_next(w, expected, pos, state)
            target_verify_ms += (Time.instant - tstart).total_milliseconds
            pos += 1
            rejected = true
            break
          end
        end
      else
        target_backup_state.copy_from!(state)
        tstart = Time.instant
        target_nexts = with_guarded_full_rows_disabled do
          ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, candidates, cycle_start_pos, state)
        end
        target_verify_ms += (Time.instant - tstart).total_milliseconds

        expected = target_next
        candidates.each_with_index do |cand, i|
          if cand == expected
            output_ids << cand
            correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0]
            break if cand == tok.eos_id
          else
            output_ids << expected
            correction_or_accepted << expected
            rejected = true
            break
          end
        end

        if rejected
          state.copy_from!(target_backup_state)
          tstart = Time.instant
          corrected = with_guarded_full_rows_disabled do
            ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, correction_or_accepted, cycle_start_pos, state)
          end
          target_verify_ms += (Time.instant - tstart).total_milliseconds
          target_next = corrected[-1][0]
        else
          target_next = target_nexts[correction_or_accepted.size - 1][0]
        end
        pos += correction_or_accepted.size
      end

      if rejected
        will_plain_fallback_after_reject = spec_plain_fallback_enabled &&
                                           spec_skip_draft_before_fallback &&
                                           Math.max(1, current_gamma // 2) <= spec_plain_fallback_gamma
        if will_plain_fallback_after_reject || output_ids.size >= n_gen
          draft_resync_skips += 1
        else
          raise "draft backup missing before required resync" if skip_draft_backup_for_fallback
          tstart = Time.instant
          draft_next = resync_draft!(draft_weights, draft_state, draft_cycle_base, correction_or_accepted, cycle_start_pos)
          draft_resync_ms += (Time.instant - tstart).total_milliseconds
        end
      end

      if trace_steps
        STDOUT << "  spec cycle=#{cycles} accepted=#{accepted}/#{proposed} emitted=#{correction_or_accepted.size} gamma=#{current_gamma} rejected=#{rejected}\n"
        STDOUT.flush
      end
    end

    if rejected
      full_accept_streak = 0
      adaptive_growth_allowed = false
      current_gamma = Math.max(1, current_gamma // 2)
    elsif adaptive_growth_allowed && candidates.size == cycle_gamma && current_gamma < spec_max_gamma
      full_accept_streak += 1
      if spec_bootstrap_gamma > current_gamma && current_gamma == spec_gamma
        if full_accept_streak >= spec_bootstrap_streak
          current_gamma = Math.min(spec_max_gamma, spec_bootstrap_gamma)
          full_accept_streak = 0
        end
      else
        required = if spec_fast_regrow_min_gamma > 0 && current_gamma >= spec_fast_regrow_min_gamma
                     1
                   else
                     spec_full_accept_streak
                   end
        if full_accept_streak >= required
          current_gamma = Math.min(spec_max_gamma, current_gamma * 2)
          full_accept_streak = 0
        end
      end
    end

    break if output_ids.last? == tok.eos_id
  end

  decode_ms = (Time.instant - decode_t0).total_milliseconds
  rate = proposed > 0 ? (accepted.to_f64 * 100.0 / proposed.to_f64) : 0.0
  STDOUT << "  speculative summary: accepted=#{accepted}/#{proposed} rate=#{rate.round(2)}% cycles=#{cycles} fallback_steps=#{plain_fallback_steps} early_rejects=#{early_rejects} single_fast=#{single_fast} wall_ms=#{decode_ms.round(1)} ms_per_tok=#{(decode_ms / output_ids.size).round(2)} draft_ms=#{draft_ms.round(1)} target_ms=#{target_verify_ms.round(1)} draft_resync_ms=#{draft_resync_ms.round(1)} draft_backup_skips=#{draft_backup_skips} draft_resync_skips=#{draft_resync_skips}\n"
elsif ngram_decode_enabled && !output_ids.empty?
  puts "\nGenerating #{n_gen} tokens with exact n-gram speculative decode..."
  puts "  ngram gamma=#{ngram_gamma} min=#{ngram_min} max=#{ngram_max} min_candidates=#{ngram_min_candidates} cache_min_remaining=#{ngram_cache_min_remaining} stage_min=#{ngram_stage_min} stage_gate=#{ngram_stage_gate} risk_gate=#{ngram_risk_gate} risk_min_size=#{ngram_risk_min_size} corridor_gate=#{ngram_corridor_gate} corridor_min_size=#{ngram_corridor_min_size} corridor_match_len_min=#{ngram_corridor_match_len_min} corridor_lag4_min=#{ngram_corridor_lag4_min} corridor_lag8_min=#{ngram_corridor_lag8_min} corridor_entropy_max=#{ngram_corridor_entropy_max} recursive=#{ngram_recursive} disable_after_reject=#{ngram_disable_after_reject} replay_on_reject=#{ngram_replay_on_reject} index=#{ngram_index_enabled}"
  decode_t0 = Time.instant
  next_id = output_ids.pop
  history = ids.dup
  ngram_history = ngram_index_enabled ? ML::GGUF::NgramDraft::IndexedHistory.new(history, ngram_max, ngram_min) : nil
  ngram_disabled = decode_policy == "auto" && prompt_cache_reused && n_gen < ngram_cache_min_remaining
  if ngram_disabled
    STDOUT << "  ngram auto cache gate: requested=#{n_gen} min=#{ngram_cache_min_remaining}; using exact target fallback\n"
  end
  ngram_cycles = 0
  ngram_accepted = 0
  ngram_proposed = 0
  ngram_corridor_skips = 0
  ngram_cursor_hits = 0
  ngram_cursor_accepts = 0
  ngram_cursor_rejects = 0
  ngram_cursor_serial_advances = 0
  ngram_cursor_serial_drops = 0
  plain_steps = 0
  target_replay_ms = 0.0
  backup = nil.as(ML::GGUF::Qwen35CPU::State?)

  while output_ids.size < n_gen
    remaining = n_gen - output_ids.size
    ngram_pending_replay_cursor = nil.as(Int32?)
    ngram_from_source = false
    candidates = [] of Int32
    match_len = 0
    unless ngram_disabled
      if cursor = ngram_replay_cursor
        replay_count = Math.min(Math.min(ngram_gamma, remaining), ngram_source_history.size - cursor)
        if replay_count > 0
          candidates = ngram_source_history[cursor, replay_count]
          if ngram_min_candidates > 0 && candidates.size < ngram_min_candidates
            candidates = [] of Int32
          else
            ngram_pending_replay_cursor = cursor + candidates.size
            ngram_from_source = true
            ngram_cursor_hits += 1
            match_len = ngram_max
          end
        end
      end
      if candidates.empty?
        if index = ngram_history
          if span = index.candidate_span(Math.min(ngram_gamma, remaining),
               recursive: ngram_recursive, min_candidates: ngram_min_candidates)
            candidates = span.ids
            match_len = span.match_len
          else
            match_len = index.match_len
          end
        else
          candidates = ML::GGUF::NgramDraft.candidates(
            history, Math.min(ngram_gamma, remaining), ngram_max, ngram_min,
            recursive: ngram_recursive, min_candidates: ngram_min_candidates)
          match_len = ML::GGUF::NgramDraft.match_len(history, ngram_max, ngram_min)
        end
      end
    end
    if ngram_risk_gate && !ngram_from_source && ML::GGUF::NgramDraft.risky_candidate_shape?(candidates, ngram_risk_min_size, match_len)
      ngram_disabled = true
      candidates = [] of Int32
    end
    if ngram_corridor_gate && !ngram_from_source && !candidates.empty? &&
       !ML::GGUF::NgramDraft.corridor_candidate_shape?(candidates,
         match_len: match_len,
         min_size: ngram_corridor_min_size,
         match_len_min: ngram_corridor_match_len_min,
         lag4_min: ngram_corridor_lag4_min,
         lag8_min: ngram_corridor_lag8_min,
         entropy_max: ngram_corridor_entropy_max)
      ngram_corridor_skips += 1
      candidates = [] of Int32
    end

    if candidates.empty?
      tstart = Time.instant
      emitted = next_id
      output_ids << emitted
      history << emitted
      ngram_history.try &.append(emitted)
      if cursor = ngram_replay_cursor
        if cursor < ngram_source_history.size && ngram_source_history[cursor]? == emitted
          ngram_replay_cursor = cursor + 1
          ngram_cursor_serial_advances += 1
        else
          ngram_replay_cursor = nil
          ngram_cursor_serial_drops += 1
        end
      end
      if output_ids.size < n_gen && emitted != tok.eos_id
        top, _top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, emitted, pos, state)
        next_id = top
      end
      dt = (Time.instant - tstart).total_seconds
      piece = tok.decode_single(emitted)
      if trace_steps
        STDOUT << "  gen #{output_ids.size}/#{n_gen} pos=#{pos} id=#{emitted} piece=#{piece.inspect} mode=plain took #{dt.round(2)}s\n"
        STDOUT.flush
      end
      pos += 1
      plain_steps += 1
      break if emitted == tok.eos_id
      next
    end

    ngram_cycles += 1
    ngram_proposed += candidates.size
    unless ngram_replay_on_reject
      unless backup
        backup = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: max_seq)
        ML::GGUF::Qwen35CPU.prepare_state_metal!(backup.not_nil!, hp) if prepare_state_metal
      end
    end
    accepted_or_corrected = [] of Int32
    rejected = false
    stage_ngram = ngram_stage_min > 0 && candidates.size >= ngram_stage_min
    ngram_offset = 0
    tstart = Time.instant
    while ngram_offset < candidates.size && output_ids.size < n_gen
      remaining_stage = candidates.size - ngram_offset
      stage_len = stage_ngram ? Math.min(ngram_stage_gate, remaining_stage) : remaining_stage
      stage_candidates = candidates[ngram_offset, stage_len]
      final_stage = output_ids.size + stage_candidates.size >= n_gen
      verify_candidates = final_stage ? stage_candidates[0, stage_candidates.size - 1] : stage_candidates
      stage_pos = pos
      stage_accepted_or_corrected = [] of Int32

      backup.not_nil!.copy_from!(state) unless ngram_replay_on_reject
      target_nexts = verify_candidates.empty? ? [] of {Int32, Float32} : with_guarded_full_rows_disabled do
        ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, verify_candidates, stage_pos, state)
      end

      expected = next_id
      stage_candidates.each_with_index do |cand, i|
        break if output_ids.size >= n_gen
        if cand == expected
          output_ids << cand
          history << cand
          ngram_history.try &.append(cand)
          accepted_or_corrected << cand
          stage_accepted_or_corrected << cand
          ngram_accepted += 1
          expected = target_nexts[i][0] if i < target_nexts.size
          break if cand == tok.eos_id
        else
          output_ids << expected
          history << expected
          ngram_history.try &.append(expected)
          accepted_or_corrected << expected
          stage_accepted_or_corrected << expected
          rejected = true
          break
        end
      end

      if rejected
        ngram_disabled = true if ngram_disable_after_reject
        if ngram_pending_replay_cursor
          ngram_replay_cursor = nil
          ngram_cursor_rejects += 1
        end
        if output_ids.size < n_gen && output_ids.last? != tok.eos_id
          if ngram_replay_on_reject
            replay_t0 = Time.instant
            state, next_id = replay_target_state(w, ids, output_ids, max_seq, prepare_state_metal)
            target_replay_ms += (Time.instant - replay_t0).total_milliseconds
          else
            state.copy_from!(backup.not_nil!)
            corrected = with_guarded_full_rows_disabled do
              ML::GGUF::Qwen35CPU.prefill_tokens_top1s(w, stage_accepted_or_corrected, stage_pos, state)
            end
            next_id = corrected[-1][0]
          end
        end
        pos += stage_accepted_or_corrected.size
        break
      else
        pos += stage_accepted_or_corrected.size
        ngram_offset += stage_candidates.size
        next_id = target_nexts[stage_accepted_or_corrected.size - 1][0] if output_ids.size < n_gen && stage_accepted_or_corrected.size - 1 < target_nexts.size
        break if output_ids.last? == tok.eos_id
      end
    end
    if ngram_pending_replay_cursor && !rejected
      ngram_replay_cursor = ngram_pending_replay_cursor
      ngram_cursor_accepts += 1
    end
    dt = (Time.instant - tstart).total_seconds

    if trace_steps
      STDOUT << "  ngram cycle=#{ngram_cycles} accepted=#{ngram_accepted}/#{ngram_proposed} emitted=#{accepted_or_corrected.size} pos=#{pos} rejected=#{rejected} took=#{dt.round(2)}s\n"
      STDOUT.flush
    end
    break if output_ids.last? == tok.eos_id
  end

  decode_ms = (Time.instant - decode_t0).total_milliseconds
  rate = ngram_proposed > 0 ? (ngram_accepted.to_f64 * 100.0 / ngram_proposed.to_f64) : 0.0
  STDOUT << "  ngram summary: accepted=#{ngram_accepted}/#{ngram_proposed} rate=#{rate.round(2)}% cycles=#{ngram_cycles} plain_steps=#{plain_steps} disabled=#{ngram_disabled} corridor_skips=#{ngram_corridor_skips} cursor_hits=#{ngram_cursor_hits} cursor_accepts=#{ngram_cursor_accepts} cursor_rejects=#{ngram_cursor_rejects} cursor_serial_advances=#{ngram_cursor_serial_advances} cursor_serial_drops=#{ngram_cursor_serial_drops} wall_ms=#{decode_ms.round(1)} ms_per_tok=#{(decode_ms / output_ids.size).round(2)} target_replay_ms=#{target_replay_ms.round(1)}\n"
else
  puts "\nGenerating #{n_gen} tokens greedily..."
  decode_t0 = Time.instant
  (n_gen - 1).times do |g_i|
    break if constrained_tool_call_prefix_enabled && tool_literal_stage == "done"

    prev = output_ids.last
    tstart = Time.instant
    constrained_generated = false
    if literal_remaining.empty?
      top, top_logit = ML::GGUF::Qwen35CPU.forward_top1(w, prev, pos, state)
    else
      constrained_stage = tool_literal_stage
      top, top_logit, literal_remaining, emitted_piece, constrained = advance_next_maybe_literal_constrained(
        w, tok, constraint_token_index.not_nil!, prev, pos, state, literal_remaining)
      constrained_generated = constrained
      if constrained
        literal_constrained_steps += 1
        tool_literal_stage_counts[constrained_stage] += 1 if constrained_tool_call_prefix_enabled
        literal_emitted += emitted_piece
        if literal_remaining.empty? && constrained_tool_call_prefix_enabled
          tool_literal_stage, literal_remaining, tool_required_sequence, tool_parameter_index, tool_value_options_by_parameter, reset_tool_value = advance_tool_literal_stage(
            tool_literal_stage, literal_emitted, tool_required_parameters, tool_required_sequence, tool_parameter_index, tool_finite_value_options, tool_value_options_by_parameter)
          tool_value_text = "" if reset_tool_value
        end
      end
    end
    dt = (Time.instant - tstart).total_seconds
    piece = tok.decode_single(top)
    if !constrained_generated && constrained_tool_call_prefix_enabled && literal_remaining.empty? && tool_literal_stage == "value"
      tool_freeform_value_steps += 1
      tool_value_text += piece
      next_stage, next_literals, next_parameter_index, next_value_text = maybe_start_tool_value_close(
        tool_literal_stage, tool_value_text, tool_required_sequence, tool_parameter_index)
      if next_stage != tool_literal_stage || !next_literals.empty?
        tool_value_boundary_hits += 1
        tool_literal_stage = next_stage
        literal_remaining = next_literals
        tool_parameter_index = next_parameter_index
        tool_value_text = next_value_text
      end
    end
    if trace_steps
      STDOUT << "  gen #{g_i + 1}/#{n_gen} pos=#{pos} id=#{top} piece=#{piece.inspect} took #{dt.round(2)}s\n"
      STDOUT.flush
    end
    output_ids << top
    pos += 1
    break if constrained_tool_call_prefix_enabled && tool_literal_stage == "done"
    break if top == tok.eos_id
  end
  decode_ms = (Time.instant - decode_t0).total_milliseconds
  STDOUT << "  greedy summary: wall_ms=#{decode_ms.round(1)} ms_per_tok=#{(decode_ms / output_ids.size).round(2)} literal_constrained_steps=#{literal_constrained_steps}\n"
end

{% unless flag?(:cpu_only) %}
  if metal_profile_enabled
    ML::GGUF::Qwen35Metal::Profile.disable!
    STDOUT << ML::GGUF::Qwen35Metal::Profile.report_io
  end
{% end %}

if prompt_cache_enabled && prompt_cache_source_history_enabled && cache_store
  full_history = ids.dup
  full_history.concat(output_ids)
  output_text = tok.decode(output_ids)
  if prompt_cache_fast_forward_used
    STDOUT << "  skipped source-history save after validated fast-forward hit\n"
  else
    source_save_t0 = Time.instant
    exact_known_span_entry = nil.as(ML::GGUF::Qwen35PromptCache::Entry?)
    if prompt_cache_fast_forward_enabled && !output_ids.empty?
      cached_prefix = full_history[0, full_history.size - 1]
      exact_known_span_entry = cache_store.not_nil!.save(
        session_id: session_id,
        turn_id: turn_id,
        model_id: cache_model,
        tokenizer_id: cache_tokenizer,
        prompt_text: "",
        token_ids: cached_prefix,
        state: state,
        artifact_codec: prompt_cache_artifact_codec,
        artifact_codec_block: prompt_cache_artifact_codec ? prompt_cache_artifact_codec_block : nil,
        artifact_live_kv_tokens: prompt_cache_live_kv_artifacts ? cached_prefix.size : nil,
        artifact_validation_kind: ML::GGUF::Qwen35PromptCache::EXACT_KNOWN_SPAN_VALIDATION_KIND,
        artifact_validation_steps: output_ids.size,
        artifact_validation_hash: ML::GGUF::Qwen35PromptCache.token_hash(full_history),
        next_token_id: output_ids[-1],
      )
    end
    saved_source = cache_store.not_nil!.save_source_history(
      session_id: session_id,
      turn_id: turn_id,
      model_id: cache_model,
      tokenizer_id: cache_tokenizer,
      token_ids: full_history,
      generated_token_count: output_ids.size,
      generated_text: output_text,
    )
    if prompt_cache_fast_forward_enabled && (exact_entry = exact_known_span_entry) && output_text
      cache_store.not_nil!.save_output_fast_forward(
        session_id: session_id,
        turn_id: turn_id,
        model_id: cache_model,
        tokenizer_id: cache_tokenizer,
        prompt_text: model_prompt,
        prompt_token_ids: ids,
        output_token_ids: output_ids,
        generated_text: output_text.not_nil!,
        exact_entry: exact_entry,
      )
    end
    source_history_save_ms = (Time.instant - source_save_t0).total_milliseconds
    STDOUT << "  saved source-history tokens=#{saved_source.token_count} hash=#{saved_source.token_hash[0, 12]}\n"
  end
end

total_ms = (Time.instant - request_t0).total_milliseconds
if constrained_tool_call_prefix_enabled
  stage_steps = tool_literal_stage_counts.to_a.sort_by { |entry| entry[0] }.map { |stage, count| "#{stage}:#{count}" }.join(",")
  stage_steps = "none" if stage_steps.empty?
  finite_value_params = tool_value_options_by_parameter.size
  STDOUT << "  tool constraint summary: final_stage=#{tool_literal_stage} stage_steps=#{stage_steps} freeform_value_steps=#{tool_freeform_value_steps} value_boundary_hits=#{tool_value_boundary_hits} finite_value_params=#{finite_value_params}\n"
end
STDOUT << "  request summary: total_ms=#{total_ms.round(1)} model_load_ms=#{model_load_ms.round(1)} draft_load_ms=#{draft_load_ms.round(1)} tokenize_ms=#{tokenize_ms.round(1)} token_cache_hit=#{token_cache_hit} cache_route=#{cache_route} state_prepare_ms=#{state_prepare_ms.round(1)} source_history_lookup_ms=#{source_history_lookup_ms.round(1)} cache_restore_ms=#{cache_restore_ms.round(1)} prefill_ms=#{prefill_ms.round(1)} decode_ms=#{decode_ms.round(1)} source_history_save_ms=#{source_history_save_ms.round(1)} prompt_tokens=#{ids.size} output_tokens=#{output_ids.size}\n"

puts "\n=== Generated token ids ==="
puts output_ids.inspect
final_generated_text = output_text || tok.decode(output_ids)
puts "\n=== Generated text ==="
puts final_generated_text
puts "\n=== Full output ==="
puts model_prompt + final_generated_text
print_qwen_tool_calls_if_any(final_generated_text, chat_mode, tool_response_json_format, chat_tools)
