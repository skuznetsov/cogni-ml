require "./qwen35_prompt_cache"
require "./qwen35_weights"

module ML::GGUF
  # Exact resident serving route for repeated Qwen session/cache hits.
  #
  # Direct output certificates are valid only for terminal cached spans. If the
  # caller needs continuation state, the route must restore the validated
  # recurrent/KV boundary instead of merely emitting cached token ids.
  module Qwen35ServingRoute
    extend self

    DIRECT_OUTPUT                    = "direct_output"
    STATE_FAST_FORWARD_CONTINUATION  = "state_fast_forward_continuation"
    STATE_FAST_FORWARD_FALLBACK      = "state_fast_forward_fallback"

    record Result,
      route : String,
      output_token_ids : Array(Int32),
      prompt_token_count : Int32,
      replay : Qwen35PromptCache::ReplayResult?

    def serve_exact_cached_span(store : Qwen35PromptCache::Store,
                                weights : Qwen35Weights,
                                model_id : String,
                                session_id : String,
                                prompt_text : String,
                                output_token_ids : Array(Int32),
                                exact_entry : Qwen35PromptCache::Entry,
                                full_history_tokens : Array(Int32),
                                continuation_required : Bool = false,
                                turn_id : String? = nil,
                                prefer_metal : Bool = Qwen35Metal.available?,
                                reuse_state : Qwen35CPU::State? = nil) : Result
      raise ArgumentError.new("output_token_ids must not be empty") if output_token_ids.empty?

      unless continuation_required
        hit = store.lookup_output_fast_forward(
          model_id,
          session_id,
          prompt_text,
          output_token_ids.size,
          turn_id: turn_id,
        )
        if hit
          raise "direct output token mismatch" unless hit.output_token_ids == output_token_ids
          return Result.new(DIRECT_OUTPUT, hit.output_token_ids, hit.prompt_token_count, nil)
        end
      end

      unless Qwen35PromptCache.exact_known_span_entry_valid?(exact_entry, full_history_tokens, output_token_ids.size)
        raise "exact cached span validation mismatch"
      end

      cached_prefix_tokens = full_history_tokens[0, exact_entry.prefix_len]
      replay = store.restore_and_replay_suffix(
        exact_entry,
        weights,
        cached_prefix_tokens,
        prefer_metal: prefer_metal,
        reuse_state: reuse_state,
      )
      raise "serving route restored prefix mismatch" unless replay.reused_prefix_len == cached_prefix_tokens.size
      raise "serving route replayed unexpected suffix" unless replay.replayed_tokens == 0
      raise "serving route restored next token mismatch" unless replay.next_token_id == output_token_ids[-1]

      route = continuation_required ? STATE_FAST_FORWARD_CONTINUATION : STATE_FAST_FORWARD_FALLBACK
      Result.new(route, output_token_ids.dup, full_history_tokens.size - output_token_ids.size, replay)
    end
  end
end
