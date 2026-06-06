require "./gemma4_prompt_cache"

module ML::GGUF
  # Exact serving route for repeated Gemma session/cache hits.
  #
  # Direct output certificates are terminal output shortcuts. Gemma continuation
  # state replay is deliberately not implemented here yet; if continuation state
  # is required, callers must use normal decode until a validated state route is
  # added.
  module Gemma4ServingRoute
    extend self

    DIRECT_OUTPUT           = "direct_output"
    EXACT_METADATA_FALLBACK = "exact_metadata_direct_output_fallback"

    record Result,
      route : String,
      output_token_ids : Array(Int32),
      prompt_token_count : Int32

    def serve_exact_cached_span(store : Gemma4PromptCache::Store,
                                model_id : String,
                                session_id : String,
                                prompt_text : String,
                                output_token_ids : Array(Int32),
                                exact_entry : Gemma4PromptCache::Entry,
                                full_history_tokens : Array(Int32),
                                full_history_len : Int32 = full_history_tokens.size,
                                continuation_required : Bool = false,
                                turn_id : String? = nil) : Result
      raise ArgumentError.new("output_token_ids must not be empty") if output_token_ids.empty?
      raise ArgumentError.new("full_history_len out of range") if full_history_len <= 0 || full_history_len > full_history_tokens.size
      raise "Gemma4 cached continuation state route is not implemented" if continuation_required

      hit = store.lookup_output_fast_forward(
        model_id,
        session_id,
        prompt_text,
        output_token_ids.size,
        tokenizer_id: exact_entry.tokenizer_id,
        turn_id: turn_id,
      )
      if hit
        raise "direct output token mismatch" unless hit.output_token_ids == output_token_ids
        return Result.new(DIRECT_OUTPUT, hit.output_token_ids, hit.prompt_token_count)
      end

      unless Gemma4PromptCache.exact_known_span_entry_valid?(exact_entry, full_history_tokens, output_token_ids.size, full_history_len)
        raise "exact cached span validation mismatch"
      end

      prompt_token_count = full_history_len - output_token_ids.size
      raise "serving route output span underflows full history" if prompt_token_count < 0
      expected_output_ids = full_history_tokens[prompt_token_count, output_token_ids.size]
      raise "serving route source-history output mismatch" unless expected_output_ids == output_token_ids

      Result.new(EXACT_METADATA_FALLBACK, output_token_ids.dup, prompt_token_count)
    end
  end
end
