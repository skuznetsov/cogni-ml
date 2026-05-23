require "./qwen35_serving_route"

module ML::GGUF
  # Resident exact-cache session facade for Qwen serving code.
  #
  # It owns route attribution and an optional active continuation cursor. Shared
  # Store state is never returned by alias; only a cursor owned by this session
  # may be consumed without the reusable-cache copy.
  class Qwen35ResidentSession
    ACTIVE_CURSOR = "state_fast_forward_active_cursor"

    record Result,
      route : String,
      output_token_ids : Array(Int32),
      prompt_token_count : Int32,
      replay : Qwen35PromptCache::ReplayResult?

    private record ActiveCursor,
      key : String,
      replay : Qwen35PromptCache::ReplayResult,
      output_token_ids : Array(Int32),
      prompt_token_count : Int32

    @route_counts : Hash(String, Int64)
    @active_cursor : ActiveCursor?

    def initialize(@store : Qwen35PromptCache::Store,
                   @weights : Qwen35Weights,
                   @model_id : String,
                   @session_id : String,
                   @turn_id : String? = nil,
                   @prefer_metal : Bool = Qwen35Metal.available?)
      @route_counts = Hash(String, Int64).new(0_i64)
      @active_cursor = nil
    end

    def route_counts : Hash(String, Int64)
      @route_counts.dup
    end

    def route_count(route : String) : Int64
      @route_counts[route]
    end

    def route_summary : String
      @route_counts.keys.sort.map { |route| "#{route}=#{@route_counts[route]}" }.join(",")
    end

    def active_cursor? : Bool
      !@active_cursor.nil?
    end

    def clear_active_cursor : Nil
      @active_cursor = nil
    end

    def prewarm_continuation_cursor(prompt_text : String,
                                    output_token_ids : Array(Int32),
                                    exact_entry : Qwen35PromptCache::Entry,
                                    full_history_tokens : Array(Int32),
                                    full_history_len : Int32 = full_history_tokens.size,
                                    reuse_state : Qwen35CPU::State? = nil) : Result
      result = Qwen35ServingRoute.serve_exact_cached_span(
        @store,
        @weights,
        @model_id,
        @session_id,
        prompt_text,
        output_token_ids,
        exact_entry,
        full_history_tokens,
        full_history_len: full_history_len,
        continuation_required: true,
        turn_id: @turn_id,
        prefer_metal: @prefer_metal,
        reuse_state: reuse_state,
      )
      replay = result.replay
      raise "resident session prewarm did not restore continuation state" unless replay

      @active_cursor = ActiveCursor.new(
        cursor_key(output_token_ids, full_history_tokens, full_history_len),
        replay,
        result.output_token_ids.dup,
        result.prompt_token_count,
      )
      record_route(result.route)
      Result.new(result.route, result.output_token_ids, result.prompt_token_count, nil)
    end

    def serve_exact_cached_span(prompt_text : String,
                                output_token_ids : Array(Int32),
                                exact_entry : Qwen35PromptCache::Entry,
                                full_history_tokens : Array(Int32),
                                full_history_len : Int32 = full_history_tokens.size,
                                continuation_required : Bool = false,
                                reuse_state : Qwen35CPU::State? = nil) : Result
      if continuation_required
        key = cursor_key(output_token_ids, full_history_tokens, full_history_len)
        if cursor = @active_cursor
          if cursor.key == key
            @active_cursor = nil
            record_route(ACTIVE_CURSOR)
            return Result.new(ACTIVE_CURSOR, cursor.output_token_ids.dup, cursor.prompt_token_count, cursor.replay)
          end
        end
      end

      result = Qwen35ServingRoute.serve_exact_cached_span(
        @store,
        @weights,
        @model_id,
        @session_id,
        prompt_text,
        output_token_ids,
        exact_entry,
        full_history_tokens,
        full_history_len: full_history_len,
        continuation_required: continuation_required,
        turn_id: @turn_id,
        prefer_metal: @prefer_metal,
        reuse_state: reuse_state,
      )
      record_route(result.route)
      Result.new(result.route, result.output_token_ids, result.prompt_token_count, result.replay)
    end

    private def record_route(route : String) : Nil
      @route_counts[route] += 1_i64
    end

    private def cursor_key(output_token_ids : Array(Int32),
                           full_history_tokens : Array(Int32),
                           full_history_len : Int32) : String
      history_hash = Qwen35PromptCache.token_hash(full_history_tokens, full_history_len)
      output_hash = Qwen35PromptCache.token_hash(output_token_ids)
      "#{@model_id}\0#{@session_id}\0#{@turn_id}\0#{full_history_len}\0#{output_token_ids.size}\0#{history_hash}\0#{output_hash}"
    end
  end
end
