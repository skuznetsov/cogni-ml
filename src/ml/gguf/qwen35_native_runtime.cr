require "digest/sha256"
require "./qwen35_engine_contract"
require "./reader"
require "./qwen35_chat"
require "./qwen35_cpu"
require "./qwen35_proposal_route"
require "./qwen35_tokenizer"
require "./qwen35_weights"

module ML::GGUF
  # First product-facing runtime for the Qwen35Engine contract.
  #
  # The runtime owns model resources and mutable decode state. State never
  # crosses the engine boundary; every request receives a fresh State under the
  # process-wide lock that also protects GGUF mmap registration and teardown.
  class Qwen35NativeRuntime < Qwen35Engine::Runtime
    NATIVE_PREWARM_SESSION_ID = "qwen35-native-prewarm"

    record PromptCacheStats,
      hits : Int64,
      misses : Int64,
      restore_failures : Int64,
      reused_prefix_tokens : Int64,
      replayed_suffix_tokens : Int64

    @@process_mutex = Mutex.new
    # Qwen35Metal keeps one process-global no-copy mmap registration. Until
    # that registration becomes model-keyed, two live native runtimes cannot
    # safely coexist: loading the second model would redirect the first
    # runtime's Metal weight slots. The slot is claimed before any mmap-backed
    # load and released only after a successful close.
    @@active_model_id : String? = nil

    getter model_path : String
    getter max_seq : Int32
    getter reasoning_effort_supported : Bool

    @model_id : String
    @tokenizer : Qwen35Tokenizer?
    @weights : Qwen35Weights?
    @prompt_cache : Qwen35PromptCache::Store?
    @prompt_cache_model_id : String?
    @prompt_cache_tokenizer_id : String?
    @prompt_cache_hits = 0_i64
    @prompt_cache_misses = 0_i64
    @prompt_cache_restore_failures = 0_i64
    @prompt_cache_reused_prefix_tokens = 0_i64
    @prompt_cache_replayed_suffix_tokens = 0_i64
    @reasoning_effort_supported = false
    @closed = false
    @llama_tokenize_bin : String

    def initialize(
      @model_path : String,
      @max_seq : Int32 = 4096,
      llama_tokenize_bin : String? = nil,
      prompt_cache_root : String? = nil,
      prompt_cache_resident_states : Int32? = nil,
    )
      raise ArgumentError.new("Qwen35NativeRuntime model path must not be empty") if @model_path.strip.empty?
      raise ArgumentError.new("Qwen35NativeRuntime model not found: #{@model_path}") unless File.exists?(@model_path)
      raise ArgumentError.new("Qwen35NativeRuntime max_seq must be positive") unless @max_seq > 0
      if resident_states = prompt_cache_resident_states
        raise ArgumentError.new("prompt_cache_resident_states must be non-negative") if resident_states < 0
      end

      @llama_tokenize_bin = llama_tokenize_bin || ENV["LLAMA_TOKENIZE_BIN"]? || ""
      @model_id = self.class.model_id_for(@model_path)
      @prompt_cache = nil
      @prompt_cache_model_id = nil
      @prompt_cache_tokenizer_id = nil
      effective_cache_root = prompt_cache_root
      if effective_cache_root.nil? && ENV["QWEN35_PROMPT_CACHE"]? == "1"
        effective_cache_root = ENV["QWEN35_PROMPT_CACHE_ROOT"]? || Qwen35PromptCache.default_root
      end
      if cache_root = effective_cache_root
        raise ArgumentError.new("prompt_cache_root must not be empty") if cache_root.strip.empty?
      end

      @@process_mutex.synchronize do
        if active_model_id = @@active_model_id
          raise Qwen35Engine::BackendMismatch.new(
            "another Qwen35NativeRuntime is active for #{active_model_id}; close it before loading #{@model_id}"
          )
        end

        @@active_model_id = @model_id
        begin
          tokenizer = load_tokenizer
          @tokenizer = tokenizer
          @reasoning_effort_supported = Qwen35Chat.supports_reasoning_effort?(tokenizer.chat_template)
          @weights = Qwen35Weights.from_gguf(@model_path)
          if cache_root = effective_cache_root
            @prompt_cache = Qwen35PromptCache::Store.new(
              cache_root,
              resident_state_cache_entries: prompt_cache_resident_states,
            )
            cache_model_id = Qwen35ProposalRoute.model_id(@model_path)
            @prompt_cache_model_id = cache_model_id
            @prompt_cache_tokenizer_id = Qwen35ProposalRoute.tokenizer_id(cache_model_id, tokenizer)
          end
        rescue ex
          # A failed constructor must not strand the process-wide ownership
          # slot. If weight loading completed before a later failure, release
          # the mmap registration before allowing another runtime to start.
          begin
            @weights.try(&.close)
          rescue
          end
          @weights = nil
          @tokenizer = nil
          @prompt_cache = nil
          @prompt_cache_model_id = nil
          @prompt_cache_tokenizer_id = nil
          @@active_model_id = nil
          raise ex
        end
      end
    end

    def self.model_id_for(path : String) : String
      info = File.info(path)
      digest = Digest::SHA256.hexdigest(
        "qwen35-native\0#{File.expand_path(path)}\0#{info.size}\0#{info.modification_time.to_unix}"
      )
      "qwen35-native:#{digest}"
    end

    # Pure backend-selection helper used by preflight and model-independent
    # tests. CUDA is intentionally guard-only until a real runtime can report it.
    def self.backend_identity_for(
      requested : Qwen35Engine::Backend,
      model_id : String,
      *,
      metal_available : Bool,
      decode_wave_forced_off : Bool,
    ) : Qwen35Engine::BackendIdentity
      if requested == Qwen35Engine::Backend::CUDA
        raise Qwen35Engine::BackendMismatch.new(
          "required CUDA, but Qwen35NativeRuntime has no CUDA adapter"
        )
      end

      # The decode-wave flag disables one fused route only. Attention,
      # recurrent, prefill, and output-head paths can still dispatch Metal, so
      # it must not be used as a proof of CPU-only execution.
      metal_selected = metal_available
      primary = case requested
                when Qwen35Engine::Backend::Metal
                  detail = metal_selected ? "execution attribution is unavailable" : "native Metal routing is unavailable"
                  raise Qwen35Engine::BackendMismatch.new(
                    "required Metal with observed attribution, but #{detail}"
                  )
                when Qwen35Engine::Backend::CPU
                  if metal_available
                    raise Qwen35Engine::BackendMismatch.new(
                      "required CPU is unavailable in a native Metal-capable build; use a cpu_only runtime"
                    )
                  end
                  Qwen35Engine::Backend::CPU
                else
                  metal_selected ? Qwen35Engine::Backend::Metal : Qwen35Engine::Backend::CPU
                end
      components = primary == Qwen35Engine::Backend::Metal ? [Qwen35Engine::Backend::Metal, Qwen35Engine::Backend::CPU] : [Qwen35Engine::Backend::CPU]
      Qwen35Engine::BackendIdentity.new(
        requested: requested,
        primary: primary,
        components: components,
        model_id: model_id,
        attribution: metal_selected ? Qwen35Engine::Attribution::Planned : Qwen35Engine::Attribution::Observed,
      )
    end

    # Resolve tokenizer output before any State is allocated. The caller may
    # supply a token id as a cross-check, but the runtime remains tokenizer-owning.
    def self.resolve_label_ids(
      labels : Array(Qwen35Engine::Label),
      encoded : Array(Array(Int32)),
    ) : Array(Int32)
      raise ArgumentError.new("label count does not match tokenizer results") unless labels.size == encoded.size

      ids = labels.map_with_index do |label, index|
        tokens = encoded[index]
        raise ArgumentError.new("label #{label.name.inspect} must resolve to exactly one token") unless tokens.size == 1

        token_id = tokens[0]
        raise ArgumentError.new("label #{label.name.inspect} resolved to a negative token id") if token_id < 0
        if expected = label.token_id
          raise ArgumentError.new("label #{label.name.inspect} token id mismatch") unless expected == token_id
        end
        token_id
      end
      raise ArgumentError.new("label scoring labels require unique token ids") unless ids.uniq.size == ids.size
      ids
    end

    def self.effective_max_seq(runtime_max_seq : Int32, request_max_seq : Int32?) : Int32
      raise ArgumentError.new("max_seq must be positive") unless runtime_max_seq > 0
      limit = request_max_seq || runtime_max_seq
      raise ArgumentError.new("max_seq must be positive") unless limit > 0
      raise ArgumentError.new("max_seq exceeds runtime capacity") if limit > runtime_max_seq
      limit
    end

    def self.validate_reasoning_effort_supported!(
      effort : Qwen35Engine::ReasoningEffort,
      supported : Bool,
    ) : Nil
      return if effort.none? || supported

      raise ArgumentError.new(
        "loaded tokenizer chat template does not support reasoning_effort=#{effort}"
      )
    end

    def preflight(
      operation : Qwen35Engine::Route,
      requested_backend : Qwen35Engine::Backend,
    ) : Qwen35Engine::PreflightRoute
      @@process_mutex.synchronize do
        ensure_open!
        backend = self.class.backend_identity_for(
          requested_backend,
          @model_id,
          metal_available: Qwen35Metal.available?,
          decode_wave_forced_off: decode_wave_forced_off?,
        )
        Qwen35Engine::PreflightRoute.new(operation, backend)
      end
    end

    # Persist an explicitly selected common chat prefix without a generation
    # prompt. Future requests whose rendered tokens extend this exact prefix can
    # restore it from disk and replay only their request-specific suffix.
    # Callers must not pass tenant- or user-private messages to a shared root.
    def prewarm_prefix(
      messages : Array(Qwen35Engine::Message),
      max_seq : Int32? = nil,
      requested_backend : Qwen35Engine::Backend = Qwen35Engine::Backend::Auto,
      reasoning_effort : Qwen35Engine::ReasoningEffort = Qwen35Engine::ReasoningEffort::None,
    ) : Qwen35PromptCache::Entry
      @@process_mutex.synchronize do
        ensure_open!
        raise ArgumentError.new("prewarm prefix requires at least one message") if messages.empty?
        messages.each do |message|
          raise ArgumentError.new("prewarm message role must not be empty") if message.role.strip.empty?
          raise ArgumentError.new("prewarm message content must not be empty") if message.content.strip.empty?
        end
        self.class.validate_reasoning_effort_supported!(reasoning_effort, @reasoning_effort_supported)

        cache, cache_model_id, cache_tokenizer_id = prompt_cache_resources
        tokenizer, weights = resources
        limit = self.class.effective_max_seq(@max_seq, max_seq)
        rendered = render_messages(
          messages,
          add_generation_prompt: false,
          reasoning_effort: reasoning_effort,
        )
        prefix_ids = tokenizer.encode(rendered, add_bos_override: false)
        raise ArgumentError.new("Qwen35NativeRuntime prewarm prefix is empty") if prefix_ids.empty?
        if prefix_ids.size >= limit
          raise ArgumentError.new("prewarm prefix must leave room within max_seq #{limit}")
        end

        backend = self.class.backend_identity_for(
          requested_backend,
          @model_id,
          metal_available: Qwen35Metal.available?,
          decode_wave_forced_off: decode_wave_forced_off?,
        )
        route = Qwen35Engine::PreflightRoute.new(Qwen35Engine::Route::GenerateGreedy, backend)
        state = Qwen35CPU::State.new(weights.hparams, max_seq: limit)
        prepare_state_metal!(state, weights, route)
        next_token, next_logit = Qwen35CPU.prefill_tokens_top1(weights, prefix_ids, 0, state)
        cache.save(
          session_id: NATIVE_PREWARM_SESSION_ID,
          model_id: cache_model_id,
          tokenizer_id: cache_tokenizer_id,
          prompt_text: rendered,
          token_ids: prefix_ids,
          state: state,
          prompt_preview: nil,
          artifact_live_kv_tokens: prefix_ids.size.to_i32,
          next_token_id: next_token,
          next_token_logit: next_logit,
        )
      end
    end

    def prompt_cache_stats : PromptCacheStats
      @@process_mutex.synchronize do
        PromptCacheStats.new(
          hits: @prompt_cache_hits,
          misses: @prompt_cache_misses,
          restore_failures: @prompt_cache_restore_failures,
          reused_prefix_tokens: @prompt_cache_reused_prefix_tokens,
          replayed_suffix_tokens: @prompt_cache_replayed_suffix_tokens,
        )
      end
    end

    def generate(
      request : Qwen35Engine::GenerateRequest,
      route : Qwen35Engine::PreflightRoute,
    ) : Qwen35Engine::GenerateResult
      @@process_mutex.synchronize do
        ensure_open!
        validate_route!(Qwen35Engine::Route::GenerateGreedy, route)
        validate_generate_request!(request)
        self.class.validate_reasoning_effort_supported!(request.reasoning_effort, @reasoning_effort_supported)
        tokenizer, weights = resources
        limit = self.class.effective_max_seq(@max_seq, request.max_seq)
        rendered = render_messages(
          request.messages,
          reasoning_effort: request.reasoning_effort,
        )
        prompt_ids = tokenizer.encode(rendered, add_bos_override: false)
        raise ArgumentError.new("Qwen35NativeRuntime generated prompt is empty") if prompt_ids.empty?
        if prompt_ids.size + request.max_tokens > limit
          raise ArgumentError.new("generation request exceeds max_seq #{limit}")
        end

        state = nil.as(Qwen35CPU::State?)
        next_token = nil.as(Int32?)
        if cache = @prompt_cache
          cache_model_id = @prompt_cache_model_id.not_nil!
          cache_tokenizer_id = @prompt_cache_tokenizer_id.not_nil!
          required_max_seq = (prompt_ids.size + request.max_tokens).to_i32
          hit = cache.lookup_longest_prefix(
            cache_model_id,
            cache_tokenizer_id,
            prompt_ids,
            required_max_seq: required_max_seq,
            maximum_max_seq: limit,
          )
          if hit && hit.prefix_len == prompt_ids.size && hit.next_token_id.nil? && prompt_ids.size > 1
            hit = cache.lookup_longest_prefix(
              cache_model_id,
              cache_tokenizer_id,
              prompt_ids,
              max_prefix_len: prompt_ids.size - 1,
              required_max_seq: required_max_seq,
              maximum_max_seq: limit,
            )
          end

          if cache_hit = hit
            begin
              replay = cache.restore_and_replay_suffix(
                cache_hit,
                weights,
                prompt_ids,
                prefer_metal: route.backend.primary.metal?,
              )
              if restored_next_token = replay.next_token_id
                state = replay.state
                next_token = restored_next_token
                @prompt_cache_hits += 1
                @prompt_cache_reused_prefix_tokens += replay.reused_prefix_len
                @prompt_cache_replayed_suffix_tokens += replay.replayed_tokens
              else
                @prompt_cache_misses += 1
              end
            rescue ex : ArgumentError | IO::Error
              @prompt_cache_restore_failures += 1
            end
          else
            @prompt_cache_misses += 1
          end
        end

        unless state && next_token
          state = Qwen35CPU::State.new(weights.hparams, max_seq: limit)
          prepare_state_metal!(state, weights, route)
          next_token, _logit = Qwen35CPU.prefill_tokens_top1(weights, prompt_ids, 0, state)
        end

        decode_state = state.not_nil!
        decode_token = next_token.not_nil!
        output_ids = [] of Int32
        pos = prompt_ids.size
        while output_ids.size < request.max_tokens
          break if stop_token?(tokenizer, decode_token)

          output_ids << decode_token
          break if output_ids.size >= request.max_tokens

          decode_token, _logit = Qwen35CPU.forward_top1(weights, decode_token, pos, decode_state)
          pos += 1
        end

        Qwen35Engine::GenerateResult.new(
          text: tokenizer.decode(output_ids),
          token_ids: output_ids,
          prompt_tokens: prompt_ids.size,
          completion_tokens: output_ids.size,
          backend: route.backend,
          route: route.operation,
          reasoning_effort: request.reasoning_effort,
        )
      end
    end

    def score_labels(
      request : Qwen35Engine::ScoreLabelsRequest,
      route : Qwen35Engine::PreflightRoute,
    ) : Qwen35Engine::ScoreLabelsResult
      @@process_mutex.synchronize do
        ensure_open!
        validate_route!(Qwen35Engine::Route::ScoreLabels, route)
        validate_score_labels_request!(request)
        tokenizer, weights = resources
        limit = self.class.effective_max_seq(@max_seq, request.max_seq)
        prompt_ids = tokenizer.encode(request.prompt, add_bos_override: false)
        raise ArgumentError.new("Qwen35NativeRuntime label-scoring prompt is empty") if prompt_ids.empty?
        raise ArgumentError.new("label-scoring prompt must leave room for one token") if prompt_ids.size >= limit

        raise ArgumentError.new("label scoring requires at least two labels") if request.labels.size < 2

        encoded = request.labels.map do |label|
          tokenizer.encode(label.text, add_bos_override: false)
        end
        label_ids = self.class.resolve_label_ids(request.labels, encoded)

        state = Qwen35CPU::State.new(weights.hparams, max_seq: limit)
        prepare_state_metal!(state, weights, route)
        if prompt_ids.size > 1
          Qwen35CPU.prefill_tokens(weights, prompt_ids[0...-1], 0, state)
        end
        logits = Qwen35CPU.forward(weights, prompt_ids[-1], prompt_ids.size - 1, state)
        best_index, second_index = top_two_indices(logits, label_ids)

        Qwen35Engine::ScoreLabelsResult.new(
          best: Qwen35Engine::LabelScore.new(request.labels[best_index], label_ids[best_index], logits[label_ids[best_index]]),
          second: Qwen35Engine::LabelScore.new(request.labels[second_index], label_ids[second_index], logits[label_ids[second_index]]),
          backend: route.backend,
          route: route.operation,
        )
      end
    end

    # Qwen35Weights#close unregisters the process-global mmap wrapper before
    # unmapping GGUF. Keep the runtime open if that cleanup raises so a caller
    # can retry, matching the Engine lifecycle contract.
    def close : Nil
      @@process_mutex.synchronize do
        return if @closed

        weights = @weights
        weights.try(&.close)
        @prompt_cache = nil
        @prompt_cache_model_id = nil
        @prompt_cache_tokenizer_id = nil
        @weights = nil
        @tokenizer = nil
        @closed = true
        @@active_model_id = nil if @@active_model_id == @model_id
      end
    end

    def finalize
      close
    end

    private def load_tokenizer : Qwen35Tokenizer
      gguf = GGUFFile.new(@model_path)
      begin
        Qwen35Tokenizer.from_gguf(gguf, @model_path, @llama_tokenize_bin)
      ensure
        gguf.close
      end
    end

    private def resources : {Qwen35Tokenizer, Qwen35Weights}
      tokenizer = @tokenizer
      weights = @weights
      raise Qwen35Engine::Closed.new("Qwen35NativeRuntime is closed") unless tokenizer && weights
      {tokenizer, weights}
    end

    private def prompt_cache_resources : {Qwen35PromptCache::Store, String, String}
      cache = @prompt_cache
      model_id = @prompt_cache_model_id
      tokenizer_id = @prompt_cache_tokenizer_id
      unless cache && model_id && tokenizer_id
        raise ArgumentError.new("Qwen35NativeRuntime prompt cache is disabled")
      end
      {cache, model_id, tokenizer_id}
    end

    private def prepare_state_metal!(state : Qwen35CPU::State,
                                     weights : Qwen35Weights,
                                     route : Qwen35Engine::PreflightRoute) : Nil
      # An observed CPU-only route must not allocate Metal state. Auto on a
      # Metal-capable build remains an explicitly planned hybrid even when the
      # fused decode wave is disabled, because lower-level paths can still use
      # Metal.
      return unless route.backend.primary.metal?
      Qwen35CPU.prepare_state_metal!(state, weights.hparams)
    end

    private def validate_generate_request!(request : Qwen35Engine::GenerateRequest) : Nil
      raise ArgumentError.new("generation requires at least one message") if request.messages.empty?
      raise ArgumentError.new("generation max_tokens must be positive") unless request.max_tokens > 0
      raise ArgumentError.new("only deterministic temperature=0 generation is admitted") unless request.temperature == 0.0
      raise ArgumentError.new("generation requires non-empty message content") if request.messages.all? { |message| message.content.strip.empty? }
      request.messages.each do |message|
        raise ArgumentError.new("message role must not be empty") if message.role.strip.empty?
      end
    end

    private def validate_score_labels_request!(request : Qwen35Engine::ScoreLabelsRequest) : Nil
      raise ArgumentError.new("label scoring prompt must not be empty") if request.prompt.strip.empty?
      raise ArgumentError.new("label scoring requires at least two labels") if request.labels.size < 2

      names = request.labels.map(&.name.strip)
      texts = request.labels.map(&.text.strip)
      raise ArgumentError.new("label scoring name must not be empty") if names.any?(&.empty?)
      raise ArgumentError.new("label scoring text must not be empty") if texts.any?(&.empty?)
      raise ArgumentError.new("label scoring names require uniqueness") unless names.uniq.size == names.size
      raise ArgumentError.new("label scoring texts require uniqueness") unless texts.uniq.size == texts.size

      token_ids = request.labels.compact_map(&.token_id)
      unless token_ids.empty? || token_ids.size == request.labels.size
        raise ArgumentError.new("label scoring labels must either all provide token ids or none")
      end
      raise ArgumentError.new("label scoring labels require non-negative token ids") if token_ids.any? { |id| id < 0 }
      raise ArgumentError.new("label scoring labels require unique token ids") unless token_ids.uniq.size == token_ids.size
    end

    private def ensure_open! : Nil
      raise Qwen35Engine::Closed.new("Qwen35NativeRuntime is closed") if @closed
    end

    private def validate_route!(operation : Qwen35Engine::Route, route : Qwen35Engine::PreflightRoute) : Nil
      raise Qwen35Engine::RouteMismatch.new("runtime received #{route.operation}, expected #{operation}") unless route.operation == operation
      unless route.backend.model_id == @model_id
        raise Qwen35Engine::RouteMismatch.new("runtime route model identity drifted")
      end
      expected = self.class.backend_identity_for(
        route.backend.requested,
        @model_id,
        metal_available: Qwen35Metal.available?,
        decode_wave_forced_off: decode_wave_forced_off?,
      )
      raise Qwen35Engine::RouteMismatch.new("runtime route backend identity drifted") unless expected == route.backend
    end

    private def render_messages(
      messages : Array(Qwen35Engine::Message),
      add_generation_prompt : Bool = true,
      reasoning_effort : Qwen35Engine::ReasoningEffort = Qwen35Engine::ReasoningEffort::None,
    ) : String
      qwen_messages = messages.map { |message| Qwen35Chat::Message.new(message.role, message.content) }
      Qwen35Chat.render(
        qwen_messages,
        add_generation_prompt: add_generation_prompt,
        reasoning_effort: reasoning_effort,
      )
    end

    private def stop_token?(tokenizer : Qwen35Tokenizer, token_id : Int32) : Bool
      token_id == tokenizer.eos_id || token_id == tokenizer.pad_id ||
        tokenizer.token_to_id["<|im_end|>"]? == token_id
    end

    private def top_two_indices(logits : Array(Float32), allowed_ids : Array(Int32)) : {Int32, Int32}
      raise ArgumentError.new("label scoring requires at least two labels") if allowed_ids.size < 2
      best_index = 0
      second_index = 1
      allowed_ids.each_with_index do |token_id, index|
        raise ArgumentError.new("label token id #{token_id} is out of vocabulary") if token_id < 0 || token_id >= logits.size
        if index == 0
          next
        elsif index == 1
          if better_token?(logits[token_id], token_id, logits[allowed_ids[best_index]], allowed_ids[best_index])
            second_index = best_index
            best_index = index
          end
          next
        end

        if better_token?(logits[token_id], token_id, logits[allowed_ids[best_index]], allowed_ids[best_index])
          second_index = best_index
          best_index = index
        elsif better_token?(logits[token_id], token_id, logits[allowed_ids[second_index]], allowed_ids[second_index])
          second_index = index
        end
      end
      {best_index, second_index}
    end

    private def better_token?(logit : Float32, token_id : Int32, other_logit : Float32, other_token_id : Int32) : Bool
      logit > other_logit || (logit == other_logit && token_id < other_token_id)
    end

    private def decode_wave_forced_off? : Bool
      ENV["QWEN35_DECODE_WAVE_OFF"]? == "1"
    end
  end
end
