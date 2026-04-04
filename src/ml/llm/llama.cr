# High-level Crystal wrapper for llama.cpp
# Provides idiomatic Crystal interface for LLM inference

require "./llama_ffi"

module ML
  module LLM
    @@backend_mutex = Mutex.new
    @@backend_initialized = false
    @@models = [] of Model
    @@contexts = [] of Context

    # Initialize llama backend (call once at program start)
    def self.init
      @@backend_mutex.synchronize do
        next if @@backend_initialized
        LlamaFFI.ggml_backend_load_all  # Load Metal/CUDA/etc backends (required since b8200)
        LlamaFFI.llama_backend_init
        @@backend_initialized = true
      end
    end

    # Cleanup llama backend (call at program end)
    def self.cleanup
      contexts = [] of Context
      models = [] of Model
      should_cleanup = false

      @@backend_mutex.synchronize do
        if @@backend_initialized
          should_cleanup = true
          contexts = @@contexts.reverse_each.to_a
          models = @@models.reverse_each.to_a
          @@contexts.clear
          @@models.clear
          @@backend_initialized = false
        end
      end

      return unless should_cleanup

      contexts.each(&.free)
      models.each(&.free)
      LlamaFFI.llama_backend_free
    end

    # Check if GPU offload is supported
    def self.gpu_available? : Bool
      LlamaFFI.llama_supports_gpu_offload
    end

    # Get system info string
    def self.system_info : String
      String.new(LlamaFFI.llama_print_system_info)
    end

    def self.register_model(model : Model) : Nil
      @@backend_mutex.synchronize { @@models << model }
    end

    def self.unregister_model(model : Model) : Nil
      @@backend_mutex.synchronize { @@models.reject!(&.same?(model)) }
    end

    def self.register_context(context : Context) : Nil
      @@backend_mutex.synchronize { @@contexts << context }
    end

    def self.unregister_context(context : Context) : Nil
      @@backend_mutex.synchronize { @@contexts.reject!(&.same?(context)) }
    end

    # LLM Model - loads and manages a GGUF model file
    class Model
      getter path : String
      getter n_ctx_train : Int32
      getter n_embd : Int32
      getter n_layers : Int32
      getter n_heads : Int32
      getter size_bytes : UInt64
      getter n_params : UInt64
      getter description : String

      @handle : LlamaFFI::LlamaModel
      @vocab : LlamaFFI::LlamaVocab
      @freed : Bool = false

      def initialize(@path : String, n_gpu_layers : Int32 = 99)
        params = LlamaFFI.llama_model_default_params
        params.n_gpu_layers = n_gpu_layers
        params.use_mmap = true

        @handle = LlamaFFI.llama_model_load_from_file(@path.to_unsafe, params)
        raise "Failed to load model: #{@path}" if @handle.null?

        @vocab = LlamaFFI.llama_model_get_vocab(@handle)
        @n_ctx_train = LlamaFFI.llama_model_n_ctx_train(@handle)
        @n_embd = LlamaFFI.llama_model_n_embd(@handle)
        @n_layers = LlamaFFI.llama_model_n_layer(@handle)
        @n_heads = LlamaFFI.llama_model_n_head(@handle)
        @size_bytes = LlamaFFI.llama_model_size(@handle)
        @n_params = LlamaFFI.llama_model_n_params(@handle)

        # Get model description
        buf = Bytes.new(256)
        len = LlamaFFI.llama_model_desc(@handle, buf.to_unsafe.as(LibC::Char*), buf.size)
        @description = len > 0 ? String.new(buf[0, len]) : "unknown"
        ML::LLM.register_model(self)
      end

      def finalize
        free
      end

      def free : Nil
        unless @handle.null? || @freed
          LlamaFFI.llama_model_free(@handle)
          @freed = true
          ML::LLM.unregister_model(self)
        end
      end

      # Vocab accessors
      def vocab_size : Int32
        LlamaFFI.llama_vocab_n_tokens(@vocab)
      end

      def bos_token : Int32
        LlamaFFI.llama_vocab_bos(@vocab)
      end

      def eos_token : Int32
        LlamaFFI.llama_vocab_eos(@vocab)
      end

      def eot_token : Int32
        LlamaFFI.llama_vocab_eot(@vocab)
      end

      def is_eog?(token : Int32) : Bool
        LlamaFFI.llama_vocab_is_eog(@vocab, token)
      end

      # Tokenize text to token IDs
      def tokenize(text : String, add_bos : Bool = true) : Array(Int32)
        max_tokens = text.bytesize + 32
        tokens = Array(Int32).new(max_tokens, 0)

        n = LlamaFFI.llama_tokenize(
          @vocab,
          text.to_unsafe,
          text.bytesize.to_i32,
          tokens.to_unsafe,
          max_tokens.to_i32,
          add_bos,
          false
        )

        if n < 0
          # Need more space
          max_tokens = -n + 16
          tokens = Array(Int32).new(max_tokens, 0)
          n = LlamaFFI.llama_tokenize(
            @vocab,
            text.to_unsafe,
            text.bytesize.to_i32,
            tokens.to_unsafe,
            max_tokens.to_i32,
            add_bos,
            false
          )
        end

        raise "Tokenization failed" if n < 0
        tokens[0, n]
      end

      # Convert token ID to text piece
      def token_to_piece(token : Int32) : String
        buf = Bytes.new(64)
        len = LlamaFFI.llama_token_to_piece(
          @vocab,
          token,
          buf.to_unsafe.as(LibC::Char*),
          buf.size.to_i32,
          0,
          false
        )
        return "" if len <= 0
        String.new(buf[0, len])
      end

      # Detokenize tokens to text
      def detokenize(tokens : Array(Int32)) : String
        max_len = tokens.size * 16
        buf = Bytes.new(max_len)

        len = LlamaFFI.llama_detokenize(
          @vocab,
          tokens.to_unsafe,
          tokens.size.to_i32,
          buf.to_unsafe.as(LibC::Char*),
          max_len.to_i32,
          false,
          false
        )

        return "" if len <= 0
        String.new(buf[0, len])
      end

      # Create inference context from this model
      def create_context(
        n_ctx : Int32 = 0,
        n_batch : Int32 = 512,
        n_ubatch : Int32 = 0,
        n_threads : Int32 = 0,
        flash_attn : Bool = true,
        embeddings : Bool = false
      ) : Context
        Context.new(self, n_ctx: n_ctx, n_batch: n_batch, n_ubatch: n_ubatch, n_threads: n_threads, flash_attn: flash_attn, embeddings: embeddings)
      end

      protected def handle : LlamaFFI::LlamaModel
        @handle
      end

      protected def vocab : LlamaFFI::LlamaVocab
        @vocab
      end
    end

    # Inference context - manages KV cache and generation state
    class Context
      getter model : Model
      getter n_ctx : UInt32
      getter n_batch : UInt32

      @handle : LlamaFFI::LlamaContext
      @sampler : LlamaFFI::LlamaSampler?
      @pos : Int32 = 0
      @freed : Bool = false

      def initialize(
        @model : Model,
        n_ctx : Int32 = 0,
        n_batch : Int32 = 512,
        n_ubatch : Int32 = 0,
        n_threads : Int32 = 0,
        flash_attn : Bool = true,
        embeddings : Bool = false
      )
        params = LlamaFFI.llama_context_default_params
        params.n_ctx = n_ctx > 0 ? n_ctx.to_u32 : @model.n_ctx_train.to_u32
        params.n_batch = n_batch.to_u32
        params.n_ubatch = n_ubatch > 0 ? n_ubatch.to_u32 : n_batch.to_u32
        params.n_threads = n_threads > 0 ? n_threads : System.cpu_count.to_i32
        params.n_threads_batch = params.n_threads
        params.embeddings = embeddings
        params.offload_kqv = true
        params.flash_attn_type = flash_attn ? LlamaFFI::LlamaFlashAttnType::Enabled : LlamaFFI::LlamaFlashAttnType::Disabled

        @handle = LlamaFFI.llama_init_from_model(@model.handle, params)
        raise "Failed to create context" if @handle.null?

        @n_ctx = LlamaFFI.llama_n_ctx(@handle)
        @n_batch = LlamaFFI.llama_n_batch(@handle)
        ML::LLM.register_context(self)
      end

      def finalize
        free
      end

      def free : Nil
        if sampler = @sampler
          LlamaFFI.llama_sampler_free(sampler)
          @sampler = nil
        end
        unless @handle.null? || @freed
          LlamaFFI.llama_free(@handle)
          @freed = true
          ML::LLM.unregister_context(self)
        end
      end

      # Setup sampler chain with default parameters
      def setup_sampler(
        temperature : Float32 = 0.8_f32,
        top_k : Int32 = 40,
        top_p : Float32 = 0.95_f32,
        min_p : Float32 = 0.05_f32,
        seed : UInt32 = LlamaFFI::LLAMA_DEFAULT_SEED
      )
        if sampler = @sampler
          LlamaFFI.llama_sampler_free(sampler)
        end

        sparams = LlamaFFI.llama_sampler_chain_default_params
        chain = LlamaFFI.llama_sampler_chain_init(sparams)

        LlamaFFI.llama_sampler_chain_add(chain, LlamaFFI.llama_sampler_init_top_k(top_k))
        LlamaFFI.llama_sampler_chain_add(chain, LlamaFFI.llama_sampler_init_top_p(top_p, 1))
        LlamaFFI.llama_sampler_chain_add(chain, LlamaFFI.llama_sampler_init_min_p(min_p, 1))
        LlamaFFI.llama_sampler_chain_add(chain, LlamaFFI.llama_sampler_init_temp(temperature))
        LlamaFFI.llama_sampler_chain_add(chain, LlamaFFI.llama_sampler_init_dist(seed))

        @sampler = chain
      end

      # Greedy sampler (deterministic)
      def setup_greedy_sampler
        if sampler = @sampler
          LlamaFFI.llama_sampler_free(sampler)
        end

        sparams = LlamaFFI.llama_sampler_chain_default_params
        chain = LlamaFFI.llama_sampler_chain_init(sparams)
        LlamaFFI.llama_sampler_chain_add(chain, LlamaFFI.llama_sampler_init_greedy)

        @sampler = chain
      end

      # Process prompt tokens (prefill)
      def eval(tokens : Array(Int32)) : Bool
        return true if tokens.empty?

        remaining = tokens.size
        offset = 0

        while remaining > 0
          n = Math.min(remaining, @n_batch.to_i32)
          batch_tokens = tokens[offset, n]

          batch = LlamaFFI.llama_batch_get_one(batch_tokens.to_unsafe, n)
          result = LlamaFFI.llama_decode(@handle, batch)

          return false if result != 0

          @pos += n
          offset += n
          remaining -= n
        end

        true
      end

      # Encode tokens for BERT/encoder models (uses llama_encode instead of llama_decode)
      def encode(tokens : Array(Int32)) : Bool
        return true if tokens.empty?

        remaining = tokens.size
        offset = 0

        while remaining > 0
          n = Math.min(remaining, @n_batch.to_i32)
          batch_tokens = tokens[offset, n]

          batch = LlamaFFI.llama_batch_get_one(batch_tokens.to_unsafe, n)
          result = LlamaFFI.llama_encode(@handle, batch)

          return false if result != 0

          @pos += n
          offset += n
          remaining -= n
        end

        true
      end

      # Get sequence embeddings (for BERT/encoder models with pooling)
      def get_seq_embeddings(seq_id : Int32) : Slice(Float32)
        ptr = LlamaFFI.llama_get_embeddings_seq(@handle, seq_id)
        raise "Failed to get sequence embeddings" if ptr.null?
        Slice.new(ptr, @model.n_embd)
      end

      # Sample next token
      def sample : Int32
        sampler = @sampler
        raise "Sampler not initialized. Call setup_sampler first." unless sampler

        LlamaFFI.llama_sampler_sample(sampler, @handle, -1)
      end

      # Get logits for last token
      def get_logits : Slice(Float32)
        ptr = LlamaFFI.llama_get_logits_ith(@handle, -1)
        raise "Failed to get logits" if ptr.null?
        Slice.new(ptr, @model.vocab_size)
      end

      # Get embeddings (requires embeddings=true in context)
      def get_embeddings : Slice(Float32)
        ptr = LlamaFFI.llama_get_embeddings_ith(@handle, -1)
        raise "Failed to get embeddings" if ptr.null?
        Slice.new(ptr, @model.n_embd)
      end

      # Reset position + sampler (safe for all context types including BERT)
      def reset
        @pos = 0
        if sampler = @sampler
          LlamaFFI.llama_sampler_reset(sampler)
        end
      end

      # Clear KV cache + recurrent state (decoder/hybrid models only, NOT BERT)
      def clear_memory
        LlamaFFI.llama_memory_clear(@handle) unless @freed
      end

      # ── KV Cache State Management ──

      # Save full context state (KV cache + logits) to file
      def state_save(path : String, tokens : Array(Int32)) : Bool
        LlamaFFI.llama_state_save_file(@handle, path.to_unsafe, tokens.to_unsafe, tokens.size.to_u64)
      end

      # Load full context state from file. Returns restored tokens.
      def state_load(path : String, max_tokens : Int32 = 131072) : Array(Int32)?
        tokens = Array(Int32).new(max_tokens, 0)
        n_loaded = uninitialized LibC::SizeT
        ok = LlamaFFI.llama_state_load_file(@handle, path.to_unsafe, tokens.to_unsafe, max_tokens.to_u64, pointerof(n_loaded))
        return nil unless ok
        @pos = n_loaded.to_i32
        tokens[0, n_loaded]
      end

      # Get state size in bytes (for allocation)
      def state_size : UInt64
        LlamaFFI.llama_state_get_size(@handle).to_u64
      end

      # Save per-sequence state to file (lighter than full state)
      def seq_state_save(path : String, seq_id : Int32, tokens : Array(Int32)) : Bool
        result = LlamaFFI.llama_state_seq_save_file(@handle, path.to_unsafe, seq_id, tokens.to_unsafe, tokens.size.to_u64)
        result > 0
      end

      # Load per-sequence state from file into a target sequence
      def seq_state_load(path : String, dest_seq_id : Int32, max_tokens : Int32 = 131072) : Array(Int32)?
        tokens = Array(Int32).new(max_tokens, 0)
        n_loaded = uninitialized LibC::SizeT
        result = LlamaFFI.llama_state_seq_load_file(@handle, path.to_unsafe, dest_seq_id, tokens.to_unsafe, max_tokens.to_u64, pointerof(n_loaded))
        return nil if result == 0
        tokens[0, n_loaded]
      end

      # Get per-sequence state size
      def seq_state_size(seq_id : Int32) : UInt64
        LlamaFFI.llama_state_seq_get_size(@handle, seq_id).to_u64
      end

      # ── KV Cache Sequence Operations ──

      # Remove tokens [p0, p1) from sequence. -1 for full range.
      def kv_seq_rm(seq_id : Int32, p0 : Int32 = -1, p1 : Int32 = -1) : Bool
        LlamaFFI.llama_kv_self_seq_rm(@handle, seq_id, p0, p1)
      end

      # Copy sequence src to dst in range [p0, p1)
      def kv_seq_cp(src_seq : Int32, dst_seq : Int32, p0 : Int32 = 0, p1 : Int32 = -1) : Nil
        LlamaFFI.llama_kv_self_seq_cp(@handle, src_seq, dst_seq, p0, p1)
      end

      # Keep only this sequence, remove all others
      def kv_seq_keep(seq_id : Int32) : Nil
        LlamaFFI.llama_kv_self_seq_keep(@handle, seq_id)
      end

      # Shift positions in sequence by delta
      def kv_seq_shift(seq_id : Int32, p0 : Int32, p1 : Int32, delta : Int32) : Nil
        LlamaFFI.llama_kv_self_seq_shift(@handle, seq_id, p0, p1, delta)
      end

      # Defragment KV cache (reclaim gaps)
      def kv_defrag : Nil
        LlamaFFI.llama_kv_self_defrag(@handle)
      end

      # Max position in sequence
      def kv_seq_pos_max(seq_id : Int32) : Int32
        LlamaFFI.llama_kv_self_seq_pos_max(@handle, seq_id)
      end

      # Number of used KV cells
      def kv_used_cells : Int32
        LlamaFFI.llama_kv_self_used_cells(@handle)
      end

      # Clear entire KV cache
      def kv_clear : Nil
        LlamaFFI.llama_kv_self_clear(@handle)
        @pos = 0
      end

      # Current position in context
      def position : Int32
        @pos
      end

      def position=(@pos : Int32)
      end

      # Remaining context space
      def remaining_ctx : Int32
        @n_ctx.to_i32 - @pos
      end

      # Get performance stats
      def perf_stats : LlamaFFI::LlamaPerfContextData
        LlamaFFI.llama_perf_context(@handle)
      end

      # Print performance stats
      def print_perf
        LlamaFFI.llama_perf_context_print(@handle)
      end

      # Reset performance counters
      def reset_perf
        LlamaFFI.llama_perf_context_reset(@handle)
      end

      protected def handle : LlamaFFI::LlamaContext
        @handle
      end
    end

    # Chat message for prompt formatting
    record ChatMessage, role : String, content : String

    # Prompt formatting modes
    enum PromptMode
      Raw        # No formatting, use prompt as-is
      ChatML     # Standard ChatML format
      GptOss     # gpt-oss format with channel control
      GptOssDirect # gpt-oss direct mode (skip reasoning)
    end

    # High-level text generation interface
    class Generator
      getter model : Model
      getter context : Context
      property prompt_mode : PromptMode

      # Per-token log-probabilities from last generation (log-softmax of sampled token)
      getter token_logprobs : Array(Float32) = [] of Float32

      def mean_logprob : Float32
        return 0.0_f32 if @token_logprobs.empty?
        @token_logprobs.sum / @token_logprobs.size
      end

      def min_logprob : Float32
        @token_logprobs.min? || 0.0_f32
      end

      def initialize(
        @model : Model,
        n_ctx : Int32 = 0,
        n_batch : Int32 = 512,
        n_ubatch : Int32 = 0,
        n_threads : Int32 = 0,
        flash_attn : Bool = true,
        @prompt_mode : PromptMode = PromptMode::Raw,
        sampler_seed : UInt32? = nil,
      )
        @seed = sampler_seed || LlamaFFI::LLAMA_DEFAULT_SEED
        @context = @model.create_context(n_ctx: n_ctx, n_batch: n_batch, n_ubatch: n_ubatch, n_threads: n_threads, flash_attn: flash_attn)
        @context.setup_sampler(seed: @seed)
      end

      # Format prompt based on current mode
      def format_prompt(user_message : String, system_message : String? = nil) : String
        case @prompt_mode
        when .raw?
          user_message
        when .chat_ml?
          build_chatml(user_message, system_message)
        when .gpt_oss?
          build_gpt_oss(user_message, system_message, direct: false)
        when .gpt_oss_direct?
          build_gpt_oss(user_message, system_message, direct: true)
        else
          user_message
        end
      end

      # Format with explicit messages
      def format_messages(messages : Array(ChatMessage)) : String
        case @prompt_mode
        when .raw?
          messages.map(&.content).join("\n")
        when .chat_ml?, .gpt_oss?, .gpt_oss_direct?
          String.build do |s|
            messages.each do |msg|
              s << "<|im_start|>#{msg.role}\n#{msg.content}<|im_end|>\n"
            end
            s << "<|im_start|>assistant\n"
            s << "<|channel|>final\n" if @prompt_mode.gpt_oss_direct?
          end
        else
          messages.map(&.content).join("\n")
        end
      end

      private def build_chatml(user : String, system : String?) : String
        String.build do |s|
          if sys = system
            s << "<|im_start|>system\n#{sys}<|im_end|>\n"
          end
          s << "<|im_start|>user\n#{user}<|im_end|>\n"
          s << "<|im_start|>assistant\n"
        end
      end

      private def build_gpt_oss(user : String, system : String?, direct : Bool) : String
        String.build do |s|
          if sys = system
            s << "<|im_start|>system\n#{sys}<|im_end|>\n"
          end
          s << "<|im_start|>user\n#{user}"
          s << "\nRespond directly without analysis." if direct
          s << "<|im_end|>\n"
          s << "<|im_start|>assistant\n"
          s << "<|channel|>final\n" if direct
        end
      end

      def finalize
        free
      end

      def free : Nil
        @context.free
      end

      # Default stop strings for gpt-oss models
      GPT_OSS_STOPS = ["<|end|>", "<|im_end|>"]

      # Ask a question and get response (formats prompt automatically)
      def ask(
        question : String,
        system : String? = nil,
        max_tokens : Int32 = 256,
        temperature : Float32 = 0.7_f32
      ) : String
        @context.reset
        @context.setup_sampler(temperature: temperature, seed: @seed)
        prompt = format_prompt(question, system)

        # Use stop strings for gpt-oss modes
        stops = @prompt_mode.gpt_oss? || @prompt_mode.gpt_oss_direct? ? GPT_OSS_STOPS : nil
        raw = generate(prompt, max_tokens: max_tokens, stop_strings: stops, temperature: temperature)

        # For GptOssDirect: output is already the answer (we prepended the marker)
        # For GptOss: need to extract content after <|channel|>final
        # For others: return as-is
        if @prompt_mode.gpt_oss_direct?
          raw.gsub(/<\|[^|]+\|>/, "").strip
        else
          extract_final_answer(raw)
        end
      end

      # Extract final answer from gpt-oss output (removes reasoning)
      private def extract_final_answer(text : String) : String
        # Look for <|channel|>final marker and extract content after it
        if idx = text.index("<|channel|>final")
          # Skip the marker and any following whitespace/newline
          start = idx + "<|channel|>final".size
          result = text[start..].lstrip
          # Remove any trailing special tokens
          result = result.gsub(/<\|[^|]+\|>/, "").strip
          result
        elsif idx = text.index("<|end|>")
          # If we hit <|end|> first, take content before it
          text[0...idx].strip
        else
          text.strip
        end
      end

      # Stream response to a question (filtered - only final answer for gpt-oss)
      def ask_stream(
        question : String,
        system : String? = nil,
        max_tokens : Int32 = 256,
        temperature : Float32 = 0.7_f32,
        &block : String ->
      ) : Int32
        @context.reset
        @context.setup_sampler(temperature: temperature, seed: @seed)
        prompt = format_prompt(question, system)

        # Use stop strings for gpt-oss modes
        stops = @prompt_mode.gpt_oss? || @prompt_mode.gpt_oss_direct? ? GPT_OSS_STOPS : nil

        # For GptOssDirect: we already prepended <|channel|>final, so output is already final
        # For GptOss: need to wait for and skip past <|channel|>final marker
        # For others: pass through directly
        buffer = IO::Memory.new
        in_final = !@prompt_mode.gpt_oss?  # Direct mode starts in final immediately
        final_marker = "<|channel|>final"
        tokens_streamed = 0

        stream(prompt, max_tokens: max_tokens, stop_strings: stops) do |piece|
          if in_final
            clean = piece.gsub(/<\|[^|]+\|>/, "")
            unless clean.empty?
              yield clean
              tokens_streamed += 1
            end
          else
            # Buffering until we see the final channel marker (GptOss mode only)
            buffer << piece
            content = buffer.to_s
            if idx = content.index(final_marker)
              in_final = true
              after = content[idx + final_marker.size..]
              clean = after.gsub(/<\|[^|]+\|>/, "").lstrip
              unless clean.empty?
                yield clean
                tokens_streamed += 1
              end
            end
          end
        end

        tokens_streamed
      end

      # Stream raw response (no filtering)
      def ask_stream_raw(
        question : String,
        system : String? = nil,
        max_tokens : Int32 = 256,
        temperature : Float32 = 0.7_f32,
        &block : String ->
      ) : Int32
        @context.reset
        @context.setup_sampler(temperature: temperature, seed: @seed)
        prompt = format_prompt(question, system)
        stream(prompt, max_tokens: max_tokens, &block)
      end

      # Generate text completion
      def generate(
        prompt : String,
        max_tokens : Int32 = 256,
        stop_on_eos : Bool = true,
        stop_strings : Array(String)? = nil,
        temperature : Float32 = 0.8_f32,
        top_k : Int32 = 40,
        top_p : Float32 = 0.95_f32
      ) : String
        @context.reset
        @token_logprobs.clear
        @context.setup_sampler(temperature: temperature, top_k: top_k, top_p: top_p, seed: @seed)

        tokens = @model.tokenize(prompt)
        return "" unless @context.eval(tokens)

        output = IO::Memory.new
        generated = 0
        recent_buffer = IO::Memory.new  # Buffer for stop string detection

        while generated < max_tokens
          token = @context.sample

          # Collect logprob for sampled token (logits valid between sample and next eval)
          @token_logprobs << token_log_softmax(token)

          # Check for end of generation
          break if stop_on_eos && @model.is_eog?(token)

          # Decode and append piece
          piece = @model.token_to_piece(token)
          output << piece

          # Check for stop strings
          if stops = stop_strings
            recent_buffer << piece
            recent = recent_buffer.to_s
            # Keep only recent characters for efficiency
            if recent.size > 100
              recent_buffer.clear
              recent_buffer << recent[-100..]
            end
            if stops.any? { |s| recent.includes?(s) }
              result = output.to_s
              stops.each { |s| result = result.gsub(s, "") }
              return result.strip
            end
          end

          # Eval the generated token
          break unless @context.eval([token])

          generated += 1
        end

        output.to_s
      end

      # Stream generation with callback
      def stream(
        prompt : String,
        max_tokens : Int32 = 256,
        stop_on_eos : Bool = true,
        stop_strings : Array(String)? = nil,
        &block : String ->
      ) : Int32
        @context.reset
        @context.reset_perf
        @token_logprobs.clear

        tokens = @model.tokenize(prompt)
        return 0 unless @context.eval(tokens)

        generated = 0
        recent_buffer = stops = stop_strings ? IO::Memory.new : nil

        while generated < max_tokens
          token = @context.sample

          # Collect logprob for sampled token (logits valid between sample and next eval)
          @token_logprobs << token_log_softmax(token)

          break if stop_on_eos && @model.is_eog?(token)

          piece = @model.token_to_piece(token)

          # Check for stop strings
          if (stops = stop_strings) && (buf = recent_buffer)
            buf << piece
            recent = buf.to_s
            if recent.size > 100
              buf.clear
              buf << recent[-100..]
            end
            break if stops.any? { |s| recent.includes?(s) }
          end

          yield piece

          break unless @context.eval([token])
          generated += 1
        end

        generated
      end

      # Get tokens per second from last generation
      # Note: llama.cpp perf stats may not populate correctly, use wall clock time instead
      def tokens_per_second : Float64
        stats = @context.perf_stats
        return 0.0 if stats.n_eval == 0 || stats.t_eval_ms <= 0
        (stats.n_eval * 1000.0) / stats.t_eval_ms
      end

      # Calculate tokens per second from wall clock time
      def calculate_tps(tokens : Int32, elapsed : Time::Span) : Float64
        return 0.0 if tokens == 0 || elapsed.total_milliseconds <= 0
        (tokens * 1000.0) / elapsed.total_milliseconds
      end

      # Numerically stable log-softmax for a single token
      # Returns log P(token) = logit[token] - log(Σ exp(logit - max))
      private def token_log_softmax(token_id : Int32) : Float32
        logits = @context.get_logits
        max_logit = logits[0]
        logits.each { |l| max_logit = l if l > max_logit }
        sum_exp = 0.0_f64
        logits.each { |l| sum_exp += Math.exp((l - max_logit).to_f64) }
        log_sum_exp = max_logit.to_f64 + Math.log(sum_exp)
        (logits[token_id].to_f64 - log_sum_exp).to_f32
      end
    end
  end
end
