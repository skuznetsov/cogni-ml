# High-level Crystal wrapper for llama.cpp
# Provides idiomatic Crystal interface for LLM inference

require "./llama_ffi"

module ML
  module LLM
    # Initialize llama backend (call once at program start)
    def self.init
      LlamaFFI.llama_backend_init
    end

    # Cleanup llama backend (call at program end)
    def self.cleanup
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
      end

      def finalize
        LlamaFFI.llama_model_free(@handle) unless @handle.null?
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
      end

      def finalize
        if sampler = @sampler
          LlamaFFI.llama_sampler_free(sampler)
        end
        LlamaFFI.llama_free(@handle) unless @handle.null?
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

      # Reset KV cache
      def reset
        @pos = 0
        if sampler = @sampler
          LlamaFFI.llama_sampler_reset(sampler)
        end
      end

      # Current position in context
      def position : Int32
        @pos
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

      def initialize(
        @model : Model,
        n_ctx : Int32 = 0,
        n_batch : Int32 = 512,
        n_ubatch : Int32 = 0,
        n_threads : Int32 = 0,
        flash_attn : Bool = true,
        @prompt_mode : PromptMode = PromptMode::Raw
      )
        @context = @model.create_context(n_ctx: n_ctx, n_batch: n_batch, n_ubatch: n_ubatch, n_threads: n_threads, flash_attn: flash_attn)
        @context.setup_sampler
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
        # Context cleanup handled by its own finalizer
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
        @context.setup_sampler(temperature: temperature)
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
        @context.setup_sampler(temperature: temperature)
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
        @context.setup_sampler(temperature: temperature)
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
        @context.setup_sampler(temperature: temperature, top_k: top_k, top_p: top_p)

        tokens = @model.tokenize(prompt)
        return "" unless @context.eval(tokens)

        output = IO::Memory.new
        generated = 0
        recent_buffer = IO::Memory.new  # Buffer for stop string detection

        while generated < max_tokens
          token = @context.sample

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

        tokens = @model.tokenize(prompt)
        return 0 unless @context.eval(tokens)

        generated = 0
        recent_buffer = stops = stop_strings ? IO::Memory.new : nil

        while generated < max_tokens
          token = @context.sample
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
    end
  end
end
