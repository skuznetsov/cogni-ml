# Crystal FFI bindings for llama.cpp
# Version: 7340 (Dec 2024)
# API docs: https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h

module ML
  module LLM
    # Link against llama library
    @[Link("llama")]
    lib LlamaFFI
      # Type aliases
      alias LlamaToken = Int32
      alias LlamaPos = Int32
      alias LlamaSeqId = Int32

      # Opaque struct pointers
      type LlamaModel = Void*
      type LlamaContext = Void*
      type LlamaSampler = Void*
      type LlamaVocab = Void*

      # Constants
      LLAMA_DEFAULT_SEED = 0xFFFFFFFF_u32
      LLAMA_TOKEN_NULL   = -1

      # Enums
      enum LlamaSplitMode
        None  = 0
        Layer = 1
        Row   = 2
      end

      enum LlamaRopeScalingType
        Unspecified = -1
        None        = 0
        Linear      = 1
        Yarn        = 2
        LongRope    = 3
      end

      enum LlamaPoolingType
        Unspecified = -1
        None        = 0
        Mean        = 1
        Cls         = 2
        Last        = 3
        Rank        = 4
      end

      enum LlamaAttentionType
        Unspecified = -1
        Causal      = 0
        NonCausal   = 1
      end

      enum LlamaFlashAttnType
        Auto     = -1
        Disabled = 0
        Enabled  = 1
      end

      # Batch structure for token processing
      struct LlamaBatch
        n_tokens : Int32
        token : LlamaToken*
        embd : Float32*
        pos : LlamaPos*
        n_seq_id : Int32*
        seq_id : LlamaSeqId**
        logits : Int8*
      end

      # Model parameters
      struct LlamaModelParams
        devices : Void*
        tensor_buft_overrides : Void*
        n_gpu_layers : Int32
        split_mode : LlamaSplitMode
        main_gpu : Int32
        tensor_split : Float32*
        progress_callback : Void*
        progress_callback_user_data : Void*
        kv_overrides : Void*
        vocab_only : Bool
        use_mmap : Bool
        use_mlock : Bool
        check_tensors : Bool
        use_extra_bufts : Bool
        no_host : Bool
      end

      # Context parameters
      struct LlamaContextParams
        n_ctx : UInt32
        n_batch : UInt32
        n_ubatch : UInt32
        n_seq_max : UInt32
        n_threads : Int32
        n_threads_batch : Int32
        rope_scaling_type : LlamaRopeScalingType
        pooling_type : LlamaPoolingType
        attention_type : LlamaAttentionType
        flash_attn_type : LlamaFlashAttnType
        rope_freq_base : Float32
        rope_freq_scale : Float32
        yarn_ext_factor : Float32
        yarn_attn_factor : Float32
        yarn_beta_fast : Float32
        yarn_beta_slow : Float32
        yarn_orig_ctx : UInt32
        defrag_thold : Float32
        cb_eval : Void*
        cb_eval_user_data : Void*
        type_k : Int32  # ggml_type
        type_v : Int32  # ggml_type
        abort_callback : Void*
        abort_callback_data : Void*
        embeddings : Bool
        offload_kqv : Bool
        no_perf : Bool
        op_offload : Bool
        swa_full : Bool
        kv_unified : Bool
      end

      # Sampler chain parameters
      struct LlamaSamplerChainParams
        no_perf : Bool
      end

      # Performance data
      struct LlamaPerfContextData
        t_start_ms : Float64
        t_load_ms : Float64
        t_p_eval_ms : Float64
        t_eval_ms : Float64
        n_p_eval : Int32
        n_eval : Int32
        n_reused : Int32
      end

      # Default parameters
      fun llama_model_default_params : LlamaModelParams
      fun llama_context_default_params : LlamaContextParams
      fun llama_sampler_chain_default_params : LlamaSamplerChainParams

      # Backend initialization
      fun llama_backend_init : Void
      fun llama_backend_free : Void

      # Model loading/freeing
      fun llama_model_load_from_file(path_model : LibC::Char*, params : LlamaModelParams) : LlamaModel
      fun llama_model_free(model : LlamaModel) : Void

      # Context creation/freeing
      fun llama_init_from_model(model : LlamaModel, params : LlamaContextParams) : LlamaContext
      fun llama_free(ctx : LlamaContext) : Void

      # Model info
      fun llama_model_n_ctx_train(model : LlamaModel) : Int32
      fun llama_model_n_embd(model : LlamaModel) : Int32
      fun llama_model_n_layer(model : LlamaModel) : Int32
      fun llama_model_n_head(model : LlamaModel) : Int32
      fun llama_model_size(model : LlamaModel) : UInt64
      fun llama_model_n_params(model : LlamaModel) : UInt64
      fun llama_model_desc(model : LlamaModel, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
      fun llama_model_get_vocab(model : LlamaModel) : LlamaVocab

      # Context info
      fun llama_n_ctx(ctx : LlamaContext) : UInt32
      fun llama_n_batch(ctx : LlamaContext) : UInt32
      fun llama_n_ubatch(ctx : LlamaContext) : UInt32
      fun llama_get_model(ctx : LlamaContext) : LlamaModel

      # Vocab info
      fun llama_vocab_n_tokens(vocab : LlamaVocab) : Int32
      fun llama_vocab_bos(vocab : LlamaVocab) : LlamaToken
      fun llama_vocab_eos(vocab : LlamaVocab) : LlamaToken
      fun llama_vocab_eot(vocab : LlamaVocab) : LlamaToken
      fun llama_vocab_nl(vocab : LlamaVocab) : LlamaToken
      fun llama_vocab_is_eog(vocab : LlamaVocab, token : LlamaToken) : Bool

      # Tokenization
      fun llama_tokenize(
        vocab : LlamaVocab,
        text : LibC::Char*,
        text_len : Int32,
        tokens : LlamaToken*,
        n_tokens_max : Int32,
        add_special : Bool,
        parse_special : Bool
      ) : Int32

      fun llama_token_to_piece(
        vocab : LlamaVocab,
        token : LlamaToken,
        buf : LibC::Char*,
        length : Int32,
        lstrip : Int32,
        special : Bool
      ) : Int32

      fun llama_detokenize(
        vocab : LlamaVocab,
        tokens : LlamaToken*,
        n_tokens : Int32,
        text : LibC::Char*,
        text_len_max : Int32,
        remove_special : Bool,
        unparse_special : Bool
      ) : Int32

      # Batch operations
      fun llama_batch_get_one(tokens : LlamaToken*, n_tokens : Int32) : LlamaBatch
      fun llama_batch_init(n_tokens : Int32, embd : Int32, n_seq_max : Int32) : LlamaBatch
      fun llama_batch_free(batch : LlamaBatch) : Void

      # Decoding
      fun llama_decode(ctx : LlamaContext, batch : LlamaBatch) : Int32
      fun llama_encode(ctx : LlamaContext, batch : LlamaBatch) : Int32
      fun llama_synchronize(ctx : LlamaContext) : Void

      # Output retrieval
      fun llama_get_logits(ctx : LlamaContext) : Float32*
      fun llama_get_logits_ith(ctx : LlamaContext, i : Int32) : Float32*
      fun llama_get_embeddings(ctx : LlamaContext) : Float32*
      fun llama_get_embeddings_ith(ctx : LlamaContext, i : Int32) : Float32*

      # Sampler chain
      fun llama_sampler_chain_init(params : LlamaSamplerChainParams) : LlamaSampler
      fun llama_sampler_chain_add(chain : LlamaSampler, smpl : LlamaSampler) : Void
      fun llama_sampler_chain_n(chain : LlamaSampler) : Int32
      fun llama_sampler_free(smpl : LlamaSampler) : Void

      # Built-in samplers
      fun llama_sampler_init_greedy : LlamaSampler
      fun llama_sampler_init_dist(seed : UInt32) : LlamaSampler
      fun llama_sampler_init_top_k(k : Int32) : LlamaSampler
      fun llama_sampler_init_top_p(p : Float32, min_keep : LibC::SizeT) : LlamaSampler
      fun llama_sampler_init_min_p(p : Float32, min_keep : LibC::SizeT) : LlamaSampler
      fun llama_sampler_init_temp(t : Float32) : LlamaSampler
      fun llama_sampler_init_temp_ext(t : Float32, delta : Float32, exponent : Float32) : LlamaSampler

      # Sampling
      fun llama_sampler_sample(smpl : LlamaSampler, ctx : LlamaContext, idx : Int32) : LlamaToken
      fun llama_sampler_accept(smpl : LlamaSampler, token : LlamaToken) : Void
      fun llama_sampler_reset(smpl : LlamaSampler) : Void

      # Threading
      fun llama_set_n_threads(ctx : LlamaContext, n_threads : Int32, n_threads_batch : Int32) : Void
      fun llama_n_threads(ctx : LlamaContext) : Int32

      # Performance
      fun llama_perf_context(ctx : LlamaContext) : LlamaPerfContextData
      fun llama_perf_context_print(ctx : LlamaContext) : Void
      fun llama_perf_context_reset(ctx : LlamaContext) : Void

      # System info
      fun llama_print_system_info : LibC::Char*

      # GPU support check
      fun llama_supports_gpu_offload : Bool
      fun llama_supports_mmap : Bool
      fun llama_supports_mlock : Bool
    end
  end
end
