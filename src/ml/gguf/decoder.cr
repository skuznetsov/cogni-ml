# CausalDecoder — autoregressive decoder with KV cache for token generation.
#
# Architecture:
#   - Causal self-attention (prefill + incremental decode)
#   - Optional cross-attention to external facts (Cogniformerus)
#   - GEMM via existing Q5_K/Q6_K Metal kernels
#   - KV cache: pre-allocated GPU buffers, grows incrementally
#
# Designed to pair with NomicBertMoE encoder:
#   encoder: text → [seq_len, 768] token representations (bidirectional)
#   decoder: token-by-token generation with KV cache + cross-attention to encoder/facts
#
# Usage:
#   decoder = CausalDecoder.new(config, backend)
#   # Prefill
#   decoder.prefill(input_tokens, facts_kv: encoded_facts)
#   # Generate
#   loop do
#     logits = decoder.forward_one(token)
#     next_token = decoder.sample(logits, temperature: 0.7)
#     break if next_token == eos_token
#   end

require "./compute"
require "./profile"

module ML::GGUF
  struct DecoderConfig
    getter dim : Int32            # hidden dimension (768 to match nomic BERT)
    getter n_heads : Int32        # attention heads (12)
    getter n_kv_heads : Int32     # KV heads for GQA (4 or 12)
    getter head_dim : Int32       # dim / n_heads (64)
    getter n_layers : Int32       # decoder layers (6-12)
    getter ffn_dim : Int32        # FFN intermediate (3072)
    getter vocab_size : Int32     # output vocabulary size
    getter max_seq_len : Int32    # maximum sequence length for KV cache
    getter use_cross_attn : Bool  # enable cross-attention to external facts
    getter cross_attn_every_n : Int32  # cross-attend every N layers (e.g. 2)

    def initialize(
      @dim = 768,
      @n_heads = 12,
      @n_kv_heads = 4,
      @head_dim = 64,
      @n_layers = 6,
      @ffn_dim = 3072,
      @vocab_size = 32000,
      @max_seq_len = 2048,
      @use_cross_attn = true,
      @cross_attn_every_n = 2
    )
    end
  end

  # KV cache for one layer — pre-allocated GPU buffers
  class KVCache
    getter k_buf : MetalBuffer   # [max_seq, n_kv_heads * head_dim] float32
    getter v_buf : MetalBuffer   # [max_seq, n_kv_heads * head_dim] float32
    getter v_t_buf : MetalBuffer # [n_kv_heads, head_dim, max_seq] transposed for flash attn
    property position : Int32 = 0

    def initialize(max_seq : Int32, n_kv_heads : Int32, head_dim : Int32)
      kv_dim = n_kv_heads * head_dim
      @k_buf = MetalBuffer.new((max_seq * kv_dim * 4).to_i64)
      @v_buf = MetalBuffer.new((max_seq * kv_dim * 4).to_i64)
      @v_t_buf = MetalBuffer.new((n_kv_heads * head_dim * max_seq * 4).to_i64)
    end

    def reset : Nil
      @position = 0
    end

    # Append new K/V vectors at current position
    def append(k_data : Array(Float32), v_data : Array(Float32), kv_dim : Int32) : Nil
      n_tokens = k_data.size // kv_dim
      byte_offset = (@position * kv_dim * 4).to_i64
      @k_buf.write_bytes((k_data.to_unsafe + 0).as(Pointer(UInt8)), k_data.size * 4, offset: byte_offset)
      @v_buf.write_bytes((v_data.to_unsafe + 0).as(Pointer(UInt8)), v_data.size * 4, offset: byte_offset)
      @position += n_tokens
    end
  end

  # Decoder workspace — pre-allocated GPU buffers for forward pass
  class DecoderWorkspace
    getter hidden : MetalBuffer      # [max_seq, dim]
    getter hidden2 : MetalBuffer     # [max_seq, dim] (double buffer)
    getter q_buf : MetalBuffer       # [n_heads, max_seq, head_dim]
    getter attn_out : MetalBuffer    # [max_seq, dim]
    getter ffn_mid : MetalBuffer     # [max_seq, ffn_dim]
    getter ffn_out : MetalBuffer     # [max_seq, dim]
    getter logits : MetalBuffer      # [max_seq, vocab_size] (only last token used in decode)
    getter kv_caches : Array(KVCache)

    # Cross-attention buffers (facts)
    getter cross_k : MetalBuffer?    # [n_facts, dim]
    getter cross_v : MetalBuffer?    # [n_facts, dim]
    getter cross_v_t : MetalBuffer?  # [n_heads, head_dim, n_facts]

    def initialize(config : DecoderConfig)
      max = config.max_seq_len
      dim = config.dim
      @hidden = MetalBuffer.new((max * dim * 4).to_i64)
      @hidden2 = MetalBuffer.new((max * dim * 4).to_i64)
      @q_buf = MetalBuffer.new((config.n_heads * max * config.head_dim * 4).to_i64)
      @attn_out = MetalBuffer.new((max * dim * 4).to_i64)
      @ffn_mid = MetalBuffer.new((max * config.ffn_dim * 4).to_i64)
      @ffn_out = MetalBuffer.new((max * dim * 4).to_i64)
      @logits = MetalBuffer.new((max * config.vocab_size * 4).to_i64)

      @kv_caches = Array(KVCache).new(config.n_layers) do
        KVCache.new(max, config.n_kv_heads, config.head_dim)
      end
    end

    def reset : Nil
      @kv_caches.each(&.reset)
    end

    # Set cross-attention facts (from Cogniformerus)
    def set_facts(fact_embeddings : Array(Float32), n_facts : Int32, dim : Int32) : Nil
      @cross_k = k = MetalBuffer.new((n_facts * dim * 4).to_i64)
      @cross_v = v = MetalBuffer.new((n_facts * dim * 4).to_i64)
      k.write(fact_embeddings)
      v.write(fact_embeddings)  # K=V for facts (content-based, no separate V projection yet)
    end
  end

  # Placeholder for the full decoder model
  # Will load weights from GGUF and run forward pass on Metal
  class CausalDecoder(B)
    getter config : DecoderConfig
    getter workspace : DecoderWorkspace
    getter backend : B
    @position : Int32 = 0

    def initialize(@config : DecoderConfig, @backend : B)
      @workspace = DecoderWorkspace.new(@config)
    end

    # Reset state for new sequence
    def reset : Nil
      @workspace.reset
      @position = 0
    end

    # Set external facts for cross-attention
    def set_facts(embeddings : Array(Float32), n_facts : Int32) : Nil
      @workspace.set_facts(embeddings, n_facts, @config.dim)
    end

    # Prefill: process multiple tokens at once (builds KV cache)
    def prefill(tokens : Array(Int32)) : Array(Float32)
      # TODO: implement full prefill forward pass
      #   1. Token embedding lookup
      #   2. For each layer:
      #      a. RMSNorm
      #      b. QKV projection (GEMM)
      #      c. Causal self-attention (attention_causal_prefill kernel)
      #      d. Residual + RMSNorm
      #      e. Cross-attention to facts (if enabled, every N layers)
      #      f. Residual + RMSNorm
      #      g. FFN (GEMM + activation + GEMM)
      #      h. Residual
      #   3. Final RMSNorm
      #   4. LM head projection → logits
      #   5. Return logits for last token
      @position = tokens.size
      [] of Float32  # placeholder
    end

    # Decode: process single token incrementally (extends KV cache)
    def forward_one(token : Int32) : Array(Float32)
      # TODO: implement incremental decode
      #   1. Token embedding lookup (single token)
      #   2. For each layer:
      #      a. RMSNorm
      #      b. Q projection (single token), K/V projection → append to KV cache
      #      c. Causal decode attention (attention_causal_decode kernel)
      #      d. Residual + RMSNorm
      #      e. Cross-attention to facts (attention_cross kernel)
      #      f. Residual + RMSNorm
      #      g. FFN
      #      h. Residual
      #   3. Final RMSNorm → LM head → logits
      @position += 1
      [] of Float32  # placeholder
    end

    # Sample next token from logits
    def sample(logits : Array(Float32), temperature : Float32 = 0.7_f32, top_k : Int32 = 40) : Int32
      return logits.each_with_index.max_by { |v, _| v }[1] if temperature <= 0.01

      # Temperature scaling
      scaled = logits.map { |l| l / temperature }

      # Top-k filtering
      if top_k > 0 && top_k < scaled.size
        threshold = scaled.sort.reverse[top_k - 1]
        scaled.map! { |l| l >= threshold ? l : -1e30_f32 }
      end

      # Softmax
      max_val = scaled.max
      exps = scaled.map { |l| Math.exp((l - max_val).to_f64) }
      sum = exps.sum
      probs = exps.map { |e| e / sum }

      # Multinomial sampling
      r = Random.rand
      cumulative = 0.0
      probs.each_with_index do |p, i|
        cumulative += p
        return i if cumulative >= r
      end
      probs.size - 1
    end
  end
end
