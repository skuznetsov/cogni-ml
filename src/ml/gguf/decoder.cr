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
require "../core/buffer"

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
      raise "dim must equal n_heads * head_dim (#{@dim} != #{@n_heads} * #{@head_dim})" unless @dim == @n_heads * @head_dim
      raise "n_heads must be divisible by n_kv_heads (#{@n_heads} % #{@n_kv_heads} != 0)" unless @n_heads % @n_kv_heads == 0
      raise "n_kv_heads must be <= n_heads" unless @n_kv_heads <= @n_heads
      raise "head_dim must be even for RoPE" unless @head_dim.even?
    end
  end

  # KV cache for one layer — pre-allocated GPU buffers
  class KVCache
    getter k_buf : MetalBuffer   # [max_seq, n_kv_heads * head_dim] float32
    getter v_buf : MetalBuffer   # [max_seq, n_kv_heads * head_dim] float32
    getter v_t_buf : MetalBuffer # [n_kv_heads, head_dim, max_seq] transposed for flash attn
    getter max_seq : Int32
    getter n_kv_heads : Int32
    getter head_dim : Int32
    property position : Int32 = 0

    def initialize(@max_seq : Int32, @n_kv_heads : Int32, @head_dim : Int32)
      kv_dim = @n_kv_heads * @head_dim
      @k_buf = MetalBuffer.new((@max_seq * kv_dim * 4).to_i64)
      @v_buf = MetalBuffer.new((@max_seq * kv_dim * 4).to_i64)
      @v_t_buf = MetalBuffer.new((@n_kv_heads * @head_dim * @max_seq * 4).to_i64)
    end

    def reset : Nil
      @position = 0
    end

    def kv_dim : Int32
      @n_kv_heads * @head_dim
    end

    # Append new K/V vectors at current position, including V transpose for flash attn
    def append(k_data : Array(Float32), v_data : Array(Float32)) : Nil
      kv_d = kv_dim
      raise "K/V size mismatch: #{k_data.size} vs #{v_data.size}" unless k_data.size == v_data.size
      raise "K data not divisible by kv_dim #{kv_d}" unless k_data.size % kv_d == 0
      n_tokens = k_data.size // kv_d
      raise "KV cache overflow: #{@position} + #{n_tokens} > #{@max_seq}" if @position + n_tokens > @max_seq

      byte_offset = (@position * kv_d * 4).to_i64
      @k_buf.write_bytes(k_data.to_unsafe.as(Pointer(UInt8)), k_data.size * 4, offset: byte_offset)
      @v_buf.write_bytes(v_data.to_unsafe.as(Pointer(UInt8)), v_data.size * 4, offset: byte_offset)

      # Transpose V into v_t_buf: [n_kv_heads, head_dim, max_seq]
      # v_data layout: [n_tokens, n_kv_heads * head_dim]
      # v_t layout:    [kv_head, d, seq_pos]
      n_tokens.times do |t|
        @n_kv_heads.times do |kh|
          @head_dim.times do |d|
            src_idx = t * kv_d + kh * @head_dim + d
            dst_idx = kh * @head_dim * @max_seq + d * @max_seq + (@position + t)
            # Write single float to transposed position
            val = v_data[src_idx]
            @v_t_buf.write_bytes(pointerof(val).as(Pointer(UInt8)), 4, offset: (dst_idx * 4).to_i64)
          end
        end
      end

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

  # CausalDecoder — autoregressive decoder on Metal GPU
  #
  # Forward pass stages (same pattern as NomicBertMoE encoder):
  #   embed → [rms_norm → qkv_proj → causal_attn → residual_norm →
  #            cross_attn? → residual_norm → ffn_up → ffn_down_norm] × N_layers
  #   → final_norm → lm_head → logits
  #
  # Two modes:
  #   prefill(tokens):     process full sequence, build KV cache  (batch GEMM)
  #   forward_one(token):  single token, extend KV cache          (vector GEMM)
  class CausalDecoder(B)
    CAUSAL_ATTN_SOURCE = {{ read_file("#{__DIR__}/kernels/attention_causal.metal") }}

    getter config : DecoderConfig
    getter workspace : DecoderWorkspace
    getter backend : B
    getter position : Int32 = 0

    # Layer weights (loaded from GGUF)
    @token_embd : Array(Float32) = [] of Float32
    @final_norm_w : Array(Float32) = [] of Float32
    @lm_head_w : Array(Float32) = [] of Float32  # [vocab_size, dim] — dense F32 (or tied to token_embd)
    @tie_word_embeddings : Bool = false
    @layers = [] of DecoderLayerWeights

    # GPU buffers for weights
    @gpu_token_embd : MetalBuffer?
    @gpu_final_norm_w : MetalBuffer?
    @gpu_lm_head_w : MetalBuffer?

    # RoPE or ALiBi tables
    @rope_cos = [] of Float32
    @rope_sin = [] of Float32

    struct DecoderLayerWeights
      # Self-attention
      getter q_proj : QuantWeight
      getter k_proj : QuantWeight
      getter v_proj : QuantWeight
      getter o_proj : QuantWeight
      getter attn_norm_w : Array(Float32)  # RMSNorm (no bias)

      # Cross-attention (optional — only every N layers)
      getter cross_q_proj : QuantWeight?
      getter cross_kv_proj : QuantWeight?
      getter cross_o_proj : QuantWeight?
      getter cross_norm_w : Array(Float32)?

      # FFN (SwiGLU: gate_proj * up_proj → down_proj)
      getter gate_proj : QuantWeight
      getter up_proj : QuantWeight
      getter down_proj : QuantWeight
      getter ffn_norm_w : Array(Float32)

      def initialize(
        @q_proj, @k_proj, @v_proj, @o_proj, @attn_norm_w,
        @gate_proj, @up_proj, @down_proj, @ffn_norm_w,
        @cross_q_proj = nil, @cross_kv_proj = nil, @cross_o_proj = nil, @cross_norm_w = nil
      )
      end

      def has_cross_attn? : Bool
        !@cross_q_proj.nil?
      end
    end

    def initialize(@config : DecoderConfig, @backend : B)
      @workspace = DecoderWorkspace.new(@config)
    end

    def reset : Nil
      @workspace.reset
      @position = 0
    end

    def set_facts(embeddings : Array(Float32), n_facts : Int32) : Nil
      @workspace.set_facts(embeddings, n_facts, @config.dim)
    end

    # ── Forward pass: prefill mode ──
    # Processes full sequence, builds KV cache for all layers
    def prefill(tokens : Array(Int32)) : Array(Float32)
      return [] of Float32 if tokens.empty?
      raise "prefill must start at position 0 (current: #{@position})" unless @position == 0

      seq_len = tokens.size
      dim = @config.dim
      n_heads = @config.n_heads
      n_kv_heads = @config.n_kv_heads
      head_dim = @config.head_dim
      scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)

      # 1. Token embedding lookup
      hidden = Array(Float32).new(seq_len * dim, 0.0_f32)
      tokens.each_with_index do |tid, pos|
        tid = tid.clamp(0, @config.vocab_size - 1)
        dim.times { |j| hidden[pos * dim + j] = @token_embd[tid * dim + j] }
      end

      # 2. Layer-by-layer processing
      @config.n_layers.times do |layer_idx|
        lw = @layers[layer_idx]
        kv_cache = @workspace.kv_caches[layer_idx]

        # a. RMSNorm → QKV projection
        normed = rms_norm(hidden, seq_len, dim, lw.attn_norm_w)
        q = matmul_cpu(normed, lw.q_proj, seq_len, dim, n_heads * head_dim)
        k = matmul_cpu(normed, lw.k_proj, seq_len, dim, n_kv_heads * head_dim)
        v = matmul_cpu(normed, lw.v_proj, seq_len, dim, n_kv_heads * head_dim)

        # Apply RoPE to Q and K
        apply_rope!(q, seq_len, n_heads, head_dim, @position)
        apply_rope!(k, seq_len, n_kv_heads, head_dim, @position)

        # Store K/V in cache
        kv_cache.append(k, v)

        # b. Causal self-attention (CPU reference — GPU version uses kernel)
        attn_out = causal_attention_cpu(q, k, v, seq_len, n_heads, n_kv_heads, head_dim, scale)

        # c. Output projection + residual
        proj = matmul_cpu(attn_out, lw.o_proj, seq_len, dim, dim)
        hidden.size.times { |i| hidden[i] += proj[i] }

        # d. Cross-attention to facts (every N layers)
        if lw.has_cross_attn? && (cross_k = @workspace.cross_k)
          cross_normed = rms_norm(hidden, seq_len, dim, lw.cross_norm_w.not_nil!)
          # Cross-attention Q from decoder, K/V from facts — NO positional encoding
          cq = matmul_cpu(cross_normed, lw.cross_q_proj.not_nil!, seq_len, dim, n_heads * head_dim)
          # Facts K/V already set in workspace
          # TODO: cross-attention computation
          # cross_out = cross_attention_cpu(cq, facts_k, facts_v, ...)
          # hidden += cross_o_proj @ cross_out
        end

        # e. FFN: RMSNorm → gate/up → SwiGLU → down → residual
        ffn_normed = rms_norm(hidden, seq_len, dim, lw.ffn_norm_w)
        gate = matmul_cpu(ffn_normed, lw.gate_proj, seq_len, dim, @config.ffn_dim)
        up = matmul_cpu(ffn_normed, lw.up_proj, seq_len, dim, @config.ffn_dim)

        # SwiGLU activation: gate * silu(up) — wait, it's silu(gate) * up
        ffn_act = Array(Float32).new(seq_len * @config.ffn_dim) do |i|
          g = gate[i]
          silu_g = g * (1.0_f32 / (1.0_f32 + Math.exp(-g.to_f64).to_f32))
          silu_g * up[i]
        end

        down = matmul_cpu(ffn_act, lw.down_proj, seq_len, @config.ffn_dim, dim)
        hidden.size.times { |i| hidden[i] += down[i] }
      end

      # 3. Final RMSNorm
      rms_norm!(hidden, seq_len, dim, @final_norm_w)

      # 4. LM head → logits (only for last token)
      last_hidden = hidden[(seq_len - 1) * dim, dim]
      logits = lm_head_matmul(last_hidden)

      @position += seq_len
      logits
    rescue ex
      Log.error { "decoder prefill failed: #{ex.message}" }
      [] of Float32
    end

    # ── Forward pass: decode mode (single token) ──
    def forward_one(token : Int32) : Array(Float32)
      dim = @config.dim
      n_heads = @config.n_heads
      n_kv_heads = @config.n_kv_heads
      head_dim = @config.head_dim
      scale = 1.0_f32 / Math.sqrt(head_dim.to_f32)

      # 1. Token embedding
      hidden = @token_embd[token.clamp(0, @config.vocab_size - 1) * dim, dim].dup

      # 2. Layers
      @config.n_layers.times do |layer_idx|
        lw = @layers[layer_idx]
        kv_cache = @workspace.kv_caches[layer_idx]

        normed = rms_norm_single(hidden, dim, lw.attn_norm_w)

        # Q for single token, K/V for single token → append to cache
        q = matmul_cpu([normed], lw.q_proj, 1, dim, n_heads * head_dim)
        k = matmul_cpu([normed], lw.k_proj, 1, dim, n_kv_heads * head_dim)
        v = matmul_cpu([normed], lw.v_proj, 1, dim, n_kv_heads * head_dim)

        apply_rope!(q, 1, n_heads, head_dim, @position)
        apply_rope!(k, 1, n_kv_heads, head_dim, @position)
        kv_cache.append(k, v)

        # Attend to full KV cache
        attn_out = decode_attention_cpu(q, kv_cache, n_heads, n_kv_heads, head_dim, scale)

        proj = matmul_cpu([attn_out], lw.o_proj, 1, dim, dim)
        dim.times { |i| hidden[i] += proj[i] }

        # Cross-attention (if applicable)
        # TODO: similar to prefill cross-attention

        # FFN
        ffn_normed = rms_norm_single(hidden, dim, lw.ffn_norm_w)
        gate = matmul_cpu([ffn_normed], lw.gate_proj, 1, dim, @config.ffn_dim)
        up = matmul_cpu([ffn_normed], lw.up_proj, 1, dim, @config.ffn_dim)

        ffn_act = Array(Float32).new(@config.ffn_dim) do |i|
          g = gate[i]
          silu_g = g * (1.0_f32 / (1.0_f32 + Math.exp(-g.to_f64).to_f32))
          silu_g * up[i]
        end

        down = matmul_cpu([ffn_act], lw.down_proj, 1, @config.ffn_dim, dim)
        dim.times { |i| hidden[i] += down[i] }
      end

      # Final norm → logits
      rms_norm_single!(hidden, dim, @final_norm_w)
      logits = lm_head_matmul(hidden)

      @position += 1
      logits
    end

    # ── Sampling ──
    def sample(logits : Array(Float32), temperature : Float32 = 0.7_f32, top_k : Int32 = 40) : Int32
      return logits.each_with_index.max_by { |v, _| v }[1] if temperature <= 0.01

      scaled = logits.map { |l| l / temperature }
      if top_k > 0 && top_k < scaled.size
        threshold = scaled.sort.reverse[top_k - 1]
        scaled.map! { |l| l >= threshold ? l : -1e30_f32 }
      end
      max_val = scaled.max
      exps = scaled.map { |l| Math.exp((l - max_val).to_f64) }
      sum = exps.sum
      r = Random.rand * sum
      cumulative = 0.0
      exps.each_with_index do |e, i|
        cumulative += e
        return i if cumulative >= r
      end
      exps.size - 1
    end

    # ── LM head: hidden → logits (dense F32 matmul, optionally tied to embeddings) ──
    private def lm_head_matmul(hidden : Array(Float32)) : Array(Float32)
      dim = @config.dim
      vocab = @config.vocab_size
      weights = @tie_word_embeddings ? @token_embd : @lm_head_w
      raise "LM head weights not loaded" if weights.empty?

      Array(Float32).new(vocab) do |v|
        sum = 0.0_f32
        dim.times { |d| sum += hidden[d] * weights[v * dim + d] }
        sum
      end
    end

    # ── CPU reference implementations (to be replaced with GPU) ──

    private def rms_norm(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32)) : Array(Float32)
      result = Array(Float32).new(x.size, 0.0_f32)
      n_pos.times do |pos|
        off = pos * dim
        ss = 0.0_f64
        dim.times { |j| ss += x[off + j].to_f64 ** 2 }
        inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + 1e-6)).to_f32
        dim.times { |j| result[off + j] = x[off + j] * inv_rms * w[j] }
      end
      result
    end

    private def rms_norm!(x : Array(Float32), n_pos : Int32, dim : Int32, w : Array(Float32)) : Nil
      n_pos.times do |pos|
        off = pos * dim
        ss = 0.0_f64
        dim.times { |j| ss += x[off + j].to_f64 ** 2 }
        inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + 1e-6)).to_f32
        dim.times { |j| x[off + j] = x[off + j] * inv_rms * w[j] }
      end
    end

    private def rms_norm_single(x : Array(Float32), dim : Int32, w : Array(Float32)) : Array(Float32)
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 ** 2 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + 1e-6)).to_f32
      Array(Float32).new(dim) { |j| x[j] * inv_rms * w[j] }
    end

    private def rms_norm_single!(x : Array(Float32), dim : Int32, w : Array(Float32)) : Nil
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 ** 2 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + 1e-6)).to_f32
      dim.times { |j| x[j] = x[j] * inv_rms * w[j] }
    end

    # CPU matmul: x [batch, in_dim] @ W [out_dim, in_dim]^T → [batch, out_dim]
    private def matmul_cpu(x : Array(Float32) | Array(Array(Float32)),
                           w : QuantWeight, batch : Int32, in_dim : Int32, out_dim : Int32) : Array(Float32)
      flat_x = x.is_a?(Array(Array(Float32))) ? x.flatten : x
      # Dequantize weights (TODO: use on-GPU dequant for production)
      weights = w.dequantize
      result = Array(Float32).new(batch * out_dim, 0.0_f32)
      batch.times do |b|
        out_dim.times do |o|
          sum = 0.0_f32
          in_dim.times do |i|
            sum += flat_x[b * in_dim + i] * weights[o * in_dim + i]
          end
          result[b * out_dim + o] = sum
        end
      end
      result
    end

    # RoPE: apply rotary position embedding to Q or K
    private def apply_rope!(x : Array(Float32), n_pos : Int32, n_heads : Int32,
                            head_dim : Int32, base_pos : Int32) : Nil
      half = head_dim // 2
      n_pos.times do |pos|
        abs_pos = base_pos + pos
        n_heads.times do |h|
          off = pos * n_heads * head_dim + h * head_dim
          half.times do |j|
            theta = abs_pos.to_f64 / (10000.0 ** (2.0 * j / head_dim))
            cos_t = Math.cos(theta).to_f32
            sin_t = Math.sin(theta).to_f32
            x0 = x[off + j]
            x1 = x[off + j + half]
            x[off + j] = x0 * cos_t - x1 * sin_t
            x[off + j + half] = x0 * sin_t + x1 * cos_t
          end
        end
      end
    end

    # Causal attention (CPU reference): Q [seq, n_heads, head_dim] × K,V
    private def causal_attention_cpu(q : Array(Float32), k : Array(Float32), v : Array(Float32),
                                     seq_len : Int32, n_heads : Int32, n_kv_heads : Int32,
                                     head_dim : Int32, scale : Float32) : Array(Float32)
      kv_repeat = n_heads // n_kv_heads  # GQA repeat factor
      out = Array(Float32).new(seq_len * n_heads * head_dim, 0.0_f32)

      seq_len.times do |i|
        n_heads.times do |h|
          kv_h = h // kv_repeat
          q_off = i * n_heads * head_dim + h * head_dim

          # Compute scores for positions 0..i (causal)
          max_score = -1e30_f32
          scores = Array(Float32).new(i + 1) do |j|
            k_off = j * n_kv_heads * head_dim + kv_h * head_dim
            dot = 0.0_f32
            head_dim.times { |d| dot += q[q_off + d] * k[k_off + d] }
            s = dot * scale
            max_score = s if s > max_score
            s
          end

          # Softmax
          sum = 0.0_f32
          scores.map! { |s| e = Math.exp((s - max_score).to_f64).to_f32; sum += e; e }

          # Weighted sum of V
          o_off = i * n_heads * head_dim + h * head_dim
          (i + 1).times do |j|
            v_off = j * n_kv_heads * head_dim + kv_h * head_dim
            w = scores[j] / sum
            head_dim.times { |d| out[o_off + d] += w * v[v_off + d] }
          end
        end
      end
      out
    end

    # Decode attention: single Q against full KV cache
    private def decode_attention_cpu(q : Array(Float32), cache : KVCache,
                                      n_heads : Int32, n_kv_heads : Int32,
                                      head_dim : Int32, scale : Float32) : Array(Float32)
      kv_repeat = n_heads // n_kv_heads
      cache_len = cache.position
      kv_dim = n_kv_heads * head_dim
      out = Array(Float32).new(n_heads * head_dim, 0.0_f32)

      # Read K/V from cache (CPU fallback — production uses GPU buffers directly)
      k_data = cache.k_buf.read(cache_len * kv_dim)
      v_data = cache.v_buf.read(cache_len * kv_dim)

      n_heads.times do |h|
        kv_h = h // kv_repeat
        q_off = h * head_dim

        max_score = -1e30_f32
        scores = Array(Float32).new(cache_len) do |j|
          k_off = j * kv_dim + kv_h * head_dim
          dot = 0.0_f32
          head_dim.times { |d| dot += q[q_off + d] * k_data[k_off + d] }
          s = dot * scale
          max_score = s if s > max_score
          s
        end

        sum = 0.0_f32
        scores.map! { |s| e = Math.exp((s - max_score).to_f64).to_f32; sum += e; e }

        cache_len.times do |j|
          v_off = j * kv_dim + kv_h * head_dim
          w = scores[j] / sum
          head_dim.times { |d| out[h * head_dim + d] += w * v_data[v_off + d] }
        end
      end
      out
    end
  end
end
