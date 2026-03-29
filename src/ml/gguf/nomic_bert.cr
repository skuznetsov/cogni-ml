# NomicBertMoE — inference-only nomic-embed-text-v2-moe from GGUF.
#
# Architecture: 12-block BERT encoder with MoE FFN every 2 layers.
# Parameterized on ComputeBackend (B) for precision control:
#   F32Backend     — CPU Float32 (default)
#   F16SimBackend  — CPU with FP16 truncation (matches GPU)
#   MetalBackend   — GPU (future)
#
# Usage:
#   model = ML::GGUF::NomicBertMoE.from_gguf("model.gguf")                     # F32
#   model = ML::GGUF::NomicBertMoE.from_gguf("model.gguf", F16SimBackend.new)  # FP16 sim

require "./reader"
require "./tokenizer"
require "./compute"

module ML::GGUF
  class NomicBertMoE(B)
    getter dim : Int32           # 768
    getter n_heads : Int32       # 12
    getter n_layers : Int32      # 12
    getter head_dim : Int32      # 64
    getter ffn_dim : Int32       # 3072
    getter n_experts : Int32     # 8
    getter n_experts_used : Int32 # 2
    getter moe_every_n : Int32   # 2
    getter vocab_size : Int32
    getter max_seq_len : Int32   # 512
    getter rope_theta : Float32  # 10000
    getter backend : B

    # Weights (dequantized to F32)
    @token_embd = [] of Float32        # [vocab_size, dim]
    @token_types = [] of Float32       # [dim] (type 0 embedding)
    @embd_norm_w = [] of Float32       # [dim]
    @embd_norm_b = [] of Float32       # [dim]
    @layers = [] of LayerWeights

    # Tokenizer
    @tokenizer : UnigramTokenizer?

    # Precomputed RoPE cos/sin tables [max_seq_len, head_dim/2]
    @rope_cos = [] of Float32
    @rope_sin = [] of Float32

    struct LayerWeights
      getter attn_qkv_w : QuantWeight       # [3*dim, dim]
      getter attn_qkv_b : Array(Float32)     # [3*dim]
      getter attn_out_w : QuantWeight        # [dim, dim]
      getter attn_out_b : Array(Float32)     # [dim]
      getter norm1_w : Array(Float32)         # [dim]
      getter norm1_b : Array(Float32)         # [dim]
      getter norm2_w : Array(Float32)         # [dim]
      getter norm2_b : Array(Float32)         # [dim]
      # Dense FFN (non-MoE layers)
      getter ffn_up_w : QuantWeight?         # [ffn_dim, dim]
      getter ffn_up_b : Array(Float32)?       # [ffn_dim]
      getter ffn_down_w : QuantWeight?       # [dim, ffn_dim]
      getter ffn_down_b : Array(Float32)?     # [dim]
      # MoE FFN (MoE layers)
      getter gate_w : Array(Float32)?         # [n_experts, dim] — small, kept as F32
      getter expert_up_w : QuantWeight?      # [n_experts * ffn_dim, dim]
      getter expert_down_w : QuantWeight?    # [n_experts * dim, ffn_dim]

      def initialize(@attn_qkv_w, @attn_qkv_b, @attn_out_w, @attn_out_b,
                     @norm1_w, @norm1_b, @norm2_w, @norm2_b,
                     @ffn_up_w = nil, @ffn_up_b = nil, @ffn_down_w = nil, @ffn_down_b = nil,
                     @gate_w = nil, @expert_up_w = nil, @expert_down_w = nil)
      end

      def moe? : Bool
        !@gate_w.nil?
      end
    end

    def self.from_gguf(path : String) : NomicBertMoE(F32Backend)
      from_gguf(path, F32Backend.new)
    end

    def self.from_gguf(path : String, backend : B) : NomicBertMoE(B) forall B
      gguf = GGUFFile.new(path)
      model = NomicBertMoE(B).new(gguf, backend)
      gguf.close
      model
    end

    def initialize(gguf : GGUFFile, @backend : B)
      prefix = gguf.get_string("general.architecture") || "nomic-bert-moe"

      @dim = (gguf.get_int("#{prefix}.embedding_length") || 768).to_i32
      @n_heads = (gguf.get_int("#{prefix}.attention.head_count") || 12).to_i32
      @n_layers = (gguf.get_int("#{prefix}.block_count") || 12).to_i32
      @ffn_dim = (gguf.get_int("#{prefix}.feed_forward_length") || 3072).to_i32
      @n_experts = (gguf.get_int("#{prefix}.expert_count") || 8).to_i32
      @n_experts_used = (gguf.get_int("#{prefix}.expert_used_count") || 2).to_i32
      @moe_every_n = (gguf.get_int("#{prefix}.moe_every_n_layers") || 2).to_i32
      @max_seq_len = (gguf.get_int("#{prefix}.context_length") || 512).to_i32
      @rope_theta = (gguf.get_float("#{prefix}.rope.freq_base") || 10000.0).to_f32
      @head_dim = @dim // @n_heads
      @vocab_size = 0

      # Load tokenizer
      @tokenizer = UnigramTokenizer.new(gguf)
      @vocab_size = @tokenizer.not_nil!.vocab_size

      # Precompute RoPE tables
      @rope_cos, @rope_sin = build_rope_tables

      # Load weights
      load_weights(gguf)
    end

    # Embed a single text → Float32 array (dim=768)
    def embed(text : String) : Array(Float32)
      tokens = tokenize(text)
      forward(tokens)
    end

    # Embed multiple texts (sequential for now)
    def embed_batch(texts : Array(String)) : Array(Array(Float32))
      texts.map { |t| embed(t) }
    end

    # Tokenize text → token IDs (delegates to SentencePiece unigram tokenizer)
    def tokenize(text : String) : Array(Int32)
      ids = @tokenizer.not_nil!.encode(text)
      # Truncate to max_seq_len
      ids.size > @max_seq_len ? ids[0, @max_seq_len] : ids
    end

    # Forward pass: token IDs → mean-pooled L2-normalized embedding
    private def forward(tokens : Array(Int32)) : Array(Float32)
      seq_len = tokens.size

      # 1. Token embedding lookup + type embedding + layer norm
      hidden = Array(Float32).new(seq_len * @dim, 0.0_f32)
      seq_len.times do |pos|
        tid = tokens[pos].clamp(0, @vocab_size - 1)
        @dim.times do |j|
          hidden[pos * @dim + j] = @token_embd[tid * @dim + j] + @token_types[j]
        end
      end
      layer_norm!(hidden, seq_len, @embd_norm_w, @embd_norm_b)

      # 2. Transformer blocks
      # nomic-bert uses post-norm: output = norm(x + sublayer(x))
      # The norm names (attn_output_norm, layer_output_norm) confirm post-norm.
      @n_layers.times do |layer_idx|
        lw = @layers[layer_idx]

        # Attention sublayer + residual + norm
        attn_out = self_attention(hidden, seq_len, lw)
        hidden.size.times { |i| hidden[i] += attn_out[i] }
        layer_norm!(hidden, seq_len, lw.norm1_w, lw.norm1_b)

        # FFN sublayer + residual + norm
        ffn_out = if lw.moe?
                    moe_ffn(hidden, seq_len, lw)
                  else
                    dense_ffn(hidden, seq_len, lw)
                  end
        hidden.size.times { |i| hidden[i] += ffn_out[i] }
        layer_norm!(hidden, seq_len, lw.norm2_w, lw.norm2_b)
      end

      # 3. Mean pooling over sequence
      result = Array(Float32).new(@dim, 0.0_f32)
      seq_len.times do |pos|
        @dim.times { |j| result[j] += hidden[pos * @dim + j] }
      end
      inv_len = 1.0_f32 / seq_len
      @dim.times { |j| result[j] *= inv_len }

      # 4. L2 normalize
      norm = 0.0_f32
      result.each { |v| norm += v * v }
      norm = Math.sqrt(norm)
      if norm > 1e-8_f32
        result.map! { |v| v / norm }
      end

      result
    end

    # Matmul through backend (precision depends on B)
    private def q_matmul_add(x : Array(Float32), rows : Int32, qw : QuantWeight, bias : Array(Float32)) : Array(Float32)
      @backend.matmul(x, rows, qw, bias)
    end

    # Self-attention with RoPE
    private def self_attention(x : Array(Float32), seq_len : Int32, lw : LayerWeights) : Array(Float32)
      # QKV projection: [seq_len, dim] × [3*dim, dim]^T → [seq_len, 3*dim]
      qkv = q_matmul_add(x, seq_len, lw.attn_qkv_w, lw.attn_qkv_b)

      # Split into Q, K, V and reshape to [n_heads, seq_len, head_dim]
      # Then apply RoPE to Q and K
      q = Array(Float32).new(@n_heads * seq_len * @head_dim, 0.0_f32)
      k = Array(Float32).new(@n_heads * seq_len * @head_dim, 0.0_f32)
      v = Array(Float32).new(@n_heads * seq_len * @head_dim, 0.0_f32)

      seq_len.times do |pos|
        @n_heads.times do |h|
          off_q = pos * 3 * @dim + h * @head_dim
          off_k = pos * 3 * @dim + @dim + h * @head_dim
          off_v = pos * 3 * @dim + 2 * @dim + h * @head_dim
          dst_off = h * seq_len * @head_dim + pos * @head_dim

          @head_dim.times do |j|
            q[dst_off + j] = qkv[off_q + j]
            k[dst_off + j] = qkv[off_k + j]
            v[dst_off + j] = qkv[off_v + j]
          end

          # Apply RoPE to Q and K
          apply_rope!(q, dst_off, pos)
          apply_rope!(k, dst_off, pos)
        end
      end

      # Attention: softmax(Q @ K^T / sqrt(head_dim)) @ V
      scale = 1.0_f32 / Math.sqrt(@head_dim.to_f32)
      attn_out = Array(Float32).new(seq_len * @dim, 0.0_f32)

      @n_heads.times do |h|
        h_off = h * seq_len * @head_dim

        # Compute attention scores [seq_len, seq_len]
        scores = Array(Float32).new(seq_len * seq_len, 0.0_f32)
        seq_len.times do |i|
          seq_len.times do |j|
            d = @backend.dot(q, h_off + i * @head_dim, k, h_off + j * @head_dim, @head_dim)
            scores[i * seq_len + j] = d * scale
          end
          # Softmax over row i
          softmax_row!(scores, i * seq_len, seq_len)
        end

        # Weighted sum of V
        seq_len.times do |i|
          @head_dim.times do |d|
            sum = 0.0_f32
            seq_len.times do |j|
              sum += scores[i * seq_len + j] * v[h_off + j * @head_dim + d]
            end
            attn_out[i * @dim + h * @head_dim + d] = sum
          end
        end
      end

      # Output projection
      q_matmul_add(attn_out, seq_len, lw.attn_out_w, lw.attn_out_b)
    end

    # Dense FFN: up → GELU → down
    # Note: llama.cpp uses SiLU for nomic-bert but empirical testing shows
    # GELU gives much higher cosine similarity (0.79 vs 0.26) for this GGUF.
    private def dense_ffn(x : Array(Float32), seq_len : Int32, lw : LayerWeights) : Array(Float32)
      up_w = lw.ffn_up_w.not_nil!
      up_b = lw.ffn_up_b.not_nil!
      down_w = lw.ffn_down_w.not_nil!
      down_b = lw.ffn_down_b.not_nil!

      # Up: [seq, dim] → [seq, ffn_dim]
      h = q_matmul_add(x, seq_len, up_w, up_b)
      # GELU activation
      h.map! { |v| gelu(v) }
      # Down: [seq, ffn_dim] → [seq, dim]
      q_matmul_add(h, seq_len, down_w, down_b)
    end

    # MoE FFN: gate → top-2 routing → expert FFNs → weighted sum
    # Expert matmuls use fused dequant for each expert slice.
    private def moe_ffn(x : Array(Float32), seq_len : Int32, lw : LayerWeights) : Array(Float32)
      gate_w = lw.gate_w.not_nil!          # [n_experts, dim] F32
      exp_up_qw = lw.expert_up_w.not_nil!   # QuantWeight for all experts
      exp_down_qw = lw.expert_down_w.not_nil!

      result = Array(Float32).new(seq_len * @dim, 0.0_f32)

      # Precompute per-expert byte offsets
      # Expert weights are [ffn_dim, dim] per expert, stored sequentially
      up_expert_rows = @ffn_dim
      down_expert_rows = @dim
      up_row_bytes = expert_row_bytes(exp_up_qw.type, @dim)
      down_row_bytes = expert_row_bytes(exp_down_qw.type, @ffn_dim)
      up_expert_bytes = up_expert_rows * up_row_bytes
      down_expert_bytes = down_expert_rows * down_row_bytes

      zero_bias_ffn = Array(Float32).new(@ffn_dim, 0.0_f32)
      zero_bias_dim = Array(Float32).new(@dim, 0.0_f32)

      seq_len.times do |pos|
        x_off = pos * @dim
        x_pos = x[x_off, @dim]

        # Gate logits
        gate_logits = Array(Float32).new(@n_experts, 0.0_f32)
        @n_experts.times do |e|
          dot = 0.0_f32
          @dim.times { |j| dot += x[x_off + j] * gate_w[e * @dim + j] }
          gate_logits[e] = dot
        end

        # Top-2
        top2 = gate_logits.each_with_index.to_a.sort_by { |v, _| -v }.first(@n_experts_used)
        max_g = top2.max_of(&.[0])
        exps = top2.map { |v, i| {Math.exp(v - max_g), i} }
        sum_exp = exps.sum(&.[0])

        top2.size.times do |ti|
          weight = (exps[ti][0] / sum_exp).to_f32
          expert_idx = exps[ti][1]

          # Per-expert sliced QuantWeight for up and down
          up_slice = Bytes.new(exp_up_qw.raw.to_unsafe + expert_idx * up_expert_bytes, up_expert_bytes, read_only: true)
          up_qw_e = QuantWeight.new(up_slice, exp_up_qw.type, @ffn_dim, @dim)

          # Up + GELU
          h = QuantMatmul.matmul_add(x_pos, 1, @dim, up_qw_e.raw, up_qw_e.type, @ffn_dim, zero_bias_ffn)
          h.map! { |v| gelu(v) }

          # Down
          down_slice = Bytes.new(exp_down_qw.raw.to_unsafe + expert_idx * down_expert_bytes, down_expert_bytes, read_only: true)
          down_qw_e = QuantWeight.new(down_slice, exp_down_qw.type, @dim, @ffn_dim)

          d = QuantMatmul.matmul_add(h, 1, @ffn_dim, down_qw_e.raw, down_qw_e.type, @dim, zero_bias_dim)

          r_off = pos * @dim
          @dim.times { |j| result[r_off + j] += weight * d[j] }
        end
      end

      result
    end

    # Bytes per row of a quantized matrix (one row = in_dim elements)
    private def expert_row_bytes(type : TensorType, in_dim : Int32) : Int32
      if type.f32? || type.f16?
        in_dim * type.block_bytes
      else
        (in_dim // type.block_elements) * type.block_bytes
      end
    end

    # Apply RoPE rotation to a head vector at position pos
    private def apply_rope!(vec : Array(Float32), offset : Int32, pos : Int32)
      half = @head_dim // 2
      rope_off = pos * half
      half.times do |i|
        cos = @rope_cos[rope_off + i]
        sin = @rope_sin[rope_off + i]
        v0 = vec[offset + 2 * i]
        v1 = vec[offset + 2 * i + 1]
        vec[offset + 2 * i]     = v0 * cos - v1 * sin
        vec[offset + 2 * i + 1] = v0 * sin + v1 * cos
      end
    end

    # Delegate to backend
    private def layer_norm!(x : Array(Float32), n_pos : Int32, w : Array(Float32), b : Array(Float32))
      @backend.layer_norm!(x, n_pos, @dim, w, b)
    end

    private def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32)
      @backend.softmax_row!(scores, offset, len)
    end

    private def gelu(x : Float32) : Float32
      @backend.gelu(x)
    end

    # Build RoPE cos/sin tables
    private def build_rope_tables : {Array(Float32), Array(Float32)}
      half = @head_dim // 2
      cos_table = Array(Float32).new(@max_seq_len * half, 0.0_f32)
      sin_table = Array(Float32).new(@max_seq_len * half, 0.0_f32)

      @max_seq_len.times do |pos|
        half.times do |i|
          freq = 1.0_f32 / (@rope_theta ** (2.0_f32 * i / @head_dim))
          angle = pos.to_f32 * freq
          cos_table[pos * half + i] = Math.cos(angle)
          sin_table[pos * half + i] = Math.sin(angle)
        end
      end

      {cos_table, sin_table}
    end

    # Load all weights from GGUF
    private def load_weights(gguf : GGUFFile)
      @token_embd = read_f32(gguf, "token_embd.weight")  # Large but needed for lookup
      @token_types = read_f32(gguf, "token_types.weight")
      @embd_norm_w = read_f32(gguf, "token_embd_norm.weight")
      @embd_norm_b = read_f32(gguf, "token_embd_norm.bias")

      @layers = Array(LayerWeights).new(@n_layers) do |i|
        p = "blk.#{i}"
        is_moe = (i % @moe_every_n == 1) && @n_experts > 0

        if is_moe
          # Expert weights stored as [in, out, n_experts] — read as one big quant block
          up_info = gguf.tensor("#{p}.ffn_up_exps.weight").not_nil!
          down_info = gguf.tensor("#{p}.ffn_down_exps.weight").not_nil!
          up_raw = gguf.read_tensor_raw(up_info).dup
          down_raw = gguf.read_tensor_raw(down_info).dup

          LayerWeights.new(
            attn_qkv_w: read_quant(gguf, "#{p}.attn_qkv.weight"),
            attn_qkv_b: read_f32(gguf, "#{p}.attn_qkv.bias"),
            attn_out_w: read_quant(gguf, "#{p}.attn_output.weight"),
            attn_out_b: read_f32(gguf, "#{p}.attn_output.bias"),
            norm1_w: read_f32(gguf, "#{p}.attn_output_norm.weight"),
            norm1_b: read_f32(gguf, "#{p}.attn_output_norm.bias"),
            norm2_w: read_f32(gguf, "#{p}.layer_output_norm.weight"),
            norm2_b: read_f32(gguf, "#{p}.layer_output_norm.bias"),
            gate_w: read_f32(gguf, "#{p}.ffn_gate_inp.weight"),
            expert_up_w: QuantWeight.new(up_raw, up_info.type, @n_experts * @ffn_dim, @dim),
            expert_down_w: QuantWeight.new(down_raw, down_info.type, @n_experts * @dim, @ffn_dim),
          )
        else
          LayerWeights.new(
            attn_qkv_w: read_quant(gguf, "#{p}.attn_qkv.weight"),
            attn_qkv_b: read_f32(gguf, "#{p}.attn_qkv.bias"),
            attn_out_w: read_quant(gguf, "#{p}.attn_output.weight"),
            attn_out_b: read_f32(gguf, "#{p}.attn_output.bias"),
            norm1_w: read_f32(gguf, "#{p}.attn_output_norm.weight"),
            norm1_b: read_f32(gguf, "#{p}.attn_output_norm.bias"),
            norm2_w: read_f32(gguf, "#{p}.layer_output_norm.weight"),
            norm2_b: read_f32(gguf, "#{p}.layer_output_norm.bias"),
            ffn_up_w: read_quant(gguf, "#{p}.ffn_up.weight"),
            ffn_up_b: read_f32(gguf, "#{p}.ffn_up.bias"),
            ffn_down_w: read_quant(gguf, "#{p}.ffn_down.weight"),
            ffn_down_b: read_f32(gguf, "#{p}.ffn_down.bias"),
          )
        end
      end
    end

    # Read as dequantized F32 (for small tensors: bias, norm, gate, embedding)
    private def read_f32(gguf : GGUFFile, name : String) : Array(Float32)
      info = gguf.tensor(name)
      raise "Missing tensor: #{name}" unless info
      gguf.read_tensor_f32(info)
    end

    # Read as raw quantized bytes (for large weight matrices — used with fused matmul)
    private def read_quant(gguf : GGUFFile, name : String) : QuantWeight
      info = gguf.tensor(name)
      raise "Missing tensor: #{name}" unless info
      raw = gguf.read_tensor_raw(info).dup  # dup to own the bytes (mmap may unmap)
      # GGUF dims: [ne0, ne1] where ne0=in_dim, ne1=out_dim
      # But stored row-major as [out_dim rows, in_dim cols]
      in_dim = info.dims[0].to_i32
      out_dim = info.dims.size > 1 ? info.dims[1].to_i32 : 1
      QuantWeight.new(raw, info.type, out_dim, in_dim)
    end
  end
end
