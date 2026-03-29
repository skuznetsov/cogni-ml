# NomicBertMoE — inference-only nomic-embed-text-v2-moe from GGUF.
#
# Architecture: 12-block BERT encoder with MoE FFN every 2 layers.
# - RoPE position encoding (freq_base=10000)
# - MoE: 8 experts, top-2 gating
# - Mean pooling → L2 normalize
# - No autograd — pure F32 tensor math for inference speed.
#
# Usage:
#   model = ML::GGUF::NomicBertMoE.from_gguf("model.gguf")
#   embedding = model.embed("hello world")  # => Array(Float32) dim=768

require "./reader"
require "./tokenizer"

module ML::GGUF
  class NomicBertMoE
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
      getter attn_qkv_w : Array(Float32)     # [3*dim, dim]
      getter attn_qkv_b : Array(Float32)     # [3*dim]
      getter attn_out_w : Array(Float32)      # [dim, dim]
      getter attn_out_b : Array(Float32)      # [dim]
      getter norm1_w : Array(Float32)         # [dim]
      getter norm1_b : Array(Float32)         # [dim]
      getter norm2_w : Array(Float32)         # [dim]
      getter norm2_b : Array(Float32)         # [dim]
      # Dense FFN (non-MoE layers)
      getter ffn_up_w : Array(Float32)?       # [ffn_dim, dim]
      getter ffn_up_b : Array(Float32)?       # [ffn_dim]
      getter ffn_down_w : Array(Float32)?     # [dim, ffn_dim]
      getter ffn_down_b : Array(Float32)?     # [dim]
      # MoE FFN (MoE layers)
      getter gate_w : Array(Float32)?         # [n_experts, dim]
      getter expert_up_w : Array(Float32)?    # [n_experts, ffn_dim, dim]
      getter expert_down_w : Array(Float32)?  # [n_experts, dim, ffn_dim]

      def initialize(@attn_qkv_w, @attn_qkv_b, @attn_out_w, @attn_out_b,
                     @norm1_w, @norm1_b, @norm2_w, @norm2_b,
                     @ffn_up_w = nil, @ffn_up_b = nil, @ffn_down_w = nil, @ffn_down_b = nil,
                     @gate_w = nil, @expert_up_w = nil, @expert_down_w = nil)
      end

      def moe? : Bool
        !@gate_w.nil?
      end
    end

    def self.from_gguf(path : String) : NomicBertMoE
      gguf = GGUFFile.new(path)
      model = new(gguf)
      gguf.close
      model
    end

    def initialize(gguf : GGUFFile)
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

    # Self-attention with RoPE
    private def self_attention(x : Array(Float32), seq_len : Int32, lw : LayerWeights) : Array(Float32)
      # QKV projection: [seq_len, dim] × [3*dim, dim]^T → [seq_len, 3*dim]
      qkv = matmul_add(x, seq_len, @dim, lw.attn_qkv_w, 3 * @dim, lw.attn_qkv_b)

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
            dot = 0.0_f32
            @head_dim.times do |d|
              dot += q[h_off + i * @head_dim + d] * k[h_off + j * @head_dim + d]
            end
            scores[i * seq_len + j] = dot * scale
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
      matmul_add(attn_out, seq_len, @dim, lw.attn_out_w, @dim, lw.attn_out_b)
    end

    # Dense FFN: up → GELU → down
    private def dense_ffn(x : Array(Float32), seq_len : Int32, lw : LayerWeights) : Array(Float32)
      up_w = lw.ffn_up_w.not_nil!
      up_b = lw.ffn_up_b.not_nil!
      down_w = lw.ffn_down_w.not_nil!
      down_b = lw.ffn_down_b.not_nil!

      # Up: [seq, dim] → [seq, ffn_dim]
      h = matmul_add(x, seq_len, @dim, up_w, @ffn_dim, up_b)
      # GELU activation
      h.map! { |v| gelu(v) }
      # Down: [seq, ffn_dim] → [seq, dim]
      matmul_add(h, seq_len, @ffn_dim, down_w, @dim, down_b)
    end

    # MoE FFN: gate → top-2 routing → expert FFNs → weighted sum
    private def moe_ffn(x : Array(Float32), seq_len : Int32, lw : LayerWeights) : Array(Float32)
      gate_w = lw.gate_w.not_nil!          # [n_experts, dim]
      exp_up = lw.expert_up_w.not_nil!     # [n_experts * ffn_dim * dim]
      exp_down = lw.expert_down_w.not_nil! # [n_experts * dim * ffn_dim]

      result = Array(Float32).new(seq_len * @dim, 0.0_f32)
      expert_dim = @ffn_dim * @dim

      seq_len.times do |pos|
        x_off = pos * @dim

        # Compute gate logits: x @ gate_w^T → [n_experts]
        # gate_w dims=[dim, n_experts] but stored as [n_experts, dim] row-major
        gate_logits = Array(Float32).new(@n_experts, 0.0_f32)
        @n_experts.times do |e|
          dot = 0.0_f32
          @dim.times { |j| dot += x[x_off + j] * gate_w[e * @dim + j] }
          gate_logits[e] = dot
        end

        # Top-2 experts
        top2 = gate_logits.each_with_index.to_a.sort_by { |v, _| -v }.first(@n_experts_used)

        # Softmax over top-2 gate values
        max_g = top2.max_of(&.[0])
        exps = top2.map { |v, i| {Math.exp(v - max_g), i} }
        sum_exp = exps.sum(&.[0])

        top2_weights = exps.map { |e, i| {(e / sum_exp).to_f32, i} }

        # Run each expert FFN and accumulate weighted output
        top2_weights.each do |weight, expert_idx|
          # Expert weights: [in, out, n_experts] dims but [out, in] per expert in memory
          # exp_up dims=[dim, ffn_dim, 8] → per expert: [ffn_dim, dim], stride = ffn_dim*dim
          # exp_down dims=[ffn_dim, dim, 8] → per expert: [dim, ffn_dim], stride = dim*ffn_dim

          # Up: x @ exp_up[e]^T → [ffn_dim]
          up_off = expert_idx * expert_dim
          h = Array(Float32).new(@ffn_dim, 0.0_f32)
          @ffn_dim.times do |f|
            dot = 0.0_f32
            @dim.times { |j| dot += x[x_off + j] * exp_up[up_off + f * @dim + j] }
            h[f] = gelu(dot)
          end

          # Down: h @ exp_down[e]^T → [dim]
          down_off = expert_idx * expert_dim
          r_off = pos * @dim
          @dim.times do |j|
            dot = 0.0_f32
            @ffn_dim.times { |f| dot += h[f] * exp_down[down_off + j * @ffn_dim + f] }
            result[r_off + j] += weight * dot
          end
        end
      end

      result
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

    # Matrix multiply + bias: x[rows, in_dim] × w[out_dim, in_dim]^T + bias → [rows, out_dim]
    # GGUF dim[0]=in_dim, dim[1]=out_dim, but stored row-major as [out_dim, in_dim].
    private def matmul_add(x : Array(Float32), rows : Int32, in_dim : Int32,
                           w : Array(Float32), out_dim : Int32, bias : Array(Float32)) : Array(Float32)
      result = Array(Float32).new(rows * out_dim, 0.0_f32)
      rows.times do |r|
        x_off = r * in_dim
        r_off = r * out_dim
        out_dim.times do |o|
          dot = bias[o]
          w_off = o * in_dim
          in_dim.times { |j| dot += x[x_off + j] * w[w_off + j] }
          result[r_off + o] = dot
        end
      end
      result
    end

    # In-place layer norm: x[pos, dim] for all positions
    private def layer_norm!(x : Array(Float32), n_pos : Int32, w : Array(Float32), b : Array(Float32))
      eps = 1e-5_f32
      n_pos.times do |pos|
        off = pos * @dim
        # Mean
        mean = 0.0_f32
        @dim.times { |j| mean += x[off + j] }
        mean /= @dim
        # Variance
        var = 0.0_f32
        @dim.times { |j| d = x[off + j] - mean; var += d * d }
        var /= @dim
        inv_std = 1.0_f32 / Math.sqrt(var + eps)
        # Normalize + scale + shift
        @dim.times do |j|
          x[off + j] = (x[off + j] - mean) * inv_std * w[j] + b[j]
        end
      end
    end

    # Softmax in-place over a row
    private def softmax_row!(scores : Array(Float32), offset : Int32, len : Int32)
      max_val = -Float32::MAX
      len.times { |i| max_val = Math.max(max_val, scores[offset + i]) }
      sum = 0.0_f32
      len.times do |i|
        scores[offset + i] = Math.exp(scores[offset + i] - max_val)
        sum += scores[offset + i]
      end
      inv_sum = 1.0_f32 / sum
      len.times { |i| scores[offset + i] *= inv_sum }
    end

    # GELU activation (tanh approximation)
    private def gelu(x : Float32) : Float32
      0.5_f32 * x * (1.0_f32 + Math.tanh(0.7978845608_f32 * (x + 0.044715_f32 * x * x * x)))
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
      @token_embd = read_weight(gguf, "token_embd.weight")
      @token_types = read_weight(gguf, "token_types.weight")
      @embd_norm_w = read_weight(gguf, "token_embd_norm.weight")
      @embd_norm_b = read_weight(gguf, "token_embd_norm.bias")

      @layers = Array(LayerWeights).new(@n_layers) do |i|
        p = "blk.#{i}"
        is_moe = (i % @moe_every_n == 1) && @n_experts > 0

        if is_moe
          LayerWeights.new(
            attn_qkv_w: read_weight(gguf, "#{p}.attn_qkv.weight"),
            attn_qkv_b: read_weight(gguf, "#{p}.attn_qkv.bias"),
            attn_out_w: read_weight(gguf, "#{p}.attn_output.weight"),
            attn_out_b: read_weight(gguf, "#{p}.attn_output.bias"),
            norm1_w: read_weight(gguf, "#{p}.attn_output_norm.weight"),
            norm1_b: read_weight(gguf, "#{p}.attn_output_norm.bias"),
            norm2_w: read_weight(gguf, "#{p}.layer_output_norm.weight"),
            norm2_b: read_weight(gguf, "#{p}.layer_output_norm.bias"),
            gate_w: read_weight(gguf, "#{p}.ffn_gate_inp.weight"),
            expert_up_w: read_weight(gguf, "#{p}.ffn_up_exps.weight"),
            expert_down_w: read_weight(gguf, "#{p}.ffn_down_exps.weight"),
          )
        else
          LayerWeights.new(
            attn_qkv_w: read_weight(gguf, "#{p}.attn_qkv.weight"),
            attn_qkv_b: read_weight(gguf, "#{p}.attn_qkv.bias"),
            attn_out_w: read_weight(gguf, "#{p}.attn_output.weight"),
            attn_out_b: read_weight(gguf, "#{p}.attn_output.bias"),
            norm1_w: read_weight(gguf, "#{p}.attn_output_norm.weight"),
            norm1_b: read_weight(gguf, "#{p}.attn_output_norm.bias"),
            norm2_w: read_weight(gguf, "#{p}.layer_output_norm.weight"),
            norm2_b: read_weight(gguf, "#{p}.layer_output_norm.bias"),
            ffn_up_w: read_weight(gguf, "#{p}.ffn_up.weight"),
            ffn_up_b: read_weight(gguf, "#{p}.ffn_up.bias"),
            ffn_down_w: read_weight(gguf, "#{p}.ffn_down.weight"),
            ffn_down_b: read_weight(gguf, "#{p}.ffn_down.bias"),
          )
        end
      end
    end

    private def read_weight(gguf : GGUFFile, name : String) : Array(Float32)
      info = gguf.tensor(name)
      raise "Missing tensor: #{name}" unless info
      gguf.read_tensor_f32(info)
    end
  end
end
