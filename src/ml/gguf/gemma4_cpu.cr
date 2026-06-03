require "./gemma4_meta"
require "./gemma4_weights"
require "./dequant"
require "./quant_matmul"

# Gemma4 CPU reference primitives.
#
# This module is intentionally small: it captures math semantics that differ
# from Qwen before we build the full text forward path.
module ML::GGUF
  module Gemma4CPU
    extend self

    struct AttentionProjection
      getter q : Array(Float32)
      getter k : Array(Float32)
      getter v : Array(Float32)
      getter reused_k_as_v : Bool

      def initialize(@q, @k, @v, @reused_k_as_v)
      end
    end

    class LayerState
      property k_cache : Array(Float32)?
      property v_cache : Array(Float32)?
      property position : Int32 = 0

      def initialize
      end
    end

    class State
      getter layers : Array(LayerState)
      getter max_seq : Int32

      def initialize(hp : Gemma4Hparams, @max_seq : Int32 = 1024)
        @layers = Array(LayerState).new(hp.n_layer) { LayerState.new }
      end
    end

    # RMSNorm with learned weight: y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i].
    def rms_norm(x : Array(Float32), w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Array(Float32)
      raise ArgumentError.new("rms_norm weight size mismatch: #{w.size} != #{x.size}") unless w.size == x.size

      result = x.dup
      rms_norm!(result, w, eps)
      result
    end

    def rms_norm!(x : Array(Float32), w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      raise ArgumentError.new("rms_norm weight size mismatch: #{w.size} != #{x.size}") unless w.size == x.size

      ss = 0.0_f64
      x.each { |v| ss += v.to_f64 * v.to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / x.size.to_f64 + eps.to_f64)).to_f32
      x.size.times { |i| x[i] = x[i] * inv_rms * w[i] }
    end

    # RMSNorm without learned weight. Gemma4 full-attention layers use this for
    # V when `attn_v.weight` is absent and V reuses pre-norm K.
    def rms_norm_plain(x : Array(Float32), eps : Float32 = 1.0e-6_f32) : Array(Float32)
      result = x.dup
      rms_norm_plain!(result, eps)
      result
    end

    def rms_norm_plain!(x : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      ss = 0.0_f64
      x.each { |v| ss += v.to_f64 * v.to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / x.size.to_f64 + eps.to_f64)).to_f32
      x.size.times { |i| x[i] *= inv_rms }
    end

    def rms_norm_slice!(x : Array(Float32), offset : Int32, len : Int32,
                        w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      raise ArgumentError.new("rms_norm_slice weight size mismatch: #{w.size} != #{len}") unless w.size == len
      raise ArgumentError.new("rms_norm_slice out of bounds") if offset < 0 || len < 0 || offset + len > x.size

      ss = 0.0_f64
      len.times { |j| v = x[offset + j]; ss += v.to_f64 * v.to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] = x[offset + j] * inv_rms * w[j] }
    end

    def rms_norm_plain_slice!(x : Array(Float32), offset : Int32, len : Int32,
                              eps : Float32 = 1.0e-6_f32) : Nil
      raise ArgumentError.new("rms_norm_plain_slice out of bounds") if offset < 0 || len < 0 || offset + len > x.size

      ss = 0.0_f64
      len.times { |j| v = x[offset + j]; ss += v.to_f64 * v.to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] *= inv_rms }
    end

    # ggml_gelu/Metal GELU approximation used by llama.cpp for Gemma FFN.
    def gelu(x : Float32) : Float32
      0.5_f32 * x * (1.0_f32 + Math.tanh(0.7978845608028654_f32 * x * (1.0_f32 + 0.044715_f32 * x * x)))
    end

    def gelu!(x : Array(Float32)) : Nil
      x.size.times { |i| x[i] = gelu(x[i]) }
    end

    def logit_softcap!(x : Array(Float32), cap : Float32) : Nil
      return if cap <= 0.0_f32

      inv = 1.0_f32 / cap
      x.size.times { |i| x[i] = Math.tanh(x[i] * inv) * cap }
    end

    def logit_softcap(x : Array(Float32), cap : Float32) : Array(Float32)
      result = x.dup
      logit_softcap!(result, cap)
      result
    end

    def matmul(qw : QuantWeight, x : Array(Float32), rows : Int32 = 1) : Array(Float32)
      raise ArgumentError.new("matmul input size mismatch: #{x.size} != #{rows * qw.in_dim}") unless x.size == rows * qw.in_dim

      bias = Array(Float32).new(qw.out_dim, 0.0_f32)
      QuantMatmul.matmul_add(x, rows, qw.in_dim, qw.raw, qw.type, qw.out_dim, bias)
    end

    def attention_project_pre_norm(lw : Gemma4LayerWeights, x_norm : Array(Float32)) : AttentionProjection
      q = matmul(lw.attn_q_qw, x_norm)
      k = matmul(lw.attn_k_qw, x_norm)
      if v_qw = lw.attn_v_qw
        v = matmul(v_qw, x_norm)
        AttentionProjection.new(q, k, v, false)
      else
        AttentionProjection.new(q, k, k.dup, true)
      end
    end

    def attention_project_normed(lw : Gemma4LayerWeights, x : Array(Float32),
                                 hp : Gemma4Hparams, il : Int32) : AttentionProjection
      x_norm = rms_norm(x, lw.attn_norm, hp.rms_eps)
      proj = attention_project_pre_norm(lw, x_norm)
      normalize_attention_projection!(proj, lw, hp, il)
      proj
    end

    def normalize_attention_projection!(proj : AttentionProjection, lw : Gemma4LayerWeights,
                                        hp : Gemma4Hparams, il : Int32) : Nil
      head_dim = hp.head_dim_for_layer(il)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)

      raise ArgumentError.new("q projection size mismatch at layer #{il}") unless proj.q.size == n_head * head_dim
      raise ArgumentError.new("k projection size mismatch at layer #{il}") unless proj.k.size == n_head_kv * head_dim
      raise ArgumentError.new("v projection size mismatch at layer #{il}") unless proj.v.size == n_head_kv * head_dim

      n_head.times do |h|
        rms_norm_slice!(proj.q, h * head_dim, head_dim, lw.attn_q_norm, hp.rms_eps)
      end
      n_head_kv.times do |h|
        off = h * head_dim
        rms_norm_slice!(proj.k, off, head_dim, lw.attn_k_norm, hp.rms_eps)
        rms_norm_plain_slice!(proj.v, off, head_dim, hp.rms_eps)
      end
    end

    def apply_rope_to_qk!(proj : AttentionProjection, hp : Gemma4Hparams, il : Int32,
                          pos : Int32, rope_freqs : Array(Float32)? = nil) : Nil
      head_dim = hp.head_dim_for_layer(il)
      n_rot = hp.rope_dim_for_layer(il)
      base = hp.rope_freq_base_for_layer(il)

      hp.n_head.times do |h|
        rope_neox_slice!(proj.q, h * head_dim, n_rot, head_dim, pos, base, rope_freqs)
      end
      hp.n_head_kv(il).times do |h|
        rope_neox_slice!(proj.k, h * head_dim, n_rot, head_dim, pos, base, rope_freqs)
      end
    end

    def attention_context(weights : Gemma4Weights, il : Int32, x : Array(Float32),
                          pos : Int32, state : State) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      raise ArgumentError.new("layer #{il} does not own KV cache") unless hp.has_kv?(il)
      raise ArgumentError.new("position #{pos} exceeds state max_seq #{state.max_seq}") if pos < 0 || pos >= state.max_seq

      proj = attention_project_normed(lw, x, hp, il)
      apply_rope_to_qk!(proj, hp, il, pos, hp.full_attention?(il) ? weights.rope_freqs : nil)
      attention_context_from_projection!(proj, hp, il, pos, state.layers[il], state.max_seq)
    end

    def attention_projected_output(weights : Gemma4Weights, il : Int32, x : Array(Float32),
                                   pos : Int32, state : State) : Array(Float32)
      ctx = attention_context(weights, il, x, pos, state)
      matmul(weights.layers[il].attn_output_qw, ctx)
    end

    def forward_layer(weights : Gemma4Weights, il : Int32, x : Array(Float32),
                      pos : Int32, state : State) : Array(Float32)
      hp = weights.hparams
      lw = weights.layers[il]
      n_embd = hp.n_embd
      raise ArgumentError.new("forward_layer input size mismatch") unless x.size == n_embd

      attn_projected = attention_projected_output(weights, il, x, pos, state)
      attn_normed = rms_norm(attn_projected, lw.post_attention_norm, hp.rms_eps)
      attn_out = Array(Float32).new(n_embd) { |i| x[i] + attn_normed[i] }

      ffn_in = rms_norm(attn_out, lw.ffn_norm, hp.rms_eps)
      up = matmul(lw.ffn_up_qw, ffn_in)
      gate = matmul(lw.ffn_gate_qw, ffn_in)
      gate.size.times { |i| gate[i] = gelu(gate[i]) * up[i] }
      ffn = matmul(lw.ffn_down_qw, gate)
      ffn = rms_norm(ffn, lw.post_ffw_norm, hp.rms_eps)

      out = Array(Float32).new(n_embd) { |i| attn_out[i] + ffn[i] }
      if scale = lw.layer_output_scale.first?
        out.size.times { |i| out[i] *= scale }
      end
      out
    end

    def forward_hidden(weights : Gemma4Weights, token_id : Int32, pos : Int32,
                       state : State, stop_layer : Int32? = nil) : Array(Float32)
      x = scaled_embedding_lookup(weights, token_id)
      layer_count = stop_layer ? Math.min(stop_layer.not_nil!, weights.layers.size) : weights.layers.size
      layer_count.times do |il|
        x = forward_layer(weights, il, x, pos, state)
      end
      x
    end

    def forward_logits(weights : Gemma4Weights, token_id : Int32, pos : Int32,
                       state : State) : Array(Float32)
      hidden = forward_hidden(weights, token_id, pos, state)
      forward_logits_from_hidden(weights, hidden)
    end

    def forward_logits_from_hidden(weights : Gemma4Weights, hidden : Array(Float32)) : Array(Float32)
      x = rms_norm(hidden, weights.output_norm, weights.hparams.rms_eps)
      logits = matmul(weights.token_embd, x)
      logit_softcap!(logits, weights.hparams.final_logit_softcapping)
      logits
    end

    def top_k(logits : Array(Float32), k : Int32) : Array({Int32, Float32})
      raise ArgumentError.new("top_k k must be positive") unless k > 0

      best = [] of {Int32, Float32}
      logits.each_with_index do |v, i|
        if best.size < k
          best << {i, v}
          best.sort_by! { |pair| -pair[1] }
        elsif v > best[-1][1]
          best[-1] = {i, v}
          best.sort_by! { |pair| -pair[1] }
        end
      end
      best
    end

    def attention_context_from_projection!(proj : AttentionProjection, hp : Gemma4Hparams,
                                           il : Int32, pos : Int32,
                                           lstate : LayerState, max_seq : Int32) : Array(Float32)
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv(il)
      head_dim = hp.head_dim_for_layer(il)
      kv_dim = n_head_kv * head_dim
      q_dim = n_head * head_dim
      heads_per_group = n_head // n_head_kv

      k_cache = lstate.k_cache ||= Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      v_cache = lstate.v_cache ||= Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      base = pos * kv_dim
      kv_dim.times do |i|
        k_cache[base + i] = proj.k[i]
        v_cache[base + i] = proj.v[i]
      end
      lstate.position = Math.max(lstate.position, pos + 1)

      start_pos = hp.attention_start_pos(il, pos)
      len = pos - start_pos + 1
      out = Array(Float32).new(q_dim, 0.0_f32)
      scores = Array(Float32).new(len, 0.0_f32)

      n_head.times do |h|
        kvh = h // heads_per_group
        q_off = h * head_dim
        len.times do |idx|
          p = start_pos + idx
          k_off = p * kv_dim + kvh * head_dim
          score = 0.0_f32
          head_dim.times { |d| score += proj.q[q_off + d] * k_cache[k_off + d] }
          scores[idx] = score
        end
        softmax_slice!(scores, 0, len)

        out_off = h * head_dim
        len.times do |idx|
          p = start_pos + idx
          v_off = p * kv_dim + kvh * head_dim
          w = scores[idx]
          head_dim.times { |d| out[out_off + d] += w * v_cache[v_off + d] }
        end
      end
      out
    end

    def softmax_slice!(x : Array(Float32), offset : Int32, len : Int32) : Nil
      raise ArgumentError.new("softmax empty slice") unless len > 0
      raise ArgumentError.new("softmax slice out of bounds") if offset < 0 || offset + len > x.size

      maxv = x[offset]
      len.times do |i|
        v = x[offset + i]
        maxv = v if v > maxv
      end
      sum = 0.0_f32
      len.times do |i|
        e = Math.exp(x[offset + i] - maxv)
        x[offset + i] = e
        sum += e
      end
      inv = 1.0_f32 / sum
      len.times { |i| x[offset + i] *= inv }
    end

    # Default ggml RoPE path for text-mode Gemma4 short-context probes. This is
    # NeoX pairing over the first `n_rot` dims. Optional `freq_factors` mirrors
    # Gemma4 full-attention proportional RoPE (`theta / freq_factor`).
    def rope_neox_slice!(x : Array(Float32), offset : Int32, n_rot : Int32,
                         head_dim : Int32, pos : Int32, freq_base : Float32,
                         freq_factors : Array(Float32)? = nil) : Nil
      raise ArgumentError.new("rope n_rot must be even") unless n_rot.even?
      raise ArgumentError.new("rope n_rot #{n_rot} exceeds head_dim #{head_dim}") if n_rot > head_dim
      raise ArgumentError.new("rope slice out of bounds") if offset < 0 || offset + head_dim > x.size
      if factors = freq_factors
        raise ArgumentError.new("rope freq_factors too small: #{factors.size} < #{n_rot // 2}") if factors.size < n_rot // 2
      end

      half = n_rot // 2
      half.times do |i|
        i0 = 2 * i
        freq_factor = freq_factors ? freq_factors[i] : 1.0_f32
        theta = pos.to_f32 * (freq_base ** (-i0.to_f32 / n_rot.to_f32)) / freq_factor
        cos_t = Math.cos(theta)
        sin_t = Math.sin(theta)
        a = offset + i
        b = offset + i + half
        x0 = x[a]
        x1 = x[b]
        x[a] = x0 * cos_t - x1 * sin_t
        x[b] = x0 * sin_t + x1 * cos_t
      end
    end

    # Embedding lookup for one token id. GGUF stores token embeddings as
    # quantized rows [vocab_size, n_embd] even though the logical matmul weight
    # dimensions are represented as [n_embd, vocab_size].
    def embedding_lookup(token_embd : QuantWeight, token_id : Int32) : Array(Float32)
      n_embd = token_embd.in_dim
      raise ArgumentError.new("embedding token_id #{token_id} out of range 0...#{token_embd.out_dim}") if token_id < 0 || token_id >= token_embd.out_dim

      t = token_embd.type
      if (t.q4_k? || t.q5_k? || t.q6_k?) && n_embd % 256 != 0
        raise ArgumentError.new("embedding n_embd #{n_embd} not divisible by 256 for #{t.name}")
      elsif t.q8_0? && n_embd % 32 != 0
        raise ArgumentError.new("embedding n_embd #{n_embd} not divisible by 32 for Q8_0")
      end

      row_bytes = quant_row_bytes(t, n_embd)
      offset = token_id.to_i64 * row_bytes.to_i64
      raise ArgumentError.new("embedding row exceeds raw tensor bytes") if offset + row_bytes > token_embd.raw.size

      row_slice = Bytes.new(token_embd.raw.to_unsafe + offset, row_bytes, read_only: true)
      Dequant.dequantize(row_slice, t, n_embd)
    end

    def scaled_embedding_lookup(weights : Gemma4Weights, token_id : Int32) : Array(Float32)
      x = embedding_lookup(weights.token_embd, token_id)
      scale = Math.sqrt(weights.hparams.n_embd.to_f64).to_f32
      x.size.times { |i| x[i] *= scale }
      x
    end

    def quant_row_bytes(t : TensorType, n : Int32) : Int32
      case
      when t.f32?  then n * 4
      when t.f16?  then n * 2
      when t.q4_k? then (n // 256) * 144
      when t.q5_k? then (n // 256) * 176
      when t.q6_k? then (n // 256) * 210
      when t.q8_0? then (n // 32) * 34
      else              raise ArgumentError.new("unsupported embedding quant type #{t.name}")
      end
    end
  end
end
