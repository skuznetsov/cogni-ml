require "./qwen35_meta"
require "./qwen35_weights"
require "./quant_matmul"
require "./qwen35_metal"

# Qwen 3.5 / 3.6 CPU reference forward pass.
#
# Purpose: ground-truth correctness path. Not performance-tuned.
# The Metal port (Phases 2+) must match this output to cosine ≥ 0.9999.
#
# Layout conventions:
#   - All activations are Array(Float32), flattened row-major.
#   - Per-head tensors: [n_head, head_dim] flattened as [n_head * head_dim],
#     head h at offset h*head_dim.
#   - KV cache stored as [pos, n_kv_heads, head_dim] flattened.
#   - Autoregressive decode only (one token at a time). Prefill iterates decode.

module ML::GGUF
  module Qwen35CPU
    extend self

    # ─────────────────────────────────────────────────────────────────────
    # Per-sequence state: KV cache for full-attn layers + SSM state for
    # recurrent (DeltaNet) layers.
    # ─────────────────────────────────────────────────────────────────────
    class LayerState
      # Full-attn: KV cache. Populated position-by-position during decode.
      # k_cache[pos * kv_dim + h * head_dim + d], same for v.
      property k_cache : Array(Float32)?
      property v_cache : Array(Float32)?
      property position : Int32 = 0  # number of cached tokens

      # GPU-resident KV cache. Same layout as k_cache/v_cache (position-
      # major). When Metal is available the full-attn path writes K/V
      # straight into these buffers via `contents` (unified memory) and
      # dispatches the Metal attention kernel against them.
      property k_cache_buf : ML::MetalBuffer?
      property v_cache_buf : ML::MetalBuffer?

      # DeltaNet: recurrent state
      # conv_state: [conv_kernel - 1, qkv_stream_dim] — past (kernel-1) token activations
      #   for the 1D conv (padded with zeros at start).
      # ssm_state: [num_v_heads, head_v_dim, head_k_dim] — the matrix-valued recurrent state.
      property conv_state : Array(Float32)?
      property ssm_state : Array(Float32)?

      # GPU-resident SSM state. Kept in parallel to the CPU `ssm_state`
      # but only one of the two is used per sequence — whichever matches
      # the backend dispatched on the first recurrent call. Persists
      # across decode steps so the DeltaNet kernel reads and writes it
      # in place.
      property ssm_state_buf : ML::MetalBuffer?

      def initialize
      end
    end

    class State
      getter layers : Array(LayerState)
      getter max_seq : Int32

      def initialize(hp : Qwen35Hparams, @max_seq : Int32 = 1024)
        @layers = Array(LayerState).new(hp.n_layer) { LayerState.new }
      end
    end

    # ─────────────────────────────────────────────────────────────────────
    # Primitives
    # ─────────────────────────────────────────────────────────────────────

    # RMSNorm: y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
    # Single token (dim-length vector). Returns new Array.
    def rms_norm(x : Array(Float32), w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Array(Float32)
      dim = x.size
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 * x[j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
      Array(Float32).new(dim) { |j| x[j] * inv_rms * w[j] }
    end

    # In-place variant — avoids alloc for hot paths.
    def rms_norm!(x : Array(Float32), w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      dim = x.size
      ss = 0.0_f64
      dim.times { |j| ss += x[j].to_f64 * x[j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
      dim.times { |j| x[j] = x[j] * inv_rms * w[j] }
    end

    # RMSNorm on a per-head slice of a longer vector starting at `offset`,
    # with weight of size `head_dim`. Used for attn_q_norm / attn_k_norm
    # (one set of weights shared across heads).
    def rms_norm_slice!(x : Array(Float32), offset : Int32, len : Int32,
                        w : Array(Float32), eps : Float32 = 1.0e-6_f32) : Nil
      ss = 0.0_f64
      len.times { |j| ss += x[offset + j].to_f64 * x[offset + j].to_f64 }
      inv_rms = (1.0 / Math.sqrt(ss / len.to_f64 + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] = x[offset + j] * inv_rms * w[j] }
    end

    # SiLU / Swish: x * sigmoid(x). Elementwise.
    def silu!(x : Array(Float32)) : Nil
      x.size.times { |i| v = x[i]; x[i] = v / (1.0_f32 + Math.exp(-v)) }
    end

    def silu(x : Float32) : Float32
      x / (1.0_f32 + Math.exp(-x))
    end

    # Sigmoid. Elementwise.
    def sigmoid!(x : Array(Float32)) : Nil
      x.size.times { |i| x[i] = 1.0_f32 / (1.0_f32 + Math.exp(-x[i])) }
    end

    def sigmoid(x : Float32) : Float32
      1.0_f32 / (1.0_f32 + Math.exp(-x))
    end

    # L2 normalize a slice [offset..offset+len).
    # y[i] = x[i] / sqrt(sum(x^2) + eps)
    def l2_norm_slice!(x : Array(Float32), offset : Int32, len : Int32, eps : Float32 = 1.0e-6_f32) : Nil
      ss = 0.0_f64
      len.times { |j| ss += x[offset + j].to_f64 * x[offset + j].to_f64 }
      inv_norm = (1.0 / Math.sqrt(ss + eps.to_f64)).to_f32
      len.times { |j| x[offset + j] = x[offset + j] * inv_norm }
    end

    # Partial M-RoPE (NeoX-style pairing) for a single head-sized vector [head_dim],
    # rotating only the first `n_rot` dims. Keeps dims [n_rot..head_dim) unchanged.
    #
    # NeoX pairing: pair is (dst[i], dst[i + n_rot/2]) for i in 0...n_rot/2
    # Per ggml-metal.metal:4505-4509:
    #   x0 = src[i]; x1 = src[i + n_rot/2]
    #   dst[i]           = x0*cos - x1*sin
    #   dst[i + n_rot/2] = x0*sin + x1*cos
    #
    # For text-only decoding all M-RoPE sections map to the same sequence position,
    # so theta_base = pos and standard RoPE frequencies apply on the first n_rot dims.
    def rope_partial!(x : Array(Float32), head_offset : Int32,
                      n_rot : Int32, head_dim : Int32,
                      pos : Int32, freq_base : Float32) : Nil
      half = n_rot // 2
      half.times do |i|
        freq = 1.0_f32 / (freq_base ** (2.0_f32 * i / n_rot))
        theta = pos.to_f32 * freq
        cos_t = Math.cos(theta)
        sin_t = Math.sin(theta)
        i0 = head_offset + i
        i1 = head_offset + i + half
        x0 = x[i0]; x1 = x[i1]
        x[i0] = x0 * cos_t - x1 * sin_t
        x[i1] = x0 * sin_t + x1 * cos_t
      end
      # dims [n_rot..head_dim) are passed through unchanged (no action needed)
    end

    # Softmax over a slice [offset..offset+len) in place.
    def softmax_slice!(x : Array(Float32), offset : Int32, len : Int32) : Nil
      maxv = x[offset]
      len.times { |i| v = x[offset + i]; maxv = v if v > maxv }
      sum = 0.0_f32
      len.times do |i|
        e = Math.exp(x[offset + i] - maxv)
        x[offset + i] = e
        sum += e
      end
      inv = 1.0_f32 / sum
      len.times { |i| x[offset + i] *= inv }
    end

    # Threshold: only use Metal when the op is large enough to amortize
    # upload/download overhead. Tiny matmuls (e.g. ssm_alpha 4096→32)
    # stay on CPU.
    METAL_QK_MIN_OUT = 256
    METAL_QK_MIN_IN  = 256

    # DeltaNet / GatedDeltaRule state-update + output, shared between the
    # CPU path and the Metal spec reference. `state` and `y` are written
    # in place. `ghead` is the per-head decay multiplier (i.e. caller
    # computes `exp(softplus(...) * ssm_a[h])` up front).
    def delta_net_step!(state : Array(Float32),
                        q_conv : Array(Float32),
                        k_conv : Array(Float32),
                        v_conv : Array(Float32),
                        ghead : Array(Float32),
                        beta : Array(Float32),
                        y : Array(Float32),
                        h_k : Int32, h_v : Int32, s : Int32,
                        scale : Float32) : Nil
      h_v.times do |h|
        k_head = h % h_k
        q_off  = k_head * s
        k_off  = k_head * s
        v_off  = h * s
        st_base = h * s * s
        gh = ghead[h]
        bh = beta[h]

        (s * s).times { |i| state[st_base + i] *= gh }

        sk = Array(Float32).new(s) do |d2|
          row_off = st_base + d2 * s
          a = 0.0_f32
          s.times { |d1| a += state[row_off + d1] * k_conv[k_off + d1] }
          a
        end

        delt = Array(Float32).new(s) { |d2| bh * (v_conv[v_off + d2] - sk[d2]) }

        s.times do |d2|
          row_off = st_base + d2 * s
          dd = delt[d2]
          s.times { |d1| state[row_off + d1] += k_conv[k_off + d1] * dd }
        end

        y_off = h * s
        s.times do |d2|
          row_off = st_base + d2 * s
          acc = 0.0_f32
          s.times { |d1| acc += state[row_off + d1] * q_conv[q_off + d1] }
          y[y_off + d2] = acc * scale
        end
      end
    end

    # Route the DeltaNet step to Metal when available, CPU otherwise.
    # State persists across decode steps on whichever backend owns it:
    # Array(Float32) for CPU, MetalBuffer for GPU — never both at once.
    private def delta_net_step_routed(lstate : LayerState,
                                       q_conv : Array(Float32),
                                       k_conv : Array(Float32),
                                       v_conv : Array(Float32),
                                       ghead : Array(Float32),
                                       beta : Array(Float32),
                                       h_k : Int32, h_v : Int32, s : Int32,
                                       scale : Float32) : Array(Float32)
      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          bytes = (h_v * s * s).to_i64 * sizeof(Float32)
          state_buf = lstate.ssm_state_buf
          if state_buf.nil?
            state_buf = ML::MetalBuffer.new(bytes)
            state_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.ssm_state_buf = state_buf
          end
          return Qwen35Metal.delta_net_step(state_buf, q_conv, k_conv, v_conv,
                                             ghead, beta, h_k, h_v, s, scale)
        end
      {% end %}

      state = lstate.ssm_state ||= Array(Float32).new(h_v * s * s, 0.0_f32)
      y     = Array(Float32).new(h_v * s, 0.0_f32)
      delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y,
                       h_k, h_v, s, scale)
      y
    end

    # Append K, V to the layer's KV cache at `pos` and run gated GQA
    # attention. On Metal: writes K/V into persistent MetalBuffers via
    # unified-memory `contents` and dispatches `Qwen35Metal.attn_decode`.
    # On CPU: allocates the Array caches lazily and runs the per-head
    # dot-product / softmax / V-weighted-sum + sigmoid(gate) multiply.
    # Returns gated attention output of length `n_head * head_dim`.
    private def attn_decode_routed(lstate : LayerState,
                                    q : Array(Float32), gate : Array(Float32),
                                    k : Array(Float32), v : Array(Float32),
                                    pos : Int32, n_head : Int32, n_head_kv : Int32,
                                    head_dim : Int32, heads_per_group : Int32,
                                    kv_dim : Int32, max_seq : Int32,
                                    scale : Float32) : Array(Float32)
      q_dim = n_head * head_dim
      base  = pos * kv_dim

      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available? && ENV["QWEN35_ATTN_CPU"]? != "1"
          bytes = (max_seq * kv_dim).to_i64 * sizeof(Float32)
          k_buf = lstate.k_cache_buf
          v_buf = lstate.v_cache_buf
          if k_buf.nil?
            k_buf = ML::MetalBuffer.new(bytes)
            k_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.k_cache_buf = k_buf
          end
          if v_buf.nil?
            v_buf = ML::MetalBuffer.new(bytes)
            v_buf.contents.as(Pointer(UInt8)).clear(bytes)
            lstate.v_cache_buf = v_buf
          end
          k_ptr = k_buf.contents.as(Pointer(Float32)) + base
          v_ptr = v_buf.contents.as(Pointer(Float32)) + base
          kv_dim.times do |i|
            k_ptr[i] = k[i]
            v_ptr[i] = v[i]
          end
          return Qwen35Metal.attn_decode(q, gate, k_buf, v_buf,
                                          pos, n_head, n_head_kv, head_dim,
                                          heads_per_group, scale)
        end
      {% end %}

      k_cache = lstate.k_cache ||= Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      v_cache = lstate.v_cache ||= Array(Float32).new(max_seq * kv_dim, 0.0_f32)
      kv_dim.times do |i|
        k_cache[base + i] = k[i]
        v_cache[base + i] = v[i]
      end

      attn_o = Array(Float32).new(q_dim, 0.0_f32)
      scores = Array(Float32).new(pos + 1, 0.0_f32)
      n_head.times do |h|
        kv_h  = h // heads_per_group
        q_off = h * head_dim
        (pos + 1).times do |p|
          k_off = p * kv_dim + kv_h * head_dim
          s = 0.0_f32
          head_dim.times { |d| s += q[q_off + d] * k_cache[k_off + d] }
          scores[p] = s * scale
        end
        softmax_slice!(scores, 0, pos + 1)
        out_off = h * head_dim
        (pos + 1).times do |p|
          v_off = p * kv_dim + kv_h * head_dim
          w = scores[p]
          head_dim.times { |d| attn_o[out_off + d] += w * v_cache[v_off + d] }
        end
      end
      q_dim.times { |i| attn_o[i] = attn_o[i] * sigmoid(gate[i]) }
      attn_o
    end

    # Try to run a GEMV (batch=1) on Metal if the type is supported and
    # the op is large enough. Returns `nil` if the call should fall back
    # to CPU.
    private def metal_matvec_or_nil(qw : QuantWeight, x : Array(Float32)) : Array(Float32)?
      {% if flag?(:cpu_only) %}
        nil
      {% else %}
        return nil if qw.out_dim < METAL_QK_MIN_OUT || qw.in_dim < METAL_QK_MIN_IN
        return nil unless Qwen35Metal.available?
        Qwen35Metal.matmul(qw, x, 1)
      {% end %}
    end

    # Quantized matvec — wrapper that routes to QuantMatmul for a single row.
    # result = bias + W @ x, where W is [out_dim, in_dim] stored row-major.
    def qmatvec(qw : QuantWeight, x : Array(Float32), bias : Array(Float32)? = nil) : Array(Float32)
      if (out = metal_matvec_or_nil(qw, x))
        if bias
          out.size.times { |i| out[i] += bias[i] }
        end
        out
      else
        b = bias || Array(Float32).new(qw.out_dim, 0.0_f32)
        QuantMatmul.matmul_add(x, 1, qw.in_dim, qw.raw, qw.type, qw.out_dim, b)
      end
    end

    # Same but without bias (save the allocation when we know bias=0).
    def qmatvec_nobias(qw : QuantWeight, x : Array(Float32)) : Array(Float32)
      if (out = metal_matvec_or_nil(qw, x))
        out
      else
        zero = Array(Float32).new(qw.out_dim, 0.0_f32)
        QuantMatmul.matmul_add(x, 1, qw.in_dim, qw.raw, qw.type, qw.out_dim, zero)
      end
    end

    # ─────────────────────────────────────────────────────────────────────
    # Full-attention layer forward (single-token decode)
    # ─────────────────────────────────────────────────────────────────────
    #
    # Structure (matches llama.cpp qwen35::build_layer_attn + surrounding code):
    #   inpSA = input
    #   cur = RMSNorm(inpSA, attn_norm)
    #   Q_full = attn_q_qw @ cur                           # [2*head_dim*n_head]
    #   Split Q_full per-head into Q [head_dim*n_head] and gate [head_dim*n_head]
    #     (interleaved: per head, first head_dim = Q, next head_dim = gate)
    #   K = attn_k_qw @ cur                                # [head_dim*n_head_kv]
    #   V = attn_v_qw @ cur                                # [head_dim*n_head_kv]
    #   Per-head RMSNorm on Q (attn_q_norm) and K (attn_k_norm)
    #   M-RoPE partial on first rope_dim_count dims of each Q and K head
    #   Write K[pos], V[pos] into KV cache
    #   GQA attention: heads_per_group = n_head / n_head_kv
    #     scores[p] = (Q · K[p]) / sqrt(head_dim)  for p in 0..pos
    #     softmax; out = sum_p scores[p] * V[p]
    #   out *= sigmoid(gate)  (elementwise per position in out)
    #   attn_out = attn_output_qw @ out
    #   cur = inpSA + attn_out   (residual 1)
    #   ffn_res = cur
    #   cur = RMSNorm(cur, post_attention_norm)
    #   gate_ff = silu(ffn_gate_qw @ cur); up = ffn_up_qw @ cur
    #   cur = ffn_down_qw @ (gate_ff * up)
    #   cur = ffn_res + cur   (residual 2)
    #
    # Returns the new hidden state (Array(Float32) size n_embd).
    def forward_full_attn_layer(inpSA : Array(Float32), pos : Int32,
                                 lw : Qwen35FullAttnWeights,
                                 lstate : LayerState,
                                 hp : Qwen35Hparams,
                                 max_seq : Int32) : Array(Float32)
      n_embd     = hp.n_embd
      n_head     = hp.n_head
      n_head_kv  = hp.n_head_kv
      head_dim   = hp.head_dim
      n_ff       = hp.n_ff
      kv_dim     = head_dim * n_head_kv
      q_dim      = head_dim * n_head
      heads_per_group = n_head // n_head_kv

      # 1. attn_norm
      cur = rms_norm(inpSA, lw.attn_norm, hp.rms_eps)

      # 2. Q+gate combined projection
      q_full = qmatvec_nobias(lw.attn_q_qw, cur)  # [2 * head_dim * n_head]

      # 3. Split Q and gate (interleaved per head: [Q_h0, gate_h0, Q_h1, gate_h1, ...])
      q    = Array(Float32).new(q_dim, 0.0_f32)
      gate = Array(Float32).new(q_dim, 0.0_f32)
      n_head.times do |h|
        src_base = h * 2 * head_dim
        dst_base = h * head_dim
        head_dim.times do |d|
          q[dst_base + d]    = q_full[src_base + d]
          gate[dst_base + d] = q_full[src_base + head_dim + d]
        end
      end

      # 4. K, V projections
      k = qmatvec_nobias(lw.attn_k_qw, cur)  # [head_dim * n_head_kv]
      v = qmatvec_nobias(lw.attn_v_qw, cur)  # [head_dim * n_head_kv]

      # 5. Per-head RMSNorm on Q and K (shared weights across heads)
      n_head.times do |h|
        rms_norm_slice!(q, h * head_dim, head_dim, lw.attn_q_norm, hp.rms_eps)
      end
      n_head_kv.times do |h|
        rms_norm_slice!(k, h * head_dim, head_dim, lw.attn_k_norm, hp.rms_eps)
      end

      # 6. M-RoPE partial on first rope_dim_count dims of each Q and K head
      n_head.times do |h|
        rope_partial!(q, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
      end
      n_head_kv.times do |h|
        rope_partial!(k, h * head_dim, hp.rope_dim_count, head_dim, pos, hp.rope_freq_base)
      end

      # 7. Append K, V to cache at current position + 8-9. GQA attention + gate.
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      attn_o = attn_decode_routed(lstate, q, gate, k, v, pos, n_head, n_head_kv,
                                   head_dim, heads_per_group, kv_dim, max_seq, scale)

      # 10. Output projection
      attn_out = qmatvec_nobias(lw.attn_output_qw, attn_o)  # [n_embd]

      # 11. Residual
      inpL2 = Array(Float32).new(n_embd) { |i| inpSA[i] + attn_out[i] }

      # 12. post_attention_norm
      cur2 = rms_norm(inpL2, lw.post_attention_norm, hp.rms_eps)

      # 13. SwiGLU FFN
      gate_ff = qmatvec_nobias(lw.ffn_gate_qw, cur2)  # [n_ff]
      up_ff   = qmatvec_nobias(lw.ffn_up_qw,   cur2)  # [n_ff]
      silu!(gate_ff)
      combined = Array(Float32).new(n_ff) { |i| gate_ff[i] * up_ff[i] }
      ffn_out = qmatvec_nobias(lw.ffn_down_qw, combined)  # [n_embd]

      # 14. Residual
      Array(Float32).new(n_embd) { |i| inpL2[i] + ffn_out[i] }
    end

    # ─────────────────────────────────────────────────────────────────────
    # DeltaNet / GatedDeltaRule recurrent layer forward (single-token decode)
    # ─────────────────────────────────────────────────────────────────────
    #
    # Structure (matches llama.cpp qwen35::build_layer_attn_linear +
    #            delta_net_base::build_delta_net_autoregressive):
    #
    # Per step:
    #   cur = RMSNorm(inpSA, attn_norm)
    #   qkv_mixed = attn_qkv_qw @ cur                         # [qkv_dim = 2*H_k*S_k + H_v*S_v]
    #   z         = attn_gate_qw @ cur                        # [d_inner = H_v * S_v]
    #   alpha     = ssm_alpha_qw @ cur                        # [H_v]
    #   beta      = sigmoid(ssm_beta_qw @ cur)                # [H_v]
    #   a_soft    = softplus(alpha + ssm_dt_bias)             # [H_v]
    #   g         = a_soft * ssm_a                            # [H_v]  (ssm_a already pre-transformed)
    #
    #   conv_input[0..K-2] = conv_state;  conv_input[K-1] = qkv_mixed
    #   new_conv_state     = conv_input[1..K]                 # shift window
    #   conv_out[ch]       = sum_{k=0}^{K-1} conv_input[k,ch] * ssm_conv1d[k,ch]
    #   conv_out           = silu(conv_out)
    #   q_conv = conv_out[0                     .. H_k*S_k)         # [H_k, S_k]
    #   k_conv = conv_out[H_k*S_k               .. 2*H_k*S_k)       # [H_k, S_k]
    #   v_conv = conv_out[2*H_k*S_k             .. 2*H_k*S_k+H_v*S_v)  # [H_v, S_v]
    #   L2-norm each (S_k)-slice of q_conv and k_conv
    #   Repeat q_conv/k_conv so each v-head h_v maps to k-head (h_v % H_k)
    #
    #   Delta rule per v-head h (scale = 1/sqrt(S_k), ghead = exp(g[h]), bhead = beta[h]):
    #     state[h] *= ghead                                    # [S_v × S_v] decay
    #     sk[d2]   = sum_{d1} state[h, d1, d2] * K[h, d1]
    #     delt[d2] = bhead * (V[h, d2] - sk[d2])
    #     state[h, d1, d2] += K[h, d1] * delt[d2]              # outer product add
    #     out[h, d2] = sum_{d1} state[h, d1, d2] * (Q[h, d1] * scale)
    #
    #   norm[h, d] = RMSNorm_per_head(out[h,:], ssm_norm) * silu(z[h, d])
    #   attn = ssm_out_qw @ flatten(norm)                     # [n_embd]
    #
    # Then standard residual + post_attention_norm + SwiGLU FFN + residual.
    #
    # State layout in lstate.ssm_state (Array(Float32)):
    #   state[h * S_v * S_v + d2 * S_v + d1]  (h-major, d2-major, d1 contiguous)
    #
    # Conv state layout in lstate.conv_state (Array(Float32)):
    #   conv[t * qkv_dim + ch]  (time-major, channels minor)
    def forward_recurrent_layer(inpSA : Array(Float32), _pos : Int32,
                                 lw : Qwen35RecurrentWeights,
                                 lstate : LayerState,
                                 hp : Qwen35Hparams,
                                 _max_seq : Int32) : Array(Float32)
      n_embd     = hp.n_embd
      n_ff       = hp.n_ff
      h_k        = hp.ssm_group_count         # num_k_heads
      h_v        = hp.ssm_time_step_rank      # num_v_heads
      s_k        = hp.ssm_state_size          # head_k_dim
      s_v        = hp.ssm_state_size          # head_v_dim (same per qwen35)
      d_inner    = hp.ssm_inner_size          # H_v * S_v
      qkv_dim    = 2 * h_k * s_k + h_v * s_v
      conv_k     = hp.ssm_conv_kernel         # typically 4
      heads_per_k = h_v // h_k

      # 1. attn_norm
      cur = rms_norm(inpSA, lw.attn_norm, hp.rms_eps)

      # 2. QKV-mixed projection
      qkv_mixed = qmatvec_nobias(lw.attn_qkv_qw, cur)  # [qkv_dim]

      # 3. z (gate for SiLU-gated RMSNorm after delta net)
      z = qmatvec_nobias(lw.attn_gate_qw, cur)         # [d_inner = h_v * s_v]

      # 4. alpha, beta
      alpha = qmatvec_nobias(lw.ssm_alpha_qw, cur)     # [h_v]
      beta  = qmatvec_nobias(lw.ssm_beta_qw,  cur)     # [h_v]
      h_v.times { |i| beta[i] = sigmoid(beta[i]) }

      # 5. gate per head (g[h] = softplus(alpha[h] + ssm_dt_bias[h]) * ssm_a[h])
      # ssm_a is pre-transformed (-A_log.exp() in llama.cpp), multiply directly
      g = Array(Float32).new(h_v) do |i|
        xi = alpha[i] + lw.ssm_dt_bias[i]
        sp = xi > 20.0_f32 ? xi : Math.log(1.0_f32 + Math.exp(xi)).to_f32
        sp * lw.ssm_a[i]
      end

      # 6. Conv state (lazy alloc). Layout: conv[t*qkv_dim + ch], t in 0..K-2 (K=conv_k)
      conv_state = lstate.conv_state ||= Array(Float32).new((conv_k - 1) * qkv_dim, 0.0_f32)

      # 7. Convolution output for current token.
      # GGUF ssm_conv1d dims=[K, qkv_dim] with dims[0]=K innermost → layout is conv1d[ch*K + t].
      # conv_state is OUR internal buffer (layout [t*qkv_dim + ch]) — unchanged.
      #    conv_out[ch] = sum_{k=0}^{K-2} conv_state[k*qkv_dim+ch] * conv1d[ch*K + k]
      #                 + qkv_mixed[ch]                          * conv1d[ch*K + (K-1)]
      conv_out = Array(Float32).new(qkv_dim) do |ch|
        acc = 0.0_f32
        w_base = ch * conv_k
        (conv_k - 1).times do |t|
          acc += conv_state[t * qkv_dim + ch] * lw.ssm_conv1d[w_base + t]
        end
        acc += qkv_mixed[ch] * lw.ssm_conv1d[w_base + (conv_k - 1)]
        acc
      end

      # 8. Update conv_state: shift window. new_state[t] = old_state[t+1], last = qkv_mixed
      (conv_k - 2).times do |t|
        src_off = (t + 1) * qkv_dim
        dst_off = t * qkv_dim
        qkv_dim.times { |ch| conv_state[dst_off + ch] = conv_state[src_off + ch] }
      end
      last_off = (conv_k - 2) * qkv_dim
      qkv_dim.times { |ch| conv_state[last_off + ch] = qkv_mixed[ch] }

      # 9. SiLU on conv output
      silu!(conv_out)

      # 10. Split conv_out into q, k, v
      q_conv = Array(Float32).new(h_k * s_k) { |i| conv_out[i] }
      k_conv = Array(Float32).new(h_k * s_k) { |i| conv_out[h_k * s_k + i] }
      v_conv = Array(Float32).new(h_v * s_v) { |i| conv_out[2 * h_k * s_k + i] }

      # 11. L2-norm each S_k-slice of q_conv and k_conv
      h_k.times do |h|
        l2_norm_slice!(q_conv, h * s_k, s_k, hp.rms_eps)
        l2_norm_slice!(k_conv, h * s_k, s_k, hp.rms_eps)
      end

      # 12. Delta rule state update + output computation
      scale = (1.0 / Math.sqrt(s_k.to_f64)).to_f32
      # Convert g[h] to ghead = exp(g[h]) inline; kernel and reference share
      # `delta_net_step!` below, which expects the already-exp'd decay.
      ghead = Array(Float32).new(h_v) { |h| Math.exp(g[h].to_f64).to_f32 }

      y = delta_net_step_routed(lstate, q_conv, k_conv, v_conv, ghead, beta,
                                  h_k, h_v, s_k, scale)

      # 13. Gated RMSNorm: norm[h,d] = RMSNorm_per_head(y[h,:], ssm_norm) * silu(z[h,d])
      h_v.times do |h|
        rms_norm_slice!(y, h * s_v, s_v, lw.ssm_norm, hp.rms_eps)
      end
      (h_v * s_v).times { |i| y[i] = y[i] * silu(z[i]) }

      # 14. Output projection (ssm_out)
      attn_out = qmatvec_nobias(lw.ssm_out_qw, y)  # [n_embd]

      # 15. Residual 1
      inpL2 = Array(Float32).new(n_embd) { |i| inpSA[i] + attn_out[i] }

      # 16. Post-attention norm
      cur2 = rms_norm(inpL2, lw.post_attention_norm, hp.rms_eps)

      # 17. SwiGLU FFN
      gate_ff = qmatvec_nobias(lw.ffn_gate_qw, cur2)
      up_ff   = qmatvec_nobias(lw.ffn_up_qw,   cur2)
      silu!(gate_ff)
      combined = Array(Float32).new(n_ff) { |i| gate_ff[i] * up_ff[i] }
      ffn_out = qmatvec_nobias(lw.ffn_down_qw, combined)

      # 18. Residual 2
      Array(Float32).new(n_embd) { |i| inpL2[i] + ffn_out[i] }
    end

    # ─────────────────────────────────────────────────────────────────────
    # Full decoder forward (single-token autoregressive)
    # ─────────────────────────────────────────────────────────────────────
    #
    # Given a token id and the current per-sequence state + position, compute
    # the logits vector [vocab_size] for the NEXT token.
    #
    # Steps:
    #   x = embedding_lookup(token_embd, token_id)   # [n_embd]
    #   for il in 0 ... n_layer:
    #     if full_attention?(il):
    #       x = forward_full_attn_layer(x, pos, layers[il], state.layers[il], hp, state.max_seq)
    #     else:
    #       x = forward_recurrent_layer(x, pos, layers[il], state.layers[il], hp, state.max_seq)
    #   x = RMSNorm(x, output_norm)
    #   logits = output @ x                          # [vocab_size]
    #
    # Note: full-attention layers internally update their KV cache; recurrent
    # layers update conv_state / ssm_state. Caller must bump state.layers[il].position
    # (for full-attn it's tracked by pos arg anyway; position field unused in
    # current impl but reserved for future fused paths).
    def forward(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                state : State) : Array(Float32)
      hp = weights.hparams
      max_seq = state.max_seq

      x = embedding_lookup(weights.token_embd, token_id)

      weights.layers.each_with_index do |lw, il|
        case lw
        in Qwen35FullAttnWeights
          x = forward_full_attn_layer(x, pos, lw, state.layers[il], hp, max_seq)
        in Qwen35RecurrentWeights
          x = forward_recurrent_layer(x, pos, lw, state.layers[il], hp, max_seq)
        end
      end

      rms_norm!(x, weights.output_norm, hp.rms_eps)
      qmatvec_nobias(weights.output, x)
    end

    # Embedding lookup for a single token id → Array(Float32)[n_embd].
    # token_embd is QuantWeight with dims [n_embd, vocab_size] (row = one embedding).
    def embedding_lookup(token_embd : QuantWeight, token_id : Int32) : Array(Float32)
      n_embd = token_embd.in_dim
      raise "embedding: token_id #{token_id} out of range" if token_id < 0 || token_id >= token_embd.out_dim

      # Dequantize just the one row. row_bytes depends on quant type.
      # Use slice into raw and dequantize full block-aligned segment.
      t = token_embd.type
      # All K-quants: 256 elts per block, so one row of 4096 elts = 16 blocks.
      # Row bytes = (n_embd / 256) * block_bytes
      if n_embd % 256 != 0 && !(t.f32? || t.f16?)
        raise "embedding: n_embd #{n_embd} not divisible by 256 for K-quant"
      end
      row_bytes = case
                  when t.f32?     then n_embd * 4
                  when t.f16?     then n_embd * 2
                  when t.q4_k?    then (n_embd // 256) * 144
                  when t.q5_k?    then (n_embd // 256) * 176
                  when t.q6_k?    then (n_embd // 256) * 210
                  else raise "embedding: unsupported quant type #{t.name}"
                  end
      offset = token_id.to_i64 * row_bytes.to_i64
      row_slice = Bytes.new(token_embd.raw.to_unsafe + offset, row_bytes, read_only: true)
      Dequant.dequantize(row_slice, t, n_embd)
    end
  end
end
