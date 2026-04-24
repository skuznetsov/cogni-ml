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
    # Keep prompt chunks large enough to avoid CPU-side boundary overhead while
    # preserving an env override for small-memory experiments.
    DEFAULT_PREFILL_CHUNK_SIZE = 1024

    # ─────────────────────────────────────────────────────────────────────
    # Per-sequence state: KV cache for full-attn layers + SSM state for
    # recurrent (DeltaNet) layers.
    # ─────────────────────────────────────────────────────────────────────
    class LayerState
      # Full-attn: KV cache. Populated position-by-position during decode.
      # k_cache[pos * kv_dim + h * head_dim + d], same for v.
      property k_cache : Array(Float32)?
      property v_cache : Array(Float32)?
      property position : Int32 = 0 # number of cached tokens

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
      property conv_state_buf : ML::MetalBuffer?
      property ssm_state : Array(Float32)?

      # GPU-resident SSM state. Kept in parallel to the CPU `ssm_state`
      # but only one of the two is used per sequence — whichever matches
      # the backend dispatched on the first recurrent call. Persists
      # across decode steps so the DeltaNet kernel reads and writes it
      # in place.
      property ssm_state_buf : ML::MetalBuffer?

      def initialize
      end

      # Deep-copy per-layer decode state. This is the minimal primitive
      # needed by exact speculative verification: each branch must mutate
      # its own KV/SSM buffers, not alias the parent sequence.
      def fork : LayerState
        copy = LayerState.new
        copy.copy_from!(self)
        copy
      end

      def copy_from!(src : LayerState) : Nil
        @position = src.position
        @k_cache = src.k_cache.try(&.dup)
        @v_cache = src.v_cache.try(&.dup)
        @conv_state = src.conv_state.try(&.dup)
        @ssm_state = src.ssm_state.try(&.dup)
        @k_cache_buf = copy_buffer_from!(@k_cache_buf, src.k_cache_buf)
        @v_cache_buf = copy_buffer_from!(@v_cache_buf, src.v_cache_buf)
        @conv_state_buf = copy_buffer_from!(@conv_state_buf, src.conv_state_buf)
        @ssm_state_buf = copy_buffer_from!(@ssm_state_buf, src.ssm_state_buf)
      end

      private def copy_buffer_from!(dst : ML::MetalBuffer?, src : ML::MetalBuffer?) : ML::MetalBuffer?
        return nil unless src_buf = src

        dst_buf = dst
        if dst_buf.nil? || dst_buf.size != src_buf.size || dst_buf.storage_mode != src_buf.storage_mode
          dst_buf = ML::MetalBuffer.new(src_buf.size, src_buf.storage_mode)
        end
        dst_buf.copy_from(src_buf, src_buf.size)
        dst_buf
      end
    end

    class State
      getter layers : Array(LayerState)
      getter max_seq : Int32

      def initialize(hp : Qwen35Hparams, @max_seq : Int32 = 1024)
        @layers = Array(LayerState).new(hp.n_layer) { LayerState.new }
      end

      protected def initialize(@layers : Array(LayerState), @max_seq : Int32)
      end

      def fork : State
        State.new(@layers.map(&.fork), @max_seq)
      end

      def copy_from!(src : State) : Nil
        raise ArgumentError.new("max_seq mismatch: #{@max_seq} != #{src.max_seq}") unless @max_seq == src.max_seq
        raise ArgumentError.new("layer count mismatch: #{@layers.size} != #{src.layers.size}") unless @layers.size == src.layers.size

        @layers.each_with_index do |layer, i|
          layer.copy_from!(src.layers[i])
        end
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

    private def metal_qw_supported?(qw : QuantWeight) : Bool
      qw.type.q4_k? || qw.type.q5_k? || qw.type.q6_k?
    end

    private def metal_qw_eligible?(qw : QuantWeight) : Bool
      qw.out_dim >= METAL_QK_MIN_OUT && qw.in_dim >= METAL_QK_MIN_IN
    end

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
        q_off = k_head * s
        k_off = k_head * s
        v_off = h * s
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
      y = Array(Float32).new(h_v * s, 0.0_f32)
      delta_net_step!(state, q_conv, k_conv, v_conv, ghead, beta, y,
        h_k, h_v, s, scale)
      y
    end

    # Recurrent Metal fast path:
    #   delta_net_step -> RMSNorm(y)*silu(z) -> ssm_out projection
    # in one command buffer. Returns nil when disabled or unsupported.
    private def delta_net_project_routed(lstate : LayerState,
                                         q_conv : Array(Float32),
                                         k_conv : Array(Float32),
                                         v_conv : Array(Float32),
                                         ghead : Array(Float32),
                                         beta : Array(Float32),
                                         z : Array(Float32),
                                         ssm_norm : Array(Float32),
                                         out_qw : QuantWeight,
                                         h_k : Int32, h_v : Int32, s : Int32,
                                         scale : Float32,
                                         eps : Float32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_DN_FUSE_OFF"]? == "1"
        return nil unless out_qw.type.q4_k? || out_qw.type.q5_k? || out_qw.type.q6_k?
        return nil unless Qwen35Metal.available?

        bytes = (h_v * s * s).to_i64 * sizeof(Float32)
        state_buf = lstate.ssm_state_buf
        if state_buf.nil?
          state_buf = ML::MetalBuffer.new(bytes)
          state_buf.contents.as(Pointer(UInt8)).clear(bytes)
          lstate.ssm_state_buf = state_buf
        end
        return Qwen35Metal.delta_net_project(
          state_buf,
          q_conv, k_conv, v_conv, ghead, beta, z, ssm_norm, out_qw,
          h_k, h_v, s, scale, eps,
        )
      {% else %}
        nil
      {% end %}
    end

    private def recurrent_attn_project_routed(lstate : LayerState,
                                              cur : Array(Float32),
                                              lw : Qwen35RecurrentWeights,
                                              hp : Qwen35Hparams) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_RECURRENT_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        return nil unless metal_qw_supported?(lw.attn_qkv_qw) &&
                          metal_qw_supported?(lw.attn_gate_qw) &&
                          metal_qw_supported?(lw.ssm_alpha_qw) &&
                          metal_qw_supported?(lw.ssm_beta_qw) &&
                          metal_qw_supported?(lw.ssm_out_qw)

        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
        conv_buf = lstate.conv_state_buf
        if conv_buf.nil?
          conv_buf = ML::MetalBuffer.new(conv_bytes)
          conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
          lstate.conv_state_buf = conv_buf
        end

        ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
        ssm_buf = lstate.ssm_state_buf
        if ssm_buf.nil?
          ssm_buf = ML::MetalBuffer.new(ssm_bytes)
          ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
          lstate.ssm_state_buf = ssm_buf
        end

        return Qwen35Metal.recurrent_attn_project(
          cur, conv_buf, ssm_buf,
          lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw,
          lw.ssm_conv1d, lw.ssm_dt_bias, lw.ssm_a, lw.ssm_norm, lw.ssm_out_qw,
          hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size, hp.ssm_conv_kernel, hp.rms_eps,
        )
      {% else %}
        nil
      {% end %}
    end

    # Fused recurrent-layer GPU route:
    #   recurrent attention -> residual add + post-attn RMSNorm -> FFN -> residual add
    private def recurrent_layer_project_routed(inpSA : Array(Float32),
                                               cur : Array(Float32),
                                               lstate : LayerState,
                                               lw : Qwen35RecurrentWeights,
                                               hp : Qwen35Hparams) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_RECURRENT_LAYER_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_qkv_qw) &&
                    metal_qw_supported?(lw.attn_gate_qw) &&
                    metal_qw_supported?(lw.ssm_alpha_qw) &&
                    metal_qw_supported?(lw.ssm_beta_qw) &&
                    metal_qw_supported?(lw.ssm_out_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size
        conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
        conv_buf = lstate.conv_state_buf
        if conv_buf.nil?
          conv_buf = ML::MetalBuffer.new(conv_bytes)
          conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
          lstate.conv_state_buf = conv_buf
        end

        ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
        ssm_buf = lstate.ssm_state_buf
        if ssm_buf.nil?
          ssm_buf = ML::MetalBuffer.new(ssm_bytes)
          ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
          lstate.ssm_state_buf = ssm_buf
        end

        return Qwen35Metal.recurrent_layer_project(
          inpSA, cur, conv_buf, ssm_buf,
          lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw,
          lw.ssm_conv1d, lw.ssm_dt_bias, lw.ssm_a, lw.ssm_norm, lw.ssm_out_qw,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          hp.ssm_group_count, hp.ssm_time_step_rank, hp.ssm_state_size, hp.ssm_conv_kernel, hp.rms_eps,
        )
      {% else %}
        nil
      {% end %}
    end

    # Fused FFN route on Metal:
    #   gate_proj + up_proj -> swiglu -> down_proj
    private def ffn_project_routed(x : Array(Float32),
                                   gate_qw : QuantWeight,
                                   up_qw : QuantWeight,
                                   down_qw : QuantWeight) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FFN_FUSE_OFF"]? == "1"
        return nil unless metal_qw_supported?(gate_qw) && metal_qw_supported?(up_qw) && metal_qw_supported?(down_qw)
        return nil unless Qwen35Metal.available?
        return Qwen35Metal.ffn_project(x, gate_qw, up_qw, down_qw)
      {% else %}
        nil
      {% end %}
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
      base = pos * kv_dim

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
        kv_h = h // heads_per_group
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

    # Same routing as `attn_decode_routed`, but keeps the attention output on
    # GPU and immediately runs the output projection there. This is an
    # optimization-only path for full-attention decode; CPU fallback remains
    # the source of truth.
    private def attn_decode_project_routed(lstate : LayerState,
                                           q : Array(Float32), gate : Array(Float32),
                                           k : Array(Float32), v : Array(Float32),
                                           out_qw : QuantWeight,
                                           pos : Int32, n_head : Int32, n_head_kv : Int32,
                                           head_dim : Int32, heads_per_group : Int32,
                                           kv_dim : Int32, max_seq : Int32,
                                           scale : Float32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_ATTN_CPU"]? == "1"
        return nil if ENV["QWEN35_ATTN_FUSE_OFF"]? == "1"
        return nil unless out_qw.type.q4_k? || out_qw.type.q5_k? || out_qw.type.q6_k?
        return nil unless Qwen35Metal.available?

        base = pos * kv_dim
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
        return Qwen35Metal.attn_decode_project(
          q, gate, k_buf, v_buf, out_qw,
          pos, n_head, n_head_kv, head_dim, heads_per_group, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    # Full-attention GPU route:
    #   qkv projections -> split/norm/rope -> kv write -> attn -> out proj
    private def full_attn_layer_project_routed(inpSA : Array(Float32),
                                               cur : Array(Float32),
                                               lstate : LayerState,
                                               lw : Qwen35FullAttnWeights,
                                               hp : Qwen35Hparams,
                                               pos : Int32,
                                               heads_per_group : Int32,
                                               kv_dim : Int32,
                                               max_seq : Int32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FULL_LAYER_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

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

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        return Qwen35Metal.full_attn_layer_project(
          inpSA, cur,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          pos, hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          heads_per_group, hp.rope_freq_base, hp.rms_eps, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    private def full_attn_layer_chunk_project_routed(inp : Array(Float32),
                                                     n_tokens : Int32,
                                                     start_pos : Int32,
                                                     lstate : LayerState,
                                                     lw : Qwen35FullAttnWeights,
                                                     hp : Qwen35Hparams,
                                                     max_seq : Int32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
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

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        return Qwen35Metal.full_attn_layer_chunk_project(
          inp,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_norm, lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          start_pos, n_tokens,
          hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    private def final_full_attn_layer_chunk_last_routed(inp : Array(Float32),
                                                        n_tokens : Int32,
                                                        start_pos : Int32,
                                                        lstate : LayerState,
                                                        lw : Qwen35FullAttnWeights,
                                                        hp : Qwen35Hparams,
                                                        max_seq : Int32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FINAL_FULL_LAST_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        supported = metal_qw_supported?(lw.attn_q_qw) &&
                    metal_qw_supported?(lw.attn_k_qw) &&
                    metal_qw_supported?(lw.attn_v_qw) &&
                    metal_qw_supported?(lw.attn_output_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
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

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        Qwen35Metal.full_attn_layer_chunk_project_last(
          inp,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_norm, lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf,
          lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
          start_pos, n_tokens,
          hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    private def full_attn_then_recurrent_chunk_project_many_routed(inp : Array(Float32),
                                                                   n_tokens : Int32,
                                                                   start_pos : Int32,
                                                                   state : State,
                                                                   weights : Qwen35Weights,
                                                                   il : Int32,
                                                                   hp : Qwen35Hparams,
                                                                   max_seq : Int32) : {Array(Float32), Int32}?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_PREFILL_FUSE_FULL_REC_OFF"]? == "1"
        return nil if ENV["QWEN35_FULL_PREFILL_CHUNK_OFF"]? == "1"
        return nil if ENV["QWEN35_PREFILL_REC_RUN_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        full_lw = weights.layers[il].as?(Qwen35FullAttnWeights)
        return nil unless full_lw

        run_start = il + 1
        return nil if run_start >= weights.layers.size
        run_end = run_start
        while run_end < weights.layers.size
          break unless weights.layers[run_end].is_a?(Qwen35RecurrentWeights)
          run_end += 1
        end
        return nil if run_end == run_start

        supported = metal_qw_supported?(full_lw.attn_q_qw) &&
                    metal_qw_supported?(full_lw.attn_k_qw) &&
                    metal_qw_supported?(full_lw.attn_v_qw) &&
                    metal_qw_supported?(full_lw.attn_output_qw) &&
                    metal_qw_supported?(full_lw.ffn_gate_qw) &&
                    metal_qw_supported?(full_lw.ffn_up_qw) &&
                    metal_qw_supported?(full_lw.ffn_down_qw)
        return nil unless supported

        rec_layers = [] of Qwen35RecurrentWeights
        conv_bufs = [] of ML::MetalBuffer
        ssm_bufs = [] of ML::MetalBuffer
        h_k = hp.ssm_group_count
        h_v = hp.ssm_time_step_rank
        s = hp.ssm_state_size
        qkv_dim = 2 * h_k * s + h_v * s
        conv_k = hp.ssm_conv_kernel

        j = run_start
        while j < run_end
          rw = weights.layers[j].as(Qwen35RecurrentWeights)
          supported &&= metal_qw_supported?(rw.attn_qkv_qw) &&
                        metal_qw_supported?(rw.attn_gate_qw) &&
                        metal_qw_supported?(rw.ssm_alpha_qw) &&
                        metal_qw_supported?(rw.ssm_beta_qw) &&
                        metal_qw_supported?(rw.ssm_out_qw) &&
                        metal_qw_supported?(rw.ffn_gate_qw) &&
                        metal_qw_supported?(rw.ffn_up_qw) &&
                        metal_qw_supported?(rw.ffn_down_qw)
          rec_layers << rw

          lstate = state.layers[j]
          conv_bytes = ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32)
          conv_buf = lstate.conv_state_buf
          if conv_buf.nil?
            conv_buf = ML::MetalBuffer.new(conv_bytes)
            if conv_state = lstate.conv_state
              conv_buf.write(conv_state)
            else
              conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
            end
            lstate.conv_state_buf = conv_buf
          end
          conv_bufs << conv_buf

          ssm_bytes = (h_v * s * s).to_i64 * sizeof(Float32)
          ssm_buf = lstate.ssm_state_buf
          if ssm_buf.nil?
            ssm_buf = ML::MetalBuffer.new(ssm_bytes)
            if ssm_state = lstate.ssm_state
              ssm_buf.write(ssm_state)
            else
              ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
            end
            lstate.ssm_state_buf = ssm_buf
          end
          ssm_bufs << ssm_buf
          j += 1
        end
        return nil unless supported

        kv_dim = hp.head_dim * hp.n_head_kv
        bytes = (max_seq * kv_dim).to_i64 * sizeof(Float32)
        full_state = state.layers[il]
        k_buf = full_state.k_cache_buf
        v_buf = full_state.v_cache_buf
        if k_buf.nil?
          k_buf = ML::MetalBuffer.new(bytes)
          k_buf.contents.as(Pointer(UInt8)).clear(bytes)
          full_state.k_cache_buf = k_buf
        end
        if v_buf.nil?
          v_buf = ML::MetalBuffer.new(bytes)
          v_buf.contents.as(Pointer(UInt8)).clear(bytes)
          full_state.v_cache_buf = v_buf
        end

        scale = (1.0 / Math.sqrt(hp.head_dim.to_f64)).to_f32
        out = Qwen35Metal.full_attn_then_recurrent_chunk_project_many(
          inp,
          full_lw.attn_q_qw, full_lw.attn_k_qw, full_lw.attn_v_qw,
          full_lw.attn_norm, full_lw.attn_q_norm, full_lw.attn_k_norm,
          full_lw.attn_output_qw, k_buf, v_buf, full_lw.post_attention_norm,
          full_lw.ffn_gate_qw, full_lw.ffn_up_qw, full_lw.ffn_down_qw,
          start_pos, n_tokens,
          hp.n_head, hp.n_head_kv, hp.head_dim, hp.rope_dim_count,
          hp.n_head // hp.n_head_kv, hp.rope_freq_base, hp.rms_eps, scale,
          conv_bufs, ssm_bufs, rec_layers, h_k, h_v, s, conv_k)
        out ? {out, run_end} : nil
      {% else %}
        nil
      {% end %}
    end

    # Full-attention GPU route:
    #   qkv projections -> split/norm/rope -> kv write -> attn -> out proj
    private def full_attn_project_routed(lstate : LayerState,
                                         cur : Array(Float32),
                                         lw : Qwen35FullAttnWeights,
                                         hp : Qwen35Hparams,
                                         pos : Int32,
                                         heads_per_group : Int32,
                                         kv_dim : Int32,
                                         max_seq : Int32,
                                         scale : Float32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_FULL_ATTN_FUSE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        return nil unless metal_qw_supported?(lw.attn_q_qw) &&
                          metal_qw_supported?(lw.attn_k_qw) &&
                          metal_qw_supported?(lw.attn_v_qw) &&
                          metal_qw_supported?(lw.attn_output_qw)

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

        return Qwen35Metal.full_attn_project(
          cur,
          lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw,
          lw.attn_q_norm, lw.attn_k_norm, lw.attn_output_qw,
          k_buf, v_buf, pos, hp.n_head, hp.n_head_kv, hp.head_dim,
          hp.rope_dim_count, heads_per_group, hp.rope_freq_base, scale,
        )
      {% else %}
        nil
      {% end %}
    end

    # Try to run a GEMV (batch=1) on Metal if the type is supported and
    # the op is large enough. Returns `nil` if the call should fall back
    # to CPU.
    private def metal_matvec_or_nil(qw : QuantWeight, x : Array(Float32)) : Array(Float32)?
      {% if flag?(:cpu_only) %}
        nil
      {% else %}
        unless metal_qw_eligible?(qw)
          Qwen35Metal::Profile.bump_cpu_fallback
          return nil
        end
        unless Qwen35Metal.available?
          Qwen35Metal::Profile.bump_cpu_fallback
          return nil
        end
        Qwen35Metal.matmul(qw, x, 1)
      {% end %}
    end

    # Batched matvec — runs all qws against the same input in ONE Metal
    # command buffer (one commit+wait, one readback of all outputs).
    # Mixed batches are split: large eligible qws stay batched on Metal,
    # while genuinely tiny qws (e.g. alpha/beta) fall back to CPU only
    # for those slots.
    def qmatvec_many(qws : Array(QuantWeight), x : Array(Float32)) : Array(Array(Float32))
      return [] of Array(Float32) if qws.empty?
      results = Array(Array(Float32)?).new(qws.size, nil)
      {% unless flag?(:cpu_only) %}
        if ENV["QWEN35_BATCH_OFF"]? != "1"
          eligible_idx = Array(Int32).new
          eligible_qws = Array(QuantWeight).new
          qws.each_with_index do |qw, i|
            if metal_qw_supported?(qw)
              eligible_idx << i.to_i32
              eligible_qws << qw
            end
          end
          if !eligible_qws.empty? && Qwen35Metal.available?
            if gpu_results = Qwen35Metal.matmul_many(eligible_qws, x)
              eligible_idx.each_with_index do |orig_i, gpu_i|
                results[orig_i] = gpu_results[gpu_i]
              end
            end
          end
        end
      {% end %}
      qws.each_with_index do |qw, i|
        results[i] ||= qmatvec_nobias(qw, x)
      end
      results.map(&.not_nil!)
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

    # Batched quantized matmul for token-major activations:
    #   x:   [batch, in_dim]
    #   out: [batch, out_dim]
    #
    # This is the projection building block for layerwise prefill. It uses the
    # existing Metal batch path when available and falls back to the CPU fused
    # matmul without changing numerics.
    private def qmatmul_nobias(qw : QuantWeight, x : Array(Float32), batch : Int32) : Array(Float32)
      raise ArgumentError.new("qmatmul_nobias batch must be positive") unless batch > 0
      raise ArgumentError.new("qmatmul_nobias x size mismatch: expected #{batch * qw.in_dim}, got #{x.size}") unless x.size == batch * qw.in_dim

      if metal_qw_supported?(qw) && Qwen35Metal.available?
        if (gpu_out = Qwen35Metal.matmul(qw, x, batch))
          return gpu_out
        end
      end

      zero = Array(Float32).new(qw.out_dim, 0.0_f32)
      QuantMatmul.matmul_add(x, batch, qw.in_dim, qw.raw, qw.type, qw.out_dim, zero)
    end

    private def rms_norm_rows(x : Array(Float32), rows : Int32, dim : Int32,
                              w : Array(Float32), eps : Float32) : Array(Float32)
      raise ArgumentError.new("rms_norm_rows x size mismatch") unless x.size == rows * dim
      out = Array(Float32).new(x.size, 0.0_f32)
      rows.times do |r|
        base = r * dim
        ss = 0.0_f64
        dim.times { |j| ss += x[base + j].to_f64 * x[base + j].to_f64 }
        inv_rms = (1.0 / Math.sqrt(ss / dim.to_f64 + eps.to_f64)).to_f32
        dim.times { |j| out[base + j] = x[base + j] * inv_rms * w[j] }
      end
      out
    end

    private def output_project_routed(x : Array(Float32),
                                      norm_weight : Array(Float32),
                                      out_qw : QuantWeight,
                                      eps : Float32) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_HEAD_FUSE_OFF"]? == "1"
        return nil unless metal_qw_supported?(out_qw)
        return nil unless Qwen35Metal.available?
        Qwen35Metal.rmsnorm_project(x, norm_weight, out_qw, eps)
      {% else %}
        nil
      {% end %}
    end

    private def output_project_top1_routed(x : Array(Float32),
                                           norm_weight : Array(Float32),
                                           out_qw : QuantWeight,
                                           eps : Float32) : {Int32, Float32}?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_HEAD_TOP1_FUSED"]? == "0"
        return nil unless metal_qw_supported?(out_qw)
        return nil unless Qwen35Metal.available?
        if packed = Qwen35Metal.rmsnorm_project_top1(x, norm_weight, out_qw, eps)
          return {packed[0].to_i32, packed[1]} if packed.size == 2
        end
      {% end %}
      nil
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
      n_embd = hp.n_embd
      n_head = hp.n_head
      n_head_kv = hp.n_head_kv
      head_dim = hp.head_dim
      n_ff = hp.n_ff
      kv_dim = head_dim * n_head_kv
      q_dim = head_dim * n_head
      heads_per_group = n_head // n_head_kv

      # 1. attn_norm
      cur = rms_norm(inpSA, lw.attn_norm, hp.rms_eps)

      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      fused_layer = full_attn_layer_project_routed(inpSA, cur, lstate, lw, hp, pos, heads_per_group, kv_dim, max_seq)
      return fused_layer if fused_layer

      attn_out = full_attn_project_routed(
        lstate, cur, lw, hp, pos, heads_per_group, kv_dim, max_seq, scale,
      )
      unless attn_out
        # 2-4. Batched Q+gate/K/V projections (all from same `cur` → one sync)
        qkv_outs = qmatvec_many([lw.attn_q_qw, lw.attn_k_qw, lw.attn_v_qw], cur)
        q_full = qkv_outs[0] # [2 * head_dim * n_head]
        k = qkv_outs[1]      # [head_dim * n_head_kv]
        v = qkv_outs[2]      # [head_dim * n_head_kv]

        # 3. Split Q and gate (interleaved per head: [Q_h0, gate_h0, Q_h1, gate_h1, ...])
        q = Array(Float32).new(q_dim, 0.0_f32)
        gate = Array(Float32).new(q_dim, 0.0_f32)
        n_head.times do |h|
          src_base = h * 2 * head_dim
          dst_base = h * head_dim
          head_dim.times do |d|
            q[dst_base + d] = q_full[src_base + d]
            gate[dst_base + d] = q_full[src_base + head_dim + d]
          end
        end

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
        attn_out = attn_decode_project_routed(
          lstate, q, gate, k, v, lw.attn_output_qw,
          pos, n_head, n_head_kv, head_dim, heads_per_group, kv_dim, max_seq, scale,
        )
        unless attn_out
          attn_o = attn_decode_routed(lstate, q, gate, k, v, pos, n_head, n_head_kv,
            head_dim, heads_per_group, kv_dim, max_seq, scale)
          # 10. Output projection
          attn_out = qmatvec_nobias(lw.attn_output_qw, attn_o) # [n_embd]
        end
      end

      # 11. Residual
      inpL2 = Array(Float32).new(n_embd) { |i| inpSA[i] + attn_out.not_nil![i] }

      # 12. post_attention_norm
      cur2 = rms_norm(inpL2, lw.post_attention_norm, hp.rms_eps)

      # 13. SwiGLU FFN
      ffn_out = ffn_project_routed(cur2, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw)
      unless ffn_out
        gu = qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], cur2)
        gate_ff = gu[0] # [n_ff]
        up_ff = gu[1]   # [n_ff]
        silu!(gate_ff)
        combined = Array(Float32).new(n_ff) { |i| gate_ff[i] * up_ff[i] }
        ffn_out = qmatvec_nobias(lw.ffn_down_qw, combined) # [n_embd]
      end

      # 14. Residual
      Array(Float32).new(n_embd) { |i| inpL2[i] + ffn_out.not_nil![i] }
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
      n_embd = hp.n_embd
      n_ff = hp.n_ff
      h_k = hp.ssm_group_count    # num_k_heads
      h_v = hp.ssm_time_step_rank # num_v_heads
      s_k = hp.ssm_state_size     # head_k_dim
      s_v = hp.ssm_state_size     # head_v_dim (same per qwen35)
      d_inner = hp.ssm_inner_size # H_v * S_v
      qkv_dim = 2 * h_k * s_k + h_v * s_v
      conv_k = hp.ssm_conv_kernel # typically 4
      heads_per_k = h_v // h_k

      # 1. attn_norm
      cur = rms_norm(inpSA, lw.attn_norm, hp.rms_eps)

      fused_layer = recurrent_layer_project_routed(inpSA, cur, lstate, lw, hp)
      return fused_layer if fused_layer

      attn_out = recurrent_attn_project_routed(lstate, cur, lw, hp)
      unless attn_out
        # 2-4. Batched qkv/gate/alpha/beta projections (all from same `cur` → one sync)
        proj = qmatvec_many([lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw], cur)
        qkv_mixed = proj[0] # [qkv_dim]
        z = proj[1]         # [d_inner = h_v * s_v]
        alpha = proj[2]     # [h_v]
        beta = proj[3]      # [h_v]
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

        attn_out = delta_net_project_routed(
          lstate, q_conv, k_conv, v_conv, ghead, beta, z,
          lw.ssm_norm, lw.ssm_out_qw, h_k, h_v, s_k, scale, hp.rms_eps,
        )
        unless attn_out
          y = delta_net_step_routed(lstate, q_conv, k_conv, v_conv, ghead, beta,
            h_k, h_v, s_k, scale)

          # 13. Gated RMSNorm: norm[h,d] = RMSNorm_per_head(y[h,:], ssm_norm) * silu(z[h,d])
          h_v.times do |h|
            rms_norm_slice!(y, h * s_v, s_v, lw.ssm_norm, hp.rms_eps)
          end
          (h_v * s_v).times { |i| y[i] = y[i] * silu(z[i]) }

          # 14. Output projection (ssm_out)
          attn_out = qmatvec_nobias(lw.ssm_out_qw, y) # [n_embd]
        end
      end

      # 15. Residual 1
      inpL2 = Array(Float32).new(n_embd) { |i| inpSA[i] + attn_out.not_nil![i] }

      # 16. Post-attention norm
      cur2 = rms_norm(inpL2, lw.post_attention_norm, hp.rms_eps)

      # 17. SwiGLU FFN
      ffn_out = ffn_project_routed(cur2, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw)
      unless ffn_out
        gu = qmatvec_many([lw.ffn_gate_qw, lw.ffn_up_qw], cur2)
        gate_ff = gu[0]
        up_ff = gu[1]
        silu!(gate_ff)
        combined = Array(Float32).new(n_ff) { |i| gate_ff[i] * up_ff[i] }
        ffn_out = qmatvec_nobias(lw.ffn_down_qw, combined)
      end

      # 18. Residual 2
      Array(Float32).new(n_embd) { |i| inpL2[i] + ffn_out.not_nil![i] }
    end

    # Multi-token recurrent layer prefill.
    #
    # Exact semantics are the same as repeated `forward_recurrent_layer` calls:
    # the convolution and DeltaNet states are scanned in token order. The
    # speedup comes from batching projections and running the recurrent prep +
    # DeltaNet scan once per layer chunk instead of once per token.
    private def forward_recurrent_layer_chunk(inp : Array(Float32),
                                              n_tokens : Int32,
                                              lw : Qwen35RecurrentWeights,
                                              lstate : LayerState,
                                              hp : Qwen35Hparams,
                                              max_seq : Int32) : Array(Float32)
      if ENV["QWEN35_PREFILL_CHUNK_OFF"]? != "1" && n_tokens > 1
        supported = Qwen35Metal.available? &&
                    metal_qw_supported?(lw.attn_qkv_qw) &&
                    metal_qw_supported?(lw.attn_gate_qw) &&
                    metal_qw_supported?(lw.ssm_alpha_qw) &&
                    metal_qw_supported?(lw.ssm_beta_qw) &&
                    metal_qw_supported?(lw.ssm_out_qw) &&
                    metal_qw_supported?(lw.ffn_gate_qw) &&
                    metal_qw_supported?(lw.ffn_up_qw) &&
                    metal_qw_supported?(lw.ffn_down_qw)
        if supported
          h_k = hp.ssm_group_count
          h_v = hp.ssm_time_step_rank
          s = hp.ssm_state_size
          qkv_dim = 2 * h_k * s + h_v * s
          conv_k = hp.ssm_conv_kernel

          conv_bytes = ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32)
          conv_buf = lstate.conv_state_buf
          if conv_buf.nil?
            conv_buf = ML::MetalBuffer.new(conv_bytes)
            if conv_state = lstate.conv_state
              conv_buf.write(conv_state)
            else
              conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
            end
            lstate.conv_state_buf = conv_buf
          end

          ssm_bytes = (h_v * s * s).to_i64 * sizeof(Float32)
          ssm_buf = lstate.ssm_state_buf
          if ssm_buf.nil?
            ssm_buf = ML::MetalBuffer.new(ssm_bytes)
            if ssm_state = lstate.ssm_state
              ssm_buf.write(ssm_state)
            else
              ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
            end
            lstate.ssm_state_buf = ssm_buf
          end

          if gpu_out = Qwen35Metal.recurrent_layer_chunk_project(
               inp, conv_buf, ssm_buf, lw.attn_norm,
               lw.attn_qkv_qw, lw.attn_gate_qw, lw.ssm_alpha_qw, lw.ssm_beta_qw,
               lw.ssm_conv1d, lw.ssm_dt_bias, lw.ssm_a, lw.ssm_norm, lw.ssm_out_qw,
               lw.post_attention_norm, lw.ffn_gate_qw, lw.ffn_up_qw, lw.ffn_down_qw,
               h_k, h_v, s, conv_k, n_tokens, hp.rms_eps)
            return gpu_out
          end
        end
      end

      out = Array(Float32).new(inp.size, 0.0_f32)
      n_tokens.times do |t|
        row = inp[t * hp.n_embd, hp.n_embd]
        y = forward_recurrent_layer(row, 0, lw, lstate, hp, max_seq)
        hp.n_embd.times { |i| out[t * hp.n_embd + i] = y[i] }
      end
      out
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
      if logits = forward_decode_wave_routed(weights, token_id, pos, state)
        return logits
      end

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

      if logits = output_project_routed(x, weights.output_norm, weights.output, hp.rms_eps)
        logits
      else
        rms_norm!(x, weights.output_norm, hp.rms_eps)
        qmatvec_nobias(weights.output, x)
      end
    end

    # Greedy decode helper. By default, the Metal wave path avoids
    # materializing full lm-head logits and returns only top-1. Set
    # `QWEN35_HEAD_TOP1_FUSED=0` to force the full-logit fallback.
    def forward_top1(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                     state : State) : {Int32, Float32}
      if packed = forward_decode_wave_routed(weights, token_id, pos, state, top1: true)
        if packed.size == 2
          return {packed[0].to_i32, packed[1]}
        end

        maxv = packed.max
        return {packed.index(maxv).not_nil!.to_i32, maxv}
      end

      logits = forward(weights, token_id, pos, state)
      maxv = logits.max
      {logits.index(maxv).not_nil!.to_i32, maxv}
    end

    # Prefill helper for prompt tokens whose logits are not needed.
    #
    # This is exact for autoregressive state construction: every layer still
    # runs and updates full-attention KV plus DeltaNet conv/SSM state, but the
    # expensive output RMSNorm/lm-head projection is skipped. Use `forward` or
    # `forward_top1` for the final prompt token when next-token logits are
    # required.
    def prefill_token(weights : Qwen35Weights, token_id : Int32, pos : Int32,
                      state : State) : Nil
      if forward_decode_wave_routed(weights, token_id, pos, state, emit_head: false)
        return
      end

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
    end

    # Prefill a known prompt span whose logits are not observed.
    #
    # This is exact for intermediate prompt tokens. It processes recurrent
    # layers in token chunks and falls back to serial full-attention layers,
    # because full-attention prefill still needs a dedicated causal chunk path.
    def prefill_tokens(weights : Qwen35Weights,
                       token_ids : Array(Int32),
                       start_pos : Int32,
                       state : State) : Nil
      return if token_ids.empty?

      prefill_tokens_hidden(weights, token_ids, start_pos, state)
    end

    def prefill_tokens_top1(weights : Qwen35Weights,
                            token_ids : Array(Int32),
                            start_pos : Int32,
                            state : State) : {Int32, Float32}
      raise ArgumentError.new("prefill_tokens_top1 token_ids must not be empty") if token_ids.empty?

      if ENV["QWEN35_PREFILL_FINAL_CHUNK_OFF"]? != "1" &&
         ENV["QWEN35_PREFILL_CHUNK_OFF"]? != "1" &&
         token_ids.size > 1
        chunk_size = (ENV["QWEN35_PREFILL_CHUNK_SIZE"]? || DEFAULT_PREFILL_CHUNK_SIZE.to_s).to_i
        if token_ids.size > chunk_size
          prefill_tokens(weights, token_ids[0...-1], start_pos, state)
          return forward_top1(weights, token_ids[-1], start_pos + token_ids.size - 1, state)
        end

        if ENV["QWEN35_FINAL_FULL_LAST_OFF"]? != "1" &&
           (last_layer = weights.layers[-1].as?(Qwen35FullAttnWeights)) &&
           metal_qw_supported?(last_layer.attn_q_qw) &&
           metal_qw_supported?(last_layer.attn_k_qw) &&
           metal_qw_supported?(last_layer.attn_v_qw) &&
           metal_qw_supported?(last_layer.attn_output_qw) &&
           metal_qw_supported?(last_layer.ffn_gate_qw) &&
           metal_qw_supported?(last_layer.ffn_up_qw) &&
           metal_qw_supported?(last_layer.ffn_down_qw)
          x_before_last = prefill_tokens_hidden(weights, token_ids, start_pos, state, stop_layer: weights.layers.size - 1)
          if last = final_full_attn_layer_chunk_last_routed(x_before_last, token_ids.size, start_pos, state.layers[-1], last_layer, weights.hparams, state.max_seq)
            hp = weights.hparams
            if top1 = output_project_top1_routed(last, weights.output_norm, weights.output, hp.rms_eps)
              return top1
            end
            rms_norm!(last, weights.output_norm, hp.rms_eps)
            logits = qmatvec_nobias(weights.output, last)
            maxv = logits.max
            return {logits.index(maxv).not_nil!.to_i32, maxv}
          end
        end

        x = prefill_tokens_hidden(weights, token_ids, start_pos, state)
        hp = weights.hparams
        last = x[(token_ids.size - 1) * hp.n_embd, hp.n_embd]
        if top1 = output_project_top1_routed(last, weights.output_norm, weights.output, hp.rms_eps)
          return top1
        end
        rms_norm!(last, weights.output_norm, hp.rms_eps)
        logits = qmatvec_nobias(weights.output, last)
        maxv = logits.max
        return {logits.index(maxv).not_nil!.to_i32, maxv}
      end

      if token_ids.size > 1
        prefill_tokens(weights, token_ids[0...-1], start_pos, state)
      end
      forward_top1(weights, token_ids[-1], start_pos + token_ids.size - 1, state)
    end

    private def prefill_tokens_hidden(weights : Qwen35Weights,
                                      token_ids : Array(Int32),
                                      start_pos : Int32,
                                      state : State,
                                      stop_layer : Int32? = nil) : Array(Float32)
      raise ArgumentError.new("prefill_tokens_hidden token_ids must not be empty") if token_ids.empty?

      if ENV["QWEN35_PREFILL_CHUNK_OFF"]? == "1" || token_ids.size == 1
        last_x = nil.as(Array(Float32)?)
        token_ids.each_with_index do |token_id, i|
          pos = start_pos + i
          prefill_token(weights, token_id, pos, state)
          last_x = embedding_lookup(weights.token_embd, token_id) if i == token_ids.size - 1
        end
        return last_x.not_nil!
      end

      hp = weights.hparams
      max_seq = state.max_seq
      n_tokens = token_ids.size
      raise ArgumentError.new("prefill span exceeds max_seq") if start_pos < 0 || start_pos + n_tokens > max_seq

      chunk_size = (ENV["QWEN35_PREFILL_CHUNK_SIZE"]? || DEFAULT_PREFILL_CHUNK_SIZE.to_s).to_i
      raise ArgumentError.new("QWEN35_PREFILL_CHUNK_SIZE must be positive") unless chunk_size > 0
      if n_tokens > chunk_size
        offset = 0
        x = nil.as(Array(Float32)?)
        while offset < n_tokens
          len = Math.min(chunk_size, n_tokens - offset)
          x = prefill_tokens_hidden(weights, token_ids[offset, len], start_pos + offset, state, stop_layer: stop_layer)
          offset += len
        end
        return x.not_nil!
      end

      x = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
      token_ids.each_with_index do |token_id, t|
        emb = embedding_lookup(weights.token_embd, token_id)
        hp.n_embd.times { |i| x[t * hp.n_embd + i] = emb[i] }
      end

      il = 0
      layer_limit = stop_layer || weights.layers.size
      while il < layer_limit
        lw = weights.layers[il]
        case lw
        in Qwen35FullAttnWeights
          if fused = full_attn_then_recurrent_chunk_project_many_routed(x, n_tokens, start_pos, state, weights, il, hp, max_seq)
            x = fused[0]
            il = fused[1]
            next
          elsif gpu_out = full_attn_layer_chunk_project_routed(x, n_tokens, start_pos, state.layers[il], lw, hp, max_seq)
            x = gpu_out
          else
            out = Array(Float32).new(n_tokens * hp.n_embd, 0.0_f32)
            n_tokens.times do |t|
              row = x[t * hp.n_embd, hp.n_embd]
              y = forward_full_attn_layer(row, start_pos + t, lw, state.layers[il], hp, max_seq)
              hp.n_embd.times { |i| out[t * hp.n_embd + i] = y[i] }
            end
            x = out
          end
          il += 1
        in Qwen35RecurrentWeights
          if ENV["QWEN35_PREFILL_REC_RUN_OFF"]? != "1"
            run_end = il
            while run_end < weights.layers.size
              break unless weights.layers[run_end].is_a?(Qwen35RecurrentWeights)
              run_end += 1
            end

            if run_end - il > 1
              rec_layers = [] of Qwen35RecurrentWeights
              conv_bufs = [] of ML::MetalBuffer
              ssm_bufs = [] of ML::MetalBuffer
              h_k = hp.ssm_group_count
              h_v = hp.ssm_time_step_rank
              s = hp.ssm_state_size
              qkv_dim = 2 * h_k * s + h_v * s
              conv_k = hp.ssm_conv_kernel
              supported = Qwen35Metal.available?

              j = il
              while j < run_end
                rw = weights.layers[j].as(Qwen35RecurrentWeights)
                supported &&= metal_qw_supported?(rw.attn_qkv_qw) &&
                              metal_qw_supported?(rw.attn_gate_qw) &&
                              metal_qw_supported?(rw.ssm_alpha_qw) &&
                              metal_qw_supported?(rw.ssm_beta_qw) &&
                              metal_qw_supported?(rw.ssm_out_qw) &&
                              metal_qw_supported?(rw.ffn_gate_qw) &&
                              metal_qw_supported?(rw.ffn_up_qw) &&
                              metal_qw_supported?(rw.ffn_down_qw)
                rec_layers << rw

                lstate = state.layers[j]
                conv_bytes = ((conv_k - 1) * qkv_dim).to_i64 * sizeof(Float32)
                conv_buf = lstate.conv_state_buf
                if conv_buf.nil?
                  conv_buf = ML::MetalBuffer.new(conv_bytes)
                  if conv_state = lstate.conv_state
                    conv_buf.write(conv_state)
                  else
                    conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
                  end
                  lstate.conv_state_buf = conv_buf
                end
                conv_bufs << conv_buf

                ssm_bytes = (h_v * s * s).to_i64 * sizeof(Float32)
                ssm_buf = lstate.ssm_state_buf
                if ssm_buf.nil?
                  ssm_buf = ML::MetalBuffer.new(ssm_bytes)
                  if ssm_state = lstate.ssm_state
                    ssm_buf.write(ssm_state)
                  else
                    ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
                  end
                  lstate.ssm_state_buf = ssm_buf
                end
                ssm_bufs << ssm_buf
                j += 1
              end

              if supported
                if gpu_out = Qwen35Metal.recurrent_layer_chunk_project_many(
                     x, conv_bufs, ssm_bufs, rec_layers,
                     h_k, h_v, s, conv_k, n_tokens, hp.rms_eps)
                  x = gpu_out
                  il = run_end
                  next
                end
              end
            end
          end

          x = forward_recurrent_layer_chunk(x, n_tokens, lw, state.layers[il], hp, max_seq)
          il += 1
        end
      end
      x
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
                  when t.f32?  then n_embd * 4
                  when t.f16?  then n_embd * 2
                  when t.q4_k? then (n_embd // 256) * 144
                  when t.q5_k? then (n_embd // 256) * 176
                  when t.q6_k? then (n_embd // 256) * 210
                  else              raise "embedding: unsupported quant type #{t.name}"
                  end
      offset = token_id.to_i64 * row_bytes.to_i64
      row_slice = Bytes.new(token_embd.raw.to_unsafe + offset, row_bytes, read_only: true)
      Dequant.dequantize(row_slice, t, n_embd)
    end

    private def forward_decode_wave_routed(weights : Qwen35Weights,
                                           token_id : Int32,
                                           pos : Int32,
                                           state : State,
                                           top1 : Bool = false,
                                           emit_head : Bool = true) : Array(Float32)?
      {% unless flag?(:cpu_only) %}
        return nil if ENV["QWEN35_DECODE_WAVE_OFF"]? == "1"
        return nil unless Qwen35Metal.available?
        return nil unless metal_qw_supported?(weights.output)

        weights.layers.each do |lw|
          supported = case lw
                      in Qwen35FullAttnWeights
                        metal_qw_supported?(lw.attn_q_qw) &&
                          metal_qw_supported?(lw.attn_k_qw) &&
                          metal_qw_supported?(lw.attn_v_qw) &&
                          metal_qw_supported?(lw.attn_output_qw) &&
                          metal_qw_supported?(lw.ffn_gate_qw) &&
                          metal_qw_supported?(lw.ffn_up_qw) &&
                          metal_qw_supported?(lw.ffn_down_qw)
                      in Qwen35RecurrentWeights
                        metal_qw_supported?(lw.attn_qkv_qw) &&
                          metal_qw_supported?(lw.attn_gate_qw) &&
                          metal_qw_supported?(lw.ssm_alpha_qw) &&
                          metal_qw_supported?(lw.ssm_beta_qw) &&
                          metal_qw_supported?(lw.ssm_out_qw) &&
                          metal_qw_supported?(lw.ffn_gate_qw) &&
                          metal_qw_supported?(lw.ffn_up_qw) &&
                          metal_qw_supported?(lw.ffn_down_qw)
                      end
          return nil unless supported
        end

        hp = weights.hparams
        max_seq = state.max_seq
        kv_dim = hp.head_dim * hp.n_head_kv
        qkv_dim = 2 * hp.ssm_group_count * hp.ssm_state_size + hp.ssm_time_step_rank * hp.ssm_state_size

        k_cache_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        v_cache_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        conv_state_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)
        ssm_state_bufs = Array(ML::MetalBuffer?).new(hp.n_layer, nil)

        weights.layers.each_with_index do |lw, il|
          case lw
          in Qwen35FullAttnWeights
            bytes = (max_seq * kv_dim).to_i64 * sizeof(Float32)
            k_buf = state.layers[il].k_cache_buf
            if k_buf.nil?
              k_buf = ML::MetalBuffer.new(bytes)
              k_buf.contents.as(Pointer(UInt8)).clear(bytes)
              state.layers[il].k_cache_buf = k_buf
            end
            v_buf = state.layers[il].v_cache_buf
            if v_buf.nil?
              v_buf = ML::MetalBuffer.new(bytes)
              v_buf.contents.as(Pointer(UInt8)).clear(bytes)
              state.layers[il].v_cache_buf = v_buf
            end
            k_cache_bufs[il] = k_buf
            v_cache_bufs[il] = v_buf
          in Qwen35RecurrentWeights
            conv_bytes = ((hp.ssm_conv_kernel - 1) * qkv_dim).to_i64 * sizeof(Float32)
            conv_buf = state.layers[il].conv_state_buf
            if conv_buf.nil?
              conv_buf = ML::MetalBuffer.new(conv_bytes)
              conv_buf.contents.as(Pointer(UInt8)).clear(conv_bytes)
              state.layers[il].conv_state_buf = conv_buf
            end
            ssm_bytes = (hp.ssm_time_step_rank * hp.ssm_state_size * hp.ssm_state_size).to_i64 * sizeof(Float32)
            ssm_buf = state.layers[il].ssm_state_buf
            if ssm_buf.nil?
              ssm_buf = ML::MetalBuffer.new(ssm_bytes)
              ssm_buf.contents.as(Pointer(UInt8)).clear(ssm_bytes)
              state.layers[il].ssm_state_buf = ssm_buf
            end
            conv_state_bufs[il] = conv_buf
            ssm_state_bufs[il] = ssm_buf
          end
        end

        emb = embedding_lookup(weights.token_embd, token_id)
        Qwen35Metal.forward_decode_wave(
          emb, weights.layers,
          k_cache_bufs, v_cache_bufs, conv_state_bufs, ssm_state_bufs,
          weights.output_norm, weights.output, hp, pos, top1: top1, emit_head: emit_head)
      {% else %}
        nil
      {% end %}
    end
  end
end
