# Frozen recurrent Qwen3.5 block used to carry gradients between trainable
# full-attention blocks. The GGUF weights remain quantized and immutable.

require "./qwen35_qlora_projection"
require "./qwen35_qlora_delta"

module ML::GGUF::Qwen35QLoRA
  class RecurrentBlock
    private struct ForwardCache
      getter residual1 : Array(Float32)
      getter conv_preact : Array(Float32)
      getter z : Array(Float32)
      getter alpha : Array(Float32)
      getter beta : Array(Float32)
      getter decay : Array(Float32)
      getter delta_y : Array(Float32)

      def initialize(@residual1, @conv_preact, @z, @alpha, @beta, @decay, @delta_y)
      end
    end

    @input_tape : Array(Float32)? = nil
    @tape_rows : Int32 = 0

    def initialize(@lw : Qwen35RecurrentWeights, @hp : Qwen35Hparams)
      validate_topology
    end

    # Runs a complete sample from zero convolution and DeltaNet state. When
    # retain_tape is true, only the block input is retained; backward
    # recomputes this block's activations to keep a stack of recurrent blocks
    # bounded in memory.
    def forward(input : Array(Float32), *, retain_tape : Bool = true) : Array(Float32)
      release_tape
      rows = validate_rows(input, "input")
      output, _cache = forward_core(input, rows, false)
      if retain_tape
        @input_tape = input.dup
        @tape_rows = rows
      end
      output
    end

    # Propagates an output derivative through all frozen recurrent-block
    # operations and consumes the retained input tape on success.
    def backward(grad : Array(Float32)) : Array(Float32)
      input = @input_tape
      raise ArgumentError.new("recurrent block backward requires a retained forward tape") unless input
      rows = @tape_rows
      expected = rows.to_i64 * @hp.n_embd.to_i64
      unless grad.size.to_i64 == expected
        raise ArgumentError.new("recurrent block grad size mismatch: expected #{expected}, got #{grad.size}")
      end

      _output, cached = forward_core(input, rows, true)
      cache = cached.not_nil!
      input_grad = backward_from_cache(input, grad, rows, cache)
      release_tape
      input_grad
    end

    # Drops an unused tape without changing block weights or outputs.
    def release_tape : Nil
      @input_tape = nil
      @tape_rows = 0
    end

    private def validate_topology : Nil
      hp = @hp
      raise ArgumentError.new("recurrent block dimensions must be positive") unless hp.n_embd > 0 && hp.n_ff > 0
      raise ArgumentError.new("recurrent state dimensions must be positive") unless hp.ssm_group_count > 0 && hp.ssm_time_step_rank > 0 && hp.ssm_state_size > 0
      raise ArgumentError.new("recurrent convolution kernel must be at least 1") unless hp.ssm_conv_kernel > 0
      raise ArgumentError.new("recurrent inner size must equal value heads times state size") unless hp.ssm_inner_size == hp.ssm_time_step_rank * hp.ssm_state_size
      raise ArgumentError.new("recurrent value heads must be divisible by key heads") unless hp.ssm_time_step_rank % hp.ssm_group_count == 0
      qkv_dim = qkv_size
      raise ArgumentError.new("recurrent qkv projection shape mismatch") unless @lw.attn_qkv_qw.in_dim == hp.n_embd && @lw.attn_qkv_qw.out_dim == qkv_dim
      raise ArgumentError.new("recurrent gate projection shape mismatch") unless @lw.attn_gate_qw.in_dim == hp.n_embd && @lw.attn_gate_qw.out_dim == hp.ssm_inner_size
      raise ArgumentError.new("recurrent alpha projection shape mismatch") unless @lw.ssm_alpha_qw.in_dim == hp.n_embd && @lw.ssm_alpha_qw.out_dim == hp.ssm_time_step_rank
      raise ArgumentError.new("recurrent beta projection shape mismatch") unless @lw.ssm_beta_qw.in_dim == hp.n_embd && @lw.ssm_beta_qw.out_dim == hp.ssm_time_step_rank
      raise ArgumentError.new("recurrent convolution weight shape mismatch") unless @lw.ssm_conv1d.size == qkv_dim * hp.ssm_conv_kernel
      raise ArgumentError.new("recurrent decay parameter shape mismatch") unless @lw.ssm_a.size == hp.ssm_time_step_rank && @lw.ssm_dt_bias.size == hp.ssm_time_step_rank
      raise ArgumentError.new("recurrent head norm shape mismatch") unless @lw.ssm_norm.size == hp.ssm_state_size
      raise ArgumentError.new("recurrent output projection shape mismatch") unless @lw.ssm_out_qw.in_dim == hp.ssm_inner_size && @lw.ssm_out_qw.out_dim == hp.n_embd
      raise ArgumentError.new("recurrent attention norm shape mismatch") unless @lw.attn_norm.size == hp.n_embd && @lw.post_attention_norm.size == hp.n_embd
      {@lw.ffn_gate_qw, @lw.ffn_up_qw}.each do |weight|
        unless weight.in_dim == hp.n_embd && weight.out_dim == hp.n_ff
          raise ArgumentError.new("recurrent FFN projection shape mismatch")
        end
      end
      raise ArgumentError.new("recurrent FFN down projection shape mismatch") unless @lw.ffn_down_qw.in_dim == hp.n_ff && @lw.ffn_down_qw.out_dim == hp.n_embd
    end

    private def validate_rows(input : Array(Float32), label : String) : Int32
      unless @hp.n_embd > 0 && input.size > 0 && input.size % @hp.n_embd == 0
        raise ArgumentError.new("recurrent block #{label} size must be a positive multiple of n_embd")
      end
      rows = input.size // @hp.n_embd
      raise ArgumentError.new("recurrent block row count exceeds Int32") if rows > Int32::MAX
      rows.to_i32
    end

    private def qkv_size : Int32
      2 * @hp.ssm_group_count * @hp.ssm_state_size + @hp.ssm_time_step_rank * @hp.ssm_state_size
    end

    private def forward_core(input : Array(Float32), rows : Int32, capture : Bool) : Tuple(Array(Float32), ForwardCache?)
      n_embd = @hp.n_embd
      h_k = @hp.ssm_group_count
      h_v = @hp.ssm_time_step_rank
      size = @hp.ssm_state_size
      qk_dim = h_k * size
      value_dim = h_v * size
      qkv_dim = qkv_size

      normalized = rms_norm_rows(input, @lw.attn_norm, rows, n_embd, @hp.rms_eps)
      qkv_projected = Qwen35CPU.qlora_project(@lw.attn_qkv_qw, normalized, rows)
      z = Qwen35CPU.qlora_project(@lw.attn_gate_qw, normalized, rows)
      alpha = Qwen35CPU.qlora_project(@lw.ssm_alpha_qw, normalized, rows)
      beta = Qwen35CPU.qlora_project(@lw.ssm_beta_qw, normalized, rows)

      conv_preact = convolve(qkv_projected, rows, qkv_dim)
      conv_activated = capture ? conv_preact.dup : conv_preact
      Qwen35CPU.silu!(conv_activated)
      q, k, v = split_qkv(conv_activated, rows, qk_dim, value_dim)
      normalize_l2_heads!(q, rows, h_k, size, @hp.rms_eps)
      normalize_l2_heads!(k, rows, h_k, size, @hp.rms_eps)

      beta.each_index { |i| beta[i] = Qwen35CPU.sigmoid(beta[i]) }
      decay = Array(Float32).new(rows * h_v) do |i|
        head = i % h_v
        raw = alpha[i] + @lw.ssm_dt_bias[head]
        softplus = raw > 20.0_f32 ? raw : Math.log(1.0_f32 + Math.exp(raw)).to_f32
        Math.exp((softplus * @lw.ssm_a[head]).to_f64).to_f32
      end

      delta_y = Delta.forward(q, k, v, decay, beta, rows, h_k, h_v, size)
      normalized_delta = rms_norm_head_rows(delta_y, rows, h_v, size, @lw.ssm_norm, @hp.rms_eps)
      gated_delta = Array(Float32).new(rows * value_dim) do |i|
        normalized_delta[i] * Qwen35CPU.silu(z[i])
      end
      attention = Qwen35CPU.qlora_project(@lw.ssm_out_qw, gated_delta, rows)

      residual1 = Array(Float32).new(rows * n_embd) { |i| input[i] + attention[i] }
      normalized_residual = rms_norm_rows(residual1, @lw.post_attention_norm, rows, n_embd, @hp.rms_eps)
      gate = Qwen35CPU.qlora_project(@lw.ffn_gate_qw, normalized_residual, rows)
      up = Qwen35CPU.qlora_project(@lw.ffn_up_qw, normalized_residual, rows)
      gate.each_index { |i| gate[i] = Qwen35CPU.silu(gate[i]) * up[i] }
      ffn = Qwen35CPU.qlora_project(@lw.ffn_down_qw, gate, rows)
      output = Array(Float32).new(rows * n_embd) { |i| residual1[i] + ffn[i] }

      cache = if capture
                ForwardCache.new(residual1, conv_preact, z, alpha, beta, decay, delta_y)
              else
                nil
              end
      {output, cache}
    end

    private def backward_from_cache(input : Array(Float32), grad : Array(Float32), rows : Int32,
                                    cache : ForwardCache) : Array(Float32)
      hp = @hp
      lw = @lw
      n_embd = hp.n_embd
      n_ff = hp.n_ff
      h_k = hp.ssm_group_count
      h_v = hp.ssm_time_step_rank
      size = hp.ssm_state_size
      qk_dim = h_k * size
      value_dim = h_v * size
      qkv_dim = qkv_size

      # SwiGLU branch, then its post-attention RMSNorm.
      normalized_residual = rms_norm_rows(cache.residual1, lw.post_attention_norm, rows, n_embd, hp.rms_eps)
      gate_pre = Qwen35CPU.qlora_project(lw.ffn_gate_qw, normalized_residual, rows)
      up = Qwen35CPU.qlora_project(lw.ffn_up_qw, normalized_residual, rows)
      grad_combined = frozen_transpose_project(lw.ffn_down_qw, grad, rows)
      grad_gate_pre = Array(Float32).new(rows * n_ff) do |i|
        grad_combined[i] * up[i] * silu_derivative(gate_pre[i])
      end
      grad_up = Array(Float32).new(rows * n_ff) do |i|
        grad_combined[i] * Qwen35CPU.silu(gate_pre[i])
      end
      grad_normalized_residual = frozen_transpose_project(lw.ffn_gate_qw, grad_gate_pre, rows)
      add_in_place!(grad_normalized_residual, frozen_transpose_project(lw.ffn_up_qw, grad_up, rows))
      grad_residual1 = rms_norm_rows_backward(
        cache.residual1, grad_normalized_residual, lw.post_attention_norm, rows, n_embd, hp.rms_eps,
      )
      add_in_place!(grad_residual1, grad)

      # Output projection, gated head RMSNorm, and the recurrent DeltaNet scan.
      grad_gated_delta = frozen_transpose_project(lw.ssm_out_qw, grad_residual1, rows)
      normalized_delta = rms_norm_head_rows(cache.delta_y, rows, h_v, size, lw.ssm_norm, hp.rms_eps)
      grad_normalized_delta = Array(Float32).new(rows * value_dim, 0.0_f32)
      grad_z = Array(Float32).new(rows * value_dim, 0.0_f32)
      (rows * value_dim).times do |i|
        z = cache.z[i]
        silu_z = Qwen35CPU.silu(z)
        grad_normalized_delta[i] = grad_gated_delta[i] * silu_z
        grad_z[i] = grad_gated_delta[i] * normalized_delta[i] * silu_derivative(z)
      end
      grad_delta_y = rms_norm_head_rows_backward(
        cache.delta_y, grad_normalized_delta, rows, h_v, size, lw.ssm_norm, hp.rms_eps,
      )

      conv_activated = cache.conv_preact.dup
      Qwen35CPU.silu!(conv_activated)
      q, k, v = split_qkv(conv_activated, rows, qk_dim, value_dim)
      q_pre_l2 = q.dup
      k_pre_l2 = k.dup
      normalize_l2_heads!(q, rows, h_k, size, hp.rms_eps)
      normalize_l2_heads!(k, rows, h_k, size, hp.rms_eps)
      delta_grad = Delta.backward(q, k, v, cache.decay, cache.beta, grad_delta_y, rows, h_k, h_v, size)
      grad_q = delta_grad[0]
      grad_k = delta_grad[1]
      grad_v = delta_grad[2]
      grad_decay = delta_grad[3]
      grad_beta = delta_grad[4]

      grad_conv_activated = Array(Float32).new(rows * qkv_dim, 0.0_f32)
      grad_q_pre = l2_norm_rows_backward(q_pre_l2, q, grad_q, rows, h_k, size, hp.rms_eps)
      grad_k_pre = l2_norm_rows_backward(k_pre_l2, k, grad_k, rows, h_k, size, hp.rms_eps)
      rows.times do |row|
        base = row * qkv_dim
        qk_base = row * qk_dim
        v_base = row * value_dim
        (h_k * size).times do |i|
          grad_conv_activated[base + i] = grad_q_pre[qk_base + i]
          grad_conv_activated[base + qk_dim + i] = grad_k_pre[qk_base + i]
        end
        value_dim.times do |i|
          grad_conv_activated[base + 2 * qk_dim + i] = grad_v[v_base + i]
        end
      end

      grad_conv_preact = Array(Float32).new(rows * qkv_dim) do |i|
        grad_conv_activated[i] * silu_derivative(cache.conv_preact[i])
      end
      grad_qkv_projected = convolve_backward(grad_conv_preact, rows, qkv_dim)

      grad_alpha = Array(Float32).new(rows * h_v) do |i|
        head = i % h_v
        raw = cache.alpha[i] + lw.ssm_dt_bias[head]
        sigmoid = Qwen35CPU.sigmoid(raw)
        grad_decay[i] * cache.decay[i] * lw.ssm_a[head] * sigmoid
      end
      grad_beta_pre = Array(Float32).new(rows * h_v) do |i|
        value = cache.beta[i]
        grad_beta[i] * value * (1.0_f32 - value)
      end

      # The four projection paths share the same normalized attention input.
      grad_normalized_input = frozen_transpose_project(lw.attn_qkv_qw, grad_qkv_projected, rows)
      add_in_place!(grad_normalized_input, frozen_transpose_project(lw.attn_gate_qw, grad_z, rows))
      add_in_place!(grad_normalized_input, frozen_transpose_project(lw.ssm_alpha_qw, grad_alpha, rows))
      add_in_place!(grad_normalized_input, frozen_transpose_project(lw.ssm_beta_qw, grad_beta_pre, rows))
      grad_input = rms_norm_rows_backward(input, grad_normalized_input, lw.attn_norm, rows, n_embd, hp.rms_eps)
      add_in_place!(grad_input, grad_residual1)
      grad_input
    end

    private def convolve(projected : Array(Float32), rows : Int32, dim : Int32) : Array(Float32)
      kernel = @hp.ssm_conv_kernel
      output = Array(Float32).new(rows * dim, 0.0_f32)
      rows.times do |row|
        dim.times do |channel|
          weight_base = channel * kernel
          acc = 0.0_f32
          (kernel - 1).times do |tap|
            source_row = row - (kernel - 1) + tap
            next if source_row < 0
            acc += projected[source_row * dim + channel] * @lw.ssm_conv1d[weight_base + tap]
          end
          acc += projected[row * dim + channel] * @lw.ssm_conv1d[weight_base + kernel - 1]
          output[row * dim + channel] = acc
        end
      end
      output
    end

    private def convolve_backward(grad_output : Array(Float32), rows : Int32, dim : Int32) : Array(Float32)
      kernel = @hp.ssm_conv_kernel
      grad_input = Array(Float32).new(rows * dim, 0.0_f32)
      rows.times do |row|
        dim.times do |channel|
          value = grad_output[row * dim + channel]
          weight_base = channel * kernel
          (kernel - 1).times do |tap|
            source_row = row - (kernel - 1) + tap
            next if source_row < 0
            grad_input[source_row * dim + channel] += value * @lw.ssm_conv1d[weight_base + tap]
          end
          grad_input[row * dim + channel] += value * @lw.ssm_conv1d[weight_base + kernel - 1]
        end
      end
      grad_input
    end

    private def split_qkv(conv : Array(Float32), rows : Int32, qk_dim : Int32,
                          value_dim : Int32) : Tuple(Array(Float32), Array(Float32), Array(Float32))
      q = Array(Float32).new(rows * qk_dim, 0.0_f32)
      k = Array(Float32).new(rows * qk_dim, 0.0_f32)
      v = Array(Float32).new(rows * value_dim, 0.0_f32)
      rows.times do |row|
        src = row * (2 * qk_dim + value_dim)
        qk_dst = row * qk_dim
        v_dst = row * value_dim
        qk_dim.times do |i|
          q[qk_dst + i] = conv[src + i]
          k[qk_dst + i] = conv[src + qk_dim + i]
        end
        value_dim.times { |i| v[v_dst + i] = conv[src + 2 * qk_dim + i] }
      end
      {q, k, v}
    end

    private def normalize_l2_heads!(x : Array(Float32), rows : Int32, heads : Int32,
                                    size : Int32, eps : Float32) : Nil
      rows.times do |row|
        heads.times do |head|
          Qwen35CPU.l2_norm_slice!(x, row * heads * size + head * size, size, eps)
        end
      end
    end

    private def rms_norm_rows(x : Array(Float32), weights : Array(Float32), rows : Int32,
                              dim : Int32, eps : Float32) : Array(Float32)
      output = Array(Float32).new(rows * dim, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        sum = 0.0_f64
        dim.times { |i| value = x[offset + i].to_f64; sum += value * value }
        inv = (1.0 / Math.sqrt(sum / dim.to_f64 + eps.to_f64)).to_f32
        dim.times { |i| output[offset + i] = x[offset + i] * inv * weights[i] }
      end
      output
    end

    private def rms_norm_rows_backward(x : Array(Float32), grad : Array(Float32), weights : Array(Float32),
                                       rows : Int32, dim : Int32, eps : Float32) : Array(Float32)
      output = Array(Float32).new(rows * dim, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        sum_x2 = 0.0_f64
        dot = 0.0_f64
        dim.times do |i|
          value = x[offset + i].to_f64
          sum_x2 += value * value
          dot += (grad[offset + i] * weights[i]).to_f64 * value
        end
        inv = (1.0 / Math.sqrt(sum_x2 / dim.to_f64 + eps.to_f64)).to_f32
        correction = inv * inv * (dot / dim.to_f64).to_f32
        dim.times do |i|
          weighted = grad[offset + i] * weights[i]
          output[offset + i] = inv * (weighted - x[offset + i] * correction)
        end
      end
      output
    end

    private def rms_norm_head_rows(x : Array(Float32), rows : Int32, heads : Int32, size : Int32,
                                   weights : Array(Float32), eps : Float32) : Array(Float32)
      output = Array(Float32).new(rows * heads * size, 0.0_f32)
      rows.times do |row|
        heads.times do |head|
          offset = (row * heads + head) * size
          sum = 0.0_f64
          size.times { |i| value = x[offset + i].to_f64; sum += value * value }
          inv = (1.0 / Math.sqrt(sum / size.to_f64 + eps.to_f64)).to_f32
          size.times { |i| output[offset + i] = x[offset + i] * inv * weights[i] }
        end
      end
      output
    end

    private def rms_norm_head_rows_backward(x : Array(Float32), grad : Array(Float32), rows : Int32,
                                            heads : Int32, size : Int32, weights : Array(Float32),
                                            eps : Float32) : Array(Float32)
      output = Array(Float32).new(rows * heads * size, 0.0_f32)
      rows.times do |row|
        heads.times do |head|
          offset = (row * heads + head) * size
          sum_x2 = 0.0_f64
          dot = 0.0_f64
          size.times do |i|
            value = x[offset + i].to_f64
            sum_x2 += value * value
            dot += (grad[offset + i] * weights[i]).to_f64 * value
          end
          inv = (1.0 / Math.sqrt(sum_x2 / size.to_f64 + eps.to_f64)).to_f32
          correction = inv * inv * (dot / size.to_f64).to_f32
          size.times do |i|
            weighted = grad[offset + i] * weights[i]
            output[offset + i] = inv * (weighted - x[offset + i] * correction)
          end
        end
      end
      output
    end

    private def l2_norm_rows_backward(input : Array(Float32), normalized : Array(Float32),
                                      grad : Array(Float32), rows : Int32, heads : Int32,
                                      size : Int32, eps : Float32) : Array(Float32)
      output = Array(Float32).new(rows * heads * size, 0.0_f32)
      rows.times do |row|
        heads.times do |head|
          offset = (row * heads + head) * size
          dot = 0.0_f64
          sum = 0.0_f64
          size.times do |i|
            value = input[offset + i].to_f64
            dot += normalized[offset + i].to_f64 * grad[offset + i].to_f64
            sum += value * value
          end
          inv = (1.0 / Math.sqrt(sum + eps.to_f64)).to_f32
          norm_dot = dot.to_f32
          size.times do |i|
            output[offset + i] = inv * (grad[offset + i] - normalized[offset + i] * norm_dot)
          end
        end
      end
      output
    end

    private def silu_derivative(x : Float32) : Float32
      sigmoid = Qwen35CPU.sigmoid(x)
      sigmoid * (1.0_f32 + x * (1.0_f32 - sigmoid))
    end

    private def frozen_transpose_project(weight : QuantWeight, grad : Array(Float32), rows : Int32) : Array(Float32)
      dense = Dequant.dequantize(weight.raw, weight.type, weight.out_dim * weight.in_dim)
      MathOps.gemm(grad, dense, rows, weight.in_dim, weight.out_dim)
    end

    private def add_in_place!(target : Array(Float32), source : Array(Float32)) : Nil
      raise ArgumentError.new("gradient accumulation size mismatch") unless target.size == source.size
      target.size.times { |i| target[i] += source[i] }
    end
  end
end

module ML::GGUF::Qwen35CPU
  # Runs a recurrent layer in the training/generation wrapper's state owner.
  # Multi-row calls use the stock causal chunk implementation; decode calls
  # use the exact single-token implementation.
  def self.qlora_recurrent(input : Array(Float32), lw : Qwen35RecurrentWeights,
                           lstate : LayerState, hp : Qwen35Hparams, max_seq : Int32) : Array(Float32)
    unless hp.n_embd > 0 && input.size > 0 && input.size % hp.n_embd == 0
      raise ArgumentError.new("recurrent input size must be a positive multiple of n_embd")
    end
    rows = input.size // hp.n_embd
    if rows == 1
      forward_recurrent_layer(input, lstate.position, lw, lstate, hp, max_seq)
    else
      forward_recurrent_layer_chunk(input, rows.to_i32, lw, lstate, hp, max_seq)
    end
  end
end
