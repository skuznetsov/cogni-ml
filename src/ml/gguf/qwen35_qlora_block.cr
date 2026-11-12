# Differentiable final full-attention block for the bounded Qwen3.5 QLoRA
# experiment. Base GGUF tensors stay quantized and frozen; only Projection's
# FP32 LoRA factors receive gradients.

require "./qwen35_qlora_projection"
require "./qwen35_weights"

module ML::GGUF::Qwen35QLoRA
  # Incremental state owned by one QLoRA block. Keys are stored after QK
  # normalization and partial RoPE, in token-major [position, kv_dim] order.
  class DecodeState
    getter position : Int32
    getter key_cache : Array(Float32)
    getter value_cache : Array(Float32)

    @kv_dim : Int32?

    def initialize
      @position = 0
      @key_cache = [] of Float32
      @value_cache = [] of Float32
      @kv_dim = nil
    end

    def clear : Nil
      @position = 0
      @key_cache.clear
      @value_cache.clear
      @kv_dim = nil
    end

    def validate!(kv_dim : Int32, max_seq : Int32) : Nil
      raise ArgumentError.new("decode state has a negative position") if @position < 0
      raise ArgumentError.new("decode state exceeds the block context") if @position > max_seq
      expected = @position.to_i64 * kv_dim.to_i64
      raise ArgumentError.new("decode state key cache size mismatch") unless @key_cache.size.to_i64 == expected
      raise ArgumentError.new("decode state value cache size mismatch") unless @value_cache.size.to_i64 == expected
      if prior_dim = @kv_dim
        raise ArgumentError.new("decode state belongs to a different KV shape") unless prior_dim == kv_dim
      end
      @kv_dim = kv_dim
    end

    def append(keys : Array(Float32), values : Array(Float32), rows : Int32,
               kv_dim : Int32) : Nil
      raise ArgumentError.new("decode cache append row mismatch") unless keys.size == rows * kv_dim && values.size == rows * kv_dim
      @key_cache.concat(keys)
      @value_cache.concat(values)
      @position += rows
      @kv_dim = kv_dim
    end
  end

  private class TrainingTape
    getter rows : Int32
    getter input : Array(Float32)
    getter norm1 : Array(Float32)
    getter q_projected : Array(Float32)
    getter k_projected : Array(Float32)
    getter v_projected : Array(Float32)
    getter attention_input : Array(Float32)
    getter residual1 : Array(Float32)
    getter norm2 : Array(Float32)
    getter gate_pre : Array(Float32)
    getter up_pre : Array(Float32)

    def initialize(@rows : Int32, @input : Array(Float32), @norm1 : Array(Float32),
                   @q_projected : Array(Float32), @k_projected : Array(Float32),
                   @v_projected : Array(Float32), @attention_input : Array(Float32),
                   @residual1 : Array(Float32), @norm2 : Array(Float32),
                   @gate_pre : Array(Float32), @up_pre : Array(Float32))
    end
  end

  private struct ForwardPass
    getter output : Array(Float32)
    getter keys : Array(Float32)
    getter values : Array(Float32)
    getter tape : TrainingTape?

    def initialize(@output : Array(Float32), @keys : Array(Float32),
                   @values : Array(Float32), @tape : TrainingTape?)
    end
  end

  class Block
    # The attention matrix scratch is one head at a time, but sequence-wide
    # Q/K/V activations are retained for the training tape. Keep the experiment
    # bounded to the samples admitted by its frontier.
    MAX_TRAINING_SEQUENCE_LENGTH = 2_400

    getter projections : Array(Projection)
    getter hparams : Qwen35Hparams

    @attn_norm : Array(Float32)
    @q_norm : Array(Float32)
    @k_norm : Array(Float32)
    @post_attention_norm : Array(Float32)
    @tape : TrainingTape?

    def initialize(lw : Qwen35FullAttnWeights, @hparams : Qwen35Hparams,
                   seed : UInt64 = 42_u64)
      validate_shapes!(lw, @hparams)
      @attn_norm = lw.attn_norm
      @q_norm = lw.attn_q_norm
      @k_norm = lw.attn_k_norm
      @post_attention_norm = lw.post_attention_norm

      random = Random.new(seed)
      @projections = Array(Projection).new(7) do |index|
        weight = case index
                 when 0 then lw.attn_q_qw
                 when 1 then lw.attn_k_qw
                 when 2 then lw.attn_v_qw
                 when 3 then lw.attn_output_qw
                 when 4 then lw.ffn_gate_qw
                 when 5 then lw.ffn_up_qw
                 else        lw.ffn_down_qw
                 end
        Projection.new(weight, random)
      end
      @tape = nil
    end

    # Forward for a flattened row-major token sample. Positions begin at zero
    # and attention is causal across rows. Retained training tapes are limited
    # to activations needed by this block's adapter gradients; prefix weights
    # and inputs remain frozen.
    def forward(input : Array(Float32), *, retain_tape : Bool = true) : Array(Float32)
      rows = validate_input!(input)
      if retain_tape && rows > MAX_TRAINING_SEQUENCE_LENGTH
        raise ArgumentError.new("training sample exceeds #{MAX_TRAINING_SEQUENCE_LENGTH} tokens")
      end
      pass = forward_core(input, rows, 0, [] of Float32, [] of Float32, retain_tape: retain_tape)
      @tape = pass.tape
      pass.output
    end

    # Backpropagates a flattened output gradient and accumulates FP32 adapter
    # gradients. The tape is consumed once. Input gradients can be requested
    # when this block feeds an earlier trainable adapter; callers training only
    # this block keep the default empty result.
    def backward(grad : Array(Float32), *, need_input : Bool = false) : Array(Float32)
      tape = @tape
      raise ArgumentError.new("backward requires an unconsumed forward tape") unless tape
      @tape = nil

      rows = tape.rows
      hidden = @hparams.n_embd
      q_dim = @hparams.n_head * @hparams.head_dim
      n_ff = @hparams.n_ff
      expected = rows.to_i64 * hidden.to_i64
      raise ArgumentError.new("block grad size mismatch: expected #{expected}, got #{grad.size}") unless grad.size.to_i64 == expected

      combined = Array(Float32).new(rows * n_ff, 0.0_f32)
      rows.times do |row|
        n_ff.times do |i|
          gate = tape.gate_pre[row * n_ff + i]
          up = tape.up_pre[row * n_ff + i]
          combined[row * n_ff + i] = silu(gate) * up
        end
      end

      # FFN: down -> (SiLU(gate) * up) -> gate/up.
      grad_combined = @projections[6].backward(combined, grad, rows)
      grad_gate_pre = Array(Float32).new(rows * n_ff, 0.0_f32)
      grad_up_pre = Array(Float32).new(rows * n_ff, 0.0_f32)
      grad_combined.size.times do |i|
        gate = tape.gate_pre[i]
        up = tape.up_pre[i]
        sig = sigmoid(gate)
        grad_gate_pre[i] = grad_combined[i] * up * sig * (1.0_f32 + gate * (1.0_f32 - sig))
        grad_up_pre[i] = grad_combined[i] * silu(gate)
      end
      grad_norm2_gate = @projections[4].backward(tape.norm2, grad_gate_pre, rows)
      grad_norm2_up = @projections[5].backward(tape.norm2, grad_up_pre, rows)
      grad_norm2 = Array(Float32).new(rows * hidden, 0.0_f32)
      grad_norm2.size.times { |i| grad_norm2[i] = grad_norm2_gate[i] + grad_norm2_up[i] }

      # Residual and post-attention RMSNorm route the FFN gradient to O.
      grad_residual1 = rmsnorm_backward_rows(
        tape.residual1, grad_norm2, rows, hidden, @post_attention_norm, @hparams.rms_eps,
      )
      grad_residual1.size.times { |i| grad_residual1[i] += grad[i] }
      grad_attention_input = @projections[3].backward(tape.attention_input, grad_residual1, rows)

      # Attention: grouped-query heads share the accumulated K/V gradients.
      q_raw = Array(Float32).new(rows * q_dim, 0.0_f32)
      gate_pre = Array(Float32).new(rows * q_dim, 0.0_f32)
      split_q_and_gate!(tape.q_projected, q_raw, gate_pre, rows)
      q = normalized_rotated_q(q_raw, rows, 0)
      k = normalized_rotated_k(tape.k_projected, rows, 0)
      attention_grads = attention_backward(
        q, k, tape.v_projected, gate_pre, grad_attention_input, rows,
      )
      unrotate_heads!(attention_grads.q, rows, @hparams.n_head)
      unrotate_heads!(attention_grads.k, rows, @hparams.n_head_kv)
      dq_raw = rmsnorm_heads_backward(
        q_raw, attention_grads.q, rows, @hparams.n_head, @q_norm,
      )
      dk_raw = rmsnorm_heads_backward(
        tape.k_projected, attention_grads.k, rows, @hparams.n_head_kv, @k_norm,
      )
      q_projection_grad = Array(Float32).new(rows * 2 * q_dim, 0.0_f32)
      join_q_and_gate_grads!(dq_raw, attention_grads.gate, q_projection_grad, rows)

      if need_input
        q_input_grad = @projections[0].backward(tape.norm1, q_projection_grad, rows)
        k_input_grad = @projections[1].backward(tape.norm1, dk_raw, rows)
        v_input_grad = @projections[2].backward(tape.norm1, attention_grads.v, rows)
        grad_norm1 = Array(Float32).new(rows * hidden, 0.0_f32)
        grad_norm1.size.times do |i|
          grad_norm1[i] = q_input_grad[i] + k_input_grad[i] + v_input_grad[i]
        end
        grad_input_norm1 = rmsnorm_backward_rows(
          tape.input, grad_norm1, rows, hidden, @attn_norm, @hparams.rms_eps,
        )
        grad_residual1.size.times { |i| grad_residual1[i] += grad_input_norm1[i] }
        grad_residual1
      else
        @projections[0].backward(tape.norm1, q_projection_grad, rows, need_input: false)
        @projections[1].backward(tape.norm1, dk_raw, rows, need_input: false)
        @projections[2].backward(tape.norm1, attention_grads.v, rows, need_input: false)
        [] of Float32
      end
    end

    # Single-token autoregressive decode. State is mutated only after the
    # entire block has completed successfully.
    def decode(input : Array(Float32), state : DecodeState) : Array(Float32)
      prefill(input, state)
    end

    # Batched prompt prefill over the same causal math used by decode. Rows
    # attend to the cached prefix and preceding rows in this chunk.
    def prefill(input : Array(Float32), state : DecodeState) : Array(Float32)
      @tape = nil
      rows = validate_input!(input)
      kv_dim = @hparams.n_head_kv * @hparams.head_dim
      state.validate!(kv_dim, @hparams.context_length)
      prefix_rows = state.position
      pass = forward_core(
        input, rows, prefix_rows, state.key_cache, state.value_cache, retain_tape: false)
      state.append(pass.keys, pass.values, rows, kv_dim)
      pass.output
    end

    def zero_grad : Nil
      @projections.each(&.zero_grad)
    end

    private def validate_input!(input : Array(Float32)) : Int32
      hidden = @hparams.n_embd
      raise ArgumentError.new("block hidden dimension must be positive") unless hidden > 0
      raise ArgumentError.new("block input cannot be empty") if input.empty?
      raise ArgumentError.new("block input size must be a multiple of n_embd") unless input.size % hidden == 0
      rows = input.size // hidden
      raise ArgumentError.new("block token count exceeds Int32") if rows > Int32::MAX
      rows.to_i32
    end

    private def forward_core(input : Array(Float32), rows : Int32, position_start : Int32,
                             prefix_keys : Array(Float32), prefix_values : Array(Float32),
                             retain_tape : Bool) : ForwardPass
      hp = @hparams
      hidden = hp.n_embd
      heads = hp.n_head
      kv_heads = hp.n_head_kv
      head_dim = hp.head_dim
      q_dim = heads * head_dim
      kv_dim = kv_heads * head_dim
      prefix_rows = position_start
      key_rows = prefix_rows + rows

      expected_cache = prefix_rows.to_i64 * kv_dim.to_i64
      raise ArgumentError.new("prefix key cache size mismatch") unless prefix_keys.size.to_i64 == expected_cache
      raise ArgumentError.new("prefix value cache size mismatch") unless prefix_values.size.to_i64 == expected_cache
      raise ArgumentError.new("block sequence exceeds context length") if key_rows > hp.context_length

      norm1 = rmsnorm_rows(input, rows, hidden, @attn_norm, hp.rms_eps)
      q_projected = @projections[0].forward(norm1, rows)
      k_projected = @projections[1].forward(norm1, rows)
      v_projected = @projections[2].forward(norm1, rows)

      q_raw = Array(Float32).new(rows * q_dim, 0.0_f32)
      gate_pre = Array(Float32).new(rows * q_dim, 0.0_f32)
      split_q_and_gate!(q_projected, q_raw, gate_pre, rows)
      q = normalized_rotated_q(q_raw, rows, position_start)
      k = normalized_rotated_k(k_projected, rows, position_start)
      all_keys = prefix_keys + k
      all_values = prefix_values + v_projected

      attended = causal_gqa_attention(q, all_keys, all_values, rows, key_rows, prefix_rows)
      attention_input = Array(Float32).new(rows * q_dim, 0.0_f32)
      attention_input.size.times do |i|
        attention_input[i] = attended[i] * sigmoid(gate_pre[i])
      end

      projected_attention = @projections[3].forward(attention_input, rows)
      residual1 = Array(Float32).new(rows * hidden, 0.0_f32)
      residual1.size.times { |i| residual1[i] = input[i] + projected_attention[i] }
      norm2 = rmsnorm_rows(residual1, rows, hidden, @post_attention_norm, hp.rms_eps)

      gate_ff = @projections[4].forward(norm2, rows)
      up_ff = @projections[5].forward(norm2, rows)
      combined = Array(Float32).new(rows * hp.n_ff, 0.0_f32)
      combined.size.times { |i| combined[i] = silu(gate_ff[i]) * up_ff[i] }
      ffn_output = @projections[6].forward(combined, rows)
      output = Array(Float32).new(rows * hidden, 0.0_f32)
      output.size.times { |i| output[i] = residual1[i] + ffn_output[i] }

      tape = if retain_tape
               TrainingTape.new(
                 rows, input, norm1, q_projected, k_projected, v_projected,
                 attention_input, residual1, norm2, gate_ff, up_ff,
               )
             else
               nil
             end
      ForwardPass.new(output, k, v_projected, tape)
    end

    private def split_q_and_gate!(projected : Array(Float32), q : Array(Float32),
                                  gate : Array(Float32), rows : Int32) : Nil
      heads = @hparams.n_head
      head_dim = @hparams.head_dim
      q_dim = heads * head_dim
      expected = rows.to_i64 * 2 * q_dim.to_i64
      raise ArgumentError.new("Q+gate projection output size mismatch") unless projected.size.to_i64 == expected
      rows.times do |row|
        heads.times do |head|
          src = row * 2 * q_dim + head * 2 * head_dim
          dst = row * q_dim + head * head_dim
          head_dim.times do |d|
            q[dst + d] = projected[src + d]
            gate[dst + d] = projected[src + head_dim + d]
          end
        end
      end
    end

    private def join_q_and_gate_grads!(q_grad : Array(Float32), gate_grad : Array(Float32),
                                       combined : Array(Float32), rows : Int32) : Nil
      heads = @hparams.n_head
      head_dim = @hparams.head_dim
      q_dim = heads * head_dim
      rows.times do |row|
        heads.times do |head|
          dst = row * 2 * q_dim + head * 2 * head_dim
          src = row * q_dim + head * head_dim
          head_dim.times do |d|
            combined[dst + d] = q_grad[src + d]
            combined[dst + head_dim + d] = gate_grad[src + d]
          end
        end
      end
    end

    private def normalized_rotated_q(q_raw : Array(Float32), rows : Int32,
                                     position_start : Int32) : Array(Float32)
      heads = @hparams.n_head
      head_dim = @hparams.head_dim
      q_dim = heads * head_dim
      q = q_raw.dup
      rows.times do |row|
        heads.times do |head|
          offset = row * q_dim + head * head_dim
          rmsnorm_slice!(q, offset, head_dim, @q_norm)
          rope_slice!(q, offset, head_dim, position_start + row)
        end
      end
      q
    end

    private def normalized_rotated_k(k_raw : Array(Float32), rows : Int32,
                                     position_start : Int32) : Array(Float32)
      heads = @hparams.n_head_kv
      head_dim = @hparams.head_dim
      kv_dim = heads * head_dim
      k = k_raw.dup
      rows.times do |row|
        heads.times do |head|
          offset = row * kv_dim + head * head_dim
          rmsnorm_slice!(k, offset, head_dim, @k_norm)
          rope_slice!(k, offset, head_dim, position_start + row)
        end
      end
      k
    end

    private def rmsnorm_rows(input : Array(Float32), rows : Int32, dim : Int32,
                             weight : Array(Float32), eps : Float32) : Array(Float32)
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        sum_squares = 0.0_f64
        dim.times { |i| value = input[offset + i].to_f64; sum_squares += value * value }
        inv = (1.0 / Math.sqrt(sum_squares / dim.to_f64 + eps.to_f64)).to_f32
        dim.times { |i| output[offset + i] = input[offset + i] * inv * weight[i] }
      end
      output
    end

    private def rmsnorm_slice!(values : Array(Float32), offset : Int32, dim : Int32,
                               weight : Array(Float32)) : Nil
      sum_squares = 0.0_f64
      dim.times do |i|
        value = values[offset + i].to_f64
        sum_squares += value * value
      end
      inv = (1.0 / Math.sqrt(sum_squares / dim.to_f64 + @hparams.rms_eps.to_f64)).to_f32
      dim.times { |i| values[offset + i] = values[offset + i] * inv * weight[i] }
    end

    private def rope_slice!(values : Array(Float32), offset : Int32, head_dim : Int32,
                            position : Int32, inverse : Bool = false) : Nil
      n_rot = @hparams.rope_dim_count
      half = n_rot // 2
      half.times do |i|
        freq = 1.0_f32 / (@hparams.rope_freq_base ** (2.0_f32 * i / n_rot))
        theta = position.to_f32 * freq * (inverse ? -1.0_f32 : 1.0_f32)
        cos_t = Math.cos(theta)
        sin_t = Math.sin(theta)
        i0 = offset + i
        i1 = offset + i + half
        x0 = values[i0]
        x1 = values[i1]
        values[i0] = x0 * cos_t - x1 * sin_t
        values[i1] = x0 * sin_t + x1 * cos_t
      end
    end

    private def unrotate_heads!(values : Array(Float32), rows : Int32, heads : Int32) : Nil
      head_dim = @hparams.head_dim
      row_dim = heads * head_dim
      rows.times do |row|
        heads.times do |head|
          rope_slice!(values, row * row_dim + head * head_dim, head_dim, row, inverse: true)
        end
      end
    end

    private def causal_gqa_attention(q : Array(Float32), keys : Array(Float32),
                                     values : Array(Float32), rows : Int32,
                                     key_rows : Int32, prefix_rows : Int32) : Array(Float32)
      heads = @hparams.n_head
      kv_heads = @hparams.n_head_kv
      head_dim = @hparams.head_dim
      q_dim = heads * head_dim
      kv_dim = kv_heads * head_dim
      heads_per_group = heads // kv_heads
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      output = Array(Float32).new(rows * q_dim, 0.0_f32)

      heads.times do |head|
        kv_head = head // heads_per_group
        q_head = gather_head(q, rows, q_dim, head, head_dim)
        k_head = gather_head(keys, key_rows, kv_dim, kv_head, head_dim)
        v_head = gather_head(values, key_rows, kv_dim, kv_head, head_dim)
        scores = MathOps.gemm(q_head, k_head, rows, key_rows, head_dim, transpose_b: true)
        scores.size.times { |i| scores[i] *= scale }
        rows.times do |row|
          allowed = prefix_rows + row + 1
          softmax_prefix!(scores, row * key_rows, key_rows, allowed)
        end
        output_head = MathOps.gemm(scores, v_head, rows, head_dim, key_rows)
        scatter_head!(output_head, output, rows, q_dim, head, head_dim)
      end
      output
    end

    private struct AttentionGradients
      getter q : Array(Float32)
      getter k : Array(Float32)
      getter v : Array(Float32)
      getter gate : Array(Float32)

      def initialize(@q : Array(Float32), @k : Array(Float32),
                     @v : Array(Float32), @gate : Array(Float32))
      end
    end

    private def attention_backward(q : Array(Float32), k : Array(Float32),
                                   v : Array(Float32), gate_pre : Array(Float32),
                                   output_grad : Array(Float32), rows : Int32) : AttentionGradients
      heads = @hparams.n_head
      kv_heads = @hparams.n_head_kv
      head_dim = @hparams.head_dim
      q_dim = heads * head_dim
      kv_dim = kv_heads * head_dim
      heads_per_group = heads // kv_heads
      scale = (1.0 / Math.sqrt(head_dim.to_f64)).to_f32
      dq = Array(Float32).new(rows * q_dim, 0.0_f32)
      dk = Array(Float32).new(rows * kv_dim, 0.0_f32)
      dv = Array(Float32).new(rows * kv_dim, 0.0_f32)
      dgate = Array(Float32).new(rows * q_dim, 0.0_f32)

      heads.times do |head|
        kv_head = head // heads_per_group
        q_head = gather_head(q, rows, q_dim, head, head_dim)
        k_head = gather_head(k, rows, kv_dim, kv_head, head_dim)
        v_head = gather_head(v, rows, kv_dim, kv_head, head_dim)
        scores = MathOps.gemm(q_head, k_head, rows, rows, head_dim, transpose_b: true)
        scores.size.times { |i| scores[i] *= scale }
        rows.times do |row|
          softmax_prefix!(scores, row * rows, rows, row + 1)
        end

        probabilities = scores
        attended = MathOps.gemm(probabilities, v_head, rows, head_dim, rows)
        dattended = Array(Float32).new(rows * head_dim, 0.0_f32)
        row_off = head * head_dim
        rows.times do |row|
          head_dim.times do |d|
            index = row * q_dim + row_off + d
            sig = sigmoid(gate_pre[index])
            dattended[row * head_dim + d] = output_grad[index] * sig
            dgate[index] = output_grad[index] * attended[row * head_dim + d] * sig * (1.0_f32 - sig)
          end
        end

        dprobabilities = MathOps.gemm(dattended, v_head, rows, rows, head_dim, transpose_b: true)
        dvalues = MathOps.gemm(probabilities, dattended, rows, head_dim, rows, transpose_a: true)

        dscores = Array(Float32).new(rows * rows, 0.0_f32)
        rows.times do |row|
          allowed = row + 1
          dot = 0.0_f32
          allowed.times do |col|
            index = row * rows + col
            dot += dprobabilities[index] * probabilities[index]
          end
          allowed.times do |col|
            index = row * rows + col
            dscores[index] = probabilities[index] * (dprobabilities[index] - dot)
          end
        end

        dqueries = MathOps.gemm(dscores, k_head, rows, head_dim, rows)
        dkeys = MathOps.gemm(dscores, q_head, rows, head_dim, rows, transpose_a: true)
        dqueries.size.times { |i| dqueries[i] *= scale }
        dkeys.size.times { |i| dkeys[i] *= scale }

        scatter_head_add!(dqueries, dq, rows, q_dim, head, head_dim)
        scatter_head_add!(dkeys, dk, rows, kv_dim, kv_head, head_dim)
        scatter_head_add!(dvalues, dv, rows, kv_dim, kv_head, head_dim)
      end
      AttentionGradients.new(dq, dk, dv, dgate)
    end

    private def softmax_prefix!(values : Array(Float32), offset : Int32,
                                total : Int32, allowed : Int32) : Nil
      raise ArgumentError.new("invalid causal attention prefix") unless allowed > 0 && allowed <= total
      max = values[offset]
      allowed.times do |i|
        value = values[offset + i]
        max = value if value > max
      end
      sum = 0.0_f32
      allowed.times do |i|
        value = Math.exp(values[offset + i] - max)
        values[offset + i] = value
        sum += value
      end
      inverse = 1.0_f32 / sum
      allowed.times { |i| values[offset + i] *= inverse }
      (allowed...total).each { |i| values[offset + i] = 0.0_f32 }
    end

    private def gather_head(values : Array(Float32), rows : Int32, row_dim : Int32,
                            head : Int32, head_dim : Int32) : Array(Float32)
      Array(Float32).new(rows * head_dim) do |i|
        row = i // head_dim
        dim = i % head_dim
        values[row * row_dim + head * head_dim + dim]
      end
    end

    private def scatter_head!(source : Array(Float32), destination : Array(Float32),
                              rows : Int32, row_dim : Int32, head : Int32,
                              head_dim : Int32) : Nil
      rows.times do |row|
        head_dim.times do |d|
          destination[row * row_dim + head * head_dim + d] = source[row * head_dim + d]
        end
      end
    end

    private def scatter_head_add!(source : Array(Float32), destination : Array(Float32),
                                  rows : Int32, row_dim : Int32, head : Int32,
                                  head_dim : Int32) : Nil
      rows.times do |row|
        head_dim.times do |d|
          destination[row * row_dim + head * head_dim + d] += source[row * head_dim + d]
        end
      end
    end

    private def rmsnorm_backward_rows(input : Array(Float32), grad : Array(Float32),
                                      rows : Int32, dim : Int32, weight : Array(Float32),
                                      eps : Float32) : Array(Float32)
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        offset = row * dim
        sum_squares = 0.0_f64
        dim.times { |i| value = input[offset + i].to_f64; sum_squares += value * value }
        inv = 1.0 / Math.sqrt(sum_squares / dim.to_f64 + eps.to_f64)
        dot = 0.0_f64
        dim.times do |i|
          dot += grad[offset + i].to_f64 * weight[i].to_f64 * input[offset + i].to_f64
        end
        correction = inv * inv * inv * dot / dim.to_f64
        dim.times do |i|
          output[offset + i] = (grad[offset + i].to_f64 * weight[i].to_f64 * inv - input[offset + i].to_f64 * correction).to_f32
        end
      end
      output
    end

    private def rmsnorm_heads_backward(input : Array(Float32), grad : Array(Float32),
                                       rows : Int32, heads : Int32,
                                       weight : Array(Float32)) : Array(Float32)
      head_dim = @hparams.head_dim
      row_dim = heads * head_dim
      output = Array(Float32).new(input.size, 0.0_f32)
      rows.times do |row|
        heads.times do |head|
          offset = row * row_dim + head * head_dim
          sum_squares = 0.0_f64
          head_dim.times do |i|
            value = input[offset + i].to_f64
            sum_squares += value * value
          end
          inv = 1.0 / Math.sqrt(sum_squares / head_dim.to_f64 + @hparams.rms_eps.to_f64)
          dot = 0.0_f64
          head_dim.times do |i|
            dot += grad[offset + i].to_f64 * weight[i].to_f64 * input[offset + i].to_f64
          end
          correction = inv * inv * inv * dot / head_dim.to_f64
          head_dim.times do |i|
            output[offset + i] = (grad[offset + i].to_f64 * weight[i].to_f64 * inv - input[offset + i].to_f64 * correction).to_f32
          end
        end
      end
      output
    end

    private def silu(value : Float32) : Float32
      value * sigmoid(value)
    end

    private def sigmoid(value : Float32) : Float32
      1.0_f32 / (1.0_f32 + Math.exp(-value))
    end

    private def validate_shapes!(lw : Qwen35FullAttnWeights, hp : Qwen35Hparams) : Nil
      hidden = hp.n_embd
      q_dim = hp.n_head * hp.head_dim
      kv_dim = hp.n_head_kv * hp.head_dim
      unless hidden > 0 && hp.n_ff > 0 && hp.n_head > 0 && hp.n_head_kv > 0 && hp.head_dim > 0
        raise ArgumentError.new("Qwen3.5 block dimensions must be positive")
      end
      raise ArgumentError.new("n_head must be divisible by n_head_kv") unless hp.n_head % hp.n_head_kv == 0
      raise ArgumentError.new("rope dimension must be positive, even, and no larger than head_dim") unless hp.rope_dim_count > 0 && hp.rope_dim_count.even? && hp.rope_dim_count <= hp.head_dim
      raise ArgumentError.new("context length must be positive") unless hp.context_length > 0
      raise ArgumentError.new("attn_norm size mismatch") unless lw.attn_norm.size == hidden
      raise ArgumentError.new("post_attention_norm size mismatch") unless lw.post_attention_norm.size == hidden
      raise ArgumentError.new("attn_q_norm size mismatch") unless lw.attn_q_norm.size == hp.head_dim
      raise ArgumentError.new("attn_k_norm size mismatch") unless lw.attn_k_norm.size == hp.head_dim
      shape!(lw.attn_q_qw, hidden, 2 * q_dim, "Q+gate")
      shape!(lw.attn_k_qw, hidden, kv_dim, "K")
      shape!(lw.attn_v_qw, hidden, kv_dim, "V")
      shape!(lw.attn_output_qw, q_dim, hidden, "attention output")
      shape!(lw.ffn_gate_qw, hidden, hp.n_ff, "FFN gate")
      shape!(lw.ffn_up_qw, hidden, hp.n_ff, "FFN up")
      shape!(lw.ffn_down_qw, hp.n_ff, hidden, "FFN down")
    end

    private def shape!(weight : QuantWeight, input : Int32, output : Int32,
                       label : String) : Nil
      unless weight.in_dim == input && weight.out_dim == output
        raise ArgumentError.new("#{label} projection shape mismatch: expected [#{output}, #{input}], got [#{weight.out_dim}, #{weight.in_dim}]")
      end
    end
  end
end
