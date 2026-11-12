# Differentiable CPU recurrence for the Qwen 3.5 gated DeltaNet block.
# The forward pass keeps one value-head state live at a time. Backward stores
# checkpoints for one head and recomputes one bounded segment while reversing.

module ML::GGUF::Qwen35QLoRA
  module Delta
    CHECKPOINT_ROWS = 16

    {% if flag?(:darwin) %}
      @[Link(framework: "Accelerate")]
      lib LibQwen35QLoRADeltaAccelerate
        fun cblas_sgemv(layout : Int32, transpose : Int32,
                        m : Int32, n : Int32,
                        alpha : Float32, matrix : Float32*, lda : Int32,
                        vector : Float32*, inc_vector : Int32,
                        beta : Float32, output : Float32*, inc_output : Int32) : Void
      end
    {% end %}

    # Recurrent scan over token-major q/k/v arrays. `decay` and `beta` are
    # already exp/sigmoid outputs, and every value head starts from zero state.
    def self.forward(q : Array(Float32), k : Array(Float32), v : Array(Float32),
                     decay : Array(Float32), beta : Array(Float32),
                     rows : Int32, h_k : Int32, h_v : Int32,
                     size : Int32) : Array(Float32)
      validate_inputs!(q, k, v, decay, beta, nil, rows, h_k, h_v, size)

      output = Array(Float32).new(checked_count(rows, h_v, size), 0.0_f32)
      state_size = checked_count(size, size)
      state = Array(Float32).new(state_size, 0.0_f32)
      prediction = Array(Float32).new(size, 0.0_f32)
      delta = Array(Float32).new(size, 0.0_f32)
      scale = 1.0_f32 / Math.sqrt(size.to_f64).to_f32

      q_ptr = q.to_unsafe
      k_ptr = k.to_unsafe
      v_ptr = v.to_unsafe
      decay_ptr = decay.to_unsafe
      beta_ptr = beta.to_unsafe
      output_ptr = output.to_unsafe
      state_ptr = state.to_unsafe
      prediction_ptr = prediction.to_unsafe
      delta_ptr = delta.to_unsafe
      q_row = h_k * size
      v_row = h_v * size
      output_row = h_v * size

      h_v.times do |head|
        state.fill(0.0_f32)
        q_head = (head % h_k) * size
        rows.times do |row|
          q_at = q_ptr + row * q_row + q_head
          k_at = k_ptr + row * q_row + q_head
          v_at = v_ptr + row * v_row + head * size
          y_at = output_ptr + row * output_row + head * size
          decay_value = decay_ptr[row * h_v + head]
          beta_value = beta_ptr[row * h_v + head]

          advance_in_place!(state_ptr, k_at, v_at, decay_value, beta_value,
            prediction_ptr, delta_ptr, size)
          gemv_nt!(state_ptr, q_at, y_at, size)
          size.times { |d| y_at[d] *= scale }
        end
      end

      output
    end

    # Reverse-mode derivative of `forward`, returning gradients in q/k/v/
    # decay/beta order. The recurrent state is checkpointed per head, never for
    # all heads at once; within a segment, only its (at most 17) state matrices
    # are materialized.
    def self.backward(q : Array(Float32), k : Array(Float32), v : Array(Float32),
                      decay : Array(Float32), beta : Array(Float32),
                      grad : Array(Float32), rows : Int32, h_k : Int32,
                      h_v : Int32, size : Int32) : Tuple(
      Array(Float32), Array(Float32), Array(Float32),
      Array(Float32), Array(Float32),
    )
      validate_inputs!(q, k, v, decay, beta, grad, rows, h_k, h_v, size)

      grad_q = Array(Float32).new(q.size, 0.0_f32)
      grad_k = Array(Float32).new(k.size, 0.0_f32)
      grad_v = Array(Float32).new(v.size, 0.0_f32)
      grad_decay = Array(Float32).new(decay.size, 0.0_f32)
      grad_beta = Array(Float32).new(beta.size, 0.0_f32)

      state_size = checked_count(size, size)
      segment_count = ((rows.to_i64 + CHECKPOINT_ROWS - 1) // CHECKPOINT_ROWS).to_i32
      checkpoint_size = checked_count(segment_count, state_size)
      local_state_size = checked_count(CHECKPOINT_ROWS + 1, state_size)
      checkpoints = Array(Float32).new(checkpoint_size, 0.0_f32)
      local_states = Array(Float32).new(local_state_size, 0.0_f32)
      state = Array(Float32).new(state_size, 0.0_f32)
      grad_state = Array(Float32).new(state_size, 0.0_f32)
      prediction = Array(Float32).new(size, 0.0_f32)
      delta = Array(Float32).new(size, 0.0_f32)
      grad_delta = Array(Float32).new(size, 0.0_f32)
      grad_p = Array(Float32).new(size, 0.0_f32)
      grad_q_head = Array(Float32).new(size, 0.0_f32)
      grad_k_head = Array(Float32).new(size, 0.0_f32)
      grad_k_from_state = Array(Float32).new(size, 0.0_f32)

      q_ptr = q.to_unsafe
      k_ptr = k.to_unsafe
      v_ptr = v.to_unsafe
      decay_ptr = decay.to_unsafe
      beta_ptr = beta.to_unsafe
      grad_ptr = grad.to_unsafe
      grad_q_ptr = grad_q.to_unsafe
      grad_k_ptr = grad_k.to_unsafe
      grad_v_ptr = grad_v.to_unsafe
      grad_decay_ptr = grad_decay.to_unsafe
      grad_beta_ptr = grad_beta.to_unsafe
      checkpoints_ptr = checkpoints.to_unsafe
      local_states_ptr = local_states.to_unsafe
      state_ptr = state.to_unsafe
      grad_state_ptr = grad_state.to_unsafe
      prediction_ptr = prediction.to_unsafe
      delta_ptr = delta.to_unsafe
      grad_delta_ptr = grad_delta.to_unsafe
      grad_p_ptr = grad_p.to_unsafe
      grad_q_head_ptr = grad_q_head.to_unsafe
      grad_k_head_ptr = grad_k_head.to_unsafe
      grad_k_from_state_ptr = grad_k_from_state.to_unsafe

      q_row = h_k * size
      v_row = h_v * size
      grad_row = h_v * size
      scale = 1.0_f32 / Math.sqrt(size.to_f64).to_f32

      h_v.times do |head|
        # Checkpoints are state-before-token snapshots at each segment boundary.
        state.fill(0.0_f32)
        rows.times do |row|
          if row % CHECKPOINT_ROWS == 0
            segment = row // CHECKPOINT_ROWS
            copy!(state_ptr, checkpoints_ptr + segment * state_size, state_size)
          end

          k_at = k_ptr + row * q_row + (head % h_k) * size
          v_at = v_ptr + row * v_row + head * size
          decay_value = decay_ptr[row * h_v + head]
          beta_value = beta_ptr[row * h_v + head]
          advance_in_place!(state_ptr, k_at, v_at, decay_value, beta_value,
            prediction_ptr, delta_ptr, size)
        end

        grad_state.fill(0.0_f32)
        (segment_count - 1).downto(0) do |segment|
          start_row = segment * CHECKPOINT_ROWS
          segment_rows = Math.min(CHECKPOINT_ROWS, rows - start_row)
          first_state = local_states_ptr
          copy!(checkpoints_ptr + segment * state_size, first_state, state_size)

          # Recompute the bounded segment from its checkpoint, retaining the
          # state after every token only for this one head and segment.
          segment_rows.times do |offset|
            row = start_row + offset
            previous = local_states_ptr + offset * state_size
            current = local_states_ptr + (offset + 1) * state_size
            k_at = k_ptr + row * q_row + (head % h_k) * size
            v_at = v_ptr + row * v_row + head * size
            decay_value = decay_ptr[row * h_v + head]
            beta_value = beta_ptr[row * h_v + head]
            advance_to!(previous, current, k_at, v_at, decay_value, beta_value,
              prediction_ptr, delta_ptr, size)
          end

          (segment_rows - 1).downto(0) do |offset|
            row = start_row + offset
            previous = local_states_ptr + offset * state_size
            current = local_states_ptr + (offset + 1) * state_size
            q_at = q_ptr + row * q_row + (head % h_k) * size
            k_at = k_ptr + row * q_row + (head % h_k) * size
            v_at = v_ptr + row * v_row + head * size
            gy_at = grad_ptr + row * grad_row + head * size
            gq_at = grad_q_ptr + row * q_row + (head % h_k) * size
            gk_at = grad_k_ptr + row * q_row + (head % h_k) * size
            decay_index = row * h_v + head
            beta_index = decay_index
            decay_value = decay_ptr[decay_index]
            beta_value = beta_ptr[beta_index]

            # p = (decay * S_prev) k. The scalar-after-matvec form avoids a
            # second state-sized temporary and differs only by FP32 roundoff.
            gemv_nt!(previous, k_at, prediction_ptr, size)
            size.times { |d| prediction_ptr[d] *= decay_value }

            # d = beta * (v - p), and G_d = G_state k plus the output's
            # low-rank outer-product contribution without materializing it.
            gemv_nt!(grad_state_ptr, k_at, grad_delta_ptr, size)
            q_dot_k = 0.0_f32
            gy_dot_delta = 0.0_f32
            size.times do |d|
              q_dot_k += q_at[d] * k_at[d]
            end
            size.times do |d|
              grad_delta_ptr[d] += gy_at[d] * q_dot_k * scale
              delta_ptr[d] = beta_value * (v_at[d] - prediction_ptr[d])
              grad_p_ptr[d] = -beta_value * grad_delta_ptr[d]
              grad_v_ptr[row * v_row + head * size + d] = beta_value * grad_delta_ptr[d]
              gy_dot_delta += gy_at[d] * delta_ptr[d]
            end

            # q gradient uses the post-update recurrent state.
            gemv_t!(current, gy_at, grad_q_head_ptr, size)
            size.times do |d|
              gq_at[d] += grad_q_head_ptr[d] * scale
            end

            # k gradients come from the rank-one update and from its prediction.
            gemv_t!(grad_state_ptr, delta_ptr, grad_k_head_ptr, size)
            gemv_t!(previous, grad_p_ptr, grad_k_from_state_ptr, size)
            size.times do |d|
              gk_at[d] += grad_k_head_ptr[d] +
                          q_at[d] * gy_dot_delta * scale +
                          decay_value * grad_k_from_state_ptr[d]
            end

            residual_dot = 0.0_f32
            size.times do |d|
              residual = v_at[d] - prediction_ptr[d]
              residual_dot += grad_delta_ptr[d] * residual
            end
            decay_grad = 0.0_f32
            size.times do |row_in_state|
              row_offset = row_in_state * size
              output_gradient = gy_at[row_in_state] * scale
              prediction_gradient = grad_p_ptr[row_in_state]
              size.times do |column_in_state|
                index = row_offset + column_in_state
                state_gradient = grad_state_ptr[index] +
                                 output_gradient * q_at[column_in_state] +
                                 prediction_gradient * k_at[column_in_state]
                decay_grad += state_gradient * previous[index]
                grad_state_ptr[index] = decay_value * state_gradient
              end
            end
            grad_decay_ptr[decay_index] = decay_grad
            grad_beta_ptr[beta_index] = residual_dot
          end
        end
      end

      {grad_q, grad_k, grad_v, grad_decay, grad_beta}
    end

    private def self.validate_inputs!(q : Array(Float32), k : Array(Float32),
                                      v : Array(Float32), decay : Array(Float32),
                                      beta : Array(Float32), grad : Array(Float32)?,
                                      rows : Int32, h_k : Int32, h_v : Int32,
                                      size : Int32) : Nil
      raise ArgumentError.new("DeltaNet dimensions must be positive") unless rows > 0 && h_k > 0 && h_v > 0 && size > 0

      q_size = checked_count(rows, h_k, size)
      v_size = checked_count(rows, h_v, size)
      gate_size = checked_count(rows, h_v)
      state_size = checked_count(size, size)
      raise ArgumentError.new("DeltaNet q size mismatch: expected #{q_size}, got #{q.size}") unless q.size == q_size
      raise ArgumentError.new("DeltaNet k size mismatch: expected #{q_size}, got #{k.size}") unless k.size == q_size
      raise ArgumentError.new("DeltaNet v size mismatch: expected #{v_size}, got #{v.size}") unless v.size == v_size
      raise ArgumentError.new("DeltaNet decay size mismatch: expected #{gate_size}, got #{decay.size}") unless decay.size == gate_size
      raise ArgumentError.new("DeltaNet beta size mismatch: expected #{gate_size}, got #{beta.size}") unless beta.size == gate_size
      if gradient = grad
        raise ArgumentError.new("DeltaNet output gradient size mismatch: expected #{v_size}, got #{gradient.size}") unless gradient.size == v_size
      end
      raise ArgumentError.new("DeltaNet state is too large") if state_size <= 0
    end

    private def self.checked_count(a : Int32, b : Int32, c : Int32 = 1) : Int32
      raise ArgumentError.new("DeltaNet dimensions must be positive") unless a > 0 && b > 0 && c > 0
      first = a.to_i64 * b.to_i64
      raise ArgumentError.new("DeltaNet array is too large") if first > Int32::MAX
      raise ArgumentError.new("DeltaNet array is too large") if first > Int32::MAX.to_i64 // c
      (first * c).to_i32
    end

    private def self.copy!(source : Pointer(Float32), destination : Pointer(Float32),
                           count : Int32) : Nil
      count.times { |i| destination[i] = source[i] }
    end

    # One exact-shaped Qwen CPU update: decay the state, compute its key
    # prediction, apply beta*(v-p) as a rank-one update.
    private def self.advance_in_place!(state : Pointer(Float32), k : Pointer(Float32),
                                       v : Pointer(Float32), decay : Float32,
                                       beta : Float32, prediction : Pointer(Float32),
                                       delta : Pointer(Float32), size : Int32) : Nil
      state_count = size * size
      state_count.times { |i| state[i] *= decay }
      gemv_nt!(state, k, prediction, size)
      size.times { |d| delta[d] = beta * (v[d] - prediction[d]) }
      size.times do |row|
        row_offset = row * size
        row_delta = delta[row]
        size.times do |column|
          index = row_offset + column
          state[index] += k[column] * row_delta
        end
      end
    end

    private def self.advance_to!(previous : Pointer(Float32), current : Pointer(Float32),
                                 k : Pointer(Float32), v : Pointer(Float32),
                                 decay : Float32, beta : Float32,
                                 prediction : Pointer(Float32), delta : Pointer(Float32),
                                 size : Int32) : Nil
      state_count = size * size
      state_count.times { |i| current[i] = previous[i] * decay }
      gemv_nt!(current, k, prediction, size)
      size.times { |d| delta[d] = beta * (v[d] - prediction[d]) }
      size.times do |row|
        row_offset = row * size
        row_delta = delta[row]
        size.times do |column|
          index = row_offset + column
          current[index] += k[column] * row_delta
        end
      end
    end

    private def self.gemv_nt!(matrix : Pointer(Float32), vector : Pointer(Float32),
                              output : Pointer(Float32), size : Int32) : Nil
      {% if flag?(:darwin) %}
        LibQwen35QLoRADeltaAccelerate.cblas_sgemv(
          101_i32, 111_i32, size, size, 1.0_f32,
          matrix, size, vector, 1_i32, 0.0_f32, output, 1_i32,
        )
      {% else %}
        size.times do |row|
          sum = 0.0_f32
          row_offset = row * size
          size.times { |column| sum += matrix[row_offset + column] * vector[column] }
          output[row] = sum
        end
      {% end %}
    end

    private def self.gemv_t!(matrix : Pointer(Float32), vector : Pointer(Float32),
                             output : Pointer(Float32), size : Int32) : Nil
      {% if flag?(:darwin) %}
        LibQwen35QLoRADeltaAccelerate.cblas_sgemv(
          101_i32, 112_i32, size, size, 1.0_f32,
          matrix, size, vector, 1_i32, 0.0_f32, output, 1_i32,
        )
      {% else %}
        size.times do |column|
          sum = 0.0_f32
          size.times { |row| sum += matrix[row * size + column] * vector[row] }
          output[column] = sum
        end
      {% end %}
    end
  end
end
