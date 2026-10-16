require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
module ML::Audio::Drama3::Resident
  BATCH_SOURCE = {{ read_file("#{__DIR__}/kernels/resident_batch.metal") }}

  # Row batches stay resident and use the same scalar kernels/reduction order
  # as their single-row counterparts. The batch axis is always the outermost
  # axis; expert ranks within a token remain in token-major, rank-major order.
  class Context
    def matvec_rows(matrix : Matrix, x : Buffer, rows : Int32,
                    dense_reduction : String? = nil) : Buffer
      check_owner!(x)
      raise ArgumentError.new("resident matvec row count") unless rows > 0
      input_count = checked_batch_count!("matvec input", [rows, matrix.cols])
      raise ArgumentError.new("resident matvec rows width") unless x.count == input_count
      output_count = checked_batch_count!("matvec output", [rows, matrix.rows])

      if matrix.scales
        # `matvec` on a packed matrix selects Matrix#expert. Generate that
        # fixed selection on-device and delegate to the existing affine path.
        raise ArgumentError.new("resident affine groups") if matrix.cols // matrix.group_size > 8192
        validate_expert_reduction("native")
        weight_buffer(matrix.tensor)
        weight_buffer(matrix.scales.not_nil!)
        weight_buffer(matrix.biases.not_nil!)
        pipeline("dr3_experts", SOURCE)
        pipeline("dr3_batch_fixed_ids", BATCH_SOURCE)
        ids = alloc(rows)
        dispatch("dr3_batch_fixed_ids", BATCH_SOURCE, [ids],
          [rows.to_u64, matrix.expert.to_u64], rows)
        return experts(matrix, x, ids, per_rank_input: true)
      end

      reduction = dense_reduction ? validate_dense_reduction(dense_reduction) : @dense_reduction
      weight_encoding = encoding(matrix.tensor)
      weight = weight_buffer(matrix.tensor)
      output = alloc(output_count)

      threadgroup_width = 64
      selected_pipeline = if reduction == "simd32"
                            candidate = pipeline("dr3_dense_rows_simd32", BATCH_SOURCE)
                            if candidate.thread_execution_width == 32
                              threadgroup_width = 32
                              candidate
                            else
                              pipeline("dr3_dense_rows", BATCH_SOURCE)
                            end
                          else
                            name = reduction == "simd" ? "dr3_dense_rows_simd" : "dr3_dense_rows"
                            pipeline(name, BATCH_SOURCE)
                          end
      enc = encoder
      enc.set_pipeline(selected_pipeline)
      enc.set_buffer(weight.buffer, 0)
      enc.set_buffer(x.buffer, 1, offset: x.offset)
      enc.set_buffer(output.buffer, 2, offset: output.offset)
      params = [matrix.rows.to_u64, matrix.cols.to_u64, weight.offset.to_u64,
                weight_encoding, matrix.expert.to_u64]
      enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 3)
      enc.dispatch_threadgroups({matrix.rows, rows, 1}, {threadgroup_width, 1, 1})
      @pending.concat([x, output])
      output
    end

    # Scores are row-major [rows, experts]. Returned expert IDs and weights
    # are separate row-major [rows, top_k] buffers.
    def route_rows(scores : Buffer, rows : Int32, experts : Int32,
                   top_k : Int32, norm_topk : Bool) : Tuple(Buffer, Buffer)
      check_owner!(scores)
      raise ArgumentError.new("resident route dimensions") unless rows > 0 && experts.in?(1..512) && top_k.in?(1..experts)
      score_count = checked_batch_count!("route scores", [rows, experts])
      raise ArgumentError.new("resident route score shape") unless scores.count == score_count
      output_count = checked_batch_count!("route output", [rows, top_k])

      ids = alloc(output_count)
      weights = alloc(output_count)
      enc = encoder
      enc.set_pipeline(pipeline("dr3_route_rows", BATCH_SOURCE))
      enc.set_buffer(scores.buffer, 0, offset: scores.offset)
      enc.set_buffer(ids.buffer, 1, offset: ids.offset)
      enc.set_buffer(weights.buffer, 2, offset: weights.offset)
      params = [rows.to_u64, experts.to_u64, top_k.to_u64, norm_topk ? 1_u64 : 0_u64]
      enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 3)
      # One lane per token keeps the original sequential softmax and argmax
      # traversal without introducing a parallel reduction or tie change.
      enc.dispatch_threadgroups({rows, 1, 1}, {1, 1, 1})
      @pending.concat([scores, ids, weights])
      {ids, weights}
    end

    # `down` is token-major then rank-major, with `hidden` values per route.
    def expert_sum_rows(down : Buffer, weights : Buffer, rows : Int32,
                        hidden : Int32, top_k : Int32) : Buffer
      check_owner!(down)
      check_owner!(weights)
      raise ArgumentError.new("resident expert sum dimensions") unless rows > 0 && hidden > 0 && top_k > 0
      output_count = checked_batch_count!("expert sum output", [rows, hidden])
      weight_count = checked_batch_count!("expert sum weights", [rows, top_k])
      down_count = checked_batch_count!("expert sum inputs", [rows, top_k, hidden])
      raise ArgumentError.new("resident expert sum input shape") unless down.count == down_count
      raise ArgumentError.new("resident expert sum weight shape") unless weights.count == weight_count

      output = alloc(output_count)
      dispatch("dr3_expert_sum_rows", BATCH_SOURCE, [down, weights, output],
        [rows.to_u64, hidden.to_u64, top_k.to_u64], output_count)
      output
    end

    # Apply each row's scalar shared-expert gate in place. The stable sigmoid
    # expression is shared with the existing backbone weighted-add kernel.
    def weighted_add_rows!(result : Buffer, shared : Buffer, shared_gate : Buffer,
                           rows : Int32, hidden : Int32) : Nil
      check_owner!(result)
      check_owner!(shared)
      check_owner!(shared_gate)
      raise ArgumentError.new("resident weighted add dimensions") unless rows > 0 && hidden > 0
      count = checked_batch_count!("weighted add rows", [rows, hidden])
      raise ArgumentError.new("resident weighted add result shape") unless result.count == count
      raise ArgumentError.new("resident weighted add shared shape") unless shared.count == count
      raise ArgumentError.new("resident weighted add gate shape") unless shared_gate.count == rows

      dispatch("dr3_weighted_add_rows", BATCH_SOURCE,
        [result, shared, shared_gate], [rows.to_u64, hidden.to_u64], count)
    end

    # Each flattened ID represents one (token, route-rank) pair. Shared token
    # inputs are expanded on-device only when the caller has not supplied a
    # distinct input per flattened ID; the arithmetic remains in #experts.
    def experts_rows(matrix : Matrix, x : Buffer, ids : Buffer, rows : Int32,
                     top_k : Int32, per_rank_input : Bool = false,
                     reduction : String = "native") : Buffer
      check_owner!(x)
      check_owner!(ids)
      raise ArgumentError.new("resident expert row dimensions") unless rows > 0 && top_k > 0
      selected_reduction = validate_expert_reduction(reduction)
      raise ArgumentError.new("resident batched experts require affine weights") unless matrix.scales && matrix.biases
      id_count = checked_batch_count!("expert row IDs", [rows, top_k])
      raise ArgumentError.new("resident expert row ID shape") unless ids.count == id_count
      row_input_count = checked_batch_count!("expert row input", [rows, matrix.cols])
      input_count = per_rank_input ? checked_batch_count!("expert rank input", [id_count, matrix.cols]) : row_input_count
      raise ArgumentError.new("resident batched expert input width") unless x.count == input_count
      checked_batch_count!("expert row output", [matrix.rows, id_count])
      expanded_count = checked_batch_count!("expanded expert row input", [id_count, matrix.cols])
      validate_mlx_q4!(matrix) if selected_reduction == "mlx_q4"
      raise ArgumentError.new("resident affine groups") if matrix.cols // matrix.group_size > 8192

      # Resolve weight extents and pipelines before encoding the optional
      # expansion, so shape, owner, reduction, and weight errors are fail-fast.
      weight_buffer(matrix.tensor)
      weight_buffer(matrix.scales.not_nil!)
      weight_buffer(matrix.biases.not_nil!)
      expert_pipeline = selected_reduction == "mlx_q4" ? "dr3_expert_mlx_q4" : "dr3_experts"
      expert_source = selected_reduction == "mlx_q4" ? MLX_Q4_SOURCE : SOURCE
      pipeline(expert_pipeline, expert_source)

      input = if per_rank_input
                x
              else
                pipeline("dr3_batch_repeat_rows", BATCH_SOURCE)
                expanded = alloc(expanded_count)
                dispatch("dr3_batch_repeat_rows", BATCH_SOURCE, [x, expanded],
                  [rows.to_u64, matrix.cols.to_u64, top_k.to_u64], expanded_count)
                expanded
              end
      experts(matrix, input, ids, per_rank_input: true, reduction: selected_reduction)
    end

    private def checked_batch_count!(name : String, factors : Array(Int32)) : Int32
      count = 1_i64
      factors.each do |factor|
        unless factor > 0 && count <= Int32::MAX.to_i64 // factor.to_i64
          raise ArgumentError.new("resident #{name} exceeds Int32 element capacity")
        end
        count *= factor.to_i64
      end
      bytes = count * sizeof(Float32).to_i64
      if @max_buffer_length > 0 && bytes > @max_buffer_length
        raise ArgumentError.new("resident #{name} exceeds device buffer limit")
      end
      count.to_i32
    end
  end
end
{% end %}
