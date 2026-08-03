require "./tensor"

module ML::Sparse
  # Payload-allocation-free admission certificate for sparse queries attending
  # to one uniform dense context sequence per batch. It preserves the physical
  # flat query rows and never constructs a padded [B, S, C] query carrier.
  #
  # This plan does not project Q/K/V and is not a sparse attention executor.
  # Projection storage and arithmetic remain a separate, independently bounded
  # boundary because the current TensorCPU channel ceiling is below the pinned
  # production TRELLIS.2 dimensions. Mutable context payloads and parameters
  # are deliberately not read; the future synchronous executor must validate
  # them after preflight and before arithmetic.
  class CrossAttentionPlanCPU
    # Local CPU-reference policy, not an upstream TRELLIS.2 backend limit.
    MAX_SCORE_BYTES      = 1_i64 * 1024_i64 * 1024_i64
    MAX_PROJECTION_BYTES = 64_i64 * 1024_i64 * 1024_i64
    MAX_WORK_ELEMENTS    = 128_i64 * 1024_i64 * 1024_i64

    getter coordinate_map : CoordinateMap3D
    getter batch_size : Int32
    getter point_count : Int32
    getter query_channels : Int32
    getter context_channels : Int32
    getter context_length : Int32
    getter num_heads : Int32
    getter head_dim : Int32
    getter max_query_length : Int32
    getter score_elements : Int64
    getter score_bytes : Int64
    getter query_projection_elements : Int64
    getter context_kv_elements : Int64
    getter projection_elements : Int64
    getter projection_bytes : Int64
    getter output_elements : Int64
    getter output_bytes : Int64
    getter query_projection_mac_elements : Int64
    getter context_kv_projection_mac_elements : Int64
    getter attention_mac_elements : Int64
    getter output_projection_mac_elements : Int64
    getter work_elements : Int64
    getter max_score_bytes : Int64
    getter max_projection_bytes : Int64
    getter max_work_elements : Int64

    private def initialize(
      @coordinate_map : CoordinateMap3D,
      @batch_size : Int32,
      @point_count : Int32,
      @query_channels : Int32,
      @context_channels : Int32,
      @context_length : Int32,
      @num_heads : Int32,
      @head_dim : Int32,
      @max_query_length : Int32,
      @score_elements : Int64,
      @score_bytes : Int64,
      @query_projection_elements : Int64,
      @context_kv_elements : Int64,
      @projection_elements : Int64,
      @projection_bytes : Int64,
      @output_elements : Int64,
      @output_bytes : Int64,
      @query_projection_mac_elements : Int64,
      @context_kv_projection_mac_elements : Int64,
      @attention_mac_elements : Int64,
      @output_projection_mac_elements : Int64,
      @work_elements : Int64,
      @max_score_bytes : Int64,
      @max_projection_bytes : Int64,
      @max_work_elements : Int64,
    )
    end

    def self.preflight(
      query : TensorCPU,
      context : ML::Tensor,
      num_heads : Int32,
      max_score_bytes : Int64 = MAX_SCORE_BYTES,
      max_projection_bytes : Int64 = MAX_PROJECTION_BYTES,
      max_work_elements : Int64 = MAX_WORK_ELEMENTS,
    ) : CrossAttentionPlanCPU
      unless query.class == TensorCPU
        raise SparseTensorError.new(
          "sparse cross-attention preflight requires a base TensorCPU query"
        )
      end
      unless context.class == ML::Tensor
        raise SparseTensorError.new(
          "sparse cross-attention preflight requires a base Tensor context"
        )
      end
      validate_budgets!(
        max_score_bytes,
        max_projection_bytes,
        max_work_elements
      )

      coordinate_map = query.coordinate_map
      unless coordinate_map.class == CoordinateMap3D
        raise SparseTensorError.new(
          "sparse cross-attention preflight requires a base CoordinateMap3D"
        )
      end
      batch_size, coordinate_point_count = CoordinateMap3D.kernel_layout(
        coordinate_map
      )
      point_count = query.point_count
      unless point_count == coordinate_point_count
        raise SparseTensorError.new(
          "sparse cross-attention query point count #{point_count} does not match coordinate point count #{coordinate_point_count}"
        )
      end

      unless num_heads > 0
        raise SparseTensorError.new(
          "sparse cross-attention head count must be positive"
        )
      end
      query_channels = query.channels
      unless query_channels % num_heads == 0
        raise SparseTensorError.new(
          "sparse cross-attention query channels #{query_channels} must be divisible by heads #{num_heads}"
        )
      end
      head_dim = query_channels // num_heads

      unless context.on_cpu?
        raise SparseTensorError.new(
          "sparse cross-attention context must be on CPU"
        )
      end
      unless context.dtype.f32?
        raise SparseTensorError.new(
          "sparse cross-attention context must use F32"
        )
      end
      unless context.contiguous?
        raise SparseTensorError.new(
          "sparse cross-attention context must be contiguous"
        )
      end
      context_shape = context.shape
      unless context_shape.ndim == 3
        raise SparseTensorError.new(
          "sparse cross-attention context must have rank 3 [B, L, C]"
        )
      end
      context_batch_size = context_shape[0]
      context_length = context_shape[1]
      context_channels = context_shape[2]
      unless context_batch_size == batch_size
        raise SparseTensorError.new(
          "sparse cross-attention context batch size #{context_batch_size} does not match sparse batch size #{batch_size}"
        )
      end
      unless context_length > 0
        raise SparseTensorError.new(
          "sparse cross-attention context length must be positive"
        )
      end
      unless context_channels > 0
        raise SparseTensorError.new(
          "sparse cross-attention context channels must be positive"
        )
      end

      sum_query_lengths = 0_i64
      sum_query_context_pairs = 0_i64
      max_query_length = 0_i32
      batch_size.times do |batch|
        batch_slice = CoordinateMap3D.kernel_batch_slice(coordinate_map, batch)
        query_length = batch_slice.size
        if query_length < 0
          raise SparseTensorError.new(
            "sparse cross-attention batch #{batch} has a negative query length"
          )
        end
        sum_query_lengths = checked_add(
          sum_query_lengths,
          query_length.to_i64,
          "query lengths"
        )
        query_context_pairs = checked_multiply(
          query_length.to_i64,
          context_length.to_i64,
          "per-batch query/context pairs"
        )
        sum_query_context_pairs = checked_add(
          sum_query_context_pairs,
          query_context_pairs,
          "query/context pairs"
        )
        max_query_length = query_length if query_length > max_query_length

        row = batch_slice.start
        while row < batch_slice.stop
          unless CoordinateMap3D.kernel_batch_index(coordinate_map, row) == batch
            raise SparseTensorError.new(
              "sparse cross-attention row #{row} is outside its batch slice"
            )
          end
          row += 1
        end
      end
      unless sum_query_lengths == point_count
        raise SparseTensorError.new(
          "sparse cross-attention query lengths sum to #{sum_query_lengths}, expected #{point_count}"
        )
      end

      score_elements = checked_multiply(
        sum_query_context_pairs,
        num_heads.to_i64,
        "headed score elements"
      )
      score_bytes = checked_multiply(score_elements, 4_i64, "score bytes")
      if score_bytes > max_score_bytes
        raise SparseTensorBudgetError.new(
          "bounded sparse cross-attention score budget would require #{score_bytes} bytes, limit is #{max_score_bytes}"
        )
      end

      query_projection_elements = checked_multiply(
        point_count.to_i64,
        query_channels.to_i64,
        "query projection elements"
      )
      context_kv_elements = checked_multiply(
        checked_multiply(
          checked_multiply(
            batch_size.to_i64,
            context_length.to_i64,
            "context token elements"
          ),
          query_channels.to_i64,
          "context KV channel elements"
        ),
        2_i64,
        "context KV elements"
      )
      projection_elements = checked_add(
        query_projection_elements,
        context_kv_elements,
        "projection elements"
      )
      projection_bytes = checked_multiply(
        projection_elements,
        4_i64,
        "projection bytes"
      )
      if projection_bytes > max_projection_bytes
        raise SparseTensorBudgetError.new(
          "bounded sparse cross-attention projections would require #{projection_bytes} bytes, limit is #{max_projection_bytes}"
        )
      end

      output_elements = query_projection_elements
      output_bytes = checked_multiply(output_elements, 4_i64, "output bytes")
      if output_bytes > query.max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse cross-attention output would require #{output_bytes} bytes, limit is #{query.max_feature_bytes}"
        )
      end

      attention_mac_elements = checked_multiply(
        checked_multiply(
          score_elements,
          head_dim.to_i64,
          "attention MAC elements"
        ),
        2_i64,
        "attention MAC elements"
      )
      query_projection_mac_elements = checked_multiply(
        query_projection_elements,
        query_channels.to_i64,
        "query projection MAC elements"
      )
      context_kv_projection_mac_elements = checked_multiply(
        context_kv_elements,
        context_channels.to_i64,
        "context KV projection MAC elements"
      )
      output_projection_mac_elements = query_projection_mac_elements
      work_elements = checked_add(
        checked_add(
          query_projection_mac_elements,
          context_kv_projection_mac_elements,
          "projection work elements"
        ),
        checked_add(
          attention_mac_elements,
          output_projection_mac_elements,
          "attention/output work elements"
        ),
        "total work elements"
      )
      if work_elements > max_work_elements
        raise SparseTensorBudgetError.new(
          "bounded sparse cross-attention work would require #{work_elements} MAC elements, limit is #{max_work_elements}"
        )
      end

      new(
        coordinate_map,
        batch_size,
        point_count,
        query_channels,
        context_channels,
        context_length,
        num_heads,
        head_dim,
        max_query_length,
        score_elements,
        score_bytes,
        query_projection_elements,
        context_kv_elements,
        projection_elements,
        projection_bytes,
        output_elements,
        output_bytes,
        query_projection_mac_elements,
        context_kv_projection_mac_elements,
        attention_mac_elements,
        output_projection_mac_elements,
        work_elements,
        max_score_bytes,
        max_projection_bytes,
        max_work_elements
      )
    end

    def query_batch_slice(batch : Int32) : BatchSlice
      CoordinateMap3D.kernel_batch_slice(@coordinate_map, batch)
    end

    def query_batch_index(row : Int32) : Int32
      CoordinateMap3D.kernel_batch_index(@coordinate_map, row)
    end

    private def self.validate_budgets!(
      max_score_bytes : Int64,
      max_projection_bytes : Int64,
      max_work_elements : Int64,
    ) : Nil
      unless 1_i64 <= max_score_bytes <= MAX_SCORE_BYTES
        raise SparseTensorBudgetError.new(
          "sparse cross-attention score byte budget must be in 1..#{MAX_SCORE_BYTES}"
        )
      end
      unless 1_i64 <= max_projection_bytes <= MAX_PROJECTION_BYTES
        raise SparseTensorBudgetError.new(
          "sparse cross-attention projection byte budget must be in 1..#{MAX_PROJECTION_BYTES}"
        )
      end
      unless 1_i64 <= max_work_elements <= MAX_WORK_ELEMENTS
        raise SparseTensorBudgetError.new(
          "sparse cross-attention work budget must be in 1..#{MAX_WORK_ELEMENTS}"
        )
      end
    end

    private def self.checked_add(
      left : Int64,
      right : Int64,
      label : String,
    ) : Int64
      if left < 0_i64 || right < 0_i64 || left > Int64::MAX - right
        raise SparseTensorBudgetError.new(
          "sparse cross-attention #{label} overflow Int64"
        )
      end
      left + right
    end

    private def self.checked_multiply(
      left : Int64,
      right : Int64,
      label : String,
    ) : Int64
      if left < 0_i64 || right < 0_i64 ||
         (right > 0_i64 && left > Int64::MAX // right)
        raise SparseTensorBudgetError.new(
          "sparse cross-attention #{label} overflow Int64"
        )
      end
      left * right
    end
  end
end
