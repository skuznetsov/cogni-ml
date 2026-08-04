require "./self_attention_qkv"

module ML::Sparse
  # Payload-allocation-free admission certificate for the bounded CPU/F32
  # reference full-attention leaf. Full attention is dense within each batch
  # slice; it is sparse only in excluding pairs from different batches.
  class FullSelfAttentionPlanCPU
    # Local oracle policy, not an upstream TRELLIS.2 backend limit. This bounds
    # a future single F32 score plane to 1 MiB and cannot be widened by callers.
    MAX_SCORE_BYTES = 1_i64 * 1024_i64 * 1024_i64

    getter batch_size : Int32
    getter point_count : Int32
    getter num_heads : Int32
    getter head_dim : Int32
    getter max_batch_length : Int32
    getter score_elements : Int64
    getter score_bytes : Int64
    getter output_elements : Int64
    getter output_bytes : Int64
    getter attention_mac_elements : Int64
    getter max_score_bytes : Int64

    private def initialize(
      @batch_size : Int32,
      @point_count : Int32,
      @num_heads : Int32,
      @head_dim : Int32,
      @max_batch_length : Int32,
      @score_elements : Int64,
      @score_bytes : Int64,
      @output_elements : Int64,
      @output_bytes : Int64,
      @attention_mac_elements : Int64,
      @max_score_bytes : Int64,
    )
    end

    def self.build(
      qkv : SelfAttentionQKVCPU,
      max_score_bytes : Int64 = MAX_SCORE_BYTES,
    ) : FullSelfAttentionPlanCPU
      unless qkv.class == SelfAttentionQKVCPU
        raise SparseTensorError.new(
          "sparse full-attention plan requires a base SelfAttentionQKVCPU"
        )
      end
      unless 1_i64 <= max_score_bytes <= MAX_SCORE_BYTES
        raise SparseTensorBudgetError.new(
          "sparse full-attention score byte budget must be in 1..#{MAX_SCORE_BYTES}"
        )
      end

      coordinate_map = qkv.coordinate_map
      unless coordinate_map.class == CoordinateMap3D
        raise SparseTensorError.new(
          "sparse full-attention plan requires a base CoordinateMap3D"
        )
      end
      batch_size, point_count = CoordinateMap3D.kernel_layout(coordinate_map)
      unless point_count == qkv.point_count
        raise SparseTensorError.new(
          "sparse full-attention QKV point count #{qkv.point_count} does not match coordinate point count #{point_count}"
        )
      end

      num_heads = qkv.num_heads
      head_dim = qkv.head_dim
      channels = qkv.channels
      unless num_heads > 0 && head_dim > 0 && channels > 0
        raise SparseTensorError.new(
          "sparse full-attention dimensions must be positive"
        )
      end
      unless checked_multiply(num_heads.to_i64, head_dim.to_i64, "channel shape") == channels
        raise SparseTensorError.new(
          "sparse full-attention heads #{num_heads} and head dimension #{head_dim} do not reconstruct channels #{channels}"
        )
      end

      build_from_layout(
        coordinate_map,
        point_count,
        num_heads,
        head_dim,
        channels,
        qkv.max_feature_bytes,
        max_score_bytes
      )
    end

    # Admits the score and output budgets from the immutable sparse input shape
    # before QKV, normalization, or RoPE allocate their output payloads.
    def self.preflight(
      input : TensorCPU,
      num_heads : Int32,
      max_score_bytes : Int64 = MAX_SCORE_BYTES,
    ) : FullSelfAttentionPlanCPU
      unless input.class == TensorCPU
        raise SparseTensorError.new(
          "sparse full-attention preflight requires a base TensorCPU"
        )
      end
      unless num_heads > 0
        raise SparseTensorError.new(
          "sparse full-attention preflight head count must be positive"
        )
      end
      TensorCPU.standard_carrier_channel_limit(
        input,
        "sparse full-attention preflight"
      )
      channels = input.channels
      unless channels % num_heads == 0
        raise SparseTensorError.new(
          "sparse full-attention preflight channels #{channels} must be divisible by heads #{num_heads}"
        )
      end

      build_from_layout(
        input.coordinate_map,
        input.point_count,
        num_heads,
        channels // num_heads,
        channels,
        input.max_feature_bytes,
        max_score_bytes
      )
    end

    private def self.build_from_layout(
      coordinate_map : CoordinateMap3D,
      point_count : Int32,
      num_heads : Int32,
      head_dim : Int32,
      channels : Int32,
      max_feature_bytes : Int64,
      max_score_bytes : Int64,
    ) : FullSelfAttentionPlanCPU
      unless 1_i64 <= max_score_bytes <= MAX_SCORE_BYTES
        raise SparseTensorBudgetError.new(
          "sparse full-attention score byte budget must be in 1..#{MAX_SCORE_BYTES}"
        )
      end
      unless coordinate_map.class == CoordinateMap3D
        raise SparseTensorError.new(
          "sparse full-attention plan requires a base CoordinateMap3D"
        )
      end
      batch_size, coordinate_point_count = CoordinateMap3D.kernel_layout(coordinate_map)
      unless coordinate_point_count == point_count
        raise SparseTensorError.new(
          "sparse full-attention point count #{point_count} does not match coordinate point count #{coordinate_point_count}"
        )
      end
      unless num_heads > 0 && head_dim > 0 && channels > 0
        raise SparseTensorError.new(
          "sparse full-attention dimensions must be positive"
        )
      end
      unless checked_multiply(num_heads.to_i64, head_dim.to_i64, "channel shape") == channels
        raise SparseTensorError.new(
          "sparse full-attention heads #{num_heads} and head dimension #{head_dim} do not reconstruct channels #{channels}"
        )
      end

      sum_lengths = 0_i64
      sum_squared_lengths = 0_i64
      max_batch_length = 0_i32
      batch_size.times do |batch|
        batch_slice = CoordinateMap3D.kernel_batch_slice(coordinate_map, batch)
        length = batch_slice.size
        if length < 0
          raise SparseTensorError.new(
            "sparse full-attention batch #{batch} has a negative sequence length"
          )
        end
        sum_lengths = checked_add(sum_lengths, length.to_i64, "sequence lengths")
        squared_length = checked_multiply(
          length.to_i64,
          length.to_i64,
          "per-batch score elements"
        )
        sum_squared_lengths = checked_add(
          sum_squared_lengths,
          squared_length,
          "score elements"
        )
        max_batch_length = length if length > max_batch_length
      end
      unless sum_lengths == point_count
        raise SparseTensorError.new(
          "sparse full-attention batch lengths sum to #{sum_lengths}, expected #{point_count}"
        )
      end

      score_elements = checked_multiply(
        sum_squared_lengths,
        num_heads.to_i64,
        "headed score elements"
      )
      score_bytes = checked_multiply(score_elements, 4_i64, "score bytes")
      if score_bytes > max_score_bytes
        raise SparseTensorBudgetError.new(
          "bounded sparse full-attention quadratic score budget would require #{score_bytes} bytes, limit is #{max_score_bytes}"
        )
      end

      output_elements = checked_multiply(
        point_count.to_i64,
        channels.to_i64,
        "output elements"
      )
      output_bytes = checked_multiply(output_elements, 4_i64, "output bytes")
      if output_bytes > max_feature_bytes
        raise SparseTensorBudgetError.new(
          "sparse full-attention output would require #{output_bytes} bytes, limit is #{max_feature_bytes}"
        )
      end
      attention_mac_elements = checked_multiply(
        checked_multiply(score_elements, head_dim.to_i64, "attention MAC elements"),
        2_i64,
        "attention MAC elements"
      )

      new(
        batch_size,
        point_count,
        num_heads,
        head_dim,
        max_batch_length,
        score_elements,
        score_bytes,
        output_elements,
        output_bytes,
        attention_mac_elements,
        max_score_bytes
      )
    end

    private def self.checked_add(left : Int64, right : Int64, label : String) : Int64
      if left < 0_i64 || right < 0_i64 || left > Int64::MAX - right
        raise SparseTensorBudgetError.new(
          "sparse full-attention #{label} overflow Int64"
        )
      end
      left + right
    end

    private def self.checked_multiply(
      left : Int64,
      right : Int64,
      label : String,
    ) : Int64
      if left < 0_i64 || right < 0_i64 || (right > 0_i64 && left > Int64::MAX // right)
        raise SparseTensorBudgetError.new(
          "sparse full-attention #{label} overflow Int64"
        )
      end
      left * right
    end
  end
end
