require "./cross_attention_projection"

module ML::Sparse
  # Graphless CPU/F32 result at the TRELLIS.2 cross-attention boundary after
  # per-head Q/K L2 normalization and before score arithmetic. Q remains flat
  # in sparse row order, K remains dense batch-major, and V is read zero-copy
  # from the retained projection owner. No padded query or reshape payload is
  # created.
  class CrossAttentionQKNormalizedCPU
    EPSILON              = 1e-12_f32
    MAX_NORMALIZED_BYTES = CrossAttentionPlanCPU::MAX_PROJECTION_BYTES

    getter plan : CrossAttentionPlanCPU
    getter source_projection : CrossAttentionProjectionCPU
    getter normalized_bytes : Int64

    @query_features : Array(Float32)
    @key_features : Array(Float32)

    private def initialize(
      @source_projection : CrossAttentionProjectionCPU,
      @plan : CrossAttentionPlanCPU,
      @query_features : Array(Float32),
      @key_features : Array(Float32),
      @normalized_bytes : Int64,
    )
    end

    # Applies the source SparseMultiHeadRMSNorm formula independently to every
    # Q and K row/head. This is L2 normalization with a historical RMSNorm name:
    # x / max(sqrt(sum(x*x)), 1e-12) * gamma[head, channel] * sqrt(head_dim).
    #
    # The byte cap bounds only the new owned Q+K payload. The input projection
    # remains live because this carrier deliberately borrows its immutable V;
    # this is not an aggregate RSS/native/peak-memory bound. Gamma storage is
    # borrowed only for this synchronous call and must not be mutated
    # concurrently.
    def self.normalize(
      projection : CrossAttentionProjectionCPU,
      q_gamma : ML::Tensor,
      k_gamma : ML::Tensor,
      max_normalized_bytes : Int64 = MAX_NORMALIZED_BYTES,
    ) : CrossAttentionQKNormalizedCPU
      unless projection.class == CrossAttentionProjectionCPU
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization requires a base CrossAttentionProjectionCPU"
        )
      end
      plan = projection.plan
      unless plan.class == CrossAttentionPlanCPU
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization requires a base CrossAttentionPlanCPU"
        )
      end
      validate_projection_contract!(projection, plan)
      unless 1_i64 <= max_normalized_bytes <= MAX_NORMALIZED_BYTES
        raise SparseTensorBudgetError.new(
          "sparse cross-attention Q/K normalization byte budget must be in 1..#{MAX_NORMALIZED_BYTES}"
        )
      end

      query_elements = plan.query_projection_elements
      key_elements = plan.context_kv_elements // 2_i64
      normalized_elements = query_elements + key_elements
      normalized_bytes = normalized_elements * 4_i64
      if normalized_bytes > max_normalized_bytes
        raise SparseTensorBudgetError.new(
          "sparse cross-attention Q/K normalization output would require #{normalized_bytes} bytes, limit is #{max_normalized_bytes}"
        )
      end

      q_values = gamma_values(
        "q_gamma",
        q_gamma,
        projection.num_heads,
        projection.head_dim
      )
      k_values = gamma_values(
        "k_gamma",
        k_gamma,
        projection.num_heads,
        projection.head_dim
      )
      scale = Math.sqrt(projection.head_dim.to_f32)

      # Reject every numerical failure before allocating either result array.
      normalize_query!(projection, q_values, scale, nil)
      normalize_key!(projection, k_values, scale, nil)

      query_features = Array(Float32).new(query_elements.to_i)
      key_features = Array(Float32).new(key_elements.to_i)
      normalize_query!(projection, q_values, scale, query_features)
      normalize_key!(projection, k_values, scale, key_features)

      new(
        projection,
        plan,
        query_features,
        key_features,
        normalized_bytes
      )
    end

    private def self.validate_projection_contract!(
      projection : CrossAttentionProjectionCPU,
      plan : CrossAttentionPlanCPU,
    ) : Nil
      # The exact base projection's private constructor owns its two arrays.
      # Recheck the public plan geometry here so upstream plan drift fails with
      # a typed boundary error; same-process class reopening remains trusted.
      query_elements = plan.point_count.to_i64 * plan.query_channels.to_i64
      context_kv_elements = plan.batch_size.to_i64 *
                            plan.context_length.to_i64 *
                            plan.query_channels.to_i64 * 2_i64
      unless projection.num_heads > 0 && projection.head_dim > 0 &&
             projection.num_heads.to_i64 * projection.head_dim.to_i64 == plan.query_channels &&
             plan.query_projection_elements == query_elements &&
             plan.context_kv_elements == context_kv_elements &&
             plan.context_kv_elements.even?
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization requires exact projection geometry"
        )
      end
    end

    def coordinate_map : CoordinateMap3D
      @plan.coordinate_map
    end

    def point_count : Int32
      @plan.point_count
    end

    def batch_size : Int32
      @plan.batch_size
    end

    def context_length : Int32
      @plan.context_length
    end

    def channels : Int32
      @plan.query_channels
    end

    def num_heads : Int32
      @plan.num_heads
    end

    def head_dim : Int32
      @plan.head_dim
    end

    def query_feature_shape : Tuple(Int32, Int32, Int32)
      {point_count, num_heads, head_dim}
    end

    def key_value_feature_shape : Tuple(Int32, Int32, Int32, Int32)
      {batch_size, context_length, num_heads, head_dim}
    end

    def query_batch_slice(batch : Int32) : BatchSlice
      @plan.query_batch_slice(batch)
    end

    def query_batch_index(row : Int32) : Int32
      @plan.query_batch_index(row)
    end

    def query_feature(row : Int32, head : Int32, channel : Int32) : Float32
      validate_query_index!(row, head, channel)
      @query_features[(row * num_heads + head) * head_dim + channel]
    end

    def key_feature(
      batch : Int32,
      token : Int32,
      head : Int32,
      channel : Int32,
    ) : Float32
      @key_features[context_index(batch, token, head, channel)]
    end

    # V stays owned by source_projection. Retaining that owner makes this a
    # lifetime-safe zero-copy borrow rather than an exposed transient slice.
    def value_feature(
      batch : Int32,
      token : Int32,
      head : Int32,
      channel : Int32,
    ) : Float32
      @source_projection.context_kv_feature(
        batch,
        token,
        1_i32,
        head,
        channel
      )
    end

    def query_features_copy : Array(Float32)
      @query_features.dup
    end

    def key_features_copy : Array(Float32)
      @key_features.dup
    end

    # Explicit diagnostic copy; the carrier itself never allocates or stores a
    # second V payload.
    def value_features_copy : Array(Float32)
      values = Array(Float32).new(
        batch_size * context_length * num_heads * head_dim
      )
      batch_size.times do |batch|
        context_length.times do |token|
          num_heads.times do |head|
            head_dim.times do |channel|
              values << value_feature(batch, token, head, channel)
            end
          end
        end
      end
      values
    end

    private def validate_query_index!(
      row : Int32,
      head : Int32,
      channel : Int32,
    ) : Nil
      unless 0 <= row < point_count
        raise IndexError.new(
          "sparse cross-attention normalized query row #{row} is out of bounds"
        )
      end
      validate_head_channel!(head, channel)
    end

    private def context_index(
      batch : Int32,
      token : Int32,
      head : Int32,
      channel : Int32,
    ) : Int32
      unless 0 <= batch < batch_size
        raise IndexError.new(
          "sparse cross-attention normalized context batch #{batch} is out of bounds"
        )
      end
      unless 0 <= token < context_length
        raise IndexError.new(
          "sparse cross-attention normalized context token #{token} is out of bounds"
        )
      end
      validate_head_channel!(head, channel)
      (((batch * context_length + token) * num_heads + head) * head_dim) + channel
    end

    private def validate_head_channel!(head : Int32, channel : Int32) : Nil
      unless 0 <= head < num_heads
        raise IndexError.new(
          "sparse cross-attention normalized head #{head} is out of bounds"
        )
      end
      unless 0 <= channel < head_dim
        raise IndexError.new(
          "sparse cross-attention normalized head channel #{channel} is out of bounds"
        )
      end
    end

    private def self.gamma_values(
      name : String,
      parameter : ML::Tensor,
      num_heads : Int32,
      head_dim : Int32,
    ) : ML::Tensor::CPUReadView
      unless parameter.class == ML::Tensor
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization #{name} requires a base Tensor"
        )
      end
      unless parameter.on_cpu?
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization #{name} must be on CPU"
        )
      end
      unless parameter.dtype.f32?
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization #{name} must use F32"
        )
      end
      unless parameter.contiguous?
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization #{name} must be contiguous"
        )
      end
      unless parameter.ndim == 2 &&
             parameter.shape[0] == num_heads &&
             parameter.shape[1] == head_dim
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization #{name} shape must be [#{num_heads}, #{head_dim}]"
        )
      end

      values = parameter.cpu_read
      unless values.borrowed? && values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse cross-attention Q/K normalization #{name} read must not materialize storage"
        )
      end
      values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse cross-attention Q/K normalization #{name}[#{index}] must be finite"
          )
        end
      end
      values
    end

    private def self.normalize_query!(
      projection : CrossAttentionProjectionCPU,
      gamma : ML::Tensor::CPUReadView,
      scale : Float32,
      output : Array(Float32)?,
    ) : Nil
      projection.point_count.times do |row|
        projection.num_heads.times do |head|
          sum_sq = 0.0_f32
          projection.head_dim.times do |channel|
            value = projection.query_feature(row, head, channel)
            sum_sq += value * value
          end
          norm = Math.sqrt(sum_sq)
          norm = EPSILON if norm < EPSILON
          gamma_offset = head * projection.head_dim
          projection.head_dim.times do |channel|
            value = projection.query_feature(row, head, channel) /
                    norm * gamma[gamma_offset + channel] * scale
            unless value.finite?
              raise SparseTensorError.new(
                "sparse cross-attention normalized Q output at row #{row}, head #{head}, channel #{channel} must be finite"
              )
            end
            output.try { |destination| destination << value }
          end
        end
      end
    end

    private def self.normalize_key!(
      projection : CrossAttentionProjectionCPU,
      gamma : ML::Tensor::CPUReadView,
      scale : Float32,
      output : Array(Float32)?,
    ) : Nil
      projection.plan.batch_size.times do |batch|
        projection.plan.context_length.times do |token|
          projection.num_heads.times do |head|
            sum_sq = 0.0_f32
            projection.head_dim.times do |channel|
              value = projection.context_kv_feature(
                batch,
                token,
                0_i32,
                head,
                channel
              )
              sum_sq += value * value
            end
            norm = Math.sqrt(sum_sq)
            norm = EPSILON if norm < EPSILON
            gamma_offset = head * projection.head_dim
            projection.head_dim.times do |channel|
              value = projection.context_kv_feature(
                batch,
                token,
                0_i32,
                head,
                channel
              ) / norm * gamma[gamma_offset + channel] * scale
              unless value.finite?
                raise SparseTensorError.new(
                  "sparse cross-attention normalized K output at batch #{batch}, token #{token}, head #{head}, channel #{channel} must be finite"
                )
              end
              output.try { |destination| destination << value }
            end
          end
        end
      end
    end
  end
end
