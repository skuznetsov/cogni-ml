require "./cross_attention_plan"
require "../nn/linear"

module ML::Sparse
  # Owned graphless outputs of the TRELLIS.2 sparse cross-attention input
  # projections. Q is stored flat as [N, C] in sparse row order; K/V are stored
  # dense and batch-major as [B, L, 2C]. Head/component access uses only index
  # arithmetic over those arrays and does not create a reshaped buffer. No padded query,
  # normalization, attention scores, softmax, or output projection is created.
  #
  # The current source query is still TensorCPU and therefore bounded to
  # TensorCPU::MAX_CHANNELS. This leaf proves the projection contract without
  # claiming that the production C=1536 sparse carrier has been admitted.
  class CrossAttentionProjectionCPU
    getter plan : CrossAttentionPlanCPU

    @query_features : Array(Float32)
    @context_kv_features : Array(Float32)

    private def initialize(
      @plan : CrossAttentionPlanCPU,
      @query_features : Array(Float32),
      @context_kv_features : Array(Float32),
    )
    end

    # Executes only the source-owned to_q(C, C) and to_kv(Cctx, 2C) affine
    # projections. Admission precedes every payload/parameter read and output
    # allocation. Mutable dense context and parameters are then borrowed and
    # validated synchronously; callers must not mutate them concurrently.
    def self.project(
      query : TensorCPU,
      context : ML::Tensor,
      num_heads : Int32,
      to_q : ML::NN::Linear,
      to_kv : ML::NN::Linear,
      max_score_bytes : Int64 = CrossAttentionPlanCPU::MAX_SCORE_BYTES,
      max_projection_bytes : Int64 = CrossAttentionPlanCPU::MAX_PROJECTION_BYTES,
      max_work_elements : Int64 = CrossAttentionPlanCPU::MAX_WORK_ELEMENTS,
    ) : CrossAttentionProjectionCPU
      plan = CrossAttentionPlanCPU.preflight(
        query,
        context,
        num_heads,
        max_score_bytes,
        max_projection_bytes,
        max_work_elements
      )

      point_count = plan.point_count
      channels = plan.query_channels
      context_channels = plan.context_channels
      context_length = plan.context_length
      batch_size = plan.batch_size

      # TensorCPU owns immutable finite features, but rechecking at this
      # execution boundary keeps the payload contract explicit and local.
      point_count.times do |row|
        channels.times do |channel|
          unless query.feature(row, channel).finite?
            raise SparseTensorError.new(
              "sparse cross-attention query[#{row * channels + channel}] must be finite"
            )
          end
        end
      end

      context_values = context.cpu_read
      unless context_values.borrowed? && context_values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse cross-attention context read must not materialize payload storage"
        )
      end
      context_values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse cross-attention context[#{index}] must be finite"
          )
        end
      end

      q_weight, q_bias = validate_linear_values(
        to_q,
        "to_q",
        channels,
        channels
      )
      kv_channels = channels * 2
      kv_weight, kv_bias = validate_linear_values(
        to_kv,
        "to_kv",
        context_channels,
        kv_channels
      )

      # The plan has already checked the combined projection byte ceiling.
      # Both capacities are reserved before arithmetic and neither input nor
      # parameter payload is copied.
      query_features = Array(Float32).new(
        plan.query_projection_elements.to_i
      )
      context_kv_features = Array(Float32).new(
        plan.context_kv_elements.to_i
      )

      point_count.times do |row|
        channels.times do |output_channel|
          weight_offset = output_channel * channels
          sum = 0.0_f32
          channels.times do |input_channel|
            sum += query.feature(row, input_channel) *
                   q_weight[weight_offset + input_channel]
          end
          value = sum + q_bias[output_channel]
          unless value.finite?
            raise SparseTensorError.new(
              "sparse cross-attention Q output[#{query_features.size}] must be finite"
            )
          end
          query_features << value
        end
      end

      context_tokens = batch_size * context_length
      context_tokens.times do |token|
        input_offset = token * context_channels
        kv_channels.times do |output_channel|
          weight_offset = output_channel * context_channels
          sum = 0.0_f32
          context_channels.times do |input_channel|
            sum += context_values[input_offset + input_channel] *
                   kv_weight[weight_offset + input_channel]
          end
          value = sum + kv_bias[output_channel]
          unless value.finite?
            raise SparseTensorError.new(
              "sparse cross-attention K/V output[#{context_kv_features.size}] must be finite"
            )
          end
          context_kv_features << value
        end
      end

      new(plan, query_features, context_kv_features)
    end

    def coordinate_map : CoordinateMap3D
      @plan.coordinate_map
    end

    def point_count : Int32
      @plan.point_count
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

    def query_feature_shape : Tuple(Int32, Int32)
      {point_count, channels}
    end

    def context_kv_feature_shape : Tuple(Int32, Int32, Int32)
      {
        @plan.batch_size,
        @plan.context_length,
        channels * 2,
      }
    end

    def query_batch_slice(batch : Int32) : BatchSlice
      @plan.query_batch_slice(batch)
    end

    def query_batch_index(row : Int32) : Int32
      @plan.query_batch_index(row)
    end

    def query_feature(row : Int32, head : Int32, channel : Int32) : Float32
      unless 0 <= row < point_count
        raise IndexError.new(
          "sparse cross-attention query row #{row} is out of bounds"
        )
      end
      validate_head_channel!(head, channel)
      @query_features[row * channels + head * head_dim + channel]
    end

    def context_kv_feature(
      batch : Int32,
      token : Int32,
      component : Int32,
      head : Int32,
      channel : Int32,
    ) : Float32
      unless 0 <= batch < @plan.batch_size
        raise IndexError.new(
          "sparse cross-attention context batch #{batch} is out of bounds"
        )
      end
      unless 0 <= token < @plan.context_length
        raise IndexError.new(
          "sparse cross-attention context token #{token} is out of bounds"
        )
      end
      unless 0 <= component < 2
        raise IndexError.new(
          "sparse cross-attention K/V component #{component} is out of bounds"
        )
      end
      validate_head_channel!(head, channel)

      index = ((((batch * @plan.context_length + token) * 2 + component) *
                num_heads + head) * head_dim) + channel
      @context_kv_features[index]
    end

    def query_features_copy : Array(Float32)
      @query_features.dup
    end

    def context_kv_features_copy : Array(Float32)
      @context_kv_features.dup
    end

    private def validate_head_channel!(head : Int32, channel : Int32) : Nil
      unless 0 <= head < num_heads
        raise IndexError.new(
          "sparse cross-attention head #{head} is out of bounds"
        )
      end
      unless 0 <= channel < head_dim
        raise IndexError.new(
          "sparse cross-attention head channel #{channel} is out of bounds"
        )
      end
    end

    private def self.validate_linear_values(
      linear : ML::NN::Linear,
      label : String,
      input_channels : Int32,
      output_channels : Int32,
    ) : Tuple(ML::Tensor::CPUReadView, ML::Tensor::CPUReadView)
      unless linear.in_features == input_channels &&
             linear.out_features == output_channels
        raise SparseTensorError.new(
          "sparse cross-attention requires #{label} Linear(#{input_channels}, #{output_channels})"
        )
      end
      bias = linear.bias
      unless bias
        raise SparseTensorError.new(
          "sparse cross-attention requires a biased #{label}"
        )
      end
      if linear.weight.requires_grad? || bias.requires_grad?
        raise SparseTensorError.new(
          "sparse cross-attention requires frozen graphless #{label} parameters"
        )
      end

      weight_tensor = linear.weight.data
      validate_parameter_tensor!(
        weight_tensor,
        "#{label} weight",
        ML::Shape.new(output_channels, input_channels)
      )
      weight_values = weight_tensor.cpu_read
      unless weight_values.borrowed? && weight_values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse cross-attention #{label} weight read must not materialize parameter storage"
        )
      end
      weight_values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse cross-attention #{label} weight[#{index}] must be finite"
          )
        end
      end

      bias_tensor = bias.data
      validate_parameter_tensor!(
        bias_tensor,
        "#{label} bias",
        ML::Shape.new(output_channels)
      )
      bias_values = bias_tensor.cpu_read
      unless bias_values.borrowed? && bias_values.materialized_bytes == 0_i64
        raise SparseTensorError.new(
          "sparse cross-attention #{label} bias read must not materialize parameter storage"
        )
      end
      bias_values.each_with_index do |value, index|
        unless value.finite?
          raise SparseTensorError.new(
            "sparse cross-attention #{label} bias[#{index}] must be finite"
          )
        end
      end
      {weight_values, bias_values}
    end

    private def self.validate_parameter_tensor!(
      tensor : ML::Tensor,
      label : String,
      expected_shape : ML::Shape,
    ) : Nil
      unless tensor.on_cpu?
        raise SparseTensorError.new(
          "sparse cross-attention #{label} must be on CPU"
        )
      end
      unless tensor.dtype.f32?
        raise SparseTensorError.new(
          "sparse cross-attention #{label} must use F32"
        )
      end
      unless tensor.contiguous?
        raise SparseTensorError.new(
          "sparse cross-attention #{label} must be contiguous"
        )
      end
      unless tensor.shape == expected_shape
        raise SparseTensorError.new(
          "sparse cross-attention #{label} shape must be #{expected_shape}"
        )
      end
    end
  end
end
