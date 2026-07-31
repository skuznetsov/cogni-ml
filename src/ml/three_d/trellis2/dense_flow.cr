# CPU-only F32 reference composition for one tiny TRELLIS.2 dense-flow block.
# This joins layout, conditioning, rotary phases, one transformer block, and
# output layout under one observable oracle boundary. It is not a model runtime.

require "./conditioning"

module ML::ThreeD::Trellis2
  module DenseFlowCPU
    extend self

    MAX_REFERENCE_PARAMETER_BYTES    = 64_i64 * 1024_i64 * 1024_i64
    MAX_REFERENCE_PARAMETER_ELEMENTS = MAX_REFERENCE_PARAMETER_BYTES // 4_i64
    MAX_REFERENCE_TENSOR_BYTES       = 64_i64 * 1024_i64 * 1024_i64

    def bounded_product!(factors : Array(Int64), limit : Int64, label : String) : Int64
      product = 1_i64
      factors.each do |factor|
        raise ArgumentError.new("#{label} dimensions must be positive") unless factor > 0
        if product > limit // factor
          raise ArgumentError.new("#{label} exceeds #{limit} elements")
        end
        product *= factor
      end
      product
    end

    def add_parameter_term!(total : Int64, factors : Array(Int64)) : Int64
      remaining = MAX_REFERENCE_PARAMETER_ELEMENTS - total
      if remaining <= 0
        raise ArgumentError.new(
          "dense-flow parameter budget exceeds #{MAX_REFERENCE_PARAMETER_ELEMENTS} F32 elements"
        )
      end
      term = bounded_product!(factors, remaining, "dense-flow parameter budget")
      total + term
    end

    def parameter_elements(
      in_channels : Int32,
      model_channels : Int32,
      context_channels : Int32,
      out_channels : Int32,
      frequency_dim : Int32,
      mlp_ratio : Float32,
    ) : Int64
      values = {
        "in_channels"      => in_channels,
        "model_channels"   => model_channels,
        "context_channels" => context_channels,
        "out_channels"     => out_channels,
        "frequency_dim"    => frequency_dim,
      }
      values.each do |name, value|
        raise ArgumentError.new("#{name} must be positive") unless value > 0
      end
      unless mlp_ratio.finite? && mlp_ratio > 0.0_f32
        raise ArgumentError.new("mlp_ratio must be finite and positive")
      end

      c = model_channels.to_i64
      i = in_channels.to_i64
      k = context_channels.to_i64
      o = out_channels.to_i64
      f = frequency_dim.to_i64
      hidden_value = model_channels.to_f64 * mlp_ratio.to_f64
      unless 1.0 <= hidden_value <= Int32::MAX.to_f64
        raise ArgumentError.new("mlp hidden dimension must be positive and representable")
      end
      m = hidden_value.to_i64

      total = 0_i64
      # Input projection.
      total = add_parameter_term!(total, [c, i])
      total = add_parameter_term!(total, [c])
      # Timestep MLP and top shared modulation.
      total = add_parameter_term!(total, [c, f])
      total = add_parameter_term!(total, [c])
      total = add_parameter_term!(total, [c, c])
      total = add_parameter_term!(total, [c])
      total = add_parameter_term!(total, [6_i64, c, c])
      total = add_parameter_term!(total, [6_i64, c])
      # One shared-modulated cross block, matching dense_block.cr exactly.
      total = add_parameter_term!(total, [6_i64, c])
      total = add_parameter_term!(total, [2_i64, c])
      total = add_parameter_term!(total, [3_i64, c, c])
      total = add_parameter_term!(total, [3_i64, c])
      total = add_parameter_term!(total, [c, c])
      total = add_parameter_term!(total, [c])
      total = add_parameter_term!(total, [2_i64, c])
      total = add_parameter_term!(total, [c, c])
      total = add_parameter_term!(total, [c])
      total = add_parameter_term!(total, [2_i64, c, k])
      total = add_parameter_term!(total, [2_i64, c])
      total = add_parameter_term!(total, [2_i64, c])
      total = add_parameter_term!(total, [c, c])
      total = add_parameter_term!(total, [c])
      total = add_parameter_term!(total, [m, c])
      total = add_parameter_term!(total, [m])
      total = add_parameter_term!(total, [c, m])
      total = add_parameter_term!(total, [c])
      # Output projection.
      total = add_parameter_term!(total, [o, c])
      add_parameter_term!(total, [o])
    end
  end

  class DenseFlowStageCPU
    getter resolution : Int32
    getter in_channels : Int32
    getter model_channels : Int32
    getter context_channels : Int32
    getter out_channels : Int32
    getter num_heads : Int32
    getter mlp_ratio : Float32
    getter frequency_dim : Int32
    getter block_eps : Float32
    getter final_eps : Float32
    getter voxel_count : Int32
    getter max_logical_tensor_bytes : Int64
    getter parameter_elements : Int64
    getter input_layer : ML::NN::Linear
    getter conditioning : TimestepConditioningCPU
    getter block : SharedModulatedTransformerCrossBlock
    getter final_norm : NonAffineLayerNorm
    getter out_layer : ML::NN::Linear

    def initialize(
      @resolution : Int32,
      @in_channels : Int32,
      @model_channels : Int32,
      @context_channels : Int32,
      @out_channels : Int32,
      @num_heads : Int32,
      @mlp_ratio : Float32 = 4.0_f32,
      @frequency_dim : Int32 = 256,
      @block_eps : Float32 = 1e-6_f32,
      @final_eps : Float32 = 1e-5_f32,
      @max_logical_tensor_bytes : Int64 = DenseFlowCPU::MAX_REFERENCE_TENSOR_BYTES,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("DenseFlowStageCPU is CPU-only") if device.gpu?
      raise ArgumentError.new("resolution must be positive") unless @resolution > 0
      raise ArgumentError.new("num_heads must be positive and divide model_channels") unless @num_heads > 0 && @model_channels > 0 && @model_channels % @num_heads == 0
      head_dim = @model_channels // @num_heads
      raise ArgumentError.new("head dimension must be positive and even for RoPE") unless head_dim > 0 && head_dim.even?
      if head_dim // 2 // 3 <= 0
        raise ArgumentError.new("head dimension must provide at least one RoPE frequency per spatial axis")
      end
      raise ArgumentError.new("frequency_dim must be at least 2") unless @frequency_dim >= 2
      unless @block_eps.finite? && @block_eps > 0.0_f32
        raise ArgumentError.new("block_eps must be finite and positive")
      end
      unless @final_eps.finite? && @final_eps > 0.0_f32
        raise ArgumentError.new("final_eps must be finite and positive")
      end
      unless 0_i64 < @max_logical_tensor_bytes <= DenseFlowCPU::MAX_REFERENCE_TENSOR_BYTES
        raise ArgumentError.new(
          "max_logical_tensor_bytes must be positive and no greater than " \
          "#{DenseFlowCPU::MAX_REFERENCE_TENSOR_BYTES}"
        )
      end

      voxel_count = DenseFlowCPU.bounded_product!(
        [@resolution.to_i64, @resolution.to_i64, @resolution.to_i64],
        Int32::MAX.to_i64,
        "voxel count"
      )
      @voxel_count = voxel_count.to_i32
      @parameter_elements = DenseFlowCPU.parameter_elements(
        @in_channels,
        @model_channels,
        @context_channels,
        @out_channels,
        @frequency_dim,
        @mlp_ratio
      )

      @input_layer = ML::NN::Linear.new(@in_channels, @model_channels, device: Tensor::Device::CPU)
      @conditioning = TimestepConditioningCPU.new(
        channels: @model_channels,
        rotary_head_dim: head_dim,
        frequency_dim: @frequency_dim,
        spatial_dim: 3,
        max_phase_output_bytes: @max_logical_tensor_bytes,
        device: Tensor::Device::CPU
      )
      @block = SharedModulatedTransformerCrossBlock.new(
        channels: @model_channels,
        context_channels: @context_channels,
        num_heads: @num_heads,
        mlp_ratio: @mlp_ratio,
        eps: @block_eps,
        device: Tensor::Device::CPU
      )
      @final_norm = NonAffineLayerNorm.new(@model_channels, @final_eps, device: Tensor::Device::CPU)
      @out_layer = ML::NN::Linear.new(@model_channels, @out_channels, device: Tensor::Device::CPU)
      ConditioningCPU.freeze_linear(@input_layer)
      ConditioningCPU.freeze_linear(@out_layer)

      constructed = @input_layer.parameters.sum(0_i64) { |parameter| parameter.data.numel.to_i64 } +
                    @conditioning.parameter_elements + @block.parameter_elements +
                    @out_layer.parameters.sum(0_i64) { |parameter| parameter.data.numel.to_i64 }
      unless constructed == @parameter_elements
        raise "dense-flow parameter accounting mismatch: #{@parameter_elements} != #{constructed}"
      end
    end

    def forward(voxels : Tensor, timesteps : Tensor, context : Tensor) : Tensor
      forward_with_trace(voxels, timesteps, context)["output"]
    end

    def call(voxels : Tensor, timesteps : Tensor, context : Tensor) : Tensor
      forward(voxels, timesteps, context)
    end

    def forward_with_trace(
      voxels : Tensor,
      timesteps : Tensor,
      context : Tensor,
    ) : Hash(String, Tensor)
      validate_inputs!(voxels, timesteps, context)
      batch = voxels.shape[0]
      context_length = context.shape[1]
      preflight_logical_tensors!(batch, context_length)
      validate_parameters!

      voxel_values = ConditioningCPU.finite_values!(voxels, "voxel input")
      ConditioningCPU.finite_values!(timesteps, "timesteps")
      ConditioningCPU.finite_values!(context, "context")

      flattened_values = Array(Float32).new(voxels.numel, 0.0_f32)
      batch.times do |batch_index|
        @voxel_count.times do |position|
          @in_channels.times do |channel|
            source = (batch_index * @in_channels + channel) * @voxel_count + position
            destination = (batch_index * @voxel_count + position) * @in_channels + channel
            flattened_values[destination] = voxel_values[source]
          end
        end
      end
      flattened = DenseBlockCPU.tensor_from(
        flattened_values,
        Shape.new(batch, @voxel_count, @in_channels)
      )
      input_projected = DenseBlockCPU.linear(@input_layer, flattened)

      conditioning_trace = @conditioning.forward_with_trace(timesteps)
      coordinates = coordinate_tensor
      rope_trace = @conditioning.rotary.forward_with_trace(coordinates)
      block_trace = @block.forward_with_trace(
        Autograd::Variable.new(input_projected, requires_grad: false),
        Autograd::Variable.new(conditioning_trace["mod"], requires_grad: false),
        Autograd::Variable.new(context, requires_grad: false),
        rope_trace["phases"]
      )

      final_normalized = @final_norm.forward(
        Autograd::Variable.new(block_trace["output"], requires_grad: false)
      ).data
      output_tokens = DenseBlockCPU.linear(@out_layer, final_normalized)
      output_values = Array(Float32).new(output_tokens.numel, 0.0_f32)
      token_values = output_tokens.to_contiguous_cpu.cpu_data.not_nil!
      batch.times do |batch_index|
        @voxel_count.times do |position|
          @out_channels.times do |channel|
            source = (batch_index * @voxel_count + position) * @out_channels + channel
            destination = (batch_index * @out_channels + channel) * @voxel_count + position
            output_values[destination] = token_values[source]
          end
        end
      end
      output = DenseBlockCPU.tensor_from(
        output_values,
        Shape.new(batch, @out_channels, @resolution, @resolution, @resolution)
      )

      trace = {
        "coordinates"     => coordinates,
        "flattened"       => flattened,
        "input_projected" => input_projected,
      } of String => Tensor
      conditioning_trace.each { |name, tensor| trace["conditioning.#{name}"] = tensor }
      rope_trace.each { |name, tensor| trace["rope.#{name}"] = tensor }
      block_trace.each { |name, tensor| trace["block.#{name}"] = tensor }
      trace["final_norm"] = final_normalized
      trace["output_tokens"] = output_tokens
      trace["output"] = output
      trace
    end

    def parameters : Array(Autograd::Variable)
      result = [] of Autograd::Variable
      result.concat(@input_layer.parameters)
      result.concat(@conditioning.parameters)
      result.concat([
        @block.modulation,
        @block.norm2.weight,
        @block.norm2.bias,
        @block.self_attn.q_rms_norm.gamma,
        @block.self_attn.k_rms_norm.gamma,
        @block.cross_attn.q_rms_norm.gamma,
        @block.cross_attn.k_rms_norm.gamma,
      ])
      [
        @block.self_attn.to_qkv,
        @block.self_attn.to_out,
        @block.cross_attn.to_q,
        @block.cross_attn.to_kv,
        @block.cross_attn.to_out,
        @block.mlp.fc1,
        @block.mlp.fc2,
      ].each { |linear| result.concat(linear.parameters) }
      result.concat(@out_layer.parameters)
      result
    end

    private def validate_parameters! : Nil
      parameters.each_with_index do |parameter, index|
        ConditioningCPU.finite_values!(
          parameter.data,
          "dense-flow parameter #{index}"
        )
      end
    end

    private def validate_inputs!(voxels : Tensor, timesteps : Tensor, context : Tensor) : Nil
      DenseBlockCPU.reject_gpu!(voxels, "voxel input")
      DenseBlockCPU.reject_gpu!(timesteps, "timesteps")
      DenseBlockCPU.reject_gpu!(context, "context")
      unless voxels.ndim == 5
        raise ArgumentError.new("voxel input must be rank 5, got #{voxels.shape}")
      end
      expected = [voxels.shape[0], @in_channels, @resolution, @resolution, @resolution]
      unless voxels.shape[0] > 0 && voxels.shape.to_a == expected
        raise ArgumentError.new("voxel input must have shape [batch, #{@in_channels}, #{@resolution}, #{@resolution}, #{@resolution}], got #{voxels.shape}")
      end
      unless voxels.contiguous?
        raise ArgumentError.new("voxel input must be contiguous to match the upstream view layout")
      end
      unless timesteps.ndim == 1 && timesteps.shape[0] == voxels.shape[0]
        raise ArgumentError.new("timesteps must have shape [#{voxels.shape[0]}], got #{timesteps.shape}")
      end
      unless context.ndim == 3 && context.shape[0] == voxels.shape[0] && context.shape[1] > 0 && context.shape[2] == @context_channels
        raise ArgumentError.new("context must have shape [#{voxels.shape[0]}, sequence, #{@context_channels}], got #{context.shape}")
      end
    end

    private def preflight_logical_tensors!(batch : Int32, context_length : Int32) : Nil
      head_dim = @model_channels // @num_heads
      hidden = (@model_channels.to_f64 * @mlp_ratio.to_f64).to_i64
      factors = {
        "voxel input"            => [batch.to_i64, @in_channels.to_i64, @voxel_count.to_i64],
        "flattened input"        => [batch.to_i64, @voxel_count.to_i64, @in_channels.to_i64],
        "projected tokens"       => [batch.to_i64, @voxel_count.to_i64, @model_channels.to_i64],
        "context"                => [batch.to_i64, context_length.to_i64, @context_channels.to_i64],
        "timestep frequency"     => [batch.to_i64, @frequency_dim.to_i64],
        "timestep embedding"     => [batch.to_i64, @model_channels.to_i64],
        "shared modulation"      => [batch.to_i64, 6_i64, @model_channels.to_i64],
        "coordinates"            => [@voxel_count.to_i64, 3_i64],
        "rotary phases"          => [@voxel_count.to_i64, head_dim.to_i64],
        "self attention qkv"     => [batch.to_i64, @voxel_count.to_i64, 3_i64, @model_channels.to_i64],
        "self attention scores"  => [batch.to_i64, @num_heads.to_i64, @voxel_count.to_i64, @voxel_count.to_i64],
        "cross attention kv"     => [batch.to_i64, context_length.to_i64, 2_i64, @model_channels.to_i64],
        "cross attention scores" => [batch.to_i64, @num_heads.to_i64, @voxel_count.to_i64, context_length.to_i64],
        "mlp hidden"             => [batch.to_i64, @voxel_count.to_i64, hidden],
        "output tokens"          => [batch.to_i64, @voxel_count.to_i64, @out_channels.to_i64],
      }
      max_elements = @max_logical_tensor_bytes // 4_i64
      factors.each do |name, dimensions|
        begin
          DenseFlowCPU.bounded_product!(dimensions, max_elements, name)
        rescue ex : ArgumentError
          raise ArgumentError.new(
            "dense-flow logical tensor budget rejects #{name}: #{ex.message}"
          )
        end
      end
    end

    private def coordinate_tensor : Tensor
      values = Array(Float32).new(@voxel_count * 3, 0.0_f32)
      position = 0
      @resolution.times do |axis0|
        @resolution.times do |axis1|
          @resolution.times do |axis2|
            offset = position * 3
            values[offset] = axis0.to_f32
            values[offset + 1] = axis1.to_f32
            values[offset + 2] = axis2.to_f32
            position += 1
          end
        end
      end
      DenseBlockCPU.tensor_from(values, Shape.new(@voxel_count, 3_i32))
    end
  end
end
