# CPU-only F32 execution oracle for the T2N2d0b right-padding contract.
#
# This wrapper deliberately keeps DenseFlowStageCPU's logical geometry intact:
# it pads flattened token rows to a declared bucket, carries explicit valid
# lengths through the block, and trims before exposing NCDHW output. It does
# not instantiate a padded spatial resolution or perform any device work.

require "./dense_flow"
require "./device_resource_contract"

module ML::ThreeD::Trellis2
  # The contract and request are the only plan authority. Construction derives
  # and preflights that plan before admitting its keys, so a caller cannot
  # smuggle a forged/stale activation plan past the owner-scoped ledger.
  class PaddedDenseFlowStageCPU
    CPU_ORACLE_COMPILER_ABI = "t2n2d0-v1"

    getter stage : DenseFlowStageCPU
    getter contract : DenseDeviceResourceContract
    getter request : DenseActivationRequest
    getter plan : DenseActivationPlan
    getter ledger : BoundedKernelKeyLedger

    def initialize(
      @stage : DenseFlowStageCPU,
      @contract : DenseDeviceResourceContract,
      @request : DenseActivationRequest,
      @ledger : BoundedKernelKeyLedger,
    )
      @plan = @contract.plan(@request)
      validate_static_plan!
      preflight_padded_inventory!
      @ledger.admit!(@plan.kernel_keys)
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
      validate_parameters!

      batch = voxels.shape[0]
      logical_voxels = @plan.logical_voxel_tokens
      padded_voxels = @plan.padded_voxel_tokens
      logical_context = @plan.logical_context_tokens
      padded_context = @plan.padded_context_tokens

      voxel_values = ConditioningCPU.finite_values!(voxels, "voxel input")
      ConditioningCPU.finite_values!(timesteps, "timesteps")
      context_values = ConditioningCPU.finite_values!(context, "context")

      flattened = flatten_and_pad_voxels(
        voxel_values,
        batch,
        logical_voxels,
        padded_voxels
      )
      padded_memory = pad_context(
        context_values,
        batch,
        logical_context,
        padded_context
      )
      input_projected = DenseBlockCPU.linear(@stage.input_layer, flattened)

      conditioning_trace = @stage.conditioning.forward_with_trace(timesteps)
      coordinates = coordinate_tensor
      logical_rope_trace = @stage.conditioning.rotary.forward_with_trace(coordinates)
      padded_phases = pad_phases(
        logical_rope_trace["phases"],
        padded_voxels
      )
      block_trace = @stage.block.forward_with_trace(
        Autograd::Variable.new(input_projected, requires_grad: false),
        Autograd::Variable.new(conditioning_trace["mod"], requires_grad: false),
        Autograd::Variable.new(padded_memory, requires_grad: false),
        padded_phases,
        valid_voxel_tokens: logical_voxels,
        valid_context_tokens: logical_context
      )

      final_normalized = @stage.final_norm.forward(
        Autograd::Variable.new(block_trace["output"], requires_grad: false)
      ).data
      padded_output_tokens = DenseBlockCPU.linear(@stage.out_layer, final_normalized)
      output_tokens = trim_tokens(
        padded_output_tokens,
        logical_voxels
      )
      output = inverse_ncdhw(output_tokens)

      trace = {
        "coordinates"     => coordinates,
        "flattened"       => flattened,
        "input_projected" => input_projected,
      } of String => Tensor
      conditioning_trace.each { |name, tensor| trace["conditioning.#{name}"] = tensor }
      logical_rope_trace.each { |name, tensor| trace["rope.#{name}"] = tensor }
      trace["padding.phases"] = padded_phases
      block_trace.each { |name, tensor| trace["block.#{name}"] = tensor }
      trace["final_norm"] = final_normalized
      trace["output_tokens"] = output_tokens
      trace["output"] = output
      trace
    end

    private def validate_static_plan! : Nil
      validate_cpu_oracle_abi!
      unless @plan.padding_policy == DenseDeviceResourceContract::PADDING_POLICY
        raise ArgumentError.new(
          "padded stage requires #{DenseDeviceResourceContract::PADDING_POLICY}"
        )
      end
      unless @plan.logical_batch == @plan.padded_batch
        raise ArgumentError.new("padded stage does not admit batch padding")
      end
      unless @plan.logical_voxel_tokens == @stage.voxel_count
        raise ArgumentError.new(
          "plan logical voxel length #{@plan.logical_voxel_tokens} does not match " \
          "stage voxel count #{@stage.voxel_count}"
        )
      end
      validate_positive_cube!(@plan.logical_voxel_tokens, "logical voxel token")
      validate_positive_cube!(@plan.padded_voxel_tokens, "padded voxel token")
      unless @plan.padded_voxel_tokens >= @plan.logical_voxel_tokens
        raise ArgumentError.new("padded voxel token length must contain logical length")
      end
      unless @plan.logical_context_tokens > 0 &&
             @plan.padded_context_tokens >= @plan.logical_context_tokens
        raise ArgumentError.new("padded context token length must contain logical length")
      end
      keys = @plan.kernel_keys
      unless !keys.empty? && keys.all? { |key| key.dtype == ML::DType::F32 }
        raise ArgumentError.new("padded stage requires an F32 resource plan")
      end
      unless keys.all? do |key|
               key.padding_policy == @plan.padding_policy &&
               key.padded_batch == @plan.padded_batch &&
               key.padded_voxel_tokens == @plan.padded_voxel_tokens &&
               key.padded_context_tokens == @plan.padded_context_tokens
             end
        raise ArgumentError.new(
          "padded stage resource keys must match plan batch, padded dimensions, dtype, and policy"
        )
      end
    end

    private def validate_cpu_oracle_abi! : Nil
      abi = @contract.kernel_abi
      valid = abi.device_family == "cpu-oracle" &&
              abi.compiler_abi == CPU_ORACLE_COMPILER_ABI &&
              abi.weight_format == "f32-reference" &&
              abi.accumulation_dtype == ML::DType::F32 &&
              abi.activation_mode == "silu-gelu" &&
              abi.normalization_mode == "layer-rms" &&
              abi.attention_mode == "self-cross-qkrms" &&
              abi.rope_mode == DenseDeviceResourceContract::REQUIRED_ROPE_MODE &&
              abi.mask_mode == DenseDeviceResourceContract::REQUIRED_MASK_MODE &&
              abi.layout_mode == DenseDeviceResourceContract::REQUIRED_LAYOUT_MODE
      return if valid

      raise ArgumentError.new(
        "padded stage requires exact CPU oracle ABI #{CPU_ORACLE_COMPILER_ABI} " \
        "(F32, cpu-oracle, f32-reference, silu-gelu, layer-rms, self-cross-qkrms)"
      )
    end

    private def validate_inputs!(voxels : Tensor, timesteps : Tensor, context : Tensor) : Nil
      DenseBlockCPU.reject_gpu!(voxels, "voxel input")
      DenseBlockCPU.reject_gpu!(timesteps, "timesteps")
      DenseBlockCPU.reject_gpu!(context, "context")

      unless voxels.ndim == 5
        raise ArgumentError.new("voxel input must be rank 5, got #{voxels.shape}")
      end
      batch = voxels.shape[0]
      resolution = @stage.resolution
      expected_voxels = [batch, @stage.in_channels, resolution, resolution, resolution]
      unless batch > 0 && voxels.shape.to_a == expected_voxels
        raise ArgumentError.new(
          "voxel input must have logical shape [batch, #{@stage.in_channels}, " \
          "#{resolution}, #{resolution}, #{resolution}], got #{voxels.shape}"
        )
      end
      unless voxels.contiguous?
        raise ArgumentError.new("voxel input must be contiguous to match the logical NCDHW view")
      end
      unless batch == @plan.logical_batch && batch == @plan.padded_batch
        raise ArgumentError.new("plan logical batch length does not match voxel input batch")
      end
      unless timesteps.ndim == 1 && timesteps.shape[0] == batch
        raise ArgumentError.new("timesteps must have shape [#{batch}], got #{timesteps.shape}")
      end
      unless context.ndim == 3 && context.shape[0] == batch &&
             context.shape[1] == @plan.logical_context_tokens &&
             context.shape[2] == @stage.context_channels
        raise ArgumentError.new(
          "logical context length must match plan; context must have logical shape [#{batch}, #{@plan.logical_context_tokens}, " \
          "#{@stage.context_channels}], got #{context.shape}"
        )
      end
    end

    private def validate_parameters! : Nil
      @stage.parameters.each_with_index do |parameter, index|
        ConditioningCPU.finite_values!(parameter.data, "padded stage parameter #{index}")
      end
    end

    private def preflight_padded_inventory! : Nil
      b = @plan.padded_batch.to_i64
      n = @plan.padded_voxel_tokens.to_i64
      s = @plan.padded_context_tokens.to_i64
      i = @stage.in_channels.to_i64
      c = @stage.model_channels.to_i64
      k = @stage.context_channels.to_i64
      o = @stage.out_channels.to_i64
      h = @stage.num_heads.to_i64
      m = (@stage.model_channels.to_f64 * @stage.mlp_ratio.to_f64).to_i64
      f = @stage.frequency_dim.to_i64
      head_dim = c // h
      expected = {
        "voxel input"            => [b, i, n],
        "flattened input"        => [b, n, i],
        "projected tokens"       => [b, n, c],
        "context"                => [b, s, k],
        "timestep frequency"     => [b, f],
        "timestep embedding"     => [b, c],
        "shared modulation"      => [b, 6_i64, c],
        "coordinates"            => [n, 3_i64],
        "rotary phases"          => [n, head_dim],
        "self attention qkv"     => [b, n, 3_i64, c],
        "self attention scores"  => [b, h, n, n],
        "cross attention kv"     => [b, s, 2_i64, c],
        "cross attention scores" => [b, h, n, s],
        "mlp hidden"             => [b, n, m],
        "output tokens"          => [b, n, o],
        "output"                 => [b, o, n],
      }
      declared = @plan.declared_tensor_bytes
      expected.each do |name, dimensions|
        elements = checked_product!(dimensions, name)
        bytes = checked_byte_size!(elements, name)
        if bytes > @stage.max_logical_tensor_bytes
          raise ArgumentError.new(
            "resource plan #{name} padded tensor #{bytes} bytes exceeds stage " \
            "max_logical_tensor_bytes #{@stage.max_logical_tensor_bytes}"
          )
        end
        unless declared[name]? == bytes
          raise ArgumentError.new(
            "resource plan #{name} bytes do not match padded stage geometry"
          )
        end
      end
      unless declared.keys.sort == expected.keys.sort
        raise ArgumentError.new("resource plan tensor inventory does not match padded stage")
      end
      declared_total = checked_sum!(declared.values, "resource plan padded tensor budget")
      unless declared.values.all? { |bytes| bytes > 0 && bytes <= @plan.max_single_tensor_bytes } &&
             declared_total == @plan.declared_activation_bytes
        raise ArgumentError.new("resource plan padded tensor budget is inconsistent")
      end
    end

    private def checked_product!(dimensions : Array(Int64), name : String) : Int64
      product = 1_i64
      dimensions.each do |dimension|
        raise ArgumentError.new("#{name} dimensions must be positive") unless dimension > 0
        if product > Int64::MAX // dimension
          raise ArgumentError.new("#{name} dimensions overflow")
        end
        product *= dimension
      end
      product
    end

    private def checked_byte_size!(elements : Int64, name : String) : Int64
      if elements > Int64::MAX // 4_i64
        raise ArgumentError.new("#{name} dimensions overflow F32 byte size")
      end
      elements * 4_i64
    end

    private def checked_sum!(values : Array(Int64), name : String) : Int64
      total = 0_i64
      values.each do |value|
        raise ArgumentError.new("#{name} contains a negative tensor size") if value < 0
        if total > Int64::MAX - value
          raise ArgumentError.new("#{name} overflows Int64")
        end
        total += value
      end
      total
    end

    private def flatten_and_pad_voxels(
      values : Array(Float32),
      batch : Int32,
      logical_tokens : Int32,
      padded_tokens : Int32,
    ) : Tensor
      result = Array(Float32).new(batch * padded_tokens * @stage.in_channels, 0.0_f32)
      batch.times do |batch_index|
        logical_tokens.times do |position|
          @stage.in_channels.times do |channel|
            source = (batch_index * @stage.in_channels + channel) * logical_tokens + position
            destination = (batch_index * padded_tokens + position) * @stage.in_channels + channel
            result[destination] = values[source]
          end
        end
      end
      DenseBlockCPU.tensor_from(
        result,
        Shape.new(batch, padded_tokens, @stage.in_channels)
      )
    end

    private def pad_context(
      values : Array(Float32),
      batch : Int32,
      logical_tokens : Int32,
      padded_tokens : Int32,
    ) : Tensor
      result = Array(Float32).new(batch * padded_tokens * @stage.context_channels, 0.0_f32)
      batch.times do |batch_index|
        logical_tokens.times do |position|
          @stage.context_channels.times do |channel|
            source = (batch_index * logical_tokens + position) * @stage.context_channels + channel
            destination = (batch_index * padded_tokens + position) * @stage.context_channels + channel
            result[destination] = values[source]
          end
        end
      end
      DenseBlockCPU.tensor_from(
        result,
        Shape.new(batch, padded_tokens, @stage.context_channels)
      )
    end

    private def coordinate_tensor : Tensor
      logical_tokens = @plan.logical_voxel_tokens
      values = Array(Float32).new(logical_tokens * 3, 0.0_f32)
      position = 0
      @stage.resolution.times do |axis0|
        @stage.resolution.times do |axis1|
          @stage.resolution.times do |axis2|
            offset = position * 3
            values[offset] = axis0.to_f32
            values[offset + 1] = axis1.to_f32
            values[offset + 2] = axis2.to_f32
            position += 1
          end
        end
      end
      DenseBlockCPU.tensor_from(values, Shape.new(logical_tokens, 3_i32))
    end

    private def pad_phases(logical : Tensor, padded_tokens : Int32) : Tensor
      DenseBlockCPU.reject_gpu!(logical, "logical RoPE phases")
      unless logical.ndim == 3 && logical.shape[2] == 2 && logical.shape[0] == @plan.logical_voxel_tokens && logical.shape[0] <= padded_tokens
        raise ArgumentError.new("logical RoPE phases must have shape [N, pairs, 2] within padded length")
      end
      logical_tokens = logical.shape[0]
      pairs = logical.shape[1]
      source = logical.to_contiguous_cpu.cpu_data.not_nil!
      result = Array(Float32).new(padded_tokens * pairs * 2, 0.0_f32)
      source.each_with_index { |value, index| result[index] = value }
      (logical_tokens...padded_tokens).each do |position|
        pairs.times do |pair|
          offset = (position * pairs + pair) * 2
          result[offset] = 1.0_f32
          result[offset + 1] = 0.0_f32
        end
      end
      DenseBlockCPU.tensor_from(result, Shape.new(padded_tokens, pairs, 2_i32))
    end

    private def trim_tokens(tokens : Tensor, logical_tokens : Int32) : Tensor
      DenseBlockCPU.reject_gpu!(tokens, "padded output tokens")
      unless tokens.ndim == 3 && tokens.shape[1] >= logical_tokens && tokens.shape[2] == @stage.out_channels
        raise ArgumentError.new("padded output tokens have an incompatible shape")
      end
      batch = tokens.shape[0]
      physical = tokens.shape[1]
      values = tokens.to_contiguous_cpu.cpu_data.not_nil!
      result = Array(Float32).new(batch * logical_tokens * @stage.out_channels, 0.0_f32)
      batch.times do |batch_index|
        logical_tokens.times do |position|
          @stage.out_channels.times do |channel|
            source = (batch_index * physical + position) * @stage.out_channels + channel
            destination = (batch_index * logical_tokens + position) * @stage.out_channels + channel
            result[destination] = values[source]
          end
        end
      end
      DenseBlockCPU.tensor_from(
        result,
        Shape.new(batch, logical_tokens, @stage.out_channels)
      )
    end

    private def inverse_ncdhw(tokens : Tensor) : Tensor
      batch = tokens.shape[0]
      logical_tokens = tokens.shape[1]
      values = tokens.to_contiguous_cpu.cpu_data.not_nil!
      result = Array(Float32).new(batch * @stage.out_channels * logical_tokens, 0.0_f32)
      batch.times do |batch_index|
        logical_tokens.times do |position|
          @stage.out_channels.times do |channel|
            source = (batch_index * logical_tokens + position) * @stage.out_channels + channel
            destination = (batch_index * @stage.out_channels + channel) * logical_tokens + position
            result[destination] = values[source]
          end
        end
      end
      DenseBlockCPU.tensor_from(
        result,
        Shape.new(
          batch,
          @stage.out_channels,
          @stage.resolution,
          @stage.resolution,
          @stage.resolution
        )
      )
    end

    private def validate_positive_cube!(value : Int32, label : String) : Nil
      raise ArgumentError.new("#{label} length must be positive") unless value > 0
      root = Math.cbrt(value.to_f64).round.to_i64
      unless root > 0 && root * root * root == value.to_i64
        raise ArgumentError.new("#{label} length must be an exact positive cube")
      end
    end
  end
end
