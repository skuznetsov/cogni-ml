# CPU-only F32 reference formulas for TRELLIS.2 timestep conditioning and
# coordinate-derived rotary phases.  This is deliberately an observable
# boundary in front of the bounded dense-block oracle, not a model runtime.

require "./dense_block"

module ML::ThreeD::Trellis2
  module ConditioningCPU
    extend self

    MAX_REFERENCE_PARAMETER_BYTES    = 64_i64 * 1024_i64 * 1024_i64
    MAX_REFERENCE_PARAMETER_ELEMENTS = MAX_REFERENCE_PARAMETER_BYTES // 4_i64
    # Per generated tensor. This is not an aggregate peak-working-set claim.
    MAX_REFERENCE_TENSOR_BYTES    = 64_i64 * 1024_i64 * 1024_i64
    MAX_REFERENCE_TENSOR_ELEMENTS = MAX_REFERENCE_TENSOR_BYTES // 4_i64

    def ensure_dimension!(value : Int32, name : String) : Int64
      raise ArgumentError.new("#{name} must be positive") unless value > 0
      widened = value.to_i64
      if widened > MAX_REFERENCE_PARAMETER_ELEMENTS
        raise ArgumentError.new(
          "conditioning parameter budget cannot represent #{name}=#{value}"
        )
      end
      widened
    end

    def timestep_parameter_elements(channels : Int32, frequency_dim : Int32) : Int64
      c = ensure_dimension!(channels, "channels")
      f = ensure_dimension!(frequency_dim, "frequency_dim")
      f * c + c + c * c + c
    end

    def modulation_parameter_elements(channels : Int32) : Int64
      c = ensure_dimension!(channels, "channels")
      6_i64 * c * c + 6_i64 * c
    end

    def conditioning_parameter_elements(channels : Int32, frequency_dim : Int32) : Int64
      elements = timestep_parameter_elements(channels, frequency_dim) +
                 modulation_parameter_elements(channels)
      if elements > MAX_REFERENCE_PARAMETER_ELEMENTS
        raise ArgumentError.new(
          "conditioning parameter budget #{elements} elements exceeds " \
          "#{MAX_REFERENCE_PARAMETER_ELEMENTS}"
        )
      end
      elements
    end

    def ensure_parameter_budget!(elements : Int64, label : String) : Nil
      if elements > MAX_REFERENCE_PARAMETER_ELEMENTS
        raise ArgumentError.new(
          "#{label} parameter budget #{elements} elements exceeds " \
          "#{MAX_REFERENCE_PARAMETER_ELEMENTS}"
        )
      end
    end

    def ensure_tensor_budget!(rows : Int32, width : Int32, label : String) : Nil
      raise ArgumentError.new("#{label} rows must be positive") unless rows > 0
      raise ArgumentError.new("#{label} width must be positive") unless width > 0
      elements = rows.to_i64 * width.to_i64
      if elements > MAX_REFERENCE_TENSOR_ELEMENTS
        raise ArgumentError.new(
          "#{label} output budget requires #{elements} F32 elements but " \
          "#{MAX_REFERENCE_TENSOR_BYTES} bytes hold at most " \
          "#{MAX_REFERENCE_TENSOR_ELEMENTS}"
        )
      end
    end

    def silu(input : Tensor) : Tensor
      DenseBlockCPU.reject_gpu!(input, "SiLU input")
      values = input.to_contiguous_cpu.cpu_data.not_nil!
      result = Array(Float32).new(input.numel, 0.0_f32)
      values.each_with_index do |value, index|
        sigmoid = 1.0_f64 / (1.0_f64 + Math.exp(-value.to_f64))
        result[index] = (value.to_f64 * sigmoid).to_f32
      end
      DenseBlockCPU.tensor_from(result, input.shape)
    end

    def freeze_linear(linear : ML::NN::Linear) : Nil
      linear.weight.requires_grad = false
      if bias = linear.bias
        bias.requires_grad = false
      end
    end

    def finite_values!(tensor : Tensor, name : String) : Array(Float32)
      DenseBlockCPU.reject_gpu!(tensor, name)
      values = tensor.to_contiguous_cpu.cpu_data.not_nil!
      unless values.all?(&.finite?)
        raise ArgumentError.new("#{name} values must be finite")
      end
      values
    end
  end

  class TimestepEmbedderCPU
    getter channels : Int32
    getter frequency_dim : Int32
    getter max_period : Float32
    getter first_linear : ML::NN::Linear
    getter second_linear : ML::NN::Linear
    getter parameter_elements : Int64

    def initialize(
      @channels : Int32,
      @frequency_dim : Int32 = 256,
      @max_period : Float32 = 10000.0_f32,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("TimestepEmbedderCPU is CPU-only") if device.gpu?
      raise ArgumentError.new("frequency_dim must be at least 2") unless @frequency_dim >= 2
      unless @max_period.finite? && @max_period > 0.0_f32
        raise ArgumentError.new("max_period must be finite and positive")
      end
      half = @frequency_dim // 2
      highest_exponent = -Math.log(@max_period.to_f64) * (half - 1).to_f64 / half.to_f64
      highest_frequency = Math.exp(highest_exponent)
      unless highest_frequency.finite? && highest_frequency <= Float32::MAX.to_f64
        raise ArgumentError.new("timestep frequency range must be representable in F32")
      end
      @parameter_elements = ConditioningCPU.timestep_parameter_elements(@channels, @frequency_dim)
      ConditioningCPU.ensure_parameter_budget!(@parameter_elements, "timestep")
      @first_linear = ML::NN::Linear.new(@frequency_dim, @channels, device: Tensor::Device::CPU)
      @second_linear = ML::NN::Linear.new(@channels, @channels, device: Tensor::Device::CPU)
      ConditioningCPU.freeze_linear(@first_linear)
      ConditioningCPU.freeze_linear(@second_linear)
    end

    def forward(timesteps : Tensor) : Tensor
      forward_with_trace(timesteps)["t_emb"]
    end

    def call(timesteps : Tensor) : Tensor
      forward(timesteps)
    end

    def forward_with_trace(timesteps : Tensor) : Hash(String, Tensor)
      unless timesteps.ndim == 1 && timesteps.shape[0] > 0
        raise ArgumentError.new("timesteps must be a non-empty rank-1 tensor")
      end
      batch = timesteps.shape[0]
      ConditioningCPU.ensure_tensor_budget!(batch, @frequency_dim, "timestep frequency")
      ConditioningCPU.ensure_tensor_budget!(batch, @channels, "timestep embedding")
      values = ConditioningCPU.finite_values!(timesteps, "timestep")

      half = @frequency_dim // 2
      frequencies = Array(Float32).new(half, 0.0_f32)
      half.times do |index|
        exponent = -Math.log(@max_period.to_f64) * index.to_f64 / half.to_f64
        frequencies[index] = Math.exp(exponent).to_f32
      end
      embedding = Array(Float32).new(batch * @frequency_dim, 0.0_f32)
      batch.times do |batch_index|
        half.times do |frequency_index|
          argument = values[batch_index].to_f64 * frequencies[frequency_index].to_f64
          embedding[batch_index * @frequency_dim + frequency_index] = Math.cos(argument).to_f32
          embedding[batch_index * @frequency_dim + half + frequency_index] = Math.sin(argument).to_f32
        end
        if @frequency_dim.odd?
          embedding[batch_index * @frequency_dim + @frequency_dim - 1] = 0.0_f32
        end
      end

      t_freq = DenseBlockCPU.tensor_from(embedding, Shape.new(batch, @frequency_dim))
      first_linear = DenseBlockCPU.linear(@first_linear, t_freq)
      first_silu = ConditioningCPU.silu(first_linear)
      t_emb = DenseBlockCPU.linear(@second_linear, first_silu)
      {
        "t_freq"       => t_freq,
        "first_linear" => first_linear,
        "first_silu"   => first_silu,
        "t_emb"        => t_emb,
      }
    end

    def parameters : Array(Autograd::Variable)
      @first_linear.parameters + @second_linear.parameters
    end
  end

  class SharedModulationCPU
    getter channels : Int32
    getter linear : ML::NN::Linear
    getter parameter_elements : Int64

    def initialize(
      @channels : Int32,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("SharedModulationCPU is CPU-only") if device.gpu?
      @parameter_elements = ConditioningCPU.modulation_parameter_elements(@channels)
      ConditioningCPU.ensure_parameter_budget!(@parameter_elements, "shared modulation")
      output_width = 6_i64 * @channels.to_i64
      if output_width > Int32::MAX
        raise ArgumentError.new("shared modulation output width is not representable")
      end
      @linear = ML::NN::Linear.new(@channels, output_width.to_i32, device: Tensor::Device::CPU)
      ConditioningCPU.freeze_linear(@linear)
    end

    def forward(timestep_embedding : Tensor) : Tensor
      forward_with_trace(timestep_embedding)["mod"]
    end

    def call(timestep_embedding : Tensor) : Tensor
      forward(timestep_embedding)
    end

    def forward_with_trace(timestep_embedding : Tensor) : Hash(String, Tensor)
      unless timestep_embedding.ndim == 2 && timestep_embedding.shape[0] > 0 && timestep_embedding.shape[1] == @channels
        raise ArgumentError.new(
          "timestep embedding must have shape [batch, #{@channels}], got #{timestep_embedding.shape}"
        )
      end
      ConditioningCPU.ensure_tensor_budget!(timestep_embedding.shape[0], 6 * @channels, "shared modulation")
      ConditioningCPU.finite_values!(timestep_embedding, "timestep embedding")
      top_silu = ConditioningCPU.silu(timestep_embedding)
      modulation = DenseBlockCPU.linear(@linear, top_silu)
      {"top_silu" => top_silu, "mod" => modulation}
    end

    def parameters : Array(Autograd::Variable)
      @linear.parameters
    end
  end

  class RotaryPositionEmbedderCPU
    getter head_dim : Int32
    getter dim : Int32
    getter frequency_dim : Int32
    getter rope_low : Float32
    getter rope_high : Float32
    getter max_output_bytes : Int64

    def initialize(
      @head_dim : Int32,
      @dim : Int32 = 3,
      @rope_low : Float32 = 1.0_f32,
      @rope_high : Float32 = 10000.0_f32,
      @max_output_bytes : Int64 = ConditioningCPU::MAX_REFERENCE_TENSOR_BYTES,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("RotaryPositionEmbedderCPU is CPU-only") if device.gpu?
      raise ArgumentError.new("head_dim must be positive and even") unless @head_dim > 0 && @head_dim.even?
      raise ArgumentError.new("spatial dimension must be positive") unless @dim > 0
      @frequency_dim = @head_dim // 2 // @dim
      if @frequency_dim <= 0
        raise ArgumentError.new(
          "RoPE frequency dimension must be positive; head_dim must provide at least one frequency per spatial dimension"
        )
      end
      unless @rope_low.finite? && @rope_low > 0.0_f32 && @rope_high.finite? && @rope_high > 0.0_f32
        raise ArgumentError.new("RoPE frequencies must be finite and positive")
      end
      highest_exponent = (@frequency_dim - 1).to_f64 / @frequency_dim.to_f64
      highest_frequency = if @rope_high < 1.0_f32
                            @rope_low.to_f64 / (@rope_high.to_f64 ** highest_exponent)
                          else
                            @rope_low.to_f64
                          end
      unless highest_frequency.finite? && highest_frequency <= Float32::MAX.to_f64
        raise ArgumentError.new("RoPE frequency range must be representable in F32")
      end
      unless 0_i64 < @max_output_bytes <= ConditioningCPU::MAX_REFERENCE_TENSOR_BYTES
        raise ArgumentError.new(
          "phase output budget must be positive and no greater than " \
          "#{ConditioningCPU::MAX_REFERENCE_TENSOR_BYTES}"
        )
      end
    end

    def forward(coordinates : Tensor) : Tensor
      forward_with_trace(coordinates)["phases"]
    end

    def call(coordinates : Tensor) : Tensor
      forward(coordinates)
    end

    def forward_with_trace(coordinates : Tensor) : Hash(String, Tensor)
      unless coordinates.ndim == 2
        raise ArgumentError.new("coordinates must be a rank-2 tensor")
      end
      unless coordinates.shape[0] > 0
        raise ArgumentError.new("coordinates must contain at least one coordinate")
      end
      unless coordinates.shape[1] == @dim
        raise ArgumentError.new("coordinate width must be #{@dim}, got #{coordinates.shape[1]}")
      end
      count = coordinates.shape[0]
      pairs = @head_dim // 2
      output_elements = count.to_i64 * pairs.to_i64 * 2_i64
      max_output_elements = @max_output_bytes // 4_i64
      if output_elements > max_output_elements
        raise ArgumentError.new(
          "phase output budget requires #{output_elements} F32 elements but " \
          "#{@max_output_bytes} bytes hold at most #{max_output_elements}"
        )
      end
      coordinate_values = ConditioningCPU.finite_values!(coordinates, "coordinate")

      frequency_values = Array(Float32).new(@frequency_dim, 0.0_f32)
      @frequency_dim.times do |index|
        exponent = index.to_f64 / @frequency_dim.to_f64
        value = @rope_low.to_f64 / (@rope_high.to_f64 ** exponent)
        unless value.finite? && value <= Float32::MAX.to_f64
          raise ArgumentError.new("RoPE frequency range must be representable in F32")
        end
        frequency_values[index] = value.to_f32
      end
      angle_width = @dim * @frequency_dim
      angles = Array(Float32).new(count * angle_width, 0.0_f32)
      count.times do |position|
        @dim.times do |axis|
          coordinate = coordinate_values[position * @dim + axis]
          @frequency_dim.times do |frequency_index|
            offset = position * angle_width + axis * @frequency_dim + frequency_index
            angles[offset] = coordinate * frequency_values[frequency_index]
          end
        end
      end

      phases = Array(Float32).new(output_elements.to_i, 0.0_f32)
      count.times do |position|
        pairs.times do |pair|
          phase_offset = (position * pairs + pair) * 2
          if pair < angle_width
            angle = angles[position * angle_width + pair].to_f64
            phases[phase_offset] = Math.cos(angle).to_f32
            phases[phase_offset + 1] = Math.sin(angle).to_f32
          else
            phases[phase_offset] = 1.0_f32
            phases[phase_offset + 1] = 0.0_f32
          end
        end
      end

      {
        "frequencies" => DenseBlockCPU.tensor_from(frequency_values, Shape.new(@frequency_dim)),
        "angles"      => DenseBlockCPU.tensor_from(angles, Shape.new(count, angle_width)),
        "phases"      => DenseBlockCPU.tensor_from(phases, Shape.new(count, pairs, 2_i32)),
      }
    end
  end

  class TimestepConditioningCPU
    getter channels : Int32
    getter frequency_dim : Int32
    getter parameter_elements : Int64
    getter timestep : TimestepEmbedderCPU
    getter modulation : SharedModulationCPU
    getter rotary : RotaryPositionEmbedderCPU

    def initialize(
      @channels : Int32,
      rotary_head_dim : Int32,
      @frequency_dim : Int32 = 256,
      max_period : Float32 = 10000.0_f32,
      spatial_dim : Int32 = 3,
      rope_low : Float32 = 1.0_f32,
      rope_high : Float32 = 10000.0_f32,
      max_phase_output_bytes : Int64 = ConditioningCPU::MAX_REFERENCE_TENSOR_BYTES,
      device : Tensor::Device = Tensor::Device::CPU,
    )
      raise ArgumentError.new("TimestepConditioningCPU is CPU-only") if device.gpu?
      @parameter_elements = ConditioningCPU.conditioning_parameter_elements(@channels, @frequency_dim)
      @timestep = TimestepEmbedderCPU.new(
        @channels,
        @frequency_dim,
        max_period,
        device: Tensor::Device::CPU
      )
      @modulation = SharedModulationCPU.new(@channels, device: Tensor::Device::CPU)
      @rotary = RotaryPositionEmbedderCPU.new(
        head_dim: rotary_head_dim,
        dim: spatial_dim,
        rope_low: rope_low,
        rope_high: rope_high,
        max_output_bytes: max_phase_output_bytes,
        device: Tensor::Device::CPU
      )
    end

    def forward(timesteps : Tensor) : Tensor
      forward_with_trace(timesteps)["mod"]
    end

    def call(timesteps : Tensor) : Tensor
      forward(timesteps)
    end

    def forward_with_trace(timesteps : Tensor) : Hash(String, Tensor)
      timestep_trace = @timestep.forward_with_trace(timesteps)
      modulation_trace = @modulation.forward_with_trace(timestep_trace["t_emb"])
      timestep_trace.merge(modulation_trace)
    end

    def parameters : Array(Autograd::Variable)
      @timestep.parameters + @modulation.parameters
    end
  end
end
