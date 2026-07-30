# Graphless CPU normalization primitives over logical Tensor rows.

require "../core/tensor"

module ML::Ops::CPU
  record NormalizationReadStats,
    input_materialized_bytes : Int64,
    weight_materialized_bytes : Int64,
    bias_materialized_bytes : Int64,
    input_materialized : Bool,
    weight_materialized : Bool,
    bias_materialized : Bool do
    def materialized_bytes : Int64
      input_materialized_bytes + weight_materialized_bytes + bias_materialized_bytes
    end

    def materialization_count : Int32
      count = 0
      count += 1 if input_materialized
      count += 1 if weight_materialized
      count += 1 if bias_materialized
      count
    end
  end

  record LayerNormResult,
    output : ML::Tensor,
    means : Array(Float32),
    inv_stds : Array(Float32),
    read_stats : NormalizationReadStats

  record RMSNormResult,
    output : ML::Tensor,
    inv_rms : Array(Float32),
    read_stats : NormalizationReadStats

  def self.layer_norm(
    input : ML::Tensor,
    weight : ML::Tensor,
    bias : ML::Tensor,
    normalized_shape : Array(Int32),
    eps : Float32 = 1e-5_f32,
  ) : ML::Tensor
    layer_norm_with_stats(input, weight, bias, normalized_shape, eps).output
  end

  def self.layer_norm_with_stats(
    input : ML::Tensor,
    weight : ML::Tensor,
    bias : ML::Tensor,
    normalized_shape : Array(Int32),
    eps : Float32 = 1e-5_f32,
  ) : LayerNormResult
    norm_size = validate_layer_norm_inputs!(input, weight, bias, normalized_shape, eps)

    input_read = input.cpu_read
    weight_read = weight.cpu_read
    bias_read = bias.cpu_read
    batch_size = input.numel // norm_size
    output = ML::Tensor.new(input.shape, input.dtype, ML::Tensor::Device::CPU)
    output_data = output.cpu_data.not_nil!
    means = Array(Float32).new(batch_size, 0.0_f32)
    inv_stds = Array(Float32).new(batch_size, 0.0_f32)

    batch_size.times do |batch|
      offset = batch * norm_size
      sum = 0.0_f64
      norm_size.times { |index| sum += input_read[offset + index].to_f64 }
      mean = sum / norm_size.to_f64

      squared_deviation = 0.0_f64
      norm_size.times do |index|
        difference = input_read[offset + index].to_f64 - mean
        squared_deviation += difference * difference
      end
      inv_std_f64 = 1.0 / Math.sqrt(squared_deviation / norm_size.to_f64 + eps.to_f64)
      inv_std = inv_std_f64.to_f32
      means[batch] = mean.to_f32
      inv_stds[batch] = inv_std

      norm_size.times do |index|
        normalized = (input_read[offset + index].to_f64 - mean) * inv_std_f64
        output_data[offset + index] =
          (normalized * weight_read[index].to_f64 + bias_read[index].to_f64).to_f32
      end
    end

    LayerNormResult.new(
      output,
      means,
      inv_stds,
      read_stats(input_read, weight_read, bias_read)
    )
  end

  def self.rms_norm(
    input : ML::Tensor,
    weight : ML::Tensor,
    dim : Int32,
    eps : Float32 = 1e-5_f32,
  ) : ML::Tensor
    rms_norm_with_stats(input, weight, dim, eps).output
  end

  def self.rms_norm_with_stats(
    input : ML::Tensor,
    weight : ML::Tensor,
    dim : Int32,
    eps : Float32 = 1e-5_f32,
  ) : RMSNormResult
    validate_rms_norm_inputs!(input, weight, dim, eps)

    input_read = input.cpu_read
    weight_read = weight.cpu_read
    batch_size = input.numel // dim
    output = ML::Tensor.new(input.shape, input.dtype, ML::Tensor::Device::CPU)
    output_data = output.cpu_data.not_nil!
    inv_rms = Array(Float32).new(batch_size, 0.0_f32)

    batch_size.times do |batch|
      offset = batch * dim
      sum_squares = 0.0_f64
      dim.times do |index|
        value = input_read[offset + index].to_f64
        sum_squares += value * value
      end
      scale_f64 = 1.0 / Math.sqrt(sum_squares / dim.to_f64 + eps.to_f64)
      inv_rms[batch] = scale_f64.to_f32
      dim.times do |index|
        output_data[offset + index] =
          (input_read[offset + index].to_f64 * scale_f64 * weight_read[index].to_f64).to_f32
      end
    end

    RMSNormResult.new(
      output,
      inv_rms,
      read_stats(input_read, weight_read)
    )
  end

  def self.validate_layer_norm_inputs!(
    input : ML::Tensor,
    weight : ML::Tensor,
    bias : ML::Tensor,
    normalized_shape : Array(Int32),
    eps : Float32,
  ) : Int32
    validate_eps!(eps)
    norm_size = validate_normalized_shape!(input, normalized_shape)
    validate_affine_size!("weight", weight, norm_size)
    validate_affine_size!("bias", bias, norm_size)
    norm_size
  end

  def self.validate_rms_norm_inputs!(
    input : ML::Tensor,
    weight : ML::Tensor,
    dim : Int32,
    eps : Float32,
  ) : Nil
    validate_eps!(eps)
    raise ArgumentError.new("RMSNorm dimension must be positive") unless dim > 0
    unless input.shape[-1] == dim
      raise ArgumentError.new(
        "RMSNorm trailing dimension #{input.shape[-1]} does not match #{dim}"
      )
    end
    validate_affine_size!("weight", weight, dim)
  end

  private def self.validate_eps!(eps : Float32) : Nil
    unless eps.finite? && eps > 0.0_f32
      raise ArgumentError.new("eps must be finite and positive")
    end
  end

  private def self.validate_normalized_shape!(
    input : ML::Tensor,
    normalized_shape : Array(Int32),
  ) : Int32
    if normalized_shape.empty?
      raise ArgumentError.new("normalized_shape must not be empty")
    end
    unless normalized_shape.all? { |dimension| dimension > 0 }
      raise ArgumentError.new("normalized_shape dimensions must be positive")
    end
    if normalized_shape.size > input.ndim
      raise ArgumentError.new(
        "LayerNorm trailing shape rank #{normalized_shape.size} exceeds input rank #{input.ndim}"
      )
    end

    normalized_shape.each_with_index do |expected, index|
      actual = input.shape[input.ndim - normalized_shape.size + index]
      unless actual == expected
        raise ArgumentError.new(
          "LayerNorm trailing shape #{input.shape.to_a} does not end with #{normalized_shape}"
        )
      end
    end

    ML::Shape.new(normalized_shape).numel
  end

  private def self.validate_affine_size!(name : String, tensor : ML::Tensor, expected : Int32) : Nil
    unless tensor.numel == expected
      raise ArgumentError.new("#{name} size #{tensor.numel} does not match #{expected}")
    end
  end

  private def self.read_stats(
    input : ML::Tensor::CPUReadView,
    weight : ML::Tensor::CPUReadView,
    bias : ML::Tensor::CPUReadView? = nil,
  ) : NormalizationReadStats
    NormalizationReadStats.new(
      input.materialized_bytes,
      weight.materialized_bytes,
      bias.try(&.materialized_bytes) || 0_i64,
      input.materialized?,
      weight.materialized?,
      bias.try(&.materialized?) || false
    )
  end
end
