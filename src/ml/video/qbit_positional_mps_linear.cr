require "./qbit_positional_weights"
require "./metal_linear_mps"

module ML::Video::QBitPositionalMPSLinear
  # Opt-in packed-weight consumer. A single caller-owned F32 stage can serve
  # different projections serially; no decoded matrix is retained per weight.
  # This does not change checkpoint formats or select a quantization policy.
  @@prepared = false

  def self.prepare! : Nil
    QBitPositionalWeights.prepare!
    MetalLinearMPS.prepare!
    @@prepared = true
  end

  # Initialize status once before the chain and reject its error counters at
  # terminal completion. Never clear prior errors here: a later valid record
  # cannot make an earlier failed operation valid. Decode validates mutable
  # payloads on GPU and poisons invalid records before MPS consumes the stage.
  # Keep all buffers and the fixed-binding plan alive through terminal wait.
  # Concurrent command chains must not share this mutable stage or status.
  def self.project(e : ML::Metal::ComputeEncoder,
                   input : ML::MetalBuffer, offsets : ML::MetalBuffer,
                   blob : ML::MetalBuffer, lut4 : ML::MetalBuffer,
                   lut5 : ML::MetalBuffer, bias : ML::MetalBuffer,
                   output : ML::MetalBuffer, status : ML::MetalBuffer,
                   rows : Int32, input_dim : Int32, output_dim : Int32,
                   tile_values : Int32, bias_dtype : WeightDType,
                   plan : WanT5MPSLinearPlan, weight_stage : ML::MetalBuffer,
                   apply_bias : Bool = true) : Nil
    raise ArgumentError.new("Call QBitPositionalMPSLinear.prepare! before encode") unless @@prepared

    # Complete the composite preflight before decode can modify the stage.
    MetalLinearMPS.validate_preexpanded!(input, bias, output, rows, input_dim,
      output_dim, bias_dtype, plan, weight_stage)
    count = (input_dim.to_i64 * output_dim).to_i32
    QBitPositionalWeights.validate_decode!(offsets, blob, lut4, lut5,
      weight_stage, status, count, tile_values)
    MetalOps.distinct(status, input, bias, output)
    MetalOps.distinct(output, offsets, blob, lut4, lut5)

    QBitPositionalWeights.decode(e, offsets, blob, lut4, lut5,
      weight_stage, status, count, tile_values)
    MetalLinearMPS.project_preexpanded(e, input, bias, output, rows, input_dim,
      output_dim, bias_dtype, plan, weight_stage, apply_bias: apply_bias)
  end
end
