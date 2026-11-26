# Opt-in F32-storage helpers that reproduce source-BF16 tensor boundaries for
# Hy-Nomas preattention diagnostics. Each encode only validates and dispatches.
require "./metal_ops"

module ML::Video::HyNomasReferenceBF16
  ROUND_KERNEL      = "hy_nomas_reference_bf16_round"
  MODULATE_KERNEL   = "hy_nomas_reference_bf16_modulate"
  RMS_FINISH_KERNEL = "hy_nomas_reference_bf16_rms_finish"
  SOURCE            = {{ read_file("#{__DIR__}/../metal/kernels/hy_nomas/hy_nomas_block.metal") }} + "\n" +
                      {{ read_file("#{__DIR__}/../metal/kernels/hy_nomas/hy_nomas_reference_bf16.metal") }}

  # Prepared pipelines are reusable across fixed-width calls. Buffers remain
  # caller-owned and must live through the command buffer's terminal wait.
  class Plan
    getter dim : Int32

    @round_pipeline : ML::Metal::ComputePipeline
    @modulate_pipeline : ML::Metal::ComputePipeline
    @rms_finish_pipeline : ML::Metal::ComputePipeline

    # Validate dynamic row geometry before dispatch. This remains callable
    # without Metal for CPU-only shape-contract checks.
    def self.checked_count(rows : Int32, dim : Int32,
                           prepared_dim : Int32) : Int32
      MetalOps.dims(rows, dim)
      unless dim == prepared_dim
        raise ArgumentError.new("Hy-Nomas reference BF16 feature width changed")
      end
      count = rows.to_i64 * dim
      unless count <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas reference BF16 tensor exceeds the Metal count limit")
      end
      count.to_i32
    end

    def initialize(@dim : Int32)
      raise ArgumentError.new("Hy-Nomas reference BF16 width must be positive") unless @dim > 0

      @round_pipeline = ML::Metal::PipelineCache.get(ROUND_KERNEL) do
        ML::Metal::ComputePipeline.new(ROUND_KERNEL, SOURCE)
      end
      @modulate_pipeline = ML::Metal::PipelineCache.get(MODULATE_KERNEL) do
        ML::Metal::ComputePipeline.new(MODULATE_KERNEL, SOURCE)
      end
      @rms_finish_pipeline = ML::Metal::PipelineCache.get(RMS_FINISH_KERNEL) do
        ML::Metal::ComputePipeline.new(RMS_FINISH_KERNEL, SOURCE)
      end
    end

    # Round F32 values through BF16 while retaining F32 storage. Exact in-place
    # operation and disjoint buffers are supported; partial overlap is rejected.
    def round(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
              output : ML::MetalBuffer, count : Int32) : Nil
      MetalOps.dims(count)
      validate_f32!(input, count.to_i64, "round input")
      validate_f32!(output, count.to_i64, "round output")
      MetalOps.inplace_or_distinct(input, output)

      e.set_pipeline(@round_pipeline)
        .set_buffer(input, 0)
        .set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 2)
        .dispatch_1d(count)
    end

    # Match source modulation: BF16-round (1 + scale), then do a separately
    # rounded F32 product and addition. Input and output are F32 storage.
    def modulate(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                 shift : ML::MetalBuffer, scale : ML::MetalBuffer,
                 output : ML::MetalBuffer, rows : Int32, dim : Int32) : Nil
      count = checked_count(rows, dim)
      validate_f32!(input, count.to_i64, "modulation input")
      validate_f32!(shift, count.to_i64, "modulation shift")
      validate_f32!(scale, count.to_i64, "modulation scale")
      validate_f32!(output, count.to_i64, "modulation output")
      MetalOps.inplace_or_distinct(input, output)
      MetalOps.distinct(output, shift, scale)

      e.set_pipeline(@modulate_pipeline)
        .set_buffer(input, 0)
        .set_buffer(shift, 1)
        .set_buffer(scale, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        .set_value(count.to_u32, 4)
        .dispatch_1d(count)
    end

    # Normalize from caller-provided F32 row means, applying source-BF16
    # boundaries after normalization and after the BF16 gain product.
    def rms_finish(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
                   weight : ML::MetalBuffer, mean : ML::MetalBuffer,
                   output : ML::MetalBuffer, rows : Int32, head_dim : Int32,
                   eps : Float32) : Nil
      count = checked_count(rows, head_dim)
      unless eps.finite? && eps > 0.0_f32
        raise ArgumentError.new("Hy-Nomas reference BF16 RMS epsilon must be finite and positive")
      end
      validate_f32!(input, count.to_i64, "RMS input")
      validate_bf16!(weight, head_dim.to_i64, "RMS gain")
      validate_f32!(mean, rows.to_i64, "RMS mean")
      validate_f32!(output, count.to_i64, "RMS output")
      MetalOps.inplace_or_distinct(input, output)
      MetalOps.distinct(weight, mean, input, output)
      MetalOps.distinct(mean, input, output)

      desired_threads = Math.min(256_i64, ((head_dim.to_i64 + 31) // 32) * 32)
      threads = Math.min(desired_threads, Int32::MAX.to_i64 // rows).to_i32
      grid_threads = (rows.to_i64 * threads).to_i32
      e.set_pipeline(@rms_finish_pipeline)
        .set_buffer(input, 0)
        .set_buffer(weight, 1)
        .set_buffer(mean, 2)
        .set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        .set_value(rows.to_u32, 4)
        .set_value(head_dim.to_u32, 5)
        .set_value(eps, 6)
        .dispatch({grid_threads, 1, 1}, {threads, 1, 1})
    end

    private def checked_count(rows : Int32, dim : Int32) : Int32
      self.class.checked_count(rows, dim, @dim)
    end

    private def validate_f32!(buffer : ML::MetalBuffer, count : Int64,
                              label : String) : Nil
      expected_bytes = count * sizeof(Float32)
      unless buffer.valid? && buffer.size == expected_bytes && buffer.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas reference BF16 #{label} must be exact-size shared F32")
      end
      MetalOps.check(buffer, count)
    end

    private def validate_bf16!(buffer : ML::MetalBuffer, count : Int64,
                               label : String) : Nil
      expected_bytes = count * 2
      unless buffer.valid? && buffer.size == expected_bytes && buffer.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas reference BF16 #{label} must be exact-size shared BF16")
      end
      MetalOps.check_storage(buffer, count, WeightDType::BF16)
    end
  end
end
