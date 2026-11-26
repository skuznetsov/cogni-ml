# Opt-in, setup-prepared PyTorch 2.8-compatible F32 LayerNorm path for
# Hy-Nomas non-affine block norms. This is a targeted compatibility dispatch,
# not a replacement for general LayerNorm.
# Adapted from PyTorch v2.8.0; copyright and BSD-3-Clause terms are in
# `third_party/licenses/pytorch-v2.8.0-BSD-3-Clause.txt` and summarized in
# `third_party/VIDEO_NOTICES.md`.
require "./metal_ops"

module ML::Video::HyNomasMPSLayerNorm
  KERNEL_NAME = "hy_nomas_pytorch28_layer_norm_f32"
  SOURCE      = MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/hy_nomas_mps_norm.metal") }}

  # Prepared setup-time pipeline and fixed normalization geometry. `encode`
  # only validates borrowed buffers and appends a dispatch to the caller's
  # encoder; buffers must remain alive through that command's terminal wait.
  class Plan
    getter dim : Int32
    getter epsilon : Float32

    @pipeline : ML::Metal::ComputePipeline

    def initialize(@dim : Int32, @epsilon : Float32 = 1e-6_f32)
      unless @dim > 0 && @dim <= 4096
        raise ArgumentError.new("PyTorch 2.8 single-row LayerNorm supports dim in 1..4096")
      end
      unless @epsilon.finite? && @epsilon > 0.0_f32
        raise ArgumentError.new("LayerNorm epsilon must be finite and positive")
      end

      @pipeline = ML::Metal::PipelineCache.get(KERNEL_NAME) do
        ML::Metal::ComputePipeline.new(KERNEL_NAME, SOURCE)
      end
    end

    # Encode contiguous non-affine F32 LayerNorm over `rows` rows. The shape is
    # fixed by this prepared plan; no allocation, wait, or host read occurs.
    def encode(e : ML::Metal::ComputeEncoder, input : ML::MetalBuffer,
               output : ML::MetalBuffer, rows : Int32) : Nil
      raise ArgumentError.new("LayerNorm row count must be positive") unless rows > 0
      count = rows.to_i64 * @dim
      unless count <= Int32::MAX
        raise ArgumentError.new("LayerNorm tensor exceeds the native Metal count limit")
      end
      expected_bytes = count * sizeof(Float32)
      unless input.size == expected_bytes && output.size == expected_bytes
        raise ArgumentError.new("LayerNorm requires exact-size F32 buffers")
      end
      unless input.storage_mode == ML::StorageMode::Shared &&
             output.storage_mode == ML::StorageMode::Shared
        raise ArgumentError.new("LayerNorm requires shared F32 buffers")
      end
      MetalOps.check(input, count)
      MetalOps.check(output, count)
      MetalOps.distinct(output, input)

      threads = Math.min((@dim + 3) // 4, 1024)
      e.set_pipeline(@pipeline)
        .set_buffer(input, 0)
        .set_buffer(output, 1, ML::Metal::BufferAccess::Write)
        .set_value(@dim.to_u32, 2)
        .set_value(@epsilon, 3)
        .dispatch_threadgroups({rows, 1, 1}, {threads, 1, 1})
    end
  end
end
