require "./driver"
require "../gguf/compute"

module ML::CUDA
  # Device-only projection adapter over the Qwen3.5 packed kernels. All weights
  # and kernel arguments are owned for the adapter lifetime; enqueue only
  # dispatches work. It neither dequantizes weights nor reads activations back.
  class QwenImage21Projection
    Q8_PTX    = {{ read_file("src/ml/cuda/kernels/q8_0_gemv.ptx") }}
    Q5_PTX    = {{ read_file("src/ml/cuda/kernels/q5k_gemv_probe.ptx") }}
    Q6_PTX    = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}
    DENSE_PTX = {{ read_file("src/ml/cuda/kernels/qwen_image21_dense.ptx") }}

    getter in_dim : Int32
    getter out_dim : Int32

    @module : CUDAModule
    @function : KernelFunction

    def self.validate_weight!(weight : ML::GGUF::QuantWeight) : Nil
      raise ArgumentError.new("projection dimensions must be positive") unless weight.in_dim > 0 && weight.out_dim > 0
      type = weight.type
      unless type.f32? || type.bf16? || type.q8_0? || type.q5_k? || type.q6_k?
        raise ArgumentError.new("unsupported image CUDA projection type #{type}")
      end
      raise ArgumentError.new("projection input is not quant-block aligned") unless weight.in_dim % type.block_elements == 0
      expected = weight.out_dim.to_i64 * (weight.in_dim // type.block_elements) * type.block_bytes
      raise ArgumentError.new("projection weight byte count mismatch") unless weight.raw.size.to_i64 == expected
    end

    def initialize(weight : ML::GGUF::QuantWeight)
      self.class.validate_weight!(weight)
      @in_dim = weight.in_dim
      @out_dim = weight.out_dim
      @type = weight.type
      @closed = false
      @params = Pointer(Void*).malloc(6)
      @weights = DeviceBuffer.new(weight.raw.size.to_u64)
      @module = begin
        ML::CUDA.copy_htod!(@weights.ptr, weight.raw.to_unsafe.as(Void*), weight.raw.size.to_u64, "image projection weights")
        ptx = case @type
              when .q8_0? then Q8_PTX
              when .q5_k? then Q5_PTX
              when .q6_k? then Q6_PTX
              else             DENSE_PTX
              end
        CUDAModule.load(ptx, "qwen_image21_projection")
      rescue ex
        @weights.close
        raise ex
      end
      @function = begin
        name = case @type
               when .q8_0? then "q8_0_gemv_warp4_f32"
               when .q5_k? then "q5_k_gemv_warp4_f32_batched"
               when .q6_k? then "q6_k_gemv_warp4_f32_batched"
               else             "qi21_dense_gemv"
               end
        @module.function(name)
      rescue ex
        @module.close
        @weights.close
        raise ex
      end
    end

    # Pointers must cover [tokens, in_dim] and [tokens, out_dim] F32 values.
    # The caller keeps them alive until its terminal completion boundary.
    def enqueue(input_ptr : DevicePtr, output_ptr : DevicePtr, tokens : Int32) : Nil
      raise "image CUDA projection is closed" if @closed
      raise ArgumentError.new("tokens must be positive") unless tokens > 0
      raise ArgumentError.new("null projection device pointer") if input_ptr == 0 || output_ptr == 0
      input_bytes = tokens.to_u64 * @in_dim.to_u64 * 4
      output_bytes = tokens.to_u64 * @out_dim.to_u64 * 4
      if input_ptr < output_ptr + output_bytes && output_ptr < input_ptr + input_bytes
        raise ArgumentError.new("projection input and output must not overlap")
      end
      weight_ptr = @weights.ptr
      in_dim = @in_dim.to_u32
      out_dim = @out_dim.to_u32
      element_bytes = (@type.bf16? ? 2_u32 : 4_u32)
      @params[0] = pointerof(weight_ptr).as(Void*)
      @params[1] = pointerof(input_ptr).as(Void*)
      @params[2] = pointerof(output_ptr).as(Void*)
      @params[3] = pointerof(in_dim).as(Void*)
      @params[4] = pointerof(out_dim).as(Void*)
      @params[5] = pointerof(element_bytes).as(Void*)
      if @type.q8_0?
        # Reuse the original Q8 arithmetic verbatim; each token points directly
        # at its resident input/output row. No host staging or activation copy.
        tokens.times do
          ML::CUDA.launch!(@function, ((out_dim + 3) // 4), 1_u32, 1_u32,
            128_u32, 1_u32, 1_u32, @params, "image Q8_0 projection")
          input_ptr += @in_dim.to_u64 * 4
          output_ptr += @out_dim.to_u64 * 4
        end
      elsif @type.q5_k? || @type.q6_k?
        blocks = ((out_dim.to_u64 + 3) // 4) * tokens.to_u64
        raise ArgumentError.new("projection launch grid overflow") if blocks > UInt32::MAX
        ML::CUDA.launch!(@function, blocks.to_u32, 1_u32, 1_u32,
          128_u32, 1_u32, 1_u32, @params, "image packed batch projection")
      else
        raise ArgumentError.new("dense token grid exceeds CUDA limit") if tokens > 65535
        ML::CUDA.launch!(@function, ((out_dim + 127) // 128), tokens.to_u32, 1_u32,
          128_u32, 1_u32, 1_u32, @params, "image dense projection")
      end
    end

    # The caller must complete in-flight work before closing.
    def close : Nil
      return if @closed
      @module.close
      @weights.close
      @closed = true
    end
  end
end
