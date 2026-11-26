# Opt-in prepared MPSGraph mean path for Hy-Nomas Q/K RMS normalization.
# The flattened rows preserve the frozen [B,T,H,Dh] inputs as [B,T*H,Dh].
require "./hy_nomas_block"
require "./hy_nomas_reference_bf16"
require "./wan_t5"

module ML::Video::HyNomasMPSRMS
  # Fixed-input plan with a default fixed weight. The caller owns and must
  # retain this plan through the terminal wait; release is permitted only after
  # that chain has drained. This wrapper owns only its square/mean scratch
  # buffers.
  class Plan
    getter tokens : Int32
    getter rows : Int32
    getter head_dim : Int32

    @config : HyNomasConfig
    @input : ML::MetalBuffer
    @weight : ML::MetalBuffer
    @output : ML::MetalBuffer
    @weight_dtype : WeightDType
    @rows : Int32
    @count : Int32
    @reduction_rows : Int32
    @reduction_count : Int32
    @arena : Arena
    @square : ML::MetalBuffer? = nil
    @mean : ML::MetalBuffer? = nil
    @mean_plan : WanT5RMSMeanPlan? = nil
    @mps_config : WanT5Config? = nil
    @reference_bf16_plan : HyNomasReferenceBF16::Plan? = nil
    @source_bf16 : Bool
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    # Keep reduction geometry selection independently testable without Metal.
    # The source-BF16 finish consumes one mean per active row, so padded
    # reduction tokens would include square values that this plan never writes.
    def self.checked_reduction_tokens(tokens : Int32,
                                      reduction_tokens : Int32?,
                                      source_bf16 : Bool) : Int32
      reduction = reduction_tokens || tokens
      unless reduction >= tokens
        raise ArgumentError.new("Hy-Nomas MPS RMS reduction tokens must cover the active tokens")
      end
      if source_bf16 && reduction != tokens
        raise ArgumentError.new("Source-BF16 Hy-Nomas RMS requires reduction tokens to match active tokens")
      end
      reduction
    end

    def initialize(@config : HyNomasConfig,
                   @input : ML::MetalBuffer,
                   @weight : ML::MetalBuffer,
                   @output : ML::MetalBuffer,
                   @weight_dtype : WeightDType,
                   @tokens : Int32,
                   *, reduction_tokens : Int32? = nil,
                   source_bf16 : Bool = false)
      @config.validate!
      @source_bf16 = source_bf16
      if @source_bf16 && !@weight_dtype.bf16?
        raise ArgumentError.new("Source-BF16 Hy-Nomas RMS requires a BF16 gain")
      end
      unless @tokens == @config.video_tokens || @tokens == @config.text_tokens
        raise ArgumentError.new("Hy-Nomas MPS RMS token count must match an image or text stream")
      end
      reduction = self.class.checked_reduction_tokens(@tokens, reduction_tokens, @source_bf16)
      @head_dim = @config.head_dim
      @rows = HyNomasDualStreamBlock.checked_count(
        @config.batches, @tokens, @config.heads,
      )
      @count = HyNomasDualStreamBlock.checked_count(@rows, @head_dim)
      @reduction_rows = HyNomasDualStreamBlock.checked_count(
        @config.batches, reduction, @config.heads,
      )
      @reduction_count = HyNomasDualStreamBlock.checked_count(@reduction_rows, @head_dim)

      # Warm both cached dispatch pipelines before any command is encoded.
      WanT5.pipeline("wan_t5_rms_square")
      WanT5.pipeline("wan_t5_rms_from_mean")
      if @source_bf16
        @reference_bf16_plan = HyNomasReferenceBF16::Plan.new(@head_dim)
      end
      validate_tensor!(@input, @count, "input")
      validate_tensor!(@output, @count, "output")
      validate_weight!
      validate_distinct!(@input, @weight, @output)

      @arena = Arena.new
      begin
        @square = @arena.alloc("hy_nomas_mps_rms.square", @reduction_count)
        @mean = @arena.alloc("hy_nomas_mps_rms.mean", @reduction_rows)
        validate_bindings!

        flat_tokens = HyNomasDualStreamBlock.checked_count(reduction, @config.heads)
        mps_config = WanT5Config.new(
          vocab: 1, dim: @head_dim, ffn_dim: @head_dim,
          heads: 1, layers: 1, batches: @config.batches,
          tokens: flat_tokens, eps: @config.norm_eps,
        )
        mps_config.validate!
        unless mps_config.rows == @reduction_rows && mps_config.dim == @head_dim &&
               mps_config.eps == @config.norm_eps
          raise ArgumentError.new("Hy-Nomas MPS RMS flattened geometry mismatch")
        end
        @mps_config = mps_config
        @mean_plan = WanT5RMSMeanPlan.new(mps_config, square, mean)
      rescue ex
        @mean_plan.try(&.release)
        @arena.release
        @released = true
        raise ex
      end
    end

    # Setup-owned F32 square and row-mean buffers; external tensors are not
    # counted here. No additional storage is allocated by encode.
    def resident_bytes : Int64
      @arena.resident_bytes
    end

    # Append the proven T5 MPS mean operation for this plan's fixed tensors and
    # default weight. The native helper pauses/resumes around MPSGraph and
    # refreshes the encoder itself.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      encode(chain, @weight, @weight_dtype)
    end

    # Append the same operation with a per-call norm weight. Only the weight is
    # dynamic; the prepared MPSGraph remains bound to this plan's input/output
    # and owned square/mean buffers.
    def encode(chain : ML::Metal::MPSCommandChain,
               weight : ML::MetalBuffer,
               weight_dtype : WeightDType) : Nil
      raise ArgumentError.new("Hy-Nomas MPS RMS plan has been released") if @released
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas MPS RMS plan cannot span in-flight command chains")
        end
      end
      validate_bindings!(weight, weight_dtype)
      @last_chain = chain
      if @source_bf16
        encode_source_bf16(chain, weight)
      else
        WanT5.rms_norm_mps_mean(
          chain, mean_plan, @mps_config.not_nil!, @input, weight, @output,
          square, mean, weight_dtype, @rows,
        )
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas MPS RMS command chain before release") unless chain.completed?
      end
      @mean_plan.try(&.release)
      @mean_plan = nil
      @arena.release
      @square = nil
      @mean = nil
      @released = true
    end

    private def mean_plan : WanT5RMSMeanPlan
      @mean_plan || raise ArgumentError.new("Hy-Nomas MPS RMS mean plan is unavailable")
    end

    private def square : ML::MetalBuffer
      @square || raise ArgumentError.new("Hy-Nomas MPS RMS square scratch is unavailable")
    end

    private def mean : ML::MetalBuffer
      @mean || raise ArgumentError.new("Hy-Nomas MPS RMS mean scratch is unavailable")
    end

    # Keep the existing square dispatch and prepared MPSGraph mean, then
    # replace only the terminal F32 affine kernel with source-BF16 boundaries.
    # In this mode callers provide an input already rounded through BF16 and
    # expanded into the public F32 storage.
    private def encode_source_bf16(chain : ML::Metal::MPSCommandChain,
                                   weight : ML::MetalBuffer) : Nil
      encoder = chain.encoder
      encoder.set_pipeline(WanT5.pipeline("wan_t5_rms_square"))
      encoder.set_buffer(@input, 0).set_buffer(square, 1, ML::Metal::BufferAccess::Write)
      encoder.set_value(@count.to_u32, 2).dispatch_1d(@count)

      chain.pause!
      begin
        mean_plan.encode(chain.handle)
      ensure
        chain.resume!
      end
      reference_bf16_plan.rms_finish(chain.encoder, @input, weight, mean, @output,
        @rows, @head_dim, @config.norm_eps)
    end

    private def reference_bf16_plan : HyNomasReferenceBF16::Plan
      @reference_bf16_plan || raise ArgumentError.new("Hy-Nomas source-BF16 RMS plan is unavailable")
    end

    private def validate_bindings! : Nil
      validate_bindings!(@weight, @weight_dtype)
    end

    private def validate_bindings!(weight : ML::MetalBuffer,
                                   weight_dtype : WeightDType) : Nil
      validate_tensor!(@input, @count, "input")
      validate_tensor!(@output, @count, "output")
      validate_weight!(weight, weight_dtype)
      validate_scratch!(square, mean)
      validate_distinct!(@input, weight, @output, square, mean)
    end

    private def validate_tensor!(buffer : ML::MetalBuffer, count : Int32,
                                 label : String) : Nil
      expected_bytes = count.to_i64 * sizeof(Float32)
      unless buffer.valid? && buffer.size == expected_bytes &&
             buffer.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas MPS RMS #{label} must be exact-size shared F32")
      end
      MetalOps.check(buffer, count.to_i64)
    end

    private def validate_weight! : Nil
      validate_weight!(@weight, @weight_dtype)
    end

    private def validate_weight!(weight : ML::MetalBuffer,
                                 weight_dtype : WeightDType) : Nil
      if @source_bf16 && !weight_dtype.bf16?
        raise ArgumentError.new("Source-BF16 Hy-Nomas RMS requires a BF16 gain")
      end
      expected_bytes = @head_dim.to_i64 * weight_dtype.bytes_per_element
      unless weight.valid? && weight.size == expected_bytes &&
             weight.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas MPS RMS weight must be exact-size shared #{weight_dtype}")
      end
      MetalOps.check_storage(weight, @head_dim.to_i64, weight_dtype)
    end

    private def validate_scratch!(square : ML::MetalBuffer,
                                  mean : ML::MetalBuffer) : Nil
      validate_tensor!(square, @reduction_count, "square scratch")
      validate_tensor!(mean, @reduction_rows, "mean scratch")
    end

    private def validate_distinct!(input : ML::MetalBuffer,
                                   weight : ML::MetalBuffer,
                                   output : ML::MetalBuffer) : Nil
      validate_distinct_pair!(input, weight)
      validate_distinct_pair!(input, output)
      validate_distinct_pair!(weight, output)
    end

    private def validate_distinct!(input : ML::MetalBuffer,
                                   weight : ML::MetalBuffer,
                                   output : ML::MetalBuffer,
                                   square : ML::MetalBuffer,
                                   mean : ML::MetalBuffer) : Nil
      validate_distinct_pair!(input, weight)
      validate_distinct_pair!(input, output)
      validate_distinct_pair!(input, square)
      validate_distinct_pair!(input, mean)
      validate_distinct_pair!(weight, output)
      validate_distinct_pair!(weight, square)
      validate_distinct_pair!(weight, mean)
      validate_distinct_pair!(output, square)
      validate_distinct_pair!(output, mean)
      validate_distinct_pair!(square, mean)
    end

    private def validate_distinct_pair!(left : ML::MetalBuffer,
                                        right : ML::MetalBuffer) : Nil
      if MetalOps.overlap?(left, right)
        raise ArgumentError.new("Hy-Nomas MPS RMS buffers must not overlap")
      end
    end
  end
end
