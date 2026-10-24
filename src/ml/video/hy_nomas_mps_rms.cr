# Opt-in prepared MPSGraph mean path for Hy-Nomas Q/K RMS normalization.
# The flattened rows preserve the frozen [B,T,H,Dh] inputs as [B,T*H,Dh].
require "./hy_nomas_block"
require "./wan_t5"

module ML::Video::HyNomasMPSRMS
  # Fixed-input, fixed-weight plan. The caller owns and must retain this plan
  # through the terminal wait; release is permitted only after that chain has
  # drained. This wrapper owns only its square/mean scratch buffers.
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
    @arena : Arena
    @square : ML::MetalBuffer? = nil
    @mean : ML::MetalBuffer? = nil
    @mean_plan : WanT5RMSMeanPlan? = nil
    @mps_config : WanT5Config? = nil
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@config : HyNomasConfig,
                   @input : ML::MetalBuffer,
                   @weight : ML::MetalBuffer,
                   @output : ML::MetalBuffer,
                   @weight_dtype : WeightDType,
                   @tokens : Int32)
      @config.validate!
      unless @tokens == @config.video_tokens || @tokens == @config.text_tokens
        raise ArgumentError.new("Hy-Nomas MPS RMS token count must match an image or text stream")
      end
      @head_dim = @config.head_dim
      @rows = HyNomasDualStreamBlock.checked_count(
        @config.batches, @tokens, @config.heads,
      )
      @count = HyNomasDualStreamBlock.checked_count(@rows, @head_dim)

      # Warm both cached dispatch pipelines before any command is encoded.
      WanT5.pipeline("wan_t5_rms_square")
      WanT5.pipeline("wan_t5_rms_from_mean")
      validate_tensor!(@input, @count, "input")
      validate_tensor!(@output, @count, "output")
      validate_weight!
      validate_distinct!([@input, @weight, @output])

      @arena = Arena.new
      begin
        @square = @arena.alloc("hy_nomas_mps_rms.square", @count)
        @mean = @arena.alloc("hy_nomas_mps_rms.mean", @rows)
        validate_bindings!

        flat_tokens = HyNomasDualStreamBlock.checked_count(@tokens, @config.heads)
        mps_config = WanT5Config.new(
          vocab: 1, dim: @head_dim, ffn_dim: @head_dim,
          heads: 1, layers: 1, batches: @config.batches,
          tokens: flat_tokens, eps: @config.norm_eps,
        )
        mps_config.validate!
        unless mps_config.rows == @rows && mps_config.dim == @head_dim &&
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

    # Append the proven T5 MPS mean operation for the fixed inputs. The native
    # helper pauses/resumes around MPSGraph and refreshes the encoder itself.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas MPS RMS plan has been released") if @released
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas MPS RMS plan cannot span in-flight command chains")
        end
      end
      validate_bindings!
      @last_chain = chain
      WanT5.rms_norm_mps_mean(
        chain, mean_plan, @mps_config.not_nil!, @input, @weight, @output,
        square, mean, @weight_dtype,
      )
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

    private def validate_bindings! : Nil
      validate_tensor!(@input, @count, "input")
      validate_tensor!(@output, @count, "output")
      validate_weight!
      validate_scratch!(square, mean)
      validate_distinct!([@input, @weight, @output, square, mean])
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
      expected_bytes = @head_dim.to_i64 * @weight_dtype.bytes_per_element
      unless @weight.valid? && @weight.size == expected_bytes &&
             @weight.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas MPS RMS weight must be exact-size shared #{@weight_dtype}")
      end
      MetalOps.check_storage(@weight, @head_dim.to_i64, @weight_dtype)
    end

    private def validate_scratch!(square : ML::MetalBuffer,
                                  mean : ML::MetalBuffer) : Nil
      validate_tensor!(square, @count, "square scratch")
      validate_tensor!(mean, @rows, "mean scratch")
    end

    private def validate_distinct!(buffers : Array(ML::MetalBuffer)) : Nil
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Hy-Nomas MPS RMS buffers must not overlap")
          end
        end
      end
    end
  end
end
