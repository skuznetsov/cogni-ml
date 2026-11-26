require "../../src/ml/core/buffer"
require "../../src/ml/metal/device"
require "../../src/ml/video/bf16_weight_cache"
require "../../src/ml/video/metal_mps_command_chain"

# Diagnostic admission of a saved DCAE result. A caller-supplied digest binds
# bytes, not encoder provenance: callers must separately retain the encoder
# comparison/receipt if they make an upstream-equivalence claim.
module HyNomasReferenceLatent
  SHAPE      = [1_i64, 32_i64, 30_i64, 9_i64, 16_i64]
  ELEMENTS   = 138_240
  BF16_BYTES = ELEMENTS.to_i64 * 2
  F32_BYTES  = ELEMENTS.to_i64 * 4

  def self.validate_record!(record : ML::Video::CheckpointTensor) : Nil
    unless record.name == "latent" && record.dtype.bf16? &&
           record.shape == SHAPE && record.n_elements == ELEMENTS &&
           record.data_bytes == BF16_BYTES
      raise ArgumentError.new("saved reference requires BF16 latent [1,32,30,9,16]")
    end
  end

  class Input
    getter path : String
    getter sha256 : String
    @file : ML::Video::CheckpointFile
    @record : ML::Video::CheckpointTensor
    @closed = false

    def initialize(path : String, @sha256 : String)
      # Preserve the supplied leaf so authentication can reject symlinks.
      @path = File.expand_path(path)
      @file = ML::Video::BF16WeightCache.authenticate(@path, @sha256)
      @record = begin
        record = @file.tensors["latent"]? || raise ArgumentError.new("saved reference has no latent tensor")
        HyNomasReferenceLatent.validate_record!(record)
        record
      rescue ex
        @file.close
        raise ex
      end
    end

    # Setup disk-to-shared-buffer copy only; no CPU conversion or GPU readback.
    # The returned allocation is caller-owned and must survive the GPU drain.
    def upload_bf16! : ML::MetalBuffer
      raise ArgumentError.new("saved reference input is closed") if @closed
      buffer = ML::MetalBuffer.new(BF16_BYTES)
      begin
        @file.copy_raw(@record, Bytes.new(buffer.contents.as(UInt8*), BF16_BYTES.to_i32))
        buffer
      rescue ex
        buffer.release
        raise ex
      end
    end

    def close : Nil
      return if @closed
      @file.close
      @closed = true
    end
  end

  # Reuse the engine's exact BF16 byte expansion. Both owners are retained by
  # this plan; the caller must not release them before its terminal drain.
  class ExpansionPlan
    @source : ML::MetalBuffer
    @destination : ML::MetalBuffer
    @pipeline : ML::Metal::ComputePipeline

    def initialize(@source : ML::MetalBuffer, @destination : ML::MetalBuffer)
      validate_buffers!
      @pipeline = ML::Metal::PipelineCache.get("video_weight_decode") do
        ML::Metal::ComputePipeline.new("video_weight_decode", ML::Video::PreparedWeights::SOURCE)
      end
    end

    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      validate_buffers!
      chain.encoder.set_pipeline(@pipeline)
        .set_buffer(@source, 0)
        .set_buffer(@destination, 1, ML::Metal::BufferAccess::Write)
        .set_value(ELEMENTS.to_u32, 2)
        .set_value(1_u32, 3)
        .dispatch_1d(ELEMENTS)
    end

    private def validate_buffers! : Nil
      unless @source.valid? && @source.size == BF16_BYTES && @source.storage_mode.shared? &&
             @destination.valid? && @destination.size == F32_BYTES && @destination.storage_mode.shared?
        raise ArgumentError.new("reference expansion requires exact shared BF16/F32 buffers")
      end
      ML::Video::MetalOps.distinct(@source, @destination)
    end
  end
end
