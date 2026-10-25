require "./qwen25_hy_nomas"

module ML::GGUF
  # Hy-Nomas crops the system/header tokens only after its selected decoder
  # output. Both outputs stay contiguous and resident on the caller's GPU
  # chain; tokenization, byte admission and buffer preparation precede encode.
  class Qwen25VLConditioningPlan
    record NamedBytes, raw_mask : Int64, hidden : Int64, mask : Int64, total : Int64

    @@retained = {} of UInt64 => Qwen25VLConditioningPlan

    getter raw_tokens : Int32
    getter crop_start : Int32
    getter token_count : Int32
    getter named_bytes : NamedBytes

    @hidden : MetalBuffer? = nil
    @mask : MetalBuffer? = nil
    @chain : Metal::MPSCommandChain? = nil
    @released = false

    def self.resident_bytes(config : Qwen25VLConfig, raw_tokens : Int32, crop_start : Int32) : NamedBytes
      raise ArgumentError.new("conditioning raw tokens outside configured context") unless 0 < raw_tokens <= config.max_positions
      raise ArgumentError.new("conditioning crop must leave a positive suffix") unless 0 < crop_start < raw_tokens
      raw_hidden_bytes = raw_tokens.to_i64 * config.dim * 2
      hidden = (raw_tokens - crop_start).to_i64 * config.dim * 2
      mask = (raw_tokens - crop_start).to_i64 * sizeof(Int32)
      raw_mask = raw_tokens.to_i64 * sizeof(Int32)
      # The existing exact-byte blit ABI accepts signed 32-bit region sizes.
      raise ArgumentError.new("conditioning exceeds native copy range") unless raw_hidden_bytes <= Int32::MAX && raw_mask <= Int32::MAX
      NamedBytes.new(raw_mask, hidden, mask, raw_mask + hidden + mask)
    end

    def self.validate_budget!(bytes : NamedBytes, max_named_bytes : Int64) : Nil
      raise ArgumentError.new("conditioning exceeds admitted named byte budget") unless max_named_bytes >= bytes.total
    end

    def self.validate_extents!(config : Qwen25VLConfig, raw_tokens : Int32, crop_start : Int32,
                               hidden_bytes : Int64, mask_bytes : Int64) : Nil
      bytes = resident_bytes(config, raw_tokens, crop_start)
      raise ArgumentError.new("raw hidden extent mismatch") unless hidden_bytes == raw_tokens.to_i64 * config.dim * 2
      raise ArgumentError.new("raw mask extent mismatch") unless mask_bytes == bytes.raw_mask
    end

    def self.validate_liveness!(source_valid : Bool, raw_mask_valid : Bool,
                                hidden_valid : Bool, mask_valid : Bool) : Nil
      raise ArgumentError.new("conditioning source or destination was released") unless source_valid && raw_mask_valid && hidden_valid && mask_valid
    end

    # Sources are borrowed. The caller must encode the complete prefix on the
    # SAME chain before this step, and keep both sources valid and unchanged
    # until terminal drain. Buffer handles do not certify producer-chain identity.
    # The admission includes the caller's uploaded raw mask as well as owned
    # destinations, so the probe can admit the whole request before allocation.
    def initialize(@config : Qwen25VLConfig, @raw_tokens : Int32, @crop_start : Int32,
                   @source : MetalBuffer, @raw_mask : MetalBuffer, max_named_bytes : Int64)
      @token_count = @raw_tokens - @crop_start
      @named_bytes = self.class.resident_bytes(@config, @raw_tokens, @crop_start)
      self.class.validate_budget!(@named_bytes, max_named_bytes)
      self.class.validate_extents!(@config, @raw_tokens, @crop_start, @source.size, @raw_mask.size)
      raise ArgumentError.new("conditioning sources must be valid shared buffers") unless @source.valid? && @raw_mask.valid? && @source.storage_mode.shared? && @raw_mask.storage_mode.shared?
      raise ArgumentError.new("conditioning sources must be distinct buffers") if @source.handle == @raw_mask.handle
      begin
        @hidden = MetalBuffer.new(@named_bytes.hidden)
        @mask = MetalBuffer.new(@named_bytes.mask)
        @@retained[object_id] = self
      rescue error
        @mask.try(&.release)
        @hidden.try(&.release)
        raise error
      end
    end

    # Borrowed output handles. Callers must not release or modify either
    # destination while its producer/consumers are in flight.
    def hidden : MetalBuffer
      check_live!
      @hidden.not_nil!
    end

    def mask : MetalBuffer
      check_live!
      @mask.not_nil!
    end

    # Append two exact-byte copies to the existing root; no commit, device
    # wait, activation read, new command buffer or MetalBuffer allocation.
    def encode(chain : Metal::MPSCommandChain) : Nil
      check_live!
      raise ArgumentError.new("cannot encode conditioning on a completed chain") if chain.completed?
      if previous = @chain
        raise ArgumentError.new("conditioning plan is already in flight") unless previous.completed?
      end
      self.class.validate_liveness!(@source.valid?, @raw_mask.valid?, hidden.valid?, mask.valid?)
      @chain = chain
      chain.pause!
      {% unless flag?(:cpu_only) %}
        encoder = Pointer(Void).null
        begin
          encoder = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
          raise "Failed to create conditioning blit encoder" if encoder.null?
          MetalDispatchFFI.blit_copy_buffer(encoder, @source.handle,
            @crop_start.to_i64 * @config.dim * 2, hidden.handle, 0_i64, @named_bytes.hidden)
          MetalDispatchFFI.blit_copy_buffer(encoder, @raw_mask.handle,
            @crop_start.to_i64 * sizeof(Int32), mask.handle, 0_i64, @named_bytes.mask)
        ensure
          MetalDispatchFFI.blit_end_encoding(encoder) unless encoder.null?
          chain.resume!
        end
      {% end %}
    end

    def release : Nil
      return if @released
      if chain = @chain
        raise ArgumentError.new("terminal wait required before conditioning release") unless chain.completed?
      end
      @mask.try(&.release)
      @hidden.try(&.release)
      @released = true
      @@retained.delete(object_id)
    end

    private def check_live! : Nil
      raise ArgumentError.new("conditioning plan was released") if @released
    end
  end
end
