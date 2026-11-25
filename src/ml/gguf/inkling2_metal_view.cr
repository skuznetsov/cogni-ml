require "../core/buffer"
require "../metal/dispatch"

module ML::GGUF
  record Inkling2View, buffer : ML::MetalBuffer, offset : Int64 = 0_i64 do
    def slice(elements : Int) : self
      self.class.new(buffer, offset + elements.to_i64 * sizeof(Float32))
    end
  end

  # Live views stay bound to their original buffer while a command is encoded.
  # Rewind only after encoding the last read of all younger allocations.
  class Inkling2ResidentScratch
    private PAGE_BYTES          = 4_i64 * 1024 * 1024
    private FAST_BYTES_PER_ROW  = 241_i64 * 1024
    private GROWTH_ALIGNMENT    = 2_i64 * 1024 * 1024
    private MAX_RETAINED_BYTES  = 512_i64 * 1024 * 1024
    private MAX_TRANSIENT_BYTES = 1024_i64 * 1024 * 1024
    private ALIGNMENT           = 256_i64

    @buffers = [] of ML::MetalBuffer
    @page = 0
    @offset = 0_i64
    @cursor = 0_i64
    @command_peak = 0_i64
    @growable : Bool

    def initialize(@growable : Bool = false)
    end

    def reserved_bytes : Int64
      @buffers.sum(0_i64, &.size)
    end

    # Each command is synchronously completed before the next one starts, so
    # replacing this arena here cannot invalidate an encoded Metal view.
    def prepare_command(rows : Int32) : Nil
      reset
      return unless @growable

      target = Math.max(rows.to_i64 * FAST_BYTES_PER_ROW, @command_peak)
      return if target == 0

      target = (target + GROWTH_ALIGNMENT - 1) & ~(GROWTH_ALIGNMENT - 1)
      raise ArgumentError.new("Inkling 2 fast-prefill scratch request exceeds #{MAX_RETAINED_BYTES} bytes") if target > MAX_RETAINED_BYTES

      first = @buffers.first?
      if first && first.size >= target
        while @buffers.size > 1
          @buffers.pop.release
        end
      else
        old_bytes = reserved_bytes
        if old_bytes + target > MAX_TRANSIENT_BYTES
          raise ArgumentError.new("Inkling 2 fast-prefill scratch growth exceeds #{MAX_TRANSIENT_BYTES} transient bytes")
        end

        # Allocate before replacement so allocation failure leaves the prior
        # arena intact for the caller's required reset and replay.
        replacement = ML::MetalBuffer.new(target, ML::StorageMode::Private)
        old = @buffers
        @buffers = [replacement]
        old.each(&.release)
      end
      @command_peak = 0_i64
    end

    def take(elements : Int) : Inkling2View
      bytes = elements.to_i64 * sizeof(Float32)
      needed = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
      raise ArgumentError.new("Inkling 2 scratch allocation must be positive") unless elements > 0

      loop do
        page = @buffers[@page]? || begin
          page_bytes = Math.max(PAGE_BYTES, needed)
          reserved = reserved_bytes
          if @growable && reserved + page_bytes > MAX_RETAINED_BYTES
            raise ArgumentError.new("Inkling 2 fast-prefill scratch exceeds #{MAX_RETAINED_BYTES} retained bytes")
          end
          buffer = ML::MetalBuffer.new(page_bytes, ML::StorageMode::Private)
          @buffers << buffer
          buffer
        end
        if @offset + needed <= page.size
          view = Inkling2View.new(page, @offset)
          @offset += needed
          @cursor += needed
          @command_peak = Math.max(@command_peak, @cursor) if @growable
          return view
        end
        @page += 1
        @offset = 0_i64
      end
    end

    def mark : {Int32, Int64, Int64}
      {@page, @offset, @cursor}
    end

    def rewind(mark : {Int32, Int64, Int64}) : Nil
      @page, @offset, @cursor = mark
    end

    def reset : Nil
      rewind({0, 0_i64, 0_i64})
    end

    def close : Nil
      @buffers.each(&.release)
      @buffers.clear
      reset
    end
  end
end
