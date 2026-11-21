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
    private PAGE_BYTES = 4_i64 * 1024 * 1024
    private ALIGNMENT  = 256_i64

    @buffers = [] of ML::MetalBuffer
    @page = 0
    @offset = 0_i64

    def take(elements : Int) : Inkling2View
      bytes = elements.to_i64 * sizeof(Float32)
      needed = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1)
      raise ArgumentError.new("Inkling 2 scratch allocation must be positive") unless elements > 0

      loop do
        page = @buffers[@page]? || begin
          buffer = ML::MetalBuffer.new(Math.max(PAGE_BYTES, needed), ML::StorageMode::Private)
          @buffers << buffer
          buffer
        end
        if @offset + needed <= page.size
          view = Inkling2View.new(page, @offset)
          @offset += needed
          return view
        end
        @page += 1
        @offset = 0_i64
      end
    end

    def mark : {Int32, Int64}
      {@page, @offset}
    end

    def rewind(mark : {Int32, Int64}) : Nil
      @page, @offset = mark
    end

    def reset : Nil
      rewind({0, 0_i64})
    end

    def close : Nil
      @buffers.each(&.release)
      @buffers.clear
      reset
    end
  end
end
