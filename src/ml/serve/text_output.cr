module ML::Serve
  # A BPE token can end inside a UTF-8 character or a requested stop string.
  # Release only the bytes that are safe to show in this chunk.
  class TextOutput
    @pending = ""
    getter? stopped = false

    def initialize(@stops : Array(String), &@emit : String -> Nil)
    end

    def append(piece : String) : Bool
      return false if @stopped
      return true if piece.empty?

      @pending += piece
      if match = @stops.compact_map { |stop| @pending.byte_index(stop) }.min?
        emit_prefix(match)
        @pending = ""
        @stopped = true
      else
        emit_prefix(utf8_boundary(@pending.bytesize - stop_prefix_size))
      end
      !@stopped
    end

    def finish : Nil
      emit_prefix(utf8_boundary(@pending.bytesize))
      unless @pending.empty?
        @emit.call(Char::REPLACEMENT.to_s)
        @pending = ""
      end
    end

    private def stop_prefix_size : Int32
      bytes = @pending.to_slice
      longest = 0
      @stops.each do |stop|
        prefix = stop.to_slice
        size = Math.min(bytes.size, prefix.size - 1)
        while size > longest
          if bytes[bytes.size - size, size] == prefix[0, size]
            longest = size
            break
          end
          size -= 1
        end
      end
      longest
    end

    private def utf8_boundary(limit : Int32) : Int32
      first = limit - 1
      while first >= 0 && limit - first < 4 && @pending.byte_at(first) & 0xc0 == 0x80
        first -= 1
      end
      return limit if first < 0

      width = case @pending.byte_at(first)
              when 0xc2..0xdf then 2
              when 0xe0..0xef then 3
              when 0xf0..0xf4 then 4
              else                 1
              end
      limit - first < width ? first : limit
    end

    private def emit_prefix(size : Int32) : Nil
      return if size.zero?

      @emit.call(@pending.byte_slice(0, size).scrub)
      @pending = @pending.byte_slice(size)
    end
  end
end
