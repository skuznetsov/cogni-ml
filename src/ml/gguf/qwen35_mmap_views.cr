module ML::GGUF
  record Qwen35MmapWeightSpan, address : UInt64, length : Int64

  # One page-aligned no-copy Metal view over a dense set of mmap-backed
  # quantized tensors. The view owns no bytes; the GGUF mmap must outlive it.
  record Qwen35MmapWeightView,
    address : UInt64,
    length : Int64,
    tensor_bytes : Int64 do
    def density : Float64
      tensor_bytes.to_f64 / length
    end

    def contains?(span : Qwen35MmapWeightSpan) : Bool
      return false if span.length <= 0 || span.address < address
      relative = span.address - address
      relative <= length.to_u64 && span.length.to_u64 <= length.to_u64 - relative
    end

    def self.for_spans(base_address : UInt64,
                       region_size : UInt64,
                       spans : Array(Qwen35MmapWeightSpan),
                       page_size : Int64 = 16_384_i64,
                       minimum_density : Float64 = 0.95_f64) : Qwen35MmapWeightView
      raise ArgumentError.new("mmap view requires at least one weight span") if spans.empty?
      unless page_size > 0 && (page_size & (page_size - 1)) == 0
        raise ArgumentError.new("page size must be a positive power of two")
      end
      unless minimum_density > 0.0 && minimum_density <= 1.0
        raise ArgumentError.new("minimum density must be in (0, 1]")
      end
      unless base_address % page_size.to_u64 == 0
        raise ArgumentError.new("mmap base must be page-aligned")
      end
      if region_size > UInt64::MAX - base_address
        raise ArgumentError.new("mmap address range overflows")
      end

      usable_size = (region_size // page_size.to_u64) * page_size.to_u64
      raise ArgumentError.new("mmap region is smaller than one page") if usable_size == 0

      relative_start = UInt64::MAX
      relative_end = 0_u64
      tensor_bytes = 0_i64
      previous_end = nil.as(UInt64?)
      spans.sort_by(&.address).each do |span|
        raise ArgumentError.new("weight span length must be positive") unless span.length > 0
        raise ArgumentError.new("weight span starts before mmap") if span.address < base_address
        relative = span.address - base_address
        if relative > usable_size || span.length.to_u64 > usable_size - relative
          raise ArgumentError.new("weight span lies outside page-aligned mmap region")
        end
        if boundary = previous_end
          raise ArgumentError.new("weight spans overlap") if relative < boundary
        end
        span_end = relative + span.length.to_u64
        previous_end = span_end
        relative_start = Math.min(relative_start, relative)
        relative_end = Math.max(relative_end, span_end)
        if span.length > Int64::MAX - tensor_bytes
          raise ArgumentError.new("weight byte count overflow")
        end
        tensor_bytes += span.length
      end

      page = page_size.to_u64
      aligned_start = relative_start - relative_start % page
      aligned_end = ((relative_end - 1) // page + 1) * page
      raise ArgumentError.new("aligned weight view exceeds mmap") if aligned_end > usable_size
      unsigned_length = aligned_end - aligned_start
      if unsigned_length > Int64::MAX.to_u64
        raise ArgumentError.new("aligned weight view is too large")
      end
      length = unsigned_length.to_i64
      view = new(base_address + aligned_start, length, tensor_bytes)
      if view.density < minimum_density
        raise ArgumentError.new(
          "weight view density #{view.density} is below #{minimum_density} " \
          "(tensor_bytes=#{tensor_bytes}, view_bytes=#{length})",
        )
      end
      view
    end
  end

  record Qwen35MmapWeightGroup,
    first_layer : Int32,
    last_layer : Int32,
    views : Array(Qwen35MmapWeightView)
end
