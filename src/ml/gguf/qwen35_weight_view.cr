module ML::GGUF
  record Qwen35WeightSpan, address : UInt64, length : Int64
  record Qwen35WeightRun, address : UInt64, length : Int64
  record Qwen35WeightSlot, run_index : Int32, offset : Int64

  # Page-aligned no-copy view that contains one mmap-backed weight slice.
  # The view owns no bytes; its registered model mmap must outlive it.
  record Qwen35WeightView, address : UInt64, length : Int64, offset : Int64 do
    def self.for_slice(base_address : UInt64, region_size : Int64,
                       slice_address : UInt64, slice_size : Int64,
                       page_size : Int64 = 16_384_i64) : Qwen35WeightView?
      raise ArgumentError.new("region size must be positive") unless region_size > 0
      raise ArgumentError.new("slice size must be positive") unless slice_size > 0
      raise ArgumentError.new("page size must be a positive power of two") unless page_size > 0 && (page_size & (page_size - 1)) == 0
      raise ArgumentError.new("mmap base must be page-aligned") unless base_address % page_size.to_u64 == 0
      raise ArgumentError.new("mmap region must be page-aligned") unless region_size % page_size == 0

      return nil if slice_address < base_address
      relative_u = slice_address - base_address
      return nil if relative_u > Int64::MAX.to_u64
      relative = relative_u.to_i64
      return nil if relative >= region_size
      return nil if slice_size > region_size - relative

      view_relative = relative - relative % page_size
      slice_end = relative + slice_size
      view_end = ((slice_end - 1) // page_size + 1) * page_size
      return nil if view_end > region_size

      Qwen35WeightView.new(
        base_address + view_relative.to_u64,
        view_end - view_relative,
        relative - view_relative,
      )
    end
  end

  # Immutable page geometry for mmap-backed quantized weights. Runs are sorted
  # and disjoint, while slots preserve the caller's span order.
  record Qwen35WeightViewPlan,
    runs : Array(Qwen35WeightRun),
    slots : Array(Qwen35WeightSlot) do
    def self.for_spans(base_address : UInt64, region_size : Int64,
                       spans : Array(Qwen35WeightSpan),
                       page_size : Int64 = 16_384_i64,
                       coalesce_target_bytes : Int64? = nil) : Qwen35WeightViewPlan
      if target = coalesce_target_bytes
        raise ArgumentError.new("coalescing target must be page-aligned") unless target >= page_size && target % page_size == 0
      end

      views = spans.map do |span|
        Qwen35WeightView.for_slice(
          base_address,
          region_size,
          span.address,
          span.length,
          page_size,
        ) || raise ArgumentError.new("weight span lies outside mmap region")
      end

      runs = [] of Qwen35WeightRun
      views.sort_by(&.address).each do |view|
        if previous = runs.last?
          previous_end = previous.address + previous.length.to_u64
          view_end = view.address + view.length.to_u64
          merged_length = (Math.max(previous_end, view_end) - previous.address).to_i64
          if view.address < previous_end || coalesce_target_bytes.try { |target| merged_length <= target }
            runs[-1] = Qwen35WeightRun.new(
              previous.address,
              merged_length,
            )
            next
          end
        end
        runs << Qwen35WeightRun.new(view.address, view.length)
      end

      slots = spans.map do |span|
        span_end = span.address + span.length.to_u64
        run_index = runs.index do |run|
          run_end = run.address + run.length.to_u64
          run.address <= span.address && span_end <= run_end
        end
        raise "coalesced weight run does not contain source span" unless run_index
        run = runs[run_index]
        Qwen35WeightSlot.new(run_index.to_i32, (span.address - run.address).to_i64)
      end

      Qwen35WeightViewPlan.new(runs, slots)
    end
  end
end
