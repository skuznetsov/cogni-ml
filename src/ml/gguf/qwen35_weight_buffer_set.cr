require "./compute"
require "./qwen35_weight_view"

module ML::GGUF
  # Model-owned, lazy zero-copy Metal views over quantized GGUF weights.
  #
  # The immutable plan covers every weight up front, so independently created
  # runs can never overlap. Metal buffers are created only when a weight in the
  # corresponding run is first used and are released before the GGUF mmap.
  class Qwen35WeightBufferSet < MetalWeightSlotProvider
    COALESCE_TARGET_BYTES = 256_i64 * 1024 * 1024

    getter plan : Qwen35WeightViewPlan

    @weights : Array(QuantWeight)
    @weight_indices : Hash(UInt64, Int32)
    @buffers : Array(ML::MetalBuffer?)
    @closed : Bool
    @mutex : Mutex

    def initialize(base : Pointer(UInt8), region_size : UInt64,
                   weights : Array(QuantWeight),
                   page_size : Int64 = 16_384_i64)
      raise ArgumentError.new("weight buffer set requires at least one weight") if weights.empty?
      raise ArgumentError.new("mmap region is too large") if region_size > Int64::MAX.to_u64
      raise ArgumentError.new("page size must be positive") unless page_size > 0

      aligned_region_size = (region_size // page_size.to_u64) * page_size.to_u64
      raise ArgumentError.new("mmap region is smaller than one page") if aligned_region_size == 0

      @weights = [] of QuantWeight
      @weight_indices = {} of UInt64 => Int32
      weights.each do |weight|
        address = weight.raw.to_unsafe.address
        next if address < base.address
        relative = address - base.address
        next if relative > aligned_region_size
        next if weight.raw.size.to_u64 > aligned_region_size - relative
        identity = weight.object_id
        next if @weight_indices.has_key?(identity)
        @weight_indices[identity] = @weights.size.to_i32
        @weights << weight
      end
      raise ArgumentError.new("no quantized weight lies in the aligned mmap region") if @weights.empty?

      spans = @weights.map do |weight|
        Qwen35WeightSpan.new(weight.raw.to_unsafe.address, weight.raw.size.to_i64)
      end
      @plan = Qwen35WeightViewPlan.for_spans(
        base.address,
        aligned_region_size.to_i64,
        spans,
        page_size,
        COALESCE_TARGET_BYTES,
      )
      @buffers = Array(ML::MetalBuffer?).new(@plan.runs.size, nil)
      @closed = false
      @mutex = Mutex.new

      bound = [] of QuantWeight
      begin
        @weights.each do |weight|
          weight.bind_metal_slot_provider(self)
          bound << weight
        end
      rescue ex
        bound.each { |weight| weight.unbind_metal_slot_provider(self) }
        raise ex
      end
    end

    def slot_for(weight : QuantWeight) : {ML::MetalBuffer, Int64}?
      index = @weight_indices[weight.object_id]?
      return nil unless index

      @mutex.synchronize do
        raise "Qwen35 weight buffer set is closed" if @closed
        slot = @plan.slots[index]
        run_index = slot.run_index
        buffer = @buffers[run_index] || begin
          run = @plan.runs[run_index]
          created = ML::MetalBuffer.wrap_no_copy(
            Pointer(Void).new(run.address),
            run.length,
          )
          @buffers[run_index] = created
          created
        end
        {buffer, slot.offset}
      end
    end

    def stats : NamedTuple(mode: Symbol, buffers: Int32, bytes: Int64,
      planned_buffers: Int32, planned_bytes: Int64)
      @mutex.synchronize do
        created = @buffers.compact
        {
          mode:            :bounded,
          buffers:         created.size.to_i32,
          bytes:           created.sum(&.size),
          planned_buffers: @plan.runs.size.to_i32,
          planned_bytes:   @plan.runs.sum(&.length),
        }
      end
    end

    def close : Nil
      @mutex.synchronize do
        return if @closed
        @weights.each { |weight| weight.unbind_metal_slot_provider(self) }
        @buffers.each { |buffer| buffer.try(&.release) }
        @buffers.fill(nil)
        @closed = true
      end
    end
  end
end
