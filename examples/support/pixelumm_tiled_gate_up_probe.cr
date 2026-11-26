require "../../src/ml/video/pixelumm_decoder_block_mps"

{% unless flag?(:cpu_only) %}
  # Keep the diagnostic binding under the same workspace lock as encoding.
  # Production Plan exposes no raw native handle for callers to retain.
  class ML::Video::PixelUMMDecoderBlockMPS::Plan
    def with_tiled_gate_up_binding(&block : Pointer(Void) -> Nil) : Nil
      validate_shared_gate_up_change!(false)
      @workspace.not_nil!.configure_plan!(@slot) do
        validate_shared_gate_up_change!(false)
        unless @ffn_f32 && !@shared_gate_up_prepared && !@shared_gate_up
          raise ArgumentError.new("tiled gate/up conflicts with a nonbaseline FFN plan")
        end
        yield @native
      end
    end
  end
{% end %}

# Diagnostic selection only. The owner is serial and validates every plan
# before mutating any graph pointer; it never drains or waits on GPU work.
module PixelUMMTiledGateUpProbe
  extend self

  def validate_rows!(tokens : Int32, rows : Int32) : Nil
    unless {2048, 21120}.includes?(tokens) && 0 < rows < tokens
      raise ArgumentError.new("row tiling requires a positive tile smaller than the qualified M extent")
    end
  end

  def row_ranges(tokens : Int32, rows : Int32) : Array(Tuple(Int32, Int32))
    validate_rows!(tokens, rows)
    ranges = [] of Tuple(Int32, Int32)
    offset = 0
    while offset < tokens
      count = Math.min(rows, tokens - offset)
      ranges << {offset, count}
      offset += count
    end
    ranges
  end

  {% unless flag?(:cpu_only) %}
    lib Bridge
      fun prepare = gs_pixelumm_tiled_gate_up_prepare(plan : Void*, row_tile : Int32,
                                                      error : UInt8*, capacity : LibC::SizeT) : Void*
      fun select = gs_pixelumm_tiled_gate_up_select(wrapper : Void*, enabled : Int32,
                                                    error : UInt8*, capacity : LibC::SizeT) : Int32
      fun release = gs_pixelumm_tiled_gate_up_release(wrapper : Void*) : Void
    end

    class Selection
      getter selected = false
      getter row_tile : Int32
      getter plans : Array(ML::Video::PixelUMMDecoderBlockMPS::Plan)
      @wrappers = [] of Pointer(Void)
      @released = false

      def initialize(@plans, @row_tile)
        handles = @plans.map do |plan|
          address = 0_u64
          plan.with_tiled_gate_up_binding { |native| address = native.address }
          address
        end
        raise ArgumentError.new("tiled gate/up needs 36 distinct plans") unless @plans.size == 36 && handles.uniq.size == 36
        PixelUMMTiledGateUpProbe.validate_rows!(21120, @row_tile)
        validate_idle!
        begin
          @plans.each do |plan|
            error = Bytes.new(1024, 0_u8)
            wrapper = Pointer(Void).null
            plan.with_tiled_gate_up_binding do |native|
              wrapper = Bridge.prepare(native, @row_tile, error.to_unsafe, error.size)
            end
            raise ArgumentError.new("tiled gate/up preparation: #{String.new(error.to_unsafe)}") if wrapper.null?
            @wrappers << wrapper
          end
        rescue error
          release
          raise error
        end
      end

      def select!(enabled : Bool) : Nil
        validate_idle!
        raise ArgumentError.new("tiled gate/up wrapper inventory mismatch") unless @wrappers.size == @plans.size
        @wrappers.each_with_index do |wrapper, index|
          error = Bytes.new(1024, 0_u8)
          status = -1
          @plans[index].with_tiled_gate_up_binding do |_native|
            status = Bridge.select(wrapper, enabled ? 1 : 0, error.to_unsafe, error.size)
          end
          raise ArgumentError.new("tiled gate/up selection: #{status}: #{String.new(error.to_unsafe)}") unless status == 0
        end
        @selected = enabled
      end

      def release : Nil
        return if @released
        validate_idle!
        @wrappers.each_with_index do |wrapper, index|
          @plans[index].with_tiled_gate_up_binding { |_native| Bridge.release(wrapper) }
        end
        @wrappers.clear
        @selected = false
        @released = true
      end

      private def validate_idle! : Nil
        raise ArgumentError.new("tiled gate/up selection was released") if @released
        @plans.each do |plan|
          # Reuse the production Plan's actual last-chain check. Disabling
          # shared mode requires no preparation and does not mutate the plan.
          plan.validate_shared_gate_up_change!(false)
          unless plan.ffn_f32 && !plan.shared_gate_up_prepared && !plan.shared_gate_up
            raise ArgumentError.new("tiled gate/up conflicts with a nonbaseline FFN plan")
          end
        end
      end
    end
  {% end %}
end
