require "../../src/ml/video/pixelumm_decoder_block_mps"

# Keep the grouping oracle independent of native graph construction. Geometry
# equality licenses batching; the two ordered key lists stay lane-local.
module PixelUMMWindowBatchProbe
  extend self

  EXTRA_PANEL_BYTES = 888_i64 * 1024**2

  record Pair, first : Int32, second : Int32, queries : Int32, keys : Int32

  def validate_capture_inventory!(order : Array(String)) : Nil
    unless order == ["attention", "output"]
      raise ArgumentError.new("window batch requires exactly attention and output captures in order")
    end
  end

  def pairs(layout : ML::Video::PixelUMMDecoderBlockMPS::AttentionLayout) : Array(Pair)
    groups = {} of Tuple(Int32, Int32) => Array(Int32)
    layout.tile_count.times do |tile|
      shape = {layout.query_indices(tile).size.to_i32, layout.key_indices(tile).size.to_i32}
      (groups[shape] ||= [] of Int32) << tile
    end
    result = [] of Pair
    groups.each do |shape, tiles|
      raise ArgumentError.new("window batch requires even exact-geometry groups") unless tiles.size.even?
      tiles.each_slice(2) do |pair|
        result << Pair.new(pair[0], pair[1], shape[0], shape[1])
      end
    end
    validate_pairs!(layout, result)
    result
  end

  def validate_pairs!(layout : ML::Video::PixelUMMDecoderBlockMPS::AttentionLayout,
                      pairs : Array(Pair)) : Nil
    tiles = pairs.flat_map { |pair| [pair.first, pair.second] }
    unless tiles.sort == (0...layout.tile_count).to_a
      raise ArgumentError.new("window pairs must cover each tile exactly once")
    end
    pairs.each do |pair|
      {pair.first, pair.second}.each do |tile|
        unless layout.query_indices(tile).size == pair.queries && layout.key_indices(tile).size == pair.keys
          raise ArgumentError.new("window pair changed query or reduction geometry")
        end
      end
    end
  end

  def validate_lane_context!(layout : ML::Video::PixelUMMDecoderBlockMPS::AttentionLayout,
                             tile : Int32, queries : Array(Int32), keys : Array(Int32)) : Nil
    unless queries == layout.query_indices(tile) && keys == layout.key_indices(tile)
      raise ArgumentError.new("window batch lane changed its ordered query/key context")
    end
  end

  def expected_elements(mode : Int32, layout : ML::Video::PixelUMMDecoderBlockMPS::AttentionLayout) : UInt32
    batches = pairs(layout)
    count = case mode
            when 1
              batches.sum { |p| 2_i64 * 32 * p.queries * p.keys }
            when 2
              batches.sum { |p| 2_i64 * 32 * p.queries * 128 }
            when 3
              p = batches.first
              2_i64 * 32 * p.queries * p.keys
            else
              raise ArgumentError.new("expected a stage-check mode")
            end
    raise ArgumentError.new("stage metrics overflow") unless 0 < count <= UInt32::MAX
    count.to_u32
  end

  {% unless flag?(:cpu_only) %}
    lib Bridge
      fun prepare = gs_pixelumm_window_batch_prepare(plan : Void*, shared : Void*, error : UInt8*, cap : LibC::SizeT) : Void*
      fun select = gs_pixelumm_window_batch_select(wrapper : Void*, enabled : Int32, error : UInt8*, cap : LibC::SizeT) : Int32
      fun set_check_mode = gs_pixelumm_window_batch_set_check_mode(wrapper : Void*, mode : Int32, error : UInt8*, cap : LibC::SizeT) : Int32
      fun metrics_buffer = gs_pixelumm_window_batch_metrics_buffer(wrapper : Void*) : Void*
      fun metrics_bytes = gs_pixelumm_window_batch_metrics_bytes(wrapper : Void*) : UInt64
      fun poison = gs_pixelumm_window_batch_poison(wrapper : Void*, error : UInt8*, cap : LibC::SizeT) : Int32
      fun extra_bytes = gs_pixelumm_window_batch_extra_bytes(wrapper : Void*) : UInt64
      fun copy_layout = gs_pixelumm_window_batch_copy_layout(wrapper : Void*, output : UInt32*, count : UInt64) : Int32
      fun release = gs_pixelumm_window_batch_release(wrapper : Void*, error : UInt8*, cap : LibC::SizeT) : Int32
    end

    class Selection
      getter plans : Array(ML::Video::PixelUMMDecoderBlockMPS::Plan)
      getter selected = false
      getter check_mode = 0
      getter named_bytes = 0_i64
      getter metrics_read_count = 0
      @wrappers = [] of Pointer(Void)
      @released = false

      def initialize(@plans)
        unless {1, 36}.includes?(@plans.size) && @plans.uniq.size == @plans.size
          raise ArgumentError.new("window batch requires one or 36 distinct serial plans")
        end
        validate_idle!
        begin
          @plans.each do |plan|
            error = Bytes.new(1024, 0_u8)
            wrapper = Pointer(Void).null
            shared = @wrappers.first? || Pointer(Void).null
            plan.with_window_batch_binding do |native|
              wrapper = Bridge.prepare(native, shared, error.to_unsafe, error.size)
            end
            raise ArgumentError.new("window batch preparation: #{String.new(error.to_unsafe)}") if wrapper.null?
            @wrappers << wrapper
            unless Bridge.extra_bytes(wrapper) == EXTRA_PANEL_BYTES && Bridge.metrics_bytes(wrapper) == 16
              raise ArgumentError.new("window batch native panel/metrics inventory mismatch")
            end
            layout = plan.config.layout
            records = Slice(UInt32).new(120, 0_u32)
            raise ArgumentError.new("window batch layout read failed") unless Bridge.copy_layout(wrapper, records.to_unsafe, records.size.to_u64) == 0
            expected = PixelUMMWindowBatchProbe.pairs(layout).flat_map { |p| [p.first.to_u32, p.second.to_u32, p.queries.to_u32, p.keys.to_u32] }
            raise ArgumentError.new("native window batch layout differs from CPU oracle") unless records.to_a == expected
          end
          @named_bytes = EXTRA_PANEL_BYTES + @wrappers.size * 16_i64
        rescue error
          release
          raise error
        end
      end

      def configure!(selected : Bool, check_mode : Int32 = 0) : Nil
        validate_idle!
        unless 0 <= check_mode <= 3 && !(selected && check_mode != 0)
          raise ArgumentError.new("candidate inference and stage diagnostics are mutually exclusive")
        end
        @wrappers.each_with_index do |wrapper, index|
          error = Bytes.new(1024, 0_u8)
          @plans[index].with_window_batch_binding do |_native|
            # Clear both switches first when crossing modes. Every plan was
            # validated before mutation, and the caller owns the serial session.
            raise ArgumentError.new("window batch deselection failed") unless Bridge.select(wrapper, 0, error.to_unsafe, error.size) == 0
            mode = index == 0 ? check_mode : 0
            status = Bridge.set_check_mode(wrapper, mode, error.to_unsafe, error.size)
            raise ArgumentError.new("window batch check selection: #{String.new(error.to_unsafe)}") unless status == 0
            status = Bridge.select(wrapper, selected ? 1 : 0, error.to_unsafe, error.size)
            raise ArgumentError.new("window batch selection: #{String.new(error.to_unsafe)}") unless status == 0
          end
        end
        @selected, @check_mode = selected, check_mode
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(UInt32)
        unless chain.completed? && chain.wait_count == 1 && @check_mode != 0
          raise ArgumentError.new("stage metrics require the existing sole terminal wait")
        end
        validate_idle!
        words = Array(UInt32).new(4, 0_u32)
        handle = Bridge.metrics_buffer(@wrappers.first)
        raise ArgumentError.new("missing window batch metrics buffer") if handle.null?
        actual = MetalFFI.gs_buffer_read(handle, words.to_unsafe.as(Void*), 16_i64)
        raise ArgumentError.new("short window batch metrics read") unless actual == 16
        @metrics_read_count += 1
        words
      end

      def poison_transients! : Nil
        validate_idle!
        @plans.first.poison_window_batch_transients!
        error = Bytes.new(1024, 0_u8)
        @plans.first.with_window_batch_binding do |_native|
          status = Bridge.poison(@wrappers.first, error.to_unsafe, error.size)
          raise ArgumentError.new("window batch poison: #{String.new(error.to_unsafe)}") unless status == 0
        end
      end

      def release : Nil
        return if @released
        validate_idle!
        @wrappers.each_with_index do |wrapper, index|
          next if wrapper.null?
          error = Bytes.new(1024, 0_u8)
          @plans[index].with_window_batch_binding do |_native|
            status = Bridge.release(wrapper, error.to_unsafe, error.size)
            raise ArgumentError.new("window batch release: #{String.new(error.to_unsafe)}") unless status == 0
          end
          @wrappers[index] = Pointer(Void).null
        end
        @wrappers.clear
        @released = true
      end

      private def validate_idle! : Nil
        raise ArgumentError.new("window batch selection was released") if @released
        @plans.each { |plan| plan.with_window_batch_binding { |_native| } }
      end
    end
  {% end %}
end

{% unless flag?(:cpu_only) %}
  class ML::Video::PixelUMMDecoderBlockMPS::Plan
    def with_window_batch_binding(&block : Pointer(Void) -> Nil) : Nil
      validate_shared_gate_up_change!(false)
      @workspace.not_nil!.configure_plan!(@slot) do
        validate_shared_gate_up_change!(false)
        unless @ffn_f32 && !@shared_gate_up_prepared && !@shared_gate_up &&
               !@fused_kv_gather && !@scale_softmax && !@scale_softmax_prepared
          raise ArgumentError.new("window batch requires the unchanged F32-FFN baseline plan")
        end
        yield @native
      end
    end

    def poison_window_batch_transients! : Nil
      # Snapshot membership before configure_plan!: both methods take the same
      # non-recursive workspace mutex. Revalidate idle/live state under that
      # mutex before touching the snapshot's retained buffers.
      buffers = @workspace.not_nil!.mutable_shared_buffers(include_outputs: true)
      with_window_batch_binding do |_native|
        buffers.each_value do |buffer|
          buffer.contents.as(UInt8*).to_slice(buffer.size.to_i32).fill(0xff_u8)
        end
      end
    end
  end
{% end %}
