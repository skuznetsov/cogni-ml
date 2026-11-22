require "./inkling2_weights"
require "./inkling2_metal_view"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  require "../metal/device"
  require "../metal/dispatch"
  require "../core/buffer"
{% end %}

module ML::GGUF
  {% if flag?(:darwin) && !flag?(:cpu_only) %}
    # Batched causal channelwise convolution for K/V preparation and the
    # post-Wo/FFN convolutions. History contains the raw convolution inputs,
    # matching Inkling2Convolution's per-channel state.
    class Inkling2ResidentConvolution
      SOURCE = {{ read_file("#{__DIR__}/kernels/inkling2_resident_attention.metal") }}

      @history : ML::MetalBuffer? = nil
      @last_end_position : Int32? = nil
      @last_encoded_start_position : Int32? = nil
      @last_encoded_end_position : Int32? = nil
      @speculation_start_position : Int32? = nil
      @history_slots : Int32
      @closed = false

      getter channels : Int32
      getter taps : Int32
      getter rollback_rows : Int32

      def initialize(@channels : Int32, @taps : Int32, *, @rollback_rows : Int32 = 0)
        raise ArgumentError.new("Inkling 2 resident convolution width must be positive") unless @channels > 0
        raise ArgumentError.new("Inkling 2 resident convolution needs at least one tap") unless @taps > 0
        raise ArgumentError.new("Inkling 2 resident convolution rollback rows must be non-negative") if @rollback_rows < 0
        history_slots = @taps > 1 ? (@taps - 1).to_i64 + @rollback_rows : 0_i64
        raise ArgumentError.new("Inkling 2 resident convolution history slots exceed Int32") if history_slots > Int32::MAX
        @history_slots = history_slots.to_i32
      end

      # Encode a row-major [rows, channels] causal convolution. Static taps
      # are channel-major and oldest-to-current. Dynamic taps are
      # [rows, head, tap], with tap zero multiplying the current input.
      def encode(
        encoder : ML::Metal::ComputeEncoder,
        scratch : Inkling2ResidentScratch,
        start : Int32,
        rows : Int32,
        input : Inkling2View,
        taps_weights : Inkling2View,
        dynamic : Inkling2View? = nil,
        *,
        dynamic_heads : Int32 = 0,
      ) : Inkling2View
        ensure_open!
        validate_speculation_tile!(start, rows)
        validate_tile!(start, rows)
        total = checked_elements(rows, @channels, "convolution output")
        validate_view!(input, total, "convolution input")
        validate_view!(taps_weights, checked_elements(@channels, @taps, "convolution taps"), "convolution weights")

        dynamic_view = dynamic || input
        if dynamic
          unless dynamic_heads > 0 && @channels % dynamic_heads == 0
            raise ArgumentError.new("Inkling 2 resident dynamic convolution heads must divide channel width")
          end
          validate_view!(dynamic_view, checked_elements(checked_elements(rows, dynamic_heads, "dynamic tap rows"), @taps, "dynamic taps"), "dynamic convolution taps")
        elsif dynamic_heads != 0
          raise ArgumentError.new("Inkling 2 resident dynamic head count requires dynamic taps")
        end

        output = scratch.take(total)
        history_slots = @history_slots
        history_count = checked_elements(history_slots, @channels, "convolution history dispatch")
        history = @history
        if history_slots > 0
          history ||= begin
            buffer = ML::MetalBuffer.new(checked_bytes(history_slots, @channels, "convolution history"), ML::StorageMode::Shared)
            @history = buffer
            buffer
          end
        end
        history_buffer = history || input.buffer

        encoder.set_pipeline(pipeline("inkling2_resident_convolution"))
        encoder.set_buffer(input.buffer, 0, offset: input.offset)
        encoder.set_buffer(taps_weights.buffer, 1, offset: taps_weights.offset)
        encoder.set_buffer(dynamic_view.buffer, 2, offset: dynamic_view.offset)
        encoder.set_buffer(history_buffer, 3)
        encoder.set_buffer(output.buffer, 4, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(@channels.to_u32, 5)
        encoder.set_value(@taps.to_u32, 6)
        encoder.set_value(rows.to_u32, 7)
        encoder.set_value(start.to_u32, 8)
        encoder.set_value(dynamic_heads.to_u32, 9)
        encoder.set_value(history_slots.to_u32, 10)
        encoder.dispatch_1d(total)

        if history_slots > 0
          encoder.set_pipeline(pipeline("inkling2_resident_convolution_commit"))
          encoder.set_buffer(input.buffer, 0, offset: input.offset)
          encoder.set_buffer(history_buffer, 1, ML::Metal::BufferAccess::Write)
          encoder.set_value(@channels.to_u32, 2)
          encoder.set_value(rows.to_u32, 3)
          encoder.set_value(start.to_u32, 4)
          encoder.set_value(history_slots.to_u32, 5)
          encoder.dispatch_1d(history_count)
        end

        @last_end_position = start + rows
        @last_encoded_start_position = start
        @last_encoded_end_position = start + rows
        output
      end

      # `start == 0` begins a fresh visit. Physical ring contents stay
      # untouched; negative history indices are masked in the convolution.
      def reset : Nil
        @last_end_position = nil
        @last_encoded_start_position = nil
        @last_encoded_end_position = nil
        @speculation_start_position = nil
      end

      # Rewind only the logical end. The enlarged history ring preserves the
      # prefix while the caller overwrites the speculative suffix.
      def rewind(end_position : Int32) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident convolution cannot rewind during active speculation") if @speculation_start_position
        last_end = @last_end_position || raise IndexError.new("Inkling 2 resident convolution history is empty")
        encoded_start = @last_encoded_start_position || raise IndexError.new("Inkling 2 resident convolution has no encoded tile")
        unless encoded_start <= end_position <= last_end
          raise ArgumentError.new("Inkling 2 resident convolution rewind must stay within its latest encoded tile")
        end
        encoded_end = @last_encoded_end_position || raise IndexError.new("Inkling 2 resident convolution has no encoded tile")
        if encoded_end.to_i64 - end_position > @rollback_rows
          raise ArgumentError.new("Inkling 2 resident convolution rewind exceeds #{@rollback_rows} reserved rows")
        end
        @last_end_position = end_position
      end

      # Open one bounded append span. It may be filled by several consecutive
      # scalar encodes; finish_speculation chooses the retained exclusive end.
      def begin_speculation(start : Int32) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident convolution speculation requires a positive rollback reserve") unless @rollback_rows > 0
        raise ArgumentError.new("Inkling 2 resident convolution speculation is already active") if @speculation_start_position
        frontier = @last_end_position || 0_i32
        unless start == frontier
          raise ArgumentError.new("Inkling 2 resident convolution speculation must begin at frontier #{frontier}")
        end
        @speculation_start_position = start
      end

      # Keep the prefix ending at end_position and make it the append frontier.
      def finish_speculation(end_position : Int32) : Nil
        ensure_open!
        span_start = @speculation_start_position || raise ArgumentError.new("Inkling 2 resident convolution speculation is not active")
        last_end = @last_end_position || span_start
        unless span_start <= end_position <= last_end
          raise ArgumentError.new("Inkling 2 resident convolution speculation finish must stay within its encoded span")
        end
        if last_end.to_i64 - span_start > @rollback_rows
          raise ArgumentError.new("Inkling 2 resident convolution speculation exceeds #{@rollback_rows} reserved rows")
        end
        @last_end_position = end_position
        @speculation_start_position = nil
      end

      def close : Nil
        return if @closed
        @closed = true
        @history.try(&.release)
        @history = nil
      end

      private def validate_tile!(start : Int32, rows : Int32) : Nil
        raise ArgumentError.new("Inkling 2 resident convolution rows must be positive") unless rows > 0
        raise ArgumentError.new("Inkling 2 resident convolution start must be non-negative") if start < 0
        raise ArgumentError.new("Inkling 2 resident convolution end exceeds Int32") if start.to_i64 + rows > Int32::MAX
        if start == 0
          @last_end_position = nil
          @last_encoded_start_position = nil
          @last_encoded_end_position = nil
        elsif @last_end_position != start
          raise ArgumentError.new("Inkling 2 resident convolution tiles must be contiguous from position zero")
        end
      end

      private def validate_speculation_tile!(start : Int32, rows : Int32) : Nil
        span_start = @speculation_start_position
        return unless span_start
        raise ArgumentError.new("Inkling 2 resident convolution rows must be positive") unless rows > 0
        frontier = @last_end_position || span_start
        unless start == frontier
          raise ArgumentError.new("Inkling 2 resident convolution speculative tiles must be contiguous from position #{frontier}")
        end
        end_position = start.to_i64 + rows
        span_end = span_start.to_i64 + @rollback_rows
        if end_position > span_end
          raise ArgumentError.new("Inkling 2 resident convolution speculation exceeds #{@rollback_rows} reserved rows")
        end
      end

      private def validate_view!(view : Inkling2View, elements : Int32, name : String) : Nil
        bytes = elements.to_i64 * sizeof(Float32)
        raise ArgumentError.new("Inkling 2 resident #{name} view exceeds its Metal buffer") if view.offset < 0 || view.offset + bytes > view.buffer.size
      end

      private def checked_elements(rows : Int32, width : Int32, name : String) : Int32
        product = rows.to_i64 * width
        raise ArgumentError.new("Inkling 2 resident #{name} exceeds Int32 elements") unless rows >= 0 && width >= 0 && product <= Int32::MAX
        product.to_i32
      end

      private def checked_bytes(rows : Int32, width : Int32, name : String) : Int64
        elements = rows.to_i64 * width
        raise ArgumentError.new("Inkling 2 resident #{name} exceeds Int64 bytes") unless rows > 0 && width > 0 && elements <= Int64::MAX // sizeof(Float32)
        elements * sizeof(Float32)
      end

      private def ensure_open! : Nil
        raise "Inkling 2 resident convolution is closed" if @closed
      end

      private def pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get_or_compile(name, SOURCE)
      end
    end

    private lib LibInkling2Attention
      # Matches Inkling2PrepareQKVRParams in inkling2_resident_attention.metal.
      struct PrepareQKVRParams
        n_head : UInt32
        n_head_kv : UInt32
        head_dim : UInt32
        relative_dim : UInt32
        start : UInt32
        sliding : UInt32
        log_scale_floor : UInt32
        qk_clip : Float32
        rel_clip : Float32
        rms_eps : Float32
        log_scale_alpha : Float32
      end
    end

    # Encode-only per-visit Inkling 2 attention. It prepares Q/K/V and
    # relative queries, performs causal attention, then commits K/V for later
    # tiles. The caller owns submission, synchronization, and scratch lifetime.
    class Inkling2ResidentAttention
      SOURCE = Inkling2ResidentConvolution::SOURCE
      private MAX_GRID_ELEMENTS = 16_777_216_i32
      private INITIAL_CACHE_ROWS = 256_i32
      private SPLIT_KEY_THRESHOLD = 64_i32
      private SPLIT_KEY_PARTITIONS = 4_i32
      private SPLIT_SCORE_CACHE_LIMIT = 256_i32
      private SPLIT_PARALLEL_SCORE_LIMIT = 512_i32
      private SPLIT_PARALLEL_THREADS = 128_i32
      MAX_HEAD_DIM = 256_i32

      getter sliding : Bool
      getter max_seq : Int32
      getter kv_width : Int32
      getter capacity : Int32
      getter rollback_rows : Int32

      @key_convolution : Inkling2ResidentConvolution
      @value_convolution : Inkling2ResidentConvolution
      @k_cache : ML::MetalBuffer? = nil
      @v_cache : ML::MetalBuffer? = nil
      @cache_capacity = 0_i32
      @physical_cache_capacity : Int32
      @retired_caches = [] of ML::MetalBuffer
      @last_end_position : Int32? = nil
      @last_encoded_start_position : Int32? = nil
      @last_encoded_end_position : Int32? = nil
      @speculation_start_position : Int32? = nil
      @closed = false

      def initialize(@config : Inkling2Config, @sliding : Bool, @max_seq : Int32, *, @rollback_rows : Int32 = 0)
        raise ArgumentError.new("Inkling 2 resident max_seq must be positive") unless @max_seq > 0
        raise ArgumentError.new("Inkling 2 resident attention rollback rows must be non-negative") if @rollback_rows < 0
        @kv_width = @config.n_head_kv * @config.head_dim
        @capacity = @sliding ? Math.min(@max_seq, @config.sliding_window) : @max_seq
        @physical_cache_capacity = if @sliding
                                     Math.min(@max_seq.to_i64, @capacity.to_i64 + @rollback_rows).to_i32
                                   else
                                     @capacity
                                   end
        @key_convolution = Inkling2ResidentConvolution.new(@kv_width, @config.conv_size, rollback_rows: @rollback_rows)
        @value_convolution = Inkling2ResidentConvolution.new(@kv_width, @config.conv_size, rollback_rows: @rollback_rows)
      end

      # Q, K, V, relative-Q, gate, and sink are row-major projection tiles.
      # The clipping, tap, and relative-bias arguments are resident F32 views.
      # The result is gated attended context [rows, n_head*head_dim], before Wo.
      def encode(
        encoder : ML::Metal::ComputeEncoder,
        scratch : Inkling2ResidentScratch,
        start : Int32,
        rows : Int32,
        q : Inkling2View,
        k : Inkling2View,
        v : Inkling2View,
        q_rel : Inkling2View,
        gate : Inkling2View,
        sink : Inkling2View,
        q_clip : Inkling2View,
        rel_clip : Inkling2View,
        relative_bias : Inkling2View,
        k_taps : Inkling2View,
        v_taps : Inkling2View,
        dynamic_k : Inkling2View? = nil,
        dynamic_v : Inkling2View? = nil,
        *,
        decode_block : Bool = false,
      ) : Inkling2View
        ensure_open!
        validate_speculation_tile!(start, rows)
        validate_tile!(start, rows)

        n_head = @config.n_head
        n_head_kv = @config.n_head_kv
        head_dim = @config.head_dim
        rel_dim = @config.rel_dim
        q_width = n_head * head_dim
        rel_width = n_head * rel_dim
        relative_extent = @sliding ? @config.sliding_window : @config.rel_extent
        unless head_dim <= MAX_HEAD_DIM && rel_dim <= MAX_HEAD_DIM
          raise ArgumentError.new("Inkling 2 resident attention supports head and relative dimensions through 256")
        end

        validate_view!(q, checked_elements(rows, q_width, "Q tile"), "Q")
        validate_view!(k, checked_elements(rows, @kv_width, "K tile"), "K")
        validate_view!(v, checked_elements(rows, @kv_width, "V tile"), "V")
        validate_view!(q_rel, checked_elements(rows, rel_width, "relative Q tile"), "relative Q")
        validate_view!(gate, checked_elements(rows, q_width, "gate tile"), "gate")
        validate_view!(sink, checked_elements(rows, n_head, "sink tile"), "sink")
        validate_view!(q_clip, checked_elements(n_head, head_dim, "Q clip"), "Q clip")
        validate_view!(rel_clip, checked_elements(n_head, rel_dim, "relative clip"), "relative clip")
        validate_view!(relative_bias, checked_elements(relative_extent, rel_dim, "relative bias"), "relative bias")
        validate_view!(k_taps, checked_elements(@kv_width, @config.conv_size, "K convolution weights"), "K convolution weights")
        validate_view!(v_taps, checked_elements(@kv_width, @config.conv_size, "V convolution weights"), "V convolution weights")
        dynamic_k.try { |view| validate_view!(view, checked_elements(checked_elements(rows, n_head_kv, "dynamic K rows"), @config.conv_size, "dynamic K taps"), "dynamic K") }
        dynamic_v.try { |view| validate_view!(view, checked_elements(checked_elements(rows, n_head_kv, "dynamic V rows"), @config.conv_size, "dynamic V taps"), "dynamic V") }

        required_capacity = @sliding ? @capacity : start + rows
        ensure_cache_capacity(encoder, required_capacity, start)

        key = @key_convolution.encode(encoder, scratch, start, rows, k, k_taps, dynamic_k, dynamic_heads: dynamic_k ? n_head_kv : 0)
        value = @value_convolution.encode(encoder, scratch, start, rows, v, v_taps, dynamic_v, dynamic_heads: dynamic_v ? n_head_kv : 0)

        encode_prepare_qkvr(encoder, q, key, value, q_rel, q_clip, rel_clip, start, rows)

        attended = scratch.take(checked_elements(rows, q_width, "attention output"))
        # Keep this before the cache commit below: later rows in the block read
        # earlier block-local K/V from the tile until all queries are done.
        encode_attention(encoder, scratch, q, key, value, q_rel, gate, sink, relative_bias, attended,
          start, rows, relative_extent, decode_block)
        encode_cache_commit(encoder, key, value, start, rows)
        @last_end_position = start + rows
        @last_encoded_start_position = start
        @last_encoded_end_position = start + rows
        attended
      end

      # Move the logical end back within the most recently encoded tile. K/V
      # bytes stay in place; reserved physical ring rows keep the accepted
      # prefix available for the next append.
      def rewind(end_position : Int32) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident attention cannot rewind during active speculation") if @speculation_start_position
        last_end = @last_end_position || raise IndexError.new("Inkling 2 resident cache is empty")
        encoded_start = @last_encoded_start_position || raise IndexError.new("Inkling 2 resident attention has no encoded tile")
        unless encoded_start <= end_position <= last_end
          raise ArgumentError.new("Inkling 2 resident attention rewind must stay within its latest encoded tile")
        end
        encoded_end = @last_encoded_end_position || raise IndexError.new("Inkling 2 resident attention has no encoded tile")
        if encoded_end.to_i64 - end_position > @rollback_rows
          raise ArgumentError.new("Inkling 2 resident attention rewind exceeds #{@rollback_rows} reserved rows")
        end
        @key_convolution.rewind(end_position)
        @value_convolution.rewind(end_position)
        @last_end_position = end_position
      end

      # Open one bounded append span. It may be filled by several consecutive
      # scalar encodes; finish_speculation chooses the retained exclusive end.
      def begin_speculation(start : Int32) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident attention speculation requires a positive rollback reserve") unless @rollback_rows > 0
        raise ArgumentError.new("Inkling 2 resident attention speculation is already active") if @speculation_start_position
        frontier = @last_end_position || 0_i32
        unless start == frontier
          raise ArgumentError.new("Inkling 2 resident attention speculation must begin at frontier #{frontier}")
        end
        @key_convolution.begin_speculation(start)
        @value_convolution.begin_speculation(start)
        @speculation_start_position = start
      end

      # Keep the prefix ending at end_position and make it the append frontier.
      def finish_speculation(end_position : Int32) : Nil
        ensure_open!
        span_start = @speculation_start_position || raise ArgumentError.new("Inkling 2 resident attention speculation is not active")
        last_end = @last_end_position || span_start
        unless span_start <= end_position <= last_end
          raise ArgumentError.new("Inkling 2 resident attention speculation finish must stay within its encoded span")
        end
        if last_end.to_i64 - span_start > @rollback_rows
          raise ArgumentError.new("Inkling 2 resident attention speculation exceeds #{@rollback_rows} reserved rows")
        end
        @key_convolution.finish_speculation(end_position)
        @value_convolution.finish_speculation(end_position)
        @last_end_position = end_position
        @speculation_start_position = nil
      end

      # Key/value indices and end_position are absolute token positions;
      # end_position is exclusive, as in the resident engine.
      def key_value_at(index : Int32, end_position : Int32) : Tuple(Array(Float32), Array(Float32), Int32)
        ensure_open!
        last_end = @last_end_position || raise IndexError.new("Inkling 2 resident cache is empty")
        unless 0 <= index < end_position && end_position <= last_end
          raise IndexError.new("Inkling 2 resident cache position is outside the requested prefix")
        end
        if @sliding && end_position != last_end
          raise IndexError.new("Inkling 2 resident sliding cache snapshots require the current end position")
        end
        # Retention follows the actual ring head, even if a caller supplies an
        # older prefix boundary. The requested boundary cannot make an
        # overwritten physical slot valid again.
        earliest = @sliding ? Math.max(0, last_end - @capacity) : 0
        raise IndexError.new("Inkling 2 resident cache position has been evicted") if index < earliest
        key_cache = @k_cache || raise IndexError.new("Inkling 2 resident cache has no keys")
        value_cache = @v_cache || raise IndexError.new("Inkling 2 resident cache has no values")
        slot = @sliding ? index % @cache_capacity : index
        key_cache.sync
        value_cache.sync
        key_ptr = key_cache.contents.as(Pointer(Float32)) + slot * @kv_width
        value_ptr = value_cache.contents.as(Pointer(Float32)) + slot * @kv_width
        key = Array(Float32).new(@kv_width) { |channel| key_ptr[channel] }
        value = Array(Float32).new(@kv_width) { |channel| value_ptr[channel] }
        {key, value, index}
      end

      # Call after the command buffer completes (or is discarded). Growth
      # copies are encoded in the active command, so old buffers must remain
      # alive until then.
      def finish : Nil
        @retired_caches.each(&.release)
        @retired_caches.clear
      end

      def reset : Nil
        ensure_open!
        @last_end_position = nil
        @last_encoded_start_position = nil
        @last_encoded_end_position = nil
        @speculation_start_position = nil
        @key_convolution.reset
        @value_convolution.reset
      end

      def close : Nil
        return if @closed
        @closed = true
        finish
        @k_cache.try(&.release)
        @v_cache.try(&.release)
        @k_cache = nil
        @v_cache = nil
        @cache_capacity = 0
        @key_convolution.close
        @value_convolution.close
      end

      private def ensure_cache_capacity(encoder : ML::Metal::ComputeEncoder, required : Int32, start : Int32) : Nil
        if @sliding
          return if @k_cache && @v_cache
          allocate_caches(encoder, @physical_cache_capacity, 0)
          return
        end
        return if required <= @cache_capacity

        target = @cache_capacity
        target = Math.min(INITIAL_CACHE_ROWS, @max_seq) if target == 0
        while target < required
          target = Math.min(@max_seq, target.to_i64 * 2).to_i32
        end
        raise ArgumentError.new("Inkling 2 resident cache growth cannot copy an incomplete prefix") if start > @cache_capacity && @cache_capacity > 0
        allocate_caches(encoder, target, start)
      end

      private def allocate_caches(encoder : ML::Metal::ComputeEncoder, target : Int32, copy_count : Int32) : Nil
        bytes = checked_bytes(target, @kv_width, "attention cache")
        new_keys = ML::MetalBuffer.new(bytes, ML::StorageMode::Shared)
        new_values = nil.as(ML::MetalBuffer?)
        begin
          new_values = ML::MetalBuffer.new(bytes, ML::StorageMode::Shared)
          old_keys = @k_cache
          old_values = @v_cache
          if copy_count > 0
            if previous_keys = old_keys
              if previous_values = old_values
                encode_cache_copy(encoder, previous_keys, previous_values, new_keys, new_values.not_nil!, copy_count)
              end
            end
          end
          @k_cache = new_keys
          @v_cache = new_values
          @cache_capacity = target
          @retired_caches << old_keys if old_keys
          @retired_caches << old_values if old_values
        rescue ex
          new_keys.release
          new_values.try(&.release)
          raise ex
        end
      end

      private def encode_prepare_qkvr(
        encoder : ML::Metal::ComputeEncoder,
        q : Inkling2View,
        keys : Inkling2View,
        values : Inkling2View,
        q_rel : Inkling2View,
        q_clip : Inkling2View,
        rel_clip : Inkling2View,
        start : Int32,
        rows : Int32,
      ) : Nil
        params = LibInkling2Attention::PrepareQKVRParams.new(
          n_head: @config.n_head.to_u32,
          n_head_kv: @config.n_head_kv.to_u32,
          head_dim: @config.head_dim.to_u32,
          relative_dim: @config.rel_dim.to_u32,
          start: start.to_u32,
          sliding: @sliding ? 1_u32 : 0_u32,
          log_scale_floor: @config.log_scale_floor.to_u32,
          qk_clip: @config.qk_clip,
          rel_clip: @config.rel_clip,
          rms_eps: @config.rms_eps,
          log_scale_alpha: @config.log_scale_alpha,
        )
        encoder.set_pipeline(pipeline("inkling2_resident_prepare_qkvr"))
        encoder.set_buffer(q.buffer, 0, ML::Metal::BufferAccess::ReadWrite, offset: q.offset)
        encoder.set_buffer(keys.buffer, 1, ML::Metal::BufferAccess::ReadWrite, offset: keys.offset)
        encoder.set_buffer(values.buffer, 2, ML::Metal::BufferAccess::ReadWrite, offset: values.offset)
        encoder.set_buffer(q_rel.buffer, 3, ML::Metal::BufferAccess::ReadWrite, offset: q_rel.offset)
        encoder.set_buffer(q_clip.buffer, 4, offset: q_clip.offset)
        encoder.set_buffer(rel_clip.buffer, 5, offset: rel_clip.offset)
        encoder.set_bytes(pointerof(params).as(Void*), sizeof(LibInkling2Attention::PrepareQKVRParams), 6)
        encoder.dispatch({32, rows, 2 * @config.n_head + @config.n_head_kv}, {32, 1, 1})
      end

      private def encode_attention(
        encoder : ML::Metal::ComputeEncoder,
        scratch : Inkling2ResidentScratch,
        q : Inkling2View,
        k : Inkling2View,
        v : Inkling2View,
        q_rel : Inkling2View,
        gate : Inkling2View,
        sink : Inkling2View,
        relative_bias : Inkling2View,
        output : Inkling2View,
        start : Int32,
        rows : Int32,
        relative_extent : Int32,
        decode_block : Bool,
      ) : Nil
        key_count = @sliding ? Math.min(start + 1, @config.sliding_window) : start + 1
        split_begin = if decode_block
                        first_split_row(start, rows)
                      elsif rows == 1 && key_count > SPLIT_KEY_THRESHOLD
                        0
                      else
                        rows
                      end
        regular_rows = split_begin
        if regular_rows > 0
          encode_attention_regular(encoder, q, k, v, q_rel, gate, sink, relative_bias,
            output, start, regular_rows, relative_extent)
        end

        split_rows = rows - split_begin
        if split_rows > 0
          row_offset = split_begin
          q_width = @config.n_head * @config.head_dim
          relative_width = @config.n_head * @config.rel_dim
          split_q = attention_row_view(q, row_offset, q_width)
          split_q_rel = attention_row_view(q_rel, row_offset, relative_width)
          split_gate = attention_row_view(gate, row_offset, q_width)
          split_sink = attention_row_view(sink, row_offset, @config.n_head)
          split_output = attention_row_view(output, row_offset, q_width)
          # Preserve the original tile origin for K/V lookup. The query offset
          # advances the suffix while keeping prior block rows visible in-tile.
          encode_attention_split(encoder, scratch, split_q, k, v, split_q_rel,
            split_gate, split_sink, relative_bias, split_output, start, split_rows,
            row_offset, relative_extent)
        end
      end

      # A decode block can straddle the 64-visible-key boundary. Find its
      # first row that should use split-K; rows before it retain the exact
      # one-token reduction path used by serial decode.
      private def first_split_row(start : Int32, rows : Int32) : Int32
        rows.times do |row|
          key_count = start + row + 1
          key_count = Math.min(key_count, @config.sliding_window) if @sliding
          return row if key_count > SPLIT_KEY_THRESHOLD
        end
        rows
      end

      private def attention_row_view(view : Inkling2View, row : Int32, width : Int32) : Inkling2View
        view.slice(checked_elements(row, width, "attention row offset"))
      end

      private def encode_attention_regular(
        encoder : ML::Metal::ComputeEncoder,
        q : Inkling2View,
        k : Inkling2View,
        v : Inkling2View,
        q_rel : Inkling2View,
        gate : Inkling2View,
        sink : Inkling2View,
        relative_bias : Inkling2View,
        output : Inkling2View,
        start : Int32,
        rows : Int32,
        relative_extent : Int32,
      ) : Nil
        encoder.set_pipeline(pipeline("inkling2_resident_attention"))
        bind_attention_inputs(encoder, q, k, v, q_rel, gate, sink, relative_bias,
          start, rows, relative_extent, 0)
        encoder.set_buffer(output.buffer, 9, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.dispatch({32, rows, @config.n_head}, {32, 1, 1})
      end

      private def encode_attention_split(
        encoder : ML::Metal::ComputeEncoder,
        scratch : Inkling2ResidentScratch,
        q : Inkling2View,
        k : Inkling2View,
        v : Inkling2View,
        q_rel : Inkling2View,
        gate : Inkling2View,
        sink : Inkling2View,
        relative_bias : Inkling2View,
        output : Inkling2View,
        start : Int32,
        rows : Int32,
        query_row_offset : Int32,
        relative_extent : Int32,
      ) : Nil
        partial_width = checked_elements(SPLIT_KEY_PARTITIONS, @config.head_dim + 2, "split attention partition")
        partial_width = checked_elements(@config.n_head, partial_width, "split attention head")
        partial_count = checked_elements(rows, partial_width, "split attention output")
        partials = scratch.take(partial_count)
        keys = start + query_row_offset + rows
        keys = Math.min(keys, @config.sliding_window) if @sliding
        partition_size = (keys - 1) // SPLIT_KEY_PARTITIONS + 1

        # Four SIMD groups score disjoint keys, then cover one channel per lane.
        parallel = @config.head_dim == 128 && @config.rel_dim == 16 && partition_size <= SPLIT_PARALLEL_SCORE_LIMIT
        if parallel
          split_pipeline = "inkling2_resident_attention_split_parallel"
          threads = SPLIT_PARALLEL_THREADS
          score_slots = partition_size
          shared_floats = score_slots + 1 # One extra float for the partition maximum.
        else
          split_pipeline = "inkling2_resident_attention_split"
          threads = 32
          score_slots = Math.min(partition_size, SPLIT_SCORE_CACHE_LIMIT)
          shared_floats = score_slots
        end
        encoder.set_pipeline(pipeline(split_pipeline))
        bind_attention_inputs(encoder, q, k, v, q_rel, gate, sink, relative_bias,
          start, rows, relative_extent, query_row_offset)
        encoder.set_buffer(partials.buffer, 9, ML::Metal::BufferAccess::Write, offset: partials.offset)
        encoder.set_value(SPLIT_KEY_PARTITIONS.to_u32, 21)
        encoder.set_value(query_row_offset.to_u32, 22)
        encoder.set_value(score_slots.to_u32, 23)
        encoder.set_threadgroup_memory(((shared_floats + 3) // 4) * 16, 0)
        encoder.dispatch({threads * SPLIT_KEY_PARTITIONS, rows, @config.n_head}, {threads, 1, 1})

        encoder.set_pipeline(pipeline("inkling2_resident_attention_combine"))
        encoder.set_buffer(partials.buffer, 0, offset: partials.offset)
        encoder.set_buffer(gate.buffer, 1, offset: gate.offset)
        encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
        encoder.set_value(rows.to_u32, 3)
        encoder.set_value(@config.n_head.to_u32, 4)
        encoder.set_value(@config.head_dim.to_u32, 5)
        encoder.set_value(SPLIT_KEY_PARTITIONS.to_u32, 6)
        encoder.dispatch({32, rows, @config.n_head}, {32, 1, 1})
      end

      private def bind_attention_inputs(
        encoder : ML::Metal::ComputeEncoder,
        q : Inkling2View,
        k : Inkling2View,
        v : Inkling2View,
        q_rel : Inkling2View,
        gate : Inkling2View,
        sink : Inkling2View,
        relative_bias : Inkling2View,
        start : Int32,
        rows : Int32,
        relative_extent : Int32,
        query_row_offset : Int32,
      ) : Nil
        key_cache = @k_cache.not_nil!
        value_cache = @v_cache.not_nil!
        encoder.set_buffer(q.buffer, 0, offset: q.offset)
        encoder.set_buffer(k.buffer, 1, offset: k.offset)
        encoder.set_buffer(v.buffer, 2, offset: v.offset)
        encoder.set_buffer(q_rel.buffer, 3, offset: q_rel.offset)
        encoder.set_buffer(gate.buffer, 4, offset: gate.offset)
        encoder.set_buffer(sink.buffer, 5, offset: sink.offset)
        encoder.set_buffer(relative_bias.buffer, 6, offset: relative_bias.offset)
        encoder.set_buffer(key_cache, 7)
        encoder.set_buffer(value_cache, 8)
        encoder.set_value(start.to_u32, 10)
        encoder.set_value(rows.to_u32, 11)
        encoder.set_value(@cache_capacity.to_u32, 12)
        encoder.set_value(@kv_width.to_u32, 13)
        encoder.set_value(@config.n_head.to_u32, 14)
        encoder.set_value(@config.n_head_kv.to_u32, 15)
        encoder.set_value(@config.head_dim.to_u32, 16)
        encoder.set_value(@config.rel_dim.to_u32, 17)
        encoder.set_value(relative_extent.to_u32, 18)
        encoder.set_value(@config.sliding_window.to_u32, 19)
        encoder.set_value(@sliding ? 1_u32 : 0_u32, 20)
        encoder.set_value(query_row_offset.to_u32, 21)
      end

      private def encode_cache_commit(
        encoder : ML::Metal::ComputeEncoder,
        keys : Inkling2View,
        values : Inkling2View,
        start : Int32,
        rows : Int32,
      ) : Nil
        key_cache = @k_cache.not_nil!
        value_cache = @v_cache.not_nil!
        total = checked_elements(@sliding ? Math.min(rows, @cache_capacity) : rows, @kv_width, "KV cache commit")
        encoder.set_pipeline(pipeline("inkling2_resident_cache_commit"))
        encoder.set_buffer(keys.buffer, 0, offset: keys.offset)
        encoder.set_buffer(values.buffer, 1, offset: values.offset)
        encoder.set_buffer(key_cache, 2, ML::Metal::BufferAccess::ReadWrite)
        encoder.set_buffer(value_cache, 3, ML::Metal::BufferAccess::ReadWrite)
        encoder.set_value(start.to_u32, 4)
        encoder.set_value(rows.to_u32, 5)
        encoder.set_value(@cache_capacity.to_u32, 6)
        encoder.set_value(@kv_width.to_u32, 7)
        encoder.set_value(@sliding ? 1_u32 : 0_u32, 8)
        encoder.dispatch_1d(total)
      end

      private def encode_cache_copy(
        encoder : ML::Metal::ComputeEncoder,
        old_keys : ML::MetalBuffer,
        old_values : ML::MetalBuffer,
        new_keys : ML::MetalBuffer,
        new_values : ML::MetalBuffer,
        count : Int32,
      ) : Nil
        total = count.to_i64 * @kv_width
        base = 0_i64
        while base < total
          chunk = Math.min(MAX_GRID_ELEMENTS.to_i64, total - base).to_i32
          encoder.set_pipeline(pipeline("inkling2_resident_cache_copy"))
          encoder.set_buffer(old_keys, 0)
          encoder.set_buffer(old_values, 1)
          encoder.set_buffer(new_keys, 2, ML::Metal::BufferAccess::Write)
          encoder.set_buffer(new_values, 3, ML::Metal::BufferAccess::Write)
          encoder.set_value(base.to_u32, 4)
          encoder.set_value(chunk.to_u32, 5)
          encoder.dispatch_1d(chunk)
          base += chunk
        end
      end

      private def validate_tile!(start : Int32, rows : Int32) : Nil
        ensure_open!
        raise ArgumentError.new("Inkling 2 resident attention rows must be positive") unless rows > 0
        raise ArgumentError.new("Inkling 2 resident attention start must be non-negative") if start < 0
        end_position = start.to_i64 + rows
        raise ArgumentError.new("Inkling 2 resident attention exceeds max_seq #{@max_seq}") if end_position > @max_seq
        raise ArgumentError.new("Inkling 2 resident attention exceeds Int32 positions") if end_position > Int32::MAX
        if start == 0
          @last_end_position = nil
          @last_encoded_start_position = nil
          @last_encoded_end_position = nil
        elsif @last_end_position != start
          raise ArgumentError.new("Inkling 2 resident attention tiles must be contiguous from position zero")
        end
      end

      private def validate_speculation_tile!(start : Int32, rows : Int32) : Nil
        span_start = @speculation_start_position
        return unless span_start
        raise ArgumentError.new("Inkling 2 resident attention rows must be positive") unless rows > 0
        frontier = @last_end_position || span_start
        unless start == frontier
          raise ArgumentError.new("Inkling 2 resident attention speculative tiles must be contiguous from position #{frontier}")
        end
        end_position = start.to_i64 + rows
        span_end = span_start.to_i64 + @rollback_rows
        if end_position > span_end
          raise ArgumentError.new("Inkling 2 resident attention speculation exceeds #{@rollback_rows} reserved rows")
        end
      end

      private def validate_view!(view : Inkling2View, elements : Int32, name : String) : Nil
        bytes = elements.to_i64 * sizeof(Float32)
        unless view.offset >= 0 && view.offset % sizeof(Float32) == 0 && bytes <= view.buffer.size && view.offset <= view.buffer.size - bytes
          raise ArgumentError.new("Inkling 2 resident #{name} view exceeds or misaligns its Metal buffer")
        end
      end

      private def checked_elements(rows : Int32, width : Int32, name : String) : Int32
        product = rows.to_i64 * width
        raise ArgumentError.new("Inkling 2 resident #{name} exceeds Int32 elements") unless rows >= 0 && width >= 0 && product <= Int32::MAX
        product.to_i32
      end

      private def checked_bytes(rows : Int32, width : Int32, name : String) : Int64
        elements = rows.to_i64 * width
        raise ArgumentError.new("Inkling 2 resident #{name} exceeds Int64 bytes") unless rows > 0 && width > 0 && elements <= Int64::MAX // sizeof(Float32)
        elements * sizeof(Float32)
      end

      private def ensure_open! : Nil
        raise "Inkling 2 resident attention is closed" if @closed
      end

      private def pipeline(name : String) : ML::Metal::ComputePipeline
        ML::Metal::PipelineCache.get_or_compile(name, SOURCE)
      end
    end
  {% end %}
end
