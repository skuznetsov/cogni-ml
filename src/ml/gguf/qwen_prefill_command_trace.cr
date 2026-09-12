module ML::GGUF
  # Opt-in host-side diagnostics for the ordinary shared prefill command and
  # the ordinary routed full-layer call. Neither measures GPU execution.
  class QwenPrefillCommandTrace
    @sequence = 0_u64

    def self.enabled?(configured : String? = ENV["QWEN35_PREFILL_COMMAND_TRACE"]?) : Bool
      configured == "1"
    end

    def initialize(@io : IO = STDERR)
    end

    def observe(command_id : UInt64, start_pos : Int32, rows : Int32,
                cursor_before : Int32, cursor_at_flush : Int32, groups : Int32, &)
      @sequence += 1
      fields = "trace_id=#{object_id} sequence=#{@sequence} command_id=#{command_id} " \
               "start_pos=#{start_pos} rows=#{rows} " \
               "layer_cursor_before=#{cursor_before} layer_cursor_at_flush=#{cursor_at_flush} groups=#{groups}"
      emit("submit_wait_begin", fields)
      started = Time.instant
      begin
        result = yield
      rescue ex
        emit("submit_wait_failed", fields, (Time.instant - started).total_milliseconds)
        raise ex
      end
      emit("submit_wait_end", fields, (Time.instant - started).total_milliseconds)
      result
    end

    # The routed call may decline before encoding or fail during setup/readback.
    # Its exact layer is known, but no command identity or GPU phase is implied.
    def observe_full_layer(start_pos : Int32, rows : Int32, layer : Int32, &)
      @sequence += 1
      fields = "trace_id=#{object_id} sequence=#{@sequence} " \
               "route=full_attn_chunk_routed start_pos=#{start_pos} rows=#{rows} layer=#{layer}"
      emit("call_begin", fields, prefix: "qwen35_prefill_layer")
      started = Time.instant
      begin
        result = yield
      rescue ex
        emit("call_failed", fields, (Time.instant - started).total_milliseconds, prefix: "qwen35_prefill_layer")
        raise ex
      end
      phase = result.nil? ? "call_declined" : "call_end"
      emit(phase, fields, (Time.instant - started).total_milliseconds, prefix: "qwen35_prefill_layer")
      result
    end

    private def emit(phase : String, fields : String, host_elapsed_ms : Float64? = nil,
                     prefix : String = "qwen35_prefill_command")
      line = "#{prefix} phase=#{phase} #{fields}"
      line += " host_elapsed_ms=#{host_elapsed_ms.round(3)}" if host_elapsed_ms
      @io.puts(line)
      @io.flush
    rescue
      # Diagnostics must not prevent submission or mask a command exception.
    end
  end
end
