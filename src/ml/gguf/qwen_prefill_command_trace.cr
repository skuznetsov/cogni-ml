module ML::GGUF
  # Opt-in host-side diagnostics for the ordinary shared prefill command.
  # This does not time GPU execution or cover CogniGraph/internal commands.
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

    private def emit(phase : String, fields : String, host_elapsed_ms : Float64? = nil)
      line = "qwen35_prefill_command phase=#{phase} #{fields}"
      line += " host_elapsed_ms=#{host_elapsed_ms.round(3)}" if host_elapsed_ms
      @io.puts(line)
      @io.flush
    rescue
      # Diagnostics must not prevent submission or mask a command exception.
    end
  end
end
