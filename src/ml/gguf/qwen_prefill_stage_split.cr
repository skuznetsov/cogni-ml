module ML::GGUF
  # Diagnostic scheduling only: the caller owns ended encoders and creates a
  # successor command only after finish succeeds. Failed state is not reusable.
  class QwenPrefillStageSplit
    enum Stage
      PrepareKV
      Attention
      OutputFFN
    end

    @sequence = 0_u64

    def self.enabled?(standalone : Bool, ordinary_f32 : Bool,
                      configured : String? = ENV["QWEN35_FULL_PREFILL_STAGE_SPLIT"]?) : Bool
      (configured == "1" || configured == "after_attention") && standalone && ordinary_f32
    end

    def self.build(standalone : Bool, ordinary_f32 : Bool, start_pos : Int32, rows : Int32,
                   configured : String? = ENV["QWEN35_FULL_PREFILL_STAGE_SPLIT"]?, *, io : IO = STDERR) : self?
      return nil unless enabled?(standalone, ordinary_f32, configured)
      new(start_pos, rows, io, combine_prepare_attention: configured == "after_attention")
    end

    def initialize(@start_pos : Int32, @rows : Int32, @io : IO = STDERR,
                   *, @combine_prepare_attention : Bool = false)
    end

    # False means the caller must retain the current, still-unsubmitted command.
    # A true result certifies a successful wait, never permission to retry failure.
    def finish(command, stage : Stage) : Bool
      return false if @combine_prepare_attention && stage.prepare_kv?
      @sequence += 1
      name = case stage
             when .prepare_kv? then "prepare_kv"
             when .attention?  then @combine_prepare_attention ? "prepare_kv_attention" : "attention"
             else                   "output_ffn"
             end
      fields = "trace_id=#{object_id} sequence=#{@sequence} command_id=#{command.object_id} " \
               "stage=#{name} start_pos=#{@start_pos} rows=#{@rows}"
      emit("submit_wait_begin", fields)
      started = Time.instant
      begin
        command.commit
        command.wait
      rescue ex
        emit("submit_wait_failed", fields, (Time.instant - started).total_milliseconds)
        raise ex
      end
      emit("submit_wait_end", fields, (Time.instant - started).total_milliseconds)
      true
    end

    private def emit(phase : String, fields : String, host_elapsed_ms : Float64? = nil)
      line = "qwen35_prefill_stage phase=#{phase} #{fields}"
      line += " host_elapsed_ms=#{host_elapsed_ms.round(3)}" if host_elapsed_ms
      @io.puts(line)
      @io.flush
    rescue
      # Logging must not prevent submission or replace the original exception.
    end
  end
end
