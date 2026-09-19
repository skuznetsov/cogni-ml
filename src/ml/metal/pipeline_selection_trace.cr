module ML::Metal
  # Host-side binding telemetry, not evidence of GPU dispatch or completion.
  # Configuration is captured once; unset/empty disables all output.
  module PipelineSelectionTrace
    PREFIX = ENV["COGNI_METAL_PIPELINE_TRACE_PREFIX"]?

    def self.record(command_handle : UInt64, encoder_handle : UInt64,
                    pipeline : String, prefix : String? = PREFIX, io : IO = STDERR) : Nil
      return unless prefix && !prefix.empty? && pipeline.starts_with?(prefix)

      io.puts "metal_pipeline phase=selected command_handle=#{command_handle} encoder_handle=#{encoder_handle} pipeline=#{pipeline.inspect}"
      io.flush
    rescue
      # A diagnostic sink failure must not replace the encoding outcome.
    end
  end
end
