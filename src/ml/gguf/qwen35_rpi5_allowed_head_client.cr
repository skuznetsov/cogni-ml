module ML::GGUF
  # Probe-side protocol helpers for the Raspberry Pi 5 resident Q6 allowed-head
  # worker. This module deliberately owns only framing and result parsing; it
  # does not decide when the V3D route is legal for product decode.
  module Qwen35Rpi5AllowedHeadClient
    extend self

    record Request,
      hidden : Array(Float32),
      allowed_ids : Array(Int32)

    record Result,
      request : Int32,
      allowed : Int32,
      gpu_ms : Float64,
      cpu_ms : Float64,
      speedup : Float64,
      max_abs_diff : Float64,
      top1_match : Bool,
      gpu_top1_src : Int32,
      cpu_top1_src : Int32,
      gpu_top1_logit : Float64?,
      cpu_top1_logit : Float64?

    def write_binary_frame(io : IO,
                           hidden : Array(Float32),
                           allowed_ids : Array(Int32),
                           vocab_rows : Int32? = nil) : Nil
      raise ArgumentError.new("hidden must not be empty") if hidden.empty?
      validate_allowed_ids(allowed_ids, vocab_rows)

      io << "bin\t"
      allowed_ids.each_with_index do |id, i|
        io << ',' if i > 0
        io << id
      end
      io << '\n'
      hidden.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
      io << '\n'
    end

    def frame_bytes(hidden : Array(Float32),
                    allowed_ids : Array(Int32),
                    vocab_rows : Int32? = nil) : Bytes
      io = IO::Memory.new
      write_binary_frame(io, hidden, allowed_ids, vocab_rows)
      io.to_slice
    end

    def parse_result_line?(line : String) : Result?
      stripped = line.strip
      return nil unless stripped.starts_with?("resident_stdin_result\t")

      fields = {} of String => String
      stripped.split('\t')[1..].each do |field|
        key, value = field.split('=', 2)
        raise ArgumentError.new("malformed resident stdin field: #{field}") unless value
        fields[key] = value
      end

      Result.new(
        int_field(fields, "request"),
        int_field(fields, "allowed"),
        float_field(fields, "gpu_ms"),
        float_field(fields, "cpu_ms"),
        speedup_field(fields, "speedup"),
        float_field(fields, "max_abs_diff"),
        bool_field(fields, "top1_match"),
        int_field(fields, "gpu_top1_src"),
        int_field(fields, "cpu_top1_src"),
        optional_float_field(fields, "gpu_top1_logit"),
        optional_float_field(fields, "cpu_top1_logit"),
      )
    end

    def parse_results(output : String) : Array(Result)
      results = [] of Result
      output.each_line do |line|
        if result = parse_result_line?(line)
          results << result
        end
      end
      results
    end

    private def validate_allowed_ids(allowed_ids : Array(Int32),
                                     vocab_rows : Int32? = nil) : Nil
      raise ArgumentError.new("allowed_ids must not be empty") if allowed_ids.empty?
      allowed_ids.each do |id|
        if id < 0 || (vocab_rows && id >= vocab_rows)
          range = vocab_rows ? "0...#{vocab_rows}" : "non-negative"
          raise ArgumentError.new("allowed token id #{id} out of range #{range}")
        end
      end
    end

    private def int_field(fields : Hash(String, String), key : String) : Int32
      raw = required_field(fields, key)
      raw.to_i? || raise ArgumentError.new("resident stdin field #{key} is not an integer: #{raw}")
    end

    private def float_field(fields : Hash(String, String), key : String) : Float64
      raw = required_field(fields, key)
      raw.to_f64? || raise ArgumentError.new("resident stdin field #{key} is not a float: #{raw}")
    end

    private def optional_float_field(fields : Hash(String, String), key : String) : Float64?
      raw = fields[key]?
      return nil unless raw

      raw.to_f64? || raise ArgumentError.new("resident stdin field #{key} is not a float: #{raw}")
    end

    private def speedup_field(fields : Hash(String, String), key : String) : Float64
      raw = required_field(fields, key)
      normalized = raw.ends_with?('x') ? raw[0...-1] : raw
      normalized.to_f64? || raise ArgumentError.new("resident stdin field #{key} is not a speedup: #{raw}")
    end

    private def bool_field(fields : Hash(String, String), key : String) : Bool
      case raw = required_field(fields, key)
      when "true"  then true
      when "false" then false
      else
        raise ArgumentError.new("resident stdin field #{key} is not a bool: #{raw}")
      end
    end

    private def required_field(fields : Hash(String, String), key : String) : String
      fields[key]? || raise ArgumentError.new("resident stdin result missing #{key}")
    end
  end
end
