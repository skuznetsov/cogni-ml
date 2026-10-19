require "json"

module ML::Audio::Drama3::Studio
  # Sampled observations, not a reservation or a physical peak guarantee. The
  # process-group watchdog remains necessary while host code is not running.
  class MemoryGuard
    class Exceeded < Exception
    end

    getter limit_bytes : Int64?

    def initialize(@limit_bytes : Int64? = nil)
      raise ArgumentError.new("studio memory limit must be positive") if @limit_bytes && @limit_bytes.not_nil! <= 0
    end

    def self.from_env : self
      value = ENV["DRAMA3_STUDIO_MEMORY_LIMIT_MIB"]?
      return new unless value
      mib = value.to_i64?
      unless mib && mib > 0 && mib <= Int64::MAX // (1024 * 1024)
        raise ArgumentError.new("DRAMA3_STUDIO_MEMORY_LIMIT_MIB must be a positive integer")
      end
      new(mib * (1024 * 1024))
    end

    def check!(sample : JSON::Any, stage : String) : Nil
      physical = sample["physical_bytes"]?.try(&.as_i64?)
      free = sample["free_reclaimable_bytes"]?.try(&.as_i64?)
      footprint = sample["process_footprint_bytes"]?.try(&.as_i64?)
      metal = sample["metal_allocated_bytes"]?.try(&.as_i64?)
      unless sample["valid"]?.try(&.as_bool?) && physical && physical > 0 && free && free >= 0 && free <= physical &&
             footprint && footprint > 0 && metal && metal >= 0
        raise Exceeded.new("studio memory telemetry unavailable or invalid at #{stage}")
      end
      if free.to_i128 * 100 <= physical.to_i128 * 15
        raise Exceeded.new("studio free/reclaimable memory reached the 15% floor at #{stage}")
      end
      if limit = @limit_bytes
        raise Exceeded.new("studio process footprint reached memory budget at #{stage}") if footprint >= limit
        raise Exceeded.new("studio Metal allocation reached memory budget at #{stage}") if metal >= limit
      end
    end

    # These counters overlap. Use the smaller remaining allowance, never their
    # sum, to tighten advisory admission under an optional observed budget.
    def remaining_bytes(sample : JSON::Any) : Int64?
      @limit_bytes.try { |limit| limit - Math.max(sample["process_footprint_bytes"].as_i64, sample["metal_allocated_bytes"].as_i64) }
    end
  end
end
