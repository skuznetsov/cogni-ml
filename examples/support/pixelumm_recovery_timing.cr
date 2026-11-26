# Sleep is allowed to return early. Admit a treatment only after the same
# monotonic clock used by the receipt reaches its minimum deadline.
module PixelUMMRecoveryTiming
  NANOSECONDS = 1_000_000_000_u64

  struct Clock
    lib Native
      fun clock_ns = clock_gettime_nsec_np(clock_id : UInt32) : UInt64
    end

    def now_ns : UInt64
      Native.clock_ns(4_u32) # CLOCK_MONOTONIC_RAW, also used by Crystal Time.instant
    end

    def adjusted_now_ns : UInt64
      Native.clock_ns(6_u32) # CLOCK_MONOTONIC, advisory cross-clock diagnostic
    end

    def pause(seconds : Float64) : Nil
      sleep seconds.seconds
    end
  end

  def self.wait_until(clock, prior_terminal : UInt64, seconds : Int32) : Nil
    raise ArgumentError.new("recovery must be positive") unless seconds > 0
    deadline = prior_terminal + seconds.to_u64 * NANOSECONDS
    loop do
      now = clock.now_ns
      break if now >= deadline
      clock.pause(Math.min((deadline - now) / 1e9, 30.0))
    end
  end
end
