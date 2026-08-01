# Minimal CPU-only observe-before-compile seam for T2N2d1a.
#
# This is not the Metal PipelineCache. It proves only that the bounded ledger
# admits a full structured key before this adapter observes or compiles it.
# Its counters do not measure allocation, RSS, device memory, or process-wide
# artifact retention.

require "./device_resource_contract"

module ML::ThreeD::Trellis2
  class KernelCacheCancelledError < Exception
  end

  record KernelCacheDiagnostics,
    lookups : Int64,
    hits : Int64,
    misses : Int64,
    compile_attempts : Int64,
    capacity_refusals : Int64,
    entries : Int32

  class BoundedKernelCacheAdapterCPU
    CACHE_OWNER = "trellis2-dense-flow"
    LIFECYCLE   = "adapter-retained-no-eviction"

    getter capacity : Int32

    def initialize(
      @ledger : BoundedKernelKeyLedger,
      @capacity : Int32,
      @compiler : Proc(KernelSpecializationKey, String),
    )
      unless 0 < @capacity <= @ledger.capacity
        raise ArgumentError.new(
          "cache adapter capacity must be within 1..#{@ledger.capacity}"
        )
      end
      @mutex = Mutex.new
      @entries = {} of KernelSpecializationKey => String
      @lookups = 0_i64
      @hits = 0_i64
      @misses = 0_i64
      @compile_attempts = 0_i64
      @capacity_refusals = 0_i64
    end

    def fetch!(
      key : KernelSpecializationKey,
      cancelled : Proc(Bool)? = nil,
    ) : String
      raise_if_cancelled!(cancelled)
      unless key.cache_owner == CACHE_OWNER
        raise ArgumentError.new(
          "kernel key must use fixed cache owner #{CACHE_OWNER}"
        )
      end

      # No adapter counter or cache observation occurs before this returns.
      @ledger.admit!([key])
      raise_if_cancelled!(cancelled)

      @mutex.synchronize do
        @lookups += 1
        if artifact = @entries[key]?
          @hits += 1
          return artifact
        end

        @misses += 1
        if @entries.size >= @capacity
          @capacity_refusals += 1
          raise KernelCacheCapacityError.new(
            "cache adapter capacity #{@capacity} would be exceeded; no compilation attempted"
          )
        end

        @compile_attempts += 1
        # The controlled CPU fake compiles under this lock to make same-key
        # single-compilation observable. A re-entrant or blocking compiler and
        # real device compilation remain outside this adapter's admitted scope.
        artifact = @compiler.call(key)
        @entries[key] = artifact
        artifact
      end
    end

    def diagnostics : KernelCacheDiagnostics
      @mutex.synchronize do
        KernelCacheDiagnostics.new(
          @lookups,
          @hits,
          @misses,
          @compile_attempts,
          @capacity_refusals,
          @entries.size
        )
      end
    end

    def lifecycle : String
      LIFECYCLE
    end

    private def raise_if_cancelled!(cancelled : Proc(Bool)?) : Nil
      if cancelled && cancelled.call
        raise KernelCacheCancelledError.new(
          "kernel cache request cancelled before cache lookup"
        )
      end
    end
  end
end
