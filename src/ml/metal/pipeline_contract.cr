require "digest/sha256"

module ML::Metal
  class PipelineCacheCapacityError < Exception
  end

  class PipelineCacheBuildInFlightError < Exception
  end

  # Code identity for one compiled Metal pipeline. Runtime sequence/spatial
  # lengths belong to a stage/dispatch plan unless they actually change source
  # or function-constant specialization.
  struct PipelineSpecializationKey
    getter owner : String
    getter function_name : String
    getter source_digest : String
    getter compiler_abi : String
    getter device_family : String
    getter code_variant : String

    def initialize(
      @owner : String,
      @function_name : String,
      @source_digest : String,
      @compiler_abi : String,
      @device_family : String,
      @code_variant : String,
    )
      validate_cache_token!(@owner, "owner")
      unless @function_name.matches?(/\A[A-Za-z_][A-Za-z0-9_]{0,127}\z/)
        raise ArgumentError.new(
          "function_name must be a Metal identifier of at most 128 characters"
        )
      end
      unless @source_digest.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new(
          "source_digest must be 64 lowercase hexadecimal characters"
        )
      end
      validate_cache_token!(@compiler_abi, "compiler_abi")
      validate_cache_token!(@device_family, "device_family")
      validate_cache_token!(@code_variant, "code_variant")
    end

    def self.for_source(
      owner : String,
      function_name : String,
      source : String,
      compiler_abi : String,
      device_family : String,
      code_variant : String,
    ) : PipelineSpecializationKey
      new(
        owner: owner,
        function_name: function_name,
        source_digest: Digest::SHA256.hexdigest(source),
        compiler_abi: compiler_abi,
        device_family: device_family,
        code_variant: code_variant
      )
    end

    def validate_source!(source : String) : Nil
      actual = Digest::SHA256.hexdigest(source)
      return if actual == @source_digest
      raise ArgumentError.new(
        "pipeline source digest mismatch for #{@function_name}: " \
        "expected #{@source_digest}, got #{actual}"
      )
    end

    def canonical : String
      "metal-pipeline/v1/owner-#{@owner}/fn-#{@function_name}/" \
      "src-#{@source_digest}/compiler-#{@compiler_abi}/" \
      "device-#{@device_family}/variant-#{@code_variant}"
    end

    def_equals_and_hash @owner, @function_name, @source_digest, @compiler_abi,
      @device_family, @code_variant

    private def validate_cache_token!(value : String, label : String) : Nil
      unless value.matches?(/\A[a-z0-9][a-z0-9._-]{0,127}\z/)
        raise ArgumentError.new(
          "#{label} must be a lowercase ASCII cache token of at most 128 characters"
        )
      end
    end
  end

  record PipelineCacheDiagnostics,
    owner : String,
    capacity : Int32,
    lookups : Int64,
    hits : Int64,
    misses : Int64,
    compile_attempts : Int64,
    compile_failures : Int64,
    capacity_refusals : Int64,
    in_flight_refusals : Int64,
    entries : Int32,
    in_flight : Int32,
    high_water : Int32

  record PipelineCacheProcessDiagnostics,
    owners : Int32,
    declared_capacity : Int32,
    max_owners : Int32,
    max_declared_capacity : Int32

  # Owner-scoped, fail-closed cache for newly admitted pipeline families.
  # Builders run outside the mutex; a reservation prevents duplicate or
  # re-entrant compilation and is removed transactionally on failure. Entries
  # have bounded process-lifetime retention; this contract exposes no eviction.
  class BoundedPipelineCache(V)
    getter owner : String
    getter capacity : Int32

    def initialize(@owner : String, @capacity : Int32)
      unless @owner.matches?(/\A[a-z0-9][a-z0-9._-]{0,127}\z/)
        raise ArgumentError.new(
          "owner must be a lowercase ASCII cache token of at most 128 characters"
        )
      end
      raise ArgumentError.new("pipeline cache capacity must be positive") unless @capacity > 0

      @mutex = Mutex.new
      @entries = {} of PipelineSpecializationKey => V
      @in_flight = {} of PipelineSpecializationKey => Bool
      @lookups = 0_i64
      @hits = 0_i64
      @misses = 0_i64
      @compile_attempts = 0_i64
      @compile_failures = 0_i64
      @capacity_refusals = 0_i64
      @in_flight_refusals = 0_i64
      @high_water = 0_i32
    end

    def fetch(key : PipelineSpecializationKey, &builder : -> V) : V
      unless key.owner == @owner
        raise ArgumentError.new(
          "pipeline key #{key.canonical} is not owned by #{@owner}"
        )
      end

      cached = nil.as(V?)
      build = false
      @mutex.synchronize do
        @lookups += 1
        if @entries.has_key?(key)
          @hits += 1
          cached = @entries[key]
        else
          @misses += 1
          if @in_flight.has_key?(key)
            @in_flight_refusals += 1
            raise PipelineCacheBuildInFlightError.new(
              "pipeline build already in flight for #{key.canonical}"
            )
          end
          if @entries.size + @in_flight.size >= @capacity
            @capacity_refusals += 1
            raise PipelineCacheCapacityError.new(
              "pipeline cache capacity #{@capacity} for owner #{@owner} would be exceeded"
            )
          end

          @in_flight[key] = true
          @compile_attempts += 1
          occupancy = @entries.size + @in_flight.size
          @high_water = occupancy if occupancy > @high_water
          build = true
        end
      end

      return cached.not_nil! unless build

      begin
        value = builder.call
      rescue ex
        @mutex.synchronize do
          @in_flight.delete(key)
          @compile_failures += 1
        end
        raise ex
      end

      @mutex.synchronize do
        @in_flight.delete(key)
        @entries[key] = value
      end
      value
    end

    def diagnostics : PipelineCacheDiagnostics
      @mutex.synchronize do
        PipelineCacheDiagnostics.new(
          owner: @owner,
          capacity: @capacity,
          lookups: @lookups,
          hits: @hits,
          misses: @misses,
          compile_attempts: @compile_attempts,
          compile_failures: @compile_failures,
          capacity_refusals: @capacity_refusals,
          in_flight_refusals: @in_flight_refusals,
          entries: @entries.size,
          in_flight: @in_flight.size,
          high_water: @high_water
        )
      end
    end
  end
end
