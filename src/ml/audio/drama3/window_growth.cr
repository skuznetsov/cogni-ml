require "json"
require "./window_budget"

module ML::Audio::Drama3
  # Pure CPU planning and fail-closed admission for geometrically growing
  # resident attention caches. Estimates remain advisory: a successful check
  # does not reserve host or Metal memory.
  module WindowGrowth
    extend self

    MAX_CAPACITY         =   4096
    INITIAL_CAPACITY     =     64
    FLOAT_BYTES          =  4_i64
    HOST_RESERVE_PERCENT = 15_i64

    enum AdmissionFailure
      UnknownTelemetry
      InvalidTelemetry
      InvalidRequest
      InsufficientHostBudget
      InsufficientMetalBudget
      InsufficientProcessBudget
      PerBufferLimit
      Overflow
    end

    struct CacheAllocation
      include JSON::Serializable

      getter capacity : Int32
      getter backbone_bytes : Int64
      getter tower_bytes : Int64
      getter score_scratch_bytes : Int64
      getter total_bytes : Int64
      getter largest_buffer_bytes : Int64

      def initialize(
        @capacity : Int32,
        @backbone_bytes : Int64,
        @tower_bytes : Int64,
        @score_scratch_bytes : Int64,
        @total_bytes : Int64,
        @largest_buffer_bytes : Int64,
      )
      end
    end

    struct Plan
      include JSON::Serializable

      getter frames : Int32
      getter initial_capacity : Int32
      getter final_capacity : Int32
      getter capacity_schedule : Array(Int32)
      getter generation_cache_bytes : Array(Int64)
      getter prior_generation_reservation_bytes : Int64
      getter window_plan : Drama3WindowBudget::Plan

      def initialize(
        @frames : Int32,
        @initial_capacity : Int32,
        @final_capacity : Int32,
        @capacity_schedule : Array(Int32),
        @generation_cache_bytes : Array(Int64),
        @prior_generation_reservation_bytes : Int64,
        @window_plan : Drama3WindowBudget::Plan,
      )
      end
    end

    struct PlanResult
      getter plan : Plan?
      getter failure : Drama3WindowBudget::Failure?

      def initialize(@plan : Plan?, @failure : Drama3WindowBudget::Failure?)
      end

      def success? : Bool
        !!@plan
      end

      def plan? : Plan?
        @plan
      end
    end

    # The report keeps each independent budget visible. Host and Metal values
    # overlap and must never be summed into a purported physical footprint.
    struct AdmissionReport
      include JSON::Serializable

      getter admitted : Bool
      getter failure : String?
      getter new_allocation_bytes : Int64
      getter downstream_reserve_bytes : Int64
      getter margin_bytes : Int64
      getter required_bytes : Int64?
      getter host_free_bytes : Int64?
      getter host_reserved_bytes : Int64?
      getter host_after_bytes : Int64?
      getter metal_recommended_bytes : Int64?
      getter metal_current_allocated_bytes : Int64?
      getter metal_remaining_bytes : Int64?
      getter process_remaining_bytes : Int64?
      getter process_after_bytes : Int64?
      getter max_buffer_length : Int64?
      getter largest_buffer_bytes : Int64

      def initialize(
        @admitted : Bool,
        @failure : String?,
        @new_allocation_bytes : Int64,
        @downstream_reserve_bytes : Int64,
        @margin_bytes : Int64,
        @required_bytes : Int64?,
        @host_free_bytes : Int64?,
        @host_reserved_bytes : Int64?,
        @host_after_bytes : Int64?,
        @metal_recommended_bytes : Int64?,
        @metal_current_allocated_bytes : Int64?,
        @metal_remaining_bytes : Int64?,
        @process_remaining_bytes : Int64?,
        @process_after_bytes : Int64?,
        @max_buffer_length : Int64?,
        @largest_buffer_bytes : Int64,
      )
      end
    end

    struct AdmissionResult
      getter admission : AdmissionReport?
      getter failure : AdmissionFailure?

      def initialize(@admission : AdmissionReport?, @failure : AdmissionFailure?)
      end

      def success? : Bool
        @admission.try(&.admitted) || false
      end

      def admission? : AdmissionReport?
        @admission
      end
    end

    private class Rejected < Exception
      getter failure : AdmissionFailure

      def initialize(@failure : AdmissionFailure)
      end
    end

    # Begin at the prompt-sized cache or 64 rows, then double geometrically and
    # clip the last step to the final capacity. Both capacities include prompt
    # rows and are bounded by the kernel's 4096-position limit.
    def capacity_schedule(prompt_rows : Int32, final_capacity : Int32) : Array(Int32)
      unless prompt_rows.in?(1..MAX_CAPACITY) && final_capacity.in?(1..MAX_CAPACITY) && prompt_rows <= final_capacity
        raise ArgumentError.new("prompt and final capacity must fit 1..4096, with prompt no larger than final capacity")
      end

      current = Math.min(final_capacity, Math.max(prompt_rows, INITIAL_CAPACITY))
      schedule = [] of Int32
      loop do
        schedule << current
        break if current == final_capacity
        current = Math.min(final_capacity, current * 2)
      end
      schedule
    end

    # Persistent K/V allocations are retained for every earlier generation.
    # If the final kernel is wide-attention, reserve its score scratch at every
    # capacity because kernel choice is pinned for the complete request.
    def cache_bytes_per_capacity(backbone : BackboneConfig, tower : TowerConfig,
                                 capacity : Int32, final_capacity : Int32) : CacheAllocation
      unless capacity.in?(1..MAX_CAPACITY) && final_capacity.in?(capacity..MAX_CAPACITY)
        raise ArgumentError.new("cache capacities must fit 1..4096 and final must cover this generation")
      end
      raise ArgumentError.new("invalid attention configuration") unless backbone.full_interval > 0
      full_layers = (backbone.layers // backbone.full_interval).to_i64
      raise ArgumentError.new("invalid attention configuration") unless full_layers >= 0
      raise ArgumentError.new("invalid cache configuration") unless backbone.kv_heads > 0 && backbone.head_dim > 0 && backbone.heads > 0
      raise ArgumentError.new("invalid tower cache configuration") unless tower.layer_count > 0 && tower.n_local_heads > 0 && tower.head_dim > 0

      positions = capacity.to_i64
      backbone_bytes = checked_product(
        positions, backbone.kv_heads.to_i64, backbone.head_dim.to_i64,
        2_i64, full_layers, FLOAT_BYTES,
      )
      tower_bytes = checked_product(
        2_i64, tower.layer_count.to_i64, positions,
        tower.n_local_heads.to_i64, tower.head_dim.to_i64, FLOAT_BYTES,
      )
      score_scratch_bytes = if final_capacity > 1024 && full_layers > 0
                              checked_product(positions, backbone.heads.to_i64, FLOAT_BYTES)
                            else
                              0_i64
                            end
      total_bytes = checked_add(checked_add(backbone_bytes, tower_bytes), score_scratch_bytes)

      largest_backbone = if full_layers > 0
                           checked_product(positions, backbone.kv_heads.to_i64, backbone.head_dim.to_i64, FLOAT_BYTES)
                         else
                           0_i64
                         end
      largest_tower = checked_product(positions, tower.n_local_heads.to_i64, tower.head_dim.to_i64, FLOAT_BYTES)
      largest_buffer = Math.max(largest_backbone, largest_tower)
      largest_buffer = Math.max(largest_buffer, score_scratch_bytes)
      CacheAllocation.new(capacity, backbone_bytes, tower_bytes, score_scratch_bytes, total_bytes, largest_buffer)
    end

    # Select the largest output cap for which the existing cold inventory plus
    # the complete geometric cache history fits both independent budgets.
    def plan(request : Drama3WindowBudget::Request,
             telemetry : Drama3WindowBudget::Telemetry) : PlanResult
      # Let the established planner validate the pinned configs and reject
      # unknown telemetry or an infeasible user minimum before binary search.
      baseline = Drama3WindowBudget.experimental_plan(request, telemetry)
      return PlanResult.new(nil, baseline.failure) unless baseline.success?

      prompt_rows = request.prompt_frames
      unless prompt_rows.in?(1..MAX_CAPACITY)
        return PlanResult.new(nil, Drama3WindowBudget::Failure::PromptExceedsPositionCap)
      end
      max_frames = Math.min(request.frame_cap, MAX_CAPACITY - prompt_rows + 1)
      if request.min_frames > max_frames
        return PlanResult.new(nil, Drama3WindowBudget::Failure::InsufficientFrames)
      end

      low = request.min_frames.to_i64
      high = max_frames.to_i64
      best : Plan? = nil
      last_failure = Drama3WindowBudget::Failure::InsufficientBudget
      while low <= high
        candidate_frames = (low + (high - low) // 2).to_i32
        candidate = candidate_plan(request, telemetry, candidate_frames)
        if plan = candidate.plan?
          best = plan
          low = candidate_frames.to_i64 + 1
        else
          last_failure = candidate.failure || Drama3WindowBudget::Failure::InsufficientBudget
          if last_failure.in?({Drama3WindowBudget::Failure::InvalidRequest,
                               Drama3WindowBudget::Failure::UnknownTelemetry,
                               Drama3WindowBudget::Failure::InvalidTelemetry})
            return PlanResult.new(nil, last_failure)
          end
          high = candidate_frames.to_i64 - 1
        end
      end
      best ? PlanResult.new(best, nil) : PlanResult.new(nil, last_failure)
    rescue OverflowError
      PlanResult.new(nil, Drama3WindowBudget::Failure::Overflow)
    end

    # Largest single cache allocation for the replacement generation. This is
    # distinct from total generation bytes and is checked against max-buffer.
    def cache_largest_buffer_bytes(backbone : BackboneConfig, tower : TowerConfig,
                                   capacity : Int32, final_capacity : Int32) : Int64
      cache_bytes_per_capacity(backbone, tower, capacity, final_capacity).largest_buffer_bytes
    end

    # Check an allocation before it is issued. `new_allocation_bytes` is the
    # complete replacement generation; downstream and margin bytes are
    # projected concurrently against host, Metal, and optional process limits.
    def admit_growth(new_allocation_bytes : Int64, downstream_reserve_bytes : Int64,
                     margin_bytes : Int64, telemetry : Drama3WindowBudget::Telemetry,
                     largest_buffer_bytes : Int64,
                     process_remaining_bytes : Int64? = nil) : AdmissionResult
      begin
        raise Rejected.new(AdmissionFailure::InvalidRequest) if new_allocation_bytes < 0 || downstream_reserve_bytes < 0 || margin_bytes < 0 || largest_buffer_bytes < 0
        raise Rejected.new(AdmissionFailure::InvalidRequest) if process_remaining_bytes && process_remaining_bytes.not_nil! < 0
        physical, free, recommended, allocated, max_buffer = read_telemetry!(telemetry)
        required = checked_add(checked_add(new_allocation_bytes, downstream_reserve_bytes), margin_bytes)
        host_reserved = ceil_percent(physical, HOST_RESERVE_PERCENT)
        host_after = free.to_i128 - required.to_i128
        metal_remaining = recommended - allocated
        process_after = process_remaining_bytes.try { |remaining| remaining - required }

        failure = if largest_buffer_bytes > max_buffer
                    AdmissionFailure::PerBufferLimit
                  elsif host_after * 100_i128 <= physical.to_i128 * 15_i128
                    AdmissionFailure::InsufficientHostBudget
                  elsif required > metal_remaining
                    AdmissionFailure::InsufficientMetalBudget
                  elsif process_remaining_bytes && required >= process_remaining_bytes.not_nil!
                    AdmissionFailure::InsufficientProcessBudget
                  else
                    nil
                  end
        report = admission_report(
          failure.nil?, failure.try(&.to_s), new_allocation_bytes, downstream_reserve_bytes,
          margin_bytes, required, free, host_reserved, host_after.to_i64,
          recommended, allocated, metal_remaining, process_remaining_bytes,
          process_after, max_buffer, largest_buffer_bytes,
        )
        AdmissionResult.new(report, failure)
      rescue ex : Rejected
        report = admission_report(
          false, ex.failure.to_s, new_allocation_bytes,
          downstream_reserve_bytes, margin_bytes, nil, telemetry.free_reclaimable_bytes,
          nil, nil, telemetry.metal_recommended_bytes,
          telemetry.metal_current_allocated_bytes, nil, process_remaining_bytes,
          nil, telemetry.max_buffer_length, largest_buffer_bytes,
        )
        AdmissionResult.new(report, ex.failure)
      rescue OverflowError
        report = admission_report(
          false, AdmissionFailure::Overflow.to_s, new_allocation_bytes,
          downstream_reserve_bytes, margin_bytes, nil, telemetry.free_reclaimable_bytes,
          nil, nil, telemetry.metal_recommended_bytes,
          telemetry.metal_current_allocated_bytes, nil, process_remaining_bytes,
          nil, telemetry.max_buffer_length, largest_buffer_bytes,
        )
        AdmissionResult.new(report, AdmissionFailure::Overflow)
      end
    end

    private def candidate_plan(request : Drama3WindowBudget::Request,
                               telemetry : Drama3WindowBudget::Telemetry,
                               frames : Int32) : PlanResult
      final_capacity = request.prompt_frames.to_i64 + frames - 1
      return PlanResult.new(nil, Drama3WindowBudget::Failure::PromptExceedsPositionCap) if final_capacity > MAX_CAPACITY
      schedule = capacity_schedule(request.prompt_frames, final_capacity.to_i32)
      allocations = schedule.map do |capacity|
        cache_bytes_per_capacity(request.backbone_config, request.tower_config, capacity, final_capacity.to_i32)
      end
      history = 0_i64
      (0...allocations.size - 1).each do |index|
        history = checked_add(history, allocations[index].total_bytes)
      end
      margin = checked_add(request.additional_margin_bytes, history)
      recommended = telemetry.metal_recommended_bytes.not_nil!
      allocated = telemetry.metal_current_allocated_bytes.not_nil!
      return PlanResult.new(nil, Drama3WindowBudget::Failure::InsufficientBudget) if history >= recommended - allocated

      adjusted_telemetry = Drama3WindowBudget::Telemetry.new(
        telemetry.physical_bytes,
        telemetry.free_reclaimable_bytes,
        recommended - history,
        allocated,
        telemetry.max_buffer_length,
      )
      fixed_request = Drama3WindowBudget::Request.new(
        request.backbone_config,
        request.tower_config,
        request.codec_config,
        request.prompt_frames,
        frames,
        frames,
        request.checkpoint_mapped_bytes,
        request.largest_checkpoint_mapped_buffer_bytes,
        request.converted_static_bytes,
        request.largest_converted_buffer_bytes,
        margin,
        request.moe_intermediate_size,
        request.shared_expert_intermediate_size,
        request.reference_encoder_peak_bytes,
      )
      result = Drama3WindowBudget.experimental_plan(fixed_request, adjusted_telemetry)
      return PlanResult.new(nil, result.failure) unless plan = result.plan?

      generation_cache_bytes = allocations.map(&.total_bytes)
      growth_plan = Plan.new(
        frames,
        schedule.first,
        final_capacity.to_i32,
        schedule,
        generation_cache_bytes,
        history,
        plan,
      )
      PlanResult.new(growth_plan, nil)
    rescue OverflowError
      PlanResult.new(nil, Drama3WindowBudget::Failure::Overflow)
    rescue ex : ArgumentError
      PlanResult.new(nil, Drama3WindowBudget::Failure::InvalidRequest)
    end

    private def read_telemetry!(telemetry : Drama3WindowBudget::Telemetry) : Tuple(Int64, Int64, Int64, Int64, Int64)
      physical = telemetry.physical_bytes || raise Rejected.new(AdmissionFailure::UnknownTelemetry)
      free = telemetry.free_reclaimable_bytes || raise Rejected.new(AdmissionFailure::UnknownTelemetry)
      recommended = telemetry.metal_recommended_bytes || raise Rejected.new(AdmissionFailure::UnknownTelemetry)
      allocated = telemetry.metal_current_allocated_bytes || raise Rejected.new(AdmissionFailure::UnknownTelemetry)
      max_buffer = telemetry.max_buffer_length || raise Rejected.new(AdmissionFailure::UnknownTelemetry)
      unless physical > 0 && free >= 0 && free <= physical && recommended > 0 && allocated >= 0 && allocated <= recommended && max_buffer > 0
        raise Rejected.new(AdmissionFailure::InvalidTelemetry)
      end
      {physical, free, recommended, allocated, max_buffer}
    end

    private def admission_report(admitted : Bool, failure : String?, new_allocation_bytes : Int64,
                                 downstream_reserve_bytes : Int64, margin_bytes : Int64,
                                 required_bytes : Int64?, host_free_bytes : Int64?, host_reserved_bytes : Int64?,
                                 host_after_bytes : Int64?, metal_recommended_bytes : Int64?,
                                 metal_current_allocated_bytes : Int64?, metal_remaining_bytes : Int64?,
                                 process_remaining_bytes : Int64?, process_after_bytes : Int64?,
                                 max_buffer_length : Int64?, largest_buffer_bytes : Int64) : AdmissionReport
      AdmissionReport.new(
        admitted, failure, new_allocation_bytes, downstream_reserve_bytes,
        margin_bytes, required_bytes, host_free_bytes, host_reserved_bytes,
        host_after_bytes, metal_recommended_bytes, metal_current_allocated_bytes,
        metal_remaining_bytes, process_remaining_bytes, process_after_bytes,
        max_buffer_length, largest_buffer_bytes,
      )
    end

    private def ceil_percent(value : Int64, percent : Int64) : Int64
      ((value.to_i128 * percent + 99_i128) // 100_i128).to_i64
    end

    private def checked_product(*factors : Int64) : Int64
      product = 1_i64
      factors.each do |factor|
        raise ArgumentError.new("negative cache size input") if factor < 0
        if factor != 0 && product > Int64::MAX // factor
          raise OverflowError.new("window cache size overflow")
        end
        product *= factor
      end
      product
    end

    private def checked_add(left : Int64, right : Int64) : Int64
      if right < 0 || left < 0
        raise ArgumentError.new("negative byte count")
      end
      raise OverflowError.new("window byte count overflow") if left > Int64::MAX - right
      left + right
    end
  end
end
