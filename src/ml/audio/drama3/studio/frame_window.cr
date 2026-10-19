require "../window_budget"
require "../window_growth"

module ML::Audio::Drama3::Studio
  module FrameWindow
    extend self

    record Selection,
      frames : Int32,
      capacity : Int32,
      maximum_capacity : Int32,
      plan : Drama3WindowBudget::Plan? = nil,
      growth_plan : WindowGrowth::Plan? = nil,
      growth_reservation_bytes : Int64 = 0_i64

    def manual(prompt_rows : Int32, frames : Int32) : Selection
      raise ArgumentError.new("invalid frame bound") unless frames.in?(1..4096)
      capacity = prompt_rows.to_i64 + frames - 1
      unless prompt_rows > 0 && capacity.in?(1_i64..4096_i64)
        raise ArgumentError.new("prompt plus frame bound exceeds 4096 positions")
      end
      Selection.new(frames, capacity.to_i32, capacity.to_i32)
    end

    def automatic(request : Drama3WindowBudget::Request, telemetry : Drama3WindowBudget::Telemetry,
                  reference : Bool, dac_scratch : Bool) : Selection
      raise ArgumentError.new("automatic windows with voice references are not yet calibrated; choose manual") if reference
      raise ArgumentError.new("automatic windows require DRAMA3_DAC_SCRATCH=1; choose manual") unless dac_scratch
      result = WindowGrowth.plan(request, telemetry)
      growth = result.plan? || raise ArgumentError.new("automatic window rejected: #{result.failure}")
      Selection.new(
        growth.frames,
        growth.initial_capacity,
        growth.final_capacity,
        growth.window_plan,
        growth,
        growth.prior_generation_reservation_bytes,
      )
    end
  end
end
