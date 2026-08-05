# Stateless CPU/F32 guidance-interval routing for one already prepared model
# timestep tensor. The caller owns the normalized_t/model_timesteps
# correspondence: this adapter neither validates that relationship nor
# constructs a timestep. It owns only raw Float64 interval routing and one
# delegation to the existing classifier-free guidance/rescale adapter.

require "./flow_classifier_free_guidance"

module ML::ThreeD::Trellis2
  module FlowGuidanceIntervalCPU
    extend self

    # Compare raw Float64 normalized time inclusively, then pass either the
    # requested strength or the exact outside override to CFG. CFG remains the
    # sole owner of tensor validation, provider calls, and CFG arithmetic.
    def predict_velocity(
      x_t : Tensor,
      model_timesteps : Tensor,
      positive_condition : C,
      negative_condition : C,
      normalized_t : Float64,
      requested_guidance_strength : Float64,
      guidance_interval : Tuple(Float64, Float64),
      max_result_bytes : Int64 = FlowClassifierFreeGuidanceCPU::MAX_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : Tensor forall C
      effective_guidance_strength = effective_guidance_strength!(
        normalized_t,
        requested_guidance_strength,
        guidance_interval
      )

      FlowClassifierFreeGuidanceCPU.predict_velocity(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        effective_guidance_strength,
        max_result_bytes: max_result_bytes,
        &velocity_provider
      )
    end

    # Route raw normalized Float64 time before delegating one CFG/rescale call.
    # The interval owns only the effective strength; CFG remains the owner of
    # rescale validation, arithmetic, provider ordering, and result storage.
    def predict_velocity_with_rescale(
      x_t : Tensor,
      model_timesteps : Tensor,
      positive_condition : C,
      negative_condition : C,
      sigma_min : Float32,
      normalized_t : Float64,
      requested_guidance_strength : Float64,
      guidance_interval : Tuple(Float64, Float64),
      guidance_rescale : Float64,
      max_result_bytes : Int64 = FlowClassifierFreeGuidanceCPU::MAX_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : Tensor forall C
      effective_guidance_strength = effective_guidance_strength!(
        normalized_t,
        requested_guidance_strength,
        guidance_interval
      )

      FlowClassifierFreeGuidanceCPU.predict_velocity_with_rescale(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        sigma_min: sigma_min,
        normalized_t: normalized_t,
        guidance_strength: effective_guidance_strength,
        guidance_rescale: guidance_rescale,
        max_result_bytes: max_result_bytes,
        &velocity_provider
      )
    end

    private def validate_request!(
      normalized_t : Float64,
      requested_guidance_strength : Float64,
      guidance_interval : Tuple(Float64, Float64),
    ) : Nil
      unless normalized_t.finite? && 0.0_f64 <= normalized_t <= 1.0_f64
        raise ArgumentError.new("normalized_t must be finite and in [0, 1]")
      end

      lower = guidance_interval[0]
      upper = guidance_interval[1]
      unless lower.finite? && upper.finite? &&
             0.0_f64 <= lower <= 1.0_f64 &&
             0.0_f64 <= upper <= 1.0_f64
        raise ArgumentError.new(
          "guidance_interval endpoints must be finite and in [0, 1]"
        )
      end
      unless lower <= upper
        raise ArgumentError.new("guidance_interval lower must be <= upper")
      end

      # Validate the requested strength before any outside override so the
      # local request contract cannot be bypassed by the positive-only route.
      unless requested_guidance_strength.finite?
        raise ArgumentError.new("guidance_strength must be finite")
      end
      max_f32 = Float32::MAX.to_f64
      unless requested_guidance_strength.abs <= max_f32 &&
             (1.0_f64 - requested_guidance_strength).abs <= max_f32
        raise ArgumentError.new(
          "guidance_strength must be representable in F32 CFG arithmetic"
        )
      end
    end

    private def effective_guidance_strength!(
      normalized_t : Float64,
      requested_guidance_strength : Float64,
      guidance_interval : Tuple(Float64, Float64),
    ) : Float64
      validate_request!(
        normalized_t,
        requested_guidance_strength,
        guidance_interval
      )

      if guidance_interval[0] <= normalized_t <= guidance_interval[1]
        requested_guidance_strength
      else
        1.0_f64
      end
    end
  end
end
