# Bounded CPU/F32 composition of the admitted FlowEuler loop and CFG/rescale
# step. This delegates schedule, state ownership, timestep construction,
# arithmetic, validation, and retained history to their existing leaves.
# It does not own guidance intervals, RNG, pipeline defaults, stages, weights,
# model execution, GPU, Metal, or aggregate process-memory policy.

require "./flow_sampler"
require "./flow_classifier_free_guidance"

module ML::ThreeD::Trellis2
  module FlowClassifierFreeGuidanceSamplerCPU
    extend self

    # The base sampler's retained-history cap and the CFG step's two-result
    # cap remain separate. `max_result_bytes` is intentionally passed to both
    # leaves; it does not claim a bound for provider-owned or aggregate memory.
    def sample(
      noise : Tensor,
      positive_condition : C,
      negative_condition : C,
      sigma_min : Float32,
      steps : Int32 = FlowEulerSamplerCPU::DEFAULT_STEPS,
      rescale_t : Float64 = FlowEulerSamplerCPU::DEFAULT_RESCALE_T,
      guidance_strength : Float64 = FlowEulerSamplerCPU::DEFAULT_GUIDANCE_STRENGTH,
      guidance_rescale : Float64 = 0.0_f64,
      max_result_bytes : Int64 = FlowEulerSamplerCPU::MAX_RETAINED_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : FlowEulerSampleResultCPU forall C
      FlowEulerSamplerCPU.sample_with_step_provider(
        noise,
        positive_condition,
        sigma_min: sigma_min,
        steps: steps,
        rescale_t: rescale_t,
        max_result_bytes: max_result_bytes
      ) do |actual_x, normalized_t, _t_prev, model_timesteps, _actual_condition|
        FlowClassifierFreeGuidanceCPU.predict_velocity_with_rescale(
          actual_x,
          model_timesteps,
          positive_condition,
          negative_condition,
          sigma_min: sigma_min,
          normalized_t: normalized_t,
          guidance_strength: guidance_strength,
          guidance_rescale: guidance_rescale,
          max_result_bytes: max_result_bytes
        ) do |provider_x, provider_model_t, provider_condition|
          velocity_provider.call(provider_x, provider_model_t, provider_condition)
        end
      end
    end
  end
end
