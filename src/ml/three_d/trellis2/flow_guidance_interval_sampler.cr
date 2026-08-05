# Bounded CPU/F32 composition of raw Float64 guidance intervals with the
# admitted repeated CFG/rescale FlowEuler loop. The base sampler owns schedule,
# state, model timestep, retained history, and request validation. The interval
# seam owns only effective-strength routing; CFG owns provider calls, rescale
# arithmetic, and per-step output storage. No RNG, pipeline defaults, stages,
# weights, model execution, GPU, Metal, or aggregate memory policy is added.

require "./flow_sampler"
require "./flow_guidance_interval"

module ML::ThreeD::Trellis2
  module FlowGuidanceIntervalSamplerCPU
    extend self

    # TRELLIS.2 source pin: 75fbf0183001ed9876c8dbb35de6b68552ee08bd.
    # Upstream FlowEulerGuidanceIntervalSampler.sample defaults to the full
    # normalized interval [0.0, 1.0], inclusive at both endpoints.
    DEFAULT_GUIDANCE_INTERVAL = {0.0_f64, 1.0_f64}

    def sample(
      noise : Tensor,
      positive_condition : C,
      negative_condition : C,
      sigma_min : Float32,
      steps : Int32 = FlowEulerSamplerCPU::DEFAULT_STEPS,
      rescale_t : Float64 = FlowEulerSamplerCPU::DEFAULT_RESCALE_T,
      guidance_strength : Float64 = FlowEulerSamplerCPU::DEFAULT_GUIDANCE_STRENGTH,
      guidance_interval : Tuple(Float64, Float64) = DEFAULT_GUIDANCE_INTERVAL,
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
      ) do |actual_x, normalized_t, _t_prev, model_timesteps, actual_condition|
        FlowGuidanceIntervalCPU.predict_velocity_with_rescale(
          actual_x,
          model_timesteps,
          actual_condition,
          negative_condition,
          sigma_min: sigma_min,
          normalized_t: normalized_t,
          requested_guidance_strength: guidance_strength,
          guidance_interval: guidance_interval,
          guidance_rescale: guidance_rescale,
          max_result_bytes: max_result_bytes
        ) do |provider_x, provider_model_t, provider_condition|
          velocity_provider.call(
            provider_x,
            provider_model_t,
            provider_condition
          )
        end
      end
    end
  end
end
