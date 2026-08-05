# Bounded CPU/F32 arithmetic for one TRELLIS.2 base-flow Euler step.
#
# This leaf consumes an already computed velocity and normalized state times.
# It does not invoke a model, construct the model timestep `1000*t`, own a
# schedule or RNG, or implement CFG. Callers must not mutate input tensors
# concurrently while this synchronous operation holds borrowed CPU reads. A
# velocity-provider callback must not mutate x_t during its invocation.

require "../../core/tensor"

module ML::ThreeD::Trellis2
  # Shared scalar source expressions for normalized Float64 schedule time.
  # Callers own tensor/scalar admission and finite-result policy. Keeping these
  # methods private lets the Euler step and CFG rescale share exact arithmetic
  # without widening the public runtime API.
  private module FlowEulerStateConversionCPU
    private def flow_euler_schedule_coefficients(
      sigma_min : Float32,
      normalized_t : Float64,
    ) : Tuple(Float32, Float32)
      one_minus_sigma = (1.0_f32 - sigma_min).to_f32
      noise_scale = (
        sigma_min.to_f64 +
        (1.0_f64 - sigma_min.to_f64) * normalized_t
      ).to_f32
      {one_minus_sigma, noise_scale}
    end

    private def flow_euler_pred_to_xstart(
      x_t : Float32,
      prediction : Float32,
      one_minus_sigma : Float32,
      noise_scale : Float32,
    ) : Float32
      (one_minus_sigma * x_t - noise_scale * prediction).to_f32
    end

    private def flow_euler_xstart_to_pred(
      x_t : Float32,
      x_start : Float32,
      one_minus_sigma : Float32,
      noise_scale : Float32,
    ) : Float32
      numerator = (one_minus_sigma * x_t - x_start).to_f32
      (numerator / noise_scale).to_f32
    end
  end

  class FlowEulerStepResultCPU
    getter pred_x_prev : Tensor
    getter pred_x_0 : Tensor

    def initialize(@pred_x_prev : Tensor, @pred_x_0 : Tensor)
    end
  end

  class FlowEulerSampleResultCPU
    getter samples : Tensor
    getter pred_x_t : Array(Tensor)
    getter pred_x_0 : Array(Tensor)

    def initialize(
      @samples : Tensor,
      @pred_x_t : Array(Tensor),
      @pred_x_0 : Array(Tensor),
    )
    end
  end

  module FlowEulerStepCPU
    extend self
    include FlowEulerStateConversionCPU

    # Combined F32 payload of the two owned result tensors. This does not claim
    # an aggregate process or stage working-set bound.
    MAX_RESULT_BYTES = 64_i64 * 1024_i64 * 1024_i64

    # Apply the source formula inside a deliberately stricter Cogni admission
    # policy. Pinned upstream checks shape equality only; this public boundary
    # additionally requires bounded, finite, normalized, forward-time CPU/F32
    # inputs with canonical storage.
    def sample_once(
      x_t : Tensor,
      pred_v : Tensor,
      sigma_min : Float32,
      t : Float32,
      t_prev : Float32,
      max_result_bytes : Int64 = MAX_RESULT_BYTES,
    ) : FlowEulerStepResultCPU
      validate_result_budget!(max_result_bytes)
      validate_tensor!(x_t, "x_t")
      validate_tensor!(pred_v, "pred_v")
      unless x_t.shape == pred_v.shape
        raise ArgumentError.new(
          "x_t and pred_v must have the same shape, got #{x_t.shape} and #{pred_v.shape}"
        )
      end
      raise ArgumentError.new("flow Euler inputs must be non-empty") if x_t.numel == 0

      validate_scalars!(sigma_min, t, t_prev)
      validate_result_capacity!(x_t, max_result_bytes)

      x_values = borrowed_values!(x_t, "x_t")
      velocity_values = borrowed_values!(pred_v, "pred_v")
      finite_values!(x_values, "x_t")
      finite_values!(velocity_values, "pred_v")

      pred_x_prev = Tensor.new(
        x_t.shape,
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      pred_x_0 = Tensor.new(
        x_t.shape,
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      apply_source_formula!(
        x_values,
        velocity_values,
        sigma_min,
        t,
        t_prev,
        pred_x_prev.cpu_data.not_nil!,
        pred_x_0.cpu_data.not_nil!
      )
      FlowEulerStepResultCPU.new(pred_x_prev, pred_x_0)
    end

    # Construct only the source-required model timestep and invoke one caller-
    # supplied velocity provider before delegating all state arithmetic to the
    # admitted explicit-velocity leaf above. x_t is forwarded by identity; the
    # opaque condition is forwarded unchanged and preserves identity for
    # reference carriers. Provider exceptions are propagated without retry.
    # The result cap excludes this small timestep tensor and all provider-owned
    # memory; it remains only the combined payload cap for the two outputs.
    def sample_once_with_velocity_provider(
      x_t : Tensor,
      cond : C,
      sigma_min : Float32,
      t : Float32,
      t_prev : Float32,
      max_result_bytes : Int64 = MAX_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : FlowEulerStepResultCPU forall C
      # Preflight every request-owned invariant and the output pair before the
      # provider can perform expensive work. The input payload is borrowed and
      # checked without a copy, then checked again by sample_once after the
      # provider returns in case the callback mutated x_t synchronously.
      validate_result_budget!(max_result_bytes)
      validate_tensor!(x_t, "x_t")
      raise ArgumentError.new("flow Euler inputs must be non-empty") if x_t.numel == 0
      validate_scalars!(sigma_min, t, t_prev)
      validate_result_capacity!(x_t, max_result_bytes)
      finite_values!(borrowed_values!(x_t, "x_t"), "x_t")

      batch = x_t.shape[0]
      model_t = (1000.0_f32 * t).to_f32
      model_timesteps = Tensor.new(
        Shape.new(batch),
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      model_timesteps.cpu_data.not_nil!.fill(model_t)
      pred_v = velocity_provider.call(x_t, model_timesteps, cond)
      sample_once(
        x_t,
        pred_v,
        sigma_min: sigma_min,
        t: t,
        t_prev: t_prev,
        max_result_bytes: max_result_bytes
      )
    end

    # Preserve the pinned source boundary for one pair owned by a Float64
    # schedule. The model timestep, schedule delta, and time-dependent x0
    # coefficient narrow only after their Python/NumPy Float64 scalar
    # expressions are evaluated. sigma_min keeps the admitted Float32 API
    # boundary. This overload leaves that explicit one-step API intact.
    def sample_once_with_velocity_provider(
      x_t : Tensor,
      cond : C,
      sigma_min : Float32,
      t : Float64,
      t_prev : Float64,
      max_result_bytes : Int64 = MAX_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : FlowEulerStepResultCPU forall C
      validate_result_budget!(max_result_bytes)
      validate_tensor!(x_t, "x_t")
      raise ArgumentError.new("flow Euler inputs must be non-empty") if x_t.numel == 0
      validate_schedule_scalars!(sigma_min, t, t_prev)
      validate_result_capacity!(x_t, max_result_bytes)
      finite_values!(borrowed_values!(x_t, "x_t"), "x_t")

      batch = x_t.shape[0]
      model_t = (1000.0_f64 * t).to_f32
      model_timesteps = Tensor.new(
        Shape.new(batch),
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      model_timesteps.cpu_data.not_nil!.fill(model_t)
      pred_v = velocity_provider.call(x_t, model_timesteps, cond)
      sample_once_from_schedule(
        x_t,
        pred_v,
        sigma_min: sigma_min,
        t: t,
        t_prev: t_prev,
        max_result_bytes: max_result_bytes
      )
    end

    private def validate_result_budget!(max_result_bytes : Int64) : Nil
      unless 0_i64 < max_result_bytes <= MAX_RESULT_BYTES
        raise ArgumentError.new(
          "max_result_bytes must be positive and no greater than #{MAX_RESULT_BYTES}"
        )
      end
    end

    private def validate_result_capacity!(x_t : Tensor, max_result_bytes : Int64) : Nil
      required_result_bytes = x_t.numel.to_i64 * 2_i64 * DType::F32.byte_size.to_i64
      if required_result_bytes > max_result_bytes
        raise ArgumentError.new(
          "flow Euler result budget requires #{required_result_bytes} bytes but " \
          "max_result_bytes is #{max_result_bytes}"
        )
      end
    end

    private def validate_tensor!(tensor : Tensor, name : String) : Nil
      raise ArgumentError.new("#{name} must be on CPU") unless tensor.on_cpu?
      raise ArgumentError.new("#{name} must use F32") unless tensor.dtype.f32?
      unless tensor.contiguous?
        raise ArgumentError.new("#{name} must be contiguous; implicit materialization is not admitted")
      end
    end

    private def validate_scalars!(sigma_min : Float32, t : Float32, t_prev : Float32) : Nil
      unless sigma_min.finite? && 0.0_f32 <= sigma_min < 1.0_f32
        raise ArgumentError.new("sigma_min must be finite and in [0, 1)")
      end
      unless t.finite? && 0.0_f32 <= t <= 1.0_f32
        raise ArgumentError.new("normalized t must be finite and in [0, 1]")
      end
      unless t_prev.finite? && 0.0_f32 <= t_prev <= 1.0_f32
        raise ArgumentError.new("normalized t_prev must be finite and in [0, 1]")
      end
      unless t_prev < t
        raise ArgumentError.new("normalized t_prev must be strictly less than normalized t")
      end
    end

    private def validate_schedule_scalars!(
      sigma_min : Float32,
      t : Float64,
      t_prev : Float64,
    ) : Nil
      unless sigma_min.finite? && 0.0_f32 <= sigma_min < 1.0_f32
        raise ArgumentError.new("sigma_min must be finite and in [0, 1)")
      end
      unless t.finite? && 0.0_f64 <= t <= 1.0_f64
        raise ArgumentError.new("normalized t must be finite and in [0, 1]")
      end
      unless t_prev.finite? && 0.0_f64 <= t_prev <= 1.0_f64
        raise ArgumentError.new("normalized t_prev must be finite and in [0, 1]")
      end
      unless t_prev < t
        raise ArgumentError.new("normalized t_prev must be strictly less than normalized t")
      end
    end

    private def borrowed_values!(tensor : Tensor, name : String) : Tensor::CPUReadView
      values = tensor.cpu_read
      unless values.borrowed? && values.materialized_bytes == 0_i64
        raise ArgumentError.new("#{name} must provide a borrowed contiguous CPU read")
      end
      values
    end

    private def finite_values!(values : Indexable(Float32), name : String) : Nil
      values.each do |value|
        unless value.finite?
          raise ArgumentError.new("#{name} values must be finite")
        end
      end
    end

    private def sample_once_from_schedule(
      x_t : Tensor,
      pred_v : Tensor,
      sigma_min : Float32,
      t : Float64,
      t_prev : Float64,
      max_result_bytes : Int64,
    ) : FlowEulerStepResultCPU
      validate_result_budget!(max_result_bytes)
      validate_tensor!(x_t, "x_t")
      validate_tensor!(pred_v, "pred_v")
      unless x_t.shape == pred_v.shape
        raise ArgumentError.new(
          "x_t and pred_v must have the same shape, got #{x_t.shape} and #{pred_v.shape}"
        )
      end
      raise ArgumentError.new("flow Euler inputs must be non-empty") if x_t.numel == 0

      validate_schedule_scalars!(sigma_min, t, t_prev)
      validate_result_capacity!(x_t, max_result_bytes)

      x_values = borrowed_values!(x_t, "x_t")
      velocity_values = borrowed_values!(pred_v, "pred_v")
      finite_values!(x_values, "x_t")
      finite_values!(velocity_values, "pred_v")

      pred_x_prev = Tensor.new(
        x_t.shape,
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      pred_x_0 = Tensor.new(
        x_t.shape,
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      apply_source_schedule_formula!(
        x_values,
        velocity_values,
        sigma_min,
        t,
        t_prev,
        pred_x_prev.cpu_data.not_nil!,
        pred_x_0.cpu_data.not_nil!
      )
      FlowEulerStepResultCPU.new(pred_x_prev, pred_x_0)
    end

    # The operation order mirrors the pinned scalar-F32 oracle. The strict
    # public admission policy above is not attributed to upstream.
    private def apply_source_formula!(
      x_t : Indexable(Float32),
      pred_v : Indexable(Float32),
      sigma_min : Float32,
      t : Float32,
      t_prev : Float32,
      pred_x_prev : Array(Float32),
      pred_x_0 : Array(Float32),
    ) : Nil
      delta = (t - t_prev).to_f32
      one_minus_sigma = (1.0_f32 - sigma_min).to_f32
      noise_scale = (sigma_min + one_minus_sigma * t).to_f32

      x_t.each_with_index do |x, index|
        velocity = pred_v[index]
        previous = (x - delta * velocity).to_f32
        origin = flow_euler_pred_to_xstart(
          x,
          velocity,
          one_minus_sigma,
          noise_scale
        )
        unless previous.finite? && origin.finite?
          raise ArgumentError.new("flow Euler arithmetic must produce finite outputs")
        end
        pred_x_prev[index] = previous
        pred_x_0[index] = origin
      end
    end

    private def apply_source_schedule_formula!(
      x_t : Indexable(Float32),
      pred_v : Indexable(Float32),
      sigma_min : Float32,
      t : Float64,
      t_prev : Float64,
      pred_x_prev : Array(Float32),
      pred_x_0 : Array(Float32),
    ) : Nil
      delta = (t - t_prev).to_f32
      one_minus_sigma, noise_scale = flow_euler_schedule_coefficients(
        sigma_min,
        t
      )

      x_t.each_with_index do |x, index|
        velocity = pred_v[index]
        previous = (x - delta * velocity).to_f32
        origin = flow_euler_pred_to_xstart(
          x,
          velocity,
          one_minus_sigma,
          noise_scale
        )
        unless previous.finite? && origin.finite?
          raise ArgumentError.new("flow Euler arithmetic must produce finite outputs")
        end
        pred_x_prev[index] = previous
        pred_x_0[index] = origin
      end
    end
  end

  # Bounded base FlowEuler schedule execution only. This owns no model, CFG,
  # guidance interval/rescale, RNG, pipeline stage, weights, GPU, or Metal
  # behavior. The retained-result cap covers only the F32 payloads in the two
  # upstream-like history arrays; schedule storage, timestep tensors, provider
  # allocations, object overhead, and aggregate process memory are excluded.
  module FlowEulerSamplerCPU
    extend self

    MAX_STEPS                 = 4096_i32
    MAX_RETAINED_RESULT_BYTES = FlowEulerStepCPU::MAX_RESULT_BYTES
    # TRELLIS.2 source pin 75fbf0183001ed9876c8dbb35de6b68552ee08bd:
    # FlowEulerSampler.sample and its CFG subclasses default steps to 50.
    # This constant owns only that public sampler default.
    DEFAULT_STEPS = 50_i32
    # TRELLIS.2 source pin 75fbf0183001ed9876c8dbb35de6b68552ee08bd:
    # FlowEulerCfgSampler and FlowEulerGuidanceIntervalSampler both default
    # guidance_strength to 3.0. This constant is only a public sampler
    # default; it does not add pipeline or model policy.
    DEFAULT_GUIDANCE_STRENGTH = 3.0_f64
    # TRELLIS.2 source pin 75fbf0183001ed9876c8dbb35de6b68552ee08bd:
    # FlowEulerSampler.sample and its CFG subclasses default rescale_t to 1.0.
    # This constant owns only that public sampler default.
    DEFAULT_RESCALE_T = 1.0_f64

    def sample(
      noise : Tensor,
      cond : C,
      sigma_min : Float32,
      steps : Int32 = DEFAULT_STEPS,
      rescale_t : Float64 = DEFAULT_RESCALE_T,
      max_result_bytes : Int64 = MAX_RETAINED_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : FlowEulerSampleResultCPU forall C
      validate_request!(
        noise,
        sigma_min,
        steps,
        rescale_t,
        max_result_bytes
      )
      schedule = build_schedule(steps, rescale_t)
      pred_x_t = Array(Tensor).new(steps)
      pred_x_0 = Array(Tensor).new(steps)
      state = noise

      schedule.each_cons_pair do |t, t_prev|
        outcome = FlowEulerStepCPU.sample_once_with_velocity_provider(
          state,
          cond,
          sigma_min: sigma_min,
          t: t,
          t_prev: t_prev,
          max_result_bytes: max_result_bytes,
          &velocity_provider
        )
        state = outcome.pred_x_prev
        pred_x_t << state
        pred_x_0 << outcome.pred_x_0
      end

      FlowEulerSampleResultCPU.new(state, pred_x_t, pred_x_0)
    end

    # Preserve the admitted base loop while exposing the exact normalized
    # Float64 schedule pair to a composition layer. The adapter receives the
    # already-created model timestep as well, so it can route both source
    # controls without reconstructing `1000*t` from a narrowed tensor value.
    # This owns no CFG, interval, RNG, pipeline, model, GPU, or Metal policy.
    def sample_with_step_provider(
      noise : Tensor,
      cond : C,
      sigma_min : Float32,
      steps : Int32 = DEFAULT_STEPS,
      rescale_t : Float64 = DEFAULT_RESCALE_T,
      max_result_bytes : Int64 = MAX_RETAINED_RESULT_BYTES,
      &step_provider : Tensor, Float64, Float64, Tensor, C -> Tensor
    ) : FlowEulerSampleResultCPU forall C
      validate_request!(
        noise,
        sigma_min,
        steps,
        rescale_t,
        max_result_bytes
      )
      schedule = build_schedule(steps, rescale_t)
      pred_x_t = Array(Tensor).new(steps)
      pred_x_0 = Array(Tensor).new(steps)
      state = noise

      schedule.each_cons_pair do |t, t_prev|
        outcome = FlowEulerStepCPU.sample_once_with_velocity_provider(
          state,
          cond,
          sigma_min: sigma_min,
          t: t,
          t_prev: t_prev,
          max_result_bytes: max_result_bytes
        ) do |actual_x, model_t, actual_cond|
          step_provider.call(actual_x, t, t_prev, model_t, actual_cond)
        end
        state = outcome.pred_x_prev
        pred_x_t << state
        pred_x_0 << outcome.pred_x_0
      end

      FlowEulerSampleResultCPU.new(state, pred_x_t, pred_x_0)
    end

    private def validate_request!(
      noise : Tensor,
      sigma_min : Float32,
      steps : Int32,
      rescale_t : Float64,
      max_result_bytes : Int64,
    ) : Nil
      unless 0_i64 < max_result_bytes <= MAX_RETAINED_RESULT_BYTES
        raise ArgumentError.new(
          "max_result_bytes must be positive and no greater than #{MAX_RETAINED_RESULT_BYTES}"
        )
      end
      raise ArgumentError.new("noise must be on CPU") unless noise.on_cpu?
      raise ArgumentError.new("noise must use F32") unless noise.dtype.f32?
      unless noise.contiguous?
        raise ArgumentError.new("noise must be contiguous; implicit materialization is not admitted")
      end
      raise ArgumentError.new("flow Euler noise must be non-empty") if noise.numel == 0
      unless 1_i32 <= steps <= MAX_STEPS
        raise ArgumentError.new("steps must be in 1..#{MAX_STEPS}")
      end
      unless rescale_t.finite? && rescale_t > 0.0_f64
        raise ArgumentError.new("rescale_t must be finite and positive")
      end
      unless sigma_min.finite? && 0.0_f32 <= sigma_min < 1.0_f32
        raise ArgumentError.new("sigma_min must be finite and in [0, 1)")
      end
      validate_retained_capacity!(noise, steps, max_result_bytes)

      values = noise.cpu_read
      unless values.borrowed? && values.materialized_bytes == 0_i64
        raise ArgumentError.new("noise must provide a borrowed contiguous CPU read")
      end
      values.each do |value|
        unless value.finite?
          raise ArgumentError.new("noise values must be finite")
        end
      end
    end

    private def validate_retained_capacity!(
      noise : Tensor,
      steps : Int32,
      max_result_bytes : Int64,
    ) : Nil
      bytes_per_input_element = 2_i64 * DType::F32.byte_size.to_i64
      bytes_per_element = bytes_per_input_element * steps.to_i64
      max_elements = max_result_bytes // bytes_per_element
      if noise.numel.to_i64 > max_elements
        raise ArgumentError.new(
          "flow Euler retained result budget exceeded for #{steps} steps and " \
          "#{noise.numel} elements"
        )
      end
    end

    # Mirror NumPy linspace(1, 0, steps + 1): form the Float64 step, multiply
    # arange by it, add the start, and pin the final endpoint before applying
    # the source rescale expression. Rewriting this as 1 - i/steps changes
    # Float64 bits for valid step counts even when a later F32 boundary can hide
    # the difference.
    private def build_schedule(steps : Int32, rescale_t : Float64) : Array(Float64)
      linspace_step = -1.0_f64 / steps.to_f64
      rescale_delta = rescale_t - 1.0_f64
      schedule = Array(Float64).new(steps + 1)

      (steps + 1).times do |index|
        unscaled = if index == steps
                     0.0_f64
                   else
                     index.to_f64 * linspace_step + 1.0_f64
                   end
        numerator = rescale_t * unscaled
        denominator = 1.0_f64 + rescale_delta * unscaled
        value = numerator / denominator
        unless value.finite? && 0.0_f64 <= value <= 1.0_f64
          raise ArgumentError.new("rescale_t produced an invalid normalized schedule")
        end
        schedule << value
      end

      schedule.each_cons_pair do |t, t_prev|
        unless t_prev < t
          raise ArgumentError.new("rescale_t must produce a strictly decreasing schedule")
        end
      end
      schedule
    end
  end
end
