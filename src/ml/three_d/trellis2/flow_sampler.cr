# Bounded CPU/F32 arithmetic for one TRELLIS.2 base-flow Euler step.
#
# This leaf consumes an already computed velocity and normalized state times.
# It does not invoke a model, construct the model timestep `1000*t`, own a
# schedule or RNG, or implement CFG. Callers must not mutate input tensors
# concurrently while this synchronous operation holds borrowed CPU reads. A
# velocity-provider callback must not mutate x_t during its invocation.

require "../../core/tensor"

module ML::ThreeD::Trellis2
  class FlowEulerStepResultCPU
    getter pred_x_prev : Tensor
    getter pred_x_0 : Tensor

    def initialize(@pred_x_prev : Tensor, @pred_x_0 : Tensor)
    end
  end

  module FlowEulerStepCPU
    extend self

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
        origin = (one_minus_sigma * x - noise_scale * velocity).to_f32
        unless previous.finite? && origin.finite?
          raise ArgumentError.new("flow Euler arithmetic must produce finite outputs")
        end
        pred_x_prev[index] = previous
        pred_x_0[index] = origin
      end
    end
  end
end
