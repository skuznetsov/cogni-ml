# Bounded synchronous CPU/F32 classifier-free guidance for one velocity call.
#
# This adapter owns only branch routing and, for the mixed branch, one result
# tensor. It does not own Euler state arithmetic, scheduling, guidance
# intervals or rescale, RNG, a model, weights, GPU, or Metal behavior.
# Callers and providers must not mutate borrowed inputs, conditions, or earlier
# returned predictions while this synchronous operation is active.

require "./flow_sampler"

module ML::ThreeD::Trellis2
  module FlowClassifierFreeGuidanceCPU
    extend self

    # Logical F32 payload of the returned velocity. Provider-owned allocations,
    # object overhead, and aggregate process memory are outside this cap.
    MAX_RESULT_BYTES = FlowEulerStepCPU::MAX_RESULT_BYTES

    # Reproduce the pinned TRELLIS.2 CFG branch order. Exact strengths 1 and 0
    # call only the positive or negative condition respectively and return that
    # provider tensor by identity. Other finite strengths call positive first,
    # then negative, and apply the source expression without reassociation.
    def predict_velocity(
      x_t : Tensor,
      model_timesteps : Tensor,
      positive_condition : C,
      negative_condition : C,
      guidance_strength : Float64,
      max_result_bytes : Int64 = MAX_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : Tensor forall C
      validate_request!(
        x_t,
        model_timesteps,
        guidance_strength,
        max_result_bytes
      )

      if guidance_strength == 1.0_f64
        prediction = velocity_provider.call(
          x_t,
          model_timesteps,
          positive_condition
        )
        validate_after_provider!(x_t, model_timesteps, prediction)
        return prediction
      end

      if guidance_strength == 0.0_f64
        prediction = velocity_provider.call(
          x_t,
          model_timesteps,
          negative_condition
        )
        validate_after_provider!(x_t, model_timesteps, prediction)
        return prediction
      end

      positive = velocity_provider.call(
        x_t,
        model_timesteps,
        positive_condition
      )
      validate_after_provider!(x_t, model_timesteps, positive)

      negative = velocity_provider.call(
        x_t,
        model_timesteps,
        negative_condition
      )
      validate_after_provider!(x_t, model_timesteps, negative)

      positive_strength = guidance_strength.to_f32
      negative_strength = (1.0_f64 - guidance_strength).to_f32
      positive_values = borrowed_values!(positive, "prediction")
      negative_values = borrowed_values!(negative, "prediction")
      result = Tensor.new(
        x_t.shape,
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      result_values = result.cpu_data.not_nil!

      positive_values.each_with_index do |value, index|
        positive_term = (positive_strength * value).to_f32
        negative_term = (negative_strength * negative_values[index]).to_f32
        mixed = (positive_term + negative_term).to_f32
        unless mixed.finite?
          raise ArgumentError.new("CFG arithmetic must produce finite outputs")
        end
        result_values[index] = mixed
      end
      result
    end

    private def validate_request!(
      x_t : Tensor,
      model_timesteps : Tensor,
      guidance_strength : Float64,
      max_result_bytes : Int64,
    ) : Nil
      validate_result_budget!(max_result_bytes)
      validate_tensor!(x_t, "x_t")
      raise ArgumentError.new("CFG input must be non-empty") if x_t.numel == 0
      validate_tensor!(model_timesteps, "model_timesteps")
      expected_model_shape = Shape.new(x_t.shape[0])
      unless model_timesteps.shape == expected_model_shape
        raise ArgumentError.new(
          "model_timesteps must have shape #{expected_model_shape}, got " \
          "#{model_timesteps.shape}"
        )
      end

      unless guidance_strength.finite?
        raise ArgumentError.new("guidance_strength must be finite")
      end
      max_f32 = Float32::MAX.to_f64
      unless guidance_strength.abs <= max_f32 &&
             (1.0_f64 - guidance_strength).abs <= max_f32
        raise ArgumentError.new("guidance_strength must be representable in F32 CFG arithmetic")
      end

      validate_result_capacity!(x_t, max_result_bytes)
      finite_values!(borrowed_values!(x_t, "x_t"), "x_t")
      finite_values!(
        borrowed_values!(model_timesteps, "model_timesteps"),
        "model_timesteps"
      )
    end

    private def validate_after_provider!(
      x_t : Tensor,
      model_timesteps : Tensor,
      prediction : Tensor,
    ) : Nil
      # Recheck borrowed request tensors after each synchronous callback. This
      # catches obvious non-finite mutation without adding defensive copies;
      # finite mutation remains a provider contract violation.
      validate_tensor!(x_t, "x_t")
      validate_tensor!(model_timesteps, "model_timesteps")
      finite_values!(borrowed_values!(x_t, "x_t"), "x_t")
      finite_values!(
        borrowed_values!(model_timesteps, "model_timesteps"),
        "model_timesteps"
      )

      validate_tensor!(prediction, "prediction")
      unless prediction.shape == x_t.shape
        raise ArgumentError.new(
          "x_t and prediction must have the same shape, got " \
          "#{x_t.shape} and #{prediction.shape}"
        )
      end
      finite_values!(borrowed_values!(prediction, "prediction"), "prediction")
    end

    private def validate_result_budget!(max_result_bytes : Int64) : Nil
      unless 0_i64 < max_result_bytes <= MAX_RESULT_BYTES
        raise ArgumentError.new(
          "max_result_bytes must be positive and no greater than #{MAX_RESULT_BYTES}"
        )
      end
    end

    private def validate_result_capacity!(
      x_t : Tensor,
      max_result_bytes : Int64,
    ) : Nil
      required_result_bytes = x_t.numel.to_i64 * DType::F32.byte_size.to_i64
      if required_result_bytes > max_result_bytes
        raise ArgumentError.new(
          "CFG result budget requires #{required_result_bytes} bytes but " \
          "max_result_bytes is #{max_result_bytes}"
        )
      end
    end

    private def validate_tensor!(tensor : Tensor, name : String) : Nil
      raise ArgumentError.new("#{name} must be on CPU") unless tensor.on_cpu?
      raise ArgumentError.new("#{name} must use F32") unless tensor.dtype.f32?
      unless tensor.contiguous?
        raise ArgumentError.new(
          "#{name} must be contiguous; implicit materialization is not admitted"
        )
      end
    end

    private def borrowed_values!(
      tensor : Tensor,
      name : String,
    ) : Tensor::CPUReadView
      values = tensor.cpu_read
      unless values.borrowed? && values.materialized_bytes == 0_i64
        raise ArgumentError.new(
          "#{name} must provide a borrowed contiguous CPU read"
        )
      end
      values
    end

    private def finite_values!(
      values : Indexable(Float32),
      name : String,
    ) : Nil
      values.each do |value|
        unless value.finite?
          raise ArgumentError.new("#{name} values must be finite")
        end
      end
    end
  end
end
