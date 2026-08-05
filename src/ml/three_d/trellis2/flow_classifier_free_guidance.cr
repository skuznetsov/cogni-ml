# Bounded synchronous CPU/F32 classifier-free guidance for one velocity call.
#
# This adapter owns branch routing and one-call guidance rescale. The plain
# mixed branch owns one result tensor; active rescale owns that mixed tensor
# plus one final tensor and streams x0 statistics without full x0 buffers. It
# does not own scheduling, guidance intervals, RNG, a model, weights, GPU, or
# Metal behavior.
# Callers and providers must not mutate borrowed inputs, conditions, or earlier
# returned predictions while this synchronous operation is active.

require "./flow_sampler"

module ML::ThreeD::Trellis2
  module FlowClassifierFreeGuidanceCPU
    extend self
    include FlowEulerStateConversionCPU

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
      predict_velocity_validated(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        guidance_strength,
        &velocity_provider
      )
    end

    # Apply the pinned mixed-only, strictly-positive guidance-rescale branch to
    # one CFG velocity call. Exact strengths 1 and 0 return before inspecting
    # conversion controls, matching the upstream early-return boundary. For a
    # mixed branch, Cogni deliberately rejects non-finite rescale controls and
    # all non-finite conversion arithmetic rather than returning upstream's raw
    # NaN/Inf values. No epsilon or clamp is introduced.
    def predict_velocity_with_rescale(
      x_t : Tensor,
      model_timesteps : Tensor,
      positive_condition : C,
      negative_condition : C,
      sigma_min : Float32,
      normalized_t : Float64,
      guidance_strength : Float64,
      guidance_rescale : Float64,
      max_result_bytes : Int64 = MAX_RESULT_BYTES,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : Tensor forall C
      # Keep the admitted CFG validation prefix and its exception precedence.
      validate_request!(
        x_t,
        model_timesteps,
        guidance_strength,
        max_result_bytes
      )

      if guidance_strength == 1.0_f64 || guidance_strength == 0.0_f64
        return predict_velocity_validated(
          x_t,
          model_timesteps,
          positive_condition,
          negative_condition,
          guidance_strength,
          &velocity_provider
        )
      end

      validate_guidance_rescale!(guidance_rescale)
      unless guidance_rescale > 0.0_f64
        return predict_velocity_validated(
          x_t,
          model_timesteps,
          positive_condition,
          negative_condition,
          guidance_strength,
          &velocity_provider
        )
      end

      one_minus_sigma, noise_scale = validate_active_rescale!(
        x_t,
        sigma_min,
        normalized_t,
        max_result_bytes
      )

      positive_prediction = nil.as(Tensor?)
      mixed_prediction = predict_velocity_validated(
        x_t,
        model_timesteps,
        positive_condition,
        negative_condition,
        guidance_strength
      ) do |actual_x, actual_model_t, actual_condition|
        prediction = velocity_provider.call(
          actual_x,
          actual_model_t,
          actual_condition
        )
        positive_prediction ||= prediction
        prediction
      end
      positive = positive_prediction.not_nil!

      # The negative callback ran after the first validation. Revalidate the
      # retained positive tensor before using it, so an obvious non-finite
      # mutation cannot leak into conversion arithmetic.
      validate_after_provider!(x_t, model_timesteps, positive)
      apply_guidance_rescale(
        x_t,
        positive,
        mixed_prediction,
        one_minus_sigma,
        noise_scale,
        guidance_rescale
      )
    end

    private def predict_velocity_validated(
      x_t : Tensor,
      model_timesteps : Tensor,
      positive_condition : C,
      negative_condition : C,
      guidance_strength : Float64,
      &velocity_provider : Tensor, Tensor, C -> Tensor
    ) : Tensor forall C
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

    private def validate_guidance_rescale!(guidance_rescale : Float64) : Nil
      unless guidance_rescale.finite?
        raise ArgumentError.new("guidance_rescale must be finite for mixed CFG")
      end

      # Non-positive values bypass x0 arithmetic and are never narrowed by the
      # pinned source. Positive values and their blend complement are F32
      # tensor coefficients and must be representable before provider work.
      return unless guidance_rescale > 0.0_f64

      max_f32 = Float32::MAX.to_f64
      unless guidance_rescale.abs <= max_f32 &&
             (1.0_f64 - guidance_rescale).abs <= max_f32
        raise ArgumentError.new(
          "guidance_rescale must be representable in F32 rescale arithmetic"
        )
      end
    end

    private def validate_active_rescale!(
      x_t : Tensor,
      sigma_min : Float32,
      normalized_t : Float64,
      max_result_bytes : Int64,
    ) : Tuple(Float32, Float32)
      unless sigma_min.finite? && 0.0_f32 <= sigma_min < 1.0_f32
        raise ArgumentError.new("sigma_min must be finite and in [0, 1)")
      end
      unless normalized_t.finite? && 0.0_f64 <= normalized_t <= 1.0_f64
        raise ArgumentError.new("normalized_t must be finite and in [0, 1]")
      end
      unless x_t.shape.ndim >= 2
        raise ArgumentError.new(
          "guidance rescale requires at least one non-batch axis"
        )
      end

      batch = x_t.shape[0]
      sample_count = x_t.numel // batch
      unless sample_count >= 2
        raise ArgumentError.new(
          "guidance rescale requires at least two values per batch sample"
        )
      end

      one_minus_sigma, noise_scale = flow_euler_schedule_coefficients(
        sigma_min,
        normalized_t
      )
      unless one_minus_sigma.finite? && noise_scale.finite?
        raise ArgumentError.new("flow conversion coefficients must be finite")
      end
      unless noise_scale > 0.0_f32
        raise ArgumentError.new("flow conversion denominator must be positive")
      end

      # Streaming x0 statistics allocate no x0 tensors or per-batch vectors.
      # The only simultaneous adapter-owned F32 payloads are the mixed CFG
      # tensor and the final rescaled tensor. Provider memory, object overhead,
      # and aggregate process memory remain outside this logical cap.
      required_bytes = x_t.numel.to_i64 * 2_i64 * DType::F32.byte_size.to_i64
      if required_bytes > max_result_bytes
        raise ArgumentError.new(
          "guidance rescale result budget requires #{required_bytes} bytes " \
          "but max_result_bytes is #{max_result_bytes}"
        )
      end

      {one_minus_sigma, noise_scale}
    end

    private def apply_guidance_rescale(
      x_t : Tensor,
      positive_prediction : Tensor,
      mixed_prediction : Tensor,
      one_minus_sigma : Float32,
      noise_scale : Float32,
      guidance_rescale : Float64,
    ) : Tensor
      x_values = borrowed_values!(x_t, "x_t")
      positive_values = borrowed_values!(positive_prediction, "prediction")
      mixed_values = borrowed_values!(mixed_prediction, "prediction")
      batch = x_t.shape[0]
      sample_count = x_t.numel // batch
      result = Tensor.new(
        x_t.shape,
        dtype: DType::F32,
        device: Tensor::Device::CPU
      )
      result_values = result.cpu_data.not_nil!
      rescale = guidance_rescale.to_f32
      remainder = (1.0_f64 - guidance_rescale).to_f32

      batch.times do |batch_index|
        offset = batch_index * sample_count
        positive_sum = 0.0_f32
        mixed_sum = 0.0_f32

        sample_count.times do |sample_index|
          index = offset + sample_index
          positive_x0 = flow_euler_pred_to_xstart(
            x_values[index],
            positive_values[index],
            one_minus_sigma,
            noise_scale
          )
          mixed_x0 = flow_euler_pred_to_xstart(
            x_values[index],
            mixed_values[index],
            one_minus_sigma,
            noise_scale
          )
          finite_rescale_value!(positive_x0, "positive x0")
          finite_rescale_value!(mixed_x0, "guided x0")
          positive_sum = (positive_sum + positive_x0).to_f32
          mixed_sum = (mixed_sum + mixed_x0).to_f32
          finite_rescale_value!(positive_sum, "positive x0 sum")
          finite_rescale_value!(mixed_sum, "guided x0 sum")
        end

        count = sample_count.to_f32
        positive_mean = (positive_sum / count).to_f32
        mixed_mean = (mixed_sum / count).to_f32
        finite_rescale_value!(positive_mean, "positive x0 mean")
        finite_rescale_value!(mixed_mean, "guided x0 mean")
        positive_squared_sum = 0.0_f32
        mixed_squared_sum = 0.0_f32

        sample_count.times do |sample_index|
          index = offset + sample_index
          positive_x0 = flow_euler_pred_to_xstart(
            x_values[index],
            positive_values[index],
            one_minus_sigma,
            noise_scale
          )
          mixed_x0 = flow_euler_pred_to_xstart(
            x_values[index],
            mixed_values[index],
            one_minus_sigma,
            noise_scale
          )
          positive_delta = (positive_x0 - positive_mean).to_f32
          mixed_delta = (mixed_x0 - mixed_mean).to_f32
          positive_square = (positive_delta * positive_delta).to_f32
          mixed_square = (mixed_delta * mixed_delta).to_f32
          positive_squared_sum = (
            positive_squared_sum + positive_square
          ).to_f32
          mixed_squared_sum = (mixed_squared_sum + mixed_square).to_f32
          finite_rescale_value!(
            positive_squared_sum,
            "positive x0 squared sum"
          )
          finite_rescale_value!(mixed_squared_sum, "guided x0 squared sum")
        end

        correction = (sample_count - 1).to_f32
        positive_variance = (positive_squared_sum / correction).to_f32
        mixed_variance = (mixed_squared_sum / correction).to_f32
        positive_std = Math.sqrt(positive_variance).to_f32
        mixed_std = Math.sqrt(mixed_variance).to_f32
        finite_rescale_value!(positive_std, "positive x0 standard deviation")
        finite_rescale_value!(mixed_std, "guided x0 standard deviation")
        unless mixed_std > 0.0_f32
          raise ArgumentError.new(
            "guided x0 standard deviation must be positive"
          )
        end
        ratio = (positive_std / mixed_std).to_f32
        finite_rescale_value!(ratio, "guidance rescale ratio")

        sample_count.times do |sample_index|
          index = offset + sample_index
          mixed_x0 = flow_euler_pred_to_xstart(
            x_values[index],
            mixed_values[index],
            one_minus_sigma,
            noise_scale
          )
          rescaled_x0 = (mixed_x0 * ratio).to_f32
          rescaled_term = (rescale * rescaled_x0).to_f32
          original_term = (remainder * mixed_x0).to_f32
          blended_x0 = (rescaled_term + original_term).to_f32
          prediction = flow_euler_xstart_to_pred(
            x_values[index],
            blended_x0,
            one_minus_sigma,
            noise_scale
          )
          finite_rescale_value!(rescaled_x0, "rescaled x0")
          finite_rescale_value!(blended_x0, "blended x0")
          finite_rescale_value!(prediction, "guidance rescale output")
          result_values[index] = prediction
        end
      end
      result
    end

    private def finite_rescale_value!(value : Float32, name : String) : Nil
      unless value.finite?
        raise ArgumentError.new("#{name} must be finite")
      end
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
