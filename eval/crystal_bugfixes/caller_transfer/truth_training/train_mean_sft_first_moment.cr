require "./train_mean_sft_step"

# Keep the archived driver and its two single-example controls intact.
# Intercept only steps that consume the mean of the two gradients.
class ML::GGUF::Qwen35QLoRA::Trainer
  @mean_update_pending = false

  private def mean_gradient(left, right) : Array(Array(Float32))
    gradient = previous_def(left, right)
    @mean_update_pending = true
    gradient
  end

  private def apply_gradient(grads : Array(Array(Float32))) : Nil
    if @mean_update_pending
      @mean_update_pending = false
      weights = @model.factors.map(&.dup)
      moments = @adam.state_dict.transform_values { |(m, v, step)| {m.clone, v.clone, step} }
      counter = moments.values.first[2]
      check_adam_state(counter)

      # Check the exact same gradient and optimizer input against the archive.
      previous_def(grads)
      carry_sha = adapter_digest
      restore_snapshot(weights, moments)

      @adam.state_dict.each_value { |(m, _, _)| m.cpu_data.not_nil!.fill(0_f32) }
      moments.each_value { |(m, _, _)| m.cpu_data.not_nil!.fill(0_f32) }
      check_snapshot(weights, moments)

      previous_def(grads)
      check_adam_state(counter + 1)
      STDERR.puts({event: "mean_first_moment_reset", adam_before: counter,
                   adam_after: counter + 1, factors: moments.size,
                   carry_sha256: carry_sha}.to_json)
    else
      previous_def(grads)
    end
  end
end
