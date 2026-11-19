require "./train_mean_sft_step"

# Reset the moments only for the branch that consumes the mean gradient.
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

      # Verify the carry update at the exact input to the treatment.
      previous_def(grads)
      carry_sha = adapter_digest
      moments.each_value do |m, v, _|
        m.cpu_data.not_nil!.fill(0_f32)
        v.cpu_data.not_nil!.fill(0_f32)
      end
      restore_snapshot(weights, moments)

      previous_def(grads)
      check_adam_state(counter + 1)
      STDERR.puts({event: "mean_both_moments_reset", adam_before: counter,
                   adam_after: counter + 1, factors: moments.size,
                   carry_sha256: carry_sha}.to_json)
    else
      previous_def(grads)
    end
  end
end
