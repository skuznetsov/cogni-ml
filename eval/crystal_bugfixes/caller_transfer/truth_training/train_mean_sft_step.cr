require "json"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

class ML::GGUF::Qwen35QLoRA::Trainer
  def mean_sft_branch(left : Example, right : Example, start : Int32, output : String)
    unless left.targets.size == right.targets.size && left.targets.first != right.targets.first
      raise "expected equal target counts and opposite first targets"
    end

    active_first, active_second = check_adam_state(start)
    weights = @model.factors.map(&.dup)
    moments = @adam.state_dict.transform_values { |(m, v, step)| {m.clone, v.clone, step} }
    common_sha = adapter_digest

    left_loss, left_grad = full_sft_gradient(left)
    check_snapshot(weights, moments)
    right_loss, right_grad = full_sft_gradient(right)
    check_snapshot(weights, moments)

    apply_gradient(mean_gradient(left_grad, right_grad))
    check_adam_state(start + 1)
    raise "checkpoint already exists at #{output}" if File.exists?(output)
    Dir.mkdir_p(File.dirname(output))
    @model.save(output)
    mean_sha = File.open(output) { |file| Digest::SHA256.hexdigest(file) }

    # Apply each unaveraged gradient once as a control. Their adapters must
    # match the archived first updates in the two sequential orders.
    restore_snapshot(weights, moments)
    apply_gradient(left_grad)
    forward_first_sha = adapter_digest
    restore_snapshot(weights, moments)
    apply_gradient(right_grad)
    swapped_first_sha = adapter_digest

    restore_snapshot(weights, moments)
    apply_gradient(mean_gradient(right_grad, left_grad))
    check_adam_state(start + 1)
    raise "reversing the average changed the adapter" unless adapter_digest == mean_sha
    restore_snapshot(weights, moments)

    {losses:               {left_loss, right_loss},
     common_sha256:        common_sha,
     forward_first_sha256: forward_first_sha,
     swapped_first_sha256: swapped_first_sha,
     mean_sha256:          mean_sha,
     adam_before:          start,
     adam_after:           start + 1,
     active_first:         active_first,
     active_second:        active_second,
     reverse_equal:        true}
  end

  private def full_sft_gradient(example : Example) : Tuple(Float64, Array(Array(Float32)))
    @model.projections.each(&.zero_grad)
    hidden = @model.forward(example.input, retain_tape: true)
    loss, gradient = @head.calculate(hidden, example.first, example.targets, backward: true)
    raise "nonfinite QLoRA loss" unless loss.finite?
    @model.backward(gradient)
    # The second backward pass clears the projection buffers.
    {loss, @model.projections.flat_map { |proj| [proj.grad_a.dup, proj.grad_b.dup] }}
  end

  private def mean_gradient(left, right) : Array(Array(Float32))
    left.zip(right).map do |a, b|
      raise "QLoRA gradient shape mismatch" unless a.size == b.size
      Array(Float32).new(a.size) { |index| (a[index] + b[index]) * 0.5_f32 }
    end
  end

  private def apply_gradient(grads : Array(Array(Float32))) : Nil
    raise "incomplete QLoRA gradients" unless grads.size == @params.size
    @params.zip(grads).each do |param, grad|
      raise "nonfinite QLoRA factor gradient" unless grad.all?(&.finite?)
      param.grad = Tensor.from_array(grad, param.shape)
    end
    @adam.step
    @model.factors.zip(@params).each do |factor, param|
      factor.to_unsafe.copy_from(param.data.cpu_data.not_nil!.to_unsafe, factor.size)
    end
  end

  private def check_adam_state(step : Int32) : Tuple(Int32, Int32)
    state = @adam.state_dict
    raise "incomplete Adam state" unless state.size == @params.size
    active_first = 0
    active_second = 0
    state.each_value do |(m, v, updates)|
      raise "unexpected Adam step #{updates}, expected #{step}" unless updates == step
      first, second = m.cpu_data.not_nil!, v.cpu_data.not_nil!
      unless first.all?(&.finite?) && second.all? { |value| value.finite? && value >= 0_f32 }
        raise "invalid Adam moments"
      end
      active_first += 1 if first.any? { |value| !value.zero? }
      active_second += 1 if second.any?(&.positive?)
    end
    unless active_first == @params.size && active_second == @params.size
      raise "missing moment history"
    end
    {active_first, active_second}
  end

  private def check_snapshot(weights, moments) : Nil
    @model.factors.zip(@params).each_with_index do |(factor, param), index|
      raise "snapshot weights changed" unless factor == weights[index] && param.data.cpu_data == weights[index]
    end
    state = @adam.state_dict
    raise "snapshot Adam state size changed" unless state.size == moments.size
    moments.each do |id, (m, v, step)|
      live_m, live_v, live_step = state[id]
      unless live_step == step && live_m.cpu_data == m.cpu_data && live_v.cpu_data == v.cpu_data
        raise "snapshot Adam state changed"
      end
    end
  end

  private def restore_snapshot(weights, moments) : Nil
    @model.factors.zip(@params).each_with_index do |(factor, param), index|
      source = weights[index]
      factor.to_unsafe.copy_from(source.to_unsafe, source.size)
      param.data.cpu_data.not_nil!.to_unsafe.copy_from(source.to_unsafe, source.size)
    end
    # The optimizer mutates these tensors in place.
    @adam.load_state_dict(moments.transform_values { |(m, v, step)| {m.clone, v.clone, step} })
    check_snapshot(weights, moments)
  end

  private def adapter_digest : String
    file = File.tempfile("qwen35-mean-sft-", ".bin")
    begin
      file.close
      @model.save(file.path)
      File.open(file.path) { |adapter| Digest::SHA256.hexdigest(adapter) }
    ensure
      File.delete?(file.path)
    end
  end
end

module MeanSFTStep
  extend self

  TARGETS = {10, 25, 50}

  private def load_examples(model, path)
    rows = [] of QLoRA::Example
    File.each_line(path) do |line|
      next if line.strip.empty?
      messages = QChat.messages_from_openai_json(JSON.parse(line)["messages"].to_json)
      example = model.example(messages)
      unless example.input.all?(&.finite?)
        raise "nonfinite frozen prefix in #{path} at row #{rows.size + 1}"
      end
      rows << example
      STDERR.puts({event: "prefill", file: File.basename(path), row: rows.size,
                   prefix_tokens: example.first + 1, target_tokens: example.targets.size}.to_json)
    end
    raise "expected 32 training rows" unless rows.size == 32
    rows
  end

  def run(model_path, train_path, initial, visits_path, output_dir) : Nil
    visits = Array(Int32).from_json(File.read(visits_path))
    raise "expected #{TARGETS.last} visits" unless visits.size == TARGETS.last

    model = QLoRA::Model.new(model_path, 42_u64, full_blocks: 4)
    begin
      model.load(initial)
      examples = load_examples(model, train_path)
      raise "training row out of range" unless visits.all? { |row| 1 <= row <= examples.size }

      trainer = QLoRA::Trainer.new(model)
      STDERR.puts({event: "start", model_sha256: model.digest, seed: 42,
                   examples: examples.size, blocks: model.layer_indices}.to_json)

      (TARGETS.last - 2).times do |index|
        step = index + 1
        row = visits[index]
        example = examples[row - 1]
        loss = trainer.step(example)
        STDERR.puts({event: "update", step: step, row: row,
                     first_target_id: example.targets.first, loss: loss}.to_json)

        target = step + 2
        next unless TARGETS.includes?(target)
        left_row, right_row = visits[step], visits[step + 1]
        left, right = examples[left_row - 1], examples[right_row - 1]
        output = File.join(output_dir, "target-#{target}", "mean", "adapter.bin")
        audit = trainer.mean_sft_branch(left, right, step, output)
        STDERR.puts(audit.merge({
          event: "mean_step", target: target, step: step + 1, rows: {left_row, right_row},
          target_tokens: {left.targets.size, right.targets.size},
        }).to_json)
      end
    ensure
      model.close
    end
  end
end

abort "usage: native-mean MODEL TRAIN_JSONL INITIAL VISITS_JSON OUT" unless ARGV.size == 5
model, train, initial, visits, output_dir = ARGV
MeanSFTStep.run(model, train, initial, visits, output_dir)
