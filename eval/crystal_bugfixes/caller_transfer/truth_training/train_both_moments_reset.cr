require "json"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

class ML::GGUF::Qwen35QLoRA::Trainer
  def capture_moment_branch(step : Int32)
    weights = @model.factors.map(&.dup)
    moments = @adam.state_dict.transform_values do |(first, second, updates)|
      raise "unexpected Adam step #{updates}, expected #{step}" unless updates == step
      {first.clone, second.clone, updates}
    end
    raise "incomplete Adam state" unless moments.size == @params.size
    {weights, moments}
  end

  def restore_moment_branch(snapshot, *, clear_moments : Bool) : Tuple(Int32, Int32)
    weights, moments = snapshot
    @model.factors.zip(@params).each_with_index do |(factor, param), index|
      values = weights[index]
      data = param.data.cpu_data.not_nil!
      factor.to_unsafe.copy_from(values.to_unsafe, values.size)
      data.to_unsafe.copy_from(values.to_unsafe, values.size)
      raise "factor restore mismatch" unless factor == values && data == values
    end

    # Each restore needs its own tensors. Adam updates its moments in place.
    copies = moments.transform_values do |(first, second, updates)|
      first_copy, second_copy = first.clone, second.clone
      if clear_moments
        first_copy.fill!(0.0_f32)
        second_copy.fill!(0.0_f32)
      end
      {first_copy, second_copy, updates}
    end
    @adam.load_state_dict(copies)
    live = @adam.state_dict
    raise "Adam parameter identities changed" unless live.size == moments.size
    moments.each do |id, (first, second, updates)|
      live_first, live_second, live_updates = live[id]
      raise "Adam step changed" unless live_updates == updates
      first_data, second_data = live_first.cpu_data.not_nil!, live_second.cpu_data.not_nil!
      if clear_moments
        raise "Adam first moment was not cleared" unless first_data.all?(&.zero?)
        raise "Adam second moment was not cleared" unless second_data.all?(&.zero?)
      else
        raise "Adam first moment restore mismatch" unless first_data == first.cpu_data
        raise "Adam second moment restore mismatch" unless second_data == second.cpu_data
      end
    end

    active_first = moments.count { |_, (first, _, _)| first.cpu_data.not_nil!.any? { |value| !value.zero? } }
    active_second = moments.count { |_, (_, second, _)| second.cpu_data.not_nil!.any?(&.positive?) }
    raise "missing Adam moment history" if clear_moments && (active_first == 0 || active_second == 0)
    {active_first, active_second}
  end

  def check_branch_state(step : Int32) : Nil
    @adam.state_dict.each_value do |(first, second, updates)|
      raise "unexpected Adam step #{updates}, expected #{step}" unless updates == step
      raise "nonfinite Adam first moment" unless first.cpu_data.not_nil!.all?(&.finite?)
      unless second.cpu_data.not_nil!.all? { |value| value.finite? && value >= 0.0_f32 }
        raise "invalid Adam second moment"
      end
    end
    raise "nonfinite updated factors" unless @model.factors.all? { |factor| factor.all?(&.finite?) }
  end
end

module BothMomentsReset
  extend self

  TARGETS = {10, 25, 50}

  private def load_examples(model, path)
    rows = [] of QLoRA::Example
    File.each_line(path) do |line|
      next if line.strip.empty?
      conversation = QChat.messages_from_openai_json(JSON.parse(line)["messages"].to_json)
      example = model.example(conversation)
      unless example.input.all?(&.finite?)
        raise "nonfinite frozen prefix in #{path} at row #{rows.size + 1}"
      end
      rows << example
      STDERR.puts({event: "prefill", file: File.basename(path), row: rows.size,
                   prefix_tokens: example.first + 1, target_tokens: example.targets.size}.to_json)
    end
    raise "empty QLoRA dataset #{path}" if rows.empty?
    rows
  end

  private def update(trainer, examples, row, step, order, target = nil) : Nil
    example = examples[row - 1]
    loss = trainer.step(example)
    STDERR.puts({event: "update", order: order, target: target, step: step, row: row,
                 first_target_id: example.targets.first, loss: loss}.to_json)
  end

  private def save(model, output_dir, target, phase, step) : Nil
    path = File.join(output_dir, "target-#{target}", phase, "adapter.bin")
    raise "checkpoint already exists at #{path}" if File.exists?(path)
    Dir.mkdir_p(File.dirname(path))
    model.save(path)
    STDERR.puts({event: "saved", target: target, phase: phase, step: step, adapter: path}.to_json)
  end

  def run(model_path, train_path, instrument_path, initial, visits_path, output_dir) : Nil
    visits = Array(Int32).from_json(File.read(visits_path))
    raise "expected #{TARGETS.last} visits" unless visits.size == TARGETS.last

    model = QLoRA::Model.new(model_path, 42_u64, full_blocks: 4)
    begin
      model.load(initial)
      examples = load_examples(model, train_path)
      instrument = load_examples(model, instrument_path)
      raise "training row out of range" unless visits.all? { |row| 1 <= row <= examples.size }

      trainer = QLoRA::Trainer.new(model)
      validation_loss = instrument.sum { |example| trainer.loss(example) } / instrument.size
      STDERR.puts({event: "start", model_sha256: model.digest, seed: 42, updates: visits.size,
                   examples: examples.size, input_adapter: initial, blocks: model.layer_indices,
                   validation_loss: validation_loss}.to_json)

      visits.each_with_index do |row, index|
        step = index + 1
        update(trainer, examples, row, step, "control")
        if TARGETS.includes?(step + 2)
          target = step + 2
          save(model, output_dir, target, "common", step)
          snapshot = trainer.capture_moment_branch(step)
          normal = {visits[index + 1], visits[index + 2]}
          {"forward" => normal, "swapped" => {normal[1], normal[0]}}.each do |order, rows|
            active_first, active_second = trainer.restore_moment_branch(snapshot, clear_moments: true)
            STDERR.puts({event: "reset_both", target: target, order: order, step: step,
                         active_first_tensors: active_first, active_second_tensors: active_second,
                         state_tensors: snapshot[1].size}.to_json)
            rows.each_with_index do |branch_row, offset|
              updated_step = step + offset + 1
              update(trainer, examples, branch_row, updated_step, order, target)
              trainer.check_branch_state(updated_step)
              save(model, output_dir, target, "#{order}-#{offset + 1}", updated_step)
            end
          end
          trainer.restore_moment_branch(snapshot, clear_moments: false)
        elsif TARGETS.includes?(step)
          save(model, output_dir, step, "control-forward", step)
        end
      end
    ensure
      model.close
    end
  end
end

abort "usage: native-reset MODEL TRAIN_JSONL INSTRUMENT_JSONL INITIAL VISITS_JSON OUT" unless ARGV.size == 6
model, train, instrument, initial, visits, output_dir = ARGV
BothMomentsReset.run(model, train, instrument, initial, visits, output_dir)
