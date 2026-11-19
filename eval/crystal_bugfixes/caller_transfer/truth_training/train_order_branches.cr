require "json"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

class ML::GGUF::Qwen35QLoRA::Trainer
  # Adam's state_dict shares the live moment tensors.
  def order_snapshot(step : Int32)
    factors = @model.factors.map(&.dup)
    moments = @adam.state_dict.transform_values do |(first, second, updates)|
      raise "unexpected Adam step #{updates}, expected #{step}" unless updates == step
      {first.clone, second.clone, updates}
    end
    raise "incomplete Adam state" unless moments.size == @params.size
    {factors, moments}
  end

  def restore_order_snapshot(snapshot) : Nil
    weights, moments = snapshot
    @model.factors.zip(@params).each_with_index do |(factor, param), index|
      values = weights[index]
      factor.to_unsafe.copy_from(values.to_unsafe, values.size)
      param.data.cpu_data.not_nil!.to_unsafe.copy_from(values.to_unsafe, values.size)
    end
    @adam.load_state_dict(moments)
  end
end

module OrderBranches
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
        update(trainer, examples, row, step, "forward")
        if TARGETS.includes?(step + 2)
          target = step + 2
          save(model, output_dir, target, "common", step)
          snapshot = trainer.order_snapshot(step)
          update(trainer, examples, visits[index + 2], step + 1, "swapped", target)
          save(model, output_dir, target, "swapped-1", step + 1)
          update(trainer, examples, visits[index + 1], target, "swapped", target)
          save(model, output_dir, target, "swapped-2", target)
          trainer.restore_order_snapshot(snapshot)
        elsif TARGETS.includes?(step + 1)
          save(model, output_dir, step + 1, "forward-1", step)
        elsif TARGETS.includes?(step)
          save(model, output_dir, step, "forward-2", step)
        end
      end
    ensure
      model.close
    end
  end
end

abort "usage: native-branches MODEL TRAIN_JSONL INSTRUMENT_JSONL INITIAL VISITS_JSON OUT" unless ARGV.size == 6
model, train, instrument, initial, visits, output_dir = ARGV
OrderBranches.run(model, train, instrument, initial, visits, output_dir)
