require "digest/sha256"
require "json"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

class ML::GGUF::Qwen35QLoRA::Trainer
  # The two backward passes accumulate on the same factors. The two-class
  # completion head computes softplus(-margin) for each required sign.
  def step_choice_pair(left : Example, right : Example) : Tuple(Float64, Float64)
    @model.projections.each(&.zero_grad)
    left_loss = accumulate_choice_gradient(left)
    right_loss = accumulate_choice_gradient(right)

    grads = @model.projections.flat_map { |projection| [projection.grad_a, projection.grad_b] }
    @params.zip(grads).each do |param, grad|
      raise "nonfinite QLoRA factor gradient" unless grad.all?(&.finite?)
      param.grad = Tensor.from_array(grad, param.shape)
    end
    @adam.step
    @model.factors.zip(@params).each do |factor, param|
      factor.to_unsafe.copy_from(param.data.cpu_data.not_nil!.to_unsafe, factor.size)
    end
    {left_loss, right_loss}
  end

  def verify_choice_pair_adam!(expected : Int32) : Nil
    state = @adam.state_dict
    unless state.size == @params.size && state.each_value.all? { |(_, _, step)| step == expected }
      raise "expected one Adam update per pair per factor after #{expected} batches"
    end
  end

  private def accumulate_choice_gradient(example : Example) : Float64
    hidden = @model.forward(example.input, retain_tape: true)
    loss, gradient = @head.calculate(hidden, example.first, example.targets, backward: true)
    raise "nonfinite QLoRA choice loss" unless loss.finite?
    gradient.map! { |value| value * 0.5_f32 }
    @model.backward(gradient)
    loss
  end
end

module TrainChoicePairs
  extend self

  UPDATES        =         50
  TRAIN_SEED     =     42_u64
  LEARNING_RATE  = 0.0001_f32
  BLOCKS         = [11, 15, 19, 23]
  ANSWER_RESERVE = 128

  def label_id(model : QLoRA::Model, label : String) : Int32
    id = model.tokenizer.token_to_id[label]?
    unless id && model.tokenizer.encode(label, add_bos_override: false) == [id] &&
           model.tokenizer.decode_single(id) == label
      raise ArgumentError.new("#{label.inspect} must be an exact single token")
    end
    id
  end

  def load_example(model : QLoRA::Model, body : JSON::Any, labels : Hash(String, Int32)) : QLoRA::Example
    messages = QChat.messages_from_openai_json(body["messages"].to_json)
    unless messages.map(&.role) == ["system", "user", "assistant"]
      raise ArgumentError.new("unexpected choice training roles")
    end
    target = labels[messages.last.content]
    example = model.example(messages, first_token_only: true)
    prefix = example.first + 1
    unless example.targets == [target] && prefix == body["prefix_tokens"].as_i &&
           prefix + ANSWER_RESERVE <= QLoRA::MAX_SEQ && example.input.all?(&.finite?)
      raise ArgumentError.new("invalid first-token example #{body["id"]}")
    end
    event("prefill", id: body["id"].as_s, prefix_tokens: prefix, target_id: target)
    example
  end

  def event(name : String, **fields) : Nil
    STDERR.puts({event: name}.merge(fields).to_json)
    STDERR.flush
  end

  def run(model_path : String, initial : String, train_path : String, schedule_path : String, output : String) : Nil
    raise ArgumentError.new("adapter output already exists: #{output}") if File.exists?(output)
    schedule = Array(Array(Int32)).from_json(File.read(schedule_path))
    rows = File.read_lines(train_path).map { |line| JSON.parse(line) }
    unless schedule.size == UPDATES &&
           schedule.all? { |pair| pair.size == 2 && pair.uniq.size == 2 } &&
           schedule.flatten.uniq.sort == (0...rows.size).to_a
      raise ArgumentError.new("expected #{UPDATES} two-example batches visiting the full training plan")
    end

    model = QLoRA::Model.new(model_path, TRAIN_SEED, full_blocks: BLOCKS.size)
    start = Time.instant
    begin
      raise ArgumentError.new("unexpected adapter placement") unless model.layer_indices == BLOCKS
      model.load(initial)
      labels = {"X" => label_id(model, "X"), "Y" => label_id(model, "Y")}
      raise ArgumentError.new("training labels must have distinct IDs") unless labels.values.uniq.size == 2
      event("start", model_sha256: model.digest, initial_sha256: File.open(initial) { |io| Digest::SHA256.hexdigest(io) },
        rank: QLoRA::RANK, blocks: model.layer_indices, projections: model.projections.size,
        updates: UPDATES, train_seed: TRAIN_SEED, learning_rate: LEARNING_RATE, labels: labels)

      # Repeated views share the frozen prefix. Materialize lazily to avoid
      # delaying the first update for the whole cohort.
      examples = Hash(Int32, QLoRA::Example).new
      trainer = QLoRA::Trainer.new(model, LEARNING_RATE, choices: {labels["X"], labels["Y"]})
      schedule.each_with_index do |pair, index|
        left, right = pair.map { |i| examples[i] ||= load_example(model, rows[i], labels) }
        unless left.targets.first != right.targets.first
          raise ArgumentError.new("each update requires opposite label targets")
        end
        a, b = trainer.step_choice_pair(left, right)
        step = index + 1
        trainer.verify_choice_pair_adam!(step)
        event("update", step: step, rows: pair, loss_a: a, loss_b: b, loss: (a + b) * 0.5,
          target_ids: [left.targets.first, right.targets.first], seconds: (Time.instant - start).total_seconds)
      end

      raise ArgumentError.new("adapter output appeared during training: #{output}") if File.exists?(output)
      model.save(output)
      event("saved", updates: schedule.size, target_tokens: schedule.size * 2,
        train_rows: examples.size, seconds: (Time.instant - start).total_seconds)
    ensure
      model.close
    end
  end
end

abort "usage: train_choice_pairs MODEL TOOL_FOUR TRAIN_JSONL SCHEDULE_JSON OUTPUT_ADAPTER" unless ARGV.size == 5
TrainChoicePairs.run(ARGV[0], ARGV[1], ARGV[2], ARGV[3], ARGV[4])
