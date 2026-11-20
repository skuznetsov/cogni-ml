require "digest/sha256"
require "json"
require "../../../../src/ml/gguf/qwen35_qlora"
require "./choice_pair_center"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

class ML::GGUF::Qwen35QLoRA::Trainer
  # The two backward passes accumulate on the same factors. The two-class
  # completion head computes softplus(-margin) for each required sign.
  def step_choice_pair(left : Example, right : Example, *, x_token : Int32, center_weight : Float32)
    @model.projections.each(&.zero_grad)
    if center_weight > 0_f32
      # Each layer retains only one tape. Read the right margin first, then
      # replay that forward after the left backward on the same factors.
      right_margin = @head.choice_margin_and_hidden_gradient(
        @model.forward(right.input, retain_tape: false), right.first, x_token, backward: false)[:margin]
      left_hidden = @model.forward(left.input, retain_tape: true)
      left_readout = @head.choice_margin_and_hidden_gradient(left_hidden, left.first, x_token)
      center = ChoicePairCenter.calculate(left_readout[:margin], right_margin, center_weight)
      left_loss = accumulate_choice_gradient(left, left_hidden, left_readout[:gradient], center[:scale])

      right_hidden = @model.forward(right.input, retain_tape: true)
      right_readout = @head.choice_margin_and_hidden_gradient(right_hidden, right.first, x_token)
      unless right_readout[:margin].finite? && (right_readout[:margin] - right_margin).abs <= 1e-4_f32
        raise "QLoRA pair margin changed during forward replay"
      end
      right_loss = accumulate_choice_gradient(right, right_hidden, right_readout[:gradient], center[:scale])
    else
      left_loss = accumulate_choice_gradient(left, @model.forward(left.input, retain_tape: true))
      right_loss = accumulate_choice_gradient(right, @model.forward(right.input, retain_tape: true))
    end

    grads = @model.projections.flat_map { |projection| [projection.grad_a, projection.grad_b] }
    @params.zip(grads).each do |param, grad|
      raise "nonfinite QLoRA factor gradient" unless grad.all?(&.finite?)
      param.grad = Tensor.from_array(grad, param.shape)
    end
    @adam.step
    @model.factors.zip(@params).each do |factor, param|
      factor.to_unsafe.copy_from(param.data.cpu_data.not_nil!.to_unsafe, factor.size)
    end
    ce_loss = (left_loss + right_loss) * 0.5
    {loss_a: left_loss, loss_b: right_loss, ce_loss: ce_loss, center: center,
     loss: ce_loss + (center.try(&.[:penalty]) || 0.0)}
  end

  def verify_choice_pair_adam!(expected : Int32) : Nil
    state = @adam.state_dict
    unless state.size == @params.size && state.each_value.all? { |(_, _, step)| step == expected }
      raise "expected one Adam update per pair per factor after #{expected} batches"
    end
  end

  private def accumulate_choice_gradient(example : Example, hidden : Array(Float32),
                                         margin_gradient : Array(Float32)? = nil, center_scale : Float32 = 0_f32) : Float64
    loss, gradient = @head.calculate(hidden, example.first, example.targets, backward: true)
    raise "nonfinite QLoRA choice loss" unless loss.finite?
    gradient.map! { |value| value * 0.5_f32 }
    if margin_gradient
      offset = example.first * @model.weights.hparams.n_embd
      margin_gradient.each_with_index do |value, index|
        gradient[offset + index] += center_scale * value
      end
    end
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

  def run(model_path : String, initial : String, train_path : String, schedule_path : String,
          output : String, center_weight : Float32) : Nil
    raise ArgumentError.new("adapter output already exists: #{output}") if File.exists?(output)
    unless center_weight.finite? && center_weight >= 0_f32
      raise ArgumentError.new("center weight must be nonnegative and finite")
    end
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
        updates: UPDATES, train_seed: TRAIN_SEED, learning_rate: LEARNING_RATE, labels: labels,
        center_weight: center_weight)

      # Repeated views share the frozen prefix. Materialize lazily to avoid
      # delaying the first update for the whole cohort.
      examples = Hash(Int32, QLoRA::Example).new
      trainer = QLoRA::Trainer.new(model, LEARNING_RATE, choices: {labels["X"], labels["Y"]})
      schedule.each_with_index do |pair, index|
        left, right = pair.map { |i| examples[i] ||= load_example(model, rows[i], labels) }
        unless left.targets.first != right.targets.first
          raise ArgumentError.new("each update requires opposite label targets")
        end
        report = trainer.step_choice_pair(left, right, x_token: labels["X"], center_weight: center_weight)
        step = index + 1
        trainer.verify_choice_pair_adam!(step)
        event("update", **{step: step, rows: pair, target_ids: [left.targets.first, right.targets.first],
                           seconds: (Time.instant - start).total_seconds}.merge(report))
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

unless (5..6).includes?(ARGV.size)
  abort "usage: train_choice_pairs MODEL TOOL_FOUR TRAIN_JSONL SCHEDULE_JSON OUTPUT_ADAPTER [CENTER_WEIGHT]"
end
TrainChoicePairs.run(ARGV[0], ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]?.try(&.to_f32) || 0_f32)
