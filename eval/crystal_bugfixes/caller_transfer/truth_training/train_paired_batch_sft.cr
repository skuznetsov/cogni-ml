require "digest/sha256"
require "json"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

class ML::GGUF::Qwen35QLoRA::Trainer
  def paired_batch_sft(true_example : Example, false_example : Example, step : Int32) : Tuple(Float64, Float64)
    unless true_example.targets.first? == PairedBatchSFT::TRUE_ID &&
           false_example.targets.first? == PairedBatchSFT::FALSE_ID
      raise ArgumentError.new("paired batch must contain true then false examples")
    end
    unless true_example.targets.size == PairedBatchSFT::TARGET_TOKENS &&
           false_example.targets.size == PairedBatchSFT::TARGET_TOKENS
      raise ArgumentError.new("paired batch examples must each have four target tokens")
    end

    check_adam_state(step - 1)
    weights = @model.factors.map(&.dup)
    moments = snapshot_adam_state

    true_loss, true_gradient = full_sft_gradient(true_example)
    check_snapshot(weights, moments)
    false_loss, false_gradient = full_sft_gradient(false_example)
    check_snapshot(weights, moments)

    apply_gradient(mean_gradient(true_gradient, false_gradient))
    check_adam_state(step)
    {true_loss, false_loss}
  end

  private def full_sft_gradient(example : Example) : Tuple(Float64, Array(Array(Float32)))
    @model.projections.each(&.zero_grad)
    hidden = @model.forward(example.input, retain_tape: true)
    loss, output_gradient = @head.calculate(hidden, example.first, example.targets, backward: true)
    raise "nonfinite QLoRA loss" unless loss.finite?
    @model.backward(output_gradient)
    gradients = @model.projections.flat_map { |projection| [projection.grad_a.dup, projection.grad_b.dup] }
    raise "nonfinite QLoRA factor gradient" unless gradients.all? { |gradient| gradient.all?(&.finite?) }
    {loss, gradients}
  end

  private def mean_gradient(left : Array(Array(Float32)), right : Array(Array(Float32))) : Array(Array(Float32))
    raise "incomplete QLoRA gradients" unless left.size == right.size && left.size == @params.size
    left.zip(right).map do |left_factor, right_factor|
      raise "QLoRA gradient shape mismatch" unless left_factor.size == right_factor.size
      Array(Float32).new(left_factor.size) do |index|
        (left_factor[index] + right_factor[index]) * 0.5_f32
      end
    end
  end

  private def apply_gradient(gradients : Array(Array(Float32))) : Nil
    raise "incomplete QLoRA gradients" unless gradients.size == @params.size
    @params.zip(gradients).each do |parameter, gradient|
      raise "nonfinite QLoRA factor gradient" unless gradient.all?(&.finite?)
      parameter.grad = Tensor.from_array(gradient, parameter.shape)
    end
    @adam.step
    @model.factors.zip(@params).each do |factor, parameter|
      factor.to_unsafe.copy_from(parameter.data.cpu_data.not_nil!.to_unsafe, factor.size)
    end
  end

  private def snapshot_adam_state : Hash(String, {Tensor, Tensor, Int32})
    @adam.state_dict.transform_values { |(first, second, updates)| {first.clone, second.clone, updates} }
  end

  private def check_adam_state(expected_step : Int32) : Nil
    state = @adam.state_dict
    expected_size = expected_step == 0 ? 0 : @params.size
    raise "unexpected Adam state size #{state.size}, expected #{expected_size}" unless state.size == expected_size
    state.each_value do |(first, second, updates)|
      raise "unexpected Adam step #{updates}, expected #{expected_step}" unless updates == expected_step
      first_values = first.cpu_data.not_nil!
      second_values = second.cpu_data.not_nil!
      unless first_values.all?(&.finite?) && second_values.all? { |value| value.finite? && value >= 0_f32 }
        raise "invalid Adam moments"
      end
    end
  end

  private def check_snapshot(weights : Array(Array(Float32)),
                             moments : Hash(String, {Tensor, Tensor, Int32})) : Nil
    @model.factors.zip(@params).each_with_index do |(factor, parameter), index|
      snapshot = weights[index]
      unless factor == snapshot && parameter.data.cpu_data == snapshot
        raise "SFT gradient pass changed adapter weights"
      end
    end
    state = @adam.state_dict
    raise "SFT gradient pass changed Adam state size" unless state.size == moments.size
    moments.each do |id, (first, second, updates)|
      live_first, live_second, live_updates = state[id]
      unless live_updates == updates && live_first.cpu_data == first.cpu_data &&
             live_second.cpu_data == second.cpu_data
        raise "SFT gradient pass changed Adam state"
      end
    end
  end
end

module PairedBatchSFT
  extend self

  SEED          = 42_u64
  EXAMPLES      =     32
  UPDATES       =     16
  TARGET_TOKENS =      4
  TRUE_ID       =   1802
  FALSE_ID      =   3721
  BLOCKS        = [11, 15, 19, 23]

  def run(model_path : String, train_path : String, initial_path : String,
          batches_path : String, output_dir : String) : Nil
    batches = Array(Array(Int32)).from_json(File.read(batches_path))
    validate_batches!(batches)
    adapter_path = File.join(output_dir, "adapter.bin")
    raise ArgumentError.new("adapter output already exists: #{adapter_path}") if File.exists?(adapter_path)

    initial_sha256 = File.open(initial_path) { |file| Digest::SHA256.hexdigest(file) }
    model = QLoRA::Model.new(model_path, SEED, full_blocks: 4)
    begin
      unless model.layer_indices == BLOCKS
        raise ArgumentError.new("expected full-attention blocks #{BLOCKS}, got #{model.layer_indices}")
      end
      model.load(initial_path)
      validate_literal_token!(model, "true", TRUE_ID)
      validate_literal_token!(model, "false", FALSE_ID)
      eos_id = model.tokenizer.token_to_id["<|im_end|>"]?
      raise ArgumentError.new("tokenizer has no <|im_end|> token") unless eos_id

      STDERR.puts({event: "start", model_sha256: model.digest,
                   initial_adapter_sha256: initial_sha256, seed: SEED,
                   examples: EXAMPLES, blocks: model.layer_indices}.to_json)

      rows = load_examples(model, train_path, eos_id)
      batches.each_with_index do |batch, index|
        true_row = batch[0]
        false_row = batch[1]
        unless rows[true_row - 1].targets.first == TRUE_ID && rows[false_row - 1].targets.first == FALSE_ID
          raise ArgumentError.new("batch #{index + 1} must list one true row before one false row")
        end
      end

      trainer = QLoRA::Trainer.new(model)
      total_target_tokens = 0
      batches.each_with_index do |batch, index|
        step = index + 1
        true_row, false_row = batch
        true_example = rows[true_row - 1]
        false_example = rows[false_row - 1]
        losses = trainer.paired_batch_sft(true_example, false_example, step)
        total_target_tokens += true_example.targets.size + false_example.targets.size
        STDERR.puts({event: "update", step: step, rows: [true_row, false_row],
                     first_target_ids: [TRUE_ID, FALSE_ID],
                     target_tokens: [true_example.targets.size, false_example.targets.size],
                     losses: [losses[0], losses[1]], adam_after: step}.to_json)
      end
      raise "expected exactly 128 supervised target tokens, got #{total_target_tokens}" unless total_target_tokens == 128
      raise ArgumentError.new("adapter output appeared during training: #{adapter_path}") if File.exists?(adapter_path)
      Dir.mkdir_p(output_dir)
      model.save(adapter_path)
      STDERR.puts({event: "saved", adapter: adapter_path, updates: UPDATES,
                   target_tokens: total_target_tokens}.to_json)
    ensure
      model.close
    end
  end

  private def validate_batches!(batches : Array(Array(Int32))) : Nil
    raise ArgumentError.new("expected #{UPDATES} batches") unless batches.size == UPDATES
    unless batches.all? { |batch| batch.size == 2 }
      raise ArgumentError.new("each batch must contain exactly two row numbers")
    end
    rows = batches.flatten
    unless rows.all? { |row| 1 <= row <= EXAMPLES } && rows.sort == (1..EXAMPLES).to_a
      raise ArgumentError.new("batches must visit every 1-based training row exactly once")
    end
  end

  private def validate_literal_token!(model : QLoRA::Model, text : String, expected_id : Int32) : Nil
    id = model.tokenizer.token_to_id[text]?
    encoded = model.tokenizer.encode(text, add_bos_override: false)
    unless id == expected_id && encoded == [expected_id] && model.tokenizer.decode_single(expected_id) == text
      raise ArgumentError.new("#{text.inspect} must be the exact token #{expected_id}")
    end
  end

  private def load_examples(model : QLoRA::Model, path : String, eos_id : Int32) : Array(QLoRA::Example)
    rows = [] of QLoRA::Example
    true_rows = 0
    false_rows = 0
    File.each_line(path) do |line|
      next if line.strip.empty?
      row = JSON.parse(line)
      messages = QChat.messages_from_openai_json(row["messages"].to_json)
      answer = messages.last.content
      expected_id = case answer
                    when "true, false"
                      true_rows += 1
                      TRUE_ID
                    when "false, true"
                      false_rows += 1
                      FALSE_ID
                    else
                      raise ArgumentError.new("unexpected training answer #{answer.inspect}")
                    end
      example = model.example(messages)
      unless example.input.all?(&.finite?)
        raise ArgumentError.new("nonfinite frozen prefix in training row #{rows.size + 1}")
      end
      unless example.targets.size == TARGET_TOKENS && example.targets.first == expected_id && example.targets.last == eos_id
        raise ArgumentError.new("training row #{rows.size + 1} must have four targets with its expected label and EOS")
      end
      rows << example
      STDERR.puts({event: "prefill", row: rows.size, first_target_id: example.targets.first,
                   prefix_tokens: example.first + 1, target_tokens: example.targets.size}.to_json)
    end
    unless rows.size == EXAMPLES && true_rows == EXAMPLES // 2 && false_rows == EXAMPLES // 2
      raise ArgumentError.new("expected #{EXAMPLES} rows with sixteen true and sixteen false targets")
    end
    rows
  end
end

unless ARGV.size == 5
  abort "usage: train_paired_batch_sft MODEL TRAIN_JSONL INITIAL BATCHES_JSON OUT"
end

model_path, train_path, initial_path, batches_path, output_dir = ARGV
PairedBatchSFT.run(model_path, train_path, initial_path, batches_path, output_dir)
