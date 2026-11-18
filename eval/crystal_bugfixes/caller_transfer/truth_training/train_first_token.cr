require "json"
require "random/pcg32"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

module FirstTokenTraining
  extend self

  SEED          =     42_u64
  UPDATES       =         50
  EXAMPLES      =         32
  FULL_BLOCKS   =          4
  LEARNING_RATE = 0.0001_f32

  def literal_id(model : QLoRA::Model, literal : String) : Int32
    tokens = model.tokenizer.encode(literal, add_bos_override: false)
    unless tokens.size == 1 && model.tokenizer.decode_single(tokens[0]) == literal
      raise ArgumentError.new("#{literal.inspect} must be a single literal token")
    end
    tokens.first
  end

  def examples(model : QLoRA::Model, path : String, labels : Hash(String, Int32)) : Array(QLoRA::Example)
    rows = [] of QLoRA::Example
    width = model.weights.hparams.n_embd
    File.each_line(path) do |line|
      row = JSON.parse(line)
      messages = QChat.messages_from_openai_json(row["messages"].to_json)
      label = labels[messages.last.content.split(',', 2).first]
      sample = model.example(messages, first_token_only: true)
      unless sample.targets == [label] && sample.input.size == (sample.first + 1) * width
        raise ArgumentError.new("expected one Boolean target after the prompt in #{row["id"]}")
      end
      raise "nonfinite frozen prefix in #{row["id"]}" unless sample.input.all?(&.finite?)

      rows << sample
      STDERR.puts({
        event:         "prefill",
        row:           rows.size,
        id:            row["id"].as_s,
        prefix_tokens: sample.first + 1,
        target_tokens: sample.targets.size,
        label_id:      label,
      }.to_json)
    end
    raise ArgumentError.new("expected #{EXAMPLES} examples, got #{rows.size}") unless rows.size == EXAMPLES
    rows
  end

  def train(model_path : String, train_path : String, initial : String, adapter : String) : Nil
    raise ArgumentError.new("adapter output already exists: #{adapter}") if File.exists?(adapter)
    model = QLoRA::Model.new(model_path, SEED, full_blocks: FULL_BLOCKS)
    begin
      start = Time.instant
      model.load(initial)
      labels = {"false" => literal_id(model, "false"), "true" => literal_id(model, "true")}
      rows = examples(model, train_path, labels)
      trainer = QLoRA::Trainer.new(model, LEARNING_RATE, choices: {labels["false"], labels["true"]})
      STDERR.puts({
        event:         "start",
        model_sha256:  model.digest,
        input_adapter: initial,
        seed:          SEED,
        learning_rate: LEARNING_RATE,
        updates:       UPDATES,
        examples:      rows.size,
        blocks:        model.layer_indices,
        choices:       labels,
      }.to_json)
      order = (0...rows.size).to_a
      random = Random::PCG32.new(SEED)
      UPDATES.times do |step|
        order.shuffle!(random: random) if step % order.size == 0
        index = order[step % order.size]
        loss = trainer.step(rows[index])
        STDERR.puts({
          event:   "update",
          step:    step + 1,
          row:     index + 1,
          loss:    loss,
          seconds: (Time.instant - start).total_seconds,
        }.to_json)
      end
      model.save(adapter)
      STDERR.puts({event: "saved", adapter: adapter, seconds: (Time.instant - start).total_seconds}.to_json)
    ensure
      model.close
    end
  end
end

unless ARGV.size == 4
  abort "usage: train_first_token MODEL TRAIN_JSONL INPUT_ADAPTER OUTPUT_ADAPTER"
end

FirstTokenTraining.train(ARGV[0], ARGV[1], ARGV[2], ARGV[3])
