require "json"
require "random/pcg32"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

module PairedTruthTraining
  extend self

  SEED           =     42_u64
  UPDATES        =         50
  EXAMPLES       =         32
  FULL_BLOCKS    =          4
  LEARNING_RATE  = 0.0001_f32
  PAIR_WEIGHT    =    0.5_f32
  MINIMUM_MARGIN =    0.5_f32

  def examples(model : QLoRA::Model, path : String) : Array(QLoRA::Example)
    rows = [] of QLoRA::Example
    File.each_line(path) do |line|
      row = JSON.parse(line)
      messages = QChat.messages_from_openai_json(row["messages"].to_json)
      sample = model.example(messages)
      raise "nonfinite frozen prefix in #{row["id"]}" unless sample.input.all?(&.finite?)
      rows << sample
      STDERR.puts({
        event:         "prefill",
        row:           rows.size,
        id:            row["id"].as_s,
        prefix_tokens: sample.first + 1,
        target_tokens: sample.targets.size,
        label_id:      sample.targets.first,
      }.to_json)
    end
    raise ArgumentError.new("expected #{EXAMPLES} examples, got #{rows.size}") unless rows.size == EXAMPLES
    rows
  end

  def train(model_path : String, train_path : String, mates_path : String, initial : String, adapter : String) : Nil
    raise ArgumentError.new("adapter output already exists: #{adapter}") if File.exists?(adapter)
    mates = Array(Int32).from_json(File.read(mates_path)).map { |row| row - 1 }
    unless mates.size == EXAMPLES && mates.each_with_index.all? { |row, i|
             0 <= row < EXAMPLES && row != i && mates[row] == i
           }
      raise ArgumentError.new("expected an opposite-sign mate for every training row")
    end
    model = QLoRA::Model.new(model_path, SEED, full_blocks: FULL_BLOCKS)
    begin
      start = Time.instant
      model.load(initial)
      false_tokens = model.tokenizer.encode("false", add_bos_override: false)
      true_tokens = model.tokenizer.encode("true", add_bos_override: false)
      unless false_tokens.size == 1 && true_tokens.size == 1 && false_tokens != true_tokens
        raise ArgumentError.new("true and false must each encode as a distinct literal token")
      end
      choices = {false_tokens.first, true_tokens.first}
      rows = examples(model, train_path)
      trainer = QLoRA::Trainer.new(model, LEARNING_RATE, paired_choices: choices)
      STDERR.puts({
        event:          "start",
        model_sha256:   model.digest,
        input_adapter:  initial,
        seed:           SEED,
        learning_rate:  LEARNING_RATE,
        updates:        UPDATES,
        examples:       rows.size,
        blocks:         model.layer_indices,
        choices:        {"false" => choices[0], "true" => choices[1]},
        weight:         PAIR_WEIGHT,
        minimum_margin: MINIMUM_MARGIN,
      }.to_json)
      order = (0...rows.size).to_a
      random = Random::PCG32.new(SEED)
      UPDATES.times do |step|
        order.shuffle!(random: random) if step % order.size == 0
        index = order[step % order.size]
        mate_index = mates[index]
        result = trainer.step_paired(rows[index], rows[mate_index], weight: PAIR_WEIGHT,
          minimum_margin: MINIMUM_MARGIN)
        STDERR.puts({
          event:         "update",
          step:          step + 1,
          row:           index + 1,
          mate_row:      mate_index + 1,
          loss:          result[:loss],
          sft:           result[:sft],
          pair:          result[:pair],
          anchor_margin: result[:anchor_margin],
          mate_margin:   result[:mate_margin],
          seconds:       (Time.instant - start).total_seconds,
        }.to_json)
      end
      model.save(adapter)
      STDERR.puts({event: "saved", adapter: adapter, seconds: (Time.instant - start).total_seconds}.to_json)
    ensure
      model.close
    end
  end
end

unless ARGV.size == 5
  abort "usage: train_paired_truth MODEL TRAIN_JSONL MATES_JSON INPUT_ADAPTER OUTPUT_ADAPTER"
end

PairedTruthTraining.train(ARGV[0], ARGV[1], ARGV[2], ARGV[3], ARGV[4])
