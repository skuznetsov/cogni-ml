require "json"
require "../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

def messages(row : JSON::Any) : Array(QChat::Message)
  QChat.messages_from_openai_json(row["messages"].to_json)
end

def load_examples(model : QLoRA::Model, path : String) : Array(QLoRA::Example)
  rows = [] of QLoRA::Example
  File.each_line(path) do |line|
    next if line.strip.empty?
    example = model.example(messages(JSON.parse(line)))
    rows << example
    STDERR.puts({event: "prefill", file: File.basename(path), row: rows.size,
                 prefix_tokens: example.first + 1, target_tokens: example.targets.size}.to_json)
  end
  raise ArgumentError.new("empty QLoRA dataset #{path}") if rows.empty?
  rows
end

case {ARGV.first?, ARGV.size}
when {"train", 7}, {"train", 8}
  _, path, train_path, valid_path, adapter, updates_text, seed_text = ARGV
  updates = updates_text.to_i
  seed = seed_text.to_u64
  full_blocks = ARGV[7]?.try(&.to_i) || 1
  abort "updates must be positive" unless updates > 0
  abort "adapter output already exists" if File.exists?(adapter)
  model = QLoRA::Model.new(path, seed, full_blocks: full_blocks)
  begin
    start = Time.instant
    train = load_examples(model, train_path)
    valid = load_examples(model, valid_path)
    trainer = QLoRA::Trainer.new(model)
    initial = valid.sum { |row| trainer.loss(row) } / valid.size
    STDERR.puts({event: "start", model_sha256: model.digest, seed: seed, updates: updates,
                 examples: train.size, blocks: model.layer_indices, validation_loss: initial,
                 seconds: (Time.instant - start).total_seconds}.to_json)
    order = (0...train.size).to_a
    random = Random::PCG32.new(seed)
    updates.times do |i|
      order.shuffle!(random: random) if i % order.size == 0
      index = order[i % order.size]
      loss = trainer.step(train[index])
      STDERR.puts({event: "update", step: i + 1, row: index + 1, loss: loss,
                   seconds: (Time.instant - start).total_seconds}.to_json)
    end
    final = valid.sum { |row| trainer.loss(row) } / valid.size
    model.save(adapter)
    STDERR.puts({event: "saved", validation_loss: final, adapter: adapter, seconds: (Time.instant - start).total_seconds}.to_json)
  ensure
    model.close
  end
when {"generate", 4}, {"generate", 5}
  _, path, adapter, limit_text = ARGV
  limit = limit_text.to_i
  abort "invalid generation limit" unless 0 < limit < QLoRA::MAX_SEQ
  full_blocks = ARGV[4]?.try(&.to_i) || (adapter == "-" ? 1 : QLoRA::Model.adapter_full_blocks(adapter))
  model = QLoRA::Model.new(path, full_blocks: full_blocks)
  begin
    model.load(adapter) unless adapter == "-"
    STDIN.each_line do |line|
      row = JSON.parse(line)
      answer, count, stop = model.generate(messages(row), limit)
      STDOUT.puts({id: row["id"].as_s, answer: answer, tokens: count, stop: stop}.to_json)
      STDOUT.flush
    end
  ensure
    model.close
  end
else
  abort "usage: qwen35_qlora train MODEL TRAIN_JSONL VALID_JSONL ADAPTER UPDATES SEED [FULL_BLOCKS]\n" +
        "       qwen35_qlora generate MODEL ADAPTER_OR_DASH MAX_NEW_TOKENS [FULL_BLOCKS]"
end
