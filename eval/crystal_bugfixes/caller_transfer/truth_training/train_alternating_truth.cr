require "json"
require "random/pcg32"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

module AlternatingTruthTraining
  extend self

  SEED        = 42_u64
  FULL_BLOCKS =      4

  def messages(row : JSON::Any) : Array(QChat::Message)
    QChat.messages_from_openai_json(row["messages"].to_json)
  end

  # Keep this loader in sync with native-recovery.cr: the frozen prefix can
  # recover from nonfinite output by retrying the original bounded chunks.
  def load_examples(model : QLoRA::Model, path : String) : Array(QLoRA::Example)
    rows = [] of QLoRA::Example
    File.each_line(path) do |line|
      next if line.strip.empty?
      conversation = messages(JSON.parse(line))
      example = model.example(conversation)
      unless example.input.all?(&.finite?)
        original_chunk = ENV["QWEN35_PREFILL_CHUNK_SIZE"]?
        begin
          [192, 64, 160, 32].each do |chunk|
            ENV["QWEN35_PREFILL_CHUNK_SIZE"] = chunk.to_s
            example = model.example(conversation)
            finite = example.input.all?(&.finite?)
            STDERR.puts({event: "prefill_recovery", file: File.basename(path), row: rows.size + 1,
                         chunk: chunk, finite: finite}.to_json)
            break if finite
          end
        ensure
          if original_chunk
            ENV["QWEN35_PREFILL_CHUNK_SIZE"] = original_chunk
          else
            ENV.delete("QWEN35_PREFILL_CHUNK_SIZE")
          end
        end
        raise "nonfinite frozen prefix after chunk recovery for #{path}:#{rows.size + 1}" unless example.input.all?(&.finite?)
      end
      rows << example
      STDERR.puts({event: "prefill", file: File.basename(path), row: rows.size,
                   prefix_tokens: example.first + 1, target_tokens: example.targets.size}.to_json)
    end
    raise ArgumentError.new("empty QLoRA dataset #{path}") if rows.empty?
    rows
  end

  def order(rows_text : String, updates_text : String, seed_text : String) : Nil
    rows = positive_int(rows_text, "rows")
    updates = positive_int(updates_text, "updates")
    seed = seed_text.to_u64?
    raise ArgumentError.new("seed must be a nonnegative integer") unless seed

    order = (0...rows).to_a
    random = Random::PCG32.new(seed.not_nil!)
    visits = [] of Int32
    epochs = (updates + rows - 1) // rows
    epochs.times do
      order.shuffle!(random: random)
      order.each { |index| visits << index + 1 }
    end
    STDOUT.puts(visits.to_json)
  end

  def train(model_path : String, train_path : String, instrument_path : String,
            initial_adapter : String, visits_path : String, checkpoints_path : String,
            output_dir : String, seed_text : String, blocks_text : String) : Nil
    seed = seed_text.to_u64?
    raise ArgumentError.new("training seed must be #{SEED}") unless seed == SEED
    full_blocks = blocks_text.to_i?
    raise ArgumentError.new("full_blocks must be #{FULL_BLOCKS}") unless full_blocks == FULL_BLOCKS
    training_seed = seed.not_nil!
    training_blocks = full_blocks.not_nil!

    visits = Array(Int32).from_json(File.read(visits_path))
    raise ArgumentError.new("visits must not be empty") if visits.empty?
    raise ArgumentError.new("visits must contain positive 1-based row indices") unless visits.all? { |row| row > 0 }

    requested_checkpoints = Array(Int32).from_json(File.read(checkpoints_path))
    checkpoints = requested_checkpoints.sort.uniq
    unless !checkpoints.empty? && checkpoints.all? { |step| 0 <= step <= visits.size }
      raise ArgumentError.new("checkpoint steps must be within 0..#{visits.size}")
    end
    unless checkpoints.includes?(visits.size)
      raise ArgumentError.new("checkpoints must include final visit #{visits.size}")
    end

    model = QLoRA::Model.new(model_path, training_seed, full_blocks: training_blocks)
    begin
      start = Time.instant
      model.load(initial_adapter)
      train = load_examples(model, train_path)
      valid = load_examples(model, instrument_path)
      unless visits.all? { |row| row <= train.size }
        raise ArgumentError.new("visits row indices must be within 1..#{train.size}")
      end

      checkpoint_paths = checkpoints.to_h do |step|
        directory = File.join(output_dir, "step-#{step.to_s.rjust(2, '0')}")
        {step, File.join(directory, "adapter.bin")}
      end
      if checkpoint_paths.values.any? { |path| File.exists?(path) }
        raise ArgumentError.new("a checkpoint adapter already exists")
      end

      trainer = QLoRA::Trainer.new(model)
      initial_loss = valid.sum { |row| trainer.loss(row) } / valid.size
      STDERR.puts({event: "start", model_sha256: model.digest, seed: seed, updates: visits.size,
                   examples: train.size, input_adapter: initial_adapter, blocks: model.layer_indices,
                   validation_loss: initial_loss,
                   seconds: (Time.instant - start).total_seconds}.to_json)

      checkpoint_set = checkpoints.to_set
      if checkpoint_set.includes?(0)
        adapter = checkpoint_paths[0]
        raise ArgumentError.new("checkpoint adapter already exists: #{adapter}") if File.exists?(adapter)
        Dir.mkdir_p(File.dirname(adapter))
        model.save(adapter)
        STDERR.puts({event: "saved", step: 0, adapter: adapter,
                     seconds: (Time.instant - start).total_seconds}.to_json)
      end

      visits.each_with_index do |row_index, index|
        sample = train[row_index - 1]
        step = index + 1
        loss = trainer.step(sample)
        STDERR.puts({event: "update", step: step, row: row_index,
                     loss: loss, seconds: (Time.instant - start).total_seconds,
                     first_target_id: sample.targets.first}.to_json)

        next unless checkpoint_set.includes?(step)

        adapter = checkpoint_paths[step]
        raise ArgumentError.new("checkpoint adapter already exists: #{adapter}") if File.exists?(adapter)
        Dir.mkdir_p(File.dirname(adapter))
        if step == visits.size
          final_loss = valid.sum { |row| trainer.loss(row) } / valid.size
          model.save(adapter)
          STDERR.puts({event: "saved", step: step, adapter: adapter,
                       seconds: (Time.instant - start).total_seconds,
                       validation_loss: final_loss}.to_json)
        else
          model.save(adapter)
          STDERR.puts({event: "saved", step: step, adapter: adapter,
                       seconds: (Time.instant - start).total_seconds}.to_json)
        end
      end
    ensure
      model.close
    end
  end

  private def positive_int(text : String, name : String) : Int32
    value = text.to_i?
    raise ArgumentError.new("#{name} must be a positive integer") unless value && value > 0
    value
  end
end

case {ARGV.first?, ARGV.size}
when {"order", 4}
  _, rows, updates, seed = ARGV
  AlternatingTruthTraining.order(rows, updates, seed)
when {"train", 10}
  _, model, train, instrument, initial, visits, checkpoints, output_dir, seed, blocks = ARGV
  AlternatingTruthTraining.train(model, train, instrument, initial, visits, checkpoints,
    output_dir, seed, blocks)
else
  abort "usage: native-checkpoints order ROWS UPDATES SEED\n" +
        "       native-checkpoints train MODEL TRAIN_JSONL INSTRUMENT_JSONL INITIAL VISITS_JSON CHECKPOINTS_JSON OUTPUT_DIR SEED FULL_BLOCKS"
end
