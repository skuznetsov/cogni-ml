require "json"
require "../../../../src/ml/gguf/qwen35_qlora"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

struct RestartBranch
  include JSON::Serializable

  getter target : Int32
  getter order : String
  getter common : String
  getter common_sha256 : String
  getter rows : Array(Int32)
end

class ML::GGUF::Qwen35QLoRA::Trainer
  def check_adam_counter(step : Int32) : Nil
    state = @adam.state_dict
    if step.zero?
      raise "expected fresh Adam state" unless state.empty?
      return
    end

    raise "incomplete Adam state" unless state.size == @params.size == 56
    state.each_value do |(first, second, updates)|
      raise "unexpected Adam step #{updates}, expected #{step}" unless updates == step
      raise "nonfinite Adam first moment" unless first.cpu_data.not_nil!.all?(&.finite?)
      unless second.cpu_data.not_nil!.all? { |value| value.finite? && value >= 0.0_f32 }
        raise "invalid Adam second moment"
      end
    end
    raise "nonfinite updated factors" unless @model.factors.all? { |factor| factor.all?(&.finite?) }
  end
end

def load_examples(model, path)
  rows = [] of QLoRA::Example
  File.each_line(path) do |line|
    next if line.strip.empty?
    messages = QChat.messages_from_openai_json(JSON.parse(line)["messages"].to_json)
    example = model.example(messages)
    raise "nonfinite frozen prefix" unless example.input.all?(&.finite?)
    rows << example
    STDERR.puts({event: "prefill", file: File.basename(path), row: rows.size,
                 prefix_tokens: example.first + 1, target_tokens: example.targets.size}.to_json)
  end
  raise "expected 32 training rows" unless rows.size == 32
  rows
end

def restart_adam(model, examples, branch, output_dir) : Nil
  target, order = branch.target, branch.order
  raise "invalid target" unless {10, 25, 50}.includes?(target)
  raise "invalid order" unless {"forward", "swapped"}.includes?(order)
  unless branch.rows.size == 2 && branch.rows.all? { |row| 1 <= row <= examples.size }
    raise "expected two training rows"
  end

  common_sha = File.open(branch.common) { |io| Digest::SHA256.hexdigest(io) }
  raise "common checkpoint changed" unless common_sha == branch.common_sha256
  model.load(branch.common)
  trainer = QLoRA::Trainer.new(model)
  trainer.check_adam_counter(0)
  STDERR.puts({event: "reset_adam", target: target, order: order, step: target - 2,
               adam_step: 0, common_sha256: common_sha}.to_json)

  branch.rows.each_with_index do |row, index|
    adam_step = index + 1
    step = target - 2 + adam_step
    example = examples[row - 1]
    loss = trainer.step(example)
    trainer.check_adam_counter(adam_step)
    STDERR.puts({event: "update", target: target, order: order, step: step, row: row,
                 first_target_id: example.targets.first, loss: loss, adam_step: adam_step}.to_json)

    phase = "#{order}-#{adam_step}"
    adapter = File.join(output_dir, "target-#{target}", phase, "adapter.bin")
    raise "checkpoint already exists at #{adapter}" if File.exists?(adapter)
    Dir.mkdir_p(File.dirname(adapter))
    model.save(adapter)
    STDERR.puts({event: "saved", target: target, phase: phase, step: step, adapter: adapter}.to_json)
  end
end

abort "usage: native-reset MODEL TRAIN_JSONL BRANCHES_JSON OUT" unless ARGV.size == 4
model_path, train_path, branches_path, output_dir = ARGV
branches = Array(RestartBranch).from_json(File.read(branches_path))
expected = {10, 25, 50}.flat_map { |target| [{"forward", target}, {"swapped", target}] }
raise "unexpected branch schedule" unless branches.map { |branch| {branch.order, branch.target} } == expected

model = QLoRA::Model.new(model_path, 42_u64, full_blocks: 4)
begin
  examples = load_examples(model, train_path)
  STDERR.puts({event: "start", model_sha256: model.digest, seed: 42,
               examples: examples.size, blocks: model.layer_indices}.to_json)
  branches.each { |branch| restart_adam(model, examples, branch, output_dir) }
ensure
  model.close
end
