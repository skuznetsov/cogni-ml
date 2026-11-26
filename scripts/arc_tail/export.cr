# Frozen-prefix export and independent numerical parity oracle. No training/API.
require "json"
require "option_parser"
require "digest/sha256"
require "../../src/ml/gguf/qwen35_cpu"
require "../../src/ml/gguf/qwen35_tokenizer"
require "../../src/ml/gguf/qwen35_weights"

def file_sha256(path : String) : String
  digest = Digest::SHA256.new
  buffer = Bytes.new(1 << 20)
  File.open(path) do |f|
    while (n = f.read(buffer)) > 0
      digest.update(buffer[0, n])
    end
  end
  digest.final.hexstring
end

model = ""
requests = ""
output = ""
max_tokens = 256
max_records = 32
parity_only = false
OptionParser.parse do |p|
  p.on("--model PATH", "Local GGUF") { |v| model = v }
  p.on("--requests PATH", "ARC request JSONL") { |v| requests = v }
  p.on("--output PATH", "New output JSONL") { |v| output = v }
  p.on("--max-tokens N", "Reject long prompts, never truncate") { |v| max_tokens = v.to_i }
  p.on("--max-records N", "Resource cap") { |v| max_records = v.to_i }
  p.on("--parity-only", "Numerical last-block probe; no ARC training") { parity_only = true }
end
raise "local model required" unless File.file?(model)
raise "new output required" if output.empty? || File.exists?(output)
raise "positive resource caps required" unless max_tokens > 0 && max_records > 0
raise "ARC requests required" unless parity_only || File.file?(requests)

weights = ML::GGUF::Qwen35Weights.from_gguf(model)
hp = weights.hparams
last = weights.layers.last.as?(ML::GGUF::Qwen35FullAttnWeights)
raise "last model block must be full attention" unless last
tokenizer = ML::GGUF::Qwen35Tokenizer.from_gguf(ML::GGUF::GGUFFile.new(model), model)
label_ids = "ABCDEFGH".chars.map do |c|
  ids = tokenizer.encode(c.to_s, add_bos_override: false)
  raise "label #{c} is not a single token" unless ids.size == 1
  ids[0]
end

File.open(output, "w") do |file|
  file.puts({"kind" => "header", "model" => File.expand_path(model),
             "model_sha256" => file_sha256(model),
             "requests_sha256" => parity_only ? nil : file_sha256(requests),
             "label_ids" => label_ids, "last_layer" => weights.layers.size - 1,
             "hidden" => hp.n_embd, "max_tokens" => max_tokens,
             "max_records" => max_records, "parity_only" => parity_only}.to_json)
  if parity_only
    # Deliberately nonzero positions: this catches incorrect causal attention,
    # grouped-query layout, Q/gate splitting, and partial RoPE.
    state = ML::GGUF::Qwen35CPU::State.new(hp, 4)
    xs = Array(Array(Float32)).new
    ys = Array(Array(Float32)).new
    4.times do |pos|
      x = Array(Float32).new(hp.n_embd) { |i| Math.sin((i + 1) * 0.017 + pos * 0.31).to_f32 }
      xs << x
      ys << ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, last,
        state.layers.last, hp, 4)
    end
    file.puts({"kind" => "parity", "inputs" => xs, "reference" => ys}.to_json)
  else
    count = 0
    File.each_line(requests) do |line|
      next if line.strip.empty?
      raise "request count exceeds cap" if count >= max_records
      row = JSON.parse(line)
      task_id = row["task_id"].as_s
      ids = tokenizer.encode(row["prompt"].as_s, add_bos_override: false)
      raise "empty prompt for #{task_id}" if ids.empty?
      if ids.size > max_tokens
        file.puts({"kind" => "skipped", "task_id" => task_id,
                   "reason" => "token_cap", "token_count" => ids.size}.to_json)
        count += 1
        next
      end
      state = ML::GGUF::Qwen35CPU::State.new(hp, ids.size)
      ML::GGUF::Qwen35CPU.prepare_state_metal!(state, hp)
      xs = Array(Array(Float32)).new
      ys = Array(Array(Float32)).new
      ids.each_with_index do |id, pos|
        x = ML::GGUF::Qwen35CPU.embedding_lookup(weights.token_embd, id)
        weights.layers.each_with_index do |lw, layer|
          xs << x.dup if layer == weights.layers.size - 1
          case lw
          in ML::GGUF::Qwen35FullAttnWeights
            x = ML::GGUF::Qwen35CPU.forward_full_attn_layer(x, pos, lw, state.layers[layer], hp, ids.size)
          in ML::GGUF::Qwen35RecurrentWeights
            x = ML::GGUF::Qwen35CPU.forward_recurrent_layer(x, pos, lw, state.layers[layer], hp, ids.size)
          end
        end
        ys << x
      end
      file.puts({"kind" => "task", "task_id" => task_id,
                 "token_ids" => ids, "inputs" => xs, "reference" => ys}.to_json)
      STDERR.puts "exported #{task_id}: #{ids.size} tokens"
      count += 1
    end
  end
end
