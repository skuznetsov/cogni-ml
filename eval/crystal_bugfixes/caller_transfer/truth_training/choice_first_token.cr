require "json"
require "../../../../src/ml/gguf/qwen35_qlora"
require "../../../../build/qlora/scale-2b/student-errors/eval_qwen35_sampler"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

module ML::GGUF::Qwen35QLoRA
  class Model
    def choice_first_token(messages : Array(Qwen35Chat::Message), labels : Tuple(Int32, Int32),
                           temperature : Float32, top_k : Int32, seed : UInt64)
      reserve = 128
      tokens = encode_prompt(messages)
      check_length(tokens.size + reserve)
      width = @weights.hparams.n_embd
      with_prefix_state(tokens.size + reserve) do |state|
        prefix = Qwen35CPU.qlora_prefix(@weights, tokens, 0, state, @prefix_layers)
        decode_states = Array(DecodeState).new(@layer_indices.size) { DecodeState.new }
        hidden = generation_prefill(prefix, state, decode_states)[-width, width]
        logits = hidden_logits(hidden)
        sampled = EvalSampler.sample(logits, temperature, top_k, Random::PCG32.new(seed))
        x, y = labels
        {prefix_tokens: tokens.size, x_logit: logits[x], y_logit: logits[y], sampled_id: sampled}
      end
    end
  end
end

def label_id(model : QLoRA::Model, text : String) : Int32
  id = model.tokenizer.token_to_id[text]?
  unless id && model.tokenizer.encode(text, add_bos_override: false) == [id] && model.tokenizer.decode_single(id) == text
    raise ArgumentError.new("#{text.inspect} must encode and decode as one exact token")
  end
  id
end

abort "usage: choice_first_token MODEL ADAPTER" unless ARGV.size == 2
model = QLoRA::Model.new(ARGV[0], full_blocks: 4)
begin
  model.load(ARGV[1])
  x = label_id(model, "X")
  y = label_id(model, "Y")
  raise ArgumentError.new("X and Y must have distinct token IDs") if x == y
  STDIN.each_line do |line|
    next if line.strip.empty?
    row = JSON.parse(line)
    sampling = row["sampling"]
    messages = QChat.messages_from_openai_json(row["messages"].to_json)
    readout = model.choice_first_token(messages, {x, y}, sampling["temperature"].as_f.to_f32,
      sampling["top_k"].as_i.to_i32, sampling["seed"].as_i64.to_u64)
    puts readout.merge({view: row["view"].as_s, x_id: x, y_id: y}).to_json
    STDOUT.flush
  end
ensure
  model.close
end
