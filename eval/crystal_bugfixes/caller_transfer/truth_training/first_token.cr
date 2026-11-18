require "json"
require "random/pcg32"
require "../../../../src/ml/gguf/qwen35_qlora"
require "../../../../build/qlora/scale-2b/student-errors/eval_qwen35_sampler"

alias QLoRA = ML::GGUF::Qwen35QLoRA
alias QChat = ML::GGUF::Qwen35Chat

module ML::GGUF::Qwen35QLoRA
  # Materialize the first logits with the historical evaluator's exact
  # QLoRA-prefill and adapted-tail route. The state objects are request-local.
  class Model
    def first_token_logits(messages : Array(Qwen35Chat::Message), max_new : Int32,
                           temperature : Float32, top_k : Int32,
                           seed : UInt64) : Tuple(Int32, Array(Float32), Int32)
      raise ArgumentError.new("invalid QLoRA generation limit") unless 0 < max_new < MAX_SEQ
      raise ArgumentError.new("sampling temperature must be finite and positive") unless temperature.finite? && temperature > 0_f32
      raise ArgumentError.new("sampling top-k must be positive") unless top_k > 0

      tokens = encode_prompt(messages)
      check_length(tokens.size + max_new)
      width = @weights.hparams.n_embd
      with_prefix_state(tokens.size + max_new) do |state|
        prefix = Qwen35CPU.qlora_prefix(@weights, tokens, 0, state, @prefix_layers)
        decode_states = Array(DecodeState).new(@layer_indices.size) { DecodeState.new }
        hidden = generation_prefill(prefix, state, decode_states)[-width, width]
        logits = hidden_logits(hidden)
        sampled_id = EvalSampler.sample(logits, temperature, top_k, Random::PCG32.new(seed))
        {tokens.size, logits, sampled_id}
      end
    end
  end
end

module FirstTokenReadout
  extend self

  def candidate_id(model : QLoRA::Model, text : String) : Int32
    id = model.tokenizer.token_to_id[text]?
    raise ArgumentError.new("tokenizer has no exact vocabulary token #{text.inspect}") unless id
    encoded = model.tokenizer.encode(text, add_bos_override: false)
    unless encoded == [id] && model.tokenizer.decode_single(id) == text
      raise ArgumentError.new("#{text.inspect} must encode as one token and decode literally")
    end
    id
  end

  def raw_probabilities(logits : Array(Float32)) : Array(Float64)
    raise ArgumentError.new("first-token logits are empty") if logits.empty?
    raise ArgumentError.new("first-token logits must be finite") unless logits.all?(&.finite?)

    maximum = logits.max.to_f64
    masses = logits.map { |value| Math.exp(value.to_f64 - maximum) }
    total = masses.sum
    raise ArgumentError.new("raw softmax has no finite mass") unless total.finite? && total > 0.0
    masses.map! { |mass| mass / total }
    masses
  end

  # Mirrors EvalSampler.sample's Float32 temperature scaling, top-k mask, and
  # Float64 softmax so candidate probabilities use the actual sampling law.
  def sampling_probabilities(logits : Array(Float32), temperature : Float32,
                             top_k : Int32) : Array(Float64)
    scaled = logits.map { |value| value / temperature }
    indices = (0...scaled.size).to_a
    indices.sort_by! { |index| -scaled[index] }
    selected = Array(Bool).new(scaled.size, false)
    indices.first(Math.min(top_k, scaled.size)).each { |index| selected[index] = true }
    scaled.map_with_index! { |value, index| selected[index] ? value : -Float32::INFINITY }

    maximum = scaled.max
    masses = Array(Float64).new(scaled.size) do |index|
      value = scaled[index]
      value.finite? ? Math.exp((value - maximum).to_f64) : 0.0
    end
    total = masses.sum
    raise ArgumentError.new("top-k softmax has no finite mass") unless total.finite? && total > 0.0
    masses.map! { |mass| mass / total }
    masses
  end

  # Ranks sort by descending logit and break exact ties by ascending token ID.
  def stable_rank(logits : Array(Float32), target_id : Int32) : Int32
    target = logits[target_id]
    rank = 1
    logits.each_with_index do |value, id|
      rank += 1 if value > target || (value == target && id < target_id)
    end
    rank
  end

  def token_record(id : Int32, logits : Array(Float32), raw : Array(Float64),
                   sampled : Array(Float64), top_k : Int32) : JSON::Any
    rank = stable_rank(logits, id)
    if (rank <= top_k) != (sampled[id] > 0.0)
      raise ArgumentError.new("top-k boundary tie has sampler ordering that differs from stable rank")
    end
    JSON.parse({
      id:                   id,
      logit:                logits[id].to_f64,
      rank:                 rank,
      raw_probability:      raw[id],
      sampling_probability: sampled[id],
    }.to_json)
  end

  def readout(row : JSON::Any, model : QLoRA::Model) : JSON::Any
    id = row["id"].as_s
    messages = QChat.messages_from_openai_json(row["messages"].to_json)
    sampling = row["sampling"]?
    raise ArgumentError.new("request #{id.inspect} is missing sampling settings") unless sampling
    temperature = sampling["temperature"].as_f.to_f32
    top_k = sampling["top_k"].as_i.to_i32
    seed_value = sampling["seed"].as_i64
    raise ArgumentError.new("sampling seed must be nonnegative") if seed_value < 0

    prefix_tokens, logits, sampled_id = model.first_token_logits(
      messages, 128, temperature, top_k, seed_value.to_u64
    )
    true_id = candidate_id(model, "true")
    false_id = candidate_id(model, "false")
    raise ArgumentError.new("true and false must use distinct token IDs") if true_id == false_id
    raw = raw_probabilities(logits)
    sampled = sampling_probabilities(logits, temperature, top_k)

    JSON.parse({
      "id"            => id,
      "prefix_tokens" => prefix_tokens,
      "vocab_size"    => logits.size,
      "sampled_id"    => sampled_id,
      "true"          => token_record(true_id, logits, raw, sampled, top_k),
      "false"         => token_record(false_id, logits, raw, sampled, top_k),
    }.to_json)
  end
end

unless ARGV.size == 3 && ARGV[0] == "first_token"
  abort "usage: first_token MODEL ADAPTER_OR_DASH"
end

_, model_path, adapter = ARGV
model = QLoRA::Model.new(model_path, full_blocks: 4)
begin
  model.load(adapter) unless adapter == "-"
  STDIN.each_line do |line|
    next if line.strip.empty?
    STDOUT.puts(FirstTokenReadout.readout(JSON.parse(line), model).to_json)
    STDOUT.flush
  end
ensure
  model.close
end
