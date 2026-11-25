require "./openai"
require "./text_output"
require "../gguf/inkling2_greedy_decode"
require "../gguf/inkling2_tokenizer"

module ML::Serve
  class Inkling2Backend < OpenAI::Backend
    private DEFAULT_COMPLETION_TOKENS = 256

    @tokenizer : GGUF::Inkling2Tokenizer

    def initialize(@engine : GGUF::Inkling2Engine)
      @tokenizer = GGUF::Inkling2Tokenizer.from_gguf(@engine.weights.gguf)
    end

    def prepare(input : OpenAI::Input, options : OpenAI::Options) : OpenAI::Generation
      chat = input.is_a?(Array(OpenAI::Message))
      ids = case input
            when String
              @tokenizer.encode(input)
            when Array(Int32)
              input
            else
              chat_ids(input, options.reasoning_effort)
            end
      param = chat ? "messages" : "prompt"
      raise OpenAI::RequestError.new("The prompt must contain at least one token", param: param) if ids.empty?
      unless ids.all? { |id| id.in?(0...@tokenizer.vocab.size) }
        raise OpenAI::RequestError.new("The prompt contains a token outside the model vocabulary", param: param)
      end

      available = @engine.max_seq - ids.size
      limit = options.max_tokens || Math.min(DEFAULT_COMPLETION_TOKENS, available)
      if limit <= 0 || limit > available
        message = "The prompt (#{ids.size} tokens) and requested completion exceed the #{@engine.max_seq}-token context"
        raise OpenAI::RequestError.new(message, param: param, code: "context_length_exceeded")
      end

      Inkling2Generation.new(@engine, @tokenizer, ids, limit, options, chat)
    end

    private def chat_ids(messages : Array(OpenAI::Message), effort : String?) : Array(Int32)
      unless @tokenizer.chat_template_supported?
        raise OpenAI::RequestError.new("This checkpoint has no supported Inkling 2 chat template", param: "messages")
      end
      @tokenizer.encode_chat(messages.map { |message| {message.role, message.content} }, reasoning_effort: effort)
    rescue ex : ArgumentError
      raise OpenAI::RequestError.new(ex.message || "Invalid reasoning effort", param: "reasoning_effort")
    end
  end

  private class Inkling2Generation < OpenAI::Generation
    private CHAT_BARRIERS = {
      "<|message_model|>", "<|message_system|>", "<|message_user|>", "<|message_tool|>",
      "<|content_thinking|>", "<|content_invoke_tool_json|>", "<|end_message|>",
    }

    @text_token : Int32?
    @barriers : Array(Int32)
    @visible = true

    def initialize(@engine : GGUF::Inkling2Engine, @tokenizer : GGUF::Inkling2Tokenizer,
                   @ids : Array(Int32), @limit : Int32, @options : OpenAI::Options, @chat : Bool)
      @text_token = @tokenizer.control_token_id("<|content_text|>")
      @barriers = CHAT_BARRIERS.to_a.compact_map { |piece| @tokenizer.control_token_id(piece) }
    end

    def run(&emit : String -> Nil) : OpenAI::Result
      @engine.reset
      @visible = true
      output = TextOutput.new(@options.stop, &emit)
      logits = @engine.prefill(@ids)
      eos = @tokenizer.eos_id
      if @options.temperature.zero?
        first_id = GGUF::Inkling2Engine.greedy_token_id(logits)
        result = GGUF::Inkling2GreedyDecode.generate(@engine, @ids, first_id, @limit, eos,
          ngram: @engine.greedy_verifier?) { |id| output.append(piece(id)) }
        count = result.ids.size
        ended = result.ids.last == eos
      else
        count, ended = run_sampling(logits, output)
      end
      output.finish
      OpenAI::Result.new(@ids.size, count, (ended || output.stopped?) ? "stop" : "length")
    end

    private def piece(id : Int32) : String
      if @chat
        if id == @text_token
          @visible = true
        elsif @barriers.includes?(id)
          @visible = false
        end
        return "" unless @visible
      end
      @tokenizer.decode_single(id)
    end

    private def run_sampling(logits : Array(Float32), output : TextOutput) : {Int32, Bool}
      seed = @options.seed
      random = seed ? Random::PCG32.new(seed.to_u64!) : Random::PCG32.new
      count = 0
      ended = false
      while count < @limit
        id = sample(logits, random)
        count += 1
        ended = id == @tokenizer.eos_id
        break if ended || !output.append(piece(id)) || count == @limit

        logits = @engine.forward(id).not_nil!
      end
      {count, ended}
    end

    private def sample(logits : Array(Float32), random : Random) : Int32
      best = GGUF::Inkling2Engine.greedy_token_id(logits)
      pivot = logits[best].to_f64
      weights = logits.map do |logit|
        logit.finite? ? Math.exp((logit.to_f64 - pivot) / @options.temperature) : 0.0
      end
      mass = weights.sum
      if @options.top_p < 1.0
        order = (0...weights.size).to_a
        order.sort! do |a, b|
          by_weight = (weights[b] <=> weights[a]).not_nil!
          by_weight.zero? ? a <=> b : by_weight
        end
        cutoff = mass * @options.top_p
        mass = 0.0
        count = 0
        while count < order.size && mass < cutoff
          mass += weights[order[count]]
          count += 1
        end
        cursor = random.rand * mass
        count.times do |i|
          id = order[i]
          cursor -= weights[id]
          return id if cursor < 0
        end
      else
        cursor = random.rand * mass
        weights.each_with_index do |weight, id|
          cursor -= weight
          return id if cursor < 0
        end
      end
      best
    end
  end
end
