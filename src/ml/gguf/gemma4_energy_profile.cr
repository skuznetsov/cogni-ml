module ML::GGUF
  # Named token budgets for comparable Gemma 4 energy runs.
  # These defaults are policy starting points, not measured energy optima.
  struct Gemma4EnergyProfile
    MAX_TOTAL_TOKENS = 8192

    getter name : String
    getter reasoning_tokens : Int32
    getter answer_tokens : Int32

    private def initialize(@name : String, @reasoning_tokens : Int32, @answer_tokens : Int32)
    end

    def self.for_name(name : String,
                      reasoning_tokens : Int32? = nil,
                      answer_tokens : Int32? = nil) : self
      canonical_name, default_reasoning, default_answer = case name.strip.downcase
                                                          when "eco"
                                                            {"Eco", 0, 256}
                                                          when "balanced"
                                                            {"Balanced", 256, 512}
                                                          when "quality"
                                                            {"Quality", 1024, 1024}
                                                          else
                                                            raise ArgumentError.new("unknown Gemma 4 energy profile")
                                                          end

      reasoning = reasoning_tokens || default_reasoning
      answer = answer_tokens || default_answer
      raise ArgumentError.new("reasoning token budget must be nonnegative") if reasoning < 0
      raise ArgumentError.new("answer token budget must be positive") if answer <= 0
      if reasoning.to_i64 + answer.to_i64 > MAX_TOTAL_TOKENS
        raise ArgumentError.new("combined token budget exceeds the safe cap")
      end

      new(canonical_name, reasoning, answer)
    end
  end
end
