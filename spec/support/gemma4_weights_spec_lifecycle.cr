module Gemma4WeightsSpecLifecycle
  @@open_weights = [] of ML::GGUF::Gemma4Weights

  def self.load(path : String) : ML::GGUF::Gemma4Weights
    weights = ML::GGUF::Gemma4Weights.from_gguf(path)
    @@open_weights << weights
    weights
  end

  def self.close_all : Nil
    @@open_weights.each(&.close)
  ensure
    @@open_weights.clear
  end
end
