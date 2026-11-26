module ML::GGUF
  # Fixed-strength guidance from pass two to pass three.
  struct Inkling2LoopCD
    enum Mode
      Logits
      Hidden
      HiddenBlocks
    end

    DEFAULT_STRENGTH = 0.25_f32

    getter mode : Mode
    getter strength : Float32

    def initialize(@mode, @strength = DEFAULT_STRENGTH)
      unless @strength.finite? && @strength >= 0_f32
        raise ArgumentError.new("Inkling 2 LoopCD strength must be finite and nonnegative")
      end
    end
  end
end
