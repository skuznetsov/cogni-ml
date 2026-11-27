module ML::GGUF
  enum Inkling2KVCache
    F32
    F16Global
    F16All

    def self.from_env : self
      case ENV["INKLING2_KV_CACHE"]?
      when nil, "fp32"   then F32
      when "fp16-global" then F16Global
      when "fp16-all"    then F16All
      else
        raise ArgumentError.new("INKLING2_KV_CACHE must be fp32, fp16-global, or fp16-all")
      end
    end
  end
end
