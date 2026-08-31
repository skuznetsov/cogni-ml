module ML::GGUF
  module QwenQBitAdaptiveMetalPolicy
    def self.gqa6_tile(device_name : String, override : String? = nil) : Int32
      case override.try(&.strip.downcase)
      when nil, "", "auto"
        device_name == "Apple M2 Max" ? 15 : 16
      when "15"
        15
      when "16"
        16
      else
        raise ArgumentError.new("QWEN35_ADAPTIVE_GQA6_TILE must be auto, 15, or 16")
      end
    end

    def self.decode_splitk?(packed_len : Int32,
                            token_count : Int32,
                            uniform_tier : Bool,
                            enabled_override : String? = nil,
                            min_context_override : String? = nil) : Bool
      enabled = case enabled_override.try(&.strip)
                when nil, "", "1" then true
                when "0"          then false
                else
                  raise ArgumentError.new("QWEN35_ADAPTIVE_SPLITK must be 0 or 1")
                end
      min_context = if raw = min_context_override
                      raw.strip.to_i?
                    else
                      256
                    end
      unless min_context && min_context > 0
        raise ArgumentError.new("QWEN35_ADAPTIVE_SPLITK_MIN_CTX must be a positive integer")
      end

      enabled && uniform_tier && token_count == 1 &&
        packed_len.to_i64 + 1_i64 >= min_context.to_i64
    end

    def self.dequant_t4?(override : String? = nil) : Bool
      case override.try(&.strip)
      when nil, "", "0" then false
      when "1"          then true
      else
        raise ArgumentError.new("QWEN35_ADAPTIVE_DEQUANT_T4 must be 0 or 1")
      end
    end
  end
end
