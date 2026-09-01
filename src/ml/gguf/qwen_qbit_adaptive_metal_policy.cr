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

    def self.dequant_t4?(device_name : String,
                         automatic : Bool,
                         override : String? = nil) : Bool
      return automatic && device_name == "Apple M2 Max" unless override

      case override.strip
      when "0" then false
      when "1" then true
      else
        raise ArgumentError.new("QWEN35_ADAPTIVE_DEQUANT_T4 must be 0 or 1")
      end
    end

    def self.splitk_stage2_fused?(device_name : String,
                                  uniform_bf16 : Bool,
                                  override : String? = nil) : Bool
      return device_name == "Apple M2 Max" && uniform_bf16 unless override

      case override.strip
      when "0" then false
      when "1" then true
      else
        raise ArgumentError.new("QWEN35_ADAPTIVE_SPLITK_STAGE2_FUSED must be 0 or 1")
      end
    end

    def self.pack_prefix_quant?(device_name : String,
                                uniform_bf16 : Bool,
                                override : String? = nil) : Bool
      return device_name == "Apple M2 Max" && uniform_bf16 unless override

      case override.strip
      when "0" then false
      when "1" then true
      else
        raise ArgumentError.new("QWEN35_ADAPTIVE_PACK_PREFIX_QUANT must be 0 or 1")
      end
    end

    def self.automatic_dequant_t4?(token_count : Int32,
                                   splitk : Bool,
                                   uniform_p4 : Bool,
                                   uniform_bf16 : Bool) : Bool
      splitk ? uniform_bf16 : token_count > 1 && (uniform_p4 || uniform_bf16)
    end

    # The measured M2 Max corridor benefits from loading eight adjacent P4
    # values per lane. Other devices stay on the portable loader unless an
    # explicit benchmark override is supplied.
    def self.p4_splitk_t8?(device_name : String,
                           override : String? = nil) : Bool
      return device_name == "Apple M2 Max" unless override

      case override.strip
      when "0" then false
      when "1" then true
      else
        raise ArgumentError.new("QWEN35_ADAPTIVE_P4_SPLITK_T8 must be 0 or 1")
      end
    end
  end
end
