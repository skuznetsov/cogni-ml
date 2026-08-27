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
  end
end
