module ML::GGUF
  module NomicMetalPolicy
    def self.simdgroup_matrix_enabled?(device_name : String, override : String? = nil) : Bool
      case override.try(&.strip.downcase)
      when nil, "", "auto"
        device_name.starts_with?("Apple M")
      when "1", "on", "true"
        true
      when "0", "off", "false"
        false
      else
        raise ArgumentError.new("NOMIC_SIMDGROUP_MATRIX must be auto, on, or off")
      end
    end

    def self.matrix_attention_enabled?(device_name : String, override : String? = nil) : Bool
      case override.try(&.strip.downcase)
      when nil, "", "auto"
        !device_name.starts_with?("Apple M5 Max")
      when "1", "on", "true"
        true
      when "0", "off", "false"
        false
      else
        raise ArgumentError.new("NOMIC_MATRIX_ATTENTION must be auto, on, or off")
      end
    end
  end
end
