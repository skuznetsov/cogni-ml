# Diagnostic admission only; no arbitrary dimensions or production route change.
module QwenSG4ProbeShape
  CASES = [{7839, 193}, {7839, 194}, {7839, 195}, {7839, 196}, {0, 64}, {0, 195}]

  def self.parse(args : Array(String)) : Tuple(Int32, Int32)?
    return nil unless args.any?(&.starts_with?("--direct-shape"))
    raise ArgumentError.new("direct-shape takes exactly one selector") unless args.size == 1
    CASES.each do |base, rows|
      return {base, rows} if args[0] == "--direct-shape=#{base}:#{rows}"
    end
    raise ArgumentError.new("direct-shape is restricted to the six declared F32 cases")
  end
end
