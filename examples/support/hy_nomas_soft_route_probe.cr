require "./hy_nomas_dit_route_factorial_probe"

# CPU-only admission for this fixed authored-prior experiment. Runtime metadata
# is immutable after admission; no image-derived decisions occur during encode.
module HyNomasSoftRouteProbe
  extend self
  CASES = ["zero", "A1_low", "A1_high", "A2_low", "A2_high",
           "A1_sham", "A2_sham", "zero_repeat"]
  QUERY              = HyNomasDiTRouteFactorialProbe::QUERY_ROWS
  VIDEO              = HyNomasDiTRouteFactorialProbe::VIDEO_ROWS
  MAX_MANIFEST_BYTES = 2_i64 * 1024 * 1024

  class Biases
    getter path : String
    getter sha256 : String
    getter values : Hash(String, Array(Float32))
    getter document : JSON::Any

    def initialize(path : String, expected_sha256 : String)
      @path = File.realpath(path)
      unless expected_sha256.matches?(/\A[0-9a-f]{64}\z/) &&
             File.info(@path).size <= MAX_MANIFEST_BYTES
        raise ArgumentError.new("soft-route bias manifest size or trusted digest is invalid")
      end
      @sha256 = HyNomasDiTRouteFactorialProbe.sha256_file(@path)
      raise ArgumentError.new("soft-route bias manifest digest mismatch") unless @sha256 == expected_sha256
      @document = JSON.parse(File.read(@path))
      unless @document["schema"].as_s == "hy-nomas-soft-route-bias-v1" &&
             @document["case_order"].as_a.map(&.as_s) == CASES &&
             @document["geometry"]["video_rows"].as_i == VIDEO &&
             @document["geometry"]["query_rows"].as_i == QUERY
        raise ArgumentError.new("soft-route case order or geometry mismatch")
      end
      image = @document["reference_image"]
      unless HyNomasDiTRouteFactorialProbe.sha256_file(image["path"].as_s) == image["sha256"].as_s
        raise ArgumentError.new("soft-route authored reference image changed")
      end
      raw = @document["biases"].as_h
      raise ArgumentError.new("soft-route bias inventory mismatch") unless raw.keys.sort == CASES.sort
      @values = {} of String => Array(Float32)
      query_labels = HyNomasDiTRouteFactorialProbe.native_query_labels_for("M11")
      CASES.each do |name|
        array = raw[name].as_a
        raise ArgumentError.new("soft-route bias row count mismatch") unless array.size == QUERY
        values = array.map { |value| value.as_f.to_f32 }
        values.each_with_index do |value, index|
          unless value.finite? && 0.0_f32 <= value <= 2.0_f32 &&
                 (value.unsafe_as(UInt32) & 0xffff_u32).zero? &&
                 (!value.zero? || value.unsafe_as(UInt32).zero?)
            raise ArgumentError.new("soft-route bias must be finite nonnegative lossless BF16")
          end
          next if value.zero?
          action = name.starts_with?("A1_") ? 1 : (name.starts_with?("A2_") ? 2 : 0)
          unless index < VIDEO && action > 0 && query_labels[index] == action
            raise ArgumentError.new("soft-route bias escaped its action, ROI or video queries")
          end
        end
        bytes = Slice(UInt8).new(values.to_unsafe.as(UInt8*), QUERY * sizeof(Float32))
        unless Digest::SHA256.hexdigest(bytes) == @document["bias_f32le_sha256"][name].as_s
          raise ArgumentError.new("soft-route bias F32 payload digest mismatch")
        end
        @values[name] = values
      end
      unless @values["zero"].all?(&.zero?) && @values["zero_repeat"] == @values["zero"]
        raise ArgumentError.new("soft-route zero or repeat is not exact zero")
      end
      ["A1", "A2"].each do |action|
        low, high, sham = @values["#{action}_low"], @values["#{action}_high"], @values["#{action}_sham"]
        unless high.any? { |value| value > 0 } && low.each_with_index.all? { |value, index| value <= high[index] }
          raise ArgumentError.new("soft-route doses are empty or unordered")
        end
        30.times do |frame|
          slice = frame * 144
          unless high[slice, 144].sort == sham[slice, 144].sort
            raise ArgumentError.new("soft-route sham changed per-frame weight histogram")
          end
        end
      end
      unless @sha256 == HyNomasDiTRouteFactorialProbe.sha256_file(@path)
        raise ArgumentError.new("soft-route manifest changed during admission")
      end
    end
  end
end
