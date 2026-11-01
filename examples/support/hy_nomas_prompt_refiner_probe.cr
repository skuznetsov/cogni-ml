require "json"
require "digest/sha256"
require "../../src/ml/core/buffer"
require "../../src/ml/video/hy_nomas_text_refiner"
require "./hy_nomas_generation"

# Authenticated expected valid-prefix refiner outputs and native-side admission.
# This support module performs no neural operations.
module HyNomasPromptRefinerProbe
  extend self

  UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
  ATOL              = 2e-5
  RTOL              = 2e-5
  MAX_SIDECAR_BYTES = 64_i64 * 1024 * 1024
  TEXT_WEIGHTS_PATH = "/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025/weights/text-time-f32.safetensors"
  TEXT_WEIGHTS_SHA256 = "58313f6141adafc12629b46c73f1bc270aa64c3e0972a8995875ba76d9068d8f"
  TEXT_MANIFEST_SHA256 = "0863d49ba53d54db5ad7ff5313099c3e93003cf304e729f4a377098f230f462f"
  OUTPUT_ROOT = "/opt/mldata_ssd"
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int32,
    max_abs: Float64, raw_bit_differences: Int32, first_mismatch: Int32?,
    first_actual: Float32?, first_expected: Float32?)

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      buffer = Bytes.new(1024 * 1024)
      while (count = io.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.hexfinal
  end

  def tensor_count(metadata : JSON::Any) : Int32
    raise ArgumentError.new("refiner fixture tensor dtype must be F32") unless metadata["dtype"].as_s == "F32"
    shape = metadata["shape"].as_a.map(&.as_i64)
    raise ArgumentError.new("refiner fixture tensor rank is invalid") unless 1 <= shape.size <= 4
    count = 1_i64
    shape.each do |dimension|
      raise ArgumentError.new("refiner fixture tensor shape exceeds its bound") unless dimension > 0 && count <= Int32::MAX // dimension
      count *= dimension
    end
    raise ArgumentError.new("refiner fixture tensor byte count mismatch") unless metadata["nbytes"].as_i64 == count * 4
    count.to_i32
  end

  def compare_values(actual : Array(Float32), expected : Array(Float32))
    raise ArgumentError.new("refiner fixture comparison size mismatch") unless actual.size == expected.size
    failures = 0
    bit_differences = 0
    maximum = 0.0
    first_mismatch : Int32? = nil
    first_actual : Float32? = nil
    first_expected : Float32? = nil
    actual.each_with_index do |value, index|
      target = expected[index]
      bit_differences += 1 if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
      failed = false
      if !value.finite? || !target.finite?
        failures += 1
        maximum = Float64::MAX
        failed = true
      else
        delta = (value.to_f64 - target.to_f64).abs
        maximum = Math.max(maximum, delta)
        if delta > ATOL + RTOL * target.abs.to_f64
          failures += 1
          failed = true
        end
      end
      if failed && first_mismatch.nil?
        first_mismatch = index
        first_actual = value if value.finite?
        first_expected = target if target.finite?
      end
    end
    {failures: failures, max_abs: maximum, raw_bit_differences: bit_differences,
     first_mismatch: first_mismatch, first_actual: first_actual, first_expected: first_expected}
  end

  class Fixture
    getter path : String
    getter sha256 : String
    getter document : JSON::Any
    getter text_weights : String
    getter valid_tokens : Int32
    getter timestep_label : String
    getter timestep : Float32
    @snapshot : Bytes

    def initialize(path : String, expected_sha256 : String, timestep_label : String)
      @path = File.realpath(path)
      raise ArgumentError.new("refiner manifest must be a bounded regular file") unless File.file?(@path) && !File.symlink?(@path) && File.size(@path) <= 2 * 1024 * 1024
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("trusted refiner manifest digest mismatch") unless expected_sha256.matches?(/\A[0-9a-f]{64}\z/) && @sha256 == expected_sha256
      @document = JSON.parse(raw)
      raise ArgumentError.new("refiner schema/source mismatch") unless @document["schema"].as_s == "cogni_ml.hy_nomas_prompt_refiner_reference/v1" && @document["schema_version"].as_i == 1 && @document["upstream_revision"].as_s == UPSTREAM_REVISION

      geometry = @document["geometry"]
      @valid_tokens = geometry["valid_tokens"].as_i
      raise ArgumentError.new("refiner fixture admits only the actual 557- or 585-token prompts") unless {557, 585}.includes?(@valid_tokens)
      {"batches" => 1, "text_tokens" => 1280, "input_dim" => 3584,
       "dim" => 2048, "heads" => 16, "head_dim" => 128, "ffn_dim" => 8192}.each do |key, value|
        raise ArgumentError.new("refiner fixture geometry mismatch: #{key}") unless geometry[key].as_i == value
      end

      precision = @document["precision"]
      unless precision["resident_weights"].as_s == "BF16" && precision["compute"].as_s == "F32" &&
             precision["attention"].as_s == "f32" && precision["text_output_policy"].as_s == "valid_prefix_only" &&
             precision["padded_slots"].as_s == "native retains 1280 storage slots; only valid prefix rows are compared; no padding equality claim"
        raise ArgumentError.new("refiner fixture precision or padding policy mismatch")
      end

      cases = @document["timestep_cases"].as_a
      raise ArgumentError.new("refiner fixture timestep inventory mismatch") unless cases.size == 2
      expected_cases = {"1000" => 1000.0, "206.897" => 206.897}
      raise ArgumentError.new("refiner fixture timestep labels mismatch") unless cases.map { |item| item["label"].as_s }.sort == expected_cases.keys.sort
      cases.each do |item|
        label = item["label"].as_s
        expected_value = expected_cases[label]? || raise ArgumentError.new("unsupported refiner timestep label")
        raise ArgumentError.new("refiner timestep value mismatch") unless item["value"].as_f == expected_value
      end
      @timestep = expected_cases[timestep_label]?.try(&.to_f32) || raise ArgumentError.new("timestep must be 1000 or 206.897")
      @timestep_label = timestep_label

      source = @document["source"]
      {"inputs_manifest_sha256", "inputs_payload_sha256", "prompt_sha256", "qwen_reference_sha256", "qwen_sidecar_sha256"}.each do |key|
        raise ArgumentError.new("refiner fixture source digest malformed: #{key}") unless source[key].as_s.matches?(/\A[0-9a-f]{64}\z/)
      end
      raise ArgumentError.new("refiner fixture input token count mismatch") unless source["valid_tokens"].as_i == @valid_tokens
      raise ArgumentError.new("refiner fixture Qwen state/crop mismatch") unless source["state_index"].as_i == 26 && source["crop"].as_a.size == 2 && source["crop"].as_a[1].as_i - source["crop"].as_a[0].as_i == 1280

      upstream = @document["upstream_source_sha256"]
      unless upstream["hymm/models/text_encoder/token_refiner.py"].as_s == "7c67d81b81d9b183c3261eb4bf87bb04e0bda3142ce096afc2e64aa722433eab" &&
             upstream["hymm/models/dit/modules/embed_layers.py"].as_s == "5b055e383bbf532ea5aa49ee506031d96bea3e3d5b8da3a104d4bd875c2d0af3"
        raise ArgumentError.new("refiner fixture upstream source hashes mismatch")
      end
      weights = @document["weights"]
      @text_weights = File.realpath(TEXT_WEIGHTS_PATH)
      raise ArgumentError.new("refiner text weights source mismatch") unless weights["path"].as_s == @text_weights && weights["sha256"].as_s == TEXT_WEIGHTS_SHA256 && weights["manifest_sha256"].as_s == TEXT_MANIFEST_SHA256
      raise ArgumentError.new("pinned text weights must be a regular file") unless File.file?(@text_weights) && !File.symlink?(@text_weights)
      raise ArgumentError.new("pinned text weights digest mismatch") unless HyNomasPromptRefinerProbe.sha256(@text_weights) == TEXT_WEIGHTS_SHA256
      text_manifest = "#{@text_weights}.manifest.json"
      raise ArgumentError.new("pinned text manifest must be a regular file") unless File.file?(text_manifest) && !File.symlink?(text_manifest)
      raise ArgumentError.new("pinned text manifest digest mismatch") unless HyNomasPromptRefinerProbe.sha256(text_manifest) == TEXT_MANIFEST_SHA256

      compaction = @document["compaction"]
      raise ArgumentError.new("upstream compaction gate did not pass") unless compaction["passed"].as_bool && compaction["atol"].as_f == 2e-5 && compaction["rtol"].as_f == 2e-5
      reports = compaction["comparisons"].as_a
      raise ArgumentError.new("upstream compaction comparison inventory mismatch") unless reports.size == 2 && reports.all? { |row| row["passed"].as_bool && row["failures"].as_i == 0 }

      expected_tensors = {} of String => Array(Int32)
      expected_cases.each_key do |label|
        expected_tensors["projected_text.#{label}"] = [1, @valid_tokens, 2048]
        expected_tensors["vec.#{label}"] = [1, 2048]
      end
      tensors = @document["tensors"]
      raise ArgumentError.new("refiner tensor inventory mismatch") unless tensors.as_h.keys.sort == expected_tensors.keys.sort
      cursor = 0_i64
      tensors.as_h.values.sort_by { |metadata| metadata["offset_bytes"].as_i64 }.each do |metadata|
        HyNomasPromptRefinerProbe.tensor_count(metadata)
        raise ArgumentError.new("refiner sidecar ranges must be contiguous") unless metadata["offset_bytes"].as_i64 == cursor
        cursor += metadata["nbytes"].as_i64
      end
      expected_tensors.each do |name, shape|
        raise ArgumentError.new("refiner tensor shape mismatch: #{name}") unless tensors[name]["shape"].as_a.map(&.as_i) == shape
      end

      tensor_file = @document["tensor_file"]
      name = tensor_file["name"].as_s
      raise ArgumentError.new("refiner sidecar must be a sibling basename") unless name.matches?(/\A[A-Za-z0-9_.-]+\z/) && name != "." && name != ".."
      sidecar = File.join(File.dirname(@path), name)
      size = tensor_file["nbytes"].as_i64
      raise ArgumentError.new("refiner sidecar extent exceeds its bound") unless 0 < size <= MAX_SIDECAR_BYTES && cursor == size
      raise ArgumentError.new("refiner sidecar must be a regular sibling") unless File.file?(sidecar) && !File.symlink?(sidecar) && File.realpath(sidecar) == sidecar && File.size(sidecar) == size
      @snapshot = File.read(sidecar).to_slice.dup
      raise ArgumentError.new("refiner sidecar digest mismatch") unless Digest::SHA256.hexdigest(@snapshot) == tensor_file["sha256"].as_s
    end

    def values(name : String) : Array(Float32)
      metadata = @document["tensors"][name]
      count = HyNomasPromptRefinerProbe.tensor_count(metadata)
      offset = metadata["offset_bytes"].as_i64.to_i32
      Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Float32, @snapshot[offset + index * 4, 4])
      end
    end

    def validate_inputs!(inputs : HyNomasGeneration::Inputs) : Nil
      source = @document["source"]
      document = inputs.file.document
      qwen_source = document["qwen_source"]
      unless inputs.file.sha256 == source["inputs_manifest_sha256"].as_s &&
             document["payload_sha256"].as_s == source["inputs_payload_sha256"].as_s &&
             inputs.valid_text_tokens == @valid_tokens &&
             inputs.mask == Array(Int32).new(1280) { |i| i < @valid_tokens ? 1 : 0 } &&
             Digest::SHA256.hexdigest(inputs.prompt) == source["prompt_sha256"].as_s &&
             qwen_source["reference_sha256"].as_s == source["qwen_reference_sha256"].as_s &&
             qwen_source["sidecar_sha256"].as_s == source["qwen_sidecar_sha256"].as_s &&
             qwen_source["state_index"].as_i == 26 &&
             qwen_source["crop"].as_a.map(&.as_i) == source["crop"].as_a.map(&.as_i)
        raise ArgumentError.new("refiner fixture does not bind to the authenticated generation inputs")
      end
    end
  end
end
