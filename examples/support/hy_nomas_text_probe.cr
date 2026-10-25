require "json"
require "digest/sha256"
require "../../src/ml/core/buffer"

# Authenticated fixture admission and post-terminal diagnostics only. Neural
# computation belongs to the native plans, never to this fixture consumer.
module HyNomasTextProbe
  extend self
  UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
  ATOL              = 2e-5
  RTOL              = 2e-5
  MAX_SIDECAR_BYTES = 64_i64 * 1024 * 1024
  alias Comparison = NamedTuple(name: String, elements: Int32, failures: Int32,
    max_abs: Float64, raw_bit_differences: Int32, first_mismatch: Int32?,
    first_actual: Float32?, first_expected: Float32?)

  def validate_document!(document : JSON::Any, qwen_reference_sha256 : String) : Nil
    raise ArgumentError.new("text reference schema/source mismatch") unless document["schema"].as_s == "hy-nomas-text-conditioning-reference-v1" && document["schema_version"].as_i == 1 && document["upstream_revision"].as_s == UPSTREAM_REVISION
    geometry = document["geometry"]
    {"batches" => 1, "text_tokens" => 1280, "valid_tokens" => 23, "input_dim" => 3584,
     "dim" => 2048, "heads" => 16, "head_dim" => 128, "ffn_dim" => 8192,
     "video_tokens" => 6, "segments" => 1, "chunk_size_tokens" => 0,
     "text_seg_pos_emb_num" => 0}.each do |key, expected|
      raise ArgumentError.new("text reference geometry mismatch: #{key}") unless geometry[key].as_i == expected
    end
    precision = document["precision"]
    raise ArgumentError.new("text reference precision policy mismatch") unless precision["resident_weights"].as_s == "BF16" && precision["compute"].as_s == "F32" && precision["attention"].as_s == "f32"
    source = document["qwen_source"]
    unless qwen_reference_sha256.matches?(/\A[0-9a-f]{64}\z/) &&
           source["reference_sha256"].as_s == qwen_reference_sha256 &&
           source["sidecar_sha256"].as_s.matches?(/\A[0-9a-f]{64}\z/) &&
           source["state_index"].as_i == 26 && source["crop"].as_a.map(&.as_i) == [108, 1388]
      raise ArgumentError.new("text reference Qwen source mismatch")
    end
    mask = document["mask"].as_a.map(&.as_i)
    raise ArgumentError.new("text reference mask is not the qualified valid prefix") unless mask == Array(Int32).new(1280) { |i| i < 23 ? 1 : 0 }
    raise ArgumentError.new("text reference timestep mismatch") unless document["timestep"].as_f == 500.0
    raise ArgumentError.new("text reference compaction gate failed") unless document["compaction"]["passed"].as_bool
  end

  def tensor_count(metadata : JSON::Any) : Int32
    raise ArgumentError.new("reference tensor dtype must be F32") unless metadata["dtype"].as_s == "F32"
    shape = metadata["shape"].as_a.map(&.as_i64)
    raise ArgumentError.new("reference tensor rank is invalid") unless 0 < shape.size <= 4
    count = 1_i64
    shape.each do |dimension|
      raise ArgumentError.new("reference tensor shape exceeds count limit") unless dimension > 0 && count <= Int32::MAX // dimension
      count *= dimension
    end
    raise ArgumentError.new("reference tensor byte size mismatch") unless metadata["nbytes"].as_i64 == count * 4
    count.to_i32
  end

  # The manifest digest alone does not establish that two trusted manifests
  # describe the same captured activation. Cross-bind the sidecar and crop.
  def validate_qwen_binding!(document : JSON::Any, qwen : JSON::Any) : Nil
    state = qwen["capture"]["hidden_states"]
    prompt = qwen["hy_nomas_prompt"]
    unless document["qwen_source"]["sidecar_sha256"].as_s == qwen["sidecars"]["bf16"]["sha256"].as_s &&
           state["dtype"].as_s == "bfloat16" && state["shape"].as_a.map(&.as_i) == [27, 1, 1388, 3584] &&
           state["storage"]["offset_bytes"].as_i64 == 0 && state["storage"]["sidecar"].as_s == "bf16" &&
           document["qwen_source"]["state_index"].as_i == 26 &&
           document["qwen_source"]["crop"].as_a.map(&.as_i) == [prompt["crop_start"].as_i, 1388] &&
           prompt["crop_start"].as_i == 108 &&
           prompt["cropped_attention_mask"].as_a.map(&.as_i) == document["mask"].as_a.map(&.as_i)
      raise ArgumentError.new("conditioning Qwen source binding mismatch")
    end
  end

  def validate_tensor_layout!(tensors : JSON::Any, nbytes : Int64) : Nil
    raise ArgumentError.new("reference tensor sidecar size outside limit") unless 0 < nbytes <= MAX_SIDECAR_BYTES
    raise ArgumentError.new("reference tensor inventory is empty") if tensors.as_h.empty?
    cursor = 0_i64
    tensors.as_h.values.sort_by { |metadata| metadata["offset_bytes"].as_i64 }.each do |metadata|
      tensor_count(metadata)
      size = metadata["nbytes"].as_i64
      raise ArgumentError.new("reference tensor ranges must be contiguous and disjoint") unless metadata["offset_bytes"].as_i64 == cursor && size <= nbytes - cursor
      cursor += size
    end
    raise ArgumentError.new("reference sidecar has trailing or missing bytes") unless cursor == nbytes
  end

  def compare_values(actual : Array(Float32), expected : Array(Float32))
    raise ArgumentError.new("reference comparison size mismatch") unless actual.size == expected.size
    failures = 0
    different = 0
    max_abs = 0.0
    first_mismatch : Int32? = nil
    first_actual : Float32? = nil
    first_expected : Float32? = nil
    actual.each_with_index do |value, index|
      target = expected[index]
      different += 1 if value.unsafe_as(UInt32) != target.unsafe_as(UInt32)
      failed = false
      if !value.finite? || !target.finite?
        failures += 1
        max_abs = Float64::MAX
        failed = true
      else
        delta = (value.to_f64 - target.to_f64).abs
        max_abs = Math.max(max_abs, delta)
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
    {failures: failures, max_abs: max_abs, raw_bit_differences: different,
     first_mismatch: first_mismatch, first_actual: first_actual, first_expected: first_expected}
  end

  def sha256(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      bytes = Bytes.new(1024 * 1024)
      while (count = io.read(bytes)) > 0
        digest.update(bytes[0, count])
      end
    end
    digest.hexfinal
  end

  class Fixture
    getter document : JSON::Any
    getter path : String
    getter sha256 : String
    getter text_weights : String
    getter block_weights : String
    @snapshot : Bytes

    def initialize(path : String, expected_sha256 : String, qwen_reference_sha256 : String)
      @path = File.realpath(path)
      raise ArgumentError.new("text manifest must be a bounded regular file") unless File.file?(@path) && File.size(@path) <= 2 * 1024 * 1024
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      raise ArgumentError.new("trusted text manifest digest mismatch") unless expected_sha256.matches?(/\A[0-9a-f]{64}\z/) && @sha256 == expected_sha256
      @document = JSON.parse(raw)
      HyNomasTextProbe.validate_document!(@document, qwen_reference_sha256)
      @text_weights = authenticated_weights("text")
      @block_weights = authenticated_weights("block")
      tensor_file = @document["tensor_file"]
      name = tensor_file["name"].as_s
      raise ArgumentError.new("tensor sidecar must be a sibling basename") unless name.matches?(/\A[A-Za-z0-9_.-]+\z/) && name != "." && name != ".."
      sidecar = File.join(File.dirname(@path), name)
      raise ArgumentError.new("tensor sidecar must be a regular sibling file") unless File.file?(sidecar) && !File.symlink?(sidecar)
      size = tensor_file["nbytes"].as_i64
      HyNomasTextProbe.validate_tensor_layout!(@document["tensors"], size)
      raise ArgumentError.new("tensor sidecar byte count mismatch") unless File.size(sidecar) == size
      # Keep exactly the bytes authenticated here, never reopen mutable expected
      # outputs during the post-wait comparison.
      @snapshot = File.read(sidecar).to_slice.dup
      raise ArgumentError.new("tensor sidecar digest mismatch") unless Digest::SHA256.hexdigest(@snapshot) == tensor_file["sha256"].as_s
    end

    def mask : Array(Int32)
      @document["mask"].as_a.map(&.as_i)
    end

    def timestep : Float32
      @document["timestep"].as_f.to_f32
    end

    # Standalone same-input control only: ingest authenticated oracle bytes.
    # The connected runner instead binds the live native Qwen crop directly.
    def source_conditioning(qwen_path : String) : Bytes
      raise ArgumentError.new("Qwen manifest must be a bounded regular file") unless File.file?(qwen_path) && !File.symlink?(qwen_path) && File.size(qwen_path) <= 2 * 1024 * 1024
      raw = File.read(qwen_path)
      raise ArgumentError.new("conditioning Qwen manifest digest mismatch") unless Digest::SHA256.hexdigest(raw) == @document["qwen_source"]["reference_sha256"].as_s
      source = JSON.parse(raw)
      HyNomasTextProbe.validate_qwen_binding!(@document, source)
      descriptor = source["sidecars"]["bf16"]
      name = descriptor["path"].as_s
      raise ArgumentError.new("Qwen sidecar must be a sibling basename") unless name.matches?(/\A[A-Za-z0-9_.-]+\z/) && name != "." && name != ".."
      sidecar = File.join(File.dirname(File.realpath(qwen_path)), name)
      raise ArgumentError.new("Qwen sidecar must be a regular file") unless File.file?(sidecar) && !File.symlink?(sidecar)
      expected_size = 27_i64 * 1388 * 3584 * 2
      raise ArgumentError.new("Qwen sidecar size/encoding mismatch") unless descriptor["nbytes"].as_i64 == expected_size && File.size(sidecar) == expected_size && descriptor["encoding"].as_s == "bfloat16-le"
      selected = Bytes.new(1280 * 3584 * 2)
      selected_start = (26_i64 * 1388 + 108) * 3584 * 2
      selected_end = selected_start + selected.size
      digest = Digest::SHA256.new
      total = 0_i64
      File.open(sidecar) do |io|
        bytes = Bytes.new(1024 * 1024)
        while (count = io.read(bytes)) > 0
          digest.update(bytes[0, count])
          first = Math.max(selected_start, total)
          last = Math.min(selected_end, total + count)
          if last > first
            selected[(first - selected_start).to_i32, (last - first).to_i32].copy_from(bytes[(first - total).to_i32, (last - first).to_i32])
          end
          total += count
        end
      end
      observed = digest.hexfinal
      raise ArgumentError.new("Qwen sidecar digest mismatch") unless total == expected_size && observed == descriptor["sha256"].as_s && observed == @document["qwen_source"]["sidecar_sha256"].as_s
      selected
    end

    def shape(name : String) : Array(Int32)
      @document["tensors"][name]["shape"].as_a.map(&.as_i)
    end

    def values(name : String) : Array(Float32)
      metadata = @document["tensors"][name]
      count = HyNomasTextProbe.tensor_count(metadata)
      offset = metadata["offset_bytes"].as_i64.to_i32
      Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Float32, @snapshot[offset + index * 4, 4])
      end
    end

    # Call only after the caller's terminal drain; this is diagnostic readback.
    def compare(buffer : ML::MetalBuffer, name : String) : Comparison
      expected = values(name)
      raise ArgumentError.new("native checkpoint extent differs from reference: #{name}") unless buffer.size == expected.size.to_i64 * 4
      result = HyNomasTextProbe.compare_values(buffer.read(expected.size), expected)
      {name: name, elements: expected.size, failures: result[:failures],
       max_abs: result[:max_abs], raw_bit_differences: result[:raw_bit_differences],
       first_mismatch: result[:first_mismatch], first_actual: result[:first_actual],
       first_expected: result[:first_expected]}
    end

    private def authenticated_weights(name : String) : String
      metadata = @document["weights"][name]
      path = metadata["path"].as_s
      raise ArgumentError.new("weights must be an absolute regular file") unless path.starts_with?("/") && File.file?(path) && !File.symlink?(path)
      path = File.realpath(path)
      digest = metadata["sha256"].as_s
      raise ArgumentError.new("weights digest malformed") unless digest.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("weights byte count mismatch") unless File.size(path) == metadata["nbytes"].as_i64
      raise ArgumentError.new("weights digest mismatch") unless HyNomasTextProbe.sha256(path) == digest
      path
    end
  end
end
