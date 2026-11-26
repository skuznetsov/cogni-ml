require "digest/sha256"
require "json"

module HyNomasQueryCorrespondenceProbe
  extend self

  INPUT_SCHEMA  = "hy-nomas-query-correspondence-input-v1"
  OUTPUT_SCHEMA = "hy-nomas-query-correspondence-native-v1"
  HEADS         =  16
  POSITIONS     = 144
  HEAD_DIM      = 128
  TENSOR_VALUES = HEADS * POSITIONS * HEAD_DIM
  TENSOR_BYTES  = TENSOR_VALUES.to_i64 * sizeof(Float32)
  INPUT_BYTES   = 2_i64 * TENSOR_BYTES
  SCORE_VALUES  = HEADS * POSITIONS * POSITIONS
  METRIC_COUNT  = HEADS * POSITIONS
  METRIC_WIDTH  = 6
  SCORE_BYTES   = SCORE_VALUES.to_i64 * sizeof(Float32)
  METRIC_BYTES  = METRIC_COUNT.to_i64 * METRIC_WIDTH * sizeof(Float32)
  NAMED_BYTES   = INPUT_BYTES + SCORE_BYTES + METRIC_BYTES
  METRICS_ORDER = ["diagonal", "max", "margin", "greater_count", "argmax", "self_mass"]

  METAL_SOURCE = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    constant uint HY_HEADS = 16u;
    constant uint HY_POSITIONS = 144u;
    constant uint HY_HEAD_DIM = 128u;
    constant uint HY_SCORE_COUNT = HY_HEADS * HY_POSITIONS * HY_POSITIONS;
    constant float HY_SCALE = 0.08838834764831845f;

    kernel void hy_nomas_query_correspondence_scores_f32(
        device const float *query [[buffer(0)]],
        device const float *key [[buffer(1)]],
        device float *scores [[buffer(2)]],
        uint index [[thread_position_in_grid]]) {
      if (index >= HY_SCORE_COUNT) return;

      const uint key_position = index % HY_POSITIONS;
      const uint query_position = (index / HY_POSITIONS) % HY_POSITIONS;
      const uint head = index / (HY_POSITIONS * HY_POSITIONS);
      float sum = 0.0f;
      for (uint dim = 0; dim < HY_HEAD_DIM; ++dim) {
        const uint query_index = (query_position * HY_HEADS + head) * HY_HEAD_DIM + dim;
        const uint key_index = (key_position * HY_HEADS + head) * HY_HEAD_DIM + dim;
        sum += query[query_index] * key[key_index];
      }
      scores[index] = sum * HY_SCALE;
    }

    kernel void hy_nomas_query_correspondence_metrics_f32(
        device const float *scores [[buffer(0)]],
        device float *metrics [[buffer(1)]],
        uint row_index [[thread_position_in_grid]]) {
      if (row_index >= HY_HEADS * HY_POSITIONS) return;

      const uint row = row_index % HY_POSITIONS;
      const uint head = row_index / HY_POSITIONS;
      const uint score_base = (head * HY_POSITIONS + row) * HY_POSITIONS;
      const float diagonal = scores[score_base + row];
      float maximum = -INFINITY;
      float maximum_other = -INFINITY;
      uint argmax = 0;
      uint greater_count = 0;

      // Scan in increasing index order and update only on strict greater-than,
      // which gives the lower key index for tied maxima.
      for (uint key_position = 0; key_position < HY_POSITIONS; ++key_position) {
        const float score = scores[score_base + key_position];
        if (score > maximum) {
          maximum = score;
          argmax = key_position;
        }
        if (key_position != row && score > maximum_other) {
          maximum_other = score;
        }
        if (score > diagonal) {
          ++greater_count;
        }
      }

      float exponential_sum = 0.0f;
      for (uint key_position = 0; key_position < HY_POSITIONS; ++key_position) {
        exponential_sum += exp(scores[score_base + key_position] - maximum);
      }

      const uint metric_base = row_index * 6;
      metrics[metric_base + 0] = diagonal;
      metrics[metric_base + 1] = maximum;
      metrics[metric_base + 2] = diagonal - maximum_other;
      metrics[metric_base + 3] = float(greater_count);
      metrics[metric_base + 4] = float(argmax);
      metrics[metric_base + 5] = exp(diagonal - maximum) / exponential_sum;
    }
  METAL

  class Fixture
    getter manifest_path : String
    getter manifest_sha256 : String
    getter payload_path : String
    getter payload_sha256 : String
    getter query_sha256 : String
    getter key_sha256 : String
    getter query_bytes : String
    getter key_bytes : String
    getter provenance : JSON::Any?

    def initialize(path : String, expected_sha256 : String)
      unless HyNomasQueryCorrespondenceProbe.valid_sha256?(expected_sha256)
        raise ArgumentError.new("manifest SHA-256 must be 64 hexadecimal characters")
      end
      @manifest_path = HyNomasQueryCorrespondenceProbe.canonical_file(path, "input manifest")
      unless File.size(@manifest_path) <= 65_536
        raise ArgumentError.new("input manifest exceeds 64 KiB")
      end
      raw_manifest = File.read(@manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(raw_manifest)
      unless @manifest_sha256 == expected_sha256.downcase
        raise ArgumentError.new("input manifest digest mismatch")
      end

      document = JSON.parse(raw_manifest)
      root_keys = document.as_h.keys.sort
      allowed_root_keys = %w(geometry payload schema schema_version tensors)
      if document.as_h.has_key?("provenance")
        allowed_root_keys << "provenance"
        @provenance = document["provenance"]
        raise ArgumentError.new("input provenance must be a JSON object") unless @provenance.not_nil!.as_h?
      end
      HyNomasQueryCorrespondenceProbe.exact_keys!(root_keys, allowed_root_keys.sort, "manifest")
      unless document["schema"].as_s == INPUT_SCHEMA && document["schema_version"].as_i == 1
        raise ArgumentError.new("input manifest schema or version mismatch")
      end
      geometry = document["geometry"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(geometry.as_h.keys.sort, %w(head_dim heads positions), "geometry")
      unless geometry["heads"].as_i == HEADS && geometry["positions"].as_i == POSITIONS &&
             geometry["head_dim"].as_i == HEAD_DIM
        raise ArgumentError.new("query correspondence input geometry mismatch")
      end

      payload = document["payload"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(payload.as_h.keys.sort, %w(bytes path sha256), "payload")
      payload_name = payload["path"].as_s
      unless !payload_name.empty? && File.basename(payload_name) == payload_name &&
             payload_name != "." && payload_name != ".."
        raise ArgumentError.new("payload path must be a simple sibling basename")
      end
      payload_bytes = payload["bytes"].as_i64
      @payload_sha256 = payload["sha256"].as_s
      unless payload_bytes == INPUT_BYTES && HyNomasQueryCorrespondenceProbe.valid_sha256?(@payload_sha256)
        raise ArgumentError.new("input payload size or SHA-256 descriptor mismatch")
      end
      @payload_path = HyNomasQueryCorrespondenceProbe.canonical_file(
        File.join(File.dirname(@manifest_path), payload_name), "input payload")
      if @payload_path == @manifest_path
        raise ArgumentError.new("manifest and payload paths must be distinct")
      end
      unless File.size(@payload_path).to_i64 == payload_bytes &&
             HyNomasQueryCorrespondenceProbe.sha256_file(@payload_path) == @payload_sha256
        raise ArgumentError.new("input payload byte count or digest mismatch")
      end

      tensors = document["tensors"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(tensors.as_h.keys.sort, %w(key query), "tensor inventory")
      query = tensor_record!(tensors["query"], "query", 0_i64)
      key = tensor_record!(tensors["key"], "key", TENSOR_BYTES)
      unless query[0] == 0_i64 && query[1] == TENSOR_BYTES &&
             key[0] == TENSOR_BYTES && key[1] == TENSOR_BYTES &&
             query[0] + query[1] == key[0] && key[0] + key[1] == payload_bytes
        raise ArgumentError.new("query/key tensor ranges overlap or do not cover the payload exactly")
      end
      raw_payload = File.read(@payload_path)
      @query_bytes = raw_payload.byte_slice(query[0].to_i, query[1].to_i)
      @key_bytes = raw_payload.byte_slice(key[0].to_i, key[1].to_i)
      HyNomasQueryCorrespondenceProbe.validate_expanded_bf16!(@query_bytes, "query")
      HyNomasQueryCorrespondenceProbe.validate_expanded_bf16!(@key_bytes, "key")
      @query_sha256 = Digest::SHA256.hexdigest(@query_bytes)
      @key_sha256 = Digest::SHA256.hexdigest(@key_bytes)
    end

    private def tensor_record!(record : JSON::Any, name : String, expected_offset : Int64) : Tuple(Int64, Int64)
      HyNomasQueryCorrespondenceProbe.exact_keys!(record.as_h.keys.sort,
        %w(byte_order dtype nbytes offset_bytes sha256 shape source_dtype), "#{name} tensor")
      shape = record["shape"].as_a.map(&.as_i64)
      offset = record["offset_bytes"].as_i64
      nbytes = record["nbytes"].as_i64
      digest = record["sha256"].as_s
      unless record["dtype"].as_s == "F32" && record["source_dtype"].as_s == "BF16" &&
             record["byte_order"].as_s == "little" && shape == [POSITIONS.to_i64, HEADS.to_i64, HEAD_DIM.to_i64] &&
             offset == expected_offset && nbytes == TENSOR_BYTES &&
             HyNomasQueryCorrespondenceProbe.valid_sha256?(digest) &&
             offset >= 0 && offset <= INPUT_BYTES && nbytes <= INPUT_BYTES - offset
        raise ArgumentError.new("#{name} tensor dtype, shape, layout, range, or digest descriptor mismatch")
      end
      range = File.open(@payload_path, "rb") do |io|
        io.seek(offset)
        bytes = Bytes.new(nbytes.to_i32)
        io.read_fully(bytes)
        bytes
      end
      unless Digest::SHA256.hexdigest(range) == digest
        raise ArgumentError.new("#{name} tensor range digest mismatch")
      end
      {offset, nbytes}
    end
  end

  def self.valid_sha256?(value : String) : Bool
    /\A[0-9a-fA-F]{64}\z/.matches?(value)
  end

  def self.sha256_file(path : String) : String
    digest = Digest::SHA256.new
    File.open(path, "rb") do |io|
      scratch = Bytes.new(256 * 1024)
      while count = io.read(scratch)
        break if count == 0
        digest.update(scratch[0, count])
      end
    end
    digest.hexfinal
  end

  def self.canonical_file(path : String, label : String) : String
    expanded = File.expand_path(path)
    unless File.file?(expanded) && !File.symlink?(expanded) && File.realpath(expanded) == expanded
      raise ArgumentError.new("#{label} must be a canonical regular file without symlink components")
    end
    expanded
  end

  def self.exact_keys!(actual : Array(String), expected : Array(String), label : String) : Nil
    raise ArgumentError.new("#{label} fields differ from the pinned schema") unless actual == expected
  end

  def self.validate_expanded_bf16!(bytes : String, label : String) : Nil
    slice = bytes.to_slice
    raise ArgumentError.new("#{label} F32 tensor has an invalid byte length") unless slice.size == TENSOR_BYTES
    index = 0
    while index < slice.size
      bits = IO::ByteFormat::LittleEndian.decode(UInt32, slice[index, sizeof(UInt32)])
      unless (bits & 0x0000_ffff_u32) == 0
        raise ArgumentError.new("#{label} contains F32 values that are not exact BF16 expansions")
      end
      unless bits.unsafe_as(Float32).finite?
        raise ArgumentError.new("#{label} contains a non-finite F32 value")
      end
      index += sizeof(UInt32)
    end
  end

  def self.write_fresh_bytes(path : String, bytes : Bytes) : Nil
    write_fresh_parts(path, [bytes])
  end

  def self.write_fresh_parts(path : String, parts : Array(Bytes)) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("refusing to replace output: #{path}") if fd < 0
    begin
      parts.each do |bytes|
        offset = 0
        while offset < bytes.size
          written = LibC.write(fd, bytes.to_unsafe + offset, (bytes.size - offset).to_u64)
          raise IO::Error.new("failed writing fresh output: #{path}") if written <= 0
          offset += written.to_i
        end
      end
    ensure
      LibC.close(fd)
    end
  end
end
