require "digest/sha256"
require "json"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/hy_nomas_mps_pointwise"
require "./hy_nomas_query_correspondence_probe"

module HyNomasQueryFixedPhaseProbe
  extend self

  INPUT_SCHEMA  = "hy-nomas-query-fixed-phase-input-v1"
  OUTPUT_SCHEMA = "hy-nomas-query-fixed-phase-native-v1"

  HEADS         =  16
  POSITIONS     = 144
  HEAD_DIM      = 128
  TENSOR_VALUES = HEADS * POSITIONS * HEAD_DIM
  TENSOR_BYTES  = TENSOR_VALUES.to_i64 * sizeof(Float32)
  PHASE_VALUES  = POSITIONS * HEAD_DIM
  PHASE_BYTES   = PHASE_VALUES.to_i64 * sizeof(Float32)
  SCORE_VALUES  = HEADS * POSITIONS * POSITIONS
  METRIC_COUNT  = HEADS * POSITIONS
  METRIC_WIDTH  = 6
  SCORE_BYTES   = SCORE_VALUES.to_i64 * sizeof(Float32)
  METRIC_BYTES  = METRIC_COUNT.to_i64 * METRIC_WIDTH * sizeof(Float32)
  OUTPUT_BYTES  = 2_i64 * TENSOR_BYTES + SCORE_BYTES + METRIC_BYTES
  NAMED_BYTES   = 2_i64 * TENSOR_BYTES + 2_i64 * PHASE_BYTES +
                  4_i64 * TENSOR_BYTES + SCORE_BYTES + METRIC_BYTES
  BUFFER_COUNT   = 10
  DISPATCH_COUNT =  8
  METRICS_ORDER  = HyNomasQueryCorrespondenceProbe::METRICS_ORDER
  FRAMES         = [1, 14, 29]
  ARMS           = %w(identity moving fixed)
  KEY_ROLL       = 17

  # These are the production equation sources. The BF16 helper is declared in
  # the block source; the uncontracted RoPE kernel is selected from pointwise.
  PRODUCTION_ROPE_SOURCE = ML::Video::HyNomasMPSPointwise::SOURCE
  PRODUCTION_BF16_SOURCE = ML::Video::HyNomasDualStreamBlock::SOURCE

  SUPPORT_KERNELS = <<-METAL
    #include <metal_stdlib>
    using namespace metal;

    constant uint HY_FP_HEADS = 16u;
    constant uint HY_FP_POSITIONS = 144u;
    constant uint HY_FP_HEAD_DIM = 128u;
    constant uint HY_FP_TENSOR_VALUES = HY_FP_HEADS * HY_FP_POSITIONS * HY_FP_HEAD_DIM;

    kernel void hy_nomas_query_fixed_phase_gather_f32(
        device const float *input [[buffer(0)]], device float *output [[buffer(1)]],
        constant uint &roll [[buffer(2)]], uint id [[thread_position_in_grid]]) {
      if (id >= HY_FP_TENSOR_VALUES) return;
      const uint row_width = HY_FP_HEADS * HY_FP_HEAD_DIM;
      const uint position = id / row_width;
      const uint source_position = (position + HY_FP_POSITIONS - roll) % HY_FP_POSITIONS;
      const uint source_index = source_position * row_width + id % row_width;
      output[id] = input[source_index];
    }

    kernel void hy_nomas_query_fixed_phase_bf16_roundtrip_inplace_f32(
        device float *values [[buffer(0)]], constant uint &count [[buffer(1)]],
        uint id [[thread_position_in_grid]]) {
      if (id < count) values[id] = hy_nomas_bf16_roundtrip_f32(values[id]);
    }
  METAL

  # All kernels are compiled from the exact production RoPE and BF16 helper
  # source plus the already-used correspondence score/metric kernels.
  METAL_SOURCE = PRODUCTION_BF16_SOURCE + "\n" + PRODUCTION_ROPE_SOURCE + "\n" +
                 HyNomasQueryCorrespondenceProbe::METAL_SOURCE + "\n" + SUPPORT_KERNELS

  class Fixture
    getter manifest_path : String
    getter manifest_sha256 : String
    getter phase_payload_path : String
    getter phase_payload_sha256 : String
    getter qk_manifest_sha256 : String
    getter qk : HyNomasQueryCorrespondenceProbe::Fixture
    getter frame : Int32
    getter arm : String
    getter key_roll : Int32
    getter phase_cosine_bytes : String
    getter phase_sine_bytes : String
    getter provenance : JSON::Any

    def initialize(path : String, expected_sha256 : String)
      unless HyNomasQueryCorrespondenceProbe.valid_sha256?(expected_sha256)
        raise ArgumentError.new("manifest SHA-256 must be 64 hexadecimal characters")
      end
      @manifest_path = HyNomasQueryCorrespondenceProbe.canonical_file(path, "fixed-phase input manifest")
      unless File.size(@manifest_path) <= 65_536
        raise ArgumentError.new("fixed-phase input manifest exceeds 64 KiB")
      end
      raw_manifest = File.read(@manifest_path)
      @manifest_sha256 = Digest::SHA256.hexdigest(raw_manifest)
      unless @manifest_sha256 == expected_sha256.downcase
        raise ArgumentError.new("fixed-phase input manifest digest mismatch")
      end

      document = JSON.parse(raw_manifest)
      HyNomasQueryCorrespondenceProbe.exact_keys!(document.as_h.keys.sort,
        %w(case payload provenance qk schema schema_version tensors), "fixed-phase manifest")
      unless document["schema"].as_s == INPUT_SCHEMA && document["schema_version"].as_i == 1
        raise ArgumentError.new("fixed-phase input schema or version mismatch")
      end

      case_document = document["case"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(case_document.as_h.keys.sort,
        %w(arm frame key_roll), "fixed-phase case")
      @frame = case_document["frame"].as_i
      @arm = case_document["arm"].as_s
      @key_roll = case_document["key_roll"].as_i
      unless FRAMES.includes?(@frame) && ARMS.includes?(@arm) &&
             @key_roll == (@arm == "identity" ? 0 : KEY_ROLL)
        raise ArgumentError.new("fixed-phase case differs from the frozen frame/arm matrix")
      end

      qk_descriptor = document["qk"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(qk_descriptor.as_h.keys.sort,
        %w(path sha256), "Q/K binding")
      qk_path = qk_descriptor["path"].as_s
      @qk_manifest_sha256 = qk_descriptor["sha256"].as_s
      unless absolute_canonical_string?(qk_path) &&
             HyNomasQueryCorrespondenceProbe.valid_sha256?(@qk_manifest_sha256)
        raise ArgumentError.new("Q/K binding path or digest is invalid")
      end
      @qk = HyNomasQueryCorrespondenceProbe::Fixture.new(qk_path, @qk_manifest_sha256)
      validate_qk_selection!(@qk.provenance || raise ArgumentError.new("Q/K input requires provenance"))

      payload = document["payload"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(payload.as_h.keys.sort,
        %w(bytes path sha256), "phase payload")
      payload_name = payload["path"].as_s
      payload_bytes = payload["bytes"].as_i64
      @phase_payload_sha256 = payload["sha256"].as_s
      unless !payload_name.empty? && File.basename(payload_name) == payload_name &&
             payload_name != "." && payload_name != ".." && payload_bytes == 2_i64 * PHASE_BYTES &&
             HyNomasQueryCorrespondenceProbe.valid_sha256?(@phase_payload_sha256)
        raise ArgumentError.new("phase payload path, size, or digest descriptor mismatch")
      end
      @phase_payload_path = HyNomasQueryCorrespondenceProbe.canonical_file(
        File.join(File.dirname(@manifest_path), payload_name), "phase payload")
      if @phase_payload_path == @manifest_path || @phase_payload_path == @qk.manifest_path ||
         File.size(@phase_payload_path).to_i64 != payload_bytes ||
         HyNomasQueryCorrespondenceProbe.sha256_file(@phase_payload_path) != @phase_payload_sha256
        raise ArgumentError.new("phase payload identity or digest mismatch")
      end

      tensors = document["tensors"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(tensors.as_h.keys.sort,
        %w(cosine sine), "phase tensor inventory")
      cosine_range = tensor_range!(tensors["cosine"], "cosine", 0_i64)
      sine_range = tensor_range!(tensors["sine"], "sine", PHASE_BYTES)
      unless cosine_range == {0_i64, PHASE_BYTES} && sine_range == {PHASE_BYTES, PHASE_BYTES} &&
             cosine_range[0] + cosine_range[1] == sine_range[0] &&
             sine_range[0] + sine_range[1] == payload_bytes
        raise ArgumentError.new("phase tensor ranges overlap or do not cover the payload exactly")
      end
      raw_phase = File.read(@phase_payload_path)
      @phase_cosine_bytes = raw_phase.byte_slice(0, PHASE_BYTES.to_i)
      @phase_sine_bytes = raw_phase.byte_slice(PHASE_BYTES.to_i, PHASE_BYTES.to_i)
      validate_finite_f32!(@phase_cosine_bytes, "cosine")
      validate_finite_f32!(@phase_sine_bytes, "sine")

      @provenance = document["provenance"]
      validate_provenance!(@provenance)
    end

    def query_bytes : String
      @qk.query_bytes
    end

    def key_bytes : String
      @qk.key_bytes
    end

    def pre_key_roll : Int32
      @arm == "fixed" ? KEY_ROLL : 0
    end

    def post_key_roll : Int32
      @arm == "moving" ? KEY_ROLL : 0
    end

    private def validate_qk_selection!(value : JSON::Any) : Nil
      source = value.as_h
      HyNomasQueryCorrespondenceProbe.exact_keys!(source.keys.sort,
        %w(boundary frame source), "Q/K provenance")
      unless source["frame"].as_i == @frame && source["boundary"].as_s == "pre" &&
             source["source"].as_h? != nil
        raise ArgumentError.new("Q/K provenance does not identify this frame's pre-RoPE capture")
      end
    end

    private def validate_provenance!(value : JSON::Any) : Nil
      expected = %w(kind phase_manifest phase_payload_sha256 phase_ranges_sha256 scope
        selection source_artifact_sha256 source_receipt_sha256 source_rows)
      HyNomasQueryCorrespondenceProbe.exact_keys!(value.as_h.keys.sort, expected.sort,
        "fixed-phase provenance")
      unless value["kind"].as_s == "same_pass_fixed_phase_content_null" &&
             value["selection"].as_h.keys.sort == %w(arm frame key_roll) &&
             value["selection"]["frame"].as_i == @frame &&
             value["selection"]["arm"].as_s == @arm &&
             value["selection"]["key_roll"].as_i == @key_roll &&
             value["source_rows"].as_a.map(&.as_i) == [@frame * POSITIONS, (@frame + 1) * POSITIONS] &&
             value["scope"].as_s == "Restricted same-pass affinity; no semantic ground truth, transport or acting"
        raise ArgumentError.new("fixed-phase provenance does not match the selected case")
      end

      # This provenance digest identifies the full authenticated parent phase
      # bundle; @phase_payload_sha256 identifies only the selected cosine/sine
      # rows copied into this case's local payload. The CPU qualifier verifies
      # the parent-to-slice lineage, while phase_ranges_sha256 binds these bytes.
      {"source_receipt_sha256"  => value["source_receipt_sha256"].as_s,
       "source_artifact_sha256" => value["source_artifact_sha256"].as_s,
       "phase_payload_sha256"   => value["phase_payload_sha256"].as_s}.each do |name, digest|
        unless HyNomasQueryCorrespondenceProbe.valid_sha256?(digest)
          raise ArgumentError.new("#{name} must be a SHA-256 digest")
        end
      end
      phase_manifest = value["phase_manifest"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(phase_manifest.as_h.keys.sort,
        %w(path sha256), "source phase manifest")
      unless absolute_canonical_string?(phase_manifest["path"].as_s) &&
             HyNomasQueryCorrespondenceProbe.valid_sha256?(phase_manifest["sha256"].as_s)
        raise ArgumentError.new("source phase manifest binding is invalid")
      end
      ranges = value["phase_ranges_sha256"]
      HyNomasQueryCorrespondenceProbe.exact_keys!(ranges.as_h.keys.sort,
        %w(cosine sine), "source phase range hashes")
      unless ranges["cosine"].as_s == Digest::SHA256.hexdigest(@phase_cosine_bytes) &&
             ranges["sine"].as_s == Digest::SHA256.hexdigest(@phase_sine_bytes)
        raise ArgumentError.new("selected phase ranges differ from their provenance digests")
      end
    end

    private def tensor_range!(record : JSON::Any, name : String,
                              expected_offset : Int64) : Tuple(Int64, Int64)
      HyNomasQueryCorrespondenceProbe.exact_keys!(record.as_h.keys.sort,
        %w(dtype nbytes offset_bytes sha256 shape), "#{name} phase tensor")
      shape = record["shape"].as_a.map(&.as_i)
      offset = record["offset_bytes"].as_i64
      nbytes = record["nbytes"].as_i64
      digest = record["sha256"].as_s
      unless record["dtype"].as_s == "F32" && shape == [POSITIONS, HEAD_DIM] &&
             offset == expected_offset && nbytes == PHASE_BYTES &&
             HyNomasQueryCorrespondenceProbe.valid_sha256?(digest) &&
             offset >= 0 && offset <= 2_i64 * PHASE_BYTES && nbytes <= 2_i64 * PHASE_BYTES - offset
        raise ArgumentError.new("#{name} phase dtype, shape, range, or digest descriptor mismatch")
      end
      range = File.open(@phase_payload_path, "rb") do |io|
        io.seek(offset)
        bytes = Bytes.new(nbytes.to_i32)
        io.read_fully(bytes)
        bytes
      end
      unless Digest::SHA256.hexdigest(range) == digest
        raise ArgumentError.new("#{name} phase range digest mismatch")
      end
      validate_finite_f32!(String.new(range), name)
      {offset, nbytes}
    end

    private def validate_finite_f32!(bytes : String, label : String) : Nil
      slice = bytes.to_slice
      unless slice.size == PHASE_BYTES
        raise ArgumentError.new("#{label} phase data has an invalid byte length")
      end
      offset = 0
      while offset < slice.size
        bits = IO::ByteFormat::LittleEndian.decode(UInt32, slice[offset, sizeof(UInt32)])
        unless bits.unsafe_as(Float32).finite?
          raise ArgumentError.new("#{label} phase data contains a non-finite F32 value")
        end
        offset += sizeof(UInt32)
      end
    end

    private def absolute_canonical_string?(path : String) : Bool
      !path.empty? && path.starts_with?("/") && File.expand_path(path) == path
    end
  end
end
