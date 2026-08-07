# Opt-in fresh-process Darwin CPU parity probe for the real DINOv3 embedding
# boundary. A separate Python reference writes little-endian F32 outputs; this
# probe compares the complete patch/prefix/coordinate/RoPE boundary.

{% if flag?(:dinov3_real_embedding_parity_probe) %}
{% unless flag?(:darwin) %}
  {% raise "DINOv3 real embedding parity probe is admitted only for local Darwin measurement" %}
{% end %}

require "json"
require "digest/sha256"
require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

module DinoV3RealEmbeddingParityProbe
  extend self

  PROBE_EDGE                 = 512_i32
  EXPECTED_FILE_BYTES        = 1_212_559_808_i64
  EXPECTED_PARAMETER_BYTES   = 3_170_304_i64
  EXPECTED_PARAMETER_SHA256  = "8124dfb9859ecbb6d33908bcd013e32a469897140cb413c2aa5b629699a1f56f"
  EXPECTED_MULTIPLY_ADDS     = 805_306_368_i64
  REFERENCE_SCHEMA            = "cogni-ml/trellis2/dino-v3-real-embedding-reference/v1"
  MAX_REFERENCE_ABSOLUTE_TOLERANCE = 1.0e-5_f64
  MAX_REFERENCE_RELATIVE_TOLERANCE = 1.0e-5_f64

  record BoundaryMetrics,
    name : String,
    elements : Int64,
    max_absolute_error : Float64,
    max_relative_error : Float64,
    max_error_index : Int64

  def fixture_path(relative : String) : String
    File.expand_path(File.join(__DIR__, "../../fixtures", relative))
  end

  def input : ML::Tensor
    tensor = ML::Tensor.new(
      ML::Shape.new(1_i32, 3_i32, PROBE_EDGE, PROBE_EDGE),
      device: ML::Tensor::Device::CPU
    )
    values = tensor.cpu_data.not_nil!
    3.times do |channel|
      PROBE_EDGE.times do |y|
        PROBE_EDGE.times do |x|
          offset = channel * PROBE_EDGE * PROBE_EDGE + y * PROBE_EDGE + x
          values[offset] = (
            ((17 * channel + 3 * y + 5 * x) % 257) - 128
          ).to_f32 / 128.0_f32
        end
      end
    end
    tensor
  end

  def f32le_sha256(values : Indexable(Float32)) : String
    bytes = Bytes.new(values.size * 4, 0_u8)
    values.each_with_index do |value, index|
      IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
    end
    Digest::SHA256.hexdigest(bytes)
  end

  def shape_from(metadata : JSON::Any) : Array(Int32)
    metadata.as_a.map(&.as_i.to_i32)
  end

  def compare_boundary(
    name : String,
    tensor : ML::Tensor,
    payload : String,
    metadata : JSON::Any,
    absolute_tolerance : Float64,
    relative_tolerance : Float64,
  ) : BoundaryMetrics
    values = tensor.cpu_data.not_nil!
    expected_shape = shape_from(metadata["shape"])
    tensor.shape.to_a.should eq(expected_shape), "#{name} shape"
    expected_bytes = values.size.to_i64 * 4_i64
    metadata["byte_length"].as_i.should eq(expected_bytes)
    payload.bytesize.to_i64.should eq(expected_bytes), "#{name} byte length"
    Digest::SHA256.hexdigest(payload.to_slice).should eq(
      metadata["f32le_sha256"].as_s
    ), "#{name} reference digest"

    max_absolute_error = 0.0_f64
    max_relative_error = 0.0_f64
    max_error_index = 0_i64
    index = 0
    payload_bytes = payload.to_slice
    while index < values.size
      expected = IO::ByteFormat::LittleEndian.decode(
        Float32,
        payload_bytes[index * 4, 4]
      ).to_f64
      actual = values[index].to_f64
      raise "#{name}[#{index}] is non-finite" unless actual.finite? && expected.finite?
      absolute_error = (actual - expected).abs
      relative_error = absolute_error / {expected.abs, 1.0e-12_f64}.max
      allowed_error = absolute_tolerance + relative_tolerance * expected.abs
      if absolute_error > allowed_error
        raise "#{name}[#{index}] exceeds parity tolerance: " \
              "actual=#{actual}, reference=#{expected}, " \
              "absolute_error=#{absolute_error}, allowed=#{allowed_error}"
      end
      if absolute_error > max_absolute_error
        max_absolute_error = absolute_error
        max_error_index = index.to_i64
      end
      max_relative_error = relative_error if relative_error > max_relative_error
      index += 1
    end

    BoundaryMetrics.new(
      name,
      values.size.to_i64,
      max_absolute_error,
      max_relative_error,
      max_error_index
    )
  end
end

describe "DINOv3 real embedding numerical parity" do
  it "matches the independently generated source-backed reference" do
    reference_path = ENV["DINO_V3_PARITY_REFERENCE"]? ||
                     raise "DINO_V3_PARITY_REFERENCE must name a generated reference directory"
    raise "reference directory does not exist: #{reference_path}" unless File.directory?(reference_path)
    reference_file = File.join(reference_path, "reference.json")
    raise "reference metadata is missing: #{reference_file}" unless File.file?(reference_file)
    reference = JSON.parse(File.read(reference_file))

    reference["schema"].as_s.should eq(
      DinoV3RealEmbeddingParityProbe::REFERENCE_SCHEMA
    )
    reference["reference"]["trellis_revision"].as_s.should eq(
      ML::Vision::DinoV3::TRELLIS2_SOURCE_REVISION
    )
    reference["reference"]["transformers_modeling_sha256"].as_s.should eq(
      ML::Vision::DinoV3::TRANSFORMERS_MODELING_SHA256
    )
    reference["reference"]["transformers_config_sha256"].as_s.should eq(
      ML::Vision::DinoV3::TRANSFORMERS_CONFIG_SHA256
    )
    reference["reference"]["transformers_dinov3_runtime_available"].as_bool.should be_false
    reference["reference"]["transformers_version"].raw.nil?.should be_true

    model = reference["model"]
    model["edge"].as_i.should eq(DinoV3RealEmbeddingParityProbe::PROBE_EDGE)
    model["patch_size"].as_i.should eq(16)
    model["num_channels"].as_i.should eq(3)
    model["hidden_size"].as_i.should eq(1024)
    model["num_attention_heads"].as_i.should eq(16)
    model["num_register_tokens"].as_i.should eq(4)
    model["rope_theta"].as_f.should eq(100.0)

    tolerance = reference["tolerance"]
    absolute_tolerance = tolerance["absolute"].as_f
    relative_tolerance = tolerance["relative"].as_f
    raise "reference absolute tolerance is too wide" unless
      0.0_f64 < absolute_tolerance <= DinoV3RealEmbeddingParityProbe::MAX_REFERENCE_ABSOLUTE_TOLERANCE
    raise "reference relative tolerance is too wide" unless
      0.0_f64 < relative_tolerance <= DinoV3RealEmbeddingParityProbe::MAX_REFERENCE_RELATIVE_TOLERANCE

    checkpoint = ENV["DINO_V3_CHECKPOINT"]? ||
                 raise "DINO_V3_CHECKPOINT must name the user-local model.safetensors"
    max_parameter_bytes = (
      ENV["DINO_V3_MAX_PARAMETER_BYTES"]? ||
      ML::Vision::DinoV3::DinoV3EmbeddingCaller::MAX_PARAMETER_BYTES.to_s
    ).to_i64
    rss_limit_mb = CogniSpecRSSGuard.limit_mb?
    raise "COGNI_SPEC_MAX_RSS_MB must be set for the real parity probe" if rss_limit_mb <= 0
    max_multiply_adds = ENV["DINO_V3_FORWARD_MAX_MULTIPLY_ADDS"]?.try(&.to_i64?) ||
                        raise "DINO_V3_FORWARD_MAX_MULTIPLY_ADDS must be set explicitly"
    max_multiply_adds.should eq(
      DinoV3RealEmbeddingParityProbe::EXPECTED_MULTIPLY_ADDS
    )

    checkpoint_metadata = reference["checkpoint"]
    checkpoint_metadata["byte_length"].as_i.should eq(
      DinoV3RealEmbeddingParityProbe::EXPECTED_FILE_BYTES
    )
    checkpoint_metadata["sha256"].as_s.should eq(
      ML::Vision::DinoV3::CheckpointManifest::PINNED_WEIGHTS_SHA256
    )

    certificate = ML::Vision::DinoV3::ConfigCertificate.parse(
      File.read(DinoV3RealEmbeddingParityProbe.fixture_path("trellis2/dino_v3_config_certificate_v1.json")),
      source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
      source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
    )
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      File.read(DinoV3RealEmbeddingParityProbe.fixture_path("trellis2/dino_v3_checkpoint_manifest_v1.json")),
      certificate: certificate
    )
    inventory = ML::Vision::DinoV3::CheckpointInventory.load(checkpoint, manifest: manifest)
    semantic = ML::Vision::DinoV3::SemanticInventory.bind(
      inventory,
      certificate: certificate,
      runtime: runtime
    )
    receipt = ML::Vision::DinoV3::CheckpointDigestReceipt.verify(
      inventory,
      manifest: manifest
    )
    inventory.file_byte_length.should eq(
      DinoV3RealEmbeddingParityProbe::EXPECTED_FILE_BYTES
    )
    receipt.sha256.should eq(
      checkpoint_metadata["sha256"].as_s
    )

    default_budget_rejected = false
    begin
      ML::Vision::DinoV3::DinoV3EmbeddingCaller.preflight(
        certificate,
        runtime,
        DinoV3RealEmbeddingParityProbe::PROBE_EDGE
      )
    rescue ML::Vision::DinoV3::EmbeddingBudgetError
      default_budget_rejected = true
    end
    default_budget_rejected.should be_true

    caller = ML::Vision::DinoV3::DinoV3EmbeddingCaller.from_checkpoint(
      inventory,
      certificate: certificate,
      runtime: runtime,
      semantic: semantic,
      digest: receipt,
      max_parameter_bytes: max_parameter_bytes
    )
    caller.parameters.parameter_byte_length.should eq(
      DinoV3RealEmbeddingParityProbe::EXPECTED_PARAMETER_BYTES
    )
    caller.parameters.f32le_sha256.should eq(
      DinoV3RealEmbeddingParityProbe::EXPECTED_PARAMETER_SHA256
    )
    reference["parameter_f32le_sha256"].as_s.should eq(
      caller.parameters.f32le_sha256
    )
    reference["parameter_order"].as_a.map(&.as_s).should eq(
      [
        "embedding.patch_weight",
        "embedding.patch_bias",
        "embedding.cls_token",
        "embedding.register_tokens",
      ]
    )

    input = DinoV3RealEmbeddingParityProbe.input
    input_values = input.cpu_data.not_nil!
    input_values.min.should be < 0.0_f32
    input_values.max.should be > 0.0_f32
    input_metadata = reference["input"]
    input.shape.to_a.should eq(
      DinoV3RealEmbeddingParityProbe.shape_from(input_metadata["shape"])
    )
    DinoV3RealEmbeddingParityProbe.f32le_sha256(input_values).should eq(
      input_metadata["f32le_sha256"].as_s
    )

    plan = ML::Vision::DinoV3::EmbeddingCPU.preflight_model_scale_probe(
      caller.config,
      DinoV3RealEmbeddingParityProbe::PROBE_EDGE,
      max_multiply_adds: max_multiply_adds
    )
    plan.multiply_adds.should eq(
      DinoV3RealEmbeddingParityProbe::EXPECTED_MULTIPLY_ADDS
    )
    result = caller.forward_for_model_scale_probe(
      input,
      max_multiply_adds: max_multiply_adds
    )

    boundaries = [
      {"patches", result.patches},
      {"embeddings", result.embeddings},
      {"coordinates", result.patch_coordinates},
      {"rope_cos", result.rope_cos},
      {"rope_sin", result.rope_sin},
    ]
    metrics = [] of DinoV3RealEmbeddingParityProbe::BoundaryMetrics
    boundaries.each do |boundary|
      name = boundary[0]
      output_metadata = reference["outputs"][name]
      output_file = File.join(reference_path, output_metadata["file"].as_s)
      raise "reference output is missing: #{output_file}" unless File.file?(output_file)
      payload = File.read(output_file)
      metrics << DinoV3RealEmbeddingParityProbe.compare_boundary(
        name,
        boundary[1],
        payload,
        output_metadata,
        absolute_tolerance,
        relative_tolerance
      )
    end

    measurement = JSON.build do |json|
      json.object do
        json.field "probe", "dinov3-real-embedding-parity-v1"
        json.field "scope", "darwin-local-manual-512px-embedding-boundary"
        json.field "checkpoint_bytes", inventory.file_byte_length
        json.field "parameter_bytes", caller.parameters.parameter_byte_length
        json.field "parameter_f32le_sha256", caller.parameters.f32le_sha256
        json.field "input_f32le_sha256", input_metadata["f32le_sha256"].as_s
        json.field "default_budget_rejected", default_budget_rejected
        json.field "multiply_adds", plan.multiply_adds
        json.field "forward", "cpu-reference-ran"
        json.field "blocks_loaded", 0
        json.field "metal", "not-run"
        json.field "reference_backend", "torch.nn.functional.conv2d"
        json.field "reference_transformers_version", nil
        json.field "rss_guard_limit_mb", rss_limit_mb
        json.field "boundaries" do
          json.object do
            metrics.each do |measurement|
              json.field measurement.name do
                json.object do
                  json.field "elements", measurement.elements
                  json.field "max_absolute_error", measurement.max_absolute_error
                  json.field "max_relative_error", measurement.max_relative_error
                  json.field "max_error_index", measurement.max_error_index
                end
              end
            end
          end
        end
      end
    end
    puts measurement
  end
end
{% end %}
