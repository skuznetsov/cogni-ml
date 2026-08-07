# Opt-in fresh-process Darwin CPU parity probe for one real DINOv3 layer.
# The probe uses the pinned layer-0 checkpoint roles with a deterministic
# eight-token boundary; it never admits the full 24-layer encoder.

{% if flag?(:dinov3_real_block_parity_probe) %}
{% unless flag?(:darwin) %}
  {% raise "DINOv3 real block parity probe is admitted only for local Darwin measurement" %}
{% end %}

require "json"
require "digest/sha256"
require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

describe "DINOv3 real layer-0 numerical parity" do
  it "matches the independently generated source-backed block reference" do
    reference_path = ENV["DINO_V3_BLOCK_PARITY_REFERENCE"]? ||
                     raise "DINO_V3_BLOCK_PARITY_REFERENCE must name a generated reference directory"
    raise "reference directory does not exist: #{reference_path}" unless File.directory?(reference_path)
    reference_file = File.join(reference_path, "reference.json")
    raise "reference metadata is missing: #{reference_file}" unless File.file?(reference_file)
    reference = JSON.parse(File.read(reference_file))

    reference["schema"].as_s.should eq(
      "cogni-ml/trellis2/dino-v3-real-block-reference/v1"
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

    model = reference["model"]
    model["layer_index"].as_i.should eq(0)
    model["token_count"].as_i.should eq(8)
    model["patch_count"].as_i.should eq(3)
    model["hidden_size"].as_i.should eq(1024)
    model["intermediate_size"].as_i.should eq(4096)
    model["num_attention_heads"].as_i.should eq(16)
    model["head_dim"].as_i.should eq(64)
    model["num_register_tokens"].as_i.should eq(4)

    tolerance = reference["tolerance"]
    absolute_tolerance = tolerance["absolute"].as_f
    relative_tolerance = tolerance["relative"].as_f
    raise "reference absolute tolerance is too wide" unless
      0.0_f64 < absolute_tolerance <= 5.0e-4_f64
    raise "reference relative tolerance is too wide" unless
      0.0_f64 < relative_tolerance <= 5.0e-4_f64

    checkpoint = ENV["DINO_V3_CHECKPOINT"]? ||
                 raise "DINO_V3_CHECKPOINT must name the user-local model.safetensors"
    max_parameter_bytes = (
      ENV["DINO_V3_BLOCK_MAX_PARAMETER_BYTES"]? ||
      ML::Vision::DinoV3::DinoV3BlockCaller::MAX_PARAMETER_BYTES.to_s
    ).to_i64
    max_multiply_adds = ENV["DINO_V3_BLOCK_MAX_MULTIPLY_ADDS"]?.try(&.to_i64?) ||
                        raise "DINO_V3_BLOCK_MAX_MULTIPLY_ADDS must be set explicitly"
    rss_limit_mb = CogniSpecRSSGuard.limit_mb?
    raise "COGNI_SPEC_MAX_RSS_MB must be set for the real block parity probe" if rss_limit_mb <= 0

    certificate = ML::Vision::DinoV3::ConfigCertificate.parse(
      File.read(File.expand_path(File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json"))),
      source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
      source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
    )
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    manifest = ML::Vision::DinoV3::CheckpointManifest.parse(
      File.read(File.expand_path(File.join(__DIR__, "../../fixtures/trellis2/dino_v3_checkpoint_manifest_v1.json"))),
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
    caller = ML::Vision::DinoV3::DinoV3BlockCaller.from_checkpoint(
      inventory,
      certificate: certificate,
      runtime: runtime,
      semantic: semantic,
      digest: receipt,
      layer_index: 0,
      max_parameter_bytes: max_parameter_bytes
    )

    model["parameter_bytes"].as_i64.should eq(caller.parameters.parameter_byte_length)
    reference["parameter_f32le_sha256"].as_s.should eq(caller.parameters.f32le_sha256)
    max_multiply_adds.should eq(reference["budgets"]["multiply_adds"].as_i64)

    input_values = Array(Float32).new(8 * 1024) do |index|
      (((index.to_i64 * 17 + 5) % 43) - 21).to_f32 / 19.0_f32
    end
    input = ML::Tensor.from_array(input_values, ML::Shape.new(1_i32, 8_i32, 1024_i32))
    rope_cos = ML::Tensor.from_array(
      reference["inputs"]["rope_cos"]["values"].as_a.map(&.as_f.to_f32),
      ML::Shape.new(3_i32, 64_i32)
    )
    rope_sin = ML::Tensor.from_array(
      reference["inputs"]["rope_sin"]["values"].as_a.map(&.as_f.to_f32),
      ML::Shape.new(3_i32, 64_i32)
    )
    expect_raises(ML::Vision::DinoV3::BlockBudgetError, /exceeds/) do
      caller.forward_for_model_scale_probe(
        input,
        rope_cos,
        rope_sin,
        max_multiply_adds: max_multiply_adds - 1_i64
      )
    end
    trace = caller.forward_for_model_scale_probe(
      input,
      rope_cos,
      rope_sin,
      max_multiply_adds: max_multiply_adds
    )

    boundaries = {
      "input" => trace.input,
      "norm1" => trace.norm1,
      "q_heads" => trace.q_heads,
      "k_heads" => trace.k_heads,
      "v_heads" => trace.v_heads,
      "q_rope" => trace.q_rope,
      "k_rope" => trace.k_rope,
      "scores" => trace.scores,
      "probabilities" => trace.probabilities,
      "attention_context" => trace.attention_context,
      "output_projection" => trace.output_projection,
      "layer_scale1" => trace.layer_scale1,
      "first_residual" => trace.first_residual,
      "norm2" => trace.norm2,
      "mlp_up" => trace.mlp_up,
      "exact_gelu" => trace.exact_gelu,
      "mlp_down" => trace.mlp_down,
      "layer_scale2" => trace.layer_scale2,
      "block_output" => trace.block_output,
      "extractor_final" => trace.extractor_final,
    }
    boundaries.each do |name, tensor|
      metadata = reference["outputs"][name]
      tensor.shape.to_a.should eq(metadata["shape"].as_a.map(&.as_i.to_i32)), name
      expected_path = File.join(reference_path, metadata["file"].as_s)
      raise "reference output is missing: #{expected_path}" unless File.file?(expected_path)
      payload = File.read(expected_path).to_slice
      values = tensor.cpu_data.not_nil!
      payload.bytesize.should eq(values.size * 4), name
      expected_digest = Digest::SHA256.hexdigest(payload)
      expected_digest.should eq(metadata["f32le_sha256"].as_s), name
      values.each_with_index do |actual, index|
        expected = IO::ByteFormat::LittleEndian.decode(Float32, payload[index * 4, 4]).to_f64
        delta = (actual.to_f64 - expected).abs
        allowed = absolute_tolerance + relative_tolerance * expected.abs
        raise "#{name}[#{index}] actual=#{actual} expected=#{expected} delta=#{delta} allowed=#{allowed}" if delta > allowed
      end
    end

    puts(JSON.build do |json|
      json.object do
        json.field "probe", "dinov3-real-block-parity-v1"
        json.field "scope", "darwin-local-manual-layer-0-eight-token"
        json.field "checkpoint_bytes", inventory.file_byte_length
        json.field "layer_index", caller.layer_index
        json.field "parameter_bytes", caller.parameters.parameter_byte_length
        json.field "parameter_f32le_sha256", caller.parameters.f32le_sha256
        json.field "multiply_adds", max_multiply_adds
        json.field "rss_guard_limit_mb", rss_limit_mb
        json.field "forward", "cpu-reference-ran"
        json.field "full_encoder", "not-run"
        json.field "metal", "not-run"
      end
    end)
  end
end
{% end %}
