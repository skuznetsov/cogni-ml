require "spec"
require "../examples/support/hy_nomas_reference_block0_probe"

private def reference_block0_manifest(valid_tokens : Int32 = 529) : JSON::Any
  JSON.parse(JSON.build do |json|
    json.object do
      json.field "schema", HyNomasReferenceCacheProbe::SCHEMA
      json.field "schema_version", 1
      json.field "upstream_revision", HyNomasReferenceCacheProbe::UPSTREAM_REVISION
      json.field "scope", HyNomasReferenceCacheProbe::SCOPE
      json.field "preprocessing" do
        json.object do
          json.field "producer", "native"
          json.field "stages", ["timeline_reference_scatter", "timeline_cache_noise", "patch_projection", "text_refiner"]
          json.field "full_upstream_t0_preprocessing_parity", false
          json.field "timeline_upstream_revision", "911cea6458a89c2663513ff7099561ec190dd491"
        end
      end
      json.field "reference_latent_artifact_sha256", "a" * 64
      json.field "timeline_manifest_sha256", "b" * 64
      json.field "valid_text_tokens", valid_tokens
      json.field "cache_writer" do
        json.object do
          json.field "intent", "single_reference_initialization_forward"
          json.field "technical_label", "T0"
          json.field "segment_index", 0
          json.field "denoising_steps", 0
          json.field "sigma", 0.206897
          json.field "timestep", 206.897
        end
      end
      json.field "payload", "cache-inputs.tensors.bin"
      offset = 0_i64
      tensor_bytes = {} of String => Int64
      HyNomasReferenceCacheProbe::TENSOR_NAMES.each do |name|
        dtype, shape = HyNomasReferenceBlock0Probe.tensor_spec(name, valid_tokens)
        count = shape.reduce(1_i64) { |n, extent| n * extent }
        tensor_bytes[name] = count * (dtype == "BF16" ? 2 : 4)
      end
      payload_bytes = tensor_bytes.values.sum
      json.field "payload_bytes", payload_bytes
      json.field "payload_sha256", "c" * 64
      json.field "tensors" do
        json.object do
          HyNomasReferenceCacheProbe::TENSOR_NAMES.each do |name|
            dtype, shape = HyNomasReferenceBlock0Probe.tensor_spec(name, valid_tokens)
            nbytes = tensor_bytes[name]
            json.field name do
              json.object do
                json.field "dtype", dtype
                json.field "shape", shape
                json.field "offset", offset
                json.field "nbytes", nbytes
                json.field "sha256", "d" * 64
              end
            end
            offset += nbytes
          end
        end
      end
    end
  end)
end

describe HyNomasReferenceBlock0Probe do
  it "admits the authenticated dynamic valid prefix and exact input tensor inventory" do
    {1, 529, 1280}.each do |valid_tokens|
      info = HyNomasReferenceBlock0Probe.validate_manifest_document!(
        reference_block0_manifest(valid_tokens), "e" * 64)
      info.valid_text_tokens.should eq(valid_tokens)
      info.tensors.size.should eq(9)
      info.payload_bytes.should eq(info.tensors.values.sum(&.nbytes))
      info.tensors["block.txt_valid_f32"].shape.should eq([1, valid_tokens, 2048])
      info.tensors["block.img_f32"].shape.should eq([1, 4320, 2048])
    end
  end

  it "rejects noncanonical or invalid manifests before any Metal allocation" do
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_manifest_document!(reference_block0_manifest(0), "e" * 64)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_manifest_document!(reference_block0_manifest(1281), "e" * 64)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_manifest_document!(reference_block0_manifest, "not-a-sha")
    end
  end

  it "rejects payload traversal, tensor-range overlap, and unqualified preprocessing claims" do
    traversal = reference_block0_manifest
    traversal.as_h["payload"] = JSON::Any.new("../cache-inputs.tensors.bin")
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_manifest_document!(traversal, "e" * 64)
    end

    overlap = reference_block0_manifest
    overlap.as_h["tensors"].as_h["block.img_f32"].as_h["offset"] = JSON::Any.new(0_i64)
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_manifest_document!(overlap, "e" * 64)
    end

    parity_claim = reference_block0_manifest
    parity_claim.as_h["preprocessing"].as_h["full_upstream_t0_preprocessing_parity"] = JSON::Any.new(true)
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_manifest_document!(parity_claim, "e" * 64)
    end
  end

  it "keeps the source comparison keys and native cache semantics distinct" do
    HyNomasReferenceBlock0Probe::SOURCE_CAPTURE_NAMES.size.should eq(14)
    HyNomasReferenceBlock0Probe::SOURCE_CAPTURE_NAMES.should contain("img.q_rope")
    HyNomasReferenceBlock0Probe::SOURCE_CAPTURE_NAMES.should contain("cache.k_native_f32")
    HyNomasReferenceBlock0Probe.capture_shape("img.q_projection", 529).should eq([1, 4320, 16, 128])
    HyNomasReferenceBlock0Probe.capture_shape("txt.q_qknorm", 529).should eq([1, 529, 16, 128])
    HyNomasReferenceBlock0Probe.capture_shape("cache.k_native_f32", 529).should eq([1, 4849, 16, 128])
    HyNomasReferenceBlock0Probe.capture_byte_count("cache.v_native_f32", 529).should eq(
      4849 * 2048 * sizeof(Float32))
    HyNomasReferenceBlock0Probe::BLOCK0_WEIGHT_SHA256.should eq(
      "01be7388fa4267f1f566c3ced405418f94d0ed2b4ee969cc880fb2518ab7c87b")
  end

  it "builds exact byte extents and safetensors offsets for all 14 captures" do
    valid_tokens = 529
    entries = HyNomasReferenceBlock0Probe.capture_header_entries(valid_tokens)
    entries.map(&.name).should eq(HyNomasReferenceBlock0Probe::SOURCE_CAPTURE_NAMES)

    offset = 0_i64
    entries.each do |entry|
      entry.shape.should eq(HyNomasReferenceBlock0Probe.capture_shape(entry.name, valid_tokens))
      entry.byte_count.should eq(HyNomasReferenceBlock0Probe.capture_byte_count(entry.name, valid_tokens))
      offset += entry.byte_count
    end

    header = HyNomasReferenceBlock0Probe.capture_header(entries)
    (header.bytesize % 8).should eq(0)
    parsed = JSON.parse(header).as_h
    offset = 0_i64
    entries.each do |entry|
      tensor = parsed[entry.name].as_h
      tensor["dtype"].as_s.should eq("F32")
      tensor["shape"].as_a.map(&.as_i.to_i32).should eq(entry.shape)
      tensor["data_offsets"].as_a.map(&.as_i64).should eq([offset, offset + entry.byte_count])
      offset += entry.byte_count
    end
    file_bytes = HyNomasReferenceBlock0Probe.safetensors_file_bytes(header, offset)
    file_bytes.should eq(sizeof(UInt64).to_i64 + header.bytesize.to_i64 + offset)
    file_bytes.should be > offset

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.safetensors_file_bytes(header, -1_i64)
    end

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.capture_header_entries(0)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.capture_header_entries(1281)
    end
  end

  it "defines the opt-in pre-QKV companion inventory and schema metadata" do
    valid_tokens = 529
    HyNomasReferenceBlock0Probe::PREQKV_SCHEMA.should eq(
      "cogni_ml.hy_nomas_reference_block0_preqkv_probe/v1")
    HyNomasReferenceBlock0Probe::PREQKV_SCOPE.should eq(
      "diagnostic pre-QKV boundaries from one standalone native block-0 forward using authenticated native preblock inputs; not a complete upstream T0 handoff")
    HyNomasReferenceBlock0Probe::PREQKV_CAPTURE_NAMES.should eq([
      "img.mod_silu", "img.mod_params", "img.norm1", "img.qkv_input",
      "txt.mod_silu", "txt.mod_params", "txt.norm1", "txt.qkv_input",
    ])
    expected_shapes = [
      [1, 2048], [1, 12_288], [1, 4320, 2048], [1, 4320, 2048],
      [1, 2048], [1, 12_288], [1, valid_tokens, 2048], [1, valid_tokens, 2048],
    ]
    entries = HyNomasReferenceBlock0Probe.preqkv_header_entries(valid_tokens)
    entries.map(&.shape).should eq(expected_shapes)
    entries.each do |entry|
      count = entry.shape.reduce(1) { |n, extent| n * extent }
      entry.byte_count.should eq(count * sizeof(Float32))
    end

    header = HyNomasReferenceBlock0Probe.capture_header(entries,
      HyNomasReferenceBlock0Probe::PREQKV_CAPTURE_NAMES,
      HyNomasReferenceBlock0Probe.preqkv_metadata_fields(
        "f32-control", "a" * 64, "b" * 64))
    parsed = JSON.parse(header).as_h
    parsed.size.should eq(HyNomasReferenceBlock0Probe::PREQKV_CAPTURE_NAMES.size + 1)
    parsed["__metadata__"].as_h.should eq({
      "format"                        => JSON::Any.new(HyNomasReferenceBlock0Probe::PREQKV_SCHEMA),
      "native_revision"               => JSON::Any.new(HyNomasReferenceBlock0Probe::NATIVE_REVISION),
      "source_revision"               => JSON::Any.new(HyNomasReferenceBlock0Probe::SOURCE_REVISION),
      "precision_control"             => JSON::Any.new("F32Control"),
      "parent_native_receipt_sha256"  => JSON::Any.new("a" * 64),
      "parent_native_artifact_sha256" => JSON::Any.new("b" * 64),
    })

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.preqkv_capture_shape("txt.norm1", 0)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.preqkv_capture_shape("txt.ffn", valid_tokens)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.preqkv_metadata_fields("f32-control", "not-a-sha", "b" * 64)
    end

    HyNomasReferenceBlock0Probe.preqkv_precision_label("f32-control").should eq("F32Control")
    HyNomasReferenceBlock0Probe.preqkv_precision_label("bf16-boundary-control").should eq("BF16BoundaryControl")
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.preqkv_precision_label("unknown")
    end
  end

  it "requires an explicit opt-in and paired companion outputs for execution" do
    HyNomasReferenceBlock0Probe.validate_preqkv_options!(false, false, "", "")
    HyNomasReferenceBlock0Probe.validate_preqkv_options!(true, false, "", "")
    HyNomasReferenceBlock0Probe.validate_preqkv_options!(true, true, "/tmp/capture.safetensors", "/tmp/capture.json")

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_preqkv_options!(false, true, "/tmp/capture.safetensors", "/tmp/capture.json")
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_preqkv_options!(true, true, "", "")
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_preqkv_options!(true, true, "/tmp/capture.safetensors", "")
    end
  end

  it "admits the reference BF16 pre-attention mode only with a supported linear variant" do
    HyNomasReferenceBlock0Probe.validate_precision_options!(
      "reference-bf16-preattention", "bf16-graph", false)
    HyNomasReferenceBlock0Probe.validate_precision_options!(
      "reference-bf16-preattention", "f32-accum", true)
    HyNomasReferenceBlock0Probe.validate_precision_options!("bf16-boundary-control", "bf16-graph", false)
    HyNomasReferenceBlock0Probe.validate_precision_options!("f32-control", "bf16-graph", false)

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_precision_options!(
        "reference-bf16-preattention", "unknown", true)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_precision_options!("bf16-boundary-control", "f32-accum", true)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.validate_precision_options!("unknown", "bf16-graph", false)
    end
  end

  it "binds pre-QKV metadata to the reference BF16 variant without widening legacy metadata" do
    legacy = HyNomasReferenceBlock0Probe.preqkv_metadata_fields(
      "bf16-boundary-control", "a" * 64, "b" * 64)
    legacy.has_key?("bf16_linear_variant").should be_false

    metadata = HyNomasReferenceBlock0Probe.preqkv_metadata_fields(
      "reference-bf16-preattention", "a" * 64, "b" * 64, "bf16-graph")
    metadata["precision_control"].should eq("reference-bf16-preattention")
    metadata["bf16_linear_variant"].should eq("bf16-graph")

    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.preqkv_metadata_fields(
        "reference-bf16-preattention", "a" * 64, "b" * 64)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.preqkv_metadata_fields(
        "reference-bf16-preattention", "a" * 64, "b" * 64, "unknown")
    end
  end

  it "rejects capture shapes outside the exact block-0 comparison inventory" do
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.capture_shape("cache.attention", 529)
    end
    expect_raises(ArgumentError) do
      HyNomasReferenceBlock0Probe.capture_shape("txt.q_projection", 0)
    end
  end
end
