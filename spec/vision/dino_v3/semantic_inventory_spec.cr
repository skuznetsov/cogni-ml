require "json"

require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_semantic_certificate : ML::Vision::DinoV3::ConfigCertificate
  source = File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json")
  )
  ML::Vision::DinoV3::ConfigCertificate.parse(
    source,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

private def dino_v3_semantic_manifest : ML::Vision::DinoV3::CheckpointManifest
  ML::Vision::DinoV3::CheckpointManifest.parse(
    File.read(
      File.join(__DIR__, "../../fixtures/trellis2/dino_v3_checkpoint_manifest_v1.json")
    ),
    certificate: dino_v3_semantic_certificate
  )
end

private def unique_dino_v3_semantic_checkpoint_path : Tuple(String, String)
  root = File.tempname("pic3d-dino-v3-semantic")
  Dir.mkdir(root)
  {root, File.join(root, "model.safetensors")}
end

private def write_dino_v3_semantic_checkpoint(
  path : String,
  certificate : ML::Vision::DinoV3::ConfigCertificate,
  runtime : ML::Vision::DinoV3::RuntimeAdapter,
  *,
  drop_name : String? = nil,
  wrong_dtype_name : String? = nil,
  wrong_shape_name : String? = nil,
  extra_name : String? = nil,
)
  specs = ML::Vision::DinoV3::SemanticInventory.expected_specs(certificate, runtime)
  cursor = 0_i64
  ordered_specs = specs.sort_by do |spec|
    elements = 1_i64
    spec.shape.each { |dimension| elements *= dimension }
    elements
  end
  header = JSON.build do |json|
    json.object do
      ordered_specs.each do |spec|
        next if drop_name == spec.name

        dtype = wrong_dtype_name == spec.name ? "BF16" : spec.dtype
        shape = wrong_shape_name == spec.name ? [1_i64] : spec.shape
        bytes_per_element = dtype == "BF16" ? 2_i64 : 4_i64
        elements = 1_i64
        shape.each { |dimension| elements *= dimension }
        data_bytes = elements * bytes_per_element

        json.field(spec.name) do
          json.object do
            json.field("dtype", dtype)
            json.field("shape") do
              json.array { shape.each { |dimension| json.number(dimension) } }
            end
            json.field("data_offsets") do
              json.array do
                json.number(cursor)
                json.number(cursor + data_bytes)
              end
            end
          end
        end
        cursor += data_bytes
      end

      if extra_name
        json.field(extra_name) do
          json.object do
            json.field("dtype", "U8")
            json.field("shape") { json.array { json.number(0) } }
            json.field("data_offsets") do
              json.array do
                json.number(cursor)
                json.number(cursor)
              end
            end
          end
        end
      end
    end
  end

  File.open(path, "wb") do |file|
    header_byte_length = 41_400_i64
    header_byte_length.should be >= header.bytesize
    cursor.should be <= ML::Vision::DinoV3::CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH -
                        8_i64 - header_byte_length
    file.write_bytes(header_byte_length.to_u64, IO::ByteFormat::LittleEndian)
    file.write(header.to_slice)
    file.write((" " * (header_byte_length - header.bytesize).to_i).to_slice)
    file.truncate(ML::Vision::DinoV3::CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH)
  end
end

describe ML::Vision::DinoV3::SemanticInventory do
  it "derives the source-backed 415-tensor L/16 schema" do
    certificate = dino_v3_semantic_certificate
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    specs = ML::Vision::DinoV3::SemanticInventory.expected_specs(certificate, runtime)

    specs.size.should eq(415)
    patch_weight = specs.find { |spec| spec.role == "embedding.patch_weight" }.not_nil!
    patch_weight.name.should eq("embeddings.patch_embeddings.weight")
    patch_weight.dtype.should eq("F32")
    patch_weight.shape.should eq([1024_i64, 3_i64, 16_i64, 16_i64])

    final_norm = specs.find { |spec| spec.role == "final_norm.weight" }.not_nil!
    final_norm.name.should eq("norm.weight")
    final_norm.shape.should eq([1024_i64])

    last_layer = specs.find { |spec| spec.role == "block.23.mlp.down_weight" }.not_nil!
    last_layer.name.should eq("layer.23.mlp.down_proj.weight")
    last_layer.shape.should eq([1024_i64, 4096_i64])
  end

  it "binds all expected roles to a header without reading payload bytes" do
    certificate = dino_v3_semantic_certificate
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    root, path = unique_dino_v3_semantic_checkpoint_path

    begin
      write_dino_v3_semantic_checkpoint(path, certificate, runtime)
      inventory = ML::Vision::DinoV3::CheckpointInventory.load(
        path,
        manifest: dino_v3_semantic_manifest
      )
      semantic = ML::Vision::DinoV3::SemanticInventory.bind(
        inventory,
        certificate: certificate,
        runtime: runtime
      )

      semantic.schema.should eq(
        "cogni-ml/vision/dino-v3/semantic-inventory/v1"
      )
      semantic.tensor_count.should eq(415)
      semantic.tensor_for("embedding.patch_weight").name.should eq(
        "embeddings.patch_embeddings.weight"
      )
      semantic.tensor_for("block.23.mlp.down_weight").shape.should eq(
        [1024_i64, 4096_i64]
      )
      semantic.bindings.size.should eq(415)

      expect_raises(
        ML::Vision::DinoV3::SemanticInventoryError,
        /unknown DINOv3 semantic role/
      ) do
        semantic.tensor_for("missing.role")
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end

  it "rejects missing, mistyped, malformed, and unexpected semantic tensors" do
    certificate = dino_v3_semantic_certificate
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    scenarios = ["missing", "dtype", "shape", "extra"]

    scenarios.each do |label|
      root, path = unique_dino_v3_semantic_checkpoint_path
      begin
        case label
        when "missing"
          write_dino_v3_semantic_checkpoint(
            path,
            certificate,
            runtime,
            drop_name: "layer.23.mlp.down_proj.weight"
          )
        when "dtype"
          write_dino_v3_semantic_checkpoint(
            path,
            certificate,
            runtime,
            wrong_dtype_name: "layer.0.attention.k_proj.weight"
          )
        when "shape"
          write_dino_v3_semantic_checkpoint(
            path,
            certificate,
            runtime,
            wrong_shape_name: "layer.0.attention.q_proj.weight"
          )
        when "extra"
          write_dino_v3_semantic_checkpoint(
            path,
            certificate,
            runtime,
            extra_name: ""
          )
        end
        inventory = ML::Vision::DinoV3::CheckpointInventory.load(
          path,
          manifest: dino_v3_semantic_manifest
        )
        expect_raises(
          ML::Vision::DinoV3::SemanticInventoryError,
          /DINOv3 tensor/
        ) do
          ML::Vision::DinoV3::SemanticInventory.bind(
            inventory,
            certificate: certificate,
            runtime: runtime
          )
        end
      ensure
        File.delete(path) if File.exists?(path)
        Dir.delete(root) if Dir.exists?(root)
      end
    end
  end
end
