require "digest/sha256"
require "json"

require "../../../src/ml/vision/dino_v3"
require "../../spec_helper"

private def dino_v3_decoder_certificate : ML::Vision::DinoV3::ConfigCertificate
  source = File.read(
    File.join(__DIR__, "../../fixtures/trellis2/dino_v3_config_certificate_v1.json")
  )
  ML::Vision::DinoV3::ConfigCertificate.parse(
    source,
    source_model: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_MODEL,
    source_revision: ML::Vision::DinoV3::ConfigCertificate::PINNED_SOURCE_REVISION
  )
end

private def dino_v3_decoder_manifest : ML::Vision::DinoV3::CheckpointManifest
  certificate = dino_v3_decoder_certificate
  ML::Vision::DinoV3::CheckpointManifest.parse(
    File.read(
      File.join(__DIR__, "../../fixtures/trellis2/dino_v3_checkpoint_manifest_v1.json")
    ),
    certificate: certificate
  )
end

private def unique_dino_v3_decoder_checkpoint_path : Tuple(String, String)
  root = File.tempname("pic3d-dino-v3-decoder")
  Dir.mkdir(root)
  {root, File.join(root, "model.safetensors")}
end

private def write_dino_v3_decoder_checkpoint(
  path : String,
  certificate : ML::Vision::DinoV3::ConfigCertificate,
  runtime : ML::Vision::DinoV3::RuntimeAdapter,
) : Nil
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
        elements = 1_i64
        spec.shape.each { |dimension| elements *= dimension }
        data_bytes = elements * 4_i64
        json.field(spec.name) do
          json.object do
            json.field("dtype", "F32")
            json.field("shape") do
              json.array { spec.shape.each { |dimension| json.number(dimension) } }
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

describe ML::Vision::DinoV3::CheckpointF32Decoder do
  it "verifies a digest receipt and decodes one semantic F32 role under a resident cap" do
    certificate = dino_v3_decoder_certificate
    runtime = ML::Vision::DinoV3::RuntimeAdapter.new(certificate)
    manifest = dino_v3_decoder_manifest
    root, path = unique_dino_v3_decoder_checkpoint_path

    begin
      write_dino_v3_decoder_checkpoint(path, certificate, runtime)
      inventory = ML::Vision::DinoV3::CheckpointInventory.load(path, manifest: manifest)
      semantic = ML::Vision::DinoV3::SemanticInventory.bind(
        inventory,
        certificate: certificate,
        runtime: runtime
      )

      File.open(path, "r+b") do |file|
        file.seek(8_i64 + 41_400_i64)
        file.write(Bytes[0, 0, 128, 63])
      end
      expected_sha256 = Digest::SHA256.new.file(path).hexfinal
      receipt = ML::Vision::DinoV3::CheckpointDigestReceipt.verify(
        inventory,
        expected_sha256: expected_sha256
      )
      decoder = ML::Vision::DinoV3::CheckpointF32Decoder.new(
        inventory,
        semantic: semantic,
        digest: receipt
      )

      decoded = decoder.decode("embedding.cls_token")
      decoded.role.should eq("embedding.cls_token")
      decoded.name.should eq("embeddings.cls_token")
      decoded.shape.should eq([1_i64, 1_i64, 1024_i64])
      decoded.values.size.should eq(1024)
      decoded.values[0].should eq(1.0_f32)
      decoded.values[1].should eq(0.0_f32)

      expect_raises(
        ML::Vision::DinoV3::CheckpointDigestError,
        /SHA-256 mismatch/
      ) do
        ML::Vision::DinoV3::CheckpointDigestReceipt.verify(
          inventory,
          expected_sha256: "0" * 64
        )
      end
      expect_raises(
        ML::Vision::DinoV3::CheckpointDigestError,
        /SHA-256 mismatch/
      ) do
        ML::Vision::DinoV3::CheckpointDigestReceipt.verify(
          inventory,
          manifest: manifest
        )
      end
      expect_raises(
        ML::Vision::DinoV3::CheckpointDecoderError,
        /resident byte budget/
      ) do
        ML::Vision::DinoV3::CheckpointF32Decoder.new(
          inventory,
          semantic: semantic,
          digest: receipt,
          max_resident_bytes: 1024_i64
        ).decode("embedding.cls_token")
      end
    ensure
      File.delete(path) if File.exists?(path)
      Dir.delete(root) if Dir.exists?(root)
    end
  end
end
