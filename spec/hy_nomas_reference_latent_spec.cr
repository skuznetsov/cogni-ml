require "spec"
require "../examples/support/hy_nomas_reference_latent"

private def reference_tensor(name = "latent", dtype = ML::GGUF::SafeTensorDType::BF16,
                             shape = HyNomasReferenceLatent::SHAPE,
                             bytes = HyNomasReferenceLatent::BF16_BYTES,
                             elements = HyNomasReferenceLatent::ELEMENTS.to_i64)
  ML::Video::CheckpointTensor.new(name, dtype, shape, 0_i64, bytes, elements)
end

describe HyNomasReferenceLatent do
  it "admits the exact saved reference representation" do
    HyNomasReferenceLatent.validate_record!(reference_tensor).should be_nil
    HyNomasReferenceLatent::BF16_BYTES.should eq(276_480_i64)
    HyNomasReferenceLatent::F32_BYTES.should eq(552_960_i64)
  end

  it "rejects same-size tensors with a different meaning or geometry" do
    invalid = [
      reference_tensor(name: "noise"),
      reference_tensor(dtype: ML::GGUF::SafeTensorDType::F16),
      reference_tensor(shape: [1_i64, 32_i64, 30_i64, 16_i64, 9_i64]),
      reference_tensor(bytes: HyNomasReferenceLatent::BF16_BYTES - 2),
      reference_tensor(elements: HyNomasReferenceLatent::ELEMENTS.to_i64 - 1),
    ]
    invalid.each do |record|
      expect_raises(ArgumentError, /requires BF16 latent/) do
        HyNomasReferenceLatent.validate_record!(record)
      end
    end
  end

  it "authenticates file bytes and rejects a missing latent before GPU setup" do
    File.tempfile("reference-latent", ".safetensors") do |file|
      header = %({"other":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})
      file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      file.write(header.to_slice)
      file.write_bytes(0_u16, IO::ByteFormat::LittleEndian)
      file.flush
      digest = Digest::SHA256.hexdigest(File.read(file.path))
      expect_raises(ArgumentError, /SHA-256 mismatch/) do
        HyNomasReferenceLatent::Input.new(file.path, "0" * 64)
      end
      expect_raises(ArgumentError, /no latent tensor/) do
        HyNomasReferenceLatent::Input.new(file.path, digest)
      end
      link = "#{file.path}.link"
      begin
        File.symlink(file.path, link)
        expect_raises(ArgumentError, /non-symlink regular file/) do
          HyNomasReferenceLatent::Input.new(link, digest)
        end
      ensure
        File.delete(link) if File.symlink?(link)
      end
    end
  end

  it "admits an authenticated saved latent without creating GPU owners" do
    File.tempfile("reference-latent-valid", ".safetensors") do |file|
      header = %({"latent":{"dtype":"BF16","shape":[1,32,30,9,16],"data_offsets":[0,276480]}})
      file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      file.write(header.to_slice)
      file.write(Bytes.new(HyNomasReferenceLatent::BF16_BYTES.to_i32, 0_u8))
      file.flush
      digest = Digest::SHA256.hexdigest(File.read(file.path))
      input = HyNomasReferenceLatent::Input.new(file.path, digest)
      input.sha256.should eq(digest)
      input.close
      input.close
      expect_raises(ArgumentError, /input is closed/) { input.upload_bf16! }
    end
  end
end
