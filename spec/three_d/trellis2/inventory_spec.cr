require "../../spec_helper"
require "file_utils"
require "../../../src/ml/three_d/trellis2"
require "./support"

private class PayloadSentinelIO < IO
  getter bytes_read = 0_i64

  def initialize(@inner : IO::Memory, @read_limit : Int64)
  end

  def read(slice : Bytes) : Int32
    return 0 if slice.empty?
    remaining = @read_limit - @bytes_read
    if remaining <= 0
      raise IO::Error.new("safetensors payload was read")
    end
    allowed = Math.min(slice.size.to_i64, remaining).to_i
    read = @inner.read(slice[0, allowed])
    @bytes_read += read
    read
  end

  def write(slice : Bytes) : Nil
    raise IO::Error.new("sentinel is read-only")
  end
end

private def header_for(tensors : Array({String, String, Array(Int64), Int64, Int64})) : String
  JSON.build do |json|
    json.object do
      tensors.each do |name, dtype, shape, start_offset, end_offset|
        json.field name do
          json.object do
            json.field "dtype", dtype
            json.field "shape" do
              json.array { shape.each { |dim| json.number(dim) } }
            end
            json.field "data_offsets" do
              json.array do
                json.number start_offset
                json.number end_offset
              end
            end
          end
        end
      end
    end
  end
end

private def expect_inventory_error(path : String, pattern : Regex)
  expect_raises(ML::ThreeD::Trellis2::InventoryError, pattern) do
    ML::ThreeD::Trellis2::SafetensorsInventory.read(path)
  end
end

describe ML::ThreeD::Trellis2::SafetensorsInventory do
  it "reads header metadata without materializing tensor payloads" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      path = File.join(dir, "tiny.safetensors")
      header = header_for([
        {"weight", "F16", [2_i64, 2_i64], 0_i64, 8_i64},
        {"bias", "F32", [2_i64], 8_i64, 16_i64},
      ])
      Trellis2SpecSupport.write_safetensors(path, header, Bytes.new(16, 7_u8))

      inventory = ML::ThreeD::Trellis2::SafetensorsInventory.read(path)
      inventory.tensors.map(&.name).should eq(["bias", "weight"])
      inventory.tensors.find! { |tensor| tensor.name == "weight" }.n_elements.should eq(4)
      inventory.tensors.find! { |tensor| tensor.name == "bias" }.data_bytes.should eq(8)
      inventory.data_length.should eq(16)
    end
  end

  it "never reads payload bytes while building the inventory" do
    payload_bytes = 1_000_000_i64
    header = header_for([
      {"weight", "F16", [payload_bytes // 2], 0_i64, payload_bytes},
    ])
    prefix = IO::Memory.new
    prefix.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    prefix.write(header.to_slice)
    header_bytes = prefix.size.to_i64
    sentinel = PayloadSentinelIO.new(IO::Memory.new(prefix.to_slice), header_bytes)

    inventory = ML::ThreeD::Trellis2::SafetensorsInventory.read(
      sentinel,
      header_bytes + payload_bytes,
      "payload-sentinel.safetensors"
    )

    inventory.data_length.should eq(payload_bytes)
    sentinel.bytes_read.should eq(header_bytes)
  end

  it "accepts safetensors scalar and empty tensor shapes" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      path = File.join(dir, "scalar-and-empty.safetensors")
      header = header_for([
        {"scalar", "F16", [] of Int64, 0_i64, 2_i64},
        {"empty", "F16", [0_i64, 4_i64], 2_i64, 2_i64},
      ])
      Trellis2SpecSupport.write_safetensors(path, header, Bytes.new(2))

      inventory = ML::ThreeD::Trellis2::SafetensorsInventory.read(path)
      inventory.tensors.find! { |tensor| tensor.name == "scalar" }
        .n_elements.should eq(1)
      inventory.tensors.find! { |tensor| tensor.name == "empty" }
        .n_elements.should eq(0)
    end
  end

  it "accepts an empty safetensors container without payload bytes" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      path = File.join(dir, "empty.safetensors")
      Trellis2SpecSupport.write_safetensors(path, "{}", Bytes.empty)

      inventory = ML::ThreeD::Trellis2::SafetensorsInventory.read(path)
      inventory.tensors.should be_empty
      inventory.data_length.should eq(0)
    end
  end

  it "accepts string metadata including empty metadata keys" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      path = File.join(dir, "metadata-only.safetensors")
      Trellis2SpecSupport.write_safetensors(
        path,
        %({"__metadata__":{"":"value"}}),
        Bytes.empty
      )

      ML::ThreeD::Trellis2::SafetensorsInventory.read(path)
        .tensors.should be_empty
    end
  end

  it "represents an empty safetensors tensor name at the format boundary" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      path = File.join(dir, "empty-name.safetensors")
      header = header_for([
        {"", "F16", [1_i64], 0_i64, 2_i64},
      ])
      Trellis2SpecSupport.write_safetensors(path, header, Bytes.new(2))

      ML::ThreeD::Trellis2::SafetensorsInventory.read(path)
        .tensors.map(&.name).should eq([""])
    end
  end

  it "rejects a safetensors header that does not begin with an object byte" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      path = File.join(dir, "leading-space.safetensors")
      valid_header = header_for([
        {"weight", "F16", [1_i64], 0_i64, 2_i64},
      ])
      Trellis2SpecSupport.write_safetensors(
        path,
        " " + valid_header,
        Bytes.new(2)
      )

      expect_inventory_error(path, /header must begin with/)
    end
  end

  it "accepts only ASCII spaces as trailing safetensors header padding" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      padded = File.join(dir, "space-padded.safetensors")
      Trellis2SpecSupport.write_safetensors(padded, "{}    ", Bytes.empty)
      ML::ThreeD::Trellis2::SafetensorsInventory.read(padded)
        .tensors.should be_empty

      newline_padded = File.join(dir, "newline-padded.safetensors")
      Trellis2SpecSupport.write_safetensors(
        newline_padded,
        "{}\n",
        Bytes.empty
      )
      expect_inventory_error(newline_padded, /header padding/)
    end
  end

  it "rejects symlink tensor paths and symlinked ancestors" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      real_parent = File.join(dir, "real")
      Dir.mkdir(real_parent)
      real_path = File.join(real_parent, "weights.safetensors")
      header = header_for([
        {"weight", "F16", [1_i64], 0_i64, 2_i64},
      ])
      Trellis2SpecSupport.write_safetensors(real_path, header, Bytes.new(2))

      file_alias = File.join(dir, "weights-link.safetensors")
      File.symlink(real_path, file_alias)
      expect_inventory_error(file_alias, /symlink/)

      ancestor_alias = File.join(dir, "ancestor-link")
      File.symlink(real_parent, ancestor_alias)
      expect_inventory_error(
        File.join(ancestor_alias, "weights.safetensors"),
        /symlink/
      )
    end
  end

  it "rejects truncated and overlapping tensor ranges" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      truncated = File.join(dir, "truncated.safetensors")
      header = header_for([
        {"weight", "F16", [2_i64, 2_i64], 0_i64, 8_i64},
      ])
      Trellis2SpecSupport.write_safetensors(truncated, header, Bytes.new(4))
      expect_inventory_error(truncated, /outside payload|truncated/)

      overlapping = File.join(dir, "overlapping.safetensors")
      overlap_header = header_for([
        {"weight", "F16", [2_i64, 2_i64], 0_i64, 8_i64},
        {"other", "F16", [2_i64, 2_i64], 4_i64, 12_i64},
      ])
      Trellis2SpecSupport.write_safetensors(overlapping, overlap_header, Bytes.new(12))
      expect_inventory_error(overlapping, /overlapping tensor ranges/)
    end
  end

  it "rejects duplicate tensor keys and malformed tensor metadata" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      duplicate = File.join(dir, "duplicate.safetensors")
      tensor = %({"dtype":"F16","shape":[1],"data_offsets":[0,2]})
      header = %({"weight":#{tensor},"weight":#{tensor}})
      Trellis2SpecSupport.write_safetensors(duplicate, header, Bytes.new(2))
      expect_inventory_error(duplicate, /duplicate JSON key.*weight/)

      unknown_key = File.join(dir, "unknown-key.safetensors")
      bad_header = %({"weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2],"surprise":true}})
      Trellis2SpecSupport.write_safetensors(unknown_key, bad_header, Bytes.new(2))
      expect_inventory_error(unknown_key, /unknown tensor metadata key.*surprise/)
    end
  end

  it "cross-checks file hashes and the exact expected tensor inventory" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      stages_dir = File.join(dir, "stages")
      configs_dir = File.join(dir, "configs")
      Dir.mkdir(stages_dir)
      Dir.mkdir(configs_dir)
      File.write(File.join(configs_dir, "dino_v3.json"), "{}")
      config_path = File.join(configs_dir, "dino_v3.json")

      model_path = File.join(stages_dir, "dino_v3.safetensors")
      header = header_for([
        {"model.weight", "F16", [2_i64, 2_i64], 0_i64, 8_i64},
      ])
      Trellis2SpecSupport.write_safetensors(model_path, header, Bytes.new(8, 1_u8))
      file_sha = Digest::SHA256.hexdigest(File.read(model_path).to_slice)

      manifest_json = Trellis2SpecSupport.sealed_manifest_json(
        file_byte_length: File.size(model_path),
        file_sha256: file_sha,
        config_byte_length: File.size(config_path),
        config_sha256: Digest::SHA256.hexdigest(File.read(config_path).to_slice)
      )
      File.write(File.join(dir, "manifest.json"), manifest_json)

      manifest = ML::ThreeD::Trellis2::Manifest.load(dir)
      manifest.validate_pack!(dir)

      config_alias = Trellis2SpecSupport.mutate(manifest_json) do |root|
        stage = root["stages"].as_a.first.as_h
        stage["config_path"] = JSON::Any.new("stages/dino_v3.safetensors")
        stage["config_sha256"] = JSON::Any.new(file_sha)
      end
      expect_raises(
        ML::ThreeD::Trellis2::ManifestError,
        /config.*(?:role|JSON)|weights.*config/
      ) do
        aliased = ML::ThreeD::Trellis2::Manifest.parse(
          Trellis2SpecSupport.seal_manifest_json(config_alias)
        )
        aliased.validate_pack!(dir)
      end

      File.open(model_path, "ab") { |io| io.write_byte(0_u8) }
      expect_raises(ML::ThreeD::Trellis2::ManifestError, /byte length|SHA-256/) do
        ML::ThreeD::Trellis2::Manifest.load(dir)
      end
    end
  end

  it "rejects missing and unexpected tensors relative to the manifest" do
    Trellis2SpecSupport.with_temp_dir do |dir|
      stages_dir = File.join(dir, "stages")
      configs_dir = File.join(dir, "configs")
      Dir.mkdir(stages_dir)
      Dir.mkdir(configs_dir)
      File.write(File.join(configs_dir, "dino_v3.json"), "{}")
      config_path = File.join(configs_dir, "dino_v3.json")

      model_path = File.join(stages_dir, "dino_v3.safetensors")
      header = header_for([
        {"other.weight", "F16", [2_i64, 2_i64], 0_i64, 8_i64},
      ])
      Trellis2SpecSupport.write_safetensors(model_path, header, Bytes.new(8))
      file_sha = Digest::SHA256.hexdigest(File.read(model_path).to_slice)

      manifest_json = Trellis2SpecSupport.sealed_manifest_json(
        file_byte_length: File.size(model_path),
        file_sha256: file_sha,
        config_byte_length: File.size(config_path),
        config_sha256: Digest::SHA256.hexdigest(File.read(config_path).to_slice)
      )
      File.write(File.join(dir, "manifest.json"), manifest_json)
      manifest = ML::ThreeD::Trellis2::Manifest.parse(manifest_json)

      expect_raises(ML::ThreeD::Trellis2::ManifestError, /missing tensor.*model.weight/) do
        manifest.validate_pack!(dir)
      end
      expect_raises(ML::ThreeD::Trellis2::ManifestError, /unexpected tensor.*other.weight/) do
        manifest.validate_pack!(dir)
      end
    end
  end
end
