require "digest/sha256"
require "json"

module Trellis2SpecSupport
  extend self

  PACK_ID_DOMAIN = "cogni-ml/trellis2/pack-id/v1"
  SHA_A          = "a" * 64
  SHA_B          = "b" * 64
  SHA_C          = "c" * 64
  REV_A          = "1" * 40
  REV_B          = "2" * 40

  def manifest_json(
    pack_id : String = "0" * 64,
    file_byte_length : Int64 = 128_i64,
    file_sha256 : String = SHA_C,
    config_byte_length : Int64 = 2_i64,
    config_sha256 : String = SHA_B,
    tensors : Array(Hash(String, JSON::Any)) = [tensor_json],
  ) : String
    JSON.build do |json|
      json.object do
        json.field "schema_version", 1
        json.field "converter_version", "0.1.0"
        json.field "converter_config_sha256", SHA_A
        json.field "source" do
          json.object do
            json.field "repository", "https://github.com/microsoft/TRELLIS.2"
            json.field "revision", REV_A
            json.field "license", "MIT"
          end
        end
        json.field "model" do
          json.object do
            json.field "repository", "microsoft/TRELLIS.2-4B"
            json.field "revision", REV_B
            json.field "license", "MIT"
          end
        end
        json.field "external_dependencies" do
          json.array { }
        end
        json.field "execution_order" do
          json.array { json.string "dino_v3" }
        end
        json.field "files" do
          json.array do
            json.object do
              json.field "path", "stages/dino_v3.safetensors"
              json.field "role", "weights"
              json.field "byte_length", file_byte_length
              json.field "sha256", file_sha256
              json.field "license", "MIT"
            end
            json.object do
              json.field "path", "configs/dino_v3.json"
              json.field "role", "config"
              json.field "byte_length", config_byte_length
              json.field "sha256", config_sha256
              json.field "license", "MIT"
            end
          end
        end
        json.field "stages" do
          json.array do
            json.object do
              json.field "id", "dino_v3"
              json.field "order", 0
              json.field "config_path", "configs/dino_v3.json"
              json.field "config_sha256", config_sha256
              json.field "tensors" do
                json.array do
                  tensors.each { |tensor| tensor.to_json(json) }
                end
              end
            end
          end
        end
        json.field "pack_id", pack_id
      end
    end
  end

  def tensor_json(
    source_name : String = "model.weight",
    destination_name : String = "dino_v3.model.weight",
    file : String = "stages/dino_v3.safetensors",
    dtype : String = "F16",
    shape : Array(Int64) = [2_i64, 2_i64],
    data_offsets : Array(Int64) = [0_i64, 8_i64],
    byte_order : String = "little",
    layout : String = "identity",
  ) : Hash(String, JSON::Any)
    {
      "source_name"      => JSON::Any.new(source_name),
      "destination_name" => JSON::Any.new(destination_name),
      "file"             => JSON::Any.new(file),
      "dtype"            => JSON::Any.new(dtype),
      "shape"            => JSON::Any.new(shape.map { |v| JSON::Any.new(v) }),
      "data_offsets"     => JSON::Any.new(data_offsets.map { |v| JSON::Any.new(v) }),
      "byte_order"       => JSON::Any.new(byte_order),
      "layout"           => JSON::Any.new(layout),
    }
  end

  def sealed_manifest_json(**args) : String
    seal_manifest_json(manifest_json(**args))
  end

  def seal_manifest_json(draft : String) : String
    pack_id = ML::ThreeD::Trellis2::Manifest.compute_pack_id_for_draft(draft)
    root = JSON.parse(draft).as_h
    root["pack_id"] = JSON::Any.new(pack_id)
    root.to_json
  end

  def independently_framed_pack_id(
    manifest : ML::ThreeD::Trellis2::Manifest,
  ) : String
    bytes = IO::Memory.new
    write_frame(bytes, PACK_ID_DOMAIN)
    write_frame(bytes, manifest.canonical_identity_json)
    manifest.files.sort_by(&.path).each do |file|
      write_frame(bytes, file.path)
      bytes.write_bytes(file.byte_length.to_u64, IO::ByteFormat::LittleEndian)
      write_frame(bytes, file.sha256)
    end
    Digest::SHA256.hexdigest(bytes.to_slice)
  end

  def mutate(json : String, &block : Hash(String, JSON::Any) ->) : String
    root = JSON.parse(json).as_h
    yield root
    root.to_json
  end

  def write_safetensors(
    path : String,
    header : String,
    payload : Bytes,
  ) : Nil
    File.open(path, "wb") do |io|
      length = Bytes.new(8)
      IO::ByteFormat::LittleEndian.encode(header.bytesize.to_u64, length)
      io.write(length)
      io.write(header.to_slice)
      io.write(payload)
    end
  end

  def write_valid_pack(path : String) : Nil
    stages_dir = File.join(path, "stages")
    configs_dir = File.join(path, "configs")
    Dir.mkdir_p(stages_dir)
    Dir.mkdir_p(configs_dir)

    config_path = File.join(configs_dir, "dino_v3.json")
    File.write(config_path, "{}")

    weights_path = File.join(stages_dir, "dino_v3.safetensors")
    header = %({"model.weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}})
    write_safetensors(weights_path, header, Bytes.new(8, 1_u8))

    manifest = sealed_manifest_json(
      file_byte_length: File.size(weights_path),
      file_sha256: Digest::SHA256.new.file(weights_path).hexfinal,
      config_byte_length: File.size(config_path),
      config_sha256: Digest::SHA256.new.file(config_path).hexfinal
    )
    File.write(File.join(path, "manifest.json"), manifest)
  end

  def with_temp_dir(&)
    path = File.tempname(
      "cogni-trellis2-spec",
      nil,
      dir: File.realpath(Dir.tempdir)
    )
    Dir.mkdir(path)
    begin
      yield path
    ensure
      FileUtils.rm_rf(path)
    end
  end

  private def write_frame(io : IO, value : String) : Nil
    io.write_bytes(value.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    io.write(value.to_slice)
  end
end
