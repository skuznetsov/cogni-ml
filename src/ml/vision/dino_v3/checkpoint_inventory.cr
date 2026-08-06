require "json"
require "set"

require "../../three_d/trellis2/strict_json"
require "./checkpoint_manifest"

module ML::Vision::DinoV3
  # Header inspection proves only that a local file has a bounded, structurally
  # valid safetensors header. It does not hash or decode the payload.
  class CheckpointInventoryError < CheckpointManifestError
  end

  struct CheckpointTensor
    getter data_start : Int64
    getter data_end : Int64

    @name : String
    @dtype : String
    @shape : Array(Int64)

    def initialize(
      name : String,
      dtype : String,
      shape : Array(Int64),
      @data_start : Int64,
      @data_end : Int64,
    )
      @name = name.dup
      @dtype = dtype.dup
      @shape = shape.dup
    end

    def shape : Array(Int64)
      @shape.dup
    end

    def name : String
      @name.dup
    end

    def dtype : String
      @dtype.dup
    end

    def data_bytes : Int64
      @data_end - @data_start
    end
  end

  class CheckpointInventory
    MAX_HEADER_BYTE_LENGTH = 16_i64 * 1024 * 1024

    SUPPORTED_DTYPES = Set{
      "BOOL",
      "U8",
      "I8",
      "U16",
      "I16",
      "U32",
      "I32",
      "U64",
      "I64",
      "F8_E4M3",
      "F8_E5M2",
      "BF16",
      "F16",
      "F32",
      "F64",
      "C64",
      "C128",
    }

    getter file_byte_length : Int64
    getter header_byte_length : Int64
    getter data_offset : Int64

    @path : String
    @metadata : Hash(String, String)
    @tensors : Array(CheckpointTensor)

    private def initialize(
      path : String,
      @file_byte_length : Int64,
      @header_byte_length : Int64,
      @data_offset : Int64,
      metadata : Hash(String, String),
      tensors : Array(CheckpointTensor),
    )
      @path = path.dup
      @metadata = metadata.dup
      @tensors = tensors.dup
    end

    def metadata : Hash(String, String)
      @metadata.each_with_object({} of String => String) do |(key, value), copy|
        copy[key.dup] = value.dup
      end
    end

    def tensors : Array(CheckpointTensor)
      @tensors.dup
    end

    def path : String
      @path.dup
    end

    def self.load(
      path : String,
      *,
      manifest : CheckpointManifest,
    ) : CheckpointInventory
      validate_manifest!(manifest)
      unless File.basename(path) == manifest.weights_path
        raise CheckpointInventoryError.new(
          "materialized checkpoint basename does not match the manifest"
        )
      end

      file_byte_length = File.size(path)
      unless file_byte_length == manifest.weights_byte_length
        raise CheckpointInventoryError.new(
          "materialized checkpoint file size does not match the manifest"
        )
      end

      File.open(path, "rb") do |file|
        header_byte_length, data_offset, metadata, tensors = parse_header(
          file,
          file_byte_length
        )
        new(
          path,
          file_byte_length,
          header_byte_length,
          data_offset,
          metadata,
          tensors
        )
      end
    rescue ex : CheckpointInventoryError
      raise ex
    rescue ex : File::Error
      raise CheckpointInventoryError.new("cannot load materialized checkpoint: #{ex.message}")
    rescue ex : IO::EOFError
      raise CheckpointInventoryError.new("materialized checkpoint header is truncated")
    rescue ex : ML::ThreeD::Trellis2::StrictJSONError
      raise CheckpointInventoryError.new("invalid safetensors header JSON: #{ex.message}")
    rescue ex : JSON::ParseException
      raise CheckpointInventoryError.new("invalid safetensors header JSON: #{ex.message}")
    rescue ex : TypeCastError
      raise CheckpointInventoryError.new("invalid safetensors header field")
    end

    private def self.validate_manifest!(manifest : CheckpointManifest) : Nil
      unless manifest.model == ConfigCertificate::PINNED_SOURCE_MODEL &&
             manifest.revision == ConfigCertificate::PINNED_SOURCE_REVISION &&
             manifest.access_mode == ConfigCertificate::PINNED_ACCESS_MODE &&
             manifest.license_name == ConfigCertificate::PINNED_LICENSE_NAME &&
             manifest.license_link == ConfigCertificate::PINNED_LICENSE_LINK &&
             manifest.config_path == CheckpointManifest::PINNED_CONFIG_PATH &&
             manifest.config_byte_length == CheckpointManifest::PINNED_CONFIG_BYTE_LENGTH &&
             manifest.config_sha256 == ConfigCertificate::PINNED_CONFIG_SHA256 &&
             manifest.weights_path == CheckpointManifest::PINNED_WEIGHTS_PATH &&
             manifest.weights_format == CheckpointManifest::PINNED_WEIGHTS_FORMAT &&
             manifest.weights_byte_length == CheckpointManifest::PINNED_WEIGHTS_BYTE_LENGTH &&
             manifest.weights_sha256 == CheckpointManifest::PINNED_WEIGHTS_SHA256
        raise CheckpointInventoryError.new(
          "checkpoint inventory requires the pinned DINOv3 checkpoint manifest"
        )
      end
    end

    private def self.parse_header(
      file : File,
      file_byte_length : Int64,
    ) : Tuple(Int64, Int64, Hash(String, String), Array(CheckpointTensor))
      length_bytes = Bytes.new(8)
      file.read_fully(length_bytes)
      header_byte_length_u64 = IO::ByteFormat::LittleEndian.decode(
        UInt64,
        length_bytes
      )
      if header_byte_length_u64 == 0_u64 ||
         header_byte_length_u64 > MAX_HEADER_BYTE_LENGTH.to_u64
        raise CheckpointInventoryError.new("safetensors header length is outside the admitted bound")
      end

      header_byte_length = header_byte_length_u64.to_i64
      data_offset = 8_i64 + header_byte_length
      if data_offset > file_byte_length
        raise CheckpointInventoryError.new("safetensors header exceeds the file size")
      end

      header = Bytes.new(header_byte_length.to_i32)
      file.read_fully(header)
      root = ML::ThreeD::Trellis2::StrictJSON.parse(String.new(header)).as_h
      data_byte_length = file_byte_length - data_offset
      metadata = {} of String => String
      tensors = [] of CheckpointTensor

      root.each do |name, value|
        if name == "__metadata__"
          parse_metadata(value, metadata)
        else
          tensors << parse_tensor(name, value, data_byte_length)
        end
      end

      if tensors.empty?
        raise CheckpointInventoryError.new("safetensors header has no tensor entries")
      end
      reject_overlapping_ranges!(tensors)
      {header_byte_length, data_offset, metadata, tensors}
    end

    private def self.parse_metadata(
      value : JSON::Any,
      metadata : Hash(String, String),
    ) : Nil
      object = value.as_h
      object.each do |key, item|
        metadata[key] = item.as_s.dup
      end
    rescue ex : TypeCastError
      raise CheckpointInventoryError.new("safetensors metadata must contain only strings")
    end

    private def self.parse_tensor(
      name : String,
      value : JSON::Any,
      data_byte_length : Int64,
    ) : CheckpointTensor
      object = value.as_h
      expect_exact_keys!(object, ["dtype", "shape", "data_offsets"], name)

      dtype = object["dtype"].as_s
      unless SUPPORTED_DTYPES.includes?(dtype)
        raise CheckpointInventoryError.new("unsupported safetensors dtype for #{name.inspect}")
      end

      shape = object["shape"].as_a.map do |item|
        dimension = item.as_i64
        unless dimension >= 0
          raise CheckpointInventoryError.new("negative tensor dimension for #{name.inspect}")
        end
        dimension
      end

      expected_bytes = expected_data_bytes(dtype, shape, name)

      offsets = object["data_offsets"].as_a
      unless offsets.size == 2
        raise CheckpointInventoryError.new("tensor offsets must have two values for #{name.inspect}")
      end
      data_start = nonnegative_integer(offsets[0], "tensor data start")
      data_end = nonnegative_integer(offsets[1], "tensor data end")
      unless data_start <= data_end && data_end <= data_byte_length
        raise CheckpointInventoryError.new("tensor offsets exceed the file data region for #{name.inspect}")
      end
      unless data_end - data_start == expected_bytes
        raise CheckpointInventoryError.new("tensor byte range does not match dtype and shape for #{name.inspect}")
      end

      CheckpointTensor.new(name, dtype, shape, data_start, data_end)
    rescue ex : CheckpointInventoryError
      raise ex
    rescue ex : TypeCastError
      raise CheckpointInventoryError.new("invalid safetensors tensor entry for #{name.inspect}")
    end

    private def self.expected_data_bytes(
      dtype : String,
      shape : Array(Int64),
      name : String,
    ) : Int64
      elements = 1_i64
      shape.each do |dimension|
        if dimension != 0_i64 && elements > Int64::MAX // dimension
          raise CheckpointInventoryError.new("tensor shape overflows byte accounting for #{name.inspect}")
        end
        elements *= dimension
      end

      bytes_per_element = case dtype
                          when "BOOL", "U8", "I8", "F8_E4M3", "F8_E5M2" then 1_i64
                          when "U16", "I16", "BF16", "F16"              then 2_i64
                          when "U32", "I32", "F32"                      then 4_i64
                          when "U64", "I64", "F64", "C64"               then 8_i64
                          when "C128"                                   then 16_i64
                          else
                            raise CheckpointInventoryError.new("unsupported safetensors dtype for #{name.inspect}")
                          end
      if bytes_per_element != 0_i64 && elements > Int64::MAX // bytes_per_element
        raise CheckpointInventoryError.new("tensor shape overflows byte accounting for #{name.inspect}")
      end
      elements * bytes_per_element
    end

    private def self.reject_overlapping_ranges!(tensors : Array(CheckpointTensor)) : Nil
      ordered = tensors.sort_by(&.data_start)
      (1...ordered.size).each do |index|
        previous = ordered[index - 1]
        current = ordered[index]
        if current.data_start < previous.data_end
          raise CheckpointInventoryError.new("safetensors tensor data ranges overlap")
        end
      end
    end

    private def self.nonnegative_integer(value : JSON::Any, context : String) : Int64
      integer = value.as_i64
      unless integer >= 0
        raise CheckpointInventoryError.new("#{context} must be non-negative")
      end
      integer
    rescue ex : CheckpointInventoryError
      raise ex
    rescue ex : TypeCastError
      raise CheckpointInventoryError.new("#{context} must be an integer")
    end

    private def self.expect_exact_keys!(
      object : Hash(String, JSON::Any),
      allowed : Array(String),
      context : String,
    ) : Nil
      object.each_key do |key|
        unless allowed.includes?(key)
          raise CheckpointInventoryError.new(
            "unknown safetensors tensor key #{key.inspect} for #{context.inspect}"
          )
        end
      end
      allowed.each do |key|
        unless object.has_key?(key)
          raise CheckpointInventoryError.new(
            "missing safetensors tensor key #{key.inspect} for #{context.inspect}"
          )
        end
      end
    end
  end
end
