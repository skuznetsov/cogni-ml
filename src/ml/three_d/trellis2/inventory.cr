require "json"

require "./strict_json"

module ML::ThreeD::Trellis2
  class InventoryError < Exception
  end

  struct InventoryTensor
    getter name : String
    getter dtype : String
    getter shape : Array(Int64)
    getter data_start : Int64
    getter data_end : Int64

    def initialize(@name, @dtype, @shape, @data_start, @data_end)
    end

    def n_elements : Int64
      checked_elements(@shape)
    end

    def data_bytes : Int64
      @data_end - @data_start
    end

    def bytes_per_element : Int64
      case @dtype
      when "BF16", "F16" then 2_i64
      when "F32"         then 4_i64
      else
        raise InventoryError.new("unsupported dtype #{@dtype.inspect}")
      end
    end

    def validate_size! : Nil
      elements = n_elements
      if elements > Int64::MAX // bytes_per_element
        raise InventoryError.new("tensor #{@name.inspect} byte size overflow")
      end
      expected = elements * bytes_per_element
      unless data_bytes == expected
        raise InventoryError.new(
          "tensor #{@name.inspect} byte size #{data_bytes} != expected #{expected}"
        )
      end
    end

    private def checked_elements(shape : Array(Int64)) : Int64
      shape.each do |dimension|
        if dimension < 0
          raise InventoryError.new(
            "tensor #{@name.inspect} has negative shape dimension #{dimension}"
          )
        end
      end
      return 0_i64 if shape.any?(&.zero?)

      shape.reduce(1_i64) do |product, dimension|
        if product > Int64::MAX // dimension
          raise InventoryError.new("tensor #{@name.inspect} shape overflow")
        end
        product * dimension
      end
    end
  end

  class SafetensorsInventory
    MAX_HEADER_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter path : String
    getter header_length : Int64
    getter data_offset : Int64
    getter data_length : Int64
    getter tensors : Array(InventoryTensor)

    private def initialize(
      @path,
      @header_length,
      @data_offset,
      @data_length,
      @tensors,
    )
    end

    def self.read(path : String) : SafetensorsInventory
      normalized = validated_tensor_path(path)
      total_size = File.size(normalized)
      File.open(normalized, "rb") do |io|
        read(io, total_size, normalized)
      end
    rescue ex : InventoryError
      raise ex
    rescue ex : File::Error | IO::Error
      raise InventoryError.new("cannot read safetensors #{path.inspect}: #{ex.message}")
    end

    def self.read(
      io : IO,
      total_size : Int64,
      path : String,
    ) : SafetensorsInventory
      if total_size < 8
        raise InventoryError.new("safetensors file is truncated before header length")
      end

      header_length = 0_i64
      header = Bytes.empty
      length_bytes = Bytes.new(8)
      io.read_fully(length_bytes)
      raw_length = IO::ByteFormat::LittleEndian.decode(UInt64, length_bytes)
      if raw_length == 0 || raw_length > Int64::MAX.to_u64
        raise InventoryError.new("invalid safetensors header length #{raw_length}")
      end
      header_length = raw_length.to_i64
      if header_length > MAX_HEADER_BYTES
        raise InventoryError.new(
          "safetensors header length #{header_length} exceeds #{MAX_HEADER_BYTES}"
        )
      end
      if header_length > total_size - 8
        raise InventoryError.new("safetensors file is truncated inside header")
      end

      header = Bytes.new(header_length.to_i32)
      io.read_fully(header)
      unless header[0] == '{'.ord
        raise InventoryError.new(
          "safetensors header must begin with an object byte"
        )
      end
      validate_header_padding!(header)

      data_offset = 8_i64 + header_length
      data_length = total_size - data_offset
      tensors = parse_header(String.new(header), data_length)
      new(path, header_length, data_offset, data_length, tensors)
    rescue ex : InventoryError
      raise ex
    rescue ex : IO::Error
      raise InventoryError.new("cannot read safetensors #{path.inspect}: #{ex.message}")
    end

    private def self.validated_tensor_path(path : String) : String
      normalized = Path.new(File.expand_path(path)).normalize.to_s
      current = "/"
      normalized.split('/', remove_empty: true).each do |component|
        current = File.join(current, component)
        if info = File.info?(current, follow_symlinks: false)
          if info.symlink?
            raise InventoryError.new(
              "safetensors path must not contain symlink component #{current.inspect}"
            )
          end
        end
      end
      info = File.info(normalized, follow_symlinks: false)
      unless info.file?
        raise InventoryError.new("safetensors path must be a regular file")
      end
      normalized
    rescue ex : InventoryError
      raise ex
    rescue ex : File::Error
      raise InventoryError.new(
        "cannot inspect safetensors #{path.inspect}: #{ex.message}"
      )
    end

    private def self.parse_header(
      source : String,
      data_length : Int64,
    ) : Array(InventoryTensor)
      root = begin
        StrictJSON.parse(source).as_h
      rescue ex : StrictJSONError
        raise InventoryError.new(ex.message)
      rescue
        raise InventoryError.new("safetensors header must be a JSON object")
      end

      tensors = [] of InventoryTensor
      root.each do |name, value|
        if name == "__metadata__"
          validate_metadata(value)
          next
        end
        tensors << parse_tensor(name, value)
      end
      validate_ranges!(tensors, data_length)
      tensors.sort_by!(&.name)
      tensors
    end

    private def self.validate_header_padding!(header : Bytes) : Nil
      index = header.size - 1
      while index >= 0 && header[index] == 0x20_u8
        index -= 1
      end
      unless index >= 0 && header[index] == '}'.ord
        raise InventoryError.new(
          "safetensors header padding must contain only ASCII spaces"
        )
      end
    end

    private def self.validate_metadata(value : JSON::Any) : Nil
      metadata = value.as_h
      metadata.each_value(&.as_s)
    rescue
      raise InventoryError.new("safetensors __metadata__ values must be strings")
    end

    private def self.parse_tensor(
      name : String,
      value : JSON::Any,
    ) : InventoryTensor
      object = value.as_h
      expect_exact_keys!(
        object,
        ["dtype", "shape", "data_offsets"],
        "tensor metadata"
      )

      dtype = object["dtype"].as_s
      unless {"BF16", "F16", "F32"}.includes?(dtype)
        raise InventoryError.new("unsupported dtype #{dtype.inspect} for #{name.inspect}")
      end

      shape = object["shape"].as_a.map(&.as_i64)
      offsets = object["data_offsets"].as_a.map(&.as_i64)
      unless offsets.size == 2
        raise InventoryError.new("tensor #{name.inspect} data_offsets must have 2 values")
      end
      start_offset = offsets[0]
      end_offset = offsets[1]
      if start_offset < 0 || end_offset < start_offset
        raise InventoryError.new("tensor #{name.inspect} has invalid data_offsets")
      end

      tensor = InventoryTensor.new(name, dtype, shape, start_offset, end_offset)
      tensor.validate_size!
      tensor
    rescue ex : InventoryError
      raise ex
    rescue
      raise InventoryError.new("invalid tensor metadata for #{name.inspect}")
    end

    private def self.expect_exact_keys!(
      object : Hash(String, JSON::Any),
      allowed : Array(String),
      context : String,
    ) : Nil
      object.each_key do |key|
        unless allowed.includes?(key)
          raise InventoryError.new("unknown #{context} key #{key.inspect}")
        end
      end
      allowed.each do |key|
        unless object.has_key?(key)
          raise InventoryError.new("missing #{context} key #{key.inspect}")
        end
      end
    end

    private def self.validate_ranges!(
      tensors : Array(InventoryTensor),
      data_length : Int64,
    ) : Nil
      ordered = tensors.sort_by { |tensor| {tensor.data_start, tensor.data_end} }
      expected_start = 0_i64

      ordered.each do |tensor|
        if tensor.data_end > data_length
          raise InventoryError.new(
            "tensor #{tensor.name.inspect} is outside payload; file is truncated"
          )
        end
        if tensor.data_start < expected_start
          raise InventoryError.new("overlapping tensor ranges at #{tensor.name.inspect}")
        end
        if tensor.data_start > expected_start
          raise InventoryError.new("gap before tensor range #{tensor.name.inspect}")
        end
        expected_start = tensor.data_end
      end

      if expected_start != data_length
        raise InventoryError.new(
          "unreferenced safetensors payload bytes: #{data_length - expected_start}"
        )
      end
    end
  end
end
