# Bounded dense CPU storage for floating element bit patterns.
#
# This type admits storage only. It does not imply ML::Tensor integration,
# floating arithmetic, autograd, device transfer, or Metal execution.

require "./dtype"

module ML
  class FloatingStorageError < Exception
  end

  class FloatingStorage
    # T2N2a admits only small CPU oracle values, not production weight tensors.
    # Keeping this equal to the T2N1 RawTensor limit prevents one valid-looking
    # request from creating multi-gigabyte process pressure.
    MAX_BYTES = 64_i32 * 1024 * 1024

    getter dtype : DType
    getter numel : Int32

    @bytes : Bytes

    private def initialize(
      @numel : Int32,
      @dtype : DType,
      @bytes : Bytes,
    )
    end

    def self.zeros(numel : Int32, dtype : DType) : FloatingStorage
      byte_size = checked_byte_size(numel, dtype)
      new(numel, dtype, Bytes.new(byte_size, 0_u8))
    end

    def self.from_bytes(
      bytes : Bytes,
      numel : Int32,
      dtype : DType,
    ) : FloatingStorage
      expected = checked_byte_size(numel, dtype)
      unless bytes.size == expected
        raise FloatingStorageError.new(
          "storage byte length #{bytes.size} does not match expected #{expected}"
        )
      end
      new(numel, dtype, bytes.dup)
    end

    def byte_size : Int32
      @bytes.size
    end

    def to_bytes : Bytes
      @bytes.dup
    end

    def clone : FloatingStorage
      FloatingStorage.from_bytes(@bytes, @numel, @dtype)
    end

    def read_u16_bits(index : Int32) : UInt16
      ensure_u16_dtype!
      offset = checked_offset(index, 2)
      # The storage wire representation is canonical little-endian on every host.
      IO::ByteFormat::LittleEndian.decode(UInt16, @bytes[offset, 2])
    end

    def write_u16_bits(index : Int32, value : UInt16) : Nil
      ensure_u16_dtype!
      offset = checked_offset(index, 2)
      IO::ByteFormat::LittleEndian.encode(value, @bytes[offset, 2])
    end

    def read_f32(index : Int32) : Float32
      ensure_f32_dtype!
      offset = checked_offset(index, 4)
      IO::ByteFormat::LittleEndian.decode(Float32, @bytes[offset, 4])
    end

    def write_f32(index : Int32, value : Float32) : Nil
      ensure_f32_dtype!
      offset = checked_offset(index, 4)
      IO::ByteFormat::LittleEndian.encode(value, @bytes[offset, 4])
    end

    private def self.checked_byte_size(
      numel : Int32,
      dtype : DType,
    ) : Int32
      unless dtype.floating?
        raise FloatingStorageError.new(
          "floating storage requires a floating dtype, got #{dtype}"
        )
      end
      if numel < 0
        raise FloatingStorageError.new(
          "floating storage element count must be non-negative"
        )
      end

      total = numel.to_i64 * dtype.byte_size
      if total > Int32::MAX
        raise FloatingStorageError.new(
          "floating storage byte size overflow: #{total}"
        )
      end
      if total > MAX_BYTES
        raise FloatingStorageError.new(
          "floating storage byte limit #{MAX_BYTES} exceeded: #{total}"
        )
      end
      total.to_i32
    end

    private def checked_offset(index : Int32, element_bytes : Int32) : Int32
      unless 0 <= index < @numel
        raise IndexError.new(
          "floating storage index #{index} is outside 0...#{@numel}"
        )
      end
      index * element_bytes
    end

    private def ensure_u16_dtype! : Nil
      unless @dtype.f16? || @dtype.bf16?
        raise FloatingStorageError.new(
          "u16 bit access requires F16 or BF16 storage, got #{@dtype}"
        )
      end
    end

    private def ensure_f32_dtype! : Nil
      unless @dtype.f32?
        raise FloatingStorageError.new(
          "Float32 access requires F32 storage, got #{@dtype}"
        )
      end
    end
  end
end
