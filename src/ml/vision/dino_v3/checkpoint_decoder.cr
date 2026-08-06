require "./checkpoint_digest"
require "./semantic_inventory"

module ML::Vision::DinoV3
  class CheckpointDecoderError < CheckpointPayloadError
  end

  struct DecodedF32Tensor
    getter role : String
    getter name : String

    @shape : Array(Int64)
    @values : Array(Float32)

    def initialize(
      @role : String,
      name : String,
      shape : Array(Int64),
      values : Array(Float32),
    )
      @name = name.dup
      @shape = shape.dup
      @values = values.dup
    end

    def shape : Array(Int64)
      @shape.dup
    end

    def name : String
      @name.dup
    end

    def values : Array(Float32)
      @values.dup
    end
  end

  class CheckpointF32Decoder
    MAX_RESIDENT_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter max_resident_bytes : Int64

    @inventory : CheckpointInventory
    @semantic : SemanticInventory
    @digest : CheckpointDigestReceipt
    @reader : CheckpointPayloadReader

    def initialize(
      @inventory : CheckpointInventory,
      *,
      @semantic : SemanticInventory,
      @digest : CheckpointDigestReceipt,
      max_resident_bytes : Int64 = MAX_RESIDENT_BYTES,
    )
      unless 0_i64 < max_resident_bytes <= MAX_RESIDENT_BYTES
        raise CheckpointDecoderError.new(
          "checkpoint decoder resident byte budget must be in 1..#{MAX_RESIDENT_BYTES}"
        )
      end
      unless @digest.matches?(@inventory)
        raise CheckpointDecoderError.new(
          "checkpoint digest receipt does not match the inspected inventory"
        )
      end
      @max_resident_bytes = max_resident_bytes
      @reader = CheckpointPayloadReader.new(
        @inventory,
        max_read_bytes: max_resident_bytes
      )
    end

    def decode(role : String) : DecodedF32Tensor
      tensor = @semantic.tensor_for(role)
      unless tensor.dtype == "F32"
        raise CheckpointDecoderError.new(
          "DINOv3 semantic role #{role.inspect} is not F32"
        )
      end

      raw_bytes = tensor.data_bytes
      unless raw_bytes >= 0_i64 && raw_bytes % 4_i64 == 0_i64
        raise CheckpointDecoderError.new(
          "DINOv3 F32 tensor #{tensor.name.inspect} has an invalid byte length"
        )
      end
      if raw_bytes > @max_resident_bytes // 2_i64
        raise CheckpointDecoderError.new(
          "DINOv3 F32 tensor #{tensor.name.inspect} exceeds the resident byte budget"
        )
      end
      unless @digest.matches?(@inventory)
        raise CheckpointDecoderError.new(
          "checkpoint digest receipt no longer matches the inspected inventory"
        )
      end

      bytes = @reader.read_tensor(tensor, max_bytes: @max_resident_bytes // 2_i64)
      values = Array(Float32).new((raw_bytes // 4_i64).to_i32)
      offset = 0
      while offset < bytes.size
        values << IO::ByteFormat::LittleEndian.decode(Float32, bytes[offset, 4])
        offset += 4
      end
      DecodedF32Tensor.new(role, tensor.name, tensor.shape, values)
    rescue ex : CheckpointPayloadError
      raise ex
    rescue ex : SemanticInventoryError
      raise ex
    end
  end
end
