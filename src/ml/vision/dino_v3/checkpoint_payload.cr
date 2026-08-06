require "./checkpoint_inventory"

module ML::Vision::DinoV3
  # A bounded local payload reader. It reads only one already-registered
  # safetensors range and never hashes, decodes, or maps the checkpoint.
  class CheckpointPayloadError < CheckpointInventoryError
  end

  class CheckpointPayloadReader
    MAX_READ_BYTES = 64_i64 * 1024_i64 * 1024_i64

    getter max_read_bytes : Int64

    @inventory : CheckpointInventory

    def initialize(
      @inventory : CheckpointInventory,
      *,
      max_read_bytes : Int64 = MAX_READ_BYTES,
    )
      unless 0_i64 < max_read_bytes <= MAX_READ_BYTES
        raise CheckpointPayloadError.new(
          "checkpoint payload byte budget must be in 1..#{MAX_READ_BYTES}"
        )
      end
      @max_read_bytes = max_read_bytes
    end

    def read_tensor(name : String, *, max_bytes : Int64? = nil) : Bytes
      tensor = @inventory.tensors.find { |candidate| candidate.name == name }
      unless tensor
        raise CheckpointPayloadError.new(
          "unknown DINOv3 checkpoint tensor #{name.inspect}"
        )
      end
      read_tensor(tensor, max_bytes: max_bytes)
    end

    def read_tensor(
      tensor : CheckpointTensor,
      *,
      max_bytes : Int64? = nil,
    ) : Bytes
      owned = @inventory.tensors.find { |candidate| same_descriptor?(candidate, tensor) }
      unless owned
        raise CheckpointPayloadError.new(
          "checkpoint tensor descriptor does not belong to the inspected inventory"
        )
      end

      limit = max_bytes || @max_read_bytes
      unless 0_i64 < limit <= @max_read_bytes
        raise CheckpointPayloadError.new(
          "checkpoint payload byte budget must be in 1..#{@max_read_bytes}"
        )
      end
      if owned.data_bytes > limit
        raise CheckpointPayloadError.new(
          "checkpoint tensor #{owned.name.inspect} requires #{owned.data_bytes} " \
          "bytes, exceeding the #{limit}-byte budget"
        )
      end
      unless File.size(@inventory.path) == @inventory.file_byte_length
        raise CheckpointPayloadError.new(
          "checkpoint file changed after inventory"
        )
      end

      absolute_offset = @inventory.data_offset + owned.data_start
      bytes = Bytes.new(owned.data_bytes.to_i32)
      File.open(@inventory.path, "rb") do |file|
        file.seek(absolute_offset)
        file.read_fully(bytes)
      end
      bytes
    rescue ex : CheckpointPayloadError
      raise ex
    rescue ex : File::Error
      raise CheckpointPayloadError.new(
        "cannot read checkpoint tensor payload: #{ex.message}"
      )
    rescue ex : IO::EOFError
      raise CheckpointPayloadError.new("checkpoint tensor payload is truncated")
    end

    private def same_descriptor?(
      left : CheckpointTensor,
      right : CheckpointTensor,
    ) : Bool
      left.name == right.name &&
        left.dtype == right.dtype &&
        left.shape == right.shape &&
        left.data_start == right.data_start &&
        left.data_end == right.data_end
    end
  end
end
