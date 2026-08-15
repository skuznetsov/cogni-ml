require "./qwen_qbit_gaussian_codec"

module ML::GGUF
  # Minimal revision-zero ClickHouse Native block writer for prepared p7 state
  # tiles. It writes QBit's nested FixedString streams directly in plane-major
  # order, avoiding ClickHouse's Array -> QBit transpose on insertion.
  #
  # This is a transport primitive, not a TCP client: packet framing,
  # compression, authentication, retries, and cache admission live outside this
  # bounded experimental surface.
  module QwenQBitNativeWriter
    extend self

    record Record,
      cache_id : UInt64,
      layer : Int32,
      kind : UInt8,
      encoded : QwenQBitGaussianCodec::Encoded

    COLUMN_COUNT = 8_u64
    P7_PRECISION  = 7

    def encode(records : Array(Record)) : Bytes
      raise ArgumentError.new("QBit Native batch must not be empty") if records.empty?

      block_size = records.first.encoded.block_size
      raise ArgumentError.new("QBit Native block size exceeds UInt16 value_count") if block_size > UInt16::MAX
      records.each do |record|
        encoded = record.encoded
        QwenQBitGaussianCodec.validate(encoded)
        raise ArgumentError.new("QBit Native writer requires p7 precision") unless encoded.precision == P7_PRECISION
        raise ArgumentError.new("mixed QBit Native block sizes") unless encoded.block_size == block_size
      end

      row_count = records.sum(0_i64) { |record| QwenQBitGaussianCodec.tile_count(record.encoded).to_i64 }
      raise ArgumentError.new("QBit Native batch has no tiles") if row_count == 0

      io = IO::Memory.new
      write_varuint(io, COLUMN_COUNT)
      write_varuint(io, row_count.to_u64)

      write_column_header(io, "cache_id", "UInt64")
      each_tile(records) { |record, _tile| io.write_bytes(record.cache_id, IO::ByteFormat::LittleEndian) }

      write_column_header(io, "layer", "Int32")
      each_tile(records) { |record, _tile| io.write_bytes(record.layer, IO::ByteFormat::LittleEndian) }

      write_column_header(io, "kind", "UInt8")
      each_tile(records) { |record, _tile| io.write_byte(record.kind) }

      write_column_header(io, "tile", "UInt32")
      each_tile(records) { |_record, tile| io.write_bytes(tile.to_u32, IO::ByteFormat::LittleEndian) }

      write_column_header(io, "value_count", "UInt16")
      each_tile(records) do |record, tile|
        count = QwenQBitGaussianCodec.tile_value_count(record.encoded, tile)
        io.write_bytes(count.to_u16, IO::ByteFormat::LittleEndian)
      end

      write_column_header(io, "mean", "Float32")
      each_tile(records) do |record, tile|
        mean, _sigma = QwenQBitGaussianCodec.tile_moments(record.encoded, tile)
        io.write_bytes(mean, IO::ByteFormat::LittleEndian)
      end

      write_column_header(io, "sigma", "Float32")
      each_tile(records) do |record, tile|
        _mean, sigma = QwenQBitGaussianCodec.tile_moments(record.encoded, tile)
        io.write_bytes(sigma, IO::ByteFormat::LittleEndian)
      end

      write_column_header(io, "codes", "QBit(Int8, #{block_size})")
      P7_PRECISION.times do |plane|
        each_tile(records) do |record, tile|
          io.write(QwenQBitGaussianCodec.tile_plane(record.encoded, tile, plane))
        end
      end
      # QBit(Int8, N) has eight physical streams. P7 intentionally leaves the
      # raw-code LSB zero; protocol compression makes this stream negligible.
      zero_plane = Bytes.new(block_size // 8, 0_u8)
      each_tile(records) { |_record, _tile| io.write(zero_plane) }

      io.to_slice
    end

    private def each_tile(records : Array(Record), &block : Record, Int32 ->) : Nil
      records.each do |record|
        QwenQBitGaussianCodec.tile_count(record.encoded).times do |tile|
          yield record, tile
        end
      end
    end

    private def write_column_header(io : IO, name : String, type : String) : Nil
      write_string(io, name)
      write_string(io, type)
    end

    private def write_string(io : IO, value : String) : Nil
      bytes = value.to_slice
      write_varuint(io, bytes.size.to_u64)
      io.write(bytes)
    end

    private def write_varuint(io : IO, value : UInt64) : Nil
      remaining = value
      loop do
        byte = (remaining & 0x7f_u64).to_u8
        remaining >>= 7
        if remaining == 0
          io.write_byte(byte)
          return
        end
        io.write_byte(byte | 0x80_u8)
      end
    end
  end
end
