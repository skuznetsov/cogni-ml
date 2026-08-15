module ML::GGUF
  # Strict parser for revision-zero Native blocks emitted by QwenQBitNativeWriter
  # or an equivalent ordered ClickHouse SELECT ... FORMAT Native. It exposes
  # zero-copy per-block column offsets and validates logical records across
  # response-block boundaries for the Metal restore path.
  module QwenQBitNativeBlock
    extend self

    record RecordSpan,
      cache_id : UInt64,
      layer : Int32,
      kind : UInt8,
      row_start : Int32,
      tile_count : Int32,
      value_count : Int32

    record RecordChunk,
      cache_id : UInt64,
      layer : Int32,
      kind : UInt8,
      row_start : Int32,
      tile_start : Int32,
      tile_count : Int32,
      value_count : Int32,
      last_value_count : Int32

    record StreamChunk,
      block_index : Int32,
      row_start : Int32,
      tile_start : Int32,
      tile_count : Int32,
      value_start : Int32,
      value_count : Int32

    record StreamRecordSpan,
      cache_id : UInt64,
      layer : Int32,
      kind : UInt8,
      tile_count : Int32,
      value_count : Int32,
      chunks : Array(StreamChunk)

    class Block
      getter bytes : Bytes
      getter row_count : Int32
      getter block_size : Int32
      getter mean_offset : Int32
      getter sigma_offset : Int32
      getter codes_offset : Int32
      getter record_chunks : Array(RecordChunk)

      def initialize(@bytes : Bytes,
                     @row_count : Int32,
                     @block_size : Int32,
                     @mean_offset : Int32,
                     @sigma_offset : Int32,
                     @codes_offset : Int32,
                     @record_chunks : Array(RecordChunk))
      end
    end

    class Parsed < Block
      getter record_spans : Array(RecordSpan)

      def initialize(bytes : Bytes,
                     row_count : Int32,
                     block_size : Int32,
                     mean_offset : Int32,
                     sigma_offset : Int32,
                     codes_offset : Int32,
                     record_chunks : Array(RecordChunk),
                     @record_spans : Array(RecordSpan))
        super(bytes, row_count, block_size, mean_offset, sigma_offset, codes_offset, record_chunks)
      end
    end

    class Stream
      getter bytes : Bytes
      getter blocks : Array(Block)
      getter row_count : Int32
      getter block_size : Int32
      getter record_spans : Array(StreamRecordSpan)

      def initialize(@bytes : Bytes,
                     @blocks : Array(Block),
                     @row_count : Int32,
                     @block_size : Int32,
                     @record_spans : Array(StreamRecordSpan))
      end
    end

    COLUMN_COUNT = 8_u64

    def parse(bytes : Bytes) : Parsed
      reader = Reader.new(bytes)
      block = parse_block(reader, bytes)
      raise ArgumentError.new("QBit Native block has trailing bytes") unless reader.offset == bytes.size
      spans = build_stream_spans([block])
      record_spans = spans.map do |span|
        chunk = span.chunks.first
        RecordSpan.new(span.cache_id, span.layer, span.kind, chunk.row_start, span.tile_count, span.value_count)
      end
      Parsed.new(
        block.bytes,
        block.row_count,
        block.block_size,
        block.mean_offset,
        block.sigma_offset,
        block.codes_offset,
        block.record_chunks,
        record_spans,
      )
    rescue ex : IndexError
      raise ArgumentError.new("truncated QBit Native block: #{ex.message}")
    end

    def parse_stream(bytes : Bytes) : Stream
      raise ArgumentError.new("QBit Native stream must contain at least one block") if bytes.empty?
      reader = Reader.new(bytes)
      blocks = [] of Block
      while reader.offset < bytes.size
        blocks << parse_block(reader, bytes)
      end

      block_size = blocks.first.block_size
      unless blocks.all? { |block| block.block_size == block_size }
        raise ArgumentError.new("QBit Native stream block size mismatch")
      end
      row_count = blocks.sum(0_i64, &.row_count.to_i64)
      raise ArgumentError.new("QBit Native stream row count exceeds Int32") if row_count > Int32::MAX
      Stream.new(bytes, blocks, row_count.to_i32, block_size, build_stream_spans(blocks))
    rescue ex : IndexError
      raise ArgumentError.new("truncated QBit Native stream: #{ex.message}")
    end

    private def parse_block(reader : Reader, source : Bytes) : Block
      block_start = reader.offset
      raise ArgumentError.new("QBit Native column count mismatch") unless reader.read_varuint == COLUMN_COUNT

      row_count_u64 = reader.read_varuint
      raise ArgumentError.new("QBit Native block must contain rows") if row_count_u64 == 0
      raise ArgumentError.new("QBit Native row count exceeds Int32") if row_count_u64 > Int32::MAX
      row_count = row_count_u64.to_i32

      expect_header(reader, "cache_id", "UInt64")
      cache_ids = Array(UInt64).new(row_count) { reader.read_u64 }

      expect_header(reader, "layer", "Int32")
      layers = Array(Int32).new(row_count) { reader.read_i32 }

      expect_header(reader, "kind", "UInt8")
      kinds = Array(UInt8).new(row_count) { reader.read_u8 }

      expect_header(reader, "tile", "UInt32")
      tiles = Array(UInt32).new(row_count) { reader.read_u32 }

      expect_header(reader, "value_count", "UInt16")
      value_counts = Array(UInt16).new(row_count) { reader.read_u16 }

      expect_header(reader, "mean", "Float32")
      mean_offset = reader.skip(column_bytes(row_count, sizeof(Float32), "mean"))

      expect_header(reader, "sigma", "Float32")
      sigma_offset = reader.skip(column_bytes(row_count, sizeof(Float32), "sigma"))
      validate_moments(source, mean_offset, sigma_offset, row_count)

      name = reader.read_string
      raise ArgumentError.new("QBit Native column name mismatch: expected codes, got #{name}") unless name == "codes"
      type = reader.read_string
      block_size = parse_qbit_type(type)
      plane_bytes = block_size // 8
      codes_offset = reader.skip(column_bytes(row_count, plane_bytes * 8, "codes"))

      block_size_bytes = reader.offset - block_start
      block_bytes = source[block_start, block_size_bytes]
      local_mean_offset = mean_offset - block_start
      local_sigma_offset = sigma_offset - block_start
      local_codes_offset = codes_offset - block_start
      validate_zero_lsb(block_bytes, local_codes_offset, row_count, plane_bytes)
      chunks = build_record_chunks(cache_ids, layers, kinds, tiles, value_counts, block_size)
      Block.new(block_bytes, row_count, block_size, local_mean_offset, local_sigma_offset, local_codes_offset, chunks)
    end

    private def expect_header(reader : Reader, expected_name : String, expected_type : String) : Nil
      name = reader.read_string
      type = reader.read_string
      unless name == expected_name && type == expected_type
        raise ArgumentError.new("QBit Native column mismatch: expected #{expected_name} #{expected_type}, got #{name} #{type}")
      end
    end

    private def parse_qbit_type(type : String) : Int32
      match = /\AQBit\(Int8, ([0-9]+)\)\z/.match(type)
      raise ArgumentError.new("QBit Native codes type is unsupported: #{type}") unless match
      block_size = match[1].to_i64
      unless block_size > 0 && block_size <= UInt16::MAX && block_size % 8 == 0
        raise ArgumentError.new("QBit Native block size must be positive, divisible by eight, and fit UInt16")
      end
      block_size.to_i32
    rescue ex : ArgumentError
      raise ex if ex.message.try(&.starts_with?("QBit Native"))
      raise ArgumentError.new("QBit Native block size is invalid")
    end

    private def column_bytes(rows : Int32, bytes_per_row : Int32, name : String) : Int32
      size = rows.to_i64 * bytes_per_row
      raise ArgumentError.new("QBit Native #{name} column is too large") if size > Int32::MAX
      size.to_i32
    end

    private def validate_zero_lsb(bytes : Bytes, codes_offset : Int32, rows : Int32, plane_bytes : Int32) : Nil
      lsb_offset = codes_offset.to_i64 + 7_i64 * rows * plane_bytes
      lsb_size = rows.to_i64 * plane_bytes
      lsb_size.to_i32.times do |i|
        raise ArgumentError.new("QBit Native p7 LSB stream must be zero") unless bytes[lsb_offset.to_i32 + i] == 0_u8
      end
    end

    private def validate_moments(bytes : Bytes, mean_offset : Int32, sigma_offset : Int32, rows : Int32) : Nil
      rows.times do |row|
        mean = read_f32_le(bytes, mean_offset + row * sizeof(Float32))
        sigma = read_f32_le(bytes, sigma_offset + row * sizeof(Float32))
        unless mean.finite? && sigma.finite? && sigma >= 0.0_f32
          raise ArgumentError.new("QBit Native block moments are invalid")
        end
      end
    end

    private def read_f32_le(bytes : Bytes, offset : Int32) : Float32
      bits = bytes[offset].to_u32 |
             (bytes[offset + 1].to_u32 << 8) |
             (bytes[offset + 2].to_u32 << 16) |
             (bytes[offset + 3].to_u32 << 24)
      bits.unsafe_as(Float32)
    end

    private def build_record_chunks(cache_ids : Array(UInt64),
                                    layers : Array(Int32),
                                    kinds : Array(UInt8),
                                    tiles : Array(UInt32),
                                    value_counts : Array(UInt16),
                                    block_size : Int32) : Array(RecordChunk)
      chunks = [] of RecordChunk
      row = 0
      while row < cache_ids.size
        cache_id = cache_ids[row]
        layer = layers[row]
        kind = kinds[row]
        start = row
        tile_start = tiles[row]
        total = 0_i64
        while row < cache_ids.size && cache_ids[row] == cache_id && layers[row] == layer && kinds[row] == kind
          expected_tile = tile_start.to_u64 + (row - start).to_u64
          unless expected_tile <= UInt32::MAX && tiles[row] == expected_tile.to_u32
            raise ArgumentError.new("QBit Native record tile sequence mismatch")
          end
          count = value_counts[row].to_i32
          raise ArgumentError.new("QBit Native tile value count is invalid") unless count > 0 && count <= block_size
          total += count
          row += 1
        end

        tile_count = row - start
        (tile_count - 1).times do |tile|
          unless value_counts[start + tile] == block_size
            raise ArgumentError.new("QBit Native non-final tile must be full")
          end
        end
        raise ArgumentError.new("QBit Native record value count exceeds Int32") if total > Int32::MAX
        raise ArgumentError.new("QBit Native tile ordinal exceeds Int32") if tile_start > Int32::MAX
        chunks << RecordChunk.new(
          cache_id,
          layer,
          kind,
          start.to_i32,
          tile_start.to_i32,
          tile_count.to_i32,
          total.to_i32,
          value_counts[row - 1].to_i32,
        )
      end
      chunks
    end

    private def build_stream_spans(blocks : Array(Block)) : Array(StreamRecordSpan)
      spans = [] of StreamRecordSpan
      seen = Set({UInt64, Int32, UInt8}).new
      have_pending = false
      pending_cache_id = 0_u64
      pending_layer = 0_i32
      pending_kind = 0_u8
      pending_tile_count = 0_i64
      pending_value_count = 0_i64
      pending_last_value_count = 0_i32
      pending_chunks = [] of StreamChunk

      blocks.each_with_index do |block, block_index|
        block.record_chunks.each do |chunk|
          key = {chunk.cache_id, chunk.layer, chunk.kind}
          pending_key = {pending_cache_id, pending_layer, pending_kind}
          unless have_pending && key == pending_key
            if have_pending
              spans << StreamRecordSpan.new(
                pending_cache_id,
                pending_layer,
                pending_kind,
                pending_tile_count.to_i32,
                pending_value_count.to_i32,
                pending_chunks,
              )
              seen << pending_key
            end
            raise ArgumentError.new("QBit Native record rows are not contiguous") if seen.includes?(key)
            raise ArgumentError.new("QBit Native record tile sequence mismatch") unless chunk.tile_start == 0
            have_pending = true
            pending_cache_id = chunk.cache_id
            pending_layer = chunk.layer
            pending_kind = chunk.kind
            pending_tile_count = 0_i64
            pending_value_count = 0_i64
            pending_last_value_count = 0_i32
            pending_chunks = [] of StreamChunk
          else
            unless pending_last_value_count == block.block_size
              raise ArgumentError.new("QBit Native non-final tile must be full")
            end
            unless chunk.tile_start.to_i64 == pending_tile_count
              raise ArgumentError.new("QBit Native record tile sequence mismatch")
            end
          end

          next_tile_count = pending_tile_count + chunk.tile_count
          next_value_count = pending_value_count + chunk.value_count
          if next_tile_count > Int32::MAX || next_value_count > Int32::MAX
            raise ArgumentError.new("QBit Native record value count exceeds Int32")
          end
          pending_chunks << StreamChunk.new(
            block_index.to_i32,
            chunk.row_start,
            chunk.tile_start,
            chunk.tile_count,
            pending_value_count.to_i32,
            chunk.value_count,
          )
          pending_tile_count = next_tile_count
          pending_value_count = next_value_count
          pending_last_value_count = chunk.last_value_count
        end
      end

      if have_pending
        spans << StreamRecordSpan.new(
          pending_cache_id,
          pending_layer,
          pending_kind,
          pending_tile_count.to_i32,
          pending_value_count.to_i32,
          pending_chunks,
        )
      end
      spans
    end

    private class Reader
      getter offset : Int32

      def initialize(@bytes : Bytes)
        @offset = 0
      end

      def read_u8 : UInt8
        ensure_remaining(1)
        value = @bytes[@offset]
        @offset += 1
        value
      end

      def read_u16 : UInt16
        value = read_u8.to_u16
        value |= read_u8.to_u16 << 8
        value
      end

      def read_u32 : UInt32
        value = 0_u32
        4.times { |i| value |= read_u8.to_u32 << (i * 8) }
        value
      end

      def read_i32 : Int32
        read_u32.unsafe_as(Int32)
      end

      def read_u64 : UInt64
        value = 0_u64
        8.times { |i| value |= read_u8.to_u64 << (i * 8) }
        value
      end

      def read_varuint : UInt64
        value = 0_u64
        shift = 0
        loop do
          byte = read_u8
          raise ArgumentError.new("QBit Native VarUInt overflow") if shift == 63 && byte > 1
          value |= (byte & 0x7f_u8).to_u64 << shift
          return value if byte & 0x80_u8 == 0
          shift += 7
          raise ArgumentError.new("QBit Native VarUInt overflow") if shift > 63
        end
      end

      def read_string : String
        size_u64 = read_varuint
        raise ArgumentError.new("QBit Native string is too large") if size_u64 > Int32::MAX
        size = size_u64.to_i32
        start = skip(size)
        String.new(@bytes[start, size])
      end

      def skip(size : Int32) : Int32
        ensure_remaining(size)
        start = @offset
        @offset += size
        start
      end

      private def ensure_remaining(size : Int32) : Nil
        raise IndexError.new("negative read") if size < 0
        raise IndexError.new("end of block") if size > @bytes.size - @offset
      end
    end
  end
end
