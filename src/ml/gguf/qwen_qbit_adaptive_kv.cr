require "./qwen_qbit_gaussian_codec"

module ML::GGUF
  # A deliberately small adaptive row codec for resident Q/K/V state.
  #
  # The wire layout is canonical and intentionally boring:
  #
  #   [all dense p4 rows][one 8-byte metadata record per row][sidecar stream]
  #
  # A row is exactly 256 values. Every row owns a 136-byte QBit p4 payload,
  # even when its selected tier is a replacement. Metadata stores a UInt32 tier
  # followed by a UInt32 offset into the sidecar stream. Sidecar offsets are
  # relative to the start of the sidecar stream, and the stream is append-only
  # in row order. This makes validation independent of allocation history and
  # rejects holes, aliases, reordering, and trailing bytes.
  module QwenQBitAdaptiveKV
    extend self

    ROW_VALUES         = 256
    TIER_BYTES         =   4
    OFFSET_BYTES       =   4
    METADATA_BYTES     = TIER_BYTES + OFFSET_BYTES
    BASE_ROW_BYTES     = 8 + 4 * (ROW_VALUES // 8)
    P4_SIDECAR_BYTES   = 0
    P5_SIDECAR_BYTES   = ROW_VALUES // 8
    BF16_SIDECAR_BYTES = ROW_VALUES * sizeof(UInt16)
    F32_SIDECAR_BYTES  = ROW_VALUES * sizeof(Float32)

    # Metadata values are stable on-wire integers. Keeping the enum's backing
    # type UInt32 prevents an accidental signed conversion at the boundary.
    enum Tier : UInt32
      P4   = 0
      P5   = 1
      BF16 = 2
      F32  = 3
    end

    record Encoded,
      value_count : Int32,
      block_size : Int32,
      payload : Bytes do
      def payload_bytes : Int32
        payload.size.to_i32
      end
    end

    # Zero-copy views used to bind the canonical artifact to three Metal
    # buffers. `regions` validates the artifact once before exposing them.
    record Regions, base : Bytes, metadata : Bytes, sidecar : Bytes

    # Immutable allocation plan for an append-only resident cache. Tier
    # metadata and sidecar offsets depend only on row order, so the exact Metal
    # allocation can be known before any K/V values are produced.
    class Plan
      getter row_count : Int32
      getter base_bytes : Int32
      getter metadata_bytes : Int32
      getter sidecar_bytes : Int32
      getter payload_bytes : Int32
      @prefix_sidecar_bytes : Array(Int32)

      private def initialize(@row_count : Int32,
                             @base_bytes : Int32,
                             @metadata_bytes : Int32,
                             @sidecar_bytes : Int32,
                             @payload_bytes : Int32,
                             @metadata : Bytes,
                             @prefix_sidecar_bytes : Array(Int32))
      end

      def self.from_tiers(tiers : Array(Tier)) : self
        row_count = tiers.size.to_i32
        base_bytes64 = row_count.to_i64 * BASE_ROW_BYTES
        metadata_bytes64 = row_count.to_i64 * METADATA_BYTES
        prefix_sidecar = Array(Int32).new(row_count + 1, 0_i32)
        sidecar_cursor = 0_i64

        tiers.each_with_index do |selected, row|
          sidecar_cursor += QwenQBitAdaptiveKV.sidecar_size(selected)
          raise ArgumentError.new("adaptive QBit sidecar offset exceeds UInt32") if sidecar_cursor > UInt32::MAX
          raise ArgumentError.new("adaptive QBit payload is too large") if sidecar_cursor > Int32::MAX
          prefix_sidecar[row + 1] = sidecar_cursor.to_i32
        end

        total_bytes = base_bytes64 + metadata_bytes64 + sidecar_cursor
        raise ArgumentError.new("adaptive QBit payload is too large") if total_bytes > Int32::MAX
        metadata = Bytes.new(metadata_bytes64.to_i, 0_u8)
        tiers.each_with_index do |selected, row|
          offset = row * METADATA_BYTES
          write_u32_le(metadata, offset, selected.value)
          write_u32_le(metadata, offset + TIER_BYTES, prefix_sidecar[row].to_u32)
        end

        new(
          row_count,
          base_bytes64.to_i32,
          metadata_bytes64.to_i32,
          sidecar_cursor.to_i32,
          total_bytes.to_i32,
          metadata,
          prefix_sidecar,
        )
      end

      def prefix_sidecar_bytes(prefix_rows : Int) : Int32
        unless prefix_rows >= 0 && prefix_rows <= @row_count
          raise ArgumentError.new("adaptive QBit plan prefix row count is out of range")
        end
        @prefix_sidecar_bytes[prefix_rows]
      end

      # Callers receive a copy so the canonical plan cannot be changed after
      # its offsets have been admitted or uploaded.
      def metadata : Bytes
        @metadata.dup
      end

      private def self.write_u32_le(payload : Bytes, offset : Int, value : UInt32) : Nil
        payload[offset] = (value & 0xff).to_u8
        payload[offset + 1] = ((value >> 8) & 0xff).to_u8
        payload[offset + 2] = ((value >> 16) & 0xff).to_u8
        payload[offset + 3] = (value >> 24).to_u8
      end
    end

    # Build canonical metadata without allocating or touching the much larger
    # value payload. Prefix sidecar sizes make live append snapshots exact.
    def plan(tiers : Array(Tier)) : Plan
      Plan.from_tiers(tiers)
    end

    # Allocate the canonical prefix layout with zero values. Device packers
    # can fill its base and sidecar regions and then pass the result through the
    # ordinary strict validator/decoder.
    def empty_encoded(plan : Plan, prefix_rows : Int) : Encoded
      prefix_sidecar = plan.prefix_sidecar_bytes(prefix_rows)
      base_bytes = prefix_rows.to_i64 * BASE_ROW_BYTES
      metadata_bytes = prefix_rows.to_i64 * METADATA_BYTES
      total_bytes = base_bytes + metadata_bytes + prefix_sidecar
      value_count = prefix_rows.to_i64 * ROW_VALUES
      raise ArgumentError.new("adaptive QBit payload is too large") if total_bytes > Int32::MAX
      raise ArgumentError.new("adaptive QBit value count exceeds Int32") if value_count > Int32::MAX

      payload = Bytes.new(total_bytes.to_i, 0_u8)
      payload[base_bytes.to_i, metadata_bytes.to_i].copy_from(
        plan.metadata[0, metadata_bytes.to_i]
      ) unless metadata_bytes == 0
      Encoded.new(value_count.to_i32, ROW_VALUES, payload)
    end

    # Join separately resident base and sidecar prefixes back into the
    # canonical transport artifact. This is intentionally a verification and
    # snapshot boundary, not part of the attention hot path.
    def encoded_from_regions(plan : Plan, prefix_rows : Int,
                             base : Bytes, sidecar : Bytes) : Encoded
      expected_base = prefix_rows.to_i64 * BASE_ROW_BYTES
      expected_sidecar = plan.prefix_sidecar_bytes(prefix_rows)
      unless base.size == expected_base
        raise ArgumentError.new("adaptive QBit base prefix size mismatch")
      end
      unless sidecar.size == expected_sidecar
        raise ArgumentError.new("adaptive QBit sidecar prefix size mismatch")
      end

      encoded = empty_encoded(plan, prefix_rows)
      metadata_bytes = prefix_rows * METADATA_BYTES
      encoded.payload[0, base.size].copy_from(base) unless base.empty?
      sidecar_offset = base.size + metadata_bytes
      encoded.payload[sidecar_offset, sidecar.size].copy_from(sidecar) unless sidecar.empty?
      validate(encoded)
      encoded
    end

    # Encode all rows as dense p4 plus the requested per-row tier.
    def encode(values : Array(Float32), tiers : Array(Tier), block_size : Int32 = ROW_VALUES) : Encoded
      validate_block_size(block_size)
      validate_values_shape(values, tiers)

      row_count = rows_for(values.size)
      sidecar_bytes = sidecar_bytes_for(tiers)
      total_bytes = row_count.to_i64 * (BASE_ROW_BYTES + METADATA_BYTES) + sidecar_bytes
      raise ArgumentError.new("adaptive QBit payload is too large") if total_bytes > Int32::MAX

      payload = Bytes.new(total_bytes.to_i, 0_u8)
      metadata_base = row_count * BASE_ROW_BYTES
      sidecar_base = metadata_base + row_count * METADATA_BYTES
      sidecar_cursor = 0_i64

      row_count.times do |row|
        row_values = values[row * ROW_VALUES, ROW_VALUES]
        p4 = QwenQBitGaussianCodec.encode(row_values, ROW_VALUES, 4)
        raise ArgumentError.new("adaptive QBit p4 row has an unexpected size") unless p4.payload.size == BASE_ROW_BYTES

        base_offset = row * BASE_ROW_BYTES
        payload[base_offset, BASE_ROW_BYTES].copy_from(p4.payload)

        selected = tiers[row]
        metadata_offset = metadata_base + row * METADATA_BYTES
        raise ArgumentError.new("adaptive QBit sidecar offset exceeds UInt32") if sidecar_cursor > UInt32::MAX
        write_u32_le(payload, metadata_offset, selected.value)
        write_u32_le(payload, metadata_offset + TIER_BYTES, sidecar_cursor.to_u32)

        sidecar_offset = sidecar_base + sidecar_cursor
        case selected
        when Tier::P4
          # The dense p4 base is already the complete representation.
        when Tier::P5
          p5 = QwenQBitGaussianCodec.encode(row_values, ROW_VALUES, 5)
          plane = QwenQBitGaussianCodec.tile_plane(p5, 0, 4)
          payload[sidecar_offset, P5_SIDECAR_BYTES].copy_from(plane)
        when Tier::BF16
          row_values.each_with_index do |value, i|
            bits = bf16_bits(value)
            raise ArgumentError.new("adaptive QBit BF16 replacement must be finite") unless bf16_bits_finite?(bits)
            write_u16_le(payload, sidecar_offset + i * sizeof(UInt16), bits)
          end
        when Tier::F32
          row_values.each_with_index do |value, i|
            raise ArgumentError.new("adaptive QBit F32 replacement must be finite") unless value.finite?
            write_u32_le(payload, sidecar_offset + i * sizeof(Float32), value.unsafe_as(UInt32))
          end
        end
        sidecar_cursor += sidecar_size(selected)
      end

      encoded = Encoded.new(values.size.to_i32, block_size, payload)
      validate(encoded)
      encoded
    end

    # Decode after strict validation. Replacement tiers bypass the p4 base;
    # p4/p5 use the existing Gaussian codec's exact reconstruction tables.
    def decode(encoded : Encoded) : Array(Float32)
      validate(encoded)
      values = Array(Float32).new(encoded.value_count, 0.0_f32)
      row_count = rows_for(encoded.value_count)
      row_count.times do |row|
        selected = read_tier(encoded, row)
        row_values = case selected
                     when Tier::P4
                       decode_p4_row(encoded, row)
                     when Tier::P5
                       decode_p5_row(encoded, row)
                     when Tier::BF16
                       decode_bf16_row(encoded, row)
                     when Tier::F32
                       decode_f32_row(encoded, row)
                     else
                       raise ArgumentError.new("invalid adaptive QBit tier #{selected.value}")
                     end
        row_values.each_with_index do |value, i|
          values[row * ROW_VALUES + i] = value
        end
      end
      values
    end

    # Validate shape, canonical metadata, bounds, exact sidecar consumption,
    # base-row moments, finite p4/p5 reconstruction bounds, and replacement
    # finiteness without returning values.
    def validate(encoded : Encoded) : Nil
      validate_block_size(encoded.block_size)
      raise ArgumentError.new("adaptive QBit value count must be non-negative") if encoded.value_count < 0
      unless encoded.value_count % ROW_VALUES == 0
        raise ArgumentError.new("adaptive QBit value count must be aligned to 256-value rows")
      end

      row_count = rows_for(encoded.value_count)
      metadata_base = row_count * BASE_ROW_BYTES
      sidecar_base = metadata_base + row_count * METADATA_BYTES
      if encoded.payload.size < sidecar_base
        raise ArgumentError.new("corrupt adaptive QBit metadata size")
      end

      sidecar_cursor = 0_i64
      row_count.times do |row|
        validate_p4_row(encoded, row)

        metadata_offset = metadata_base + row * METADATA_BYTES
        raw_tier = read_u32_le(encoded.payload, metadata_offset)
        selected = tier_from_raw(raw_tier)
        case selected
        when Tier::P4
          validate_quantized_reconstruction(encoded, row, 4)
        when Tier::P5
          validate_quantized_reconstruction(encoded, row, 5)
        end
        offset = read_u32_le(encoded.payload, metadata_offset + TIER_BYTES)
        unless offset.to_i64 == sidecar_cursor
          raise ArgumentError.new("adaptive QBit sidecar offset is not canonical for row #{row}")
        end

        size = sidecar_size(selected).to_i64
        sidecar_end = sidecar_cursor + size
        if sidecar_end > UInt32::MAX
          raise ArgumentError.new("adaptive QBit sidecar offset exceeds UInt32")
        end
        if sidecar_base.to_i64 + sidecar_end > encoded.payload.size
          raise ArgumentError.new("adaptive QBit sidecar bounds exceed payload")
        end

        sidecar_offset = sidecar_base + sidecar_cursor
        case selected
        when Tier::BF16
          validate_bf16_sidecar(encoded.payload, sidecar_offset)
        when Tier::F32
          validate_f32_sidecar(encoded.payload, sidecar_offset)
        end
        sidecar_cursor = sidecar_end
      end

      expected_size = sidecar_base.to_i64 + sidecar_cursor
      unless expected_size == encoded.payload.size
        raise ArgumentError.new("adaptive QBit payload has non-exact sidecar consumption")
      end
      nil
    end

    # Public metadata accessors. They validate the complete record so callers
    # cannot observe a tier or offset from an unadmitted/corrupt payload.
    def tier(encoded : Encoded, row : Int32) : Tier
      validate(encoded)
      validate_row_index(encoded, row)
      read_tier(encoded, row)
    end

    def sidecar_offset(encoded : Encoded, row : Int32) : UInt32
      validate(encoded)
      validate_row_index(encoded, row)
      metadata_offset = rows_for(encoded.value_count) * BASE_ROW_BYTES + row * METADATA_BYTES
      read_u32_le(encoded.payload, metadata_offset + TIER_BYTES)
    end

    # Split the canonical payload once after one strict validation pass.
    def regions(encoded : Encoded) : Regions
      validate(encoded)
      row_count = rows_for(encoded.value_count)
      metadata_offset = row_count * BASE_ROW_BYTES
      sidecar_offset = metadata_offset + row_count * METADATA_BYTES
      Regions.new(
        encoded.payload[0, metadata_offset],
        encoded.payload[metadata_offset, sidecar_offset - metadata_offset],
        encoded.payload[sidecar_offset, encoded.payload.size - sidecar_offset],
      )
    end

    def sidecar_size(selected : Tier) : Int32
      case selected
      when Tier::P4   then P4_SIDECAR_BYTES
      when Tier::P5   then P5_SIDECAR_BYTES
      when Tier::BF16 then BF16_SIDECAR_BYTES
      when Tier::F32  then F32_SIDECAR_BYTES
      else
        raise ArgumentError.new("invalid adaptive QBit tier #{selected.value}")
      end
    end

    # Calculate the exact wire size before encoding. This mirrors the
    # canonical cursor policy and is intentionally independent of value data.
    def payload_bytes(value_count : Int, tiers : Array(Tier), block_size : Int32 = ROW_VALUES) : Int32
      validate_block_size(block_size)
      unless value_count >= 0 && value_count % ROW_VALUES == 0
        raise ArgumentError.new("adaptive QBit value count must be aligned to 256-value rows")
      end
      rows = rows_for(value_count)
      unless tiers.size == rows
        raise ArgumentError.new("adaptive QBit tier count must equal row count")
      end
      total = rows.to_i64 * (BASE_ROW_BYTES + METADATA_BYTES) + sidecar_bytes_for(tiers)
      raise ArgumentError.new("adaptive QBit payload is too large") if total > Int32::MAX
      total.to_i32
    end

    private def validate_values_shape(values : Array(Float32), tiers : Array(Tier)) : Nil
      unless values.size % ROW_VALUES == 0
        raise ArgumentError.new("adaptive QBit values must be aligned to 256-value rows")
      end
      rows = rows_for(values.size)
      unless tiers.size == rows
        raise ArgumentError.new("adaptive QBit tier count must equal row count")
      end
      values.each do |value|
        raise ArgumentError.new("adaptive QBit values must be finite") unless value.finite?
      end
      nil
    end

    private def validate_block_size(block_size : Int32) : Nil
      raise ArgumentError.new("adaptive QBit block size must equal 256") unless block_size == ROW_VALUES
    end

    private def rows_for(value_count : Int) : Int32
      raise ArgumentError.new("adaptive QBit value count must be non-negative") if value_count < 0
      raise ArgumentError.new("adaptive QBit value count exceeds Int32") if value_count > Int32::MAX
      (value_count // ROW_VALUES).to_i32
    end

    private def sidecar_bytes_for(tiers : Array(Tier)) : Int64
      tiers.sum(0_i64) { |selected| sidecar_size(selected).to_i64 }
    end

    private def validate_row_index(encoded : Encoded, row : Int32) : Nil
      row_count = rows_for(encoded.value_count)
      raise ArgumentError.new("adaptive QBit row out of range: #{row}") unless row >= 0 && row < row_count
    end

    private def validate_p4_row(encoded : Encoded, row : Int32) : Nil
      offset = row * BASE_ROW_BYTES
      base = QwenQBitGaussianCodec::Encoded.new(
        ROW_VALUES, ROW_VALUES, 4, encoded.payload[offset, BASE_ROW_BYTES]
      )
      QwenQBitGaussianCodec.validate(base)
    end

    private def validate_quantized_reconstruction(encoded : Encoded, row : Int32, precision : Int32) : Nil
      offset = row * BASE_ROW_BYTES
      mean = read_u32_le(encoded.payload, offset).unsafe_as(Float32)
      sigma = read_u32_le(encoded.payload, offset + sizeof(Float32)).unsafe_as(Float32)
      max_centroid = QwenQBitGaussianCodec.reconstruct_raw_code(0x7f_u8, precision)
      low = mean - sigma * max_centroid
      high = mean + sigma * max_centroid
      unless low.finite? && high.finite?
        raise ArgumentError.new("adaptive QBit p#{precision} row has non-finite reconstruction")
      end
    end

    private def read_tier(encoded : Encoded, row : Int32) : Tier
      metadata_base = rows_for(encoded.value_count) * BASE_ROW_BYTES
      tier_from_raw(read_u32_le(encoded.payload, metadata_base + row * METADATA_BYTES))
    end

    private def read_sidecar_offset(encoded : Encoded, row : Int32) : UInt32
      metadata_base = rows_for(encoded.value_count) * BASE_ROW_BYTES
      read_u32_le(encoded.payload, metadata_base + row * METADATA_BYTES + TIER_BYTES)
    end

    private def decode_p4_row(encoded : Encoded, row : Int32) : Array(Float32)
      base = QwenQBitGaussianCodec::Encoded.new(ROW_VALUES, ROW_VALUES, 4, base_row_unchecked(encoded, row))
      QwenQBitGaussianCodec.decode(base)
    end

    private def decode_p5_row(encoded : Encoded, row : Int32) : Array(Float32)
      base_payload = base_row_unchecked(encoded, row)
      p5_payload = Bytes.new(QwenQBitGaussianCodec::BLOCK_HEADER_BYTES + 5 * P5_SIDECAR_BYTES, 0_u8)
      p5_payload[0, QwenQBitGaussianCodec::BLOCK_HEADER_BYTES].copy_from(
        base_payload[0, QwenQBitGaussianCodec::BLOCK_HEADER_BYTES]
      )
      4.times do |plane|
        source = QwenQBitGaussianCodec::BLOCK_HEADER_BYTES + plane * P5_SIDECAR_BYTES
        p5_payload[source, P5_SIDECAR_BYTES].copy_from(base_payload[source, P5_SIDECAR_BYTES])
      end
      p5_payload[QwenQBitGaussianCodec::BLOCK_HEADER_BYTES + 4 * P5_SIDECAR_BYTES, P5_SIDECAR_BYTES].copy_from(
        sidecar_row_unchecked(encoded, row)
      )
      p5 = QwenQBitGaussianCodec::Encoded.new(ROW_VALUES, ROW_VALUES, 5, p5_payload)
      QwenQBitGaussianCodec.decode(p5)
    end

    private def decode_bf16_row(encoded : Encoded, row : Int32) : Array(Float32)
      sidecar = sidecar_row_unchecked(encoded, row)
      Array(Float32).new(ROW_VALUES) do |i|
        (read_u16_le(sidecar, i * sizeof(UInt16)).to_u32 << 16).unsafe_as(Float32)
      end
    end

    private def decode_f32_row(encoded : Encoded, row : Int32) : Array(Float32)
      sidecar = sidecar_row_unchecked(encoded, row)
      Array(Float32).new(ROW_VALUES) { |i| read_u32_le(sidecar, i * sizeof(Float32)).unsafe_as(Float32) }
    end

    private def base_row_unchecked(encoded : Encoded, row : Int32) : Bytes
      encoded.payload[row * BASE_ROW_BYTES, BASE_ROW_BYTES]
    end

    private def sidecar_row_unchecked(encoded : Encoded, row : Int32) : Bytes
      selected = read_tier(encoded, row)
      size = sidecar_size(selected)
      sidecar_base = rows_for(encoded.value_count) * (BASE_ROW_BYTES + METADATA_BYTES)
      offset = sidecar_base + read_sidecar_offset(encoded, row).to_i
      encoded.payload[offset, size]
    end

    private def validate_bf16_sidecar(payload : Bytes, offset : Int32) : Nil
      ROW_VALUES.times do |i|
        bits = read_u16_le(payload, offset + i * sizeof(UInt16))
        raise ArgumentError.new("adaptive QBit BF16 replacement must be finite") unless bf16_bits_finite?(bits)
      end
    end

    private def validate_f32_sidecar(payload : Bytes, offset : Int32) : Nil
      ROW_VALUES.times do |i|
        value = read_u32_le(payload, offset + i * sizeof(Float32)).unsafe_as(Float32)
        raise ArgumentError.new("adaptive QBit F32 replacement must be finite") unless value.finite?
      end
    end

    private def tier_from_raw(raw : UInt32) : Tier
      case raw
      when Tier::P4.value   then Tier::P4
      when Tier::P5.value   then Tier::P5
      when Tier::BF16.value then Tier::BF16
      when Tier::F32.value  then Tier::F32
      else
        raise ArgumentError.new("invalid adaptive QBit tier #{raw}")
      end
    end

    private def bf16_bits(value : Float32) : UInt16
      bits = value.unsafe_as(UInt32)
      lsb = (bits >> 16) & 1_u32
      ((bits + 0x7fff_u32 + lsb) >> 16).to_u16
    end

    private def bf16_bits_finite?(bits : UInt16) : Bool
      (bits & 0x7f80_u16) != 0x7f80_u16
    end

    private def read_u16_le(bytes : Bytes, offset : Int32) : UInt16
      bytes[offset].to_u16 | (bytes[offset + 1].to_u16 << 8)
    end

    private def write_u16_le(bytes : Bytes, offset : Int32, value : UInt16) : Nil
      bytes[offset] = (value & 0xff_u16).to_u8
      bytes[offset + 1] = (value >> 8).to_u8
    end

    private def read_u32_le(bytes : Bytes, offset : Int32) : UInt32
      bytes[offset].to_u32 |
        (bytes[offset + 1].to_u32 << 8) |
        (bytes[offset + 2].to_u32 << 16) |
        (bytes[offset + 3].to_u32 << 24)
    end

    private def write_u32_le(bytes : Bytes, offset : Int32, value : UInt32) : Nil
      bytes[offset] = (value & 0xff_u32).to_u8
      bytes[offset + 1] = ((value >> 8) & 0xff_u32).to_u8
      bytes[offset + 2] = ((value >> 16) & 0xff_u32).to_u8
      bytes[offset + 3] = ((value >> 24) & 0xff_u32).to_u8
    end
  end
end
