module ML::GGUF
  # Experimental CPU codec for evaluating QBit-style progressive compression of
  # recurrent Qwen state. It intentionally does not participate in the durable
  # .qkv artifact format yet.
  #
  # Each block stores an affine normalization (mean and standard deviation),
  # followed by the most-significant bit planes of a 256-level Gaussian
  # Lloyd-Max code. Prefixes are reconstructed with the conditional mean of the
  # standard-normal interval represented by the retained planes, matching the
  # reconstruction rule used by ClickHouse QBit's truncated Gaussian codes.
  module QwenQBitGaussianCodec
    extend self

    record Encoded,
      value_count : Int32,
      block_size : Int32,
      precision : Int32,
      payload : Bytes

    MIN_PRECISION      = 6
    MAX_PRECISION      = 8
    BLOCK_HEADER_BYTES = 2 * sizeof(Float32)

    # Positive half of ClickHouse's 256-level standard-normal Lloyd-Max table
    # (Apache-2.0, src/Common/LloydMaxQuantizer.h, PR #111867). Negative levels
    # are obtained by exact reflection. The eight-plane decoder rounds these
    # values to BF16, as ClickHouse's full-code reconstruction does.
    private POSITIVE_LEVELS = [
      0.00491977_f32, 0.01475981_f32, 0.02460130_f32, 0.03444523_f32,
      0.04429256_f32, 0.05414428_f32, 0.06400137_f32, 0.07386480_f32,
      0.08373558_f32, 0.09361469_f32, 0.10350313_f32, 0.11340192_f32,
      0.12331206_f32, 0.13323459_f32, 0.14317053_f32, 0.15312092_f32,
      0.16308682_f32, 0.17306929_f32, 0.18306942_f32, 0.19308828_f32,
      0.20312698_f32, 0.21318664_f32, 0.22326841_f32, 0.23337343_f32,
      0.24350287_f32, 0.25365792_f32, 0.26383980_f32, 0.27404974_f32,
      0.28428900_f32, 0.29455885_f32, 0.30486060_f32, 0.31519559_f32,
      0.32556517_f32, 0.33597074_f32, 0.34641372_f32, 0.35689557_f32,
      0.36741778_f32, 0.37798188_f32, 0.38858944_f32, 0.39924207_f32,
      0.40994142_f32, 0.42068918_f32, 0.43148712_f32, 0.44233703_f32,
      0.45324075_f32, 0.46420020_f32, 0.47521736_f32, 0.48629424_f32,
      0.49743295_f32, 0.50863566_f32, 0.51990462_f32, 0.53124215_f32,
      0.54265067_f32, 0.55413266_f32, 0.56569073_f32, 0.57732756_f32,
      0.58904597_f32, 0.60084886_f32, 0.61273927_f32, 0.62472037_f32,
      0.63679545_f32, 0.64896799_f32, 0.66124158_f32, 0.67362001_f32,
      0.68610723_f32, 0.69870743_f32, 0.71142496_f32, 0.72426444_f32,
      0.73723071_f32, 0.75032892_f32, 0.76356447_f32, 0.77694313_f32,
      0.79047101_f32, 0.80415460_f32, 0.81800086_f32, 0.83201723_f32,
      0.84621168_f32, 0.86059283_f32, 0.87516997_f32, 0.88995319_f32,
      0.90495349_f32, 0.92018290_f32, 0.93565460_f32, 0.95138317_f32,
      0.96738473_f32, 0.98367719_f32, 1.00028055_f32, 1.01721718_f32,
      1.03451219_f32, 1.05219386_f32, 1.07029404_f32, 1.08884872_f32,
      1.10789853_f32, 1.12748941_f32, 1.14767324_f32, 1.16850859_f32,
      1.19006145_f32, 1.21240610_f32, 1.23562592_f32, 1.25981428_f32,
      1.28507552_f32, 1.31152588_f32, 1.33929452_f32, 1.36852460_f32,
      1.39937446_f32, 1.43201890_f32, 1.46665067_f32, 1.50348227_f32,
      1.54274811_f32, 1.58470730_f32, 1.62964731_f32, 1.67788877_f32,
      1.72979202_f32, 1.78576605_f32, 1.84628084_f32, 1.91188474_f32,
      1.98322915_f32, 2.06110438_f32, 2.14649281_f32, 2.24065008_f32,
      2.34523372_f32, 2.46251620_f32, 2.59575919_f32, 2.74992207_f32,
      2.93314607_f32, 3.16034096_f32, 3.46399932_f32, 3.94331723_f32,
    ] of Float32

    # Exact Float32 bit patterns from ClickHouse
    # LloydMax::POSITIVE_PREFIX_CENTROIDS. Keeping the generated values avoids
    # platform-dependent libm rounding in the cache representation.
    private P6_POSITIVE_PREFIX_CENTROID_BITS = [
      0x3ca13bf4_u32, 0x3d71fa9c_u32, 0x3dc9dcd5_u32, 0x3e0d8782_u32,
      0x3e365b21_u32, 0x3e5f7b2c_u32, 0x3e847d3d_u32, 0x3e997688_u32,
      0x3eaeb43f_u32, 0x3ec44216_u32, 0x3eda2cee_u32, 0x3ef08319_u32,
      0x3f03aa56_u32, 0x3f0f59fc_u32, 0x3f1b5b0a_u32, 0x3f27b9d1_u32,
      0x3f3484f0_u32, 0x3f41ce0e_u32, 0x3f4fab03_u32, 0x3f5e37b8_u32,
      0x3f6d9967_u32, 0x3f7e04ad_u32, 0x3f87e447_u32, 0x3f91b0cf_u32,
      0x3f9ccc1b_u32, 0x3fa9d54e_u32, 0x3fb9c094_u32, 0x3fcdf6e5_u32,
      0x3fe889ed_u32, 0x40065d39_u32, 0x402059ac_u32, 0x4048b7dd_u32,
    ] of UInt32

    private P7_POSITIVE_PREFIX_CENTROID_BITS = [
      0x3c2137cb_u32, 0x3cf1dbd9_u32, 0x3d499a25_u32, 0x3d8d2d69_u32,
      0x3db59c1c_u32, 0x3dde1d58_u32, 0x3e035aab_u32, 0x3e17b432_u32,
      0x3e2c1d6f_u32, 0x3e40989f_u32, 0x3e55280b_u32, 0x3e69ce0c_u32,
      0x3e7e8d0e_u32, 0x3e89b3c9_u32, 0x3e943016_u32, 0x3e9ebcc7_u32,
      0x3ea95b42_u32, 0x3eb40cfc_u32, 0x3ebed37e_u32, 0x3ec9b061_u32,
      0x3ed4a557_u32, 0x3edfb429_u32, 0x3eeadeb8_u32, 0x3ef62707_u32,
      0x3f00c79c_u32, 0x3f068cca_u32, 0x3f0c6445_u32, 0x3f124f5c_u32,
      0x3f184f7b_u32, 0x3f1e662a_u32, 0x3f249511_u32, 0x3f2ade02_u32,
      0x3f3142fb_u32, 0x3f37c629_u32, 0x3f3e69fa_u32, 0x3f453121_u32,
      0x3f4c1ea5_u32, 0x3f5335f4_u32, 0x3f5a7afc_u32, 0x3f61f246_u32,
      0x3f69a12b_u32, 0x3f718e08_u32, 0x3f79c099_u32, 0x3f812130_u32,
      0x3f858f9f_u32, 0x3f8a331c_u32, 0x3f8f1532_u32, 0x3f94420a_u32,
      0x3f99c91e_u32, 0x3f9fbe05_u32, 0x3fa6394f_u32, 0x3fad597a_u32,
      0x3fb543ff_u32, 0x3fbe2689_u32, 0x3fc83887_u32, 0x3fd3bd81_u32,
      0x3fe10908_u32, 0x3ff085be_u32, 0x4001615b_u32, 0x400c4727_u32,
      0x40199826_u32, 0x402a7bfd_u32, 0x40416dd8_u32, 0x4064c7a3_u32,
    ] of UInt32

    @@reconstruction_luts = {} of Int32 => Array(Float32)

    def encode(values : Array(Float32), block_size : Int32, precision : Int32) : Encoded
      validate_shape(block_size, precision)
      values.each do |value|
        raise ArgumentError.new("QBit state values must be finite") unless value.finite?
      end

      count = values.size.to_i32
      payload = Bytes.new(payload_size(count, block_size, precision), 0_u8)
      return Encoded.new(count, block_size, precision, payload) if count == 0

      plane_bytes = plane_bytes(block_size)
      block_stride = BLOCK_HEADER_BYTES + precision * plane_bytes
      blocks(count, block_size).times do |block|
        value_offset = block * block_size
        block_count = Math.min(block_size, count - value_offset)
        mean, sigma = block_moments(values, value_offset, block_count)
        payload_offset = block * block_stride
        write_f32_le(payload, payload_offset, mean)
        write_f32_le(payload, payload_offset + sizeof(Float32), sigma)

        block_count.times do |within|
          normalized = sigma == 0.0_f32 ? 0.0_f32 : (values[value_offset + within] - mean) / sigma
          raw_code = quantize_raw_code(normalized)
          precision.times do |plane|
            next if (raw_code & (1_u8 << (7 - plane))) == 0

            byte_offset, bit_mask = transposed_position(within, plane_bytes)
            plane_offset = payload_offset + BLOCK_HEADER_BYTES + plane * plane_bytes
            payload[plane_offset + byte_offset] |= bit_mask
          end
        end
      end

      Encoded.new(count, block_size, precision, payload)
    end

    def decode(encoded : Encoded) : Array(Float32)
      validate_shape(encoded.block_size, encoded.precision)
      expected = payload_size(encoded.value_count, encoded.block_size, encoded.precision)
      raise ArgumentError.new("corrupt QBit state payload") unless encoded.payload.size == expected

      values = Array(Float32).new(encoded.value_count, 0.0_f32)
      return values if encoded.value_count == 0

      plane_bytes = plane_bytes(encoded.block_size)
      block_stride = BLOCK_HEADER_BYTES + encoded.precision * plane_bytes
      lut = reconstruction_lut(encoded.precision)
      blocks(encoded.value_count, encoded.block_size).times do |block|
        value_offset = block * encoded.block_size
        block_count = Math.min(encoded.block_size, encoded.value_count - value_offset)
        payload_offset = block * block_stride
        mean = read_f32_le(encoded.payload, payload_offset)
        sigma = read_f32_le(encoded.payload, payload_offset + sizeof(Float32))
        raise ArgumentError.new("corrupt QBit state block moments") unless mean.finite? && sigma.finite? && sigma >= 0.0_f32

        block_count.times do |within|
          raw_code = 0_u8
          byte_offset, bit_mask = transposed_position(within, plane_bytes)
          encoded.precision.times do |plane|
            plane_offset = payload_offset + BLOCK_HEADER_BYTES + plane * plane_bytes
            raw_code |= 1_u8 << (7 - plane) if (encoded.payload[plane_offset + byte_offset] & bit_mask) != 0
          end
          values[value_offset + within] = mean + sigma * lut[raw_code]
        end
      end
      values
    end

    def payload_size(value_count : Int, block_size : Int32, precision : Int32) : Int32
      validate_shape(block_size, precision)
      raise ArgumentError.new("QBit value count must be non-negative") if value_count < 0

      blocks(value_count.to_i32, block_size) * (BLOCK_HEADER_BYTES + precision * plane_bytes(block_size))
    end

    def reconstruct_raw_code(raw_code : UInt8, precision : Int32) : Float32
      validate_precision(precision)
      reconstruction_lut(precision)[raw_code]
    end

    private def validate_shape(block_size : Int32, precision : Int32) : Nil
      raise ArgumentError.new("QBit block size must be positive") unless block_size > 0
      raise ArgumentError.new("QBit block size must be a multiple of 8") unless block_size % 8 == 0
      validate_precision(precision)
    end

    private def validate_precision(precision : Int32) : Nil
      unless precision >= MIN_PRECISION && precision <= MAX_PRECISION
        raise ArgumentError.new("QBit precision must be between #{MIN_PRECISION} and #{MAX_PRECISION}")
      end
    end

    private def plane_bytes(block_size : Int32) : Int32
      (block_size + 7) // 8
    end

    private def blocks(value_count : Int32, block_size : Int32) : Int32
      return 0_i32 if value_count == 0
      (value_count + block_size - 1) // block_size
    end

    private def block_moments(values : Array(Float32), offset : Int32, count : Int32) : {Float32, Float32}
      sum = 0.0_f64
      count.times { |i| sum += values[offset + i].to_f64 }
      mean64 = sum / count
      squared = 0.0_f64
      count.times do |i|
        delta = values[offset + i].to_f64 - mean64
        squared += delta * delta
      end
      {mean64.to_f32, Math.sqrt(squared / count).to_f32}
    end

    private def quantize_raw_code(value : Float32) : UInt8
      negative = value.sign_bit < 0
      magnitude = value.abs
      low = 0
      high = POSITIVE_LEVELS.size - 1
      while low < high
        mid = (low + high) // 2
        boundary = (POSITIVE_LEVELS[mid] + POSITIVE_LEVELS[mid + 1]) * 0.5_f32
        if magnitude <= boundary
          high = mid
        else
          low = mid + 1
        end
      end
      raw = low.to_u8
      negative ? (0xff_u8 - raw) : raw
    end

    private def reconstruction_lut(precision : Int32) : Array(Float32)
      @@reconstruction_luts[precision] ||= build_reconstruction_lut(precision)
    end

    private def build_reconstruction_lut(precision : Int32) : Array(Float32)
      lut = Array(Float32).new(256, 0.0_f32)
      if precision == 8
        128.times do |raw|
          level = bf16_round(POSITIVE_LEVELS[raw])
          lut[raw] = level
          lut[0xff - raw] = -level
        end
        return lut
      end

      suffix_bits = 8 - precision
      prefix_width = 1 << suffix_bits
      positive_prefixes = 128 // prefix_width
      centroid_bits = positive_prefix_centroid_bits(precision)
      positive_prefixes.times do |prefix|
        first = prefix * prefix_width
        last = first + prefix_width - 1
        centroid = centroid_bits[prefix].unsafe_as(Float32)
        first.upto(last) do |raw|
          lut[raw] = centroid
          lut[0xff - raw] = -centroid
        end
      end
      lut
    end

    private def positive_prefix_centroid_bits(precision : Int32) : Array(UInt32)
      precision == 6 ? P6_POSITIVE_PREFIX_CENTROID_BITS : P7_POSITIVE_PREFIX_CENTROID_BITS
    end

    private def bf16_round(value : Float32) : Float32
      bits = value.unsafe_as(UInt32)
      lsb = (bits >> 16) & 1_u32
      rounded = (bits + 0x7fff_u32 + lsb) & 0xffff0000_u32
      rounded.unsafe_as(Float32)
    end

    # Match ClickHouse QBit's per-stride bit ordering so each block can later be
    # mapped to FixedString bit-plane subcolumns without reordering its bytes.
    private def transposed_position(row : Int32, plane_bytes : Int32) : {Int32, UInt8}
      total_bits = plane_bytes * 8
      bit_index = (total_bits - 1) - (row ^ 7)
      {bit_index // 8, (1_u8 << (bit_index % 8))}
    end

    private def read_u32_le(bytes : Bytes, offset : Int32) : UInt32
      bytes[offset].to_u32 |
        (bytes[offset + 1].to_u32 << 8) |
        (bytes[offset + 2].to_u32 << 16) |
        (bytes[offset + 3].to_u32 << 24)
    end

    private def write_u32_le(bytes : Bytes, offset : Int32, value : UInt32) : Nil
      bytes[offset] = (value & 0xff).to_u8
      bytes[offset + 1] = ((value >> 8) & 0xff).to_u8
      bytes[offset + 2] = ((value >> 16) & 0xff).to_u8
      bytes[offset + 3] = ((value >> 24) & 0xff).to_u8
    end

    private def read_f32_le(bytes : Bytes, offset : Int32) : Float32
      read_u32_le(bytes, offset).unsafe_as(Float32)
    end

    private def write_f32_le(bytes : Bytes, offset : Int32, value : Float32) : Nil
      write_u32_le(bytes, offset, value.unsafe_as(UInt32))
    end
  end
end
