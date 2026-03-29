# Dequantization routines for GGUF quantized tensors.
# Ported from llama.cpp ggml-quants.c
#
# Supports: F32, F16, Q5_K, Q6_K (used by nomic-embed-text-v2-moe.Q5_K_M)
# QK_K = 256 elements per super-block

module ML::GGUF::Dequant
  QK_K         = 256
  K_SCALE_SIZE =  12

  # Main entry: dequantize raw bytes to Float32 array
  def self.dequantize(data : Bytes, type : TensorType, n_elements : Int32) : Array(Float32)
    case type
    when .f32?  then dequantize_f32(data, n_elements)
    when .f16?  then dequantize_f16(data, n_elements)
    when .q5_k? then dequantize_q5_k(data, n_elements)
    when .q6_k? then dequantize_q6_k(data, n_elements)
    else
      raise "Unsupported dequantization type: #{type.name}"
    end
  end

  # F32: direct copy
  def self.dequantize_f32(data : Bytes, n : Int32) : Array(Float32)
    result = Array(Float32).new(n, 0.0_f32)
    n.times do |i|
      result[i] = IO::ByteFormat::LittleEndian.decode(Float32, data[i * 4, 4])
    end
    result
  end

  # F16: convert half-float to float32
  def self.dequantize_f16(data : Bytes, n : Int32) : Array(Float32)
    result = Array(Float32).new(n, 0.0_f32)
    n.times do |i|
      result[i] = fp16_to_f32(data[i * 2, 2])
    end
    result
  end

  # Q5_K: 5.5 bits per weight, QK_K=256 block, 176 bytes/block
  # Block layout: [d:f16][dmin:f16][scales:12B][qh:32B][qs:128B]
  def self.dequantize_q5_k(data : Bytes, n : Int32) : Array(Float32)
    nb = n // QK_K
    result = Array(Float32).new(n, 0.0_f32)
    block_size = 176  # 2+2+12+32+128
    yi = 0

    nb.times do |i|
      off = i * block_size
      d = fp16_to_f32(data[off, 2])
      dmin = fp16_to_f32(data[off + 2, 2])
      scales_ptr = (data.to_unsafe + off + 4)    # 12 bytes
      qh_ptr = (data.to_unsafe + off + 4 + 12)   # 32 bytes
      ql_ptr = (data.to_unsafe + off + 4 + 12 + 32) # 128 bytes

      u1 = 1_u8
      u2 = 2_u8
      is = 0

      4.times do |_j|  # 4 iterations × 64 elements = 256
        sc, m = get_scale_min_k4(is, scales_ptr)
        d1 = d * sc
        m1 = dmin * m
        sc2, m2_val = get_scale_min_k4(is + 1, scales_ptr)
        d2 = d * sc2
        m2 = dmin * m2_val

        32.times do |l|
          q_low = ql_ptr[l] & 0x0F
          q_high = (qh_ptr[l] & u1) != 0 ? 16 : 0
          result[yi] = d1 * (q_low + q_high) - m1
          yi += 1
        end
        32.times do |l|
          q_low = (ql_ptr[l] >> 4) & 0x0F
          q_high = (qh_ptr[l] & u2) != 0 ? 16 : 0
          result[yi] = d2 * (q_low + q_high) - m2
          yi += 1
        end

        ql_ptr += 32
        is += 2
        u1 = u1 << 2
        u2 = u2 << 2
      end
    end

    result
  end

  # Q6_K: 6.5625 bits per weight, QK_K=256 block, 210 bytes/block
  # Block layout: [ql:128B][qh:64B][scales:16B][d:f16]
  def self.dequantize_q6_k(data : Bytes, n : Int32) : Array(Float32)
    nb = n // QK_K
    result = Array(Float32).new(n, 0.0_f32)
    block_size = 210  # 128+64+16+2
    yi = 0

    nb.times do |i|
      off = i * block_size
      ql_ptr = data.to_unsafe + off               # 128 bytes (QK_K/2)
      qh_ptr = data.to_unsafe + off + 128         # 64 bytes (QK_K/4)
      sc_ptr = data.to_unsafe + off + 128 + 64    # 16 bytes (QK_K/16) — signed int8
      d = fp16_to_f32(data[off + 128 + 64 + 16, 2])

      2.times do |_n|  # 2 iterations × 128 elements = 256
        32.times do |l|
          is = l // 16
          q1 = ((ql_ptr[l].to_i32 & 0xF) | (((qh_ptr[l].to_i32 >> 0) & 3) << 4)) - 32
          q2 = ((ql_ptr[l + 32].to_i32 & 0xF) | (((qh_ptr[l].to_i32 >> 2) & 3) << 4)) - 32
          q3 = ((ql_ptr[l].to_i32 >> 4) | (((qh_ptr[l].to_i32 >> 4) & 3) << 4)) - 32
          q4 = ((ql_ptr[l + 32].to_i32 >> 4) | (((qh_ptr[l].to_i32 >> 6) & 3) << 4)) - 32

          sc0 = sc_ptr[is].unsafe_as(Int8).to_f32
          sc2 = sc_ptr[is + 2].unsafe_as(Int8).to_f32
          sc4 = sc_ptr[is + 4].unsafe_as(Int8).to_f32
          sc6 = sc_ptr[is + 6].unsafe_as(Int8).to_f32

          result[yi + l]      = d * sc0 * q1
          result[yi + l + 32] = d * sc2 * q2
          result[yi + l + 64] = d * sc4 * q3
          result[yi + l + 96] = d * sc6 * q4
        end
        yi += 128
        ql_ptr += 64
        qh_ptr += 32
        sc_ptr += 8
      end
    end

    result
  end

  # Extract 6-bit scale and min from packed scales array (K_SCALE_SIZE=12 bytes)
  # Ported from get_scale_min_k4() in ggml-quants.c
  def self.get_scale_min_k4(j : Int32, scales : Pointer(UInt8)) : {Float32, Float32}
    if j < 4
      sc = (scales[j] & 63).to_f32
      m = (scales[j + 4] & 63).to_f32
    else
      sc = ((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)).to_f32
      m = ((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)).to_f32
    end
    {sc, m}
  end

  # Convert IEEE 754 half-precision (2 bytes LE) to Float32
  def self.fp16_to_f32(bytes : Bytes) : Float32
    h = IO::ByteFormat::LittleEndian.decode(UInt16, bytes)
    fp16_to_f32(h)
  end

  def self.fp16_to_f32(h : UInt16) : Float32
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x03FF

    if exp == 0
      # Subnormal or zero
      if mant == 0
        return sign == 1 ? -0.0_f32 : 0.0_f32
      end
      # Subnormal
      f = mant.to_f32 / 1024.0_f32
      f *= (2.0_f32 ** -14)
      return sign == 1 ? -f : f
    elsif exp == 31
      # Inf or NaN
      if mant == 0
        return sign == 1 ? Float32::INFINITY * -1 : Float32::INFINITY
      else
        return Float32::NAN
      end
    end

    # Normal
    f = (1.0_f32 + mant.to_f32 / 1024.0_f32) * (2.0_f32 ** (exp.to_i32 - 15))
    sign == 1 ? -f : f
  end
end
