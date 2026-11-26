module ML::Float16Bits
  # Truncate the mantissa and flush values below the normal half range to signed zero.
  # Preserve the GPU upload encoding, including signed infinity for NaN inputs.
  def self.truncate_f32(value : Float32) : UInt16
    bits = value.unsafe_as(UInt32)
    sign = (bits >> 16) & 0x8000_u32
    exponent = ((bits >> 23) & 0xff).to_i32 - 127 + 15
    mantissa = (bits >> 13) & 0x03ff_u32
    if exponent <= 0
      sign.to_u16
    elsif exponent >= 31
      (sign | 0x7c00_u32).to_u16
    else
      (sign | (exponent.to_u32 << 10) | mantissa).to_u16
    end
  end
end
