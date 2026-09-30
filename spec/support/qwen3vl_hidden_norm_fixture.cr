require "base64"

module Qwen3VLHiddenNormFixture
  DIM       = 4096
  ROW_BYTES = DIM * 2

  def self.small_rows : Hash(String, Bytes)
    sections = {} of String => String
    current = ""
    path = File.join(__DIR__, "../fixtures/qwen3vl_hidden_norm_row50.b64")
    File.read(path).each_line do |line|
      value = line.strip
      next if value.empty?
      if value.starts_with?("@")
        current = value[1..]
        sections[current] = ""
      else
        raise "fixture data appeared before a section label" if current.empty?
        sections[current] += value
      end
    end
    decoded = {} of String => Bytes
    sections.each do |name, encoded|
      decoded[name] = Base64.decode(encoded)
    end
    decoded
  end

  def self.decode_bf16(bytes : Bytes) : Array(Float32)
    raise "BF16 fixture byte count must be even" unless bytes.size.even?
    Array(Float32).new(bytes.size // 2) do |index|
      offset = index * 2
      bits = bytes[offset].to_u32 | (bytes[offset + 1].to_u32 << 8)
      (bits << 16).unsafe_as(Float32)
    end
  end

  def self.encode_bf16(values : Array(Float32)) : Bytes
    bytes = Bytes.new(values.size * 2)
    values.each_with_index do |value, index|
      bits = value.unsafe_as(UInt32) >> 16
      bytes[index * 2] = (bits & 0xff_u32).to_u8
      bytes[index * 2 + 1] = ((bits >> 8) & 0xff_u32).to_u8
    end
    bytes
  end

  def self.mismatch_count(actual : Bytes, expected : Bytes) : Int32
    raise "BF16 byte lengths differ" unless actual.size == expected.size
    mismatches = 0_i32
    (actual.size // 2).times do |index|
      offset = index * 2
      mismatches += 1 if actual[offset] != expected[offset] || actual[offset + 1] != expected[offset + 1]
    end
    mismatches
  end
end
