require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_adaptive_kv"

module QwenQBitAdaptiveKVSpec
  extend self

  def deterministic_values(rows : Int32) : Array(Float32)
    Array(Float32).new(rows * 256) do |i|
      (((i * 37) % 257) - 128).to_f32 / 91.0_f32
    end
  end

  def bf16_reference(values : Array(Float32)) : Array(Float32)
    values.map do |value|
      bits = value.unsafe_as(UInt32)
      lsb = (bits >> 16) & 1_u32
      ((bits + 0x7fff_u32 + lsb) & 0xffff0000_u32).unsafe_as(Float32)
    end
  end

  def assert_close(actual : Array(Float32), expected : Array(Float32), tolerance : Float32)
    actual.size.should eq(expected.size)
    actual.each_with_index do |value, i|
      (value - expected[i]).abs.should be <= tolerance
    end
  end

  def write_u16(bytes : Bytes, offset : Int32, value : UInt16) : Nil
    bytes[offset] = (value & 0xff_u16).to_u8
    bytes[offset + 1] = (value >> 8).to_u8
  end

  def write_u32(bytes : Bytes, offset : Int32, value : UInt32) : Nil
    bytes[offset] = (value & 0xff_u32).to_u8
    bytes[offset + 1] = ((value >> 8) & 0xff_u32).to_u8
    bytes[offset + 2] = ((value >> 16) & 0xff_u32).to_u8
    bytes[offset + 3] = ((value >> 24) & 0xff_u32).to_u8
  end
end

describe ML::GGUF::QwenQBitAdaptiveKV do
  codec = ML::GGUF::QwenQBitAdaptiveKV
  tier = ML::GGUF::QwenQBitAdaptiveKV::Tier

  it "reconstructs mixed p4, p5, BF16, and F32 rows from canonical references" do
    values = QwenQBitAdaptiveKVSpec.deterministic_values(4)
    tiers = [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
             ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
             ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16,
             ML::GGUF::QwenQBitAdaptiveKV::Tier::F32]
    encoded = codec.encode(values, tiers)

    p4 = ML::GGUF::QwenQBitGaussianCodec.decode(
      ML::GGUF::QwenQBitGaussianCodec.encode(values[0, 256], 256, 4)
    )
    p5 = ML::GGUF::QwenQBitGaussianCodec.decode(
      ML::GGUF::QwenQBitGaussianCodec.encode(values[256, 256], 256, 5)
    )
    bf16 = QwenQBitAdaptiveKVSpec.bf16_reference(values[512, 256])
    f32 = values[768, 256]
    expected = p4 + p5 + bf16 + f32

    QwenQBitAdaptiveKVSpec.assert_close(codec.decode(encoded), expected, 1.0e-6_f32)
    tiers.each_with_index do |expected_tier, row|
      codec.tier(encoded, row).should eq(expected_tier)
    end
    [0_u32, 0_u32, 32_u32, 544_u32].each_with_index do |expected_offset, row|
      codec.sidecar_offset(encoded, row).should eq(expected_offset)
    end
    regions = codec.regions(encoded)
    regions.base.size.should eq(4 * ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES)
    regions.metadata.size.should eq(4 * ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES)
    regions.sidecar.size.should eq(544 + ML::GGUF::QwenQBitAdaptiveKV::F32_SIDECAR_BYTES)
  end

  it "accounts for uniform tiers and a sparse approximately five-bit mixture exactly" do
    values = QwenQBitAdaptiveKVSpec.deterministic_values(8)
    base_and_metadata = ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES +
                        ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES

    [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
     ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
     ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16,
     ML::GGUF::QwenQBitAdaptiveKV::Tier::F32].each do |selected|
      tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(8, selected)
      encoded = codec.encode(values, tiers)
      expected = 8 * base_and_metadata + 8 * codec.sidecar_size(selected)
      encoded.payload_bytes.should eq(expected)
      codec.payload_bytes(values.size, tiers).should eq(expected)
    end

    sparse_tiers = Array(ML::GGUF::QwenQBitAdaptiveKV::Tier).new(8, ML::GGUF::QwenQBitAdaptiveKV::Tier::P4)
    sparse_tiers[-1] = ML::GGUF::QwenQBitAdaptiveKV::Tier::P5
    sparse = codec.encode(values, sparse_tiers)
    sparse.payload_bytes.should eq(8 * base_and_metadata + ML::GGUF::QwenQBitAdaptiveKV::P5_SIDECAR_BYTES)
    bits_per_value = sparse.payload_bytes.to_f64 * 8.0 / values.size
    bits_per_value.should be > 4.5
    bits_per_value.should be < 5.0
  end

  it "plans canonical metadata and exact prefix sidecar capacity without value payloads" do
    tiers = [
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::P5,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16,
      ML::GGUF::QwenQBitAdaptiveKV::Tier::F32,
    ]
    plan = codec.plan(tiers)

    plan.row_count.should eq(4)
    plan.base_bytes.should eq(4 * ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES)
    plan.metadata_bytes.should eq(4 * ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES)
    plan.sidecar_bytes.should eq(32 + 512 + 1024)
    plan.payload_bytes.should eq(plan.base_bytes + plan.metadata_bytes + plan.sidecar_bytes)
    [0, 0, 32, 544, 1568].each_with_index do |expected, prefix_rows|
      plan.prefix_sidecar_bytes(prefix_rows).should eq(expected)
    end
    mutable_copy = plan.metadata
    mutable_copy[0] = 0xff_u8
    plan.metadata[0].should eq(ML::GGUF::QwenQBitAdaptiveKV::Tier::P4.value.to_u8)

    encoded = codec.empty_encoded(plan, 3)
    encoded.value_count.should eq(3 * 256)
    encoded.payload_bytes.should eq(
      3 * (ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES +
           ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES) + 544,
    )
    regions = codec.regions(encoded)
    regions.metadata.should eq(plan.metadata[0, 3 * ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES])
  end

  it "rejects invalid adaptive plan prefixes" do
    plan = codec.plan([ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    expect_raises(ArgumentError, /prefix/) { plan.prefix_sidecar_bytes(-1) }
    expect_raises(ArgumentError, /prefix/) { plan.prefix_sidecar_bytes(2) }
    expect_raises(ArgumentError, /prefix/) { codec.empty_encoded(plan, 2) }
  end

  it "rejects non-row-aligned input and tier-count mismatches" do
    expect_raises(ArgumentError, /256/) do
      codec.encode(Array(Float32).new(255, 0.0_f32), [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    end
    expect_raises(ArgumentError, /tier count/) do
      codec.encode(QwenQBitAdaptiveKVSpec.deterministic_values(2), [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4])
    end
    expect_raises(ArgumentError, /block size/) do
      codec.encode(QwenQBitAdaptiveKVSpec.deterministic_values(1), [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4], block_size: 128)
    end
  end

  it "fails closed on malformed metadata, offsets, bounds, and sidecar consumption" do
    values = QwenQBitAdaptiveKVSpec.deterministic_values(2)
    valid = codec.encode(values, [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
                                  ML::GGUF::QwenQBitAdaptiveKV::Tier::F32])
    metadata_offset = 2 * ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES

    invalid_tier = Bytes.new(valid.payload.size, 0_u8)
    invalid_tier.copy_from(valid.payload)
    QwenQBitAdaptiveKVSpec.write_u32(invalid_tier, metadata_offset, 99_u32)
    expect_raises(ArgumentError, /tier/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(512, 256, invalid_tier))
    end

    invalid_offset = Bytes.new(valid.payload.size, 0_u8)
    invalid_offset.copy_from(valid.payload)
    QwenQBitAdaptiveKVSpec.write_u32(invalid_offset, metadata_offset + ML::GGUF::QwenQBitAdaptiveKV::TIER_BYTES, 1_u32)
    expect_raises(ArgumentError, /canonical|offset/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(512, 256, invalid_offset))
    end

    truncated_metadata = valid.payload[0, metadata_offset + ML::GGUF::QwenQBitAdaptiveKV::TIER_BYTES]
    expect_raises(ArgumentError, /metadata|payload/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(512, 256, truncated_metadata))
    end

    truncated_sidecar = valid.payload[0, valid.payload.size - 1]
    expect_raises(ArgumentError, /sidecar|payload/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(512, 256, truncated_sidecar))
    end

    trailing = Bytes.new(valid.payload.size + 1, 0_u8)
    trailing[0, valid.payload.size].copy_from(valid.payload)
    expect_raises(ArgumentError, /consumption|payload/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(512, 256, trailing))
    end
  end

  it "rejects non-finite BF16 and F32 replacement values" do
    values = QwenQBitAdaptiveKVSpec.deterministic_values(1)
    bf16 = codec.encode(values, [ML::GGUF::QwenQBitAdaptiveKV::Tier::BF16])
    bf16_bad = Bytes.new(bf16.payload.size, 0_u8)
    bf16_bad.copy_from(bf16.payload)
    bf16_sidecar = 1 * (ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES +
                        ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES) + codec.sidecar_offset(bf16, 0)
    QwenQBitAdaptiveKVSpec.write_u16(bf16_bad, bf16_sidecar, 0x7fc1_u16)
    expect_raises(ArgumentError, /finite/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(256, 256, bf16_bad))
    end

    f32 = codec.encode(values, [ML::GGUF::QwenQBitAdaptiveKV::Tier::F32])
    f32_bad = Bytes.new(f32.payload.size, 0_u8)
    f32_bad.copy_from(f32.payload)
    f32_sidecar = 1 * (ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES +
                       ML::GGUF::QwenQBitAdaptiveKV::METADATA_BYTES) + codec.sidecar_offset(f32, 0)
    QwenQBitAdaptiveKVSpec.write_u32(f32_bad, f32_sidecar, Float32::INFINITY.unsafe_as(UInt32))
    expect_raises(ArgumentError, /finite/) do
      codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(256, 256, f32_bad))
    end
  end

  it "rejects finite moments that overflow p4 or p5 reconstruction" do
    values = QwenQBitAdaptiveKVSpec.deterministic_values(2)
    encoded = codec.encode(values, [ML::GGUF::QwenQBitAdaptiveKV::Tier::P4,
                                    ML::GGUF::QwenQBitAdaptiveKV::Tier::P5])

    [0, 1].each do |row|
      malformed = Bytes.new(encoded.payload.size, 0_u8)
      malformed.copy_from(encoded.payload)
      base_offset = row * ML::GGUF::QwenQBitAdaptiveKV::BASE_ROW_BYTES
      QwenQBitAdaptiveKVSpec.write_u32(malformed, base_offset, Float32::MAX.unsafe_as(UInt32))
      QwenQBitAdaptiveKVSpec.write_u32(malformed, base_offset + sizeof(Float32), Float32::MAX.unsafe_as(UInt32))

      expect_raises(ArgumentError, /finite reconstruction/) do
        codec.validate(ML::GGUF::QwenQBitAdaptiveKV::Encoded.new(512, 256, malformed))
      end
    end
  end
end
