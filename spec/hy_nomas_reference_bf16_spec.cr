require "spec"
require "../src/ml/core/buffer"
require "../src/ml/metal/device"
require "../src/ml/video/hy_nomas_mps_rms"

private def hy_nomas_source_bf16_rms_plan_type
  config = uninitialized (ML::Video::HyNomasConfig)
  buffer = uninitialized (ML::MetalBuffer)
  typeof(ML::Video::HyNomasMPSRMS::Plan.new(config, buffer, buffer, buffer,
    ML::Video::WeightDType::BF16, 1, source_bf16: true))
end

private def hy_nomas_spec_bf16_roundtrip(value : Float32) : Float32
  bits = value.unsafe_as(UInt32)
  magnitude = bits & 0x7fffffff_u32
  if magnitude > 0x7f800000_u32
    nan_bf16 = (bits >> 16) | 0x40_u32
    (nan_bf16 << 16).unsafe_as(Float32)
  else
    rounded = bits + 0x7fff_u32 + ((bits >> 16) & 1_u32)
    (rounded & 0xffff0000_u32).unsafe_as(Float32)
  end
end

describe ML::Video::HyNomasReferenceBF16::Plan do
  it "keeps the optional source flag on the MPS RMS plan constructor" do
    hy_nomas_source_bf16_rms_plan_type.should eq(ML::Video::HyNomasMPSRMS::Plan)
  end

  it "validates positive prepared width before Metal pipeline setup" do
    expect_raises(ArgumentError, /width must be positive/) do
      ML::Video::HyNomasReferenceBF16::Plan.new(0)
    end
  end

  it "checks row geometry and Int32 dispatch overflow on the CPU" do
    ML::Video::HyNomasReferenceBF16::Plan.checked_count(3, 128, 128).should eq(384)

    expect_raises(ArgumentError, /dimensions must be positive/) do
      ML::Video::HyNomasReferenceBF16::Plan.checked_count(0, 128, 128)
    end
    expect_raises(ArgumentError, /feature width changed/) do
      ML::Video::HyNomasReferenceBF16::Plan.checked_count(1, 64, 128)
    end
    expect_raises(ArgumentError, /Metal count limit/) do
      ML::Video::HyNomasReferenceBF16::Plan.checked_count(Int32::MAX // 2 + 1, 2, 2)
    end
  end

  it "rejects padded reductions only for source-BF16 RMS" do
    rms_plan = ML::Video::HyNomasMPSRMS::Plan
    rms_plan.checked_reduction_tokens(4, nil, true).should eq(4)
    rms_plan.checked_reduction_tokens(4, 4, true).should eq(4)

    # The default MPS path keeps its existing padded-reduction behavior.
    rms_plan.checked_reduction_tokens(4, 8, false).should eq(8)

    expect_raises(ArgumentError, /requires reduction tokens to match active tokens/) do
      rms_plan.checked_reduction_tokens(4, 8, true)
    end
  end

  it "uses the shared RNE helper and keeps source BF16 operation boundaries ordered" do
    source = ML::Video::HyNomasReferenceBF16::SOURCE
    source.should contain("inline float hy_nomas_bf16_roundtrip_f32(float value)")
    source.should contain("0x7fffu + ((bits >> 16) & 1u)")
    source.should contain("kernel void hy_nomas_reference_bf16_round(")
    source.should contain("#pragma clang fp contract(off)")
    source.should contain("const float factor = hy_nomas_bf16_roundtrip_f32(1.0f + scale[id]);")
    source.should contain("const float product = input[id] * factor;")
    source.should contain("output[id] = product + shift[id];")
    source.should contain("const float normalized_bf16 = hy_nomas_bf16_roundtrip_f32(normalized);")
    source.should contain("hy_nomas_bf16_roundtrip_f32(normalized_bf16 * gain)")
  end

  it "distinguishes BF16 rounding of 1 plus scale from rounding after modulation" do
    factor = hy_nomas_spec_bf16_roundtrip(1.0_f32 + 0.00390625_f32)
    factor.should eq(1.0_f32)

    source_order = 2.0_f32 * factor + 0.25_f32
    late_rounding = 2.0_f32 * (1.0_f32 + 0.00390625_f32) + 0.25_f32
    source_order.should eq(2.25_f32)
    late_rounding.should eq(2.2578125_f32)
    source_order.should_not eq(late_rounding)
  end
end
