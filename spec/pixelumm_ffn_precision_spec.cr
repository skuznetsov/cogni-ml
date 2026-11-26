require "spec"
require "../src/ml/video/pixelumm_decoder_block_mps"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
private def pixelumm_create_precision_without_device(mode : Int32, error : Bytes) : Pointer(Void)
  null = Pointer(Void).null
  handles = Pointer(Pointer(Void)).null
  PixelUMMDecoderBlockMPSFFI.plan_create_with_ffn_f32(
    null, 2, 32, 32, 8, 4096, 12288, 32, 8, 128, 1e-6_f32,
    null, null, null, handles, handles, handles, 0_u64,
    handles, 128, 1152, 16, mode, error.to_unsafe, error.size)
end

describe "PixelUMM FFN graph precision admission" do
  it "rejects a nonboolean mode before inspecting handles or allocating GPU resources" do
    {-1, 2, Int32::MAX}.each do |mode|
      error = Bytes.new(1024, 0_u8)
      pixelumm_create_precision_without_device(mode, error).null?.should be_true
      String.new(error.to_unsafe).should eq("PixelUMM FFN F32 mode must be 0 or 1")
    end
  end

  it "retains binding validation for both admitted modes" do
    {0, 1}.each do |mode|
      error = Bytes.new(1024, 0_u8)
      pixelumm_create_precision_without_device(mode, error).null?.should be_true
      String.new(error.to_unsafe).should eq("unsupported or incomplete PixelUMM plan configuration")
    end
  end

  it "preserves the original C create symbol and its default-zero validation" do
    null = Pointer(Void).null
    handles = Pointer(Pointer(Void)).null
    error = Bytes.new(1024, 0_u8)
    PixelUMMDecoderBlockMPSFFI.plan_create(
      null, 2, 32, 32, 8, 4096, 12288, 32, 8, 128, 1e-6_f32,
      null, null, null, handles, handles, handles, 0_u64,
      handles, 128, 1152, 16, error.to_unsafe, error.size).null?.should be_true
    expected = Bytes.new(1024, 0_u8)
    pixelumm_create_precision_without_device(0, expected).null?.should be_true
    String.new(error.to_unsafe).should eq(String.new(expected.to_unsafe))
  end
end
{% end %}
