require "spec"
require "../examples/support/hy_nomas_pretrained_serial_probe"

reference_path = ENV["HY_NOMAS_PRETRAINED_SERIAL_REFERENCE"]?
reference_sha256 = ENV["HY_NOMAS_PRETRAINED_SERIAL_REFERENCE_SHA256"]?

if reference_path && reference_sha256
  describe "optional authenticated two-block Hy-Nomas admission" do
    it "rejects the aggregate budget before any named Metal allocation or readback" do
      conditioning_path = ENV["HY_NOMAS_TEXT_CONDITIONING_REFERENCE"]? ||
                          "/opt/mldata_ssd/cogni-ml/hy-nomas-text-conditioning-20261025/fixtures/text-conditioning-v2.json"
      conditioning_sha256 = HyNomasPretrainedSerialProbe::PINNED_CONDITIONING_SHA

      reference = HyNomasPretrainedSerialProbe::ReferenceFixture.new(
        reference_path, reference_sha256
      )
      qwen_sha256 = HyNomasPretrainedSerialProbe.conditioning_qwen_sha256(
        conditioning_path, conditioning_sha256
      )
      conditioning = HyNomasTextProbe::Fixture.new(
        conditioning_path, conditioning_sha256, qwen_sha256
      )
      reference.validate_conditioning_inputs!(conditioning)
      reference.authenticate_weight_artifacts!(conditioning)

      allocations_before = ML::MetalBuffer.hy_text_probe_allocations
      readbacks_before = ML::MetalBuffer.hy_text_probe_readbacks
      expect_raises(ArgumentError, /independent two-plan control named-byte peak .* exceeds budget 1/) do
        HyNomasPretrainedSerialProbe::RunOwner.new(reference, conditioning, 1_i64)
      end
      ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations_before)
      ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks_before)
    end
  end
end
