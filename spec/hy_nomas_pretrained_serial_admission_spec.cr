require "spec"
require "../examples/support/hy_nomas_pretrained_serial_probe"

describe HyNomasPretrainedSerialProbe do
  it "rejects a named-byte admission underflow before setup can allocate" do
    expect_raises(ArgumentError) do
      HyNomasPretrainedSerialProbe.admit_named_peak!("shared workspace", 101_i64, 100_i64)
    end
  end

  it "admits an exact named-byte budget without slack arithmetic" do
    HyNomasPretrainedSerialProbe.admit_named_peak!("shared workspace", 100_i64, 100_i64)
  end
end
