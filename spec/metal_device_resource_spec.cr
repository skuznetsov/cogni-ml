require "./spec_helper"
require "../src/ml/metal/device"

describe ML::Metal::Device do
  it "reports the device working-set and current allocation sizes" do
    pending!("Metal not available") unless ML::Metal::Device.available?

    device = ML::Metal::Device.instance
    device.recommended_working_set_size.should be > 0_i64
    device.current_allocated_size.should be >= 0_i64
  end
end
