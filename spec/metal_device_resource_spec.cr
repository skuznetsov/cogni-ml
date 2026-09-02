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

describe ML::Metal::CommandBuffer do
  it "completes an empty command buffer through the bounded wait path" do
    pending!("Metal not available") unless ML::Metal::Device.available?

    command = ML::Metal::CommandBuffer.new
    command.commit_and_wait
    command.completed_successfully?.should be_true
  end

  it "discards an uncommitted native command handle exactly once" do
    pending!("Metal not available") unless ML::Metal::Device.available?

    command = ML::Metal::CommandBuffer.new
    command.handle.null?.should be_false

    command.discard
    command.handle.null?.should be_true
    command.discard
  end

  it "captures optional GPU timing while waiting for an already committed command" do
    pending!("Metal not available") unless ML::Metal::Device.available?

    command = ML::Metal::CommandBuffer.new
    expect_raises(ArgumentError, /uncommitted/) do
      command.wait_gpu_elapsed_seconds?
    end

    command.commit
    elapsed = command.wait_gpu_elapsed_seconds?
    command.completed_successfully?.should be_true
    elapsed.try { |seconds| seconds.should be > 0.0_f64 }

    # Completion is idempotent and must not touch the released native handle.
    command.wait_gpu_elapsed_seconds?.should eq(elapsed)
  end
end
