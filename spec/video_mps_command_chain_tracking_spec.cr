require "spec"
require "./support/video_mps_command_chain_root_tracking"

describe "MPS command-chain root tracking" do
  it "uses weak pointer identity without scanning roots through equality" do
    status, retained, equality_calls =
      VideoMPSCommandChainRootTrackingProbe.run(4096)

    status.should eq(0)
    retained.should eq(4097) # 4096 roots plus a distinct equal-valued clone
    equality_calls.should eq(0)
  end

  it "releases completed roots and records callback or fallback metrics once" do
    completed = 2048
    status, summaries, pending, peak_pending, errors, callbacks, fallbacks,
      gpu_seconds, released, tracker_released =
        VideoMPSCommandChainRootTrackingProbe.retention(completed)

    status.should eq(0)
    summaries.should eq(completed + 2)
    pending.should eq(0)
    peak_pending.should eq(2)
    errors.should eq(2)
    callbacks.should eq(completed)
    fallbacks.should eq(2)
    gpu_seconds.should eq(completed.to_f64 * 0.25 + 3.5)
    released.should eq(completed)
    tracker_released.should eq(1)
  end
end
