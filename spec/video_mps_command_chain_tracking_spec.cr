require "spec"
require "./support/video_mps_command_chain_root_tracking"

describe "MPS command-chain root tracking" do
  it "deduplicates by identity without scanning roots through equality" do
    status, retained, equality_calls =
      VideoMPSCommandChainRootTrackingProbe.run(4096)

    status.should eq(0)
    retained.should eq(4097) # 4096 distinct roots plus one equal-valued clone
    equality_calls.should eq(0)
  end
end
