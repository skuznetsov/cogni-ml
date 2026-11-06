require "spec"
require "../examples/support/hy_nomas_soft_route_render"

describe HyNomasSoftRouteRender do
  it "admits four ordered staged-weight passes and rejects the fifth on one chain" do
    HyNomasSoftRouteRender::OUTER_STEPS.should eq(4)
    HyNomasSoftRouteRender::WEIGHT_STAGE_PASSES_PER_CHAIN.should eq(4)
    HyNomasSoftRouteRender::LOGICAL_FULL_FORWARD_CALLS.should eq(16)

    contract = HyNomasSoftRouteRender.validate_stack_cursor_lifecycle!
    contract[:weight_stage_passes].should eq(4)
    contract[:logical_full_forward_calls].should eq(16)
    contract[:blocks_per_pass].should eq(54)
    contract[:fifth_pass_rejected].should be_true
  end
end
