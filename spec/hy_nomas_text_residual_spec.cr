require "spec"
require "../src/ml/video/hy_nomas_text_refiner"
require "../examples/support/hy_nomas_probe_instrumentation"
require "../examples/support/qwen25_prefix_probe"

describe "Hy-Nomas text refiner residual" do
  it "rounds the broadcast gate product before adding the F32 residual" do
    arena = ML::Video::Arena.new
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      pipeline = ML::Metal::ComputePipeline.new(
        ML::Video::HyNomasTextRefinerMPSPlan::RESIDUAL_KERNEL,
        ML::Video::HyNomasTextRefinerMPSPlan::SOURCE)
      # The exact product is slightly above one but rounds to one in F32.
      # A fused multiply-add would instead preserve a nonzero cancellation.
      arena.upload("input", [-1.0_f32, -1.0_f32, 4.0_f32, -4.0_f32])
      arena.upload("delta", [1.00000011920928955078125_f32,
                             1.00000011920928955078125_f32, 1.0_f32, 1.0_f32])
      arena.upload("gate", [0.999999940395355224609375_f32,
                            0.999999940395355224609375_f32])
      arena.alloc("output", 4)
      chain = ML::Metal::MPSCommandChain.new
      allocations = ML::MetalBuffer.hy_text_probe_allocations
      readbacks = ML::MetalBuffer.hy_text_probe_readbacks
      chain.encoder.set_pipeline(pipeline)
        .set_buffer(arena["input"], 0).set_buffer(arena["delta"], 1)
        .set_buffer(arena["gate"], 2)
        .set_buffer(arena["output"], 3, ML::Metal::BufferAccess::Write)
        .set_value(4_u32, 4).set_value(2_u32, 5).dispatch_1d(4)
      ML::MetalBuffer.hy_text_probe_allocations.should eq(allocations)
      ML::MetalBuffer.hy_text_probe_readbacks.should eq(readbacks)
      chain.wait_count.should eq(0)
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      arena["output"].read(4).map(&.unsafe_as(UInt32)).should eq(
        [0.0_f32, 0.0_f32, 5.0_f32, -3.0_f32].map(&.unsafe_as(UInt32)))
    ensure
      Qwen25PrefixProbe.cleanup_after_drain(chain) do
        chain.try(&.release)
        arena.release
      end
    end
  end
end
