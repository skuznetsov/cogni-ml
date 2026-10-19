require "spec"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_mps_command_chain"
require "./support/video_allocation_counter"

describe ML::Metal::MPSCommandChain do
  it "orders native work across continued roots with one terminal wait" do
    arena = ML::Video::Arena.new
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      source = arena.upload("source", Array(Float32).new(257) { |i| i.to_f32 })
      a = arena.alloc("a", 257)
      b = arena.alloc("b", 257)
      output = arena.alloc("output", 257)
      pipeline = ML::Metal::ComputePipeline.new("mps_chain_increment", <<-METAL)
        #include <metal_stdlib>
        using namespace metal;
        kernel void mps_chain_increment(device const float *x [[buffer(0)]],
                                        device float *y [[buffer(1)]],
                                        uint i [[thread_position_in_grid]]) {
          if (i < 257) y[i] = x[i] + 1.0f;
        }
        METAL
      before = ML::MetalBuffer.video_spec_allocations
      c = ML::Metal::MPSCommandChain.new
      chain = c
      expect_raises(ArgumentError) { c.handle }
      expect_raises(ArgumentError) { c.resume! }
      expect_raises(ArgumentError) { c.release }
      { {source, a}, {a, b}, {b, output} }.each_with_index do |buffers, i|
        e = c.encoder
        e.set_pipeline(pipeline).set_buffer(buffers[0], 0).set_buffer(buffers[1], 1).dispatch_1d(257)
        c.pause!
        expect_raises(ArgumentError) { c.encoder }
        expect_raises(ArgumentError) { c.pause! }
        unless i == 2
          c.continue_async!
          c.resume!
        end
      end
      c.wait_count.should eq(0)
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      c.commit_and_wait
      c.commit_and_wait
      c.wait_count.should eq(1)
      c.segment_count.should eq(3)
      c.gpu_elapsed_seconds.finite?.should be_true
      c.gpu_elapsed_seconds.should be > 0.0
      output.read(257).should eq(Array(Float32).new(257) { |i| i.to_f32 + 3.0_f32 })
      expect_raises(ArgumentError) { c.resume! }
      expect_raises(ArgumentError) { c.handle }
      expect_raises(ArgumentError) { c.continue_async! }
      c.release
      c.release
    ensure
      if c = chain
        c.commit_and_wait unless c.completed?
        c.release
      end
      arena.release
    end
  end
end
