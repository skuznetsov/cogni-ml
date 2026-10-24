require "spec"
require "../src/ml/video/metal_ops"
require "../src/ml/video/metal_mps_command_chain"
require "./support/video_allocation_counter"

lib MPSCommandChainScopeProbeFFI
  struct Scope
    owner_data : Void*
    start_wall_ns : UInt64
    start_thread_cpu_ns : UInt64
    serial : UInt64
    parent_graph_serial : UInt64
    operation : UInt32
    phase : UInt32
    active : UInt32
    clocks_valid : UInt32
    nested_in_graph : UInt32
    reserved : UInt32
  end

  fun begin = gs_mps_api_profile_scope_begin(operation : Int32, command_context : Void*, scope : Scope*) : Void
  fun finish = gs_mps_api_profile_scope_end(scope : Scope*) : Void
end

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
      c.observed_root_count.should eq(1)
      expect_raises(ArgumentError) { c.completed_root_timings }
      expect_raises(ArgumentError) { c.handle }
      expect_raises(ArgumentError) { c.resume! }
      expect_raises(ArgumentError) { c.release }
      { {source, a}, {a, b}, {b, output} }.each_with_index do |buffers, i|
        e = c.encoder
        e.set_pipeline(pipeline).set_buffer(buffers[0], 0).set_buffer(buffers[1], 1).dispatch_1d(257)
        # Observing roots must keep the existing encoder usable and create no
        # extra root or wait, even while native work is being encoded.
        borrowed_encoder = c.encoder
        c.observed_root_count.should eq(i + 1)
        c.encoder.should eq(borrowed_encoder)
        c.wait_count.should eq(0)
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
      roots = c.completed_root_timings
      roots.map(&.sequence).should eq([0, 1, 2])
      roots.all?(&.completed).should be_true
      roots.any?(&.has_error).should be_false
      roots.sum { |root| root.gpu_end_seconds - root.gpu_start_seconds }.should eq(c.gpu_elapsed_seconds)
      c.completed_root_timings.should eq(roots)
      expect_raises(ArgumentError) { c.observed_root_count }
      output.read(257).should eq(Array(Float32).new(257) { |i| i.to_f32 + 3.0_f32 })
      expect_raises(ArgumentError) { c.resume! }
      expect_raises(ArgumentError) { c.handle }
      expect_raises(ArgumentError) { c.continue_async! }
      c.release
      expect_raises(ArgumentError) { c.completed_root_timings }
      c.release
    ensure
      if c = chain
        c.commit_and_wait unless c.completed?
        c.release
      end
      arena.release
    end
  end

  it "profiles existing roots after one terminal wait without changing native output or allocations" do
    arena = ML::Video::Arena.new
    chain = nil.as(ML::Metal::MPSCommandChain?)
    begin
      source = arena.upload("profile_source", Array(Float32).new(4097) { |i| i.to_f32 })
      a = arena.alloc("profile_a", 4097)
      b = arena.alloc("profile_b", 4097)
      output = arena.alloc("profile_output", 4097)
      pipeline = ML::Metal::ComputePipeline.new("mps_chain_profile_increment", <<-METAL)
        #include <metal_stdlib>
        using namespace metal;
        kernel void mps_chain_profile_increment(device const float *x [[buffer(0)]],
                                                device float *y [[buffer(1)]],
                                                uint i [[thread_position_in_grid]]) {
          if (i < 4097) y[i] = x[i] + 1.0f;
        }
        METAL
      expected = Array(Float32).new(4097) { |i| i.to_f32 + 4.0_f32 }
      baseline = nil.as(Array(Float32)?)
      names = {"self_prepare", "self_attention_projection_residual",
               "cross_attention_projection_residual"}

      {false, true}.each do |profiled|
        before = ML::MetalBuffer.video_spec_allocations
        c = ML::Metal::MPSCommandChain.new
        chain = c
        c.enable_phase_root_profile! if profiled
        { {source, a}, {a, b}, {b, a}, {a, output} }.each_with_index do |buffers, i|
          c.encoder.set_pipeline(pipeline).set_buffer(buffers[0], 0)
            .set_buffer(buffers[1], 1).dispatch_1d(4097)
          if i < 3
            end_count = profiled ? c.record_phase_root_boundary!(names[i]) : nil
            c.pause!
            c.continue_async!
            c.resume!
            c.verify_phase_root_continuation!(end_count.not_nil!) if profiled
          end
        end
        c.finish_phase_root_profile! if profiled
        expect_raises(ArgumentError) { c.phase_root_profile }
        c.wait_count.should eq(0)
        ML::MetalBuffer.video_spec_allocations.should eq(before)
        # There is deliberately no same-root caller suffix before the sole wait.
        c.commit_and_wait
        c.wait_count.should eq(1)
        c.segment_count.should eq(4)
        ML::MetalBuffer.video_spec_allocations.should eq(before)
        if profiled
          profile = c.phase_root_profile
          profile.root_count.should eq(4)
          profile.root_duration_seconds.should eq(c.gpu_elapsed_seconds)
          profile.phase_union_seconds.keys.should eq(names.to_a + ["ffn"])
          profile.phase_union_seconds.values.each { |seconds| seconds.should be > 0.0 }
          profile.union_seconds.should be > 0.0
          profile.envelope_seconds.should be >= profile.union_seconds
          profile.overlap_seconds.should be >= 0.0
          c.wait_count.should eq(1)
          ML::MetalBuffer.video_spec_allocations.should eq(before)
        end
        actual = output.read(4097)
        actual.should eq(expected)
        if profiled
          actual.map(&.unsafe_as(UInt32)).should eq(baseline.not_nil!.map(&.unsafe_as(UInt32)))
        else
          baseline = actual
        end
        c.release
        chain = nil
      end
    ensure
      if c = chain
        c.commit_and_wait unless c.completed?
        c.release
      end
      arena.release
    end
  end

  it "keeps a pre-wait profile scope rejection retryable until the native chain drains" do
    chain = nil.as(ML::Metal::MPSCommandChain?)
    scope = MPSCommandChainScopeProbeFFI::Scope.new
    scope_open = false
    begin
      c = ML::Metal::MPSCommandChain.new
      chain = c
      c.pause!
      c.enable_api_profile!(phase_roots: false)

      # Reproduce an unclosed graph API scope at the terminal-wait boundary.
      MPSCommandChainScopeProbeFFI.begin(0, c.handle, pointerof(scope))
      scope.active.should eq(1_u32)
      scope_open = true

      expect_raises(Exception, /-5/) { c.commit_and_wait }
      c.drained?.should be_false
      c.completed?.should be_false
      c.wait_count.should eq(0)
      expect_raises(ArgumentError) { c.release }

      MPSCommandChainScopeProbeFFI.finish(pointerof(scope))
      scope_open = false
      scope.active.should eq(0_u32)

      c.commit_and_wait
      c.drained?.should be_true
      c.completed?.should be_true
      c.wait_count.should eq(1)
      c.release
      chain = nil
    ensure
      MPSCommandChainScopeProbeFFI.finish(pointerof(scope)) if scope_open
      if c = chain
        c.commit_and_wait unless c.drained?
        c.release
      end
    end
  end
end
