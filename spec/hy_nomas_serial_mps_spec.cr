require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_serial_mps"

# A deliberately different successor catches a shared plan accidentally using
# its first block's projection or RMS weights. The pinned upstream fixture
# separately qualifies the actual checkpoint block applied twice.
describe ML::Video::HyNomasMPSSerialWorkspace do
  it "matches separate plans with different successor weights and one terminal wait" do
    c = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 6, text_tokens: 3,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256,
      attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl)
    arenas = [] of ML::Video::Arena
    workspaces = [] of ML::Video::HyNomasMPSWorkspace
    chains = [] of ML::Metal::MPSCommandChain
    serial = nil.as(ML::Video::HyNomasMPSSerialWorkspace?)
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
    begin
      weights = Array(ML::Video::Arena).new(2) do |block_index|
        arena = ML::Video::Arena.new
        arenas << arena
        ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(c).each_with_index do |binding, tensor|
          count = binding.shape.reduce(1_i64) { |n, d| n * d }.to_i32
          values = Array(Float32).new(count) do |i|
            if binding.target.includes?("norm_weight")
              1.0_f32 + block_index * 0.125_f32 + (i % 7) * 0.003_f32
            else
              Math.sin(i * 0.17 + tensor + block_index * 0.31).to_f32 * 0.04_f32
            end
          end
          arena.upload(binding.target, values)
        end
        arena
      end
      inputs = ML::Video::Arena.new
      arenas << inputs
      {"img" => c.image_rows * c.dim, "txt" => c.text_rows * c.dim,
       "vec" => c.dim, "vec_txt" => c.dim}.each_with_index do |(name, count), salt|
        inputs.upload(name, Array(Float32).new(count) { |i| Math.sin(i * 0.13 + salt).to_f32 * 0.25_f32 })
      end
      inputs.upload("cos", Array(Float32).new(c.video_tokens * c.head_dim, 1.0_f32))
      inputs.upload("sin", Array(Float32).new(c.video_tokens * c.head_dim, 0.0_f32))
      controls = weights.map do |weight|
        scratch = ML::Video::Arena.new
        arenas << scratch
        ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, c)
        ML::Video::HyNomasDualStreamBlock.new(c, weight, scratch)
      end
      controls.each_with_index do |block, index|
        img = index == 0 ? inputs["img"] : controls[0].image_output
        txt = index == 0 ? inputs["txt"] : controls[0].text_output
        workspaces << ML::Video::HyNomasMPSWorkspace.new(block, img, txt,
          inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"],
          attention_query_chunk: 9, attention_heads_per_group: c.heads)
      end
      expect_raises(ArgumentError, /intermediate/) do
        workspaces[0].register_serial_blocks!(controls,
          [{inputs["img"], inputs["txt"]}, {controls[0].image_output, controls[0].text_output}])
      end
      serial = ML::Video::HyNomasMPSSerialWorkspace.new(c, weights,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["cos"], inputs["sin"], attention_query_chunk: 9, attention_heads_per_group: c.heads)
      allocations = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /exactly two/) do
        ML::Video::HyNomasMPSSerialWorkspace.new(c, [weights[0]],
          inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"])
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      retained = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
      expect_raises(ArgumentError, /alias/) do
        ML::Video::HyNomasMPSSerialWorkspace.new(c, weights,
          inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["cos"],
          attention_query_chunk: 9, attention_heads_per_group: c.heads)
      end
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(retained)
      serial.blocks[0].image_output.same?(serial.blocks[1].image_output).should be_false
      serial.blocks[0].text_output.same?(serial.blocks[1].text_output).should be_false
      serial.blocks[0].checkpoints.each do |name, buffer|
        next if {"img.output", "txt.output"}.includes?(name)
        buffer.same?(serial.blocks[1].checkpoints[name]).should be_true
      end
      serial.mps_workspace_bytes.should eq(workspaces[0].resident_bytes)
      serial.scratch_bytes.should be < arenas[-1].resident_bytes * 2

      # Establish both paths' root schedule on fresh chains, with one wait per
      # request. Diagnostic readback happens only after their terminal drain.
      chain = ML::Metal::MPSCommandChain.new
      chains << chain
      allocations = ML::MetalBuffer.video_spec_allocations
      controls[0].encode_mps(chain, inputs["img"], inputs["txt"], inputs["vec"],
        inputs["vec_txt"], inputs["cos"], inputs["sin"], workspaces[0])
      controls[1].encode_mps(chain, controls[0].image_output, controls[0].text_output,
        inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"], workspaces[1])
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      chain.wait_count.should eq(0)
      chain.commit_and_wait
      chain.wait_count.should eq(1)
      control_roots = chain.completed_root_timings.size
      expected = controls[1].checkpoints.transform_values { |buffer| buffer.read(buffer.element_count) }
      different = controls[0].image_output.read(c.image_rows * c.dim)

      poison = ML::Metal::ComputePipeline.new("hy_serial_spec_poison", <<-METAL)
        #include <metal_stdlib>
        using namespace metal;
        kernel void hy_serial_spec_poison(device float *output [[buffer(0)]],
          constant uint &count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
          if (i < count) output[i] = as_type<float>(0x7fc00000u);
        }
        METAL
      poison_buffers = [] of ML::MetalBuffer
      serial.blocks.each do |block|
        scratch, _ = block.mps_workspace_buffers
        ML::Video::HyNomasDualStreamBlock.scratch_plan(c).each_with_index do |(name, _), index|
          next if name == "hy_nomas.zero_head_bias"
          buffer = scratch[index]
          poison_buffers << buffer unless poison_buffers.any? { |other| other.same?(buffer) }
        end
      end

      2.times do |iteration|
        current = ML::Metal::MPSCommandChain.new
        chains << current
        allocations = ML::MetalBuffer.video_spec_allocations
        # NaNs in every nonconstant scratch slot expose reads of stale data.
        # These diagnostic dispatches share the existing first command root.
        poison_buffers.each do |buffer|
          current.encoder.set_pipeline(poison)
            .set_buffer(buffer, 0, ML::Metal::BufferAccess::Write)
            .set_value(buffer.element_count.to_u32, 1)
            .dispatch_1d(buffer.element_count)
        end
        serial.encode(current)
        current.wait_count.should eq(0)
        ML::MetalBuffer.video_spec_allocations.should eq(allocations)
        expect_raises(ArgumentError, /Complete/) { serial.release }
        current.commit_and_wait
        current.wait_count.should eq(1)
        current.completed_root_timings.size.should eq(control_roots)
        serial.blocks[1].checkpoints.each do |name, buffer|
          actual = buffer.read(buffer.element_count)
          actual.zip(expected[name]).all? do |value, reference|
            value.finite? && reference.finite? && value.unsafe_as(UInt32) == reference.unsafe_as(UInt32)
          end.should be_true, "different-weight successor #{name}, iteration #{iteration} must match separate plans exactly"
        end
        serial.blocks[1].image_output.read(c.image_rows * c.dim).should_not eq(different)
      end
    ensure
      chains.each { |chain| chain.commit_and_wait unless chain.completed? }
      serial.try(&.release)
      workspaces.reverse_each(&.release)
      arenas.reverse_each(&.release)
      chains.each(&.release)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    end
  end
end
