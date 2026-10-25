require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_serial_mps"

# Deliberately different successor weights catch a shared plan accidentally
# reusing the first block's projection or RMS weights. The bounded owner is
# exercised at the two-, three-, and four-block admission boundaries.
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
      # A block caches LinearWeights and NormWeights at construction. Replacing
      # an equal-shaped/equal-dtype Arena entry afterward must not make the map
      # appear to describe the buffers the block will actually dispatch.
      {"img_attn_q_weight", "img_attn_q_bias", "img_attn_q_norm_weight"}.each do |target|
        original = weights[0][target]
        replacement = weights[1][target]
        original.same?(replacement).should be_false
        weights[0].dtype(target).should eq(weights[1].dtype(target))
        weights[0].buffers[target] = replacement
        allocations_before_rebind_check = ML::MetalBuffer.video_spec_allocations
        begin
          expect_raises(ArgumentError, /cached weight binding changed/) do
            candidate = ML::Video::HyNomasMPSWorkspace.new(controls[0],
              inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"])
            candidate.release
          end
        ensure
          # Restore the owner map before either Arena cleanup runs; replacement
          # remains owned by weights[1] throughout this falsifier.
          weights[0].buffers[target] = original
        end
        weights[0][target].same?(original).should be_true
        ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_rebind_check)
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
      expect_raises(ArgumentError, /between two and four/) do
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
      # The registered workspace must re-check cached RMS weight identity
      # before appending either block. Rebind after registration to isolate the
      # preflight guard from constructor-time checks.
      registered_target = "txt_attn_k_norm_weight"
      registered_original = weights[1][registered_target]
      registered_replacement = weights[0][registered_target]
      registered_original.same?(registered_replacement).should be_false
      weights[1].dtype(registered_target).should eq(weights[0].dtype(registered_target))
      roots_before_rebind = chain.observed_root_count
      allocations_before_registered_rebind = ML::MetalBuffer.video_spec_allocations
      weights[1].buffers[registered_target] = registered_replacement
      begin
        expect_raises(ArgumentError, /cached weight binding changed/) do
          serial.encode(chain)
        end
      ensure
        weights[1].buffers[registered_target] = registered_original
      end
      chain.observed_root_count.should eq(roots_before_rebind)
      chain.wait_count.should eq(0)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_registered_rebind)
      weights[1][registered_target].same?(registered_original).should be_true
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

  it "runs four distinct consecutive blocks with retained output banks and repeatable shared scratch" do
    c = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 6, text_tokens: 3,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256,
      attention_precision: ML::Video::HyNomasAttentionPrecision::BF16BoundaryControl)
    arenas = [] of ML::Video::Arena
    workspaces = [] of ML::Video::HyNomasMPSWorkspace
    chains = [] of ML::Metal::MPSCommandChain
    serial = nil.as(ML::Video::HyNomasMPSSerialWorkspace?)
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count
    begin
      weights = Array(ML::Video::Arena).new(4) do |block_index|
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

      # An invalid block count is rejected before the serial owner allocates
      # scratch or its MPS workspace.
      allocations_before_count_rejections = ML::MetalBuffer.video_spec_allocations
      expect_raises(ArgumentError, /between two and four/) do
        ML::Video::HyNomasMPSSerialWorkspace.new(c, [weights[0]], inputs["img"], inputs["txt"],
          inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"])
      end
      five_weights = Array(ML::Video::Arena).new(5) { weights[0] }
      expect_raises(ArgumentError, /between two and four/) do
        ML::Video::HyNomasMPSSerialWorkspace.new(c, five_weights, inputs["img"], inputs["txt"],
          inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"])
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_count_rejections)

      controls = weights.map do |weight|
        scratch = ML::Video::Arena.new
        arenas << scratch
        ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, c)
        ML::Video::HyNomasDualStreamBlock.new(c, weight, scratch)
      end
      controls.each_with_index do |block, index|
        img = index == 0 ? inputs["img"] : controls[index - 1].image_output
        txt = index == 0 ? inputs["txt"] : controls[index - 1].text_output
        workspaces << ML::Video::HyNomasMPSWorkspace.new(block, img, txt,
          inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"],
          attention_query_chunk: 9, attention_heads_per_group: c.heads)
      end
      three_serial = nil.as(ML::Video::HyNomasMPSSerialWorkspace?)
      begin
        three_serial = ML::Video::HyNomasMPSSerialWorkspace.new(c,
          [weights[0], weights[1], weights[2]], inputs["img"], inputs["txt"],
          inputs["vec"], inputs["vec_txt"], inputs["cos"], inputs["sin"],
          attention_query_chunk: 9, attention_heads_per_group: c.heads)
        three_serial.blocks.size.should eq(3)
      ensure
        three_serial.try(&.release)
      end
      serial = ML::Video::HyNomasMPSSerialWorkspace.new(c, weights,
        inputs["img"], inputs["txt"], inputs["vec"], inputs["vec_txt"],
        inputs["cos"], inputs["sin"], attention_query_chunk: 9,
        attention_heads_per_group: c.heads)

      blocks = serial.blocks
      blocks.size.should eq(4)
      blocks.each_with_index do |block, index|
        block.checkpoints.keys.sort.should eq(controls[index].checkpoints.keys.sort)
        block.image_output.same?(controls[index].image_output).should be_false
        block.text_output.same?(controls[index].text_output).should be_false
        blocks.each_with_index do |other, other_index|
          next if other_index <= index
          block.image_output.same?(other.image_output).should be_false
          block.image_output.same?(other.text_output).should be_false
          block.text_output.same?(other.image_output).should be_false
          block.text_output.same?(other.text_output).should be_false
        end
        next if index == 0
        block.checkpoints.each do |name, buffer|
          next if {"img.output", "txt.output"}.includes?(name)
          buffer.same?(blocks[0].checkpoints[name]).should be_true
        end
      end
      # Reordering a returned registry view must not mutate the owner's order.
      registered_order = blocks.dup
      blocks.reverse!
      serial.blocks.each_with_index { |block, index| block.same?(registered_order[index]).should be_true }
      serial.scratch_bytes.should be > 0
      serial.mps_workspace_bytes.should eq(workspaces[0].resident_bytes)

      # Mutating the last successor's cached weight binding must be rejected
      # before any block is appended, not just when block 1 is the successor.
      registered_target = "txt_attn_k_norm_weight"
      original = weights[3][registered_target]
      replacement = weights[2][registered_target]
      weights[3].buffers[registered_target] = replacement
      mutated_chain = ML::Metal::MPSCommandChain.new
      chains << mutated_chain
      roots_before_mutation = mutated_chain.observed_root_count
      allocations_before_mutation = ML::MetalBuffer.video_spec_allocations
      begin
        expect_raises(ArgumentError, /cached weight binding changed/) do
          serial.encode(mutated_chain)
        end
      ensure
        weights[3].buffers[registered_target] = original
      end
      mutated_chain.observed_root_count.should eq(roots_before_mutation)
      mutated_chain.wait_count.should eq(0)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_mutation)

      # Build a separate-plan four-block result with the same direct
      # predecessor handoff. Only output banks persist per block; final shared
      # scratch persists for the last block.
      reference_chain = ML::Metal::MPSCommandChain.new
      chains << reference_chain
      allocations_before_reference = ML::MetalBuffer.video_spec_allocations
      controls.each_with_index do |block, index|
        img = index == 0 ? inputs["img"] : controls[index - 1].image_output
        txt = index == 0 ? inputs["txt"] : controls[index - 1].text_output
        block.encode_mps(reference_chain, img, txt, inputs["vec"],
          inputs["vec_txt"], inputs["cos"], inputs["sin"], workspaces[index])
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_reference)
      reference_chain.wait_count.should eq(0)
      reference_chain.commit_and_wait
      reference_chain.wait_count.should eq(1)
      control_roots = reference_chain.completed_root_timings.size
      control_expected = controls.map do |block|
        block.checkpoints.transform_values { |buffer| buffer.read(buffer.element_count) }
      end

      2.times do |iteration|
        # A failed late-block preflight must not bind the owner to its empty
        # chain. Switch to a fresh chain without waiting on the rejected one.
        chain = ML::Metal::MPSCommandChain.new
        chains << chain
        mutated_chain.wait_count.should eq(0)
        allocations_before_encode = ML::MetalBuffer.video_spec_allocations
        callback_order = [] of Int32
        returned_img, returned_txt = serial.encode(chain, after_block: ->(index : Int32, _block : ML::Video::HyNomasDualStreamBlock) do
          callback_order << index
          nil
        end)
        returned_img.same?(serial.blocks[3].image_output).should be_true
        returned_txt.same?(serial.blocks[3].text_output).should be_true
        callback_order.should eq([0, 1, 2, 3])
        chain.wait_count.should eq(0)
        ML::MetalBuffer.video_spec_allocations.should eq(allocations_before_encode)
        expect_raises(ArgumentError, /Complete/) { serial.release }
        chain.commit_and_wait
        chain.wait_count.should eq(1)
        mutated_chain.wait_count.should eq(0)
        chain.completed_root_timings.size.should eq(control_roots)
        blocks = serial.blocks
        blocks.each_with_index do |block, index|
          block.checkpoints.each do |name, buffer|
            # Each block owns its image/text output banks, but all remaining
            # checkpoint names share one scratch set and therefore contain the
            # last block's values at terminal readback.
            next unless index == 3 || {"img.output", "txt.output"}.includes?(name)
            actual = buffer.read(buffer.element_count)
            actual.zip(control_expected[index][name]).all? do |value, reference|
              value.finite? && reference.finite? && value.unsafe_as(UInt32) == reference.unsafe_as(UInt32)
            end.should be_true, "four-block #{name}, block #{index}, iteration #{iteration} must match separate plans"
          end
        end
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
