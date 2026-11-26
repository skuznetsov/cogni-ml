require "spec"
require "./support/video_allocation_counter"
require "../src/ml/video/hy_nomas_block_mps"

describe ML::Video::HyNomasMPSWorkspace do
  it "matches compact-prefix attention and clears poisoned padded attention rows" do
    compact = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 1, text_tokens: 1,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)
    padded = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 1, text_tokens: 3,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)
    arenas = [] of ML::Video::Arena
    blocks = [] of ML::Video::HyNomasDualStreamBlock
    workspaces = [] of ML::Video::HyNomasMPSWorkspace
    chains = [] of ML::Metal::MPSCommandChain
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count

    begin
      weights = ML::Video::Arena.new
      arenas << weights
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(compact).each_with_index do |binding, tensor|
        count = binding.shape.reduce(1_i64) { |n, dim| n * dim }.to_i32
        values = Array(Float32).new(count) do |index|
          if binding.target.includes?("norm_weight")
            1.0_f32 + (index % 7) * 0.003_f32
          else
            Math.sin(index * 0.17 + tensor).to_f32 * 0.04_f32
          end
        end
        weights.upload(binding.target, values)
      end

      inputs = ML::Video::Arena.new
      arenas << inputs
      img = inputs.upload("img", Array(Float32).new(compact.image_rows * compact.dim) do |i|
        Math.sin(i * 0.13).to_f32 * 0.25_f32
      end)
      compact_txt = inputs.upload("compact.txt", Array(Float32).new(compact.text_rows * compact.dim) do |i|
        Math.cos(i * 0.11).to_f32 * 0.2_f32
      end)
      padded_txt = inputs.upload("padded.txt", Array(Float32).new(padded.text_rows * padded.dim) do |i|
        # The first valid row is bit-identical to the compact path; padded rows
        # are finite but distinct so only the valid prefix can match.
        i < compact.dim ? Math.cos(i * 0.11).to_f32 * 0.2_f32 : Math.sin(i * 0.07).to_f32 * 0.3_f32
      end)
      vec = inputs.upload("vec", Array(Float32).new(compact.dim) { |i| Math.sin(i * 0.03).to_f32 * 0.1_f32 })
      vec_txt = inputs.upload("vec_txt", Array(Float32).new(compact.dim) { |i| Math.cos(i * 0.05).to_f32 * 0.1_f32 })
      cos = inputs.upload("cos", Array(Float32).new(compact.video_tokens * compact.head_dim, 1.0_f32))
      sin = inputs.upload("sin", Array(Float32).new(compact.video_tokens * compact.head_dim, 0.0_f32))

      {compact, padded}.each_with_index do |config, index|
        scratch = ML::Video::Arena.new
        arenas << scratch
        ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
        block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)
        blocks << block
        workspaces << ML::Video::HyNomasMPSWorkspace.new(block, img,
          index == 0 ? compact_txt : padded_txt, vec, vec_txt, cos, sin,
          valid_text_tokens: index == 0 ? nil : 1)
      end

      # Without the prefix clear, MPSGraph writes only the first two joint
      # rows and this NaN tail propagates through padded text projection/FFN.
      poison = ML::Metal::ComputePipeline.new("hy_nomas_padded_prefix_poison", <<-METAL)
        #include <metal_stdlib>
        using namespace metal;
        kernel void hy_nomas_padded_prefix_poison(device float *output [[buffer(0)]],
          constant uint &count [[buffer(1)]], uint i [[thread_position_in_grid]]) {
          if (i < count) output[i] = as_type<float>(0x7fc00000u);
        }
        METAL

      compact_chain = ML::Metal::MPSCommandChain.new
      chains << compact_chain
      allocations = ML::MetalBuffer.video_spec_allocations
      blocks[0].encode_mps(compact_chain, img, compact_txt, vec, vec_txt, cos, sin, workspaces[0])
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      compact_chain.wait_count.should eq(0)
      compact_chain.commit_and_wait
      compact_chain.wait_count.should eq(1)

      padded_chain = ML::Metal::MPSCommandChain.new
      chains << padded_chain
      padded_attention = blocks[1].checkpoints["joint.attention"]
      padded_count = padded.joint_rows * padded.dim
      padded_chain.encoder.set_pipeline(poison)
        .set_buffer(padded_attention, 0, ML::Metal::BufferAccess::Write)
        .set_value(padded_count.to_u32, 1)
        .dispatch_1d(padded_count)
      allocations = ML::MetalBuffer.video_spec_allocations
      blocks[1].encode_mps(padded_chain, img, padded_txt, vec, vec_txt, cos, sin, workspaces[1])
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
      padded_chain.wait_count.should eq(0)
      padded_chain.commit_and_wait
      padded_chain.wait_count.should eq(1)

      compact_joint = blocks[0].checkpoints["joint.attention"].read(compact.joint_rows * compact.dim)
      padded_joint = padded_attention.read(padded.joint_rows * padded.dim)
      padded_prefix = compact.joint_rows * compact.dim
      padded_joint.first(padded_prefix).zip(compact_joint).all? do |actual, expected|
        actual.finite? && expected.finite? && (actual - expected).abs <= 2e-5_f32
      end.should be_true, "padded-prefix joint attention must match the compact sequence"
      padded_joint.skip(padded_prefix).all? { |value| value == 0.0_f32 }.should be_true,
        "the skipped padded query rows must be exactly cleared"

      compact_img = blocks[0].image_output.read(compact.image_rows * compact.dim)
      padded_img = blocks[1].image_output.read(padded.image_rows * padded.dim)
      compact_img.zip(padded_img).all? do |expected, actual|
        expected.finite? && actual.finite? && (actual - expected).abs <= 2e-5_f32
      end.should be_true, "padded-prefix image output must match compact conditioning"
      compact_txt_output = blocks[0].text_output.read(compact.text_rows * compact.dim)
      padded_txt_output = blocks[1].text_output.read(padded.text_rows * padded.dim)
      compact_txt_output.zip(padded_txt_output.first(compact.text_rows * compact.dim)).all? do |expected, actual|
        expected.finite? && actual.finite? && (actual - expected).abs <= 2e-5_f32
      end.should be_true, "valid padded-prefix text output must match compact conditioning"
      padded_txt_output.all?(&.finite?).should be_true
    ensure
      chains.each { |chain| chain.commit_and_wait unless chain.completed? }
      workspaces.reverse_each(&.release)
      arenas.reverse_each(&.release)
      chains.each(&.release)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    end
  end
end
