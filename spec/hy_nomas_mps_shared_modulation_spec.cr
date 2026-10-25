require "spec"
require "../src/ml/video/hy_nomas_block_mps"

describe ML::Video::HyNomasMPSWorkspace do
  it "allows only the exact vec/vec_txt object to be shared when opted in" do
    config = ML::Video::HyNomasConfig.new(batches: 1, video_tokens: 1, text_tokens: 1,
      dim: 64, heads: 2, head_dim: 32, ffn_dim: 256)
    arenas = [] of ML::Video::Arena
    workspace = nil.as(ML::Video::HyNomasMPSWorkspace?)
    roots_before = ML::Video::HyNomasMPSWorkspace.retained_workspace_count

    begin
      weights = ML::Video::Arena.new
      arenas << weights
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config).each do |binding|
        count = binding.shape.reduce(1_i64) { |n, dim| n * dim }.to_i32
        weights.upload(binding.target, Array(Float32).new(count, 0.0_f32))
      end

      scratch = ML::Video::Arena.new
      arenas << scratch
      ML::Video::HyNomasDualStreamBlock.prepare_scratch!(scratch, config)
      block = ML::Video::HyNomasDualStreamBlock.new(config, weights, scratch)

      inputs = ML::Video::Arena.new
      arenas << inputs
      inputs.upload("img", Array(Float32).new(config.dim, 0.0_f32))
      inputs.upload("txt", Array(Float32).new(config.dim, 0.0_f32))
      shared_modulation = inputs.upload("modulation", Array(Float32).new(config.dim, 0.0_f32))
      inputs.upload("rope_cos", Array(Float32).new(config.head_dim, 1.0_f32))
      inputs.upload("rope_sin", Array(Float32).new(config.head_dim, 0.0_f32))

      expect_raises(ArgumentError, /fixed input buffers must not alias/) do
        ML::Video::HyNomasMPSWorkspace.new(block, inputs["img"], inputs["txt"],
          shared_modulation, shared_modulation, inputs["rope_cos"], inputs["rope_sin"])
      end
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)

      workspace = ML::Video::HyNomasMPSWorkspace.new(block, inputs["img"], inputs["txt"],
        shared_modulation, shared_modulation, inputs["rope_cos"], inputs["rope_sin"],
        allow_shared_modulation: true)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before + 1)
      workspace.release
      workspace = nil
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)

      expect_raises(ArgumentError, /fixed input buffers must not alias/) do
        ML::Video::HyNomasMPSWorkspace.new(block, inputs["img"], inputs["img"],
          shared_modulation, shared_modulation, inputs["rope_cos"], inputs["rope_sin"],
          allow_shared_modulation: true)
      end
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    ensure
      workspace.try(&.release)
      arenas.reverse_each(&.release)
      ML::Video::HyNomasMPSWorkspace.retained_workspace_count.should eq(roots_before)
    end
  end
end
