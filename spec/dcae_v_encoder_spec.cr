require "spec"
require "../src/ml/core/shape"
require "../src/ml/core/buffer"
require "../src/ml/core/tensor"
require "../src/ml/metal/device"
require "../src/ml/video/dcae_v_encoder"

describe ML::Video::DCAEVEncoderConfig do
  it "plans the released 512x288, 120-frame reference without allocating Metal buffers" do
    cfg = ML::Video::DCAEVEncoderConfig.new(288, 512)

    cfg.tile_count.should eq(3)
    cfg.max_tile_frames.should eq(40)
    cfg.tile_frame_count(2).should eq(40)
    cfg.latent_frames.should eq(30)
    cfg.latent_height.should eq(9)
    cfg.latent_width.should eq(16)
    cfg.input_elements.should eq(53_084_160)
    cfg.input_bytes.should eq(106_168_320_i64)
    cfg.output_elements.should eq(138_240)
    cfg.output_bytes.should eq(276_480_i64)
    cfg.rgb_bytes.should eq(442_368_i64)

    ML::Video::DCAEVEncoder.weight_bytes(cfg).should eq(1_833_310_016_i64)
    ML::Video::DCAEVEncoder.workspace_required_bytes(cfg).should eq(2_714_664_960_i64)
    ML::Video::DCAEVEncoder.trace_bytes(cfg).should eq(8_847_360_i64)
    ML::Video::DCAEVEncoder.project_in_trace_bytes(cfg).should eq(754_974_720_i64)
  end

  it "preserves upstream encoder-prefixed BF16 checkpoint bindings and shapes" do
    cfg = ML::Video::DCAEVEncoderConfig.new(288, 512)
    bindings = ML::Video::DCAEVEncoder.checkpoint_bindings(cfg)
    by_name = bindings.to_h { |binding| {binding.target, binding} }
    dispatch_names = ML::Video::DCAEVEncoder.dispatch_weight_names(cfg)

    bindings.size.should eq(169)
    dispatch_names.size.should eq(169)
    dispatch_names.sort.should eq(bindings.map(&.target).sort)
    bindings.each do |binding|
      binding.target.should eq(binding.source)
      binding.target.starts_with?("encoder.").should be_true
      binding.dtype.should eq(ML::GGUF::SafeTensorDType::BF16)
    end
    by_name["encoder.project_in.conv.conv.weight"].shape.should eq([64_i64, 3_i64, 3_i64, 3_i64, 3_i64])
    by_name["encoder.stages.1.op_list.0.main.conv1.conv.weight"].shape.should eq([256_i64, 256_i64, 3_i64, 3_i64, 3_i64])
    by_name["encoder.stages.1.op_list.0.main.conv2.norm.weight"].shape.should eq([256_i64])
    by_name["encoder.stages.1.op_list.5.main.conv.conv.weight"].shape.should eq([128_i64, 256_i64, 3_i64, 3_i64, 3_i64])
    by_name["encoder.stages.5.op_list.4.main.conv.conv.weight"].shape.should eq([256_i64, 1024_i64, 3_i64, 3_i64, 3_i64])
    by_name["encoder.project_out.conv.weight"].shape.should eq([32_i64, 1024_i64, 3_i64, 3_i64, 3_i64])
    bindings.any? { |binding| binding.target.ends_with?(".conv2.conv.bias") }.should be_false
    by_name.has_key?("encoder.stages.1.op_list.0.main.conv1.conv.bias").should be_true
    by_name.has_key?("encoder.stages.1.op_list.0.main.conv2.conv.bias").should be_false
  end

  it "keeps convolution operation bias roles aligned with checkpoint bindings" do
    roles = {
      ML::Video::DCAEVConvRole::ProjectIn     => true,
      ML::Video::DCAEVConvRole::ResidualConv1 => true,
      ML::Video::DCAEVConvRole::ResidualConv2 => false,
      ML::Video::DCAEVConvRole::Downsample    => true,
      ML::Video::DCAEVConvRole::ProjectOut    => true,
    }
    roles.each do |role, expected|
      ML::Video::DCAEVEncoder.conv_role_uses_bias?(role).should eq(expected)
    end

    cfg = ML::Video::DCAEVEncoderConfig.new(32, 32, 4)
    binding_names = ML::Video::DCAEVEncoder.checkpoint_bindings(cfg).map(&.target)
    binding_names.should contain("encoder.project_in.conv.conv.bias")
    binding_names.should contain("encoder.stages.1.op_list.0.main.conv1.conv.bias")
    binding_names.should_not contain("encoder.stages.1.op_list.0.main.conv2.conv.bias")
    binding_names.should contain("encoder.stages.1.op_list.5.main.conv.conv.bias")
    binding_names.should contain("encoder.project_out.conv.bias")
  end

  it "accepts the upstream short final tile while keeping 40-frame cache boundaries" do
    tiny = ML::Video::DCAEVEncoderConfig.new(32, 32, 4)
    tiny.tile_count.should eq(1)
    tiny.tile_frame_count(0).should eq(4)
    tiny.latent_frames.should eq(1)
    tiny.output_elements.should eq(32)
    ML::Video::DCAEVEncoder.workspace_required_bytes(tiny).should eq(4_659_200_i64)
    ML::Video::DCAEVEncoder.project_in_trace_bytes(tiny).should eq(524_288_i64)

    two_tiles = ML::Video::DCAEVEncoderConfig.new(32, 32, 44)
    two_tiles.tile_count.should eq(2)
    two_tiles.tile_frame_count(0).should eq(40)
    two_tiles.tile_frame_count(1).should eq(4)
    two_tiles.latent_frames.should eq(11)
    ML::Video::DCAEVEncoder.trace_bytes(two_tiles).should eq(22_528_i64)
    ML::Video::DCAEVEncoder.workspace_required_bytes(two_tiles).should eq(18_851_840_i64)
  end

  it "rejects incomplete temporal compression groups and non-upstream tile sizes" do
    expect_raises(ArgumentError, /divisible by 4/) do
      ML::Video::DCAEVEncoderConfig.new(32, 32, 6)
    end
    expect_raises(ArgumentError, /40-frame tiles/) do
      ML::Video::DCAEVEncoderConfig.new(32, 32, 40, 20)
    end
    expect_raises(ArgumentError, /tile index/) do
      ML::Video::DCAEVEncoderConfig.new(32, 32, 4).tile_frame_count(1)
    end
  end
end

describe ML::Video::DCAEVEncoderMPSPlanSet do
  it "rejects an absent prepared MPS plan set" do
    expect_raises(ArgumentError, /requires prepared MPSGraph plans/) do
      ML::Video::DCAEVEncoderMPSPlanSet.require_prepared!(nil)
    end
  end

  it "returns a supplied prepared MPS plan set unchanged" do
    plans = ML::Video::DCAEVEncoderMPSPlanSet.new
    ML::Video::DCAEVEncoderMPSPlanSet.require_prepared!(plans).should be(plans)
  end
end
