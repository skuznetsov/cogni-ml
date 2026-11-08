require "spec"
require "../src/ml/metal/device"
require "../src/ml/video/dcae_v_decoder"

describe ML::Video::DCAEVDecoderConfig do
  it "plans the original decoder and its reusable scratch without Metal" do
    cfg = ML::Video::DCAEVDecoderConfig.new(9, 16, 60)
    cfg.tile_count.should eq(6)
    cfg.output_frames.should eq(240)
    cfg.output_height.should eq(288)
    cfg.output_width.should eq(512)
    cfg.input_bytes.should eq(552_960_i64)
    cfg.output_bytes.should eq(212_336_640_i64)
    ML::Video::DCAEVDecoder.weight_bytes(cfg).should eq(2_443_960_856_i64)
    ML::Video::DCAEVDecoder.scratch_values(cfg).should eq(377_487_360)
    ML::Video::DCAEVDecoder.workspace_required_bytes(cfg).should be > 2_700_000_000_i64
    ML::Video::DCAEVDecoder.workspace_required_bytes(cfg).should be < 2_800_000_000_i64
  end

  it "preserves every upstream decoder-prefixed BF16 tensor and bias role" do
    cfg = ML::Video::DCAEVDecoderConfig.new(1, 1, 20)
    bindings = ML::Video::DCAEVDecoder.checkpoint_bindings(cfg)
    bindings.size.should eq(171)
    by_name = bindings.to_h { |b| {b.target, b} }
    bindings.each do |binding|
      binding.target.should eq(binding.source)
      binding.target.starts_with?("decoder.").should be_true
      binding.dtype.should eq(ML::GGUF::SafeTensorDType::BF16)
    end
    by_name["decoder.project_in.conv.weight"].shape.should eq([1024_i64, 32_i64, 3_i64, 3_i64, 3_i64])
    by_name["decoder.stages.5.op_list.0.main.conv.conv.weight"].shape.should eq([4096_i64, 1024_i64, 3_i64, 3_i64, 3_i64])
    by_name["decoder.stages.1.op_list.0.main.conv.conv.weight"].shape.should eq([1024_i64, 512_i64, 3_i64, 3_i64, 3_i64])
    by_name["decoder.project_out.op_list.2.conv.conv.weight"].shape.should eq([12_i64, 256_i64, 3_i64, 3_i64, 3_i64])
    by_name.has_key?("decoder.stages.6.op_list.0.main.conv2.conv.bias").should be_false
    by_name.has_key?("decoder.stages.6.op_list.0.main.conv1.conv.bias").should be_true
  end

  it "exposes stage captures on the same ten-latent cache grid" do
    cfg = ML::Video::DCAEVDecoderConfig.new(1, 2, 20)
    shapes = ML::Video::DCAEVDecoder.trace_shapes(cfg)
    shapes.size.should eq(16)
    shapes["trace.project_in.tile0"].should eq([1_i64, 1024_i64, 10_i64, 1_i64, 2_i64])
    shapes["trace.stages.5.tile1"].should eq([1_i64, 1024_i64, 40_i64, 1_i64, 2_i64])
    shapes["trace.stages.1.tile1"].should eq([1_i64, 256_i64, 40_i64, 16_i64, 32_i64])
    shapes["trace.project_out.tile1"].should eq([1_i64, 3_i64, 40_i64, 32_i64, 64_i64])
    cfg.tile_offset(1).should eq(10)
    ML::Video::DCAEVDecoder.workspace_required_bytes(cfg).should eq(
      ML::Video::DCAEVDecoder.workspace_required_bytes(ML::Video::DCAEVDecoderConfig.new(1, 2, 10)))
  end

  it "rejects unsupported or overflowing geometry before GPU work" do
    expect_raises(ArgumentError) { ML::Video::DCAEVDecoderConfig.new(0, 1, 10) }
    expect_raises(ArgumentError, /multiples of 10/) { ML::Video::DCAEVDecoderConfig.new(1, 1, 11) }
    expect_raises(ArgumentError) { ML::Video::DCAEVDecoderConfig.new(Int32::MAX, 2, 10) }
    expect_raises(ArgumentError, /tile index/) { ML::Video::DCAEVDecoderConfig.new(1, 1, 10).tile_offset(1) }
  end
end
