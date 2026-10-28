require "spec"
require "../examples/support/hy_nomas_generation"

describe "preview DiT weight selection" do
  it "keeps BF16 as the control and makes Q4_K explicit" do
    HyNomasGeneration.parse_dit_weights("bf16").should eq(HyNomasGeneration::DiTWeights::BF16)
    HyNomasGeneration.parse_dit_weights("q4-k").should eq(HyNomasGeneration::DiTWeights::Q4K)
    expect_raises(ArgumentError, /bf16 or q4-k/) do
      HyNomasGeneration.parse_dit_weights("q4")
    end
  end

  it "admits only the selected block residents plus the same fixed stage" do
    bf16 = HyNomasGeneration::DiTWeights::BF16
    q4 = HyNomasGeneration::DiTWeights::Q4K
    HyNomasGeneration.dit_resident_weight_bytes(bf16, 302_113_792_i64, 4_593_162_240_i64).should eq(16_616_258_560_i64)
    HyNomasGeneration.dit_resident_weight_bytes(q4, 302_113_792_i64, 4_593_162_240_i64).should eq(4_895_276_032_i64)
    expect_raises(ArgumentError, /positive/) do
      HyNomasGeneration.dit_resident_weight_bytes(q4, 302_113_792_i64, 0_i64)
    end
  end
end

describe "preview terminal output" do
  it "writes a fresh file and refuses to replace it" do
    path = File.tempname("hy-nomas-terminal-output", ".bin")
    begin
      HyNomasGeneration.write_fresh_output(path) { |io| io.write(Bytes[0, 127, 255]) }
      File.read(path).to_slice.should eq(Bytes[0, 127, 255])
      expect_raises(IO::Error, /fresh and writable/) do
        HyNomasGeneration.write_fresh_output(path) { |io| io.puts("replacement") }
      end
      File.read(path).to_slice.should eq(Bytes[0, 127, 255])
    ensure
      File.delete?(path)
    end
  end
end

describe "bounded first-video input identity" do
  it "rejects caller-selected provenance before loading files or Metal" do
    HyNomasGeneration.validate_input_identity!(HyNomasGeneration::INPUTS_SHA)
    expect_raises(ArgumentError, /pinned producer-authenticated/) do
      HyNomasGeneration::Inputs.new("/nonexistent/inputs.json", "a" * 64)
    end
  end
end

describe "bounded preview weight-cache geometry" do
  it "authenticates the original cache geometry with identical runtime parameters" do
    cache = HyNomasGeneration.weight_cache_config
    cache.video_tokens.should eq(5280)
    HyNomasGeneration::CONFIG.video_tokens.should eq(4320)
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(cache).should eq(
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(HyNomasGeneration::CONFIG))
  end

  it "rejects changes to architecture or another fixture dimension" do
    [HyNomasGeneration::CONFIG.copy_with(heads: 8, head_dim: 256),
     HyNomasGeneration::CONFIG.copy_with(text_tokens: 64)].each do |changed|
      expect_raises(ArgumentError, /authenticated weight-cache architecture/) do
        HyNomasGeneration.weight_cache_config(changed)
      end
    end
  end
end
