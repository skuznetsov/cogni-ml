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

describe "preview Euler step admission" do
  it "keeps eight steps as the default and accepts the supported endpoints" do
    HyNomasGeneration::DEFAULT_STEPS.should eq(8)
    HyNomasGeneration.validate_steps!(1).should eq(1)
    [8, 16, 24, 32, 64, 96, 128].each do |steps|
      HyNomasGeneration.validate_steps!(steps).should eq(steps)
    end
  end

  it "rejects step counts outside the scheduler range before allocation" do
    [0, 129].each do |steps|
      expect_raises(ArgumentError, /between 1 and 128/) do
        HyNomasGeneration.validate_steps!(steps)
      end
    end
  end

  it "keeps the unqualified Q4 path limited to eight steps" do
    HyNomasGeneration.validate_steps!(8, HyNomasGeneration::DiTWeights::Q4K).should eq(8)
    [16, 64, 96, 128].each do |steps|
      expect_raises(ArgumentError, /Q4.*eight/) do
        HyNomasGeneration.validate_steps!(steps, HyNomasGeneration::DiTWeights::Q4K)
      end
    end
  end

  it "accounts for two F32 schedule values per requested step" do
    HyNomasGeneration.euler_schedule_bytes(2).should eq(16_i64)
    HyNomasGeneration.euler_schedule_bytes(8).should eq(64_i64)
    HyNomasGeneration.euler_schedule_bytes(32).should eq(256_i64)
    HyNomasGeneration.euler_schedule_bytes(128).should eq(1024_i64)
    expect_raises(ArgumentError, /between 1 and 128/) do
      HyNomasGeneration.euler_schedule_bytes(129)
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

private def conditioned_document(valid : Int32 = 67) : JSON::Any
  prompt = "A tram follows the rails through the intersection."
  mask = Array.new(1280) { |i| i < valid ? 1 : 0 }
  contract = {prompt: prompt, prompt_sha256: Digest::SHA256.hexdigest(prompt),
              upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
              template_name: "li-dit-encode-video-json", template_sha256: HyNomasGeneration::PROMPT_TEMPLATE_SHA,
              crop_start: 108, raw_length: 1388, cropped_attention_mask: mask,
              untruncated_preflight: {truncation_required: false}}
  JSON.parse({schema: "cogni_ml.hy_nomas_generation_inputs/v2", schema_version: 2,
              upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION, prompt: prompt,
              geometry: {qwen_valid_tokens: valid},
              qwen_source: {prompt_contract: contract, prompt_sha256: Digest::SHA256.hexdigest(prompt),
                            upstream_revision: PixelUMMBlockProbe::UPSTREAM_REVISION,
                            template_name: "li-dit-encode-video-json", template_sha256: HyNomasGeneration::PROMPT_TEMPLATE_SHA,
                            state_index: 26, hidden_size: 3584, crop_tokens: 1280, crop: [108, 1388],
                            raw_sequence_length: 1388, sidecar_bytes: 27_i64 * 1388 * 3584 * 2,
                            cropped_attention_mask: mask, valid_tokens: valid}}.to_json)
end

describe "conditioned first-segment input admission" do
  it "uses the actual visible prefix instead of the legacy 23-token mask" do
    mask = HyNomasGeneration.conditioned_mask!(conditioned_document)
    mask.size.should eq(1280)
    mask.sum.should eq(67)
    mask[66].should eq(1)
    mask[67].should eq(0)
    HyNomasGeneration.dit_scratch_bytes(67, 128).should be > HyNomasGeneration.dit_scratch_bytes
  end

  it "rejects a non-prefix mask even when its population is unchanged" do
    doc = conditioned_document
    mask = doc["qwen_source"]["cropped_attention_mask"].as_a
    mask[2] = JSON::Any.new(0_i64)
    mask[100] = JSON::Any.new(1_i64)
    expect_raises(ArgumentError, /visible text prefix/) { HyNomasGeneration.conditioned_mask!(doc) }
  end

  it "rejects lost prompt text, changed templates and silent truncation" do
    doc = conditioned_document
    doc.as_h["prompt"] = JSON::Any.new("A different scene")
    expect_raises(ArgumentError, /identity or truncation/) { HyNomasGeneration.conditioned_mask!(doc) }
    doc = conditioned_document
    doc["qwen_source"].as_h["template_sha256"] = JSON::Any.new("a" * 64)
    expect_raises(ArgumentError, /schema\/template/) { HyNomasGeneration.conditioned_mask!(doc) }
    doc = conditioned_document
    doc["qwen_source"]["prompt_contract"]["untruncated_preflight"].as_h["truncation_required"] = JSON::Any.new(true)
    expect_raises(ArgumentError, /identity or truncation/) { HyNomasGeneration.conditioned_mask!(doc) }
  end

  it "rejects unsupported token extents before constructing GPU owners" do
    [0, 1001, 1280].each do |valid|
      expect_raises(ArgumentError, /visible text prefix/) { HyNomasGeneration.conditioned_mask!(conditioned_document(valid)) }
    end
  end

  it "hashes exactly the authenticated byte range" do
    path = File.tempname("hy-nomas-crop", ".bin")
    begin
      File.write(path, "abcdef")
      HyNomasGeneration.sha_range(path, 2_i64, 3_i64).should eq(Digest::SHA256.hexdigest("cde"))
      expect_raises(ArgumentError, /invalid authenticated/) { HyNomasGeneration.sha_range(path, 4_i64, 3_i64) }
    ensure
      File.delete?(path)
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
