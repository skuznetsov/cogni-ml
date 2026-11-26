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

describe "opt-in Hy-Nomas spatial geometry" do
  it "shares the patch/final finite token gate with CPU preflight" do
    [1, 4320, 5280, 9360].each do |tokens|
      ML::Video::HyNomasDiTBoundaryMPSPlan.validate_token_count!(tokens).should be_nil
    end
    [0, 9361, Int32::MAX].each do |tokens|
      expect_raises(ArgumentError, /1\.\.9360/) do
        ML::Video::HyNomasDiTBoundaryMPSPlan.validate_token_count!(tokens)
      end
    end

    candidate = HyNomasGeneration::SpatialGeometry::CANDIDATE_768.dit_config
    HyNomasGeneration.validate_dit_config!(candidate).should eq(candidate)
    expect_raises(ArgumentError, /1\.\.9360/) do
      HyNomasGeneration.validate_dit_config!(candidate.copy_with(video_tokens: 9361))
    end
  end

  it "keeps the default shape and admits only the pinned 768x416 candidate" do
    default = HyNomasGeneration::SpatialGeometry.parse("512x288")
    default.video_tokens.should eq(4320)
    default.pixelumm_config.decoder_config.tokens.should eq(5760)
    HyNomasGeneration.input_shapes(default)["input.latent"][1].should eq([1, 32, 30, 9, 16])

    candidate = HyNomasGeneration::SpatialGeometry.parse("768x416")
    candidate.high_resolution?.should be_true
    candidate.pixelumm_config.latent_shape.should eq([1, 32, 30, 13, 24])
    candidate.video_tokens.should eq(9360)
    candidate.pixelumm_config.decoder_config.tokens.should eq(12_480)
    candidate.pixelumm_config.decoder_config.layout.tile_count.should eq(24)
    HyNomasGeneration.input_shapes(candidate)["input.rope_cos"][1].should eq([9360, 128])
    runtime = candidate.dit_config
    runtime.video_tokens.should eq(9360)
    cache = HyNomasGeneration.weight_cache_config(runtime)
    ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(cache).should eq(
      ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(HyNomasGeneration::CONFIG))

    expect_raises(ArgumentError, /resolution must be/) do
      HyNomasGeneration::SpatialGeometry.parse("1024x576")
    end
  end

  it "requires the v3 conditioned manifest to declare every candidate extent" do
    valid = 67
    crop = [108, 1388]
    geometry = {
      pixel_width:                         768,
      pixel_height:                        416,
      dit_latent_shape:                    [1, 32, 30, 13, 24],
      dit_rope_grid:                       [30, 13, 24],
      decoder_latent_grid:                 [30, 13, 24],
      decoder_noise_shape:                 [1, 3, 120, 416, 768],
      decoder_rope_grid:                   [10, 26, 48],
      decoder_rope_shape:                  [1, 12_480, 1, 128],
      decoder_local_attention_tile_tokens: 8,
      decoder_local_attention_tile_pixels: 128,
      qwen_valid_tokens:                   valid,
      qwen_crop:                           crop,
    }
    doc = conditioned_document(valid)
    doc.as_h["schema"] = JSON::Any.new("cogni_ml.hy_nomas_generation_inputs/v3")
    doc.as_h["schema_version"] = JSON::Any.new(3_i64)
    doc.as_h["geometry"] = JSON.parse(geometry.to_json)

    mask = HyNomasGeneration.conditioned_mask!(doc,
      geometry: HyNomasGeneration::SpatialGeometry.parse("768x416"))
    mask.sum.should eq(valid)

    mutated = JSON.parse(doc.to_json)
    mutated["geometry"].as_h["decoder_rope_grid"] = JSON.parse([10, 18, 48].to_json)
    expect_raises(ArgumentError, /geometry mismatch/) do
      HyNomasGeneration.conditioned_mask!(mutated,
        geometry: HyNomasGeneration::SpatialGeometry.parse("768x416"))
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
