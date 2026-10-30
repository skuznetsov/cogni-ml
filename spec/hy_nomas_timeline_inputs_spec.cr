require "spec"
require "../examples/support/hy_nomas_timeline_inputs"

private def timeline_input_qwen_source(prompt : String) : Hash(String, JSON::Any)
  prompt_sha = Digest::SHA256.hexdigest(prompt)
  mask = Array.new(1280) { |index| JSON::Any.new(index < 67 ? 1_i64 : 0_i64) }
  contract = JSON.parse({
    "prompt"                 => prompt,
    "prompt_sha256"          => prompt_sha,
    "upstream_revision"      => HyNomasGeneration::TimelineInputs::PINNED_REVISION,
    "template_name"          => "li-dit-encode-video-json",
    "template_sha256"        => HyNomasGeneration::PROMPT_TEMPLATE_SHA,
    "crop_start"             => 108,
    "raw_length"             => 1388,
    "cropped_attention_mask" => mask,
    "untruncated_preflight"  => {"truncation_required" => false},
  }.to_json)
  JSON.parse({
    "prompt_sha256"          => prompt_sha,
    "prompt_contract"        => contract,
    "upstream_revision"      => HyNomasGeneration::TimelineInputs::PINNED_REVISION,
    "template_name"          => "li-dit-encode-video-json",
    "template_sha256"        => HyNomasGeneration::PROMPT_TEMPLATE_SHA,
    "state_index"            => 26,
    "hidden_size"            => 3584,
    "crop_tokens"            => 1280,
    "crop"                   => [108, 1388],
    "raw_sequence_length"    => 1388,
    "sidecar_bytes"          => 27_i64 * 1388 * 3584 * 2,
    "cropped_attention_mask" => mask,
    "valid_tokens"           => 67,
  }.to_json).as_h
end

private def timeline_inputs_metadata : JSON::Any
  inputs = HyNomasGeneration::TimelineInputs
  prompt = "A tram follows the rails through the intersection."
  records = {} of String => JSON::Any
  offset = 0_i64
  HyNomasGeneration::TimelineInputs::TENSOR_SPECS.each do |name, (dtype, shape)|
    nbytes = shape.reduce(1_i64) { |product, extent| product * extent } * (dtype == "BF16" ? 2_i64 : 4_i64)
    records[name] = JSON.parse({
      "dtype"  => dtype,
      "shape"  => shape,
      "offset" => offset,
      "nbytes" => nbytes,
      "sha256" => "a" * 64,
    }.to_json)
    offset += nbytes
  end
  geometry = JSON.parse({
    "total_latents"                       => 180,
    "segments"                            => 6,
    "segment_latents"                     => 30,
    "total_pixel_frames"                  => 720,
    "segment_pixel_frames"                => 120,
    "latent_bank_shape"                   => [1, 32, 180, 9, 16],
    "full_latent_draw_shape"              => [1, 32, 720, 9, 16],
    "dit_rope_offsets"                    => [0, 30, 60, 90, 120, 150],
    "dit_rope_grid"                       => [30, 9, 16],
    "dit_rope_axis_dims"                  => [16, 56, 56],
    "dit_rope_theta"                      => 256.0,
    "cache_eps_shape"                     => [6, 1, 32, 30, 9, 16],
    "qwen_shape"                          => [1, 1280, 3584],
    "decoder_noise_shape"                 => [1, 3, 720, 288, 512],
    "decoder_rope_grid"                   => [10, 18, 32],
    "decoder_rope_shape"                  => [1, 5760, 1, 128],
    "decoder_rope_sections"               => [64, 32, 32],
    "decoder_rope_theta"                  => [1_000_000.0, 10_000.0, 10_000.0],
    "decoder_local_attention_tile_tokens" => 8,
  }.to_json)
  noise_seeds = JSON.parse({
    "latent_seed"               => 42,
    "latent_generator_device"   => "mps",
    "latent_draw_shape"         => [1, 32, 720, 9, 16],
    "latent_export_slice"       => [0, 180],
    "upstream_interactive_bank" => {
      "seconds"          => 120,
      "video_fps"        => 24,
      "latent_frames"    => 720,
      "seed"             => 42,
      "generator_device" => "cpu",
      "note"             => "MPS values are not claimed identical to upstream CPU torch.randn values",
    },
    "pixel_noise_seed"                        => 42,
    "pixel_noise_seed_formula"                => "seed * 1000003 + global_latent_frame_index",
    "pixel_noise_latent_indices"              => (0...180).to_a,
    "cache_eps_seed"                          => 42,
    "cache_eps_generator_device"              => "mps",
    "cache_eps_sample_dtype"                  => "F32",
    "cache_eps_output_dtype"                  => "BF16",
    "cache_eps_draw_shape"                    => [1, 32, 30, 9, 16],
    "cache_eps_draw_count"                    => 6,
    "cache_eps_draw_order"                    => "six sequential segment-shaped draws from a separate MPS generator",
    "cache_eps_stock_upstream_cpu_rng_parity" => false,
  }.to_json)
  cache_writer = JSON.parse({
    "config_path"                             => "configs/experiment/inference.yaml",
    "config_sha256"                           => HyNomasGeneration::TimelineInputs::SOURCE_SHA256["configs/experiment/inference.yaml"],
    "source_config_default_infer_steps_video" => 8,
    "resolved_override"                       => {"inference.infer_steps_video" => 4},
    "preserve_kv_cache"                       => true,
    "fixed_initial_noise_timeline"            => {
      "seconds"                             => 120,
      "video_fps"                           => 24,
      "latent_frames"                       => 720,
      "seed"                                => 42,
      "generator_device"                    => "mps",
      "sample_dtype"                        => "F32",
      "stock_upstream_generator_device"     => "cpu",
      "stock_upstream_sample_dtype"         => "BF16",
      "stock_upstream_rng_values_identical" => false,
    },
    "cache_at_step"                     => -1,
    "cache_noise_sigma"                 => 0.206897,
    "cache_timestep"                    => 206.897,
    "cache_latent_source"               => "final denoised segment latent selected by cache_at_step=-1",
    "cache_write_formula"               => "((1-sigma)*cache_latents.float()+sigma*cache_eps.float()).to(cache_latents.dtype)",
    "cache_eps_tensor"                  => "cache.eps_bf16",
    "cache_eps_seed"                    => 42,
    "cache_eps_generator_device"        => "mps",
    "cache_eps_sample_dtype"            => "F32 then cast to BF16",
    "cache_eps_dtype"                   => "BF16",
    "cache_eps_upstream_cpu_rng_parity" => false,
    "cache_eps_draws"                   => Array.new(6) do |index|
      {
        "segment_index"       => index,
        "global_latent_start" => index * 30,
        "global_latent_stop"  => (index + 1) * 30,
        "shape"               => [1, 32, 30, 9, 16],
        "dtype"               => "BF16",
        "generator_device"    => "mps",
        "seed"                => 42,
        "draw_order"          => index,
      }
    end,
  }.to_json)
  qwen_source = timeline_input_qwen_source(prompt)
  bundle = JSON.parse({
    "manifest_path"         => "/tmp/inputs-detailed.json",
    "manifest_sha256"       => "b" * 64,
    "payload_path"          => "/tmp/inputs-detailed.tensors.bin",
    "payload_sha256"        => "c" * 64,
    "qwen_tensor_sha256"    => records["input.qwen_bf16"]["sha256"],
    "first_rope_cos_sha256" => "d" * 64,
    "first_rope_sin_sha256" => "e" * 64,
    "first_rope_atol"       => 2e-5,
    "first_rope_rtol"       => 2e-5,
  }.to_json)
  producer = JSON.parse({
    "device"                   => "mps",
    "mps_fallback"             => "disabled",
    "model_inference"          => false,
    "tensor_generation_device" => "mps",
    "host_export"              => "one packed terminal MPS-to-host readback",
    "torch_version"            => "2.6.0",
    "python_version"           => "3.12.0",
    "rope_first_segment_check" => {
      "cos_source_sha256"      => bundle["first_rope_cos_sha256"],
      "cos_output_sha256"      => "f" * 64,
      "cos_max_abs_error"      => 0.0,
      "sin_source_sha256"      => bundle["first_rope_sin_sha256"],
      "sin_output_sha256"      => "1" * 64,
      "sin_max_abs_error"      => 0.0,
      "atol"                   => 2e-5,
      "rtol"                   => 2e-5,
      "source_manifest_sha256" => bundle["manifest_sha256"],
    },
  }.to_json)
  JSON.parse({
    "schema"            => HyNomasGeneration::TimelineInputs::SCHEMA,
    "schema_version"    => HyNomasGeneration::TimelineInputs::SCHEMA_VERSION,
    "payload"           => "continuation.tensors.bin",
    "payload_bytes"     => offset,
    "payload_sha256"    => "2" * 64,
    "tensors"           => records,
    "geometry"          => geometry,
    "prompt"            => prompt,
    "qwen_source"       => qwen_source,
    "qwen_input_bundle" => bundle,
    "noise_seeds"       => noise_seeds,
    "cache_writer"      => cache_writer,
    "upstream_revision" => HyNomasGeneration::TimelineInputs::PINNED_REVISION,
    "upstream_sources"  => HyNomasGeneration::TimelineInputs::SOURCE_SHA256,
    "producer"          => producer,
  }.to_json)
end

describe HyNomasGeneration::TimelineInputs do
  it "admits the exact six-segment inventory without a payload allocation" do
    document = timeline_inputs_metadata
    bytes = document["payload_bytes"].as_i64

    HyNomasGeneration::TimelineInputs.validate_document!(document, bytes)
    bytes.should eq(1_320_611_840_i64)
    document["tensors"]["decoder.noise"]["nbytes"].as_i64.should eq(1_274_019_840_i64)
  end

  it "rejects changed tensor dimensions, offsets, or payload sizes" do
    document = timeline_inputs_metadata
    document["tensors"].as_h["decoder.noise"].as_h["nbytes"] = JSON::Any.new(1_i64)
    expect_raises(ArgumentError, /tensor size/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end

    document = timeline_inputs_metadata
    document["geometry"].as_h["dit_rope_offsets"] = JSON.parse("[0, 30, 60, 90, 120, 151]")
    expect_raises(ArgumentError, /RoPE offsets/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end
  end

  it "rejects cache timestep, cache mask, and prompt identity drift" do
    document = timeline_inputs_metadata
    document["cache_writer"].as_h["cache_timestep"] = JSON::Any.new(0.0)
    expect_raises(ArgumentError, /cache writer/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end

    document = timeline_inputs_metadata
    mask = document["qwen_source"]["cropped_attention_mask"].as_a
    mask[2] = JSON::Any.new(0_i64)
    mask[100] = JSON::Any.new(1_i64)
    expect_raises(ArgumentError, /visible text prefix/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end

    document = timeline_inputs_metadata
    document.as_h["prompt"] = JSON::Any.new("A different scene")
    expect_raises(ArgumentError, /prompt and pinned geometry/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end
  end

  it "rejects changed Qwen lineage, source pins, or RNG claims" do
    document = timeline_inputs_metadata
    document["qwen_input_bundle"].as_h["qwen_tensor_sha256"] = JSON::Any.new("3" * 64)
    expect_raises(ArgumentError, /Qwen input bundle/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end

    document = timeline_inputs_metadata
    document["upstream_sources"].as_h["hymm/pipeline/causal_ar.py"] = JSON::Any.new("4" * 64)
    expect_raises(ArgumentError, /pinned upstream source/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end

    document = timeline_inputs_metadata
    document["noise_seeds"].as_h["cache_eps_stock_upstream_cpu_rng_parity"] = JSON::Any.new(true)
    expect_raises(ArgumentError, /RNG provenance/) do
      HyNomasGeneration::TimelineInputs.validate_document!(document, document["payload_bytes"].as_i64)
    end
  end
end
