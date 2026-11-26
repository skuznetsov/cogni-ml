require "spec"
require "json"
require "../src/ml/audio/drama3/window_budget"
require "../src/ml/audio/drama3/studio/frame_window"

include ML::Audio::Drama3

describe ML::Audio::Drama3::Studio::FrameWindow do
  it "allocates all requested manual positions and rejects overflowing prompt plus frames" do
    window = ML::Audio::Drama3::Studio::FrameWindow.manual(269, 512)
    window.frames.should eq(512)
    window.capacity.should eq(780)
    window.maximum_capacity.should eq(780)
    window.growth_reservation_bytes.should eq(0_i64)
    window.growth_plan.should be_nil
    window.plan.should be_nil
    ML::Audio::Drama3::Studio::FrameWindow.manual(4096, 1).capacity.should eq(4096)
    expect_raises(ArgumentError) { ML::Audio::Drama3::Studio::FrameWindow.manual(4096, 2) }
    expect_raises(ArgumentError) { ML::Audio::Drama3::Studio::FrameWindow.manual(0, 10) }
  end

  it "uses live advisory admission up to the explicit ceiling and rejects unknown telemetry" do
    window = ML::Audio::Drama3::Studio::FrameWindow.automatic(budget_request, budget_telemetry, false, true)
    window.frames.should eq(8)
    window.capacity.should eq(10)
    window.maximum_capacity.should eq(10)
    window.growth_plan.not_nil!.capacity_schedule.should eq([10])
    window.plan.not_nil!.advisory_only.should be_true
    expect_raises(ArgumentError, /UnknownTelemetry/) do
      ML::Audio::Drama3::Studio::FrameWindow.automatic(budget_request, Drama3WindowBudget::Telemetry.new, false, true)
    end
  end

  it "separates automatic initial cache capacity from its pre-admitted final capacity" do
    request = budget_request(prompt_frames: 80, frame_cap: 100)
    window = ML::Audio::Drama3::Studio::FrameWindow.automatic(request, budget_telemetry, false, true)
    window.frames.should eq(100)
    window.capacity.should eq(80)
    window.maximum_capacity.should eq(179)
    window.growth_plan.not_nil!.capacity_schedule.should eq([80, 160, 179])
    window.growth_reservation_bytes.should eq(window.growth_plan.not_nil!.prior_generation_reservation_bytes)
    window.plan.should eq(window.growth_plan.not_nil!.window_plan)
  end

  it "rejects uncalibrated reference encoding and incompatible DAC layouts" do
    expect_raises(ArgumentError, /voice references/) do
      ML::Audio::Drama3::Studio::FrameWindow.automatic(budget_request, budget_telemetry, true, true)
    end
    expect_raises(ArgumentError, /DRAMA3_DAC_SCRATCH/) do
      ML::Audio::Drama3::Studio::FrameWindow.automatic(budget_request, budget_telemetry, false, false)
    end
  end
end

private def budget_backbone_config : BackboneConfig
  BackboneConfig.new(JSON.parse(<<-JSON
    {
      "hidden_size": 8,
      "num_hidden_layers": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 4,
      "num_experts": 4,
      "num_experts_per_tok": 2,
      "full_attention_interval": 1,
      "linear_num_key_heads": 1,
      "linear_num_value_heads": 1,
      "linear_key_head_dim": 2,
      "linear_value_head_dim": 2,
      "linear_conv_kernel_dim": 2,
      "rope_parameters": {"partial_rotary_factor": 0.5},
      "layer_types": ["full_attention", "full_attention"]
    }
    JSON
  ))
end

private def budget_tower_config : TowerConfig
  TowerConfig.new(
    vocab_size: 8,
    num_codebooks: 3,
    dim: 4,
    n_head: 2,
    n_local_heads: 1,
    head_dim: 2,
    intermediate_size: 8,
    rotary_percent: 1.0,
    shared_enrichment_layers: 1,
  )
end

private def budget_codec_config : CodecConfig
  CodecConfig.new(
    latent_channels: 4,
    codebook_dim: 2,
    semantic_codebook_size: 8,
    residual_codebooks: 2,
    residual_codebook_size: 8,
    post_layers: 1,
    transformer_heads: 2,
    transformer_head_dim: 2,
    transformer_ff_dim: 8,
    transformer_window: 4,
    quantizer_upsample_rates: [2],
    decoder_channels: 8,
    decoder_rates: [2],
  )
end

private def budget_request(
  prompt_frames : Int32 = 3,
  min_frames : Int32 = 1,
  frame_cap : Int32 = 8,
  checkpoint_mapped_bytes : Int64 = 1_000_i64,
  largest_checkpoint_mapped_buffer_bytes : Int64 = 1_000_i64,
  converted_static_bytes : Int64 = 500_i64,
  largest_converted_buffer_bytes : Int64 = 300_i64,
  additional_margin_bytes : Int64 = 1_000_i64,
  moe_intermediate_size : Int32? = 8,
  shared_expert_intermediate_size : Int32? = 8,
  reference_encoder_peak_bytes : Int64? = nil,
) : Drama3WindowBudget::Request
  Drama3WindowBudget::Request.new(
    budget_backbone_config,
    budget_tower_config,
    budget_codec_config,
    prompt_frames,
    min_frames,
    frame_cap,
    checkpoint_mapped_bytes,
    largest_checkpoint_mapped_buffer_bytes,
    converted_static_bytes,
    largest_converted_buffer_bytes,
    additional_margin_bytes,
    moe_intermediate_size,
    shared_expert_intermediate_size,
    reference_encoder_peak_bytes,
  )
end

private def budget_telemetry(
  physical_bytes : Int64 = 1_000_000_i64,
  free_reclaimable_bytes : Int64 = 900_000_i64,
  metal_recommended_bytes : Int64 = 1_000_000_i64,
  metal_current_allocated_bytes : Int64 = 0_i64,
  max_buffer_length : Int64 = 1_000_000_i64,
) : Drama3WindowBudget::Telemetry
  Drama3WindowBudget::Telemetry.new(
    physical_bytes,
    free_reclaimable_bytes,
    metal_recommended_bytes,
    metal_current_allocated_bytes,
    max_buffer_length,
  )
end

describe Drama3WindowBudget do
  it "fails closed when any telemetry field is unknown" do
    request = budget_request
    result = Drama3WindowBudget.experimental_plan(request, Drama3WindowBudget::Telemetry.new)
    result.success?.should be_false
    result.failure.should eq(Drama3WindowBudget::Failure::UnknownTelemetry)
  end

  it "retains a fifteen percent physical-memory reserve plus the extra margin" do
    result = Drama3WindowBudget.experimental_plan(budget_request, budget_telemetry)
    plan = result.plan.not_nil!
    plan.host_reserved_bytes.should eq(150_000_i64)
    plan.host_budget_bytes.should eq(749_000_i64)
    plan.additional_margin_bytes.should eq(1_000_i64)
    plan.advisory_only.should be_true
    plan.host_gc_retention_certified.should be_false
    plan.breakdown.codec_weightnorm_temporaries_included_bytes.should be > 0_i64
    report = JSON.parse(plan.to_json)
    report["advisory_only"].as_bool.should be_true
    report["host_gc_retention_certified"].as_bool.should be_false
    report["breakdown"]["codec_weightnorm_temporaries_included_bytes"].as_i64.should be > 0_i64
  end

  it "rejects prompts beyond the 4096-position ceiling" do
    result = Drama3WindowBudget.experimental_plan(
      budget_request(prompt_frames: 4097),
      budget_telemetry,
    )
    result.success?.should be_false
    result.failure.should eq(Drama3WindowBudget::Failure::PromptExceedsPositionCap)
  end

  it "admits an exact memory boundary and rejects a one-byte-short budget" do
    request = budget_request(prompt_frames: 1, frame_cap: 1, additional_margin_bytes: 0_i64)
    roomy = budget_telemetry(physical_bytes: 1_000_000_000_i64,
      free_reclaimable_bytes: 1_000_000_000_i64,
      metal_recommended_bytes: 1_000_000_000_i64,
      max_buffer_length: 1_000_000_i64)
    fitted = Drama3WindowBudget.experimental_plan(request, roomy).plan.not_nil!
    reserve = fitted.host_reserved_bytes
    exact_free = reserve + fitted.estimated_bytes

    exact = Drama3WindowBudget.experimental_plan(request, budget_telemetry(
      physical_bytes: 1_000_000_000_i64,
      free_reclaimable_bytes: exact_free,
      metal_recommended_bytes: 1_000_000_000_i64,
      max_buffer_length: 1_000_000_i64,
    ))
    exact.success?.should be_true
    exact.plan.not_nil!.estimated_bytes.should eq(fitted.estimated_bytes)

    short = Drama3WindowBudget.experimental_plan(request, budget_telemetry(
      physical_bytes: 1_000_000_000_i64,
      free_reclaimable_bytes: exact_free - 1,
      metal_recommended_bytes: 1_000_000_000_i64,
      max_buffer_length: 1_000_000_i64,
    ))
    short.success?.should be_false
    short.failure.should eq(Drama3WindowBudget::Failure::InsufficientBudget)
  end

  it "rejects checked Int64 overflow instead of wrapping an estimate" do
    request = budget_request(
      checkpoint_mapped_bytes: Int64::MAX,
      largest_checkpoint_mapped_buffer_bytes: Int64::MAX,
    )
    result = Drama3WindowBudget.experimental_plan(request, budget_telemetry)
    result.success?.should be_false
    result.failure.should eq(Drama3WindowBudget::Failure::Overflow)
  end

  it "enforces the maximum individual Metal buffer length" do
    result = Drama3WindowBudget.experimental_plan(
      budget_request,
      budget_telemetry(max_buffer_length: 1_i64),
    )
    result.success?.should be_false
    result.failure.should eq(Drama3WindowBudget::Failure::PerBufferLimit)
  end

  it "admits the additive QKV buffer at its exact size and rejects one byte less" do
    backbone = BackboneConfig.new(JSON.parse(<<-JSON
      {
        "hidden_size": 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "full_attention_interval": 2,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 14,
        "linear_value_head_dim": 1,
        "linear_conv_kernel_dim": 1,
        "rope_parameters": {"partial_rotary_factor": 1.0}
      }
      JSON
    ))
    tower = TowerConfig.new(
      vocab_size: 1,
      num_codebooks: 2,
      dim: 1,
      n_head: 1,
      n_local_heads: 1,
      head_dim: 2,
      intermediate_size: 1,
      rotary_percent: 1.0,
      shared_enrichment_layers: 1,
      vq_backbone_codebooks: 0,
    )
    codec = CodecConfig.new(
      latent_channels: 1,
      codebook_dim: 1,
      semantic_codebook_size: 1,
      residual_codebooks: 1,
      residual_codebook_size: 1,
      post_layers: 0,
      transformer_heads: 1,
      transformer_head_dim: 1,
      transformer_ff_dim: 1,
      transformer_window: 1,
      quantizer_upsample_rates: [1],
      decoder_channels: 2,
      decoder_rates: [1],
    )
    request = Drama3WindowBudget::Request.new(
      backbone, tower, codec,
      1, 1, 1,
      0_i64, 0_i64, 0_i64, 0_i64, 0_i64,
      1, 1,
    )

    exact = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(max_buffer_length: 116_i64),
    )
    exact.success?.should be_true
    exact.plan.not_nil!.breakdown.largest_buffer_bytes.should eq(116_i64)

    one_byte_short = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(max_buffer_length: 115_i64),
    )
    one_byte_short.success?.should be_false
    one_byte_short.failure.should eq(Drama3WindowBudget::Failure::PerBufferLimit)
  end

  it "admits linear convolution history at its exact size and rejects one byte less" do
    backbone = BackboneConfig.new(JSON.parse(<<-JSON
      {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "full_attention_interval": 2,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 16,
        "rope_parameters": {"partial_rotary_factor": 1.0}
      }
      JSON
    ))
    tower = TowerConfig.new(
      vocab_size: 1,
      num_codebooks: 2,
      dim: 2,
      n_head: 1,
      n_local_heads: 1,
      head_dim: 2,
      intermediate_size: 1,
      rotary_percent: 1.0,
      shared_enrichment_layers: 1,
      vq_backbone_codebooks: 0,
    )
    codec = CodecConfig.new(
      latent_channels: 1,
      codebook_dim: 1,
      semantic_codebook_size: 1,
      residual_codebooks: 1,
      residual_codebook_size: 1,
      post_layers: 0,
      transformer_heads: 1,
      transformer_head_dim: 1,
      transformer_ff_dim: 1,
      transformer_window: 1,
      quantizer_upsample_rates: [1],
      decoder_channels: 2,
      decoder_rates: [1],
    )
    request = Drama3WindowBudget::Request.new(
      backbone, tower, codec,
      1, 1, 1,
      0_i64, 0_i64, 0_i64, 0_i64, 0_i64,
      1, 1,
    )

    exact = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(max_buffer_length: 768_i64),
    )
    exact.success?.should be_true
    exact.plan.not_nil!.breakdown.largest_buffer_bytes.should eq(768_i64)

    one_byte_short = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(max_buffer_length: 767_i64),
    )
    one_byte_short.success?.should be_false
    one_byte_short.failure.should eq(Drama3WindowBudget::Failure::PerBufferLimit)
  end

  it "admits full-attention q projection at its exact size and rejects one byte less" do
    backbone = BackboneConfig.new(JSON.parse(<<-JSON
      {
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "full_attention_interval": 1,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_key_head_dim": 2,
        "linear_value_head_dim": 2,
        "linear_conv_kernel_dim": 1,
        "rope_parameters": {"partial_rotary_factor": 1.0},
        "layer_types": ["full_attention"]
      }
      JSON
    ))
    tower = TowerConfig.new(
      vocab_size: 1,
      num_codebooks: 2,
      dim: 2,
      n_head: 1,
      n_local_heads: 1,
      head_dim: 2,
      intermediate_size: 1,
      rotary_percent: 1.0,
      shared_enrichment_layers: 1,
      vq_backbone_codebooks: 0,
    )
    codec = CodecConfig.new(
      latent_channels: 1,
      codebook_dim: 1,
      semantic_codebook_size: 1,
      residual_codebooks: 1,
      residual_codebook_size: 1,
      post_layers: 0,
      transformer_heads: 1,
      transformer_head_dim: 1,
      transformer_ff_dim: 1,
      transformer_window: 1,
      quantizer_upsample_rates: [1],
      decoder_channels: 2,
      decoder_rates: [1],
    )
    request = Drama3WindowBudget::Request.new(
      backbone, tower, codec,
      1, 1, 1,
      0_i64, 0_i64, 0_i64, 0_i64, 0_i64,
      1, 1,
    )

    exact = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(max_buffer_length: 256_i64),
    )
    exact.success?.should be_true
    exact.plan.not_nil!.breakdown.largest_buffer_bytes.should eq(256_i64)

    one_byte_short = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(max_buffer_length: 255_i64),
    )
    one_byte_short.success?.should be_false
    one_byte_short.failure.should eq(Drama3WindowBudget::Failure::PerBufferLimit)
  end

  it "estimates monotonically and never shrinks the selected window when free memory grows" do
    request = budget_request(prompt_frames: 2, frame_cap: 8)
    roomy = budget_telemetry
    one = Drama3WindowBudget.experimental_plan(
      budget_request(prompt_frames: 2, frame_cap: 1), roomy,
    ).plan.not_nil!
    two = Drama3WindowBudget.experimental_plan(
      budget_request(prompt_frames: 2, frame_cap: 2), roomy,
    ).plan.not_nil!
    eight = Drama3WindowBudget.experimental_plan(request, roomy).plan.not_nil!
    one.estimated_bytes.should be <= two.estimated_bytes
    two.estimated_bytes.should be <= eight.estimated_bytes

    constrained = Drama3WindowBudget.experimental_plan(
      request,
      budget_telemetry(free_reclaimable_bytes: 200_000_i64),
    ).plan.not_nil!
    unconstrained = Drama3WindowBudget.experimental_plan(request, roomy).plan.not_nil!
    unconstrained.generated_frames.should be >= constrained.generated_frames
  end
end
