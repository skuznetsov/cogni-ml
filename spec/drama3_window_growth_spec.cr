require "spec"
require "json"
require "../src/ml/audio/drama3/window_budget"
require "../src/ml/audio/drama3/window_growth"

include ML::Audio::Drama3

describe WindowGrowth do
  it "starts at the prompt or 64 rows and clips each geometric step" do
    WindowGrowth.capacity_schedule(80, 179).should eq([80, 160, 179])
    WindowGrowth.capacity_schedule(3, 10).should eq([10])
    WindowGrowth.capacity_schedule(3, 150).should eq([64, 128, 150])
    WindowGrowth.capacity_schedule(100, 120).should eq([100, 120])
    WindowGrowth.capacity_schedule(4096, 4096).should eq([4096])
  end

  it "rejects invalid prompt and capacity bounds" do
    expect_raises(ArgumentError) { WindowGrowth.capacity_schedule(0, 10) }
    expect_raises(ArgumentError) { WindowGrowth.capacity_schedule(10, 9) }
    expect_raises(ArgumentError) { WindowGrowth.capacity_schedule(1, 4097) }
  end

  it "charges wide-attention score scratch at every generation when the final kernel is wide" do
    config = growth_backbone_config
    tower = growth_tower_config
    cache = WindowGrowth.cache_bytes_per_capacity(config, tower, 64, 1100)
    cache.score_scratch_bytes.should eq(64_i64 * config.heads.to_i64 * 4_i64)
    cache.backbone_bytes.should be > cache.score_scratch_bytes
    cache.tower_bytes.should be > 0_i64
    cache.total_bytes.should eq(cache.backbone_bytes + cache.tower_bytes + cache.score_scratch_bytes)
    cache.largest_buffer_bytes.should be >= cache.score_scratch_bytes

    narrow = WindowGrowth.cache_bytes_per_capacity(config, tower, 64, 1024)
    narrow.score_scratch_bytes.should eq(0_i64)
  end

  it "admits the largest candidate only after charging every prior generation on both budgets" do
    request = growth_request
    roomy = growth_telemetry
    full = WindowGrowth.plan(request, roomy).plan.not_nil!
    full.frames.should eq(100)
    full.final_capacity.should eq(179)
    full.capacity_schedule.should eq([80, 160, 179])
    expected_history = full.generation_cache_bytes[0] + full.generation_cache_bytes[1]
    full.prior_generation_reservation_bytes.should eq(expected_history)
    full.window_plan.additional_margin_bytes.should eq(request.additional_margin_bytes + expected_history)
    full.window_plan.metal_budget_bytes.should eq(roomy.metal_recommended_bytes.not_nil! - expected_history)
    growth_report = JSON.parse(full.to_json)
    growth_report["capacity_schedule"].as_a.map(&.as_i).should eq([80, 160, 179])
    growth_report["prior_generation_reservation_bytes"].as_i64.should eq(expected_history)

    exact_metal = growth_telemetry(
      metal_recommended_bytes: full.window_plan.estimated_bytes +
                               full.prior_generation_reservation_bytes,
    )
    exact = WindowGrowth.plan(request, exact_metal).plan.not_nil!
    exact.frames.should eq(100)

    one_byte_short = WindowGrowth.plan(request, growth_telemetry(
      metal_recommended_bytes: exact_metal.metal_recommended_bytes.not_nil! - 1_i64,
    )).plan.not_nil!
    one_byte_short.frames.should be < 100
  end

  it "rejects unknown telemetry and preserves the cold-plan minimum failure" do
    unknown = WindowGrowth.plan(growth_request, Drama3WindowBudget::Telemetry.new)
    unknown.success?.should be_false
    unknown.failure.should eq(Drama3WindowBudget::Failure::UnknownTelemetry)

    tiny = WindowGrowth.plan(
      growth_request(min_frames: 1, frame_cap: 100),
      growth_telemetry(physical_bytes: 10_000_i64, free_reclaimable_bytes: 1_600_i64,
        metal_recommended_bytes: 1_600_i64, max_buffer_length: 10_000_i64),
    )
    tiny.success?.should be_false
    tiny.failure.should eq(Drama3WindowBudget::Failure::InsufficientBudget)
  end

  it "checks replacement plus downstream reserve against host floor, Metal, process and buffer limits" do
    request_bytes = 80_i64
    downstream = 20_i64
    margin = 10_i64
    exact = WindowGrowth.admit_growth(
      request_bytes, downstream, margin,
      growth_telemetry(physical_bytes: 10_000_i64, free_reclaimable_bytes: 1_611_i64,
        metal_recommended_bytes: 1_110_i64, metal_current_allocated_bytes: 1_000_i64,
        max_buffer_length: 80_i64),
      largest_buffer_bytes: 80_i64,
      process_remaining_bytes: 111_i64,
    )
    exact.success?.should be_true
    exact_admission = exact.admission?.not_nil!
    exact_admission.required_bytes.should eq(110_i64)
    exact_report = JSON.parse(exact_admission.to_json)
    exact_report["admitted"].as_bool.should be_true
    exact_report["metal_remaining_bytes"].as_i64.should eq(110_i64)

    host_floor = WindowGrowth.admit_growth(
      request_bytes, downstream, margin,
      growth_telemetry(physical_bytes: 10_000_i64, free_reclaimable_bytes: 1_610_i64,
        metal_recommended_bytes: 2_000_i64, max_buffer_length: 80_i64),
      largest_buffer_bytes: 80_i64,
    )
    host_floor.failure.should eq(WindowGrowth::AdmissionFailure::InsufficientHostBudget)

    fractional_floor = WindowGrowth.admit_growth(
      10_i64, 0_i64, 0_i64,
      growth_telemetry(physical_bytes: 10_001_i64, free_reclaimable_bytes: 1_511_i64,
        metal_recommended_bytes: 100_i64, max_buffer_length: 10_i64),
      largest_buffer_bytes: 10_i64,
    )
    fractional_floor.success?.should be_true

    metal_short = WindowGrowth.admit_growth(
      request_bytes, downstream, margin,
      growth_telemetry(metal_recommended_bytes: 1_109_i64, metal_current_allocated_bytes: 1_000_i64),
      largest_buffer_bytes: 80_i64,
    )
    metal_short.failure.should eq(WindowGrowth::AdmissionFailure::InsufficientMetalBudget)

    process_short = WindowGrowth.admit_growth(
      request_bytes, downstream, margin, growth_telemetry,
      largest_buffer_bytes: 80_i64, process_remaining_bytes: 110_i64,
    )
    process_short.failure.should eq(WindowGrowth::AdmissionFailure::InsufficientProcessBudget)

    buffer_short = WindowGrowth.admit_growth(
      request_bytes, downstream, margin,
      growth_telemetry(max_buffer_length: 79_i64), largest_buffer_bytes: 80_i64)
    buffer_short.failure.should eq(WindowGrowth::AdmissionFailure::PerBufferLimit)
  end

  it "fails closed on missing telemetry and arithmetic overflow" do
    WindowGrowth.admit_growth(
      1_i64, 0_i64, 0_i64, Drama3WindowBudget::Telemetry.new,
      largest_buffer_bytes: 1_i64,
    ).failure.should eq(WindowGrowth::AdmissionFailure::UnknownTelemetry)

    WindowGrowth.admit_growth(
      Int64::MAX, 1_i64, 0_i64, growth_telemetry,
      largest_buffer_bytes: 1_i64,
    ).failure.should eq(WindowGrowth::AdmissionFailure::Overflow)
  end
end

private def growth_backbone_config : BackboneConfig
  BackboneConfig.new(JSON.parse(<<-JSON
    {
      "hidden_size": 8,
      "num_hidden_layers": 2,
      "num_attention_heads": 4,
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

private def growth_tower_config : TowerConfig
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

private def growth_codec_config : CodecConfig
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

private def growth_request(
  prompt_frames : Int32 = 80,
  min_frames : Int32 = 1,
  frame_cap : Int32 = 100,
  additional_margin_bytes : Int64 = 20_i64,
) : Drama3WindowBudget::Request
  Drama3WindowBudget::Request.new(
    growth_backbone_config,
    growth_tower_config,
    growth_codec_config,
    prompt_frames,
    min_frames,
    frame_cap,
    100_i64,
    100_i64,
    50_i64,
    50_i64,
    additional_margin_bytes,
    8,
    8,
  )
end

private def growth_telemetry(
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
