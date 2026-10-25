require "spec"
require "../examples/support/hy_nomas_residency_plan"
require "./support/video_allocation_counter"

private def residency_config(video = 6, text = 23)
  ML::Video::HyNomasConfig.new(batches: 1, video_tokens: video,
    text_tokens: text, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192)
end

describe HyNomasResidencyPlan do
  it "reconciles the actual two-block short scratch/workspace receipt without allocating" do
    allocations = ML::MetalBuffer.video_spec_allocations
    geometry = HyNomasResidencyPlan.geometry(residency_config)
    geometry.scratch_bytes.should eq(7_815_680)
    geometry.output_pair_bytes.should eq(237_568)
    geometry.workspace_bytes.should eq(102_092_416)
    geometry.shared_bytes(2).should eq(110_145_664)
    geometry.input_bytes.should eq(260_096)
    ML::MetalBuffer.video_spec_allocations.should eq(allocations)
  end

  it "uses the native full diagnostic geometry with compact C256/G16 attention" do
    geometry = HyNomasResidencyPlan.geometry(residency_config(5280, 1280), 256, 16)
    geometry.scratch_bytes.should eq(1_806_287_360)
    geometry.output_pair_bytes.should eq(53_739_520)
    geometry.weight_expansion_bytes.should eq(100_663_296)
    geometry.attention_stage_bytes.should eq(111_673_344)
    geometry.rms_bytes.should eq(108_318_720)
    geometry.score_tile_bytes.should eq(107_479_040)
    geometry.graph_calls.should eq(26)
    geometry.shared_bytes(54).should eq(4_975_137_280)
  end

  it "keeps distinct output retention separate from proposed two-bank storage" do
    geometry = HyNomasResidencyPlan.geometry(residency_config(5280, 1280), 256, 16)
    geometry.proposed_two_bank_bytes(54).should eq(geometry.shared_bytes(2))
    (geometry.shared_bytes(54) - geometry.proposed_two_bank_bytes(54)).should eq(52_i64 * 53_739_520)
    expect_raises(ArgumentError) { geometry.shared_bytes(0) }
    expect_raises(ArgumentError) { geometry.proposed_two_bank_bytes(1) }
    expect_raises(ArgumentError) { HyNomasResidencyPlan.geometry(residency_config, 4, 4) }
  end

  it "counts all same-chain F32 conversion sources as simultaneously retained" do
    geometry = HyNomasResidencyPlan.geometry(residency_config)
    weights = 2_i64 * 604_227_584
    scenario = HyNomasResidencyPlan::Scenario.new(geometry, 2, weights)
    scenario.bf16_weights_bytes.should eq(604_227_584)
    scenario.f32_conversion_staging_bytes.should eq(1_208_455_168)
    scenario.cold_named_bytes.should eq(weights + weights // 2 + 260_096 + 110_145_664)
    (scenario.cold_named_bytes - scenario.warm_named_bytes).should eq(weights)
    scenario.retained_checkpoint_bytes.should eq(12_746_752)
  end

  it "treats reserve heuristics and omitted pipeline allocations explicitly" do
    geometry = HyNomasResidencyPlan.geometry(residency_config(5280, 1280), 256, 16)
    scenario = HyNomasResidencyPlan::Scenario.new(geometry, 54, 34_213_738_368_i64)
    scenario.reserve_heuristic_bytes.should eq(3_i64 * 107_479_040 + 256_i64 * 1024 * 1024)
    scenario.estimated_cold_bytes.should eq(scenario.cold_named_bytes + scenario.reserve_heuristic_bytes)
    scenario.estimated_warm_bytes.should eq(scenario.warm_named_bytes + scenario.reserve_heuristic_bytes)
    scenario.bf16_weights_bytes.should eq(17_106_869_184)
    scenario.implemented?.should be_false
    scenario.full_pipeline_accounted?.should be_false
  end

  it "rejects a named underestimate, including at a one-byte boundary" do
    HyNomasResidencyPlan.require_budget!(100_i64, 100_i64)
    expect_raises(ArgumentError) { HyNomasResidencyPlan.require_budget!(101_i64, 100_i64) }
    expect_raises(ArgumentError) { HyNomasResidencyPlan.require_budget!(0_i64, 0_i64) }
    expect_raises(ArgumentError) { HyNomasResidencyPlan.require_budget!(-1_i64, 100_i64) }
    expect_raises(ArgumentError) do
      HyNomasResidencyPlan::Scenario.new(HyNomasResidencyPlan.geometry(residency_config), 2, 1_i64)
    end
  end

  if path = ENV["HY_NOMAS_RESIDENCY_HEADER"]?
    it "authenticates and enumerates the complete pinned header against native block bindings" do
      allocations = ML::MetalBuffer.video_spec_allocations
      inventory = HyNomasResidencyPlan::Inventory.load!(path)
      inventory.tensor_count.should eq(2268)
      inventory.block_count.should eq(54)
      inventory.f32_bytes.should eq(34_213_738_368)
      inventory.block_f32_bytes.should eq(32_628_289_536)
      inventory.groups.size.should eq(9)
      inventory.groups["action_modules"].tensors.should eq(486)
      inventory.groups["txt_in"].f32_bytes.should eq(564_420_608)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    end

    it "rejects a same-length tampered header before any GPU allocation" do
      allocations = ML::MetalBuffer.video_spec_allocations
      raw = File.read(path)
      bytes = raw.to_slice.dup
      bytes[100] ^= 1_u8
      expect_raises(ArgumentError, /header digest/) do
        HyNomasResidencyPlan::Inventory.from_authenticated_bytes!(String.new(bytes))
      end
      expect_raises(ArgumentError, /header length/) do
        HyNomasResidencyPlan::Inventory.from_authenticated_bytes!("{}")
      end
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    end
  else
    pending "set HY_NOMAS_RESIDENCY_HEADER for the authenticated external header controls" do
    end
  end
end
