require "spec"
require "../src/ml/video/wan_s2v_block"

private def wan_scratch_config(ffn_dim : Int32 = 13_824) : ML::Video::WanS2VConfig
  ML::Video::WanS2VConfig.new(
    batches: 1, seq: 36_608, context_tokens: 512, dim: 5_120, heads: 40,
    ffn_dim: ffn_dim, split: 34_944, theta: 10_000.0_f32,
    video_frames: 21, spatial_tokens: 1_664,
  )
end

describe "Wan block scratch lifetime plan" do
  it "keeps the default layout and saves five hidden-state buffers only when requested" do
    cfg = wan_scratch_config
    ML::Video::WanS2VBlock.scratch_plan(cfg).size.should eq(12)
    ML::Video::WanS2VBlock.scratch_bytes(cfg).should eq(8_792_855_040_i64)
    ML::Video::WanS2VBlock.scratch_plan(cfg, reuse_scratch: true).size.should eq(7)
    compact = ML::Video::WanS2VBlock.scratch_bytes(cfg, reuse_scratch: true)
    compact.should eq(5_044_195_840_i64)
    (ML::Video::WanS2VBlock.scratch_bytes(cfg) - compact).should eq(5_i64 * cfg.values * 4)
  end

  it "keeps self residual storage separate from the later cross attention output" do
    plan = ML::Video::WanS2VBlock.scratch_plan(wan_scratch_config, reuse_scratch: true).to_h
    plan["wan_b_self_res"].should eq(187_432_960)
    plan.has_key?("wan_b_self_attn").should be_false
    ML::Video::WanS2VBlock::SCRATCH_REUSE_ALIASES.has_key?("wan_b_self_res").should be_false
  end

  it "sizes the Q and FFN owner for the larger lifetime even for a narrow FFN" do
    cfg = wan_scratch_config(2_560)
    plan = ML::Video::WanS2VBlock.scratch_plan(cfg, reuse_scratch: true).to_h
    plan["wan_b_self_q"].should eq(cfg.values)
    expected = (4_i64 * cfg.values + cfg.values + 2_i64 * cfg.context_rows * cfg.dim + cfg.dim + cfg.head_dim) * 4
    ML::Video::WanS2VBlock.scratch_bytes(cfg, reuse_scratch: true).should eq(expected)
  end

  it "preserves the independent RoPE table accounting in both layouts" do
    cfg = wan_scratch_config
    [false, true].each do |reuse|
      base = ML::Video::WanS2VBlock.scratch_bytes(cfg, reuse_scratch: reuse)
      tables = ML::Video::WanS2VBlock.scratch_bytes(cfg, precompute_rope: true, reuse_scratch: reuse)
      (tables - base).should eq(cfg.batches.to_i64 * cfg.seq * cfg.head_dim * 4)
    end
  end
end
