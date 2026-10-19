require "spec"
require "../src/ml/video/metal_ops"

describe ML::Video::MetalStagePool do
  it "uses page-aligned no-copy views, phase overlap, and disjoint lane allocations" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    page = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
    pool = ML::Video::MetalStagePool.new(page * 5, page * 4)
    views = [] of ML::Video::MetalStageBufferView
    begin
      first = pool.allocate!("t5", "weights", page + 1)
      views << first
      second = pool.allocate!("t5", "constants", page)
      views << second
      dit = pool.allocate!("dit", "weights", page + 1)
      views << dit
      dit_scratch = pool.allocate!("dit", "scratch", page)
      views << dit_scratch

      first.offset.should eq(0_i64)
      first.requested_bytes.should eq(page + 1)
      first.mapped_bytes.should eq(page * 2)
      first.buffer.size.should eq(page * 2)
      (first.buffer.contents.address % page.to_u64).should eq(0_u64)
      second.offset.should eq(page * 2)
      dit.offset.should eq(0_i64)
      dit.buffer.contents.should eq(first.buffer.contents)
      dit_scratch.offset.should eq(page * 2)
      pool.phase_cursor("t5").should eq(page * 3)
      pool.phase_cursor("dit").should eq(page * 3)
      pool.lane_cursor("dit", "weights").should eq(page * 2)
      pool.lane_cursor("dit", "scratch").should eq(page * 3)

      expect_raises(ArgumentError, /live stage views/) { pool.reset_phase!("dit") }
      expect_raises(ArgumentError, /live views/) { pool.release }
      expect_raises(ArgumentError, /maxBufferLength/) { pool.allocate!("too-large", "lane", page * 5) }
      expect_raises(ArgumentError, /capacity exceeded/) { pool.allocate!("t5", "overflow", page * 4) }
    ensure
      views.each do |view|
        view.release! unless view.released?
      end
      pool.release unless pool.released?
    end
  end

  it "lets Arena allocate pooled scratch while setup uploads stay outside the pool" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    page = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
    pool = ML::Video::MetalStagePool.new(page * 8, page * 4)
    arena = ML::Video::Arena.new(stage_pool: pool, stage_phase: "scratch", stage_lane: "dit")
    begin
      scratch = arena.alloc("scratch", page.to_i32 // sizeof(Float32))
      constant = arena.upload("constant", [1.0_f32, 2.0_f32])
      scratch.size.should eq(page)
      ML::Video::MetalOps.overlap?(scratch, constant).should be_false
      pool.live_view_count.should eq(1)
      expect_raises(ArgumentError, /live views/) { pool.release }
    ensure
      arena.release
      pool.live_view_count.should eq(0)
      pool.release unless pool.released?
    end
  end

  it "adopts BF16 weights only with an explicit typed view and releases its lease with the Arena" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    page = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
    pool = ML::Video::MetalStagePool.new(page * 4, page * 2)
    weights = ML::Video::Arena.new
    begin
      view = pool.allocate!("weights", "dit", page)
      buffer = weights.adopt_weight("linear.weight", view, ML::Video::WeightDType::BF16)
      weights["linear.weight"].same?(buffer).should be_true
      weights.dtype("linear.weight").should eq(ML::Video::WeightDType::BF16)
      weights.check_weight("linear.weight", 16_i64)
      expect_raises(ArgumentError, /Arena-owned/) { view.release! }
      expect_raises(ArgumentError, /live views/) { pool.release }
    ensure
      weights.release
      pool.live_view_count.should eq(0)
      pool.release unless pool.released?
    end
  end
end
