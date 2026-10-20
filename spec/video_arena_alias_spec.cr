require "spec"
require "../src/ml/video/metal_ops"
require "./support/video_allocation_counter"

describe ML::Video::Arena do
  it "aliases normal F32 buffers by identity without allocating or double-counting resident bytes" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    arena = ML::Video::Arena.new
    begin
      owner = arena.alloc("owner", 16)
      allocations = ML::MetalBuffer.video_spec_allocations

      alias_buffer = arena.alias_f32("alias", "owner")

      alias_buffer.same?(owner).should be_true
      arena["alias"].same?(owner).should be_true
      arena.dtype("alias").should eq(ML::Video::WeightDType::F32)
      arena.resident_bytes.should eq(owner.size)
      ML::MetalBuffer.video_spec_allocations.should eq(allocations)
    ensure
      arena.release
    end
  end

  it "keeps ordinary storage alive until its last normal or typed name is removed" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    arena = ML::Video::Arena.new
    begin
      owner = arena.alloc("owner_first", 8)
      alias_buffer = arena.alias_f32("alias_first", "owner_first")
      other_owner = arena.alloc("owner_last", 4)
      other_alias = arena.alias_f32("alias_last", "owner_last")

      arena.resident_bytes.should eq(owner.size + other_owner.size)

      arena.remove("owner_first")
      alias_buffer.valid?.should be_true
      arena.resident_bytes.should eq(other_owner.size + alias_buffer.size)
      arena.remove("alias_first")
      alias_buffer.valid?.should be_false

      arena.remove("alias_last")
      other_owner.valid?.should be_true
      arena.resident_bytes.should eq(other_owner.size)
      arena.remove("owner_last")
      other_owner.valid?.should be_false
      arena.resident_bytes.should eq(0_i64)
    ensure
      arena.release
    end
  end

  it "rejects duplicate targets, missing or released owners, and typed owners" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    arena = ML::Video::Arena.new
    begin
      arena.alloc("existing", 8)
      expect_raises(ArgumentError, /Duplicate buffer/) do
        arena.alias_f32("existing", "existing")
      end
      expect_raises(ArgumentError, /owner/) do
        arena.alias_f32("missing_alias", "missing_owner")
      end

      arena.alloc_weight("typed_f32", 8, ML::Video::WeightDType::F32)
      expect_raises(ArgumentError, /normal F32/) do
        arena.alias_f32("typed_alias", "typed_f32")
      end

      released = arena.alloc("released_owner", 8)
      released.release
      expect_raises(ArgumentError, /valid/) do
        arena.alias_f32("released_alias", "released_owner")
      end
    ensure
      arena.release
    end
  end

  it "rejects stage-backed owners without changing stage-pool ownership" do
    unless ML::Metal::Device.init!
      pending!("requires a Darwin Metal device")
    end

    page = LibC.sysconf(LibC::SC_PAGESIZE).to_i64
    pool = ML::Video::MetalStagePool.new(page * 2, page)
    arena = ML::Video::Arena.new(stage_pool: pool, stage_phase: "scratch", stage_lane: "test")
    begin
      arena.alloc("stage_owner", 4)
      expect_raises(ArgumentError, /stage-backed/) do
        arena.alias_f32("stage_alias", "stage_owner")
      end
      arena.allocated?("stage_alias").should be_false
      pool.live_view_count.should eq(1)
    ensure
      arena.release
      pool.live_view_count.should eq(0)
      pool.release unless pool.released?
    end
  end
end
