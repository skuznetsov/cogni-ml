require "spec"
require "../src/ml/video/hy_nomas_timeline_ops"

describe ML::Video::HyNomasTimelineOps::Geometry do
  it "accounts for six global latent and RoPE segments with exact offsets" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new

    geometry.total_latent_frames.should eq(180)
    geometry.segment_count.should eq(6)
    geometry.segment_elements.should eq(138_240)
    geometry.timeline_elements.should eq(829_440)
    geometry.rope_elements_per_segment.should eq(552_960)
    geometry.rope_bank_elements.should eq(3_317_760)
    geometry.segment_start(5).should eq(150)
    geometry.rope_offset(5).should eq(2_764_800)
    geometry.cache_noise_offset(5).should eq(691_200)
    geometry.timeline_bytes.should eq(3_317_760_i64)
    geometry.cache_noise_bank_bytes.should eq(1_658_880_i64)
    geometry.rope_bank_bytes.should eq(13_271_040_i64)
    geometry.cache_sigma.should eq(0.206897_f32)
    geometry.cache_timestep.should eq(206.897_f32)
  end

  it "rejects segment lengths that do not tile the global timeline" do
    expect_raises(ArgumentError, /divide/) do
      ML::Video::HyNomasTimelineOps::Geometry.new(
        total_latent_frames: 181,
        segment_latent_frames: 30
      )
    end
  end

  it "bounds active runs to two through all resident segments" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new

    expect_raises(ArgumentError, /run segment count/) do
      ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 1)
    end
    expect_raises(ArgumentError, /run segment count/) do
      ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 7)
    end
    ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 2)
    ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 6)
  end
end

describe ML::Video::HyNomasTimelineOps::PrefixGeometry do
  it "admits the 180-to-120 latent and 720-to-480 pixel-noise prefixes" do
    latent = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
      channels: 32, total_frames: 180, prefix_frames: 120, height: 9, width: 16)
    pixel_noise = ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
      channels: 3, total_frames: 720, prefix_frames: 480, height: 288, width: 512)

    latent.source_elements.should eq(829_440)
    latent.prefix_elements.should eq(552_960)
    latent.source_bytes.should eq(3_317_760_i64)
    latent.prefix_bytes.should eq(2_211_840_i64)
    pixel_noise.source_elements.should eq(318_504_960)
    pixel_noise.prefix_elements.should eq(212_336_640)
    pixel_noise.source_bytes.should eq(1_274_019_840_i64)
    pixel_noise.prefix_bytes.should eq(849_346_560_i64)
  end

  it "rejects empty dimensions and a prefix longer than the source" do
    expect_raises(ArgumentError, /positive/) do
      ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: 0, total_frames: 8, prefix_frames: 4, height: 2, width: 2)
    end
    expect_raises(ArgumentError, /cannot exceed/) do
      ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: 1, total_frames: 4, prefix_frames: 5, height: 2, width: 2)
    end
  end

  it "rejects source or output element counts over the Metal dispatch bound" do
    expect_raises(ArgumentError, /dispatch limit/) do
      ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: 2, total_frames: 1, prefix_frames: 1,
        height: Int32::MAX, width: 2)
    end
    expect_raises(ArgumentError, /dispatch limit/) do
      ML::Video::HyNomasTimelineOps::PrefixGeometry.new(
        channels: 1, total_frames: Int32::MAX, prefix_frames: 1,
        height: 1, width: 2)
    end
  end
end

describe ML::Video::HyNomasTimelineOps::Schedule do
  it "requires denoise, scatter, noisy cache write, and cache forward before advancing" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new
    schedule = ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 2)

    schedule.expect_prepare!(0)
    schedule.begin_prepare!(0)
    schedule.record_prepared!(0)
    schedule.record_denoise_complete!(0)
    schedule.record_scattered!(0)
    expect_raises(ArgumentError, /cache-noise/) { schedule.advance_segment!(0) }
    schedule.record_cache_noise_written!(0)
    expect_raises(ArgumentError, /cache forward/) { schedule.advance_segment!(0) }
    schedule.record_cache_forward_enqueued!(0)
    schedule.advance_segment!(0)
    schedule.next_segment.should eq(1)
    schedule.phase.should eq(ML::Video::HyNomasTimelineOps::SchedulePhase::Ready)
  end

  it "finishes a bounded run at its final scatter without mutating its clean output" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new
    schedule = ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 2)

    2.times do |index|
      i = index.to_i32
      schedule.expect_prepare!(i)
      schedule.begin_prepare!(i)
      schedule.record_prepared!(i)
      schedule.record_denoise_complete!(i)
      schedule.record_scattered!(i)
      if i == 0
        schedule.record_cache_noise_written!(i)
        schedule.record_cache_forward_enqueued!(i)
      else
        schedule.cache_write_required?(i).should be_false
      end
      schedule.advance_segment!(i)
    end

    schedule.phase.should eq(ML::Video::HyNomasTimelineOps::SchedulePhase::Finished)
    schedule.next_segment.should eq(2)
    expect_raises(ArgumentError, /finished/) { schedule.expect_prepare!(0) }
  end

  it "cache-writes the first five segments of a full six-segment export" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new
    schedule = ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 6)
    cache_writes = 0

    6.times do |index|
      i = index.to_i32
      schedule.expect_prepare!(i)
      schedule.begin_prepare!(i)
      schedule.record_prepared!(i)
      schedule.record_denoise_complete!(i)
      schedule.expect_scatter!(i)
      schedule.record_scattered!(i)
      if schedule.cache_write_required?(i)
        schedule.record_cache_noise_written!(i)
        schedule.record_cache_forward_enqueued!(i)
        cache_writes += 1
      end
      schedule.advance_segment!(i)
    end

    cache_writes.should eq(5)
    schedule.phase.should eq(ML::Video::HyNomasTimelineOps::SchedulePhase::Finished)
  end

  it "rejects skipped, repeated, and out-of-order transitions" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new
    schedule = ML::Video::HyNomasTimelineOps::Schedule.new(geometry, run_segments: 2)

    expect_raises(ArgumentError, /next active segment/) { schedule.expect_prepare!(1) }
    expect_raises(ArgumentError, /phase/) { schedule.record_denoise_complete!(0) }
    expect_raises(ArgumentError, /phase/) { schedule.record_prepared!(0) }
  end
end
