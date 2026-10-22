require "spec"
require "json"
require "../src/ml/video/wan_unipc"
require "./support/video_allocation_counter"

describe ML::Video::WanUniPC do
  it "rejects unsupported schedules and keeps default endpoints finite" do
    expect_raises(ArgumentError) { ML::Video::WanUniPC.new(0) }
    expect_raises(ArgumentError) { ML::Video::WanUniPC.new(3, shift: Float64::NAN) }
    expect_raises(ArgumentError) { ML::Video::WanUniPC.new(3, shift: 1e-300) }
    expect_raises(ArgumentError) { ML::Video::WanUniPC.new(3, shift: 1e300) }
    expect_raises(ArgumentError, "distinct integer timesteps") { ML::Video::WanUniPC.new(3, 1000) }
    scheduler = ML::Video::WanUniPC.new(257)
    scheduler.sigmas.size.should eq(41)
    scheduler.sigmas.last.should eq(0_f32)
    scheduler.timesteps.first.should eq(999)
    scheduler.scratch_bytes.should eq(5_i64 * 257 * 4)
    scheduler.sigmas.clear
    scheduler.timesteps.clear
    scheduler.sigmas.size.should eq(41)
    scheduler.timesteps.size.should eq(40)
  end

  it "matches every upstream corrector and predictor with resident repeated history" do
    unless path = ENV["WAN_UNIPC_FIXTURE"]?
      pending! "Set WAN_UNIPC_FIXTURE to a pinned MPS fixture"
    end
    fixture = JSON.parse(File.read(path))
    fixture["revision"].as_s.should eq("1ea34ff48f87168174e12956e200b1d908b1c5ff")
    fixture["source_sha256"].as_s.should eq("0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9")
    fixture["device"].as_s.should eq("mps")
    steps = fixture["steps"].as_i
    if expected_steps = ENV["WAN_UNIPC_EXPECT_STEPS"]?
      steps.should eq(expected_steps.to_i), "Fixture does not contain the requested step count"
    end
    count = fixture["count"].as_i
    scheduler = ML::Video::WanUniPC.new(count, steps)
    scheduler.sigmas.should eq(fixture["sigmas"].as_a.map(&.as_f.to_f32))
    scheduler.timesteps.should eq(fixture["timesteps"].as_a.map(&.as_i))
    a = ML::Video::Arena.new
    begin
      scheduler.prepare_scratch!(a)
      initial = a.upload("initial", fixture["initial"].as_a.map(&.as_f.to_f32))
      sample = a.alloc("sample", count)
      model = a.alloc("model", count)
      steps.times do |i|
        a.upload("cond.#{i}", fixture["conditional"][i].as_a.map(&.as_f.to_f32))
        a.upload("uncond.#{i}", fixture["unconditional"][i].as_a.map(&.as_f.to_f32))
        {"guidance", "current", "corrected", "output"}.each { |n| a.alloc("trace.#{i}.#{n}", count) }
      end
      a.alloc("repeated", count)
      rejected = ML::Metal::CommandBuffer.new
      rejected_encoder = ML::Metal::ComputeEncoder.new(rejected)
      expect_raises(ArgumentError, "Unsafe video buffer alias") do
        scheduler.encode_step(rejected_encoder, a, model, sample, model)
      end
      expect_raises(ArgumentError, "Unsafe video buffer alias") do
        scheduler.encode_step(rejected_encoder, a, model, sample, a["unipc.m0"])
      end
      scheduler.index.should eq(0)
      rejected_encoder.end_encoding
      rejected.discard
      before = ML::MetalBuffer.video_spec_allocations
      command = ML::Metal::CommandBuffer.new
      e = ML::Metal::ComputeEncoder.new(command)
      2.times do |repeat|
        scheduler.reset!
        ML::Video::MetalOps.copy(e, initial, sample, count)
        steps.times do |i|
          scheduler.guidance(e, a["cond.#{i}"], a["uncond.#{i}"], model)
          scheduler.encode_step(e, a, model, sample, sample)
          if repeat == 0
            ML::Video::MetalOps.copy(e, model, a["trace.#{i}.guidance"], count)
            ML::Video::MetalOps.copy(e, a["unipc.current"], a["trace.#{i}.current"], count)
            ML::Video::MetalOps.copy(e, a["unipc.last"], a["trace.#{i}.corrected"], count)
            ML::Video::MetalOps.copy(e, sample, a["trace.#{i}.output"], count)
          end
        end
        ML::Video::MetalOps.copy(e, sample, a["repeated"], count) if repeat == 1
        expect_raises(ArgumentError, "Wan UniPC schedule exhausted") { scheduler.encode_step(e, a, model, sample, sample) }
      end
      e.end_encoding
      ML::MetalBuffer.video_spec_allocations.should eq(before)
      command.commit
      command.wait
      if actual_path = ENV["WAN_UNIPC_ACTUAL"]?
        File.write(actual_path, Array.new(steps) do |i|
          {"guidance", "current", "corrected", "output"}.to_h do |name|
            {name, a["trace.#{i}.#{name}"].read(count)}
          end
        end.to_json)
      end
      steps.times do |i|
        {"guidance", "current", "corrected", "output"}.each do |name|
          actual = a["trace.#{i}.#{name}"].read(count)
          expected = fixture["expected"][i][name].as_a
          violations = 0
          worst = 0_f64
          actual.each_with_index do |x, j|
            ref = expected[j].as_f.to_f32
            ratio = (x - ref).abs.to_f64 / (2e-5 + 2e-5 * ref.abs)
            violations += 1 unless x.finite? && ref.finite? && ratio <= 1
            worst = Math.max(worst, ratio)
          end
          violations.should eq(0), "step #{i} #{name}: #{violations} violations, worst=#{worst}"
        end
      end
      a["repeated"].read(count).should eq(a["trace.#{steps - 1}.output"].read(count))
    ensure
      a.release
    end
  end
end
