require "spec"
require "../src/ml/audio/drama3/studio/memory_guard"

private def studio_memory_sample(**overrides) : JSON::Any
  values = JSON.parse({valid: true, physical_bytes: 1000, free_reclaimable_bytes: 500,
                       process_footprint_bytes: 100, metal_allocated_bytes: 200}.to_json).as_h
  JSON.parse(overrides.to_h.to_json).as_h.each { |key, value| values[key] = value }
  JSON::Any.new(values)
end

describe ML::Audio::Drama3::Studio::MemoryGuard do
  it "checks independent observed counters without adding overlapping memory" do
    guard = ML::Audio::Drama3::Studio::MemoryGuard.new(250_i64)
    guard.check!(studio_memory_sample, "request_begin")
    guard.remaining_bytes(studio_memory_sample).should eq(50)
    ML::Audio::Drama3::Studio::MemoryGuard.new.remaining_bytes(studio_memory_sample).should be_nil
    expect_raises(ML::Audio::Drama3::Studio::MemoryGuard::Exceeded, /Metal/) do
      guard.check!(studio_memory_sample(metal_allocated_bytes: 250), "generation")
    end
    expect_raises(ML::Audio::Drama3::Studio::MemoryGuard::Exceeded, /footprint/) do
      guard.check!(studio_memory_sample(process_footprint_bytes: 251), "wav_end")
    end
  end

  it "stops at the 15 percent floor and rejects missing or invalid telemetry" do
    guard = ML::Audio::Drama3::Studio::MemoryGuard.new
    guard.check!(studio_memory_sample(free_reclaimable_bytes: 151), "test")
    expect_raises(ML::Audio::Drama3::Studio::MemoryGuard::Exceeded, /15%/) do
      guard.check!(studio_memory_sample(free_reclaimable_bytes: 150), "test")
    end
    [studio_memory_sample(valid: false), studio_memory_sample(physical_bytes: 0),
     studio_memory_sample(process_footprint_bytes: 0), studio_memory_sample(free_reclaimable_bytes: 1001),
     studio_memory_sample(metal_allocated_bytes: -1), JSON.parse("{}")].each do |sample|
      expect_raises(ML::Audio::Drama3::Studio::MemoryGuard::Exceeded, /telemetry/) { guard.check!(sample, "test") }
    end
    expect_raises(ArgumentError) { ML::Audio::Drama3::Studio::MemoryGuard.new(0_i64) }
  end

  it "rejects malformed or overflowing environment budgets before model loading" do
    previous = ENV["DRAMA3_STUDIO_MEMORY_LIMIT_MIB"]?
    begin
      ENV.delete("DRAMA3_STUDIO_MEMORY_LIMIT_MIB")
      ML::Audio::Drama3::Studio::MemoryGuard.from_env.limit_bytes.should be_nil
      ENV["DRAMA3_STUDIO_MEMORY_LIMIT_MIB"] = "40960"
      ML::Audio::Drama3::Studio::MemoryGuard.from_env.limit_bytes.should eq(40_i64 * 1024 * 1024 * 1024)
      ["", "0", "-1", "1.5", "8796093022208", "9223372036854775808"].each do |value|
        ENV["DRAMA3_STUDIO_MEMORY_LIMIT_MIB"] = value
        expect_raises(ArgumentError, /positive integer/) { ML::Audio::Drama3::Studio::MemoryGuard.from_env }
      end
    ensure
      if previous
        ENV["DRAMA3_STUDIO_MEMORY_LIMIT_MIB"] = previous
      else
        ENV.delete("DRAMA3_STUDIO_MEMORY_LIMIT_MIB")
      end
    end
  end
end
