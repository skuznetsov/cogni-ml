require "spec"
require "digest/sha256"
require "file/tempfile"
require "../examples/support/pixelumm_cpu_diagnostic_probe"

private def with_pixelumm_cpu_diagnostic_dir(&block : String ->)
  root = File.join(Dir.tempdir, "pixelumm-cpu-diagnostic-#{Process.pid}-#{Random.rand(1_000_000_000)}")
  Dir.mkdir(root)
  begin
    yield root
  ensure
    Dir.children(root).each do |entry|
      File.delete?(File.join(root, entry))
    end
    Dir.delete?(root)
  end
end

private def pixelumm_cpu_diagnostic_captures : Array(PixelUMMCPUDiagnosticProbe::Capture)
  [PixelUMMCPUDiagnosticProbe::Capture.new("block0.output", 4_i64),
   PixelUMMCPUDiagnosticProbe::Capture.new("head.output", 2_i64)]
end

private def pixelumm_cpu_diagnostic_words : Hash(String, Bytes)
  {"block0.output" => Bytes[0x80, 0x3f, 0x00, 0x40], "head.output" => Bytes[0x00, 0x40]}
end

describe PixelUMMCPUDiagnosticProbe do
  it "builds the six-arm eager/deferred crossover and its reverse" do
    normal = PixelUMMCPUDiagnosticProbe.schedule
    reverse = PixelUMMCPUDiagnosticProbe.schedule(reverse: true)

    normal.size.should eq(6)
    reverse.size.should eq(6)
    normal.map(&.kind).should eq(%w[warm condition_0 probe_0 condition_1 probe_1 tail])
    normal.map(&.role).should eq(%w[warm condition probe condition probe tail])
    normal.map(&.policy).should eq(%w[standard eager standard deferred standard standard])
    normal[1].snapshot_slot.should eq(0)
    normal[2].preceding_policy.should eq("eager")
    normal[3].snapshot_slot.should eq(1)
    normal[4].preceding_policy.should eq("deferred")
    reverse.map(&.policy).should eq(%w[standard deferred standard eager standard standard])
    reverse[1].snapshot_slot.should eq(0)
    reverse[2].preceding_policy.should eq("deferred")
    reverse[3].snapshot_slot.should eq(1)
    reverse[4].preceding_policy.should eq("eager")
  end

  it "matches eager and deferred comparisons after the live actual source is overwritten" do
    with_pixelumm_cpu_diagnostic_dir do |root|
      paths = [File.join(root, "slot0.bin"), File.join(root, "slot1.bin")]
      captures = pixelumm_cpu_diagnostic_captures
      expected = pixelumm_cpu_diagnostic_words
      actual = pixelumm_cpu_diagnostic_words
      expected_calls = 0
      expected_reader = ->(stage : String, dst : Bytes) do
        expected_calls += 1
        expected[stage].copy_to(dst)
      end
      actual_calls = 0
      actual_reader = ->(stage : String, dst : Bytes) do
        actual_calls += 1
        actual[stage].copy_to(dst)
      end
      store = PixelUMMCPUDiagnosticProbe::Store.new(paths, captures,
        PixelUMMTwoBlockProbe::ComparisonScratch.new(4))
      begin
        eager = store.snapshot(0, 1, true, expected_reader, actual_reader)
        store.snapshot(1, 3, false, expected_reader, actual_reader).should be_empty
        actual_calls.should eq(4)
        expected_calls.should eq(2)
        actual.each_value(&.fill(0_u8))
        store.finish_requests!(6)
        deferred = store.compare_deferred(1, expected_reader)
        expected_calls.should eq(4)
        deferred.should eq(eager)
        store.slot_completed?(0).should be_true
        store.slot_completed?(1).should be_true
        store.completed?.should be_true
        store.total_bytes.should eq(12_i64)
        store.per_slot_bytes.should eq(6_i64)
        store.release
        paths.each { |path| File.exists?(path).should be_true }
      ensure
        store.release
      end
    end
  end

  it "processes the complete 43-stage runtime inventory with one actual read per stage and slot" do
    with_pixelumm_cpu_diagnostic_dir do |root|
      captures = Array.new(43) do |index|
        PixelUMMCPUDiagnosticProbe::Capture.new("stage#{index}", 2_i64)
      end
      expected = {} of String => Bytes
      actual = {} of String => Bytes
      captures.each_with_index do |capture, index|
        word = Bytes[index.to_u8, 0x3f]
        expected[capture.stage] = word.dup
        actual[capture.stage] = word.dup
      end
      expected_calls = 0
      expected_reader = ->(stage : String, dst : Bytes) do
        expected_calls += 1
        expected[stage].copy_to(dst)
      end
      actual_calls = 0
      actual_reader = ->(stage : String, dst : Bytes) do
        actual_calls += 1
        actual[stage].copy_to(dst)
      end
      store = PixelUMMCPUDiagnosticProbe::Store.new(
        [File.join(root, "slot0.bin"), File.join(root, "slot1.bin")], captures,
        PixelUMMTwoBlockProbe::ComparisonScratch.new(2))
      begin
        eager = store.snapshot(0, 1, true, expected_reader, actual_reader)
        store.snapshot(1, 3, false, expected_reader, actual_reader).should be_empty
        eager.size.should eq(43)
        expected_calls.should eq(43)
        actual_calls.should eq(86)
        store.total_bytes.should eq(172_i64)
        actual.each_value(&.fill(0_u8))
        store.finish_requests!(6)
        deferred = store.compare_deferred(1, expected_reader)
        deferred.should eq(eager)
        expected_calls.should eq(86)
      ensure
        store.release
      end
    end
  end

  it "qualifies the comparator with a non-finite positive control" do
    with_pixelumm_cpu_diagnostic_dir do |root|
      expected = {"nan" => Bytes[0x80, 0x3f]}
      actual = {"nan" => Bytes[0xc0, 0x7f]}
      expected_reader = ->(stage : String, dst : Bytes) { expected[stage].copy_to(dst) }
      actual_reader = ->(stage : String, dst : Bytes) { actual[stage].copy_to(dst) }
      store = PixelUMMCPUDiagnosticProbe::Store.new(
        [File.join(root, "slot0.bin"), File.join(root, "slot1.bin")],
        [PixelUMMCPUDiagnosticProbe::Capture.new("nan", 2_i64)],
        PixelUMMTwoBlockProbe::ComparisonScratch.new(2))
      begin
        result = store.snapshot(0, 1, true, expected_reader, actual_reader).first
        result[:finite].should be_false
        result[:failures].should eq(1_i64)
        result[:actual_sha256].should eq(Digest::SHA256.hexdigest(actual["nan"]))
      ensure
        store.release
      end
    end
  end

  it "rejects empty, duplicate, odd, oversized, and under-capacity inventories before creating files" do
    with_pixelumm_cpu_diagnostic_dir do |root|
      paths = [File.join(root, "slot0.bin"), File.join(root, "slot1.bin")]
      scratch = PixelUMMTwoBlockProbe::ComparisonScratch.new(4)
      expect_raises(ArgumentError) { PixelUMMCPUDiagnosticProbe::Store.new(paths, [] of PixelUMMCPUDiagnosticProbe::Capture, scratch) }
      duplicate = [PixelUMMCPUDiagnosticProbe::Capture.new("same", 2_i64), PixelUMMCPUDiagnosticProbe::Capture.new("same", 2_i64)]
      expect_raises(ArgumentError) { PixelUMMCPUDiagnosticProbe::Store.new(paths, duplicate, scratch) }
      expect_raises(ArgumentError) do
        PixelUMMCPUDiagnosticProbe::Store.new(paths, [PixelUMMCPUDiagnosticProbe::Capture.new("odd", 3_i64)], scratch)
      end
      expect_raises(ArgumentError) do
        PixelUMMCPUDiagnosticProbe::Store.new(paths,
          [PixelUMMCPUDiagnosticProbe::Capture.new("oversized", Int32::MAX.to_i64 + 1)], scratch)
      end
      expect_raises(ArgumentError) do
        PixelUMMCPUDiagnosticProbe::Store.new(paths, [PixelUMMCPUDiagnosticProbe::Capture.new("large", 6_i64)], scratch)
      end
      expect_raises(ArgumentError) do
        PixelUMMCPUDiagnosticProbe::Store.new([paths[0], paths[0]], pixelumm_cpu_diagnostic_captures, scratch)
      end
      paths.each { |path| File.exists?(path).should be_false }
    end
  end

  it "creates exclusively and preserves existing files when construction fails" do
    with_pixelumm_cpu_diagnostic_dir do |root|
      fresh = File.join(root, "fresh.bin")
      existing = File.join(root, "existing.bin")
      File.write(existing, "preserve me")
      expect_raises(Exception) do
        PixelUMMCPUDiagnosticProbe::Store.new([fresh, existing], pixelumm_cpu_diagnostic_captures,
          PixelUMMTwoBlockProbe::ComparisonScratch.new(4))
      end
      File.exists?(fresh).should be_false
      File.read(existing).should eq("preserve me")
    end
  end

  it "rejects wrong slot, premature deferred comparison, duplicate snapshots, and truncated backing files" do
    with_pixelumm_cpu_diagnostic_dir do |root|
      paths = [File.join(root, "slot0.bin"), File.join(root, "slot1.bin")]
      expected = pixelumm_cpu_diagnostic_words
      actual = pixelumm_cpu_diagnostic_words
      expected_reader = ->(stage : String, dst : Bytes) { expected[stage].copy_to(dst) }
      actual_reader = ->(stage : String, dst : Bytes) { actual[stage].copy_to(dst) }
      store = PixelUMMCPUDiagnosticProbe::Store.new(paths, pixelumm_cpu_diagnostic_captures,
        PixelUMMTwoBlockProbe::ComparisonScratch.new(4))
      begin
        expect_raises(ArgumentError) { store.snapshot(1, 1, false, expected_reader, actual_reader) }
        store.snapshot(0, 1, true, expected_reader, actual_reader)
        expect_raises(ArgumentError) { store.snapshot(0, 1, true, expected_reader, actual_reader) }
        store.snapshot(1, 3, false, expected_reader, actual_reader)
        store.slot_request(0).should eq(1)
        store.slot_request(1).should eq(3)
        expect_raises(ArgumentError) { store.compare_deferred(1, expected_reader) }
        expect_raises(ArgumentError) { store.finish_requests!(5) }
        store.finish_requests!(6)
        expect_raises(ArgumentError) { store.compare_deferred(0, expected_reader) }
        File.open(paths[1], "r+") { |io| io.truncate(2_i64) }
        expect_raises(ArgumentError) { store.compare_deferred(1, expected_reader) }
      ensure
        store.release
      end
    end
  end
end
