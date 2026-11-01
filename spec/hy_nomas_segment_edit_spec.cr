require "spec"
require "file/tempfile"
require "../src/ml/video/hy_nomas_segment_edit"
require "../src/ml/video/hy_nomas_resume_bundle"
require "../src/ml/video/hy_nomas_timeline_ops"
require "../examples/support/hy_nomas_cache_qbit_checkpoint"

describe ML::Video::HyNomasSegmentNoiseEdit do
  it "invalidates B and C while preserving the post-A contract" do
    edit = ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, 5)
    edit.source_for(0).should eq(0)
    edit.source_for(1).should eq(5)
    edit.source_for(2).should eq(2)
    edit.bind_contract("original-after-A", 1).should eq("original-after-A")
    edit.bind_contract("original-after-B", 2).should_not eq("original-after-B")
    document = JSON.parse(edit.bind_contract("original-after-B", 2))
    document["base_contract_sha256"].as_s.should eq(Digest::SHA256.hexdigest("original-after-B"))
    document["noise_edit"]["target_segment"].as_i.should eq(1)
    document["noise_edit"]["source_segment"].as_i.should eq(5)
  end

  it "does not accept a generated or active segment as a noise donor" do
    [-1, 0, 1, 2, 6].each do |source|
      expect_raises(ArgumentError) { ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, source) }
    end
    [0, -1, 3].each do |target|
      expect_raises(ArgumentError) { ML::Video::HyNomasSegmentNoiseEdit.new(3, target, 5) }
    end
    expect_raises(ArgumentError) { ML::Video::HyNomasSegmentNoiseEdit.new(6, 1, 5) }
    expect_raises(ArgumentError) { ML::Video::HyNomasSegmentNoiseEdit.new(1, 0, 5) }
  end

  it "binds each applied edit and rejects a stale or differently edited checkpoint" do
    path = File.tempname("hy-nomas-edit-spec", ".hnrs")
    begin
      edit = ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, 5)
      other = ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, 4)
      inventory = {"timeline" => 4_i64}
      contract = edit.bind_contract("original-after-B", 2)
      sha = ML::Video::HyNomasResumeBundle.write(path, contract, inventory) { |_, io| io.write(Bytes[1, 2, 3, 4]) }
      reader = ML::Video::HyNomasResumeBundle.open(path, sha, contract, inventory)
      reader.close
      ["original-after-B", other.bind_contract("original-after-B", 2)].each do |stale|
        expect_raises(ArgumentError, /contract/) do
          ML::Video::HyNomasResumeBundle.open(path, sha, stale, inventory)
        end
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects invalid boundaries and selectors rather than hiding a missed edit" do
    edit = ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, 5)
    [-1, 3].each { |index| expect_raises(ArgumentError) { edit.source_for(index) } }
    [0, 3].each { |boundary| expect_raises(ArgumentError) { edit.bind_contract("base", boundary) } }
  end

  it "rejects stale QBit branch identity before parsing compact records" do
    path = File.tempname("hy-nomas-edit-qbit-spec", ".hnrs")
    begin
      edit = ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, 5)
      other = ML::Video::HyNomasSegmentNoiseEdit.new(3, 1, 4)
      base = "original-after-B"
      contract = edit.bind_contract(base, 2)
      inventory = {"timeline" => 4_i64}
      # Deliberately no codec manifest: stale identity must fail before the
      # compact payload is parsed. Valid codec parsing has separate fixtures.
      sha = ML::Video::HyNomasResumeBundle.write(path,
        HyNomasCacheQBitCheckpoint.contract_for(contract), inventory) do |_, io|
        io.write(Bytes[1, 2, 3, 4])
      end
      reader = ML::Video::HyNomasResumeBundle.open_bounded(path, sha,
        HyNomasCacheQBitCheckpoint.contract_for(contract), inventory)
      reader.close
      [base, other.bind_contract(base, 2)].each do |stale|
        expect_raises(ArgumentError, /contract/) do
          HyNomasCacheQBitCheckpoint::Preflight.open(path, sha, stale, inventory)
        end
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end

describe ML::Video::HyNomasTimelineOps::Geometry do
  it "admits only the active target or an unused future source before encoding" do
    geometry = ML::Video::HyNomasTimelineOps::Geometry.new
    geometry.validate_noise_source!(1, 1, 3)
    geometry.validate_noise_source!(1, 5, 3)
    [-1, 0, 2, 6].each do |source|
      expect_raises(ArgumentError) { geometry.validate_noise_source!(1, source, 3) }
    end
    expect_raises(ArgumentError) { geometry.validate_noise_source!(3, 5, 3) }
  end
end
