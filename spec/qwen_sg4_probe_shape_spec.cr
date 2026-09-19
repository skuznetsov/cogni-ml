require "spec"
require "../bin/support/qwen_sg4_probe_shape"

describe QwenSG4ProbeShape do
  it "admits only the six bounded full-shape direct cases" do
    expected = [{7839, 193}, {7839, 194}, {7839, 195}, {7839, 196}, {0, 64}, {0, 195}]
    QwenSG4ProbeShape::CASES.should eq(expected)
    expected.each do |base, rows|
      QwenSG4ProbeShape.parse(["--direct-shape=#{base}:#{rows}"]).should eq({base, rows})
    end
  end

  it "rejects malformed, unbounded and combined selectors before GPU admission" do
    {"", "=", "=7839:192", "=7839:197", "=-1:195", "=0:0", "=0:196",
     "=99999999999999:195", "=07839:195", "=7839:195 ", "=7839:195:1"}.each do |suffix|
      expect_raises(ArgumentError) { QwenSG4ProbeShape.parse(["--direct-shape#{suffix}"]) }
    end
    expect_raises(ArgumentError) { QwenSG4ProbeShape.parse(["--direct-shape=7839:195", "--run"]) }
  end

  it "does not widen or reinterpret existing modes" do
    {[] of String, ["--self-test"], ["--single-command=direct"], ["--run"]}.each do |args|
      QwenSG4ProbeShape.parse(args).should be_nil
    end
  end

  it "keeps full-shape dispatch distinct from the legacy 64-row slice" do
    source = File.read(Path[__DIR__] / "../bin/qwen35_sg4_tail_probe.cr")
    start = source.index("  if shape = direct_shape").not_nil!
    finish = source.index("\n  if ARGV == [\"--pipeline-info\"]", start).not_nil!
    branch = source[start...finish]
    branch.should contain("ShapeFixture.new(base, rows, false)")
    branch.should contain("dispatch!(pipe, fixture, true, 0, rows, trace: true)")
    branch.should contain("validate_output!(fixture.read_output, fixture.expected, fixture.count,")
    branch.should contain("fixture.release")
    branch.should contain("lease.close")
    branch.should contain("commands=1")
    branch.should_not contain("pregate")
    branch.should_not contain("sleep")
    branch.should_not contain("retry")
    source.index("direct_shape = QwenSG4ProbeShape.parse(ARGV)").not_nil!.should be < start
  end
end
