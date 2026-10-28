require "spec"
require "../examples/support/pixelumm_ffn_f32_kernel_probe"

private def f32_probe_words(values : Array(UInt16)) : Bytes
  bytes = Bytes.new(values.size * 2)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 2, 2])
  end
  bytes
end

describe PixelUMMFFNF32KernelProbe do
  it "budgets only the selected projection buffers and all eleven metric slots" do
    gate = PixelUMMFFNF32KernelProbe.resources("gate", 21120)
    gate.keys.sort.should eq(["expected", "input", "metrics", "output", "weight"])
    gate.should eq({
      "input"    => 173_015_040_i64,
      "weight"   => 100_663_296_i64,
      "expected" => 519_045_120_i64,
      "output"   => 519_045_120_i64,
      "metrics"  => 176_i64,
    })
    PixelUMMFFNF32KernelProbe.named_bytes("gate", 21120).should eq(1_311_768_752_i64)
    PixelUMMFFNF32KernelProbe.named_bytes("down", 21120).should eq(965_738_672_i64)

    exact = PixelUMMFFNF32KernelProbe.named_bytes("up", 2048)
    PixelUMMFFNF32KernelProbe.validate_budget!("up", 2048, exact).should eq(exact)
    expect_raises(ArgumentError) do
      PixelUMMFFNF32KernelProbe.validate_budget!("up", 2048, exact - 1)
    end
    expect_raises(ArgumentError) do
      PixelUMMFFNF32KernelProbe.validate_budget!("up", 2048, PixelUMMFFNF32KernelProbe::MAX_NAMED_BYTES + 1)
    end
  end

  it "uses full-tile groups for each selected projection and both exact row counts" do
    {
      {"gate", 2048, 4096, 12288, {384, 32, 1}, 25_165_824_i64},
      {"up", 21120, 4096, 12288, {384, 330, 1}, 259_522_560_i64},
      {"down", 2048, 12288, 4096, {128, 32, 1}, 8_388_608_i64},
      {"down", 21120, 12288, 4096, {128, 330, 1}, 86_507_520_i64},
    }.each do |name, rows, k, n, groups, outputs|
      shape = PixelUMMFFNF32KernelProbe.projection(name)
      shape[:k].should eq(k)
      shape[:n].should eq(n)
      geometry = PixelUMMFFNF32KernelProbe.geometry(rows, k, n)
      geometry[:groups].should eq(groups)
      geometry[:threads].should eq({128, 1, 1})
      geometry[:output_elements].should eq(outputs)
    end

    expect_raises(ArgumentError) do
      PixelUMMFFNF32KernelProbe.geometry(21121, 4096, 12288)
    end
    expect_raises(ArgumentError) do
      PixelUMMFFNF32KernelProbe.geometry(2048, 4096, 12289)
    end
  end

  it "balances eight measured requests and reverses the whole schedule for the paired order cell" do
    forward = PixelUMMFFNF32KernelProbe.schedule(false)
    reverse = PixelUMMFFNF32KernelProbe.schedule(true)
    forward.map(&.[:arm]).should eq([
      "mps.f32", "native.f32",
      "mps.f32", "native.f32", "native.f32", "mps.f32",
      "native.f32", "mps.f32", "mps.f32", "native.f32",
    ])
    reverse[0, 2].map(&.[:arm]).should eq(forward[0, 2].map(&.[:arm]).reverse)
    reverse[2, 8].map(&.[:arm]).should eq(forward[2, 8].map(&.[:arm]).reverse)
    forward.count { |entry| entry[:warmup] }.should eq(2)
    forward.count { |entry| entry[:arm] == "mps.f32" && !entry[:warmup] }.should eq(4)
    forward.count { |entry| entry[:arm] == "native.f32" && !entry[:warmup] }.should eq(4)
    forward.map(&.[:index]).should eq((0...10).map(&.to_i32))
  end

  it "preserves the existing BF16 output tolerance and rejects nonfinite comparator values" do
    PixelUMMFFNIsolatedProbe::ATOL.should eq(2e-5_f64)
    PixelUMMFFNIsolatedProbe::RTOL.should eq(2e-5_f64)

    expected = f32_probe_words([0x3f80_u16, 0x4000_u16, 0x0000_u16])
    within = f32_probe_words([0x3f80_u16, 0x4000_u16, 0x3700_u16])
    PixelUMMFFNIsolatedProbe.compare(expected, within).should eq(
      {failures: 0_u32, different_words: 1_u32, nonfinite: 0_u32, elements: 3_u32})

    outside = f32_probe_words([0x3f81_u16, 0x4000_u16, 0x0000_u16])
    PixelUMMFFNIsolatedProbe.compare(expected, outside).should eq(
      {failures: 1_u32, different_words: 1_u32, nonfinite: 0_u32, elements: 3_u32})

    forced_nan = f32_probe_words([0x7fc0_u16, 0x4000_u16, 0x0000_u16])
    PixelUMMFFNIsolatedProbe.compare(expected, forced_nan).should eq(
      {failures: 1_u32, different_words: 1_u32, nonfinite: 1_u32, elements: 3_u32})
  end

  it "rejects unmeasured outside roots before declaring projection timing isolated" do
    timed = ML::Video::MPSCommandRootTiming.new(0, true, false, 1.0, 2.0)
    outside = ML::Video::MPSCommandRootTiming.new(1, true, false, 3.0, 4.0)
    PixelUMMFFNF32KernelProbe.validate_root_timings!([timed, outside])
    [
      ML::Video::MPSCommandRootTiming.new(1, true, false, 0.0, 0.0),
      ML::Video::MPSCommandRootTiming.new(1, false, false, 3.0, 4.0),
      ML::Video::MPSCommandRootTiming.new(1, true, true, 3.0, 4.0),
      ML::Video::MPSCommandRootTiming.new(1, true, false, 4.0, 3.0),
      ML::Video::MPSCommandRootTiming.new(1, true, false, 3.0, 3.0),
      ML::Video::MPSCommandRootTiming.new(1, true, false, Float64::NAN, 4.0),
      ML::Video::MPSCommandRootTiming.new(1, true, false, 3.0, Float64::INFINITY),
      ML::Video::MPSCommandRootTiming.new(2, true, false, 3.0, 4.0),
    ].each do |invalid|
      expect_raises(ArgumentError) do
        PixelUMMFFNF32KernelProbe.validate_root_timings!([timed, invalid])
      end
    end
    expect_raises(ArgumentError) do
      PixelUMMFFNF32KernelProbe.validate_root_timings!([] of ML::Video::MPSCommandRootTiming)
    end
  end

  it "holds plans and buffers until one successful terminal wait after the first submission attempt" do
    lifecycle = PixelUMMFFNIsolatedProbe::SubmissionLifecycle.new
    lifecycle.release_allowed?(false, 0).should be_true

    lifecycle.mark_attempt! # Poison blit, projection encode, or GPU comparison.
    lifecycle.release_allowed?(true, 1).should be_false
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(false, 1) }
    expect_raises(ArgumentError) { lifecycle.mark_terminal_wait!(true, 2) }
    lifecycle.terminal_wait_succeeded?.should be_false

    lifecycle.mark_terminal_wait!(true, 1)
    lifecycle.release_allowed?(true, 1).should be_true
    expect_raises(ArgumentError) { lifecycle.mark_attempt! }
  end
end
