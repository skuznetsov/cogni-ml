require "spec"
require "../examples/support/pixelumm_recovery_timing"

private class RecoveryClock
  getter now_ns : UInt64
  getter pauses = [] of Float64

  def initialize(@now_ns : UInt64, @early = false)
  end

  def pause(seconds : Float64) : Nil
    @pauses << seconds
    advance = (seconds * 1e9).round.to_u64
    if @early
      advance //= 2
      @early = false
    end
    @now_ns += advance
  end
end

describe PixelUMMRecoveryTiming do
  it "waits again after an early wake without shortening the treatment" do
    clock = RecoveryClock.new(10_u64, early: true)
    PixelUMMRecoveryTiming.wait_until(clock, 10_u64, 1)
    clock.now_ns.should eq(1_000_000_010_u64)
    clock.pauses.should eq([1.0, 0.5])
  end

  it "caps each host wait at thirty seconds" do
    clock = RecoveryClock.new(0_u64)
    PixelUMMRecoveryTiming.wait_until(clock, 0_u64, 120)
    clock.now_ns.should eq(120_000_000_000_u64)
    clock.pauses.should eq([30.0] * 4)
  end

  it "counts restart bookkeeping as part of the recovery interval" do
    clock = RecoveryClock.new(200_000_000_u64)
    PixelUMMRecoveryTiming.wait_until(clock, 0_u64, 1)
    clock.pauses.should eq([0.8])
  end

  it "does not add another pause when the deadline already elapsed" do
    clock = RecoveryClock.new(2_000_000_000_u64)
    PixelUMMRecoveryTiming.wait_until(clock, 0_u64, 1)
    clock.pauses.should be_empty
  end
end
