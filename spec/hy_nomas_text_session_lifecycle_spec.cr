require "spec"
require "../examples/support/hy_nomas_text_session"

private class HyNomasTextLifecycleTestChain
  @terminal = false

  def encoder : Nil
    raise ArgumentError.new("test chain is terminal") if @terminal
  end

  def completed? : Bool
    @terminal
  end

  def finish! : Nil
    @terminal = true
  end
end

describe HyNomasTextProbe::SessionChainLease do
  it "refuses release after partial encode until the chain reaches terminal completion" do
    lease = HyNomasTextProbe::SessionChainLease(HyNomasTextLifecycleTestChain).new
    chain = HyNomasTextLifecycleTestChain.new

    # Model the first conversion being appended, followed by a later encode
    # failure. The lease must already exist before that failure can occur.
    expect_raises(Exception, /injected partial encode failure/) do
      lease.begin_encode!(chain)
      raise "injected partial encode failure"
    end

    expect_raises(ArgumentError, /before release/) { lease.ensure_terminal! }
    chain.finish!
    lease.ensure_terminal!
  end

  it "does not replace an incomplete chain with a later chain" do
    lease = HyNomasTextProbe::SessionChainLease(HyNomasTextLifecycleTestChain).new
    first = HyNomasTextLifecycleTestChain.new
    second = HyNomasTextLifecycleTestChain.new

    lease.begin_encode!(first)
    expect_raises(ArgumentError, /in-flight command chains/) { lease.begin_encode!(second) }
    expect_raises(ArgumentError, /before release/) { lease.ensure_terminal! }

    first.finish!
    lease.ensure_terminal!
  end
end
