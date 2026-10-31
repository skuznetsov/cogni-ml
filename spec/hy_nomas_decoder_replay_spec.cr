require "spec"
require "../examples/support/hy_nomas_decoder_replay"

private def restart_state(chain_present : Bool = true,
                          terminal_completed : Bool = true,
                          wait_count : Int32 = 1,
                          released : Bool = false,
                          failed : Bool = false,
                          schedule_complete : Bool = true,
                          finalized : Bool = true) : HyNomasDecoderReplay::RestartState
  HyNomasDecoderReplay::RestartState.new(chain_present, terminal_completed,
    wait_count, released, failed, schedule_complete, finalized)
end

describe HyNomasDecoderReplay::RestartGuard do
  it "admits a complete decoder pass after exactly one terminal wait" do
    HyNomasDecoderReplay::RestartGuard.validate!(restart_state, "decoder",
      require_finalized: true).should be_nil
  end

  it "rejects released, failed, incomplete, or chainless owners" do
    [
      {restart_state(released: true), /has been released/},
      {restart_state(failed: true), /encode failure/},
      {restart_state(schedule_complete: false), /complete configured window schedule/},
      {restart_state(chain_present: false, terminal_completed: false, wait_count: 0), /prior command chain/},
    ].each do |state, message|
      expect_raises(ArgumentError, message) do
        HyNomasDecoderReplay::RestartGuard.validate!(state, "decoder", require_finalized: true)
      end
    end
  end

  it "rejects an undrained chain and any terminal wait count other than one" do
    expect_raises(ArgumentError, /terminal chain completion/) do
      HyNomasDecoderReplay::RestartGuard.validate!(restart_state(terminal_completed: false), "decoder")
    end
    [0, 2].each do |wait_count|
      expect_raises(ArgumentError, /exactly one terminal wait/) do
        HyNomasDecoderReplay::RestartGuard.validate!(restart_state(wait_count: wait_count), "decoder")
      end
    end
  end

  it "requires finalized output for a full decoder restart" do
    expect_raises(ArgumentError, /requires finalized output/) do
      HyNomasDecoderReplay::RestartGuard.validate!(restart_state(finalized: false), "decoder",
        require_finalized: true)
    end
  end
end
