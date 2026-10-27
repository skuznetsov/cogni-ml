require "spec"
require "../examples/support/pixelumm_serial_probe_runner"

private def with_pixelumm_runner_argv(args : Array(String), &)
  original = [] of String
  begin
    original = ARGV.dup
    ARGV.clear
    args.each { |value| ARGV << value }
    yield
  ensure
    ARGV.clear
    original.each { |value| ARGV << value }
  end
end

describe PixelUMMSerialProbeRunner do
  it "rejects thirty-six-block mode before accepting inputs unless F32 FFN and output banks are enabled" do
    expect_raises(ArgumentError, /--ffn-f32/) do
      with_pixelumm_runner_argv([] of String) do
        PixelUMMSerialProbeRunner.run(36)
      end
    end
    expect_raises(ArgumentError, /--reuse-output-banks/) do
      with_pixelumm_runner_argv(["--ffn-f32"]) do
        PixelUMMSerialProbeRunner.run(36)
      end
    end
  end

  it "keeps prior serial depths on their existing default precision and output mode" do
    {2, 4, 8, 16}.each do |block_count|
      expect_raises(ArgumentError, /authenticated fixtures\/cache/) do
        with_pixelumm_runner_argv([] of String) do
          PixelUMMSerialProbeRunner.run(block_count)
        end
      end
    end
  end

  it "accepts the required thirty-six-block flags then fails closed before report creation when fixtures are absent" do
    expect_raises(ArgumentError, /authenticated fixtures\/cache/) do
      with_pixelumm_runner_argv(["--ffn-f32", "--reuse-output-banks"]) do
        PixelUMMSerialProbeRunner.run(36)
      end
    end
  end
end
