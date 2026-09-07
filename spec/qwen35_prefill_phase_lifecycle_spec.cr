require "./spec_helper"
require "../src/ml/gguf/qwen35_metal"

{% unless flag?(:cpu_only) %}
  # Test-only access to the private profiling checkpoint; no model or kernels.
  module ML::GGUF::Qwen35Metal
    def self.phase_checkpoint_for_spec(cmd : ML::Metal::CommandBuffer, next_command : Bool)
      prefill_phase_checkpoint(cmd, "lifecycle_spec", Time.instant, next_command: next_command)[0]
    end
  end

  describe "prefill profiling command lifecycle" do
    it "does not allocate terminal successors across repeated groups" do
      pending "Metal unavailable" unless ML::Metal::Device.available?
      80.times do
        cmd = ML::Metal::CommandBuffer.new
        terminal = ML::GGUF::Qwen35Metal.phase_checkpoint_for_spec(cmd, false)
        terminal.same?(cmd).should be_true
        terminal.completed_successfully?.should be_true
      end
    end

    it "creates a live successor only for another phase" do
      pending "Metal unavailable" unless ML::Metal::Device.available?
      cmd = ML::Metal::CommandBuffer.new
      successor = ML::GGUF::Qwen35Metal.phase_checkpoint_for_spec(cmd, true)
      cmd.completed_successfully?.should be_true
      successor.same?(cmd).should be_false
      successor.committed?.should be_false
      terminal = ML::GGUF::Qwen35Metal.phase_checkpoint_for_spec(successor, false)
      terminal.same?(successor).should be_true
      terminal.completed_successfully?.should be_true
    end
  end
{% end %}
