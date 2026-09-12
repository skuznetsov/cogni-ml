require "spec"
require "../src/ml/gguf/qwen_prefill_stage_split"

private class StageSplitFakeCommand
  getter events = [] of Symbol

  def initialize(@failure : Exception? = nil, @fail_commit = false)
  end

  def commit
    @events << :commit
    raise @failure.not_nil! if @fail_commit && @failure
  end

  def wait
    @events << :wait
    if failure = @failure
      raise failure
    end
  end
end

describe ML::GGUF::QwenPrefillStageSplit do
  it "requires explicit opt-in and an owned ordinary F32 command" do
    {nil, "", "0", "true", "2"}.each do |value|
      ML::GGUF::QwenPrefillStageSplit.enabled?(true, true, value).should be_false
    end
    ML::GGUF::QwenPrefillStageSplit.enabled?(true, true, "1").should be_true
    ML::GGUF::QwenPrefillStageSplit.enabled?(false, true, "1").should be_false
    ML::GGUF::QwenPrefillStageSplit.enabled?(true, false, "1").should be_false
  end

  it "submits and waits each stage with paired non-GPU timing records" do
    io = IO::Memory.new
    split = ML::GGUF::QwenPrefillStageSplit.new(7839, 195, io)
    ML::GGUF::QwenPrefillStageSplit::Stage.each do |stage|
      command = StageSplitFakeCommand.new
      split.finish(command, stage)
      command.events.should eq([:commit, :wait])
    end
    io.to_s.lines.size.should eq(6)
    {"prepare_kv", "attention", "output_ffn"}.each do |stage|
      lines = io.to_s.lines.select { |line| line.includes?("stage=#{stage} ") }
      lines.size.should eq(2)
      lines[0].should contain("phase=submit_wait_begin")
      lines[1].should contain("phase=submit_wait_end")
      lines[0].should contain("start_pos=7839 rows=195")
      lines[1].should contain("host_elapsed_ms=")
    end
  end

  it "stops on a failed stage and rethrows the exact exception without its payload" do
    ML::GGUF::QwenPrefillStageSplit::Stage.each do |failed_stage|
      io = IO::Memory.new
      split = ML::GGUF::QwenPrefillStageSplit.new(7839, 195, io)
      original = Exception.new("private failure payload")
      visited = [] of ML::GGUF::QwenPrefillStageSplit::Stage
      caught = nil.as(Exception?)
      begin
        ML::GGUF::QwenPrefillStageSplit::Stage.each do |stage|
          visited << stage
          split.finish(StageSplitFakeCommand.new(stage == failed_stage ? original : nil), stage)
        end
      rescue ex
        caught = ex
      end
      caught.should be(original)
      visited.last.should eq(failed_stage)
      io.to_s.lines.last.should contain("phase=submit_wait_failed")
      io.to_s.should_not contain("private failure payload")
    end
  end

  it "preserves success and commit/wait errors even when logging is closed" do
    io = IO::Memory.new
    io.close
    split = ML::GGUF::QwenPrefillStageSplit.new(0, 4, io)
    split.finish(StageSplitFakeCommand.new, ML::GGUF::QwenPrefillStageSplit::Stage::PrepareKV)
    {false, true}.each do |fail_commit|
      original = Exception.new("failure")
      cmd = StageSplitFakeCommand.new(original, fail_commit)
      caught = nil.as(Exception?)
      begin
        split.finish(cmd, ML::GGUF::QwenPrefillStageSplit::Stage::Attention)
      rescue ex
        caught = ex
      end
      caught.should be(original)
      cmd.events.should eq(fail_commit ? [:commit] : [:commit, :wait])
    end
  end

  it "cuts only after ended encoders and allocates no final successor" do
    source = File.read(File.join(__DIR__, "../src/ml/gguf/qwen35_metal.cr"))
    first = source.index("def self.full_attn_layer_chunk_project(").not_nil!
    last = source.index("def self.rmsnorm_project_top1_rows_buffer(", first).not_nil!
    helper = source[first...last]
    helper.should contain("QwenPrefillStageSplit.enabled?(append_command_buffer.nil?, adaptive_prefill_encoder.nil? && !kv_cache_f16)")
    helper.should contain("kvwrite_enc.end_encoding\n\n            if split = stage_split")
    prepare = helper.index("split.finish(cmd, QwenPrefillStageSplit::Stage::PrepareKV)").not_nil!
    attention = helper.index("split.finish(cmd, QwenPrefillStageSplit::Stage::Attention)").not_nil!
    output = helper.index("split.finish(cmd, QwenPrefillStageSplit::Stage::OutputFFN)").not_nil!
    prepare.should be < attention
    attention.should be < output
    helper[prepare...attention].scan("cmd = ML::Metal::CommandBuffer.new").size.should eq(1)
    helper[attention...output].scan("cmd = ML::Metal::CommandBuffer.new").size.should eq(1)
    helper[output..].should_not contain("CommandBuffer.new")
    helper.should contain("attn_enc.end_encoding\n          end\n\n          if split = stage_split")
    helper.index("return [] of Float32 if appended").not_nil!.should be < output
    helper.should_not contain("sleep")
    helper.should_not contain("retry")
  end
end
