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

  it "admits both modes only for owned ordinary F32 commands" do
    {"1", "after_attention"}.each do |mode|
      ML::GGUF::QwenPrefillStageSplit.enabled?(true, true, mode).should be_true
      ML::GGUF::QwenPrefillStageSplit.build(false, true, 7839, 195, mode).should be_nil
      ML::GGUF::QwenPrefillStageSplit.build(true, false, 7839, 195, mode).should be_nil
    end
    {nil, "", "0", "2", "true", "AFTER_ATTENTION", "after_attention "}.each do |mode|
      ML::GGUF::QwenPrefillStageSplit.build(true, true, 7839, 195, mode).should be_nil
    end
  end

  it "keeps the same command across preparation and records only two completions" do
    io = IO::Memory.new
    split = ML::GGUF::QwenPrefillStageSplit.build(true, true, 7839, 195, "after_attention", io: io).not_nil!
    commands = [StageSplitFakeCommand.new]
    first = commands.last
    if split.finish(commands.last, ML::GGUF::QwenPrefillStageSplit::Stage::PrepareKV)
      commands << StageSplitFakeCommand.new
    end
    commands.size.should eq(1)
    commands.last.should be(first)
    first.events.should be_empty
    io.to_s.should be_empty
    if split.finish(commands.last, ML::GGUF::QwenPrefillStageSplit::Stage::Attention)
      commands << StageSplitFakeCommand.new
    end
    split.finish(commands.last, ML::GGUF::QwenPrefillStageSplit::Stage::OutputFFN).should be_true
    commands.size.should eq(2)
    commands.each { |command| command.events.should eq([:commit, :wait]) }
    lines = io.to_s.lines
    lines.size.should eq(4)
    lines[0].should contain("sequence=1 command_id=#{first.object_id} stage=prepare_kv_attention ")
    lines[1].should contain("phase=submit_wait_end")
    lines[2].should contain("sequence=2 command_id=#{commands.last.object_id} stage=output_ffn ")
    lines[3].should contain("phase=submit_wait_end")
    io.to_s.should_not contain("stage=prepare_kv ")
    io.to_s.should_not contain("stage=attention ")
  end

  it "captures the mode once when building the diagnostic" do
    previous = ENV["QWEN35_FULL_PREFILL_STAGE_SPLIT"]?
    begin
      {"1", "after_attention"}.each do |mode|
        ENV["QWEN35_FULL_PREFILL_STAGE_SPLIT"] = mode
        io = IO::Memory.new
        split = ML::GGUF::QwenPrefillStageSplit.build(true, true, 7839, 195, io: io).not_nil!
        ENV["QWEN35_FULL_PREFILL_STAGE_SPLIT"] = mode == "1" ? "after_attention" : "1"
        command = StageSplitFakeCommand.new
        split.finish(command, ML::GGUF::QwenPrefillStageSplit::Stage::PrepareKV).should eq(mode == "1")
        command.events.should eq(mode == "1" ? [:commit, :wait] : [] of Symbol)
      end
    ensure
      if previous
        ENV["QWEN35_FULL_PREFILL_STAGE_SPLIT"] = previous
      else
        ENV.delete("QWEN35_FULL_PREFILL_STAGE_SPLIT")
      end
    end
  end

  it "stops two-stage execution before creating a successor after a failed command" do
    {true, false}.each do |fail_first|
      {true, false}.each do |fail_commit|
        io = IO::Memory.new
        split = ML::GGUF::QwenPrefillStageSplit.build(true, true, 7839, 195, "after_attention", io: io).not_nil!
        original = Exception.new("private failure payload")
        commands = [StageSplitFakeCommand.new(fail_first ? original : nil, fail_commit)]
        caught = nil.as(Exception?)
        begin
          split.finish(commands.last, ML::GGUF::QwenPrefillStageSplit::Stage::PrepareKV).should be_false
          if split.finish(commands.last, ML::GGUF::QwenPrefillStageSplit::Stage::Attention)
            commands << StageSplitFakeCommand.new(fail_first ? nil : original, fail_commit)
          end
          split.finish(commands.last, ML::GGUF::QwenPrefillStageSplit::Stage::OutputFFN)
        rescue ex
          caught = ex
        end
        caught.should be(original)
        commands.size.should eq(fail_first ? 1 : 2)
        commands.last.events.should eq(fail_commit ? [:commit] : [:commit, :wait])
        io.to_s.lines.last.should contain("phase=submit_wait_failed")
        io.to_s.should_not contain("private failure payload")
      end
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
    helper.should contain("QwenPrefillStageSplit.build(")
    helper.should contain("append_command_buffer.nil?, adaptive_prefill_encoder.nil? && !kv_cache_f16,")
    helper.should contain("kvwrite_enc.end_encoding\n\n            if split = stage_split")
    prepare = helper.index("split.finish(cmd, QwenPrefillStageSplit::Stage::PrepareKV)").not_nil!
    attention = helper.index("split.finish(cmd, QwenPrefillStageSplit::Stage::Attention)").not_nil!
    output = helper.index("split.finish(cmd, QwenPrefillStageSplit::Stage::OutputFFN)").not_nil!
    prepare.should be < attention
    attention.should be < output
    helper[prepare...attention].scan("cmd = ML::Metal::CommandBuffer.new").size.should eq(1)
    helper.should contain("if split.finish(cmd, QwenPrefillStageSplit::Stage::PrepareKV)\n                cmd = ML::Metal::CommandBuffer.new\n              end")
    helper[attention...output].scan("cmd = ML::Metal::CommandBuffer.new").size.should eq(1)
    helper[output..].should_not contain("CommandBuffer.new")
    helper.should contain("attn_enc.end_encoding\n          end\n\n          if split = stage_split")
    helper.index("return [] of Float32 if appended").not_nil!.should be < output
    helper.should_not contain("sleep")
    helper.should_not contain("retry")
  end
end
