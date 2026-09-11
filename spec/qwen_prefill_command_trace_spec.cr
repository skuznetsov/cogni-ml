require "spec"
require "../src/ml/gguf/qwen_prefill_command_trace"

private class TraceFlushSpy < IO::Memory
  getter flush_count = 0

  def flush
    @flush_count += 1
    super
  end
end

describe ML::GGUF::QwenPrefillCommandTrace do
  it "enables only for an explicit one" do
    {nil, "0", "true", "", "2"}.each do |setting|
      ML::GGUF::QwenPrefillCommandTrace.enabled?(setting).should be_false
    end
    ML::GGUF::QwenPrefillCommandTrace.enabled?("1").should be_true
  end

  it "flushes a pre-submit record before work and preserves the result" do
    io = TraceFlushSpy.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    value = trace.observe(17_u64, 7839, 195, 0, 7, 1) do
      io.to_s.should contain("phase=submit_wait_begin")
      io.to_s.should_not contain("phase=submit_wait_end")
      io.flush_count.should eq(1)
      42.0
    end
    value.should eq(42.0)
    io.to_s.should contain("command_id=17")
    io.to_s.should contain("layer_cursor_before=0 layer_cursor_at_flush=7 groups=1")
    io.to_s.should contain("start_pos=7839 rows=195")
    io.to_s.should contain("phase=submit_wait_end")
    io.flush_count.should eq(2)
  end

  it "records failed wait and rethrows the original exception without its payload" do
    io = IO::Memory.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    original = Exception.new("private payload")
    caught = nil.as(Exception?)
    begin
      trace.observe(19_u64, 0, 2048, 7, 11, 1) { raise original }
    rescue ex
      caught = ex
    end
    caught.should be(original)
    io.to_s.should contain("phase=submit_wait_failed")
    io.to_s.should_not contain("phase=submit_wait_end")
    io.to_s.should_not contain("private payload")
  end

  it "does not prevent command execution or replace its error when logging fails" do
    io = IO::Memory.new
    io.close
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    trace.observe(1_u64, 0, 1, 0, 0, 0) { 3 }.should eq(3)
    original = Exception.new("command failed")
    caught = nil.as(Exception?)
    begin
      trace.observe(2_u64, 0, 1, 0, 0, 0) { raise original }
    rescue ex
      caught = ex
    end
    caught.should be(original)
  end

  it "distinguishes reused command identities within one trace" do
    io = IO::Memory.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    2.times { trace.observe(17_u64, 0, 1, 0, 1, 1) { nil } }
    io.to_s.should contain("sequence=1 command_id=17")
    io.to_s.should contain("sequence=2 command_id=17")
  end

  it "wraps only ordinary submit/wait and does not claim an encoded layer interval" do
    source = File.read(File.join(__DIR__, "../src/ml/gguf/qwen35_cpu.cr"))
    source.scan("trace.observe(").size.should eq(1)
    ordinary = source.index("elsif cmd = append_prefill_cmd\n            begin").not_nil!
    observe = source.index("gpu_elapsed_ms = trace.observe(").not_nil!
    publish = source.index("QwenQBitAdaptiveResidentKV.finish_pending_appends!(pending_adaptive_caches, cmd)", observe).not_nil!
    ordinary.should be < observe
    observe.should be < publish
    source.should contain("append_prefill_cursor_before, il, append_prefill_group_count")
    source.should contain("append_prefill_cursor_before = il")
    # This route is why an exclusive encoded-layer-end label would be false.
    adaptive = source.index("adaptive_input = gpu_hidden").not_nil!
    flush = source.index("flush_prefill_cmd.call", adaptive).not_nil!
    advance = source.index("il += 1", adaptive).not_nil!
    flush.should be < advance
  end
end
