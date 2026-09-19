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

  it "attributes the full-layer call without pretending to time a command wait" do
    io = TraceFlushSpy.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    expected = [] of Float32
    result = trace.observe_full_layer(7839, 195, 27) do
      io.to_s.should contain("qwen35_prefill_layer phase=call_begin")
      io.flush_count.should eq(1)
      expected
    end
    result.should be(expected)
    io.to_s.should contain("route=full_attn_chunk_routed start_pos=7839 rows=195 layer=27")
    io.to_s.should contain("phase=call_end")
    io.to_s.should_not contain("phase=call_declined")
    io.to_s.should_not contain("command_id=")
    io.to_s.should_not contain("submit_wait")
    io.flush_count.should eq(2)
  end

  it "distinguishes a declined full-layer route from a completed call" do
    io = IO::Memory.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    trace.observe_full_layer(0, 4, 3) { nil }.should be_nil
    io.to_s.should contain("phase=call_declined")
    io.to_s.should_not contain("phase=call_end")
    io.to_s.should_not contain("phase=call_failed")
  end

  it "pairs full-layer failures and preserves the exception without logging its payload" do
    io = IO::Memory.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    original = Exception.new("private payload")
    caught = nil.as(Exception?)
    begin
      trace.observe_full_layer(7839, 195, 31) { raise original }
    rescue ex
      caught = ex
    end
    caught.should be(original)
    lines = io.to_s.lines
    lines.size.should eq(2)
    lines[0].should contain("phase=call_begin")
    lines[1].should contain("phase=call_failed")
    lines.all? { |line| line.includes?("sequence=1 route=full_attn_chunk_routed start_pos=7839 rows=195 layer=31") }.should be_true
    io.to_s.should_not contain("private payload")
    io.to_s.should_not contain("phase=call_end")
  end

  it "preserves full-layer calls with a broken diagnostic sink" do
    io = IO::Memory.new
    io.close
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)
    trace.observe_full_layer(0, 1, 3) { [1.0_f32] }.should eq([1.0_f32])
    trace.observe_full_layer(0, 1, 3) { nil }.should be_nil
    original = Exception.new("failed")
    caught = nil.as(Exception?)
    begin
      trace.observe_full_layer(0, 1, 3) { raise original }
    rescue ex
      caught = ex
    end
    caught.should be(original)
  end

  it "wraps only the ordinary routed full-layer call after the existing flush" do
    source = File.read(File.join(__DIR__, "../src/ml/gguf/qwen35_cpu.cr"))
    source.scan("trace.observe_full_layer(").size.should eq(1)
    observe = source.index("trace.observe_full_layer(start_pos, n_tokens, il)").not_nil!
    boundary = source.rindex("flush_prefill_cmd.call if read_output", observe).not_nil!
    source[boundary...observe].should contain("if trace = prefill_command_trace")
    finish = source.index("if gpu_out = full_result", observe).not_nil!
    body = source[observe...finish]
    # One call in each exclusive branch: enabled and default-off.
    body.scan("full_attn_layer_chunk_project_routed(x, n_tokens, start_pos, state.layers[il], lw, hp, max_seq, read_output: read_output, output_buf: full_output_buf)").size.should eq(2)
    body.should_not contain("append_command_buffer:")
    body.should_not contain("flush_prefill_cmd.call")
    body.should_not contain("il +=")
  end

  it "wraps only ordinary submit/wait and does not claim an encoded layer interval" do
    source = File.read(File.join(__DIR__, "../src/ml/gguf/qwen35_cpu.cr"))
    source.scan("trace.observe(").size.should eq(1)
    ordinary = source.index("elsif cmd = append_prefill_cmd\n", source.index("flush_prefill_cmd = -> {").not_nil!).not_nil!
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
