require "spec"
require "../src/ml/metal/pipeline_selection_trace"

private class PipelineTraceFlushSpy < IO::Memory
  getter flushes = 0

  def flush
    @flushes += 1
    super
  end
end

describe ML::Metal::PipelineSelectionTrace do
  it "records the actual selected name and native handles before submission" do
    io = PipelineTraceFlushSpy.new
    ML::Metal::PipelineSelectionTrace.record(17_u64, 23_u64,
      "qwen35_attn_decode_rows_sg4_pregate", "qwen35_attn_", io)
    io.to_s.should eq("metal_pipeline phase=selected command_handle=17 encoder_handle=23 pipeline=\"qwen35_attn_decode_rows_sg4_pregate\"\n")
    io.flushes.should eq(1)
    io.to_s.should_not contain("completed")
    io.to_s.should_not contain("gpu_elapsed")
  end

  it "stays silent for unset, empty and nonmatching prefixes" do
    {nil, "", "other_"}.each do |prefix|
      io = PipelineTraceFlushSpy.new
      ML::Metal::PipelineSelectionTrace.record(1_u64, 2_u64, "qwen35_attn_rows", prefix, io)
      io.to_s.should be_empty
      io.flushes.should eq(0)
    end
  end

  it "distinguishes direct, pregate, H16 and flash names without inferring a route" do
    io = IO::Memory.new
    names = {"qwen35_attn_decode_rows_sg4", "qwen35_attn_decode_rows_sg4_pregate",
             "qwen35_attn_decode_rows_sg4_h16", "qwen35_attn_flash_d256"}
    names.each_with_index do |name, i|
      ML::Metal::PipelineSelectionTrace.record(7_u64, i.to_u64, name, "qwen35_attn_", io)
    end
    io.to_s.lines.size.should eq(names.size)
    names.each { |name| io.to_s.should contain("pipeline=#{name.inspect}") }
  end

  it "escapes names and cannot interrupt encoding when its sink fails" do
    io = IO::Memory.new
    ML::Metal::PipelineSelectionTrace.record(1_u64, 2_u64, "kernel\nforged", "kernel", io)
    io.to_s.lines.size.should eq(1)
    io.close
    ML::Metal::PipelineSelectionTrace.record(1_u64, 2_u64, "kernel", "kernel", io).should be_nil
  end

  it "records after the actual binding without changing dispatch or command boundaries" do
    source = File.read(Path[__DIR__] / "../src/ml/metal/dispatch.cr")
    start = source.index("        @pipeline = pipeline\n").not_nil!
    finish = source.index("\n      end", start).not_nil!
    body = source[start...finish]
    bind = body.index("MetalDispatchFFI.encoder_set_pipeline(@encoder, pipeline.handle)").not_nil!
    trace = body.index("PipelineSelectionTrace.record(@cmd_buffer.address, @encoder.address, pipeline.name)").not_nil!
    bind.should be < trace
    body.scan("PipelineSelectionTrace.record").size.should eq(1)
    {"commit", ".wait", "dispatch", "barrier", "pipeline.name ="}.each do |forbidden|
      body.should_not contain(forbidden)
    end
  end
end
