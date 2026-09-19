require "spec"
require "../src/ml/gguf/qwen_first_command_probe"

describe ML::GGUF::QwenFirstCommandProbe do
  it "keeps the unwind after successful wait, publication and cleanup, behind a build flag" do
    source = File.read(File.join(__DIR__, "../src/ml/gguf/qwen35_cpu.cr"))
    wait = source.index("gpu_elapsed_ms = trace.observe(").not_nil!
    publish = source.index("QwenQBitAdaptiveResidentKV.finish_pending_appends!(pending_adaptive_caches, cmd)", wait).not_nil!
    rethrow = source.index("raise ex", publish).not_nil!
    cleanup = source.index("ensure", rethrow).not_nil!
    detach = source.index("append_prefill_cmd = nil", cleanup).not_nil!
    hook = source.index("QwenFirstCommandProbe.after_flush!(diagnostic_boundary)", detach).not_nil!
    source[detach...hook].should contain("{% if flag?(:qwen_first_command_probe) %}")
    source.scan("QwenFirstCommandProbe.after_flush!(").size.should eq(1)
    wait.should be < publish
    publish.should be < rethrow
    rethrow.should be < cleanup
    cleanup.should be < detach
    detach.should be < hook
  end

  it "admits only the original first shared-command boundary" do
    expect_raises(ML::GGUF::QwenFirstCommandProbe::Completed) do
      ML::GGUF::QwenFirstCommandProbe.after_flush!({true, 0, 2048, 0, 7, 1, 0})
    end
  end

  it "rejects empty, shifted, shortened, regrouped, or adaptive boundaries" do
    [{false, 0, 2048, 0, 7, 1, 0}, {true, 1, 2048, 0, 7, 1, 0},
     {true, 0, 1024, 0, 7, 1, 0}, {true, 0, 2048, 3, 7, 1, 0},
     {true, 0, 2048, 0, 3, 1, 0}, {true, 0, 2048, 0, 7, 2, 0},
     {true, 0, 2048, 0, 7, 1, 1}].each do |boundary|
      expect_raises(ArgumentError) { ML::GGUF::QwenFirstCommandProbe.after_flush!(boundary) }
    end
  end
end
