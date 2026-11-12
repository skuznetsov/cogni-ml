require "spec"
require "../src/ml/gguf/qwen_qbit_async_writer"

private alias Writer = ML::GGUF::QwenQBitAsyncWriter(Int32, Int32)

describe ML::GGUF::QwenQBitAsyncWriter do
  it "retains a durability failure for a retried close" do
    writer = Writer.new("qbit-retry-close-spec") do |_value|
      raise IO::Error.new("injected teardown publication failure")
    end
    writer.enqueue(5)

    first = writer.close
    first.should_not be_nil
    first.not_nil!.failed?.should be_true
    failure = first.not_nil!.error_message.not_nil!
    failure.should contain("injected teardown publication failure")

    retry = writer.close
    retry.should_not be_nil
    retry.not_nil!.failed?.should be_true
    retry.not_nil!.error_message.not_nil!.should eq(failure)
    writer.stats.failures.should eq(1)
    writer.stats.pending.should eq(0)
  end
end
