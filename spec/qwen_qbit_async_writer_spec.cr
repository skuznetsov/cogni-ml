require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_async_writer"

private alias QBitAsyncWriter = ML::GGUF::QwenQBitAsyncWriter(Int32, Int32)

describe ML::GGUF::QwenQBitAsyncWriter do
  it "publishes one job off-thread and rejects a second resident job" do
    entered = Thread::Mutex.new
    entered_condition = Thread::ConditionVariable.new
    worker_entered = false
    release_worker = false

    writer = QBitAsyncWriter.new("qbit-spec") do |value|
      entered.lock
      begin
        worker_entered = true
        entered_condition.broadcast
        until release_worker
          entered_condition.wait(entered)
        end
      ensure
        entered.unlock
      end
      value * 2
    end

    begin
      writer.enqueue(21)
      entered.lock
      begin
        until worker_entered
          entered_condition.wait(entered)
        end
      ensure
        entered.unlock
      end

      stats = writer.stats
      stats.enqueued.should eq(1)
      stats.pending.should eq(1)
      expect_raises(ArgumentError, /single-flight/) { writer.enqueue(22) }

      entered.synchronize do
        release_worker = true
        entered_condition.broadcast
      end
      completion = writer.flush.not_nil!
      completion.failed?.should be_false
      completion.result.should eq(42)
      writer.stats.completed.should eq(1)
      writer.stats.pending.should eq(0)
    ensure
      writer.close
    end
  end

  it "reports a background failure exactly once without counting a success" do
    writer = QBitAsyncWriter.new("qbit-failure-spec") do |_value|
      raise IO::Error.new("injected async publication failure")
    end

    begin
      writer.enqueue(7)
      completion = writer.flush.not_nil!
      completion.failed?.should be_true
      completion.error_class.not_nil!.should contain("IO::Error")
      completion.error_message.not_nil!.should contain("injected async publication failure")
      writer.flush.should be_nil

      stats = writer.stats
      stats.enqueued.should eq(1)
      stats.completed.should eq(0)
      stats.failures.should eq(1)
      stats.pending.should eq(0)
    ensure
      writer.close
    end
  end

  it "drains a resident job before close returns" do
    writer = QBitAsyncWriter.new("qbit-close-spec") do |value|
      Thread.sleep(10.milliseconds)
      value + 1
    end

    writer.enqueue(8)
    completion = writer.close.not_nil!
    completion.failed?.should be_false
    completion.result.should eq(9)
    writer.stats.completed.should eq(1)
    writer.stats.pending.should eq(0)
  end

  it "runs fiber-aware blocking work on a scheduler-backed thread" do
    writer = QBitAsyncWriter.new("qbit-scheduler-spec") do |value|
      sleep 1.millisecond
      value + 1
    end

    writer.enqueue(40)
    completion = writer.close.not_nil!
    completion.failed?.should be_false
    completion.result.should eq(41)
  end

  it "reserves the single-flight slot before preparing a large job" do
    entered = Thread::Mutex.new
    entered_condition = Thread::ConditionVariable.new
    preparation_entered = false
    release_preparation = false
    writer = QBitAsyncWriter.new("qbit-reservation-spec") { |value| value * 2 }

    producer = Thread.new do
      writer.enqueue_with do
        entered.lock
        begin
          preparation_entered = true
          entered_condition.broadcast
          until release_preparation
            entered_condition.wait(entered)
          end
        ensure
          entered.unlock
        end
        21
      end
    end

    begin
      entered.lock
      begin
        until preparation_entered
          entered_condition.wait(entered)
        end
      ensure
        entered.unlock
      end

      writer.stats.pending.should eq(1)
      expect_raises(ArgumentError, /single-flight/) do
        writer.enqueue_with { 22 }
      end

      releaser = Thread.new do
        Thread.sleep(10.milliseconds)
        entered.synchronize do
          release_preparation = true
          entered_condition.broadcast
        end
      end
      completion = writer.close.not_nil!
      producer.join
      releaser.join
      completion.failed?.should be_false
      completion.result.should eq(42)
    ensure
      entered.synchronize do
        release_preparation = true
        entered_condition.broadcast
      end
      writer.close
    end
  end

  it "releases a reservation when job preparation fails" do
    writer = QBitAsyncWriter.new("qbit-reservation-failure-spec") { |value| value }

    begin
      expect_raises(ArgumentError, /injected capture failure/) do
        writer.enqueue_with do
          raise ArgumentError.new("injected capture failure")
        end
      end
      writer.stats.pending.should eq(0)

      writer.enqueue(9)
      completion = writer.close.not_nil!
      completion.failed?.should be_false
      completion.result.should eq(9)
    ensure
      writer.close
    end
  end
end
