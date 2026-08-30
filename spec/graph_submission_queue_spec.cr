require "./spec_helper"
require "../src/ml/metal/compute_graph"

private GRAPH_SUBMISSION_QUEUE_TEST_KERNEL = <<-METAL
#include <metal_stdlib>
using namespace metal;

kernel void graph_submission_queue_write_i32(device int* out [[buffer(0)]],
                                             constant int& value [[buffer(1)]],
                                             uint id [[thread_position_in_grid]]) {
  if (id == 0) {
    out[0] = value;
  }
}
METAL

private class FakeGraphCommand
  getter name : String

  def initialize(@name : String, @events : Array(String), @fail_wait = false, @fail_commit = false, @fail_discard = false)
  end

  def enqueue : Nil
    @events << "enqueue:#{@name}"
  end

  def commit : Nil
    @events << "commit:#{@name}"
    raise "commit failed: #{@name}" if @fail_commit
  end

  def wait : Nil
    @events << "wait:#{@name}"
    raise "wait failed: #{@name}" if @fail_wait
  end

  def discard : Nil
    @events << "discard:#{@name}"
    raise "discard failed: #{@name}" if @fail_discard
  end
end

describe ML::Metal::GraphSubmissionQueue do
  it "bounds ordered in-flight submissions and drains before reuse" do
    events = [] of String
    names = ["a", "b", "c"]
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      FakeGraphCommand.new(names[sequence.to_i], events)
    end

    lease_a = queue.begin_submission
    events << "encode:a:#{lease_a.slot}"
    queue.submit(lease_a)
    expect_raises(ArgumentError, /before graph submission/) do
      lease_a.retain("late")
    end
    lease_b = queue.begin_submission
    events << "encode:b:#{lease_b.slot}"
    queue.submit(lease_b)
    lease_c = queue.begin_submission
    events << "encode:c:#{lease_c.slot}"
    queue.submit(lease_c)

    events.should eq([
      "encode:a:0",
      "enqueue:a", "commit:a",
      "encode:b:1",
      "enqueue:b", "commit:b",
      "wait:a",
      "encode:c:0",
      "enqueue:c", "commit:c",
    ])
    queue.pending_count.should eq(2)
    queue.max_pending_seen.should eq(2)

    queue.drain
    events.last(2).should eq(["wait:b", "wait:c"])
    queue.pending_count.should eq(0)
    queue.submitted_count.should eq(3_i64)
    queue.completed_count.should eq(3_i64)
  end

  it "drains every submitted command after a failure and poisons the queue" do
    events = [] of String
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      name = sequence == 0 ? "a" : "b"
      FakeGraphCommand.new(name, events, fail_wait: sequence == 0)
    end
    lease_a = queue.begin_submission
    queue.submit(lease_a)
    lease_b = queue.begin_submission
    queue.submit(lease_b)

    expect_raises(Exception, /wait failed: a/) { queue.drain }
    events.last(2).should eq(["wait:a", "wait:b"])
    queue.failed?.should be_true
    queue.pending_count.should eq(0)
    expect_raises(ArgumentError, /failed/) do
      queue.begin_submission
    end
  end

  it "drains prior submissions when a later submit fails" do
    events = [] of String
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      name = sequence == 0 ? "a" : "b"
      FakeGraphCommand.new(name, events, fail_commit: sequence == 1)
    end
    lease_a = queue.begin_submission
    queue.submit(lease_a)
    lease_b = queue.begin_submission

    expect_raises(Exception, /commit failed: b/) do
      queue.submit(lease_b)
    end
    events.should contain("wait:a")
    events.should contain("wait:b")
    queue.failed?.should be_true
    queue.pending_count.should eq(0)
  end

  it "drains prior submissions when command construction fails" do
    events = [] of String
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      raise "factory failed" if sequence == 1
      FakeGraphCommand.new("a", events)
    end
    lease = queue.begin_submission
    released = [false]
    lease.retain("held") { |resource| released[0] = resource == "held" }
    queue.submit(lease)

    expect_raises(Exception, /factory failed/) do
      queue.begin_submission
    end

    events.should eq(["enqueue:a", "commit:a", "wait:a"])
    lease.state.completed?.should be_true
    released[0].should be_true
    queue.failed?.should be_true
    queue.pending_count.should eq(0)
  end

  it "drains prior submissions when cancelling an open command fails" do
    events = [] of String
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      FakeGraphCommand.new(sequence.to_s, events, fail_discard: sequence == 1)
    end
    submitted = queue.begin_submission
    released = [false, false]
    submitted.retain("submitted") { |resource| released[0] = resource == "submitted" }
    queue.submit(submitted)
    open = queue.begin_submission
    open.retain("open") { |resource| released[1] = resource == "open" }

    expect_raises(Exception, /discard failed: 1/) do
      queue.cancel(open)
    end

    events.should eq(["enqueue:0", "commit:0", "discard:1", "wait:0"])
    submitted.state.completed?.should be_true
    open.state.cancelled?.should be_true
    released.should eq([true, true])
    queue.failed?.should be_true
    queue.pending_count.should eq(0)
  end

  it "owns leases, retains resources, and can cancel abandoned encoding" do
    events = [] of String
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      FakeGraphCommand.new(sequence.to_s, events)
    end
    foreign = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(1) do |slot, sequence|
      FakeGraphCommand.new("foreign", events)
    end

    lease = queue.begin_submission
    released = [false]
    lease.retain("held") { |resource| released[0] = resource == "held" }
    lease.retained_count.should eq(1)
    expect_raises(ArgumentError, /another queue/) do
      foreign.submit(lease)
    end
    expect_raises(ArgumentError, /open lease/) { queue.drain }
    slot = lease.slot
    queue.cancel(lease)
    lease.state.cancelled?.should be_true
    lease.retained_count.should eq(0)
    released[0].should be_true
    events.last.should eq("discard:0")

    reused = queue.begin_submission
    reused.slot.should eq(slot)
    queue.cancel(reused)
  end

  it "aborts an open lease only after draining earlier submissions" do
    events = [] of String
    queue = ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(2) do |slot, sequence|
      FakeGraphCommand.new(sequence.to_s, events)
    end
    submitted = queue.begin_submission
    queue.submit(submitted)
    open = queue.begin_submission
    open.retain("held")

    queue.abort

    events.should eq(["enqueue:0", "commit:0", "discard:1", "wait:0"])
    submitted.state.completed?.should be_true
    open.state.cancelled?.should be_true
    open.retained_count.should eq(0)
    queue.failed?.should be_true
    expect_raises(ArgumentError, /failed/) { queue.begin_submission }
  end

  it "rejects an unbounded or oversized submission window" do
    expect_raises(ArgumentError, /between 1 and 2/) do
      ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(0) { |slot, sequence| FakeGraphCommand.new("a", [] of String) }
    end
    expect_raises(ArgumentError, /between 1 and 2/) do
      ML::Metal::GraphSubmissionQueue(FakeGraphCommand, String).new(3) { |slot, sequence| FakeGraphCommand.new("a", [] of String) }
    end
  end

  it "preserves command-buffer order on the native Metal queue" do
    pending!("Metal not available") unless ML::Metal::Device.init!

    pipeline = ML::Metal::ComputePipeline.new(
      "graph_submission_queue_write_i32",
      GRAPH_SUBMISSION_QUEUE_TEST_KERNEL,
    )
    out_buf = ML::MetalBuffer.new(sizeof(Int32).to_i64)
    command_queue = ML::Metal::CommandQueue.new
    queue = ML::Metal::GraphSubmissionQueue(ML::Metal::CommandBuffer, ML::MetalBuffer).new(2) do |slot, sequence|
      ML::Metal::CommandBuffer.new(queue: command_queue)
    end

    {11_i32, 29_i32}.each do |value|
      lease = queue.begin_submission
      cmd = lease.command
      enc = ML::Metal::ComputeEncoder.new(cmd)
      enc.set_pipeline(pipeline)
      enc.set_buffer(out_buf, 0, ML::Metal::BufferAccess::Write)
      enc.set_value(value, 1)
      enc.dispatch_1d(1, 1)
      enc.end_encoding
      queue.submit(lease)
    end
    queue.drain

    out_buf.contents.as(Pointer(Int32)).value.should eq(29)
    queue.max_pending_seen.should eq(2)
  end
end
