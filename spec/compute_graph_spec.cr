require "./spec_helper"
require "../src/ml/metal/compute_graph"

private COMPUTE_GRAPH_TEST_KERNEL = <<-METAL
#include <metal_stdlib>
using namespace metal;

kernel void compute_graph_test(device float* out [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
  if (id == 0) {
    out[0] = 0.0f;
  }
}

kernel void compute_graph_write_i32(device int* out [[buffer(0)]],
                                    constant int& value [[buffer(1)]],
                                    uint id [[thread_position_in_grid]]) {
  if (id == 0) {
    out[0] = value;
  }
}
METAL

private def compute_graph_test_pipeline : ML::Metal::ComputePipeline
  ML::Metal::ComputePipeline.new("compute_graph_test", COMPUTE_GRAPH_TEST_KERNEL)
end

private def compute_graph_i32_pipeline : ML::Metal::ComputePipeline
  ML::Metal::ComputePipeline.new("compute_graph_write_i32", COMPUTE_GRAPH_TEST_KERNEL)
end

describe ML::Metal::ComputeGraph do
  pending!("Metal not available") unless ML::Metal::Device.init!

  it "groups independent ops into waves and inserts one barrier for dependent fan-in" do
    pipeline = compute_graph_test_pipeline
    a = ML::MetalBuffer.new(4_i64)
    b = ML::MetalBuffer.new(4_i64)
    c = ML::MetalBuffer.new(4_i64)

    graph = ML::Metal::ComputeGraph.new
    graph.add_op(pipeline) do |op|
      op.buffer(a, 0, ML::Metal::BufferAccess::Read)
      op.buffer(b, 1, ML::Metal::BufferAccess::Write)
      op.dispatch_1d(1, 1)
    end
    graph.add_op(pipeline) do |op|
      op.buffer(a, 0, ML::Metal::BufferAccess::Read)
      op.buffer(c, 1, ML::Metal::BufferAccess::Write)
      op.dispatch_1d(1, 1)
    end
    graph.add_op(pipeline) do |op|
      op.buffer(b, 0, ML::Metal::BufferAccess::Read)
      op.buffer(c, 1, ML::Metal::BufferAccess::Read)
      op.buffer(a, 2, ML::Metal::BufferAccess::Write)
      op.dispatch_1d(1, 1)
    end

    graph.compile!
    stats = graph.stats
    stats.n_ops.should eq(3)
    stats.n_waves.should eq(2)
    stats.n_barriers.should eq(1)
    stats.max_wave_width.should eq(2)
  end

  it "preserves partition keys so disjoint writes can share a wave" do
    pipeline = compute_graph_test_pipeline
    buf = ML::MetalBuffer.new(8_i64)

    graph = ML::Metal::ComputeGraph.new
    graph.add_op(pipeline) do |op|
      op.buffer(buf, 0, ML::Metal::BufferAccess::Write, partition: 0)
      op.dispatch_1d(1, 1)
    end
    graph.add_op(pipeline) do |op|
      op.buffer(buf, 0, ML::Metal::BufferAccess::Write, partition: 1)
      op.dispatch_1d(1, 1)
    end

    graph.compile!
    stats = graph.stats
    stats.n_ops.should eq(2)
    stats.n_waves.should eq(1)
    stats.n_barriers.should eq(0)
    stats.max_wave_width.should eq(2)
  end

  it "keeps GraphEncoder scalar and raw-byte bindings API-compatible with ComputeEncoder" do
    pipeline = compute_graph_i32_pipeline
    out_buf = ML::MetalBuffer.new(sizeof(Int32).to_i64)

    graph = ML::Metal::ComputeGraph.new
    enc = ML::Metal::GraphEncoder.new(graph)
    enc.set_pipeline(pipeline)
    enc.set_buffer(out_buf, 0, ML::Metal::BufferAccess::Write)
    enc.set_value(-7_i32, 1)
    enc.dispatch_1d(1, 1)
    graph.compile!

    cmd = ML::Metal::CommandBuffer.new
    graph.encode(cmd)
    cmd.commit_and_wait
    out_buf.contents.as(Pointer(Int32)).value.should eq(-7)

    graph = ML::Metal::ComputeGraph.new
    enc = ML::Metal::GraphEncoder.new(graph)
    raw = Bytes.new(sizeof(Int32))
    IO::ByteFormat::LittleEndian.encode(42_i32, raw)
    enc.set_pipeline(pipeline)
    enc.set_buffer(out_buf, 0, ML::Metal::BufferAccess::Write)
    enc.set_bytes(raw.to_unsafe.as(Pointer(Void)), raw.size, 1)
    enc.dispatch_1d(1, 1)
    graph.compile!

    cmd = ML::Metal::CommandBuffer.new
    graph.encode(cmd)
    cmd.commit_and_wait
    out_buf.contents.as(Pointer(Int32)).value.should eq(42)
  end
end
