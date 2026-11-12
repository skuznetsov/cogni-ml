require "spec"
require "../src/ml/metal/compute_graph"

class ML::Metal::ComputePipeline
  def self.regression_fixture
    new("compile-only", Pointer(Void).null)
  end
end

class ML::MetalBuffer
  def initialize(*, regression_bytes : Int64)
    @handle = Pointer(Void).new(1_u64)
    @size = regression_bytes
    @storage_mode = StorageMode::Shared
    @owned = false
    @valid = true
  end
end

class ML::Metal::ComputeGraph
  def regression_binding(index : Int32)
    @ops[index].buffers.first
  end
end

describe ML::Metal::ComputeGraph do
  it "schedules writes to disjoint partitions together" do
    graph = ML::Metal::ComputeGraph.new
    pipeline = ML::Metal::ComputePipeline.regression_fixture
    buffer = ML::MetalBuffer.new(regression_bytes: 8_i64)

    2.times do |partition|
      graph.add_op(pipeline) do |op|
        op.buffer(buffer, 0, :write, offset: 2_i64, length: 4_i64, partition: partition)
        op.dispatch_1d(1, 1)
      end
    end
    binding = graph.regression_binding(1)
    {binding.access, binding.offset, binding.length, binding.partition}.should eq(
      {ML::Metal::BufferAccess::Write, 2_i64, 4_i64, 1},
    )
    graph.compile!

    graph.stats.n_ops.should eq(2)
    graph.stats.n_waves.should eq(1)
    graph.stats.n_barriers.should eq(0)
  end
end
