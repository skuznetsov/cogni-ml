require "spec"

private def sg4_kernel(name : String) : String
  source = File.read(Path[__DIR__] / "../src/ml/gguf/kernels/fullattn_qwen35.metal")
  start = source.index("kernel void #{name}(").not_nil!
  finish = source.index("\nkernel void ", start + 1) || source.size
  source[start...finish].gsub(/\/\/[^\n]*/, "")
end

describe "Qwen35 SG4 partial-row synchronization" do
  it "keeps experimental gate staging lane-local and opt-in" do
    kernel = sg4_kernel("qwen35_attn_decode_rows_sg4_pregate")
    kernel.should contain("#if defined(QWEN35_SG4_REGISTER_GATE) && QWEN35_SG4_REGISTER_GATE == 1")
    kernel.should contain("float gate_local[8];")
    kernel.should contain("gate_local[d / 32] = gate[(t * n_head + h) * head_dim + d];")
    kernel.should contain("const float g = gate_local[dl];")
    {4, 32, 64, 128, 252, 256}.each do |dim|
      32.times do |lane|
        loaded = (lane...dim).step(32).map { |d| {d // 32, d} }.to_a
        consumed = (0...8).map { |dl| {dl, lane + dl * 32} }.select { |_, d| d < dim }
        loaded.should eq(consumed)
      end
    end
  end

  it "keeps pipeline inspection on a terminating compile-only branch" do
    source = File.read(Path[__DIR__] / "../bin/qwen35_sg4_tail_probe.cr")
    start = source.index(%(  if ARGV == ["--pipeline-info"])).not_nil!
    finish = source.index("\n  if ARGV.any?", start).not_nil!
    branch = source[start...finish]
    branch.should contain(%({"qwen35_attn_decode_rows_sg4", "qwen35_attn_decode_rows_sg4_pregate"}))
    branch.should contain("ComputePipeline.new(name, SOURCE)")
    branch.should contain("pipe.static_threadgroup_memory_length")
    branch.should contain("pipe.thread_execution_width")
    branch.should contain("pipe.max_total_threads_per_threadgroup")
    branch.should contain("lease.close")
    branch.should end_with("    exit\n  end\n")
    {"ShapeFixture", "Buffer.new", "CommandBuffer", "Encoder", "dispatch!", "synchronize"}.each do |forbidden|
      branch.should_not contain(forbidden)
    end
  end

  {"qwen35_attn_decode_rows_sg4", "qwen35_attn_decode_rows_sg4_pregate"}.each do |name|
    it "keeps #{name} barriers within independently active SIMD groups" do
      kernel = sg4_kernel(name)
      kernel.should contain("const uint t = tgpig.y * 4 + sgitg;")
      kernel.should contain("if (h >= n_head || t >= n_tokens) return;")
      kernel.should contain("q_tg_all[sgitg]")
      kernel.should contain("tile_scores_all[sgitg]")
      kernel.should contain("gate_tg_all[sgitg]") if name.ends_with?("pregate")
      kernel.should_not contain("threadgroup_barrier(")
      load = kernel.index("q_tg[d] = Q[").not_nil!
      barrier = kernel.index("simdgroup_barrier(mem_flags::mem_threadgroup);", load).not_nil!
      consume = kernel.index("float dot =", load).not_nil!
      barrier.should be < consume
    end
  end

  it "exhibits partial SIMD-group participation for each tail remainder" do
    {1, 2, 3, 5, 6, 7, 193, 194, 195}.each do |rows|
      last_group = (rows - 1) // 4
      active = (0...4).count { |sg| last_group * 4 + sg < rows }
      active.should eq(rows % 4)
      active.should be < 4
    end
    {4, 196, 1668, 2048}.each do |rows|
      last_group = (rows - 1) // 4
      (0...4).count { |sg| last_group * 4 + sg < rows }.should eq(4)
    end
  end
end
