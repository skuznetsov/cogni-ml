require "spec"

private def sg4_kernel(name : String) : String
  source = File.read(Path[__DIR__] / "../src/ml/gguf/kernels/fullattn_qwen35.metal")
  start = source.index("kernel void #{name}(").not_nil!
  finish = source.index("\nkernel void ", start + 1) || source.size
  source[start...finish].gsub(/\/\/[^\n]*/, "")
end

describe "Qwen35 SG4 partial-row synchronization" do
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
