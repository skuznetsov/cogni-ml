require "spec"

describe "simdgroup matrix shared-memory handoff" do
  it "has exactly one terminal barrier before every staging-to-output reuse" do
    sources = [
      File.read(Path[__DIR__] / "../src/ml/gguf/kernels/gemm_mm.metal"),
      File.read(Path[__DIR__] / "../src/ml/gguf/kernels/gemm_mm_f16.metal"),
    ]

    reuse_count = 0
    sources.each do |source|
      lines = source.lines
      lines.each_index do |index|
        next unless lines[index].includes?("threadgroup float * temp = (threadgroup float *)shmem")

        reuse_count += 1
        handoff_window = lines[{index - 12, 0}.max...index]
        barrier_count = handoff_window.count do |line|
          line.includes?("threadgroup_barrier(mem_flags::mem_threadgroup)")
        end
        barrier_count.should eq(1), "expected one handoff barrier before #{lines[index].strip.inspect}"
      end
    end

    reuse_count.should eq(10)
  end
end
