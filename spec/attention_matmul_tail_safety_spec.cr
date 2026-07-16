require "spec"

ATTENTION_MATMUL_SOURCE = File.read(Path[__DIR__] / "../src/ml/gguf/kernels/attention_matmul.metal")
METAL_BACKEND_SOURCE    = File.read(Path[__DIR__] / "../src/ml/gguf/metal_backend.cr")

def kernel_source(source : String, name : String) : String
  start_marker = "kernel void #{name}("
  start = source.index(start_marker).not_nil!
  next_kernel = source.index("\nkernel void ", start + start_marker.size) || source.size
  source[start...next_kernel]
end

describe "attention_matmul tail safety source shape" do
  it "stages and zero-pads partial K and V rows before simdgroup loads" do
    helper_start = ATTENTION_MATMUL_SOURCE.index("inline void attention_stage_tail_8(").not_nil!
    first_kernel = ATTENTION_MATMUL_SOURCE.index("kernel void attention_matmul(").not_nil!
    helper = ATTENTION_MATMUL_SOURCE[helper_start...first_kernel]

    guard = helper.index("if (row < valid_rows)").not_nil!
    device_load = helper.index("src[(base_row + row) * row_stride + col]").not_nil!
    device_load.should be > guard
    helper.includes?("dst[idx] = half(0);").should be_true
    helper.includes?("simdgroup_barrier(mem_flags::mem_threadgroup);").should be_true
    helper.includes?("threadgroup_barrier(mem_flags::mem_threadgroup);").should be_false

    ATTENTION_MATMUL_SOURCE.scan("attention_stage_tail_8(k_src").size.should eq(2)
    ATTENTION_MATMUL_SOURCE.scan("attention_stage_tail_8(v_src").size.should eq(2)

    ["attention_matmul", "attention_matmul_batch"].each do |name|
      kernel = kernel_source(ATTENTION_MATMUL_SOURCE, name)

      q_guard = kernel.index("if (q_pos <").not_nil!
      q_pointer = kernel.index("device const half4* q4", q_guard).not_nil!
      q_pointer.should be > q_guard

      k_guard = kernel.index("if (k_rows == ATTENTION_TAIL_ROWS)").not_nil!
      k_direct = kernel.index("simdgroup_load(mk[0], pk", k_guard).not_nil!
      k_else = kernel.index("} else {", k_direct).not_nil!
      k_staged = kernel.index("simdgroup_load(mk[0], tail_scratch", k_else).not_nil!
      k_direct.should be < k_else
      k_staged.should be > k_else
      kernel.includes?("if (k_rows < ATTENTION_TAIL_ROWS)").should be_true

      v_guard = kernel.index("if (v_rows == ATTENTION_TAIL_ROWS)").not_nil!
      v_pointer = kernel.index("device const half* pv", v_guard).not_nil!
      v_direct = kernel.index("simdgroup_load(mv, pv", v_guard).not_nil!
      v_else = kernel.index("} else {", v_direct).not_nil!
      v_staged = kernel.index("simdgroup_load(mv, tail_scratch", v_else).not_nil!
      v_pointer.should be < v_else
      v_direct.should be < v_else
      v_staged.should be > v_else
      kernel.includes?("if (v_rows < ATTENTION_TAIL_ROWS)").should be_true
    end
  end

  it "allocates the per-simdgroup tail scratch at every attention dispatch" do
    METAL_BACKEND_SOURCE.includes?("ATTN_MATMUL_TAIL_ROWS = 8").should be_true
    METAL_BACKEND_SOURCE.includes?("sh_tail = ATTN_MATMUL_NSG * ATTN_MATMUL_TAIL_ROWS * head_dim * 2").should be_true
    METAL_BACKEND_SOURCE.includes?("sh_q + sh_o + sh_s + sh_tail").should be_true
    METAL_BACKEND_SOURCE.scan("enc.set_threadgroup_memory(attention_matmul_shmem_bytes(head_dim), 0)").size.should eq(4)
    METAL_BACKEND_SOURCE.scan("enc.set_threadgroup_memory(sh_q + sh_o + sh_s, 0)").size.should eq(0)
  end
end
