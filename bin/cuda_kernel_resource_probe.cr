# Reports CUDA Driver JIT resource attributes for hot embedded PTX kernels.
# This uses cuFuncGetAttribute, so it reflects the active driver/GPU JIT even
# when standalone ptxas/nvdisasm are unavailable on the host.

require "option_parser"
require "../src/ml/cuda/driver"

record KernelSpec, module_label : String, ptx : String, functions : Array(String)

Q4K_PTX       = {{ read_file("src/ml/cuda/kernels/q4k_gemv_probe.ptx") }}
Q5K_PTX       = {{ read_file("src/ml/cuda/kernels/q5k_gemv_probe.ptx") }}
Q6K_PTX       = {{ read_file("src/ml/cuda/kernels/q6k_gemv_probe.ptx") }}
DN_PTX        = {{ read_file("src/ml/cuda/kernels/deltanet_step_probe.ptx") }}
FULL_ATTN_PTX = {{ read_file("src/ml/cuda/kernels/fullattn_post_probe.ptx") }}

format = "table"
only = [] of String

OptionParser.parse do |p|
  p.banner = "Usage: cuda_kernel_resource_probe [--format table|csv] [--only SUBSTR]"
  p.on("--format FORMAT", "Output format: table or csv") { |v| format = v }
  p.on("--only SUBSTR", "Report only functions/modules containing substring; repeatable") { |v| only << v }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "unsupported format #{format}" unless {"table", "csv"}.includes?(format)

specs = [
  KernelSpec.new("q4", Q4K_PTX, [
    "q4_k_gemv_warp4_f32",
    "q4_k_gemv_warp4_f32_xsum",
    "q4_k_gemv_add_warp4_f32",
    "q4_k_gemv_warp4_f32_batched",
    "q4_k_gemv_add_warp4_f32_batched",
  ]),
  KernelSpec.new("q5", Q5K_PTX, [
    "q5_k_gemv_warp4_f32",
    "q5_k_gemv_warp4_f32_batched",
  ]),
  KernelSpec.new("q6", Q6K_PTX, [
    "q6_k_gemv_warp4_f32",
    "q6_k_gemv_add_warp4_f32",
    "q6_k_gemv_warp4_f32_batched",
    "q6_k_gemv_add_warp4_f32_batched",
    "q6_k_gemv_top1_partial_f32",
  ]),
  KernelSpec.new("delta", DN_PTX, [
    "rmsnorm_vec_parallel_probe",
    "add_rmsnorm_vec_parallel_probe",
    "swiglu_probe",
    "deltanet_step_128_probe",
  ]),
  KernelSpec.new("fullattn", FULL_ATTN_PTX, [
    "full_attn_decode_cache_probe",
    "full_attn_decode_cache_parallel_probe",
    "full_attn_q_split_norm_rope_probe",
    "full_attn_k_norm_rope_cache_probe",
  ]),
]

ctx = ML::CUDA::Context.create
modules = [] of ML::CUDA::CUDAModule
begin
  rows = [] of Tuple(String, String, Int32, Int32, Int32, Int32, Int32, Int32)
  specs.each do |spec|
    next unless only.empty? || only.any? { |needle| spec.module_label.includes?(needle) || spec.functions.any?(&.includes?(needle)) }

    mod = ML::CUDA::CUDAModule.load(spec.ptx, spec.module_label)
    modules << mod
    spec.functions.each do |fn_name|
      next unless only.empty? || only.any? { |needle| spec.module_label.includes?(needle) || fn_name.includes?(needle) }

      fn = mod.function(fn_name)
      rows << {
        spec.module_label,
        fn_name,
        fn.attribute(ML::CUDA::FunctionAttribute::NumRegs),
        fn.attribute(ML::CUDA::FunctionAttribute::SharedSizeBytes),
        fn.attribute(ML::CUDA::FunctionAttribute::LocalSizeBytes),
        fn.attribute(ML::CUDA::FunctionAttribute::MaxThreadsPerBlock),
        fn.attribute(ML::CUDA::FunctionAttribute::PtxVersion),
        fn.attribute(ML::CUDA::FunctionAttribute::BinaryVersion),
      }
    end
  end

  puts "device=#{ctx.device_name} cc=#{ctx.compute_capability_major}.#{ctx.compute_capability_minor}"
  if format == "csv"
    puts "module,function,num_regs,shared_bytes,local_bytes,max_threads,ptx_version,binary_version"
    rows.each do |row|
      puts row.join(",")
    end
  else
    puts "module,function,num_regs,shared_bytes,local_bytes,max_threads,ptx_version,binary_version"
    rows.each do |mod, fn, regs, shared, local, max_threads, ptx, bin|
      puts "#{mod},#{fn},#{regs},#{shared},#{local},#{max_threads},#{ptx},#{bin}"
    end
  end
ensure
  modules.each(&.close)
  ctx.close
end
