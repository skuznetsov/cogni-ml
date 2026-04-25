#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_deltanet_scan_model"

prompts = [64, 256, 1024, 2048]
blocks = [8, 16, 32]
target = 1.25
compose_values = [8.0, 32.0, ML::GGUF::Qwen35DeltaNetScanModel.naive_dense_compose_token_equiv(128)]
show_rank_growth = false

private def parse_i32_list(value : String) : Array(Int32)
  value.split(',').map(&.strip).reject(&.empty?).map(&.to_i)
end

private def parse_f64_list(value : String) : Array(Float64)
  value.split(',').map(&.strip).reject(&.empty?).map(&.to_f64)
end

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_deltanet_scan_model [--prompts LIST] [--blocks LIST] [--compose LIST] [--target N]"
  p.on("--prompts=LIST", "Prompt lengths, comma-separated (default: 64,256,1024,2048)") { |v| prompts = parse_i32_list(v) }
  p.on("--blocks=LIST", "Block sizes, comma-separated (default: 8,16,32)") { |v| blocks = parse_i32_list(v) }
  p.on("--compose=LIST", "Summary compose costs in token-equivalent steps (default: 8,32,dense-s128)") { |v| compose_values = parse_f64_list(v) }
  p.on("--target=N", "Target speedup gate (default: 1.25)") { |v| target = v.to_f64 }
  p.on("--rank-growth", "Also print compact rank-growth adversary model") { show_rank_growth = true }
  p.on("-h", "--help", "Show this help") do
    puts p
    exit
  end
end

model = ML::GGUF::Qwen35DeltaNetScanModel
dense = model.naive_dense_compose_token_equiv(128)

puts "Qwen35 DeltaNet associative block-scan critical-path model"
puts "target_speedup=#{target}"
puts "dense_s128_compose_token_equiv=#{dense.round(2)}"
puts
printf "%8s %6s %12s %6s %12s %8s %14s %8s\n",
  "prompt", "block", "compose_eq", "levels", "depth_eq", "speedup", "max_compose", "pass?"

prompts.each do |prompt|
  blocks.each do |block|
    compose_values.each do |compose|
      est = model.estimate(prompt, block, compose, target)
      pass = est.speedup >= target
      printf "%8d %6d %12.2f %6d %12.2f %8.2f %14.2f %8s\n",
        est.prompt_tokens,
        est.block_size,
        est.compose_token_equiv,
        est.scan_levels,
        est.parallel_depth,
        est.speedup,
        est.max_compose_for_target,
        pass ? "yes" : "no"
    end
  end
  puts
end

puts "Interpretation:"
puts "- compose_eq is the cost of one block-summary composition measured in serial-token-equivalent DeltaNet steps."
puts "- depth_eq is critical-path depth, not total GPU work or memory traffic."
puts "- Dense summaries should fail this model for s=128; compact low-rank summaries must stay below max_compose."

if show_rank_growth
  puts
  puts "Compact rank-growth adversary model"
  puts "Assumes exact compact summary composition cost grows with rank^2; rank_cap=128 is an optimistic exact compression gate."
  printf "%8s %6s %10s %6s %10s %12s %8s %8s\n",
    "prompt", "block", "rank_cap", "levels", "max_rank", "compose_eq", "depth", "speedup"
  prompts.each do |prompt|
    blocks.each do |block|
      {nil, 128}.each do |cap|
        est = model.rank_growth_estimate(prompt, block, 128, cap)
        printf "%8d %6d %10s %6d %10d %12.2f %8.2f %8.2f\n",
          est.prompt_tokens,
          est.block_size,
          cap ? cap.to_s : "none",
          est.scan_levels,
          est.max_rank_on_path,
          est.compose_depth,
          est.parallel_depth,
          est.speedup
      end
    end
    puts
  end
end
