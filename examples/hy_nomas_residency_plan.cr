require "option_parser"
require "./support/hy_nomas_residency_plan"
require "../spec/support/video_allocation_counter"

header = ""
budget = 4_i64 * 1024 * 1024 * 1024
required_scenario = ""
OptionParser.parse do |parser|
  parser.banner = "Header-only Hy-Nomas DiT parameter and serial-block residency estimates; no model execution."
  parser.on("--header PATH", "Pinned full checkpoint JSON header only (252904 bytes)") { |value| header = value }
  parser.on("--estimate-budget-bytes N", "Comparison budget; not runtime admission or a peak guarantee") { |value| budget = value.to_i64 }
  parser.on("--require-cold-scenario NAME", "Reject if the selected cold estimate exceeds the comparison budget") { |value| required_scenario = value }
end
raise ArgumentError.new("header and positive estimate budget required") if header.empty? || budget <= 0
raise ArgumentError.new("unexpected positional arguments") unless ARGV.empty?
before = ML::MetalBuffer.video_spec_allocations
inventory = HyNomasResidencyPlan::Inventory.load!(header)
short = HyNomasResidencyPlan.geometry(HyNomasResidencyPlan.config(6, 23))
long = HyNomasResidencyPlan.geometry(HyNomasResidencyPlan.config(5280, 1280), 256, 16)
two_weights = inventory.block_f32_bytes // inventory.block_count * 2
scenarios = {
  "short_two_blocks"         => HyNomasResidencyPlan::Scenario.new(short, 2, two_weights),
  "j6560_two_blocks"         => HyNomasResidencyPlan::Scenario.new(long, 2, two_weights),
  "j6560_all_dit_parameters" => HyNomasResidencyPlan::Scenario.new(long, inventory.block_count, inventory.f32_bytes),
}
unless required_scenario.empty?
  selected = scenarios[required_scenario]? || raise ArgumentError.new("unknown required scenario")
  HyNomasResidencyPlan.require_budget!(selected.estimated_cold_bytes, budget)
end
raise "metadata planner allocated a MetalBuffer" unless before == ML::MetalBuffer.video_spec_allocations
puts({schema: "hy_nomas_residency_plan_v1", status: "metadata_accounted",
      inventory: inventory.report, estimate_budget_bytes: budget,
      gpu_buffer_allocations: ML::MetalBuffer.video_spec_allocations - before,
      runtime_launch_authorized: false, full_pipeline_accounted: false,
      scenarios: scenarios.map { |name, value| value.report(name, budget) },
      exclusions: ["Qwen encoder weights/workspaces", "PixelUMM weights/workspaces",
                   "camera/history/KV state and camera execution workspace",
                   "refiner, input and final layer execution workspace",
                   "scheduler/latents and non-block conditioning buffers",
                   "host/runtime objects and allocations beyond the stated MPS heuristic"],
      proposals: ["N-block shared workspace with explicit output lifetime policy",
                  "two-bank outputs (not implemented)",
                  "authenticated BF16-source cache (not present; would avoid raw F32 staging)"],
      interpretation: "All checkpoint parameters are counted. Workspace estimates cover block execution only. Estimates and budget comparisons do not authorize a model run or establish physical peaks, quality or speed."}.to_pretty_json)
