require "option_parser"
require "../src/ml/gguf/diffusion_gemma_runtime"

plan_path = ""
format = "env"
require_artifacts = true

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_mixed_route_plan_env --plan PATH [--format env|tsv] [--allow-missing-artifacts]"
  p.on("--plan PATH", "Mixed route-plan JSONL path") { |v| plan_path = v }
  p.on("--format FORMAT", "Output format: env or tsv (default: env)") { |v| format = v }
  p.on("--allow-missing-artifacts", "Do not require variant_fast artifacts to exist") { require_artifacts = false }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

if plan_path.empty?
  STDERR.puts "error: --plan is required"
  exit 2
end

unless format == "env" || format == "tsv"
  STDERR.puts "error: --format must be env or tsv"
  exit 2
end

plan = ML::GGUF::DiffusionGemmaMixedRoutePlan.from_jsonl(plan_path)

if require_artifacts
  plan.windows.each do |window|
    artifact = window.selected_runtime_route_artifact
    next unless artifact
    next if File.file?(artifact)

    STDERR.puts "error: selected route artifact missing for #{window.prompt_token}:#{window.canvas_token}: #{artifact}"
    exit 4
  end
end

fast_windows = plan.windows.select(&.variant_fast?).map { |window| "#{window.prompt_token}:#{window.canvas_token}" }.join(",")
fallback_windows = plan.exact_fallback_windows_spec
fast_artifact_map = plan.variant_route_artifact_map
fallback_artifact_map = plan.exact_fallback_route_artifact_map
selected_artifact_map = plan.selected_runtime_route_artifact_map

def shell_quote(value : String) : String
  return "''" if value.empty?

  "'#{value.gsub("'", "'\"'\"'")}'"
end

case format
when "env"
  puts "DIFFUSION_GEMMA_MIXED_ROUTE_PLAN=#{shell_quote(plan_path)}"
  puts "DIFFUSION_GEMMA_MIXED_DECISION=#{shell_quote(plan.decision)}"
  puts "DIFFUSION_GEMMA_MIXED_SPEEDUP=#{plan.mixed_speedup}"
  puts "DIFFUSION_GEMMA_MIXED_CANDIDATE_WINDOWS=#{plan.candidate_windows}"
  puts "DIFFUSION_GEMMA_MIXED_FALLBACK_WINDOWS=#{plan.fallback_windows}"
  puts "DIFFUSION_GEMMA_MIXED_FAST_WINDOWS=#{shell_quote(fast_windows)}"
  puts "DIFFUSION_GEMMA_MIXED_EXACT_FALLBACK_WINDOWS=#{shell_quote(fallback_windows)}"
  puts "DIFFUSION_GEMMA_MIXED_FAST_ROUTE_ARTIFACT_MAP=#{shell_quote(fast_artifact_map)}"
  puts "DIFFUSION_GEMMA_MIXED_EXACT_FALLBACK_ROUTE_ARTIFACT_MAP=#{shell_quote(fallback_artifact_map)}"
  puts "DIFFUSION_GEMMA_MIXED_SELECTED_ROUTE_ARTIFACT_MAP=#{shell_quote(selected_artifact_map)}"
when "tsv"
  puts "kind\tprompt_token\tcanvas_token\tselected_route\tvariant_env_role\treason\tmixed_speedup\tbase_route_artifact\tvariant_route_artifact\tselected_route_artifact\tselected_route_artifact_arm\tselected_route_artifact_env_role\tchild_log"
  plan.windows.each do |window|
    puts [
      "window",
      window.prompt_token,
      window.canvas_token,
      window.selected_route,
      window.variant_env_role,
      window.reason,
      window.mixed_speedup,
      window.base_route_artifact,
      window.variant_route_artifact,
      window.selected_runtime_route_artifact || "",
      window.selected_runtime_route_artifact_arm,
      window.selected_runtime_route_artifact_env_role,
      window.child_log,
    ].join('\t')
  end
end
