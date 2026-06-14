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
    next unless window.variant_fast?
    next if File.file?(window.variant_route_artifact)

    STDERR.puts "error: variant_fast artifact missing for #{window.prompt_token}:#{window.canvas_token}: #{window.variant_route_artifact}"
    exit 4
  end
end

fast_windows = plan.windows.select(&.variant_fast?).map { |window| "#{window.prompt_token}:#{window.canvas_token}" }.join(",")
fallback_windows = plan.exact_fallback_windows_spec
artifact_map = plan.variant_route_artifact_map

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
  puts "DIFFUSION_GEMMA_MIXED_FAST_ROUTE_ARTIFACT_MAP=#{shell_quote(artifact_map)}"
when "tsv"
  puts "kind\tprompt_token\tcanvas_token\tselected_route\treason\tmixed_speedup\tvariant_route_artifact\tchild_log"
  plan.windows.each do |window|
    puts [
      "window",
      window.prompt_token,
      window.canvas_token,
      window.selected_route,
      window.reason,
      window.mixed_speedup,
      window.variant_route_artifact,
      window.child_log,
    ].join('\t')
  end
end
