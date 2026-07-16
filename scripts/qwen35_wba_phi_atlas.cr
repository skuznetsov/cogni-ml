#!/usr/bin/env crystal

require "option_parser"
require "../src/ml/gguf/qwen35_wba_phi_atlas"

paths = [] of String
combine = false
summary = false

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: scripts/qwen35_wba_phi_atlas.cr [--summary] [--combine] [profile-or-ab.log ...]"
  p.on("--summary", "Print one TSV summary row per analyzed log") { summary = true }
  p.on("--combine", "Combine multiple logs into one explicit aggregate analysis") { combine = true }
  p.on("-h", "--help", "Show help") { puts p; exit }
  p.unknown_args { |args| paths = args }
end

analyses = ML::GGUF::Qwen35WbaPhiAtlas.analyze_paths(paths, combine: combine)
if summary
  puts "source\tverdict\tactive_window\tactive_signal_ms\ttied_routes\tbad_corners\tremaining_work_mib\tenv\tdefault_p50_ms\tother_p50_ms\tdefault_wins"
  analyses.each { |analysis| puts analysis.summary_tsv }
else
  analyses.each_with_index do |analysis, index|
    puts unless index == 0
    puts analysis.render
  end
end
