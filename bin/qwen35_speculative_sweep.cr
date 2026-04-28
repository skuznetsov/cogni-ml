#!/usr/bin/env crystal

# Multi-prompt policy sweep for the Qwen35 speculative acceptance harness.
#
# Build the runner once, then sweep policies without hand-copying metrics:
#
#   crystal build --release bin/qwen35_speculative_accept.cr -o /tmp/qwen35_speculative_accept
#   crystal run bin/qwen35_speculative_sweep.cr -- --runner /tmp/qwen35_speculative_accept
#
# This is intentionally a measurement driver. It does not load models itself and
# does not change inference semantics. Policies are interleaved inside each
# prompt/repetition block to reduce host-drift bias.

require "option_parser"

record Policy, name : String, args : Array(String), env : Hash(String, String)

record RunResult,
  policy : String,
  prompt : String,
  rep : Int32,
  ok : Bool,
  spec_ms_tok : Float64?,
  plain_ms_tok : Float64?,
  speedup : Float64?,
  accept_rate : Float64?,
  accepted : Int32?,
  proposed : Int32?,
  cycles : Int32?,
  max_gamma : Int32?,
  final_gamma : Int32?,
  stderr : String,
  stdout : String

runner = ENV["QWEN35_SPEC_SWEEP_RUNNER"]? || "/tmp/qwen35_speculative_accept"
tokens = 32
gamma = 4
max_gamma = 32
reps = 1
prompts = [
  "The capital of France is",
  "The quick brown fox",
  "def fibonacci(n):",
]
policy_names = ["default", "guard", "bootstrap32", "bootstrap32_s2", "bootstrap32_guard", "hybrid", "ngram", "ngram_guard"]
extra_args = [] of String

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_speculative_sweep [--runner PATH] [--tokens N] [--reps N] [--policies LIST] [--prompt TEXT] [--extra-arg ARG]"
  p.on("--runner PATH", "Compiled qwen35_speculative_accept binary (default: /tmp/qwen35_speculative_accept)") { |v| runner = v }
  p.on("--tokens N", "Generated tokens per run (default: 32)") { |v| tokens = v.to_i }
  p.on("--gamma N", "Initial speculative gamma (default: 4)") { |v| gamma = v.to_i }
  p.on("--max-gamma N", "Maximum adaptive gamma (default: 32)") { |v| max_gamma = v.to_i }
  p.on("--reps N", "Repeat each policy/prompt this many times (default: 1)") { |v| reps = v.to_i }
  p.on("--policies LIST", "Comma-separated policies: default,guard,bootstrap32,bootstrap32_s2,bootstrap32_guard,hybrid,ngram,ngram_guard; *_guard explicitly enables research-only guarded verifier") do |v|
    policy_names = v.split(',').map(&.strip).reject(&.empty?)
  end
  p.on("--prompt TEXT", "Add one prompt; can be passed multiple times") { |v| prompts << v }
  p.on("--only-prompts LIST", "Replace prompt set with pipe-separated prompts") do |v|
    prompts = v.split('|').map(&.strip).reject(&.empty?)
  end
  p.on("--extra-arg ARG", "Extra arg forwarded to every runner invocation; can be repeated") { |v| extra_args << v }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--tokens must be positive" unless tokens > 0
raise "--gamma must be positive" unless gamma > 0
raise "--max-gamma must be positive" unless max_gamma > 0
raise "--reps must be positive" unless reps > 0
raise "runner not found: #{runner}" unless File.exists?(runner)
raise "at least one prompt is required" if prompts.empty?

policies = {
  "default" => Policy.new("default", [] of String, {} of String => String),
  "guard"   => Policy.new("guard", ["--allow-guarded-verifier"], {
    "QWEN35_HEAD_FULL_ROWS_GUARDED" => "1",
  }),
  "bootstrap32"       => Policy.new("bootstrap32", ["--bootstrap-gamma", "32"], {} of String => String),
  "bootstrap32_s2"    => Policy.new("bootstrap32_s2", ["--bootstrap-gamma", "32", "--bootstrap-streak", "2"], {} of String => String),
  "bootstrap32_guard" => Policy.new("bootstrap32_guard", ["--bootstrap-gamma", "32", "--allow-guarded-verifier"], {
    "QWEN35_HEAD_FULL_ROWS_GUARDED" => "1",
  }),
  "hybrid"      => Policy.new("hybrid", ["--verify", "hybrid"], {} of String => String),
  "ngram"       => Policy.new("ngram", ["--ngram"], {} of String => String),
  "ngram_guard" => Policy.new("ngram_guard", ["--ngram", "--allow-guarded-verifier"], {
    "QWEN35_HEAD_FULL_ROWS_GUARDED" => "1",
  }),
}

selected = policy_names.map do |name|
  policies[name]? || raise "unknown policy #{name.inspect}; known: #{policies.keys.join(", ")}"
end

def parse_f64(pattern : Regex, text : String) : Float64?
  match = text.match(pattern)
  match ? match[1].to_f64 : nil
end

def parse_i32(pattern : Regex, text : String) : Int32?
  match = text.match(pattern)
  match ? match[1].to_i : nil
end

def run_one(runner : String,
            policy : Policy,
            prompt : String,
            rep : Int32,
            tokens : Int32,
            gamma : Int32,
            max_gamma : Int32,
            extra_args : Array(String)) : RunResult
  args = [
    "--tokens", tokens.to_s,
    "--gamma", gamma.to_s,
    "--max-gamma", max_gamma.to_s,
  ]
  args.concat(policy.args)
  args.concat(extra_args)
  args << prompt

  stdout = IO::Memory.new
  stderr = IO::Memory.new
  status = Process.run(runner, args, env: policy.env, output: stdout, error: stderr)
  stdout_text = stdout.to_s
  err = stderr.to_s

  spec = parse_f64(/spec_wall=.*?\(([0-9.]+) ms\/tok,/, stdout_text)
  plain = parse_f64(/plain_target_wall=.*?\(([0-9.]+) ms\/tok,/, stdout_text)
  accept_rate = parse_f64(/accept_rate=([0-9.]+)%/, stdout_text)
  accepted = parse_i32(/accepted=(\d+)\/\d+/, stdout_text)
  proposed = parse_i32(/accepted=\d+\/(\d+)/, stdout_text)
  cycles = parse_i32(/accept_rate=.*?cycles=(\d+)/, stdout_text)
  max_seen = parse_i32(/gamma_stats .*?max_seen=(\d+)/, stdout_text)
  final_gamma = parse_i32(/gamma_stats .*?final=(\d+)/, stdout_text)
  speedup = if spec && plain
              plain / spec
            end

  RunResult.new(
    policy: policy.name,
    prompt: prompt,
    rep: rep,
    ok: status.success? && !!spec && !!plain,
    spec_ms_tok: spec,
    plain_ms_tok: plain,
    speedup: speedup,
    accept_rate: accept_rate,
    accepted: accepted,
    proposed: proposed,
    cycles: cycles,
    max_gamma: max_seen,
    final_gamma: final_gamma,
    stderr: err,
    stdout: stdout_text,
  )
end

results = [] of RunResult

# Interleave policies inside each prompt/rep block so host drift affects the
# candidates more evenly than a policy-major run order.
reps.times do |rep|
  prompts.each do |prompt|
    selected.each do |policy|
      result = run_one(runner, policy, prompt, rep, tokens, gamma, max_gamma, extra_args)
      results << result
      unless result.ok
        STDERR.puts "FAILED policy=#{policy.name} prompt=#{prompt.inspect}"
        STDERR.puts result.stderr
      end
    end
  end
end

baseline_by_key = Hash({String, Int32}, RunResult).new
results.each do |r|
  baseline_by_key[{r.prompt, r.rep}] = r if r.policy == "default" && r.ok
end

puts "Qwen35 speculative policy sweep"
puts "runner=#{runner}"
puts "tokens=#{tokens} gamma=#{gamma} max_gamma=#{max_gamma} reps=#{reps}"
puts
printf "%-18s %-26s %9s %9s %8s %9s %9s %7s %7s\n",
  "policy", "prompt", "spec", "plain", "speedup", "accept", "accepted", "cycles", "gamma"

results.each do |r|
  prompt = r.prompt.size > 25 ? "#{r.prompt[0, 22]}..." : r.prompt
  accepted = if r.accepted && r.proposed
               "#{r.accepted}/#{r.proposed}"
             else
               "?"
             end
  gamma_s = r.max_gamma ? "#{r.max_gamma}/#{r.final_gamma || 0}" : "?"
  printf "%-18s %-26s %9s %9s %8s %8s%% %9s %7s %7s\n",
    r.policy,
    prompt.inspect,
    r.spec_ms_tok.try(&.round(2).to_s) || "FAIL",
    r.plain_ms_tok.try(&.round(2).to_s) || "FAIL",
    r.speedup.try { |v| "#{v.round(3)}x" } || "?",
    r.accept_rate.try(&.round(2).to_s) || "?",
    accepted,
    r.cycles.try(&.to_s) || "?",
    gamma_s
end

puts
puts "Averages by policy"
printf "%-18s %9s %9s %8s %8s%% %7s\n", "policy", "spec", "plain", "speedup", "accept", "runs"
selected.each do |policy|
  rows = results.select { |r| r.policy == policy.name && r.ok }
  next if rows.empty?
  spec_avg = rows.compact_map(&.spec_ms_tok).sum / rows.size
  plain_avg = rows.compact_map(&.plain_ms_tok).sum / rows.size
  speed_avg = rows.compact_map(&.speedup).sum / rows.size
  accept_avg = rows.compact_map(&.accept_rate).sum / rows.size
  printf "%-18s %9.2f %9.2f %7.3fx %8.2f%% %7d\n",
    policy.name, spec_avg, plain_avg, speed_avg, accept_avg, rows.size
end

if baseline_by_key.any?
  puts
  puts "Paired vs default"
  printf "%-18s %7s %7s %10s %10s\n", "policy", "wins", "pairs", "delta_ms", "avg_ratio"
  selected.each do |policy|
    next if policy.name == "default"
    pairs = results.compact_map do |r|
      next unless r.policy == policy.name && r.ok
      base = baseline_by_key[{r.prompt, r.rep}]?
      next unless base && base.spec_ms_tok && r.spec_ms_tok
      {r.spec_ms_tok.not_nil! - base.spec_ms_tok.not_nil!, r.spec_ms_tok.not_nil! / base.spec_ms_tok.not_nil!}
    end
    next if pairs.empty?
    wins = pairs.count { |(delta, _ratio)| delta < 0.0 }
    avg_delta = pairs.sum { |(delta, _ratio)| delta } / pairs.size
    avg_ratio = pairs.sum { |(_delta, ratio)| ratio } / pairs.size
    printf "%-18s %7d %7d %10.2f %9.3fx\n",
      policy.name, wins, pairs.size, avg_delta, avg_ratio
  end
end
