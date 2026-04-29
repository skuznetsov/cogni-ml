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
require "json"

record Policy, name : String, args : Array(String), env : Hash(String, String)
record PromptCase, name : String, text : String

record RunResult,
  policy : String,
  prompt_name : String,
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
  cycle_dump_path : String?,
  stderr : String,
  stdout : String

def json_i(rec : JSON::Any, key : String) : Int32
  if value = rec[key]?
    value.as_i.to_i32
  else
    0
  end
end

def json_f(rec : JSON::Any, key : String) : Float64
  if value = rec[key]?
    value.as_f? || 0.0
  else
    0.0
  end
end

class CycleStats
  property count : Int32 = 0
  property proposed : Int32 = 0
  property accepted : Int32 = 0
  property rejects : Int32 = 0
  property generated : Int32 = 0
  property gain_ms : Float64 = 0.0
  property wall_ms : Float64 = 0.0
  property draft_ms : Float64 = 0.0
  property verify_ms : Float64 = 0.0
  property backup_ms : Float64 = 0.0

  def add(rec : JSON::Any)
    @count += 1
    @proposed += json_i(rec, "proposed_count")
    @accepted += json_i(rec, "accepted_count")
    @rejects += 1 if json_i(rec, "reject_index") >= 0
    @generated += json_i(rec, "generated_count")
    @gain_ms += json_f(rec, "expected_gain_ms")
    @wall_ms += json_f(rec, "wall_ms")
    @draft_ms += json_f(rec, "draft_ms")
    @verify_ms += json_f(rec, "target_verify_ms")
    @backup_ms += json_f(rec, "target_backup_ms") + json_f(rec, "draft_backup_ms") + json_f(rec, "draft_resync_ms")
  end
end

def fnv1a64_hex(bytes : Bytes) : String
  hash = 0xcbf29ce484222325_u64
  bytes.each do |b|
    hash = (hash ^ b.to_u64) &* 0x100000001b3_u64
  end
  hash.to_s(16)
end

def prompt_case_from_spec(spec : String, index : Int32) : PromptCase
  if match = spec.match(/\A([A-Za-z0-9_.-]+)::(.+)\z/m)
    PromptCase.new(match[1], match[2])
  else
    PromptCase.new("prompt#{index}", spec)
  end
end

def safe_name(name : String) : String
  safe = name.gsub(/[^A-Za-z0-9_.-]/, "_")
  safe.empty? ? "prompt" : safe
end

def write_prompt_manifest(dir : String, prompts : Array(PromptCase), include_text : Bool)
  Dir.mkdir_p(dir)
  path = File.join(dir, "prompt_manifest.jsonl")
  File.open(path, "w") do |io|
    prompts.each_with_index do |prompt, index|
      base = {
        prompt_index: index,
        prompt_name:  prompt.name,
        prompt_hash:  fnv1a64_hex(prompt.text.to_slice),
        prompt_bytes: prompt.text.bytesize,
      }
      if include_text
        base.merge({prompt_text: prompt.text}).to_json(io)
      else
        base.to_json(io)
      end
      io.puts
    end
  end
end

runner = ENV["QWEN35_SPEC_SWEEP_RUNNER"]? || "/tmp/qwen35_speculative_accept"
tokens = 32
gamma = 4
max_gamma = 32
reps = 1
prompts = [
  PromptCase.new("france", "The capital of France is"),
  PromptCase.new("quick_brown_fox", "The quick brown fox"),
  PromptCase.new("code_fibonacci", "def fibonacci(n):"),
]
policy_names = ["default", "guard", "bootstrap32", "bootstrap32_s2", "bootstrap32_guard", "router16", "fixed16", "staged16", "hybrid", "ngram", "ngram_bootstrap32_s2", "ngram_router16", "ngram_fixed16", "ngram_staged16", "ngram_guard"]
extra_args = [] of String
dump_cycles_dir = ENV["QWEN35_SPEC_SWEEP_DUMP_CYCLES_DIR"]?
dump_cycle_token_ids = ENV["QWEN35_SPEC_DUMP_TOKEN_IDS"]? == "1"
dump_prompt_texts = ENV["QWEN35_SPEC_DUMP_PROMPT_TEXTS"]? == "1"

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_speculative_sweep [--runner PATH] [--tokens N] [--reps N] [--policies LIST] [--prompt TEXT] [--extra-arg ARG]"
  p.on("--runner PATH", "Compiled qwen35_speculative_accept binary (default: /tmp/qwen35_speculative_accept)") { |v| runner = v }
  p.on("--tokens N", "Generated tokens per run (default: 32)") { |v| tokens = v.to_i }
  p.on("--gamma N", "Initial speculative gamma (default: 4)") { |v| gamma = v.to_i }
  p.on("--max-gamma N", "Maximum adaptive gamma (default: 32)") { |v| max_gamma = v.to_i }
  p.on("--reps N", "Repeat each policy/prompt this many times (default: 1)") { |v| reps = v.to_i }
  p.on("--policies LIST", "Comma-separated policies: default,guard,bootstrap32,bootstrap32_s2,bootstrap32_guard,router16,fixed16,staged16,hybrid,ngram,ngram_bootstrap32_s2,ngram_router16,ngram_fixed16,ngram_staged16,ngram_guard; *_guard explicitly enables research-only guarded verifier") do |v|
    policy_names = v.split(',').map(&.strip).reject(&.empty?)
  end
  p.on("--prompt TEXT", "Add one prompt; can be passed multiple times. Optional NAME::TEXT gives a stable dataset label.") do |v|
    prompts << prompt_case_from_spec(v, prompts.size)
  end
  p.on("--only-prompts LIST", "Replace prompt set with pipe-separated prompts. Entries may be NAME::TEXT.") do |v|
    prompts = v.split('|').map(&.strip).reject(&.empty?).map_with_index { |spec, i| prompt_case_from_spec(spec, i) }
  end
  p.on("--extra-arg ARG", "Extra arg forwarded to every runner invocation; can be repeated") { |v| extra_args << v }
  p.on("--dump-cycles-dir DIR", "Write per-run cycle JSONL files under DIR by forwarding --dump-cycles to the runner") { |v| dump_cycles_dir = v }
  p.on("--dump-cycle-token-ids", "Forward --dump-cycle-token-ids to the runner; default sweep dumps only candidate hashes") { dump_cycle_token_ids = true }
  p.on("--dump-prompt-texts", "Include raw prompt text in dump prompt_manifest.jsonl; default writes labels and hashes only") { dump_prompt_texts = true }
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
  "router16"             => Policy.new("router16", ["--gamma", "4", "--max-gamma", "16", "--bootstrap-gamma", "16", "--bootstrap-streak", "1"], {} of String => String),
  "fixed16"              => Policy.new("fixed16", ["--gamma", "16", "--max-gamma", "16", "--no-adaptive"], {} of String => String),
  "staged16"             => Policy.new("staged16", ["--gamma", "16", "--max-gamma", "16", "--no-adaptive", "--verify", "staged", "--stage-gate", "4"], {} of String => String),
  "hybrid"               => Policy.new("hybrid", ["--verify", "hybrid"], {} of String => String),
  "ngram"                => Policy.new("ngram", ["--ngram"], {} of String => String),
  "ngram_bootstrap32_s2" => Policy.new("ngram_bootstrap32_s2", ["--ngram", "--bootstrap-gamma", "32", "--bootstrap-streak", "2"], {} of String => String),
  "ngram_router16"       => Policy.new("ngram_router16", ["--ngram", "--gamma", "4", "--max-gamma", "16", "--bootstrap-gamma", "16", "--bootstrap-streak", "1"], {} of String => String),
  "ngram_fixed16"        => Policy.new("ngram_fixed16", ["--ngram", "--gamma", "16", "--max-gamma", "16", "--no-adaptive"], {} of String => String),
  "ngram_staged16"       => Policy.new("ngram_staged16", ["--ngram", "--gamma", "16", "--max-gamma", "16", "--no-adaptive", "--verify", "staged", "--stage-gate", "4"], {} of String => String),
  "ngram_guard"          => Policy.new("ngram_guard", ["--ngram", "--allow-guarded-verifier"], {
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
            prompt : PromptCase,
            prompt_index : Int32,
            rep : Int32,
            tokens : Int32,
            gamma : Int32,
            max_gamma : Int32,
            extra_args : Array(String),
            dump_cycles_dir : String?,
            dump_cycle_token_ids : Bool) : RunResult
  args = [
    "--tokens", tokens.to_s,
    "--gamma", gamma.to_s,
    "--max-gamma", max_gamma.to_s,
  ]
  args.concat(policy.args)
  args.concat(extra_args)
  cycle_dump_path = nil.as(String?)
  if dir = dump_cycles_dir
    Dir.mkdir_p(dir)
    safe_policy = policy.name.gsub(/[^A-Za-z0-9_.-]/, "_")
    cycle_dump_path = File.join(dir, "rep#{rep}_prompt#{prompt_index}_#{safe_name(prompt.name)}_#{safe_policy}.jsonl")
    args.concat(["--dump-cycles", cycle_dump_path])
    args << "--dump-cycle-token-ids" if dump_cycle_token_ids
  end
  args << prompt.text

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
    prompt_name: prompt.name,
    prompt: prompt.text,
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
    cycle_dump_path: cycle_dump_path,
    stderr: err,
    stdout: stdout_text,
  )
end

results = [] of RunResult
write_prompt_manifest(dump_cycles_dir.not_nil!, prompts, dump_prompt_texts) if dump_cycles_dir

# Interleave policies inside each prompt/rep block so host drift affects the
# candidates more evenly than a policy-major run order.
reps.times do |rep|
  prompts.each_with_index do |prompt, prompt_index|
    selected.each do |policy|
      result = run_one(runner, policy, prompt, prompt_index, rep, tokens, gamma, max_gamma, extra_args, dump_cycles_dir, dump_cycle_token_ids)
      results << result
      unless result.ok
        STDERR.puts "FAILED policy=#{policy.name} prompt=#{prompt.name.inspect}"
        STDERR.puts result.stderr
      end
    end
  end
end

baseline_by_key = Hash({String, Int32}, RunResult).new
results.each do |r|
  baseline_by_key[{r.prompt_name, r.rep}] = r if r.policy == "default" && r.ok
end

puts "Qwen35 speculative policy sweep"
puts "runner=#{runner}"
puts "tokens=#{tokens} gamma=#{gamma} max_gamma=#{max_gamma} reps=#{reps} dump_cycles_dir=#{dump_cycles_dir || ""} dump_token_ids=#{dump_cycle_token_ids}"
puts
printf "%-18s %-26s %9s %9s %8s %9s %9s %7s %7s\n",
  "policy", "prompt", "spec", "plain", "speedup", "accept", "accepted", "cycles", "gamma"

results.each do |r|
  prompt = r.prompt_name
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
      base = baseline_by_key[{r.prompt_name, r.rep}]?
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

if dump_cycles_dir
  dumped = results.compact_map(&.cycle_dump_path).count { |path| path && File.exists?(path) }
  puts
  puts "Cycle JSONL dumps: #{dumped}/#{results.size} files in #{dump_cycles_dir}"
  stats = Hash(String, CycleStats).new { |hash, key| hash[key] = CycleStats.new }
  results.each do |result|
    path = result.cycle_dump_path
    next unless path && File.exists?(path)
    File.each_line(path) do |line|
      next if line.empty?
      rec = JSON.parse(line)
      key = "#{rec["policy"].as_s}/#{rec["kind"].as_s}"
      stats[key].add(rec)
    end
  end
  unless stats.empty?
    puts
    puts "Cycle JSONL summary"
    printf "%-28s %7s %9s %9s %8s %10s %9s %9s %9s\n",
      "policy/kind", "cycles", "accepted", "proposed", "rejects", "gain_ms", "wall_ms", "draft_ms", "verify_ms"
    stats.keys.sort.each do |key|
      stat = stats[key]
      printf "%-28s %7d %9d %9d %8d %10.1f %9.1f %9.1f %9.1f\n",
        key, stat.count, stat.accepted, stat.proposed, stat.rejects,
        stat.gain_ms, stat.wall_ms, stat.draft_ms, stat.verify_ms
    end
  end
end
