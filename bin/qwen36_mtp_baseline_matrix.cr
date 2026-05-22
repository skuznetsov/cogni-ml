#!/usr/bin/env crystal

require "option_parser"

record Candidate,
  name : String,
  kind : String,
  hf_spec : String?,
  local_globs : Array(String),
  native_supported : Bool,
  notes : String

HOME_DIR             = ENV["HOME"]
DEFAULT_LLAMA_CLI    = "#{HOME_DIR}/SrcArchives/AI/llama.cpp/build/bin/llama-cli"
DEFAULT_LLAMA_SERVER = "#{HOME_DIR}/SrcArchives/AI/llama.cpp/build/bin/llama-server"
DEFAULT_NATIVE_BENCH = "./build/benchmark_qwen_vs_llama"

CANDIDATES = [
  Candidate.new(
    "local_q4_k_m_no_mtp",
    "plain",
    nil,
    ["#{HOME_DIR}/.cache/lm-studio/models/**/Qwen3.6-27B-Q4_K_M.gguf"],
    true,
    "Current native target. No built-in GGUF MTP head; native sidecar MTP exists separately.",
  ),
  Candidate.new(
    "unsloth_iq4_nl_mtp",
    "mtp",
    "unsloth/Qwen3.6-27B-MTP-GGUF:IQ4_NL",
    [
      "#{HOME_DIR}/.cache/**/Qwen3.6-27B-IQ4_NL.gguf",
      "#{HOME_DIR}/.cache/**/Qwen3.6-27B*IQ4_NL*mtp*.gguf",
      "#{HOME_DIR}/.cache/**/Qwen3.6-27B*IQ4_NL*MTP*.gguf",
    ],
    false,
    "Reddit finalist target. llama.cpp/other runtime baseline first; native IQ4_NL support is not implemented yet.",
  ),
  Candidate.new(
    "unsloth_ud_q4_k_xl_mtp",
    "mtp",
    "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q4_K_XL",
    [
      "#{HOME_DIR}/.cache/**/Qwen3.6-27B-UD-Q4_K_XL.gguf",
      "#{HOME_DIR}/.cache/**/Qwen3.6-27B*UD-Q4_K_XL*mtp*.gguf",
      "#{HOME_DIR}/.cache/**/Qwen3.6-27B*UD-Q4_K_XL*MTP*.gguf",
    ],
    false,
    "Reddit finalist alternative. Useful for MTP throughput/quality comparison; native UD quant support is not implemented yet.",
  ),
  Candidate.new(
    "atomic_q4_k_xl_mtp",
    "mtp",
    "AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF:Q4_K_XL_MTP",
    ["#{HOME_DIR}/.cache/**/Qwen3.6-27B*Q4_K_XL_MTP*.gguf"],
    false,
    "Reproducible UDT MTP artifact with documented HF alias; benchmark as an external MTP baseline.",
  ),
]

llama_cli = ENV["LLAMA_CLI"]? || DEFAULT_LLAMA_CLI
llama_server = ENV["LLAMA_SERVER"]? || DEFAULT_LLAMA_SERVER
native_bench = ENV["QWEN35_NATIVE_BENCH"]? || DEFAULT_NATIVE_BENCH
prompt = "The capital of France is"
gen = 128
ctx = 8192
ngl = 99
cache_k = "q8_0"
cache_v = "q4_0"
spec_draft_n_max = 3
run_available = false
show_commands = true
repeats = 1
compare_plain = false

OptionParser.parse do |p|
  p.banner = "Usage: qwen36_mtp_baseline_matrix [options]"
  p.on("--llama-cli=PATH", "Path to llama-cli") { |v| llama_cli = v }
  p.on("--llama-server=PATH", "Path to llama-server") { |v| llama_server = v }
  p.on("--native-bench=PATH", "Path to build/benchmark_qwen_vs_llama") { |v| native_bench = v }
  p.on("--prompt=TEXT", "Prompt for llama-cli MTP timing smoke") { |v| prompt = v }
  p.on("--gen=N", "Generated tokens for llama-cli timing smoke (default: 128)") { |v| gen = v.to_i }
  p.on("--ctx=N", "Context size for llama.cpp commands (default: 8192)") { |v| ctx = v.to_i }
  p.on("--ngl=N", "llama.cpp GPU layers (default: 99)") { |v| ngl = v.to_i }
  p.on("--cache-k=TYPE", "llama.cpp K cache type (default: q8_0)") { |v| cache_k = v }
  p.on("--cache-v=TYPE", "llama.cpp V cache type (default: q4_0)") { |v| cache_v = v }
  p.on("--spec-draft-n-max=N", "MTP draft max for llama.cpp server/CLI (default: 3)") { |v| spec_draft_n_max = v.to_i }
  p.on("--run-available", "Run llama-cli timing smokes for local MTP files and native bench for supported local files") { run_available = true }
  p.on("--repeats=N", "Repeat each llama-cli timing row and print min/median/max (default: 1)") { |v| repeats = v.to_i }
  p.on("--compare-plain", "For MTP candidates, also time the same file without draft-mtp") { compare_plain = true }
  p.on("--no-commands", "Only print the candidate table") { show_commands = false }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

raise "--repeats must be >= 1" if repeats < 1

def shell_quote(s : String) : String
  "'" + s.gsub("'", "'\"'\"'") + "'"
end

def first_existing(globs : Array(String)) : String?
  globs.each do |glob|
    if path = Dir.glob(glob).find { |candidate| File.file?(candidate) }
      return path
    end
  end
  nil
end

def binary_help(binary : String) : String
  return "" unless executable_file?(binary)
  output = IO::Memory.new
  error = IO::Memory.new
  Process.run(binary, ["--help"], output: output, error: error)
  output.to_s + "\n" + error.to_s
rescue
  ""
end

def executable_file?(path : String) : Bool
  File.file?(path)
end

def llama_model_arg(candidate : Candidate, local_path : String?) : Array(String)
  if local_path
    ["-m", local_path]
  elsif hf = candidate.hf_spec
    ["-hf", hf]
  else
    [] of String
  end
end

def print_cmd(label : String, args : Array(String)) : Nil
  puts "  #{label}:"
  puts "    #{args.map { |a| shell_quote(a) }.join(" ")}"
end

def parse_llama_cli_timing(output : String) : {Float64?, Float64?}
  prompt_ts = nil.as(Float64?)
  decode_ts = nil.as(Float64?)
  output.each_line do |line|
    if line =~ /\[\s*Prompt:\s*([0-9.]+)\s*t\/s\s*\|\s*Generation:\s*([0-9.]+)\s*t\/s\s*\]/
      prompt_ts = $1.to_f
      decode_ts = $2.to_f
    elsif line.includes?("prompt eval time") && line =~ /,\s*([0-9.]+)\s+tokens per second\)/
      prompt_ts = $1.to_f
    elsif line.includes?("eval time") && line =~ /,\s*([0-9.]+)\s+tokens per second\)/
      decode_ts = $1.to_f
    end
  end
  {prompt_ts, decode_ts}
end

record TimingSample, status : Int32, prompt_tok_s : Float64?, decode_tok_s : Float64?

def run_llama_cli_timing(llama_cli : String, args : Array(String)) : TimingSample
  output = IO::Memory.new
  error = IO::Memory.new
  status = Process.run(llama_cli, args, output: output, error: error)
  timing = parse_llama_cli_timing(output.to_s + "\n" + error.to_s)
  TimingSample.new(status.exit_code, timing[0], timing[1])
end

def median(values : Array(Float64)) : Float64?
  return nil if values.empty?
  sorted = values.sort
  mid = sorted.size // 2
  if sorted.size.odd?
    sorted[mid]
  else
    (sorted[mid - 1] + sorted[mid]) / 2.0
  end
end

def fmt_rate(value : Float64?) : String
  value ? value.round(2).to_s : "?"
end

def print_timing_summary(label : String, samples : Array(TimingSample)) : Nil
  prompts = [] of Float64
  decodes = [] of Float64
  statuses = [] of Int32
  samples.each do |sample|
    statuses << sample.status
    if value = sample.prompt_tok_s
      prompts << value
    end
    if value = sample.decode_tok_s
      decodes << value
    end
  end

  puts "  run #{label}: statuses=#{statuses.join(",")}"
  puts "    prompt_tok_s min=#{fmt_rate(prompts.min?)} median=#{fmt_rate(median(prompts))} max=#{fmt_rate(prompts.max?)} samples=#{prompts.map { |v| v.round(2) }.join(",")}"
  puts "    decode_tok_s min=#{fmt_rate(decodes.min?)} median=#{fmt_rate(median(decodes))} max=#{fmt_rate(decodes.max?)} samples=#{decodes.map { |v| v.round(2) }.join(",")}"
end

llama_cli_help = binary_help(llama_cli)
llama_server_help = binary_help(llama_server)
llama_cli_has_spec_type = llama_cli_help.includes?("--spec-type")
llama_server_has_spec_type = llama_server_help.includes?("--spec-type")
llama_cli_has_draft_mtp = llama_cli_help.includes?("draft-mtp")
llama_server_has_draft_mtp = llama_server_help.includes?("draft-mtp")
native_bench_help = binary_help(native_bench)
native_bench_has_cache_args = native_bench_help.includes?("--llama-cache-k") && native_bench_help.includes?("--llama-cache-v")

puts "Qwen3.6 MTP baseline matrix"
puts "llama_cli=#{llama_cli} spec_type=#{llama_cli_has_spec_type} draft_mtp=#{llama_cli_has_draft_mtp}"
puts "llama_server=#{llama_server} spec_type=#{llama_server_has_spec_type} draft_mtp=#{llama_server_has_draft_mtp}"
puts "settings: gen=#{gen} ctx=#{ctx} ngl=#{ngl} cache_k=#{cache_k} cache_v=#{cache_v} spec_draft_n_max=#{spec_draft_n_max} repeats=#{repeats} compare_plain=#{compare_plain}"
puts

CANDIDATES.each do |candidate|
  local_path = first_existing(candidate.local_globs)
  puts "#{candidate.name}"
  puts "  kind=#{candidate.kind} native_supported=#{candidate.native_supported} local=#{local_path || "missing"} hf=#{candidate.hf_spec || "-"}"
  puts "  notes=#{candidate.notes}"

  if show_commands
    model_args = llama_model_arg(candidate, local_path)
    if model_args.empty?
      puts "  commands: unavailable until local path or HF spec is provided"
    else
      mtp_args = [
        llama_cli,
      ] + model_args + [
        "-ngl", ngl.to_s,
        "-c", ctx.to_s,
        "-n", gen.to_s,
        "-p", prompt,
        "--temp", "0",
        "--no-display-prompt",
        "--single-turn",
        "--no-warmup",
        "-fa", "on",
        "-ctk", cache_k,
        "-ctv", cache_v,
      ]
      if candidate.kind == "mtp"
        mtp_args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", spec_draft_n_max.to_s]
      end
      print_cmd("llama-cli #{candidate.kind == "mtp" ? "MTP" : "plain"} timing", mtp_args)

      if candidate.kind == "mtp" && candidate.hf_spec
        server_args = [
          llama_server,
        ] + model_args + [
          "-ngl", ngl.to_s,
          "-c", ctx.to_s,
          "-np", "1",
          "-fa", "on",
          "-ctk", cache_k,
          "-ctv", cache_v,
          "--spec-type", "draft-mtp",
          "--spec-draft-n-max", spec_draft_n_max.to_s,
        ]
        print_cmd("llama-server MTP baseline", server_args)
      end

      if candidate.native_supported && local_path
        native_args = [
          native_bench,
          "--model", local_path,
          "--prompt=64",
          "--gen", gen.to_s,
          "--reps=3",
          "--warmup=1",
          "--flash-attn",
          "--llama-cache-k", cache_k,
          "--llama-cache-v", cache_v,
        ]
        print_cmd("cogni-ml native/plain vs llama-bench", native_args)
      end
    end
  end

  if run_available
    if candidate.kind == "mtp" && !llama_cli_has_draft_mtp
      puts "  run: skipped; llama-cli does not advertise draft-mtp. Rebuild/update llama.cpp first."
    elsif candidate.kind == "mtp" && local_path
      plain_args = llama_model_arg(candidate, local_path) + [
        "-ngl", ngl.to_s,
        "-c", ctx.to_s,
        "-n", gen.to_s,
        "-p", prompt,
        "--temp", "0",
        "--no-display-prompt",
        "--single-turn",
        "--no-warmup",
        "-fa", "on",
        "-ctk", cache_k,
        "-ctv", cache_v,
      ]
      if compare_plain
        samples = [] of TimingSample
        repeats.times { samples << run_llama_cli_timing(llama_cli, plain_args) }
        print_timing_summary("plain", samples)
      end

      mtp_args = plain_args + ["--spec-type", "draft-mtp", "--spec-draft-n-max", spec_draft_n_max.to_s]
      samples = [] of TimingSample
      repeats.times { samples << run_llama_cli_timing(llama_cli, mtp_args) }
      print_timing_summary("mtp", samples)
    elsif candidate.native_supported && local_path && executable_file?(native_bench)
      if compare_plain
        plain_args = llama_model_arg(candidate, local_path) + [
          "-ngl", ngl.to_s,
          "-c", ctx.to_s,
          "-n", gen.to_s,
          "-p", prompt,
          "--temp", "0",
          "--no-display-prompt",
          "--single-turn",
          "--no-warmup",
          "-fa", "on",
          "-ctk", cache_k,
          "-ctv", cache_v,
        ]
        samples = [] of TimingSample
        repeats.times { samples << run_llama_cli_timing(llama_cli, plain_args) }
        print_timing_summary("plain", samples)
      end
      unless native_bench_has_cache_args
        puts "  run: skipped; native bench exists but does not advertise --llama-cache-k/--llama-cache-v. Rebuild build/benchmark_qwen_vs_llama first."
        next
      end
      status = Process.run(native_bench, ["--model", local_path, "--prompt=64", "--gen", gen.to_s, "--reps=3", "--warmup=1", "--flash-attn", "--llama-cache-k", cache_k, "--llama-cache-v", cache_v], output: STDOUT, error: STDERR)
      puts "  run: native bench exit=#{status.exit_code}"
    else
      puts "  run: skipped; local file missing or native bench unavailable"
    end
  end

  puts
end

unless llama_cli_has_draft_mtp || llama_server_has_draft_mtp
  STDERR.puts "warning: current llama.cpp binaries do not advertise draft-mtp; MTP baselines require a newer/rebuilt llama.cpp even though the source tree documents draft-mtp."
end
