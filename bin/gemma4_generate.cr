require "json"
require "option_parser"
require "../src/ml/gguf/gemma4_chat"

DEFAULT_PROFILE_BIN = "#{Dir.current}/build/gemma4_metal_decode_profile"

profile_bin = ENV["GEMMA4_PROFILE_BIN"]? || ENV["GEMMA4_GENERATE_PROFILE_BIN"]? || DEFAULT_PROFILE_BIN
prompt = nil.as(String?)
generate = (ENV["GEMMA4_MAX_TOKENS"]? || ENV["GEMMA4_GENERATE"]? || "128").to_i
warmups = (ENV["GEMMA4_WARMUPS"]? || "0").to_i
runs = (ENV["GEMMA4_RUNS"]? || "1").to_i
prefill_chunk = (ENV["GEMMA4_PREFILL_CHUNK"]? || "4").to_i
max_seq = ENV["GEMMA4_MAX_SEQ"]?.try(&.to_i)
model = ENV["GEMMA4_MODEL"]?
tool_response_format = (ENV["GEMMA4_TOOL_RESPONSE_JSON"]? || "simple").downcase
quiet_contract = ENV["GEMMA4_GENERATE_QUIET_CONTRACT"]? == "1"
dry_run = false
extra_args = [] of String

OptionParser.parse(ARGV) do |p|
  p.banner = "usage: gemma4_generate [prompt] [n_tokens] [--profile-bin PATH]"
  p.on("--profile-bin PATH", "Compiled gemma4_metal_decode_profile binary") { |v| profile_bin = v }
  p.on("--generate N", "Maximum generated tokens") { |v| generate = v.to_i }
  p.on("--warmups N", "Profiler warmup runs") { |v| warmups = v.to_i }
  p.on("--runs N", "Profiler measured runs") { |v| runs = v.to_i }
  p.on("--max-seq N", "Profiler max sequence length") { |v| max_seq = v.to_i }
  p.on("--model PATH", "Gemma4 GGUF model path") { |v| model = v }
  p.on("--prefill-chunk N", "Gemma4 row prefill chunk") { |v| prefill_chunk = v.to_i }
  p.on("--tool-response-json FORMAT", "Emit parsed tool response JSON: simple or openai") { |v| tool_response_format = v.downcase }
  p.on("--quiet-contract", "Only print the profiler output contract; currently keeps profiler sentinel output unchanged") { quiet_contract = true }
  p.on("--dry-run", "Print argv JSON instead of executing the profiler") { dry_run = true }
  p.on("--", "Pass remaining args to profiler") do
    extra_args.concat(ARGV)
    ARGV.clear
  end
  p.on("-h", "--help", "Show help") { puts p; exit }
  p.unknown_args do |before_dash, after_dash|
    if arg_prompt = before_dash[0]?
      prompt = arg_prompt
    end
    if arg_gen = before_dash[1]?
      generate = arg_gen.to_i
    end
    extra_args.concat(after_dash)
  end
end

raise "GEMMA4_TOOL_RESPONSE_JSON must be simple or openai" unless tool_response_format == "simple" || tool_response_format == "openai"
raise "GEMMA4_MAX_TOKENS / --generate must be positive" unless generate > 0
raise "GEMMA4_WARMUPS / --warmups must be non-negative" unless warmups >= 0
raise "GEMMA4_RUNS / --runs must be positive" unless runs > 0
raise "GEMMA4_PREFILL_CHUNK / --prefill-chunk must be positive" unless prefill_chunk > 0

messages_json = ENV["GEMMA4_MESSAGES_JSON"]?
tools_json = ENV["GEMMA4_TOOLS_JSON"]?
tools = if raw = tools_json
          raw.empty? ? [] of JSON::Any : ML::GGUF::Gemma4Chat.parse_tools_json(raw)
        else
          [] of JSON::Any
        end

argv = [] of String
argv << profile_bin
argv.concat(["--generate", generate.to_s, "--warmups", warmups.to_s, "--runs", runs.to_s])
argv.concat(["--profile-decode-only", "--decode-wave", "--top1-wave-resident", "--prefill-mode", "rows", "--prefill-chunk", prefill_chunk.to_s])
argv << "--print-generated-text"
argv.concat(["--tool-response-json", tool_response_format])
argv.concat(["--max-seq", max_seq.not_nil!.to_s]) if max_seq
argv.concat(["--model", model.not_nil!]) if model

if raw_tools = tools_json
  argv.concat(["--tools-json", raw_tools]) unless raw_tools.empty?
end

if raw_messages = messages_json
  messages = ML::GGUF::Gemma4Chat.messages_from_openai_json(raw_messages)
  rendered = ML::GGUF::Gemma4Chat.render(messages, tools: tools, add_generation_prompt: true)
  argv.concat(["--prompt", rendered])
else
  user_prompt = prompt || ENV["GEMMA4_PROMPT"]? || "Hello"
  argv.concat(["--chat-user", user_prompt])
end

if !tools.empty? && ENV["GEMMA4_CONSTRAINED_TOOLS_OFF"]? != "1"
  finite_options = ML::GGUF::Gemma4Chat.native_tool_finite_call_options(tools)
  if !finite_options.empty?
    argv << "--constrained-gemma-tool-finite-call"
    argv << "--literal-stop-after-complete"
    argv << "--literal-force-single-off" if ENV["GEMMA4_LITERAL_FORCE_SINGLE_OFF"]? == "1"
    argv << "--literal-force-span-off" if ENV["GEMMA4_LITERAL_FORCE_SPAN_OFF"]? == "1"
  end
end

argv.concat(extra_args)

if dry_run
  puts({"argv" => argv, "tool_response_json" => tool_response_format, "quiet_contract" => quiet_contract}.to_json)
  exit
end

unless File::Info.executable?(profile_bin)
  raise "Gemma4 profile binary is not executable: #{profile_bin}. Build it or set GEMMA4_PROFILE_BIN=/path/to/gemma4_metal_decode_profile"
end

status = Process.run(argv[0], argv[1..], output: STDOUT, error: STDERR)
exit(status.exit_code)
