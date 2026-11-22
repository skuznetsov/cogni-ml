require "option_parser"
require "../src/ml/gguf/inkling2_greedy_decode"
require "../src/ml/gguf/inkling2_tokenizer"

private def parse_cli_integer(text : String, option : String) : Int32
  text.to_i32? || raise ArgumentError.new("#{option} expects a 32-bit integer, got #{text.inspect}")
end

private def parse_token_id_csv(text : String) : Array(Int32)
  parts = text.split(',')
  raise ArgumentError.new("--token-ids must contain at least one token id") if parts.empty?

  parts.map_with_index do |part, index|
    value = part.strip
    raise ArgumentError.new("--token-ids contains an empty field at position #{index + 1}") if value.empty?
    id = parse_cli_integer(value, "--token-ids")
    raise ArgumentError.new("--token-ids cannot contain negative ids") if id < 0
    id
  end
end

model_path = nil.as(String?)
prompt_text = nil.as(String?)
forced_ids = nil.as(Array(Int32)?)
max_new_tokens = 32
context = 2048
force_cpu = false
staged_metal = false
raw_prompt = false
print_ids = false
ngram = false
trace_loops = false
passes = nil.as(Int32?)
prefill_tile = nil.as(Int32?)
auto_prefill_tile = false
stop_delta = nil.as(Float32?)
help = false

parser = OptionParser.new
parser.banner = "Usage: ./build/inkling2_generate --model PATH --prompt TEXT [options]"
parser.on("--model PATH", "Inkling 2 Mini or TML-d24-varloop-research GGUF path (required)") { |value| model_path = value }
parser.on("--prompt TEXT", "One user text message (omit when --token-ids is supplied)") { |value| prompt_text = value }
parser.on("--token-ids CSV", "Force prompt token IDs instead of tokenizing --prompt") { |value| forced_ids = parse_token_id_csv(value) }
parser.on("--tokens N", "Maximum new tokens (default: 32)") { |value| max_new_tokens = parse_cli_integer(value, "--tokens") }
parser.on("--context N", "Model context size (default: 2048)") { |value| context = parse_cli_integer(value, "--context") }
parser.on("--cpu", "Force CPU inference; Metal is the default on Darwin unless built with -Dcpu_only") { force_cpu = true }
parser.on("--staged-metal", "Use the staged Metal backend with token-by-token prefill") { staged_metal = true }
parser.on("--raw", "Use --prompt as a raw completion, without the chat template") { raw_prompt = true }
parser.on("--passes N", "Pass override for variable-loop checkpoints; defaults to checkpoint depth") { |value| passes = parse_cli_integer(value, "--passes") }
parser.on("--prefill-tile SIZE", "Resident Metal tile: auto or 1–#{ML::GGUF::Inkling2Engine::MAX_PREFILL_TILE} (default: #{ML::GGUF::Inkling2Engine::DEFAULT_PREFILL_TILE})") do |value|
  auto_prefill_tile = value == "auto"
  prefill_tile = auto_prefill_tile ? nil : parse_cli_integer(value, "--prefill-tile")
end
parser.on("--stop-delta DELTA", "Research only: stop after two successive loop deltas at or below DELTA") do |value|
  stop_delta = value.to_f32? || raise ArgumentError.new("--stop-delta expects a number, got #{value.inspect}")
end
parser.on("--trace-loops", "Write position, completed_pass, and relative_delta records to stderr") { trace_loops = true }
parser.on("--print-ids", "Write generated token IDs as CSV to stderr") { print_ids = true }
parser.on("--ngram", "Verify repeated continuations in four-token blocks on fixed-depth resident Metal") { ngram = true }
parser.on("--help", "Show this help") { help = true }

begin
  parser.parse
rescue ex : Exception
  STDERR.puts("inkling2_generate: #{ex.message}")
  STDERR.puts(parser)
  exit 2
end

if help
  puts parser
  exit 0
end

begin
  raise ArgumentError.new("--model is required") unless model_path
  model = model_path.not_nil!
  raise ArgumentError.new("model not found: #{model}") unless File.exists?(model)
  raise ArgumentError.new("--prompt is required unless --token-ids is supplied") if prompt_text.nil? && forced_ids.nil?
  raise ArgumentError.new("--prompt and --token-ids cannot be used together") if prompt_text && forced_ids
  raise ArgumentError.new("--raw cannot be combined with --token-ids") if raw_prompt && forced_ids
  raise ArgumentError.new("--tokens must be zero or greater") if max_new_tokens < 0
  raise ArgumentError.new("--context must be greater than zero") if context <= 0
  if pass_count = passes
    raise ArgumentError.new("--passes must be greater than zero") if pass_count <= 0
  end

  engine = nil.as(ML::GGUF::Inkling2Engine?)
  begin
    use_metal = false
    {% if flag?(:darwin) && !flag?(:cpu_only) %}
      use_metal = !force_cpu
    {% end %}

    trace = if trace_loops
              ->(position : Int32, completed_pass : Int32, relative_delta : Float32) do
                STDERR.puts("loop_trace position=#{position} completed_pass=#{completed_pass} relative_delta=#{relative_delta}")
              end
            else
              nil
            end

    engine = ML::GGUF::Inkling2Engine.new(model, context, metal: use_metal, resident: !staged_metal,
      passes: passes, trace: trace, stop_delta: stop_delta, prefill_tile: prefill_tile,
      auto_prefill_tile: auto_prefill_tile, greedy_verifier: ngram)
    tokenizer = ML::GGUF::Inkling2Tokenizer.from_gguf(engine.weights.gguf)
    prompt_ids = if ids = forced_ids
                   ids
                 else
                   text = prompt_text.not_nil!
                   rendered = raw_prompt ? text : tokenizer.render_user_prompt(text)
                   tokenizer.encode(rendered)
                 end

    raise ArgumentError.new("prompt token sequence is empty") if prompt_ids.empty?
    prompt_ids.each do |id|
      raise ArgumentError.new("prompt token id #{id} is outside 0...#{tokenizer.vocab.size}") if id < 0 || id >= tokenizer.vocab.size
    end
    if prompt_ids.size.to_i64 + max_new_tokens > context
      raise ArgumentError.new("prompt (#{prompt_ids.size}) plus requested output (#{max_new_tokens}) exceeds context (#{context})")
    end

    generated_ids = [] of Int32
    if max_new_tokens > 0
      first_id = ML::GGUF::Inkling2Engine.greedy_token_id(engine.prefill(prompt_ids))
      generated_ids = ML::GGUF::Inkling2GreedyDecode.generate(engine, prompt_ids, first_id,
        max_new_tokens, tokenizer.eos_id, ngram: ngram).ids
    end

    STDERR.puts("backfilled_core_passes=#{engine.backfilled_passes}") if trace_loops && stop_delta
    STDERR.puts("generated_token_ids=#{generated_ids.join(',')}") if print_ids
    puts tokenizer.decode(generated_ids)
  ensure
    engine.try(&.close)
  end
rescue ex : Exception
  STDERR.puts("inkling2_generate: #{ex.message}")
  exit 1
end
