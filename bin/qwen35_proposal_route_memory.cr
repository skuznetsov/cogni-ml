#!/usr/bin/env crystal

require "json"
require "option_parser"
require "./../src/ml/gguf/reader"
require "./../src/ml/gguf/qwen35_prompt_cache"
require "./../src/ml/gguf/qwen35_tokenizer"

DEFAULT_MODEL     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

mode = "save"
root = ENV["QWEN35_PROMPT_CACHE_ROOT"]? || ML::GGUF::Qwen35PromptCache.default_root
model = ENV["QWEN35_MODEL"]? || DEFAULT_MODEL
tokenizer_bin = ENV["QWEN35_TOKENIZER"]? || DEFAULT_TOKENIZER
prompt = nil.as(String?)
prompt_file = nil.as(String?)
route_key = nil.as(String?)
route = nil.as(String?)
rank = nil.as(Int32?)
layers = [] of Int32
trigger = nil.as(String?)
evidence = nil.as(String?)
tokens_limit = 0
json_output = false

parse_int_list = ->(raw : String) do
  raw.split(',').map(&.strip).reject(&.empty?).map(&.to_i)
end

OptionParser.parse(ARGV) do |p|
  p.banner = "Usage: qwen35_proposal_route_memory --save|--lookup|--list [options]"
  p.on("--save", "Save a proposal-route certificate (default)") { mode = "save" }
  p.on("--lookup", "Lookup a proposal-route certificate by --route-key or exact prompt tokens") { mode = "lookup" }
  p.on("--list", "List proposal-route certificates in the cache root") { mode = "list" }
  p.on("--root=PATH", "Prompt-cache root (default: QWEN35_PROMPT_CACHE_ROOT or #{ML::GGUF::Qwen35PromptCache.default_root})") { |v| root = v }
  p.on("--model=PATH", "GGUF model path used for model/tokenizer ids") { |v| model = v }
  p.on("--tokenizer=PATH", "Optional llama-tokenize fallback path") { |v| tokenizer_bin = v }
  p.on("--prompt=TEXT", "Prompt text for exact prompt-token certificate") { |v| prompt = v }
  p.on("--prompt-file=PATH", "Read prompt text from a UTF-8 file") { |v| prompt_file = v }
  p.on("--tokens=N", "Keep only the first N prompt tokens for the certificate; 0 means all") { |v| tokens_limit = v.to_i }
  p.on("--route-key=KEY", "Stable caller-certified task/session route key") { |v| route_key = v }
  p.on("--route=NAME", "Route to save: baseline or pca_updown") { |v| route = v }
  p.on("--rank=N", "PCA-updown rank for --route=pca_updown") { |v| rank = v.to_i }
  p.on("--layers=LIST", "Comma-separated PCA-updown layers") { |v| layers = parse_int_list.call(v) }
  p.on("--trigger=TEXT", "Human-readable trigger/evidence source") { |v| trigger = v }
  p.on("--evidence=TEXT", "Human-readable measurement evidence") { |v| evidence = v }
  p.on("--json", "Emit JSON instead of text rows") { json_output = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

abort "--tokens must be non-negative" if tokens_limit < 0
abort "model not found: #{model}" unless File.exists?(model)

gguf = ML::GGUF::GGUFFile.new(model)
tok = ML::GGUF::Qwen35Tokenizer.from_gguf(gguf, model, tokenizer_bin)
model_info = File.info(model)
model_id = ML::GGUF::Qwen35PromptCache.short_hash("model\0#{model}\0#{model_info.size}\0#{model_info.modification_time.to_unix}")
tokenizer_id = ML::GGUF::Qwen35PromptCache.short_hash("tokenizer\0#{model_id}\0#{tok.vocab.size}\0#{tok.eos_id}\0#{tok.pad_id}")
store = ML::GGUF::Qwen35PromptCache::Store.new(root)

load_prompt = -> do
  text = prompt
  if file = prompt_file
    text = File.read(file)
  end
  text
end

encode_prompt = ->(text : String) do
  ids = tok.encode(text, add_bos_override: false)
  if tokens_limit > 0 && ids.size > tokens_limit
    ids = ids[0, tokens_limit]
  end
  abort "prompt produced no tokens" if ids.empty?
  ids
end

emit_entry = ->(entry : ML::GGUF::Qwen35PromptCache::ProposalRouteEntry) do
  if json_output
    entry.to_json(STDOUT)
    STDOUT << '\n'
  else
    rank_text = entry.route_rank ? entry.route_rank.to_s : "na"
    layers_text = entry.route_layers.empty? ? "default" : entry.route_layers.join(',')
    key_text = entry.route_key_preview || "exact_prompt"
    puts "proposal_route route=#{entry.route} rank=#{rank_text} layers=#{layers_text} key=#{key_text} prompt_tokens=#{entry.prompt_token_count} trigger=#{entry.trigger || "unknown"} evidence=#{entry.evidence || "none"}"
  end
end

case mode
when "save"
  abort "--save requires --route=baseline|pca_updown" unless route
  route_value = route.not_nil!
  text = load_prompt.call || abort "--save requires --prompt or --prompt-file"
  ids = encode_prompt.call(text)
  entry = store.save_proposal_route(
    model_id: model_id,
    tokenizer_id: tokenizer_id,
    prompt_text: text,
    token_ids: ids,
    route: route_value,
    route_rank: rank,
    route_layers: layers,
    route_key: route_key,
    trigger: trigger,
    evidence: evidence,
  )
  emit_entry.call(entry)
when "lookup"
  hit = if key = route_key
          store.lookup_proposal_route_key(model_id, tokenizer_id, key)
        else
          text = load_prompt.call || abort "--lookup requires --route-key or --prompt/--prompt-file"
          ids = encode_prompt.call(text)
          store.lookup_proposal_route(model_id, tokenizer_id, text, ids)
        end
  if entry = hit
    emit_entry.call(entry)
  else
    puts(json_output ? "null" : "proposal_route miss")
    exit 2
  end
when "list"
  entries = store.proposal_route_entries.select do |entry|
    ML::GGUF::Qwen35PromptCache.proposal_route_entry_valid?(entry, model_id, tokenizer_id)
  end
  if json_output
    entries.to_json(STDOUT)
    STDOUT << '\n'
  else
    entries.each { |entry| emit_entry.call(entry) }
  end
else
  abort "unknown mode #{mode.inspect}"
end
