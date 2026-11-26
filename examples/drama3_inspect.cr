require "../src/ml/audio/drama3/prompt"
require "../src/ml/audio/drama3/backbone"

# A small native asset/tokenizer probe, also useful before allocating caches.
abort "usage: drama3_inspect MODEL_DIR [TEXT]" unless ARGV.size.in?(1..2)
path = File.expand_path(ARGV[0])
root = JSON.parse(File.read(File.join(path, "config.json")))
config = ML::Audio::Drama3::BackboneConfig.new(root)
tokenizer = ML::Audio::Drama3::Tokenizer.new(File.join(path, "tokenizer.json"), root["eos_token_id"].as_i, root["pad_token_id"]?.try(&.as_i) || root["eos_token_id"].as_i)
store = ML::Audio::Drama3::WeightStore.new(path)
text = ARGV[1]? || "Hello world."
puts({"ids" => tokenizer.encode(text), "prompt" => ML::Audio::Drama3::Prompt.build(text, tokenizer).map(&.[0]), "embedding" => store.matrix("backbone.embed_tokens.weight").row(tokenizer.encode(text)[0]), "layers" => config.layers}.to_json)
