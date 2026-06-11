require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"
require "../src/ml/gguf/gemma4_tokenizer"
require "../src/ml/gguf/reader"

DEFAULT_MODEL          = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
DEFAULT_LLAMA_TOKENIZE = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp-diffusiongemma-pr/build-dg/bin/llama-tokenize"

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
llama_tokenize = ENV["DIFFUSION_GEMMA_LLAMA_TOKENIZE"]? || ENV["LLAMA_TOKENIZE"]? || DEFAULT_LLAMA_TOKENIZE
prompt_text = nil.as(String?)
prompt_token = 1
prompt_tokens_arg = nil.as(String?)
prompt_lengths_arg = nil.as(String?)
canvas_text = nil.as(String?)
canvas_token = 0
canvas_tokens_arg = nil.as(String?)
canvas_lengths_arg = nil.as(String?)
candidate_ids_arg = nil.as(String?)
candidate_texts_arg = nil.as(String?)
candidate_count = nil.as(Int32?)
candidate_counts_arg = nil.as(String?)
max_layers = 1
steps = 1
proposal_top_k = 1
entropy_bound = 0.0_f32
stability_threshold = 1
seed = 7
adaptive = true
single_route = true
format = "keyvalue"
repeats = 1
warmups = 0
decode_canvas_text = false

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_sparse_loop_smoke [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--llama-tokenize PATH", "llama-tokenize binary for --prompt text") { |v| llama_tokenize = v }
  p.on("--prompt TEXT", "Text prompt, tokenized through the Gemma4 GGUF tokenizer") { |v| prompt_text = v }
  p.on("--prompt-token ID", "Prompt token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--prompt-tokens CSV", "Prompt token ids, overrides --prompt-token") { |v| prompt_tokens_arg = v }
  p.on("--prompt-lengths LIST", "Generate multiple synthetic prompt token lists, comma or space separated") { |v| prompt_lengths_arg = v }
  p.on("--canvas TEXT", "Initial canvas text, tokenized without BOS through the Gemma4 GGUF tokenizer") { |v| canvas_text = v }
  p.on("--canvas-token ID", "Initial canvas token id (default: 0)") { |v| canvas_token = v.to_i }
  p.on("--canvas-tokens CSV", "Initial canvas token ids, overrides --canvas-token") { |v| canvas_tokens_arg = v }
  p.on("--canvas-lengths LIST", "Generate multiple synthetic canvas token lists, comma or space separated") { |v| canvas_lengths_arg = v }
  p.on("--candidate-ids CSV", "Sparse candidate token ids for the canvas row (default: canvas token)") { |v| candidate_ids_arg = v }
  p.on("--candidate-texts LIST", "Pipe-separated one-token candidate texts, tokenized without BOS") { |v| candidate_texts_arg = v }
  p.on("--candidate-count N", "Generate N sparse candidate ids starting at the canvas token") { |v| candidate_count = v.to_i }
  p.on("--candidate-counts LIST", "Generate multiple candidate-count rows, comma or space separated") { |v| candidate_counts_arg = v }
  p.on("--max-layers N", "Bounded decode layers (default: 1)") { |v| max_layers = v.to_i }
  p.on("--steps N", "Sparse denoise steps / adaptive budget (default: 1)") { |v| steps = v.to_i }
  p.on("--proposal-top-k N", "Adaptive proposal top-k (default: 1)") { |v| proposal_top_k = v.to_i }
  p.on("--entropy-bound F", "Entropy-bound acceptance budget (default: 0.0)") { |v| entropy_bound = v.to_f32 }
  p.on("--stability-threshold N", "Convergence threshold (default: 1)") { |v| stability_threshold = v.to_i }
  p.on("--seed N", "Deterministic sampling seed (default: 7)") { |v| seed = v.to_i }
  p.on("--fixed", "Use fixed candidate steps instead of adaptive proposals") { adaptive = false }
  p.on("--full-routes", "Use full top-k MoE routing instead of the single-route smoke shortcut") { single_route = false }
  p.on("--format FORMAT", "Output format: keyvalue or tsv (default: keyvalue)") { |v| format = v.downcase }
  p.on("--repeats N", "Repeat sparse loop after one model/cache load (default: 1)") { |v| repeats = v.to_i }
  p.on("--warmups N", "Run sparse loop warmups before measured repeats (default: 0)") { |v| warmups = v.to_i }
  p.on("--decode-canvas-text", "Decode initial/final canvas token ids for qualitative probe output") { decode_canvas_text = true }
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

def parse_token_ids(raw : String, label : String) : Array(Int32)
  ids = raw.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
  raise "#{label} must contain at least one id" if ids.empty?
  ids
end

def parse_candidate_ids(raw : String?, default_token : Int32) : Array(Int32)
  text = raw || default_token.to_s
  ids = parse_token_ids(text, "--candidate-ids")
  raise "--candidate-ids must contain at least one id" if ids.empty?
  ids.sort.uniq
end

def parse_candidate_counts(raw : String) : Array(Int32)
  counts = raw.split(/[,\s]+/).map(&.strip).reject(&.empty?).map(&.to_i)
  raise "--candidate-counts must contain at least one count" if counts.empty?
  counts
end

def parse_positive_counts(raw : String, label : String) : Array(Int32)
  counts = raw.split(/[,\s]+/).map(&.strip).reject(&.empty?).map(&.to_i)
  raise "#{label} must contain at least one count" if counts.empty?
  counts.each do |count|
    raise "#{label} entries must be positive" unless count > 0
  end
  counts
end

def generated_token_sequence(default_token : Int32, count : Int32, vocab_size : Int32, label : String) : Array(Int32)
  raise "#{label} must be positive" unless count > 0
  raise "#{label} exceeds vocab size" if count > vocab_size
  Array(Int32).new(count) { |i| (default_token + i) % vocab_size }
end

def generated_candidate_ids(default_token : Int32, count : Int32, vocab_size : Int32) : Array(Int32)
  generated_token_sequence(default_token, count, vocab_size, "--candidate-count").sort
end

def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  sorted[sorted.size // 2]
end

def encode_text_tokens(model : String, llama_tokenize : String, text : String, label : String, *, add_bos : Bool) : Array(Int32)
  raise "#{label} must not be empty" if text.empty?
  raise "--llama-tokenize not found: #{llama_tokenize}" unless File.exists?(llama_tokenize)

  g = ML::GGUF::GGUFFile.new(model)
  ids = begin
    tokenizer = ML::GGUF::Gemma4Tokenizer.from_gguf(g, model, llama_tokenize)
    tokenizer.encode(text, add_bos: add_bos)
  ensure
    g.close
  end
  raise "#{label} produced no token ids" if ids.empty?
  ids
end

def load_gemma4_tokenizer(model : String, llama_tokenize : String) : ML::GGUF::Gemma4Tokenizer
  g = ML::GGUF::GGUFFile.new(model)
  begin
    ML::GGUF::Gemma4Tokenizer.from_gguf(g, model, llama_tokenize)
  ensure
    g.close
  end
end

def output_safe_text(text : String) : String
  text.gsub('\t', ' ').gsub('\n', "\\n").gsub('\r', "\\r")
end

def output_safe_list_text(text : String) : String
  output_safe_text(text).gsub('|', "\\|")
end

def output_safe_nested_text(text : String) : String
  output_safe_list_text(text).gsub(',', "\\,")
end

def decode_candidate_rows(tokenizer : ML::GGUF::Gemma4Tokenizer, candidate_rows : Array(Array(Int32))) : String
  candidate_rows.map do |row|
    row.map { |id| output_safe_nested_text(tokenizer.decode([id])) }.join(",")
  end.join("|")
end

def format_f32(value : Float32) : String
  value.round(6).to_s
end

def format_prediction_f32_rows(predictions : Array(ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction),
                               &block : ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction -> Array(Float32)) : String
  predictions.map do |prediction|
    block.call(prediction).map { |value| format_f32(value) }.join(",")
  end.join("|")
end

def format_prediction_i32s(predictions : Array(ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction),
                           &block : ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction -> Int32) : String
  predictions.map { |prediction| block.call(prediction).to_s }.join(",")
end

def format_argmax_probabilities(predictions : Array(ML::GGUF::DiffusionGemmaCPU::BoundedDenoisePrediction)) : String
  predictions.map do |prediction|
    index = prediction.candidate_token_ids.index(prediction.argmax_token_id)
    index ? format_f32(prediction.probabilities[index]) : ""
  end.join(",")
end

def parse_candidate_texts(model : String, llama_tokenize : String, raw : String) : Tuple(Array(Int32), Array(String))
  texts = raw.split('|')
  raise "--candidate-texts must contain at least one text" if texts.empty? || texts.any?(&.empty?)

  pairs = [] of Tuple(Int32, String)
  texts.each do |text|
    ids = encode_text_tokens(model, llama_tokenize, text, "--candidate-texts entry", add_bos: false)
    raise "--candidate-texts entry #{text.inspect} encoded to #{ids.size} tokens; expected exactly one" unless ids.size == 1
    pairs << {ids[0], text}
  end

  unique = {} of Int32 => String
  pairs.each do |id, text|
    unique[id] ||= text
  end
  sorted = unique.to_a.sort_by { |id, _| id }
  {sorted.map(&.[0]), sorted.map { |_, text| text }}
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--max-layers must be positive" unless max_layers > 0
raise "--steps must be positive" unless steps > 0
raise "--proposal-top-k must be positive" unless proposal_top_k > 0
raise "--stability-threshold must be positive" unless stability_threshold > 0
raise "--entropy-bound must be finite and non-negative" unless entropy_bound.finite? && entropy_bound >= 0.0_f32
raise "--format must be keyvalue or tsv" unless {"keyvalue", "tsv"}.includes?(format)
raise "--repeats must be positive" unless repeats > 0
raise "--warmups must be non-negative" unless warmups >= 0
candidate_mode_count = 0
candidate_mode_count += 1 if candidate_ids_arg
candidate_mode_count += 1 if candidate_texts_arg
candidate_mode_count += 1 if candidate_count
candidate_mode_count += 1 if candidate_counts_arg
raise "--candidate-ids, --candidate-texts, --candidate-count, and --candidate-counts are mutually exclusive" if candidate_mode_count > 1
raise "--candidate-counts requires --format tsv" if candidate_counts_arg && format != "tsv"
raise "--prompt-tokens and --prompt-lengths are mutually exclusive" if prompt_tokens_arg && prompt_lengths_arg
raise "--prompt and --prompt-tokens are mutually exclusive" if prompt_text && prompt_tokens_arg
raise "--prompt and --prompt-lengths are mutually exclusive" if prompt_text && prompt_lengths_arg
raise "--prompt-lengths requires --format tsv" if prompt_lengths_arg && format != "tsv"
raise "--canvas and --canvas-tokens are mutually exclusive" if canvas_text && canvas_tokens_arg
raise "--canvas and --canvas-lengths are mutually exclusive" if canvas_text && canvas_lengths_arg
raise "--canvas-tokens and --canvas-lengths are mutually exclusive" if canvas_tokens_arg && canvas_lengths_arg
raise "--canvas-lengths requires --format tsv" if canvas_lengths_arg && format != "tsv"
raise "single-route smoke currently supports --max-layers 1; pass --full-routes for deeper smoke" if single_route && max_layers != 1
prompt_source = prompt_text ? "text" : (prompt_tokens_arg ? "token_ids" : (prompt_lengths_arg ? "synthetic_lengths" : "single_token"))
canvas_source = canvas_text ? "text" : (canvas_tokens_arg ? "token_ids" : (canvas_lengths_arg ? "synthetic_lengths" : "single_token"))
canvas_lengths = canvas_lengths_arg ? parse_positive_counts(canvas_lengths_arg.not_nil!, "--canvas-lengths") : nil
canvas_sets = [] of Array(Int32)
unless lengths = canvas_lengths
  canvas_tokens = if text = canvas_text
                    encode_text_tokens(model, llama_tokenize, text, "--canvas", add_bos: false)
                  else
                    canvas_tokens_arg ? parse_token_ids(canvas_tokens_arg.not_nil!, "--canvas-tokens") : [canvas_token]
                  end
  canvas_tokens.each do |token_id|
    raise "--canvas-token must be non-negative" if token_id < 0
  end
  canvas_sets << canvas_tokens
end
prompt_tokens = if text = prompt_text
                  encode_text_tokens(model, llama_tokenize, text, "--prompt", add_bos: true)
                else
                  prompt_tokens_arg ? parse_token_ids(prompt_tokens_arg.not_nil!, "--prompt-tokens") : [prompt_token]
                end
prompt_lengths = prompt_lengths_arg ? parse_positive_counts(prompt_lengths_arg.not_nil!, "--prompt-lengths") : nil
candidate_text_spec = if raw_texts = candidate_texts_arg
                        parse_candidate_texts(model, llama_tokenize, raw_texts)
                      else
                        nil
                      end

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams
canvas_decoder = decode_canvas_text ? load_gemma4_tokenizer(model, llama_tokenize) : nil

if lengths = canvas_lengths
  lengths.each do |length|
    raise "--canvas-lengths entry exceeds context length" if length > hp.context_length
    canvas_sets << generated_token_sequence(canvas_token, length, hp.vocab_size, "--canvas-lengths entry")
  end
end
canvas_sets.each do |tokens|
  tokens.each do |token_id|
    raise "--canvas-token out of range" if token_id < 0 || token_id >= hp.vocab_size
  end
end

prompt_sets = [] of Array(Int32)
if lengths = prompt_lengths
  lengths.each do |length|
    prompt_sets << generated_token_sequence(prompt_token, length, hp.vocab_size, "--prompt-lengths entry")
  end
else
  prompt_sets << prompt_tokens
end
prompt_sets.each do |tokens|
  tokens.each do |token_id|
    raise "--prompt-token out of range" if token_id < 0 || token_id >= hp.vocab_size
  end
end
prompt_sets.each do |prompt_set|
  canvas_sets.each do |canvas_set|
    raise "prompt+canvas exceeds context_length" if prompt_set.size + canvas_set.size > hp.context_length
  end
end

candidate_specs = [] of Tuple(Int32?, Array(Int32), Array(String)?)
if raw_counts = candidate_counts_arg
  parse_candidate_counts(raw_counts).each do |count|
    candidate_specs << {count, generated_candidate_ids(canvas_token, count, hp.vocab_size), nil}
  end
elsif count = candidate_count
  candidate_specs << {count, generated_candidate_ids(canvas_token, count, hp.vocab_size), nil}
elsif spec = candidate_text_spec
  candidate_ids, candidate_texts = spec
  candidate_specs << {nil, candidate_ids, candidate_texts}
else
  candidate_specs << {nil, parse_candidate_ids(candidate_ids_arg, canvas_token), nil}
end
candidate_specs.each do |_, candidate_ids, _|
  candidate_ids.each do |candidate_id|
    raise "--candidate-ids contains out-of-range id #{candidate_id}" if candidate_id < 0 || candidate_id >= hp.vocab_size
  end
end

result_rows = [] of Array(Tuple(String, String))
baseline_cache_ms = nil.as(Float64?)
baseline_loop_ms = nil.as(Float64?)
baseline_candidate_tokens_per_ms = nil.as(Float64?)
prompt_sets.each_with_index do |tokens, prompt_set_index|
  canvas_sets.each_with_index do |canvas_tokens, canvas_set_index|
    canvas_rows = ML::GGUF::DiffusionGemmaCPU.canvas_rows_from_tokens(weights, canvas_tokens)
    canvas_routes = nil.as(Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?)
    if single_route
      canvas_route_rows = canvas_tokens.map_with_index do |_, i|
        row = canvas_rows[i * hp.n_embd, hp.n_embd]
        ML::GGUF::DiffusionGemmaCPU.route_experts(weights, 0, row)[0, 1]
      end
      canvas_routes = [canvas_route_rows]
    end

    sample_us = ML::GGUF::DiffusionGemmaCPU.sample_u_steps(seed, steps, canvas_tokens.size)
    prompt_rows = [] of Float32
    tokens.each do |token_id|
      prompt_rows.concat(ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(weights, token_id))
    end
    mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: tokens.size, canvas_len: canvas_tokens.size, sliding_window: hp.sliding_window)

    prompt_routes = nil.as(Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?)
    if single_route
      prompt_route_rows = tokens.map_with_index do |_, i|
        row = prompt_rows[i * hp.n_embd, hp.n_embd]
        ML::GGUF::DiffusionGemmaCPU.route_experts(weights, 0, row)[0, 1]
      end
      prompt_routes = [prompt_route_rows]
    end

    cache_t0 = Time.instant
    prompt_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
      weights,
      prompt_rows,
      mask,
      max_layers: max_layers,
      routes_by_layer_by_prompt_row: prompt_routes,
    )
    cache_ms = (Time.instant - cache_t0).total_milliseconds
    baseline_cache_ms ||= cache_ms
    prompt_cache_ms_ratio_vs_first = baseline_cache_ms.not_nil! > 0.0 ? cache_ms / baseline_cache_ms.not_nil! : 0.0
    prompt_cache_tokens_per_ms = cache_ms > 0.0 ? tokens.size.to_f64 / cache_ms : 0.0

    candidate_specs.each_with_index do |candidate_spec, candidate_set_index|
      generated_count, candidate_ids, candidate_texts = candidate_spec
      candidate_rows = generated_count ? ML::GGUF::DiffusionGemmaCPU.generated_candidate_rows(canvas_tokens, generated_count.not_nil!, hp.vocab_size) : canvas_tokens.map { candidate_ids.dup }
      loop_samples = [] of Float64
      loop = nil.as(ML::GGUF::DiffusionGemmaCPU::BoundedDenoiseLoopResult?)
      (warmups + repeats).times do |run_index|
        loop_t0 = Time.instant
        loop = if adaptive
                 ML::GGUF::DiffusionGemmaCPU.decode_canvas_adaptive_bounded_loop(
                   weights,
                   canvas_tokens,
                   canvas_rows,
                   mask,
                   prompt_cache,
                   candidate_rows,
                   entropy_bound: entropy_bound,
                   stability_threshold: stability_threshold,
                   max_steps: steps,
                   proposal_top_k: proposal_top_k,
                   max_layers: max_layers,
                   sample_us_by_step_by_canvas_row: sample_us,
                   routes_by_layer_by_canvas_row: canvas_routes,
                 )
               else
                 candidate_steps = Array(Array(Array(Int32))).new(steps) { candidate_rows.map(&.dup) }
                 ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_loop(
                   weights,
                   canvas_tokens,
                   canvas_rows,
                   mask,
                   prompt_cache,
                   candidate_steps,
                   entropy_bound: entropy_bound,
                   stability_threshold: stability_threshold,
                   max_layers: max_layers,
                   sample_us_by_step_by_canvas_row: sample_us,
                   routes_by_layer_by_canvas_row: canvas_routes,
                 )
               end
        elapsed_ms = (Time.instant - loop_t0).total_milliseconds
        loop_samples << elapsed_ms if run_index >= warmups
      end
      loop = loop.not_nil!
      summary = loop.summary
      loop_ms_min = loop_samples.min
      loop_ms_median = median(loop_samples)
      loop_ms_max = loop_samples.max
      loop_candidate_tokens_per_ms = loop_ms_median > 0.0 ? summary.total_candidate_tokens.to_f64 / loop_ms_median : 0.0
      loop_predictions_per_ms = loop_ms_median > 0.0 ? summary.prediction_count.to_f64 / loop_ms_median : 0.0
      last_predictions = loop.updates.last.predictions
      baseline_loop_ms ||= loop_ms_median
      baseline_candidate_tokens_per_ms ||= loop_candidate_tokens_per_ms
      loop_ms_ratio_vs_first = baseline_loop_ms.not_nil! > 0.0 ? loop_ms_median / baseline_loop_ms.not_nil! : 0.0
      candidate_tokens_per_ms_ratio_vs_first = baseline_candidate_tokens_per_ms.not_nil! > 0.0 ? loop_candidate_tokens_per_ms / baseline_candidate_tokens_per_ms.not_nil! : 0.0

      result_rows << [
        {"status", "ok"},
        {"model", model},
        {"mode", adaptive ? "adaptive" : "fixed"},
        {"warmups", warmups.to_s},
        {"repeats", repeats.to_s},
        {"max_layers", max_layers.to_s},
        {"steps_budget", steps.to_s},
        {"steps_run", summary.steps_run.to_s},
        {"converged", summary.converged.to_s},
        {"stop_reason", summary.stop_reason},
        {"prompt_set_index", prompt_set_index.to_s},
        {"prompt_source", prompt_source},
        {"prompt_text_bytes", prompt_text.try(&.bytesize).try(&.to_s) || "0"},
        {"prompt_token", tokens[0].to_s},
        {"prompt_len", tokens.size.to_s},
        {"prompt_tokens", tokens.join(",")},
        {"canvas_set_index", canvas_set_index.to_s},
        {"canvas_source", canvas_source},
        {"canvas_text_bytes", canvas_text.try(&.bytesize).try(&.to_s) || "0"},
        {"canvas_len", canvas_tokens.size.to_s},
        {"initial_canvas_token", canvas_tokens[0].to_s},
        {"initial_canvas_tokens", canvas_tokens.join(",")},
        {"initial_canvas_text", canvas_decoder ? output_safe_text(canvas_decoder.not_nil!.decode(canvas_tokens)) : ""},
        {"final_canvas_token", loop.final_canvas_tokens[0].to_s},
        {"final_canvas_tokens", loop.final_canvas_tokens.join(",")},
        {"final_canvas_text", canvas_decoder ? output_safe_text(canvas_decoder.not_nil!.decode(loop.final_canvas_tokens)) : ""},
        {"candidate_set_index", candidate_set_index.to_s},
        {"candidate_count", candidate_rows.first.size.to_s},
        {"candidate_ids", candidate_rows.first.join(",")},
        {"candidate_texts", candidate_texts ? candidate_texts.not_nil!.map { |text| output_safe_list_text(text) }.join("|") : ""},
        {"candidate_rows", candidate_rows.map { |row| row.join(",") }.join("|")},
        {"candidate_row_texts", canvas_decoder ? decode_candidate_rows(canvas_decoder.not_nil!, candidate_rows) : ""},
        {"prediction_count", summary.prediction_count.to_s},
        {"accepted_count", summary.accepted_count.to_s},
        {"acceptance_rate", summary.acceptance_rate.to_s},
        {"total_candidate_tokens", summary.total_candidate_tokens.to_s},
        {"max_candidate_tokens", summary.max_candidate_tokens.to_s},
        {"mean_candidate_tokens", summary.mean_candidate_tokens.to_s},
        {"mean_entropy", summary.mean_entropy.to_s},
        {"last_argmax_tokens", format_prediction_i32s(last_predictions, &.argmax_token_id)},
        {"last_sampled_tokens", format_prediction_i32s(last_predictions, &.sampled_token_id)},
        {"last_argmax_probabilities", format_argmax_probabilities(last_predictions)},
        {"last_entropies", last_predictions.map { |prediction| format_f32(prediction.entropy) }.join(",")},
        {"last_candidate_probability_rows", format_prediction_f32_rows(last_predictions, &.probabilities)},
        {"load_ms", load_ms.round(3).to_s},
        {"prompt_cache_ms", cache_ms.round(3).to_s},
        {"prompt_cache_ms_ratio_vs_first", prompt_cache_ms_ratio_vs_first.round(6).to_s},
        {"prompt_cache_tokens_per_ms", prompt_cache_tokens_per_ms.round(6).to_s},
        {"loop_ms", loop_ms_median.round(3).to_s},
        {"loop_ms_min", loop_ms_min.round(3).to_s},
        {"loop_ms_median", loop_ms_median.round(3).to_s},
        {"loop_ms_max", loop_ms_max.round(3).to_s},
        {"loop_ms_samples", loop_samples.map { |v| v.round(3) }.join(",")},
        {"loop_candidate_tokens_per_ms", loop_candidate_tokens_per_ms.round(6).to_s},
        {"loop_predictions_per_ms", loop_predictions_per_ms.round(6).to_s},
        {"loop_ms_ratio_vs_first", loop_ms_ratio_vs_first.round(6).to_s},
        {"candidate_tokens_per_ms_ratio_vs_first", candidate_tokens_per_ms_ratio_vs_first.round(6).to_s},
      ]
    end
  end
end

case format
when "keyvalue"
  puts "diffusion_gemma_sparse_loop_smoke_result status=ok"
  result_rows[0].each do |key, value|
    next if key == "status"
    puts "#{key}=#{value}"
  end
when "tsv"
  puts result_rows[0].map(&.[0]).join('\t')
  result_rows.each do |rows|
    puts rows.map(&.[1]).join('\t')
  end
else
  raise "unreachable output format: #{format}"
end
