require "option_parser"
require "../src/ml/gguf/diffusion_gemma_cpu"

DEFAULT_MODEL = "#{ENV["HOME"]}/.cache/lm-studio/models/unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"

model = ENV["DIFFUSION_GEMMA_MODEL"]? || DEFAULT_MODEL
prompt_token = 1
canvas_token = 0
candidate_ids_arg = nil.as(String?)
candidate_count = nil.as(Int32?)
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

OptionParser.parse do |p|
  p.banner = "Usage: diffusion_gemma_sparse_loop_smoke [options]"
  p.on("--model PATH", "DiffusionGemma GGUF path") { |v| model = v }
  p.on("--prompt-token ID", "Prompt token id (default: 1)") { |v| prompt_token = v.to_i }
  p.on("--canvas-token ID", "Initial canvas token id (default: 0)") { |v| canvas_token = v.to_i }
  p.on("--candidate-ids CSV", "Sparse candidate token ids for the canvas row (default: canvas token)") { |v| candidate_ids_arg = v }
  p.on("--candidate-count N", "Generate N sparse candidate ids starting at the canvas token") { |v| candidate_count = v.to_i }
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
  p.on("-h", "--help", "Show help") do
    puts p
    exit
  end
end

def parse_candidate_ids(raw : String?, default_token : Int32) : Array(Int32)
  text = raw || default_token.to_s
  ids = text.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
  raise "--candidate-ids must contain at least one id" if ids.empty?
  ids.sort.uniq
end

def generated_candidate_ids(default_token : Int32, count : Int32, vocab_size : Int32) : Array(Int32)
  raise "--candidate-count must be positive" unless count > 0
  raise "--candidate-count exceeds vocab size" if count > vocab_size
  Array(Int32).new(count) { |i| (default_token + i) % vocab_size }.sort
end

def median(values : Array(Float64)) : Float64
  raise "median requires at least one value" if values.empty?
  sorted = values.sort
  sorted[sorted.size // 2]
end

raise "model not found: #{model}" unless File.exists?(model)
raise "--max-layers must be positive" unless max_layers > 0
raise "--steps must be positive" unless steps > 0
raise "--proposal-top-k must be positive" unless proposal_top_k > 0
raise "--stability-threshold must be positive" unless stability_threshold > 0
raise "--entropy-bound must be finite and non-negative" unless entropy_bound.finite? && entropy_bound >= 0.0_f32
raise "--format must be keyvalue or tsv" unless {"keyvalue", "tsv"}.includes?(format)
raise "--repeats must be positive" unless repeats > 0
raise "--candidate-ids and --candidate-count are mutually exclusive" if candidate_ids_arg && candidate_count
raise "single-route smoke currently supports --max-layers 1; pass --full-routes for deeper smoke" if single_route && max_layers != 1

load_t0 = Time.instant
weights = ML::GGUF::DiffusionGemmaWeights.from_gguf(model)
load_ms = (Time.instant - load_t0).total_milliseconds
hp = weights.hparams

raise "--prompt-token out of range" if prompt_token < 0 || prompt_token >= hp.vocab_size
raise "--canvas-token out of range" if canvas_token < 0 || canvas_token >= hp.vocab_size
candidate_ids = if count = candidate_count
                  generated_candidate_ids(canvas_token, count, hp.vocab_size)
                else
                  parse_candidate_ids(candidate_ids_arg, canvas_token)
                end
candidate_ids.each do |candidate_id|
  raise "--candidate-ids contains out-of-range id #{candidate_id}" if candidate_id < 0 || candidate_id >= hp.vocab_size
end

prompt_row = ML::GGUF::DiffusionGemmaCPU.scaled_embedding_lookup(weights, prompt_token)
canvas_row = ML::GGUF::DiffusionGemmaCPU.zero_sc_canvas_embedding(weights, canvas_token)
mask = ML::GGUF::DiffusionGemmaAttentionMask.new(prompt_len: 1, canvas_len: 1, sliding_window: hp.sliding_window)

prompt_routes = nil.as(Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?)
canvas_routes = nil.as(Array(Array(Array(ML::GGUF::DiffusionGemmaCPU::ExpertRoute)))?)
if single_route
  prompt_route = ML::GGUF::DiffusionGemmaCPU.route_experts(weights, 0, prompt_row)[0, 1]
  canvas_route = ML::GGUF::DiffusionGemmaCPU.route_experts(weights, 0, canvas_row)[0, 1]
  prompt_routes = [[prompt_route]]
  canvas_routes = [[canvas_route]]
end

cache_t0 = Time.instant
prompt_cache = ML::GGUF::DiffusionGemmaCPU.build_prompt_layer_cache(
  weights,
  prompt_row,
  mask,
  max_layers: max_layers,
  routes_by_layer_by_prompt_row: prompt_routes,
)
cache_ms = (Time.instant - cache_t0).total_milliseconds

sample_us = ML::GGUF::DiffusionGemmaCPU.sample_u_steps(seed, steps, 1)
loop_samples = [] of Float64
loop = nil.as(ML::GGUF::DiffusionGemmaCPU::BoundedDenoiseLoopResult?)
repeats.times do
  loop_t0 = Time.instant
  loop = if adaptive
           ML::GGUF::DiffusionGemmaCPU.decode_canvas_adaptive_bounded_loop(
             weights,
             [canvas_token],
             canvas_row,
             mask,
             prompt_cache,
             [candidate_ids],
             entropy_bound: entropy_bound,
             stability_threshold: stability_threshold,
             max_steps: steps,
             proposal_top_k: proposal_top_k,
             max_layers: max_layers,
             sample_us_by_step_by_canvas_row: sample_us,
             routes_by_layer_by_canvas_row: canvas_routes,
           )
         else
           candidate_steps = Array(Array(Array(Int32))).new(steps) { [candidate_ids.dup] }
           ML::GGUF::DiffusionGemmaCPU.decode_canvas_bounded_loop(
             weights,
             [canvas_token],
             canvas_row,
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
  loop_samples << (Time.instant - loop_t0).total_milliseconds
end
loop = loop.not_nil!
summary = loop.summary
loop_ms_min = loop_samples.min
loop_ms_median = median(loop_samples)
loop_ms_max = loop_samples.max

rows = {
  {"status", "ok"},
  {"model", model},
  {"mode", adaptive ? "adaptive" : "fixed"},
  {"repeats", repeats.to_s},
  {"max_layers", max_layers.to_s},
  {"steps_budget", steps.to_s},
  {"steps_run", summary.steps_run.to_s},
  {"converged", summary.converged.to_s},
  {"stop_reason", summary.stop_reason},
  {"prompt_token", prompt_token.to_s},
  {"initial_canvas_token", canvas_token.to_s},
  {"final_canvas_token", loop.final_canvas_tokens[0].to_s},
  {"candidate_ids", candidate_ids.join(",")},
  {"prediction_count", summary.prediction_count.to_s},
  {"accepted_count", summary.accepted_count.to_s},
  {"acceptance_rate", summary.acceptance_rate.to_s},
  {"total_candidate_tokens", summary.total_candidate_tokens.to_s},
  {"max_candidate_tokens", summary.max_candidate_tokens.to_s},
  {"mean_candidate_tokens", summary.mean_candidate_tokens.to_s},
  {"mean_entropy", summary.mean_entropy.to_s},
  {"load_ms", load_ms.round(3).to_s},
  {"prompt_cache_ms", cache_ms.round(3).to_s},
  {"loop_ms", loop_ms_median.round(3).to_s},
  {"loop_ms_min", loop_ms_min.round(3).to_s},
  {"loop_ms_median", loop_ms_median.round(3).to_s},
  {"loop_ms_max", loop_ms_max.round(3).to_s},
  {"loop_ms_samples", loop_samples.map { |v| v.round(3) }.join(",")},
}

case format
when "keyvalue"
  puts "diffusion_gemma_sparse_loop_smoke_result status=ok"
  rows.each do |key, value|
    next if key == "status"
    puts "#{key}=#{value}"
  end
when "tsv"
  puts rows.map(&.[0]).join('\t')
  puts rows.map(&.[1]).join('\t')
else
  raise "unreachable output format: #{format}"
end
