# Greedy speculative-decode acceptance probe for Qwen35 target/draft pairs.
#
# The chunk verifiers process each gamma-sized target candidate span through the
# chunked prefill body and then emit one top1 per row. This is still not the
# final fully batched verifier, but it measures exact speed steps after a purely
# serial target verifier.

require "../src/ml/gguf/reader"
require "../src/ml/gguf/ngram_draft"
require "../src/ml/gguf/qwen35_cpu"
require "../src/ml/gguf/qwen35_tokenizer"
require "../src/ml/gguf/qwen35_weights"
require "json"
require "option_parser"
require "set"

DEFAULT_TARGET    = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
DEFAULT_DRAFT     = "#{ENV["HOME"]}/.cache/lm-studio/models/lmstudio-community/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
DEFAULT_TOKENIZER = "#{ENV["HOME"]}/SrcArchives/AI/llama.cpp/build/bin/llama-tokenize"

target_path = ENV["QWEN35_TARGET"]? || DEFAULT_TARGET
draft_path = ENV["QWEN35_DRAFT"]? || DEFAULT_DRAFT
tokenizer_bin = ENV["LLAMA_TOKENIZE_BIN"]? || DEFAULT_TOKENIZER
prompt = "The capital of France is"
n_gen = 32
gamma = 4
adaptive_gamma = ENV["QWEN35_SPEC_ADAPTIVE"]? != "0"
adaptive_regrow = ENV["QWEN35_SPEC_ADAPTIVE_REGROW"]? == "1"
adaptive_full_accept_streak = (ENV["QWEN35_SPEC_FULL_ACCEPT_STREAK"]? || "2").to_i
adaptive_fast_regrow_min_gamma = (ENV["QWEN35_SPEC_FAST_REGROW_MIN_GAMMA"]? || "8").to_i
adaptive_bootstrap_gamma = (ENV["QWEN35_SPEC_BOOTSTRAP_GAMMA"]? || "0").to_i
adaptive_bootstrap_streak = (ENV["QWEN35_SPEC_BOOTSTRAP_STREAK"]? || "1").to_i
max_gamma = (ENV["QWEN35_SPEC_MAX_GAMMA"]? || "32").to_i
verify_mode = ENV["QWEN35_SPEC_VERIFY"]? || "chunk-inplace"
stage_gate = (ENV["QWEN35_SPEC_STAGE_GATE"]? || gamma.to_s).to_i
trace = ENV["QWEN35_SPEC_TRACE"]? == "1"
early_reject_enabled = ENV["QWEN35_SPEC_EARLY_REJECT_OFF"]? != "1"
single_accept_fast_enabled = ENV["QWEN35_SPEC_SINGLE_FAST_OFF"]? != "1"
plain_fallback_enabled = ENV["QWEN35_SPEC_PLAIN_FALLBACK_OFF"]? != "1"
plain_fallback_gamma = (ENV["QWEN35_SPEC_PLAIN_FALLBACK_GAMMA"]? || "2").to_i
skip_draft_before_fallback_enabled = ENV["QWEN35_SPEC_SKIP_DRAFT_BEFORE_FALLBACK_OFF"]? != "1"
skip_draft_backup_before_fallback_enabled = ENV["QWEN35_SPEC_SKIP_DRAFT_BACKUP_BEFORE_FALLBACK_OFF"]? != "1"
ngram_enabled = ENV["QWEN35_SPEC_NGRAM"]? == "1"
ngram_gamma = (ENV["QWEN35_SPEC_NGRAM_GAMMA"]? || "32").to_i
ngram_min = (ENV["QWEN35_SPEC_NGRAM_MIN"]? || "6").to_i
ngram_max = (ENV["QWEN35_SPEC_NGRAM_MAX"]? || "8").to_i
ngram_stage_min_env = ENV["QWEN35_SPEC_NGRAM_STAGE_MIN"]?
ngram_stage_min = (ngram_stage_min_env || (ngram_gamma + 1).to_s).to_i
ngram_stage_min_explicit = !ngram_stage_min_env.nil?
ngram_risk_min_size = (ENV["QWEN35_SPEC_NGRAM_RISK_MIN_SIZE"]? || "16").to_i
ngram_risk_gate = ENV["QWEN35_SPEC_NGRAM_RISK_GATE"]? == "1"
ngram_recursive = ENV["QWEN35_SPEC_NGRAM_RECURSIVE_OFF"]? != "1"
ngram_disable_after_reject = ENV["QWEN35_SPEC_NGRAM_DISABLE_AFTER_REJECT_OFF"]? != "1"
prepare_state_metal = ENV["QWEN35_PREPARE_STATE_OFF"]? != "1"
warm_verifier = ENV["QWEN35_SPEC_WARM_VERIFIER_OFF"]? != "1"
allow_guarded_verifier = ENV["QWEN35_SPEC_ALLOW_GUARDED_VERIFIER"]? == "1"
dump_cycles_path = ENV["QWEN35_SPEC_DUMP_CYCLES"]?
dump_cycle_token_ids = ENV["QWEN35_SPEC_DUMP_TOKEN_IDS"]? == "1"
router_model_path = ENV["QWEN35_SPEC_ROUTER_MODEL"]?

OptionParser.parse(ARGV) do |parser|
  parser.banner = "Usage: qwen35_speculative_accept [--target PATH] [--draft PATH] [--gamma N] [--max-gamma N] [--bootstrap-gamma N] [--adaptive|--no-adaptive] [--tokens N] [--verify serial|chunk|chunk-inplace|hybrid|staged] [--ngram] [prompt]"
  parser.on("--target PATH", "Target GGUF path (default: Qwen3.5 9B Q4_K_M)") { |path| target_path = path }
  parser.on("--draft PATH", "Draft GGUF path (default: Qwen3.5 0.8B Q8_0)") { |path| draft_path = path }
  parser.on("--tokenizer-bin PATH", "llama.cpp tokenizer helper path") { |path| tokenizer_bin = path }
  parser.on("--gamma N", "Draft candidates per cycle") { |value| gamma = value.to_i }
  parser.on("--max-gamma N", "Maximum adaptive draft candidates per cycle (default: 32)") { |value| max_gamma = value.to_i }
  parser.on("--bootstrap-gamma N", "After enough fully accepted initial chunks, jump to this gamma (default: env QWEN35_SPEC_BOOTSTRAP_GAMMA or 0/off)") { |value| adaptive_bootstrap_gamma = value.to_i }
  parser.on("--bootstrap-streak N", "Full accepts at initial gamma before --bootstrap-gamma jump (default: env QWEN35_SPEC_BOOTSTRAP_STREAK or 1)") { |value| adaptive_bootstrap_streak = value.to_i }
  parser.on("--adaptive", "Adapt gamma: double after fully accepted cycles, halve after rejection (default)") { adaptive_gamma = true }
  parser.on("--no-adaptive", "Use fixed --gamma for every speculative cycle") { adaptive_gamma = false }
  parser.on("--tokens N", "Generated tokens to compare") { |value| n_gen = value.to_i }
  parser.on("--verify MODE", "Target verifier: serial, chunk, chunk-inplace, hybrid, or staged (default: chunk-inplace)") { |value| verify_mode = value }
  parser.on("--stage-gate N", "For --verify staged, verify this many candidates before drafting/verifying the rest") { |value| stage_gate = value.to_i }
  parser.on("--ngram", "Try exact n-gram/cache draft chunks before the neural draft") { ngram_enabled = true }
  parser.on("--allow-guarded-verifier", "Research only: allow guarded full-row verifier inside speculative target chunks") { allow_guarded_verifier = true }
  parser.on("--ngram-gamma N", "Maximum n-gram candidates per chunk (default: env QWEN35_SPEC_NGRAM_GAMMA or 32)") { |value| ngram_gamma = value.to_i }
  parser.on("--ngram-min N", "Minimum repeated suffix length before n-gram drafting (default: env QWEN35_SPEC_NGRAM_MIN or 6)") { |value| ngram_min = value.to_i }
  parser.on("--ngram-max N", "Maximum repeated suffix length to search (default: env QWEN35_SPEC_NGRAM_MAX or 8)") { |value| ngram_max = value.to_i }
  parser.on("--ngram-stage-min N", "For --verify staged, only split n-gram chunks with at least this many candidates (default: ngram_gamma + 1)") do |value|
    ngram_stage_min = value.to_i
    ngram_stage_min_explicit = true
  end
  parser.on("--ngram-risk-min-size N", "Minimum candidate size for the n-gram risk gate (default: 16)") { |value| ngram_risk_min_size = value.to_i }
  parser.on("--ngram-risk-gate", "Research: skip n-gram chunks whose candidate-token shape matches known bad repeat tails") { ngram_risk_gate = true }
  parser.on("--no-recursive-ngram", "Do not recursively extend n-gram candidates through scratch history") { ngram_recursive = false }
  parser.on("--keep-ngram-after-reject", "Keep trying n-gram draft chunks after a rejected n-gram chunk") { ngram_disable_after_reject = false }
  parser.on("--no-warm-verifier", "Do not warm the target chunk-verifier route before decode timing") { warm_verifier = false }
  parser.on("--trace", "Print per-cycle verifier decisions") { trace = true }
  parser.on("--dump-cycles PATH", "Write per-cycle speculative policy/timing records as JSONL") { |path| dump_cycles_path = path }
  parser.on("--dump-cycle-token-ids", "Include raw token ids in --dump-cycles records; default only writes stable hashes") { dump_cycle_token_ids = true }
  parser.on("--router-model PATH", "Research: logistic router JSON used to gate n-gram chunks before verification") { |path| router_model_path = path }
  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
  parser.unknown_args do |before_dash, _after_dash|
    prompt = before_dash.join(" ") unless before_dash.empty?
  end
end

ngram_stage_min = ngram_gamma + 1 unless ngram_stage_min_explicit

raise ArgumentError.new("--gamma must be positive") unless gamma > 0
stage_gate = gamma if stage_gate <= 0
raise ArgumentError.new("QWEN35_SPEC_FULL_ACCEPT_STREAK must be positive") unless adaptive_full_accept_streak > 0
raise ArgumentError.new("QWEN35_SPEC_FAST_REGROW_MIN_GAMMA must be non-negative") unless adaptive_fast_regrow_min_gamma >= 0
raise ArgumentError.new("QWEN35_SPEC_BOOTSTRAP_GAMMA must be non-negative") unless adaptive_bootstrap_gamma >= 0
raise ArgumentError.new("QWEN35_SPEC_BOOTSTRAP_STREAK must be positive") unless adaptive_bootstrap_streak > 0
raise ArgumentError.new("--max-gamma must be positive") unless max_gamma > 0
max_gamma = Math.max(max_gamma, gamma)
raise ArgumentError.new("QWEN35_SPEC_PLAIN_FALLBACK_GAMMA must be positive") unless plain_fallback_gamma > 0
raise ArgumentError.new("--tokens must be positive") unless n_gen > 0
raise ArgumentError.new("--verify must be serial, chunk, chunk-inplace, hybrid, or staged") unless {"serial", "chunk", "chunk-inplace", "hybrid", "staged"}.includes?(verify_mode)
raise ArgumentError.new("QWEN35_SPEC_NGRAM_GAMMA must be positive") unless ngram_gamma > 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_MIN must be positive") unless ngram_min > 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_MAX must be >= QWEN35_SPEC_NGRAM_MIN") unless ngram_max >= ngram_min
raise ArgumentError.new("QWEN35_SPEC_NGRAM_STAGE_MIN must be positive") unless ngram_stage_min > 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_RISK_MIN_SIZE must be positive") unless ngram_risk_min_size > 0
raise ArgumentError.new("router model not found: #{router_model_path}") if router_model_path && !File.file?(router_model_path.not_nil!)

def load_tokenizer(model_path : String, tokenizer_bin : String) : ML::GGUF::Qwen35Tokenizer
  g = ML::GGUF::GGUFFile.new(model_path)
  ML::GGUF::Qwen35Tokenizer.from_gguf(g, model_path, tokenizer_bin)
ensure
  g.try(&.close)
end

def prefill_next(weights : ML::GGUF::Qwen35Weights,
                 token_ids : Array(Int32),
                 state : ML::GGUF::Qwen35CPU::State) : Int32
  top, _logit = ML::GGUF::Qwen35CPU.prefill_tokens_top1(weights, token_ids, 0, state)
  top.to_i32
end

def advance_next(weights : ML::GGUF::Qwen35Weights,
                 token_id : Int32,
                 pos : Int32,
                 state : ML::GGUF::Qwen35CPU::State) : Int32
  top, _logit = ML::GGUF::Qwen35CPU.forward_top1(weights, token_id, pos, state)
  top.to_i32
end

def greedy_sequence(weights : ML::GGUF::Qwen35Weights,
                    prompt_ids : Array(Int32),
                    n_gen : Int32) : {Array(Int32), Float64, Float64}
  state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: prompt_ids.size + n_gen + 8)
  prefill0 = Time.instant
  next_id = prefill_next(weights, prompt_ids, state)
  prefill_ms = (Time.instant - prefill0).total_milliseconds
  pos = prompt_ids.size
  ids = [] of Int32
  decode0 = Time.instant
  n_gen.times do
    ids << next_id
    break if ids.size >= n_gen
    next_id = advance_next(weights, next_id, pos, state)
    pos += 1
  end
  decode_ms = (Time.instant - decode0).total_milliseconds
  {ids, decode_ms, prefill_ms}
end

def resync_draft!(weights : ML::GGUF::Qwen35Weights,
                  state : ML::GGUF::Qwen35CPU::State,
                  base : ML::GGUF::Qwen35CPU::State,
                  accepted_or_corrected : Array(Int32),
                  start_pos : Int32) : Int32
  state.copy_from!(base)
  next_id = -1
  accepted_or_corrected.each_with_index do |tok, i|
    next_id = advance_next(weights, tok, start_pos + i, state)
  end
  next_id
end

def with_guarded_full_rows_disabled(&)
  old_guard = ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"]?
  ENV.delete("QWEN35_HEAD_FULL_ROWS_GUARDED")
  yield
ensure
  if old_guard
    ENV["QWEN35_HEAD_FULL_ROWS_GUARDED"] = old_guard
  else
    ENV.delete("QWEN35_HEAD_FULL_ROWS_GUARDED")
  end
end

def target_prefill_top1s_exact(weights : ML::GGUF::Qwen35Weights,
                               token_ids : Array(Int32),
                               start_pos : Int32,
                               state : ML::GGUF::Qwen35CPU::State,
                               allow_guarded_verifier : Bool) : Array({Int32, Float32})
  if allow_guarded_verifier
    ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, token_ids, start_pos, state)
  else
    with_guarded_full_rows_disabled do
      ML::GGUF::Qwen35CPU.prefill_tokens_top1s(weights, token_ids, start_pos, state)
    end
  end
end

def target_prefill_top1s_for_future(weights : ML::GGUF::Qwen35Weights,
                                    token_ids : Array(Int32),
                                    start_pos : Int32,
                                    state : ML::GGUF::Qwen35CPU::State,
                                    allow_guarded_verifier : Bool,
                                    generated_before : Int32,
                                    n_gen : Int32) : Array({Int32, Float32})
  # The final generated token does not need its next-token logits. Skipping that
  # tail row keeps speculative timings aligned with the CLI decode path.
  remaining = Math.max(n_gen - generated_before, 0)
  verify_len = Math.min(token_ids.size, remaining)
  verify_len -= 1 if verify_len > 0 && generated_before + verify_len >= n_gen
  verify_ids = token_ids[0, verify_len]
  return [] of {Int32, Float32} if verify_ids.empty?
  target_prefill_top1s_exact(weights, verify_ids, start_pos, state, allow_guarded_verifier)
end

def fnv1a64_hex(bytes : Bytes) : String
  hash = 0xcbf29ce484222325_u64
  bytes.each do |b|
    hash = (hash ^ b.to_u64) &* 0x100000001b3_u64
  end
  hash.to_s(16)
end

def token_ids_hash(ids : Array(Int32)) : String
  bytes = Bytes.new(ids.size * 4)
  ids.each_with_index do |id, i|
    value = id.to_u32
    offset = i * 4
    bytes[offset] = (value & 0xff).to_u8
    bytes[offset + 1] = ((value >> 8) & 0xff).to_u8
    bytes[offset + 2] = ((value >> 16) & 0xff).to_u8
    bytes[offset + 3] = ((value >> 24) & 0xff).to_u8
  end
  fnv1a64_hex(bytes)
end

def ngram_match_len(history : Array(Int32), max_ngram : Int32, min_ngram : Int32) : Int32
  return 0 if history.empty?
  max_len = Math.min(max_ngram, history.size)
  max_len.downto(min_ngram) do |n|
    suffix_start = history.size - n
    i = history.size - n - 1
    while i >= 0
      matched = true
      n.times do |j|
        if history[i + j] != history[suffix_start + j]
          matched = false
          break
        end
      end
      return n if matched && i + n < history.size
      i -= 1
    end
  end
  0
end

class SpecRouterModel
  getter threshold : Float64
  getter feature_names : Array(String)
  getter weights : Array(Float64)
  getter path : String

  def initialize(@path : String, @threshold : Float64, @feature_names : Array(String), @weights : Array(Float64))
    raise ArgumentError.new("router feature/weight size mismatch") unless @feature_names.size == @weights.size
  end

  def self.load(path : String) : self
    rec = JSON.parse(File.read(path))
    kind = rec["kind"]?.try(&.as_s) || ""
    raise ArgumentError.new("unsupported router model kind: #{kind}") unless kind == "qwen35_spec_router_logistic"

    threshold = rec["threshold"].as_f
    feature_names = rec["feature_names"].as_a.map(&.as_s)
    weights = rec["weights"].as_a.map(&.as_f)
    new(path, threshold, feature_names, weights)
  end

  def score(features : Hash(String, Float64)) : Float64
    z = 0.0
    @feature_names.each_with_index do |name, i|
      z += @weights[i] * (features[name]? || 0.0)
    end
    sigmoid(z)
  end

  private def sigmoid(z : Float64) : Float64
    if z >= 0.0
      1.0 / (1.0 + Math.exp(-z))
    else
      ez = Math.exp(z)
      ez / (1.0 + ez)
    end
  end
end

def add_candidate_features(features : Hash(String, Float64), ids : Array(Int32))
  features["candidate_features_present"] = ids.empty? ? 0.0 : 1.0
  return if ids.empty?

  counts = Hash(Int32, Int32).new(0)
  ids.each { |id| counts[id] += 1 }
  features["candidate_unique_ratio"] = counts.size.to_f / ids.size
  features["candidate_pair_unique_ratio"] = ML::GGUF::NgramDraft.pair_unique_ratio(ids)

  entropy = 0.0
  counts.each_value do |count|
    p = count.to_f / ids.size
    entropy -= p * (Math.log(p) / Math.log(2.0))
  end
  max_entropy = ids.size > 1 ? Math.log(ids.size.to_f) / Math.log(2.0) : 1.0
  features["candidate_entropy_norm"] = max_entropy > 0.0 ? entropy / max_entropy : 0.0

  longest = 1
  run = 1
  1.upto(ids.size - 1) do |i|
    if ids[i] == ids[i - 1]
      run += 1
    else
      longest = Math.max(longest, run)
      run = 1
    end
  end
  longest = Math.max(longest, run)
  features["candidate_longest_run_ratio"] = longest.to_f / ids.size

  period = ML::GGUF::NgramDraft.exact_period(ids, 8)
  features["candidate_exact_period_over_8"] = period > 0 ? period.to_f / 8.0 : 0.0
  features["candidate_lag1_ratio"] = ML::GGUF::NgramDraft.lag_ratio(ids, 1)
  features["candidate_lag2_ratio"] = ML::GGUF::NgramDraft.lag_ratio(ids, 2)
  features["candidate_lag4_ratio"] = ML::GGUF::NgramDraft.lag_ratio(ids, 4)
  features["candidate_lag8_ratio"] = ML::GGUF::NgramDraft.lag_ratio(ids, 8)
end

def ngram_router_features(candidates : Array(Int32),
                          generated_before : Int32,
                          match_len : Int32,
                          ngram_max : Int32,
                          ngram_disabled_before : Bool,
                          verify_mode : String,
                          draft_model_id : String) : Hash(String, Float64)
  proposed = candidates.size
  features = Hash(String, Float64).new(0.0)
  features["bias"] = 1.0
  features["gamma_over_32"] = proposed.clamp(0, 64).to_f / 32.0
  features["proposed_over_32"] = proposed.clamp(0, 64).to_f / 32.0
  features["proposed_to_gamma_ratio"] = proposed > 0 ? 1.0 : 0.0
  features["generated_before_over_128"] = generated_before.clamp(0, 512).to_f / 128.0
  features["ngram_match_ratio"] = ngram_max > 0 ? match_len.clamp(0, ngram_max).to_f / ngram_max : 0.0
  features["ngram_disabled_before"] = ngram_disabled_before ? 1.0 : 0.0
  add_candidate_features(features, candidates)
  features["kind=ngram"] = 1.0
  features["verify=#{verify_mode}"] = 1.0
  features["draft=#{draft_model_id}"] = 1.0
  features
end

private class CycleDump
  include JSON::Serializable

  property prompt_hash : String
  property target_model : String
  property draft_model : String
  property kind : String
  property policy : String
  property verify_mode : String
  property position : Int32
  property generated_before : Int32
  property generated_count : Int32
  property gamma : Int32
  property proposed_count : Int32
  property accepted_count : Int32
  property reject_index : Int32
  property ngram_match_len : Int32
  property ngram_min : Int32
  property ngram_max : Int32
  property ngram_recursive : Bool
  property ngram_disabled_before : Bool
  property ngram_disabled_after : Bool
  property candidate_hash : String
  property candidates : Array(Int32)?
  property draft_ms : Float64
  property target_verify_ms : Float64
  property target_backup_ms : Float64
  property draft_backup_ms : Float64
  property draft_resync_ms : Float64
  property wall_ms : Float64
  property expected_gain_ms : Float64?

  def initialize(@prompt_hash : String,
                 @target_model : String,
                 @draft_model : String,
                 @kind : String,
                 @policy : String,
                 @verify_mode : String,
                 @position : Int32,
                 @generated_before : Int32,
                 @generated_count : Int32,
                 @gamma : Int32,
                 @proposed_count : Int32,
                 @accepted_count : Int32,
                 @reject_index : Int32,
                 @ngram_match_len : Int32,
                 @ngram_min : Int32,
                 @ngram_max : Int32,
                 @ngram_recursive : Bool,
                 @ngram_disabled_before : Bool,
                 @ngram_disabled_after : Bool,
                 @candidate_hash : String,
                 @candidates : Array(Int32)?,
                 @draft_ms : Float64,
                 @target_verify_ms : Float64,
                 @target_backup_ms : Float64,
                 @draft_backup_ms : Float64,
                 @draft_resync_ms : Float64,
                 @wall_ms : Float64,
                 @expected_gain_ms : Float64? = nil)
  end
end

puts "Loading tokenizer and models..."
t0 = Time.instant
tok = load_tokenizer(target_path, tokenizer_bin)
target = ML::GGUF::Qwen35Weights.from_gguf(target_path)
draft = ML::GGUF::Qwen35Weights.from_gguf(draft_path)
load_s = (Time.instant - t0).total_seconds

unless target.output.out_dim == draft.output.out_dim
  raise ArgumentError.new("target/draft vocab mismatch: #{target.output.out_dim} != #{draft.output.out_dim}")
end

prompt_ids = tok.encode(prompt)
raise ArgumentError.new("prompt encoded to no tokens") if prompt_ids.empty?
prompt_hash = fnv1a64_hex(prompt.to_slice)
target_model_id = File.basename(target_path)
draft_model_id = File.basename(draft_path)
router_model = router_model_path ? SpecRouterModel.load(router_model_path.not_nil!) : nil
cycle_dumps = [] of CycleDump

puts "Loaded in #{load_s.round(2)}s"
puts "target: layers=#{target.hparams.n_layer} dim=#{target.hparams.n_embd} vocab=#{target.output.out_dim}"
puts "draft:  layers=#{draft.hparams.n_layer} dim=#{draft.hparams.n_embd} vocab=#{draft.output.out_dim}"
puts "prompt tokens=#{prompt_ids.size} prompt_hash=#{prompt_hash} gamma=#{gamma} max_gamma=#{max_gamma} adaptive=#{adaptive_gamma} adaptive_regrow=#{adaptive_regrow} full_accept_streak=#{adaptive_full_accept_streak} fast_regrow_min_gamma=#{adaptive_fast_regrow_min_gamma} bootstrap_gamma=#{adaptive_bootstrap_gamma} bootstrap_streak=#{adaptive_bootstrap_streak} ngram=#{ngram_enabled} ngram_gamma=#{ngram_gamma} ngram_min=#{ngram_min} ngram_max=#{ngram_max} ngram_stage_min=#{ngram_stage_min} ngram_risk_gate=#{ngram_risk_gate} ngram_risk_min_size=#{ngram_risk_min_size} ngram_recursive=#{ngram_recursive} ngram_disable_after_reject=#{ngram_disable_after_reject} router_model=#{router_model_path || ""} early_reject=#{early_reject_enabled} single_fast=#{single_accept_fast_enabled} plain_fallback=#{plain_fallback_enabled} fallback_gamma=#{plain_fallback_gamma} skip_draft_before_fallback=#{skip_draft_before_fallback_enabled} skip_draft_backup_before_fallback=#{skip_draft_backup_before_fallback_enabled} prepare_state=#{prepare_state_metal} warm_verifier=#{warm_verifier} stage_gate=#{stage_gate} n_gen=#{n_gen} verify=#{verify_mode} allow_guarded_verifier=#{allow_guarded_verifier} dump_cycles=#{dump_cycles_path || ""} dump_token_ids=#{dump_cycle_token_ids}"

max_seq = prompt_ids.size + n_gen + Math.max(gamma, ngram_gamma) + 8
target_state = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: max_seq)
draft_state = ML::GGUF::Qwen35CPU::State.new(draft.hparams, max_seq: max_seq)
target_backup_state = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: max_seq)
draft_cycle_base = ML::GGUF::Qwen35CPU::State.new(draft.hparams, max_seq: max_seq)
if prepare_state_metal
  ML::GGUF::Qwen35CPU.prepare_state_metal!(target_state, target.hparams)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(draft_state, draft.hparams)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(target_backup_state, target.hparams)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(draft_cycle_base, draft.hparams)
end

target_next = prefill_next(target, prompt_ids, target_state)
draft_next = prefill_next(draft, prompt_ids, draft_state)
verifier_warmup_ms = 0.0
if warm_verifier && n_gen > 1
  warm_len = Math.min(gamma, n_gen)
  if warm_len > 1
    warm_state = target_state.fork
    warm_candidates = Array.new(warm_len) { target_next }
    tw0 = Time.instant
    target_prefill_top1s_exact(target, warm_candidates, prompt_ids.size, warm_state, allow_guarded_verifier)
    verifier_warmup_ms = (Time.instant - tw0).total_milliseconds
  end
end

profile_spec_region = ENV["QWEN35_SPEC_PROFILE"]? == "1"
if profile_spec_region
  ML::GGUF::Qwen35Metal::Profile.reset
  ML::GGUF::Qwen35Metal::Profile.enable!
end

generated_ids = [] of Int32
history = prompt_ids.dup
pending_draft_tokens = [] of Int32
pending_draft_start_pos = 0
ngram_disabled = false
pos = prompt_ids.size
accepted = 0
proposed = 0
cycles = 0
ngram_cycles = 0
ngram_accepted = 0
ngram_proposed = 0
ngram_router_checks = 0
ngram_router_skips = 0
ngram_router_score_sum = 0.0
target_verify_ms = 0.0
draft_ms = 0.0
target_backup_ms = 0.0
draft_backup_ms = 0.0
draft_resync_ms = 0.0
current_gamma = gamma
full_accept_streak = 0
adaptive_growth_allowed = true
gamma_sum = 0
gamma_max_seen = 0
early_rejects = 0
single_accept_fast = 0
plain_fallback_tokens = 0
draft_skips_before_fallback = 0
draft_backup_skips = 0

wall0 = Time.instant
while generated_ids.size < n_gen
  history_size_before = generated_ids.size

  if ngram_enabled && !ngram_disabled
    ngram_candidates = ML::GGUF::NgramDraft.candidates(
      history,
      Math.min(ngram_gamma, n_gen - generated_ids.size),
      ngram_max,
      ngram_min,
      recursive: ngram_recursive)
    match_len = ngram_match_len(history, ngram_max, ngram_min)
    if ngram_risk_gate && ML::GGUF::NgramDraft.risky_candidate_shape?(ngram_candidates, ngram_risk_min_size, match_len)
      ngram_disabled = true
      ngram_candidates = [] of Int32
    end
    if router_model && !ngram_candidates.empty?
      score = router_model.not_nil!.score(ngram_router_features(
        ngram_candidates, generated_ids.size, match_len, ngram_max, ngram_disabled, verify_mode, draft_model_id))
      ngram_router_checks += 1
      ngram_router_score_sum += score
      if score < router_model.not_nil!.threshold
        ngram_router_skips += 1
        ngram_candidates = [] of Int32
      end
    end

    unless ngram_candidates.empty?
      cycle_wall0 = Time.instant
      cycle_draft0 = draft_ms
      cycle_target_verify0 = target_verify_ms
      cycle_target_backup0 = target_backup_ms
      cycle_draft_backup0 = draft_backup_ms
      cycle_draft_resync0 = draft_resync_ms
      ngram_disabled_before = ngram_disabled
      match_len = ngram_match_len(history, ngram_max, ngram_min)
      ngram_cycles += 1
      ngram_proposed += ngram_candidates.size
      proposed += ngram_candidates.size
      cycle_start_pos = pos
      correction_or_accepted = [] of Int32

      rejected = false
      accepted_in_cycle = 0
      reject_index = -1
      ngram_offset = 0
      stage_ngram = verify_mode == "staged" && ngram_candidates.size >= ngram_stage_min
      while ngram_offset < ngram_candidates.size
        remaining = ngram_candidates.size - ngram_offset
        stage_len = stage_ngram ? Math.min(stage_gate, remaining) : remaining
        stage_len = Math.min(stage_len, n_gen - generated_ids.size)
        break if stage_len <= 0
        stage_candidates = ngram_candidates[ngram_offset, stage_len]
        stage_start_pos = cycle_start_pos + ngram_offset
        stage_correction_or_accepted = [] of Int32

        tb0 = Time.instant
        target_backup_state.copy_from!(target_state)
        target_backup_ms += (Time.instant - tb0).total_milliseconds
        tv0 = Time.instant
        target_nexts = target_prefill_top1s_for_future(target, stage_candidates, stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size, n_gen)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "ngram_cycle=#{ngram_cycles} stage_offset=#{ngram_offset} pos=#{stage_start_pos} expected0=#{target_next} candidates=#{stage_candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        stage_candidates.each_with_index do |cand, i|
          break if generated_ids.size >= n_gen
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            stage_correction_or_accepted << cand
            accepted += 1
            accepted_in_cycle += 1
            ngram_accepted += 1
            expected = target_nexts[i][0] if i < target_nexts.size
          else
            generated_ids << expected
            correction_or_accepted << expected
            stage_correction_or_accepted << expected
            rejected = true
            reject_index = ngram_offset + i
            break
          end
        end

        if rejected
          ngram_disabled = true if ngram_disable_after_reject
          if generated_ids.size < n_gen
            target_state.copy_from!(target_backup_state)
            tv1 = Time.instant
            corrected = target_prefill_top1s_for_future(target, stage_correction_or_accepted, stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size - stage_correction_or_accepted.size, n_gen)
            target_verify_ms += (Time.instant - tv1).total_milliseconds
            target_next = corrected[-1][0]
          end
          pos += stage_correction_or_accepted.size
          break
        else
          target_next = target_nexts[-1][0] if generated_ids.size < n_gen
          pos += stage_correction_or_accepted.size
          ngram_offset += stage_correction_or_accepted.size
        end
      end

      unless correction_or_accepted.empty?
        pending_draft_start_pos = cycle_start_pos if pending_draft_tokens.empty?
        pending_draft_tokens.concat(correction_or_accepted)
      end
      if generated_ids.size > history_size_before
        history.concat(generated_ids[history_size_before, generated_ids.size - history_size_before])
      end
      if dump_cycles_path
        record_candidates = dump_cycle_token_ids ? ngram_candidates.dup : nil
        cycle_dumps << CycleDump.new(
          prompt_hash, target_model_id, draft_model_id,
          "ngram", "ngram", verify_mode,
          cycle_start_pos, history_size_before, generated_ids.size - history_size_before,
          ngram_candidates.size, ngram_candidates.size, accepted_in_cycle, reject_index,
          match_len, ngram_min, ngram_max, ngram_recursive,
          ngram_disabled_before, ngram_disabled,
          token_ids_hash(ngram_candidates), record_candidates,
          draft_ms - cycle_draft0,
          target_verify_ms - cycle_target_verify0,
          target_backup_ms - cycle_target_backup0,
          draft_backup_ms - cycle_draft_backup0,
          draft_resync_ms - cycle_draft_resync0,
          (Time.instant - cycle_wall0).total_milliseconds)
      end
      next
    end
  end

  if plain_fallback_enabled && adaptive_gamma && !adaptive_growth_allowed && current_gamma <= plain_fallback_gamma
    cycle_wall0 = Time.instant
    cycle_draft0 = draft_ms
    cycle_target_verify0 = target_verify_ms
    cycle_target_backup0 = target_backup_ms
    cycle_draft_backup0 = draft_backup_ms
    cycle_draft_resync0 = draft_resync_ms
    cycle_start_pos = pos
    generated_ids << target_next
    if generated_ids.size < n_gen
      tv0 = Time.instant
      target_next = advance_next(target, target_next, pos, target_state)
      target_verify_ms += (Time.instant - tv0).total_milliseconds
    end
    pos += 1
    plain_fallback_tokens += 1
    history.concat(generated_ids[history_size_before, generated_ids.size - history_size_before])
    if dump_cycles_path
      cycle_dumps << CycleDump.new(
        prompt_hash, target_model_id, draft_model_id,
        "target_only", "plain_fallback", verify_mode,
        cycle_start_pos, history_size_before, 1,
        1, 0, 0, -1,
        0, ngram_min, ngram_max, ngram_recursive,
        ngram_disabled, ngram_disabled,
        token_ids_hash([] of Int32), nil,
        draft_ms - cycle_draft0,
        target_verify_ms - cycle_target_verify0,
        target_backup_ms - cycle_target_backup0,
        draft_backup_ms - cycle_draft_backup0,
        draft_resync_ms - cycle_draft_resync0,
        (Time.instant - cycle_wall0).total_milliseconds)
    end
    next
  end

  unless pending_draft_tokens.empty?
    tr0 = Time.instant
    pending_draft_tokens.each_with_index do |tok_id, i|
      draft_next = advance_next(draft, tok_id, pending_draft_start_pos + i, draft_state)
    end
    draft_resync_ms += (Time.instant - tr0).total_milliseconds
    pending_draft_tokens.clear
  end

  cycles += 1
  cycle_wall0 = Time.instant
  cycle_draft0 = draft_ms
  cycle_target_verify0 = target_verify_ms
  cycle_target_backup0 = target_backup_ms
  cycle_draft_backup0 = draft_backup_ms
  cycle_draft_resync0 = draft_resync_ms
  cycle_start_pos = pos
  cycle_gamma = adaptive_gamma ? current_gamma : gamma
  draft_next_at_cycle = draft_next
  cycle_ngram_match_len = ngram_match_len(history, ngram_max, ngram_min)
  gamma_sum += cycle_gamma
  gamma_max_seen = Math.max(gamma_max_seen, cycle_gamma)
  correction_or_accepted = [] of Int32
  candidates = [] of Int32
  rejected = false
  cycle_done = false

  if early_reject_enabled && draft_next != target_next
    will_plain_fallback_after_reject = plain_fallback_enabled &&
                                       skip_draft_before_fallback_enabled &&
                                       adaptive_gamma &&
                                       !adaptive_regrow &&
                                       Math.max(1, current_gamma // 2) <= plain_fallback_gamma
    generated_ids << target_next
    correction_or_accepted << target_next
    proposed += 1
    if generated_ids.size < n_gen
      tv0 = Time.instant
      target_next = advance_next(target, target_next, pos, target_state)
      target_verify_ms += (Time.instant - tv0).total_milliseconds
    end
    if will_plain_fallback_after_reject || generated_ids.size >= n_gen
      draft_skips_before_fallback += 1
    else
      td0 = Time.instant
      draft_next = advance_next(draft, correction_or_accepted[0], pos, draft_state)
      draft_ms += (Time.instant - td0).total_milliseconds
    end
    pos += 1
    rejected = true
    early_rejects += 1
    cycle_done = true
  elsif single_accept_fast_enabled && cycle_gamma == 1 && draft_next == target_next
    accepted_token = draft_next
    generated_ids << accepted_token
    correction_or_accepted << accepted_token
    accepted += 1
    proposed += 1
    if generated_ids.size < n_gen
      td0 = Time.instant
      draft_next = advance_next(draft, accepted_token, pos, draft_state)
      draft_ms += (Time.instant - td0).total_milliseconds
      tv0 = Time.instant
      target_next = advance_next(target, accepted_token, pos, target_state)
      target_verify_ms += (Time.instant - tv0).total_milliseconds
    end
    pos += 1
    single_accept_fast += 1
    rejected = false
    cycle_done = true
  end

  unless cycle_done
    skip_draft_backup_for_fallback = plain_fallback_enabled &&
                                     skip_draft_before_fallback_enabled &&
                                     skip_draft_backup_before_fallback_enabled &&
                                     adaptive_gamma &&
                                     !adaptive_regrow &&
                                     Math.max(1, current_gamma // 2) <= plain_fallback_gamma
    unless skip_draft_backup_for_fallback
      tdb0 = Time.instant
      draft_cycle_base.copy_from!(draft_state)
      draft_backup_ms += (Time.instant - tdb0).total_milliseconds
    else
      draft_backup_skips += 1
    end

    if verify_mode == "staged"
      remaining_gamma = cycle_gamma
      stage_index = 0
      while remaining_gamma > 0 && generated_ids.size < n_gen
        if early_reject_enabled && draft_next != target_next
          generated_ids << target_next
          correction_or_accepted << target_next
          proposed += 1
          if generated_ids.size < n_gen
            tv0 = Time.instant
            target_next = advance_next(target, target_next, pos, target_state)
            target_verify_ms += (Time.instant - tv0).total_milliseconds
          end
          pos += 1
          rejected = true
          early_rejects += 1
          break
        end

        stage_len = if stage_index == 0 && remaining_gamma > stage_gate
                      stage_gate
                    else
                      remaining_gamma
                    end
        stage_len = Math.min(stage_len, n_gen - generated_ids.size)
        break if stage_len <= 0

        stage_start_pos = pos
        stage_candidates = [] of Int32
        td0 = Time.instant
        stage_len.times do |i|
          break if generated_ids.size + stage_candidates.size >= n_gen
          stage_candidates << draft_next
          break if generated_ids.size + stage_candidates.size >= n_gen
          draft_next = advance_next(draft, draft_next, stage_start_pos + i, draft_state)
        end
        draft_ms += (Time.instant - td0).total_milliseconds
        proposed += stage_candidates.size
        candidates.concat(stage_candidates)
        break if stage_candidates.empty?

        tb0 = Time.instant
        target_backup_state.copy_from!(target_state)
        target_backup_ms += (Time.instant - tb0).total_milliseconds
        tv0 = Time.instant
        target_nexts = target_prefill_top1s_for_future(target, stage_candidates, stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size, n_gen)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "cycle=#{cycles} stage=#{stage_index} pos=#{stage_start_pos} expected0=#{target_next} candidates=#{stage_candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        stage_correction_or_accepted = [] of Int32
        stage_candidates.each_with_index do |cand, i|
          break if generated_ids.size >= n_gen
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            stage_correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0] if i < target_nexts.size
          else
            generated_ids << expected
            correction_or_accepted << expected
            stage_correction_or_accepted << expected
            rejected = true
            break
          end
        end

        if rejected
          if generated_ids.size < n_gen
            target_state.copy_from!(target_backup_state)
            tv1 = Time.instant
            corrected = target_prefill_top1s_for_future(target, stage_correction_or_accepted, stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size - stage_correction_or_accepted.size, n_gen)
            target_verify_ms += (Time.instant - tv1).total_milliseconds
            target_next = corrected[-1][0]
          end
          pos += stage_correction_or_accepted.size
          break
        else
          target_next = target_nexts[-1][0] if generated_ids.size < n_gen
          pos += stage_candidates.size
          remaining_gamma -= stage_candidates.size
          stage_index += 1
        end
      end
    else
      td0 = Time.instant
      cycle_gamma.times do |i|
        break if generated_ids.size + candidates.size >= n_gen
        candidates << draft_next
        break if generated_ids.size + candidates.size >= n_gen
        draft_next = advance_next(draft, draft_next, pos + i, draft_state)
      end
      draft_ms += (Time.instant - td0).total_milliseconds
      proposed += candidates.size

      if verify_mode == "serial" || (verify_mode == "hybrid" && cycles == 1)
        candidates.each do |cand|
          if cand == target_next
            generated_ids << cand
            correction_or_accepted << cand
            accepted += 1
            if generated_ids.size < n_gen
              tv0 = Time.instant
              target_next = advance_next(target, cand, pos, target_state)
              target_verify_ms += (Time.instant - tv0).total_milliseconds
            end
            pos += 1
          else
            generated_ids << target_next
            correction_or_accepted << target_next
            if generated_ids.size < n_gen
              tv0 = Time.instant
              target_next = advance_next(target, target_next, pos, target_state)
              target_verify_ms += (Time.instant - tv0).total_milliseconds
            end
            pos += 1
            rejected = true
            break
          end
        end
      elsif verify_mode == "chunk"
        target_base = target_state
        verify_state = target_base.fork
        tv0 = Time.instant
        target_nexts = target_prefill_top1s_for_future(target, candidates, cycle_start_pos, verify_state, allow_guarded_verifier, generated_ids.size, n_gen)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "cycle=#{cycles} pos=#{cycle_start_pos} expected0=#{target_next} candidates=#{candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        reject_at = nil.as(Int32?)
        candidates.each_with_index do |cand, i|
          break if generated_ids.size >= n_gen
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0] if i < target_nexts.size
          else
            generated_ids << expected
            correction_or_accepted << expected
            reject_at = i
            rejected = true
            break
          end
        end

        if rejected
          if generated_ids.size < n_gen
            tv1 = Time.instant
            corrected = target_prefill_top1s_for_future(target, correction_or_accepted, cycle_start_pos, target_state, allow_guarded_verifier, generated_ids.size - correction_or_accepted.size, n_gen)
            target_verify_ms += (Time.instant - tv1).total_milliseconds
            target_next = corrected[-1][0]
          end
          pos += correction_or_accepted.size
        else
          target_state.copy_from!(verify_state) if generated_ids.size < n_gen
          target_next = target_nexts[-1][0] if generated_ids.size < n_gen
          pos += candidates.size
        end
      else
        tb0 = Time.instant
        target_backup_state.copy_from!(target_state)
        target_backup_ms += (Time.instant - tb0).total_milliseconds
        tv0 = Time.instant
        target_nexts = target_prefill_top1s_for_future(target, candidates, cycle_start_pos, target_state, allow_guarded_verifier, generated_ids.size, n_gen)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "cycle=#{cycles} pos=#{cycle_start_pos} expected0=#{target_next} candidates=#{candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        candidates.each_with_index do |cand, i|
          break if generated_ids.size >= n_gen
          if cand == expected
            generated_ids << cand
            correction_or_accepted << cand
            accepted += 1
            expected = target_nexts[i][0] if i < target_nexts.size
          else
            generated_ids << expected
            correction_or_accepted << expected
            rejected = true
            break
          end
        end

        if rejected
          if generated_ids.size < n_gen
            target_state.copy_from!(target_backup_state)
            tv1 = Time.instant
            corrected = target_prefill_top1s_for_future(target, correction_or_accepted, cycle_start_pos, target_state, allow_guarded_verifier, generated_ids.size - correction_or_accepted.size, n_gen)
            target_verify_ms += (Time.instant - tv1).total_milliseconds
            target_next = corrected[-1][0]
          end
          pos += correction_or_accepted.size
        else
          target_next = target_nexts[-1][0] if generated_ids.size < n_gen
          pos += candidates.size
        end
      end
    end

    if rejected
      will_plain_fallback_after_reject = plain_fallback_enabled &&
                                         skip_draft_before_fallback_enabled &&
                                         adaptive_gamma &&
                                         !adaptive_regrow &&
                                         Math.max(1, current_gamma // 2) <= plain_fallback_gamma
      if will_plain_fallback_after_reject || generated_ids.size >= n_gen
        draft_skips_before_fallback += 1
      else
        raise "draft backup missing before required resync" if skip_draft_backup_for_fallback
        tr0 = Time.instant
        draft_next = resync_draft!(draft, draft_state, draft_cycle_base, correction_or_accepted, cycle_start_pos)
        draft_resync_ms += (Time.instant - tr0).total_milliseconds
      end
    end
  end

  if generated_ids.size > history_size_before
    history.concat(generated_ids[history_size_before, generated_ids.size - history_size_before])
  end

  if dump_cycles_path
    candidate_snapshot = candidates.empty? && !correction_or_accepted.empty? ? [draft_next_at_cycle] : candidates
    accepted_in_cycle = rejected ? Math.max(correction_or_accepted.size - 1, 0) : correction_or_accepted.size
    reject_index = rejected ? accepted_in_cycle : -1
    kind = if candidate_snapshot.size == 1 && rejected && candidates.empty?
             "neural_early_reject"
           elsif candidate_snapshot.size == 1 && !rejected && candidates.empty?
             "neural_single_fast"
           elsif verify_mode == "staged"
             "neural_staged"
           else
             "neural"
           end
    record_candidates = dump_cycle_token_ids ? candidate_snapshot.dup : nil
    cycle_dumps << CycleDump.new(
      prompt_hash, target_model_id, draft_model_id,
      kind, "neural", verify_mode,
      cycle_start_pos, history_size_before, generated_ids.size - history_size_before,
      cycle_gamma, candidate_snapshot.size, accepted_in_cycle, reject_index,
      cycle_ngram_match_len, ngram_min, ngram_max, ngram_recursive,
      ngram_disabled, ngram_disabled,
      token_ids_hash(candidate_snapshot), record_candidates,
      draft_ms - cycle_draft0,
      target_verify_ms - cycle_target_verify0,
      target_backup_ms - cycle_target_backup0,
      draft_backup_ms - cycle_draft_backup0,
      draft_resync_ms - cycle_draft_resync0,
      (Time.instant - cycle_wall0).total_milliseconds)
  end

  if adaptive_gamma
    if rejected
      full_accept_streak = 0
      adaptive_growth_allowed = false unless adaptive_regrow
      current_gamma = Math.max(1, current_gamma // 2)
    elsif adaptive_growth_allowed && candidates.size == cycle_gamma && current_gamma < max_gamma
      full_accept_streak += 1
      if adaptive_bootstrap_gamma > current_gamma && current_gamma == gamma
        if full_accept_streak >= adaptive_bootstrap_streak
          current_gamma = Math.min(max_gamma, adaptive_bootstrap_gamma)
          full_accept_streak = 0
        end
      else
        required_full_accept_streak = if adaptive_fast_regrow_min_gamma > 0 && current_gamma >= adaptive_fast_regrow_min_gamma
                                        1
                                      else
                                        adaptive_full_accept_streak
                                      end
        if full_accept_streak >= required_full_accept_streak
          current_gamma = Math.min(max_gamma, current_gamma * 2)
          full_accept_streak = 0
        end
      end
    end
  end
end
wall_ms = (Time.instant - wall0).total_milliseconds
profile_report = nil.as(String?)
if profile_spec_region
  ML::GGUF::Qwen35Metal::Profile.disable!
  profile_report = ML::GGUF::Qwen35Metal::Profile.report_io
end

plain, plain_ms, plain_prefill_ms = greedy_sequence(target, prompt_ids, n_gen)
unless plain == generated_ids
  first_diff = plain.zip(generated_ids).index { |(a, b)| a != b } || Math.min(plain.size, generated_ids.size)
  raise "speculative output diverged from target greedy at #{first_diff}: plain=#{plain.inspect} speculative=#{generated_ids.inspect}"
end

if path = dump_cycles_path
  plain_ms_per_token = plain_ms / n_gen
  cycle_dumps.each do |record|
    record.expected_gain_ms = record.generated_count * plain_ms_per_token - record.wall_ms
  end
  dir = File.dirname(path)
  Dir.mkdir_p(dir) unless dir.empty? || dir == "."
  File.open(path, "w") do |io|
    cycle_dumps.each do |record|
      record.to_json(io)
      io.puts
    end
  end
end

accept_rate = accepted.to_f64 / proposed.to_f64
tokens_s = n_gen.to_f64 / (wall_ms / 1000.0)
plain_tokens_s = n_gen.to_f64 / (plain_ms / 1000.0)

puts
puts "accept_rate=#{(accept_rate * 100.0).round(2)}% accepted=#{accepted}/#{proposed} cycles=#{cycles}"
if ngram_enabled
  ngram_rate = ngram_proposed > 0 ? (ngram_accepted.to_f64 * 100.0 / ngram_proposed.to_f64) : 0.0
  puts "ngram_stats accepted=#{ngram_accepted}/#{ngram_proposed} rate=#{ngram_rate.round(2)}% cycles=#{ngram_cycles} disabled=#{ngram_disabled} pending_draft=#{pending_draft_tokens.size}"
  if router_model
    avg_router_score = ngram_router_checks > 0 ? (ngram_router_score_sum / ngram_router_checks).round(4) : 0.0
    puts "ngram_router_stats checks=#{ngram_router_checks} skips=#{ngram_router_skips} threshold=#{router_model.not_nil!.threshold} avg_score=#{avg_router_score}"
  end
end
avg_gamma = cycles > 0 ? (gamma_sum.to_f64 / cycles.to_f64).round(2) : 0.0
puts "gamma_stats avg=#{avg_gamma} max_seen=#{gamma_max_seen} final=#{current_gamma} early_rejects=#{early_rejects} single_fast=#{single_accept_fast} plain_fallback=#{plain_fallback_tokens} draft_skip=#{draft_skips_before_fallback} draft_backup_skip=#{draft_backup_skips}"
puts "spec_wall=#{wall_ms.round(1)} ms (#{(wall_ms / n_gen).round(2)} ms/tok, #{tokens_s.round(2)} tok/s, verify=#{verify_mode})"
puts "plain_target_wall=#{plain_ms.round(1)} ms (#{(plain_ms / n_gen).round(2)} ms/tok, #{plain_tokens_s.round(2)} tok/s, decode_only=true)"
puts "plain_target_prefill_wall=#{plain_prefill_ms.round(1)} ms"
puts "verifier_warmup_wall=#{verifier_warmup_ms.round(1)} ms"
puts "time_breakdown draft=#{draft_ms.round(1)} ms target_verify=#{target_verify_ms.round(1)} ms target_backup=#{target_backup_ms.round(1)} ms draft_backup=#{draft_backup_ms.round(1)} ms draft_resync=#{draft_resync_ms.round(1)} ms"
puts profile_report.not_nil! if profile_report
puts "note=exact speculative probe; speedup still needs lower draft cost and/or verifier rollback overhead removal"
puts "generated=#{tok.decode(generated_ids).inspect}"
