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

def parse_i32_list(value : String) : Array(Int32)
  value.split(",").map(&.strip).reject(&.empty?).map(&.to_i)
end

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
target_only = ENV["QWEN35_SPEC_TARGET_ONLY"]? == "1"
ngram_enabled = ENV["QWEN35_SPEC_NGRAM"]? == "1"
ngram_gamma = (ENV["QWEN35_SPEC_NGRAM_GAMMA"]? || "32").to_i
ngram_min = (ENV["QWEN35_SPEC_NGRAM_MIN"]? || "6").to_i
ngram_max = (ENV["QWEN35_SPEC_NGRAM_MAX"]? || "8").to_i
ngram_stage_min_env = ENV["QWEN35_SPEC_NGRAM_STAGE_MIN"]?
ngram_stage_min = (ngram_stage_min_env || (ngram_gamma + 1).to_s).to_i
ngram_stage_min_explicit = !ngram_stage_min_env.nil?
ngram_probe_gate = (ENV["QWEN35_SPEC_NGRAM_PROBE_GATE"]? || "0").to_i
ngram_probe_min = (ENV["QWEN35_SPEC_NGRAM_PROBE_MIN"]? || "2").to_i
ngram_risk_min_size = (ENV["QWEN35_SPEC_NGRAM_RISK_MIN_SIZE"]? || "16").to_i
ngram_min_candidates = (ENV["QWEN35_SPEC_NGRAM_MIN_CANDIDATES"]? || "0").to_i
ngram_risk_gate = ENV["QWEN35_SPEC_NGRAM_RISK_GATE"]? == "1"
ngram_corridor_gate = ENV["QWEN35_SPEC_NGRAM_CORRIDOR_GATE"]? == "1"
ngram_corridor_min_size = (ENV["QWEN35_SPEC_NGRAM_CORRIDOR_MIN_SIZE"]? || "4").to_i
ngram_corridor_match_len_min = (ENV["QWEN35_SPEC_NGRAM_CORRIDOR_MATCH_LEN_MIN"]? || "0").to_i
ngram_corridor_lag4_min = (ENV["QWEN35_SPEC_NGRAM_CORRIDOR_LAG4_MIN"]? || "0.25").to_f
ngram_corridor_lag8_min = (ENV["QWEN35_SPEC_NGRAM_CORRIDOR_LAG8_MIN"]? || "0.5").to_f
ngram_corridor_entropy_max = (ENV["QWEN35_SPEC_NGRAM_CORRIDOR_ENTROPY_MAX"]? || "0.6").to_f
ngram_recursive = ENV["QWEN35_SPEC_NGRAM_RECURSIVE_OFF"]? != "1"
ngram_disable_after_reject = ENV["QWEN35_SPEC_NGRAM_DISABLE_AFTER_REJECT_OFF"]? != "1"
ngram_replay_on_reject = ENV["QWEN35_SPEC_NGRAM_REPLAY_ON_REJECT"]? == "1"
ngram_target_only = ENV["QWEN35_SPEC_NGRAM_TARGET_ONLY"]? == "1"
ngram_index_enabled = ENV["QWEN35_SPEC_NGRAM_INDEX_OFF"]? != "1"
ngram_source_history = ENV["QWEN35_SPEC_NGRAM_SOURCE_HISTORY"]?.try { |v| parse_i32_list(v) } || [] of Int32
ngram_replay_start = (ENV["QWEN35_SPEC_NGRAM_REPLAY_START"]? || "-1").to_i
ngram_cursor_only = ENV["QWEN35_SPEC_NGRAM_CURSOR_ONLY"]? == "1"
ngram_trusted_source = ENV["QWEN35_SPEC_NGRAM_TRUSTED_SOURCE"]? == "1"
ngram_source_prefix_gate = ENV["QWEN35_SPEC_NGRAM_SOURCE_PREFIX_GATE_OFF"]? != "1"
current_hidden_trace = ENV["QWEN35_SPEC_CURRENT_HIDDEN_TRACE"]? == "1"
current_hidden_trace_topk = (ENV["QWEN35_SPEC_CURRENT_HIDDEN_TOPK"]? || "8").to_i
current_hidden_tree = ENV["QWEN35_SPEC_CURRENT_HIDDEN_TREE"]? == "1"
current_hidden_tree_depth = (ENV["QWEN35_SPEC_CURRENT_HIDDEN_TREE_DEPTH"]? || "4").to_i
current_hidden_tree_width = (ENV["QWEN35_SPEC_CURRENT_HIDDEN_TREE_WIDTH"]? || current_hidden_trace_topk.to_s).to_i
current_hidden_ctx = ENV["QWEN35_SPEC_CURRENT_HIDDEN_CTX"]? == "1"
current_hidden_ctx_seed = (ENV["QWEN35_SPEC_CURRENT_HIDDEN_CTX_SEED"]? || "2").to_i
current_hidden_ctx_depth = (ENV["QWEN35_SPEC_CURRENT_HIDDEN_CTX_DEPTH"]? || "8").to_i
current_hidden_ctx_width = (ENV["QWEN35_SPEC_CURRENT_HIDDEN_CTX_WIDTH"]? || current_hidden_trace_topk.to_s).to_i
prepare_state_metal = ENV["QWEN35_PREPARE_STATE_OFF"]? != "1"
warm_verifier = ENV["QWEN35_SPEC_WARM_VERIFIER_OFF"]? != "1"
allow_guarded_verifier = ENV["QWEN35_SPEC_ALLOW_GUARDED_VERIFIER"]? == "1"
dump_cycles_path = ENV["QWEN35_SPEC_DUMP_CYCLES"]?
dump_cycle_token_ids = ENV["QWEN35_SPEC_DUMP_TOKEN_IDS"]? == "1"
router_model_path = ENV["QWEN35_SPEC_ROUTER_MODEL"]?
prompt_category = ENV["QWEN35_SPEC_PROMPT_CATEGORY"]? || "unknown"
router_long_threshold = ENV["QWEN35_SPEC_ROUTER_LONG_THRESHOLD"]?.try(&.to_f)
router_long_min = (ENV["QWEN35_SPEC_ROUTER_LONG_MIN"]? || "16").to_i

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
  parser.on("--target-only", "Research: generate through the exact target decode loop without neural or n-gram proposals") { target_only = true }
  parser.on("--allow-guarded-verifier", "Research only: allow guarded full-row verifier inside speculative target chunks") { allow_guarded_verifier = true }
  parser.on("--ngram-gamma N", "Maximum n-gram candidates per chunk (default: env QWEN35_SPEC_NGRAM_GAMMA or 32)") { |value| ngram_gamma = value.to_i }
  parser.on("--ngram-min N", "Minimum repeated suffix length before n-gram drafting (default: env QWEN35_SPEC_NGRAM_MIN or 6)") { |value| ngram_min = value.to_i }
  parser.on("--ngram-max N", "Maximum repeated suffix length to search (default: env QWEN35_SPEC_NGRAM_MAX or 8)") { |value| ngram_max = value.to_i }
  parser.on("--ngram-stage-min N", "For --verify staged, only split n-gram chunks with at least this many candidates (default: ngram_gamma + 1)") do |value|
    ngram_stage_min = value.to_i
    ngram_stage_min_explicit = true
  end
  parser.on("--ngram-probe-gate N", "Research: verify the first N n-gram candidates before bulk-verifying the rest; 0 disables (default)") { |value| ngram_probe_gate = value.to_i }
  parser.on("--ngram-probe-min N", "Minimum n-gram chunk size for --ngram-probe-gate (default: 2)") { |value| ngram_probe_min = value.to_i }
  parser.on("--ngram-risk-min-size N", "Minimum candidate size for the n-gram risk gate (default: 16)") { |value| ngram_risk_min_size = value.to_i }
  parser.on("--ngram-min-candidates N", "Skip n-gram chunks shorter than N candidates; 0 preserves historical behavior") { |value| ngram_min_candidates = value.to_i }
  parser.on("--ngram-risk-gate", "Research: skip n-gram chunks whose candidate-token shape matches known bad repeat tails") { ngram_risk_gate = true }
  parser.on("--ngram-corridor-gate", "Research: require periodic or low-entropy n-gram continuation before entering the verifier corridor") { ngram_corridor_gate = true }
  parser.on("--ngram-corridor-min-size N", "Minimum candidate length for --ngram-corridor-gate evidence (default: 4)") { |value| ngram_corridor_min_size = value.to_i }
  parser.on("--ngram-corridor-match-len-min N", "Accept candidate corridors with at least this suffix match length; 0 disables (default)") { |value| ngram_corridor_match_len_min = value.to_i }
  parser.on("--ngram-corridor-lag4-min F", "Minimum lag-4 ratio for --ngram-corridor-gate (default: 0.25)") { |value| ngram_corridor_lag4_min = value.to_f }
  parser.on("--ngram-corridor-lag8-min F", "Minimum lag-8 ratio for --ngram-corridor-gate (default: 0.5)") { |value| ngram_corridor_lag8_min = value.to_f }
  parser.on("--ngram-corridor-entropy-max F", "Maximum normalized entropy for --ngram-corridor-gate (default: 0.6)") { |value| ngram_corridor_entropy_max = value.to_f }
  parser.on("--no-recursive-ngram", "Do not recursively extend n-gram candidates through scratch history") { ngram_recursive = false }
  parser.on("--keep-ngram-after-reject", "Keep trying n-gram draft chunks after a rejected n-gram chunk") { ngram_disable_after_reject = false }
  parser.on("--ngram-replay-on-reject", "Research: skip n-gram state backups and rebuild exact target state only after a non-final reject") { ngram_replay_on_reject = true }
  parser.on("--ngram-target-only", "Research: after n-gram has no chunk, continue with exact target-only steps instead of neural draft") { ngram_target_only = true }
  parser.on("--ngram-source-history LIST", "Comma-separated validated source/cache token IDs for cursor replay") { |value| ngram_source_history = parse_i32_list(value) }
  parser.on("--ngram-replay-start N", "Start n-gram cursor replay at source-history offset N") { |value| ngram_replay_start = value.to_i }
  parser.on("--ngram-cursor-only", "Only propose from an active source-history cursor; fallback instead of suffix-searching") { ngram_cursor_only = true }
  parser.on("--ngram-trusted-source", "Bypass untrusted candidate-shape gates for source cursor proposals after prefix validation") { ngram_trusted_source = true }
  parser.on("--ngram-no-source-prefix-gate", "Disable prompt-prefix validation before source cursor replay") { ngram_source_prefix_gate = false }
  parser.on("--current-hidden-trace", "Trace-only: score prompt hidden-table replay against exact generated tokens without changing policy") { current_hidden_trace = true }
  parser.on("--current-hidden-topk N", "Candidate width for current-hidden trace/tree/ctx routes (default: env QWEN35_SPEC_CURRENT_HIDDEN_TOPK or 8)") { |value| current_hidden_trace_topk = value.to_i; current_hidden_tree_width = value.to_i; current_hidden_ctx_width = value.to_i }
  parser.on("--current-hidden-tree", "Research: exact-first current-hidden topK beam verifier route") { current_hidden_tree = true }
  parser.on("--current-hidden-tree-depth N", "Chunk depth for --current-hidden-tree (default: env QWEN35_SPEC_CURRENT_HIDDEN_TREE_DEPTH or 4)") { |value| current_hidden_tree_depth = value.to_i }
  parser.on("--current-hidden-tree-width N", "Beam width/topK for --current-hidden-tree (default: --current-hidden-topk)") { |value| current_hidden_tree_width = value.to_i }
  parser.on("--current-hidden-ctx", "Research: seed exact tokens, then verify one context-selected current-hidden path") { current_hidden_ctx = true }
  parser.on("--current-hidden-ctx-seed N", "Exact seed tokens before --current-hidden-ctx proposals (default: env QWEN35_SPEC_CURRENT_HIDDEN_CTX_SEED or 2)") { |value| current_hidden_ctx_seed = value.to_i }
  parser.on("--current-hidden-ctx-depth N", "Maximum context-selected proposal length for --current-hidden-ctx (default: env QWEN35_SPEC_CURRENT_HIDDEN_CTX_DEPTH or 8)") { |value| current_hidden_ctx_depth = value.to_i }
  parser.on("--current-hidden-ctx-width N", "TopK candidate width for --current-hidden-ctx selector (default: --current-hidden-topk)") { |value| current_hidden_ctx_width = value.to_i }
  parser.on("--no-warm-verifier", "Do not warm the target chunk-verifier route before decode timing") { warm_verifier = false }
  parser.on("--trace", "Print per-cycle verifier decisions") { trace = true }
  parser.on("--dump-cycles PATH", "Write per-cycle speculative policy/timing records as JSONL") { |path| dump_cycles_path = path }
  parser.on("--dump-cycle-token-ids", "Include raw token ids in --dump-cycles records; default only writes stable hashes") { dump_cycle_token_ids = true }
  parser.on("--router-model PATH", "Research: logistic router JSON used to gate n-gram chunks before verification") { |path| router_model_path = path }
  parser.on("--router-long-threshold X", "Research: stricter router threshold for long n-gram chunks") { |value| router_long_threshold = value.to_f }
  parser.on("--router-long-min N", "Candidate count where --router-long-threshold applies (default: 16)") { |value| router_long_min = value.to_i }
  parser.on("--prompt-category NAME", "Research: prompt category feature for router models (default: env QWEN35_SPEC_PROMPT_CATEGORY or unknown)") { |value| prompt_category = value }
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
raise ArgumentError.new("QWEN35_SPEC_NGRAM_PROBE_GATE must be non-negative") unless ngram_probe_gate >= 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_PROBE_MIN must be positive") unless ngram_probe_min > 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_RISK_MIN_SIZE must be positive") unless ngram_risk_min_size > 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_MIN_CANDIDATES must be non-negative") unless ngram_min_candidates >= 0
raise ArgumentError.new("QWEN35_SPEC_NGRAM_REPLAY_START must be >= -1") unless ngram_replay_start >= -1
raise ArgumentError.new("--ngram-cursor-only requires --ngram") if ngram_cursor_only && !ngram_enabled
raise ArgumentError.new("--ngram-replay-start requires --ngram-source-history") if ngram_replay_start >= 0 && ngram_source_history.empty?
raise ArgumentError.new("--ngram-trusted-source requires --ngram-source-history") if ngram_trusted_source && ngram_source_history.empty?
raise ArgumentError.new("--ngram-trusted-source requires --ngram-cursor-only") if ngram_trusted_source && !ngram_cursor_only
raise ArgumentError.new("--ngram-trusted-source requires --ngram-replay-start") if ngram_trusted_source && ngram_replay_start < 0
raise ArgumentError.new("--ngram-trusted-source requires source prefix gate") if ngram_trusted_source && !ngram_source_prefix_gate
raise ArgumentError.new("QWEN35_SPEC_ROUTER_LONG_MIN must be positive") unless router_long_min > 0
raise ArgumentError.new("QWEN35_SPEC_ROUTER_LONG_THRESHOLD must be in [0,1]") if router_long_threshold && !(0.0..1.0).includes?(router_long_threshold.not_nil!)
raise ArgumentError.new("router model not found: #{router_model_path}") if router_model_path && !File.file?(router_model_path.not_nil!)
raise ArgumentError.new("QWEN35_SPEC_CURRENT_HIDDEN_TOPK must be positive") unless current_hidden_trace_topk > 0
raise ArgumentError.new("QWEN35_SPEC_CURRENT_HIDDEN_TREE_DEPTH must be positive") unless current_hidden_tree_depth > 0
raise ArgumentError.new("QWEN35_SPEC_CURRENT_HIDDEN_TREE_WIDTH must be positive") unless current_hidden_tree_width > 0
raise ArgumentError.new("QWEN35_SPEC_CURRENT_HIDDEN_CTX_SEED must be positive") unless current_hidden_ctx_seed > 0
raise ArgumentError.new("QWEN35_SPEC_CURRENT_HIDDEN_CTX_DEPTH must be positive") unless current_hidden_ctx_depth > 0
raise ArgumentError.new("QWEN35_SPEC_CURRENT_HIDDEN_CTX_WIDTH must be positive") unless current_hidden_ctx_width > 0

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

def replay_target_state(weights : ML::GGUF::Qwen35Weights,
                        prompt_ids : Array(Int32),
                        generated_ids : Array(Int32),
                        max_seq : Int32,
                        prepare_state_metal : Bool) : {ML::GGUF::Qwen35CPU::State, Int32}
  replay_state = ML::GGUF::Qwen35CPU::State.new(weights.hparams, max_seq: max_seq)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(replay_state, weights.hparams) if prepare_state_metal
  replay_ids = prompt_ids.dup
  replay_ids.concat(generated_ids)
  {replay_state, prefill_next(weights, replay_ids, replay_state)}
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

def hidden_row_norm_spec(hidden : Array(Float32), row : Int32, dim : Int32) : Float64
  base = row * dim
  acc = 0.0
  dim.times do |i|
    x = hidden[base + i].to_f64
    acc += x * x
  end
  Math.sqrt(acc)
end

def hidden_row_cosine_spec(hidden : Array(Float32),
                           dim : Int32,
                           a : Int32,
                           b : Int32,
                           norms : Array(Float64)) : Float64
  denom = norms[a] * norms[b]
  return -Float64::INFINITY if denom <= 0.0

  abase = a * dim
  bbase = b * dim
  acc = 0.0
  dim.times { |i| acc += hidden[abase + i].to_f64 * hidden[bbase + i].to_f64 }
  acc / denom
end

def hidden_vector_norm_spec(v : Array(Float32)) : Float64
  acc = 0.0
  v.each do |x|
    xf = x.to_f64
    acc += xf * xf
  end
  Math.sqrt(acc)
end

def current_hidden_trace_topk_for_row(hidden : Array(Float32),
                                      labels : Array(Int32),
                                      norms : Array(Float64),
                                      dim : Int32,
                                      eval_row : Int32,
                                      train_count : Int32,
                                      top_k : Int32) : NamedTuple(ids: Array(Int32), best_row: Int32, best_cos: Float64)
  by_label = {} of Int32 => Float64
  best_row = -1
  best_cos = -Float64::INFINITY
  train_count.times do |row|
    sim = hidden_row_cosine_spec(hidden, dim, eval_row, row, norms)
    if sim > best_cos
      best_cos = sim
      best_row = row
    end
    label = labels[row]
    prev = by_label[label]?
    by_label[label] = sim if prev.nil? || sim > prev
  end

  ranked = by_label.to_a.sort_by { |pair| -pair[1] }
  {
    ids:      ranked.first(top_k).map { |pair| pair[0] },
    best_row: best_row,
    best_cos: best_cos,
  }
end

def current_hidden_chain_candidate_rows(hidden : Array(Float32),
                                        labels : Array(Int32),
                                        norms : Array(Float64),
                                        dim : Int32,
                                        cursor_row : Int32,
                                        train_count : Int32,
                                        require_next : Bool,
                                        top_k : Int32) : Array(NamedTuple(id: Int32, row: Int32, score: Float64))
  by_label = {} of Int32 => NamedTuple(row: Int32, score: Float64)
  train_count.times do |row|
    next if require_next && row + 1 >= train_count
    sim = hidden_row_cosine_spec(hidden, dim, cursor_row, row, norms)
    label = labels[row]
    prev = by_label[label]?
    by_label[label] = {row: row, score: sim} if prev.nil? || sim > prev[:score]
  end

  by_label.to_a
    .sort_by { |pair| -pair[1][:score] }
    .first(top_k)
    .map { |pair| {id: pair[0], row: pair[1][:row], score: pair[1][:score]} }
end

def current_hidden_chain_candidates(hidden : Array(Float32),
                                    labels : Array(Int32),
                                    norms : Array(Float64),
                                    dim : Int32,
                                    cursor_row : Int32,
                                    train_count : Int32,
                                    require_next : Bool,
                                    top_k : Int32) : NamedTuple(ids: Array(Int32), rows: Array(Int32), best_cos: Float64)
  picked = current_hidden_chain_candidate_rows(hidden, labels, norms, dim, cursor_row, train_count, require_next, top_k)
  {
    ids:      picked.map { |pair| pair[:id] },
    rows:     picked.map { |pair| pair[:row] },
    best_cos: picked.empty? ? -Float64::INFINITY : picked[0][:score],
  }
end

def current_hidden_tree_paths(hidden : Array(Float32),
                              labels : Array(Int32),
                              norms : Array(Float64),
                              dim : Int32,
                              cursor_row : Int32,
                              train_count : Int32,
                              depth : Int32,
                              width : Int32) : Array(NamedTuple(ids: Array(Int32), rows: Array(Int32), cursor: Int32, score: Float64))
  beams = [{ids: [] of Int32, rows: [] of Int32, cursor: cursor_row, score: 0.0}]
  depth.times do |step|
    expanded = [] of NamedTuple(ids: Array(Int32), rows: Array(Int32), cursor: Int32, score: Float64)
    require_next = step < depth - 1
    beams.each do |beam|
      current_hidden_chain_candidate_rows(hidden, labels, norms, dim, beam[:cursor], train_count, require_next, width).each do |candidate|
        next_cursor = candidate[:row] + 1 < train_count ? candidate[:row] + 1 : candidate[:row]
        expanded << {
          ids:    beam[:ids] + [candidate[:id]],
          rows:   beam[:rows] + [candidate[:row]],
          cursor: next_cursor,
          score:  beam[:score] + candidate[:score],
        }
      end
    end
    break if expanded.empty?
    beams = expanded.sort_by { |beam| -beam[:score] }.first(width)
  end
  beams.reject { |beam| beam[:ids].empty? }
end

def token_suffix_overlap_at_prompt_row(prompt_ids : Array(Int32),
                                       emitted_ids : Array(Int32),
                                       prompt_row : Int32,
                                       max_len : Int32 = 24) : Int32
  return 0 if emitted_ids.empty? || prompt_row < 0

  overlap = 0
  while overlap < max_len && overlap < emitted_ids.size && overlap <= prompt_row
    return overlap unless emitted_ids[emitted_ids.size - 1 - overlap] == prompt_ids[prompt_row - overlap]
    overlap += 1
  end
  overlap
end

def choose_current_hidden_context_candidate(prompt_ids : Array(Int32),
                                            emitted_ids : Array(Int32),
                                            candidates : Array(NamedTuple(id: Int32, row: Int32, score: Float64))) : NamedTuple(id: Int32, row: Int32, score: Float64, overlap: Int32)?
  best = nil.as(NamedTuple(id: Int32, row: Int32, score: Float64, overlap: Int32)?)
  candidates.each do |candidate|
    overlap = token_suffix_overlap_at_prompt_row(prompt_ids, emitted_ids, candidate[:row])
    item = {id: candidate[:id], row: candidate[:row], score: candidate[:score], overlap: overlap}
    if best.nil? || item[:overlap] > best.not_nil![:overlap] ||
       (item[:overlap] == best.not_nil![:overlap] && item[:score] > best.not_nil![:score])
      best = item
    end
  end
  best
end

def current_hidden_chain_step(hidden : Array(Float32),
                              labels : Array(Int32),
                              norms : Array(Float64),
                              dim : Int32,
                              cursor_row : Int32,
                              train_count : Int32,
                              require_next : Bool) : NamedTuple(id: Int32, next_row: Int32, best_row: Int32, best_cos: Float64)
  candidates = current_hidden_chain_candidates(hidden, labels, norms, dim, cursor_row, train_count, require_next, 1)
  return {id: -1, next_row: cursor_row, best_row: -1, best_cos: -Float64::INFINITY} if candidates[:ids].empty?

  best_row = candidates[:rows][0]
  next_row = best_row + 1 < train_count ? best_row + 1 : best_row
  {id: candidates[:ids][0], next_row: next_row, best_row: best_row, best_cos: candidates[:best_cos]}
end

def current_hidden_replay_trace(weights : ML::GGUF::Qwen35Weights,
                                prompt_ids : Array(Int32),
                                generated_ids : Array(Int32),
                                top_k : Int32) : NamedTuple(eval_samples: Int32, top_k: Int32, top1_hits: Int32, topk_hits: Int32, chain_hits: Int32, chain_steps: Int32, chain_topk_hits: Int32, chain_topk_steps: Int32, seed1_topk_hits: Int32, seed1_topk_steps: Int32, seed1_ctx_hits: Int32, seed1_ctx_steps: Int32, seed2_ctx_hits: Int32, seed2_ctx_steps: Int32, collect_ms: Float64, proposal_ms: Float64, avg_best_cos: Float64, top1_rate: Float64, topk_rate: Float64, chain_rate: Float64, chain_topk_rate: Float64, seed1_topk_rate: Float64, seed1_ctx_rate: Float64, seed2_ctx_rate: Float64, exact_ids: Array(Int32), chain_ids: Array(Int32), chain_topk_ids: Array(Int32), seed1_topk_ids: Array(Int32), seed1_ctx_ids: Array(Int32), seed2_ctx_ids: Array(Int32))
  raise ArgumentError.new("current_hidden_replay_trace needs non-empty prompt_ids") if prompt_ids.empty?
  raise ArgumentError.new("current_hidden_replay_trace top_k must be positive") unless top_k > 0

  return {
    eval_samples: 0,
    top_k:        top_k,
    top1_hits:    0,
    topk_hits:    0,
    chain_hits:   0,
    chain_steps:  0,
    chain_topk_hits:  0,
    chain_topk_steps: 0,
    seed1_topk_hits:  0,
    seed1_topk_steps: 0,
    seed1_ctx_hits:  0,
    seed1_ctx_steps: 0,
    seed2_ctx_hits:  0,
    seed2_ctx_steps: 0,
    collect_ms:   0.0,
    proposal_ms:  0.0,
    avg_best_cos: 0.0,
    top1_rate:    0.0,
    topk_rate:    0.0,
    chain_rate:   0.0,
    chain_topk_rate: 0.0,
    seed1_topk_rate: 0.0,
    seed1_ctx_rate: 0.0,
    seed2_ctx_rate: 0.0,
    exact_ids:    [] of Int32,
    chain_ids:    [] of Int32,
    chain_topk_ids: [] of Int32,
    seed1_topk_ids: [] of Int32,
    seed1_ctx_ids: [] of Int32,
    seed2_ctx_ids: [] of Int32,
  } if generated_ids.empty?

  hp = weights.hparams
  full_ids = prompt_ids.dup
  full_ids.concat(generated_ids)
  trace_state = ML::GGUF::Qwen35CPU::State.new(hp, max_seq: full_ids.size + 4)

  t_collect = Time.instant
  pair = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(weights, full_ids, 0, trace_state)
  collect_ms = (Time.instant - t_collect).total_milliseconds

  hidden = pair[:hidden]
  labels = pair[:top1s].map { |row| row[0] }
  dim = hp.n_embd
  rows = labels.size
  train_count = prompt_ids.size
  norms = Array(Float64).new(rows) { |row| hidden_row_norm_spec(hidden, row, dim) }

  top1_hits = 0
  topk_hits = 0
  best_cosines = [] of Float64
  exact_ids = [] of Int32
  chain_ids = [] of Int32
  chain_hits = 0
  chain_steps = 0
  chain_cursor = train_count - 1
  chain_topk_ids = [] of Int32
  chain_topk_hits = 0
  chain_topk_steps = 0
  chain_topk_cursor = train_count - 1
  seed1_topk_ids = [] of Int32
  seed1_topk_hits = 0
  seed1_topk_steps = 0
  seed1_topk_cursor = train_count
  seed1_ctx_ids = [] of Int32
  seed1_ctx_hits = 0
  seed1_ctx_steps = 0
  seed1_ctx_cursor = train_count
  seed1_ctx_emitted = generated_ids[0, Math.min(generated_ids.size, 1)]
  seed2_ctx_ids = [] of Int32
  seed2_ctx_hits = 0
  seed2_ctx_steps = 0
  seed2_ctx_cursor = train_count + 1
  seed2_ctx_emitted = generated_ids[0, Math.min(generated_ids.size, 2)]

  t_probe = Time.instant
  generated_ids.each_with_index do |exact_id, step|
    eval_row = train_count - 1 + step
    break if eval_row >= rows

    proposal = current_hidden_trace_topk_for_row(hidden, labels, norms, dim, eval_row, train_count, top_k)
    ids = proposal[:ids]
    exact_ids << exact_id
    top1_hits += 1 if ids[0]? == exact_id
    topk_hits += 1 if ids.includes?(exact_id)
    best_cosines << proposal[:best_cos]

    if chain_cursor >= 0 && chain_cursor < train_count
      chain = current_hidden_chain_step(hidden, labels, norms, dim, chain_cursor, train_count, step < generated_ids.size - 1)
      if chain[:id] >= 0
        chain_ids << chain[:id]
        chain_hits += 1 if chain[:id] == exact_id
        chain_steps += 1
        chain_cursor = chain[:next_row]
      end
    end

    if chain_topk_cursor >= 0 && chain_topk_cursor < train_count
      candidates = current_hidden_chain_candidates(hidden, labels, norms, dim, chain_topk_cursor, train_count, step < generated_ids.size - 1, top_k)
      unless candidates[:ids].empty?
        exact_index = candidates[:ids].index(exact_id)
        chosen_index = exact_index || 0
        chosen_id = candidates[:ids][chosen_index]
        chosen_row = candidates[:rows][chosen_index]
        chain_topk_ids << chosen_id
        chain_topk_hits += 1 if exact_index
        chain_topk_steps += 1
        chain_topk_cursor = chosen_row + 1 < train_count ? chosen_row + 1 : chosen_row
      end
    end

    if step > 0 && seed1_topk_cursor >= 0 && seed1_topk_cursor < rows
      candidates = current_hidden_chain_candidates(hidden, labels, norms, dim, seed1_topk_cursor, train_count, step < generated_ids.size - 1, top_k)
      unless candidates[:ids].empty?
        exact_index = candidates[:ids].index(exact_id)
        chosen_index = exact_index || 0
        chosen_id = candidates[:ids][chosen_index]
        chosen_row = candidates[:rows][chosen_index]
        seed1_topk_ids << chosen_id
        seed1_topk_hits += 1 if exact_index
        seed1_topk_steps += 1
        seed1_topk_cursor = chosen_row + 1 < train_count ? chosen_row + 1 : chosen_row
      end
    end

    if step > 0 && seed1_ctx_cursor >= 0 && seed1_ctx_cursor < rows
      candidates = current_hidden_chain_candidate_rows(hidden, labels, norms, dim, seed1_ctx_cursor, train_count, step < generated_ids.size - 1, top_k)
      if chosen = choose_current_hidden_context_candidate(prompt_ids, seed1_ctx_emitted, candidates)
        seed1_ctx_ids << chosen[:id]
        seed1_ctx_hits += 1 if chosen[:id] == exact_id
        seed1_ctx_steps += 1
        seed1_ctx_emitted << chosen[:id]
        seed1_ctx_cursor = chosen[:row] + 1 < train_count ? chosen[:row] + 1 : chosen[:row]
      end
    end

    if step > 1 && seed2_ctx_cursor >= 0 && seed2_ctx_cursor < rows
      candidates = current_hidden_chain_candidate_rows(hidden, labels, norms, dim, seed2_ctx_cursor, train_count, step < generated_ids.size - 1, top_k)
      if chosen = choose_current_hidden_context_candidate(prompt_ids, seed2_ctx_emitted, candidates)
        seed2_ctx_ids << chosen[:id]
        seed2_ctx_hits += 1 if chosen[:id] == exact_id
        seed2_ctx_steps += 1
        seed2_ctx_emitted << chosen[:id]
        seed2_ctx_cursor = chosen[:row] + 1 < train_count ? chosen[:row] + 1 : chosen[:row]
      end
    end
  end
  proposal_ms = (Time.instant - t_probe).total_milliseconds

  eval_samples = exact_ids.size
  {
    eval_samples: eval_samples,
    top_k:        top_k,
    top1_hits:    top1_hits,
    topk_hits:    topk_hits,
    chain_hits:   chain_hits,
    chain_steps:  chain_steps,
    chain_topk_hits:  chain_topk_hits,
    chain_topk_steps: chain_topk_steps,
    seed1_topk_hits:  seed1_topk_hits,
    seed1_topk_steps: seed1_topk_steps,
    seed1_ctx_hits:  seed1_ctx_hits,
    seed1_ctx_steps: seed1_ctx_steps,
    seed2_ctx_hits:  seed2_ctx_hits,
    seed2_ctx_steps: seed2_ctx_steps,
    collect_ms:   collect_ms,
    proposal_ms:  proposal_ms,
    avg_best_cos: best_cosines.empty? ? 0.0 : best_cosines.sum / best_cosines.size,
    top1_rate:    eval_samples > 0 ? 100.0 * top1_hits / eval_samples : 0.0,
    topk_rate:    eval_samples > 0 ? 100.0 * topk_hits / eval_samples : 0.0,
    chain_rate:   chain_steps > 0 ? 100.0 * chain_hits / chain_steps : 0.0,
    chain_topk_rate: chain_topk_steps > 0 ? 100.0 * chain_topk_hits / chain_topk_steps : 0.0,
    seed1_topk_rate: seed1_topk_steps > 0 ? 100.0 * seed1_topk_hits / seed1_topk_steps : 0.0,
    seed1_ctx_rate: seed1_ctx_steps > 0 ? 100.0 * seed1_ctx_hits / seed1_ctx_steps : 0.0,
    seed2_ctx_rate: seed2_ctx_steps > 0 ? 100.0 * seed2_ctx_hits / seed2_ctx_steps : 0.0,
    exact_ids:    exact_ids,
    chain_ids:    chain_ids,
    chain_topk_ids: chain_topk_ids,
    seed1_topk_ids: seed1_topk_ids,
    seed1_ctx_ids: seed1_ctx_ids,
    seed2_ctx_ids: seed2_ctx_ids,
  }
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

  features["candidate_unique_ratio"] = ML::GGUF::NgramDraft.unique_ratio(ids)
  features["candidate_pair_unique_ratio"] = ML::GGUF::NgramDraft.pair_unique_ratio(ids)
  features["candidate_entropy_norm"] = ML::GGUF::NgramDraft.entropy_norm(ids)

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

def ascii_alpha?(ch : Char) : Bool
  (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
end

def ascii_digit?(ch : Char) : Bool
  ch >= '0' && ch <= '9'
end

def ascii_alnum?(ch : Char) : Bool
  ascii_alpha?(ch) || ascii_digit?(ch)
end

def all_chars?(text : String, &block : Char -> Bool) : Bool
  return false if text.empty?
  text.each_char do |ch|
    return false unless yield ch
  end
  true
end

def add_candidate_token_class_features(features : Hash(String, Float64),
                                       ids : Array(Int32),
                                       tokenizer : ML::GGUF::Qwen35Tokenizer)
  return if ids.empty?

  newline = 0
  single_letter = 0
  word_like = 0
  numeric = 0
  punct_like = 0
  non_ascii = 0

  ids.each do |id|
    raw = tokenizer.decode_single(id)
    newline += 1 if raw.includes?('\n') || raw.includes?('\r')
    piece = raw.strip
    if piece.each_char.any? { |ch| ch.ord > 127 }
      non_ascii += 1
      next
    end

    if all_chars?(piece) { |ch| ascii_alpha?(ch) }
      if piece.size == 1
        single_letter += 1
      else
        word_like += 1
      end
    elsif all_chars?(piece) { |ch| ascii_digit?(ch) }
      numeric += 1
    elsif !piece.empty? && all_chars?(piece) { |ch| !ascii_alnum?(ch) }
      punct_like += 1
    end
  end

  denom = ids.size.to_f
  features["candidate_newline_token_ratio"] = newline.to_f / denom
  features["candidate_single_letter_ratio"] = single_letter.to_f / denom
  features["candidate_word_like_ratio"] = word_like.to_f / denom
  features["candidate_numeric_ratio"] = numeric.to_f / denom
  features["candidate_punct_like_ratio"] = punct_like.to_f / denom
  features["candidate_non_ascii_ratio"] = non_ascii.to_f / denom
end

private record PrefillPolicyHint,
  policy : String,
  score : Float64,
  reason : String,
  features : Hash(String, Float64)

def add_policy_hint_features(features : Hash(String, Float64), hint : PrefillPolicyHint) : Nil
  features["policy_hint_score"] = hint.score
  features["policy_hint_is_ngram"] = hint.policy == "ngram" ? 1.0 : 0.0
  features["policy_hint_is_neural"] = hint.policy == "neural" ? 1.0 : 0.0
  features["policy_hint_is_target_only"] = hint.policy == "target_only" ? 1.0 : 0.0
  features["policy_hint_features_present"] = hint.features.empty? ? 0.0 : 1.0
  hint.features.each do |name, value|
    features[name] = value if name.starts_with?("prefill_")
  end
end

def ngram_router_features(candidates : Array(Int32),
                          generated_before : Int32,
                          match_len : Int32,
                          ngram_max : Int32,
                          ngram_disabled_before : Bool,
                          verify_mode : String,
                          draft_model_id : String,
                          prompt_category : String,
                          tokenizer : ML::GGUF::Qwen35Tokenizer,
                          policy_hint : PrefillPolicyHint? = nil) : Hash(String, Float64)
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
  add_candidate_token_class_features(features, candidates, tokenizer)
  features["kind=ngram"] = 1.0
  features["verify=#{verify_mode}"] = 1.0
  features["draft=#{draft_model_id}"] = 1.0
  features["category=#{prompt_category}"] = 1.0 unless prompt_category.empty?
  add_policy_hint_features(features, policy_hint) if policy_hint
  features
end

def ngram_candidate_feature_dump(candidates : Array(Int32),
                                 match_len : Int32,
                                 ngram_max : Int32,
                                 tokenizer : ML::GGUF::Qwen35Tokenizer) : Hash(String, Float64)
  features = Hash(String, Float64).new(0.0)
  features["ngram_match_ratio"] = ngram_max > 0 ? match_len.clamp(0, ngram_max).to_f / ngram_max : 0.0
  add_candidate_features(features, candidates)
  add_candidate_token_class_features(features, candidates, tokenizer)
  features
end

def ngram_corridor_gate_pass?(candidates : Array(Int32),
                              features : Hash(String, Float64),
                              match_len : Int32,
                              min_size : Int32,
                              match_len_min : Int32,
                              lag4_min : Float64,
                              lag8_min : Float64,
                              entropy_max : Float64) : Bool
  ML::GGUF::NgramDraft.corridor_candidate_shape?(candidates,
    match_len: match_len,
    min_size: min_size,
    match_len_min: match_len_min,
    lag4_min: lag4_min,
    lag8_min: lag8_min,
    entropy_max: entropy_max)
end

def prompt_marker_features(prompt : String) : Hash(String, Float64)
  lower = prompt.downcase
  features = Hash(String, Float64).new(0.0)
  code_like = lower.includes?("def ") ||
              lower.includes?("function ") ||
              lower.includes?("class ") ||
              lower.includes?("import ") ||
              lower.includes?("return ") ||
              prompt.includes?("```") ||
              prompt.includes?("=>") ||
              prompt.includes?("::")
  math_like = lower.includes?("prove") ||
              lower.includes?("solve") ||
              lower.includes?("integral") ||
              lower.includes?("derivative") ||
              lower.includes?("equation") ||
              lower.includes?("sqrt") ||
              prompt.includes?("=")
  structured_like = prompt.includes?("{") ||
                    prompt.includes?("[") ||
                    prompt.includes?("|") ||
                    prompt.includes?("- ") ||
                    prompt.includes?(":") ||
                    lower.includes?("json") ||
                    lower.includes?("yaml") ||
                    lower.includes?("table")
  features["prefill_marker_code_like"] = code_like ? 1.0 : 0.0
  features["prefill_marker_math_like"] = math_like ? 1.0 : 0.0
  features["prefill_marker_structured_like"] = structured_like ? 1.0 : 0.0
  features["prefill_marker_newline_count_over_16"] = prompt.count('\n').clamp(0, 16).to_f / 16.0
  features
end

def prefill_policy_hint(prompt : String,
                        prompt_ids : Array(Int32),
                        tokenizer : ML::GGUF::Qwen35Tokenizer,
                        target_next : Int32,
                        draft_next : Int32,
                        ngram_gamma : Int32,
                        ngram_min : Int32,
                        ngram_max : Int32,
                        ngram_recursive : Bool,
                        ngram_risk_min_size : Int32) : PrefillPolicyHint
  features = Hash(String, Float64).new(0.0)
  features["prefill_prompt_tokens_over_128"] = prompt_ids.size.clamp(0, 512).to_f / 128.0
  features["prefill_prompt_bytes_over_1024"] = prompt.bytesize.clamp(0, 4096).to_f / 1024.0
  features["prefill_target_draft_top1_agree"] = target_next == draft_next ? 1.0 : 0.0

  prompt_token_features = Hash(String, Float64).new(0.0)
  add_candidate_token_class_features(prompt_token_features, prompt_ids, tokenizer)
  prompt_token_features.each do |name, value|
    features["prefill_prompt_#{name.sub("candidate_", "")}"] = value
  end
  prompt_marker_features(prompt).each { |name, value| features[name] = value }

  max_candidates = Math.min(ngram_gamma, 32)
  ngram_candidates = if prompt_ids.empty?
                       [] of Int32
                     else
                       ML::GGUF::NgramDraft.candidates(prompt_ids, max_candidates, ngram_max, ngram_min, recursive: ngram_recursive)
                     end
  match_len = ML::GGUF::NgramDraft.match_len(prompt_ids, ngram_max, ngram_min)
  ngram_risky = ML::GGUF::NgramDraft.risky_candidate_shape?(ngram_candidates, ngram_risk_min_size, match_len)
  features["prefill_ngram_candidates_over_32"] = ngram_candidates.size.clamp(0, 32).to_f / 32.0
  features["prefill_ngram_match_ratio"] = ngram_max > 0 ? match_len.clamp(0, ngram_max).to_f / ngram_max : 0.0
  features["prefill_ngram_risky"] = ngram_risky ? 1.0 : 0.0

  if ngram_candidates.size >= 8 && !ngram_risky
    score = 0.55 + 0.25 * features["prefill_ngram_candidates_over_32"] + 0.15 * features["prefill_ngram_match_ratio"]
    return PrefillPolicyHint.new("ngram", Math.min(score, 0.95), "prefill_repeat_candidate_#{ngram_candidates.size}_match_#{match_len}", features)
  end

  if features["prefill_marker_code_like"] > 0.0 || features["prefill_marker_math_like"] > 0.0
    return PrefillPolicyHint.new("target_only", 0.65, "code_or_math_marker_without_safe_repeat", features)
  end

  if features["prefill_target_draft_top1_agree"] > 0.0
    return PrefillPolicyHint.new("neural", 0.62, "target_draft_prefill_top1_agree", features)
  end

  if features["prefill_marker_structured_like"] > 0.0 && ngram_candidates.size >= 4 && !ngram_risky
    return PrefillPolicyHint.new("ngram", 0.55, "structured_prompt_with_short_repeat_candidate_#{ngram_candidates.size}", features)
  end

  PrefillPolicyHint.new("target_only", 0.50, "no_strong_prefill_spec_signal", features)
end

private class CycleDump
  include JSON::Serializable

  property prompt_hash : String
  property target_model : String
  property draft_model : String
  property kind : String
  property policy : String
  property verify_mode : String
  property prompt_category : String = ""
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
  property proposal_ms : Float64 = 0.0
  property accept_scan_ms : Float64 = 0.0
  property commit_ms : Float64 = 0.0
  property target_replay_ms : Float64 = 0.0
  property draft_ms : Float64
  property target_verify_ms : Float64
  property target_backup_ms : Float64
  property draft_backup_ms : Float64
  property draft_resync_ms : Float64
  property wall_ms : Float64
  property expected_gain_ms : Float64?
  property router_score : Float64?
  property router_candidate_count : Int32 = 0
  property candidate_features : Hash(String, Float64)?
  property policy_hint : String?
  property policy_hint_score : Float64?
  property policy_hint_reason : String?
  property policy_hint_features : Hash(String, Float64)?

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
                 @expected_gain_ms : Float64? = nil,
                 @router_score : Float64? = nil)
  end
end

def attach_policy_hint(record : CycleDump, hint : PrefillPolicyHint) : Nil
  record.policy_hint = hint.policy
  record.policy_hint_score = hint.score
  record.policy_hint_reason = hint.reason
  record.policy_hint_features = hint.features
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
ngram_source_prefix_checked = false
ngram_source_prefix_match = true
if ngram_enabled && ngram_replay_start >= 0
  raise ArgumentError.new("--ngram-replay-start out of source history range") unless ngram_replay_start < ngram_source_history.size
  if ngram_source_prefix_gate
    source_prefix_start = ngram_replay_start - prompt_ids.size
    ngram_source_prefix_checked = true
    ngram_source_prefix_match = source_prefix_start >= 0 &&
                                 source_prefix_start + prompt_ids.size <= ngram_source_history.size &&
                                 ngram_source_history[source_prefix_start, prompt_ids.size] == prompt_ids
  end
end
prompt_hash = fnv1a64_hex(prompt.to_slice)
target_model_id = File.basename(target_path)
draft_model_id = File.basename(draft_path)
router_model = router_model_path ? SpecRouterModel.load(router_model_path.not_nil!) : nil
cycle_dumps = [] of CycleDump

puts "Loaded in #{load_s.round(2)}s"
puts "target: layers=#{target.hparams.n_layer} dim=#{target.hparams.n_embd} vocab=#{target.output.out_dim}"
puts "draft:  layers=#{draft.hparams.n_layer} dim=#{draft.hparams.n_embd} vocab=#{draft.output.out_dim}"
puts "prompt tokens=#{prompt_ids.size} prompt_hash=#{prompt_hash} prompt_category=#{prompt_category} gamma=#{gamma} max_gamma=#{max_gamma} adaptive=#{adaptive_gamma} adaptive_regrow=#{adaptive_regrow} full_accept_streak=#{adaptive_full_accept_streak} fast_regrow_min_gamma=#{adaptive_fast_regrow_min_gamma} bootstrap_gamma=#{adaptive_bootstrap_gamma} bootstrap_streak=#{adaptive_bootstrap_streak} target_only=#{target_only} ngram=#{ngram_enabled} ngram_gamma=#{ngram_gamma} ngram_min=#{ngram_min} ngram_max=#{ngram_max} ngram_min_candidates=#{ngram_min_candidates} ngram_stage_min=#{ngram_stage_min} ngram_probe_gate=#{ngram_probe_gate} ngram_probe_min=#{ngram_probe_min} ngram_risk_gate=#{ngram_risk_gate} ngram_corridor_gate=#{ngram_corridor_gate} ngram_corridor_min_size=#{ngram_corridor_min_size} ngram_corridor_match_len_min=#{ngram_corridor_match_len_min} ngram_corridor_lag4_min=#{ngram_corridor_lag4_min} ngram_corridor_lag8_min=#{ngram_corridor_lag8_min} ngram_corridor_entropy_max=#{ngram_corridor_entropy_max} ngram_risk_min_size=#{ngram_risk_min_size} ngram_recursive=#{ngram_recursive} ngram_disable_after_reject=#{ngram_disable_after_reject} ngram_replay_on_reject=#{ngram_replay_on_reject} ngram_target_only=#{ngram_target_only} ngram_index=#{ngram_index_enabled} ngram_source_history=#{ngram_source_history.size} ngram_replay_start=#{ngram_replay_start} ngram_cursor_only=#{ngram_cursor_only} ngram_trusted_source=#{ngram_trusted_source} ngram_source_prefix_gate=#{ngram_source_prefix_gate} ngram_source_prefix_match=#{ngram_source_prefix_match} router_model=#{router_model_path || ""} early_reject=#{early_reject_enabled} single_fast=#{single_accept_fast_enabled} plain_fallback=#{plain_fallback_enabled} fallback_gamma=#{plain_fallback_gamma} skip_draft_before_fallback=#{skip_draft_before_fallback_enabled} skip_draft_backup_before_fallback=#{skip_draft_backup_before_fallback_enabled} prepare_state=#{prepare_state_metal} warm_verifier=#{warm_verifier} stage_gate=#{stage_gate} n_gen=#{n_gen} verify=#{verify_mode} allow_guarded_verifier=#{allow_guarded_verifier} dump_cycles=#{dump_cycles_path || ""} dump_token_ids=#{dump_cycle_token_ids} current_hidden_trace=#{current_hidden_trace} current_hidden_topk=#{current_hidden_trace_topk} current_hidden_tree=#{current_hidden_tree} current_hidden_tree_depth=#{current_hidden_tree_depth} current_hidden_tree_width=#{current_hidden_tree_width} current_hidden_ctx=#{current_hidden_ctx} current_hidden_ctx_seed=#{current_hidden_ctx_seed} current_hidden_ctx_depth=#{current_hidden_ctx_depth} current_hidden_ctx_width=#{current_hidden_ctx_width}"

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
current_hidden_tree_prepare_ms = 0.0
current_hidden_tree_hidden = [] of Float32
current_hidden_tree_labels = [] of Int32
current_hidden_tree_norms = [] of Float64
current_hidden_tree_dim = target.hparams.n_embd
current_hidden_tree_train_count = prompt_ids.size
if current_hidden_tree || current_hidden_ctx
  prep0 = Time.instant
  prep_state = ML::GGUF::Qwen35CPU::State.new(target.hparams, max_seq: prompt_ids.size + n_gen + 8)
  ML::GGUF::Qwen35CPU.prepare_state_metal!(prep_state, target.hparams) if prepare_state_metal
  pair = ML::GGUF::Qwen35CPU.prefill_tokens_hidden_top1s(target, prompt_ids, 0, prep_state)
  current_hidden_tree_hidden = pair[:hidden]
  current_hidden_tree_labels = pair[:top1s].map { |row| row[0] }
  current_hidden_tree_norms = Array(Float64).new(current_hidden_tree_labels.size) { |row| hidden_row_norm_spec(current_hidden_tree_hidden, row, current_hidden_tree_dim) }
  current_hidden_tree_prepare_ms = (Time.instant - prep0).total_milliseconds
end
policy_hint = prefill_policy_hint(
  prompt, prompt_ids, tok, target_next, draft_next,
  ngram_gamma, ngram_min, ngram_max, ngram_recursive, ngram_risk_min_size)
puts "policy_hint=#{policy_hint.policy} score=#{policy_hint.score.round(4)} reason=#{policy_hint.reason}"
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
ngram_history = ngram_index_enabled ? ML::GGUF::NgramDraft::IndexedHistory.new(history, ngram_max, ngram_min) : nil
ngram_replay_cursor = (ngram_enabled && ngram_replay_start >= 0 && (!ngram_source_prefix_gate || ngram_source_prefix_match)) ? ngram_replay_start : nil.as(Int32?)
ngram_replay_limit = ngram_source_history.size
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
ngram_corridor_skips = 0
ngram_cursor_hits = 0
ngram_cursor_accepts = 0
ngram_cursor_rejects = 0
ngram_cursor_serial_advances = 0
ngram_cursor_serial_drops = 0
ngram_router_score_sum = 0.0
target_verify_ms = 0.0
proposal_ms = 0.0
accept_scan_ms = 0.0
commit_ms = 0.0
draft_ms = 0.0
target_backup_ms = 0.0
target_replay_ms = 0.0
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
target_only_tokens = 0
ngram_target_only_tokens = 0
draft_skips_before_fallback = 0
draft_backup_skips = 0
current_hidden_tree_seeded = false
current_hidden_tree_cursor = current_hidden_tree_train_count - 1
current_hidden_tree_cycles = 0
current_hidden_tree_proposed = 0
current_hidden_tree_accepted = 0
current_hidden_tree_rejects = 0
current_hidden_tree_fallback_tokens = 0
current_hidden_tree_fork_ms = 0.0
current_hidden_tree_commit_ms = 0.0
current_hidden_ctx_seeded_tokens = 0
current_hidden_ctx_cursor = current_hidden_tree_train_count - 1
current_hidden_ctx_cycles = 0
current_hidden_ctx_proposed = 0
current_hidden_ctx_accepted = 0
current_hidden_ctx_rejects = 0
current_hidden_ctx_fallback_tokens = 0
current_hidden_ctx_select_ms = 0.0
current_hidden_ctx_commit_ms = 0.0

wall0 = Time.instant
while generated_ids.size < n_gen
  history_size_before = generated_ids.size
  cycle_proposal_ms = 0.0
  cycle_router_score = nil.as(Float64?)
  cycle_router_candidate_count = 0
  cycle_candidate_features = nil.as(Hash(String, Float64)?)
  ngram_pending_replay_cursor = nil.as(Int32?)
  ngram_from_trusted_source = false

  if current_hidden_tree
    cycle_wall0 = Time.instant
    cycle_start_pos = pos
    cycle_target_verify0 = target_verify_ms
    cycle_commit0 = commit_ms
    cycle_tree_fork0 = current_hidden_tree_fork_ms
    cycle_tree_commit0 = current_hidden_tree_commit_ms
    cycle_tree_proposed0 = current_hidden_tree_proposed
    cycle_tree_accepted0 = current_hidden_tree_accepted
    cycle_tree_reject0 = current_hidden_tree_rejects
    cycle_tree_fallback0 = current_hidden_tree_fallback_tokens

    if !current_hidden_tree_seeded
      generated_ids << target_next
      if generated_ids.size < n_gen
        tv0 = Time.instant
        hidden = ML::GGUF::Qwen35CPU.forward_hidden(target, target_next, pos, target_state)
        target_next = ML::GGUF::Qwen35CPU.hidden_top1(target, hidden)[0]
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        current_hidden_tree_hidden.concat(hidden)
        current_hidden_tree_norms << hidden_vector_norm_spec(hidden)
        current_hidden_tree_cursor = current_hidden_tree_train_count
      end
      pos += 1
      current_hidden_tree_seeded = true
      current_hidden_tree_fallback_tokens += 1
      commit0 = Time.instant
      new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
      history.concat(new_history)
      ngram_history.try &.append(new_history)
      commit_ms += (Time.instant - commit0).total_milliseconds
      next
    end

    remaining = n_gen - generated_ids.size
    depth = Math.min(current_hidden_tree_depth, remaining)
    paths = current_hidden_tree_paths(current_hidden_tree_hidden, current_hidden_tree_labels, current_hidden_tree_norms, current_hidden_tree_dim, current_hidden_tree_cursor, current_hidden_tree_train_count, depth, current_hidden_tree_width)
    proposal_ms += 0.0
    current_hidden_tree_cycles += 1
    current_hidden_tree_proposed += paths.sum { |path| path[:ids].size }

    best_path = nil.as(NamedTuple(ids: Array(Int32), rows: Array(Int32), cursor: Int32, score: Float64)?)
    best_accept = 0
    best_full_state = nil.as(ML::GGUF::Qwen35CPU::State?)
    best_next = target_next
    paths.each do |path|
      fork0 = Time.instant
      verify_state = target_state.fork
      current_hidden_tree_fork_ms += (Time.instant - fork0).total_milliseconds
      tv0 = Time.instant
      target_nexts = target_prefill_top1s_for_future(target, path[:ids], pos, verify_state, allow_guarded_verifier, generated_ids.size, n_gen)
      target_verify_ms += (Time.instant - tv0).total_milliseconds
      expected = target_next
      accepted_prefix = 0
      path[:ids].each_with_index do |cand, i|
        break if generated_ids.size + accepted_prefix >= n_gen
        break unless cand == expected
        accepted_prefix += 1
        expected = target_nexts[i][0] if i < target_nexts.size
      end
      if accepted_prefix > best_accept
        best_accept = accepted_prefix
        best_path = path
        if accepted_prefix == path[:ids].size
          best_full_state = verify_state
          best_next = expected
        else
          best_full_state = nil
          best_next = expected
        end
      end
    end

    if best_accept > 0 && (path = best_path)
      accepted_ids = path[:ids][0, best_accept]
      generated_ids.concat(accepted_ids)
      current_hidden_tree_accepted += best_accept
      if generated_ids.size < n_gen
        commit_tree0 = Time.instant
        if best_accept == path[:ids].size && (state = best_full_state)
          target_state.copy_from!(state)
          target_next = best_next
        else
          replay = target_prefill_top1s_for_future(target, accepted_ids, cycle_start_pos, target_state, allow_guarded_verifier, generated_ids.size - accepted_ids.size, n_gen)
          target_next = replay[-1][0] unless replay.empty?
        end
        current_hidden_tree_commit_ms += (Time.instant - commit_tree0).total_milliseconds
        current_hidden_tree_cursor = path[:rows][best_accept - 1] + 1 < current_hidden_tree_train_count ? path[:rows][best_accept - 1] + 1 : path[:rows][best_accept - 1]
      end
      pos += best_accept
    else
      current_hidden_tree_rejects += 1 unless paths.empty?
      generated_ids << target_next
      if generated_ids.size < n_gen
        tv0 = Time.instant
        hidden = ML::GGUF::Qwen35CPU.forward_hidden(target, target_next, pos, target_state)
        target_next = ML::GGUF::Qwen35CPU.hidden_top1(target, hidden)[0]
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        current_hidden_tree_hidden.concat(hidden)
        current_hidden_tree_norms << hidden_vector_norm_spec(hidden)
        current_hidden_tree_cursor = current_hidden_tree_hidden.size // current_hidden_tree_dim - 1
      end
      pos += 1
      current_hidden_tree_fallback_tokens += 1
    end

    commit0 = Time.instant
    new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
    history.concat(new_history)
    ngram_history.try &.append(new_history)
    commit_ms += (Time.instant - commit0).total_milliseconds
    if dump_cycles_path
      record = CycleDump.new(
        prompt_hash, target_model_id, draft_model_id,
        "current_hidden_tree", "current_hidden_tree", verify_mode,
        cycle_start_pos, history_size_before, generated_ids.size - history_size_before,
        current_hidden_tree_width, current_hidden_tree_proposed - cycle_tree_proposed0, current_hidden_tree_accepted - cycle_tree_accepted0, best_accept > 0 ? -1 : 0,
        0, ngram_min, ngram_max, ngram_recursive,
        false, false,
        token_ids_hash(best_path.try(&.[:ids]) || [] of Int32), dump_cycle_token_ids ? (best_path.try(&.[:ids]) || [] of Int32) : nil,
        0.0,
        target_verify_ms - cycle_target_verify0,
        target_backup_ms,
        0.0,
        0.0,
        (Time.instant - cycle_wall0).total_milliseconds)
      record.commit_ms = commit_ms - cycle_commit0
      record.prompt_category = prompt_category
      record.candidate_features = {
        "tree_depth" => depth.to_f64,
        "tree_width" => current_hidden_tree_width.to_f64,
        "tree_paths" => paths.size.to_f64,
        "tree_best_accept" => best_accept.to_f64,
        "tree_fork_ms" => current_hidden_tree_fork_ms - cycle_tree_fork0,
        "tree_commit_ms" => current_hidden_tree_commit_ms - cycle_tree_commit0,
        "tree_rejects" => (current_hidden_tree_rejects - cycle_tree_reject0).to_f64,
        "tree_fallback_tokens" => (current_hidden_tree_fallback_tokens - cycle_tree_fallback0).to_f64,
      }
      cycle_dumps << record
    end
    next
  end

  if !target_only && ngram_enabled && !ngram_disabled
    proposal0 = Time.instant
    ngram_limit = Math.min(ngram_gamma, n_gen - generated_ids.size)
    match_len = 0
    ngram_candidates = [] of Int32
    if cursor = ngram_replay_cursor
      replay_count = Math.min(ngram_limit, ngram_replay_limit - cursor)
      if replay_count > 0
        ngram_candidates = ngram_source_history[cursor, replay_count]
        if ngram_min_candidates > 0 && ngram_candidates.size < ngram_min_candidates
          ngram_candidates = [] of Int32
        else
          ngram_pending_replay_cursor = cursor + ngram_candidates.size
          ngram_from_trusted_source = ngram_trusted_source && (!ngram_source_prefix_gate || ngram_source_prefix_match)
          ngram_cursor_hits += 1
          match_len = ngram_max
        end
      end
    end
    if ngram_candidates.empty? && !ngram_cursor_only
      if index = ngram_history
        if span = index.candidate_span(
             ngram_limit,
             recursive: ngram_recursive,
             min_candidates: ngram_min_candidates)
          ngram_candidates = span.ids
          match_len = span.match_len
        else
          match_len = index.match_len
        end
      else
        ngram_candidates = ML::GGUF::NgramDraft.candidates(
          history,
          ngram_limit,
          ngram_max,
          ngram_min,
          recursive: ngram_recursive,
          min_candidates: ngram_min_candidates)
        match_len = ML::GGUF::NgramDraft.match_len(history, ngram_max, ngram_min)
      end
    end
    cycle_candidate_features = ngram_candidates.empty? ? nil : ngram_candidate_feature_dump(ngram_candidates, match_len, ngram_max, tok)
    if ngram_corridor_gate && !ngram_from_trusted_source && (features = cycle_candidate_features) && !ngram_corridor_gate_pass?(
         ngram_candidates,
         features,
         match_len,
         ngram_corridor_min_size,
         ngram_corridor_match_len_min,
         ngram_corridor_lag4_min,
         ngram_corridor_lag8_min,
         ngram_corridor_entropy_max)
      ngram_corridor_skips += 1
      ngram_candidates = [] of Int32
    end
    if ngram_risk_gate && !ngram_from_trusted_source && ML::GGUF::NgramDraft.risky_candidate_shape?(ngram_candidates, ngram_risk_min_size, match_len)
      ngram_disabled = true
      ngram_candidates = [] of Int32
    end
    if router_model && !ngram_candidates.empty?
      cycle_router_candidate_count = ngram_candidates.size
      score = router_model.not_nil!.score(ngram_router_features(
        ngram_candidates, generated_ids.size, match_len, ngram_max, ngram_disabled, verify_mode, draft_model_id, prompt_category, tok, policy_hint))
      cycle_router_score = score
      ngram_router_checks += 1
      ngram_router_score_sum += score
      threshold = router_model.not_nil!.threshold
      if long_threshold = router_long_threshold
        threshold = Math.max(threshold, long_threshold) if ngram_candidates.size >= router_long_min
      end
      if score < threshold
        ngram_router_skips += 1
        ngram_candidates = [] of Int32
      end
    end
    cycle_proposal_ms = (Time.instant - proposal0).total_milliseconds
    proposal_ms += cycle_proposal_ms

    unless ngram_candidates.empty?
      cycle_wall0 = Time.instant
      cycle_draft0 = draft_ms
      cycle_target_verify0 = target_verify_ms
      cycle_target_backup0 = target_backup_ms
      cycle_draft_backup0 = draft_backup_ms
      cycle_draft_resync0 = draft_resync_ms
      cycle_accept_scan0 = accept_scan_ms
      cycle_commit0 = commit_ms
      cycle_target_replay0 = target_replay_ms
      ngram_disabled_before = ngram_disabled
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
      probe_ngram = ngram_probe_gate > 0 &&
                    ngram_candidates.size >= ngram_probe_min &&
                    ngram_candidates.size > ngram_probe_gate
      while ngram_offset < ngram_candidates.size
        remaining = ngram_candidates.size - ngram_offset
        stage_len = if probe_ngram && ngram_offset == 0
                      Math.min(ngram_probe_gate, remaining)
                    elsif probe_ngram
                      remaining
                    elsif stage_ngram
                      Math.min(stage_gate, remaining)
                    else
                      remaining
                    end
        stage_len = Math.min(stage_len, n_gen - generated_ids.size)
        break if stage_len <= 0
        if probe_ngram && ngram_offset > 0 && ngram_candidates[ngram_offset] != target_next
          stage_start_pos = cycle_start_pos + ngram_offset
          correction = target_next
          generated_ids << correction
          correction_or_accepted << correction
          rejected = true
          reject_index = ngram_offset
          ngram_disabled = true if ngram_disable_after_reject
          if generated_ids.size < n_gen
            tv1 = Time.instant
            corrected = target_prefill_top1s_for_future(target, [correction], stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size - 1, n_gen)
            target_verify_ms += (Time.instant - tv1).total_milliseconds
            target_next = corrected[-1][0] unless corrected.empty?
          end
          pos += 1
          break
        end
        stage_candidates = ngram_candidates[ngram_offset, stage_len]
        stage_start_pos = cycle_start_pos + ngram_offset
        stage_correction_or_accepted = [] of Int32

        unless ngram_replay_on_reject
          tb0 = Time.instant
          target_backup_state.copy_from!(target_state)
          target_backup_ms += (Time.instant - tb0).total_milliseconds
        end
        tv0 = Time.instant
        target_nexts = target_prefill_top1s_for_future(target, stage_candidates, stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size, n_gen)
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        if trace
          puts "ngram_cycle=#{ngram_cycles} stage_offset=#{ngram_offset} pos=#{stage_start_pos} expected0=#{target_next} candidates=#{stage_candidates.inspect} target_nexts=#{target_nexts.map(&.[0]).inspect}"
        end

        expected = target_next
        accept0 = Time.instant
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
        accept_scan_ms += (Time.instant - accept0).total_milliseconds

        if rejected
          ngram_disabled = true if ngram_disable_after_reject
          if generated_ids.size < n_gen
            if ngram_replay_on_reject
              tr0 = Time.instant
              target_state, target_next = replay_target_state(target, prompt_ids, generated_ids, max_seq, prepare_state_metal)
              target_replay_ms += (Time.instant - tr0).total_milliseconds
            else
              target_state.copy_from!(target_backup_state)
              tv1 = Time.instant
              corrected = target_prefill_top1s_for_future(target, stage_correction_or_accepted, stage_start_pos, target_state, allow_guarded_verifier, generated_ids.size - stage_correction_or_accepted.size, n_gen)
              target_verify_ms += (Time.instant - tv1).total_milliseconds
              target_next = corrected[-1][0]
            end
          end
          pos += stage_correction_or_accepted.size
          break
        else
          target_next = target_nexts[-1][0] if generated_ids.size < n_gen
          pos += stage_correction_or_accepted.size
          ngram_offset += stage_correction_or_accepted.size
        end
      end

      unless correction_or_accepted.empty? || ngram_target_only
        pending_draft_start_pos = cycle_start_pos if pending_draft_tokens.empty?
        pending_draft_tokens.concat(correction_or_accepted)
      end
      if generated_ids.size > history_size_before
        commit0 = Time.instant
        new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
        history.concat(new_history)
        ngram_history.try &.append(new_history)
        commit_ms += (Time.instant - commit0).total_milliseconds
      end
      if ngram_pending_replay_cursor
        if rejected
          ngram_replay_cursor = nil
          ngram_cursor_rejects += 1
        else
          ngram_replay_cursor = ngram_pending_replay_cursor
          ngram_cursor_accepts += 1
        end
      end
      if dump_cycles_path
        record_candidates = dump_cycle_token_ids ? ngram_candidates.dup : nil
        record = CycleDump.new(
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
        record.proposal_ms = cycle_proposal_ms
        record.accept_scan_ms = accept_scan_ms - cycle_accept_scan0
        record.commit_ms = commit_ms - cycle_commit0
        record.target_replay_ms = target_replay_ms - cycle_target_replay0
        record.router_score = cycle_router_score
        record.router_candidate_count = cycle_router_candidate_count
        record.prompt_category = prompt_category
        record.candidate_features = cycle_candidate_features
        cycle_dumps << record
      end
      next
    end
  end

  if current_hidden_ctx && !target_only && !(ngram_enabled && ngram_target_only)
    cycle_wall0 = Time.instant
    cycle_start_pos = pos
    cycle_target_verify0 = target_verify_ms
    cycle_target_backup0 = target_backup_ms
    cycle_commit0 = commit_ms
    cycle_select0 = current_hidden_ctx_select_ms
    cycle_ctx_proposed0 = current_hidden_ctx_proposed
    cycle_ctx_accepted0 = current_hidden_ctx_accepted
    cycle_ctx_reject0 = current_hidden_ctx_rejects
    cycle_ctx_fallback0 = current_hidden_ctx_fallback_tokens
    path_ids = [] of Int32
    path_rows = [] of Int32
    accepted_prefix = 0
    rejected_path = false

    if current_hidden_ctx_seeded_tokens < current_hidden_ctx_seed
      generated_ids << target_next
      if generated_ids.size < n_gen
        tv0 = Time.instant
        hidden = ML::GGUF::Qwen35CPU.forward_hidden(target, target_next, pos, target_state)
        target_next = ML::GGUF::Qwen35CPU.hidden_top1(target, hidden)[0]
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        current_hidden_tree_hidden.concat(hidden)
        current_hidden_tree_norms << hidden_vector_norm_spec(hidden)
        current_hidden_ctx_cursor = current_hidden_tree_hidden.size // current_hidden_tree_dim - 1
      end
      pos += 1
      current_hidden_ctx_seeded_tokens += 1
      current_hidden_ctx_fallback_tokens += 1
      commit0 = Time.instant
      new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
      history.concat(new_history)
      ngram_history.try &.append(new_history)
      commit_ms += (Time.instant - commit0).total_milliseconds
      next
    end

    remaining = n_gen - generated_ids.size
    depth = Math.min(current_hidden_ctx_depth, remaining)
    select0 = Time.instant
    cursor = current_hidden_ctx_cursor
    emitted_for_select = generated_ids.dup
    best_suffix_row = -1
    best_suffix_overlap = 0
    current_hidden_tree_train_count.times do |row|
      overlap = token_suffix_overlap_at_prompt_row(prompt_ids, emitted_for_select, row)
      if overlap > best_suffix_overlap || (overlap == best_suffix_overlap && row > best_suffix_row)
        best_suffix_overlap = overlap
        best_suffix_row = row
      end
    end

    if best_suffix_row >= 0 && best_suffix_overlap >= current_hidden_ctx_seed
      direct_len = Math.min(depth, current_hidden_tree_train_count - best_suffix_row)
      direct_len.times do |i|
        row = best_suffix_row + i
        path_ids << current_hidden_tree_labels[row]
        path_rows << row
      end
    else
      candidates = current_hidden_chain_candidate_rows(
        current_hidden_tree_hidden,
        current_hidden_tree_labels,
        current_hidden_tree_norms,
        current_hidden_tree_dim,
        cursor,
        current_hidden_tree_train_count,
        depth > 1,
        current_hidden_ctx_width)
      chosen = choose_current_hidden_context_candidate(prompt_ids, emitted_for_select, candidates)
      if chosen
        corridor_len = Math.min(depth, current_hidden_tree_train_count - chosen[:row])
        corridor_len.times do |i|
          row = chosen[:row] + i
          path_ids << current_hidden_tree_labels[row]
          path_rows << row
        end
      end
    end
    elapsed_select = (Time.instant - select0).total_milliseconds
    current_hidden_ctx_select_ms += elapsed_select
    proposal_ms += elapsed_select
    current_hidden_ctx_cycles += 1
    current_hidden_ctx_proposed += path_ids.size

    if path_ids.empty?
      generated_ids << target_next
      if generated_ids.size < n_gen
        tv0 = Time.instant
        hidden = ML::GGUF::Qwen35CPU.forward_hidden(target, target_next, pos, target_state)
        target_next = ML::GGUF::Qwen35CPU.hidden_top1(target, hidden)[0]
        target_verify_ms += (Time.instant - tv0).total_milliseconds
        current_hidden_tree_hidden.concat(hidden)
        current_hidden_tree_norms << hidden_vector_norm_spec(hidden)
        current_hidden_ctx_cursor = current_hidden_tree_hidden.size // current_hidden_tree_dim - 1
      end
      pos += 1
      current_hidden_ctx_fallback_tokens += 1
    else
      tb0 = Time.instant
      target_backup_state.copy_from!(target_state)
      target_backup_ms += (Time.instant - tb0).total_milliseconds

      tv0 = Time.instant
      target_nexts = target_prefill_top1s_for_future(target, path_ids, pos, target_state, allow_guarded_verifier, generated_ids.size, n_gen)
      target_verify_ms += (Time.instant - tv0).total_milliseconds

      expected = target_next
      path_ids.each_with_index do |cand, i|
        break if generated_ids.size + accepted_prefix >= n_gen
        break unless cand == expected
        accepted_prefix += 1
        expected = target_nexts[i][0] if i < target_nexts.size
      end

      if accepted_prefix == path_ids.size
        accepted_ids = path_ids
        generated_ids.concat(accepted_ids)
        current_hidden_ctx_accepted += accepted_prefix
        current_hidden_ctx_cursor = path_rows[accepted_prefix - 1] + 1 < current_hidden_tree_train_count ? path_rows[accepted_prefix - 1] + 1 : path_rows[accepted_prefix - 1] if accepted_prefix > 0
        target_next = expected if generated_ids.size < n_gen
        pos += accepted_prefix
      elsif accepted_prefix > 0
        rejected_path = true
        current_hidden_ctx_rejects += 1
        current_hidden_ctx_accepted += accepted_prefix
        accepted_ids = path_ids[0, accepted_prefix]
        target_state.copy_from!(target_backup_state)
        generated_ids.concat(accepted_ids)
        if generated_ids.size < n_gen
          commit_ctx0 = Time.instant
          replay = target_prefill_top1s_for_future(target, accepted_ids, cycle_start_pos, target_state, allow_guarded_verifier, generated_ids.size - accepted_ids.size, n_gen)
          target_next = replay[-1][0] unless replay.empty?
          current_hidden_ctx_commit_ms += (Time.instant - commit_ctx0).total_milliseconds
        end
        current_hidden_ctx_cursor = path_rows[accepted_prefix - 1] + 1 < current_hidden_tree_train_count ? path_rows[accepted_prefix - 1] + 1 : path_rows[accepted_prefix - 1]
        pos += accepted_prefix
      else
        rejected_path = true
        current_hidden_ctx_rejects += 1
        target_state.copy_from!(target_backup_state)
        generated_ids << target_next
        if generated_ids.size < n_gen
          tv1 = Time.instant
          hidden = ML::GGUF::Qwen35CPU.forward_hidden(target, target_next, pos, target_state)
          target_next = ML::GGUF::Qwen35CPU.hidden_top1(target, hidden)[0]
          target_verify_ms += (Time.instant - tv1).total_milliseconds
          current_hidden_tree_hidden.concat(hidden)
          current_hidden_tree_norms << hidden_vector_norm_spec(hidden)
          current_hidden_ctx_cursor = current_hidden_tree_hidden.size // current_hidden_tree_dim - 1
        end
        pos += 1
        current_hidden_ctx_fallback_tokens += 1
      end
    end

    commit0 = Time.instant
    new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
    history.concat(new_history)
    ngram_history.try &.append(new_history)
    commit_ms += (Time.instant - commit0).total_milliseconds
    if dump_cycles_path
      record = CycleDump.new(
        prompt_hash, target_model_id, draft_model_id,
        "current_hidden_ctx", "current_hidden_ctx", verify_mode,
        cycle_start_pos, history_size_before, generated_ids.size - history_size_before,
        current_hidden_ctx_width, current_hidden_ctx_proposed - cycle_ctx_proposed0, current_hidden_ctx_accepted - cycle_ctx_accepted0, rejected_path ? accepted_prefix : -1,
        0, ngram_min, ngram_max, ngram_recursive,
        false, false,
        token_ids_hash(path_ids), dump_cycle_token_ids ? path_ids : nil,
        0.0,
        target_verify_ms - cycle_target_verify0,
        target_backup_ms - cycle_target_backup0,
        0.0,
        0.0,
        (Time.instant - cycle_wall0).total_milliseconds)
      record.proposal_ms = current_hidden_ctx_select_ms - cycle_select0
      record.commit_ms = commit_ms - cycle_commit0
      record.prompt_category = prompt_category
      record.candidate_features = {
        "ctx_seed" => current_hidden_ctx_seed.to_f64,
        "ctx_depth" => depth.to_f64,
        "ctx_width" => current_hidden_ctx_width.to_f64,
        "ctx_path_len" => path_ids.size.to_f64,
        "ctx_accepted" => accepted_prefix.to_f64,
        "ctx_rejects" => (current_hidden_ctx_rejects - cycle_ctx_reject0).to_f64,
        "ctx_fallback_tokens" => (current_hidden_ctx_fallback_tokens - cycle_ctx_fallback0).to_f64,
      }
      cycle_dumps << record
    end
    next
  end

  if target_only || (ngram_enabled && ngram_target_only)
    cycle_wall0 = Time.instant
    cycle_draft0 = draft_ms
    cycle_target_verify0 = target_verify_ms
    cycle_target_backup0 = target_backup_ms
    cycle_draft_backup0 = draft_backup_ms
    cycle_draft_resync0 = draft_resync_ms
    cycle_commit0 = commit_ms
    cycle_start_pos = pos
    cycle_ngram_match_len = if ngram_enabled
                              if index = ngram_history
                                index.match_len
                              else
                                ML::GGUF::NgramDraft.match_len(history, ngram_max, ngram_min)
                              end
                            else
                              0
                            end
    generated_ids << target_next
    if generated_ids.size < n_gen
      tv0 = Time.instant
      target_next = advance_next(target, target_next, pos, target_state)
      target_verify_ms += (Time.instant - tv0).total_milliseconds
    end
    pos += 1
    if target_only
      target_only_tokens += 1
    else
      ngram_target_only_tokens += 1
    end
    commit0 = Time.instant
    new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
    history.concat(new_history)
    ngram_history.try &.append(new_history)
    if ngram_enabled
      if cursor = ngram_replay_cursor
        exact_id = new_history[0]?
        if exact_id && cursor < ngram_replay_limit && ngram_source_history[cursor]? == exact_id
          ngram_replay_cursor = cursor + 1
          ngram_cursor_serial_advances += 1
        else
          ngram_replay_cursor = nil
          ngram_cursor_serial_drops += 1
        end
      end
    end
    commit_ms += (Time.instant - commit0).total_milliseconds
    if dump_cycles_path
      record = CycleDump.new(
        prompt_hash, target_model_id, draft_model_id,
        "target_only", target_only ? "target_only" : "ngram_target_only", verify_mode,
        cycle_start_pos, history_size_before, 1,
        1, 0, 0, -1,
        cycle_ngram_match_len, ngram_min, ngram_max, ngram_recursive,
        ngram_disabled, ngram_disabled,
        token_ids_hash([] of Int32), nil,
        draft_ms - cycle_draft0,
        target_verify_ms - cycle_target_verify0,
        target_backup_ms - cycle_target_backup0,
        draft_backup_ms - cycle_draft_backup0,
        draft_resync_ms - cycle_draft_resync0,
        (Time.instant - cycle_wall0).total_milliseconds)
      record.proposal_ms = cycle_proposal_ms
      record.commit_ms = commit_ms - cycle_commit0
      record.router_score = cycle_router_score
      record.router_candidate_count = cycle_router_candidate_count
      record.prompt_category = prompt_category
      record.candidate_features = cycle_candidate_features
      cycle_dumps << record
    end
    next
  end

  if plain_fallback_enabled && adaptive_gamma && !adaptive_growth_allowed && current_gamma <= plain_fallback_gamma
    cycle_wall0 = Time.instant
    cycle_draft0 = draft_ms
    cycle_target_verify0 = target_verify_ms
    cycle_target_backup0 = target_backup_ms
    cycle_draft_backup0 = draft_backup_ms
    cycle_draft_resync0 = draft_resync_ms
    cycle_commit0 = commit_ms
    cycle_start_pos = pos
    generated_ids << target_next
    if generated_ids.size < n_gen
      tv0 = Time.instant
      target_next = advance_next(target, target_next, pos, target_state)
      target_verify_ms += (Time.instant - tv0).total_milliseconds
    end
    pos += 1
    plain_fallback_tokens += 1
    commit0 = Time.instant
    new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
    history.concat(new_history)
    ngram_history.try &.append(new_history)
    commit_ms += (Time.instant - commit0).total_milliseconds
    if dump_cycles_path
      record = CycleDump.new(
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
      record.commit_ms = commit_ms - cycle_commit0
      record.prompt_category = prompt_category
      cycle_dumps << record
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
  cycle_commit0 = commit_ms
  cycle_start_pos = pos
  cycle_gamma = adaptive_gamma ? current_gamma : gamma
  draft_next_at_cycle = draft_next
  cycle_ngram_match_len = if index = ngram_history
                            index.match_len
                          else
                            ML::GGUF::NgramDraft.match_len(history, ngram_max, ngram_min)
                          end
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
    commit0 = Time.instant
    new_history = generated_ids[history_size_before, generated_ids.size - history_size_before]
    history.concat(new_history)
    ngram_history.try &.append(new_history)
    commit_ms += (Time.instant - commit0).total_milliseconds
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
    record = CycleDump.new(
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
    record.commit_ms = commit_ms - cycle_commit0
    record.prompt_category = prompt_category
    cycle_dumps << record
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
current_hidden_trace_result = current_hidden_trace ? current_hidden_replay_trace(target, prompt_ids, generated_ids, current_hidden_trace_topk) : nil

if path = dump_cycles_path
  plain_ms_per_token = plain_ms / n_gen
  cycle_dumps.each do |record|
    record.expected_gain_ms = record.generated_count * plain_ms_per_token - record.wall_ms
    attach_policy_hint(record, policy_hint)
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

accept_rate = proposed > 0 ? accepted.to_f64 / proposed.to_f64 : 1.0
tokens_s = n_gen.to_f64 / (wall_ms / 1000.0)
plain_tokens_s = n_gen.to_f64 / (plain_ms / 1000.0)

puts
puts "accept_rate=#{(accept_rate * 100.0).round(2)}% accepted=#{accepted}/#{proposed} cycles=#{cycles}"
if ngram_enabled
  ngram_rate = ngram_proposed > 0 ? (ngram_accepted.to_f64 * 100.0 / ngram_proposed.to_f64) : 0.0
  puts "ngram_stats accepted=#{ngram_accepted}/#{ngram_proposed} rate=#{ngram_rate.round(2)}% cycles=#{ngram_cycles} disabled=#{ngram_disabled} corridor_skips=#{ngram_corridor_skips} cursor_hits=#{ngram_cursor_hits} cursor_accepts=#{ngram_cursor_accepts} cursor_rejects=#{ngram_cursor_rejects} cursor_serial_advances=#{ngram_cursor_serial_advances} cursor_serial_drops=#{ngram_cursor_serial_drops} source_prefix_checked=#{ngram_source_prefix_checked} source_prefix_match=#{ngram_source_prefix_match} pending_draft=#{pending_draft_tokens.size}"
  if router_model
    avg_router_score = ngram_router_checks > 0 ? (ngram_router_score_sum / ngram_router_checks).round(4) : 0.0
    puts "ngram_router_stats checks=#{ngram_router_checks} skips=#{ngram_router_skips} threshold=#{router_model.not_nil!.threshold} avg_score=#{avg_router_score}"
  end
end
avg_gamma = cycles > 0 ? (gamma_sum.to_f64 / cycles.to_f64).round(2) : 0.0
puts "gamma_stats avg=#{avg_gamma} max_seen=#{gamma_max_seen} final=#{current_gamma} early_rejects=#{early_rejects} single_fast=#{single_accept_fast} plain_fallback=#{plain_fallback_tokens} target_only=#{target_only_tokens} ngram_target_only=#{ngram_target_only_tokens} draft_skip=#{draft_skips_before_fallback} draft_backup_skip=#{draft_backup_skips}"
puts "spec_wall=#{wall_ms.round(1)} ms (#{(wall_ms / n_gen).round(2)} ms/tok, #{tokens_s.round(2)} tok/s, verify=#{verify_mode})"
puts "plain_target_wall=#{plain_ms.round(1)} ms (#{(plain_ms / n_gen).round(2)} ms/tok, #{plain_tokens_s.round(2)} tok/s, decode_only=true)"
puts "plain_target_prefill_wall=#{plain_prefill_ms.round(1)} ms"
if trace_result = current_hidden_trace_result
  proposal_per_eval = trace_result[:eval_samples] > 0 ? trace_result[:proposal_ms] / trace_result[:eval_samples] : 0.0
  puts "current_hidden_trace top_k=#{trace_result[:top_k]} eval=#{trace_result[:eval_samples]} top1=#{trace_result[:top1_rate].round(2)}% topk=#{trace_result[:topk_rate].round(2)}% hits=#{trace_result[:top1_hits]}/#{trace_result[:topk_hits]}/#{trace_result[:eval_samples]} chain=#{trace_result[:chain_rate].round(2)}% chain_hits=#{trace_result[:chain_hits]}/#{trace_result[:chain_steps]} chain_topk=#{trace_result[:chain_topk_rate].round(2)}% chain_topk_hits=#{trace_result[:chain_topk_hits]}/#{trace_result[:chain_topk_steps]} seed1_topk=#{trace_result[:seed1_topk_rate].round(2)}% seed1_topk_hits=#{trace_result[:seed1_topk_hits]}/#{trace_result[:seed1_topk_steps]} seed1_ctx=#{trace_result[:seed1_ctx_rate].round(2)}% seed1_ctx_hits=#{trace_result[:seed1_ctx_hits]}/#{trace_result[:seed1_ctx_steps]} seed2_ctx=#{trace_result[:seed2_ctx_rate].round(2)}% seed2_ctx_hits=#{trace_result[:seed2_ctx_hits]}/#{trace_result[:seed2_ctx_steps]} collect_ms=#{trace_result[:collect_ms].round(3)} proposal_ms=#{trace_result[:proposal_ms].round(3)} proposal_ms_per_eval=#{proposal_per_eval.round(6)} avg_best_cos=#{trace_result[:avg_best_cos].round(6)} exact_ids=#{trace_result[:exact_ids].join(',')} chain_ids=#{trace_result[:chain_ids].join(',')} chain_topk_ids=#{trace_result[:chain_topk_ids].join(',')} seed1_topk_ids=#{trace_result[:seed1_topk_ids].join(',')} seed1_ctx_ids=#{trace_result[:seed1_ctx_ids].join(',')} seed2_ctx_ids=#{trace_result[:seed2_ctx_ids].join(',')} note=postrun_trace_only_not_in_wall"
end
if current_hidden_tree
  puts "current_hidden_tree_stats prepare_ms=#{current_hidden_tree_prepare_ms.round(3)} cycles=#{current_hidden_tree_cycles} proposed=#{current_hidden_tree_proposed} accepted=#{current_hidden_tree_accepted} rejects=#{current_hidden_tree_rejects} fallback_tokens=#{current_hidden_tree_fallback_tokens} fork_ms=#{current_hidden_tree_fork_ms.round(3)} commit_ms=#{current_hidden_tree_commit_ms.round(3)} depth=#{current_hidden_tree_depth} width=#{current_hidden_tree_width} note=prepare_excluded_like_prefill"
end
if current_hidden_ctx
  ctx_rate = current_hidden_ctx_proposed > 0 ? (current_hidden_ctx_accepted.to_f64 * 100.0 / current_hidden_ctx_proposed.to_f64) : 0.0
  puts "current_hidden_ctx_stats prepare_ms=#{current_hidden_tree_prepare_ms.round(3)} cycles=#{current_hidden_ctx_cycles} proposed=#{current_hidden_ctx_proposed} accepted=#{current_hidden_ctx_accepted} rate=#{ctx_rate.round(2)}% rejects=#{current_hidden_ctx_rejects} fallback_tokens=#{current_hidden_ctx_fallback_tokens} select_ms=#{current_hidden_ctx_select_ms.round(3)} commit_ms=#{current_hidden_ctx_commit_ms.round(3)} seed=#{current_hidden_ctx_seed} depth=#{current_hidden_ctx_depth} width=#{current_hidden_ctx_width} note=prepare_excluded_like_prefill"
end
puts "verifier_warmup_wall=#{verifier_warmup_ms.round(1)} ms"
puts "time_breakdown draft=#{draft_ms.round(1)} ms target_verify=#{target_verify_ms.round(1)} ms target_backup=#{target_backup_ms.round(1)} ms target_replay=#{target_replay_ms.round(1)} ms draft_backup=#{draft_backup_ms.round(1)} ms draft_resync=#{draft_resync_ms.round(1)} ms"
puts "spec_accounting proposal=#{proposal_ms.round(3)} ms accept_scan=#{accept_scan_ms.round(3)} ms commit=#{commit_ms.round(3)} ms"
puts profile_report.not_nil! if profile_report
puts "note=exact speculative probe; speedup still needs lower draft cost and/or verifier rollback overhead removal"
puts "generated=#{tok.decode(generated_ids).inspect}"
